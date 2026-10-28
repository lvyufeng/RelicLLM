"""Qwen4-Exp routed-expert backends.

Two implementations share one interface:

* `InMemoryMoE` keeps `gate_up_proj`/`down_proj` as dense tensors on whatever
  device they were handed.  Used by tests and small models.
* `HostExpertMoE` reads expert rows from the host through
  `reader.expert_rows()` and runs the SwiGLU on the GPU, staging only the active
  experts.  This is what makes the 225 GiB expert set usable on 4x22 GiB cards: a
  decode step touches at most `top_k` experts per layer (6.6 MiB each in BF16).
  The reader normally serves those rows from this rank's resident host shard (see
  `weights.HostExpertShard`); the mmap is only the fallback.

When the reader is FP8 it hands back `FP8Tensor`s instead of plain tensors.  Only
the codes and their 128x128 block scales cross PCIe — half the bytes of BF16 —
and the dequantization happens on the device right before the GEMM.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from typing import Protocol

import torch

# The module and not the name: one H2D stream a device is a property of the link rather than of a
# bank, so this is the same stream a per-device arena stages on, and `movement.copy_stream` is the
# one place a test or a probe replaces it.
from relicllm.components.moe import movement
from relicllm.models.qwen4_exp.layers import swiglu_expert
from relicllm.models.qwen4_exp.quant import FP8Tensor, dequantize_block_fp8
from relicllm.runtime.ops import load_ops


@dataclass
class _PendingExpert:
    gate_up: torch.Tensor | FP8Tensor
    down: torch.Tensor | FP8Tensor
    ready: torch.cuda.Event | None


class MoEBackend(Protocol):
    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """hidden_states: (tokens, hidden); indices/weights: (tokens, top_k)."""
        ...


def _expert_matrix(
    weight,
    compute_dtype: torch.dtype | None,
    default_dtype: torch.dtype,
) -> torch.Tensor:
    """One expert projection as a GEMM-ready dense matrix.

    An `FP8Tensor` is dequantized here, at the last moment before the GEMM, so
    only the codes and scales occupy device memory and cross PCIe.
    """
    if isinstance(weight, FP8Tensor):
        return weight.dequantize(compute_dtype or default_dtype)
    if compute_dtype is not None:
        return weight.to(compute_dtype)
    return weight


def _run_experts_fixed(
    hidden_states: torch.Tensor,
    weights: torch.Tensor,
    indices: torch.Tensor,
    num_experts: int,
    expert_rows,
    num_rows: int,
    *,
    compute_dtype: torch.dtype | None = None,
    rows_are_local: bool = True,
) -> torch.Tensor:
    """One token's top-k experts with a fixed amount of work and no host read.

    `_dispatch_experts` brings the routing back to the host twice (`unique.tolist()` and
    `counts.tolist()`) and then runs a Python loop whose length is how many *distinct* experts the
    token drew. Neither is needed: the slot count is `top_k` whatever the draw, so the work can be
    fixed and the loop replaced by `top_k` bound Python calls that are identical every step, which
    is what a captured graph holds. The arithmetic is the same as the loop's, slot by slot.

    `expert_rows(row)` answers one row's `(gate_up, down)` -- dense, `FP8Tensor`s or fp16, cast by
    `_expert_matrix` as before -- and what a "row" is, is the backend's: the id the route names
    (`ShardedMoE` does not renumber the shard, it masks; the deep cache's rows are its own ids,
    `id % world_size`, so the same id reads either), where an id at or past `num_experts` is
    another rank's expert and takes row 0, whose zero weight multiplies it out -- the loop's
    answer for skipping it -- or, with `rows_are_local`, a 0-based line of the backend's own
    table and nothing to remap.

    """
    top_k = int(indices.numel())
    flat = indices.reshape(-1)
    # A foreign id takes row 0 and a zero weight keeps that row out of the sum. Local rows are the
    # route's own ids, and none of them can be the sentinel: a zero-route rank has none at all.
    owned = (flat < num_experts) if not rows_are_local else torch.ones_like(flat, dtype=torch.bool)
    rows = flat if rows_are_local else torch.where(owned, flat, torch.zeros_like(flat))
    owned = owned.to(weights.dtype)
    out = torch.zeros_like(hidden_states)
    for slot in range(top_k):
        # One host read per slot, and this is the only one: the fixed shape of the step is what
        # a captured graph needs, and hosting this branch is the price of having it in Python.
        gate_up, down = expert_rows(int(rows[slot]))
        gate_up = _expert_matrix(gate_up, compute_dtype, hidden_states.dtype)
        down = _expert_matrix(down, compute_dtype, hidden_states.dtype)
        contrib = swiglu_expert(hidden_states, gate_up, down)
        weight = (weights.reshape(-1)[slot] * owned[slot]).to(contrib.dtype)
        out += (contrib * weight).to(out.dtype)
    return out


def _dispatch_experts(
    hidden_states: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
    num_experts: int,
    expert_weights,
    stats: dict | None = None,
    stager=None,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Group tokens by expert, run each expert once, scatter-add the results.

    `expert_weights(expert_id)` returns `(gate_up, down)` on the compute device,
    either as dense tensors or as `FP8Tensor`s.  Experts are visited in ascending
    id so the accumulation order is fixed and the result is run-to-run
    reproducible.

    When `stats` is given, per-phase wall time is accumulated into it.  The
    timing path synchronizes and is meant for profiling runs only.
    """
    out = torch.zeros_like(hidden_states)
    flat_experts = indices.reshape(-1)
    order = torch.argsort(flat_experts, stable=True)
    sorted_experts = flat_experts[order]
    unique, counts = torch.unique_consecutive(sorted_experts, return_counts=True)
    top_k = indices.shape[1]

    if stats is None:
        offset = 0
        unique_list = unique.tolist()
        counts_list = counts.tolist()
        expert_ids = [int(expert_id) for expert_id in unique_list if expert_id < num_experts]
        prefetched = stager(expert_ids) if stager is not None else None
        for expert_id, count in zip(unique_list, counts_list):
            slot = order[offset : offset + count]
            offset += count
            if expert_id >= num_experts:
                continue
            token_idx = slot // top_k
            k_idx = slot % top_k
            gate_up, down = next(prefetched) if prefetched is not None else expert_weights(int(expert_id))
            expert_input = hidden_states[token_idx]
            if compute_dtype is not None:
                expert_input = expert_input.to(compute_dtype)
            gate_up = _expert_matrix(gate_up, compute_dtype, expert_input.dtype)
            down = _expert_matrix(down, compute_dtype, expert_input.dtype)
            contrib = swiglu_expert(expert_input, gate_up, down)
            contrib = contrib * weights[token_idx, k_idx].unsqueeze(-1).to(contrib.dtype)
            out.index_add_(0, token_idx, contrib.to(out.dtype))
        return out

    import time

    dev = hidden_states.device
    sync = torch.cuda.synchronize if dev.type == "cuda" else lambda *a: None

    offset = 0
    unique_list = unique.tolist()
    counts_list = counts.tolist()
    for expert_id, count in zip(unique_list, counts_list):
        slot = order[offset : offset + count]
        offset += count
        if expert_id >= num_experts:
            continue
        token_idx = slot // top_k
        k_idx = slot % top_k

        sync(dev)
        t0 = time.perf_counter()
        gate_up, down = expert_weights(int(expert_id))
        sync(dev)
        t1 = time.perf_counter()
        expert_input = hidden_states[token_idx]
        if compute_dtype is not None:
            expert_input = expert_input.to(compute_dtype)
        gate_up = _expert_matrix(gate_up, compute_dtype, expert_input.dtype)
        down = _expert_matrix(down, compute_dtype, expert_input.dtype)
        contrib = swiglu_expert(expert_input, gate_up, down)
        contrib = contrib * weights[token_idx, k_idx].unsqueeze(-1).to(contrib.dtype)
        out.index_add_(0, token_idx, contrib.to(out.dtype))
        sync(dev)
        t2 = time.perf_counter()

        stats["stage_s"] = stats.get("stage_s", 0.0) + (t1 - t0)
        stats["compute_s"] = stats.get("compute_s", 0.0) + (t2 - t1)
        stats["experts"] = stats.get("experts", 0) + 1
    stats["calls"] = stats.get("calls", 0) + 1
    return out


class InMemoryMoE:
    """All experts resident as dense per-layer tensors."""

    def __init__(self, num_experts: int, gate_up: dict[int, torch.Tensor], down: dict[int, torch.Tensor]) -> None:
        self.num_experts = num_experts
        self.gate_up = gate_up  # layer_idx -> (num_experts, 2*inter, hidden)
        self.down = down  # layer_idx -> (num_experts, hidden, inter)

    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        gu = self.gate_up[layer_idx]
        dn = self.down[layer_idx]
        return _dispatch_experts(
            hidden_states,
            indices,
            weights,
            self.num_experts,
            lambda e: (gu[e], dn[e]),
        )


class HostExpertMoE:
    """Experts stay in host RAM; active ones are staged to GPU per step.

    The reader must expose `expert_rows(layer_idx, expert_id)` returning host
    tensors — resident shard rows when preloaded, mmap views otherwise.  A small
    LRU keeps recently used experts on device, which pays off during prefill where
    consecutive chunks reuse the same hot experts.
    """

    def __init__(
        self,
        reader,
        num_experts: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        cache_capacity: int = 0,
    ) -> None:
        self.reader = reader
        self.num_experts = num_experts
        self.device = device
        self.dtype = dtype
        self.cache_capacity = cache_capacity
        self._cache: OrderedDict[tuple[int, int], tuple[torch.Tensor | FP8Tensor, torch.Tensor | FP8Tensor]] = OrderedDict()
        self.stage_bytes = 0
        self.stage_calls = 0
        self.cache_hits = 0
        # Set to a dict to collect per-phase timings (profiling only).
        self.phase_stats: dict | None = None
        self.prefetch_enabled = os.environ.get("POCKETLLM_QWEN4_MOE_PREFETCH", "1").lower() in {
            "1", "true", "yes"
        }
        self._cuda_ext = None
        self.grouped_enabled = os.environ.get("POCKETLLM_QWEN4_MOE_GROUPED", "0").lower() in {
            "1", "true", "yes"
        }
        self.grouped_min_tokens = int(os.environ.get("POCKETLLM_QWEN4_MOE_GROUPED_MIN_TOKENS", "128"))
        self.grouped_max_padding = float(os.environ.get("POCKETLLM_QWEN4_MOE_GROUPED_MAX_PADDING", "2.5"))
        self.grouped_min_free_gib = float(
            os.environ.get("POCKETLLM_QWEN4_MOE_GROUPED_MIN_FREE_GIB", "2.0")
        )
        self.fp16_compute_min_tokens = max(
            1,
            int(os.environ.get("POCKETLLM_QWEN4_MOE_FP16_MIN_TOKENS", "128")),
        )

    def _cache_put(self, key: tuple[int, int], pair: tuple[torch.Tensor, torch.Tensor]) -> None:
        if self.cache_capacity <= 0:
            return
        self._cache[key] = pair
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_capacity:
            self._cache.popitem(last=False)

    def _cache_get(self, key: tuple[int, int]):
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.cache_hits += 1
        return hit

    @staticmethod
    def _stage_weight(weight, device: torch.device, dtype: torch.dtype):
        if isinstance(weight, FP8Tensor):
            return weight.to(device, non_blocking=True)
        return weight.to(device, dtype=dtype, non_blocking=True)

    @staticmethod
    def _weight_nbytes(weight) -> int:
        return weight.nbytes() if isinstance(weight, FP8Tensor) else weight.numel() * weight.element_size()

    @staticmethod
    def _record_stream(weight, stream) -> None:
        if isinstance(weight, FP8Tensor):
            weight.code.record_stream(stream)
            weight.scale.record_stream(stream)
        else:
            weight.record_stream(stream)

    def _fetch(self, layer_idx: int, expert_id: int):
        key = (layer_idx, expert_id)
        hit = self._cache_get(key)
        if hit is not None:
            return hit
        gate_up_cpu, down_cpu = self.reader.expert_rows(layer_idx, expert_id)
        gate_up = self._stage_weight(gate_up_cpu, self.device, self.dtype)
        down = self._stage_weight(down_cpu, self.device, self.dtype)
        self.stage_bytes += self._weight_nbytes(gate_up) + self._weight_nbytes(down)
        self.stage_calls += 1
        self._cache_put(key, (gate_up, down))
        return gate_up, down

    def _dequantized_layer_tensors(self, layer_idx: int):
        """Return a dense local layer only when a grouped path can consume it."""
        tensors = self.reader.expert_shard.layer_tensors(layer_idx)
        if isinstance(tensors[0], FP8Tensor):
            return None
        return tensors

    def _cache_clear(self) -> None:
        self._cache.clear()


    def _grouped_bf16(self, layer_idx: int, hidden_states, indices, weights):
        """Run a dense-local-expert BF16 grouped prefill, or return None."""
        if hidden_states.shape[0] < self.grouped_min_tokens:
            return None
        shard = getattr(self.reader, "expert_shard", None)
        if shard is None or not shard.has_layer(layer_idx):
            return None
        if self.dtype is not torch.bfloat16:
            return None
        if shard.world_size <= 1:
            # A single-rank shard contains all experts; copying a full layer is
            # unnecessarily large, so keep the per-expert path there.
            return None
        if self._cuda_ext is None:
            self._cuda_ext = load_ops()
        if self._cuda_ext is None or not hasattr(
            self._cuda_ext, "qwen4_exp_moe_prefill_bf16_forward"
        ):
            return None

        layer_tensors = self._dequantized_layer_tensors(layer_idx)
        if layer_tensors is None:
            return None
        gate_up_cpu, down_cpu = layer_tensors
        layer_bytes = (
            gate_up_cpu.numel() * gate_up_cpu.element_size()
            + down_cpu.numel() * down_cpu.element_size()
        )
        try:
            free_bytes, _ = torch.cuda.mem_get_info(self.device)
        except RuntimeError:
            return None
        reserve_bytes = max(0, int(self.grouped_min_free_gib * (1 << 30)))
        if free_bytes < layer_bytes + reserve_bytes:
            return None

        flat = indices.reshape(-1)
        top_k = indices.shape[1]
        order = torch.argsort(flat, stable=True)
        sorted_ids = flat[order]
        unique, counts = torch.unique_consecutive(sorted_ids, return_counts=True)

        unique_list = [int(e) for e in unique.tolist()]
        counts_list = [int(c) for c in counts.tolist()]
        local_experts = shard.num_local_experts
        offset = 0
        route_tokens = []
        route_weights = []
        counts_by_local = [0] * local_experts
        for expert_id, count in zip(unique_list, counts_list):
            begin, offset = offset, offset + count
            if expert_id >= self.num_experts or not shard.owns(expert_id):
                continue
            local_id = shard.local_index(expert_id)
            if local_id < 0 or local_id >= local_experts:
                return None
            slots = order[begin:offset]
            route_tokens.append((slots // top_k).to(torch.int64))
            route_weights.append(weights.reshape(-1)[slots].to(torch.float32))
            counts_by_local[local_id] = count
        if not route_tokens:
            # This is the expected result for a ShardedMoE rank with no local
            # routes.  The caller's all-reduce will combine the zero partial.
            return torch.zeros_like(hidden_states)

        route_count = sum(counts_by_local)
        max_count = max(counts_by_local)
        padding_ratio = shard.num_local_experts * max_count / max(1, route_count)
        if padding_ratio > self.grouped_max_padding:
            return None

        route_tokens = torch.cat(route_tokens).contiguous()
        route_weights = torch.cat(route_weights).contiguous()
        seg = [0]
        for count in counts_by_local:
            seg.append(seg[-1] + count)
        seg_starts = torch.tensor(seg, device=hidden_states.device, dtype=torch.int32)

        # Keep the local shard's expert-id layout intact.  This replaces many
        # small H2D copies with two contiguous layer copies; zero-route experts
        # are represented by empty segments and never contribute to the output.
        stream = movement.copy_stream(self.device)
        current = torch.cuda.current_stream(self.device)
        with torch.cuda.stream(stream):
            gate_up = self._stage_weight(gate_up_cpu, self.device, self.dtype)
            down = self._stage_weight(down_cpu, self.device, self.dtype)
            ready = torch.cuda.Event()
            ready.record(stream)
        current.wait_event(ready)
        self._record_stream(gate_up, current)
        self._record_stream(down, current)
        self.stage_bytes += self._weight_nbytes(gate_up) + self._weight_nbytes(down)
        self.stage_calls += 1
        return self._cuda_ext.qwen4_exp_moe_prefill_bf16_forward(
            hidden_states.contiguous(),
            route_tokens,
            route_weights,
            seg_starts,
            gate_up,
            down,
            0.0,
        )

    def _prefetch(self, layer_idx: int, expert_ids: list[int]):
        """Stage one layer's active experts on the copy stream in expert-id order.

        FP8 entries are cached as code/scale pairs; dequantization is performed
        after the copy-stream event and immediately before their GEMM in the
        dispatch loop.  This preserves the FP8 residency advantage.

        The stream is the movement layer's (`components.moe.movement.copy_stream`),
        the same one a per-device arena stages on, and not one of this class's:
        one H2D stream a device is a property of the link and not of a bank.

        The ordering is one edge and not the arena path's two.  A staged expert
        lands in freshly allocated device tensors that an LRU holds until it
        evicts them, so a copy overwrites nothing a kernel may still be reading
        and only the compute stream has to be made to wait: on this stream's
        event for a miss, and on nothing for a hit.  The `record_stream` in
        `_record_stream` is what keeps the allocator from handing the block
        back before the compute stream is done reading it.
        """
        stream = movement.copy_stream(self.device)
        current = torch.cuda.current_stream(self.device)
        pending: list[_PendingExpert] = []
        with torch.cuda.stream(stream):
            for expert_id in expert_ids:
                key = (layer_idx, expert_id)
                hit = self._cache_get(key)
                if hit is not None:
                    pending.append(_PendingExpert(hit[0], hit[1], None))
                    continue
                gate_up_cpu, down_cpu = self.reader.expert_rows(layer_idx, expert_id)
                gate_up = self._stage_weight(gate_up_cpu, self.device, self.dtype)
                down = self._stage_weight(down_cpu, self.device, self.dtype)
                event = torch.cuda.Event()
                event.record(stream)
                pending.append(_PendingExpert(gate_up, down, event))
                self.stage_bytes += self._weight_nbytes(gate_up) + self._weight_nbytes(down)
                self.stage_calls += 1
                self._cache_put(key, (gate_up, down))

        def consume():
            for item in pending:
                if item.ready is not None:
                    current.wait_event(item.ready)
                self._record_stream(item.gate_up, current)
                self._record_stream(item.down, current)
                yield item.gate_up, item.down

        return consume()

    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.grouped_enabled
            and self.phase_stats is None
            and hidden_states.device.type == "cuda"
        ):
            grouped = self._grouped_bf16(layer_idx, hidden_states, indices, weights)
            if grouped is not None:
                return grouped

        use_prefetch = (
            self.prefetch_enabled
            and self.phase_stats is None
            and hidden_states.device.type == "cuda"
        )
        fp16_compute_enabled = os.environ.get(
            "POCKETLLM_QWEN4_MOE_FP16_COMPUTE", "0"
        ).lower() in {"1", "true", "yes"}
        compute_dtype = (
            torch.float16
            if fp16_compute_enabled
            and hidden_states.device.type == "cuda"
            and hidden_states.dtype is torch.bfloat16
            and hidden_states.shape[0] >= self.fp16_compute_min_tokens
            else None
        )
        return _dispatch_experts(
            hidden_states,
            indices,
            weights,
            self.num_experts,
            lambda e: self._fetch(layer_idx, e),
            stats=None if use_prefetch else self.phase_stats,
            stager=(lambda expert_ids: self._prefetch(layer_idx, expert_ids)) if use_prefetch else None,
            compute_dtype=compute_dtype,
        )


class ShardedMoE:
    """Wraps a backend so each rank only runs the experts it owns.

    Experts are partitioned round-robin by id across `world_size` ranks; the
    per-layer all-reduce that already follows the MLP block sums the partial
    results, so no extra communication is needed.
    """

    def __init__(self, inner: MoEBackend, rank: int, world_size: int) -> None:
        self.inner = inner
        self.rank = rank
        self.world_size = world_size

    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.inner(layer_idx, hidden_states, self.local_ids(indices), weights)

    def local_ids(self, indices: torch.Tensor) -> torch.Tensor:
        """Ids rewritten to this rank's local rows, with foreign experts sentinel-ised.

        The sentinel is `num_experts`, which both dispatch loops skip; it is chosen at or above the
        global count so it can never name a real expert on any rank.
        """
        if self.world_size == 1:
            return indices
        mine = (indices % self.world_size) == self.rank
        return torch.where(mine, indices, torch.full_like(indices, self.inner.num_experts))


def _fp8_slots_ops():
    """The decode kernel that runs a top-k of FP8 experts in one launch pair, or None."""
    from relicllm.runtime.ops import load_ops

    ops = load_ops()
    if ops is None or not hasattr(ops, "qwen4_exp_fp8_slots_swiglu"):
        return None
    return ops


def encode_route(indices: torch.Tensor, world_size: int, rank: int) -> torch.Tensor:
    """A route as a local-row *code*: this rank's row, or `-1` for another rank's expert.

    The expert partition is round-robin by id across the TP width (`expert_id % world_size`), so
    this rank's row for an id is `expert_id // world_size` and `ShardedMoE`'s sentinel -- an id at
    or past the global count -- is not this rank's either.  A backend with a row-addressed table
    wants this form, and it is a code rather than an id so that a foreign slot reads an empty row
    instead of someone else's expert.

    Written with integer ops on the tensors as given: it runs inside a captured decode graph, where
    a host read or a shape that depended on the route would be the one thing the graph cannot hold.
    """
    if world_size == 1:
        return indices.reshape(-1).to(torch.int32)
    flat = indices.reshape(-1)
    mine = (flat % world_size) == rank
    local = torch.div(flat, world_size, rounding_mode="floor")
    return torch.where(mine, local, torch.full_like(flat, -1)).to(torch.int32)


class Fp8SlotExpertMoE:
    """The rank's FP8 experts for every layer, one launch pair a layer a step.

    The placement is `DeviceExpertMoE`'s -- the shard's codes and scales on the card, nothing staged
    a step -- and the difference is the arithmetic's *shape*: one launch computes every routed
    expert's gate/up activation for the token and a second reduces them, both reading e4m3 and the
    block scales in the load, so no expert is expanded to bf16 on the way.  What the per-slot loop
    cost on a 3090 was not the table read but the bound Python calls and their launches, about 8us a
    slot, and the GPU's own launch-to-launch time is 8.3us eager or 0.92us in a graph -- so the point
    of this backend is not this op's few microseconds but that it is *two* launches for the whole
    route, the same two whatever the draw, which a captured step can hold.

    The route arrives as `encode_route`'s code and the caller zeroes a foreign slot's weight, so an
    empty slot reads an empty row and adds nothing: the loop's answer for skipping it.
    """

    def __init__(self, shard, num_experts: int, device: torch.device, *, blocks_per_expert: int = 40):
        self.shard = shard
        self.num_experts = int(num_experts)
        self.world_size = int(getattr(shard, "world_size", 1))
        self.rank = int(getattr(shard, "rank", 0))
        self.device = device
        #: How many blocks an expert's row range is split across, in both phases of the fused
        #: kernel.  Four is the default because a slot's `inter` is 640 rows and eight warps a
        #: block, so four blocks is 32 warps an expert -- enough to fill a card's scheduler without
        #: splitting the reduction past the point where the launch dominates.  Overridable because
        #: the right value is a property of the card, and 3090 / 4090 / 2080 Ti are not one card.
        self.blocks_per_expert = int(
            os.environ.get("POCKETLLM_QWEN4_FP8_EXPERT_BLOCKS", blocks_per_expert)
        )
        #: The counters `/metrics` reads; nothing is staged, as on `DeviceExpertMoE`.
        self.stage_bytes = 0
        self.stage_calls = 0
        self.cache_hits = 0
        self.cache_capacity = 0
        self.phase_stats = None
        self.ops = _fp8_slots_ops()
        if self.ops is None:
            raise RuntimeError("Fp8SlotExpertMoE needs the relic_core CUDA extension")
        if not getattr(shard, "packed_gate_up", False):
            # The plane is a third of the table's bytes *on top of* the row copy, and on a card
            # that is already holding 14 GiB of experts it does not fit beside it. The shard is
            # therefore asked to store the planes instead of the rows -- one copy, one layout.
            raise TypeError(
                "Fp8SlotExpertMoE reads gate/up *passing through* its planes as they are allocated; "
                "the resident shard has to be filled in that layout (preload_experts(packed_gate_up=True))"
            )

    def _is_decode_step(self, hidden_states: torch.Tensor, layer_idx: int) -> bool:
        """Whether this call is one token, which is the only shape the fused kernel takes.

        The kernel is a decode kernel and says so in its own checks: `x` must be exactly one row of
        `hidden`, and the slots it computes are the routes of *one* token.  A prefill hands it
        `prompt_tokens x hidden` rows and the assertion fires, so the two phases cannot share this
        path -- and they should not, because they are different arithmetic.  Routing prefill to the
        grouped loop is what lets the flag be turned on for a whole run rather than only for a
        decode of one step.

        The answer comes from the *planes*, not from `layer_tensors`: a `packed_gate_up` shard
        answers `layer_tensors` by unpacking the whole layer back into rows, a 300 MiB permute and
        copy on a 3090 that costs more than the step's arithmetic, and a step that asked only for
        the width would pay it twice a layer.
        """
        """The planes are `[local, ceil(hidden/16), 2*inter, 16]`, so the hidden width is dim -3 times 16."""
        hidden = int(self.shard.gate_up_planes(layer_idx).shape[-3]) * 16
        return hidden_states.numel() == hidden

    def _prefill(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """The many-token path: group by expert and run each once, the shared loop's arithmetic.

        Same seam `DeviceExpertMoE` uses, so prefill and decode agree on the accumulation order and
        on how an FP8 row is dequantized; only the launch shape differs.

        The row is cut out of `layer_tensors`, not taken from `rows`: a `packed_gate_up` shard --
        which this backend requires -- stores the fused kernel's planes, and `rows` indexes those
        planes as if they were expert matrices.  `layer_tensors` is the accessor that unpacks them
        back into the row layout a grouped GEMM takes, so prefill reads the same numbers decode
        would have computed from.
        """
        gate_up, down = self.shard.layer_tensors(layer_idx)

        def expert_row(expert_id: int):
            local = self.shard.local_index(expert_id)
            if isinstance(gate_up, FP8Tensor):
                return (
                    FP8Tensor(gate_up.code[local], gate_up.scale[local], gate_up.block),
                    FP8Tensor(down.code[local], down.scale[local], down.block),
                )
            return gate_up[local], down[local]

        return _dispatch_experts(
            hidden_states,
            indices,
            weights,
            self.num_experts,
            expert_row,
            stats=self.phase_stats,
        )

    def _table(self, layer_idx: int):
        """The layer's gate/up planes and their scale, and its down codes.

        The kernel reads the planes as the shard stores them, so the row-layout form never has to
        exist: asking `layer_tensors` for it answers by unpacking the whole layer back into rows,
        a permute and copy of ~300 MiB once a layer and once a step -- for bytes the step already
        has, in the order it wants them.
        """
        planes = self.shard.gate_up_planes(layer_idx)
        scale = self.shard.gate_up_plane_scale(layer_idx)
        down = self.shard.down_tensors(layer_idx)
        if not isinstance(down, FP8Tensor):
            raise TypeError(
                f"layer {layer_idx} holds {type(down).__name__}, not FP8 codes: the fused "
                "backend runs the block-scaled layout only"
            )
        return planes, scale, down

    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self._is_decode_step(hidden_states, layer_idx):
            return self._prefill(layer_idx, hidden_states, indices, weights)
        routed = encode_route(indices, self.world_size, self.rank)
        empty = routed < 0
        # A foreign slot's weight is the zeroing: the kernel still runs its (empty) row, so the
        # work and the shapes are the route's, not the draw's.
        wts = torch.where(empty, torch.zeros((), dtype=weights.dtype, device=weights.device),
                          weights.to(torch.bfloat16).reshape(-1))
        planes, plane_scale, down = self._table(layer_idx)
        slot_rows = self.ops.qwen4_exp_fp8_slots_swiglu(
            hidden_states.reshape(-1),
            routed,
            wts,
            planes,
            plane_scale,
            planes,
            down.code,
            down.scale,
            self.blocks_per_expert,
        )
        # Slot order, folded once: the loop adds its contributions in ascending slot too, and this
        # keeps a `[top_k, hidden]` from travelling up as rows a caller would add one at a time.
        return slot_rows.sum(0).to(hidden_states.dtype).reshape_as(hidden_states)


class DeepCacheMoE:
    """This rank's experts for every layer it can hold, decoded once and read in place after.

    The resident mode is *wide*: at tensor-parallel width 8 a rank holds a sixth of every layer's
    routed experts, 64 a layer, every id in its `48 x 64` set present.  That is 14 GiB, and it sets
    the width.  This mode is *deep*: every layer, this rank's shard of each, as many layers as the
    card leaves room for.  Both keep the bytes across steps; they differ in which axis of the
    expert table the card holds, and a rank cannot hold both -- the first is what a step reads in
    full and the second is 39 GiB decoded, so on a 24 GiB card a layer's fit is a measurement.

    The bytes kept are not the checkpoint's.  Stored, a layer of these experts is FP8 codes under
    128x128 block scales, and decoding one expert per slot under the GEMM costs a copy per expert.
    Decoded once at load, a layer is bf16 at *one* fixed scale -- bounded by the product of the two
    largest FP8 magnitudes, so no exponent is lost outside the checked zero -- and then a step's
    expert cost is a row read.  What that gives up is the block scales: a block whose own scale is
    small loses its low bits, and at the 128x128 blocks this checkpoint uses the per-block maxima
    run within a factor of a few of each other, which is the headroom that makes one scale work.

    Nothing is staged, nothing is cached and nothing is decided per step: a decode step reads
    `top_k` rows out of one tensor whatever the layer, and a token's draw does not change the
    amount of work.  Layers beyond the card's room are not here at all -- `has_layer` is false and
    the runtime sends them to the host stove.

    The table holds this rank's rows only, one per id it owns (`id % world_size`), so it is
    `num_experts / world_size` deep however wide the tensor-parallel group is.  The step reads it
    through `local_rows`, the same masking every other backend gets from `ShardedMoE`: a foreign id
    becomes the sentinel and takes a zeroed row.
    """

    def __init__(
        self,
        num_experts: int,
        device: torch.device,
        *,
        scale: float,
        world_size: int = 1,
        rank: int = 0,
    ) -> None:
        self.num_experts = int(num_experts)
        #: The rank's width the rows are dealt over; this rank holds every `world_size`-th id.
        self.world_size = max(1, int(world_size))
        #: Which `id % world_size` this rank's stored rows are.
        self.rank = int(rank) % self.world_size
        self.device = device
        #: Never incremented: there is no staging on this path, which is the point.  Named like
        #: `DeviceExpertMoE`'s for the same reason -- `/metrics` reads them to tell residency from
        #: a hot cache.
        self.stage_bytes = 0
        self.stage_calls = 0
        self.cache_hits = 0
        self.cache_capacity = 0
        self.phase_stats = None
        self.scale = float(scale)
        self.gate_up: dict[int, torch.Tensor] = {}
        self.down: dict[int, torch.Tensor] = {}
        self.decoded_bytes = 0

    @property
    def num_local_experts(self) -> int:
        """Rows per layer: the experts this rank owns."""
        for layer in self.gate_up:
            return int(self.gate_up[layer].shape[0])
        return 0

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self.gate_up

    def local_rows(self, indices: torch.Tensor) -> torch.Tensor:
        """The route's ids with foreign experts sentinel-ised, as `ShardedMoE` spells it."""
        if self.world_size == 1:
            return indices
        mine = (indices % self.world_size) == self.rank
        return torch.where(mine, indices, torch.full_like(indices, self.num_experts))

    def line_of(self, masked: torch.Tensor) -> torch.Tensor:
        """Masked ids to this table's lines: an owned id is line `(id - rank) // world_size`.

        The sentinel keeps reading the last line.  That line is a real expert of this rank, but
        the loop's owned-weight test zeroes a sentinel's contribution, so the row is never summed;
        it only has to be a legal index into a table that is shorter than the id axis.
        """
        if self.world_size == 1:
            return masked
        lines = torch.div(masked - self.rank, self.world_size, rounding_mode="floor")
        last = self.gate_up and next(iter(self.gate_up.values())).shape[0] or 1
        return lines.clamp_(0, last - 1)

    def decode_layer(self, layer_idx: int, stored) -> int:
        """Decode one layer's local rows into fixed-scale bf16; returns the bytes it holds.

        `stored` is the pair the checkpoint has: FP8 codes, or packed dense rows, which are scaled
        the same way so one mode serves both fixtures and the real thing.  Calling this twice for a
        layer keeps the first decode rather than decoding the codes again.
        """
        if layer_idx in self.gate_up:
            return 0
        gate_up, down = _decode_pair(stored, self.scale, self.device)
        self.gate_up[layer_idx] = gate_up
        self.down[layer_idx] = down
        held = (gate_up.numel() + down.numel()) * 2
        self.decoded_bytes += held
        return held

    def drop_layer(self, layer_idx: int) -> None:
        self.decoded_bytes -= (
            self.gate_up.pop(layer_idx).numel() + self.down.pop(layer_idx).numel()
        ) * 2

    def rows(self, layer_idx: int, row: int):
        return self.gate_up[layer_idx][row], self.down[layer_idx][row]

    def layer_row(self, layer_idx: int, row: int, gate_up: torch.Tensor | None = None):
        """One row of a layer, from the tensors the caller already looked up if it has them."""
        if gate_up is None:
            gate_up, down = self.rows(layer_idx, row)
            return gate_up, down
        return gate_up[row], self.down[layer_idx][row]

    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        rows_count = int(self.gate_up[layer_idx].shape[0])
        masked = self.local_rows(indices)
        # This table *is* compact, so the route's ids are not its lines: only the owned
        # `id % world_size` is in it, and the masked sentinel has no line at all.  The weight is
        # zeroed here, both for the sentinel and for a zero draw, because the loop cannot see the
        # row is a substitute.
        owned = (masked < self.num_experts).reshape(weights.shape)
        weights = torch.where(owned, weights, torch.zeros((), dtype=weights.dtype, device=weights.device))
        lines = self.line_of(masked).contiguous()
        gate_up, down = self.gate_up[layer_idx], self.down[layer_idx]
        if hidden_states.shape[0] == 1 and self.phase_stats is None:
            # Decode: one row per slot, from one tensor per layer, whatever an id names.
            def rows(line: int):
                return self.layer_row(layer_idx, line, gate_up)

            # Every slot's line is legal here (the mask spelled it), so nothing is sentinel-ised.
            return _run_experts_fixed(
                hidden_states, weights, lines, self.num_experts, rows, rows_count,
                rows_are_local=True,
            )
        return _dispatch_experts(
            hidden_states,
            lines,
            weights,
            rows_count,
            lambda line: (gate_up[line], down[line]),
            stats=self.phase_stats,
        )


class DeepFallbackMoE:
    """The deep cache for the layers it holds, the host stove for the ones it does not.

    A cache that could not decode every layer is not a broken run: the layers it holds are the
    cheap ones and the rest are staged per step exactly as `--expert-resident` off would.  Which
    layers are which is the cache's own `has_layer`, so the split is one place and the runtime that
    decodes them does not have to tell this class anything.
    """

    def __init__(self, cache: DeepCacheMoE, host: "HostExpertMoE") -> None:
        self.cache = cache
        self.host = host
        self.num_experts = cache.num_experts
        # The host half owns the counters and the phase stats, and `__getattr__` answers all of
        # them from it, because it is the half that moves bytes.

    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.cache.has_layer(layer_idx):
            return self.cache(layer_idx, hidden_states, indices, weights)
        return self.host(layer_idx, hidden_states, indices, weights)

    def __getattr__(self, name):
        return getattr(self.__dict__["host"], name)


def _decode_pair(stored, scale: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """A layer's `(gate_up, down)` in bf16 at one fixed `scale`, on `device`."""
    gate_up, down = stored
    if isinstance(gate_up, FP8Tensor):
        return _decode_fp8(gate_up, scale, device), _decode_fp8(down, scale, device)
    return (
        (gate_up / scale).to(device, torch.bfloat16),
        (down / scale).to(device, torch.bfloat16),
    )


def _decode_fp8(weight: FP8Tensor, scale: float, device) -> torch.Tensor:
    """`weight / scale` as bf16 on `device`.

    `weight` is a layer's local rows (leading axis an expert) with one grid per expert, so the
    expansion applies to the stacked tensor as it stands: `dequantize_block_fp8` tiles only the
    last two axes and broadcasts over the leading one.

    The division is by the scale itself rather than by its reciprocal: `x / t` on the card rounds
    the quotient once, while `x * (1/t)` rounds `1/t` to bf16 first and then the product, which is
    a different number for most `x`.

    The codes are not assumed to be on `device`: an untouched checkpoint is a file-backed mapping,
    and expanding it in place is how a layer's rows come over.
    """
    code, grid, block = weight.code, weight.scale, weight.block
    if code.device == device and code.device.type == "cuda":
        raise ValueError("decoding reads codes from the host or from this device")
    decoded = dequantize_block_fp8(code, grid, block, torch.bfloat16).to(device)
    return decoded / decoded.new_tensor(scale)


class DeviceExpertMoE:
    """This rank's experts resident on the device; nothing is staged per step.

    The same shard `HostExpertMoE` stages from, held where it is read instead of
    where it is stored.  It is a *substitution at `_dispatch_experts`'s
    `expert_weights` seam* rather than a second dispatch loop: grouping, the
    ascending-id accumulation order, the dequantization at the GEMM and the
    scatter-add are the host path's, shared rather than reimplemented, so the two
    modes cannot drift apart in arithmetic while differing in placement.

    What it drops is the staging policy, and with it the whole reason
    `HostExpertMoE` exists: no `_fetch`, no `_prefetch`, no LRU, no per-expert
    event or copy stream, because `rows()` already answers with device tensors.
    `_stage_weight` is the identity here.

    The three counters are kept and stay at zero.  They are not decorative:
    `/metrics` reads `stage_bytes` / `stage_calls` / `cache_hits` to tell
    *residency* from *a hot cache*, and a deployment that sizes an LRU to hold
    every expert it owns stops staging too -- so a zero here is the evidence that
    the experts are present, not merely being hit.  A backend without the
    attributes could not be told apart from one that never staged.
    """

    def __init__(self, shard, num_experts: int, device: torch.device) -> None:
        self.shard = shard
        self.num_experts = num_experts
        self.device = device
        #: Never incremented: there is no staging on this path, which is the point.
        self.stage_bytes = 0
        self.stage_calls = 0
        self.cache_hits = 0
        #: No cache, so no capacity -- reported as 0 rather than the shard's expert count so a
        #: metrics reader cannot mistake residency for an LRU sized to fit.
        self.cache_capacity = 0
        self.phase_stats = None

    def layer_row(self, layer_idx: int, row: int):
        """This rank's `row`-th expert of `layer_idx`, indexing the place its rows actually sit.

        `_run_experts_fixed` is handed rows, not ids: the sentinel rewrite has already happened,
        so every value is local by construction and 0..num_rows.  The resident shard is indexed by
        *global* expert id (`ShardedMoE` does not renumber, it masks), so "row" here means the
        id -- an intermediate id indexes nothing and must not be mapped into the dense table.
        The deep cache, whose table is local and 0-based, indexes it directly.
        """
        raise NotImplementedError

    def expert_weights(self, layer_idx: int, expert_id: int):
        """One expert's `(gate_up, down)` straight off the card, already there."""
        if not self.shard.owns(expert_id):
            raise ValueError(
                f"expert {expert_id} is not this rank's: rank {self.shard.rank} of "
                f"{self.shard.world_size} owns ids where id % {self.shard.world_size} "
                f"== {self.shard.rank}"
            )
        return self.shard.rows(layer_idx, expert_id)

    def layer_row(self, layer_idx: int, row: int, gate_up=None):
        """One row.  The optional `gate_up` is not used here -- a decode step already holds the
        layer's tensors and `_run_experts_fixed` passes it, so the signature has to accept it."""
        gate_up, down = self.shard.layer_tensors(layer_idx)
        if isinstance(gate_up, FP8Tensor):
            return (
                FP8Tensor(gate_up.code[row], gate_up.scale[row], gate_up.block),
                FP8Tensor(down.code[row], down.scale[row], down.block),
            )
        return gate_up[row], down[row]

    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        if (
            hidden_states.shape[0] == 1
            and self.phase_stats is None
            and os.environ.get("POCKETLLM_QWEN4_MOE_FIXED", "1").lower() in {"1", "true", "yes"}
        ):
            # A decode step: `top_k` bound calls whatever the draw, and no host read, which is the
            # shape a captured graph needs. The arithmetic is the loop's, slot by slot
            # (tests/test_qwen4_exp_moe_fixed.py); it buys ~10% at ten distinct experts, and the
            # fixed work is what a graph would then remove the launch overhead from.
            gate_up, _ = self.shard.layer_tensors(layer_idx)

            def rows(local_row: int):
                return self.layer_row(layer_idx, local_row, gate_up)

            # The table is this rank's, one row per owned id, so the route's ids are lines here:
            # `ShardedMoE`'s sentinel is the only value at or past the count and has no line.
            # Its weight goes to zero, which is what keeps the substituted row out of the sum --
            # and a row that reads another id's expert under a zero weight is never summed either.
            local_rows = int(gate_up.shape[0])
            owned = (indices < self.num_experts).reshape(weights.shape)
            weights = torch.where(
                owned, weights, torch.zeros((), dtype=weights.dtype, device=weights.device)
            )
            lines = indices.clamp_(0, local_rows - 1).contiguous()
            return _run_experts_fixed(
                hidden_states, weights, lines, self.num_experts, rows, local_rows
            )
        return _dispatch_experts(
            hidden_states,
            indices,
            weights,
            self.num_experts,
            lambda expert_id: self.expert_weights(layer_idx, expert_id),
            stats=self.phase_stats,
        )
