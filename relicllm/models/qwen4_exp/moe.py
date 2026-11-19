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
    # `owned` is all ones when the rows are the backend's own lines, and the caller that masks a
    # foreign route has already zeroed that weight -- so multiplying by it again is a real kernel
    # whose result is known.  Fold it once here instead of once a slot.
    # The ownership multiply is the same value every slot, so it is done once here rather than
    # once a slot.  When the rows are the backend's own lines `owned` is all ones and the caller
    # has already masked a foreign route, so the multiply is dropped entirely.
    share = weights.reshape(-1) if rows_are_local else weights.reshape(-1) * owned
    for slot in range(top_k):
        # One host read per slot, and this is the only one: the fixed shape of the step is what
        # a captured graph needs, and hosting this branch is the price of having it in Python.
        gate_up, down = expert_rows(int(rows[slot]))
        gate_up = _expert_matrix(gate_up, compute_dtype, hidden_states.dtype)
        down = _expert_matrix(down, compute_dtype, hidden_states.dtype)
        contrib = swiglu_expert(hidden_states, gate_up, down)
        weight = share[slot].to(contrib.dtype)
        # Deliberately not `out.addcmul_(contrib, weight)`, which is one launch shorter: a fused
        # multiply-add rounds once where this rounds the product and then the sum, and in bf16
        # that is a different number -- 256 off on a 50k value in the fixed-slot test.  The step
        # keeps the two-rounding form so every measurement on this branch compares against the
        # same tokens.
        out += (contrib * weight).to(out.dtype)
    return out



def _batched_experts(
    x: torch.Tensor,
    gate_up: torch.Tensor,
    down: torch.Tensor,
    indices: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """Every routed expert of one token in one batched pass, with no host read.

    The per-expert loop visits `top_k` experts with ~5 kernels each and, at decode, that loop is
    the step: measured on 4x2080Ti a layer's MoE phase was 2.46 ms while the weights it touched
    were 65 MiB, which at the card's bandwidth is 0.11 ms -- the rest was launch and index work.
    Stacking the experts into one `[k, ...]` bmm turns about 50 small kernels into 6.

    This is the decode shape only: one token, so every expert is a GEMV and the batch axis is
    the expert. A prefill visits an expert with many tokens and keeps the grouped path.

    `gate_up`/`down` are already on the device, `[k, 2*inter, hidden]` / `[k, hidden, inter]`,
    and `indices` names which expert each row is. No `.tolist()`: the reason the per-expert loop
    reads the routing back to the host is to index a Python list of weights, and a stack does not
    need that -- which is also what lets this run inside a capture.
    """
    xb = x.expand(indices.numel(), -1).unsqueeze(1)          # [k, 1, hidden]
    gate_up_x = torch.bmm(xb, gate_up.transpose(1, 2))        # [k, 1, 2*inter]
    gate, up = gate_up_x.chunk(2, dim=-1)
    act = torch.nn.functional.silu(gate) * up                 # [k, 1, inter]
    per_expert = torch.bmm(act, down.transpose(1, 2))         # [k, 1, hidden]
    return (per_expert.squeeze(1).to(x.dtype) * weights.reshape(-1, 1).to(x.dtype)).sum(0)


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
        resident_layers: int = 0,
    ) -> None:
        self.reader = reader
        self.num_experts = num_experts
        self.device = device
        self.dtype = dtype
        self.cache_capacity = cache_capacity
        #: How many layers' experts are held on the device for the whole run, from layer 0 up.
        #: A cache answers "this expert was used recently"; this answers "this layer never
        #: leaves", and on a box that cannot hold all 48 layers it is the difference between
        #: staging 100% of the layer set every step and staging the tail of it. Filled by
        #: `pin_layers`, which is what decides how many actually fit.
        self._resident: dict[int, dict[int, tuple]] = {}

        #: An `nvfp4_moe.NvFp4ExpertBank` when the checkpoint's experts are NVFP4, or None. When
        #: it is set it answers for the layers it holds and this class never stages anything from
        #: them -- the FP8 codes a ModelOpt checkpoint does not have.
        self._nvfp4 = None
        self._resident_layers = int(resident_layers)
        #: Hold the pinned experts widened to bf16 rather than as the checkpoint's FP8 codes.
        self.wide_bank = os.environ.get("POCKETLLM_QWEN4_PIN_BF16", "0").lower() in {
            "1", "true", "yes"
        }
        #: Batch the routed experts of *staged* layers too. Those need a temporary bf16 stack,
        #: which is why this is asked for rather than assumed: a stack a layer is 94 MiB and the
        #: card may have none to spare.
        self.batch_staged = os.environ.get("POCKETLLM_QWEN4_BATCH_STAGED", "0").lower() in {
            "1", "true", "yes"
        }
        #: Per pinned layer, its experts stacked once: `(order, gate_up [E, ...], down [E, ...])`.
        self._stack_cache: dict[int, tuple] = {}
        self._position_map: dict[int, torch.Tensor] = {}
        #: Per pinned layer with a widened bank, `(order, [E, 2I, H] bf16, [E, H, I] bf16)`.
        self._wide_keys: dict[int, tuple] = {}
        #: Stacks built from an FP8 bank for a capture: same bytes as the bank, device-side gather.
        self._fp8_stacks: dict[int, tuple] = {}
        #: Expert ids in stack order, and the id -> stack-row table for the gather.
        self._fp8_order: dict[int, list] = {}
        self._fp8_lookup: dict[int, torch.Tensor] = {}
        #: Block geometry kept alongside a stack, since the bank that carried it is dropped.
        self._fp8_blocks: dict[int, tuple] = {}
        #: Whether a graph may be recorded over this backend's step.  Off unless asked for,
        #: because building these stacks costs a layer's worth of pinned memory per layer.
        self._capture_safe = os.environ.get("POCKETLLM_QWEN4_CAPTURE_BANK", "0").lower() in {
            "1", "true", "yes", "on"}
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
        # A pinned layer never reads the host again: no key, no eviction, no counter. It is the
        # cheapest hit there is and it does not compete for cache capacity.
        pinned = self._resident.get(layer_idx)
        if os.environ.get("POCKETLLM_QWEN4_TRACE_PIN"):
            print(f"[pin] fetch layer {layer_idx} expert {expert_id}: bank "
                  f"{'hit' if pinned is not None else 'miss'}, id={id(self)}", flush=True)
        if pinned is not None:
            held = pinned.get(expert_id)
            if held is not None:
                self.cache_hits += 1
                return held
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

    def use_nvfp4(self, bank) -> None:
        """Answer from a whole-device NVFP4 expert bank instead of the host shard."""
        self._nvfp4 = bank

    def pin_layers(self, layer_indices, budget_bytes: int) -> int:
        """Hold as many of `layer_indices` on the device as `budget_bytes` allows.

        Layers are taken in the order given until the next one does not fit, so the set is a
        prefix and a caller can reason about it. Returns how many were pinned.

        Resident experts are held in the checkpoint's own form -- FP8 codes and scales, never a
        bf16 expansion -- because that is what the staging path would have copied anyway; widening
        them here would trade 4.69 MiB for 9.38 MiB a expert and halve how many layers fit.

        A layer is all-or-nothing. Half a layer pinned would leave the other experts of that layer
        staging, which is the case this exists to avoid, and the caller's budget is what decides
        the boundary rather than a per-expert approximation of it.
        """
        expert_ids = list(range(self.num_experts))
        if self.reader.expert_shard is not None:
            owns = self.reader.expert_shard.owns
            expert_ids = [e for e in expert_ids if owns(e)]
        if not expert_ids:
            return 0
        used = 0
        pinned = 0
        for layer_idx in layer_indices:
            # Read the layer's rows on the host and weigh them there. The budget has to decide
            # before anything moves: moving first and weighing after is what made an
            # all-or-nothing budget behave as "move the whole layer anyway and then drop it",
            # which fails with an out-of-memory during the move instead of declining the layer.
            cpu_rows = {}
            layer_bytes = 0
            for expert_id in expert_ids:
                gate_up_cpu, down_cpu = self.reader.expert_rows(layer_idx, expert_id)
                cpu_rows[expert_id] = (gate_up_cpu, down_cpu)
                # Widening doubles the bank, which the comment below explains. Weighing the host
                # copy in its stored form and doubling is the same number the widened device copy
                # takes, without having to allocate it to find out.
                layer_bytes += self._weight_nbytes(gate_up_cpu)
                layer_bytes += self._weight_nbytes(down_cpu)
            if self.wide_bank:
                layer_bytes *= 2
            if used + layer_bytes > budget_bytes:
                # This layer is over the line; it never moved, so there is nothing to drop.
                break
            rows = {}
            for expert_id, (gate_up_cpu, down_cpu) in cpu_rows.items():
                gate_up = gate_up_cpu.to(self.device, non_blocking=False)
                down = down_cpu.to(self.device, non_blocking=False)
                if self.wide_bank:
                    # Widened once here instead of every step: a decode routes to ten experts and
                    # `_wide` on each is the elementwise pass the batched path would otherwise pay
                    # 48 times a step. The cost is the bank itself, 2x the FP8 bytes a layer.
                    gate_up = self._wide(gate_up)
                    down = self._wide(down)
                rows[expert_id] = (gate_up, down)
            del cpu_rows
            self._resident[layer_idx] = rows
            if self.wide_bank:
                # A whole widened bank, stacked once, so a step's routing is an `index_select`
                # with no host read anywhere in it -- which is what a capture needs. This is the
                # expensive shape: 1.17 GiB a layer against 0.59, so it halves the layers that fit.
                order = sorted(rows)
                gate_all = torch.stack([rows[e][0] for e in order])
                down_all = torch.stack([rows[e][1] for e in order])
                self._wide_keys[layer_idx] = (order, gate_all, down_all)
                # The per-expert bank is the same memory the stack just copied, so drop it and
                # keep only the stack for the gather.
                self._resident[layer_idx] = {}
            used += layer_bytes
            pinned += 1
            torch.cuda.synchronize()
        self._resident_layers = pinned
        if os.environ.get("POCKETLLM_QWEN4_TRACE_PIN"):
            print(f"[pin] pinned {pinned} layers, resident dict has {len(self._resident)} "
                  f"entries, id={id(self)}", flush=True)
        return pinned

    def _batched_from_bank(self, layer_idx: int, hidden_states, indices, weights):
        """The routed experts of a pinned layer, batched, or None if this layer is not pinned.

        A pinned layer holds all of its experts in one place, so the batch is an `index_select`
        on the device and the arithmetic is `_batched_experts`. Nothing here reads the routing
        back to the host, which is what the per-expert loop does and what makes it the slow path
        at decode.
        """
        # A wider bank is held only in `_wide_keys`, with `_resident` deliberately emptied, so the
        # test is "is this layer pinned". A layer that is not pinned still takes the batched
        # arithmetic; `_stacked_bank` stages it through the cache first and frees the stack after,
        # so 48 layers do not hold 48 stacks at once.
        if (not self.batch_staged and layer_idx not in self._resident
                and layer_idx not in self._wide_keys
                and layer_idx not in self._fp8_stacks):
            return None
        ids = indices.reshape(-1)
        gate_up, down = self._stacked_bank(layer_idx, ids)
        if gate_up is None:
            return None
        out = _batched_experts(
            hidden_states.reshape(-1).to(self.dtype), gate_up, down, ids, weights.reshape(-1)
        )
        if layer_idx not in self._resident and layer_idx not in self._wide_keys:
            # A staged layer's stack is this step's alone; dropping the reference here lets the
            # allocator reuse the block for the next layer instead of growing the cache by one
            # stack a layer.
            del gate_up, down
        self.cache_hits += int(ids.numel())
        return out.reshape_as(hidden_states)

    def _wide(self, weight):
        """One banked expert projection as a dense bf16 tensor."""
        if isinstance(weight, FP8Tensor):
            return dequantize_block_fp8(weight.code, weight.scale, weight.block, self.dtype)
        return weight if weight.dtype is self.dtype else weight.to(self.dtype)

    def _fp8_block(self, layer_idx: int):
        """The block geometry every expert of a layer shares."""
        if layer_idx in self._fp8_blocks:
            return self._fp8_blocks[layer_idx]
        bank = self._resident.get(layer_idx) or {}
        for weight in bank.values():
            for w in weight:
                if isinstance(w, FP8Tensor):
                    return w.block
        raise RuntimeError(f"layer {layer_idx} has no FP8 expert bank to stack")

    def _from_fp8_stack(self, layer_idx: int, ids: torch.Tensor):
        """Gather this step's experts out of the layer's FP8 stack, or `None` if there is none.

        Built lazily on first use, which is what keeps a run that was never captured from paying
        for it.  The gather and the widen both happen on device, so a graph can record this step;
        the alternative here is a `tolist()` to pick rows out of a Python dict, which cannot.
        """
        stacked = self._fp8_stacks.get(layer_idx) or self._fp8_stack(layer_idx)
        if stacked is None:
            return None
        gate_code, gate_scale, down_code, down_scale = stacked
        order = self._fp8_order[layer_idx]
        flat = ids.reshape(-1).to(torch.long)
        # A routing table can hand out an id this rank does not hold, and a capture makes any
        # out-of-range gather a device-side assert rather than a Python IndexError -- so the
        # lookup is clamped and the "not mine" rows are zeroed after, same as the host path.
        valid = (flat >= 0) & (flat < self._fp8_lookup[layer_idx].numel())
        slots = self._fp8_lookup[layer_idx].index_select(0, torch.where(valid, flat, 0))
        slots = torch.where(valid, slots, torch.full_like(slots, -1))
        empty = slots < 0
        safe = torch.where(empty, torch.zeros_like(slots), slots)
        gate = dequantize_block_fp8(
            gate_code.index_select(0, safe), gate_scale.index_select(0, safe),
            self._fp8_block(layer_idx), self.dtype,
        )
        down = dequantize_block_fp8(
            down_code.index_select(0, safe), down_scale.index_select(0, safe),
            self._fp8_block(layer_idx), self.dtype,
        )
        keep = (~empty).to(gate.dtype).reshape(-1, *([1] * (gate.dim() - 1)))
        return gate * keep, down * keep

    def _fp8_stack(self, layer_idx: int):
        """One layer's FP8 experts stacked `[experts, ...]`, built once and kept.

        `None` when the layer is not a plain FP8 bank, which leaves the caller on the host-read
        path -- a capture then declines rather than recording a step that reads the host.
        """
        cached = self._fp8_stacks.get(layer_idx)
        if cached is not None:
            return cached
        bank = self._resident.get(layer_idx)
        if not bank:
            return None
        order = sorted(bank)
        if not all(isinstance(w, FP8Tensor) for e in order for w in bank[e]):
            return None
        try:
            gate_code = torch.stack([bank[e][0].code for e in order])
            gate_scale = torch.stack([bank[e][0].scale for e in order])
            down_code = torch.stack([bank[e][1].code for e in order])
            down_scale = torch.stack([bank[e][1].scale for e in order])
        except RuntimeError:
            return None
        # The per-expert dict held the same bytes; keeping both would double the layer.  The
        # block geometry goes with it, because the dict was where `_fp8_block` read it from.
        bank = self._resident[layer_idx]
        for weight in bank.values():
            for w in weight:
                if isinstance(w, FP8Tensor):
                    self._fp8_blocks[layer_idx] = w.block
                    break
            else:
                continue
            break
        self._resident[layer_idx] = {}
        self._position_map.pop(layer_idx, None)
        self._stack_cache.pop(layer_idx, None)
        self._fp8_stacks[layer_idx] = (gate_code, gate_scale, down_code, down_scale)
        self._fp8_order[layer_idx] = order
        lookup = torch.full((self.num_experts,), -1, dtype=torch.long, device=self.device)
        lookup[torch.tensor(order, dtype=torch.long, device=self.device)] = torch.arange(
            len(order), dtype=torch.long, device=self.device
        )
        self._fp8_lookup[layer_idx] = lookup
        return self._fp8_stacks[layer_idx]

    def _stacked_bank(self, layer_idx: int, ids: torch.Tensor):
        """The routed experts of a pinned layer as two `[k, ...]` bf16 stacks.

        Gathered per step rather than stacked once: a whole layer is 128 experts and 1.15 GiB in
        bf16, which is what the pinning budget is spent on holding, so keeping a second bf16 copy
        of the layer does not fit. A decode routes to ten of them, and ten `index_select`s plus a
        widen is what the per-expert loop was doing anyway -- the difference is that the arithmetic
        afterwards is one bmm pair instead of ten GEMV triples.
        """
        if self._capture_safe and layer_idx not in self._wide_keys:
            # Served entirely from a device-side stack: no host read, so a capture can record it.
            served = self._from_fp8_stack(layer_idx, ids)
            if served is not None:
                return served
        bank = self._resident.get(layer_idx)
        # A layer that has been stacked is still this rank's resident layer: the dict is empty
        # because the stack holds the same bytes, not because the layer was never pinned.  Left
        # unmentioned here, every stacked layer would fall through to staging after its first
        # step -- which is what turned a working capture into wrong output.
        if not bank and layer_idx not in self._wide_keys and layer_idx not in self._fp8_stacks:
            return self._stage_routed(layer_idx, ids)
        lookup = self._position_map.get(layer_idx)
        if lookup is None:
            order = sorted(bank)
            width = max(max(order) if order else 0, self.num_experts) + 1
            lookup = torch.full((width,), -1, dtype=torch.long, device=self.device)
            lookup[torch.tensor(order, dtype=torch.long, device=self.device)] = torch.arange(
                len(order), dtype=torch.long, device=self.device
            )
            self._position_map[layer_idx] = lookup
            self._stack_cache[layer_idx] = order
        order = self._stack_cache[layer_idx]
        if self.wide_bank and layer_idx in self._wide_keys:
            # The bank is already widened and whole on the device, so the routing is an index and
            # nothing crosses to the host: this is the shape a capture needs.
            order_g, gate_all, down_all = self._wide_keys[layer_idx]
            slots = lookup.index_select(0, ids.reshape(-1).to(torch.long))
            empty = slots < 0
            slots = torch.where(empty, torch.zeros_like(slots), slots)
            keep = (~empty).to(gate_all.dtype).reshape(-1, *([1] * (gate_all.dim() - 1)))
            return gate_all.index_select(0, slots) * keep, down_all.index_select(0, slots) * keep
        slots_host = lookup.index_select(0, ids.reshape(-1).to(torch.long)).tolist()
        # Row `i` of the batch is routing slot `i`, always: a rank that owns fewer than all ten
        # still returns ten rows, zeroed where it owns nothing, because the all-reduce sums the
        # owners and `_batched_experts` multiplies row `i` by weight `i`. Dropping the rows it
        # does not own would shift every later row onto the wrong expert.
        grid, dgrid = [], []
        for pos in slots_host:
            if pos < 0:
                grid.append(None)
                dgrid.append(None)
                continue
            grid.append(self._wide(bank[order[pos]][0]))
            dgrid.append(self._wide(bank[order[pos]][1]))
        donor_g = next((t for t in grid if t is not None), None)
        donor_d = next((t for t in dgrid if t is not None), None)
        if donor_g is None:
            return None, None
        gate = torch.stack([t if t is not None else torch.zeros_like(donor_g) for t in grid])
        down = torch.stack([t if t is not None else torch.zeros_like(donor_d) for t in dgrid])
        return gate.to(self.dtype), down.to(self.dtype)

    def _stage_routed(self, layer_idx: int, ids: torch.Tensor):
        """The routed experts of an unpinned layer, staged into one stack through the cache.

        The cache is what makes a staged layer worth having -- without it every step re-reads the
        same experts over PCIe -- so this goes through `_fetch` and keeps its hit rate.

        One widened row is live at a time rather than ten: widening each expert into its own
        tensor first is what OOM'd when this path was first tried. Widening the whole stack in one
        `dequantize_block_fp8` call was also tried and measured slower (4.44 against 4.72) -- the
        call count fell by half but each call handled a `[k, ...]` grid, and the total time went
        up, so the per-expert widen is what stays.
        """
        slots = ids.reshape(-1).tolist()
        live = [int(s) for s in slots if int(s) < self.num_experts]
        if not live:
            return None, None
        probe_g, probe_d = self._fetch(layer_idx, live[0])
        gate = torch.empty((len(slots), *probe_g.shape), dtype=self.dtype, device=self.device)
        down = torch.empty((len(slots), *probe_d.shape), dtype=self.dtype, device=self.device)
        gate.zero_()
        down.zero_()
        gate[0] = self._wide(probe_g)
        down[0] = self._wide(probe_d)
        del probe_g, probe_d
        for row, slot in enumerate(slots):
            if row == 0:
                continue
            expert_id = int(slot)
            if expert_id >= self.num_experts:
                continue
            g, d = self._fetch(layer_idx, expert_id)
            gate[row] = self._wide(g)
            down[row] = self._wide(d)
            del g, d
        return gate, down

    def _direct_rows(self, layer_idx: int, expert_id: int):
        """One expert's rows from the reader, without the cache.

        `_fetch` exists to keep a staged expert resident across steps; this path widens what it
        gets and keeps only the widened stack, so a cached FP8 pair beside it is memory the card
        does not have.
        """
        gate_up_cpu, down_cpu = self.reader.expert_rows(layer_idx, expert_id)
        gate_up = gate_up_cpu.to(self.device, non_blocking=False)
        down = down_cpu.to(self.device, non_blocking=False)
        self.stage_bytes += self._weight_nbytes(gate_up) + self._weight_nbytes(down)
        self.stage_calls += 1
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
        pinned = self._resident.get(layer_idx)
        if os.environ.get("POCKETLLM_QWEN4_TRACE_PIN") and layer_idx < 2:
            print(f"[pin] prefetch layer {layer_idx}: bank {'hit' if pinned is not None else 'miss'}",
                  flush=True)
        with torch.cuda.stream(stream):
            for expert_id in expert_ids:
                # A pinned layer is already on the device: no copy, no event to wait on. This is
                # the path a decode actually takes -- `_prefetch` is what the sharded call uses,
                # and the `_fetch` shortcut alone left every pinned layer staging anyway.
                if pinned is not None:
                    held = pinned.get(expert_id)
                    if held is not None:
                        self.cache_hits += 1
                        pending.append(_PendingExpert(held[0], held[1], None))
                        continue
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
        if hidden_states.shape[0] == 1 and getattr(self, "_resident", None):
            # One token and this layer is pinned: the batch is on the device, so the step can be
            # a handful of kernels instead of the per-expert loop's fifty.
            batched = self._batched_from_bank(layer_idx, hidden_states, indices, weights)
            if batched is not None:
                return batched
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
        if self._nvfp4 is not None:
            held = self._nvfp4(layer_idx, hidden_states, indices, weights)
            if held is not None:
                return held
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
        #: Run a multi-row call one row at a time. Set by a caller that records the MTP head in a
        #: graph: its expert bank's one-row path is the only one with no host read.
        self.rowwise = False

    def __call__(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        # A speculative verify round feeds K positions in one forward, and above a single token the
        # experts switch from the decode kernel to a grouped prefill kernel. Those are different
        # arithmetic, not different tilings: measured on one cache state, the same row at the same
        # position differs by up to 0.766 in the logits depending only on how many rows travelled
        # with it, so a verify round would judge drafts against an answer the one-row decode chain
        # never gives. Doing the experts a row at a time on the verify path keeps the decode kernel
        # and makes the two halves of a round agree; attention and the rest stay batched, which is
        # where a round's time actually goes.
        if hidden_states.shape[0] > 1 and (
            self.rowwise or os.environ.get("POCKETLLM_QWEN4_MOE_VERIFY_ROWWISE", "0") == "1"
        ):
            local = self.local_ids(indices)
            return torch.cat(
                [
                    self.inner(layer_idx, hidden_states[i : i + 1], local[i : i + 1], weights[i : i + 1])
                    for i in range(hidden_states.shape[0])
                ],
                dim=0,
            )
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


def encode_route(
    indices: torch.Tensor, world_size: int, rank: int, num_experts: int | None = None
) -> torch.Tensor:
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
    if num_experts is not None:
        # `ShardedMoE` masks a foreign route with `num_experts` as its sentinel, and that value is
        # not an id: it is the first one past the table.  `% world_size` would still make it look
        # like this rank's -- 512 % 8 is 0 on rank 0 -- and `// world_size` would then index row
        # 64 of a 64-row table, reading past the end of the expert.  The weight is zero, but zero
        # times garbage is nan in the sum, not zero.
        mine = mine & (flat < num_experts)
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

    def _grouped_prefill(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor | None:
        """The many-token path through the *decode* kernel, one row a slot.

        The fused kernel already walks slots on an axis of its grid, so the only thing that made it
        a decode kernel was that its `x` was one row shared by every slot.  Give each
        (token, expert) pair its own row and the same launch is a grouped GEMM: the routes are the
        slot axis, the arithmetic is per row, and the per-expert Python loop -- which measured at
        42% of a prefill -- never runs.

        The rows come back one a slot, so the fold is a weighted sum over the top-k axis, in the
        same order the loop accumulates in.
        """
        if os.environ.get("POCKETLLM_QWEN4_MOE_SLOTS_PREFILL", "1") != "1":
            return None
        if hidden_states.device.type != "cuda" or self.ops is None:
            return None
        try:
            planes, plane_scale, down = self._table(layer_idx)
        except Exception:
            return None
        if planes is None or down is None:
            return None

        shape = hidden_states.shape
        rows = hidden_states.reshape(-1, shape[-1])
        tokens = rows.shape[0]
        top_k = indices.shape[-1]
        routed = encode_route(indices, self.world_size, self.rank, self.num_experts).reshape(-1)
        if routed.numel() != tokens * top_k:
            return None
        empty = routed < 0
        wts = torch.where(
            empty,
            torch.zeros((), dtype=weights.dtype, device=weights.device),
            weights.to(torch.bfloat16).reshape(-1),
        )
        # A row a slot: the kernel strides `x` by its hidden width when it is handed more than one.
        x = rows.repeat_interleave(top_k, dim=0).reshape(-1)
        slot_rows = self.ops.qwen4_exp_fp8_slots_swiglu(
            x,
            routed,
            wts,
            planes,
            plane_scale,
            planes,
            down.code,
            down.scale,
            self.blocks_per_expert,
        )
        hidden = rows.shape[-1]
        # No weight here: the kernel's down phase already multiplies each slot by `wts`, which is
        # why the decode path can sum its slots as they come back.  Applying it again squares the
        # routing weight and the answer diverges from the first token on.
        folded = slot_rows.view(tokens, top_k, hidden).to(rows.dtype).sum(1)
        return folded.reshape(shape)

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
            grouped = self._grouped_prefill(layer_idx, hidden_states, indices, weights)
            if grouped is not None:
                return grouped
            return self._prefill(layer_idx, hidden_states, indices, weights)
        routed = encode_route(indices, self.world_size, self.rank, self.num_experts)
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


class NvFp4SlotExpertMoE:
    """The rank's experts as NVFP4, one fused launch pair a layer a step.

    Same seam as `Fp8SlotExpertMoE` -- decode only, one token, route as `encode_route`'s code -- but
    the table is 4-bit, so a rank's bank falls from 14.06 GiB to about 7.85 and the room can go to
    something else (the MTP head, which needs a fixed 4.6 GiB and does not fit beside an FP8 bank).

    The format is the one `relic_core`'s `nvfp4_moe_token_forward` reads, and its convention was
    measured rather than assumed:

    * E2M1 codes, two a byte, even K in the low nibble;
    * one **E4M3** scale per 16 weights, holding `block_amax / expert_amax` so it lands in (0, 1];
    * one float32 scale per expert holding `expert_amax / 6`, so the block step comes out
      `block_amax / 6` and the codebook spans its +-6.

    The codebook is chosen against `block_amax / 6`, not against the block amax: normalising by the
    amax would use +-1 of the available +-6 and throw away six times the resolution. Gate and up
    share one per-expert scale because the kernel applies a single `w_scale1[local]` to both
    accumulators, so both are quantized against the larger of the two block maxima.

    Measured against the same weights computed densely, this reconstruction reaches cosine 0.988 and
    relative L2 0.152 at the expert-MLP output, which is what a 4.5-bit codebook costs.
    """

    #: Experts dequantized at a time. A whole layer in float is ~1.7 GiB, which a card holding a
    #: resident NVFP4 bank and long-context caches does not have spare; eight is ~210 MiB.
    _QUANT_CHUNK = 8

    def __init__(self, shard, num_experts: int, device: torch.device, num_layers: int | None = None):
        self.shard = shard
        self.num_experts = int(num_experts)
        self.num_layers = None if num_layers is None else int(num_layers)
        self.world_size = int(getattr(shard, "world_size", 1))
        self.rank = int(getattr(shard, "rank", 0))
        self.device = device
        self.stage_bytes = 0
        self.stage_calls = 0
        self.cache_hits = 0
        self.cache_capacity = 0
        self.phase_stats = None
        from relicllm.runtime.ops import load_ops

        _ops = load_ops()
        if _ops is None or not hasattr(_ops, "nvfp4_moe_token_forward"):
            raise RuntimeError("NvFp4SlotExpertMoE needs nvfp4_moe_token_forward from relic_core")
        self._ops = _ops
        self._tables: dict[int, tuple] = {}
        # Quantize every layer now, not on first use. The long-context arena is allocated before the
        # first token, so a bank that is quantized lazily still has the FP8 bytes resident while the
        # arena asks for its own ordinary GiB -- at 256K the two do not fit (22.66 GiB allocated,
        # out of memory on an 800 MiB request) even though the quantized bank alone would. Doing it
        # here frees each layer's FP8 bytes before that allocation happens.
        if self.num_layers:
            for layer in range(self.num_layers):
                self._table(layer)
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    # -- quantization -----------------------------------------------------

    @staticmethod
    def _quantize(w: torch.Tensor, shared_amax: torch.Tensor | None = None):
        """[E, N, K] float -> packed codes, E4M3 block bytes, per-expert float32 scale."""
        from relic_core.kernels.ops import _quantize_fp4_codes

        block, fp4_max = 16, 6.0
        e, n, k = w.shape
        blocks = w.float().view(e, n, k // block, block)
        ablock = blocks.abs().amax(dim=-1).clamp_min(1e-30)
        amax = ablock.amax(dim=(1, 2)).clamp_min(1e-30)
        if shared_amax is not None:
            amax = torch.maximum(amax, shared_amax.to(amax.dtype))
        w_scale = amax / fp4_max
        byte_val = (ablock / (fp4_max * w_scale.view(e, 1, 1))).clamp_min(1e-30)
        s8 = byte_val.to(torch.float8_e4m3fn)
        sf = s8.to(torch.float32)
        sf = torch.where(sf > 0, sf, torch.full_like(sf, 1e-30))
        step = sf * w_scale.view(e, 1, 1)
        norm = (blocks / step.unsqueeze(-1)).clamp(-fp4_max, fp4_max).reshape(e, n, k)
        codes = _quantize_fp4_codes(norm)
        packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous().to(torch.uint8)
        return packed, s8.view(torch.uint8), w_scale.contiguous(), amax

    @staticmethod
    def _dequant_rows(weight, start: int, stop: int) -> torch.Tensor:
        """`weight[start:stop]` as float, for either a plain tensor or a block-scaled `FP8Tensor`.

        `FP8Tensor` is not subscriptable -- it is a `code`/`scale` pair over a tile -- so an expert
        range is taken from both halves and the scale expanded to the code's last two dims. The
        ratios are derived rather than assumed: the gate/up table's scale grid is not the plain
        128x128 one, and hardcoding a tile is how the first version of this raised
        "size of tensor a (320) must match b (1280)".
        """
        code = getattr(weight, "code", None)
        if code is None:
            return weight[start:stop].float()
        code = code[start:stop]
        scale = weight.scale[start:stop].to(torch.float32)
        lead = code.shape[:-2]
        if scale.shape[:-2] != lead:
            scale = scale.reshape(lead + scale.shape[-2:])
        ry = code.shape[-2] // scale.shape[-2]
        rx = code.shape[-1] // scale.shape[-1]
        expanded = scale.repeat_interleave(ry, dim=-2).repeat_interleave(rx, dim=-1)
        return code.to(torch.float32) * expanded

    def _table(self, layer_idx: int) -> tuple:
        cached = self._tables.get(layer_idx)
        if cached is not None:
            self.cache_hits += 1
            return cached
        gate_up, down = self.shard.layer_tensors(layer_idx)
        if os.environ.get("POCKETLLM_QWEN4_NVFP4_DEBUG", "0") == "1" and not self._tables:
            gu_code = getattr(gate_up, "code", gate_up)
            dn_code = getattr(down, "code", down)
            print(
                f"[nvfp4] layer {layer_idx}: gate_up code {tuple(gu_code.shape)} "
                f"scale {tuple(getattr(gate_up, 'scale', torch.empty(0)).shape)} | "
                f"down code {tuple(dn_code.shape)} "
                f"scale {tuple(getattr(down, 'scale', torch.empty(0)).shape)}",
                flush=True,
            )
        n_local = int(gate_up.shape[0])
        # The down table is `[experts, hidden, inter]`, so its row count is the model width and its
        # column count is the expert intermediate width -- the two are easy to transpose and were,
        # which allocated the packed tables with 2560 and 320 where 640 and 1280 belonged.
        hidden = int(down.shape[1])
        inter = int(down.shape[2])
        if int(gate_up.shape[1]) != 2 * inter:
            raise ValueError(
                f"gate/up table is {tuple(gate_up.shape)}, expected rows of 2*inter={2 * inter}"
            )
        w1q = torch.empty(n_local, inter, hidden // 2, dtype=torch.uint8, device=self.device)
        w1s = torch.empty(n_local, inter, hidden // 16, dtype=torch.uint8, device=self.device)
        w3q = torch.empty_like(w1q)
        w3s = torch.empty_like(w1s)
        w2q = torch.empty(n_local, hidden, inter // 2, dtype=torch.uint8, device=self.device)
        w2s = torch.empty(n_local, hidden, inter // 16, dtype=torch.uint8, device=self.device)
        scale1 = torch.empty(n_local, dtype=torch.float32, device=self.device)
        scale2 = torch.empty(n_local, dtype=torch.float32, device=self.device)
        for start in range(0, n_local, self._QUANT_CHUNK):
            stop = min(start + self._QUANT_CHUNK, n_local)
            gu = self._dequant_rows(gate_up, start, stop).to(self.device)
            gate, up = gu.chunk(2, dim=1)
            dn = self._dequant_rows(down, start, stop).to(self.device)
            # The kernel has one global scale for both projections. Compute their joint range
            # first; choosing the gate's scale and discarding a larger up scale would shrink the
            # reconstructed up projection by gate_amax / up_amax for those experts.
            shared_amax = torch.maximum(gate.abs().amax(dim=(1, 2)), up.abs().amax(dim=(1, 2)))
            q1, s1, t1, _ = self._quantize(gate, shared_amax=shared_amax)
            q3, s3, _t3, _ = self._quantize(up, shared_amax=shared_amax)
            q2, s2, t2, _ = self._quantize(dn)
            w1q[start:stop].copy_(q1); w1s[start:stop].copy_(s1)
            w3q[start:stop].copy_(q3); w3s[start:stop].copy_(s3)
            w2q[start:stop].copy_(q2); w2s[start:stop].copy_(s2)
            scale1[start:stop].copy_(t1); scale2[start:stop].copy_(t2)
        table = (w1q, w1s, w3q, w3s, w2q, w2s, scale1, scale2)
        self._tables[layer_idx] = table
        # Drop the FP8 bytes for this layer now that its NVFP4 table exists. Holding both is not a
        # transient: the resident bank is 14.06 GiB and the quantized one is 7.85, and the first
        # end-to-end run died at 22.94 GiB of 23.52 for exactly that reason. Freeing a layer at a
        # time keeps the peak at the FP8 bank's own size and lets it decline from there, which is
        # the room this whole exercise is trying to find.
        for attr in ("_gate_up", "_down", "_gate_up_scale", "_down_scale"):
            tables = getattr(self.shard, attr, None)
            if isinstance(tables, dict):
                tables.pop(layer_idx, None)
        # `has_layer` is answered from `_loaded`, a separate set, so dropping the tables without
        # touching it leaves the shard claiming bytes it no longer holds. That is not cosmetic:
        # `mtp_expert_rows` asks `has_layer` first and then reads `rows`, so a stale entry sends the
        # MTP head's pin into a popped dict -- the `KeyError: 0` this raised. Clearing it makes the
        # shard answer honestly, and the MTP reader falls through to the checkpoint tensors, which is
        # the right source once the bank is quantized.
        loaded = getattr(self.shard, "_loaded", None)
        if isinstance(loaded, set):
            loaded.discard(layer_idx)
        return table

    # -- forward ----------------------------------------------------------

    def __call__(self, layer_idx: int, hidden_states, indices, weights) -> torch.Tensor:
        shape = hidden_states.shape
        rows = hidden_states.reshape(-1, shape[-1])
        table = self._table(layer_idx)
        self.stage_calls += 1
        exact_rows = getattr(self._ops, "nvfp4_moe_rows_forward_exact_fast", None)
        fast_requested = os.environ.get("POCKETLLM_QWEN4_NVFP4_EXACT_FAST", "0") == "1"
        batch_allowed = rows.shape[0] == 1 or os.environ.get("POCKETLLM_QWEN4_NVFP4_ROWS", "1") == "1"
        compact_requested = os.environ.get("POCKETLLM_QWEN4_NVFP4_COMPACT", "0") == "1"
        store_requested = os.environ.get("POCKETLLM_QWEN4_NVFP4_STORE", "0") == "1"
        if compact_requested and store_requested:
            raise RuntimeError("compact and store NVFP4 modes are mutually exclusive")
        if compact_requested:
            if not fast_requested or not batch_allowed:
                raise RuntimeError("compact NVFP4 requires the exact fast rows path")
            minimum_rows = int(os.environ.get("POCKETLLM_QWEN4_NVFP4_COMPACT_MIN_ROWS", "128"))
            if minimum_rows < 1:
                raise RuntimeError("compact NVFP4 minimum rows must be positive")
            # Compact can win at prefill rows but lose at one-row decode. An explicit threshold
            # leaves the small-row arithmetic and launch exactly as the qualified baseline.
            if rows.shape[0] >= minimum_rows:
                exact_rows = getattr(self._ops, "nvfp4_moe_rows_forward_exact_fast_compact", None)
                if exact_rows is None:
                    raise RuntimeError("the native binary lacks the requested compact NVFP4 entry")
        if store_requested:
            if not fast_requested or not batch_allowed:
                raise RuntimeError("store NVFP4 requires the exact fast rows path")
            exact_rows = getattr(self._ops, "nvfp4_moe_rows_forward_exact_fast_store", None)
            if exact_rows is None:
                raise RuntimeError("the native binary lacks the requested store NVFP4 entry")
        if fast_requested and batch_allowed and exact_rows is not None:
            # The independent GPU oracle must prove identical operand/accumulation bits before
            # enabling this flag. Apply it to decode as well as verify so their baselines agree.
            out = self._rows_forward(exact_rows, table, rows, indices, weights)
        elif rows.shape[0] == 1:
            out = self._one_row(table, rows, indices.reshape(-1), weights.reshape(-1))
        else:
            out = self._prefill(table, rows, indices, weights)
        return out.to(hidden_states.dtype).reshape(shape)

    def _one_row(self, table, row, route_ids, route_weights) -> torch.Tensor:
        w1q, w1s, w3q, w3s, w2q, w2s, s1, s2 = table
        # A foreign route stays negative, and both kernels skip a route whose local index is out of
        # range, so the zeroed-weight dance the FP8 path needs is not needed here.
        idx = encode_route(route_ids, self.world_size, self.rank, self.num_experts)
        return self._ops.nvfp4_moe_token_forward(
            row.reshape(1, -1).to(torch.float32).contiguous(),
            idx.reshape(-1).to(torch.int64).contiguous(),
            route_weights.reshape(-1).to(torch.float32).contiguous(),
            w1q, w1s, w3q, w3s, w2q, w2s,
            0, 0.0, 1.0, 1.0, s1, s2, False,
        )

    def _prefill(self, table, rows, indices, weights) -> torch.Tensor:
        """The many-row path: one launch pair for the whole chunk when the kernel has the K-row form.

        `nvfp4_moe_rows_forward` takes `x [rows, hidden]` and per-row routes, which is exactly what a
        prefill chunk and an MTP verify round both are; a chunk of 512 then costs two launches instead
        of 1024. The row loop below stays as the fallback for a build without that entry, and both
        run the same arithmetic, so which one answered cannot change an expert's result.
        """
        rows_fn = getattr(self._ops, "nvfp4_moe_rows_forward", None)
        if os.environ.get("POCKETLLM_QWEN4_NVFP4_ROWS", "1") != "1":
            # Same arithmetic, one row a launch. Kept reachable because it is the only way to ask
            # whether a difference between a decode step and a K-row step belongs to the expert
            # kernel's batching or to something else in the step.
            rows_fn = None
        if rows_fn is None:
            return self._prefill_loop(table, rows, indices, weights)
        return self._rows_forward(rows_fn, table, rows, indices, weights)

    def _rows_forward(self, rows_fn, table, rows, indices, weights) -> torch.Tensor:
        """Prepare the same route/activation operands for either independently named rows entry."""
        tokens = rows.shape[0]
        top_k = indices.reshape(tokens, -1).shape[1]
        routed = encode_route(indices, self.world_size, self.rank, self.num_experts)
        idx = routed.reshape(tokens, top_k).to(torch.int64)
        route_w = weights.to(torch.float32).reshape(tokens, top_k)
        w1q, w1s, w3q, w3s, w2q, w2s, s1, s2 = table
        return rows_fn(
            rows.to(torch.float32).contiguous(),
            idx.contiguous(),
            route_w.contiguous(),
            w1q, w1s, w3q, w3s, w2q, w2s,
            0, 0.0, 1.0, 1.0, s1, s2, False,
        )

    def _prefill_loop(self, table, rows, indices, weights) -> torch.Tensor:
        """A token at a time through the decode kernel.

        `nvfp4_moe_token_forward` is a single-row kernel by construction -- it checks
        `x.size(0) == 1` -- so unlike the FP8 slot kernel there is no grouped form to hand a prefill
        chunk to, and this loop is the honest option until the kernel takes K rows. That extension is
        the same one the MTP verify round needs (one row is the whole reason that round cannot be
        captured), so this loop is a placeholder with a known replacement rather than a design.

        Correctness is what matters here: it runs the identical arithmetic the decode step runs, so a
        prefill and a decode of the same prompt cannot disagree about which expert computed what.
        """
        tokens = rows.shape[0]
        out = torch.empty(tokens, rows.shape[-1], dtype=torch.float32, device=rows.device)
        for t in range(tokens):
            out[t : t + 1] = self._one_row(table, rows[t : t + 1], indices[t], weights[t])
        return out


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
