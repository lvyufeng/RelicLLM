"""Explicit original-byte expert TP/EP residency and batch-one NPU dispatch.

Dense/attention/shared-expert TP stays unchanged. EP groups own cyclic expert
identities; each TP member owns an original output slice of W1/W3 and the matching
input slice of W2. Down FP32 values are TP-reduced before BF16 narrowing. Only a
TP leader emits the EP routed sum into the existing global TP8 MoE reduction.
"""
from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

import torch
import torch.distributed as dist

from relic_core.kernels.ascendc_tile_moe import make_tile_moe_workspace
from relic_core.kernels.expert_tp_moe import tp_moe_down_partials, finish_tp_moe, CONTRACT
from .ascend_packed_experts import AscendPackedRoutedExperts, _stored_bytes, packed_expert_bytes
from .loader import scale_key
from .ascend_tile_major_experts import pack_code_tiles, pack_scale_tiles
from .ascend_device_routing import _OWNER


@dataclass
class HybridExpertBank:
    layer_id: int
    dim: int
    inter_dim: int
    topk: int
    n_experts: int
    ep_rank: int
    ep_world: int
    tp_rank: int
    tp_world: int
    device: torch.device
    swiglu_limit: float
    banks: dict
    partial: bool = True

    @property
    def resident_bytes(self):
        return sum(t.numel() * t.element_size() for pair in self.banks.values() for t in pair)

    @property
    def cache_stats(self):
        return {"name": f"original-expert-TP/layer{self.layer_id}", "preloaded": True,
                "retained_bytes": self.resident_bytes, "entries": len(range(self.ep_rank, self.n_experts, self.ep_world)),
                "owned_experts": len(range(self.ep_rank, self.n_experts, self.ep_world)),
                "EP_world": self.ep_world, "EP_rank": self.ep_rank, "TP_world": self.tp_world,
                "TP_rank": self.tp_rank, "local_inter_dim": self.inter_dim, "misses": 0, "evictions": 0}


def shard_original_projection(code, scale, *, projection, tp_rank, tp_world, full_inter_dim):
    """Slice only original code bytes/K32 scales; never recover or requantize."""
    if (isinstance(tp_world, bool) or not isinstance(tp_world, int) or tp_world not in (1, 2, 4, 8)
            or isinstance(tp_rank, bool) or not isinstance(tp_rank, int) or not 0 <= tp_rank < tp_world
            or isinstance(full_inter_dim, bool) or not isinstance(full_inter_dim, int)
            or full_inter_dim <= 0 or full_inter_dim % (32 * tp_world)):
        raise ValueError("expert TP must preserve complete original K32 intermediate blocks")
    code, scale = _stored_bytes(code, "original FP4 codes"), _stored_bytes(scale, "original E8M0 scales")
    width = full_inter_dim // tp_world
    start, stop = tp_rank * width, (tp_rank + 1) * width
    if projection in ("w1", "w3"):
        if code.shape[0] != full_inter_dim or scale.shape != (full_inter_dim, code.shape[1] * 2 // 32):
            raise ValueError("up/gate source does not have the original output-row/K32 layout")
        return code[start:stop].contiguous(), scale[start:stop].contiguous()
    if projection == "w2":
        if code.shape[1] * 2 != full_inter_dim or scale.shape != (code.shape[0], full_inter_dim // 32):
            raise ValueError("down source does not have the original input-column/K32 layout")
        return code[:, start // 2:stop // 2].contiguous(), scale[:, start // 32:stop // 32].contiguous()
    raise ValueError("only the original w1/w3/w2 projection triplet is supported")


class ExpertTPDispatch:
    """Reversible private resharding; original prefills precede installation."""

    def __init__(self, model, *, consumer=None, tp_world=2, backend="triton", rank=None, world=None,
                 reduce_fp32=None, capture_graphs=True, restore_residency=True, reserve_bytes=4 * 2**30,
                 max_workspace_bytes=512 * 2**20):
        if (backend not in ("torch", "triton") or isinstance(tp_world, bool)
                or not isinstance(tp_world, int) or tp_world not in (1, 2, 4, 8)):
            raise ValueError("choose explicit original-byte TP geometry and backend")
        initialized = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if rank is None and initialized else (0 if rank is None else rank)
        self.world = dist.get_world_size() if world is None and initialized else (1 if world is None else world)
        if (not isinstance(self.rank, int) or isinstance(self.rank, bool) or not isinstance(self.world, int)
                or isinstance(self.world, bool) or self.world % tp_world or not 0 <= self.rank < self.world):
            raise ValueError("TP/EP topology must divide the physical world")
        if backend == "triton" and (consumer is None or not initialized):
            raise ValueError("NPU expert TP needs an explicit native consumer and initialized HCCL")
        if initialized and reduce_fp32 is None and (self.rank != dist.get_rank() or self.world != dist.get_world_size()):
            raise ValueError("expert TP topology must match its default process group")
        if capture_graphs and backend != "triton":
            raise ValueError("the CPU logic oracle cannot claim NPU graph capture")
        self.model, self.backend, self.consumer = model, backend, consumer
        self.tp_world, self.tp_rank = tp_world, self.rank % tp_world
        self.ep_world, self.ep_rank = self.world // tp_world, self.rank // tp_world
        if reduce_fp32 is not None and not callable(reduce_fp32):
            raise ValueError("the explicit FP32 TP collective must be callable")
        self.reduce_fp32, self.tp_group = reduce_fp32, None
        if (not isinstance(capture_graphs, bool) or not isinstance(restore_residency, bool)
                or isinstance(reserve_bytes, bool) or not isinstance(reserve_bytes, int) or reserve_bytes < 0):
            raise ValueError("graph/restoration policies and device reserve must be explicit")
        self.capture_graphs, self.restore_residency = capture_graphs, restore_residency
        self.reserve_bytes, self.budget = reserve_bytes, max_workspace_bytes
        self.entries, self.graphs, self.replacements = [], {}, []
        self.workspace = self.graph_api = self.graph_pool = None
        self.graph_allocation_start = None
        self.lock = RLock()
        self.owner_stream = None
        self.installed = self.closed = False
        self.calls = 0
        for block in model.layers:
            store = block.ffn.routed
            if not isinstance(store, AscendPackedRoutedExperts) or store._arenas is None:
                raise ValueError("start from preloaded original packed EP stores, never an expanded weight bank")
            if store.world != self.world or store.rank != self.rank:
                raise ValueError("the original store must describe this physical EP rank")
            if getattr(store, _OWNER, None) is not None or getattr(store, "_relic_m1_routed_dispatch_owner", None) is not None:
                raise RuntimeError("close existing expert dispatch before changing its TP topology")
            if store.inter_dim % (32 * tp_world) or store.dim % 32:
                raise ValueError("TP slices must retain complete original K32 activation/weight grids")
            if store.dtype != torch.bfloat16:
                raise ValueError("expert TP retains the original BF16 projection boundaries")
            self.entries.append((block.ffn, store, store.inter_dim))
        if not self.entries:
            raise ValueError("expert TP requires a nonempty original backbone")
        if len({id(store) for _, store, _ in self.entries}) != len(self.entries) or len({store.layer_id for _, store, _ in self.entries}) != len(self.entries):
            raise ValueError("expert TP requires distinct original stores and layer identities")
        first = self.entries[0][1]
        self.device = first.device
        geometry = (first.dim, first.inter_dim, first.topk, first.n_experts, first.device)
        if any((store.dim, full_mid, store.topk, store.n_experts, store.device) != geometry for _, store, full_mid in self.entries):
            raise ValueError("shared TP workspace requires identical expert layer geometry")
        group_k = consumer.group_k if consumer is not None else 1
        self.workspace = make_tile_moe_workspace(first.dim, first.inter_dim // tp_world, first.topk,
            device=self.device, group_k=group_k, block_dim=consumer.block_dim if consumer is not None else 24,
            max_bytes=max_workspace_bytes)

    def _make_bank(self, store, full_mid):
        local_mid = full_mid // self.tp_world
        owned = tuple(range(self.ep_rank, store.n_experts, self.ep_world))
        count = len(owned)
        required = count * packed_expert_bytes(store.dim, local_mid)
        if self.backend == "triton":
            free, _ = torch.npu.mem_get_info(self.device)
            if int(free) < required + self.reserve_bytes:
                raise RuntimeError("original expert TP shard would consume the reserved device headroom")
        banks = {}
        # Allocate only the new TP shard. The caller has released this layer's
        # old full-expert EP arena; total resident bytes stay approximately equal.
        for name, n, k in (("w1", local_mid, store.dim), ("w3", local_mid, store.dim), ("w2", store.dim, local_mid)):
            banks[name] = (torch.empty((count, n // 16, k // 32, 16, 16), dtype=torch.uint8, device=self.device),
                           torch.empty((count, n // 16, k // 32, 16), dtype=torch.uint8, device=self.device))
            for slot, expert in enumerate(owned):
                key = store._key(expert, name)
                skey = scale_key(key)
                if not store.checkpoint.is_packed_fp4(key) or store.checkpoint.block_size(key) != 32:
                    raise ValueError("expert TP requires original E2M1 and row/K32 E8M0 scales")
                code, scale = shard_original_projection(store.checkpoint.packed(key), store.checkpoint.packed(skey),
                    projection=name, tp_rank=self.tp_rank, tp_world=self.tp_world, full_inter_dim=full_mid)
                if int(scale.min()) < 2 or int(scale.max()) > 252:
                    raise ValueError("the original expert shard is not certified for exact BF16 operands")
                banks[name][0][slot].copy_(pack_code_tiles(code, n, k))
                banks[name][1][slot].copy_(pack_scale_tiles(scale, n, k))
        return HybridExpertBank(store.layer_id, store.dim, local_mid, store.topk, store.n_experts,
            self.ep_rank, self.ep_world, self.tp_rank, self.tp_world, self.device, store.swiglu_limit, banks)

    def install(self):
        with self.lock:
            if self.installed or self.closed:
                raise RuntimeError("use a fresh expert TP owner")
            for ffn, store, _ in self.entries:
                if (ffn.routed is not store or store._arenas is None
                        or getattr(store, _OWNER, None) is not None
                        or getattr(store, "_relic_m1_routed_dispatch_owner", None) is not None):
                    raise RuntimeError("another owner changed original expert residency before TP installation")
            if self.backend == "triton":
                if self.device.type != "npu":
                    raise ValueError("expert TP NPU backend needs NPU resident weights")
                # Every physical rank creates all groups in the same order.
                for ep in range(self.ep_world):
                    group = dist.new_group(list(range(ep * self.tp_world, (ep + 1) * self.tp_world)), backend="hccl")
                    if ep == self.ep_rank:
                        self.tp_group = group
                if self.reduce_fp32 is None:
                    self.reduce_fp32 = lambda value: dist.all_reduce(value, op=dist.ReduceOp.SUM, group=self.tp_group)
                from .ascend_graphs import npu_graph_api
                self.graph_api = npu_graph_api()
                self.graph_pool = self.graph_api.graph_pool_handle()
                self.graph_allocation_start = self.graph_api.memory_allocated()
            elif self.reduce_fp32 is None and self.tp_world > 1:
                raise ValueError("the CPU TP oracle requires an explicit collective fixture")
            changed = []
            try:
                for ffn, original, full_mid in self.entries:
                    original.release()
                    changed.append(original)
                    shard = self._make_bank(original, full_mid)
                    shard.forward = self._forward_for(shard)
                    ffn.routed = shard
                    if self.backend == "triton" and self.rank == 0:
                        print(f"EXPERT_TP_RESIDENT layer={original.layer_id} TP={self.tp_world} EP={self.ep_world} bytes={shard.resident_bytes}", flush=True)
                    self.replacements.append((ffn, original, shard))
                self.installed = True
            except BaseException:
                for ffn, original, shard in reversed(self.replacements):
                    ffn.routed = original
                    shard.banks.clear()
                self.replacements.clear()
                for original in changed:
                    original.preload()
                raise
            return self

    def _execute(self, shard, x, weights, indices):
        partial = tp_moe_down_partials(x, weights, indices, shard.banks,
            ep_rank=self.ep_rank, ep_world=self.ep_world, n_experts=shard.n_experts,
            workspace=self.workspace, consumer=self.consumer, swiglu_limit=shard.swiglu_limit, backend=self.backend)
        if self.reduce_fp32 is not None:
            if self.reduce_fp32(partial) is not None:
                raise RuntimeError("the FP32 TP collective must update the provided partial in place")
        return finish_tp_moe(partial, self.workspace, backend=self.backend, emit=self.tp_rank == 0)

    def _forward_for(self, shard):
        def forward(x, weights, indices):
            with self.lock:
                if not self.installed:
                    raise RuntimeError("expert TP dispatch is not installed")
                if x.shape != (1, shard.dim) or x.dtype != torch.bfloat16 or x.device != self.device:
                    raise ValueError("this explicit TP experiment is decode-only; close and restore residency for prefill")
                if (weights.shape != (1, shard.topk) or indices.shape != (1, shard.topk)
                        or weights.dtype not in (torch.float32, torch.bfloat16)
                        or indices.dtype not in (torch.int32, torch.int64)
                        or weights.device != self.device or indices.device != self.device
                        or any(value.requires_grad for value in (x, weights, indices))):
                    raise ValueError("TP graph inputs must retain original routing extent, dtypes and device")
                if self.backend == "triton":
                    stream = torch.npu.current_stream(self.device).npu_stream
                    if self.owner_stream is None:
                        self.owner_stream = stream
                    elif self.owner_stream != stream:
                        raise RuntimeError("TP shared ring requires one execution-stream owner")
                self.calls += 1
                if not self.capture_graphs:
                    return self._execute(shard, x.contiguous(), weights.contiguous(), indices.contiguous())
                entry = self.graphs.get(shard.layer_id)
                if entry is None:
                    from .ascend_graphs import _record
                    inputs = tuple(torch.empty_like(value) for value in (x, weights, indices))
                    for dest, src in zip(inputs, (x, weights, indices)):
                        dest.copy_(src)
                    captured = {}
                    def body():
                        captured["output"] = self._execute(shard, *inputs)
                    graph = _record(body, self.graph_pool, 1, self.graph_api)
                    allocated = self.graph_api.memory_allocated() - self.graph_allocation_start
                    if allocated > self.budget:
                        raise RuntimeError("expert TP graph capture exceeded its explicit allocation budget")
                    entry = (graph, inputs, captured["output"])
                    self.graphs[shard.layer_id] = entry
                graph, inputs, output = entry
                for dest, src in zip(inputs, (x, weights, indices)):
                    dest.copy_(src)
                graph.replay()
                return output.clone()
        return forward

    def describe(self):
        return {"installed": self.installed, "expert_TP": self.tp_world, "expert_EP": self.ep_world,
                "TP_rank": self.tp_rank, "EP_rank": self.ep_rank, "physical_world": self.world,
                "dense_attention_and_shared_expert_TP_unchanged": True,
                "FP32_down_TP_reduction_before_BF16_narrowing": True,
                "one_TP_leader_emits_into_original_global_MoE_reduce": True,
                "original_E2M1_E8M0_bytes_and_FP8_activation_grids_unchanged": True,
                "numerical_contract": CONTRACT, "decode_only": True, "routing_D2H_calls": 0,
                "resident_bytes": sum(shard.resident_bytes for _, _, shard in self.replacements),
                "shared_workspace_bytes": self.workspace.nbytes if self.workspace is not None else 0,
                "graphs": len(self.graphs), "calls": self.calls,
                "graph_allocator_budget_is_not_a_physical_memory_measurement": True,
                "restore_residency_default": self.restore_residency,
                "per_layer_residency": [shard.cache_stats for _, _, shard in self.replacements]}

    def close(self, *, restore_residency=None):
        with self.lock:
            if not self.installed:
                return
            restore = self.restore_residency if restore_residency is None else restore_residency
            if not isinstance(restore, bool):
                raise ValueError("expert TP restore policy must be an explicit boolean")
            for ffn, _, shard in self.replacements:
                if ffn.routed is not shard:
                    raise RuntimeError("another owner changed expert TP residency")
            errors = []
            if self.backend == "triton":
                try:
                    torch.npu.synchronize(self.device)
                except BaseException as error:
                    errors.append(error)
            self.graphs.clear()
            for ffn, original, shard in reversed(self.replacements):
                ffn.routed = original
                shard.banks.clear()
                if restore and not errors:
                    try:
                        original.preload()
                    except BaseException as error:
                        errors.append(error)
            self.replacements.clear()
            self.graph_pool = self.graph_api = self.workspace = None
            self.installed, self.closed = False, True
            if errors:
                raise errors[0]

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
