"""Opt-in one-collective complete FFN with original-byte TP expert residency.

The global buffer keeps topk independent FP32 expert rows plus the shared row.
Each expert narrows only AFTER the all-rank partial sum. Attention TP is untouched.
The changed global FP32 addition order is an explicit numerical experiment.
"""
from __future__ import annotations

import functools
from threading import RLock
import types

import torch
import torch.distributed as dist

from relic_core.kernels.expert_tp_moe import tp_moe_down_partials
from relic_core.kernels.global_ffn_slots import prepare_global_ffn_slots, finish_global_ffn_slots
from .ascend_expert_tp import ExpertTPDispatch

_OWNER = "_relic_full_FFN_global_slots_owner"


class FullFFNGlobalDispatch:
    def __init__(self, model, *, consumer, tp_world=2, restore_residency=True, max_workspace_bytes=512 * 2**20):
        self.model = model
        self.residency = ExpertTPDispatch(model, consumer=consumer, tp_world=tp_world,
            capture_graphs=False, restore_residency=restore_residency, max_workspace_bytes=max_workspace_bytes)
        self.device = self.residency.device
        self.entries = [(block.ffn, block.ffn.forward, "forward" in block.ffn.__dict__, block.ffn.__dict__.get("forward")) for block in model.layers]
        if any(block.decode_graph is not None or getattr(block.ffn, _OWNER, None) is not None for block in model.layers):
            raise ValueError("install complete-FFN dispatch before recording any block graph")
        if any(block.ffn.tp is None or block.ffn.tp.world != self.residency.world
               or block.ffn.tp.rank != self.residency.rank for block in model.layers):
            raise ValueError("complete FFN global reduction must match the original physical TP plan")
        first = self.residency.entries[0][1]
        self.slots = torch.empty((first.topk + 1, first.dim), dtype=torch.float32, device=self.device)
        if self.slots.numel() * 4 + self.residency.workspace.nbytes > max_workspace_bytes:
            raise ValueError("the global complete-FFN slots exceed the explicit workspace budget")
        self.lock = RLock()
        self.graphs, self.adapted, self.replacements = {}, [], []
        self.installed = self.closed = False
        self.execution_stream = None
        self.calls = 0
        self.graph_budget = max_workspace_bytes
        self.graph_allocation_start = None

    def install(self):
        with self.lock:
            if self.installed or self.closed:
                raise RuntimeError("use a fresh global FFN owner")
            for ffn, original, _, _ in self.entries:
                if ffn.forward != original or getattr(ffn, _OWNER, None) is not None:
                    raise RuntimeError("another owner changed the original FFN before install")
            self.residency.install()
            try:
                for ffn, original, _, _ in self.entries:
                    @functools.wraps(original)
                    @torch.no_grad()
                    def forward(x, image_mask=None, _ffn=ffn):
                        if image_mask is not None or x.numel() != _ffn.dim or x.dtype != torch.bfloat16:
                            raise ValueError("the explicit global-FFN experiment is one-row text decode only")
                        source = x.reshape(1, _ffn.dim).contiguous()
                        weights, ids = _ffn.gate(source)
                        return self.forward_routed_and_shared(_ffn, source, weights, ids).type_as(x).view(x.shape)
                    self.replacements.append((ffn, forward))
                    ffn.forward = forward
                    setattr(ffn, _OWNER, self)
            except BaseException:
                for ffn, _, had, prior in reversed(self.entries):
                    replacement = next((method for block, method in self.replacements if block is ffn), None)
                    if ffn.forward is replacement:
                        if had:
                            ffn.forward = prior
                        else:
                            delattr(ffn, "forward")
                    if getattr(ffn, _OWNER, None) is self:
                        delattr(ffn, _OWNER)
                self.replacements.clear()
                try:
                    self.residency.close(restore_residency=True)
                finally:
                    self.slots = None
                    self.closed = True
                raise
            self.installed = True
            return self

    def _execute(self, ffn, x, weights, ids):
        shard = ffn.routed
        down = tp_moe_down_partials(x, weights, ids, shard.banks,
            ep_rank=self.residency.ep_rank, ep_world=self.residency.ep_world, n_experts=shard.n_experts,
            workspace=self.residency.workspace, consumer=self.residency.consumer,
            swiglu_limit=shard.swiglu_limit, backend="triton")
        shared = ffn.shared_experts(x)
        prepare_global_ffn_slots(down, self.residency.workspace.selected, ids, shared, self.slots)
        dist.all_reduce(self.slots, op=dist.ReduceOp.SUM)
        return finish_global_ffn_slots(self.slots, self.residency.workspace.route_error)

    @torch.no_grad()
    def forward_routed_and_shared(self, ffn, x, weights, ids):
        with self.lock:
            if not self.installed or getattr(ffn, _OWNER, None) is not self:
                raise RuntimeError("complete FFN owner is not installed")
            if (x.shape != (1, ffn.dim) or x.dtype != torch.bfloat16 or x.device != self.device
                    or weights.shape != (1, ffn.n_activated_experts) or ids.shape != weights.shape
                    or weights.dtype != torch.float32 or ids.dtype not in (torch.int32, torch.int64)
                    or any(value.requires_grad for value in (x, weights, ids))
                    or any(value.device != self.device for value in (weights, ids))):
                raise ValueError("complete FFN graph routing/input metadata changed")
            x, weights, ids = (value.contiguous() for value in (x, weights, ids))
            stream = torch.npu.current_stream(self.device).npu_stream
            if self.execution_stream is None:
                self.execution_stream = stream
            elif self.execution_stream != stream:
                raise RuntimeError("complete FFN scratch requires one execution-stream owner")
            entry = self.graphs.get(ffn.layer_id)
            if entry is None:
                from .ascend_graphs import _record, npu_graph_api
                api = npu_graph_api()
                if self.graph_allocation_start is None:
                    self.graph_allocation_start = api.memory_allocated()
                staging = tuple(torch.empty_like(value) for value in (x, weights, ids))
                for a, b in zip(staging, (x, weights, ids)):
                    a.copy_(b)
                result = {}
                def body():
                    result['out'] = self._execute(ffn, *staging)
                pool = api.graph_pool_handle()
                graph = _record(body, pool, 1, api)
                if api.memory_allocated() - self.graph_allocation_start > self.graph_budget:
                    raise RuntimeError("complete FFN graph capture exceeded the explicit allocation budget")
                entry = (graph, staging, result['out'], pool)
                self.graphs[ffn.layer_id] = entry
            graph, staging, output, _ = entry
            if any(a.dtype != b.dtype for a,b in zip(staging,(x,weights,ids))):
                raise ValueError("complete FFN graph cannot change its captured routing dtype")
            for a, b in zip(staging,(x,weights,ids)):
                a.copy_(b)
            graph.replay(); self.calls += 1
            return output.clone()

    def adapt_decode_graphs(self, graph_driver):
        """Before capture, replace routed+B bodies with full FFN+HC-only B."""
        from . import graphs
        if not self.installed or self.adapted:
            raise RuntimeError("adapt each decode driver once while the owner is installed")
        for layer in graph_driver.layers:
            if layer.ready or getattr(layer.block.ffn, _OWNER, None) is not self:
                raise ValueError("global FFN graph adaptation must precede capture")
        for layer in graph_driver.layers:
            original_routed, original_b = layer._routed, layer._body_b
            had_routed, prior_routed = "_routed" in layer.__dict__, layer.__dict__.get("_routed")
            had_b, prior_b = "_body_b" in layer.__dict__, layer.__dict__.get("_body_b")
            def routed(this):
                sink, ffn = this.sink, this.block.ffn
                graphs._put(sink, "y_in", self.forward_routed_and_shared(ffn, sink.ffn_in.view(1, ffn.dim), sink.weights, sink.indices))
            def body_b(this):
                sink, block = this.sink, this.block
                y = sink.y_in.type_as(sink.ffn_in).view(sink.ffn_in.shape)
                graphs._put(sink, "x_out", block.hc_post(y, sink.residual, sink.ffn_post, sink.ffn_comb))
            new_routed, new_b = types.MethodType(routed, layer), types.MethodType(body_b, layer)
            layer._routed, layer._body_b = new_routed, new_b
            self.adapted.append((layer, original_routed, original_b, new_routed, new_b,
                                 had_routed, prior_routed, had_b, prior_b))

    def describe(self):
        placement = self.residency.describe()
        for key in ("FP32_down_TP_reduction_before_BF16_narrowing",
                    "one_TP_leader_emits_into_original_global_MoE_reduce", "numerical_contract", "graphs", "calls"):
            placement.pop(key, None)
        placement["placement_metadata_only_not_the_TP_dispatch_execution_path"] = True
        return {"installed":self.installed,"one_global_FFN_collective_per_layer":True,
                "numerical_contract":"original_byte_FP32_expert_slots_global_reduce_then_per_expert_BF16",
                "independent_expert_FP32_slots_before_BF16_narrowing":True,
                "shared_partial_in_same_global_buffer":True,"global_buffer_bytes":self.slots.numel()*4 if self.slots is not None else 0,
                "ordered_expert_combination_after_collective":True,"original_quantization_unchanged":True,
                "changed_global_FP32_addition_order_is_explicit_experiment":True,
                "graph_allocation_budget_bytes":self.graph_budget,
                "graph_allocation_budget_not_physical_memory_measurement":True,
                "decode_only":True,"routing_D2H_calls":0,
                "graphs":len(self.graphs),"calls":self.calls,"residency":placement}

    def close(self, *, restore_residency=None):
        with self.lock:
            if restore_residency is not None and not isinstance(restore_residency, bool):
                raise ValueError("complete FFN restore policy must be an explicit boolean")
            if self.closed:
                return
            if not self.installed:
                self.residency.close(restore_residency=restore_residency)
                self.slots = None
                self.closed = True
                return
            for ffn, replacement in self.replacements:
                if getattr(ffn, _OWNER, None) is not self or ffn.forward is not replacement:
                    raise RuntimeError("another owner changed complete FFN dispatch")
            for ffn, _, shard in self.residency.replacements:
                if ffn.routed is not shard:
                    raise RuntimeError("another owner changed complete FFN expert residency")
            for layer, _, _, new_routed, new_b, *_ in self.adapted:
                if layer._routed != new_routed or layer._body_b != new_b:
                    raise RuntimeError("another owner changed adapted full FFN graph methods")
                if (getattr(layer.block, "decode_graph", None) is layer
                        or any(getattr(layer, name, None) is not None for name in
                               ("graph_a", "graph_a_emit", "graph_a_fill", "graph_b"))):
                    raise RuntimeError("release adapted decode graphs before closing complete FFN residency")
            errors = []
            try:
                torch.npu.synchronize(self.device)
            except BaseException as error:
                errors.append(error)
            self.graphs.clear()
            for layer, _, _, _, _, had_routed, prior_routed, had_b, prior_b in self.adapted:
                for name, had, prior in (("_routed", had_routed, prior_routed), ("_body_b", had_b, prior_b)):
                    if had:
                        setattr(layer, name, prior)
                    else:
                        delattr(layer, name)
            self.adapted.clear()
            for ffn, _, had, previous in reversed(self.entries):
                if had:
                    ffn.forward = previous
                else:
                    delattr(ffn, "forward")
                delattr(ffn, _OWNER)
            self.replacements.clear()
            try:
                self.residency.close(restore_residency=False if errors else restore_residency)
            except BaseException as error:
                errors.append(error)
            finally:
                self.slots = None
                self.graph_allocation_start = None
                self.execution_stream = None
                self.installed, self.closed = False, True
            if errors:
                raise errors[0]

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
