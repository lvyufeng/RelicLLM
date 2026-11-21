"""Explicit original-checkpoint MoE experiment with device-side expert selection.

Only routed decode dispatch is replaced. Prefill, complete host Engram,
checkpoint residency, serving guards and the exact default provider are not
changed. The native Cube K accumulation differs; the expert-ID FP32 sum order
and every activation quantization/narrowing boundary remain the original ones.
"""
from __future__ import annotations

import threading

import torch

_OWNER = "_relic_device_selected_moe_owner"


class DeviceRoutedDispatch:
    """Install a no-readback one-row dispatcher with one shared stage workspace."""

    def __init__(self, stores, *, certified_bf16_operands=False, backend="triton",
                 max_workspace_bytes=512 * 2**20, capture_graphs=False, projection_mode="grouped",
                 fused_epilogue=False):
        self.stores = tuple(stores)
        if not self.stores or len({id(store) for store in self.stores}) != len(self.stores):
            raise ValueError("supply distinct, nonempty routed stores")
        if backend not in ("torch", "triton"):
            raise ValueError("backend must be the explicit CPU oracle or Triton Ascend")
        if certified_bf16_operands is not True:
            raise ValueError("certify original source operands before installing native Cube dispatch")
        self.backend = backend
        if projection_mode not in ("grouped", "weight_quant"):
            raise ValueError("only grouped or original weight_quant projections are exposed")
        self.projection_mode = projection_mode
        if not isinstance(fused_epilogue, bool):
            raise ValueError("fused_epilogue must be a bool")
        self.fused_epilogue = fused_epilogue
        self.enqueue_lock = threading.RLock()
        self.execution_stream = None
        if not isinstance(capture_graphs, bool) or (capture_graphs and backend != "triton"):
            raise ValueError("device routed graph capture requires the NPU backend")
        self.capture_graphs = capture_graphs
        self.graph_entries = {}
        self.graph_api = self.graph_pool = None
        self.graph_allocation_start = None
        self.budget = max_workspace_bytes
        self.certified = certified_bf16_operands
        self.replacements = []
        self.workspace = None
        self.calls = self.passthroughs = 0
        self.installed = False
        self.banks = {}

    def install(self):
        if self.installed:
            raise RuntimeError("device routing is already installed")
        from relic_core.kernels.device_selected_moe import make_device_moe_workspace

        first = self.stores[0]
        expected_type = "cpu" if self.backend == "torch" else "npu"
        if first.device.type != expected_type:
            raise ValueError("the explicit dispatch backend must match the store device")
        geometry = (first.dim, first.inter_dim, first.topk, first.device, first.dtype)
        for store in self.stores:
            if (store.dim, store.inter_dim, store.topk, store.device, store.dtype) != geometry:
                raise ValueError("all stores sharing stage scratch must have identical geometry")
            if store.dtype != torch.bfloat16 or store._arenas is None:
                raise ValueError("device routing needs preloaded BF16 original packed stores")
            if tuple(store.owned) != tuple(range(store.rank, store.n_experts, store.world)):
                raise ValueError("the device selector requires the cyclic original expert deal")
            if getattr(store, _OWNER, None) is not None or getattr(store, "_relic_m1_routed_dispatch_owner", None) is not None:
                raise RuntimeError("another dispatcher already owns a routed store")
            for which in ("w1", "w2", "w3"):
                arena = store._arenas[which]
                if arena.codes.ndim != 3 or arena.scales.ndim != 3:
                    raise ValueError("the original row-major packed arenas are required")
            self.banks[id(store)] = {name: (arena.codes, arena.scales)
                                    for name, arena in store._arenas.items()}
        self.workspace = make_device_moe_workspace(first.dim, first.inter_dim, first.topk,
                                                   device=first.device, max_bytes=self.budget,
                                                   projection_mode=self.projection_mode)
        self.execution_stream = None
        if self.capture_graphs:
            from .ascend_graphs import npu_graph_api
            self.graph_api = npu_graph_api()
            self.graph_pool = self.graph_api.graph_pool_handle()
            self.graph_allocation_start = self.graph_api.memory_allocated()
        for store in self.stores:
            original = store.forward
            replacement = self._forward_for(store, original)
            store.forward = replacement
            setattr(store, _OWNER, self)
            self.replacements.append((store, original, replacement))
        self.installed = True
        return self

    def _forward_for(self, store, original):
        def forward(x, weights, indices):
            expected_type = "cpu" if self.backend == "torch" else "npu"
            if x.device.type != expected_type or x.ndim != 2 or x.shape[0] != 1:
                self.passthroughs += 1
                return original(x, weights, indices)
            with self.enqueue_lock:
                if not self.installed:
                    raise RuntimeError("device routing was closed")
                if self.backend == "triton":
                    stream = torch.npu.current_stream(x.device).npu_stream
                    if self.execution_stream is None:
                        self.execution_stream = stream
                    elif stream != self.execution_stream:
                        raise RuntimeError("shared stage workspace requires one serialized execution stream")
                store._check_forward_inputs(x, weights, indices)
                x, weights, indices = x.contiguous(), weights.contiguous(), indices.contiguous()
                if self.capture_graphs:
                    result = self._recorded(store, x, weights, indices)
                else:
                    result = self._execute(store, x, weights, indices)
                self.calls += 1
                return result
        return forward

    def _execute(self, store, x, weights, indices):
        from relic_core.kernels.device_selected_moe import device_selected_fp4_moe

        return device_selected_fp4_moe(
            x, weights, indices, self.banks[id(store)], rank=store.rank, world=store.world,
            n_experts=store.n_experts, swiglu_limit=store.swiglu_limit,
            workspace=self.workspace, certified_bf16_operands=self.certified,
            backend=self.backend, projection_mode=self.projection_mode, fused_epilogue=self.fused_epilogue,
        )

    def _recorded(self, store, x, weights, indices):
        entry = self.graph_entries.get(id(store))
        if entry is None:
            from .ascend_graphs import _record

            staged = tuple(torch.empty_like(tensor) for tensor in (x, weights, indices))
            for destination, source in zip(staged, (x, weights, indices)):
                destination.copy_(source)
            captured = {}
            def body():
                captured["result"] = self._execute(store, *staged)
            graph = _record(body, self.graph_pool, 1, self.graph_api)
            allocated = self.graph_api.memory_allocated() - self.graph_allocation_start
            if allocated > self.budget:
                raise RuntimeError("device routed capture exceeded the explicit graph allocation budget")
            entry = (graph, staged, captured["result"])
            self.graph_entries[id(store)] = entry
        graph, staged, result = entry
        inputs = (x, weights, indices)
        # Check the entire triple before any copy: a late mismatch must not
        # partially overwrite a recorded input or silently narrow routing weights.
        for name, destination, source in zip(("activation", "routing weights", "routing indices"), staged, inputs):
            if (destination.dtype != source.dtype or destination.shape != source.shape
                    or destination.device != source.device):
                raise ValueError(f"device routed graph cannot change captured {name} dtype, shape or device")
        for destination, source in zip(staged, inputs):
            destination.copy_(source)
        graph.replay()
        return result

    def describe(self):
        from relic_core.kernels.device_selected_moe import CONTRACT, WEIGHT_QUANT_CONTRACT
        contract = WEIGHT_QUANT_CONTRACT if self.projection_mode == "weight_quant" else CONTRACT

        return {"installed": self.installed, "stores": len(self.stores),
                "device_routed_calls": self.calls, "passthrough_calls": self.passthroughs,
                "routing_D2H_calls": 0, "workspace_bytes": self.workspace.nbytes if self.workspace else 0,
                "device_routed_graphs": len(self.graph_entries), "projection_mode": self.projection_mode,
                "resident_weight_bank_duplicated": False, "complete_Engram_CPU": True,
                "numerical_contract": contract, "native_K_accumulation_changed": True,
                "fused_epilogue": self.fused_epilogue, "single_execution_stream_required": True,
                "physical_graph_pool_footprint_measured": False,
                "routing_readback_scope": "one-row replaced decode; original passthroughs are not audited",
                "ascending_expert_FP32_sum_unchanged": True,
                "original_FP8_activation_quantization_unchanged": True}

    def close(self):
        with self.enqueue_lock:
            self._close()

    def _close(self):
        if not self.installed:
            return
        for store, _, replacement in self.replacements:
            if store.forward is not replacement or getattr(store, _OWNER, None) is not self:
                raise RuntimeError("another owner changed installed device dispatch")
        if self.backend == "triton":
            torch.npu.synchronize(self.workspace.weight_storage.device)
        for store, original, _ in reversed(self.replacements):
            store.forward = original
            delattr(store, _OWNER)
        self.replacements.clear()
        self.graph_entries.clear()
        self.graph_pool = self.graph_api = None
        self.banks.clear()
        self.workspace = None
        self.installed = False

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
