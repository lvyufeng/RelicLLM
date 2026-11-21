"""Explicit tile-resident decode binding; complete host Engram and prefill stay original."""
import torch

from .ascend_device_routing import DeviceRoutedDispatch, _OWNER


class TileRoutedDispatch(DeviceRoutedDispatch):
    """Share bounded tile-ring scratch across layers on one serialized stream.

    The native consumer is supplied explicitly after its codec/capture proof;
    this binding never builds a library, installs a default, or copies a bank.
    """

    def __init__(self, stores, *, consumer=None, certified_bf16_operands=False,
                 backend="triton", capture_graphs=False, max_workspace_bytes=512 * 2**20):
        super().__init__(stores, certified_bf16_operands=certified_bf16_operands,
                         backend=backend, capture_graphs=capture_graphs,
                         max_workspace_bytes=max_workspace_bytes, fused_epilogue=True)
        if backend == "triton" and (consumer is None or not callable(consumer)):
            raise ValueError("supply the explicitly validated native tile consumer")
        self.consumer = consumer
        self.projection_mode = "tile_ring"

    def install(self):
        if self.installed:
            raise RuntimeError("tile routing is already installed")
        from relic_core.kernels.ascendc_tile_moe import make_tile_moe_workspace

        first = self.stores[0]
        expected_type = "cpu" if self.backend == "torch" else "npu"
        geometry = (first.dim, first.inter_dim, first.topk, first.device, first.dtype)
        if first.device.type != expected_type:
            raise ValueError("tile backend must match the original store device")
        for store in self.stores:
            if (store.dim, store.inter_dim, store.topk, store.device, store.dtype) != geometry:
                raise ValueError("shared tile scratch requires identical layer geometry")
            if store.dtype != torch.bfloat16 or store._arenas is None:
                raise ValueError("tile dispatch requires preloaded original packed arenas")
            if tuple(store.owned) != tuple(range(store.rank, store.n_experts, store.world)):
                raise ValueError("tile selection requires the original cyclic expert deal")
            if getattr(store, _OWNER, None) is not None or getattr(store, "_relic_m1_routed_dispatch_owner", None) is not None:
                raise RuntimeError("another dispatcher already owns the original store")
            for which, n, k in (("w1", store.inter_dim, store.dim), ("w3", store.inter_dim, store.dim),
                                ("w2", store.dim, store.inter_dim)):
                arena = store._arenas[which]
                if (arena.codes.shape != (len(store.owned), n // 16, k // 32, 16, 16)
                        or arena.scales.shape != (len(store.owned), n // 16, k // 32, 16)):
                    raise ValueError("tile dispatch requires lossless tile-major banks, not row-major or NZ guesses")
            self.banks[id(store)] = {name: (arena.codes, arena.scales) for name, arena in store._arenas.items()}
        self.workspace = make_tile_moe_workspace(
            first.dim, first.inter_dim, first.topk, device=first.device, max_bytes=self.budget,
            block_dim=self.consumer.block_dim if self.consumer is not None else 24,
            group_k=self.consumer.group_k if self.consumer is not None else 1)
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

    def _execute(self, store, x, weights, indices):
        from relic_core.kernels.ascendc_tile_moe import device_tile_fp4_moe
        return device_tile_fp4_moe(
            x, weights, indices, self.banks[id(store)], rank=store.rank, world=store.world,
            n_experts=store.n_experts, workspace=self.workspace, consumer=self.consumer,
            swiglu_limit=store.swiglu_limit, certified_bf16_operands=self.certified, backend=self.backend)

    def describe(self):
        from relic_core.kernels.ascendc_tile_moe import CONTRACT
        report = super().describe()
        report.update(numerical_contract=CONTRACT, whole_matrix_expanded_weight_scratch_bytes=0,
                      original_tile_bank_duplicated=False,
                      global_expanded_weight_scratch_bytes=(self.workspace.weight_storage.numel()
                                                           * self.workspace.weight_storage.element_size()) if self.workspace else 0,
                      native_consumer=self.consumer.describe() if self.consumer is not None else "explicit CPU oracle",
                      original_FP32_SiLU_unchanged=True)
        return report
