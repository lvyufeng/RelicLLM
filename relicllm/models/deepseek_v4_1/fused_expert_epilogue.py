"""Explicit fused clamp/SwiGLU/routing/FP8 epilogue at the original expert seam.

The GEMVs and their reduction are unchanged by this binding. Only the stage
between w1/w3 and w2 is composed into fewer launches. Prefill and CPU Engram
use the original body. Install before RoutedExpertGraphs so the recorded
bodies contain the fusion rather than a Python-only replacement after capture.
"""
from __future__ import annotations

import torch

_MARKER = "_relic_fused_expert_epilogue_owner"


class FusedExpertEpilogue:
    """Install one reversible module-function replacement, no weight residency."""

    def __init__(self, *, operator=None):
        self.operator = operator
        self.original = self.replacement = None
        self.installed = False
        self.calls = self.passthroughs = 0

    def install(self):
        if self.installed:
            raise RuntimeError("fused expert epilogue already installed")
        from . import ascend_packed_experts as packed
        from relic_core.kernels.device_selected_moe import fused_moe_epilogue

        original = packed.packed_expert_forward
        if getattr(original, _MARKER, None) is not None:
            raise RuntimeError("another fused epilogue owns the expert body")
        operator = fused_moe_epilogue if self.operator is None else self.operator
        if not callable(operator):
            raise TypeError("the epilogue operator must be callable")

        def forward(x, resident, *, swiglu_limit=0.0, weights=None, dtype=None,
                    backend="auto", packed_gemv=None, prefill=None):
            chosen_dtype = x.dtype if dtype is None else dtype
            if (x.device.type != "npu" or x.shape[0] != 1 or prefill is True
                    or chosen_dtype != torch.bfloat16 or backend == "torch" or x.requires_grad
                    or (weights is not None and (not isinstance(weights, torch.Tensor)
                        or weights.dtype not in (torch.bfloat16, torch.float32) or weights.requires_grad))):
                self.passthroughs += 1
                return original(x, resident, swiglu_limit=swiglu_limit, weights=weights,
                                dtype=dtype, backend=backend, packed_gemv=packed_gemv, prefill=prefill)
            if x.ndim != 2 or x.shape[1] != resident.w1.shape[1] or x.dtype != chosen_dtype:
                raise ValueError("fused expert epilogue requires the original BF16 decode shape")
            from .fused_activation_binding import remember_original_decode

            block_fp8_quant, _, _ = packed._core()
            quantize = block_fp8_quant.quantize_block_fp8
            gemv = packed._resolve_packed_gemv(packed_gemv)
            codes, scales = quantize(x.contiguous(), backend=backend)
            gate = packed._project(codes, scales, resident.w1, gemv=gemv, backend=backend).to(chosen_dtype)
            up = packed._project(codes, scales, resident.w3, gemv=gemv, backend=backend).to(chosen_dtype)
            if weights is None:
                routing = torch.ones(1, dtype=torch.float32, device=x.device)
            else:
                if tuple(weights.shape) != (1, 1) or weights.device != x.device:
                    raise ValueError("routing weight must be the original [1,1] device value")
                routing = weights.reshape(1).to(torch.float32)
            decoded, middle_codes, middle_scales = operator(
                gate.contiguous(), routing, up=up.contiguous(), swiglu_limit=swiglu_limit,
                backend="triton", return_codes=True,
            )
            remember_original_decode(middle_codes, middle_scales, decoded)
            result = packed._project(middle_codes, middle_scales, resident.w2,
                                      gemv=gemv, backend=backend).to(chosen_dtype)
            self.calls += 1
            return result

        setattr(forward, _MARKER, self)
        self.original, self.replacement = original, forward
        packed.packed_expert_forward = forward
        self.installed = True
        return self

    def describe(self):
        return {"installed": self.installed, "fused_expert_calls": self.calls,
                "original_passthroughs": self.passthroughs,
                "original_FP8_E8M0_quantization_preserved": True,
                "gemv_reduction_unchanged_by_binding": True, "complete_Engram_CPU": True,
                "weight_cache_bytes": 0}

    def close(self):
        if not self.installed:
            return
        from . import ascend_packed_experts as packed

        if packed.packed_expert_forward is not self.replacement:
            raise RuntimeError("release outer expert graphs before the fused epilogue owner")
        packed.packed_expert_forward = self.original
        self.installed = False
        self.replacement = self.original = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
