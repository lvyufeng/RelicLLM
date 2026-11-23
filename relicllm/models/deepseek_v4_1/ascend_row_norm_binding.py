"""Bind the fused row normalization into the decode path, reversibly and only on request.

`RMSNorm.forward` casts to FP32, squares, takes the mean, adds epsilon, takes rsqrt, scales, applies the
weight and casts back: eight dispatches for a 5120-wide row. The fused path is two kernels from
relic-core with the rsqrt kept in the framework, which is the combination measured to reproduce the
eager chain exactly.

Admission is by shape and dtype, and everything else keeps the original method: this binding never
approximates a case it does not cover. `relic_core.kernels.ascend_row_norm` owns the arithmetic; this
module owns only the decision to take that path and the ability to undo it.
"""
from __future__ import annotations

import threading

import torch

from relic_core.kernels import ascend_row_norm as _fused

_OWNER = "_relic_row_norm_owner"
_FALLBACK = "_relic_row_norm_fallback"


class RowNormBinding:
    """The fused normalization installed on `RMSNorm.forward`, with what it displaced recorded."""

    def __init__(self, module, original):
        self._module = module
        self._original = original
        self._thread = threading.get_ident()
        self.calls = 0
        self.bypasses = 0

    def describe(self):
        return {
            "installed": True,
            "awareness": "relic_core.kernels.ascend_row_norm",
            "kernels_per_call": 2,
            "rstd_provider": "framework",
            "mean_is_reciprocal_multiply": True,
            "calls": self.calls,
            "bypasses_to_the_eager_body": self.bypasses,
            "numerical_or_performance_acceptance_claimed": False,
        }

    def forward(self, rms_norm, x):
        """Take the fused path when it covers the case, otherwise hand back to the original."""
        weight = getattr(rms_norm, "weight", None)
        eligible = (
            isinstance(x, torch.Tensor)
            and x.dim() >= 2
            and weight is not None
            and x.shape[-1] == getattr(rms_norm, "dim", -1)
            and x.dtype in (torch.bfloat16, torch.float16)
            and weight.dtype == torch.float32
            and weight.is_contiguous()
            and weight.numel() == x.shape[-1]
        )
        if not eligible:
            self.bypasses += 1
            return self._original(rms_norm, x)
        try:
            self.calls += 1
            return _fused.fused_row_norm(x, weight, rms_norm.eps, backend="triton")
        except Exception:
            # A device or compiler refusal is exactly the case the original body covers; taking it
            # costs the gain on that call and nothing else.
            self.calls -= 1
            self.bypasses += 1
            return self._original(rms_norm, x)

    def close(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError("the row normalization binding has to be closed from the thread that installed it")
        if self._module.RMSNorm.forward is not self._patched:
            raise RuntimeError("the replaced method is no longer the one this binding installed")
        self._module.RMSNorm.forward = self._original
        for name in (_OWNER, _FALLBACK):
            if hasattr(self._module.RMSNorm, name):
                delattr(self._module.RMSNorm, name)
        return True


def install():
    """Replace `RMSNorm.forward` with the fused dispatch, keeping the original as the fallback."""
    from relicllm.models.deepseek_v4_1 import attention

    if getattr(attention.RMSNorm, _OWNER, None) is not None:
        raise RuntimeError("the row normalization binding is already installed")
    original = attention.RMSNorm.forward
    owner = RowNormBinding(attention, original)

    def patched(self, x):
        return owner.forward(self, x)

    owner._patched = patched
    setattr(attention.RMSNorm, _FALLBACK, original)
    attention.RMSNorm.forward = patched
    setattr(attention.RMSNorm, _OWNER, owner)
    return owner