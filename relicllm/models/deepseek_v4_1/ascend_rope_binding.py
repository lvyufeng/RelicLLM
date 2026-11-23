"""Bind the fused rotary pair rotation into the decode path, reversibly and only on request.

`apply_rotary_emb_pairs` is the eager body the decode path runs several times a layer. The fused form
performs the same four products and two sums in the same order and writes into the activation's own
storage, so the eager path's copy disappears without the values moving.

The fused path covers the single-position decode table only; prefill and any multi-position table keep
the original body. `relic_core.kernels.ascend_rope_pairs` owns the arithmetic; this module owns the
decision and the undo.
"""
from __future__ import annotations

import threading

import torch

from relic_core.kernels import ascend_rope_pairs as _fused

_OWNER = "_relic_rope_owner"
_FALLBACK = "_relic_rope_fallback"


class RopeBinding:
    """The fused rotation installed on `apply_rotary_emb_pairs`, with what it displaced recorded."""

    def __init__(self, module, original):
        self._module = module
        self._original = original
        self._thread = threading.get_ident()
        self.calls = 0
        self.bypasses = 0

    def describe(self):
        return {
            "installed": True,
            "awareness": "relic_core.kernels.ascend_rope_pairs",
            "kernels_per_call": 1,
            "arithmetic_order_unchanged": True,
            "calls": self.calls,
            "bypasses_to_the_eager_body": self.bypasses,
            "numerical_or_performance_acceptance_claimed": False,
        }

    def call(self, x, freqs, inverse=False):
        eligible = (
            isinstance(x, torch.Tensor)
            and isinstance(freqs, torch.Tensor)
            and x.dim() >= 2
            and x.shape[1] == 1
            and x.shape[-1] % 2 == 0
            and x.dtype in (torch.bfloat16, torch.float16)
        )
        if not eligible:
            self.bypasses += 1
            return self._original(x, freqs, inverse)
        try:
            self.calls += 1
            return _fused.fused_rope_pairs(x, freqs, inverse=inverse, backend="triton")
        except Exception:
            self.calls -= 1
            self.bypasses += 1
            return self._original(x, freqs, inverse)

    def close(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError("the rope binding has to be closed from the thread that installed it")
        if self._module.apply_rotary_emb_pairs is not self._patched:
            raise RuntimeError("the replaced function is no longer the one this binding installed")
        self._module.apply_rotary_emb_pairs = self._original
        for name in (_OWNER, _FALLBACK):
            if hasattr(self._module, name):
                delattr(self._module, name)
        return True


def install():
    """Replace the module-level pair rotation, keeping the eager body as the fallback."""
    from relicllm.models.deepseek_v4_1 import attention

    if getattr(attention, _OWNER, None) is not None:
        raise RuntimeError("the rope binding is already installed")
    original = attention.apply_rotary_emb_pairs
    owner = RopeBinding(attention, original)

    def patched(x, freqs, inverse=False):
        return owner.call(x, freqs, inverse)

    owner._patched = patched
    setattr(attention, _FALLBACK, original)
    attention.apply_rotary_emb_pairs = patched
    setattr(attention, _OWNER, owner)
    return owner