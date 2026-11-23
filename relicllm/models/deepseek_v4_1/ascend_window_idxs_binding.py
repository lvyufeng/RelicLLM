"""Bind the fused sliding-window index table into the decode path, reversibly and only on request.

`get_window_topk_idxs` builds a decode step's table with eight launches for 128 int32 values, once a
layer. The fused form builds the same integers in one kernel; bitwise parity is a property of the
expression, since every operand is a bounded integer.

Only the decode case is replaced: prefill needs one row per query, a later chunk reads its window out
of two tensors, and a batch above one is not this case at all. Each of those keeps the original body.
"""
from __future__ import annotations

import threading

import torch

from relic_core.kernels import ascend_window_idxs as _fused

_OWNER = "_relic_window_idxs_owner"
_FALLBACK = "_relic_window_idxs_fallback"


class WindowIdxsBinding:
    """The fused table build installed on `get_window_topk_idxs`, with what it displaced recorded."""

    def __init__(self, module, original):
        self._module = module
        self._original = original
        self._thread = threading.get_ident()
        self.calls = 0
        self.bypasses = 0

    def describe(self):
        return {
            "installed": True,
            "awareness": "relic_core.kernels.ascend_window_idxs",
            "kernels_per_call": 1,
            "displaced_operators_per_call": 8,
            "bit_exact_by_construction": True,
            "calls": self.calls,
            "bypasses_to_the_eager_build": self.bypasses,
            "numerical_or_performance_acceptance_claimed": False,
        }

    def call(self, window_size, bsz, seqlen, pos, device=None):
        # `pos.slot(window_size)` is already the ring position; `pos.row()` is the head. Both are
        # plain integers on the device path, so no host read is needed to build the table.
        slot_reader = getattr(pos, "slot", None)
        row_reader = getattr(pos, "row", None)
        eligible = (
            bsz == 1
            and getattr(pos, "on_device", False)
            and callable(slot_reader)
            and callable(row_reader)
            and device is not None
        )
        if not eligible:
            self.bypasses += 1
            return self._original(window_size, bsz, seqlen, pos, device=device)
        try:
            slot = int(slot_reader(window_size))
            row = int(row_reader())
            self.calls += 1
            return _fused.window_topk_idxs(window_size, slot, row, backend="triton", device=device)
        except Exception:
            self.calls -= 1
            self.bypasses += 1
            return self._original(window_size, bsz, seqlen, pos, device=device)

    def close(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError("the window table binding has to be closed from the thread that installed it")
        if self._module.get_window_topk_idxs is not self._patched:
            raise RuntimeError("the replaced function is no longer the one this binding installed")
        self._module.get_window_topk_idxs = self._original
        for name in (_OWNER, _FALLBACK):
            if hasattr(self._module, name):
                delattr(self._module, name)
        return True


def install():
    """Replace the module-level table builder, keeping the eager body as the fallback."""
    from relicllm.models.deepseek_v4_1 import attention

    if getattr(attention, _OWNER, None) is not None:
        raise RuntimeError("the window table binding is already installed")
    original = attention.get_window_topk_idxs
    owner = WindowIdxsBinding(attention, original)

    def patched(window_size, bsz, seqlen, pos, device=None):
        return owner.call(window_size, bsz, seqlen, pos, device=device)

    owner._patched = patched
    setattr(attention, _FALLBACK, original)
    attention.get_window_topk_idxs = patched
    setattr(attention, _OWNER, owner)
    return owner