"""Opt-in M1 routed dispatch without copies of the constant row index.

Host expert-ID routing and ascending-expert FP32 summation are unchanged. For
one row, indexing that row through a device tensor only copies identical bits;
the routing weight is a one-column view followed by the same FP32 cast. The
original expert body, prefills and out-of-scope calls remain unchanged. This is a
binding/scheduling change, not a kernel or a new numerical rule.
"""
from __future__ import annotations

import torch

_OWNER = "_relic_m1_routed_dispatch_owner"


def _on_npu(x):
    return isinstance(x, torch.Tensor) and x.device.type == "npu"


class M1RoutedDispatch:
    """Replace each supplied store's forward explicitly, and restore it on close."""

    def __init__(self, stores):
        self.stores = tuple(stores)
        if len({id(store) for store in self.stores}) != len(self.stores):
            raise ValueError("a routed store was supplied more than once")
        self.replacements = []
        self.optimized_calls = self.original_calls = self.contributions = 0
        self.installed = False

    def install(self):
        if self.installed:
            raise RuntimeError("the M1 routed dispatch is already installed")
        for store in self.stores:
            if getattr(store, _OWNER, None) is not None:
                raise RuntimeError("another M1 dispatch owns this routed store")
            for name in ("forward", "_check_forward_inputs", "_routing", "_owns", "_expert_at"):
                if not callable(getattr(store, name, None)):
                    raise TypeError(f"the routed store is missing {name}")
        for store in self.stores:
            original = store.forward
            replacement = self._forward_for(store, original)
            setattr(store, _OWNER, self)
            store.forward = replacement
            self.replacements.append((store, original, replacement))
        self.installed = True
        return self

    def _forward_for(self, store, original):
        def forward(x, weights, indices):
            if not _on_npu(x) or x.ndim != 2 or x.shape[0] != 1 or store._arenas is None:
                self.original_calls += 1
                return original(x, weights, indices)
            store._check_forward_inputs(x, weights, indices)
            buckets = store._routing(indices)
            for rows, slots in buckets.values():
                if rows != [0] or len(slots) != 1 or not isinstance(slots[0], int) or not 0 <= slots[0] < weights.shape[1]:
                    self.original_calls += 1
                    return original(x, weights, indices)
            from . import ascend_packed_experts as packed
            gemv = packed._resolve_packed_gemv(store.packed_gemv)
            value = x.contiguous()
            result = torch.zeros((1, store.dim), dtype=torch.float32, device=store.device)
            for expert in sorted(buckets):
                if not store._owns(expert):
                    continue
                slot = buckets[expert][1][0]
                gate_weights = weights[:, slot:slot+1].to(torch.float32)
                store.lookups += 1
                # Resolve the module seam at the call. An explicitly installed
                # expert graph manager can reuse this same original expert body.
                contribution = packed.packed_expert_forward(
                    value, store._expert_at(expert), swiglu_limit=store.swiglu_limit,
                    weights=gate_weights, dtype=store.dtype, backend=store.backend,
                    packed_gemv=gemv, prefill=False)
                result = result + contribution.to(torch.float32)
                self.contributions += 1
            self.optimized_calls += 1
            return result
        return forward

    def close(self):
        if not self.installed:
            return
        for store, _, replacement in self.replacements:
            if store.forward is not replacement or getattr(store, _OWNER, None) is not self:
                raise RuntimeError("another owner changed an installed M1 routed store")
        for store, original, _ in reversed(self.replacements):
            store.forward = original
            delattr(store, _OWNER)
        self.replacements.clear()
        self.installed = False

    def describe(self):
        return {"installed": self.installed, "stores": len(self.stores),
                "M1_calls": self.optimized_calls, "original_calls": self.original_calls,
                "expert_contributions": self.contributions,
                "host_routing_unchanged": True, "ascending_expert_FP32_sum_unchanged": True,
                "complete_Engram_CPU": True}

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
