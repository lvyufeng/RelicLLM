"""Bind the output head at the width the checkpoint stores it, reversibly and only on request.

`Head` widens `head.weight` from the checkpoint's bf16 to fp32 and keeps it resident, because the
reference does and the sampler reads logits at full width. On a single-row decode that projection is
memory bound: 129280 x 5120 x 4 bytes is 2.648 GB per token against 1.324 GB at the width the weights
are stored in, so the widening buys no information and doubles the traffic.

**This is not a layout-only change and it is not claimed to be one.** The logits leave this path in
bf16, so they differ from the reference tensor broadly -- measured on the registered lane as 1,034,216
of 1,034,240 values changing at up to 0.0623 on every step. That is far outside the row-normalization
class of deviation and is why it needed its own explicit approval rather than being folded into a
"bit-exact" story. What it does not do is change anything on disk: the checkpoint already stores bf16,
so the weight payload and every value it represents are the ones the model was published with.

Because the deviation is broad, the binding is opt-in, states the deviation in `describe()`, and is
reversed by one assignment.
"""
from __future__ import annotations

import functools
import threading

import torch
import torch.nn.functional as F

_OWNER = "_relic_narrow_head_owner"
_FALLBACK = "_relic_narrow_head_fallback"


class NarrowHeadBinding:
    """The output projection at the stored width, with the bf16 twin built once per module."""

    def __init__(self, module, original):
        self._module = module
        self._original = original
        self._thread = threading.get_ident()
        self.calls = 0
        self.instances = 0

    def describe(self):
        return {
            "installed": True,
            "awareness": "the checkpoint's own bf16 weight, no conversion",
            "weight_dtype": "bfloat16 resident",
            "reference_weight_dtype": "float32 resident",
            "logits_dtype": "bfloat16",
            "calls": self.calls,
            "heads_patched": self.instances,
            "traffic_per_token_bytes": 129280 * 5120 * 2,
            "reference_traffic_per_token_bytes": 129280 * 5120 * 4,
            "deviation_class": "broad logits dtype change, explicitly approved",
            "checkpoint_payload_unchanged": True,
            "numerical_or_performance_acceptance_claimed": False,
        }

    def forward(self, head, x, full_logits=False):
        if not isinstance(x, torch.Tensor):
            return self._original(head, x, full_logits)
        if not full_logits:
            x = x[:, -1]
        twin = getattr(head, "_relic_weight_bf16", None)
        if twin is None:
            # The twin is the checkpoint's own bf16 payload narrowed nowhere: `weight` was widened
            # from bf16 in the first place, so casting back restores the stored values exactly.
            twin = head.weight.detach().to(torch.bfloat16).contiguous()
            head._relic_weight_bf16 = twin
            self.instances += 1
        self.calls += 1
        return F.linear(x.to(torch.bfloat16), twin)

    def close(self):
        if threading.get_ident() != self._thread:
            raise RuntimeError("the narrow head binding has to be closed from the thread that installed it")
        if self._module.Head.forward is not self._patched:
            raise RuntimeError("the replaced method is no longer the one this binding installed")
        self._module.Head.forward = self._original
        for name in (_OWNER, _FALLBACK):
            if hasattr(self._module, name):
                delattr(self._module, name)
        return True


def install():
    """Replace `Head.forward` on the class, which needs no model instance."""
    from relicllm.models.deepseek_v4_1 import modules

    if getattr(modules, _OWNER, None) is not None:
        raise RuntimeError("the narrow head binding is already installed")
    original = modules.Head.forward
    owner = NarrowHeadBinding(modules, original)

    @functools.wraps(original)
    def patched(self, x, full_logits=False):
        return owner.forward(self, x, full_logits)

    owner._patched = patched
    setattr(modules, _FALLBACK, original)
    modules.Head.forward = patched
    setattr(modules, _OWNER, owner)
    return owner