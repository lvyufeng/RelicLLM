"""Per-architecture kernel choices for the qwen4_exp runtime.

The three cards this tree runs Qwen3.8-Flash-Next on are not the same machine with different clock
speeds, and treating them as one is what makes a choice that is right on one card silently wrong on
another:

* **Turing** (``sm_75``, RTX 2080 Ti) and **Ampere** (``sm_86``, RTX 3090) have no 8-bit matrix
  multiply worth the name. An FP8 weight here is a storage format, and the arithmetic is done after
  expanding it to bf16 -- `quant.dequantize_block_fp8`, then an ordinary GEMM.
* **Ada** (``sm_89``, RTX 4090) has an 8-bit tensor core (``m16n8k16``). An FP8 weight can go to
  `torch._scaled_mm` as codes and stay codes, and the dequantize is a cost that only makes sense
  when something else needs the dense matrix.

The split is made here once, on the capability the card reports, so that a call site asks a
question about its workload instead of guessing from a device string. Two things this module
deliberately does **not** decide:

* **What is fastest.** `supports_fp8_tensor_core` says the card can take an FP8 operand; it does
  not say the FP8 path beats the bf16 one at this shape. On a decode step ``M`` is the token count
  -- 64 on this model -- and a caller that picks by capability alone is choosing on the wrong
  evidence; the measured numbers belong in the call site.
* **What relic-core exports.** A capability predicate answers what the card has. Whether the
  extension built on this host exports the kernel for it is a separate question, and a call site
  still has to check what it loaded.

The predicates come from `relicllm.runtime.device`, which already names the fork relic-core itself
takes; this module only groups the decisions that fork implies.

## Why an sm_89 card still cannot use `_scaled_mm` here

The obvious use of this plan -- route `_expert_matrix` through `torch._scaled_mm` on a card that
reports an FP8 tensor core -- **does not work, and the reason is the weights, not the card.**

`_scaled_mm` accepts a per-tensor scale. This checkpoint carries a **128x128 block grid**, one bf16
scale a tile, and torch refuses that layout outright::

    per-tensor   OK
    block grid   rejects: RuntimeError: Invalid scaling configuration.
    2D scale_b   rejects: RuntimeError: Invalid scaling configuration.

What does reach the tensor core is the fused kernel (`--expert-fused-fp8`, relic-core's
`qwen4_exp_fp8_expert.cu`): it applies the block scale as it widens each code in registers, so the
weight is never materialised in bf16, and it takes `__nv_cvt_fp8x2_to_halfraw2` on Ada and a field
shift elsewhere. That kernel is already what runs on a 4090, and it reaches the decode step
**without `_expert_matrix` in the path** -- which is why this module has no call site yet rather
than an unfinished one.

Measured numbers were reported for this at various points and **none of them should be quoted**:
one pass compared dequantize-then-GEMM against `_scaled_mm` with `torch.ones(1)` as the scale, i.e.
a layout the checkpoint does not have, and reported 7-10%. The layout decides this question, not
the card, and a benchmark that does not carry the block grid is not measuring the decision.

## Why a capability plan and not a file per architecture

A `sm75.py` / `sm89.py` split was considered and rejected for now, because **there is nothing to put
in them yet**. The Python implementations of the FP8 steps -- `quant.dequantize_block_fp8`,
`moe._expert_matrix` -- are one function each, called on all three cards; the per-architecture
difference lives in relic-core's CUDA code, selected by `ops.py`, and this repository has no kernels
by design (`AGENTS.md`). Splitting today would copy the same logic into three files to give each
architecture a home, which is duplication dressed as abstraction, and it would hide the thing that
actually varies: **which strategy a call site picks**, which is a decision made per call site on
this card's capability, not a whole module per card.

**When that stops being true** -- i.e. when this repository should grow `sm75.py` and `sm89.py` --
is when one of these happens:

* an implementation here has to differ *numerically* between architectures, not just in what it
  selects (for example an accumulation that has to be chunked on sm_75 but not on sm_89);
* a card needs a Python-side workaround for a kernel or a torch path that is wrong only there;
* a call site needs so much per-architecture logic that reading it means reading three branches.

Until then, the per-call-site choice is the honest shape: one place to ask the question, and the
call site keeping the measurement that decides it.
"""

from __future__ import annotations

from dataclasses import dataclass

from relicllm.runtime.device import CardCapability, probe_card_capability

__all__ = ["KernelPlan", "plan_for", "fp8_is_native"]


@dataclass(frozen=True, slots=True)
class KernelPlan:
    """Which implementation of each FP8-dependent step this card should take.

    Everything here is a *claim about the card*, not a measurement.
    """

    #: The card this was planned for, so a caller can log what it chose and why.
    capability: CardCapability

    #: Whether an FP8 matrix multiply is a tensor-core operation, i.e. codes can go to the GEMM
    #: without being expanded first.
    fp8_native: bool

    @property
    def explain(self) -> str:
        card = self.capability.name or f"sm_{self.capability.major}{self.capability.minor}"
        return (
            f"{card}: FP8 "
            + ("has a tensor core, so weights may stay as codes" if self.fp8_native
               else "has none, so weights are expanded to bf16 before the GEMM")
        )


def plan_for(capability: CardCapability | None = None) -> KernelPlan:
    """The plan for `capability`, or for the card this process is running on."""
    cap = capability if capability is not None else probe_card_capability()
    return KernelPlan(capability=cap, fp8_native=cap.supports_fp8_tensor_core)


def fp8_is_native(capability: CardCapability | None = None) -> bool:
    """Whether this card multiplies FP8 without expanding it first."""
    return plan_for(capability).fp8_native