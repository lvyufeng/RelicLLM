"""The per-architecture plan: which card takes which FP8 path, and what it does not decide.

`plan_for` is a claim about the card, so these check it against capabilities rather than against
the machine the suite happens to run on — a test that needed an sm_89 card to pass would skip on
every box this repository is developed on.
"""

from __future__ import annotations

import pytest

from relicllm.models.qwen4_exp.kernels import KernelPlan, fp8_is_native, plan_for
from relicllm.runtime.device import CardCapability, probe_card_capability

SM75 = CardCapability(known=True, major=7, minor=5, name="Turing / RTX 2080 Ti")
SM86 = CardCapability(known=False, major=8, minor=6, name="Ampere / RTX 3090")
SM89 = CardCapability(known=True, major=8, minor=9, name="Ada / RTX 4090")


def test_a_card_without_an_fp8_tensor_core_expands_first():
    """Turing's 8-bit multiply is a sequence of m16n8k8 instructions, so an FP8 weight is a storage
    format there and the arithmetic happens on the expanded matrix."""
    plan = plan_for(SM75)
    assert plan.fp8_native is False
    assert "has none" in plan.explain


@pytest.mark.parametrize("capability", [SM86, SM89])
def test_a_card_with_one_may_keep_the_codes(capability):
    plan = plan_for(capability)
    assert plan.fp8_native is True
    assert "has a tensor core" in plan.explain


def test_the_plan_carries_the_card_it_was_made_for():
    """So a caller can log what it chose and why, rather than a bare boolean."""
    plan = plan_for(SM89)
    assert isinstance(plan, KernelPlan)
    assert plan.capability is SM89
    assert "Ada / RTX 4090" in plan.explain


def test_the_live_plan_answers_for_this_process():
    """Whatever card is under the process, the shortcut agrees with the full call."""
    assert fp8_is_native() == plan_for().fp8_native
    assert plan_for().capability == probe_card_capability()


def test_an_unknown_card_is_planned_for_not_refused():
    """`probe_card_capability` reports what it read even for a card the tree was never validated
    on; the plan follows the capability, not the membership in KNOWN_CAPABILITIES."""
    unknown = CardCapability(known=False, major=9, minor=0, name="Hopper / H100")
    assert plan_for(unknown).fp8_native is True
    assert "Hopper" in plan_for(unknown).explain


def test_the_explain_line_falls_back_to_the_capability():
    """A capability handed in without a name still produces a readable line."""
    nameless = CardCapability(known=False, major=8, minor=6)
    assert "sm_86" in plan_for(nameless).explain


def test_capability_is_not_a_claim_about_which_path_is_faster():
    """The docstring's promise, as a test: the plan says an FP8 operand is *allowed* on sm_86 and
    sm_89, and says nothing about the speed of either path at a given shape. Measured on a 4090,
    `_scaled_mm` at the decode shapes is 7-10% faster than dequant-then-GEMM — not the 2x an
    operator reading this boolean as "use FP8" would assume."""
    assert plan_for(SM86).fp8_native is plan_for(SM89).fp8_native
    assert "may" in plan_for(SM89).explain
    assert "should" not in plan_for(SM89).explain