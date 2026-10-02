"""Gate 3 in isolation. Every case here is a pure function of two inputs and needs no hardware.

The thresholds themselves are not asserted as constants -- that would make this test a second copy
of the module. What is asserted is the behaviour around them, including the boundaries, because the
difference between ``<`` and ``<=`` is one token per second that a report would otherwise never
explain.
"""

from __future__ import annotations

import inspect

import pytest

from relicllm.triage.tier import DEFAULT_THRESHOLDS, Measured, Thresholds, Tier, tier


def _measured(**overrides: float | str | None) -> Measured:
    """A run that comfortably meets both SLOs, which each test then moves one field of."""
    fields: dict[str, float | str | None] = {
        "ttft_1k_seconds": 8.0,
        "tpot_seconds": 0.120,
        "prefill_8k_tok_per_second": 300.0,
        "source": "tests/bench_serving.py, this host",
    }
    fields.update(overrides)
    return Measured(**fields)  # type: ignore[arg-type]


def test_a_model_that_does_not_fit_is_impossible_whatever_was_measured() -> None:
    verdict = tier(fits=False, measured=_measured(prefill_8k_tok_per_second=1.0))

    assert verdict.tier is Tier.IMPOSSIBLE
    assert verdict.decided_by == "min_cards"


def test_no_measurement_yields_candidate_not_demo() -> None:
    """"Not tested" and "tested and slow" are different answers and must not collapse."""
    verdict = tier(fits=True, measured=None)

    assert verdict.tier is Tier.CANDIDATE
    assert verdict.unknown, "a CANDIDATE must say what it is waiting for"


def test_an_incomplete_run_stays_candidate_and_names_the_gap() -> None:
    verdict = tier(fits=True, measured=Measured(ttft_1k_seconds=8.0))

    assert verdict.tier is Tier.CANDIDATE
    assert set(verdict.unknown) == {"tpot_seconds", "prefill_8k_tok_per_second"}
    assert "tpot_seconds" in verdict.reason


@pytest.mark.parametrize(
    "field,value",
    [
        ("ttft_1k_seconds", DEFAULT_THRESHOLDS.ttft_1k_seconds + 0.1),
        ("tpot_seconds", DEFAULT_THRESHOLDS.tpot_seconds + 0.001),
    ],
)
def test_either_latency_threshold_being_missed_makes_it_demo(field: str, value: float) -> None:
    verdict = tier(fits=True, measured=_measured(**{field: value}))

    assert verdict.tier is Tier.DEMO
    assert verdict.decided_by == field


def test_meeting_latency_but_missing_throughput_is_chat() -> None:
    verdict = tier(
        fits=True, measured=_measured(prefill_8k_tok_per_second=DEFAULT_THRESHOLDS.prefill_8k_tok_per_second - 1)
    )

    assert verdict.tier is Tier.CHAT
    assert verdict.decided_by == "prefill_8k_tok_per_second"


def test_meeting_both_is_production() -> None:
    verdict = tier(fits=True, measured=_measured())

    assert verdict.tier is Tier.PRODUCTION


@pytest.mark.parametrize(
    "overrides",
    [
        {"ttft_1k_seconds": DEFAULT_THRESHOLDS.ttft_1k_seconds},
        {"tpot_seconds": DEFAULT_THRESHOLDS.tpot_seconds},
        {"prefill_8k_tok_per_second": DEFAULT_THRESHOLDS.prefill_8k_tok_per_second},
    ],
    ids=["ttft-at-the-line", "tpot-at-the-line", "prefill-at-the-line"],
)
def test_the_thresholds_are_inclusive(overrides: dict[str, float]) -> None:
    """A run exactly on the line meets the SLO. The reverse would make the number unquotable."""
    assert tier(fits=True, measured=_measured(**overrides)).tier is not Tier.DEMO


def test_the_tier_function_cannot_see_reachability() -> None:
    """Ruling: a missing adapter, loader or kernel is a note, never a downgrade.

    Asserted on the signature rather than on behaviour, because the failure mode is somebody adding
    the parameter back "for completeness" and a model silently dropping a tier over a kernel that an
    agent could write.
    """
    parameters = set(inspect.signature(tier).parameters)

    assert parameters == {"fits", "measured", "thresholds"}, parameters


def test_the_thresholds_carry_their_own_justification() -> None:
    """A report must be able to print a threshold next to the measurements it was placed between."""
    basis = DEFAULT_THRESHOLDS.basis

    assert "TTFT" in basis and "TPOT" in basis
    assert str(DEFAULT_THRESHOLDS.input_len) in basis


def test_a_custom_threshold_set_moves_only_the_line_it_names() -> None:
    strict = Thresholds(tpot_seconds=0.050)

    assert tier(fits=True, measured=_measured(), thresholds=strict).tier is Tier.DEMO
    assert tier(fits=True, measured=_measured()).tier is Tier.PRODUCTION
