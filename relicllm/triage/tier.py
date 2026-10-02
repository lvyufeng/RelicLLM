"""Gate 3: turn a static fit and a measurement into one of five answers.

The thresholds are the two the project settled on, and both were chosen because they fall in a gap
that the recorded runs leave rather than because they are round numbers:

* **chat** — ``TTFT@1k <= 15 s`` and ``TPOT <= 250 ms``. TTFT@1k across the live roster is 7.4, 7.6
  and 13.5 seconds, and the next value after those is 137 seconds. Nothing at all was measured
  between 13.5 and 137, so any threshold in that window separates the same two groups.
* **production** — box-level aggregate prefill ``>= 222 tok/s`` at ``input=8192, output=128``. The
  recorded 8k prefill figures are 290 and 170 tok/s, and again nothing sits between them.

**This module does not take a reachability result, and that is the point.** A missing kernel, a
missing loader and a missing backend adapter are all *notes*: an agent can write a kernel, so a gap
in `relic-core` costs work and not feasibility. The only thing that makes a model impossible here is
that it does not fit on any card count this hardware offers. Keeping the parameter out of the
signature is how that rule stays true rather than being restated in a docstring.

Vocabulary follows ``docs/guides/latency_metrics.md``: TTFT, TPOT, ITL and E2EL, summarised as a
mean and as P50/P99. Note that ITL and TPOT coincide only while ``supports_batch`` is false, which
is every runtime in this tree today (`relicllm/backends/capabilities.py:257`).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "DEFAULT_THRESHOLDS",
    "Measured",
    "Thresholds",
    "Tier",
    "Verdict",
    "tier",
]


class Tier(StrEnum):
    """The five answers. Four are conclusions about this hardware; the fifth is an absence of one."""

    PRODUCTION = "production"
    """Meets the box-level throughput SLO as well as the latency one."""

    CHAT = "chat"
    """Meets the single-stream latency SLO; does not reach the throughput one."""

    DEMO = "demo"
    """Produces correct tokens on this hardware and meets neither SLO."""

    IMPOSSIBLE = "impossible"
    """Does not fit at any precision, on max cards plus max host memory."""

    CANDIDATE = "candidate"
    """Passes the free static gates and has no measurement yet -- 候选, not a rank."""


@dataclass(frozen=True)
class Thresholds:
    """The two SLOs, with the measurements they were placed between."""

    ttft_1k_seconds: float = 15.0
    """Time to first token at a 1k-token input. Measured roster: 7.4, 7.6, 13.5, then 137."""

    tpot_seconds: float = 0.250
    """Time per output token, i.e. 4.0 tok/s, which is about reading speed."""

    prefill_8k_tok_per_second: float = 222.0
    """Box-level aggregate prefill at input=8192. Measured: 290 and 170, nothing between."""

    input_len: int = 8192
    """The production input length the throughput SLO is stated at."""

    output_len: int = 128
    """The output length it is stated at, which is also the 128-token rollout the runs used."""

    @property
    def basis(self) -> str:
        """One line a report can print so a threshold never appears without its justification."""
        return (
            f"chat = TTFT@1k <= {self.ttft_1k_seconds:g}s and TPOT <= {self.tpot_seconds * 1000:g}ms; "
            f"production = >= {self.prefill_8k_tok_per_second:g} tok/s aggregate prefill at "
            f"input={self.input_len}, output={self.output_len}"
        )


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class Measured:
    """What a run on this hardware produced. Every field optional, because a partial run is normal.

    A partial measurement is not promoted to a full one by defaulting the missing field: the tier
    function reports which fields it could not see, so a report never says "CHAT" on the strength of
    a TTFT with no TPOT.
    """

    ttft_1k_seconds: float | None = None
    tpot_seconds: float | None = None
    prefill_8k_tok_per_second: float | None = None
    source: str = ""
    """The run this came from, so a tier can be traced back to it."""

    ttft_1k_p99_seconds: float | None = None
    tpot_p99_seconds: float | None = None

    @property
    def missing(self) -> tuple[str, ...]:
        """The fields the tier function needs and this run did not produce."""
        absent = []
        if self.ttft_1k_seconds is None:
            absent.append("ttft_1k_seconds")
        if self.tpot_seconds is None:
            absent.append("tpot_seconds")
        if self.prefill_8k_tok_per_second is None:
            absent.append("prefill_8k_tok_per_second")
        return tuple(absent)


@dataclass(frozen=True)
class Verdict:
    """A tier, the one line that decided it, and what could not be seen."""

    tier: Tier
    reason: str
    decided_by: str
    """The threshold the answer turns on -- a field name, so a report can link to it."""

    unknown: tuple[str, ...] = ()
    """Measurements that were absent. An entry here is why an answer is CANDIDATE or partial."""


def tier(
    *,
    fits: bool,
    measured: Measured | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> Verdict:
    """Decide one of five tiers from a static fit and, sometimes, a measurement.

    ``fits`` is the answer to the only disqualifying question: does it fit, at *some* precision, on
    some card count this hardware offers, counting host memory the way the runtime does. Static
    arithmetic decides ``False`` definitively -- it cannot be wrong in that direction, because
    workspace and fragmentation only ever make things worse -- but its ``True`` is only provisional,
    which is why passing the static gate yields :attr:`Tier.CANDIDATE` and never a rank.

    ``measured`` absent means CANDIDATE, not DEMO: one is "not tested", the other is "tested and
    slow", and collapsing them loses the only actionable thing the tool knows.
    """
    if not fits:
        return Verdict(
            tier=Tier.IMPOSSIBLE,
            reason="does not fit on any precision or card count this hardware offers",
            decided_by="min_cards",
        )

    if measured is None:
        return Verdict(
            tier=Tier.CANDIDATE,
            reason="passes the static gates; no measurement on this hardware yet",
            decided_by="measurement",
            unknown=Measured().missing,
        )

    absent = measured.missing
    if absent:
        return Verdict(
            tier=Tier.CANDIDATE,
            reason=f"the run is incomplete: no {', '.join(absent)}",
            decided_by="measurement",
            unknown=absent,
        )

    if measured.ttft_1k_seconds > thresholds.ttft_1k_seconds:
        return Verdict(
            tier=Tier.DEMO,
            reason=(
                f"TTFT@1k {measured.ttft_1k_seconds:.1f}s exceeds the {thresholds.ttft_1k_seconds:g}s "
                f"chat threshold"
            ),
            decided_by="ttft_1k_seconds",
        )
    if measured.tpot_seconds > thresholds.tpot_seconds:
        return Verdict(
            tier=Tier.DEMO,
            reason=(
                f"TPOT {measured.tpot_seconds * 1000:.0f}ms exceeds the "
                f"{thresholds.tpot_seconds * 1000:g}ms chat threshold"
            ),
            decided_by="tpot_seconds",
        )
    if measured.prefill_8k_tok_per_second < thresholds.prefill_8k_tok_per_second:
        return Verdict(
            tier=Tier.CHAT,
            reason=(
                f"meets the latency SLO; aggregate prefill {measured.prefill_8k_tok_per_second:.0f} "
                f"tok/s is below the {thresholds.prefill_8k_tok_per_second:g} tok/s production one"
            ),
            decided_by="prefill_8k_tok_per_second",
        )
    return Verdict(
        tier=Tier.PRODUCTION,
        reason=(
            f"meets both SLOs: TTFT@1k {measured.ttft_1k_seconds:.1f}s, "
            f"TPOT {measured.tpot_seconds * 1000:.0f}ms, prefill "
            f"{measured.prefill_8k_tok_per_second:.0f} tok/s at 8k"
        ),
        decided_by="prefill_8k_tok_per_second",
    )
