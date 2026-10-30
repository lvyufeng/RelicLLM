#!/usr/bin/env python3
"""Diff a benchmark run against the recorded performance baseline, per metric.

``relicllm bench serve`` (P0.1) answers "how fast is this, on this box, for this scenario" and writes
the answer as a JSON record. This script is the other half: it holds the answer that was recorded,
and it says which metric moved, in which direction, by how much -- or that nothing did.

The design is deliberately the test baseline's (`scripts/check_test_baseline.py`), with one
substitution. That script's unit is a **node id** because the failure set is a set: three tests fixed
and one broken is a net improvement in a count and a regression in the tree. This script's unit
cannot be a set, because a metric is a *number* and every good number moves a little every run; the
unit is a **key** of the runtime, the card, the scenario and the metric, and the comparison is a ratio
against a threshold.

Four properties that are the reason this is a script and not a `diff`:

- **A threshold, not equality.** Two identical runs do not produce identical numbers -- the v41 record
  in ``tests/fixtures/perf/`` has a 4.8x spread on the eager path and 1.1x on the graphed one. A
  gate that asked for equality would be red on every run and would be turned off within a week.
- **A ratio, and both directions.** Faster is not the only direction that matters: a change that
  halves TPOT and doubles E2EL is a regression, and one that quietly improves throughput by dropping
  tokens is not an improvement. Both ends are reported, on the same threshold.
- **``skip`` without a GPU, not ``fail``.** The suite is run by hand on a box that may not be this
  one; a record that cannot be compared (no committed baseline, a runtime the baseline does not
  cover) is reported and not enforced. The same rule the P0.1 runner uses: absent is absent, and
  never a zero standing in for it.
- **The card is in the key.** The record's ``cuda`` block names the device it was measured on, and the
  key is ``runtime@card/scenario/metric``. Without the card, an 8x RTX 4090 reading and a 4x RTX 2080
  Ti reading are both ``v41/decode/tpot.median``, and the gate reported the first as a 2.6x
  regression against the second. Absence is still absence: a record with no ``cuda`` block, and a
  baseline entry with no card, keep the unqualified key and the comparison they have always had.

Usage::

    python scripts/check_perf_baseline.py                        # diff tests/fixtures/perf/baseline.json
    python scripts/check_perf_baseline.py --observed run.json    # diff a specific run
    python scripts/check_perf_baseline.py --update --observed run.json   # record it as the baseline
    python scripts/check_perf_baseline.py --threshold 0.10       # 10% either way (the default is 15%)

One record per runtime **and per card**, all in the one ``--update``: a second card's numbers are keys
of their own, so the two sets live side by side in one file rather than replacing each other.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "tests" / "fixtures" / "perf" / "baseline.json"

#: How far a metric may move before the gate calls it a change, either direction.
#:
#: 15% is a starting point argued from the records rather than from taste. The two v41 records in
#: ``tests/fixtures/perf/`` are the same binary a flag apart and their *stable* metrics (TPOT on the
#: graphed path, output throughput, E2EL) agree far inside this, while the eager path's TPOT spread
#: across eight requests of one prompt is 4.8x -- which no threshold distinguishes from a regression,
#: and which is exactly why the eager configuration is not what the baseline records. A metric whose
#: own run-to-run spread is wider than the threshold does not belong in a baseline; the threshold is
#: not the lever for that, removing the metric is.
DEFAULT_THRESHOLD = 0.15

#: The metrics the gate compares, and which direction is an improvement.
#:
#: Explicit rather than "every number in ``metrics``": the record also carries ``duration_seconds``,
#: ``max_concurrent_requests``, per-request ``output_lens`` and a per-run ``std``, and a gate that
#: compared those would report a change every time the scheduler happened to interleave differently.
#: ``-1`` means smaller is better, ``+1`` larger.
_METRICS: dict[str, int] = {
    # Latency: a call that answers sooner is the whole point.
    "ttft.median": -1,
    "tpot.median": -1,
    "itl.median": -1,
    "e2el.median": -1,
    # Throughput: more tokens a second, holding the latency above.
    "output_throughput": +1,
}

#: Nothing here is compared by ratio against a baseline of exactly zero, and the reason is not
#: numerical: ``after / 0`` is a claim about a run that served nothing, not a measurement of how the
#: second run differs from the first. A zero is reported in the ``unenforced`` list rather than
#: divided by, which is also why there is no epsilon here -- a magic constant that made some small
#: numbers comparable and others not would be a threshold hidden inside the threshold.
_ZERO = 0.0


def _dig(record: Mapping[str, Any], dotted: str) -> float | None:
    """Read a dotted key out of a metric block, or ``None`` when the run did not produce it.

    ``None`` and ``0.0`` are different answers here and the distinction is load-bearing: a metric the
    run did not report is one this gate has nothing to say about, while a metric reported as zero is
    a claim the server served nothing, which is a regression of the first order.
    """
    value: Any = record
    for part in dotted.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    return None if math.isnan(number) or math.isinf(number) else number


def observations(record: Mapping[str, Any]) -> dict[str, float]:
    """One run as ``runtime[@card]/scenario/metric -> value``.

    Keyed by everything needed to know what the number is about. The runtime comes from
    ``launch.argv`` rather than from a filename, because the record is the thing being compared and a
    moved or renamed file must not change what it says; the card comes from the record's ``cuda``
    block, for the same reason and one more -- a latency is a property of the box as much as of the
    code, so a number measured elsewhere is not a number this one moved from.
    """
    token = _token_of_record(record)
    scenarios = record.get("scenarios")
    if not isinstance(scenarios, Mapping):
        return {}
    found: dict[str, float] = {}
    for scenario, block in scenarios.items():
        if not isinstance(block, Mapping):
            continue
        metrics = block.get("metrics")
        if not isinstance(metrics, Mapping):
            continue
        for metric in _METRICS:
            value = _dig(metrics, metric)
            if value is not None:
                found[f"{token}/{scenario}/{metric}"] = value
    return found


def _runtime_of(record: Mapping[str, Any]) -> str:
    """The runtime a record measured, from the launch argv it recorded.

    ``--backend`` is read where ``relicllm serve`` reads it, including the ``--backend=name``
    spelling. A record with no ``--backend`` (a run against an already-running server) is keyed
    ``unknown`` rather than guessed at -- see the module docstring on absent versus invented.
    """
    launch = record.get("launch")
    argv = list(launch.get("argv") or []) if isinstance(launch, Mapping) else []
    for index, token in enumerate(argv):
        if token == "--backend" and index + 1 < len(argv):
            return str(argv[index + 1])
        if token.startswith("--backend="):
            return token.split("=", 1)[1]
    return "unknown"


#: What separates a runtime from the card its numbers were measured on, inside a key's first field.
_CARD_SEPARATOR = "@"


def _card_of(record: Mapping[str, Any]) -> str | None:
    """The card a record was measured on, from the envelope's ``cuda.devices``, or ``None``.

    The device names as the probe spelled them, joined when a box is not uniform. Absent is absent:
    a record with no ``cuda`` block -- the probe is best-effort and writes nothing when neither
    ``torch`` nor ``nvidia-smi`` answers (``relicllm/bench/metadata.py``) -- names no card, and its
    keys keep the unqualified form they have always had rather than gaining a made-up one.
    """
    cuda = record.get("cuda")
    if not isinstance(cuda, Mapping):
        return None
    devices = cuda.get("devices")
    if not isinstance(devices, (list, tuple)):
        return None
    names = [str(name).strip() for name in devices if str(name).strip()]
    unique = list(dict.fromkeys(names))
    return " + ".join(unique) if unique else None


def _token_of_record(record: Mapping[str, Any]) -> str:
    """``runtime`` or ``runtime@card``: what a record's keys are headed by."""
    card = _card_of(record)
    runtime = _runtime_of(record)
    return f"{runtime}{_CARD_SEPARATOR}{card}" if card else runtime


def _split_token(token: str) -> tuple[str, str | None]:
    """A key's first field as ``(runtime, card)``; the card is ``None`` when none was recorded."""
    runtime, separator, card = token.partition(_CARD_SEPARATOR)
    return runtime, (card if separator else None)


def _card_conflict(one: str | None, other: str | None) -> bool:
    """Whether two cards are *known* to be different.

    Absence is not a difference: a baseline recorded before the card was read, or a record from a box
    with neither torch nor nvidia-smi, has no card to disagree with, and forcing them apart would be
    the suite gaining a failure from a missing field. Two named cards that are not the same are the
    one thing this refuses -- TPOT 1.09 s on a 4090 and 0.41 s on a 2080 Ti are both
    ``v41/decode/tpot.median``, and the second is not a baseline the first moved from.
    """
    return bool(one) and bool(other) and one != other


def _key_parts(key: str) -> tuple[str, str, str] | None:
    """A metric key as ``(token, scenario, metric)``, or ``None`` when it is not one.

    A device name carries no slash, so a card in the token does not change this split.
    """
    parts = key.split("/")
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _token_of_key(key: str) -> str:
    """The runtime token a metric key is headed by, card and all."""
    return key.split("/", 1)[0]


@dataclass(frozen=True)
class Delta:
    """One compared metric: what it was, what it is, and how far it moved."""

    key: str
    before: float
    after: float

    @property
    def ratio(self) -> float:
        """``after / before``: 1.0 is unchanged, 0.5 is half, 2.0 is double."""
        if self.before == 0.0:
            return math.inf if self.after else 1.0
        return self.after / self.before

    def moved(self, threshold: float) -> bool:
        return abs(self.ratio - 1.0) > threshold

    def worse(self, threshold: float) -> bool:
        """Whether the movement is in the direction the metric does not want.

        The direction is per metric, and the same ratio can be one or the other: E2EL rising is a
        regression, output throughput rising is not.
        """
        return (self.ratio - 1.0) * _METRICS[self.key.rsplit("/", 1)[-1]] < -threshold


@dataclass(frozen=True)
class Report:
    regressions: tuple[Delta, ...]
    improvements: tuple[Delta, ...]
    unchanged: tuple[Delta, ...]
    missing: tuple[str, ...]
    unenforced: tuple[Delta, ...]

    @property
    def ok(self) -> bool:
        return not self.regressions


def compare(
    baseline: Mapping[str, float],
    observed: Mapping[str, float],
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> Report:
    """Every key the two runs share, split three ways, plus the keys only one of them has.

    A key in the baseline that the run did not produce is **not** a pass. It is reported in
    ``missing`` and does not fail on its own -- a runtime skipped for want of a checkpoint is the
    ordinary case -- but it is printed, because "the gate is green" and "the gate compared nothing"
    must not look the same. The reverse (a key the run produced that no baseline holds) is ignored:
    a new metric, or a new scenario, is not a regression and is recorded by the next ``--update``.

    Two keys meet on their text, with one exception: a baseline entry that names no card is answered
    by the run's key for the same runtime, scenario and metric whatever card *it* names (see
    ``_comparable_key``). Two keys that name different cards never meet, which is the whole point of
    the card being in the key -- see the module docstring.
    """
    regressions: list[Delta] = []
    improvements: list[Delta] = []
    unchanged: list[Delta] = []
    unenforced: list[Delta] = []
    missing: list[str] = []

    for key in sorted(baseline):
        answered = key if key in observed else _comparable_key(key, observed)
        if answered is None:
            missing.append(key)
            continue
        delta = Delta(key=key, before=baseline[key], after=observed[answered])
        if delta.before == _ZERO:
            unenforced.append(delta)
        elif not delta.moved(threshold):
            unchanged.append(delta)
        elif delta.worse(threshold):
            regressions.append(delta)
        else:
            improvements.append(delta)
    return Report(
        regressions=tuple(regressions),
        improvements=tuple(improvements),
        unchanged=tuple(unchanged),
        missing=tuple(missing),
        unenforced=tuple(unenforced),
    )


def _comparable_key(key: str, observed: Mapping[str, float]) -> str | None:
    """The observed key that answers a baseline key the two runs do not spell identically.

    Only the card can differ here, and only by one side not naming one: two named cards have already
    been kept apart by the key, and a record that was measured on the same card always spells the key
    the same way. This is the tolerance for a baseline recorded before the card was read, or for a
    record from a box that could not read it -- those keep the comparison they have always had.

    Several observed cards for one card-less baseline key is not a comparison this script can make:
    which of them is the number the entry recorded? It is left to be reported as missing, which is a
    statement about what could not be compared rather than a guess.
    """
    wanted = _key_parts(key)
    if wanted is None:
        return None
    token, scenario, metric = wanted
    runtime, card = _split_token(token)
    candidates: list[str] = []
    for other in observed:
        parts = _key_parts(other)
        if parts is None:
            continue
        other_runtime, other_card = _split_token(parts[0])
        if (
            other_runtime == runtime
            and parts[1] == scenario
            and parts[2] == metric
            and not _card_conflict(card, other_card)
        ):
            candidates.append(other)
    return candidates[0] if len(candidates) == 1 else None


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_from(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """The baseline a set of runs defines: every key any of them produced.

    A key two runs disagree on is taken from the last one, which is stated here because the
    alternative -- silently averaging them -- would hide exactly the spread a threshold has to be
    chosen against. In practice a baseline is built from one run per runtime, which is why the
    caller takes several records: `--observed` is repeatable.
    """
    merged: dict[str, float] = {}
    for record in records:
        merged.update(observations(record))
    return merged


#: The launch shape a run's numbers are a claim about, read off its recorded argv.
#:
#: Kept beside the value because a latency is not comparable without it. This runtime set does not
#: share one width: `xing4` serves one card one process at a time -- `factory` has no supervised
#: branch for it and its adapter never joins a group -- while the rest run TP4 here. Two numbers for
#: one metric from two widths are not two measurements of one thing, and a reader who cannot tell a
#: 4-card aggregate from a 1-card reading has no way to know which they are looking at.
_SHAPE_FLAGS = ("--tensor-parallel-size", "--max-model-len")


def _shape_of(record: Mapping[str, Any]) -> str:
    launch = record.get("launch")
    argv = list(launch.get("argv") or []) if isinstance(launch, Mapping) else []
    parts: list[str] = []
    for index, token in enumerate(argv):
        for flag in _SHAPE_FLAGS:
            if token == flag and index + 1 < len(argv):
                parts.append(f"{flag.lstrip('-')}={argv[index + 1]}")
            elif token.startswith(flag + "="):
                parts.append(f"{flag.lstrip('-')}={token.split('=', 1)[1]}")
    return " ".join(parts) or "unknown"


def shapes_of(records: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """``runtime[@card] -> launch shape``, one line per runtime and card, for the baseline's `runs`.

    Keyed the way the metrics are, because the shape is a property of the same pair: a TP4 reading on
    a 4090 and a TP1 reading on a 2080 Ti are two different claims, and a `runs` block that named
    only the runtime could hold one of them.
    """
    shapes: dict[str, str] = {}
    for record in records:
        shapes[_token_of_record(record)] = _shape_of(record)
    return shapes


def render_baseline(
    values: Mapping[str, float],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    shapes: Mapping[str, str] | None = None,
) -> str:
    """The baseline as JSON, sorted, with the provenance a reader needs to judge it.

    JSON rather than a line-per-entry text file: a value here is a float, and a format that renders
    one is a format that rounds one. The provenance travels *inside* the file because a baseline
    whose commit and box are in a docstring goes stale the first time either changes, and this one is
    compared against numbers those two facts move. ``runs`` carries the launch shape per runtime for
    the reason the numbers cannot: a TP4 latency and a TP1 latency are both "tpot.median" in the
    metrics map.
    """
    document: dict[str, Any] = {
        "threshold": threshold,
        "note": (
            "Recorded from `relicllm bench serve`; regenerate with "
            "scripts/check_perf_baseline.py --update --observed <run.json> (repeat --observed, one "
            "run per runtime and per card). Keyed runtime[@card]/scenario/metric, the card being the "
            "device name the record's `cuda` block reports: numbers taken on another card are keys of "
            "their own and are never diffed against these. Compare with "
            "scripts/check_perf_baseline.py. `runs` is the launch shape each runtime's numbers were "
            "measured at -- they are not the same width, so it is not a detail."
        ),
        "metrics": values,
    }
    if shapes:
        document["runs"] = dict(shapes)
    return json.dumps(document, indent=2, sort_keys=True) + "\n"


def _format(delta: Delta) -> str:
    return (
        f"  {delta.key:<48} {delta.before:>12.4f} -> {delta.after:>12.4f}  "
        f"({delta.ratio:>6.2f}x)"
    )


def _resolve_records(args: argparse.Namespace) -> list[Mapping[str, Any]]:
    if args.observed:
        return [load(path) for path in args.observed]
    raise SystemExit(
        "no run to check: pass --observed FILE (a relicllm bench serve --json-out record), or "
        "launch one yourself:\n"
        "  relicllm bench serve --scenario decode --scenario prefill-8k \\\n"
        "      --json-out /tmp/run.json -- --model /ckpt --backend <name> --tensor-parallel-size 4"
    )


def _is_covered(token: str, covered: Iterable[str]) -> bool:
    """Whether a record's token is one the baseline holds numbers for.

    Same rule as the keys themselves: the runtime has to match, and a card on both sides has to be the
    same card. A baseline token with no card covers any card, because it makes no claim about one.
    """
    runtime, card = _split_token(token)
    return any(
        other_runtime == runtime and not _card_conflict(card, other_card)
        for other_runtime, other_card in (_split_token(other) for other in covered)
    )


def _skip_reason(token: str, covered: Iterable[str]) -> str:
    """Why a record was not compared, in the terms the reader needs to act on it.

    A runtime the baseline does not hold at all is the ordinary case -- a checkpoint this host does
    not have. A runtime it holds *on another card* is the one this script exists to refuse, and which
    card, on both sides, is the whole of the explanation.
    """
    runtime, _ = _split_token(token)
    others: set[str] = set()
    for other in covered:
        other_runtime, other_card = _split_token(other)
        if other_runtime == runtime:
            others.add(other_card or "no recorded card")
    if not others:
        return "the baseline does not cover this runtime"
    return (
        f"the baseline's numbers for {runtime} were measured on {', '.join(sorted(others))}, and a "
        f"number from one box is not a number from another"
    )


def _recorded_tokens(path: Path) -> set[str]:
    """The runtime tokens a baseline file already holds, or none when there is nothing to read."""
    try:
        document = load(path)
        keys = document.get("metrics", document)
        return {_token_of_key(key) for key in keys}
    except (OSError, ValueError, AttributeError):
        return set()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--observed",
        type=Path,
        action="append",
        default=None,
        metavar="FILE",
        help=(
            "a relicllm bench record to compare against the baseline (or to record with --update). "
            "Repeatable: the baseline covers several runtimes, and a run has one runtime in it"
        ),
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=BASELINE_PATH,
        help=f"the recorded baseline (default: {BASELINE_PATH.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="rewrite the baseline from --observed instead of comparing against it",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=f"how far a metric may move either way (default: the baseline's own, else {DEFAULT_THRESHOLD})",
    )
    parser.add_argument("--verbose", action="store_true", help="print the unchanged keys too")
    args = parser.parse_args(argv)

    records = _resolve_records(args)
    observed = baseline_from(records)

    if args.update:
        if not observed:
            named = ", ".join(str(path) for path in args.observed)
            raise SystemExit(
                f"{named} carries no comparable metric: the record's `scenarios` block is empty or "
                f"has no `metrics` -- nothing was measured, so there is nothing to record"
            )
        replaced = _recorded_tokens(args.baseline)
        tokens = sorted({_token_of_key(key) for key in observed})
        args.baseline.write_text(
            render_baseline(
                observed,
                threshold=args.threshold or DEFAULT_THRESHOLD,
                shapes=shapes_of(records),
            ),
            encoding="utf-8",
        )
        scenarios = sorted({key.split("/")[1] for key in observed})
        print(
            f"{args.baseline}: {len(observed)} metrics from "
            f"{', '.join(tokens)} x {', '.join(scenarios)}"
        )
        # `--update` replaces the file rather than merging into it, which was already true of a second
        # runtime and is now true of a second card. Saying which tokens the file used to hold is the
        # difference between recording the 4090 numbers and losing the 2080 Ti ones.
        dropped = sorted(replaced - set(tokens))
        if dropped:
            print(
                f"note: {args.baseline} no longer holds {', '.join(dropped)} -- a baseline is "
                f"replaced, not merged; pass every runtime and card you mean to keep"
            )
        return 0

    if not args.baseline.exists():
        raise SystemExit(
            f"{args.baseline} is missing. Record one with:\n"
            f"  python scripts/check_perf_baseline.py --update --observed {args.observed or 'RUN.json'}"
        )
    document = load(args.baseline)
    baseline = document.get("metrics", document)
    threshold = args.threshold if args.threshold is not None else float(
        document.get("threshold", DEFAULT_THRESHOLD)
    )

    covered = sorted({_token_of_key(key) for key in baseline})
    # A record whose runtime the baseline does not cover is reported and not compared -- a run
    # against an already-running server (which carries no --backend to read) or against a runtime
    # this host cannot serve has nothing to be compared with. Exit 0, the same rule as a missing
    # checkpoint; only a runtime the baseline *does* cover is allowed to move a metric. The same rule
    # applies to the card: a 4090 run is not a 2080 Ti baseline that moved, and the two are kept apart
    # here rather than being compared and reported as a regression.
    comparable: dict[str, float] = {}
    skipped: list[str] = []
    for record in records:
        token = _token_of_record(record)
        if not _is_covered(token, covered):
            skipped.append(token)
            continue
        comparable.update(observations(record))
    if skipped:
        for token in sorted(set(skipped)):
            print(f"skipped: measured {token} -- {_skip_reason(token, covered)}")
        print(f"the baseline covers {', '.join(covered)}")
    if not comparable:
        print("\nnothing was compared -- no named run measured a runtime the baseline covers")
        return 1

    report = compare(baseline, comparable, threshold=threshold)
    print(
        f"threshold {threshold:.0%}  "
        f"{len(report.unchanged) + len(report.regressions) + len(report.improvements)} compared, "
        f"{len(report.regressions)} regression(s), {len(report.improvements)} improvement(s)"
    )

    if report.regressions:
        print(f"\nREGRESSIONS ({len(report.regressions)}) -- past the threshold, the wrong way:")
        for delta in report.regressions:
            print(_format(delta))
    if report.improvements:
        print(f"\nimprovements ({len(report.improvements)}) -- past the threshold, the good way:")
        for delta in report.improvements:
            print(_format(delta))
    if report.missing:
        print(
            f"\nnot measured this run ({len(report.missing)}) -- not a pass, not a failure; the "
            f"baseline covers more than this run did:"
        )
        for key in report.missing:
            print(f"  {key}")
    if report.unenforced:
        print(
            f"\nnot enforced ({len(report.unenforced)}) -- the baseline recorded a zero, so there is "
            f"no ratio to take:"
        )
        for delta in report.unenforced:
            print(_format(delta))
    if args.verbose and report.unchanged:
        print(f"\nunchanged ({len(report.unchanged)}):")
        for delta in report.unchanged:
            print(_format(delta))

    if report.regressions:
        return 1
    if not report.unchanged and not report.improvements:
        print("\nnothing was compared -- the run and the baseline share no metric")
        return 1
    print("\nevery compared metric is within the threshold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
