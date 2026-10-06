#!/usr/bin/env python3
"""Diff a benchmark run against the recorded performance baseline, per metric.

``relicllm bench serve`` (P0.1) answers "how fast is this, on this box, for this scenario" and writes
the answer as a JSON record. This script is the other half: it holds the answer that was recorded,
and it says which metric moved, in which direction, by how much -- or that nothing did.

The design is deliberately the test baseline's (`scripts/check_test_baseline.py`), with one
substitution. That script's unit is a **node id** because the failure set is a set: three tests fixed
and one broken is a net improvement in a count and a regression in the tree. This script's unit
cannot be a set, because a metric is a *number* and every good number moves a little every run; the
unit is a **(runtime, scenario, metric) key** and the comparison is a ratio against a threshold.

Three properties that are the reason this is a script and not a `diff`:

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

Usage::

    python scripts/check_perf_baseline.py                        # diff tests/fixtures/perf/baseline.json
    python scripts/check_perf_baseline.py --observed run.json    # diff a specific run
    python scripts/check_perf_baseline.py --update --observed run.json   # record it as the baseline
    python scripts/check_perf_baseline.py --threshold 0.10       # 10% either way (the default is 15%)
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
    """One run as ``runtime/scenario/metric -> value``.

    Keyed by everything needed to know what the number is about. The runtime comes from
    ``launch.argv`` rather than from a filename, because the record is the thing being compared and a
    moved or renamed file must not change what it says.
    """
    runtime = _runtime_of(record)
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
                found[f"{runtime}/{scenario}/{metric}"] = value
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
    """
    regressions: list[Delta] = []
    improvements: list[Delta] = []
    unchanged: list[Delta] = []
    unenforced: list[Delta] = []
    missing: list[str] = []

    for key in sorted(baseline):
        if key not in observed:
            missing.append(key)
            continue
        delta = Delta(key=key, before=baseline[key], after=observed[key])
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
    """``runtime -> launch shape``, one line per runtime, for the baseline's `runs` block."""
    shapes: dict[str, str] = {}
    for record in records:
        shapes[_runtime_of(record)] = _shape_of(record)
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
            "run per runtime). Keyed runtime/scenario/metric. Compare with "
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
        args.baseline.write_text(
            render_baseline(
                observed,
                threshold=args.threshold or DEFAULT_THRESHOLD,
                shapes=shapes_of(records),
            ),
            encoding="utf-8",
        )
        runtimes = sorted({key.split("/", 1)[0] for key in observed})
        scenarios = sorted({key.split("/")[1] for key in observed})
        print(
            f"{args.baseline}: {len(observed)} metrics from "
            f"{', '.join(runtimes)} x {', '.join(scenarios)}"
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

    covered = sorted({key.split("/", 1)[0] for key in baseline})
    # A record whose runtime the baseline does not cover is reported and not compared -- a run
    # against an already-running server (which carries no --backend to read) or against a runtime
    # this host cannot serve has nothing to be compared with. Exit 0, the same rule as a missing
    # checkpoint; only a runtime the baseline *does* cover is allowed to move a metric.
    comparable: dict[str, float] = {}
    skipped: list[str] = []
    for record in records:
        runtime = _runtime_of(record)
        if runtime not in covered:
            skipped.append(runtime)
            continue
        comparable.update(observations(record))
    if skipped:
        print(f"skipped: measured {', '.join(sorted(set(skipped)))}, and the baseline covers {', '.join(covered)}")
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