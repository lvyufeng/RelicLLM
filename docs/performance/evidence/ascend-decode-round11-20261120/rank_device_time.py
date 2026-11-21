"""Rank a CANN kernel_details.csv by device time, then by input shape within the top families.

This is the probe that found the round-27 gain. Reasoning about the model for fifteen rounds produced
nothing; ranking the profile by device time put a *copy* kernel first, and grouping those copies by
input shape named the rope slice that was paying for a format conversion per call.

Usage::

    python .coord/rank_device_time.py PROFILER_DIR [--steps 3] [--family aclnnInplaceCopy]

`PROFILER_DIR` is the `ASCEND_PROFILER_OUTPUT` directory holding `kernel_details.csv`; `--steps` is how
many decode steps the recording covers, so the per-step columns divide by it.

Two things the output deliberately does not do. It never reports a duty cycle: the wall span of an
instrumented recording is inflated by the profiler and cannot say what fraction of time the device was
busy. And it marks `hcom_*` rows, which are HCCL collective attributions whose durations under the
profiler are far above their cost in a captured graph, so they must not be ranked alongside kernels.
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path


def load(path: Path):
    rows = []
    with path.open() as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append((row["Name"], float(row["Duration(us)"]), row))
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def rank(rows, steps: int, top: int) -> None:
    by = collections.defaultdict(lambda: [0, 0.0])
    for name, duration, _ in rows:
        entry = by[name]
        entry[0] += 1
        entry[1] += duration
    collective = sum(t for n, (_, t) in by.items() if n.startswith("hcom_"))
    kernels = sum(v[1] for n, v in by.items() if not n.startswith("hcom_"))
    print(f"{len(rows)} kernels over {steps} steps: {len(rows)/steps:.0f} per step")
    print(f"non-collective device time {kernels/1000.0/steps:.1f} ms/step; "
          f"{collective/1000.0/steps:.1f} ms/step attributed to hcom_* (profiler-inflated, not ranked)")
    print()
    print(f"{'kernel':<46}{'calls/step':>11}{'ms/step':>10}{'share':>8}{'us each':>10}")
    for name, (count, total) in sorted(by.items(), key=lambda kv: -kv[1][1])[:top]:
        if name.startswith("hcom_"):
            continue
        print(f"{name[:46]:<46}{count/steps:>11.0f}{total/1000.0/steps:>10.2f}"
              f"{100.0*total/kernels:>7.1f}%{total/count:>10.1f}")


def by_shape(rows, steps: int, family: str, top: int) -> None:
    agg = collections.defaultdict(lambda: [0, 0.0])
    for name, duration, row in rows:
        if family not in name or name.startswith("hcom_"):
            continue
        key = (name[:40], row.get("Input Shapes", "")[:110],
               row.get("Input Formats", "")[:50], row.get("Output Formats", "")[:30])
        entry = agg[key]
        entry[0] += 1
        entry[1] += duration
    total = sum(v[1] for v in agg.values())
    print()
    print(f"=== {family}: {total/1000.0/steps:.2f} ms/step, grouped by input shape ===")
    for (name, shapes, in_fmt, out_fmt), (count, seconds) in sorted(
            agg.items(), key=lambda kv: -kv[1][1])[:top]:
        print(f"  {count/steps:>5.0f} calls {seconds/1000.0/steps:>6.2f} ms {seconds/count:>7.1f} us each")
        print(f"     {name}")
        print(f"     shapes {shapes}")
        print(f"     {in_fmt} -> {out_fmt}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiler_dir", type=Path,
                        help="the ASCEND_PROFILER_OUTPUT directory holding kernel_details.csv")
    parser.add_argument("--steps", type=int, default=3,
                        help="decode steps the recording covers (the driver profiles three)")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--family", default="aclnnInplaceCopy",
                        help="kernel-name fragment to break down by input shape")
    args = parser.parse_args()
    details = args.profiler_dir / "kernel_details.csv"
    if not details.is_file():
        raise SystemExit(f"no kernel_details.csv under {args.profiler_dir}")
    rows = load(details)
    rank(rows, args.steps, args.top)
    by_shape(rows, args.steps, args.family, 8)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
