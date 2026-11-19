"""Attribute a `POCKETLLM_QWEN4_ROUND_TRACE` Chrome trace to full kernel names, by round.

Why this exists. The first-generation analyzer collapsed kernel names before reporting them
(`tests/analyze_round_trace.py`), which produced buckets that named nothing: the top verify entry
was the literal string ``std::enable_if`` at 7.173 ms, and three more were ``""``, ``void`` and
``void at::native::``. Those are not kernel families, they are artifacts of truncating a mangled
template name at its first ``<`` or ``(``; together they were 20% of the measured busy time, so a
fifth of the verifier's cost could not be named. This tool keeps every full name and classifies on
it, and prints anything it cannot classify instead of merging it into a catch-all.

What it does. Each ``cudaGraphLaunch`` runtime call is joined to the ``user_annotation`` range
(`round.verify`, `round.draft`, ...) that contains it, and each kernel/memcpy/memset event is joined
to its launch by Chrome's ``correlation`` id. Per round it then reports:

* ``span_ms_mean`` -- first kernel start to last kernel end inside one replay. This is a *span*,
  not device busy time, and it is measured under the profiler, so it is deliberately NOT an
  achievable floor or a TPS input. Every document says so next to the number.
* ``busy_ms_mean`` -- the sum of the replay's kernel durations. Distinct from the span: the
  difference is inter-kernel gap, not recoverable work.
* ``families`` and ``top_kernels`` -- per replay and summed, with full names retained.

Usage::

    python tests/analyze_qwen4_exp_kernel_census.py trace.json [--top 30] [--json-out out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict

SCHEMA = "qwen4-exp-kernel-census-v1"

#: Launches belong to a round when they fall inside one of these annotated ranges.
ROUND_PREFIX = "round."

#: Event categories whose `dur` is device work inside a launch.
KERNEL_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})

#: Ordered family rules; first match wins. The order is meaningful: a cuBLAS GEMV template also
#: contains library words, project kernels must be matched before the generic `qwen4_exp_` rule,
#: and `copy` is safe before `elementwise` because a memcpy name never contains "elementwise".
_FAMILY_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("nccl", ("nccl",)),
    # The family the previous analyzer hid behind `std::enable_if`: templates whose *body* is
    # `internal::gemvx::kernel`.
    ("cublas_gemvx", ("gemvx::kernel",)),
    ("cublaslt", ("cublaslt",)),
    ("cutlass", ("cutlass",)),
    ("gemm", ("sgemm", "hgemm", "gemm", "kernel2")),
    ("gdn_recurrent", ("qwen4_exp_gated_delta", "normalize_qk_bf16", "gated_delta_bf16_kernel")),
    ("hyper_connection", ("qwen4_exp_hc_", "hc_up_mix", "qwen4_exp_inject")),
    ("nvfp4_moe", ("nvfp4",)),
    ("norm", ("grouped_rms_norm", "rms_norm")),
    ("qsa_attention", ("qsa",)),
    ("qwen4_exp_other", ("qwen4_exp_", "relic_core")),
    # Torch's top-k machinery is many kernels with one cause: selection over many blocks.
    ("topk_sort", ("topk", "radixsort", "bitonicsort", "scan_by_key", "sortkv")),
    ("copy", ("memcpy", "memset", "copy_kernel", "catarraybatchedcopy")),
    ("convolution", ("conv_depthwise", "conv1d", "convolution")),
    ("elementwise", ("elementwise",)),
    ("reduction", ("reduce", "dot_kernel", "sum_kernel")),
)

#: Names no rule matches keep this family and are still listed with their full text.
UNCLASSIFIED = "unclassified"


def _events(data) -> list:
    """The event list from either Chrome export shape (``{"traceEvents": [...]}`` or a bare list)."""
    if isinstance(data, dict):
        events = data.get("traceEvents")
        if not isinstance(events, list):
            raise ValueError("a Chrome trace dict must carry a traceEvents list")
        return events
    if isinstance(data, list):
        return data
    raise ValueError("a trace must be a JSON list or an object with traceEvents")


def classify(name: str) -> str:
    """The family a full kernel name belongs to, or :data:`UNCLASSIFIED`.

    Matching is on the untruncated lowercased name, so a cuBLAS GEMV whose text starts with
    ``std::enable_if`` is classified by the ``gemvx::kernel`` inside it, not by its leading token.
    """
    lowered = str(name).lower()
    for family, markers in _FAMILY_RULES:
        if any(marker in lowered for marker in markers):
            return family
    return UNCLASSIFIED


def _correlation(event) -> object | None:
    """The launch/kernel correlation id, under the spellings Chrome exporters use."""
    args = event.get("args")
    if not isinstance(args, dict):
        return None
    for key in ("correlation", "Correlation", "external id", "External id"):
        if key in args:
            return args[key]
    return None


def _launch_kind(event) -> str | None:
    """``graph`` for a CUDA graph launch, else ``None``.

    Only graph launches are joined: an ordinary kernel launch has no correlation the trace can pair
    with a replay, and the round annotations here wrap recorded replays.
    """
    name = str(event.get("name", ""))
    if event.get("cat") != "cuda_runtime":
        return None
    return "graph" if name.startswith("cudaGraphLaunch") else None


def _ranges(events: list) -> list[dict]:
    """Every ``round.*`` user annotation, with its interval, in start order."""
    found = []
    for event in events:
        if event.get("cat") != "user_annotation":
            continue
        name = str(event.get("name", ""))
        if not name.startswith(ROUND_PREFIX):
            continue
        ts = event.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        duration = event.get("dur")
        length = float(duration) if isinstance(duration, (int, float)) else 0.0
        found.append({"name": name, "ts": float(ts), "end": float(ts) + length, "dur": length})
    found.sort(key=lambda row: (row["ts"], row["end"]))
    return found


def _owning_range(ranges: list[dict], ts: float) -> dict | None:
    """The innermost round range containing ``ts``.

    Innermost, not first, because ranges can nest (a startup or tail recording can open one range
    inside a wider one). Choosing the shortest containing interval attributes a launch to the
    tightest annotation that claims it.
    """
    best = None
    for row in ranges:
        if row["ts"] <= ts <= row["end"] and (best is None or row["dur"] < best["dur"]):
            best = row
    return best


def census(data) -> dict:
    """Build the census document from a parsed Chrome trace.

    The join is kernel --correlation--> graph launch --containment--> round annotation. A kernel
    with no correlation, or a launch outside every round range, is counted rather than dropped, so
    a missing attribute is visible instead of silently shrinking the totals. The two are counted
    separately: ``kernels_outside_rounds`` is startup/tail work whose launch was seen, and
    ``kernels_unmatched_correlation`` is the stricter case where no launch carried that id at all.
    """
    events = _events(data)
    ranges = _ranges(events)
    categories = Counter(str(event.get("cat", "")) for event in events)

    rounds: dict[str, dict] = {}

    def round_stats(label: str) -> dict:
        stats = rounds.get(label)
        if stats is None:
            stats = {"launch_count": 0, "span_us": 0.0, "busy_us": 0.0, "kernels": 0,
                     "families": defaultdict(lambda: {"kernels": 0, "busy_us": 0.0,
                                                      "names": Counter()}),
                     "name_busy": defaultdict(float)}
            rounds[label] = stats
        return stats

    launches_by_correlation: dict[object, str] = {}
    #: Correlations of graph launches that ran *outside* every round range (startup/tail replays).
    #: Kept separately so their kernels are reported as outside-round work rather than as a failed
    #: correlation join: the launch exists, it simply belongs to no steady round.
    outside_launch_correlations: dict[object, str] = {}
    outside = Counter()
    for event in events:
        if _launch_kind(event) is None:
            continue
        ts = event.get("ts")
        if not isinstance(ts, (int, float)):
            continue
        name = str(event.get("name", ""))
        correlation = _correlation(event)
        owner = _owning_range(ranges, float(ts))
        if owner is None:
            outside[name] += 1
            if correlation is not None:
                outside_launch_correlations[correlation] = name
            continue
        if correlation is None:
            # A launch inside a round with no id could not be paired with its kernels; count it as
            # outside so a round's replay count never silently includes a nameless launch.
            outside[name + ":no-correlation"] += 1
            continue
        # Two launches sharing a correlation would double count kernels; the trace does not do
        # this, and if it ever did the label map would keep the last, which is at least visible.
        launches_by_correlation[correlation] = owner["name"]
        round_stats(owner["name"])["launch_count"] += 1

    kinds = Counter(str(event.get("cat", "")) for event in events)
    kernels_without_correlation = 0
    kernels_outside_rounds = 0
    kernels_unmatched = 0
    launch_bounds: dict[object, list[float]] = {}
    for event in events:
        if event.get("cat") not in KERNEL_CATEGORIES:
            continue
        correlation = _correlation(event)
        if correlation is None:
            kernels_without_correlation += 1
            continue
        label = launches_by_correlation.get(correlation)
        if label is None:
            if correlation in outside_launch_correlations:
                # The launch was seen, it just ran outside every round annotation. That is
                # startup/tail work, not a broken join, and must not inflate the unmatched count.
                kernels_outside_rounds += 1
            else:
                kernels_unmatched += 1
            continue
        name = str(event.get("name", ""))
        ts = event.get("ts")
        duration = event.get("dur")
        length = float(duration) if isinstance(duration, (int, float)) else 0.0
        stats = round_stats(label)
        stats["kernels"] += 1
        stats["busy_us"] += length
        family = classify(name)
        bucket = stats["families"][family]
        bucket["kernels"] += 1
        bucket["busy_us"] += length
        bucket["names"][name] += 1
        stats["name_busy"][(family, name)] += length
        if isinstance(ts, (int, float)):
            start = float(ts)
            bounds = launch_bounds.get(correlation)
            if bounds is None:
                launch_bounds[correlation] = [start, start + length]
            else:
                bounds[0] = min(bounds[0], start)
                bounds[1] = max(bounds[1], start + length)

    # A replay's span needs that launch's own first/last kernel: a replay with no observed kernels
    # contributes no span, so it is not counted as zero.
    for correlation, (_start, _end) in launch_bounds.items():
        stats = round_stats(launches_by_correlation[correlation])
        stats["span_us"] += _end - _start

    report = {}
    for label in sorted(rounds):
        stats = rounds[label]
        replays = max(1, len(
            [correlation for correlation, owner in launches_by_correlation.items()
             if owner == label and correlation in launch_bounds]))
        families = [
            {
                "family": family,
                "kernels_per_replay": round(bucket["kernels"] / replays, 3),
                "busy_ms_per_replay": round(bucket["busy_us"] / replays / 1000.0, 4),
                "busy_ms_sum": round(bucket["busy_us"] / 1000.0, 4),
                "share_of_busy": (round(bucket["busy_us"] / stats["busy_us"], 6)
                                  if stats["busy_us"] > 0 else 0.0),
                "distinct_full_names": len(bucket["names"]),
            }
            for family, bucket in sorted(stats["families"].items(),
                                         key=lambda item: -item[1]["busy_us"])
        ]
        top_kernels = [
            {
                "full_name": name,
                "family": family,
                "kernels_sum": stats["families"][family]["names"][name],
                "kernels_per_replay": round(
                    stats["families"][family]["names"][name] / replays, 3),
                "busy_ms_sum": round(busy / 1000.0, 4),
                "busy_ms_per_replay": round(busy / replays / 1000.0, 4),
            }
            for (family, name), busy in sorted(stats["name_busy"].items(),
                                               key=lambda item: -item[1])
        ]
        report[label] = {
            "launches": stats["launch_count"],
            "replays": replays,
            "span_ms_mean": round(stats["span_us"] / replays / 1000.0, 3),
            "busy_ms_mean": round(stats["busy_us"] / replays / 1000.0, 3),
            "busy_ms_sum": round(stats["busy_us"] / 1000.0, 3),
            "kernels_per_replay": round(stats["kernels"] / replays, 1),
            "kernels_sum": stats["kernels"],
            "families": families,
            "top_kernels": top_kernels,
            "unclassified_full_names": sorted(
                {name for (family, name) in stats["name_busy"] if family == UNCLASSIFIED}),
        }

    return {
        "schema": SCHEMA,
        "instrumented_scope": {
            "span_is_instrumented_not_floor": True,
            "note": ("span_ms_mean is a profiled wall span from the first to the last kernel of one "
                     "graph replay. It contains inter-kernel gaps the profiler itself widens, so it "
                     "must not be quoted as an achievable lower bound, a device-busy floor, or a "
                     "TPS denominator. busy_ms_mean is the summed kernel duration and is a separate "
                     "number, not a corrected span."),
        },
        "event_counts": {
            "user_annotations": categories.get("user_annotation", 0),
            "round_annotations": len(ranges),
            "cuda_graph_launches": sum(outside.values()) + len(launches_by_correlation),
            "kernels": sum(kinds.get(category, 0) for category in KERNEL_CATEGORIES),
        },
        "rounds": report,
        "launches_outside_rounds": {
            "count": sum(outside.values()),
            "by_name": dict(sorted(outside.items())),
            "note": ("startup and one-row tail recordings replay outside round.* ranges; they are "
                     "counted here rather than attributed to a steady round"),
        },
        "kernels_without_correlation": kernels_without_correlation,
        "kernels_outside_rounds": kernels_outside_rounds,
        "kernels_unmatched_correlation": kernels_unmatched,
        "notes": [
            "Families are first-match rules over the full untruncated name; unmatched names keep "
            "family 'unclassified' and are listed, never merged into a catch-all bucket.",
            "cublas_gemvx is cuBLAS's internal::gemvx::kernel, the family the previous analyzer "
            "reported as std::enable_if.",
            "kernels_outside_rounds are kernels whose graph launch was found outside every "
            "round.* range (startup and one-row tail replays); kernels_unmatched_correlation is the "
            "stricter case where no launch carried that correlation at all.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", help="Chrome trace JSON exported by POCKETLLM_QWEN4_ROUND_TRACE")
    parser.add_argument("--top", type=int, default=30,
                        help="full-name kernels printed per round; families are always complete")
    parser.add_argument("--json-out", help="also write the census document to this path")
    args = parser.parse_args(argv)
    if args.top < 1:
        raise SystemExit("--top must be positive")
    with open(args.trace) as stream:
        data = json.load(stream)
    document = census(data)
    for stats in document["rounds"].values():
        stats["top_kernels_shown"] = min(args.top, len(stats["top_kernels"]))
        stats["top_kernels_total"] = len(stats["top_kernels"])
        stats["top_kernels"] = stats["top_kernels"][: args.top]
    text = json.dumps(document, indent=1, sort_keys=True)
    if args.json_out:
        with open(args.json_out, "w") as stream:
            stream.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())