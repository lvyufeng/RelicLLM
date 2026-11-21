"""Analyze the current eight-rank scopes and CANN kernel details without TPS extrapolation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re


METRICS = ("aic_mac_ratio", "aic_scalar_ratio", "aic_mte1_ratio", "aic_mte2_ratio", "aic_fixpipe_ratio",
           "aiv_vec_ratio", "aiv_scalar_ratio", "aiv_mte2_ratio", "aiv_mte3_ratio", "cube_utilization(%)")


def number(value):
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def collective(row):
    return any(fragment in (row.get("Name", "") + " " + row.get("Type", "")).lower()
               for fragment in ("hcom_", "hccl", "hcomall"))


def merge(intervals):
    result = []
    for begin, end in sorted(intervals):
        if end <= begin:
            continue
        if result and begin <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([begin, end])
    return result


def length(intervals):
    return sum(end - begin for begin, end in merge(intervals))


def load_details(path):
    with path.open(newline="") as stream:
        data = list(csv.DictReader(stream))
    result = []
    for row in data:
        begin, duration = number(row.get("Start Time(us)")), number(row.get("Duration(us)"))
        if begin is not None and duration is not None and duration >= 0:
            result.append({**row, "_begin": begin, "_duration": duration, "_end": begin + duration})
    return result


def aggregate(rows, steps):
    families = defaultdict(lambda: {"calls": 0, "duration_us": 0.0, "metric_sums": {}, "metric_weights": {}})
    shapes = defaultdict(lambda: {"calls": 0, "duration_us": 0.0})
    for row in rows:
        name, duration = row["Name"], row["_duration"]
        entry = families[name]
        entry["calls"] += 1
        entry["duration_us"] += duration
        for metric in METRICS:
            value = number(row.get(metric))
            if value is not None:
                entry["metric_sums"][metric] = entry["metric_sums"].get(metric, 0.0) + value * duration
                entry["metric_weights"][metric] = entry["metric_weights"].get(metric, 0.0) + duration
        key = (name, row.get("Type", ""), row.get("Input Shapes", ""), row.get("Input Formats", ""), row.get("Output Formats", ""))
        shapes[key]["calls"] += 1
        shapes[key]["duration_us"] += duration
    ranked = []
    for name, row in families.items():
        ranked.append({"name": name, "calls_per_forward": row["calls"] / steps,
                       "duration_ms_per_forward": row["duration_us"] / steps / 1000,
                       "mean_duration_us_per_call": row["duration_us"] / row["calls"],
                       "duration_weighted_pipeline_ratios": {metric: total / row["metric_weights"][metric]
                           for metric, total in row["metric_sums"].items() if row["metric_weights"][metric] > 0}})
    ranked.sort(key=lambda row: -row["duration_ms_per_forward"])
    shape_rows = [{"name": key[0], "type": key[1], "input_shapes": key[2], "input_formats": key[3],
                   "output_formats": key[4], "calls_per_forward": row["calls"] / steps,
                   "duration_ms_per_forward": row["duration_us"] / steps / 1000}
                  for key, row in shapes.items()]
    shape_rows.sort(key=lambda row: -row["duration_ms_per_forward"])
    return ranked, shape_rows


def trace_windows(path):
    with path.open() as stream:
        trace = json.load(stream)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    windows = []
    host = defaultdict(lambda: {"calls": 0, "inclusive_us": 0.0})
    for event in events:
        if event.get("ph") != "X":
            continue
        name = event.get("name", "")
        ts, duration = number(event.get("ts")), number(event.get("dur"))
        if ts is None or duration is None:
            continue
        if re.fullmatch(r"relic\.decode_step\.\d+", name):
            windows.append({"name": name, "begin_us": ts, "end_us": ts + duration, "duration_ms": duration / 1000})
        if name.startswith("relic.scope.") or any(name.startswith(prefix) for prefix in ("aclrtSynchronize", "aclrtMemcpy", "aclrtLaunch", "aclmdlRIC")):
            host[name]["calls"] += 1
            host[name]["inclusive_us"] += duration
    windows.sort(key=lambda value: value["begin_us"])
    return windows, sorted([{"name": name, **row} for name, row in host.items()], key=lambda row: -row["inclusive_us"])


def analyze(root):
    modes = [json.loads((root / f"Cube-model-rank-{rank}.json").read_text()) for rank in range(8)]
    scopes = [json.loads((root / f"decode-scopes-rank-{rank}.json").read_text()) for rank in range(8)]
    profiles = [json.loads((root / f"profile-scope-rank-{rank}.json").read_text()) for rank in range(8)]
    assert [row["rank"] for row in modes] == list(range(8))
    assert all(row["timed_steps"] == 16 and row["timed_context_start"] == 24 and row["timed_context_end"] == 40 for row in modes)
    assert all(row["same_token_ids_as_uninstrumented_window"] for row in scopes + profiles)
    assert all(row["token_ids"] == scopes[0]["token_ids"] for row in scopes)
    assert all(row["timed_token_ids"] == modes[0]["timed_token_ids"] for row in modes)
    kernels = []
    for rank, profile in enumerate(profiles):
        profile_root = root / f"profiler-rank{rank}"
        details = list(profile_root.glob("**/kernel_details.csv"))
        traces = list(profile_root.glob("**/trace_view.json"))
        if len(details) != 1 or len(traces) != 1:
            raise RuntimeError(f"rank {rank} does not have exactly one kernel/trace output: {details}, {traces}")
        rows = load_details(details[0])
        windows, host_scopes = trace_windows(traces[0])
        if len(windows) != profile["steps"]:
            raise RuntimeError(f"rank {rank} trace does not contain exactly the declared decode forward markers")
        selected = [row for row in rows if any(window["begin_us"] <= row["_begin"] < window["end_us"] for window in windows)]
        clock_aligned = bool(selected)
        if not clock_aligned:
            raise RuntimeError(f"rank {rank} CPU trace/kernel timestamp domains do not align; do not infer attribution")
        compute = [row for row in selected if not collective(row)]
        comm = [row for row in selected if collective(row)]
        family, shapes = aggregate(compute, profile["steps"])
        comm_family, _ = aggregate(comm, profile["steps"])
        per_step = []
        for window in windows:
            current = [row for row in selected if window["begin_us"] <= row["_begin"] < window["end_us"]]
            non = [(max(row["_begin"], window["begin_us"]), min(row["_end"], window["end_us"])) for row in current if not collective(row)]
            com = [(max(row["_begin"], window["begin_us"]), min(row["_end"], window["end_us"])) for row in current if collective(row)]
            covered = length(non + com)
            per_step.append({"name": window["name"], "profiled_host_span_ms": window["duration_ms"],
                             "noncollective_kernel_interval_union_ms": length(non) / 1000,
                             "collective_duration_interval_union_ms_including_waits": length(com) / 1000,
                             "covered_interval_union_ms_including_waits": covered / 1000,
                             "outside_recorded_kernel_or_collective_intervals_ms": window["duration_ms"] - covered / 1000})
        native = [row for row in compute if "online_tile_cube" in row["Name"]]
        native_family, _ = aggregate(native, profile["steps"])
        kernels.append({"rank": rank, "steps": profile["steps"], "details_rows_in_forward_markers": len(selected),
                        "noncollective_calls_per_forward": len(compute) / profile["steps"],
                        "noncollective_duration_sum_ms_per_forward": sum(row["_duration"] for row in compute) / profile["steps"] / 1000,
                        "top_noncollective_families": family[:30], "top_input_shape_families": shapes[:50],
                        "collective_attributions_include_waiting_and_profiler_inflation": comm_family,
                        "native_tile_families": native_family, "profile_interval_unions": per_step,
                        "CPU_trace_host_scopes": host_scopes[:30], "timestamp_domains_verified_by_forward_marker_overlap": clock_aligned,
                        "notes": ["kernel duration sums may overlap; interval unions are reported separately",
                                  "HCCL durations include synchronization/rank waits; never call this link-only cost",
                                  "pipeline ratios are profiler measurements, not whole-model wall percentages",
                                  "this four-forward trace is not the TPS measurement"]})
    source_bytes_per_expert = 3 * (2304 * 5120 // 2 + 2304 * 5120 // 32)
    telemetry = [row for rank in scopes for row in rank["excluded_route_telemetry"]]
    if len(telemetry) != 8 * 8 * 40:
        raise RuntimeError("route telemetry must cover every layer/rank for every event-scoped forward")
    counts = defaultdict(list)
    for rank in scopes:
        for row in rank["excluded_route_telemetry"]:
            counts[(row["step"], row["layer"])].append(row["owned"])
    if any(sum(values) != 6 or len(values) != 8 for values in counts.values()):
        raise RuntimeError("cyclic route ownership is not a complete six-expert partition")
    max_owned = [max(values) for values in counts.values()]
    slowest = max(row["timed_decode_seconds"] for row in modes)
    output = {"directory": str(root), "ranks": 8, "actual_configuration": modes[0]["expert_contract"],
              "uninstrumented_forwards": 16, "uninstrumented_context": [24, 40],
              "uninstrumented_slowest_rank_seconds": slowest, "sustained_batch_one_TPS": 16 / slowest,
              "uninstrumented_ms_per_forward": slowest / 16 * 1000,
              "all_rank_finite": all(row["finite_final_logits"] for row in modes),
              "timed_token_ids": modes[0]["timed_token_ids"],
              "event_scopes": scopes, "CANN_kernels": kernels,
              "routing": {"all_rank_six_expert_partition_verified": True,
                          "per_layer_max_rank_owned_histogram": {str(count):max_owned.count(count) for count in sorted(set(max_owned))},
                          "mean_max_rank_owned_per_layer": sum(max_owned) / len(max_owned),
                          "mean_owned_per_rank_per_layer": 6 / 8,
                          "original_packed_bytes_per_owned_expert_three_projections": source_bytes_per_expert,
                          "original_packed_bytes_all_ranks_per_forward": source_bytes_per_expert * 6 * 40,
                          "mean_original_packed_bytes_per_rank_per_forward": source_bytes_per_expert * 6 * 40 / 8,
                          "logical_FP4_weight_elements_per_expert_three_projections": 3 * 2304 * 5120,
                          "tile_ring_write_plus_read_bytes_per_owned_expert_if_tile_mode": 3 * 2304 * 5120 * 2 * 2,
                          "traffic_fields_are_geometry_derived_not_measured_DRAM_bandwidth": True},
              "attribution_limits": ["event scopes, CPU scopes and CANN kernel durations are separate measurements, not additive",
                                     "all-rank CANN profile prevents one-rank-only profiler setup skew but still adds overhead",
                                     "no synthetic zero-work ablation is treated as production latency",
                                     "no HCCL waiting interval is labeled as removable network cost"]}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root)
    if args.save.exists():
        raise RuntimeError("do not overwrite a completed profile analysis")
    with args.save.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"directory": str(args.root), "TPS": report["sustained_batch_one_TPS"],
                      "uninstrumented_ms": report["uninstrumented_ms_per_forward"],
                      "rank0_scope_totals": report["event_scopes"][0]["totals"],
                      "rank0_top_kernels": report["CANN_kernels"][0]["top_noncollective_families"][:10],
                      "routing": report["routing"]}), flush=True)


if __name__ == "__main__":
    main()
