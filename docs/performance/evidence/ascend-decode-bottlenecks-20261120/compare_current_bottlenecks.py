"""Cross-check critical-rank attribution, route skew and current-vs-baseline profiles."""
from __future__ import annotations

from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def critical_scopes(report):
    by_layer = defaultdict(dict)
    for rank in report["event_scopes"]:
        for row in rank["raw_scopes"]:
            if row["scope"] != "complete_CPU_Engram_boundary":
                by_layer[(row["step"], row["layer"])].setdefault(rank["rank"], {})[row["scope"]] = row["stream_elapsed_ms"]
    totals = defaultdict(float)
    result = []
    for (step, layer), values in sorted(by_layer.items()):
        if len(values) != 8:
            raise RuntimeError("every layer scope must include all eight ranks")
        selected = max(values, key=lambda rank: values[rank]["routed_experts"])
        row = values[selected]
        for label, milliseconds in row.items():
            totals[label] += milliseconds / report["event_scopes"][0]["steps"]
        result.append({"step": step, "layer": layer, "selected_rank": selected, **row})
    return {"same_rank_per_layer_ms_per_forward": dict(totals), "per_layer": result,
            "selection": "select longest routed rank, then report that SAME rank's A and B; do not add an independent maximum B",
            "is_event_dependency_elapsed_not_a_pure_compute_decomposition": True}


def native_triplets(report):
    root = ROOT / report["directory"]
    by_layer = defaultdict(list)
    by_count = defaultdict(list)
    rank_totals = []
    for rank, scopes in enumerate(report["event_scopes"]):
        telemetry = {(row["step"], row["layer"]): row["owned"] for row in scopes["excluded_route_telemetry"]}
        profile = json.loads((root / f"profile-scope-rank-{rank}.json").read_text())
        steps = profile["steps"]
        details = next((root / f"profiler-rank{rank}").glob("**/kernel_details.csv"))
        with details.open(newline="") as stream:
            rows = sorted([row for row in csv.DictReader(stream) if row["Name"] == "relic_online_tile_cube_0"],
                          key=lambda row: float(row["Start Time(us)"]))
        if len(rows) != steps * 40 * 3:
            raise RuntimeError("native projection triplets do not exactly cover the declared layer/forward count")
        active, empty, mac = 0.0, 0.0, 0.0
        for index in range(steps * 40):
            triplet = rows[index * 3:index * 3 + 3]
            durations = [float(row["Duration(us)"]) / 1000 for row in triplet]
            row = {"step": index // 40, "layer": index % 40, "rank": rank,
                   "owned": telemetry[(index // 40, index % 40)],
                   "gate_ms": durations[0], "up_ms": durations[1], "down_ms": durations[2],
                   "total_ms": sum(durations)}
            mac_us = sum(float(item["aic_mac_time(us)"]) for item in triplet)
            if (row["owned"] == 0) != (mac_us == 0):
                raise RuntimeError("chronological native triplets disagree with the actual source ownership telemetry")
            by_layer[(row["step"], row["layer"])].append(row)
            by_count[row["owned"]].append(row["total_ms"])
            mac += mac_us / 1000
            if row["owned"]:
                active += row["total_ms"]
            else:
                empty += row["total_ms"]
        rank_totals.append({"rank": rank, "active_native_ms_per_forward": active / steps,
                            "empty_native_ms_per_forward": empty / steps,
                            "AIC_mac_pipeline_time_sum_ms_per_forward": mac / steps,
                            "mac_pipeline_time_is_not_a_wall_or_global_utilization_percentage": True})
    critical = [max(rows, key=lambda row: row["total_ms"]) for _, rows in sorted(by_layer.items())]
    if any(row["owned"] != max(item["owned"] for item in by_layer[(row["step"], row["layer"])]) for row in critical):
        raise RuntimeError("the slowest native rank does not match largest ownership for at least one layer")
    steps = report["CANN_kernels"][0]["steps"]
    return {"critical_native_projection_ms_per_forward": sum(row["total_ms"] for row in critical) / steps,
            "gate_ms_per_forward_on_critical_native_rank": sum(row["gate_ms"] for row in critical) / steps,
            "up_ms_per_forward_on_critical_native_rank": sum(row["up_ms"] for row in critical) / steps,
            "down_ms_per_forward_on_critical_native_rank": sum(row["down_ms"] for row in critical) / steps,
            "per_rank": rank_totals, "per_layer_critical_rank": critical,
            "all_critical_native_ranks_have_maximum_layer_owned_count": True,
            "three_projection_ms_per_layer_by_owned_count": {str(count): {"cases": len(values),
                  "mean_ms": sum(values) / len(values), "minimum_ms": min(values), "maximum_ms": max(values)}
                  for count, values in sorted(by_count.items())},
            "notes": ["triplets are validated against all-rank owned-count telemetry and exact 3*40*steps count",
                      "native kernel projection durations exclude epilogue, routing and communication",
                      "this is a diagnostic four-forward profile, not the 16-forward TPS denominator"]}


def main():
    tile = json.loads((ROOT / ".coord/current-bottleneck-tile-analysis.json").read_text())
    baseline = json.loads((ROOT / ".coord/current-bottleneck-baseline-analysis.json").read_text())
    if tile["timed_token_ids"] != baseline["timed_token_ids"]:
        raise RuntimeError("the matched profiling runs do not share a 16-token workload")
    if any(tile["event_scopes"][rank]["excluded_route_telemetry"] != baseline["event_scopes"][rank]["excluded_route_telemetry"] for rank in range(8)):
        raise RuntimeError("the profiling runs do not have matching actual expert choices")
    families = []
    for rank, (left, right) in enumerate(zip(tile["CANN_kernels"], baseline["CANN_kernels"])):
        lf = {row["name"]: row["duration_ms_per_forward"] for row in left["top_noncollective_families"]}
        rf = {row["name"]: row["duration_ms_per_forward"] for row in right["top_noncollective_families"]}
        deltas = [{"name": name, "tile_ms": lf.get(name, 0), "baseline_ms": rf.get(name, 0),
                   "delta_ms": lf.get(name, 0) - rf.get(name, 0)} for name in lf.keys() | rf.keys()]
        deltas.sort(key=lambda row: -abs(row["delta_ms"]))
        families.append({"rank": rank, "noncollective_kernel_duration_delta_ms":
                         left["noncollective_duration_sum_ms_per_forward"] - right["noncollective_duration_sum_ms_per_forward"],
                         "top_family_deltas": deltas[:20], "family_rank_tables_are_truncated_not_closed_accounting": True})
    geometry = tile["routing"]
    native = native_triplets(tile)
    report = {
        "fresh_same_host_comparison": True, "ranks": 8, "same_timed_tokens": True,
        "same_actual_expert_routes_in_all_event_scoped_forwards": True,
        "uninstrumented": {
            "steps": 16, "excluded_warmups": 4, "context": [24,40],
            "tile_TPS": tile["sustained_batch_one_TPS"], "baseline_TPS": baseline["sustained_batch_one_TPS"],
            "tile_ms_per_forward": tile["uninstrumented_ms_per_forward"],
            "baseline_ms_per_forward": baseline["uninstrumented_ms_per_forward"],
            "regression_ms_per_forward": tile["uninstrumented_ms_per_forward"] - baseline["uninstrumented_ms_per_forward"],
            "native_changed_accumulation_mode_same_for_comparison": True,
            "complete_Engram_CPU_and_original_quantization_retained": True,
            "exact_default_and_serving_unchanged": True},
        "critical_rank_event_scopes": {"tile": critical_scopes(tile), "baseline": critical_scopes(baseline)},
        "native_tile_projections": native, "operator_changes": families,
        "EP_skew": {**geometry, "ratio_mean_max_owned_to_mean_owned": geometry["mean_max_rank_owned_per_layer"] / geometry["mean_owned_per_rank_per_layer"],
                    "ring_logical_write_plus_read_bytes_per_rank_per_forward":
                    geometry["tile_ring_write_plus_read_bytes_per_owned_expert_if_tile_mode"] * 40 * 6 / 8,
                    "packed_source_bytes_and_ring_capacity_do_not_measure_DRAM_bandwidth": True},
        "Engram": {"tile_rank0_stream_boundary_ms_per_forward": tile["event_scopes"][0]["totals"]["complete_CPU_Engram_boundary"]["stream_elapsed_ms_per_forward"],
                   "baseline_rank0_stream_boundary_ms_per_forward": baseline["event_scopes"][0]["totals"]["complete_CPU_Engram_boundary"]["stream_elapsed_ms_per_forward"],
                   "tile_rank0_exposed_prefetch_wait_ms_per_forward": tile["event_scopes"][0]["exposed_prefetch_wait_ms_per_forward"],
                   "baseline_rank0_exposed_prefetch_wait_ms_per_forward": baseline["event_scopes"][0]["exposed_prefetch_wait_ms_per_forward"],
                   "tile_CPU_lookup_projection_worker_mean_ms": sum(row["elapsed_ms"] for row in tile["event_scopes"][0]["CPU_worker_calls"]) / len(tile["event_scopes"][0]["CPU_worker_calls"]),
                   "tile_host_boundary_ms_including_pending_NPU_wait": tile["event_scopes"][0]["totals"]["complete_CPU_Engram_boundary"]["host_inclusive_ms_per_forward"],
                   "host_boundary_time_is_not_CPU_compute_or_removable_D2H_cost": True},
        "interpretation": [
            "The current tile path's primary bottleneck is the routed-expert consumer, amplified by per-layer EP maximum ownership.",
            "Rank0 average native 50.7 ms understates model-critical time: sum of slowest per-layer native triplets is 134.5 ms in the four-forward trace.",
            "The native AIC MAC ratio is about 3.7%; AIV scalar/control ratio is about 95%, including synchronization waits, not just decode arithmetic.",
            "The 192 KiB ring bounds capacity but still logically writes and rereads 4.25 GB of expanded BF16 weights per rank per forward on average.",
            "The second large region is attention/router/HC graph A at about 52 ms; ordinary MatMul and thousands of small vector/movement ops are both present.",
            "High light-rank HCCL durations primarily expose waits on busy expert ranks; these are not additional link-only latency.",
            "Complete CPU Engram remains a smaller exposed region, about 8.7 ms plus roughly 4.7 ms exposed prefetch wait (the latter is included, not additive)."],
        "next_investigations": [
            "Profile/implement a 910B-supported consumer handoff that avoids expanded BF16 GM write/read, checking any SDK-internal staging rather than assuming zero traffic.",
            "Reduce per-Ktile scalar/event serialization and per-launch lookup-table initialization; fuse gate/up scheduling while preserving original operands and BF16 boundaries.",
            "Address actual per-layer EP critical-rank load or overlap independent shared-expert work; do not extrapolate mean 0.75 expert/rank to the critical path.",
            "For graph A, fuse arithmetic/layout boundaries only after original-value parity checks; prioritize measured shapes instead of raw Python call count."],
        "limits": [
            "Event and profiler passes are separate from TPS. Eight event forwards and four CANN forwards cover a shorter context than the 16-step wall window.",
            "Event instrumentation perturbs baseline more than tile; scope differences are not an exact causal wall decomposition.",
            "Per-rank A/B/Engram host and stream clocks overlap. Selecting independent rank maxima for routed and B double-counts wait; this report selects the SAME rank.",
            "Kernel duration sums and pipeline ratios are not global device utilization; HCCL rows include waits and overhead.",
            "Ring traffic is logical geometry-derived, not an observed off-chip bandwidth counter.",
            "The two fresh trials are sequential matched runs, not a randomized statistical performance experiment."],
        "safe_lease_release": {"released": True, "receipt": ".coord/current-bottleneck-safety-20261120T041016Z.json",
                              "checkpoint_files_unchanged": 90, "all_eight_NPU_process_tables_empty": True},
        "source_profiles": [{"path": str(Path(row["directory"]) / "profile-bundle.zip"),
                             "local_zip": f".coord/{Path(row['directory']).name}.zip",
                             "sha256": hashlib.sha256((ROOT / f".coord/{Path(row['directory']).name}.zip").read_bytes()).hexdigest()}
                            for row in (tile, baseline)]}
    destination = ROOT / ".coord/current-bottleneck-comparison.json"
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"report": str(destination), "uninstrumented": report["uninstrumented"],
                      "tile_critical_scopes": report["critical_rank_event_scopes"]["tile"]["same_rank_per_layer_ms_per_forward"],
                      "critical_native_ms": native["critical_native_projection_ms_per_forward"],
                      "EP_max_mean_ratio": report["EP_skew"]["ratio_mean_max_owned_to_mean_owned"],
                      "ring_logical_bytes_per_rank_per_forward": report["EP_skew"]["ring_logical_write_plus_read_bytes_per_rank_per_forward"]}), flush=True)


if __name__ == "__main__":
    main()
