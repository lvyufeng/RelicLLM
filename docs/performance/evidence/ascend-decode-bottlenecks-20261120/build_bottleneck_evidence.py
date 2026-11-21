"""Freeze matched bottleneck receipts and derived profiles, not a new speed milestone."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "docs/performance/evidence/ascend-decode-bottlenecks-20261120"


def main():
    if TARGET.exists():
        raise RuntimeError("bottleneck evidence is frozen; do not overwrite")
    comparison = json.loads((ROOT / ".coord/current-bottleneck-comparison.json").read_text())
    if not comparison["same_timed_tokens"] or not comparison["same_actual_expert_routes_in_all_event_scoped_forwards"]:
        raise RuntimeError("the profile comparison did not validate the workload")
    TARGET.mkdir(parents=True)
    files = []
    def copy(path, relative):
        destination = TARGET / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        files.append(destination)
    for mode, name in (("tile", "current-bottleneck-tile-20261120T034608Z"),
                        ("baseline", "current-bottleneck-baseline-20261120T035802Z")):
        source = ROOT / ".coord" / name
        for rank in range(8):
            for prefix in ("Cube-model", "decode-scopes", "profile-scope"):
                copy(source / f"{prefix}-rank-{rank}.json", f"{mode}/{prefix}-rank-{rank}.json")
        for base in ("preconditions.json", "summary.json"):
            copy(source / base, f"{mode}/{base}")
        copy(ROOT / f".coord/current-bottleneck-{mode}-analysis.json", f"{mode}/analysis.json")
    for name in ("current-bottleneck-comparison.json", "current-bottleneck-safety-20261120T033905Z.json",
                 "current-bottleneck-safety-20261120T041016Z.json", "profile_current_bottlenecks.py",
                 "run_current_bottleneck_profile.py", "analyze_current_bottleneck_profile.py",
                 "compare_current_bottlenecks.py", "pack_current_profile.py", "build_bottleneck_evidence.py"):
        copy(ROOT / ".coord" / name, name)
    manifest = {"milestone": "current tile-consumer bottleneck diagnosis against same-host native baseline",
                "results": comparison["uninstrumented"],
                "critical_rank_event_scope_totals": {mode: values["same_rank_per_layer_ms_per_forward"]
                       for mode, values in comparison["critical_rank_event_scopes"].items()},
                "critical_native_projection_ms_per_forward": comparison["native_tile_projections"]["critical_native_projection_ms_per_forward"],
                "same_actual_expert_routes_and_tokens_verified": True,
                "all_eight_ranks_profiled": True,
                "uninstrumented_timing_and_instrumented_profiles_are_separate": True,
                "EP_mean_max_owned_per_layer": comparison["EP_skew"]["mean_max_rank_owned_per_layer"],
                "EP_mean_owned_per_rank_per_layer": 0.75,
                "ring_geometry_logical_bytes_per_rank_per_forward": comparison["EP_skew"]["ring_logical_write_plus_read_bytes_per_rank_per_forward"],
                "ring_geometry_is_not_measured_DRAM_bandwidth": True,
                "device_kernel_traces_archived_separately": comparison["source_profiles"],
                "full_traces_are_retained_locally_and_private_remote_not_duplicated_into_published_site": True,
                "limits": comparison["limits"], "no_device_code_change_for_this_diagnosis": True,
                "lease_released_after_checkpoint_stat_and_empty_NPU_safety_checks": True}
    manifest["files"] = [{"path": str(path.relative_to(TARGET)), "bytes": path.stat().st_size,
                           "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in files]
    manifest["file_count"] = len(files)
    manifest["total_bytes"] = sum(row["bytes"] for row in manifest["files"])
    with (TARGET / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    print(json.dumps({"evidence": str(TARGET), "files": len(files), "bytes": manifest["total_bytes"],
                      "tile_TPS": comparison["uninstrumented"]["tile_TPS"],
                      "baseline_TPS": comparison["uninstrumented"]["baseline_TPS"]}), flush=True)


if __name__ == "__main__":
    main()
