"""Freeze original-table Engram and HC decode evidence without large tensor copies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
CORE = Path("/Users/lvyufeng/Projects/RelicLLM/.agents/worktrees/relic-core-ascend-online-unpack")
AUDIT = BASE / ".coord/engram-hc-audit-20261120T084701Z"
SNAPSHOTS = BASE / ".coord/engram-hc-final-sources/source-snapshots"
DEST = BASE / "docs/performance/evidence/ascend-host-engram-hc-20261120"
PRIVATE = Path("/data1/relic-v41-perf-20261118-0938/source")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    comparison = json.loads((AUDIT / "Engram-HC-model-comparison.json").read_bytes())
    if (not comparison["all_8_ranks_checked"] or not comparison["HC_only_bit_parity_passed_in_this_full_model_window"]
            or not comparison["native_Engram_changes_generation_trajectory"]):
        raise RuntimeError("the frozen scope requires HC bit parity and explicit Engram trajectory drift")
    preconditions = json.loads((BASE / ".coord/engram-hc-final-sources/preconditions.json").read_bytes())
    source_names = (
        "RelicLLM/relicllm/models/deepseek_v4_1/ascend_device_engram.py",
        "RelicLLM/relicllm/models/deepseek_v4_1/ascend_hc_decode.py",
        "relic-core/relic_core/kernels/original_fp8_engram.py",
        "relic-core/relic_core/kernels/hc_decode_fold.py",
        "relic-core/relic_core/kernels/_ascend_hc_decode_fold_kernels.py",
        "profile_device_engram.py", "profile_actual_cube.py", "run_device_engram_trial.py", "bounded_task.py",
    )
    staged = {}
    origins = {}
    for relative in source_names:
        path = SNAPSHOTS / relative
        payload = path.read_bytes()
        if sha(payload) != preconditions["source_hashes"][str(PRIVATE / relative)]:
            raise RuntimeError(f"the timed source snapshot failed its precondition checksum: {relative}")
        if relative.startswith("RelicLLM/"):
            current = BASE / relative.removeprefix("RelicLLM/")
        elif relative.startswith("relic-core/"):
            current = CORE / relative.removeprefix("relic-core/")
        else:
            current = BASE / ".coord" / relative
        if current.exists() and sha(current.read_bytes()) != sha(payload):
            raise RuntimeError(f"a current implementation differs from the final timed snapshot: {relative}")
        name = "source-snapshots/" + relative
        staged[name], origins[name] = payload, str(path.relative_to(BASE))
    for mode in ("host", "device", "device-hc", "device-hc-64"):
        for rank in range(8):
            path = AUDIT / f"model-{mode}-rank-{rank}.json"
            row = json.loads(path.read_bytes())
            original = row["original_raw_model_receipt"]
            if comparison["artifact_hashes"][original["path"]]["sha256"] != original["sha256"]:
                raise RuntimeError("a retained rank extraction has a different original artifact identity")
            name = "all-rank-models/" + path.name
            staged[name], origins[name] = path.read_bytes(), str(path.relative_to(BASE))
    additional = [
        (AUDIT / "Engram-HC-model-comparison.json", "Engram-HC-model-comparison.json"),
        (AUDIT / "summary.json", "all-rank-audit-ownership.json"),
        (BASE / ".coord/engram-hc-final-sources/preconditions.json", "64-step-preconditions.json"),
        (BASE / ".coord/compare_engram_hc_models.py", "sources/compare_engram_hc_models.py"),
        (Path(__file__), "sources/build_engram_hc_evidence.py"),
        (BASE / ".coord/probe_device_engram.py", "sources/probe_device_engram.py"),
        (BASE / ".coord/probe_hc_decode_fold.py", "sources/probe_hc_decode_fold.py"),
        (BASE / ".coord/perf_round2_safety.py", "sources/perf_round2_safety.py"),
        (BASE / ".coord/device-engram-probe-v2.stderr.log", "negative-results/copysign-capture-fallback.stderr.log"),
        (BASE / ".coord/device-engram-diagnostic-v4/device-engram-failure-rank-0.json", "negative-results/native-full-vs-TP-projection.json"),
        (BASE / ".coord/device-engram-probe-v5/device-engram-failure-rank-0.json", "negative-results/native-shape-not-ULP-neutral.json"),
        (BASE / ".coord/fused-HC-decode-fold-probe-20261120T070504Z/HC-fold-parity-failure.json", "negative-results/HC-tree-post-parity-failure.json"),
        (BASE / ".coord/fused-HC-decode-fold-sequential-probe-20261120T074204Z/HC-fold-parity-failure.json", "negative-results/HC-tree-pre-parity-failure.json"),
        (BASE / ".coord/fused-HC-decode-fold-both-sequential-probe-20261120T080750Z/HC-fold-probe.json", "HC-fold-probe.json"),
        (BASE / ".coord/engram-hc-regression.log", "regression.log"),
        (BASE / "tests/test_v41_ascend_device_engram.py", "tests/test_v41_ascend_device_engram.py"),
        (BASE / "tests/test_v41_ascend_hc_decode.py", "tests/test_v41_ascend_hc_decode.py"),
        (CORE / "tests/test_original_fp8_engram.py", "tests/test_original_fp8_engram.py"),
        (CORE / "tests/test_hc_decode_fold.py", "tests/test_hc_decode_fold.py"),
    ]
    for rank in range(8):
        additional.append((BASE / f".coord/device-engram-accepted-20261120T071528Z/device-engram-rank-{rank}.json",
                           f"all-rank-operator-probe/device-engram-rank-{rank}.json"))
    for path, name in additional:
        staged[name] = path.read_bytes()
        origins[name] = str(path.relative_to(BASE)) if path.is_relative_to(BASE) else str(path)
    for name in ("new-lease-safety-20261120T055851Z.json", "new-lease-safety-20261120T084751Z.json"):
        path = AUDIT / name
        row = json.loads(path.read_bytes())
        command = row.pop("jenkins_agent_identity")
        row["resident_jenkins_identity_sha256"] = sha(command.encode())
        row["resident_agent_identity_command_omitted_from_published_bundle"] = True
        row["original_private_receipt_sha256"] = sha(path.read_bytes())
        staged["safety/" + name] = (json.dumps(row, indent=2) + "\n").encode()
        origins["safety/" + name] = str(path.relative_to(BASE))
    release = {"host": "ts-216", "released": True, "detail": "lease dropped",
               "confirmed_by": "ts-fleet-bridge.ts_release after final idle/checkpoint/CANN/CI safety receipt",
               "final_safety_receipt": "safety/new-lease-safety-20261120T084751Z.json"}
    staged["safety/lease-release.json"] = (json.dumps(release, indent=2) + "\n").encode()
    origins["safety/lease-release.json"] = "confirmed tool result, not a new remote operation"
    manifest = {"scope": "original host tables; opt-in NPU Engram; exact-on-tested-consumers HC folds",
                "native_Engram_is_changed_accumulation_and_changes_generation": True,
                "Engram_observed_TPS_ratio_not_same_stream_causal_speedup": True,
                "HC_same_stream_16_step_TPS_ratio": comparison["device_to_device-hc"]["observed_TPS_ratio"],
                "64_step_sustained_batch_one_TPS": comparison["modes"]["device-hc-64"]["TPS"],
                "target_50_TPS_reached": False, "exact_production_defaults_unchanged": True,
                "full_teacher_and_final_tensor_artifacts_checksum_pinned_in_comparison_not_copied": True,
                "optional_incomplete_local_tensor_downloads_excluded": True,
                "older_bundles_not_rewritten": True,
                "file_count_excluding_manifest": len(staged), "bytes_excluding_manifest": sum(map(len, staged.values())),
                "files": {name: {"sha256": sha(payload), "bytes": len(payload), "source": origins[name]}
                          for name, payload in sorted(staged.items())}}
    DEST.mkdir(exist_ok=False)
    for relative, payload in staged.items():
        path = DEST / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    with (DEST / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    for relative, item in manifest["files"].items():
        if sha((DEST / relative).read_bytes()) != item["sha256"]:
            raise RuntimeError("frozen artifact checksum changed")
    print(json.dumps({"bundle": str(DEST.relative_to(BASE)), "files": len(staged),
                      "bytes": manifest["bytes_excluding_manifest"]}), flush=True)


if __name__ == "__main__":
    main()
