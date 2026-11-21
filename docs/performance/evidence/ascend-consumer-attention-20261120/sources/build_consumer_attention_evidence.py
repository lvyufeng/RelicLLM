"""Freeze new consumer-grain and attention-reuse evidence, leaving older bundles intact."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = Path("/home/<user>/Projects/RelicLLM/.agents/worktrees/relic-core-ascend-online-unpack")
AUDIT = ROOT / ".coord/consumer-attention-audit-20261120T110608Z"
DEST = ROOT / "docs/performance/evidence/ascend-consumer-attention-20261120"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    comparison = json.loads((AUDIT / "consumer-attention-model-comparison.json").read_bytes())
    if not comparison["all_8_ranks_checked"]:
        raise RuntimeError("all-rank model audit is missing")
    for pair in ("k128_to_k256", "native_to_native-reuse"):
        rows = comparison["pairs"][pair]["all_rank_bits"]
        if len(rows) != 8 or any(row["teacher_FP32_bit_differences"] or row["final_FP32_bit_differences"] or not row["same_timed_tokens"] for row in rows):
            raise RuntimeError("the proposed same-stream scope did not pass its complete-model bit checks")
    staged, origins = {}, {}
    def include(path, relative):
        staged[relative] = path.read_bytes()
        origins[relative] = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    include(AUDIT / "consumer-attention-model-comparison.json", "consumer-attention-model-comparison.json")
    include(AUDIT / "summary.json", "all-rank-audit-ownership.json")
    for mode in ("k128", "k256", "native", "native-reuse"):
        for rank in range(8):
            path = AUDIT / f"model-{mode}-rank-{rank}.json"
            row = json.loads(path.read_bytes())
            origin = row["original_complete_model_receipt"]
            if origin["sha256"] != comparison["artifact_hashes"][origin["path"]]["sha256"]:
                raise RuntimeError("a retained rank receipt has the wrong raw artifact checksum")
            # The store describes its default placement context. Rebind only
            # that descriptive claim to the separately installed Engram owner;
            # original complete receipts stay checksum-pinned, not rewritten.
            if row.get("host_table_device_Engram") is not None:
                store = row.get("device_selected_MoE")
                if isinstance(store, dict):
                    for key in ("complete_Engram_CPU", "complete_engram_remains_on_cpu", "complete_engram_on_cpu"):
                        if key in store:
                            store[key] = False
                row["current_Engram_placement_is_separate_host_table_NPU_compute_owner"] = True
            name = "all-rank-models/" + path.name
            staged[name] = (json.dumps(row, indent=2) + "\n").encode()
            origins[name] = str(path.relative_to(ROOT))
    pointer = ROOT / ".coord/consumer-model-source-snapshots"
    preconditions = json.loads((pointer / "preconditions.json").read_bytes())
    snapshots = pointer / "source-snapshots"
    private_source = Path("<workspace>/relic-v41-perf-20261118-0938/source")
    for path in sorted(snapshots.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(snapshots).as_posix()
        if digest(path.read_bytes()) != preconditions["source_hashes"][str(private_source / relative)]:
            raise RuntimeError("a timed source snapshot does not match the precondition certificate")
        include(path, "timed-source-snapshots/" + relative)
    include(pointer / "preconditions.json", "native-reuse-preconditions.json")
    for path in (
        ROOT / "relicllm/models/deepseek_v4_1/ascend_attention_reuse.py",
        CORE / "relic_core/kernels/ascendc_online_tile_cube.py",
        CORE / "relic_core/kernels/ascendc_tile_moe.py",
        CORE / "relic_core/csrc/ascend/online_tile_cube/README.md",
    ):
        name = path.name + (".txt" if path.suffix == ".md" else "")
        include(path, "final-source-hardening/" + name)
    for name in ("build_consumer_variants.py", "probe_consumer_variants.py", "probe_k256_corpus.py",
                 "probe_attention_fp8_reuse.py", "compare_consumer_attention_models.py", "build_consumer_attention_evidence.py"):
        include(ROOT / ".coord" / name, "sources/" + name)
    include(ROOT / "tests/test_v41_ascend_attention_reuse.py", "tests/test_v41_ascend_attention_reuse.py")
    for name in ("test_ascendc_online_tile_cube.py", "test_ascendc_tile_moe.py"):
        include(CORE / "tests" / name, "tests/" + name)
    include(ROOT / ".coord/consumer-attention-regression.log", "regression.log")
    for name in ("consumer-variants-dynamic-validation.json", "consumer-variants-real-projection-comparison.json", "summary.json"):
        include(ROOT / ".coord/consumer-variants-operator-20261120T103051Z" / name, "operator-variants/" + name)
    include(ROOT / ".coord/consumer-all-K-corpus-20261120T110616Z/K256-original-byte-scale-corpus.json", "K256-original-byte-scale-corpus.json")
    include(ROOT / ".coord/consumer-all-K-corpus-20261120T110616Z/summary.json", "K256-corpus-ownership.json")
    include(ROOT / ".coord/attention-fp8-reuse-operator-20261120T104637Z/attention-FP8-reuse-probe.json", "attention-FP8-reuse-probe.json")
    include(ROOT / ".coord/attention-fp8-reuse-operator-20261120T104637Z/summary.json", "attention-probe-ownership.json")
    for name in ("build-original-consumer-k256-L1-variants.stderr.log", "build-original-consumer-k256-L1-variants.stdout.log"):
        include(ROOT / ".coord/consumer-variants-build-failure-20261120T095625Z" / name, "negative-results/" + name)
    include(ROOT / ".coord/consumer-k256-L1-real-comparison.stderr.log", "negative-results/initial-row-grid-probe-mismatch.stderr.log")
    for name in ("new-lease-safety-20261120T094505Z.json", "new-lease-safety-20261120T110713Z.json"):
        path = AUDIT / name
        row = json.loads(path.read_bytes())
        row["resident_jenkins_identity_sha256"] = digest(row.pop("jenkins_agent_identity").encode())
        row["original_private_receipt_sha256"] = digest(path.read_bytes())
        row["resident_agent_command_omitted_from_published_bundle"] = True
        relative = "safety/" + name
        staged[relative] = (json.dumps(row, indent=2) + "\n").encode()
        origins[relative] = str(path.relative_to(ROOT))
    release = {"host": "fleet-host-2", "released": True, "detail": "lease dropped",
               "confirmed_by": "ts-fleet-bridge.ts_release after final idle/checkpoint/CANN/CI safety receipt"}
    staged["safety/lease-release.json"] = (json.dumps(release, indent=2) + "\n").encode()
    origins["safety/lease-release.json"] = "confirmed tool result"
    manifest = {"scope": "original K32 consumer grain/L1 and immutable call-scoped attention preparation",
                "K128_to_K256_TPS_ratio": comparison["pairs"]["k128_to_k256"]["TPS_ratio"],
                "K256_not_faster_than_native_control": comparison["models"]["k256"]["TPS"] < comparison["models"]["native"]["TPS"],
                "attention_reuse_observed_TPS_ratio": comparison["pairs"]["native_to_native-reuse"]["TPS_ratio"],
                "attention_reuse_effective_speedup_not_established": True,
                "L1_preload_effective_speedup_not_established": True,
                "native_reduction_experiment_not_exact_default": True,
                "original_checkpoint_quantization_and_production_defaults_unchanged": True,
                "all_8_rank_teacher_final_FP32_tensor_artifacts_compared_and_checksum_pinned_not_copied": True,
                "timed_source_snapshots_separate_from_post_trial_metadata_hardening": True,
                "older_bundles_not_rewritten": True,
                "file_count_excluding_manifest": len(staged), "bytes_excluding_manifest": sum(map(len, staged.values())),
                "files": {name: {"sha256": digest(payload), "bytes": len(payload), "source": origins[name]} for name, payload in sorted(staged.items())}}
    DEST.mkdir(exist_ok=False)
    for name, payload in staged.items():
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    with (DEST / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    for name, row in manifest["files"].items():
        if digest((DEST / name).read_bytes()) != row["sha256"]:
            raise RuntimeError("frozen evidence checksum changed")
    print(json.dumps({"bundle": str(DEST.relative_to(ROOT)), "files": len(staged),
                      "bytes": manifest["bytes_excluding_manifest"]}), flush=True)


if __name__ == "__main__":
    main()
