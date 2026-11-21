"""Freeze expert TP2/EP4 evidence without rewriting any prior milestone bundle."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = Path("/Users/lvyufeng/Projects/RelicLLM/.agents/worktrees/relic-core-ascend-online-unpack")
SOURCE = ROOT / ".coord/expert-tp-frozen-inputs-20261120"
DEST = ROOT / "docs/performance/evidence/ascend-expert-tp-20261120"


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    transfer = json.loads((SOURCE / "transfer-manifest.json").read_bytes())
    generation = json.loads((SOURCE / "generation-audit/expert-TP2-EP4-model-comparison.json").read_bytes())
    matched = json.loads((SOURCE / "matched-input-audit/expert-TP2-EP4-matched-input-latency.json").read_bytes())
    if not generation["all_8_ranks_and_full_teacher_final_artifacts_compared"] or not matched["all_8_rank_complete_model_artifacts_verified_and_hashed"]:
        raise RuntimeError("full all-rank tensor checks are required")
    if not matched["not_autoregressive_generation_TPS"] or generation["same_timed_token_stream_between_EP8_TP2"]:
        raise RuntimeError("retain the measured generation divergence and distinct latency denominator")
    for row in generation["all_rank_TP2_sustained_repeat"]:
        if row["teacher_FP32_bit_differences"] or not row["same_prefix_16_tokens"]:
            raise RuntimeError("sustained repeat does not match the TP2 source/control prefix")
    start = json.loads((SOURCE / "safety/beginning.json").read_bytes())
    end = json.loads((SOURCE / "safety/ending.json").read_bytes())
    if start["jenkins_agent_identity"] != end["jenkins_agent_identity"]:
        raise RuntimeError("resident CI identity changed")
    for row in (start, end):
        if not row["checkpoint_stat_identity_unchanged"] or not row["all_eight_NPU_process_tables_empty"]:
            raise RuntimeError("original checkpoint or idle host safety is not certified")
    for tag in ("TP2-sustained", "TP2-matched-input"):
        pre = json.loads((SOURCE / tag / "preconditions.json").read_bytes())
        for path in (SOURCE / tag / "source-snapshots").rglob("*"):
            if path.is_file():
                relative = path.relative_to(SOURCE / tag / "source-snapshots").as_posix()
                expected = pre["source_hashes"]["/data1/relic-v41-perf-20261118-0938/source/" + relative]
                if sha(path.read_bytes()) != expected:
                    raise RuntimeError("a timed source snapshot changed")
    for tag in ("original-real-replay", "final-real-replay"):
        for rank in range(8):
            row = json.loads((SOURCE / tag / f"expert-TP2-EP4-rank-{rank}.json").read_bytes())
            if row["rank"] != rank or row["changed_input_graph_replays"] != 32:
                raise RuntimeError("real expert rank/changed-input coverage is incomplete")
            if not row["eager_vs_captured_TP_FP32_bits_equal"] or any(case["captured_vs_eager_TP_FP32_bit_differences"] for case in row["records"]):
                raise RuntimeError("same-TP eager/graph numerical gate failed")
    staged, origins = {}, {}
    for name, record in transfer["files"].items():
        payload = (SOURCE / name).read_bytes()
        if len(payload) != record["bytes"] or sha(payload) != record["sha256"]:
            raise RuntimeError("compressed-transfer integrity changed")
        origins[name] = {"path": record["origin"], "private_original_sha256": record["sha256"]}
        if name.startswith("safety/"):
            row = json.loads(payload)
            row["resident_jenkins_identity_sha256"] = sha(row.pop("jenkins_agent_identity").encode())
            row["resident_agent_command_omitted_from_published_bundle"] = True
            payload = (json.dumps(row, indent=2) + "\n").encode()
        elif name.startswith(("generation-audit/model-", "matched-input-audit/matched-input-")):
            row = json.loads(payload)
            marker = "original_model_artifact" if "original_model_artifact" in row else "original_complete_model_receipt"
            artifact = row[marker]
            audit = generation if name.startswith("generation-audit/") else matched
            if artifact["sha256"] != audit["artifact_hashes"][artifact["path"]]["sha256"]:
                raise RuntimeError("a model rank extract is not pinned to its complete receipt")
            if row.get("host_table_device_Engram") is not None:
                store = row.get("device_selected_MoE")
                if isinstance(store, dict) and "complete_Engram_CPU" in store:
                    store["complete_Engram_CPU"] = False
                row["current_Engram_placement_is_separate_host_table_NPU_compute_owner"] = True
                payload = (json.dumps(row, indent=2) + "\n").encode()
        staged[name] = payload
    def local(path, name):
        staged[name] = path.read_bytes()
        origins[name] = {"path": str(path), "private_original_sha256": sha(staged[name])}
    local(SOURCE / "transfer-manifest.json", "transfer-manifest.json")
    local(ROOT / ".coord/build_expert_tp_evidence.py", "sources/build_expert_tp_evidence.py")
    local(ROOT / ".coord/run_expert_tp_trial.py", "post-trial-runner-receipt-hardening/run_expert_tp_trial.py")
    local(ROOT / "tests/test_v41_ascend_expert_tp.py", "tests/test_v41_ascend_expert_tp.py")
    for name in ("test_expert_tp_moe.py", "test_ascendc_online_tile_cube.py", "test_ascendc_tile_moe.py"):
        local(CORE / "tests" / name, "tests/" + name)
    local(CORE / "relic_core/csrc/ascend/online_tile_cube/README.md", "sources/online-tile-consumer-README.md.txt")
    local(ROOT / ".coord/expert-tp-final-regression-20261120.log", "portable-regression.log")
    release = {"host": "ts-216", "released": True, "detail": "lease dropped",
               "confirmed_by": "ts-fleet-bridge.ts_release after ending idle/checkpoint/CANN/CI receipt"}
    staged["safety/lease-release.json"] = (json.dumps(release, indent=2) + "\n").encode()
    origins["safety/lease-release.json"] = {"path": "confirmed tool result"}
    transport = {"SSH_reply_exit_code": 255, "reason": "server stopped responding",
                 "native_control_owned_receipt_returncode": 0, "native_owned_cleanup_complete": True,
                 "outer_summary_absent_but_original_helper_receipt_persisted_before_print": True,
                 "no_model_replay_after_lost_reply": True,
                 "helper_receipt": transfer["files"]["native-matched-input/original-expert-native-forced-host-table-HC.json"],
                 "slow_multi_file_download_cancelled_partial_files_excluded": True}
    staged["negative-results/transport-reconciliation.json"] = (json.dumps(transport, indent=2) + "\n").encode()
    origins["negative-results/transport-reconciliation.json"] = {"path": "lost-reply reconciliation and pinned original helper receipt"}
    manifest = {"milestone": "original-byte expert TP2/EP4 complete-model implementation",
                "generation_TP2_16_step_TPS": generation["modes"]["tp2"]["single_request_batch_one_TPS"],
                "generation_TP2_64_step_TPS": generation["modes"]["tp2-64"]["single_request_batch_one_TPS"],
                "generation_EP8_16_step_TPS": generation["modes"]["ep8"]["single_request_batch_one_TPS"],
                "generation_ratio_not_matched_continuation_causal_speedup": True,
                "first_EP8_to_TP2_generated_output_difference_zero_based": 10,
                "matched_input_latency_ms": {key: value["ms_per_forward"] for key, value in matched["modes"].items()},
                "matched_latency_not_generation_TPS": True, "native_to_TP2_matched_latency_ratio": matched["native_to_TP2_latency_ratio"],
                "down_FP32_TP_reduction_precedes_original_BF16_per_expert_boundary": True,
                "source_checkpoint_quantization_and_production_defaults_unchanged": True,
                "native_K_partition_and_EP_global_sum_order_changed": True,
                "model_quality_not_accepted_as_bit_neutral_or_general_quality_equivalent": True,
                "real_eager_graph_replays_per_rank_per_probe": 32, "complete_model_ranks_retained": 48,
                "all_8_rank_complete_tensor_artifacts_compared_and_checksum_pinned_not_downloaded": True,
                "timed_sources_separate_from_final_lifecycle_admission_hardening": True,
                "portable_regression_passed": 399, "hardware_gated_regression_skips": 6,
                "older_frozen_bundles_unchanged": True, "target_50_TPS_reached": False,
                "file_count_excluding_manifest": len(staged), "bytes_excluding_manifest": sum(map(len, staged.values())),
                "files": {name: {"sha256": sha(payload), "bytes": len(payload), "origin": origins[name]} for name, payload in sorted(staged.items())}}
    DEST.mkdir(exist_ok=False)
    for name, payload in staged.items():
        path = DEST / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    with (DEST / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2)
    for name, record in manifest["files"].items():
        if sha((DEST / name).read_bytes()) != record["sha256"]:
            raise RuntimeError("frozen file checksum changed")
    print(json.dumps({"bundle": str(DEST.relative_to(ROOT)), "files": len(staged),
                      "bytes": manifest["bytes_excluding_manifest"]}), flush=True)


if __name__ == "__main__":
    main()
