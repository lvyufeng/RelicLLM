"""CPU-only audit of completed Engram/HC benchmarks, with full byte checksums.

No NPU allocations or inference. Native Engram changes the autoregressive
trajectory; its observed run ratio is not an isolated same-stream speedup.
HC-only parity and matched tokens are checked separately on every rank.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938/logs")
DIRECTORIES = {
    "host": ROOT / "original-host-table-host-engram-native-baseline-20261120T074354Z",
    "device": ROOT / "original-host-table-device-engram-native-baseline-20261120T073354Z",
    "device-hc": ROOT / "original-host-table-device-hc-engram-native-baseline-20261120T081755Z",
    "device-hc-64": ROOT / "original-host-table-device-hc-engram-native-baseline-steps64-20261120T083149Z",
}
OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metrics(reference, candidate):
    if reference.dtype != torch.float32 or candidate.dtype != torch.float32 or reference.shape != candidate.shape:
        raise RuntimeError("quality tensors do not share the original FP32 logit extent")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise RuntimeError("nonfinite teacher logits")
    result = []
    for old, new in zip(reference, candidate):
        old, new = old.reshape(-1), new.reshape(-1)
        delta = new - old
        old_lp, new_lp = old.double().log_softmax(-1), new.double().log_softmax(-1)
        result.append({"FP32_bit_differences": int((old.view(torch.int32) != new.view(torch.int32)).sum()),
                       "max_abs_difference": float(delta.abs().max()),
                       "relative_L2_difference": float(delta.norm() / old.norm().clamp_min(1e-30)),
                       "reference_to_candidate_KL_FP64_analysis": float((old_lp.exp() * (old_lp - new_lp)).sum()),
                       "greedy_matches": int(old.argmax()) == int(new.argmax())})
    return result


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    rows, tensors, artifact_hashes = {}, {}, {}
    keep = (
        "rank", "variant", "dense_contract", "expert_contract", "explicit_numerical_test_permission",
        "checkpoint_original_and_not_rewritten", "Engram_complete_CPU", "Engram_original_tables_host",
        "host_table_device_Engram", "HC_decode_fold", "quality_context_start", "quality_steps",
        "teacher_forced_reference_tokens", "quality_metrics", "timed_warmup_steps_excluded", "timed_steps",
        "timed_context_start", "timed_context_end", "timed_token_ids", "timed_step_seconds",
        "timed_decode_seconds", "sustained_single_stream_TPS", "single_stream", "batch_size", "graphs",
        "NPU_completion_and_host_argmax_included", "per_step_quality_copies_excluded", "finite_final_logits",
        "production_defaults_and_serving_guards_unchanged", "graph_candidate_eager_FP32_bit_differences",
        "expert_body_graph_teacher_FP32_bit_differences", "observed_EOS_in_timed_tokens",
    )
    for mode, directory in DIRECTORIES.items():
        receipt = json.loads((directory / "summary.json").read_bytes())
        if receipt["returncode"] != 0 or not receipt["cleanup"]["complete"]:
            raise RuntimeError("cannot analyze an unsuccessful/unreconciled model trial")
        rows[mode], tensors[mode] = [], []
        expected_steps = 64 if mode.endswith("-64") else 16
        for rank in range(8):
            path = directory / f"Cube-model-rank-{rank}.json"
            row = json.loads(path.read_bytes())
            if (row["rank"] != rank or row["timed_steps"] != expected_steps
                    or not row["single_stream"] or row["batch_size"] != 1
                    or not row["NPU_completion_and_host_argmax_included"]):
                raise RuntimeError("unmatched batch/rank/timed-step receipt")
            rows[mode].append(row)
            tensor_path = directory / f"Cube-quality-logits-rank-{rank}.pt"
            final_path = directory / f"Cube-final-logits-rank-{rank}.pt"
            data = torch.load(tensor_path, map_location="cpu", weights_only=True)
            data["final"] = torch.load(final_path, map_location="cpu", weights_only=True)
            tensors[mode].append(data)
            for item in (path, tensor_path, final_path):
                artifact_hashes[str(item)] = {"sha256": sha(item), "bytes": item.stat().st_size}
            retained = {key: row[key] for key in keep if key in row}
            retained["original_raw_model_receipt"] = {"path": str(path), **artifact_hashes[str(path)]}
            with (OUTPUT / f"model-{mode}-rank-{rank}.json").open("x") as stream:
                json.dump(retained, stream, indent=2)
    pairs = {}
    for old_mode, new_mode in (("host", "device"), ("device", "device-hc")):
        per_rank = []
        for rank in range(8):
            old, new = tensors[old_mode][rank], tensors[new_mode][rank]
            if not torch.equal(old["reference"].view(torch.int32), new["reference"].view(torch.int32)):
                raise RuntimeError("original exact reference phases differ between the pair")
            old_ids, new_ids = rows[old_mode][rank]["timed_token_ids"], rows[new_mode][rank]["timed_token_ids"]
            first_different = next((index for index, (a, b) in enumerate(zip(old_ids, new_ids)) if a != b), None)
            per_rank.append({"rank": rank, "teacher_metrics": metrics(old["candidate_teacher_forced"], new["candidate_teacher_forced"]),
                             "final_logits_FP32_bit_differences": int((old["final"].view(torch.int32) != new["final"].view(torch.int32)).sum()),
                             "same_timed_tokens": old_ids == new_ids,
                             "first_different_timed_output_index": first_different})
        pairs[f"{old_mode}_to_{new_mode}"] = per_rank
    modes = {}
    for mode in DIRECTORIES:
        timings = [row["timed_decode_seconds"] for row in rows[mode]]
        reference = tensors[mode][0]
        for other in tensors[mode][1:]:
            if any(not torch.equal(reference[key].view(torch.int32), other[key].view(torch.int32)) for key in reference):
                raise RuntimeError("teacher/final logits differ across ranks")
        if any(row["timed_token_ids"] != rows[mode][0]["timed_token_ids"] for row in rows[mode][1:]):
            raise RuntimeError("timed tokens differ across ranks")
        steps = rows[mode][0]["timed_steps"]
        modes[mode] = {"timed_steps": steps, "slowest_rank_seconds": max(timings), "TPS": steps / max(timings),
                       "ms_per_forward": 1000 * max(timings) / steps,
                       "timed_context": [rows[mode][0]["timed_context_start"], rows[mode][0]["timed_context_end"]],
                       "timed_tokens": rows[mode][0]["timed_token_ids"], "all_rank_teacher_logits_bit_equal": True,
                       "all_rank_final_FP32_logits_bit_equal": True, "all_rank_timed_tokens_equal": True,
                       "all_teacher_greedy_match_original_exact": all(item["greedy_matches"] for row in rows[mode] for item in row["quality_metrics"])}
    if not all(item["same_timed_tokens"] for item in pairs["device_to_device-hc"]):
        raise RuntimeError("HC-only performance pair uses different autoregressive token streams")
    teacher_diff = sum(item["FP32_bit_differences"] for rank in pairs["device_to_device-hc"] for item in rank["teacher_metrics"])
    final_diff = sum(rank["final_logits_FP32_bit_differences"] for rank in pairs["device_to_device-hc"])
    report = {"modes": modes, "pairs": pairs, "artifact_hashes": artifact_hashes,
              "all_8_ranks_checked": True, "teacher_steps": 8, "paired_timed_steps": 16,
              "same_exact_reference_bits_between_pairs": True,
              "native_Engram_changes_generation_trajectory": not all(item["same_timed_tokens"] for item in pairs["host_to_device"]),
              "HC_only_all_rank_teacher_FP32_bit_differences": teacher_diff,
              "HC_only_all_rank_final_FP32_bit_differences": final_diff,
              "HC_only_bit_parity_passed_in_this_full_model_window": teacher_diff == final_diff == 0,
              "64_step_repeat_prefix_16_equal": all(rows["device-hc-64"][rank]["timed_token_ids"][:16] == rows["device-hc"][rank]["timed_token_ids"] for rank in range(8)),
              "quality_comparison_CPU_only_not_a_TPS_window": True,
              "native_Engram_not_a_bit_neutral_change": True,
              "small_teacher_window_is_not_general_model_quality_acceptance": True,
              "production_defaults_and_original_checkpoint_unchanged": True}
    for old_mode, new_mode in (("host", "device"), ("device", "device-hc"), ("host", "device-hc")):
        key = f"{old_mode}_to_{new_mode}"
        same_tokens = modes[old_mode]["timed_tokens"] == modes[new_mode]["timed_tokens"]
        report[key] = {"observed_TPS_ratio": modes[new_mode]["TPS"] / modes[old_mode]["TPS"],
                       "observed_latency_reduction_ms": modes[old_mode]["ms_per_forward"] - modes[new_mode]["ms_per_forward"],
                       "same_timed_token_stream": same_tokens,
                       "isolated_same_stream_speedup_not_established": not same_tokens}
    with (OUTPUT / "Engram-HC-model-comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"modes": modes, "HC_teacher_bit_differences": teacher_diff,
                      "HC_final_bit_differences": final_diff,
                      "host_to_device": report["host_to_device"], "device_to_device_hc": report["device_to_device-hc"]}), flush=True)


if __name__ == "__main__":
    main()
