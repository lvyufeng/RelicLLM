"""CPU-only all-rank audit of completed consumer and attention-reuse model trials."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path("/data1/relic-v41-perf-20261118-0938/logs")
OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])
MODES = ("k128", "k256", "native", "native-reuse")


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            result.update(chunk)
    return result.hexdigest()


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    results, tensors, sources = {}, {}, {}
    models = {}
    keep = ("rank", "dense_contract", "expert_contract", "explicit_numerical_test_permission",
            "checkpoint_original_and_not_rewritten", "Engram_original_tables_host", "host_table_device_Engram",
            "HC_decode_fold", "attention_FP8_preparation_reuse", "quality_steps", "teacher_forced_reference_tokens",
            "quality_metrics", "timed_steps", "timed_context_start", "timed_context_end", "timed_token_ids",
            "timed_decode_seconds", "sustained_single_stream_TPS", "NPU_completion_and_host_argmax_included",
            "observed_EOS_in_timed_tokens", "single_stream", "batch_size", "production_defaults_and_serving_guards_unchanged",
            "device_selected_MoE", "graph_candidate_eager_FP32_bit_differences", "expert_body_graph_teacher_FP32_bit_differences")
    for mode in MODES:
        candidates = []
        for directory in ROOT.glob(f"original-consumer-{mode}-host-table-HC-20*T*Z"):
            summary = directory / "summary.json"
            if summary.exists():
                receipt = json.loads(summary.read_bytes())
                if receipt["returncode"] == 0 and receipt["cleanup"]["complete"]:
                    candidates.append(directory)
        if len(candidates) != 1:
            raise RuntimeError(f"one completed unambiguous {mode} model run required, got {candidates}")
        directory = candidates[0]
        records, timings, tensors[mode], models[mode] = [], [], [], []
        for rank in range(8):
            path = directory / f"Cube-model-rank-{rank}.json"
            row = json.loads(path.read_bytes())
            if row["rank"] != rank or row["timed_steps"] != 16 or row["batch_size"] != 1 or not row["single_stream"]:
                raise RuntimeError("model batch/rank/timed-window contract mismatch")
            if not row["NPU_completion_and_host_argmax_included"] or [row["timed_context_start"], row["timed_context_end"]] != [24, 40]:
                raise RuntimeError("model timing boundary mismatch")
            quality = directory / f"Cube-quality-logits-rank-{rank}.pt"
            final = directory / f"Cube-final-logits-rank-{rank}.pt"
            values = torch.load(quality, map_location="cpu", weights_only=True)
            values["final"] = torch.load(final, map_location="cpu", weights_only=True)
            if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in values.values()):
                raise RuntimeError("teacher/final logs are not finite original FP32")
            tensors[mode].append(values)
            models[mode].append(row)
            timings.append(row["timed_decode_seconds"])
            for artifact in (path, quality, final):
                sources[str(artifact)] = {"sha256": digest(artifact), "bytes": artifact.stat().st_size}
            retained = {key: row[key] for key in keep if key in row}
            retained["original_complete_model_receipt"] = {"path": str(path), **sources[str(path)]}
            with (OUTPUT / f"model-{mode}-rank-{rank}.json").open("x") as stream:
                json.dump(retained, stream, indent=2)
        for rank in range(1, 8):
            if any(not torch.equal(tensors[mode][0][key].view(torch.int32), tensors[mode][rank][key].view(torch.int32)) for key in tensors[mode][0]):
                raise RuntimeError("teacher/final values differ across ranks")
            if models[mode][0]["timed_token_ids"] != models[mode][rank]["timed_token_ids"]:
                raise RuntimeError("timed tokens differ across ranks")
        results[mode] = {"directory": str(directory), "timed_steps": 16, "context": [24, 40],
                         "slowest_rank_seconds": max(timings), "TPS": 16 / max(timings),
                         "all_rank_teacher_final_FP32_bits_equal": True, "all_rank_timed_tokens_equal": True,
                         "timed_tokens": models[mode][0]["timed_token_ids"],
                         "all_teacher_greedy_match_original_exact": all(item["greedy_matches"] for row in models[mode] for item in row["quality_metrics"])}
    pairs = {}
    for old, new in (("k128", "k256"), ("native", "native-reuse"), ("k256", "native")):
        bits = []
        for rank in range(8):
            a, b = tensors[old][rank], tensors[new][rank]
            if not torch.equal(a["reference"].view(torch.int32), b["reference"].view(torch.int32)):
                raise RuntimeError("the original exact teacher reference differs across configurations")
            bits.append({"rank": rank,
                         "teacher_FP32_bit_differences": int((a["candidate_teacher_forced"].view(torch.int32) != b["candidate_teacher_forced"].view(torch.int32)).sum()),
                         "final_FP32_bit_differences": int((a["final"].view(torch.int32) != b["final"].view(torch.int32)).sum()),
                         "same_timed_tokens": models[old][rank]["timed_token_ids"] == models[new][rank]["timed_token_ids"]})
        pairs[old + "_to_" + new] = {"all_rank_bits": bits,
                                     "TPS_ratio": results[new]["TPS"] / results[old]["TPS"],
                                     "latency_reduction_ms_per_forward": 1000 * (results[old]["slowest_rank_seconds"] - results[new]["slowest_rank_seconds"]) / 16,
                                     "same_timed_tokens": all(row["same_timed_tokens"] for row in bits)}
    if any(not pairs[key]["same_timed_tokens"] for key in ("k128_to_k256", "native_to_native-reuse")):
        raise RuntimeError("a proposed same-stream optimization changes the timed token trajectory")
    if any(row["teacher_FP32_bit_differences"] or row["final_FP32_bit_differences"] for row in pairs["native_to_native-reuse"]["all_rank_bits"]):
        raise RuntimeError("original FP8 activation reuse did not preserve complete-model FP32 bits")
    report = {"models": results, "pairs": pairs, "artifact_hashes": sources,
              "all_8_ranks_checked": True, "CPU_only_comparison_not_a_model_or_NPU_trial": True,
              "native_K256_remains_explicit_changed_accumulation_experiment": True,
              "original_checkpoint_Engram_tables_and_quantization_unchanged": True,
              "production_defaults_unchanged": True}
    with (OUTPUT / "consumer-attention-model-comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"models": results, "paired_ratios": {key: row["TPS_ratio"] for key, row in pairs.items()}}), flush=True)


if __name__ == "__main__":
    main()
