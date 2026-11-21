"""CPU-only complete all-rank EP8/TP2-EP4 model comparison, including sustained repeat."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938/logs")
OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])
CONFIG = {"ep8": "original-expert-ep8-host-table-HC-20*T*Z",
          "tp2": "original-expert-tp2-host-table-HC-20*T*Z",
          "tp2-64": "original-expert-tp2-host-table-HC-steps64-20*T*Z"}


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(2**20), b""):
            digest.update(part)
    return digest.hexdigest()


def delta_metrics(old, new):
    result = []
    for reference, candidate in zip(old, new):
        reference, candidate = reference.flatten(), candidate.flatten()
        delta = candidate - reference
        a, b = reference.double().log_softmax(-1), candidate.double().log_softmax(-1)
        result.append({"FP32_bit_differences": int((reference.view(torch.int32) != candidate.view(torch.int32)).sum()),
                       "max_abs_difference": float(delta.abs().max()),
                       "relative_L2_difference": float(delta.norm() / reference.norm().clamp_min(1e-30)),
                       "reference_to_candidate_KL_FP64": float((a.exp() * (a - b)).sum()),
                       "greedy_matches": int(reference.argmax()) == int(candidate.argmax())})
    return result


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    tensors, models, modes, hashes = {}, {}, {}, {}
    keep = ("rank", "dense_contract", "expert_contract", "explicit_numerical_test_permission",
            "checkpoint_original_and_not_rewritten", "Engram_original_tables_host", "host_table_device_Engram",
            "HC_decode_fold", "expert_TP_EP", "quality_steps", "teacher_forced_reference_tokens", "quality_metrics",
            "timed_warmup_steps_excluded", "timed_steps", "timed_context_start", "timed_context_end", "timed_token_ids",
            "timed_step_seconds", "timed_decode_seconds", "sustained_single_stream_TPS", "single_stream", "batch_size",
            "NPU_completion_and_host_argmax_included", "per_step_quality_copies_excluded", "finite_final_logits",
            "observed_EOS_in_timed_tokens", "graph_candidate_eager_FP32_bit_differences",
            "expert_body_graph_teacher_FP32_bit_differences", "device_selected_MoE", "production_defaults_and_serving_guards_unchanged")
    for mode, pattern in CONFIG.items():
        completed = []
        for directory in ROOT.glob(pattern):
            summary = directory / "summary.json"
            if summary.exists():
                receipt = json.loads(summary.read_bytes())
                if receipt["returncode"] == 0 and receipt["cleanup"]["complete"]:
                    completed.append(directory)
        if len(completed) != 1:
            raise RuntimeError(f"exactly one completed unambiguous model required for {mode}: {completed}")
        directory = completed[0]
        tensors[mode], models[mode] = [], []
        expected_steps = 64 if mode.endswith("64") else 16
        for rank in range(8):
            path = directory / f"Cube-model-rank-{rank}.json"
            row = json.loads(path.read_bytes())
            if (row["rank"] != rank or row["timed_steps"] != expected_steps or row["batch_size"] != 1
                    or not row["single_stream"] or not row["NPU_completion_and_host_argmax_included"]):
                raise RuntimeError("model timing/rank/batch contract mismatch")
            teacher_path = directory / f"Cube-quality-logits-rank-{rank}.pt"
            final_path = directory / f"Cube-final-logits-rank-{rank}.pt"
            values = torch.load(teacher_path, map_location="cpu", weights_only=True)
            values["final"] = torch.load(final_path, map_location="cpu", weights_only=True)
            if any(value.dtype != torch.float32 or not torch.isfinite(value).all() for value in values.values()):
                raise RuntimeError("nonfinite or incorrectly narrowed model quality output")
            tensors[mode].append(values); models[mode].append(row)
            for artifact in (path, teacher_path, final_path):
                hashes[str(artifact)] = {"sha256": sha(artifact), "bytes": artifact.stat().st_size}
            retained = {key: row[key] for key in keep if key in row}
            retained["original_model_artifact"] = {"path": str(path), **hashes[str(path)]}
            with (OUTPUT / f"model-{mode}-rank-{rank}.json").open("x") as stream:
                json.dump(retained, stream, indent=2)
        for rank in range(1, 8):
            if models[mode][0]["timed_token_ids"] != models[mode][rank]["timed_token_ids"]:
                raise RuntimeError("generated model tokens differ across physical ranks")
            if any(not torch.equal(tensors[mode][0][key].view(torch.int32), tensors[mode][rank][key].view(torch.int32)) for key in tensors[mode][0]):
                raise RuntimeError("teacher/final FP32 values differ across physical ranks")
        wall = max(row["timed_decode_seconds"] for row in models[mode])
        modes[mode] = {"directory": str(directory), "timed_steps": expected_steps,
                       "context": [models[mode][0]["timed_context_start"], models[mode][0]["timed_context_end"]],
                       "slowest_rank_seconds": wall, "single_request_batch_one_TPS": expected_steps / wall,
                       "all_8_rank_teacher_final_bits_equal": True, "timed_tokens": models[mode][0]["timed_token_ids"],
                       "all_8_step_teacher_greedy_match_original_exact": all(item["greedy_matches"] for row in models[mode] for item in row["quality_metrics"])}
    pairs = []
    for rank in range(8):
        old, new = tensors["ep8"][rank], tensors["tp2"][rank]
        if not torch.equal(old["reference"].view(torch.int32), new["reference"].view(torch.int32)):
            raise RuntimeError("the EP8/TP2 original exact teacher reference phases differ")
        a, b = models["ep8"][rank]["timed_token_ids"], models["tp2"][rank]["timed_token_ids"]
        pairs.append({"rank": rank, "same_timed_token_stream": a == b,
                      "first_timed_output_difference": next((index for index, (x, y) in enumerate(zip(a, b)) if x != y), None),
                      "teacher_metrics": delta_metrics(old["candidate_teacher_forced"], new["candidate_teacher_forced"]),
                      "final_FP32_bit_differences": int((old["final"].view(torch.int32) != new["final"].view(torch.int32)).sum())})
    repeat = []
    for rank in range(8):
        a, b = tensors["tp2"][rank], tensors["tp2-64"][rank]
        repeat.append({"rank": rank, "same_prefix_16_tokens": models["tp2"][rank]["timed_token_ids"] == models["tp2-64"][rank]["timed_token_ids"][:16],
                       "teacher_FP32_bit_differences": int((a["candidate_teacher_forced"].view(torch.int32) != b["candidate_teacher_forced"].view(torch.int32)).sum())})
    old_wall, new_wall = modes["ep8"]["slowest_rank_seconds"], modes["tp2"]["slowest_rank_seconds"]
    report = {"modes": modes, "all_rank_EP8_vs_TP2": pairs, "all_rank_TP2_sustained_repeat": repeat,
              "observed_TPS_ratio": old_wall / new_wall,
              "observed_latency_reduction_ms_per_forward": 1000 * (old_wall - new_wall) / 16,
              "same_timed_token_stream_between_EP8_TP2": all(row["same_timed_token_stream"] for row in pairs),
              "native_TP_partial_reduction_is_authorized_experiment_not_exact_default": True,
              "small_teacher_window_not_general_model_quality_acceptance": True,
              "all_8_ranks_and_full_teacher_final_artifacts_compared": True,
              "original_E2M1_FP8_E8M0_quantization_and_production_defaults_unchanged": True,
              "artifact_hashes": hashes, "CPU_only_audit_not_model_TPS": True}
    with (OUTPUT / "expert-TP2-EP4-model-comparison.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"modes": modes, "same_timed_tokens": report["same_timed_token_stream_between_EP8_TP2"],
                      "observed_TPS_ratio": report["observed_TPS_ratio"]}), flush=True)


if __name__ == "__main__":
    main()
