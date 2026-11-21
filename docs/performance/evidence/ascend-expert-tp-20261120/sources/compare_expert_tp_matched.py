"""Audit full-model latency on identical input tokens; never label it generation TPS."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938/logs")
OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])


def main():
    torch.set_num_threads(4)
    modes, hashes = {}, {}
    fixed_inputs = reference = None
    for mode in ("tp2", "ep8", "native"):
        directories = []
        for path in ROOT.glob(f"original-expert-{mode}-forced-host-table-HC-20*T*Z"):
            receipt = path / "summary.json"
            if not receipt.exists():
                # The owned helper persists this receipt before printing its
                # result. A lost SSH reply can prevent the outer summary only;
                # never fabricate it or replay the completed model.
                receipt = path / f"original-expert-{mode}-forced-host-table-HC.json"
            if receipt.exists():
                status = json.loads(receipt.read_bytes())
                if status["returncode"] == 0 and status["cleanup"]["complete"]:
                    directories.append(path)
        if len(directories) != 1:
            raise RuntimeError(f"require one completed {mode} matched-input trial, got {directories}")
        directory = directories[0]
        rows, finals, teachers = [], [], []
        for rank in range(8):
            path = directory / f"Cube-model-rank-{rank}.json"
            row = json.loads(path.read_bytes())
            if (row["rank"] != rank or not row["same_token_input_latency_not_generation_TPS"]
                    or row["timed_steps"] != 16 or row["timed_context_start"] != 24
                    or row["timed_context_end"] != 40 or row["timed_warmup_steps_excluded"] != 4
                    or row["batch_size"] != 1 or not row["NPU_completion_and_host_argmax_included"]
                    or "sustained_single_stream_TPS" in row):
                raise RuntimeError("matched-input forward latency must not be called generation TPS")
            if fixed_inputs is None:
                fixed_inputs = row["fixed_input_token_ids"]
            if row["fixed_input_token_ids"] != fixed_inputs:
                raise RuntimeError("matched-input conditioning tokens differ")
            preconditions = json.loads((directory / "preconditions.json").read_bytes())
            if not preconditions["environment"].get("RELIC_TP_MATCHED_TOKEN_INPUTS") == "1":
                raise RuntimeError("the recorded conditioning override was not explicitly selected")
            teacher = torch.load(directory / f"Cube-quality-logits-rank-{rank}.pt", map_location="cpu", weights_only=True)
            final = torch.load(directory / f"Cube-final-logits-rank-{rank}.pt", map_location="cpu", weights_only=True)
            if not all(torch.isfinite(value).all() and value.dtype == torch.float32 for value in [final, *teacher.values()]):
                raise RuntimeError("nonfinite matched-input model output")
            for artifact in (path, directory / f"Cube-quality-logits-rank-{rank}.pt", directory / f"Cube-final-logits-rank-{rank}.pt"):
                hashes[str(artifact)] = {"bytes": artifact.stat().st_size, "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
            keep = ("rank", "dense_contract", "expert_contract", "host_table_device_Engram", "HC_decode_fold",
                    "expert_TP_EP", "quality_steps", "quality_metrics", "graph_candidate_eager_FP32_bit_differences",
                    "timed_steps", "timed_context_start", "timed_context_end", "timed_warmup_steps_excluded", "timed_decode_seconds",
                    "single_request_forward_rate_not_generation_TPS", "same_token_input_latency_not_generation_TPS", "fixed_input_token_ids",
                    "timed_token_ids", "NPU_completion_and_host_argmax_included", "batch_size", "single_stream",
                    "production_defaults_and_serving_guards_unchanged")
            retained = {key: row[key] for key in keep if key in row}
            retained["original_complete_model_receipt"] = {"path": str(path), **hashes[str(path)]}
            with (OUTPUT / f"matched-input-{mode}-rank-{rank}.json").open("x") as stream:
                json.dump(retained, stream, indent=2)
            rows.append(row); finals.append(final); teachers.append(teacher)
        if any(not torch.equal(finals[0].view(torch.int32), value.view(torch.int32)) for value in finals[1:]):
            raise RuntimeError("matched-input final values differ across physical ranks")
        if any(any(not torch.equal(teachers[0][key].view(torch.int32), value[key].view(torch.int32)) for key in teachers[0]) for value in teachers[1:]):
            raise RuntimeError("matched-input teacher tensors differ across physical ranks")
        if reference is None:
            reference = teachers[0]["reference"]
        elif not torch.equal(reference.view(torch.int32), teachers[0]["reference"].view(torch.int32)):
            raise RuntimeError("matched controls do not share the original exact prefill/reference")
        wall = max(row["timed_decode_seconds"] for row in rows)
        modes[mode] = {"directory": str(directory), "same_conditioning_input_tokens": fixed_inputs,
                       "slowest_rank_seconds": wall, "ms_per_forward": wall * 1000 / 16,
                       "all_8_rank_teacher_and_final_FP32_bits_equal": True,
                       "original_exact_teacher_8_greedy_tokens_match": all(item["greedy_matches"] for row in rows for item in row["quality_metrics"])}
    result = {"modes": modes, "scope": "16 full-model synchronized batch-one forwards, fixed input tokens at context24..40",
              "not_autoregressive_generation_TPS": True, "same_prompt_prefill_and_conditioning_token_inputs": True,
              "native_numerical_experiment_not_exact_default": True, "EP8_to_TP2_latency_ratio": modes["ep8"]["slowest_rank_seconds"] / modes["tp2"]["slowest_rank_seconds"],
              "native_to_TP2_latency_ratio": modes["native"]["slowest_rank_seconds"] / modes["tp2"]["slowest_rank_seconds"],
              "EP8_to_TP2_ms_saved_per_forward": modes["ep8"]["ms_per_forward"] - modes["tp2"]["ms_per_forward"],
              "native_to_TP2_ms_saved_per_forward": modes["native"]["ms_per_forward"] - modes["tp2"]["ms_per_forward"],
              "all_8_rank_complete_model_artifacts_verified_and_hashed": True, "artifact_hashes": hashes}
    with (OUTPUT / "expert-TP2-EP4-matched-input-latency.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "artifact_hashes"}), flush=True)


if __name__ == "__main__":
    main()
