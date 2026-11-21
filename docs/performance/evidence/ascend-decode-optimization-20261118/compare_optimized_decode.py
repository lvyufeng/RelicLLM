"""Compare checkpoint-backed sustained decode runs with identical prompt/context.

Kernel microbenchmarks are never accepted as a model throughput measurement.
The initial control also performed an unused model sample; its timing caveat is
recorded explicitly instead of treating it as a corrected normal-generate run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--optimized", type=Path, required=True)
    args = parser.parse_args()
    controls, optimized = [], []
    for rank in range(8):
        controls.append(json.loads((args.control / f"decode-baseline-rank-{rank}.json").read_text()))
        optimized.append(json.loads((args.optimized / f"decode-baseline-rank-{rank}.json").read_text()))
    comparisons = []
    for rank, (before, after) in enumerate(zip(controls, optimized)):
        for key in ("prompt_tokens", "timed_steps", "context_start", "context_end", "batch_size", "single_stream"):
            if before[key] != after[key]:
                raise RuntimeError(f"rank {rank}: control/optimized {key} differs")
        if before["all_timed_token_ids"] != after["all_timed_token_ids"]:
            raise RuntimeError(f"rank {rank}: actual timed greedy tokens differ")
        old = torch.load(args.control / f"baseline-final-logits-rank-{rank}.pt", weights_only=True)["final_logits"]
        new = torch.load(args.optimized / f"baseline-final-logits-rank-{rank}.pt", weights_only=True)["final_logits"]
        if not bool(torch.isfinite(old).all()) or not bool(torch.isfinite(new).all()):
            raise RuntimeError("non-finite final full-model logits")
        mismatches = int((old.contiguous().view(torch.int32) != new.contiguous().view(torch.int32)).sum())
        if mismatches:
            raise RuntimeError(f"rank {rank}: {mismatches} full-model final FP32 logit bit mismatches")
        stores = after.get("packed_residency", [])
        packed_bytes = sum(store.get("retained_bytes", 0) for store in stores)
        if len(stores) != 40 or packed_bytes != 36_097_228_800:
            raise RuntimeError(f"rank {rank}: packed residency is not all 40 original layers: {len(stores)}, {packed_bytes}")
        states = after.get("host_engram_states", [])
        if len(states) != 2 or not all(state["verified"] for state in states):
            raise RuntimeError("both CPU Engram modules must have passed cross-rank byte verification")
        comparisons.append({"rank": rank, "timed_greedy_token_ids_equal": True,
                            "final_finite_FP32_logit_bit_mismatches": mismatches,
                            "logit_elements": old.numel(), "packed_expert_HBM_bytes": packed_bytes,
                            "CPU_Engram_byte_proof_states": states,
                            "decode_seconds": after["decode_seconds"],
                            "step_median_seconds": statistics.median(after["step_seconds"])})
    control_seconds = max(row["decode_seconds"] for row in controls)
    actual_seconds = max(row["decode_seconds"] for row in optimized)
    steps = optimized[0]["timed_steps"]
    result = {"single_stream": True, "batch_size": 1, "context_start": optimized[0]["context_start"],
              "context_end": optimized[0]["context_end"], "prompt_tokens": optimized[0]["prompt_tokens"],
              "timed_steps": steps, "warmup_and_instrumentation_excluded": True,
              "NPU_completion_and_host_greedy_pick_included": True,
              "slowest_rank_decode_seconds": actual_seconds,
              "sustained_single_stream_TPS": steps / actual_seconds,
              "meets_50_TPS": steps / actual_seconds >= 50,
              "initial_control_TPS": steps / control_seconds,
              "initial_control_timing_caveat": "initial profiler performed an unused model Gumbel sample; corrected optimized driver resets state and sets model.temperature=0 like normal generate",
              "whole_model_speed_ratio_against_initial_control": control_seconds / actual_seconds,
              "rank_comparisons": comparisons}
    with (args.optimized / "verified-optimized-decode-comparison.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
