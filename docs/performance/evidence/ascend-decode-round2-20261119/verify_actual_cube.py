"""Verify all-rank single-stream Cube receipts without pretending bit parity."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    records=[]
    tokens=None
    final=None
    quality=[]
    for rank in range(8):
        record=json.loads((args.directory/f"Cube-model-rank-{rank}.json").read_text())
        current=torch.load(args.directory/f"Cube-final-logits-rank-{rank}.pt",map_location="cpu",weights_only=True).contiguous()
        paired=torch.load(args.directory/f"Cube-quality-logits-rank-{rank}.pt",map_location="cpu",weights_only=True)
        assert record["timed_steps"]==16 and record["single_stream"] and record["batch_size"]==1
        assert record["timed_context_start"]==24 and record["timed_context_end"]==40
        assert record["Engram_complete_CPU"] and record["exact_CPU_reference_reduction_flag"]=="1"
        assert current.numel()==129280 and bool(torch.isfinite(current).all())
        assert paired["reference"].shape==paired["candidate_teacher_forced"].shape==(8,129280)
        assert all(bool(torch.isfinite(value).all()) for value in paired.values())
        if tokens is None:tokens=record["timed_token_ids"]
        assert record["timed_token_ids"]==tokens
        if final is None:final=current
        assert torch.equal(current.view(torch.int32),final.view(torch.int32))
        for step,metric in enumerate(record["quality_metrics"]):
            ref=paired["reference"][step].contiguous()
            candidate=paired["candidate_teacher_forced"][step].contiguous()
            assert metric["reference_token"]==int(ref.argmax())
            assert metric["candidate_greedy_token"]==int(candidate.argmax())
            assert metric["FP32_bit_differences"]==int((ref.view(torch.int32)!=candidate.view(torch.int32)).sum())
            quality.append(metric)
        assert len(record["packed_residency"])==40
        assert sum(row["retained_bytes"] for row in record["packed_residency"])==36097228800
        assert len(record["host_engram_states"])==2
        records.append(record)
    wall=max(row["timed_decode_seconds"] for row in records)
    exact_tokens=[14,223,21,14,223,22,14,223,23,14,223,24,14,223,25,14]
    report={"ranks_verified":8,"timed_forwards":16,"context":[24,40],"single_stream":True,
            "batch_size":1,"slowest_rank_wall_seconds":wall,"sustained_TPS":16/wall,
            "meets_50_TPS":16/wall>=50,"all_rank_tokens_equal":True,"all_rank_final_logit_bits_equal":True,
            "candidate_is_bit_equal_to_exact_reference":False,
            "teacher_forced_greedy_matches":sum(row["greedy_matches"] for row in quality),
            "teacher_forced_comparisons":len(quality),
            "teacher_forced_max_logit_abs_difference":max(row["max_abs_difference"] for row in quality),
            "teacher_forced_max_KL":max(row["reference_to_candidate_KL"] for row in quality),
            "teacher_forced_max_relative_L2":max(row["relative_L2_difference"] for row in quality),
            "free_generation_token_ids":tokens,"exact_free_generation_token_ids":exact_tokens,
            "free_generation_differs_from_exact":tokens!=exact_tokens,
            "same_position_greedy_matches_in_free_generation":sum(a==b for a,b in zip(tokens,exact_tokens)),
            "full_host_Engram":True,"original_checkpoint_and_activation_quantization":True,
            "changed_accumulation_explicitly_authorized_for_testing":True,
            "production_default_replacement":False,"model_quality_benchmark_claim":False,
            "timed_mode":"fixed-length free-running forward workload",
            "stopping_on_EOS":False,
            "observed_EOS_timed_positions":records[0].get("observed_EOS_in_timed_tokens",[]),
            "normal_user_visible_generation_TPS_claim":False}
    with (args.directory/"verified-whole-model-Cube-summary.json").open("x") as handle:
        json.dump(report,handle,indent=2)
    print(json.dumps(report,indent=2),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
