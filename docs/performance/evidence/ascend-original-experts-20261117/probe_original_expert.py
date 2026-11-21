"""Checkpoint-backed bounded expert-forward comparison; no model/TPS claim."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu

from relicllm.models.deepseek_v4_1.loader import CheckpointRoutedExperts, V41Checkpoint
from relicllm.models.deepseek_v4_1.ascend_experts import AscendExpertCache, AscendRoutedExperts

ROOT = Path("/data1/relic-expert-gemm-20261117-1046")
CHECKPOINT = Path("/data1/relic-v41-20261116-210/checkpoint")


def run(store, x, weights, indices):
    if x.device.type == "npu":
        torch.npu.synchronize(x.device)
    begin=time.perf_counter()
    result=store.forward(x,weights,indices)
    if x.device.type == "npu":
        torch.npu.synchronize(x.device)
    return result, time.perf_counter()-begin


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--report",required=True)
    parser.add_argument("--rows",type=int,default=1)
    parser.add_argument("--experts",default="0,127,383")
    parser.add_argument("--iterations",type=int,default=10)
    args=parser.parse_args()
    expert_ids=[int(value) for value in args.experts.split(",")]
    torch.set_num_threads(4)
    torch.npu.set_device(0)
    checkpoint=V41Checkpoint(str(CHECKPOINT))
    first="layers.0.ffn.experts.0.w1.weight"
    shape=checkpoint.reader.entry(first).shape
    inter,dim=shape[0],shape[1]*2
    raw_hashes={}
    for expert_id in expert_ids:
        for which in ("w1","w2","w3"):
            key=f"layers.0.ffn.experts.{expert_id}.{which}.weight"
            scale=f"layers.0.ffn.experts.{expert_id}.{which}.scale"
            for name in (key,scale):
                payload=checkpoint.packed(name).contiguous().view(torch.uint8)
                raw_hashes[name] = {"shape":list(payload.shape),"bytes":payload.numel(),
                                    "sha256":hashlib.sha256(payload.numpy().tobytes()).hexdigest()}
    cpu=CheckpointRoutedExperts(checkpoint,0,n_experts=384,dim=dim,inter_dim=inter,
                                swiglu_limit=10.0,cache_size=len(expert_ids),quantized=True)
    cache=AscendExpertCache(budget_bytes=128*2**20)
    npu=AscendRoutedExperts(checkpoint,0,n_experts=384,dim=dim,inter_dim=inter,topk=len(expert_ids),
                            swiglu_limit=10.,device="npu:0",dtype=torch.bfloat16,
                            cache=cache,backend="triton")
    g=torch.Generator().manual_seed(417)
    x=(torch.randn(args.rows,dim,generator=g)*0.125).to(torch.bfloat16)
    indices=torch.tensor(expert_ids,dtype=torch.int64)[None,:].repeat(args.rows,1)
    weights=torch.tensor([.7/len(expert_ids)]*len(expert_ids),dtype=torch.float32)[None,:].repeat(args.rows,1)
    expected,cpu_seconds=run(cpu,x,weights,indices)
    ndx,nw,ni=x.to("npu:0"),weights.to("npu:0"),indices.to("npu:0")
    actual,cold_seconds=run(npu,ndx,nw,ni)
    actual_cpu=actual.cpu()
    warm,warm_seconds=run(npu,ndx,nw,ni)
    warm_cpu=warm.cpu()
    cpu_times=[];npu_times=[]
    for _ in range(args.iterations):
        reference,seconds=run(cpu,x,weights,indices);cpu_times.append(seconds)
        observed,seconds=run(npu,ndx,nw,ni);npu_times.append(seconds)
        if not torch.equal(reference.view(torch.int32),expected.view(torch.int32)):
            raise RuntimeError("CPU oracle warm repeat changed bits")
        if not torch.equal(observed.cpu().view(torch.int32),actual_cpu.view(torch.int32)):
            raise RuntimeError("NPU warm repeat changed bits")
    source_paths=[ROOT/"source/relic-core/relic_core/kernels"/name for name in
                  ("block_gemm.py","_ascend_block_gemm_kernels.py","block_fp8_quant.py","_ascend_fp8_quant_kernels.py")]
    source_paths += [ROOT/"source/RelicLLM/relicllm/models/deepseek_v4_1"/name for name in
                     ("ascend_experts.py","ascend_linear.py","loader.py","kernels.py","mma.py")]
    source_hashes={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    same_nan=torch.equal(torch.isnan(expected),torch.isnan(actual_cpu))
    finite=~torch.isnan(expected)
    mismatches=int((expected.view(torch.int32)[finite]!=actual_cpu.view(torch.int32)[finite]).sum())
    max_abs=float((expected[finite]-actual_cpu[finite]).abs().max())
    result={"timestamp":dt.datetime.now(dt.timezone.utc).isoformat(),"torch":torch.__version__,
            "torch_npu":torch_npu.__version__,"device":torch.npu.get_device_name(0),
            "checkpoint":str(CHECKPOINT),"rows":args.rows,"expert_ids":expert_ids,"dim":dim,"inter_dim":inter,
            "tested_source_hashes":source_hashes,"warm_iterations":args.iterations,
            "cpu_warm_median_seconds":statistics.median(cpu_times),
            "npu_warm_median_seconds":statistics.median(npu_times),
            "cpu_warm_samples_seconds":cpu_times,"npu_warm_samples_seconds":npu_times,
            "source_payloads":raw_hashes,"cpu_reference":"CheckpointRoutedExperts quantized=True, original E4M3 activation and stored scales",
            "npu_provider":"online original INT8x2 + original half-scale, fused block32 two-half reduction",
            "nan_classification_equal":same_nan,"float32_bit_mismatches":mismatches,
            "elements":expected.numel(),"max_absolute_error":max_abs,
            "warm_equals_cold_bits":torch.equal(actual_cpu.view(torch.int32),warm_cpu.view(torch.int32)),
            "cpu_expert_seconds":cpu_seconds,"npu_cold_expert_seconds":cold_seconds,
            "npu_warm_expert_seconds":warm_seconds,"cache":cache.snapshot(),
            "full_model_forward_claimed":False,"generated_tokens_claimed":False,
            "native_fp4_fp8_arithmetic_claimed":False,"checkpoint_rewritten":False}
    with Path(args.report).open("x") as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result,indent=2),flush=True)
    npu.clear_cache()
    checkpoint.close()
    return 0 if same_nan and mismatches==0 and result["warm_equals_cold_bits"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
