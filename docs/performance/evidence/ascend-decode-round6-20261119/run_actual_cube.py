"""Bound the explicitly authorized full-model NPU-only accumulation experiment."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path("/data1/relic-v41-perf-20261118-0938")
LEASE_END = dt.datetime.fromisoformat("2026-11-19T13:54:08+00:00")
LIMIT = 900


def main():
    if dt.datetime.now(dt.timezone.utc)+dt.timedelta(seconds=LIMIT+120) >= LEASE_END:
        raise RuntimeError("not enough time remains on the live lease for this bounded experiment")
    safety = ROOT/"source/bounded_task.py"
    if hashlib.sha256(safety.read_bytes()).hexdigest() != "7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a":
        raise RuntimeError("unreviewed owned-process helper")
    specification = importlib.util.spec_from_file_location("Cube_full_model_safety", safety)
    helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helper)
    cube = ROOT/"source/RelicLLM/relicllm/models/deepseek_v4_1/cube_experiment.py"
    if hashlib.sha256(cube.read_bytes()).hexdigest() != "f4a4f4d745277861a982053f023a45292e2087d8df577c01f476bdc8d411e536":
        raise RuntimeError("the hardware-accepted Cube binding changed")
    variant = os.environ.get("RELIC_CUBE_VARIANT", "weight_quant")
    suffix = "-host-overlap" if os.environ.get("RELIC_HOST_ENGRAM_PREFETCH") == "1" else ""
    if os.environ.get("RELIC_CUBE_NPUGRAPH") == "1": suffix += "-graphs"
    if os.environ.get("RELIC_CUBE_PROFILE") == "1": suffix += "-profile"
    if int(os.environ.get("RELIC_CUBE_WEIGHT_CACHE_BYTES","0")) > 0: suffix += "-operand-cache"
    if os.environ.get("RELIC_HC_FUSED_NORMALIZE") == "1": suffix += "-HC-normalize"
    if os.environ.get("RELIC_VECTOR_FP4_WIDEN") == "1": suffix += "-vector-widen"
    if os.environ.get("RELIC_FUSED_ORIGINAL_ACTIVATION") == "1": suffix += "-fused-activation"
    if os.environ.get("RELIC_FAST_FUSED_ACTIVATION") == "1": suffix += "-static-shifts"
    if os.environ.get("RELIC_M1_DIRECT_ROUTING") == "1": suffix += "-M1-direct"
    if os.environ.get("RELIC_EXPERT_BODY_GRAPHS") == "1": suffix += "-expert-bodies"
    name = "actual-model-Cube-" + variant + suffix
    output = ROOT/"logs"/(name+"-"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(exist_ok=False)
    env = dict(os.environ, RELIC_PERF_OUTPUT=str(output), RELIC_CUBE_VARIANT=variant,
               POCKETLLM_V41_MMA_EMULATE="1", DEEPSEEK_V41_ASCEND_PACKED_RESIDENT="1",
               DEEPSEEK_V41_ASCEND_PACKED_GEMV="prefetch", DEEPSEEK_V41_ASCEND_GROUP_K="4",
               DEEPSEEK_V41_ASCEND_ROWS_N="16", DEEPSEEK_V41_ASCEND_DECODE_GEMV="1",
               DEEPSEEK_V41_ASCEND_FP8_GEMV_IMPL="float_significand",
               DEEPSEEK_V41_CPU_EXACT_GEMV="1", DEEPSEEK_V41_CPU_GEMV_IMPL="raw_simd",
               DEEPSEEK_V41_CPU_GEMV_THREADS="96", DEEPSEEK_V41_ENGRAM_RANK0="1",
               RELIC_CPU_RAW_SIMD_GEMV_CACHE=str(ROOT/"cache/cpu"), RELIC_CPU_GEMV_CACHE=str(ROOT/"cache/cpu"))
    command = [sys.executable,"-m","torch.distributed.run","--standalone","--nnodes=1","--nproc-per-node=8",
               str(ROOT/"source/profile_actual_cube.py"),"--checkpoint","/data1/relic-v41-20261116-210/checkpoint",
               "--expert-device","npu:0","--expert-world","8","--expert-pool-rows","0","--expert-hot-rows","0",
               "--no-expert-batched","--expert-deal","id","--no-decode-graphs","--threads","4",
               "--max-seq-len","128","--max-new-tokens","16","--temperature","0","--prompt",
               "List the integers from 1 through 100 in order, separated by commas. The list is:"]
    sources = {}
    for project, directory in (("RelicLLM","relicllm/models/deepseek_v4_1"),("relic-core","relic_core/kernels")):
        for path in (ROOT/"source"/project/directory).glob("*.py"):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name_source in ("run_actual_cube.py", "profile_actual_cube.py", "bounded_task.py"):
        path = ROOT/"source"/name_source
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    experiment_fields = ("RELIC_CUBE_VARIANT", "RELIC_CUBE_WEIGHT_CACHE_BYTES", "RELIC_EXACT_CPU_POOL",
                         "RELIC_HOST_ENGRAM_PREFETCH", "RELIC_CUBE_NPUGRAPH", "RELIC_CUBE_PROFILE",
                         "RELIC_HC_FUSED_NORMALIZE", "RELIC_STATIC_FP4_WIDEN", "RELIC_VECTOR_FP4_WIDEN",
                         "RELIC_FUSED_ORIGINAL_ACTIVATION", "RELIC_FAST_FUSED_ACTIVATION",
                         "RELIC_M1_DIRECT_ROUTING", "RELIC_EXPERT_BODY_GRAPHS")
    with (output/"preconditions.json").open("x") as handle:
        json.dump({"command":command,"source_hashes":sources,"lease_end_UTC":LEASE_END.isoformat(),
                   "timeout_seconds":LIMIT,"explicit_human_permission":"test different NPU accumulation",
                   "complete_Engram_CPU_and_original_reduction":True,"original_checkpoint_unchanged":True,
                   "original_activation_FP8_quantization":True,"expert_I8_times_two_and_original_half_scales":True,
                   "dense_non_BF16_output_precision_retains_original_path":True,
                   "production_defaults_and_serving_guards_unchanged":True,"single_stream_batch":1,
                   "quality_teacher_forced_steps":8,"timed_forwards":16,"timed_context":[24,40],
                    "explicit_experiment_configuration":{name:env.get(name,"0") for name in experiment_fields},
                    "timing_is_fixed_length_forward_workload_not_EOS_stopped_service":True},handle,indent=2)
    result = helper.bounded_attempt(name,command,LIMIT,output,env,cwd=ROOT/"source/RelicLLM")
    with (output/"summary.json").open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps({"output":str(output),"returncode":result["returncode"],"cleanup":result["cleanup"]}),flush=True)
    return 0 if result["returncode"]==0 and result["cleanup"]["complete"] else 1


if __name__=="__main__":
    raise SystemExit(main())
