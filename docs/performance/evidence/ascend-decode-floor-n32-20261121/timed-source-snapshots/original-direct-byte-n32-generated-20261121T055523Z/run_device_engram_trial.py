"""Bound host-table/device-Engram full-model comparison within its live lease."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938")
LEASE_END = dt.datetime.fromisoformat("2026-11-20T09:54:06+00:00")
LIMIT = 900
HELPER_SHA = "7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a"


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in ("device", "host", "device-hc"):
        raise SystemExit("choose host, device, or device-hc compute, optionally with 16 or 64 timed forwards")
    steps = int(sys.argv[2]) if len(sys.argv) == 3 else 16
    if steps not in (16, 64):
        raise ValueError("this benchmark permits exactly 16 or 64 timed forwards")
    if dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=LIMIT + 120) >= LEASE_END:
        raise RuntimeError("not enough authorized lease remains for this bounded model trial")
    mode = sys.argv[1]
    safety = ROOT / "source/bounded_task.py"
    if hashlib.sha256(safety.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError("unreviewed owned-process helper")
    spec = importlib.util.spec_from_file_location("host_table_device_engram_ownership", safety)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    name = "original-host-table-" + mode + "-engram-native-baseline"
    if steps != 16:
        name += f"-steps{steps}"
    output = ROOT / "logs" / (name + "-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(exist_ok=False)
    env = dict(os.environ)
    for key in ("RELIC_CUBE_ABLATE", "RELIC_CUBE_PROFILE", "RELIC_REPLAY_PROBE", "RELIC_DISPATCH_TIMING",
                "RELIC_KERNEL_PROFILE", "RELIC_STATIC_FP4_WIDEN", "PYTORCH_NPU_ALLOC_CONF",
                "RELIC_EXPERT_PROJECTION", "DEEPSEEK_V41_ASCEND_TILE_MAJOR"):
        env.pop(key, None)
    env.update(RELIC_PERF_OUTPUT=str(output), RELIC_PERF_TIMED_STEPS=str(steps),
               RELIC_CUBE_VARIANT="weight_quant", RELIC_VECTOR_FP4_WIDEN="1",
               RELIC_FUSED_ORIGINAL_ACTIVATION="1", RELIC_FAST_FUSED_ACTIVATION="1",
               RELIC_EXACT_CPU_POOL="1", RELIC_HOST_ENGRAM_PREFETCH="1",
               RELIC_CPU_RAW_SIMD_POOL_CACHE=str(ROOT / "cache/cpu"), RELIC_CUBE_WEIGHT_CACHE_BYTES="0",
               RELIC_CUBE_NPUGRAPH="1", RELIC_HC_FUSED_NORMALIZE="1", RELIC_FUSED_EXPERT_EPILOGUE="0",
               RELIC_M1_DIRECT_ROUTING="1", RELIC_EXPERT_BODY_GRAPHS="1", RELIC_DEVICE_MOE="0", RELIC_DEVICE_TILE_MOE="0",
               RELIC_HOST_TABLE_DEVICE_ENGRAM="1" if mode != "host" else "0", RELIC_ENGRAM_GRAPHS="1",
               RELIC_HC_DECODE_FOLD="1" if mode == "device-hc" else "0",
               POCKETLLM_V41_MMA_EMULATE="1", DEEPSEEK_V41_ASCEND_PACKED_RESIDENT="1",
               DEEPSEEK_V41_ASCEND_PACKED_GEMV="prefetch", DEEPSEEK_V41_ASCEND_GROUP_K="4",
               DEEPSEEK_V41_ASCEND_ROWS_N="16", DEEPSEEK_V41_ASCEND_DECODE_GEMV="1",
               DEEPSEEK_V41_ASCEND_FP8_GEMV_IMPL="float_significand", DEEPSEEK_V41_CPU_EXACT_GEMV="1",
               DEEPSEEK_V41_CPU_GEMV_IMPL="raw_simd", DEEPSEEK_V41_CPU_GEMV_THREADS="96", DEEPSEEK_V41_ENGRAM_RANK0="1",
               RELIC_CPU_RAW_SIMD_GEMV_CACHE=str(ROOT / "cache/cpu"), RELIC_CPU_GEMV_CACHE=str(ROOT / "cache/cpu"))
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8",
               str(ROOT / "source/profile_device_engram.py"), "--checkpoint", "<checkpoint>",
               "--expert-device", "npu:0", "--expert-world", "8", "--expert-pool-rows", "0", "--expert-hot-rows", "0",
               "--no-expert-batched", "--expert-deal", "id", "--no-decode-graphs", "--threads", "4",
               "--max-seq-len", "128", "--max-new-tokens", str(steps), "--temperature", "0", "--prompt",
               "List the integers from 1 through 100 in order, separated by commas. The list is:"]
    sources = {}
    for project, directory in (("RelicLLM", "relicllm/models/deepseek_v4_1"), ("relic-core", "relic_core/kernels")):
        for path in (ROOT / "source" / project / directory).glob("*.py"):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for base in ("profile_device_engram.py", "profile_actual_cube.py", "run_device_engram_trial.py", "bounded_task.py"):
        path = ROOT / "source" / base
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    snapshots = output / "source-snapshots"
    relative_sources = [
        "RelicLLM/relicllm/models/deepseek_v4_1/ascend_device_engram.py",
        "RelicLLM/relicllm/models/deepseek_v4_1/ascend_hc_decode.py",
        "relic-core/relic_core/kernels/original_fp8_engram.py",
        "relic-core/relic_core/kernels/hc_decode_fold.py",
        "relic-core/relic_core/kernels/_ascend_hc_decode_fold_kernels.py",
        "profile_device_engram.py", "profile_actual_cube.py", "run_device_engram_trial.py", "bounded_task.py",
    ]
    for relative in relative_sources:
        source = ROOT / "source" / relative
        payload = source.read_bytes()
        if hashlib.sha256(payload).hexdigest() != sources[str(source)]:
            raise RuntimeError("a benchmark source changed during precondition capture")
        destination = snapshots / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            stream.write(payload)
    with (output / "preconditions.json").open("x") as stream:
        json.dump({"mode": mode, "command": command, "source_hashes": sources, "lease_end_UTC": LEASE_END.isoformat(),
                   "timeout_seconds": LIMIT, "original_checkpoint_and_quantization_unchanged": True,
                   "latest_human_permission": "Engram tables host is sufficient; begin optimization",
                   "tables_on_host": True, "complete_CPU_Engram_required": False,
                   "NPU_Engram_projection_and_gate_opt_in": mode != "host",
                   "HC_decode_collapse_and_residual_fold_opt_in": mode == "device-hc",
                   "prefill_and_exact_reference_original_host_path": True, "timed_steps": steps, "excluded_warmups": 4,
                   "timed_context": [24, 24 + steps], "single_request_batch_one": True,
                   "environment": {k:v for k,v in env.items() if k.startswith(("RELIC_", "DEEPSEEK_V41_", "POCKETLLM_V41_"))}}, stream, indent=2)
    result = helper.bounded_attempt(name, command, LIMIT, output, env, cwd=ROOT / "source/RelicLLM")
    with (output / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"mode": mode, "output": str(output), "returncode": result["returncode"],
                      "owned_cleanup_complete": result["cleanup"]["complete"]}), flush=True)
    return 0 if result["returncode"] == 0 and result["cleanup"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
