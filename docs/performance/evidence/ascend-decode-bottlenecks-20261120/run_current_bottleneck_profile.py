"""Same-host baseline/current tile profiling under an independently bounded lease."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938")
LEASE_END = dt.datetime.fromisoformat("2026-11-20T05:37:45+00:00")
LIMIT = 900
HELPER_SHA = "7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a"
LIBRARY_SHA = "0cd9111e373859c5b8d4ec09089a952467f26b6c88a3fb387842b11c0725b287"


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("tile", "baseline"):
        raise SystemExit("choose the exact measured tile or baseline configuration")
    if dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=LIMIT + 120) >= LEASE_END:
        raise RuntimeError("the profiler cannot finish safely within this lease")
    mode = sys.argv[1]
    safety = ROOT / "source/bounded_task.py"
    if hashlib.sha256(safety.read_bytes()).hexdigest() != HELPER_SHA:
        raise RuntimeError("unreviewed ownership/cleanup helper")
    specification = importlib.util.spec_from_file_location("bottleneck_profile_ownership", safety)
    helper = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helper)
    library = ROOT / "cache/online-tile-cube-k128-v2/lib/librelic_online_tile_cube.so"
    if hashlib.sha256(library.read_bytes()).hexdigest() != LIBRARY_SHA:
        raise RuntimeError("the accepted native consumer device binary changed")
    name = "current-bottleneck-" + mode
    output = ROOT / "logs" / (name + "-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(exist_ok=False)
    env = dict(os.environ)
    for key in ("RELIC_CUBE_ABLATE", "RELIC_CUBE_PROFILE", "RELIC_REPLAY_PROBE", "RELIC_DISPATCH_TIMING",
                "RELIC_STATIC_FP4_WIDEN", "PYTORCH_NPU_ALLOC_CONF", "RELIC_EXPERT_PROJECTION",
                "DEEPSEEK_V41_ASCEND_TILE_MAJOR"):
        env.pop(key, None)
    env.update(
        RELIC_PERF_OUTPUT=str(output), RELIC_CUBE_VARIANT="weight_quant",
        RELIC_VECTOR_FP4_WIDEN="1", RELIC_FUSED_ORIGINAL_ACTIVATION="1", RELIC_FAST_FUSED_ACTIVATION="1",
        RELIC_EXACT_CPU_POOL="1", RELIC_HOST_ENGRAM_PREFETCH="1",
        RELIC_CPU_RAW_SIMD_POOL_CACHE=str(ROOT / "cache/cpu"), RELIC_CUBE_WEIGHT_CACHE_BYTES="0",
        RELIC_CUBE_NPUGRAPH="1", RELIC_HC_FUSED_NORMALIZE="1", RELIC_FUSED_EXPERT_EPILOGUE="0",
        RELIC_M1_DIRECT_ROUTING="1" if mode == "baseline" else "0",
        RELIC_EXPERT_BODY_GRAPHS="1" if mode == "baseline" else "0",
        RELIC_DEVICE_MOE="0", RELIC_DEVICE_TILE_MOE="1" if mode == "tile" else "0",
        RELIC_KERNEL_PROFILE="1", POCKETLLM_V41_MMA_EMULATE="1",
        DEEPSEEK_V41_ASCEND_PACKED_RESIDENT="1", DEEPSEEK_V41_ASCEND_PACKED_GEMV="prefetch",
        DEEPSEEK_V41_ASCEND_GROUP_K="4", DEEPSEEK_V41_ASCEND_ROWS_N="16",
        DEEPSEEK_V41_ASCEND_DECODE_GEMV="1", DEEPSEEK_V41_ASCEND_FP8_GEMV_IMPL="float_significand",
        DEEPSEEK_V41_CPU_EXACT_GEMV="1", DEEPSEEK_V41_CPU_GEMV_IMPL="raw_simd",
        DEEPSEEK_V41_CPU_GEMV_THREADS="96", DEEPSEEK_V41_ENGRAM_RANK0="1",
        RELIC_CPU_RAW_SIMD_GEMV_CACHE=str(ROOT / "cache/cpu"), RELIC_CPU_GEMV_CACHE=str(ROOT / "cache/cpu"),
        RELIC_TILE_CONSUMER_LIBRARY=str(library), RELIC_TILE_CONSUMER_SHA256=LIBRARY_SHA)
    if mode == "tile":
        env.update(DEEPSEEK_V41_ASCEND_TILE_MAJOR="1", RELIC_EXPERT_PROJECTION="loader")
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=8",
               str(ROOT / "source/profile_current_bottlenecks.py"),
               "--checkpoint", "<checkpoint>", "--expert-device", "npu:0",
               "--expert-world", "8", "--expert-pool-rows", "0", "--expert-hot-rows", "0",
               "--no-expert-batched", "--expert-deal", "id", "--no-decode-graphs", "--threads", "4",
               "--max-seq-len", "128", "--max-new-tokens", "16", "--temperature", "0", "--prompt",
               "List the integers from 1 through 100 in order, separated by commas. The list is:"]
    sources = {}
    for project, directory in (("RelicLLM", "relicllm/models/deepseek_v4_1"), ("relic-core", "relic_core/kernels")):
        for path in (ROOT / "source" / project / directory).glob("*.py"):
            sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name_source in ("profile_current_bottlenecks.py", "run_current_bottleneck_profile.py", "profile_actual_cube.py", "bounded_task.py"):
        path = ROOT / "source" / name_source
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    sources[str(library)] = LIBRARY_SHA
    with (output / "preconditions.json").open("x") as stream:
        json.dump({"mode": mode, "command": command, "source_hashes": sources,
                   "lease_end_UTC": LEASE_END.isoformat(), "timeout_seconds": LIMIT,
                   "all_eight_NPU_ranks": True, "original_checkpoint_read_only": True,
                   "complete_Engram_CPU": True, "original_quantization_and_default_guards_unchanged": True,
                   "native_accumulation_experiment_previously_authorized": True,
                   "uninstrumented_timed_forwards": 16, "excluded_warmups": 4, "timed_context": [24,40],
                   "event_scoped_forwards_separate_from_TPS": 8, "CANN_profiled_forwards_separate_from_TPS": 4,
                   "profile_restarts_from_valid_context24_KV_and_hash_snapshot": True,
                   "all_ranks_profiled_not_single_rank_setup_skew": True,
                   "environment": {key:value for key,value in env.items() if key.startswith(("RELIC_", "DEEPSEEK_V41_", "POCKETLLM_V41_"))}},
                  stream, indent=2)
    result = helper.bounded_attempt(name, command, LIMIT, output, env, cwd=ROOT / "source/RelicLLM")
    with (output / "summary.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"mode": mode, "output": str(output), "returncode": result["returncode"],
                      "owned_cleanup_complete": result["cleanup"]["complete"]}), flush=True)
    return 0 if result["returncode"] == 0 and result["cleanup"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
