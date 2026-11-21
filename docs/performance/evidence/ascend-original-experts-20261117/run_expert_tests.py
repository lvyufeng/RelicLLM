"""Bounded task-private expert operator test runner with source receipts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path("/data1/relic-expert-gemm-20261117-1046")
SAFETY_SHA = "7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("gemm", "quant", "runtime"), required=True)
    parser.add_argument("--device", choices=("cpu", "npu"), required=True)
    parser.add_argument("--timeout", type=int, default=900)
    args = parser.parse_args()
    helper_path = ROOT / "source/bounded_task.py"
    if hashlib.sha256(helper_path.read_bytes()).hexdigest() != SAFETY_SHA:
        raise RuntimeError("ownership cleanup helper hash changed")
    spec = importlib.util.spec_from_file_location("expert_bounded", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    env = dict(os.environ)
    backend = "torch" if args.device == "cpu" else "triton"
    device = "cpu" if args.device == "cpu" else "npu:0"
    env.update(RELIC_EXPERT_TEST_DEVICE=device, RELIC_EXPERT_TEST_BACKEND=backend,
               RELIC_BLOCK_FP8_QUANT_TEST_DEVICE=device)
    if args.suite == "runtime" and args.device == "npu":
        env["RELIC_ASCEND_EXPERTS_DEVICE"] = device
    else:
        env.pop("RELIC_ASCEND_EXPERTS_DEVICE", None)
    project = "RelicLLM" if args.suite == "runtime" else "relic-core"
    cwd = ROOT / "source" / project
    tests = {"gemm": ["tests/test_block_gemm.py", "tests/test_block_gemm_math.py"],
             "quant": ["tests/test_block_fp8_quant_math.py", "tests/test_block_fp8_quant.py"],
             "runtime": ["tests/test_v41_ascend_experts.py", "tests/test_v41_ascend_bindings.py",
                         "tests/test_v41_online_unpack.py", "tests/test_models_deepseek_v4_1_loader.py",
                         "tests/test_v41_mma_reduction.py", "tests/test_v41_quantized_activation_path.py",
                         "tests/test_v41_quantized_activation_wiring.py", "tests/test_v41_fp4_expert_repack.py"]}[args.suite]
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = ROOT / "logs" / f"{args.suite}-{args.device}-{stamp}"
    output.mkdir(exist_ok=False)
    sources = [ROOT / "source/relic-core/relic_core/kernels" / name for name in (
        "block_gemm.py", "_ascend_block_gemm_kernels.py", "block_fp8_quant.py", "_ascend_fp8_quant_kernels.py",
        "unpack.py", "ascend_unpack.py", "_ascend_unpack_kernels.py")]
    sources += [cwd / name for name in tests]
    sources += [ROOT / "source/RelicLLM/relicllm/models/deepseek_v4_1" / name for name in
                ("ascend_experts.py", "ascend_linear.py", "loader.py", "kernels.py", "mma.py", "online_unpack.py")]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources if path.is_file()}
    with (output / "tested-source-hashes.json").open("x") as handle:
        json.dump({"sources": hashes, "python": sys.executable, "requested_backend": backend,
                   "requested_device": device}, handle, indent=2)
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *tests,
               "-q", "-rs", "--junitxml=" + str(output / "pytest.xml")]
    result = helper.bounded_attempt(f"{args.suite}-{args.device}", command, args.timeout,
                                    output, env, cwd=cwd)
    with (output / "summary.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    return 0 if result["returncode"] == 0 and result["cleanup"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
