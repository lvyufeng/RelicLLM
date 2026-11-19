"""Probe RoPE batch-shape invariance and the readiness of CPU token staging.

Run on an explicitly leased CUDA host; no model weights are loaded:
  python tests/probe_qwen4_exp_rope_invariance.py --model /workspace/qwen38-fp8

The token probe deliberately delays the CUDA stream before a GPU-to-host copy.
It is a correctness diagnostic, not an inference throughput benchmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from relicllm.models.qwen4_exp.config import Qwen4ExpConfig
from relicllm.models.qwen4_exp.layers import MRoPE
from relicllm.models.qwen4_exp.model import Qwen4ExpModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    config = Qwen4ExpConfig.from_pretrained(args.model).text_config
    rope = MRoPE(config, device)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    results = []
    try:
        with torch.inference_mode():
            for tf32 in (False, True):
                torch.backends.cuda.matmul.allow_tf32 = tf32
                tables = {}
                for length in (8192, 8448, 8704, 8960):
                    positions = torch.arange(length, device=device).reshape(1, -1)
                    tables[length] = rope(positions, torch.bfloat16)
                base_cos, base_sin = tables[8192]
                for length in (8448, 8704, 8960):
                    cos, sin = tables[length]
                    results.append({
                        "tf32": tf32, "base_length": 8192, "compared_length": length,
                        "cos_max_abs": float((base_cos.float() - cos[:, :8192].float()).abs().max()),
                        "sin_max_abs": float((base_sin.float() - sin[:, :8192].float()).abs().max()),
                        "cos_changed_elements": int((base_cos != cos[:, :8192]).sum()),
                        "sin_changed_elements": int((base_sin != sin[:, :8192]).sum()),
                    })
            print(json.dumps({"kind": "rope_prefix_invariance", "results": results}), flush=True)
            # A bare instance is enough: this helper has no model-weight dependency.
            import ctypes
            import os
            import time
            model = Qwen4ExpModel.__new__(Qwen4ExpModel)
            model._ids_pinned = torch.full((1, 1), 7, dtype=torch.long, device="cpu", pin_memory=True)
            ids = torch.tensor([[42]], dtype=torch.long, device=device)
            model._pinned_ids(ids)  # Initialize copy/event/view dispatch before the delayed test.
            torch.cuda.synchronize(device)
            pointer = model._ids_pinned.data_ptr()
            trials = []
            for trial in range(3):
                model._ids_pinned.fill_(7)
                started = time.perf_counter()
                torch.cuda._sleep(200_000_000)
                launched = time.perf_counter()
                host = model._pinned_ids(ids)
                copied = time.perf_counter()
                immediate = ctypes.c_int64.from_address(pointer).value
                torch.cuda.synchronize(device)
                drained_at = time.perf_counter()
                drained = ctypes.c_int64.from_address(pointer).value
                trials.append({
                    "trial": trial, "immediate": immediate, "after_sync": drained,
                    "ready_on_return": immediate == 42,
                    "sleep_launch_ms": 1000 * (launched - started),
                    "copy_call_ms": 1000 * (copied - launched),
                    "remaining_sync_ms": 1000 * (drained_at - copied),
                })
            print(json.dumps({
                "kind": "gpu_token_host_readiness", "expected": 42, "trials": trials,
                "cuda_launch_blocking": os.environ.get("CUDA_LAUNCH_BLOCKING"),
                "torch": torch.__version__, "cuda": torch.version.cuda,
            }), flush=True)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


if __name__ == "__main__":
    main()
