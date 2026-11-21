"""Original expert FP4 byte widening: global pair lookup versus local vector modes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa:F401

from relic_core.kernels.static_fp4_decode import unpack_fp4_pairs
from relic_core.kernels.vector_fp4_decode import unpack_fp4_vector
from relic_core.kernels.unpack import unpack_fp4_to_int8
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint


@torch.inference_mode()
def main():
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    output = Path(os.environ["RELIC_PERF_OUTPUT"])
    checkpoint = V41Checkpoint("<checkpoint>")
    records = []
    for key in ("layers.0.ffn.experts.0.w1.weight", "layers.0.ffn.experts.0.w2.weight"):
        raw = checkpoint.packed(key).view(torch.uint8).to(device)
        expected = unpack_fp4_to_int8(raw, backend="triton").cpu()
        calls = [("global_pair", 2048, lambda: unpack_fp4_pairs(raw, backend="triton"))]
        for mode in ("arithmetic", "lut"):
            for tile in (512, 1024, 2048):
                calls.append((mode, tile, lambda mode=mode, tile=tile: unpack_fp4_vector(raw, backend="triton", mode=mode, tile=tile)))
        for mode, tile, call in calls:
            actual = call()
            torch.npu.synchronize(device)
            if not torch.equal(actual.cpu(), expected):
                raise RuntimeError(f"original full expert widened bytes changed in {mode}/{tile}")
            for _ in range(3):
                call()
            torch.npu.synchronize(device)
            samples = []
            for _ in range(12):
                start = time.perf_counter()
                call()
                torch.npu.synchronize(device)
                samples.append(time.perf_counter() - start)
            record = {"key": key, "mode": mode, "tile": tile, "output_bytes": actual.numel(),
                      "original_full_matrix_byte_identity": True, "median_seconds": statistics.median(samples),
                      "seconds": samples, "whole_model_TPS_claimed": False}
            records.append(record)
            print(json.dumps(record), flush=True)
            with (output / "partial-vector-original-widen-benchmark.json").open("w") as handle:
                json.dump(records, handle, indent=2)
        del raw, actual, expected
    with (output / "vector-original-widen-benchmark.json").open("x") as handle:
        json.dump(records, handle, indent=2)
    checkpoint.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
