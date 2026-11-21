"""Price the dense projections at one row against the same weights with more rows.

Layer A reads 47.71 MB of BF16 dense weights per layer and takes 1.386 ms, which is about 34 GB/s on a
card whose bandwidth is far higher. This probe times the same matrix at M=1 and at larger M to show
whether the loss is the row dimension itself or something specific to the model's call path. Timings
only: no model arithmetic and no TPS claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa:F401


def amortized(call, repeats=50, warmup=10):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(repeats):
        call()
    torch.npu.synchronize()
    return (time.perf_counter()-started)/repeats


@torch.inference_mode()
def main():
    out = Path(os.environ["RELIC_PERF_OUTPUT"])
    torch.npu.set_device(torch.device("npu", 0))
    # The released layer-0 attention projections: (K, N, elements) from the checkpoint's own counts.
    shapes = {"wq_a": (5120, 1280), "wq_b": (1280, 4096), "wkv": (5120, 512),
              "wo_a": (4096, 1024), "wo_b": (1024, 5120)}
    report = {"shapes": {}, "device": "npu:0"}
    for name, (k, n) in shapes.items():
        weight = torch.randn((n, k), dtype=torch.bfloat16, device="npu:0")
        entry = {"k": k, "n": n, "weight_bytes": n*k*2}
        for rows in (1, 2, 8, 32):
            x = torch.randn((rows, k), dtype=torch.bfloat16, device="npu:0")
            seconds = amortized(lambda: torch.matmul(x, weight.t()))
            entry[f"M{rows}_ms"] = seconds*1e3
            entry[f"M{rows}_GB_per_s"] = n*k*2/seconds/1e9
        report["shapes"][name] = entry
        print(name, {m: round(entry[f"M{m}_ms"], 4) for m in (1, 2, 8, 32)},
              "M1 GB/s", round(entry["M1_GB_per_s"], 1), flush=True)

    # The layer A total, at M=1 and with the same weights at M=8, for the row-effect summary.
    total_bytes = sum(n*k*2 for k, n in shapes.values())
    m1_total = sum(r["M1_ms"] for r in report["shapes"].values())
    m8_total = sum(r["M8_ms"] for r in report["shapes"].values())
    report["layer_attention_weights_bytes"] = total_bytes
    report["layer_attention_M1_ms"] = m1_total
    report["layer_attention_M1_GB_per_s"] = total_bytes/m1_total/1e6
    report["layer_attention_M8_ms"] = m8_total
    report["layer_attention_M8_GB_per_s"] = total_bytes/m8_total/1e6
    report["measured_layer_A_replay_ms"] = 1.3863178342580795
    print("attention weights total MB", round(total_bytes/1e6, 2))
    print("M1 total ms", round(m1_total, 4), "GB/s", round(total_bytes/m1_total/1e6, 1))
    print("M8 total ms", round(m8_total, 4), "GB/s", round(total_bytes/m8_total/1e6, 1))
    with (out / "dense-M1-pricing.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
