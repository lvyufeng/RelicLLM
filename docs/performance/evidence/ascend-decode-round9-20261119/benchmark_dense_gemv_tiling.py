"""Sweep the dense FP8 GEMV's scheduling knobs at the model's own projection shapes.

The exact-reference dense path is `fp8_float_significand_gemv`, whose tiling decides how many
programs the card launches: `rows_n` (capped at 16, already the default) and `group_k` (default 4, max
32). The per-K-block accumulation is a separate kernel with a fixed order, so `group_k` is a schedule
knob and the result must stay bit-identical across it. This probe times every admissible combination on
the five attention projections and checks bit identity. Timings only: no model arithmetic, no TPS claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa:F401

from relic_core.kernels.floating_significand_gemv import fp8_float_significand_gemv

# The released layer-0 attention projections: (K, N) from the checkpoint's own element counts.
SHAPES = {"wq_a": (5120, 1280), "wq_b": (1280, 4096), "wkv": (5120, 512),
          "wo_a": (4096, 1024), "wo_b": (1024, 5120)}


def amortized(call, repeats=30, warmup=8):
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
    rows_n = 16                      # the documented maximum, and the default
    group_ks = (2, 4, 8, 16, 32)
    report = {"rows_n": rows_n, "group_ks": list(group_ks), "shapes": {}}
    totals = {g: 0.0 for g in group_ks}
    identity_ok = True
    for name, (k, n) in SHAPES.items():
        a_codes = torch.randint(0, 256, (1, k), dtype=torch.uint8, device="npu:0")
        a_scale = torch.full((1, k // 32), 127, dtype=torch.uint8, device="npu:0")
        b = torch.randint(0, 256, (n, k), dtype=torch.uint8, device="npu:0")
        sb = torch.full((n // 32, k // 32), 127, dtype=torch.uint8, device="npu:0")
        entry = {"k": k, "n": n, "weight_code_bytes": n*k, "group_k": {}}
        reference = None
        for g in group_ks:
            value = fp8_float_significand_gemv(
                a_codes, a_scale, b, sb, block_rows=32, backend="triton",
                rows_n=rows_n, group_k=g, variant="grouped")
            torch.npu.synchronize()
            if reference is None:
                reference = value.clone()
            same = bool(torch.equal(value.view(torch.int32), reference.view(torch.int32)))
            identity_ok &= same
            del value
            torch.npu.synchronize()
            seconds = amortized(lambda g=g: fp8_float_significand_gemv(
                a_codes, a_scale, b, sb, block_rows=32, backend="triton",
                rows_n=rows_n, group_k=g, variant="grouped"))
            entry["group_k"][str(g)] = {
                "ms": seconds*1e3,
                "GB_per_s": n*k/seconds/1e9,
                "bit_identical_to_group_k_2": same,
            }
            totals[g] += seconds*1e3
        report["shapes"][name] = entry
        print(name, {g: round(entry["group_k"][str(g)]["ms"], 4) for g in group_ks}, flush=True)
    report["layer_total_ms"] = totals
    report["all_group_k_bit_identical"] = identity_ok
    best = min(totals, key=totals.get)
    report["best_group_k"] = best
    report["layer_speedup_vs_group_k_4"] = totals[4]/totals[best]
    report["measured_layer_A_replay_ms"] = 1.3863178342580795
    print("layer total ms by group_k", {g: round(t, 4) for g, t in totals.items()})
    print("best group_k", best, "speedup vs 4: %.2fx" % report["layer_speedup_vs_group_k_4"])
    print("all bit identical", identity_ok)
    with (out / "dense-gemv-tiling.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
