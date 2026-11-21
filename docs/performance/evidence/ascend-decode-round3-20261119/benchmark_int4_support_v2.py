"""Probe which weight dtypes this build's weight-quant matmul accepts, in the model's own geometry.

The model's expert weight is [N, K] in the checkpoint; the operator wants x [1, K] times weight
[K, N]. This probe fixes that orientation and walks the candidate weight encodings for the original
4-bit payload, with the released per-group half scale. It reports what the build accepts and what it
costs; it makes no correctness or model claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch
import torch_npu  # noqa:F401

ROWS, COLS, GROUP = 2304, 5120, 32          # N, K, group width of the released expert projection


@torch.inference_mode()
def main():
    out = Path(os.environ["RELIC_PERF_OUTPUT"])
    torch.npu.set_device(torch.device("npu", 0))
    report = {"torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", "unknown"),
              "geometry": {"x": [1, COLS], "weight": [COLS, ROWS], "group": GROUP},
              "schema": None, "attempts": {}}

    # The full schema, taken from the operator's own registration.
    try:
        from torch._C import _dispatch_dump  # noqa:F401  (best effort only)
    except BaseException:
        pass
    try:
        torch_npu.npu_weight_quant_batchmatmul(
            torch.zeros((1, COLS), dtype=torch.bfloat16, device="npu:0"),
            torch.zeros((COLS, ROWS), dtype=torch.int8, device="npu:0"),
            torch.ones((ROWS, COLS // GROUP), dtype=torch.bfloat16, device="npu:0"),
            antiquant_group_size=GROUP, no_such_kwarg=1)
    except BaseException as error:
        report["schema"] = str(error)[:1500]

    x = torch.randn((1, COLS), dtype=torch.bfloat16, device="npu:0")
    scale = torch.ones((ROWS, COLS // GROUP), dtype=torch.bfloat16, device="npu:0")
    codes = torch.randint(0, 256, (COLS, ROWS // 2), dtype=torch.uint8, device="npu:0")

    def attempt(label, weight, **kwargs):
        entry = {"weight_shape": list(weight.shape), "weight_dtype": str(weight.dtype)}
        try:
            call = lambda: torch_npu.npu_weight_quant_batchmatmul(  # noqa: E731
                x, weight, scale, antiquant_group_size=GROUP, **kwargs)
            call()
            torch.npu.synchronize()
            started = time.perf_counter()
            for _ in range(20):
                call()
            torch.npu.synchronize()
            entry.update(accepted=True, median_ms=(time.perf_counter() - started) / 20 * 1e3)
            value = call()
            entry["finite"] = bool(torch.isfinite(value).all())
        except BaseException as error:
            entry.update(accepted=False, error=f"{type(error).__name__}: {str(error)[:300]}")
        report["attempts"][label] = entry

    int8_weight = torch.randint(-8, 8, (COLS, ROWS), dtype=torch.int8, device="npu:0")
    attempt("int8_control", int8_weight)
    # The 4-bit payload in the checkpoint's own byte order (two codes per byte along K).
    attempt("uint8_two_codes_per_byte_along_K", codes)
    # int4 declared explicitly, if the build's int4 dtype exists.
    if hasattr(torch, "int4"):
        try:
            int4_weight = torch.randint(0, 16, (COLS, ROWS), dtype=torch.int4, device="npu:0")
            attempt("int4_explicit", int4_weight)
        except BaseException as error:
            report["attempts"]["int4_explicit"] = {"accepted": False,
                                                   "error": f"{type(error).__name__}: {str(error)[:300]}"}
    else:
        report["attempts"]["int4_explicit"] = {"accepted": False, "error": "torch.int4 does not exist"}
    # Two codes per byte along N, as an int8 reinterpretation of the same payload.
    codes_n = codes.contiguous().view(torch.int8)
    attempt("uint8_as_int8_along_K", codes_n)

    with (out / "int4-support-probe-v2.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"schema_tail": (report["schema"] or "")[-500:],
                      "attempts": {k: (v.get("accepted"), v.get("median_ms"),
                                       (v.get("error") or "")[:120]) for k, v in report["attempts"].items()}},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())