"""Break one routed expert body into its stages on the released geometry.

The full-model ablation showed the routed-expert bodies are about half of a token, and the widening
tile is now measured, so the next decision needs the split between widening, the native weight-quant
matmul and the elementwise gate. This probe times each stage separately with the released shapes and
the released operators, and times the whole projection for reference. Timings only: it runs no model
and makes no TPS claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa:F401

from relicllm.models.deepseek_v4_1 import cube_experiment
from relic_core.kernels import unpack, vector_fp4_decode

DIM = 5120          # model width K and the w2 output width
INTER = 2304        # expert intermediate width: w1/w3 output, w2 input
BLOCK = 32          # released E8M0 group width


def timed(label, call, repeats=20, warmup=3):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    seconds = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        torch.npu.synchronize()
        seconds.append(time.perf_counter() - started)
    median = statistics.median(seconds)
    return {"median_ms": median * 1e3, "seconds": seconds}


@torch.inference_mode()
def main():
    out = Path(os.environ["RELIC_PERF_OUTPUT"])
    torch.npu.set_device(torch.device("npu", 0))
    pack = unpack.unpack_fp4_to_int8
    unpack.unpack_fp4_to_int8 = (lambda packed, **kwargs:
                                vector_fp4_decode.unpack_fp4_vector(packed, mode="arithmetic"))
    try:
        def expert_projection(rows, packed_cols, k, n):
            """One projection's operands and the stage timings for them.

            ``k`` is that projection's own activation width: 5120 for w1/w3 and 2304 for w2.
            """
            a_codes = torch.randint(0, 256, (1, k), dtype=torch.uint8, device="npu:0")
            a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8, device="npu:0")
            packed = torch.randint(0, 256, (rows, packed_cols), dtype=torch.uint8, device="npu:0")
            e8m0 = torch.full((rows, k // BLOCK), 127, dtype=torch.uint8, device="npu:0")
            stages = {}
            widened = pack(packed, backend="auto")
            torch.npu.synchronize()
            half = unpack.fp4_int8_scale(e8m0, backend="auto")
            torch.npu.synchronize()

            stages["decode_widen_only"] = timed(
                "decode", lambda: pack(packed, backend="auto"))
            stages["half_scale_only"] = timed(
                "half scale", lambda: unpack.fp4_int8_scale(e8m0, backend="auto"))
            weight = widened.transpose(0, 1).contiguous()
            half_scale = cube_experiment._half_scale_bf16(
                half.transpose(0, 1).contiguous(), name="probe", verify=False)
            torch.npu.synchronize()

            activation = unpack.unpack_fp8_to_bf16(
                a_codes, a_scale, block_shape=(1, BLOCK), backend="auto")
            op = cube_experiment._npu_weight_quant_op()
            stages["matmul_with_decoded_operands"] = timed(
                "matmul", lambda: op(activation, weight, half_scale, antiquant_offset=None,
                                     quant_scale=None, quant_offset=None, bias=None,
                                     antiquant_group_size=BLOCK, inner_precise=0))
            stages["whole_projection"] = timed(
                "whole", lambda: cube_experiment.cube_fp4_weight_quant_gemv(
                    a_codes, a_scale, packed, e8m0, antiquant_group_size=BLOCK, inner_precise=0))
            return stages

        report = {"geometry": {"dim": DIM, "inter": INTER, "block": BLOCK}, "projections": {}}
        report["projections"]["w1_or_w3"] = expert_projection(INTER, DIM // 2, DIM, INTER)
        report["projections"]["w2"] = expert_projection(DIM, INTER // 2, INTER, DIM)

        # The elementwise SwiGLU gate on the released shapes.
        w1 = torch.randn((1, INTER), dtype=torch.float32, device="npu:0")
        w3 = torch.randn((1, INTER), dtype=torch.float32, device="npu:0")
        report["elementwise_silu_gate"] = timed(
            "silu*gate", lambda: torch.nn.functional.silu(w1) * w3)

        for name, stages in report["projections"].items():
            for stage, value in stages.items():
                print(name, stage, round(value["median_ms"], 4), "ms", flush=True)
        print("elementwise_silu_gate", round(report["elementwise_silu_gate"]["median_ms"], 4), "ms",
              flush=True)
        w1_sum = report["projections"]["w1_or_w3"]["whole_projection"]["median_ms"]
        w2_sum = report["projections"]["w2"]["whole_projection"]["median_ms"]
        total = 2 * w1_sum + w2_sum + report["elementwise_silu_gate"]["median_ms"]
        report["expert_body_composed_ms"] = total
        print("expert_body_composed_ms", round(total, 4), flush=True)
        with (out / "expert-body-breakdown.json").open("x") as handle:
            json.dump(report, handle, indent=2)
    finally:
        unpack.unpack_fp4_to_int8 = pack
    return 0


if __name__ == "__main__":
    raise SystemExit(main())