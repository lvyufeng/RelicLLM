"""Time one expert projection's stages without a per-iteration synchronize.

The first breakdown timed each stage with a synchronize inside the loop, which adds a fixed
per-call cost that is the same order as the stages themselves, and it reported a 15 ms widening that
the composite contradicts. This version repeats each stage N times inside one timed region, so the
synchronize and the Python call are amortized. It also times the layout change the projection needs
(packed [N, K/2] widened to [N, K] then transposed to the operator's [K, N]) separately, because that
is the term the first breakdown could not see.

Timings only: no model arithmetic and no TPS claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch_npu  # noqa:F401

from relicllm.models.deepseek_v4_1 import cube_experiment
from relic_core.kernels import unpack, vector_fp4_decode

DIM, INTER, BLOCK = 5120, 2304, 32
REPEATS = 200


def amortized(label, call, repeats=REPEATS, warmup=20):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    started = __import__("time").perf_counter()
    for _ in range(repeats):
        call()
    torch.npu.synchronize()
    total = __import__("time").perf_counter() - started
    return {"per_call_ms": total / repeats * 1e3, "total_s": total, "repeats": repeats}


@torch.inference_mode()
def main():
    out = Path(os.environ["RELIC_PERF_OUTPUT"])
    torch.npu.set_device(torch.device("npu", 0))
    report = {"repeats": REPEATS, "projections": {}}
    for name, rows, cols, k, n in (("w1_or_w3", INTER, DIM // 2, DIM, INTER),
                                   ("w2", DIM, INTER // 2, INTER, DIM)):
        a_codes = torch.randint(0, 256, (1, k), dtype=torch.uint8, device="npu:0")
        a_scale = torch.full((1, k // BLOCK), 127, dtype=torch.uint8, device="npu:0")
        packed = torch.randint(0, 256, (rows, cols), dtype=torch.uint8, device="npu:0")
        e8m0 = torch.full((rows, k // BLOCK), 127, dtype=torch.uint8, device="npu:0")
        widened = vector_fp4_decode.unpack_fp4_vector(packed, mode="arithmetic")
        half = unpack.fp4_int8_scale(e8m0)
        torch.npu.synchronize()
        weight = widened.transpose(0, 1).contiguous()
        half_scale = cube_experiment._half_scale_bf16(
            half.transpose(0, 1).contiguous(), name="probe", verify=False)
        activation = unpack.unpack_fp8_to_bf16(a_codes, a_scale, block_shape=(1, BLOCK))
        op = cube_experiment._npu_weight_quant_op()
        torch.npu.synchronize()
        stages = {
            "activation_decode_only": amortized(
                "act", lambda: unpack.unpack_fp8_to_bf16(a_codes, a_scale, block_shape=(1, BLOCK))),
            "widen_only": amortized(
                "widen", lambda: vector_fp4_decode.unpack_fp4_vector(packed, mode="arithmetic")),
            "transpose_to_KN_only": amortized(
                "transpose", lambda: widened.transpose(0, 1).contiguous()),
            "half_scale_build_only": amortized(
                "half", lambda: cube_experiment._half_scale_bf16(
                    unpack.fp4_int8_scale(e8m0).transpose(0, 1).contiguous(),
                    name="probe", verify=False)),
            "matmul_only": amortized(
                "matmul", lambda: op(activation, weight, half_scale, antiquant_offset=None,
                                     quant_scale=None, quant_offset=None, bias=None,
                                     antiquant_group_size=BLOCK, inner_precise=0)),
            "whole_projection": amortized(
                "whole", lambda: cube_experiment.cube_fp4_weight_quant_gemv(
                    a_codes, a_scale, packed, e8m0, antiquant_group_size=BLOCK, inner_precise=0)),
        }
        report["projections"][name] = stages
        for stage, value in stages.items():
            print(name, stage, round(value["per_call_ms"], 4), "ms", flush=True)

        # What the recorded body would cost if the layout change and the scale build were
        # precomputed once per expert instead of once per token.
        precomputed = (stages["activation_decode_only"]["per_call_ms"]
                       + stages["widen_only"]["per_call_ms"]
                       + stages["matmul_only"]["per_call_ms"])
        report.setdefault("projection_of_body", {})[name] = {
            "whole_projection_ms": stages["whole_projection"]["per_call_ms"],
            "widen_plus_transpose_plus_halfscale_plus_matmul_ms": (
                stages["widen_only"]["per_call_ms"] + stages["transpose_to_KN_only"]["per_call_ms"]
                + stages["half_scale_build_only"]["per_call_ms"]
                + stages["matmul_only"]["per_call_ms"]),
            "if_layout_and_scale_were_precomputed_ms": precomputed,
        }
    body = report.setdefault("expert_body", {})
    pw = report["projection_of_body"]
    body["whole_three_projections_ms"] = (2 * pw["w1_or_w3"]["whole_projection_ms"]
                                          + pw["w2"]["whole_projection_ms"])
    body["with_layout_and_scale_precomputed_ms"] = (
        2 * pw["w1_or_w3"]["if_layout_and_scale_were_precomputed_ms"]
        + pw["w2"]["if_layout_and_scale_were_precomputed_ms"])
    print("body_whole_ms", round(body["whole_three_projections_ms"], 4), flush=True)
    print("body_if_precomputed_ms", round(body["with_layout_and_scale_precomputed_ms"], 4), flush=True)
    with (out / "expert-body-breakdown-amortized.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())