"""Bounded single-card HC decode-fold parity/replay and paired graph microbench.

Original FP32 coefficients, multiply-before-sum and BF16 boundaries are checked
against the runtime equations. Performance is one pre+post body, not model TPS.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch

from relic_core.kernels.hc_decode_fold import fused_hc_decode_pre, fused_hc_decode_post
from relicllm.models.deepseek_v4_1.ascend_graphs import npu_graph_api, _record

OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])


@torch.inference_mode()
def main():
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    dim = 5120
    x = torch.empty(1, 1, dim, dtype=torch.bfloat16, device=device)
    residual = torch.empty(1, 1, 4, dim, dtype=torch.bfloat16, device=device)
    pre = torch.empty(1, 1, 4, dtype=torch.float32, device=device)
    post = torch.empty_like(pre)
    comb = torch.empty(1, 1, 4, 4, dtype=torch.float32, device=device)
    candidate_pre, reference_pre = torch.empty_like(x), torch.empty_like(x)
    candidate_post, reference_post = torch.empty_like(residual), torch.empty_like(residual)
    generator = torch.Generator().manual_seed(0x4C41)

    def update(case):
        cpu_x = torch.randn(1, 1, dim, generator=generator).bfloat16()
        cpu_residual = torch.randn(1, 1, 4, dim, generator=generator).bfloat16()
        cpu_pre = torch.rand(1, 1, 4, generator=generator)
        cpu_post = 2 * torch.rand(1, 1, 4, generator=generator)
        cpu_comb = torch.rand(1, 1, 4, 4, generator=generator)
        cpu_comb /= cpu_comb.sum(-1, keepdim=True)
        factor = 2.0 ** ((case % 9) - 3)
        cpu_x *= factor
        cpu_residual *= factor
        if case % 8 == 0:
            cpu_pre.zero_(); cpu_pre[0, 0, case % 4] = 1
        if case == 63:
            cpu_x.fill_(-0.0); cpu_residual.fill_(-0.0)
        for destination, source in ((x, cpu_x), (residual, cpu_residual), (pre, cpu_pre),
                                    (post, cpu_post), (comb, cpu_comb)):
            destination.copy_(source)
        torch.npu.synchronize()

    def candidate():
        fused_hc_decode_pre(residual, pre, out=candidate_pre, backend="triton")
        fused_hc_decode_post(x, residual, post, comb, out=candidate_post, backend="triton")

    def reference():
        reference_pre.copy_((pre.unsqueeze(-1) * residual.float()).sum(dim=2).bfloat16())
        value = post.unsqueeze(-1) * x.unsqueeze(-2)
        value = value + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
        reference_post.copy_(value.bfloat16())

    update(0)
    api = npu_graph_api()
    pool = api.graph_pool_handle()
    graphs = {"candidate": _record(candidate, pool, 2, api), "reference": _record(reference, pool, 2, api)}
    comparisons = []
    for case in range(64):
        update(case)
        graphs["reference"].replay(); graphs["candidate"].replay()
        torch.npu.synchronize()
        pre_diff = int((candidate_pre.view(torch.int16) != reference_pre.view(torch.int16)).sum().item())
        post_diff = int((candidate_post.view(torch.int16) != reference_post.view(torch.int16)).sum().item())
        row = {"case": case, "pre_bit_differences": pre_diff, "post_bit_differences": post_diff}
        comparisons.append(row)
        if pre_diff or post_diff:
            row.update(pre_max_abs=float((candidate_pre.float() - reference_pre.float()).abs().max().item()),
                       post_max_abs=float((candidate_post.float() - reference_post.float()).abs().max().item()))
            with (OUTPUT / "HC-fold-parity-failure.json").open("x") as stream:
                json.dump({"comparisons": comparisons, "failure": row}, stream, indent=2)
            raise RuntimeError(f"fused HC fold is not bit-identical to the current device reduction: {row}")
    samples = {"candidate": [], "reference": []}
    for graph in graphs.values():
        for _ in range(4):
            graph.replay()
    torch.npu.synchronize()
    for repetition in range(16):
        order = ("candidate", "reference") if repetition % 2 else ("reference", "candidate")
        for name in order:
            start = time.perf_counter()
            graphs[name].replay()
            torch.npu.synchronize()
            samples[name].append(time.perf_counter() - start)
    report = {"physical_device": 0, "dim": dim, "hc_mult": 4, "changing_input_graph_replays": 64,
              "pre_bit_differences": 0, "post_bit_differences": 0, "original_FP32_and_BF16_boundaries": True,
              "comparisons": comparisons, "paired_synchronized_graph_seconds": samples,
              "mean_seconds": {name: sum(values) / len(values) for name, values in samples.items()},
              "not_model_TPS": True, "scope": "one HC collapse plus one residual fold"}
    with (OUTPUT / "HC-fold-probe.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
