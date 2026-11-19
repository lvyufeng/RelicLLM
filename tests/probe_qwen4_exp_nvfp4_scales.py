"""Audit the shared NVFP4 gate/up scale on real FP8 checkpoint experts, on CPU.

This measures weight reconstruction and expert-MLP output drift, not end-to-end model quality or
GPU throughput. It contrasts the historical gate-first recipe with the corrected joint-amax recipe.
Example: python tests/probe_qwen4_exp_nvfp4_scales.py --model <checkpoint>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as F

from relicllm.models.qwen4_exp.moe import NvFp4SlotExpertMoE
from relicllm.models.qwen4_exp.weights import Qwen4ExpCheckpoint


def unpack(packed, scales, global_scale):
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    codes = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8)
    codes[..., 0::2] = packed & 15
    codes[..., 1::2] = packed >> 4
    values = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    return values * scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, -1) * global_scale[:, None, None]


def metrics(reference, candidate):
    reference, candidate = reference.float().reshape(-1), candidate.float().reshape(-1)
    return {
        "relative_l2": ((candidate - reference).norm() / reference.norm().clamp_min(1e-30)).item(),
        "cosine": F.cosine_similarity(reference, candidate, dim=0).item(),
        "norm_ratio": (candidate.norm() / reference.norm().clamp_min(1e-30)).item(),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", default="0,20")
    ap.add_argument("--experts", default="0,63")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint = Qwen4ExpCheckpoint(args.model)
    records = []
    with torch.inference_mode():
        for layer in map(int, args.layers.split(",")):
            for expert in map(int, args.experts.split(",")):
                gate_up_fp8, down_fp8 = checkpoint.expert_rows(layer, expert)
                gate_up = gate_up_fp8.dequantize(torch.float32).unsqueeze(0)
                down = down_fp8.dequantize(torch.float32).unsqueeze(0)
                gate, up = gate_up.chunk(2, dim=1)
                gate_max = gate.abs().amax((1, 2))
                up_max = up.abs().amax((1, 2))

                # Exercise production _table, not a second implementation of the repaired recipe.
                backend = NvFp4SlotExpertMoE.__new__(NvFp4SlotExpertMoE)
                backend.device = torch.device("cpu")
                backend.shard = SimpleNamespace(layer_tensors=lambda _layer: (gate_up, down))
                backend._tables = {}
                backend.cache_hits = 0
                w1q, w1s, w3q, w3s, w2q, w2s, t1, t2 = backend._table(0)
                fixed_gate, fixed_up = unpack(w1q, w1s, t1), unpack(w3q, w3s, t1)
                fixed_down = unpack(w2q, w2s, t2)

                # Historical reproduction: the up bytes may use a larger scale than the gate's,
                # but the consuming kernel applies the gate's scale to BOTH projections.
                old_gq, old_gs, old_t, old_amax = backend._quantize(gate)
                old_uq, old_us, _discarded_t, _ = backend._quantize(up, shared_amax=old_amax)
                old_gate, old_up = unpack(old_gq, old_gs, old_t), unpack(old_uq, old_us, old_t)

                rng = torch.Generator().manual_seed(1234 + layer * 512 + expert)
                x = torch.randn(args.samples, gate.shape[-1], generator=rng)

                def mlp(g, u, d):
                    return F.linear(F.silu(F.linear(x, g[0])) * F.linear(x, u[0]), d[0])

                ideal = mlp(gate, up, down)
                record = {
                    "kind": "nvfp4_shared_scale_expert_audit", "layer": layer, "expert": expert,
                    "samples": args.samples, "hidden": gate.shape[-1], "intermediate": gate.shape[1],
                    "gate_amax": gate_max.item(), "up_amax": up_max.item(),
                    "up_has_larger_range": bool((up_max > gate_max).item()),
                    "historical_up_scale_multiplier": min(1.0, (gate_max / up_max).item()),
                    "old_up_weight": metrics(up, old_up), "fixed_up_weight": metrics(up, fixed_up),
                    "old_mlp": metrics(ideal, mlp(old_gate, old_up, fixed_down)),
                    "fixed_mlp": metrics(ideal, mlp(fixed_gate, fixed_up, fixed_down)),
                    "scope": "CPU FP8-dequantized reference, real expert weights, not task quality or decode TPS",
                }
                records.append(record)
                print(json.dumps(record), flush=True)
    print(json.dumps({"kind": "nvfp4_shared_scale_audit_summary", "experts": len(records),
                      "affected_experts": sum(r["up_has_larger_range"] for r in records),
                      "sample_outputs": len(records) * args.samples,
                      "fixed_mean_relative_l2": sum(r["fixed_mlp"]["relative_l2"] for r in records) / len(records),
                      "old_mean_relative_l2": sum(r["old_mlp"]["relative_l2"] for r in records) / len(records)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
