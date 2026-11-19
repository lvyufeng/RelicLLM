"""Locate first row-count-dependent arithmetic from one identical committed cache prefix.

Debug only: records first-row layer and dense-linear outputs on rank 0 and compares an ordinary
single-row step with K rows. Optionally loops dense BF16 linear calls row-wise for a controlled
causality/rounding diagnostic. It never reports decode TPS or changes the runtime's default path.
"""
from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F

from bench_qwen4_exp_baseline import _inputs
from relicllm.models.qwen4_exp.model import snapshot_cache, restore_cache
from relicllm.models.qwen4_exp.runtime import (
    init_distributed, load_model, next_token_from_logits, gather_logits,
    _begin_state_log, _begin_ple_log,
)


def difference(reference, candidate):
    delta = candidate.float() - reference.float()
    return {"equal": torch.equal(reference, candidate), "max_abs": delta.abs().max().item(),
            "mean_abs": delta.abs().mean().item(),
            "relative_l2": (delta.norm() / reference.float().norm().clamp_min(1e-30)).item()}


class LayerObserver:
    def __init__(self, inner, index, records):
        self.inner, self.index, self.records = inner, index, records

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, *args, **kwargs):
        value = self.inner(*args, **kwargs)
        if self.records is not None:
            self.records.append((self.index, value.reshape(-1, value.shape[-1])[0].float().cpu().clone()))
        return value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--context-tokens", type=int, default=1024)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--chunk-size", type=int, default=512)
    args = ap.parse_args()
    ctx = init_distributed()
    from transformers import AutoTokenizer
    model, config = load_model(args.model, ctx, expert_resident=True, expert_fused_fp8=True,
                                graph_window=0, mtp=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    inputs = _inputs(tokenizer, args.context_tokens)
    cache = model.make_cache(1, args.context_tokens + args.rows + 16)
    with torch.inference_mode():
        past = 0
        for piece in inputs.split(args.chunk_size, dim=1):
            logits = model.forward(piece, cache=cache, past_len=past)
            past += piece.shape[1]
        first_id = next_token_from_logits(logits, ctx).reshape(1, 1)
        before = snapshot_cache(cache)
        records, linears, convolutions, reductions = [], [], [], []
        original_linear, original_conv = F.linear, F.conv1d
        from relicllm.models.qwen4_exp import attention as attention_module
        original_qsa = attention_module.cuda_qsa_attention
        mode = {"rowwise": False, "conv_rowwise": False, "reduce": "normal", "qsa_rowwise": False}
        import torch.distributed as dist

        def observed_qsa(query, key, value, selection, scaling):
            if mode["qsa_rowwise"] and 1 < query.shape[2] <= args.rows:
                outputs = [original_qsa(query[:, :, row:row + 1].contiguous(), key, value,
                                        selection[:, row:row + 1].contiguous(), scaling)
                           for row in range(query.shape[2])]
                if any(output is None for output in outputs):
                    raise RuntimeError("the CUDA sparse-attention kernel declined a diagnostic row")
                return torch.cat(outputs, dim=1)
            return original_qsa(query, key, value, selection, scaling)

        def first_row(value):
            return value.reshape(-1, value.shape[-1])[0].float().cpu().clone()

        def observed_reduce(value):
            incoming = first_row(value) if ctx.rank == 0 else None
            flat = value.contiguous().reshape(-1, value.shape[-1])
            if mode["reduce"] != "normal" and 1 < flat.shape[0] <= args.rows:
                if mode["reduce"] == "coalesced":
                    with dist._coalescing_manager(device=ctx.device):
                        for row in flat:
                            dist.all_reduce(row)
                else:
                    for row in flat:
                        dist.all_reduce(row)
                value = flat.reshape_as(value)
            else:
                dist.all_reduce(value)
            if ctx.rank == 0:
                reductions.append((incoming, first_row(value)))
            return value

        def observed_conv(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
            if mode["conv_rowwise"] and stride == 1 and padding == 0 and dilation == 1:
                width = weight.shape[-1]
                count = x.shape[-1] - width + 1
                if 1 < count <= args.rows:
                    value = torch.cat([original_conv(x[..., j:j+width].contiguous(), weight, bias,
                                                     stride, padding, dilation, groups)
                                       for j in range(count)], dim=-1)
                else:
                    value = original_conv(x, weight, bias, stride, padding, dilation, groups)
            else:
                value = original_conv(x, weight, bias, stride, padding, dilation, groups)
            if ctx.rank == 0:
                convolutions.append(value[..., 0].float().cpu().clone())
            return value

        for layer in model.layers:
            layer.all_reduce = observed_reduce
        if ctx.rank == 0:
            model.layers = [LayerObserver(layer, index, records) for index, layer in enumerate(model.layers)]

        def observed_linear(x, weight, bias=None):
            rows = x.numel() // x.shape[-1]
            if mode["rowwise"] and 1 < rows <= args.rows:
                flat = x.reshape(rows, -1)
                value = torch.cat([original_linear(flat[i:i+1], weight, bias) for i in range(rows)], dim=0)
                value = value.reshape(*x.shape[:-1], weight.shape[0])
            else:
                value = original_linear(x, weight, bias)
            if ctx.rank == 0:
                frame = inspect.currentframe().f_back
                key = (Path(frame.f_code.co_filename).name, frame.f_lineno, tuple(weight.shape))
                linears.append((key, value.reshape(-1, value.shape[-1])[0].float().cpu().clone()))
            return value

        F.linear, F.conv1d = observed_linear, observed_conv
        attention_module.cuda_qsa_attention = observed_qsa
        try:
            reference_logits = model.forward(first_id, cache=cache, past_len=past)
            reference_full = gather_logits(reference_logits[0], ctx)
            reference_layers, reference_linears = list(records), list(linears)
            reference_convolutions, reference_reductions = list(convolutions), list(reductions)
            reference_full = reference_full[0].float().cpu().clone() if ctx.rank == 0 else None
            scenarios = [(False, False, "normal", False), (True, True, "normal", False),
                         (True, True, "rowwise", False), (True, True, "coalesced", False),
                         (True, True, "rowwise", True)]
            for rowwise, conv_rowwise, reduce_mode, qsa_rowwise in scenarios:
                restore_cache(before, cache)
                records.clear(); linears.clear(); convolutions.clear(); reductions.clear()
                _begin_state_log(cache)
                _begin_ple_log(cache)
                mode.update(rowwise=rowwise, conv_rowwise=conv_rowwise, reduce=reduce_mode,
                            qsa_rowwise=qsa_rowwise)
                feed = first_id.expand(1, args.rows).contiguous()
                logits = model.forward(feed, cache=cache, past_len=past)
                full = gather_logits(logits[0], ctx)
                if ctx.rank == 0:
                    actual = full[0].float().cpu().clone()
                    layer_diff = [{"layer": index, **difference(expected, value)}
                                  for (index, expected), (got_index, value) in zip(reference_layers, records)
                                  if index == got_index]
                    linear_diff = []
                    for (key, expected), (got_key, value) in zip(reference_linears, linears):
                        if key != got_key:
                            raise RuntimeError(f"linear call order differs: {key} != {got_key}")
                        d = difference(expected, value)
                        if not d["equal"]:
                            linear_diff.append({"site": key, **d})
                    print(json.dumps({"kind": "qwen4_first_row_arithmetic", "context_tokens": past,
                                      "rows": args.rows, "dense_rowwise": rowwise,
                                      "conv_rowwise": conv_rowwise, "reduction_mode": reduce_mode,
                                      "qsa_rowwise": qsa_rowwise,
                                      "convolution_differences": [
                                          {"call": i, **difference(a, b)}
                                          for i, (a, b) in enumerate(zip(reference_convolutions, convolutions))
                                          if not torch.equal(a, b)],
                                      "reduction_differences": [
                                          {"call": i, "input": difference(a0, b0), "output": difference(a1, b1)}
                                          for i, ((a0, a1), (b0, b1)) in enumerate(zip(reference_reductions, reductions))
                                          if not torch.equal(a1, b1)],
                                      "first_token": int(first_id.item()),
                                      "logits": difference(reference_full, actual),
                                      "reference_argmax": int(reference_full.argmax()),
                                      "candidate_argmax": int(actual.argmax()),
                                      "layers": layer_diff, "first_linear_differences": linear_diff[:20],
                                      "linear_calls": len(linears), "different_linear_calls": len(linear_diff)}), flush=True)
        finally:
            F.linear, F.conv1d = original_linear, original_conv
            attention_module.cuda_qsa_attention = original_qsa
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
