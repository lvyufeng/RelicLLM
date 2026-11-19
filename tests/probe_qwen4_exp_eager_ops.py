"""Which aten operations an eager decode step asks for, and how many times.

A captured graph contains kernels, not the Python that built them, so a replay profile cannot say
which part of the layer code a kernel came from -- the aten census of a replay is empty, while the
recording it replays was built from exactly those aten calls. Profiling the eager step instead
answers the question a fusion decision needs: what does the code ask for, per layer, per step.

Run on an explicitly leased CUDA host, from the repository root:
  torchrun --nproc_per_node=8 --master_port=60251 \
    tests/probe_qwen4_exp_eager_ops.py --model <checkpoint> \
    --context-tokens 8192 --steps 4

Profiling overhead is present, so the times here are not throughput numbers. The counts are the
report, and they are divided by the step count and by the layer count printed alongside.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from relicllm.models.qwen4_exp.runtime import (
    _warmup_device, init_distributed, load_model, next_token_from_logits,
)


def _inputs(tokenizer, count):
    filler = tokenizer.encode(
        "The measurement separates kernel cost from launch cost and from host copies. ",
        add_special_tokens=False,
    )
    head = tokenizer.encode("Report the decode bottleneck.\n", add_special_tokens=False)
    body = (filler * ((count - len(head) + len(filler) - 1) // len(filler)))[: count - len(head)]
    return torch.tensor([head + body], dtype=torch.long, device="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--limit", type=int, default=34)
    parser.add_argument("--shapes", action="store_true",
                        help="also report the argument shapes a aten op was called with")
    args = parser.parse_args()
    ctx = init_distributed()
    if ctx.pp_size != 1 or ctx.device.type != "cuda":
        raise RuntimeError("this probe needs pure TP on a leased CUDA host")
    from transformers import AutoTokenizer

    # No graph window: the point is to see the eager calls the recording would be built from.
    model, config = load_model(
        args.model, ctx, expert_resident=True, expert_fused_fp8=True,
        graph_window=0, mtp=False,
    )
    if config.text_config.indexer_budget != 2048:
        raise RuntimeError("the probe must run at the native indexer budget")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True
    )
    inputs = _inputs(tokenizer, args.context_tokens)
    _warmup_device(ctx, model.dtype, 3.0)
    cache = model.make_cache(1, args.context_tokens + args.steps + 8)

    layers = len([layer for layer in model.layers if layer is not None])
    report = {
        "kind": "qwen4_exp_eager_aten_census",
        "context_tokens": args.context_tokens,
        "layers": layers,
        "steps": args.steps,
        "budget": 2048,
    }
    with torch.inference_mode():
        past = 0
        logits = None
        for begin in range(0, args.context_tokens, args.chunk_size):
            block = inputs[:, begin:begin + args.chunk_size]
            logits = model.forward(block, cache=cache, past_len=begin, last_token_only=True)
            past = begin + block.shape[1]
        torch.cuda.synchronize(ctx.device)
        token = next_token_from_logits(logits, ctx)

        # One unprofiled step first so lazy initialisation is not charged to the census.
        logits = model.forward(token, cache=cache, past_len=past)
        token = next_token_from_logits(logits, ctx)
        past += 1
        torch.cuda.synchronize(ctx.device)

        from torch.profiler import ProfilerActivity, profile, record_function

        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
                     record_shapes=args.shapes) as prof:
            for _ in range(args.steps):
                with record_function("qwen4_exp_decode_step"):
                    logits = model.forward(token, cache=cache, past_len=past)
                    token = next_token_from_logits(logits, ctx)
                past += 1
            torch.cuda.synchronize(ctx.device)

    events = prof.key_averages(group_by_input_shape=args.shapes)
    aten = {}
    cuda_total = 0
    cuda_count = 0
    for entry in events:
        cuda_total += entry.self_device_time_total
        cuda_count += entry.count if not entry.key.startswith("aten::") else 0
        if entry.key.startswith("aten::"):
            aten[entry.key] = aten.get(entry.key, 0) + int(entry.count)

    steps = args.steps
    report["aten_by_count"] = [
        {"op": op, "per_step": count // steps, "per_layer_per_step": round(count / steps / max(layers, 1), 2)}
        for op, count in sorted(aten.items(), key=lambda item: item[1], reverse=True)[: args.limit]
    ]
    report["aten_total_per_step"] = sum(aten.values()) // steps
    report["cuda_kernels_per_step"] = cuda_count // steps
    report["cuda_device_ms_per_step"] = cuda_total / 1000.0 / steps
    if ctx.rank == 0:
        print(json.dumps(report), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()