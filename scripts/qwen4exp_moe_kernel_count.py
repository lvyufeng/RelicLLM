"""Count the kernels one decode step's MoE actually issues, and where its time goes.

A decode step's expert cost on a 3090 does not read as bandwidth.  One rank holds 14 GiB of
routed experts, a layer's slice of that is about 293 MiB, and at the card's own bandwidth a
layer's read is a few tenths of a millisecond -- but the measured MoE block is several
milliseconds.  That gap has two candidate explanations and they want different fixes:

* the loop issues many small kernels, so the step is launch-bound and a graph (or a fused
  kernel) is the answer;
* the loop issues few kernels but each is slow, so the arithmetic or the dequantization inside
  it is the answer.

This probe separates the two by profiling one step and printing, per kernel name, how many were
issued and how long they took.  Run it under torchrun:

    torchrun --nproc_per_node=8 scripts/qwen4exp_moe_kernel_count.py --model /dev/shm/qwen38-fp8

The report is printed by rank 0.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys

import torch


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hello, who are you?")
    parser.add_argument("--steps", type=int, default=3, help="decode steps to profile")
    parser.add_argument("--expert-resident", action="store_true")
    parser.add_argument("--expert-fused-fp8", action="store_true")
    parser.add_argument("--top", type=int, default=20, help="kernel rows to print")
    return parser.parse_args()


def main() -> int:
    args = _parse()
    sys.path.insert(0, os.getcwd())

    from relicllm.models.qwen4_exp.runtime import (
        _apply_chat_template,
        _load_tokenizer,
        init_distributed,
        load_model,
    )

    ctx = init_distributed()
    model, config = load_model(
        args.model,
        ctx,
        expert_resident=args.expert_resident,
        expert_fused_fp8=args.expert_fused_fp8,
    )

    tokenizer = _load_tokenizer(args.model)
    text = _apply_chat_template(args.model, args.prompt)
    ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.long, device=ctx.device)
    prompt_len = int(ids.shape[1])

    cache = model.make_cache(1, prompt_len + 64)
    with torch.inference_mode():
        logits = model.forward(ids, cache=cache, past_len=0, last_token_only=True)
    nxt = logits[:, -1].argmax(-1, keepdim=True)

    # Warm up so the profile sees steady state: allocator pools grown, JIT done, kernels loaded.
    token = nxt
    past = prompt_len
    for _ in range(3):
        with torch.inference_mode():
            out = model.forward(token.reshape(1, 1), cache=cache, past_len=past, last_token_only=True)
        token = out[:, -1].argmax(-1)
        past += 1
    torch.cuda.synchronize()

    if ctx.rank != 0:
        # Only rank 0 profiles; the others have to stay in lockstep through the same steps.
        for _ in range(args.steps):
            with torch.inference_mode():
                out = model.forward(token.reshape(1, 1), cache=cache, past_len=past, last_token_only=True)
            token = out[:, -1].argmax(-1)
            past += 1
        return 0

    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        for _ in range(args.steps):
            with torch.inference_mode():
                out = model.forward(token.reshape(1, 1), cache=cache, past_len=past, last_token_only=True)
            token = out[:, -1].argmax(-1)
            past += 1
        torch.cuda.synchronize()

    events = prof.key_averages()
    cuda = [e for e in events if getattr(e, "device_time_total", 0) > 0 or "cuda_time_total" in dir(e)]
    rows = []
    for e in cuda:
        total_us = getattr(e, "self_device_time_total", None)
        if total_us is None:
            total_us = getattr(e, "self_cuda_time_total", 0)
        if total_us and total_us > 0:
            rows.append((total_us, e.count, e.key))

    rows.sort(reverse=True)
    grand = sum(r[0] for r in rows)
    kernels = sum(r[1] for r in rows)

    print(f"[kernel-count] steps={args.steps} distinct={len(rows)} launches={kernels} "
          f"device_total={grand / 1000.0:.1f} ms "
          f"({grand / 1000.0 / args.steps:.1f} ms/step)", flush=True)
    print(f"[kernel-count] {'us_total':>10s} {'us/step':>9s} {'calls':>8s} {'per_step':>9s}  name", flush=True)
    for total_us, count, key in rows[: args.top]:
        print(
            f"[kernel-count] {total_us:10.1f} {total_us / args.steps:9.1f} {count:8d} "
            f"{count / args.steps:9.1f}  {key}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())