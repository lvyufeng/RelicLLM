"""Measure how much of a qwen4exp decode step a CUDA graph can take back.

A step spends about 40% of its wall clock waiting on the host: the GPU is idle between the
launches the driver has not fed it yet.  This probe answers the two questions that decide whether
the whole-graph work is worth doing, on the box that matters:

* can a decode step be captured at all, or does something in it still reach the host?
* if it captures, how much of the step does a replay remove?

It does not try to make the model faster.  It runs the same step twice -- once eagerly, once as a
captured graph -- and prints both, so the difference is the measurement and nothing else is.

    torchrun --nproc_per_node=8 scripts/qwen4exp_graph_probe.py --model /dev/shm/qwen38-fp8 \
        --expert-resident --expert-fused-fp8

A step that cannot be captured says why, from `torch.cuda`'s own message, and a full traceback for
rank 0: the failing frame is the result, because it names the call to move off the capture path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import torch


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="Hello, who are you?")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--eager-steps", type=int, default=20)
    parser.add_argument("--graph-steps", type=int, default=20)
    parser.add_argument("--expert-resident", action="store_true")
    parser.add_argument("--expert-fused-fp8", action="store_true")
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

    cache = model.make_cache(1, prompt_len + 128)
    with torch.inference_mode():
        logits = model.forward(ids, cache=cache, past_len=0, last_token_only=True)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    nxt = logits[:, -1].argmax(-1, keepdim=True)

    def step(token: torch.Tensor, past: int):
        with torch.inference_mode():
            out = model.forward(token, cache=cache, past_len=past, last_token_only=True)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        return out[:, -1].argmax(-1)

    def time_eager(n: int, first_past: int) -> float:
        token, past = nxt.clone(), first_past
        for _ in range(args.warmup):
            token = step(token, past).reshape(1, 1)
            past += 1
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            token = step(token, past).reshape(1, 1)
            past += 1
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    eager_s = time_eager(args.eager_steps, prompt_len)
    past = prompt_len + args.warmup + args.eager_steps + 4
    token_in = nxt.clone().reshape(1, 1)

    graph_s = None
    failure = None
    try:
        if ctx.device.type != "cuda":
            raise RuntimeError("no CUDA device")
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
            _ = model.forward(token_in, cache=cache, past_len=past, last_token_only=True)
        torch.cuda.synchronize()
        for _ in range(args.warmup):
            graph.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.graph_steps):
            graph.replay()
        torch.cuda.synchronize()
        graph_s = (time.perf_counter() - t0) / args.graph_steps
    except Exception as exc:  # noqa: BLE001 - the message is the result
        failure = f"{type(exc).__name__}: {exc}"
        if ctx.rank == 0:
            traceback.print_exc()

    if ctx.rank == 0:
        print(f"[graph-probe] eager = {1000.0 * eager_s:.2f} ms/step ({1.0 / eager_s:.3f} tok/s)",
              flush=True)
        if graph_s is not None:
            print(f"[graph-probe] graph = {1000.0 * graph_s:.2f} ms/step "
                  f"({1.0 / graph_s:.3f} tok/s)  speedup {eager_s / graph_s:.2f}x", flush=True)
        else:
            print(f"[graph-probe] CAPTURE FAILED: {failure}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
