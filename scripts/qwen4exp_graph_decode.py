"""Capture one graph per decode position, so a generation replays instead of running eagerly.

A step's cost is dominated by the host: the GPU finishes the work of a layer long before the
driver has fed it the next launch.  A captured step replays that work with one call, but a graph
records the position it was captured at -- `past_len` is a Python integer that selects a rope
slice and a cache offset -- so a generation cannot reuse a single capture across its positions.

The approach here is the direct one: capture `steps` graphs, one per position, each against the
same caches, and replay graph `i` at step `i`.  Capturing is expensive (a warmup pass each) and
the graphs share nothing, so the window is a trade against the cost of capturing it.

This module is the measurement path, not the shipped one: `--graph-window N` in the runtime is
what would use it, and until that exists the numbers here are what it would deliver.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", default="Hello, who are you?")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--window", type=int, default=16, help="graphs to capture per round")
    ap.add_argument("--expert-resident", action="store_true")
    ap.add_argument("--expert-fused-fp8", action="store_true")
    args = ap.parse_args()
    sys.path.insert(0, os.getcwd())

    from relicllm.models.qwen4_exp.runtime import (
        _apply_chat_template,
        _load_tokenizer,
        init_distributed,
        load_model,
    )

    ctx = init_distributed()
    model, _ = load_model(
        args.model, ctx,
        expert_resident=args.expert_resident,
        expert_fused_fp8=args.expert_fused_fp8,
    )
    tok = _load_tokenizer(args.model)
    ids = torch.tensor([tok.encode(_apply_chat_template(args.model, args.prompt)).ids],
                       dtype=torch.long, device=ctx.device)
    plen = int(ids.shape[1])

    cache = model.make_cache(1, plen + args.steps + 8)
    with torch.inference_mode():
        logits = model.forward(ids, cache=cache, past_len=0, last_token_only=True)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    nxt = logits[:, -1].argmax(-1, keepdim=True)

    # One input buffer per position: a graph reads its input at a fixed address, and each
    # position's graph has its own.
    with torch.inference_mode(False):
        inputs = [nxt.clone() for _ in range(args.steps)]

    graphs = []
    t0 = time.perf_counter()
    try:
        for i in range(args.steps):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=torch.cuda.graph_pool_handle()):
                _ = model.forward(inputs[i], cache=cache, past_len=plen + i, last_token_only=True)
            graphs.append(g)
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        if ctx.rank == 0:
            print(f"[graph-decode] capture failed at {len(graphs)}: {type(exc).__name__}: {exc}")
        return 1
    capture_s = time.perf_counter() - t0

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for g in graphs:
        g.replay()
    torch.cuda.synchronize()
    replay_s = (time.perf_counter() - t0) / len(graphs)

    if ctx.rank == 0:
        print(f"[graph-decode] captured {len(graphs)} graphs in {capture_s:.1f}s "
              f"({capture_s / max(1, len(graphs)):.2f}s each)")
        print(f"[graph-decode] replay {1000.0 * replay_s:.2f} ms/step "
              f"({1.0 / replay_s:.2f} tok/s)")
        print(f"[graph-decode] steady state over {args.steps} tokens: "
              f"{args.steps * replay_s:.2f}s, capture amortised only past "
              f"{capture_s / replay_s:.0f} tokens")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
