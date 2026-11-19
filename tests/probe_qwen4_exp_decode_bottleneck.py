"""Kernel-level decomposition of one recorded decode step at native budget.

Run on an explicitly leased CUDA host, from the repository root:
  torchrun --nproc_per_node=8 --master_port=60241 \
    tests/probe_qwen4_exp_decode_bottleneck.py --model /workspace/qwen38-fp8 \
    --context-tokens 8192 --window 32 --steps 8

CUPTI reports the kernels a graph replay launches, which is the only way to
split a monolithic recording. Profiling overhead is present, so the absolute
step time here is not a throughput number; the shares and the counts are.
No checkpoint quality claim is made.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist

from relicllm.models.qwen4_exp.runtime import (
    _warmup_device, init_distributed, load_model, next_token_from_logits,
)
from relicllm.models.qwen4_exp.graph_window import build


def _inputs(tokenizer, count):
    filler = tokenizer.encode(
        "The measurement separates kernel cost from launch cost and from host copies. ",
        add_special_tokens=False,
    )
    head = tokenizer.encode("Report the decode bottleneck.\n", add_special_tokens=False)
    body = (filler * ((count - len(head) + len(filler) - 1) // len(filler)))[: count - len(head)]
    return torch.tensor([head + body], dtype=torch.long, device="cpu")


def _table(prof, limit=22):
    keyed = prof.key_averages()
    rows = sorted(keyed, key=lambda e: -e.self_device_time_total)
    total = sum(r.self_device_time_total for r in rows)
    out = []
    for entry in rows[:limit]:
        out.append({
            "kernel": entry.key[:78],
            "count": entry.count,
            "self_device_ms": entry.self_device_time_total / 1000.0,
            "share": entry.self_device_time_total / total if total else 0.0,
        })
    return out, total / 1000.0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()
    ctx = init_distributed()
    if ctx.pp_size != 1 or ctx.device.type != "cuda":
        raise RuntimeError("this probe needs pure TP on a leased CUDA host")
    from transformers import AutoTokenizer

    model, config = load_model(
        args.model, ctx, expert_resident=True, expert_fused_fp8=True,
        graph_window=args.window, mtp=False,
    )
    if config.text_config.indexer_budget != 2048:
        raise RuntimeError("the probe must run at the native indexer budget")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    inputs = _inputs(tokenizer, args.context_tokens)
    _warmup_device(ctx, model.dtype, 3.0)
    cache = model.make_cache(1, args.context_tokens + args.window + 8)
    with torch.inference_mode():
        past = 0
        logits = None
        for begin in range(0, args.context_tokens, args.chunk_size):
            block = inputs[:, begin:begin + args.chunk_size]
            logits = model.forward(block, cache=cache, past_len=begin, last_token_only=True)
            past = begin + block.shape[1]
        torch.cuda.synchronize(ctx.device)
        token = next_token_from_logits(logits, ctx)
        report = {"kind": "qwen4_exp_decode_bottleneck", "context_tokens": args.context_tokens,
                  "window": args.window, "budget": 2048, "profiled": True}

        graph = build(model=model, cache=cache, start=past, steps=args.window, token=token,
                      select=lambda logits: next_token_from_logits(logits, ctx))
        if graph is None:
            raise RuntimeError("capture declined; this probe measures a recording, not eager steps")
        history = inputs[0].tolist()
        # The recorded step is replayed from the host, so the step's wall time is the replay plus
        # whatever the host does between replays.  Timing the two apart says which one the next
        # optimisation is about; a replay-only number would credit host work to the GPU.
        started = time.perf_counter()
        replayed = 0
        host_s = 0.0
        gpu_s = 0.0
        for _ in range(args.steps):
            host_started = time.perf_counter()
            graph.fill_history(history + [int(token.item())])
            host_s += time.perf_counter() - host_started
            replay_started = time.perf_counter()
            nxt = graph.replay(past, token.to(ctx.device))
            gpu_s += time.perf_counter() - replay_started
            if nxt is None:
                break
            token = nxt
            history.append(int(token[0, 0].item()))
            past += 1
            replayed += 1
        torch.cuda.synchronize(ctx.device)
        elapsed = time.perf_counter() - started
        report["tokens_per_s_unprofiled"] = replayed / elapsed if replayed else 0.0
        report["unprofiled_replays"] = replayed
        report["host_fill_history_ms_per_step"] = 1000 * host_s / replayed if replayed else None
        report["replay_call_ms_per_step"] = 1000 * gpu_s / replayed if replayed else None
        report["step_wall_ms"] = 1000 * elapsed / replayed if replayed else None

        from torch.profiler import ProfilerActivity, profile

        torch.cuda.synchronize(ctx.device)
        with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
            profiled = 0
            for _ in range(args.steps):
                graph.fill_history(history + [int(token.item())])
                nxt = graph.replay(past, token.to(ctx.device))
                if nxt is None:
                    break
                token = nxt
                history.append(int(token[0, 0].item()))
                past += 1
                profiled += 1
            torch.cuda.synchronize(ctx.device)
        # The aten names say *what the code asked for*, which is what a fusion decision needs;
        # the CUDA kernel names in `rows` say what the compiler made of it.
        by_op: dict[str, int] = {}
        for entry in prof.key_averages():
            name = entry.key
            if not name.startswith("aten::"):
                continue
            by_op[name] = by_op.get(name, 0) + int(entry.count)
        report["aten_by_count"] = [
            {"op": op, "count": count}
            for op, count in sorted(by_op.items(), key=lambda item: item[1], reverse=True)[:30]
        ]
        rows, total_ms = _table(prof)
        report["profiled_replays"] = profiled
        report["kernels"] = rows
        report["self_device_ms_total"] = total_ms
        report["self_device_ms_per_step"] = total_ms / profiled if profiled else None
        report["launch_count_total"] = int(sum(entry.count for entry in prof.key_averages()))
        # By count as well as by time. A step's wall time is close to a launch count times what a
        # launch costs, so the ops that dominate the count are worth fusing even when no single
        # one of them is large -- which is the opposite conclusion from a time-ordered table.
        counted = sorted(prof.key_averages(), key=lambda entry: entry.count, reverse=True)
        report["kernels_by_count"] = [
            {"kernel": entry.key, "count": int(entry.count),
             "self_device_ms": float(entry.self_device_time_total) / 1000.0}
            for entry in counted[:150]
        ]
        # The profiler charges a collective for waiting behind the recorded step, so a table of
        # self device time alone cannot separate kernel work from that wait. Keep the counts, and
        # keep the two largest CUDA-graph-level items out of the share arithmetic by name.
        report["collective_note"] = (
            "nccl entries include time spent waiting behind preceding graph work; "
            "measure their cost by removing them, not from this table"
        )
        graph.release()
    if ctx.rank == 0:
        print(json.dumps(report, ensure_ascii=False, indent=1), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()