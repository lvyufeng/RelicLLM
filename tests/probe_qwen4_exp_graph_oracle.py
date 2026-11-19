"""Compare real-checkpoint graph replay against eager on a teacher-forced trace.

Run on an explicitly leased CUDA host, from the repository root:
  torchrun --nproc_per_node=8 --master_port=60123 \
    tests/probe_qwen4_exp_graph_oracle.py --model <checkpoint>

This is a numerical/state oracle, NOT a throughput or task-quality benchmark.
Both paths share weights and kernels but have independent request caches. It
crosses QSA group boundaries and rotates graph windows on fixed token IDs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from relic_core.kernels.cuda_loader import load_cuda_kernel
from relicllm.models.qwen4_exp.graph_window import build
from relicllm.models.qwen4_exp.runtime import init_distributed, load_model, next_token_from_logits


def _state(cache, name, slot):
    layout = getattr(cache, "state_slots", None)
    if layout is not None and name in layout:
        return layout[name].stack[slot]
    return getattr(cache, name, None)


def _compare(reference, replay, slot, position, atol, rtol):
    errors = []
    for idx, (left, right) in enumerate(zip(reference.layers, replay.layers)):
        if left is None:
            continue
        for name in ("conv_state", "recurrent_state"):
            value = getattr(left, name, None)
            if value is None:
                continue
            actual = _state(right, name, slot)
            try:
                torch.testing.assert_close(actual, value, atol=atol, rtol=rtol)
            except AssertionError as exc:
                errors.append(f"layer {idx} {name}: {str(exc).splitlines()[0]}")
        if hasattr(left, "k"):
            for name in ("k", "v", "index_k"):
                a, b = getattr(left, name), getattr(right, name)
                a = a[:, :, position] if a.ndim == 4 else a[:, position]
                b = b[:, :, position] if b.ndim == 4 else b[:, position]
                try:
                    torch.testing.assert_close(b, a, atol=atol, rtol=rtol)
                except AssertionError as exc:
                    errors.append(f"layer {idx} {name}: {str(exc).splitlines()[0]}")
    for idx, left in getattr(reference, "ple_states", {}).items():
        actual = _state(replay.ple_states[idx], "conv_state", slot)
        try:
            torch.testing.assert_close(actual, left.conv_state, atol=atol, rtol=rtol)
        except AssertionError as exc:
            errors.append(f"PLE {idx}: {str(exc).splitlines()[0]}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt-tokens", type=int, default=2303)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--handback-at", type=int, default=None)
    parser.add_argument("--reuse-windows", action="store_true")
    parser.add_argument("--incremental-candidate", action="store_true")
    parser.add_argument("--logit-atol", type=float, default=0.02)
    parser.add_argument("--logit-rtol", type=float, default=0.005)
    parser.add_argument("--state-atol", type=float, default=1e-5)
    parser.add_argument("--state-rtol", type=float, default=1e-5)
    args = parser.parse_args()
    if min(args.prompt_tokens, args.steps, args.window, args.chunk_size) <= 0:
        parser.error("trace dimensions must be positive")
    import os
    os.environ["POCKETLLM_QWEN4_GRAPH_REUSE"] = "1" if args.reuse_windows else "0"
    ctx = init_distributed()
    if ctx.pp_size != 1 or ctx.device.type != "cuda":
        raise RuntimeError("oracle requires pure TP on an explicitly leased CUDA host")
    model, config = load_model(
        args.model, ctx, expert_resident=True, expert_fused_fp8=True,
        graph_window=args.window, mtp=False,
    )
    if config.text_config.indexer_budget != 2048:
        raise RuntimeError("oracle baseline must keep the checkpoint budget at 2048")
    width = args.prompt_tokens + args.steps + 2
    reference = model.make_cache(1, width)
    replay = model.make_cache(1, width)
    gen = torch.Generator(device="cpu").manual_seed(7301)
    trace = torch.randint(1, min(config.text_config.vocab_size, 4096), (1, width), generator=gen, device="cpu")
    prefix = trace[:, :args.prompt_tokens]
    history = prefix[0].tolist()
    graph = None
    windows = 0
    top1_mismatches = 0
    handbacks = 0
    rebases = 0
    max_logit_abs = 0.0
    numerical_errors = []
    try:
        with torch.inference_mode():
            for cache in (reference, replay):
                os.environ["POCKETLLM_QWEN4_INCREMENTAL_INDEXER"] = (
                    "1" if cache is replay and args.incremental_candidate else "0"
                )
                for begin in range(0, args.prompt_tokens, args.chunk_size):
                    block = prefix[:, begin:begin + args.chunk_size]
                    model.forward(block, cache=cache, past_len=begin, last_token_only=True)
                torch.cuda.synchronize(ctx.device)
            for offset in range(args.steps):
                position = args.prompt_tokens + offset
                token = trace[:, position:position + 1]
                if graph is None or (not args.reuse_windows and position >= graph.start + len(graph)):
                    if graph is not None:
                        graph.release()
                    os.environ["POCKETLLM_QWEN4_INCREMENTAL_INDEXER"] = "1" if args.incremental_candidate else "0"
                    graph = build(
                        model=model, cache=replay, start=position,
                        steps=min(args.window, args.steps - offset),
                        token=token.to(ctx.device), select=None,
                    )
                    if graph is None:
                        raise RuntimeError("capture declined: oracle cannot count eager as replay")
                    windows += 1
                os.environ["POCKETLLM_QWEN4_INCREMENTAL_INDEXER"] = "0"
                expected = model.forward(token, cache=reference, past_len=position, last_token_only=True)
                os.environ["POCKETLLM_QWEN4_INCREMENTAL_INDEXER"] = "1" if args.incremental_candidate else "0"
                graph.fill_history(history + [int(token.item())])
                forced_handback = offset == args.handback_at
                if forced_handback:
                    graph._should_handback = lambda: True
                before_rebases = getattr(graph, "rebase_count", 0)
                selected = graph.replay(position, token.to(ctx.device))
                rebases += getattr(graph, "rebase_count", 0) - before_rebases
                if selected is None:
                    if not forced_handback:
                        raise RuntimeError("unrequested handback cannot count as graph replay")
                    graph.release()  # The runtime calls this again; release must be idempotent.
                    graph = None
                    handbacks += 1
                    actual = model.forward(
                        token, cache=replay, past_len=position, last_token_only=True
                    )[:, -1].float()
                else:
                    if forced_handback:
                        raise RuntimeError("the requested handback was not exercised")
                    actual = graph.outputs
                torch.cuda.synchronize(ctx.device)
                expected = expected[:, -1].float()
                max_logit_abs = max(max_logit_abs, float((actual - expected).abs().max()))
                eager_token = next_token_from_logits(expected, ctx)
                graph_token = next_token_from_logits(actual, ctx)
                top1_mismatches += int(not torch.equal(eager_token, graph_token))
                try:
                    torch.testing.assert_close(actual, expected, atol=args.logit_atol, rtol=args.logit_rtol)
                except AssertionError as exc:
                    numerical_errors.append(f"position {position} logits: {str(exc).splitlines()[0]}")
                numerical_errors.extend(_compare(
                    reference, replay, graph._committed_slot if graph is not None else None, position,
                    args.state_atol, args.state_rtol,
                ))
                history.append(int(token.item()))
            if graph is not None:
                graph.release()
                graph = None
            numerical_errors.extend(_compare(
                reference, replay, None, args.prompt_tokens + args.steps - 1,
                args.state_atol, args.state_rtol,
            ))
            if reference.length != replay.length:
                numerical_errors.append("logical cache length differs after release")
            if not torch.equal(reference.ple_token_history, replay.ple_token_history):
                numerical_errors.append("PLE token history differs after release")
        failed = torch.tensor([bool(numerical_errors or top1_mismatches)], device=ctx.device, dtype=torch.int32)
        maximum = torch.tensor([max_logit_abs], device=ctx.device)
        if dist.is_initialized():
            dist.all_reduce(failed, op=dist.ReduceOp.SUM)
            dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        if numerical_errors:
            print(json.dumps({"rank": ctx.rank, "errors": numerical_errors[:8]}, ensure_ascii=False), flush=True)
        if ctx.rank == 0:
            native = load_cuda_kernel()
            binary = Path(native.__file__)
            print(json.dumps({
                "kind": "teacher_forced_graph_state_oracle", "passed": int(failed.item()) == 0,
                "failed_ranks": int(failed.item()), "prompt_tokens": args.prompt_tokens,
                "steps": args.steps, "graph_windows": windows, "budget": 2048,
                "forced_handbacks": handbacks, "successful_replay_steps": args.steps - handbacks,
                "graph_captures": windows, "window_rebases": rebases,
                "reuse_windows": args.reuse_windows, "incremental_candidate": args.incremental_candidate,
                "top1_mismatches": top1_mismatches, "max_logit_abs": float(maximum.item()),
                "logit_atol": args.logit_atol, "logit_rtol": args.logit_rtol,
                "state_atol": args.state_atol, "state_rtol": args.state_rtol,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                "trace_sha256": hashlib.sha256(trace.numpy().tobytes()).hexdigest(),
                "quality_benchmark": False, "throughput_benchmark": False,
            }, ensure_ascii=False), flush=True)
        return 1 if int(failed.item()) else 0
    finally:
        if graph is not None:
            graph.release()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
