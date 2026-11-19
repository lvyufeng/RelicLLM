"""ABBA single-request FP8 baseline with explicit wall/setup timing.

Run on an explicitly leased CUDA host, from the repository root:
  torchrun --nproc_per_node=8 --master_port=60217 \
    tests/bench_qwen4_exp_baseline.py --model /workspace/qwen38-fp8 \
    --context-tokens 8192 --max-new-tokens 512 --graph-window 32

The same prompt, checkpoint and budget are used for eager/graph/graph/eager.
Loading, clock warmup and cache creation are outside each measured request.
No profiler is enabled. Setup-separated rates are wall residuals, NOT CUDA
kernel times. EOS is honored and short outputs are explicitly marked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
from relic_core.kernels.cuda_loader import load_cuda_kernel
from relicllm.models.qwen4_exp.runtime import generate, init_distributed, load_model, _warmup_device


def _cuda_memory() -> dict:
    """Peak and current device memory for this rank, in GiB, as the run actually spent it."""
    if not torch.cuda.is_available():
        return {}
    gib = float(1 << 30)
    free, total = torch.cuda.mem_get_info()
    return {
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / gib, 2),
        "free_gib": round(free / gib, 2),
        "total_gib": round(total / gib, 2),
    }


def _inputs(tokenizer, count):
    marker = "RELICLLM_CONTEXT_INSERTION"
    request = marker + (
        "\nExplain, in at least 1500 words, how to design a trustworthy GPU inference "
        "benchmark. Discuss cache ownership, speculative rollback, timing scope, hardware "
        "topology, numerical validation, and long-context testing. Use numbered sections."
    )
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": request}], tokenize=False, add_generation_prompt=True,
    )
    if formatted.count(marker) != 1:
        raise RuntimeError("chat template must preserve the unique insertion marker")
    begin, end = formatted.split(marker)
    leading = tokenizer.encode(begin, add_special_tokens=False)
    trailing = tokenizer.encode(end, add_special_tokens=False)
    filler = tokenizer.encode(
        "The experiment records model weights, input tokens, accelerator topology, "
        "cache states, and measured latency. Its claims require reproducible evidence. ",
        add_special_tokens=False,
    )
    available = count - len(leading) - len(trailing)
    if available <= 0 or not filler:
        raise ValueError("context is too short for the benchmark instruction")
    body = (filler * ((available + len(filler) - 1) // len(filler)))[:available]
    return torch.tensor([leading + body + trailing], dtype=torch.long, device="cpu")


def _percentile_ms(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return 1000 * (ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--context-tokens", type=int, default=8192)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--graph-window", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--warmup-s", type=float, default=3)
    parser.add_argument("--arena-tokens", type=int, default=0)
    parser.add_argument("--order", default="eager,graph,graph,eager",
                        help="comma-separated modes from eager/graph; any combination")
    parser.add_argument("--graph-reuse-order", default="")
    parser.add_argument("--incremental-order", default="")
    parser.add_argument("--host-kv-order", default="")
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    modes = [mode.strip() for mode in args.order.split(",")]
    if any(mode not in {"eager", "graph"} for mode in modes) or not modes:
        parser.error("--order must be a comma-separated subset of eager/graph")
    toggles = {}
    for name, text in (("POCKETLLM_QWEN4_GRAPH_REUSE", args.graph_reuse_order),
                       ("POCKETLLM_QWEN4_INCREMENTAL_INDEXER", args.incremental_order),
                       ("POCKETLLM_QWEN4_HOST_KV", args.host_kv_order)):
        if text:
            values = text.split(",")
            if len(values) != len(modes) or any(value not in {"0", "1"} for value in values):
                parser.error("per-sample flags require one 0/1 value per mode")
            toggles[name] = values
    arena_tokens = args.arena_tokens or args.context_tokens + args.max_new_tokens + 1
    if arena_tokens < args.context_tokens + args.max_new_tokens:
        raise ValueError("arena cannot fit the requested prompt and generation")
    ctx = init_distributed()
    if ctx.pp_size != 1 or ctx.device.type != "cuda":
        raise RuntimeError("baseline requires pure TP on an explicitly leased CUDA host")
    from transformers import AutoTokenizer

    started = time.perf_counter()
    model, config = load_model(
        args.model, ctx, expert_resident=True, expert_fused_fp8=True,
        graph_window=args.graph_window, mtp=False,
    )
    load_s = time.perf_counter() - started
    effective_budgets = {
        layer.attn.indexer.budget for layer in model.layers
        if layer is not None and hasattr(layer.attn, "indexer")
    }
    if config.text_config.indexer_budget != 2048 or effective_budgets != {2048}:
        raise RuntimeError(f"baseline must retain effective budget 2048, got {effective_budgets}")
    if args.context_tokens + args.max_new_tokens > config.text_config.max_position_embeddings:
        raise ValueError("prompt plus requested generation exceeds native maximum context")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    inputs = _inputs(tokenizer, args.context_tokens)
    warmup_s = _warmup_device(ctx=ctx, dtype=model.dtype, seconds=args.warmup_s)
    records = []
    reference_ids = None
    try:
        import os
        for order, mode in enumerate(modes):
            for name, values in toggles.items():
                os.environ[name] = values[order]
            # Fresh sequence state with a preallocated arena; no cross-request carry.
            cache = model.make_cache(1, arena_tokens)
            emitted_at = []
            def emitted(_token):
                emitted_at.append(time.perf_counter())
            output, stats = generate(
                model=model, ctx=ctx, input_ids=inputs, chunk_size=args.chunk_size,
                max_new_tokens=args.max_new_tokens, graph_window=args.graph_window if mode == "graph" else 0,
                cache=cache, eos_token_ids=tuple(config.text_config.eos_token_id),
                on_token=emitted if ctx.rank == 0 else None,
            )
            torch.cuda.synchronize(ctx.device)
            rank_stats = [None] * ctx.world_size
            if dist.is_initialized():
                dist.all_gather_object(rank_stats, stats)
            else:
                rank_stats[0] = stats
            if reference_ids is None:
                reference_ids = list(output)
            mismatch = next((i for i, (a,b) in enumerate(zip(reference_ids, output)) if a != b), None)
            if mismatch is None and len(reference_ids) != len(output):
                mismatch = min(len(reference_ids), len(output))
            intervals = [b - a for a,b in zip(emitted_at, emitted_at[1:])]
            min_replays = min(s.get("replay_count", 0) for s in rank_stats)
            execution_valid = mode != "graph" or min_replays > 0
            record = {
                "kind": "qwen4_exp_abba_baseline", "order": order, "mode": mode,
                "context_tokens": args.context_tokens, "budget": 2048, "mtp": False,
                "effective_budgets": sorted(effective_budgets),
                "requested_tokens": args.max_new_tokens, "emitted_tokens": len(output),
                "arena_tokens": arena_tokens, "output_ids": list(output),
                "sufficient_span": len(output) >= 128,
                "reference_mode": modes[0], "matches_first_sample": mismatch is None,
                "matches_first_eager": mismatch is None if modes[0] == "eager" else None,
                "graph_reuse_enabled": os.environ.get("POCKETLLM_QWEN4_GRAPH_REUSE", "0") == "1",
                "incremental_enabled": os.environ.get("POCKETLLM_QWEN4_INCREMENTAL_INDEXER", "0") == "1",
                "host_kv": os.environ.get("POCKETLLM_QWEN4_HOST_KV", "0") == "1",
                "label": args.label,
                "first_mismatch": mismatch, "stats_rank0": rank_stats[0],
                "successful_replays_min_rank": min_replays,
                "required_execution_mode_ran": execution_valid,
                "rank0_emit_itl_p50_ms": _percentile_ms(intervals, 0.50),
                "rank0_emit_itl_p95_ms": _percentile_ms(intervals, 0.95),
                "rank0_emit_itl_p99_ms": _percentile_ms(intervals, 0.99),
                "rank0_streaming_tps": len(intervals) / sum(intervals) if intervals and sum(intervals) > 0 else None,
                "itl_scope": "rank0 callback inter-emission wall time, including rebuilds; not HTTP or GPU events",
                "decode_wall_s_max_rank": max(s["decode_wall_s"] for s in rank_stats),
                "load_s_rank0": load_s, "warmup_s_rank0": warmup_s,
                # Measured, not estimated: what the run actually asked of the card. Whether a
                # feature can be added -- an MTP head is a whole extra decoder layer -- is decided
                # by this number against 24 GiB, and quoting it from arithmetic invites an
                # argument that a measurement settles.
                "rank0_cuda": _cuda_memory(),
                "input_sha256": hashlib.sha256(inputs.numpy().tobytes()).hexdigest(),
                "output_sha256": hashlib.sha256(json.dumps(output).encode()).hexdigest(),
            }
            records.append(record)
            if ctx.rank == 0:
                print(json.dumps(record, ensure_ascii=False), flush=True)
                print("OUTPUT_PREVIEW=" + tokenizer.decode(output[:160]), flush=True)
            del cache
        if ctx.rank == 0:
            native = load_cuda_kernel()
            roots = Path(__file__).resolve().parents[1] / "relicllm/models/qwen4_exp"
            source_hashes = {name: hashlib.sha256((roots/name).read_bytes()).hexdigest()
                             for name in ("runtime.py", "model.py", "graph_window.py", "attention.py")}
            valid = all(r["matches_first_sample"] and r["sufficient_span"] and r["required_execution_mode_ran"] for r in records)
            print(json.dumps({
                "kind": "qwen4_exp_abba_summary", "matched_output_baseline": valid,
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "binary_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
                "source_sha256": source_hashes, "profiler_enabled": False,
                "clocks_locked": False, "native_max_context": config.text_config.max_position_embeddings,
                "task_quality_score": None, "matching_eager_is_not_a_quality_score": True,
            }), flush=True)
        valid = all(r["matches_first_sample"] and r["sufficient_span"] and r["required_execution_mode_ran"] for r in records)
        return 0 if valid else 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
