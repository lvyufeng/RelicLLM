"""Checkpoint-backed coarse profiling and synchronized fixed-step decode baseline.

This task-private driver replaces only the CLI generation loop in this process.
It keeps all model weights, Engram, quantization/reduction and TP bindings intact.
A fixed-length continuation may pass EOS; that is explicitly reported, not a
natural-completion quality benchmark. Instrumented steps are never used for TPS.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import functools
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])
BENCH_STEPS = int(os.environ.get("RELIC_PERF_BENCH_STEPS", "16"))
PROFILE_STEPS = int(os.environ.get("RELIC_PERF_PROFILE_STEPS", "2"))


def sync():
    torch.npu.synchronize()


class Timers:
    def __init__(self):
        self.records = defaultdict(lambda: {"calls": 0, "inclusive_s": 0.0, "exclusive_s": 0.0})
        self.stack = []
        self.replacements = []

    def patch(self, owner, name, label):
        original = getattr(owner, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            sync()
            started = time.perf_counter()
            self.stack.append(0.0)
            try:
                return original(*args, **kwargs)
            finally:
                sync()
                elapsed = time.perf_counter() - started
                children = self.stack.pop()
                if self.stack:
                    self.stack[-1] += elapsed
                row = self.records[label]
                row["calls"] += 1
                row["inclusive_s"] += elapsed
                row["exclusive_s"] += max(0.0, elapsed - children)

        setattr(owner, name, wrapped)
        self.replacements.append((owner, name, original))

    def restore(self):
        for owner, name, original in reversed(self.replacements):
            setattr(owner, name, original)
        self.replacements.clear()


def instrument(front):
    from relic_core.kernels import block_fp8_quant, block_gemm, unpack
    from relicllm.models.deepseek_v4_1 import ascend_experts, ascend_linear, attention, loader, mma, modules

    timer = Timers()
    entries = [
        (modules.Engram, "forward", "engram.total_boundary"),
        (modules.Engram, "_forward_pass", "engram.cpu_compute"),
        (loader.CheckpointEngramTable, "lookup", "engram.lookup"),
        (mma, "fp8_block_gemm", "gemm.cpu_reference"),
        (attention.Attention, "forward", "attention.total"),
        (modules.MoE, "forward", "moe.total"),
        (modules.Expert, "forward", "shared_expert.total"),
        (modules.Block, "hc_mixes", "hc.mix"),
        (modules.Block, "hc_pre", "hc.pre"),
        (modules.Block, "hc_post", "hc.post"),
        (ascend_experts.AscendRoutedExperts, "forward", "routed_experts.total"),
        (ascend_experts.AscendRoutedExperts, "expert", "expert.cache_access"),
        (ascend_experts.AscendRoutedExperts, "_load", "expert.host_read_transfer_unpack"),
        (ascend_experts, "expert_forward_ascend", "expert.compute"),
        (ascend_linear, "quantized_linear_ascend", "linear.npu_total"),
        (block_gemm, "fp4_int8_block_gemm", "gemm.npu_fp4"),
        (block_gemm, "fp8_block_gemm", "gemm.npu_fp8"),
        (block_fp8_quant, "quantize_block_fp8", "activation.fp8_quant"),
        (unpack, "unpack_fp4_to_int8", "unpack.fp4"),
        (unpack, "unpack_fp8_to_bf16", "unpack.fp8"),
        (unpack, "fp4_int8_scale", "unpack.half_scale"),
    ]
    from relic_core.kernels import cpu_block_gemv, cpu_raw_simd_gemv, decode_block_gemv, decode_grouped_gemv, floating_significand_gemv
    from relicllm.models.deepseek_v4_1 import ascend_packed_experts
    entries.extend([
        (cpu_block_gemv, "fp8_block_gemv", "gemm.cpu_compiled"),
        (cpu_raw_simd_gemv, "fp8_raw_simd_gemv", "gemm.cpu_raw_SIMD"),
        (floating_significand_gemv, "fp8_float_significand_gemv", "gemm.npu_FP8_float_significand"),
        (ascend_packed_experts.AscendPackedRoutedExperts, "forward", "routed_packed.total"),
        (ascend_packed_experts.AscendPackedRoutedExperts, "expert", "expert.packed_access"),
        (ascend_packed_experts, "packed_expert_forward", "expert.packed_compute"),
        (ascend_packed_experts, "_call_packed_gemv", "gemm.npu_packed_provider"),
        (decode_block_gemv, "fp4_packed_decode_gemv", "gemm.npu_packed_split"),
        (decode_grouped_gemv, "fp4_packed_grouped_decode_gemv", "gemm.npu_packed_grouped"),
    ])
    for owner, name, label in entries:
        timer.patch(owner, name, label)
    timer.patch(dist, "all_reduce", "collective.all_reduce")
    timer.patch(dist, "all_gather", "collective.all_gather")
    return timer


@torch.inference_mode()
def profiling_generate(front, ids, *, max_new_tokens, temperature, top_k, eos_token_id,
                       seed=None, graphs=False, prefill_chunk=None, on_token=None, **kwargs):
    from relicllm.models.deepseek_v4_1.generate import Generation
    from relicllm.models.deepseek_v4_1.ascend_experts import global_cache

    if graphs or temperature != 0:
        raise ValueError("this baseline requires eager original-semantic greedy decode")
    rank = dist.get_rank() if dist.is_initialized() else 0
    device = next(front.model.parameters()).device
    cache = global_cache()
    initial_cache = cache.snapshot()
    prompt_tokens = len(ids)
    position = prompt_tokens
    eos_seen = []
    all_tokens = []
    front.reset_state(1)
    # Match generate.generate: the model's own sample is unused by this greedy
    # loop and must not perform a softmax/Gumbel draw on each backbone forward.
    front.model.temperature = 0.0
    dist.barrier()
    sync()
    started = time.perf_counter()
    _, logits, _ = front(torch.tensor([ids]), 0, chunk=prefill_chunk)
    sync()
    prefill_s = time.perf_counter() - started
    token = int(logits[0].argmax().item())
    all_tokens.append(token)
    if token == eos_token_id:
        eos_seen.append({"phase": "prefill", "step": 0})

    # Two uninstrumented warm-up forwards pay the first decode shape/JIT cost.
    warm_s = []
    for index in range(2):
        sync()
        started = time.perf_counter()
        _, logits, _ = front(torch.tensor([[token]]), position)
        sync()
        token = int(logits[0].argmax().item())
        warm_s.append(time.perf_counter() - started)
        position += 1
        all_tokens.append(token)
        if token == eos_token_id:
            eos_seen.append({"phase": "warmup", "step": index})

    timer = instrument(front)
    profile_rows = []
    try:
        for index in range(PROFILE_STEPS):
            started = time.perf_counter()
            _, logits, _ = front(torch.tensor([[token]]), position)
            sync()
            token = int(logits[0].argmax().item())
            elapsed = time.perf_counter() - started
            position += 1
            all_tokens.append(token)
            profile_rows.append(elapsed)
            if token == eos_token_id:
                eos_seen.append({"phase": "instrumented", "step": index})
    finally:
        timer.restore()
    with (OUTPUT / f"coarse-profile-rank-{rank}.json").open("x") as handle:
        json.dump({"rank": rank, "profile_steps": PROFILE_STEPS,
                   "instrumented_step_seconds": profile_rows,
                   "records": dict(timer.records), "synchronized_nested_exclusive": True,
                   "instrumented_times_not_used_for_TPS": True}, handle, indent=2)

    # Timed steps include the required host greedy pick. No profiling wrappers,
    # per-step barrier, full-logit dump or callback runs inside this timed section.
    dist.barrier()
    sync()
    before_cache = cache.snapshot()
    timed_tokens = []
    step_s = []
    phase_started = time.perf_counter()
    for index in range(BENCH_STEPS):
        started = time.perf_counter()
        _, logits, _ = front(torch.tensor([[token]]), position)
        sync()
        token = int(logits[0].argmax().item())
        step_s.append(time.perf_counter() - started)
        position += 1
        timed_tokens.append(token)
        if token == eos_token_id:
            eos_seen.append({"phase": "timed", "step": index})
    sync()
    decode_s = time.perf_counter() - phase_started
    final_logit = logits[0].detach().float().cpu()
    if not bool(torch.isfinite(final_logit).all()):
        raise RuntimeError("non-finite final real-checkpoint logits")
    if int(final_logit.argmax()) != timed_tokens[-1]:
        raise RuntimeError("timed output does not match the actual greedy logits")
    after_cache = cache.snapshot()
    row = {"rank": rank, "device": str(device), "prompt_tokens": prompt_tokens,
           "prefill_seconds": prefill_s, "warmup_steps": 2, "warmup_seconds": warm_s,
           "instrumented_steps_excluded_from_benchmark": PROFILE_STEPS,
           "timed_steps": BENCH_STEPS, "decode_seconds": decode_s,
           "decode_token_per_second": BENCH_STEPS / decode_s,
           "step_seconds": step_s, "all_timed_token_ids": timed_tokens,
           "context_start": position - BENCH_STEPS, "context_end": position,
           "single_stream": True, "batch_size": 1, "graphs": False,
           "fixed_length_may_continue_after_eos": True, "observed_eos": eos_seen,
           "final_logits_finite_and_greedy_verified": True,
           "host_Engram_enabled": True, "original_checkpoint_rewritten": False,
           "reference_reduction_enabled": os.environ.get("POCKETLLM_V41_MMA_EMULATE"),
           "cache_initial": initial_cache, "cache_before_timed": before_cache,
           "cache_after_timed": after_cache,
           "packed_residency": [layer.ffn.routed.cache_stats for layer in front.model.layers
                                if hasattr(layer.ffn.routed, "resident_bytes")],
           "host_engram_states": [layer.engram._host_engram_state.describe() for layer in front.model.layers
                                  if layer.engram is not None and hasattr(layer.engram, "_host_engram_state")],
           "npu_allocated_bytes": torch.npu.memory_allocated(),
           "npu_reserved_bytes": torch.npu.memory_reserved(),
           "npu_peak_allocated_bytes": torch.npu.max_memory_allocated()}
    with (OUTPUT / f"decode-baseline-rank-{rank}.json").open("x") as handle:
        json.dump(row, handle, indent=2)
    torch.save({"final_logits": final_logit, "timed_tokens": timed_tokens},
               OUTPUT / f"baseline-final-logits-rank-{rank}.pt")
    print("SYNCHRONIZED_SINGLE_STREAM_BASELINE", json.dumps(row), flush=True)
    result = Generation(prompt_tokens=prompt_tokens)
    result.tokens = timed_tokens
    result.decode_seconds = decode_s
    result.stopped = "fixed_benchmark_steps"
    return result


def main():
    from relicllm.cli import generate_v41
    generate_v41.generate = profiling_generate
    return generate_v41.main()


if __name__ == "__main__":
    raise SystemExit(main())
