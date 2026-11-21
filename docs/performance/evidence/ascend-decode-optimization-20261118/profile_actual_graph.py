"""Actual-checkpoint graph/eager decode equivalence and synchronized timing.

This private probe constructs the NPU graph adapter directly after an eager
prefill; it does not enable the CLI/serving graph capability or disable reference
reduction. Host Engram and routed-expert Python work stay outside each recording.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])
STEPS = int(os.environ.get("RELIC_PERF_GRAPH_STEPS", "8"))


def sync():
    torch.npu.synchronize()


@torch.inference_mode()
def generate_probe(front, ids, *, temperature, graphs=False, **kwargs):
    from relicllm.models.deepseek_v4_1.ascend_graphs import NpuDecodeGraphs
    from relicllm.models.deepseek_v4_1.decode_pos import Pos
    from relicllm.models.deepseek_v4_1.generate import Generation
    from relicllm.models.deepseek_v4_1.graphs import snapshot, restore

    if temperature != 0 or graphs:
        raise ValueError("the private exact graph probe requires greedy eager CLI entry")
    rank = dist.get_rank()
    device = next(front.model.parameters()).device
    front.reset_state(1)
    front.model.temperature = 0.0
    _, logits, _ = front(torch.tensor([ids]), 0)
    sync()
    first = int(logits[0].argmax().item())
    position = len(ids)
    saved = snapshot(front.model)
    hashes = front.snapshot_prefix(front.model.max_seq_len)
    eager_tokens, eager_logits, eager_times = [], [], []
    token = first
    for step in range(STEPS):
        sync()
        start = time.perf_counter()
        _, logits, _ = front(torch.tensor([[token]]), position+step)
        sync()
        token = int(logits[0].argmax().item())
        eager_times.append(time.perf_counter()-start)
        eager_tokens.append(token)
        eager_logits.append(logits[0].detach().float().cpu().clone())
    restore(front.model, saved)
    front.restore_prefix(hashes)
    driver = None
    try:
        driver = NpuDecodeGraphs(front.model)
        pos = Pos.device(position, device)
        capture_start = time.perf_counter()
        driver.capture_pass(lambda: front(torch.tensor([[first]]), pos))
        sync()
        capture_s = time.perf_counter()-capture_start
        # Attention cache capture rewinds itself. Restore the host hash cache too.
        restore(front.model, saved)
        front.restore_prefix(hashes)
        token = first
        graph_tokens, graph_times, mismatches = [], [], []
        for step in range(STEPS):
            sync()
            start = time.perf_counter()
            _, logits, _ = front(torch.tensor([[token]]), pos)
            sync()
            token = int(logits[0].argmax().item())
            graph_times.append(time.perf_counter()-start)
            graph_tokens.append(token)
            current = logits[0].detach().float().cpu().contiguous()
            expected = eager_logits[step].contiguous()
            if not bool(torch.isfinite(current).all()) or not bool(torch.isfinite(expected).all()):
                raise RuntimeError("graph/eager actual checkpoint logits are nonfinite")
            count = int((current.view(torch.int32) != expected.view(torch.int32)).sum())
            mismatches.append(count)
            if count:
                raise RuntimeError(f"position {pos.host}: {count} finite FP32 logit bits differ from eager")
            if token != eager_tokens[step]:
                raise RuntimeError("actual graph/eager greedy tokens differ")
            pos.advance()
        row = {"rank": rank, "prompt_tokens": len(ids), "context_start": position,
               "steps": STEPS, "eager_tokens": eager_tokens, "graph_tokens": graph_tokens,
               "FP32_logit_bit_mismatches_per_step": mismatches, "capture_seconds_excluded": capture_s,
               "eager_seconds": eager_times, "graph_seconds": graph_times,
               "eager_median_seconds": statistics.median(eager_times),
               "graph_median_seconds": statistics.median(graph_times),
               "warm_full_model_graph_not_yet_sustained_TPS": True,
               "host_Engram_remains_CPU": True, "original_checkpoint_rewritten": False,
               "reference_reduction_enabled": os.environ.get("POCKETLLM_V41_MMA_EMULATE"),
               "recorded_layers": len(driver.layers), "pool_bytes": driver.pool_bytes}
        with (OUTPUT / f"actual-graph-eager-rank-{rank}.json").open("x") as handle:
            json.dump(row, handle, indent=2)
        print("ACTUAL_GRAPH_EAGER_BIT_PARITY", json.dumps(row), flush=True)
    finally:
        if driver is not None:
            driver.release()
    result = Generation(prompt_tokens=len(ids))
    result.tokens = graph_tokens
    result.decode_seconds = sum(graph_times)
    result.stopped = "fixed_graph_parity_probe"
    return result


def main():
    from relicllm.cli import generate_v41
    generate_v41.generate = generate_probe
    return generate_v41.main()


if __name__ == "__main__":
    raise SystemExit(main())
