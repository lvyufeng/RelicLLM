"""Synchronized sustained batch-one NPU-graph decode after actual parity proof.

Only this private CLI process constructs the adapter. Public serving guards and
normal generation selection remain unchanged. No logit copy occurs inside the
sixteen-step timed window except the required host greedy argmax.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])


@torch.inference_mode()
def generate_probe(front, ids, *, temperature, graphs=False, **kwargs):
    from relicllm.models.deepseek_v4_1.ascend_graphs import NpuDecodeGraphs
    from relicllm.models.deepseek_v4_1.decode_pos import Pos
    from relicllm.models.deepseek_v4_1.generate import Generation

    if temperature != 0 or graphs:
        raise ValueError("the private graph benchmark requires a greedy eager CLI entry")
    rank = dist.get_rank()
    model = front.model
    device = next(model.parameters()).device
    front.reset_state(1)
    model.temperature = 0.0
    dist.barrier()
    torch.npu.synchronize()
    start = time.perf_counter()
    _, logits, _ = front(torch.tensor([ids]), 0)
    torch.npu.synchronize()
    prefill_s = time.perf_counter()-start
    token = int(logits[0].argmax().item())
    pos = Pos.device(len(ids), device)
    driver = NpuDecodeGraphs(model)
    try:
        start = time.perf_counter()
        driver.capture_pass(lambda: front(torch.tensor([[token]]), pos))
        torch.npu.synchronize()
        capture_s = time.perf_counter()-start
        # Same four excluded forwards as the eager profiler (two warmups and two
        # instrumented steps), maintaining the identical timed context 24..40.
        excluded = []
        for _ in range(4):
            start = time.perf_counter()
            _, logits, _ = front(torch.tensor([[token]]), pos)
            torch.npu.synchronize()
            token = int(logits[0].argmax().item())
            pos.advance()
            excluded.append(time.perf_counter()-start)
        dist.barrier()
        torch.npu.synchronize()
        tokens, samples, eos = [], [], []
        start_phase = time.perf_counter()
        for index in range(16):
            start = time.perf_counter()
            _, logits, _ = front(torch.tensor([[token]]), pos)
            torch.npu.synchronize()
            token = int(logits[0].argmax().item())
            pos.advance()
            samples.append(time.perf_counter()-start)
            tokens.append(token)
            if token == 1:
                eos.append(index)
        torch.npu.synchronize()
        seconds = time.perf_counter()-start_phase
        last = logits[0].detach().float().cpu().contiguous()
        if not bool(torch.isfinite(last).all()) or int(last.argmax()) != tokens[-1]:
            raise RuntimeError("actual graph logits are not finite greedy output")
        row = {"rank": rank, "device": str(device), "prompt_tokens": len(ids),
               "prefill_seconds": prefill_s, "capture_seconds_excluded": capture_s,
               "warmup_steps": 4, "warmup_seconds": excluded,
               "instrumented_steps_excluded_from_benchmark": 0, "timed_steps": 16,
               "decode_seconds": seconds, "decode_token_per_second": 16/seconds,
               "step_seconds": samples, "all_timed_token_ids": tokens,
               "context_start": pos.host-16, "context_end": pos.host,
               "single_stream": True, "batch_size": 1, "graphs": True,
               "fixed_length_may_continue_after_eos": True, "observed_eos": eos,
               "final_logits_finite_and_greedy_verified": True, "host_Engram_enabled": True,
               "original_checkpoint_rewritten": False, "reference_reduction_enabled": os.environ.get("POCKETLLM_V41_MMA_EMULATE"),
               "packed_residency": [layer.ffn.routed.cache_stats for layer in model.layers],
               "host_engram_states": [layer.engram._host_engram_state.describe() for layer in model.layers if layer.engram is not None],
               "npu_allocated_bytes": torch.npu.memory_allocated(),
               "npu_reserved_bytes": torch.npu.memory_reserved(),
               "npu_peak_allocated_bytes": torch.npu.max_memory_allocated(),
               "private_graph_probe_does_not_enable_serving": True}
        with (OUTPUT / f"decode-baseline-rank-{rank}.json").open("x") as handle:
            json.dump(row, handle, indent=2)
        torch.save({"final_logits": last, "timed_tokens": tokens}, OUTPUT / f"baseline-final-logits-rank-{rank}.pt")
        print("SYNCHRONIZED_SINGLE_STREAM_GRAPH", json.dumps(row), flush=True)
        result = Generation(prompt_tokens=len(ids))
        result.tokens = tokens
        result.decode_seconds = seconds
        result.stopped = "fixed_graph_benchmark_steps"
        return result
    finally:
        driver.release()


def main():
    from relicllm.cli import generate_v41
    generate_v41.generate = generate_probe
    return generate_v41.main()


if __name__ == "__main__":
    raise SystemExit(main())
