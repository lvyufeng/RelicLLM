"""Measure the per-collective latency floor this 8-rank fabric pays for the model's messages.

The decode layer carries TP partial-sum reductions of one [1, 5120] FP32 row. This probe times
exactly that shape, eager and inside a captured graph, so the per-layer collective cost in the
full model can be compared with a measured floor instead of a guess. It is a transport probe:
it runs no model arithmetic and makes no TPS claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist
import torch_npu  # noqa:F401


def sync():
    torch.npu.synchronize()


@torch.inference_mode()
def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    device = torch.device("npu", int(os.environ["LOCAL_RANK"]))
    torch.npu.set_device(device)
    dist.init_process_group("hccl")
    output = Path(os.environ["RELIC_PERF_OUTPUT"])
    record = {"rank": rank, "world": world, "message_shape": [1, 5120], "message_dtype": "float32",
              "message_bytes": 5120 * 4, "transport_probe_only": True, "model_arithmetic_run": False}
    rows = 5120
    payload = torch.full((1, rows), float(rank + 1), dtype=torch.float32, device=device)
    for _ in range(20):
        wide = payload.clone()
        dist.all_reduce(wide, op=dist.ReduceOp.SUM)
    sync()

    eager = []
    for _ in range(40):
        start = time.perf_counter()
        wide = payload.clone()
        dist.all_reduce(wide, op=dist.ReduceOp.SUM)
        sync()
        eager.append(time.perf_counter() - start)
    record["eager_seconds"] = eager
    record["eager_median_ms"] = statistics.median(eager) * 1e3

    # Captured: the same call inside one NPUGraph, replayed and synchronized.
    try:
        pool = torch.npu.graph_pool_handle()
        side = torch.npu.Stream()
        side.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(side):
            for _ in range(3):
                inner = payload.clone()
                dist.all_reduce(inner, op=dist.ReduceOp.SUM)
        torch.npu.current_stream().wait_stream(side)
        sync()
        target = torch.empty_like(payload)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, pool=pool):
            target.copy_(payload)
            dist.all_reduce(target, op=dist.ReduceOp.SUM)
        sync()
        captured = []
        for _ in range(40):
            start = time.perf_counter()
            graph.replay()
            sync()
            captured.append(time.perf_counter() - start)
        record["captured_seconds"] = captured
        record["captured_median_ms"] = statistics.median(captured) * 1e3
        record["captured_value"] = float(target.flatten()[0].item())
        record["captured_expected"] = float(world * (world + 1) / 2)
        record["captured_value_correct"] = record["captured_value"] == record["captured_expected"]
        # An eager replay of the same two operations, for comparison with the graph replay.
        reference = payload.clone()
        dist.all_reduce(reference, op=dist.ReduceOp.SUM)
        sync()
        record["eager_value"] = float(reference.flatten()[0].item())
    except BaseException as error:
        record["captured_error"] = f"{type(error).__name__}: {error}"
        record["captured_median_ms"] = None

    # A CPU-timed pair of events with nothing between them: the host-side floor of any measurement.
    record["output"] = str(output)
    with (output / f"transport-rank-{rank}.json").open("x") as handle:
        json.dump(record, handle, indent=2)
    print(json.dumps({k: record[k] for k in ("rank", "eager_median_ms", "captured_median_ms")}), flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())