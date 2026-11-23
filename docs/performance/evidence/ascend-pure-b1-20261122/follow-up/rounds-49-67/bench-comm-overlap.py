"""Can an HCCL allReduce on one stream overlap a matmul on another on this device?

The earlier non-overlap result was compute against compute. A collective runs on the communication
engine, not the cube units, so that result does not settle this. If the parallel wall time lands near
max(matmul, collective) the wait at a barrier can be hidden under useful work; if it lands near their
sum, the 19.3% of the step spent waiting is unrecoverable on this hardware and the execution-graph
route should be abandoned rather than built.
"""
import json
import os
import statistics
import time

import torch
import torch_npu  # noqa: F401
import torch.distributed as dist

rank = int(os.environ['RANK'])
world = int(os.environ['WORLD_SIZE'])
torch.npu.set_device(rank)
dist.init_process_group(backend='hccl', rank=rank, world_size=world)
device = torch.device('npu', rank)

# 20 KB: exactly the size of one allReduce in the model (5120 floats, one row).
comm_tensor = torch.randn(5120, device=device)
# A matmul big enough to be a few hundred microseconds, so the two terms are comparable.
weight = torch.randn((5120, 16384), dtype=torch.bfloat16, device=device) * 0.02
activation = torch.randn((1, 5120), dtype=torch.bfloat16, device=device)


def timed(fn, repeats=40):
    for _ in range(8):
        fn()
    torch.npu.synchronize()
    samples = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    return statistics.median(samples)


def collective():
    dist.all_reduce(comm_tensor)


def matmul():
    torch.mm(activation, weight)


def sequential():
    matmul()
    collective()


second = torch.npu.Stream(device=device)


def parallel():
    with torch.npu.stream(second):
        collective()
    matmul()
    torch.npu.current_stream().wait_stream(second)


t_comm = timed(collective)
t_mm = timed(matmul)
t_seq = timed(sequential)
t_par = timed(parallel)

report = {
    'rank': rank,
    'world': world,
    'collective_us': round(t_comm, 1),
    'matmul_us': round(t_mm, 1),
    'sequential_us': round(t_seq, 1),
    'parallel_two_streams_us': round(t_par, 1),
    'sum_of_parts': round(t_comm + t_mm, 1),
    'max_of_parts': round(max(t_comm, t_mm), 1),
    'overlap_ratio_vs_sequential': round(t_seq / t_par, 3) if t_par else None,
}
if rank == 0:
    print(json.dumps(report, indent=2))
    hidden = t_seq - t_par
    print()
    print('sequential %.1f us, two streams %.1f us, saving %.1f us (%.1f%%)'
          % (t_seq, t_par, hidden, 100 * hidden / t_seq if t_seq else 0))
    print('if fully overlapped it would be %.1f us; if not at all, %.1f us'
          % (max(t_comm, t_mm), t_comm + t_mm))
dist.destroy_process_group()
