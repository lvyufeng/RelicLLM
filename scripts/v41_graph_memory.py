#!/usr/bin/env python3
"""Measure what capturing the emulated decode GEMM costs the driver, on the card it would run on.

`EMULATION_REFUSES_GRAPHS` is the rule that keeps a decode graph and the reference-exact mma
emulation apart, and it is a *memory* rule: a recording of the emulated body holds one node a launch,
and the driver charges for each. Its number -- ~897 MiB a layer against the ~10 GiB a 24 GiB card has
left after the weights -- is the budget a reduction has to move, so this measures the same quantity
directly and lets a change be scored on it.

`scripts/v41_emulation_cost.py` counts operators on CPU and is the cheap, per-change signal. This is
the expensive one that says what an operator count is worth in bytes, and it needs the card:

* **Does one capture succeed?** The refusal is a memory refusal, so one GEMM is the unit.
* **What does each capture commit?** Read with `torch.cuda.mem_get_info` around the capture, because
  the cost is *driver* graph memory -- the commit that added the refusal records that
  `memory_allocated` and `memory_reserved` barely move and torch's own graph pool stays under 20 MiB,
  so the allocator's counters cannot see it.
* **How does it scale with the node count?** `--k` is the lever: K/32 is the K-block count, and the
  emulation's per-block work is where its nodes come from.

Repeats stand in for the GEMMs a decode body contains: each gets its own graph object and memory
pool, so the loop reports the marginal cost rather than one pool's high-water mark.

The tree it reads is **the tree it sits in** -- run it from the checkout under test, not through an
installed `relicllm`, or two revisions are silently the same measurement.

Usage::

    cd <checkout>; python scripts/v41_graph_memory.py --m 1 --n 1280 --k 5120 --repeats 40
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from relicllm.models.deepseek_v4_1 import mma  # noqa: E402

MIB = 2 ** 20


def operands(m: int, n: int, k: int, device: torch.device):
    """Codes and e8m0 scales, deterministic so two revisions are measured on the same bytes.

    127 is the e8m0 encoding of ``2 ** 0``, the exponent that leaves the block sums alone. The file's
    own grids are not needed: the op count and the node count are shape properties, not value ones.
    """
    generator = torch.Generator(device="cpu").manual_seed(0)
    a_codes = torch.randint(0, 255, (m, k), generator=generator, dtype=torch.int32).to(torch.uint8).to(device)
    b_codes = torch.randint(0, 255, (n, k), generator=generator, dtype=torch.int32).to(torch.uint8).to(device)
    a_scale = torch.full((m, k // mma.BLOCK), 127, dtype=torch.uint8, device=device)
    b_scale = torch.full((n // mma.BLOCK, k // mma.BLOCK), 127, dtype=torch.uint8, device=device)
    return a_codes, a_scale, b_codes, b_scale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=1280)
    parser.add_argument("--k", type=int, default=5120)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    if not torch.cuda.is_available():
        print("no CUDA device: this measures driver graph memory and has nothing to say without one")
        return 2

    device = torch.device(args.device)
    print(f"torch {torch.__version__} cuda {torch.version.cuda} {torch.cuda.get_device_name(device)}")
    print(f"shape M={args.m} N={args.n} K={args.k} ({args.k // mma.BLOCK} K blocks)")

    given = operands(args.m, args.n, args.k, device)
    # Warm up *off* the graph: the emulation builds its tables on first use, and a cold cache inside a
    # capture is the pageable H2D those tables exist to keep out -- it would be measured as graph cost.
    for _ in range(3):
        out = mma.fp8_block_gemm(*given)
    torch.cuda.synchronize(device)
    del out
    torch.cuda.empty_cache()

    free0, total = torch.cuda.mem_get_info(device)
    print(f"free before: {free0 / MIB:.0f} MiB of {total / MIB:.0f} MiB")

    graphs = []
    previous = free0
    for index in range(args.repeats):
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                captured = mma.fp8_block_gemm(*given)
        except Exception as exc:  # noqa: BLE001 -- the failure is the measurement
            free_now, _ = torch.cuda.mem_get_info(device)
            print(f"capture {index} FAILED: {type(exc).__name__}: {str(exc)[:120]}")
            print(f"  committed {(free0 - free_now) / MIB:.0f} MiB over {index} captures")
            return 1
        graphs.append((graph, captured))
        torch.cuda.synchronize(device)
        free_now, _ = torch.cuda.mem_get_info(device)
        if index < 3 or (index + 1) % 10 == 0:
            print(
                f"  capture {index:3d}: +{(previous - free_now) / MIB:7.1f} MiB, "
                f"{(free0 - free_now) / MIB:8.1f} MiB total, free {free_now / MIB:.0f} MiB"
            )
        previous = free_now

    free_end, _ = torch.cuda.mem_get_info(device)
    committed = free0 - free_end
    print(
        f"OK: {len(graphs)} captures, {committed / MIB:.0f} MiB "
        f"({committed / MIB / max(1, len(graphs)):.1f} MiB a capture)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())