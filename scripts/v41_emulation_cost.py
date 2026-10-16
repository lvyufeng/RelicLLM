#!/usr/bin/env python3
"""Count what one emulated fp8 GEMM costs, in operators, at the shapes a decode runs.

`relicllm/models/deepseek_v4_1/mma.py` reproduces the DeepSeek reference's `mma.sync` reduction in
elementwise torch, which is what makes a port reproduce the reference's *tokens* rather than a more
accurate answer -- and it is why a decode step is thousands of small tensor operations instead of a
handful of tensor-core calls. That op count is the quantity that decides two things at once:

* **Whether the decode body can be captured in a CUDA graph at all.** A graph records one node per
  launch, and a recorded node costs the driver memory to hold -- the measurement in
  `EMULATION_REFUSES_GRAPHS` is ~897 MiB a layer against the ~10 GiB a 24 GiB card has left after the
  weights. Fewer ops is the only lever on that number.
* **How slow the eager fallback is.** Every op is a host dispatch, and the eager path is what a card
  with no room for the graph runs.

The split into *views* and *kernel-launching* ops is the one that matters, so this reports it. A view
(`alias`, `select`, `as_strided`, `view`, …) has no kernel and no graph node; it is host bookkeeping.
Everything else launches, and everything else is a candidate node.

`torch.profiler` counts ATen dispatches, on CPU or CUDA alike, so this runs on a laptop and needs no
card -- which is the point: the number can be tracked per change rather than per GPU run.

Usage::

    python scripts/v41_emulation_cost.py
    python scripts/v41_emulation_cost.py --json

See `docs/superpowers/notes/v41-sm89-decode-accounting.md` for the reading this produced and what it
implies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402 -- after the path insert, like the repository's other scripts
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from relicllm.models.deepseek_v4_1 import mma  # noqa: E402

#: Ops that produce a view and launch nothing. A CUDA-graph recording gives these no node, so they
#: are counted separately rather than folded into a total that is meant to bound the graph.
_VIEW_OPS = frozenset(
    {
        "aten::alias",
        "aten::as_strided",
        "aten::detach",
        "aten::expand",
        "aten::narrow",
        "aten::permute",
        "aten::reshape",
        "aten::select",
        "aten::slice",
        "aten::squeeze",
        "aten::t",
        "aten::transpose",
        "aten::unsqueeze",
        "aten::view",
    }
)

#: ``(label, M, N, K)``. The three M values are the ones the module's own cost note uses: one row is
#: a decode step, nine is the checkpoint's recorded `wq_a` oracle shape, and thirteen is a short
#: prefill. ``N``/``K`` are `wq_a`'s real geometry, so the K-block count (160) is the model's.
_SHAPES: tuple[tuple[str, int, int, int], ...] = (
    ("decode-m1", 1, 1280, 5120),
    ("decode-m9", 9, 1280, 5120),
    ("prefill-m13", 13, 1280, 5120),
)


def _operands(m: int, n: int, k: int) -> tuple[torch.Tensor, ...]:
    """Codes and e8m0 scales for one call, deterministic so two runs count the same thing.

    The values are arbitrary but legal: the op count is a shape property, not a value one.
    """
    generator = torch.Generator().manual_seed(0)
    a_codes = torch.randint(0, 255, (m, k), generator=generator, dtype=torch.int32).to(torch.uint8)
    b_codes = torch.randint(0, 255, (n, k), generator=generator, dtype=torch.int32).to(torch.uint8)
    # 127 is the e8m0 encoding of 2**0, the exponent that leaves the block sums alone.
    a_scale = torch.full((m, k // mma.BLOCK), 127, dtype=torch.uint8)
    b_scale = torch.full((n // mma.BLOCK, k // mma.BLOCK), 127, dtype=torch.uint8)
    return a_codes, a_scale, b_codes, b_scale


def count(label: str, m: int, n: int, k: int) -> dict[str, object]:
    """One call's operator census, as a record rather than a print."""
    operands = _operands(m, n, k)
    mma.fp8_block_gemm(*operands)  # warm: the first call builds the e2m1 table

    with profile(activities=[ProfilerActivity.CPU]) as profiler:
        mma.fp8_block_gemm(*operands)

    events = profiler.key_averages()
    total = sum(event.count for event in events)
    views = sum(event.count for event in events if event.key in _VIEW_OPS)
    top = sorted(
        (event for event in events if event.key not in _VIEW_OPS),
        key=lambda event: -event.count,
    )[:8]
    return {
        "label": label,
        "shape": {"M": m, "N": n, "K": k, "k_blocks": k // mma.BLOCK},
        "total_ops": total,
        "view_ops": views,
        "kernel_ops": total - views,
        "top_kernel_ops": [[event.key, event.count] for event in top],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = parser.parse_args(argv)

    records = [count(label, m, n, k) for label, m, n, k in _SHAPES]
    if args.json:
        print(json.dumps({"block": mma.BLOCK, "records": records}, indent=2))
        return 0

    print(f"mma.BLOCK = {mma.BLOCK}")
    for record in records:
        shape = record["shape"]
        print(
            f"\n{record['label']}  M={shape['M']} N={shape['N']} K={shape['K']} "
            f"({shape['k_blocks']} K blocks)"
        )
        print(
            f"    total {record['total_ops']} ops = "
            f"{record['view_ops']} views + {record['kernel_ops']} kernel-launching"
        )
        for name, count_ in record["top_kernel_ops"]:
            print(f"        {count_:6d}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())