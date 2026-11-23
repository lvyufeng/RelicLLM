"""Do the sparse indexer's mask-then-sort selection on the vector core instead of the AI CPU.

`attention.py:1173` and `attention.py:1326` both build a small int array, mask the entries the query
cannot reach, sort it ascending and cast to int32. On this build each of those steps becomes an AI CPU
operator, and an AI CPU operator occupies no cube or vector unit while serialising against the stream.
Measured on the registered lane that is 9 Greater + 9 Sort + 9 Cast + 4 Index = 31 operators a forward,
2,762 us, 3.5% of the step - almost all of it fixed dispatch cost rather than work.

The two call sites differ only in where the padding lands, because the sentinel they sort against is
different:

  site A  sentinel `tail`, a value past every real position -> padding trails the real ids
  site B  sentinel -1, which sorts below every real block id -> padding leads the real ids

This module reproduces each exactly. The selection itself is a rank: an entry's output position is the
number of reachable entries strictly below it, so the reader scatters instead of sorting. One pass per
element with the whole row compared against it, which is O(width) a program and needs no intra-block
exchange - and this Triton build offers no tensor slicing and no shuffles, so a bitonic sort in
registers is not available anyway.

Correctness rests on a property of the input rather than on the algorithm: ids are unique, and every tie
is between sentinels that all collapse to the same padding value. So the sorted result is uniquely
determined and no tie-break rule has to match `torch.sort`'s.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

#: Elements compared per iteration inside the rank kernel. Only affects register pressure.
_COMPARE_BLOCK = 256


@triton.jit
def _rank_scatter_kernel(VALUES, KEYS, OUT, WIDTH: tl.constexpr, OFFSET: tl.constexpr,
                         PAD_FIRST: tl.constexpr, COMPARE: tl.constexpr):
    """One program per row element.

    `OUT` must already hold the padding value in every slot; this kernel only writes the reachable ids
    into their ranked slots, so the padding positions keep what the caller filled in.

    The rank is the number of reachable entries strictly below this one, plus - when the padding sorts
    first, as at site B - the number of unreachable entries, which all sit ahead of every real id.
    """
    index = tl.program_id(0)
    row = index // WIDTH
    lane = index % WIDTH
    base = row * WIDTH
    value = tl.load(VALUES + base + lane)
    reachable = value > float('-inf')
    ident = (tl.load(KEYS + base + lane) + OFFSET).to(tl.int32)
    # The reference sorts the *ids*, not the scores: it builds `where(reachable, keys + offset,
    # tail)` and sorts that. So an entry's rank is the number of reachable ids below it.
    below = tl.zeros((1,), dtype=tl.int32)
    padding = tl.zeros((1,), dtype=tl.int32)
    for start in tl.static_range(0, WIDTH, COMPARE):
        offset = tl.arange(0, COMPARE)
        other_lane = start + offset
        live = other_lane < WIDTH
        other_value = tl.load(VALUES + base + other_lane, live, other=float('-inf'))
        other_reachable = live & (other_value > float('-inf'))
        other_ident = (tl.load(KEYS + base + other_lane, live, other=0) + OFFSET).to(tl.int32)
        below += tl.sum((other_reachable & (other_ident < ident)).to(tl.int32), axis=0)
        padding += tl.sum((live & ~other_reachable).to(tl.int32), axis=0)
    position = below
    if PAD_FIRST:
        position = position + padding
    tl.store(OUT + base + position, ident, mask=reachable)


def _launch(values, keys, *, offset: int, pad_first: bool, out: torch.Tensor):
    width = values.shape[-1]
    rows = values.numel() // width
    _rank_scatter_kernel[(rows * width,)](
        values.contiguous(), keys.contiguous(), out, width, int(offset), pad_first,
        COMPARE=_COMPARE_BLOCK,
    )
    return out


def masked_sort_trailing(values: torch.Tensor, keys: torch.Tensor, offset: int) -> torch.Tensor:
    """Site A: reachable ids ascending and cast to int32, then `-1` padding.

    Matches `torch.where(values > -inf, keys + offset, tail).sort().values`, then `tail -> -1`.
    """
    out = torch.full(values.shape, -1, dtype=torch.int32, device=values.device)
    return _launch(values, keys, offset=offset, pad_first=False, out=out)


def masked_sort_leading(values: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Site B: `-1` padding then reachable block ids ascending, as int32.

    Matches `torch.where(values > -inf, keys, -1).sort().values.int()`.
    """
    out = torch.full(values.shape, -1, dtype=torch.int32, device=values.device)
    return _launch(values, keys, offset=0, pad_first=True, out=out)


def masked_sort_trailing_with_tail(values, keys_plus_offset, width: int, tail: int) -> torch.Tensor:
    """Site A's exact form: ids already offset, padding value already named.

    `attention.py:1173` offsets the keys and names the sentinel at the call site, so this takes both
    rather than deriving them, which keeps the replacement a drop-in for that one expression.
    """
    out = torch.full(values.shape, -1, dtype=torch.int32, device=values.device)
    _rank_scatter_kernel[((values.numel() // values.shape[-1]) * values.shape[-1],)](
        values.contiguous(), keys_plus_offset.contiguous(), out, values.shape[-1], 0, False,
        COMPARE=_COMPARE_BLOCK,
    )
    return out
