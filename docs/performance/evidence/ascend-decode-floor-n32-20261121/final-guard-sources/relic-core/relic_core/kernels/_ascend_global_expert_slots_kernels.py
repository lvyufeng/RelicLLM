"""Preserve separate expert FP32 slots through one complete-FFN collective."""
import triton
import triton.language as tl


@triton.jit
def prepare_global_ffn_slots_kernel(DOWN, SELECTED, IDS, SHARED, OUT,
    DIM: tl.constexpr, TOPK: tl.constexpr, LANES: tl.constexpr, TILE: tl.constexpr):
    lane = tl.arange(0, LANES)
    ids = tl.load(IDS + lane, lane < TOPK, other=2147483647)
    selected = tl.load(SELECTED + lane, lane < TOPK, other=-1)
    position = tl.sum(((ids[None, :] < ids[:, None]) & (lane[None, :] < TOPK)).to(tl.int32), axis=1)
    slot = tl.program_id(1)
    column = tl.program_id(0) * TILE + tl.arange(0, TILE)
    if slot < TOPK:
        expert = tl.min(tl.where((position == slot) & (lane < TOPK), ids, 2147483647), axis=0)
        source = tl.min(tl.where((selected == expert) & (selected >= 0), lane, TOPK), axis=0)
        value = tl.load(DOWN + source * DIM + column, (column < DIM) & (source < TOPK), other=0)
    else:
        value = tl.load(SHARED + column, column < DIM, other=0).to(tl.float32)
    tl.store(OUT + slot * DIM + column, value, column < DIM)


@triton.jit
def finish_global_ffn_slots_kernel(SLOTS, ERROR, OUT, DIM: tl.constexpr, TOPK: tl.constexpr, TILE: tl.constexpr):
    column = tl.program_id(0) * TILE + tl.arange(0, TILE)
    value = tl.full((TILE,), 0.0, tl.float32)
    for expert_slot in range(TOPK):
        partial = tl.load(SLOTS + expert_slot * DIM + column, column < DIM, other=0)
        # Explicit BF16 RNE boundary AFTER global TP partial reduction.
        bits = partial.to(tl.int32, bitcast=True)
        rounded = bits + 0x7fff + ((bits >> 16) & 1)
        magnitude = bits & 0x7fffffff
        finite = magnitude < 0x7f800000
        exceptional = tl.where(magnitude > 0x7f800000, (bits & -2147483648) | 0x7fc00000, bits & -65536)
        narrowed = tl.where(finite, rounded & -65536, exceptional).to(tl.float32, bitcast=True)
        value = value + narrowed
    shared = tl.load(SLOTS + TOPK * DIM + column, column < DIM, other=0)
    value = value + shared
    bad = tl.load(ERROR) != 0
    value = tl.where(bad, float('nan'), value)
    tl.store(OUT + column, value, column < DIM)
