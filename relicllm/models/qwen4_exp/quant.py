"""FP8 block-scaled weight handling for Qwen4-Exp.

The Flash-Next FP8 checkpoint stores routed experts as `float8_e4m3fn` codes plus
a BF16 scale per 128x128 tile, and the PLE n-gram table as codes plus one scalar
scale.  Both are *multipliers*: ``value = code * scale``.  `torch._scaled_mm` is
unavailable on SM75, so the only route is to dequantize into a tensor-core dtype
and run an ordinary GEMM.

Dequantization multiplies in the destination dtype rather than in float32.  For
float16 that is exact — every `e4m3` code is representable in float16, and a
BF16 scale has fewer mantissa bits than float16 — while avoiding a 4x-wide
intermediate.  For bfloat16 the product rounds to 8 mantissa bits, which is the
same rounding a native BF16 runtime would apply.
"""

from __future__ import annotations

import math

from dataclasses import dataclass

import torch

FP8_DTYPE = torch.float8_e4m3fn


@dataclass(frozen=True)
class FP8Tensor:
    """One block-scaled weight: `code` with `scale` over `block` tiles."""

    code: torch.Tensor
    scale: torch.Tensor
    block: tuple[int, int] = (128, 128)

    @property
    def shape(self) -> torch.Size:
        return self.code.shape

    @property
    def device(self) -> torch.device:
        return self.code.device

    def nbytes(self) -> int:
        return (
            self.code.numel() * self.code.element_size()
            + self.scale.numel() * self.scale.element_size()
        )

    def to(self, device, *, non_blocking: bool = False) -> "FP8Tensor":
        """Move the codes and scales without dequantizing."""
        return FP8Tensor(
            self.code.to(device, non_blocking=non_blocking),
            self.scale.to(device, non_blocking=non_blocking),
            self.block,
        )

    def dequantize(self, dtype: torch.dtype) -> torch.Tensor:
        return dequantize_block_fp8(self.code, self.scale, self.block, dtype)


def dequantize_block_fp8(
    code: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Expand a block-scaled FP8 weight to `dtype`.

    `code` is `[out, in]`; `scale` is `[ceil(out/block[0]), ceil(in/block[1])]`.  Either may keep
    a prefix of leading axes -- a layer's local rows, one expert per axis -- and `scale` then holds
    one grid per element of that prefix; the expansion broadcasts over the prefix and only the last
    two axes are tiled.

    A matrix that does not cover whole tiles keeps its shapes: `pack_gate_up_fp8` packs it into a
    larger one whose rows are whole tiles, and the grids are each padded to that larger tiling, so
    this only ever sees the packed layout.

    Nothing is copied to get to `dtype` from a host tensor: a file-backed mapping is expanded where
    it is read, which is several times faster than a copy of the codes followed by the same
    expansion on the card.
    """
    if code.dim() < 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(code.shape)}")
    lead, rows, cols = code.shape[:-2], code.shape[-2], code.shape[-1]
    block_rows, block_cols = int(block[0]), int(block[1])
    tiles = (-(-rows // block_rows), -(-cols // block_cols))
    if tuple(scale.shape)[-2:] != tiles:
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match a {block_rows}x{block_cols} "
            f"tiling of {(rows, cols)} (expected one {tiles} grid per leading element)"
        )
    if scale.numel() != tiles[0] * tiles[1] * math.prod(lead):
        raise ValueError(
            f"scale of shape {tuple(scale.shape)} does not hold one grid per element of the "
            f"leading shape {tuple(lead)}"
        )
    values = code.to(dtype)
    grid = scale.to(dtype)
    if rows % block_rows == 0 and cols % block_cols == 0:
        # Every tile is whole, so the scale broadcasts over a reshape without
        # materializing a full-size scale tensor.
        tiled = values.reshape(*lead, tiles[0], block_rows, tiles[1], block_cols)
        return (tiled * grid.reshape(*lead, tiles[0], 1, tiles[1], 1)).reshape(*lead, rows, cols)
    # Ragged tail: expand and crop.  Not hit by this checkpoint (640, 1280 and
    # 2560 all divide by 128) but kept so the helper is not silently wrong.  Both halves are
    # padded out to whole tiles by a broadcast rather than by repeat: values can wrap over the
    # tile axis (`[t0*b0, ...]` is contiguous as `[t0, b0, ...]`), but a grid cannot -- its
    # `b0` axis is not in memory -- so the grid is the one that gets materialized.
    padded = (tiles[0] * block_rows, tiles[1] * block_cols)
    if (rows, cols) == padded:
        expanded = values
    else:
        expanded = code.new_zeros(*lead, *padded).to(dtype)
        expanded[..., :rows, :cols] = values
    # A scale covers a tile, so along the tile axis it repeats -- a plain `expand`.
    grid = (
        grid.unsqueeze(-1)
        .unsqueeze(-3)
        .expand(*lead, tiles[0], block_rows, tiles[1], block_cols)
        .reshape(*lead, padded[0], padded[1])
    )
    return (expanded * grid)[..., :rows, :cols]


def fp8_scalar_dequantize(
    code: torch.Tensor,
    scale: torch.Tensor | float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize an FP8 tensor that shares one scale (the PLE n-gram table)."""
    factor = scale.to(dtype) if isinstance(scale, torch.Tensor) else scale
    return code.to(dtype) * factor


def _stack_codes(head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """`[head; tail]` along the code's row axis: the last two axes are the matrix's.

    Done by hand because that axis moves with rank -- last for a 2-D pair, second to last for a
    stack of experts -- and so that a stack of grid rows is not involved.
    """
    axis = head.dim() - 2
    if head.shape[:axis] != tail.shape[:axis] or head.shape[axis + 1 :] != tail.shape[axis + 1 :]:
        raise ValueError(f"cannot stack {tuple(head.shape)} and {tuple(tail.shape)} by rows")
    out = head.new_empty(*head.shape[:axis], head.shape[axis] + tail.shape[axis], *head.shape[axis + 1 :])
    rows = head.shape[axis]
    out.narrow(axis, 0, rows).copy_(head)
    out.narrow(axis, rows, tail.shape[axis]).copy_(tail)
    return out


def _stack_rows(head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """`[head; tail]` along the row axis of a matrix -- the second-to-last axis."""
    if head.shape[:-2] != tail.shape[:-2] or head.shape[-1] != tail.shape[-1]:
        raise ValueError(f"cannot stack {tuple(head.shape)} and {tuple(tail.shape)} by rows")
    out = head.new_empty(*head.shape[:-2], head.shape[-2] + tail.shape[-2], head.shape[-1])
    rows = head.shape[-2]
    out.narrow(-2, 0, rows).copy_(head)
    out.narrow(-2, rows, tail.shape[-2]).copy_(tail)
    return out


def pack_gate_up_fp8(
    gate: FP8Tensor,
    up: FP8Tensor,
) -> FP8Tensor:
    """Row-concatenate gate and up into the packed `[2*inter, hidden]` layout.

    The runtime's expert convention is one fused `gate_up` weight with gate rows first, which is how
    the BF16 checkpoint ships it and what `swiglu_expert`'s `chunk(2)` assumes.

    A 2-D pair joins on the row axis, and so does a stack of experts -- on each matrix's row axis.
    Two grids are accepted there.  Per-matrix, `inter=640` is five 128-row tiles, so the packed
    1280 rows are gate's five tile rows then `up`'s five.  The checkpoint instead writes one grid
    over the fused 1280 rows (also ten by twenty for the column axis, so tile rows are one axis of
    one grid); that is what a stack of experts' rows are read from, and it is carried through as
    is once it is the right height.
    """
    if gate.block != up.block:
        raise ValueError(f"gate/up block size mismatch: {gate.block} vs {up.block}")
    if gate.code.shape != up.code.shape:
        raise ValueError(f"gate shape {tuple(gate.code.shape)} and up {tuple(up.code.shape)} differ")
    rows, block_rows = gate.code.shape[-2], gate.block[0]
    own = -(-rows // block_rows)
    tiles = -(-(2 * rows) // block_rows)
    if rows % block_rows:
        raise ValueError(
            f"gate rows {rows} do not divide the {block_rows}-row scale block, so "
            "concatenating gate and up would misalign the scale grid"
        )
    code = _stack_codes(gate.code, up.code)
    if gate.code.dim() == 2:
        if tuple(gate.scale.shape) != tuple(up.scale.shape):
            raise ValueError(f"gate grid {tuple(gate.scale.shape)} and up {tuple(up.scale.shape)} differ")
        return FP8Tensor(code, torch.cat([gate.scale, up.scale], dim=0), gate.block)
    lead = gate.code.shape[:-2]
    for name, weight in (("gate", gate), ("up", up)):
        if tuple(weight.scale.shape) != (*lead, own, weight.scale.shape[-1]) and tuple(
            weight.scale.shape
        ) != (*lead, tiles, weight.scale.shape[-1]):
            raise ValueError(
                f"{name} grid {tuple(weight.scale.shape)} is neither one {own}-tile nor one "
                f"{tiles}-tile grid per expert"
            )
    if gate.scale.shape[:-2] == lead and gate.scale.shape[-2] == tiles:
        # Already one grid over the packed rows: gate's half is its head, up's its tail.
        return FP8Tensor(code, gate.scale, gate.block)
    return FP8Tensor(code, _stack_rows(gate.scale, up.scale), gate.block)
