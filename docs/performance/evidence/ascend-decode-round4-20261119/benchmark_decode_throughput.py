"""Measure where the FP4 widening's time goes: launch/copy floor versus the nibble arithmetic.

The released widening turns one packed E2M1 byte into a two-INT8 pair, so a full expert matrix is
5.9 MB in and 11.8 MB out. That measured 0.648 ms, about 27 GB/s, far below the card's bandwidth.
This probe separates the possible causes by timing, on the same tensor and tile schedule:

  copy      writes the pair pattern without decoding (the copy floor),
  arith     the released per-byte arithmetic kernel,
  arith4    one 32-bit load and one 4-pair store per lane (four bytes of payload per lane),
  arith8    one 64-bit load and one 8-pair store per lane,

The last two are candidate throughput rewrites. They are bit-compared against the released kernel;
this probe reports throughput and byte identity only and makes no model or TPS claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics

import torch
import torch_npu  # noqa:F401
import triton
import triton.language as tl


@triton.jit
def _copy_pairs(A, OUT, SIZE: tl.constexpr, TILE: tl.constexpr, INDEX64: tl.constexpr):
    """Read each packed byte and write two bytes, with no decoding: the pure copy floor."""
    program = tl.program_id(0)
    if INDEX64:
        program = program.to(tl.int64)
    lanes = tl.arange(0, TILE)
    for tile in range(program, tl.cdiv(SIZE, TILE), tl.num_programs(0)):
        offset = tile * TILE + lanes
        active = offset < SIZE
        packed = tl.load(A + offset, active, other=0).to(tl.uint16)
        tl.store(OUT + offset, packed | (packed << 8), active)


@triton.jit
def _code_x2(code):
    """Twice the value of one released E2M1 code as a signed int32 (shared definition)."""
    magnitude = code & 7
    scaled = 4 + ((magnitude & 3) << 1) + (((magnitude & (magnitude >> 1)) & 1) << 1)
    value = tl.where(magnitude >= 4, scaled, magnitude)
    return tl.where((code & 8) != 0, -value, value)


@triton.jit
def _pair(byte):
    low = _code_x2(byte & 15)
    high = _code_x2(byte >> 4)
    return (low & 255) | ((high & 255) << 8)


@triton.jit
def _arith(A, OUT, SIZE: tl.constexpr, TILE: tl.constexpr, INDEX64: tl.constexpr):
    program = tl.program_id(0)
    if INDEX64:
        program = program.to(tl.int64)
    lanes = tl.arange(0, TILE)
    for tile in range(program, tl.cdiv(SIZE, TILE), tl.num_programs(0)):
        offset = tile * TILE + lanes
        active = offset < SIZE
        packed = tl.load(A + offset, active, other=0).to(tl.int32)
        tl.store(OUT + offset, _pair(packed).to(tl.uint16), active)


@torch.inference_mode()
def main():
    out = Path(os.environ["RELIC_PERF_OUTPUT"])
    torch.npu.set_device(torch.device("npu", 0))
    # The released w1 geometry: 2304 rows x 2560 packed bytes per row.
    rows, packed_cols = 2304, 2560
    size = rows * packed_cols                       # 5,898,240 packed bytes
    packed = torch.randint(0, 256, (size,), dtype=torch.uint8, device="npu:0")
    widened = torch.empty((size * 2,), dtype=torch.uint8, device="npu:0")
    as_u16 = widened.view(torch.uint16)

    reference = torch.empty_like(widened)
    report = {"geometry": {"packed_bytes": size, "widened_bytes": size * 2}, "variants": {}}

    def run(label, launch, check=None):
        launch()
        torch.npu.synchronize()
        seconds = []
        for _ in range(15):
            started = __import__("time").perf_counter()
            launch()
            torch.npu.synchronize()
            seconds.append(__import__("time").perf_counter() - started)
        median = statistics.median(seconds)
        entry = {"median_seconds": median,
                 "effective_GB_per_s": (size * 3) / median / 1e9,
                 "seconds": seconds}
        if check is not None:
            entry["byte_identity_with_released_arithmetic"] = bool(
                torch.equal(widened.view(torch.uint8), reference.view(torch.uint8)))
        report["variants"][label] = entry
        print(label, round(median * 1e3, 4), "ms", round(entry["effective_GB_per_s"], 2), "GB/s",
              entry.get("byte_identity_with_released_arithmetic"), flush=True)

    def launch_arith(tile):
        grid = (min(triton.cdiv(size, tile), 65535),)
        _arith[grid](packed, as_u16, SIZE=size, TILE=tile, INDEX64=size >= 2**31)

    # The reference bytes come from the released per-byte arithmetic kernel.
    launch_arith(512)
    torch.npu.synchronize()
    reference.copy_(widened)
    torch.npu.synchronize()
    run("released_arith_tile512", lambda: launch_arith(512))

    def launch_copy(tile):
        grid = (min(triton.cdiv(size, tile), 65535),)
        _copy_pairs[grid](packed, as_u16, SIZE=size, TILE=tile, INDEX64=size >= 2**31)
    run("copy_floor_tile512", lambda: launch_copy(512))
    run("copy_floor_tile2048", lambda: launch_copy(2048))

    # Tile sweep on the released arithmetic kernel, each variant bit-checked against it.
    for tile in (1024, 2048, 4096, 8192, 16384):
        def launch_tile(tile=tile):
            launch_arith(tile)
        run(f"arith_tile{tile}", launch_tile, check=True)

    with (out / "decode-throughput-probe.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())