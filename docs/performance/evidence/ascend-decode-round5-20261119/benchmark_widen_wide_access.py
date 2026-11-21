"""Try wider element accesses in the FP4 widening kernel.

The released kernel loads one packed byte per lane and stores one 16-bit pair, which measures the same
throughput as a copy that does no decoding at all (83 GB/s, about a tenth of this card's bandwidth), so
the limit is element width rather than arithmetic. This probe compares the released byte-per-lane
kernel with a two-bytes-per-lane variant that loads one 16-bit word and stores one 32-bit word.

Every variant is byte-compared against the released widening. Timings and byte identity only: no model
arithmetic and no TPS claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa:F401
import triton
import triton.language as tl


@triton.jit
def _code_x2(code):
    """Twice the value of one released E2M1 code as a signed int32."""
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
def _bytes_per_lane(A, OUT, SIZE: tl.constexpr, TILE: tl.constexpr, INDEX64: tl.constexpr):
    """The released formulation: one packed byte in, one 16-bit pair out."""
    program = tl.program_id(0)
    if INDEX64:
        program = program.to(tl.int64)
    lanes = tl.arange(0, TILE)
    for tile in range(program, tl.cdiv(SIZE, TILE), tl.num_programs(0)):
        offset = tile * TILE + lanes
        active = offset < SIZE
        packed = tl.load(A + offset, active, other=0).to(tl.int32)
        tl.store(OUT + offset, _pair(packed).to(tl.uint16), active)


@triton.jit
def _words_per_lane(A16, OUT32, SIZE: tl.constexpr, TILE: tl.constexpr, INDEX64: tl.constexpr):
    """Two packed bytes per lane: one 16-bit load, one 32-bit store, four output bytes."""
    program = tl.program_id(0)
    if INDEX64:
        program = program.to(tl.int64)
    lanes = tl.arange(0, TILE)
    for tile in range(program, tl.cdiv(SIZE, TILE), tl.num_programs(0)):
        offset = tile * TILE + lanes
        active = offset < SIZE
        word = tl.load(A16 + offset, active, other=0).to(tl.int32)
        low = _pair(word & 255) | (_pair((word >> 8) & 255) << 16)
        tl.store(OUT32 + offset, low.to(tl.uint32), active)


@torch.inference_mode()
def main():
    out = Path(os.environ["RELIC_PERF_OUTPUT"])
    torch.npu.set_device(torch.device("npu", 0))
    report = {"variants": {}}

    def measure(label, launch, reference, tensor, size_bytes):
        launch()
        torch.npu.synchronize()
        identical = bool(torch.equal(tensor.view(torch.uint8), reference.view(torch.uint8)))
        seconds = []
        for _ in range(15):
            started = time.perf_counter()
            launch()
            torch.npu.synchronize()
            seconds.append(time.perf_counter() - started)
        median = statistics.median(seconds)
        report["variants"][label] = {
            "median_seconds": median,
            "effective_GB_per_s": (size_bytes * 3) / median / 1e9,
            "byte_identity_with_released_widening": identical,
        }
        print(label, round(median * 1e3, 4), "ms",
              round(report["variants"][label]["effective_GB_per_s"], 2), "GB/s",
              "identical" if identical else "MISMATCH", flush=True)

    for name, rows, cols in (("w1_w3", 2304, 2560), ("w2", 5120, 1152)):
        size = rows * cols
        pairs = size * 2
        packed = torch.randint(0, 256, (size,), dtype=torch.uint8, device="npu:0")
        widened = torch.empty((pairs,), dtype=torch.uint8, device="npu:0")
        reference = torch.empty_like(widened)
        as_u16 = widened.view(torch.uint16)
        as_u32 = widened.view(torch.uint32)
        packed_u16 = packed.view(torch.uint16)

        grid = (min(triton.cdiv(size, 2048), 65535),)
        _bytes_per_lane[grid](packed, as_u16, SIZE=size, TILE=2048, INDEX64=size >= 2**31)
        torch.npu.synchronize()
        reference.copy_(widened)
        torch.npu.synchronize()

        measure(f"{name}_released_bytes_per_lane_2048",
                lambda: _bytes_per_lane[grid](packed, as_u16, SIZE=size, TILE=2048,
                                              INDEX64=size >= 2**31),
                reference, widened, size)
        half = size // 2
        for tile in (1024, 2048, 4096):
            words = (min(triton.cdiv(half, tile), 65535),)
            measure(f"{name}_words_per_lane_{tile}",
                    lambda tile=tile, words=words, half=half: _words_per_lane[words](
                        packed_u16, as_u32, SIZE=half, TILE=tile, INDEX64=size >= 2**31),
                    reference, widened, size)
    with (out / "widen-wide-access-probe.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())