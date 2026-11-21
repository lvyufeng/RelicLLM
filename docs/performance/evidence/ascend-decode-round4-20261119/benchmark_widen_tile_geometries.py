"""Check the larger widening tiles on both expert projection geometries, byte for byte.

The measured sweep on the w1 geometry (2304 rows x 2560 packed bytes) showed tile 4096 widening in
0.2505 ms against 0.5428 ms at tile 512. The expert body also carries w2, whose packed shape is
5120 rows x 1152 packed bytes, so the larger tile has to be shown on that geometry as well before any
caller opts into it. This probe reports per-geometry timing and full byte identity against the
released widening; it makes no model or TPS claim.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa:F401

from relic_core.kernels import unpack, vector_fp4_decode


@torch.inference_mode()
def main():
    out = Path(os.environ["RELIC_PERF_OUTPUT"])
    torch.npu.set_device(torch.device("npu", 0))
    geometries = {
        # name: (rows, packed bytes per row)
        "w1_w3": (2304, 2560),
        "w2": (5120, 1152),
    }
    tiles = (1024, 2048, 4096)
    report = {"tiles": list(tiles), "geometries": {}}
    for name, (rows, cols) in geometries.items():
        packed = torch.randint(0, 256, (rows, cols), dtype=torch.uint8, device="npu:0")
        reference = unpack.unpack_fp4_to_int8(packed, backend="torch")
        torch.npu.synchronize()
        entry = {"rows": rows, "packed_cols": cols,
                 "packed_bytes": rows * cols, "widened_bytes": rows * cols * 2, "tiles": {}}
        for tile in tiles:
            value = vector_fp4_decode.unpack_fp4_vector(packed, mode="arithmetic", tile=tile)
            torch.npu.synchronize()
            identical = bool(torch.equal(value.view(torch.uint8), reference.view(torch.uint8)))
            del value
            torch.npu.synchronize()
            seconds = []
            for _ in range(15):
                started = time.perf_counter()
                vector_fp4_decode.unpack_fp4_vector(packed, mode="arithmetic", tile=tile)
                torch.npu.synchronize()
                seconds.append(time.perf_counter() - started)
            median = statistics.median(seconds)
            entry["tiles"][str(tile)] = {
                "median_seconds": median,
                "effective_GB_per_s": (rows * cols * 3) / median / 1e9,
                "full_byte_identity_with_released_widening": identical,
            }
            print(name, tile, round(median * 1e3, 4), "ms",
                  round(entry["tiles"][str(tile)]["effective_GB_per_s"], 2), "GB/s",
                  "identical" if identical else "MISMATCH", flush=True)
        report["geometries"][name] = entry
    with (out / "widen-tile-geometries.json").open("x") as handle:
        json.dump(report, handle, indent=2)
    ok = all(t["full_byte_identity_with_released_widening"]
             for g in report["geometries"].values() for t in g["tiles"].values())
    print("all_geometries_all_tiles_identical", ok, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())