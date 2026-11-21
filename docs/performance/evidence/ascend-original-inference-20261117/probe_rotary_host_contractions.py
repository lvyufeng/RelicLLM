"""Measure this host's complex64 contraction per component; CPU only."""
from __future__ import annotations

import ctypes
import ctypes.util
import hashlib
import json
import math
from pathlib import Path
import struct

import torch

ROOT = Path("<workspace>/relic-v41-infer-20261117-1424")


def f32(value):
    try:
        return struct.unpack("f", struct.pack("f", value))[0]
    except OverflowError:
        return math.copysign(float("inf"), value)


def bits(value):
    return struct.unpack("I", struct.pack("f", value))[0]


def main():
    lib = ctypes.CDLL(ctypes.util.find_library("m") or None)
    fma = lib.fmaf
    fma.argtypes = (ctypes.c_float, ctypes.c_float, ctypes.c_float)
    fma.restype = ctypes.c_float
    generator = torch.Generator().manual_seed(8611)
    samples = torch.randn((4096, 4), generator=generator, dtype=torch.float32)
    arrays = samples.tolist()
    complex_x = torch.view_as_complex(samples[:, :2].contiguous())
    complex_table = torch.view_as_complex(samples[:, 2:].contiguous())
    measured = torch.view_as_real(complex_x * complex_table).tolist()
    counts = {component: {mode: 0 for mode in ("first", "second", "unfused")}
              for component in ("real", "imag")}
    witnesses = []
    for (a, b, c, sine), (real, imag) in zip(arrays, measured):
        possibilities = {
            "real": {
                "first": fma(a, c, -f32(b * sine)),
                "second": fma(-b, sine, f32(a * c)),
                "unfused": f32(f32(a * c) - f32(b * sine)),
            },
            "imag": {
                "first": fma(a, sine, f32(b * c)),
                "second": fma(b, c, f32(a * sine)),
                "unfused": f32(f32(a * sine) + f32(b * c)),
            },
        }
        for component, actual in (("real", real), ("imag", imag)):
            for mode, value in possibilities[component].items():
                counts[component][mode] += bits(actual) != bits(value)
        if len(witnesses) < 8 and any(len({bits(v) for v in alternatives.values()}) > 1
                                     for alternatives in possibilities.values()):
            witnesses.append({"operands_bits": [bits(v) for v in (a, b, c, sine)],
                              "measured_bits": [bits(real), bits(imag)],
                              "candidate_bits": {component: {mode: bits(value) for mode, value in alternatives.items()}
                                                 for component, alternatives in possibilities.items()}})
    result = {"torch": torch.__version__, "cpu_only": True, "sample_count": len(arrays),
              "candidate_mismatch_counts_by_component": counts, "witnesses": witnesses,
              "original_cuda_reference_parity_claimed": False,
              "sample_source_sha256": hashlib.sha256(samples.numpy().tobytes()).hexdigest()}
    with (ROOT / "logs/rotary-host-component-contractions.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
