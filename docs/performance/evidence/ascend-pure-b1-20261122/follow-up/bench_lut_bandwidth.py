#!/usr/bin/env python3
"""Bounded bandwidth micro-benchmark for the native scale-LUT expert projection kernel.

This driver answers one question and nothing else: at the real per-card expert
geometry, is the accepted ``relic_scale_lut_tile_cube`` projection kernel limited
by memory bandwidth, by per-call fixed cost, or by tile padding? It never claims a
speedup. The receipt it writes carries ``hypothesis_not_yet_proven: true`` and the
raw timings so a reviewer can recompute every GB/s number.

Measured context it is meant to explain (recorded by the root, not re-derived
here): the accepted B=1 stack runs 11.61 tok/s = 86.1 ms/token on one 8x910B3;
a rank-0 profile of four excluded forwards sums to 86.99 ms/forward of device
kernel time, so the step is fully serialized; the expert path costs 14.83 ms and
reads ~564 MB per card per token, i.e. ~38 GB/s against a ~800 GB/s HBM peak.

Geometry (real, from the checkpoint and the owner's dispatch log):
    hidden 5120, moe_intermediate 2304, TP=2, EP=4, 96 experts per card,
    three matrices per expert = 9,400,320 B/card, one matrix per card
    = 9,400,320 / 3 = 3,133,440 B = n=1152 rows * K=5120 packed 4-bit
    (codes n*K/2 = 2,949,120 B + E8M0 n*K/32 = 184,320 B).

Contracts followed (read from source, no invented APIs):
  * ``relic_core.kernels.cube_packed_format64``: ``cube_geometry64(n, k)`` and
    ``convert_fp4_cube64(codes, scales)`` build the accepted
    ``relic.fp4.cube-n64-k128.v1`` layout; conversion is CPU-only, byte exact,
    padding free, and interprets neither nibbles nor scale bytes.
  * ``relic_core.kernels.ascendc_scale_lut_tile_cube``:
    ``AscendCScaleLUTTileCubeConsumer(library_path, sha256=..., block_dim=24)``
    verifies the native library checksum and its query ABI (output_rows must be
    64 for N64 compact, group_k=4, layout id 2, table words, scale range
    117..127, prepared tables, unsigned byte cast), then
    ``consumer(x, b_tile, s_tile, selected, *, world, n_output=None,
    certified_bf16_operands=True, certified_scale_lut_operands=True,
    scratch=..., accumulation=..., output_dtype=...)``.
  * shared validator ``_validate`` from ``relic_core.kernels.online_tile_cube``:
    ``b`` contiguous uint8 ``[owned, n//16, k//32, 16, 16]``; ``s`` uint8
    ``[owned, n//16, k//32, 16]``; ``x`` contiguous bf16 ``[1,K]`` or
    ``[groups,K]``; ``selected`` contiguous int32, ``1 <= groups <= 8``;
    ``n = tiles*16``; ``blocks % group_k == 0``; output fp32 ``[groups,n]``;
    scratch bf16 ``[block_dim, 2, output_rows*128]``; six distinct storages.

Known contract limit pinned by this driver: an activation with M=16 rows (or 16
selected slots) is not expressible through the accepted LUT consumer. The guard
is ``_validate``'s ``if not 0 < groups <= 8: raise ValueError("at most eight
selected slots and a bounded launch grid are supported")`` together with
``if x.ndim != 2 or x.shape[1] != blocks * BLOCK_K or x.shape[0] not in (1,
groups)``. The smallest faithful alternative, implemented here, is the maximum
accepted width (6 selected slots, one broadcast activation row) plus a
six-single-expert vs one-six-expert comparison that separates per-call fixed cost
from per-expert bandwidth.

Safety: single process, single card (device 0), no device reset, no CI/Jenkins
interaction, no CANN or checkpoint change, no resident-bank mutation, no global
provider installation. Any precondition failure exits non-zero. Target runtime
under three minutes.

Usage (root runs this on the leased Ascend host; nothing here is executed by the author):

    python3 docs/performance/evidence/ascend-pure-b1-20261122/follow-up/bench_lut_bandwidth.py \
        --library /path/to/librelic_scale_lut_tile_cube.so \
        --cann-set-env /usr/local/Ascend/ascend-toolkit/set_env.sh \
        --receipt /tmp/bench_lut_bandwidth_receipt.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

FORMAT_64 = "relic.fp4.cube-n64-k128.v1"
N_ROWS = 1152          # 2304 // 2: per-card half intermediate
K_DIM = 5120           # hidden size
N_EXPERTS = 6          # max accepted selected slots; the real card owns 96
TP_WORLD = 2           # matches the owner's TP=2 dispatch
SELECTED_GLOBAL = [0, 2, 4, 6, 8, 10]  # global IDs; local = id // TP_WORLD < owned
BLOCK_DIM = 24         # accepted prototype block dimension
N_OUTPUT = N_ROWS      # tiles * 16, and divisible by the consumer's 64 output rows

# Pins recorded by earlier accepted runs; overridable on the command line.
ACCEPTED_SO_SHA256 = "7dce3e3dfa9aa3fb9159a425bd71c8e41bfbd437a3ea197c7d4abc8cb9f9072c"
ACCEPTED_CANN_SET_ENV_SHA256 = "bbbda1c0f925b0e8673e8c83958d6bb333e0e7d3ed0eb389db1457d92b3c3b12"


def fail(message):
    print(f"PRECONDITION FAILED: {message}", file=sys.stderr)
    raise SystemExit(2)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_library(explicit):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.environ.get("RELIC_SCALE_LUT_LIB")
    if env:
        candidates.append(Path(env))
    roots = [Path.cwd()]
    roots.extend(Path(__file__).resolve().parents[:8])
    for root in roots:
        candidates.extend([
            root / "build/extensions/librelic_scale_lut_tile_cube.so",
            root / "build/librelic_scale_lut_tile_cube.so",
            root / "relic_core/csrc/ascend/scale_lut_tile_cube/build/lib/librelic_scale_lut_tile_cube.so",
        ])
    candidates.extend(sorted(Path("/private/build").glob("**/librelic_scale_lut_tile_cube.so")) if Path("/private/build").is_dir() else [])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    fail("native library not found; pass --library /path/to/librelic_scale_lut_tile_cube.so")


def discover_cann(explicit):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    for key in ("CANN_SET_ENV", "ASCEND_TOOLKIT_HOME", "ASCEND_HOME"):
        value = os.environ.get(key)
        if value:
            candidates.append(Path(value) / "set_env.sh")
    candidates.extend([
        Path("/usr/local/Ascend/ascend-toolkit/set_env.sh"),
        Path("/usr/local/Ascend/latest/set_env.sh"),
    ])
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    fail("CANN set_env.sh not found; pass --cann-set-env /path/to/ascend-toolkit/set_env.sh")


def _device_name(torch):
    try:
        if hasattr(torch.npu, "get_device_name"):
            return torch.npu.get_device_name(0)
    except BaseException:
        return None
    return None


def percentile(values, fraction):
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil(fraction * len(ordered))) - 1))
    return ordered[index]


def stats_us(samples):
    return {
        "count": len(samples),
        "min_us": min(samples),
        "median_us": percentile(samples, 0.5),
        "p95_us": percentile(samples, 0.95),
        "max_us": max(samples),
        "mean_us": sum(samples) / len(samples),
    }


def gbps(total_bytes, microseconds):
    return total_bytes / (microseconds * 1e-6) / 1e9


def timed_calls(torch, call, warmup, iterations):
    for _ in range(warmup):
        call()
    torch.npu.synchronize()
    pairs = []
    for _ in range(iterations):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        pairs.append((start, end))
    torch.npu.synchronize()
    return [start.elapsed_time(end) * 1000.0 for start, end in pairs]


def main():
    parser = argparse.ArgumentParser(description="Bounded LUT expert-projection bandwidth micro-benchmark")
    parser.add_argument("--library", default=None, help="path to librelic_scale_lut_tile_cube.so")
    parser.add_argument("--expected-so-sha256", default=ACCEPTED_SO_SHA256,
                        help="expected native library sha256 (pass '' to accept the discovered file)")
    parser.add_argument("--cann-set-env", default=None, help="path to the CANN set_env.sh used by the container")
    parser.add_argument("--expected-cann-sha256", default=ACCEPTED_CANN_SET_ENV_SHA256,
                        help="expected CANN set_env.sh sha256 (pass '' to accept the discovered file)")
    parser.add_argument("--receipt", required=True, help="path where the JSON receipt is written")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--large-reference-mib", type=int, default=256)
    parser.add_argument("--skip-large-reference", action="store_true")
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1:
        fail("warmup must be >= 0 and iterations >= 1")

    try:
        import torch
        import torch_npu  # noqa: F401  (registers the npu backend)
    except Exception as exc:  # pragma: no cover - server-side precondition
        fail(f"torch/torch_npu import failed: {exc}")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        fail("no NPU is visible in this container")
    torch.npu.set_device(0)
    device = torch.device("npu", 0)
    torch.npu.synchronize()

    library = discover_library(args.library)
    library_sha = sha256_file(library)
    expected_library = args.expected_so_sha256 or ""
    if expected_library and library_sha != expected_library:
        fail(f"native library sha256 {library_sha} does not match the accepted {expected_library}")
    cann = discover_cann(args.cann_set_env)
    cann_sha = sha256_file(cann)
    expected_cann = args.expected_cann_sha256 or ""
    if expected_cann and cann_sha != expected_cann:
        fail(f"CANN set_env.sh sha256 {cann_sha} does not match the accepted {expected_cann}")

    from relic_core.kernels.cube_packed_format64 import convert_fp4_cube64, cube_geometry64
    from relic_core.kernels.ascendc_scale_lut_tile_cube import (
        AscendCScaleLUTTileCubeConsumer, scale_lut_table_words)

    geometry = cube_geometry64(N_ROWS, K_DIM)
    if geometry.payload_bytes != N_ROWS * K_DIM // 2 + N_ROWS * K_DIM // 32:
        fail("cube_geometry64 did not preserve the exact packed byte count")
    tiles = N_ROWS // 16
    blocks = K_DIM // 32
    consumer = AscendCScaleLUTTileCubeConsumer(str(library), sha256=library_sha, block_dim=BLOCK_DIM)
    if consumer.weight_format != FORMAT_64 or consumer.output_rows != 64:
        fail(f"library is not the N64/64-row consumer (format={consumer.weight_format}, rows={consumer.output_rows})")
    if consumer.table_words != scale_lut_table_words(64):
        fail("native LUT table words differ from the accepted N64 table")
    if consumer.layout_id != 2 or consumer.group_k != 4:
        fail("native layout id / K32 grouping differs from the accepted consumer")

    generator = torch.Generator(device="cpu").manual_seed(0)
    bank_codes = torch.empty((N_EXPERTS, tiles, blocks, 16, 16), dtype=torch.uint8, device=device)
    bank_scales = torch.empty((N_EXPERTS, tiles, blocks, 16), dtype=torch.uint8, device=device)
    conversion = {"experts": N_EXPERTS, "format": geometry.report()["format"],
                  "payload_bytes_per_expert": geometry.payload_bytes}
    for slot in range(N_EXPERTS):
        codes_cpu = torch.randint(0, 256, (N_ROWS, K_DIM // 2), generator=generator, dtype=torch.uint8)
        # The accepted LUT consumer certifies the measured E8M0 range 117..127.
        scales_cpu = torch.randint(117, 128, (N_ROWS, K_DIM // 32), generator=generator, dtype=torch.uint8)
        if int(scales_cpu.min()) < 117 or int(scales_cpu.max()) > 127:
            fail("synthetic E8M0 bank left the certified 117..127 range")
        payload = convert_fp4_cube64(codes_cpu, scales_cpu)
        if payload.n != N_ROWS or payload.k != K_DIM or payload.payload_bytes != geometry.payload_bytes:
            fail("convert_fp4_cube64 changed the declared geometry or byte count")
        bank_codes[slot].copy_(payload.codes.reshape(tiles, blocks, 16, 16))
        bank_scales[slot].copy_(payload.scales.reshape(tiles, blocks, 16))
        del payload
    torch.npu.synchronize()

    activation = torch.randn((1, K_DIM), dtype=torch.bfloat16, device=device)
    scratch = torch.empty(consumer.stage_shape, dtype=torch.bfloat16, device=device)
    consumer.prepare_device_metadata(device)
    torch.npu.synchronize()

    selected_all = torch.tensor(SELECTED_GLOBAL, dtype=torch.int32, device=device)
    local_ids = [int(value) // TP_WORLD for value in SELECTED_GLOBAL]
    if any(local < 0 or local >= N_EXPERTS for local in local_ids):
        fail("selected global IDs do not map inside the synthetic owned bank")

    n_bytes_per_expert = N_ROWS * K_DIM // 2 + N_ROWS * K_DIM // 32
    if n_bytes_per_expert != 3_133_440:
        fail(f"per-expert byte count {n_bytes_per_expert} differs from the measured 3,133,440 B")

    variants = []
    def run_variant(label, groups, output_dtype):
        selected = selected_all[:groups].contiguous()
        accumulation = torch.empty((groups, N_OUTPUT), dtype=torch.float32, device=device)
        def call():
            return consumer(activation, bank_codes, bank_scales, selected,
                            world=TP_WORLD, n_output=None,
                            certified_bf16_operands=True, certified_scale_lut_operands=True,
                            scratch=scratch, accumulation=accumulation,
                            output_dtype=output_dtype)
        out = call()
        torch.npu.synchronize()
        finite = bool(torch.isfinite(out).all().item())
        if not finite:
            fail(f"{label}: synthetic output is not finite (a scale byte left 117..127?)")
        samples = timed_calls(torch, call, args.warmup, args.iterations)
        codes_bytes = groups * N_ROWS * K_DIM // 2
        scales_bytes = groups * N_ROWS * K_DIM // 32
        activation_bytes = K_DIM * 2
        # The kernel always writes the fp32 accumulation buffer (groups*N*4 B).
        # The default runtime path then narrows it, which reads that buffer and
        # writes a bf16 copy (groups*N*(4+2) B more). The fp32-output variant stops
        # at the accumulation write and does not narrow.
        output_bytes = groups * N_OUTPUT * 4
        cast_bytes = (groups * N_OUTPUT * 6) if output_dtype == torch.bfloat16 else 0
        touched = int(codes_bytes + scales_bytes + activation_bytes + output_bytes + cast_bytes)
        timing = stats_us(samples)
        variants.append({
            "label": label, "groups": groups, "output_dtype": str(output_dtype).replace("torch.", ""),
            "timing_us": timing, "touched_bytes": touched,
            "achieved_GB_per_s": gbps(touched, timing["median_us"]),
            "touched_breakdown": {"codes_bytes": int(codes_bytes), "scales_bytes": int(scales_bytes),
                                  "activation_bytes": activation_bytes,
                                  "fp32_output_write_bytes": int(output_bytes),
                                  "runtime_bf16_narrow_bytes": int(cast_bytes)},
        })
        return timing

    single_runtime = run_variant("groups_1_runtime_bf16", 1, torch.bfloat16)
    six_runtime = run_variant("groups_6_runtime_bf16", N_EXPERTS, torch.bfloat16)
    single_kernel = run_variant("groups_1_kernel_fp32", 1, torch.float32)
    six_kernel = run_variant("groups_6_kernel_fp32", N_EXPERTS, torch.float32)

    implied_fixed_us = (N_EXPERTS * single_runtime["median_us"]
                        - six_runtime["median_us"]) / (N_EXPERTS - 1)
    m16_block = {
        "status": "blocked_by_accepted_contract",
        "guard": "_validate (online_tile_cube): 'at most eight selected slots and a bounded launch grid' "
                 "and x.shape[0] in (1, groups); the consumer's output_rows=64 and groups<=8 cannot express "
                 "an M=16 activation tile",
        "smallest_faithful_alternative": "max accepted width: 6 selected slots on one broadcast activation "
                                         "row, plus the six-single-call vs one-six-group comparison",
        "implied_fixed_cost_per_call_us": implied_fixed_us,
        "note": "negative or small values mean per-call fixed cost is not the dominant term; this is a "
                "hypothesis to be read against the timing spread, not a proven causal claim",
    }

    reference = {"practical_reference_not_spec_peak": True}
    same_bytes = N_EXPERTS * (N_ROWS * K_DIM // 2 + N_ROWS * K_DIM // 32)
    source = torch.randint(0, 256, (same_bytes,), generator=generator, dtype=torch.uint8).to(device)
    destination = torch.empty_like(source)
    read_samples = timed_calls(torch, lambda: source.sum(), args.warmup, max(10, args.iterations // 2))
    copy_samples = timed_calls(torch, lambda: destination.copy_(source), args.warmup, max(10, args.iterations // 2))
    reference["same_byte_volume"] = {
        "bytes": int(same_bytes),
        "read_only_sum": {"timing_us": stats_us(read_samples),
                          "achieved_GB_per_s": gbps(same_bytes, percentile(read_samples, 0.5))},
        "copy_read_write": {"timing_us": stats_us(copy_samples),
                            "achieved_GB_per_s": gbps(2 * same_bytes, percentile(copy_samples, 0.5))},
    }
    if not args.skip_large_reference:
        large_bytes = args.large_reference_mib * 1024 * 1024
        large_source = torch.randint(0, 256, (large_bytes,), generator=generator, dtype=torch.uint8).to(device)
        large_destination = torch.empty_like(large_source)
        large_read = timed_calls(torch, lambda: large_source.sum(), 5, 10)
        large_copy = timed_calls(torch, lambda: large_destination.copy_(large_source), 5, 10)
        reference["large_volume_practical_peak"] = {
            "bytes": int(large_bytes),
            "read_only_sum": {"timing_us": stats_us(large_read),
                              "achieved_GB_per_s": gbps(large_bytes, percentile(large_read, 0.5))},
            "copy_read_write": {"timing_us": stats_us(large_copy),
                                "achieved_GB_per_s": gbps(2 * large_bytes, percentile(large_copy, 0.5))},
        }

    receipt = {
        "schema": "relic.pure-b1.evidence.lut-bandwidth-receipt.v1",
        "generated_utc": None,
        "hypothesis_not_yet_proven": True,
        "no_speedup_claimed": True,
        "purpose": "decide P0 feasibility for the expert projection kernel: bandwidth, per-call fixed "
                   "cost or tile padding, at the real per-card expert geometry",
        "measured_context_not_re_derived": {
            "accepted_stack_TPS": 11.61, "ms_per_token": 86.1,
            "device_kernel_ms_per_forward": 86.99,
            "expert_path_ms_per_forward": 14.83,
            "expert_bytes_per_card_per_token": 564_000_000,
            "implied_expert_GB_per_s": 38.0, "hbm_peak_GB_per_s_approx": 800.0,
        },
        "geometry": {
            "rows_per_projection": N_ROWS, "K": K_DIM,
            "per_expert_matrix_bytes": int(n_bytes_per_expert),
            "layout": geometry.report(),
            "bank_codes_shape": list(bank_codes.shape), "bank_scales_shape": list(bank_scales.shape),
            "blocks": blocks, "tiles": tiles, "block_dim": BLOCK_DIM,
            "world": TP_WORLD, "selected_global_ids": SELECTED_GLOBAL,
            "local_ids_after_world_division": local_ids,
            "synthetic_bank_experts": N_EXPERTS,
            "real_card_owned_experts": 96,
        },
        "library": {"path": str(library), "sha256": library_sha, "expected_sha256": expected_library,
                    "matches_expected": (not expected_library) or library_sha == expected_library},
        "cann": {"set_env_path": str(cann), "sha256": cann_sha, "expected_sha256": expected_cann,
                 "matches_expected": (not expected_cann) or cann_sha == expected_cann},
        "consumer": {"output_rows": consumer.output_rows, "layout_id": consumer.layout_id,
                     "group_k": consumer.group_k, "weight_format": consumer.weight_format,
                     "table_words": consumer.table_words, "table_bytes_per_device": consumer.table_words * 4,
                     "stage_shape": list(consumer.stage_shape), "stage_bytes": consumer.stage_nbytes,
                     "measured_scale_range": [consumer.scale_min, consumer.scale_max],
                     "prepared_decode_tables_once_per_device": True},
        "variants": variants,
        "m1_vs_m16": m16_block,
        "reference": reference,
        "environment": {
            "torch_version": torch.__version__,
            "torch_npu_version": getattr(torch_npu, "__version__", None),
            "device_name": _device_name(torch),
            "iterations": args.iterations, "warmup": args.warmup,
        },
        "interpretation_limits": [
            "one synthetic byte-exact projection per expert slot; routing, HCCL and layer scheduling excluded",
            "synthetic codes/scales exercise the accepted layout and certified scale range, not checkpoint values",
            "GB/s uses the touched-byte formula recorded per variant; compare variants, not absolute expectations",
            "any speedup inference from these numbers is explicitly not made here",
        ],
    }
    import datetime
    receipt["generated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    receipt_path = Path(args.receipt)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "receipt": str(receipt_path),
        "library_sha256": library_sha,
        "cann_sha256": cann_sha,
        "variants": {variant["label"]: {"median_us": variant["timing_us"]["median_us"],
                                        "GB_per_s": variant["achieved_GB_per_s"],
                                        "groups": variant["groups"]} for variant in variants},
        "implied_fixed_cost_us": implied_fixed_us,
    }, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except BaseException as exc:  # pragma: no cover - server-side failure path
        print(f"BENCHMARK FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(3)
