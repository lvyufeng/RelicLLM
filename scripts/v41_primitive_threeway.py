#!/usr/bin/env python3
"""Compare relic-core's activation-quantization primitives against each other and the reference.

The port's `quantized_linear` is pure torch and therefore arch-independent, but the V4.1 attention
calls `fp4_act_quant` and `act_quant` straight out of `relic_core.kernels.ops`, where `_auto_impl`
picks triton when triton is importable. Triton compiles per architecture, so an op that selects
triton can be correct on one card and not on another from the *same* source -- which is the shape of
a defect that makes one box agree with the reference while the other does not. This measures each
primitive's triton and torch implementations against each other, and against the checkpoint's own
`kernel.py` where that is importable.

    PYTHONPATH=<relic-core>:<relicllm> python scripts/v41_primitive_threeway.py
    PYTHONPATH=<relic-core>:<relicllm> python scripts/v41_primitive_threeway.py \\
        --inference /path/to/DeepSeek-V4.1-Flash/inference

Run it on both cards and compare the tables; a primitive whose triton-vs-torch gap is card-dependent
is the one to look at.

Do not add `-I`: it implies `-E`, which drops PYTHONPATH.
"""
from __future__ import annotations

import argparse
import sys

import torch


def _e2m1_codes(t: torch.Tensor) -> torch.Tensor:
    """A packed e2m1 payload as one 4-bit code per element, low nibble first.

    Low-first is the order `_pack_fp4_codes` writes and the reference's decoder reads.
    """
    b = t.reshape(-1).view(torch.uint8).to(torch.int64)
    out = torch.empty(b.numel() * 2, dtype=torch.int64, device=b.device)
    out[0::2] = b & 0xF
    out[1::2] = b >> 4
    return out


def report(name: str, arms: dict) -> None:
    names = [n for n in arms if arms[n] is not None]
    if len(names) < 2:
        print(f"  {name}: only {names} available; nothing to compare")
        return
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            x, y = arms[names[a]], arms[names[b]]
            if tuple(x.shape) != tuple(y.shape):
                print(f"  {name:<22} {names[a]:<10} vs {names[b]:<10} shapes differ: "
                      f"{tuple(x.shape)} vs {tuple(y.shape)}")
                continue
            packed = getattr(torch, "float4_e2m1fn_x2", None)
            if packed is not None and x.dtype == packed:
                # `float4_e2m1fn_x2` packs two e2m1 values into a byte. Torch reports it as a
                # floating dtype but implements no cast to float, so arithmetic on it -- `(x - y)`
                # here, `x.float()` before -- trips a device-side assert, which reads as a fault in
                # the quantizer when it is a fault in the comparison.
                #
                # Compare the codes, not the bytes, because a differing byte has two causes that
                # want opposite responses. A differing *value* is a quantizer defect. A differing
                # *sign bit on a zero* -- e2m1 spends two of its sixteen codes on +0.0 and -0.0 --
                # is inert: both dequantize to a zero and add to a sum identically. Comparing bytes
                # merges them, and comparing dequantized floats drops the second entirely and then
                # reports a byte count it cannot explain (`+0.0 != -0.0` is False in IEEE).
                xc, yc = _e2m1_codes(x), _e2m1_codes(y)
                bad = xc != yc
                zeros = ((xc == 0) | (xc == 8)) & ((yc == 0) | (yc == 8))
                value_bad = bad & ~zeros
                if not int(bad.sum()):
                    print(f"  {name:<22} {names[a]:<10} vs {names[b]:<10} "
                          f"identical, all {xc.numel()} e2m1 codes")
                    continue
                print(f"  {name:<22} {names[a]:<10} vs {names[b]:<10} "
                      f"{int(value_bad.sum())} value differences, "
                      f"{int((bad & zeros).sum())} sign-of-zero, of {xc.numel()} codes")
                pairs = torch.stack((xc[value_bad], yc[value_bad])).T
                uniq, counts = torch.unique(pairs, dim=0, return_counts=True)
                for pair, count in sorted(zip(uniq.tolist(), counts.tolist()),
                                          key=lambda kv: -kv[1])[:6]:
                    print(f"      code {pair[0]:2d} vs {pair[1]:2d}  {count:8d} times")
                continue
            x, y = x.float().reshape(-1), y.float().reshape(-1)
            rel = float((x - y).norm() / y.norm().clamp_min(1e-30))
            print(f"  {name:<22} {names[a]:<10} vs {names[b]:<10} relL2 {rel:.5e}  "
                  f"maxabs {float((x - y).abs().max()):.4e}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inference", default=None,
                        help="the checkpoint's own inference/ dir; enables the reference arms")
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import relic_core.kernels.ops as ops

    print(f"triton available (_USE_TRITON): {ops._USE_TRITON}")
    print(f"device: {torch.cuda.get_device_name(0)} capability {torch.cuda.get_device_capability(0)}")
    if args.inference:
        sys.path.insert(0, args.inference)
    ref_kernel = None
    if args.inference:
        try:
            import kernel as ref_kernel  # the checkpoint's own implementations
        except Exception as exc:  # noqa: BLE001
            print(f"reference kernel.py unavailable: {type(exc).__name__}: {str(exc)[:120]}")

    torch.manual_seed(args.seed)
    dev = "cuda"
    # A scale near 1 with a long tail: block scales are amax-driven, so a Gaussian alone hides the
    # reduction-width differences that only show up when one element dominates its block.
    x = torch.randn(args.rows, args.dim, device=dev, dtype=torch.bfloat16)
    x[:4] *= 30.0

    print("\nprimitives")
    # `fp4_act_quant` has no `impl` parameter: its body is pure torch by construction, so it is the
    # control -- an op with no arch dependency at all. Only the reference arm varies here.
    for label, block_size in (("fp4/32", 32), ("fp4/16", 16)):
        arms = {}
        try:
            arms["torch"] = ops.fp4_act_quant(x, block_size=block_size)
        except Exception as exc:  # noqa: BLE001
            print(f"  fp4_act_quant {label} torch: {type(exc).__name__}: {str(exc)[:100]}")
            arms["torch"] = None
        if ref_kernel is not None:
            try:
                arms["reference"] = ref_kernel.fp4_act_quant(x, block_size, False)
            except Exception as exc:  # noqa: BLE001
                print(f"  fp4_act_quant {label} reference: {type(exc).__name__}: {str(exc)[:100]}")
                arms["reference"] = None
        report(f"fp4_act_quant {label}", {k: (v[0] if isinstance(v, tuple) else v)
                                          for k, v in arms.items()})

    arms = {}
    arms["torch"] = ops.blockfp8_act_quant(x, block_size=32, round_scale_to_pow2=True,
                                           eps=1e-4, impl="torch")[0].float()
    if ops._USE_TRITON:
        arms["triton"] = ops.blockfp8_act_quant(x, block_size=32, round_scale_to_pow2=True,
                                                eps=1e-4, impl="triton")[0].float()
    if ref_kernel is not None:
        arms["reference"] = ref_kernel.act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)[0].float()
    report("act_quant ue8m0/32", arms)

    print("\nhc_split_sinkhorn (`hc_pre`'s mixing coefficients)")
    # `hc_split` is the one V4.1 primitive whose `_auto_impl` picks triton on *both* cards, so it is
    # the only place besides `sparse_attn` where the same source can compile to different arithmetic.
    hc_mult = 4
    mix_hc = (2 + hc_mult) * hc_mult
    mixes = torch.randn(args.rows, mix_hc, device=dev, dtype=torch.float32)
    hc_scale = torch.randn(hc_mult, device=dev, dtype=torch.float32)
    hc_base = torch.randn(mix_hc, device=dev, dtype=torch.float32)
    names = ("pre", "post", "comb")
    by_part: dict[str, dict] = {name: {} for name in names}
    for impl in ("torch", "triton"):
        if impl == "triton" and not ops._USE_TRITON:
            continue
        try:
            out = ops.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, impl=impl)
        except Exception as exc:  # noqa: BLE001
            print(f"  {impl}: {type(exc).__name__}: {str(exc)[:100]}")
            continue
        parts = out if isinstance(out, tuple) else (out,)
        for name, part in zip(names, parts):
            by_part[name][impl] = part
    for name in names:
        report(f"hc_split_sinkhorn.{name}", by_part[name])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())