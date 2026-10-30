"""Check a patched expert kernel against the eager arithmetic it replaced.

`qwen4_exp_fp8_slots_swiglu` is the fused decode expert: one launch pair for the whole top-k of
a token.  A change to its launch geometry is easy to get subtly wrong -- a block that reads the
wrong row slice produces plausible numbers -- so this compares it against the reference it was
written to match, recomputed in plain torch from the same FP8 codes.

The reference is deliberately naive: dequantize the codes with the block scales, do the two
matmuls, apply silu, sum the slots.  That is the arithmetic, written out; the kernel is the
optimisation of it.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=10)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--inter", type=int, default=640)
    parser.add_argument("--blocks", type=int, default=4, help="blocks_per_expert (the grid split)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse()
    sys.path.insert(0, os.getcwd())
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    E, H, I = args.experts, args.hidden, args.inter
    K = args.slots

    from relic_core.kernels.cuda_loader import load_cuda_kernel

    ops = load_cuda_kernel()
    if ops is None or not hasattr(ops, "qwen4_exp_fp8_slots_swiglu"):
        print("[parity] SKIP: the extension does not export qwen4_exp_fp8_slots_swiglu")
        return 1

    # Codes and per-128x128-block scales, the layout the checkpoint stores.
    def make(rows: int, cols: int):
        # e4m3 holds finite values in +-448 and has no infinity, so the codes are drawn as float
        # in that range and cast, rather than reinterpreted from arbitrary int8 bit patterns --
        # those include the NaN encoding, which would make a mismatch look like a kernel bug.
        codes = (torch.rand(E, rows, cols, device=device) * 16.0 - 8.0).to(torch.float8_e4m3fn)
        scale = torch.rand(E, (rows + 127) // 128, (cols + 127) // 128, device=device,
                           dtype=torch.bfloat16) / 64.0 + 0.001
        return codes, scale

    gu_codes, gu_scale = make(2 * I, H)
    dw_codes, dw_scale = make(H, I)

    def dequant(codes: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """[E, R, C] codes and [E, R/128, C/128] scales -> [E, R, C] float."""
        R, C = codes.shape[1], codes.shape[2]
        s = scale.repeat_interleave(128, dim=1).repeat_interleave(128, dim=2)
        return codes.to(torch.float32) * s[:, :R, :C].to(torch.float32)

    x = torch.randn(H, device=device, dtype=torch.bfloat16)
    ids = torch.randint(0, E, (K,), device=device, dtype=torch.int32)
    wts = torch.rand(K, device=device, dtype=torch.bfloat16)

    # Reference: decode, matmul, silu, sum.
    gu = dequant(gu_codes, gu_scale)                       # [E, 2I, H]
    dw = dequant(dw_codes, dw_scale)                       # [E, H, I]
    ref = torch.zeros(H, device=device, dtype=torch.float32)
    for slot in range(K):
        e = int(ids[slot])
        g = gu[e, :I] @ x.float()
        u = gu[e, I:] @ x.float()
        a = torch.nn.functional.silu(g) * u
        ref += (dw[e] @ a) * float(wts[slot])

    # The kernel's own layout: gate/up repacked into 16-byte chunk planes.
    packed = ops.qwen4_exp_fp8_pack_t(gu_codes)
    # The kernel reads the gate/up table as the planes; `packed` is that table in its layout.
    out = ops.qwen4_exp_fp8_slots_swiglu(
        x, ids, wts, packed, gu_scale, packed, dw_codes, dw_scale, args.blocks
    )
    got = out.float().sum(0) if out.dim() == 2 else out.float().reshape(-1, H).sum(0)

    diff = (got - ref).abs()
    # Per-element relative error is the wrong yardstick here: an output element that happens to
    # land near zero divides the bf16 rounding by almost nothing and reads as a mismatch.  What
    # is asked instead is whether the whole vector agrees to bf16's own precision.
    scale = ref.abs().max().item()
    rel_norm = (diff.max().item() / scale) if scale > 0 else float("inf")
    print(f"[parity] blocks={args.blocks} slots={K} max_abs={diff.max().item():.3e} "
          f"rel_to_max={rel_norm:.3e} mean_abs={diff.mean().item():.3e}")
    if not torch.isfinite(got).all() or not torch.isfinite(ref).all():
        # A NaN compares false against everything, so an overflowed mismatch would have slipped
        # through the tolerance test; it is named rather than tolerated.
        print(
            f"[parity] FAIL: non-finite output (got finite={bool(torch.isfinite(got).all())}, "
            f"ref finite={bool(torch.isfinite(ref).all())}, blocks={args.blocks})"
        )
        return 2
    # bf16 carries 8 explicit mantissa bits, so ~4e-3 of an operand's magnitude is the floor;
    # the tolerance is that, one order looser for the fp32 accumulation order.
    if rel_norm > 5e-2:
        print("[parity] FAIL: the kernel does not match the eager arithmetic")
        return 2
    print("[parity] OK")
    return 0


# A synthetic parity check compares the kernel against the arithmetic, which is what a layout
# change must preserve -- but it feeds random codes, and a tiling bug that drops a few rows of
# one expert still agrees with itself on data the bug also sees.  What caught one was the model:
# a decode of a real prompt ran from "We need to respond" to "We!!!" when the act phase stopped
# covering the tail of an expert's rows.  Keeping the two checks apart is deliberate; this note is
# here so the synthetic one is not mistaken for the whole story.

if __name__ == "__main__":
    raise SystemExit(main())