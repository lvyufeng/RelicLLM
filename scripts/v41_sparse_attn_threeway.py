#!/usr/bin/env python3
"""Measure one prefill `sparse_attn` call three ways: CUDA headpair kernel, torch, reference.

`relic_core.kernels.ops.sparse_attn` dispatches prefill on `DEEPSEEK_PREFILL_SPARSE_ATTN_CUDA` and
`..._HEADPAIR_CUDA`, both read at import, when `q.is_cuda and q.size(1) > 1 and q.size(-1) == 512`.
The outer knob used to default to "1" and now defaults to "0" (relic-core PR #36, which made
`_sparse_attn_torch` -- the transcription of the released kernel -- the V4.1 prefill default);
the headpair knob still defaults to "1" but is inert on its own, selecting only *which* CUDA kernel
the outer opt-in gets.

The fused headpair kernel is a different function of the same inputs: warp `__shfl_xor_sync` dot,
one global max with the sink folded in, one fp32 denominator, and no bf16 numerator. Its own comment
says the summation order "does move the whole model". This script asks how far it is from the
reference, without loading the model: one synthetic call at the real shapes, read back from the
kernel, from `_sparse_attn_torch`, and -- when the checkpoint's own `kernel.py` is importable --
from the reference's `sparse_attn`, which is the thing both of the port's paths are supposed to
reproduce.

This is *not* a card-dependent difference: the 2080 Ti's extension exposes
`prefill_sparse_attn_headpair_forward` too. Host-expert runs appear to fall through to the torch
path because their attention tensors are on the CPU, so `q.is_cuda` is False and the branch is
unreachable -- no knob and no missing symbol is involved.

Run on the box that has the extension and the checkpoint:

    PYTHONPATH=<relic-core>:<relicllm> python scripts/v41_sparse_attn_threeway.py
    PYTHONPATH=<relic-core>:<relicllm> python scripts/v41_sparse_attn_threeway.py \\
        --inference /path/to/DeepSeek-V4.1-Flash/inference     # adds the reference arm

Do not add `-I`: it implies `-E`, which drops PYTHONPATH.
"""
from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inference", default=None,
                        help="the checkpoint's own inference/ dir; enables the reference arm")
    parser.add_argument("--seqlen", type=int, default=9, help="tokens in the prefill")
    parser.add_argument("--heads", type=int, default=16,
                        help="query heads; the reference's real value is read off its config")
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--topk", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch

    import relic_core.kernels.ops as ops
    from relic_core.kernels.cuda_loader import load_cuda_kernel

    dev = "cuda"
    torch.manual_seed(args.seed)
    q = torch.randn(1, args.seqlen, args.heads, args.dim, device=dev, dtype=torch.bfloat16)
    kv = torch.randn(1, args.kv_len, args.dim, device=dev, dtype=torch.bfloat16)
    # A sink is a per-head bias that sits in the denominator; the reference's are O(1)-O(10), and
    # its magnitude matters because exp(sink - max) is a term in the denominator.
    attn_sink = (torch.randn(args.heads, device=dev, dtype=torch.float32) * 2.0 + 4.0)
    idx = torch.randint(0, args.kv_len, (1, args.seqlen, args.topk), device=dev, dtype=torch.int32)
    idx[0, :, :3] = -1  # the out-of-range branch the reference turns into -inf and kv 0
    scale = float(args.dim) ** -0.5

    print(f"q {tuple(q.shape)}  kv {tuple(kv.shape)}  topk {args.topk}  scale {scale:.6f}")

    arm = {}
    try:
        ext = load_cuda_kernel()
        arm["cuda_headpair"] = ext.prefill_sparse_attn_headpair_forward(
            q.contiguous(), kv.contiguous(), attn_sink.contiguous(), idx.contiguous(), scale)
        print("cuda_headpair: ran")
    except Exception as exc:  # noqa: BLE001 - the point is to report which arms are even reachable
        print(f"cuda_headpair: unavailable ({type(exc).__name__}: {str(exc)[:120]})")

    arm["torch"] = ops._sparse_attn_torch(q, kv, attn_sink, idx, scale)
    print("torch: ran")

    if args.inference:
        sys.path.insert(0, args.inference)
        import kernel as ref_kernel  # the checkpoint's own sparse_attn, on its own kernels

        arm["reference"] = ref_kernel.sparse_attn(q, kv, attn_sink, idx, scale).to(torch.bfloat16)
        print("reference: ran")

    names = list(arm)
    print()
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            x, y = arm[names[a]].float().reshape(-1), arm[names[b]].float().reshape(-1)
            rel = float((x - y).norm() / y.norm())
            cos = float(torch.dot(x, y) / (x.norm() * y.norm()))
            print(f"  {names[a]:<15} vs {names[b]:<15} relL2 {rel:.5e}  cos {cos:.7f}  "
                  f"maxabs {float((x - y).abs().max()):.4e}")
    for name in names:
        print(f"  ||{name}|| {float(arm[name].float().norm()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())