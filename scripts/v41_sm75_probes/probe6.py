"""Confirm relic-core's fp4_gemm hardcodes act_block_size=128.

Read-only; touches no model weights. Run with:

    PYTHONPATH=<relic-core checkout> python -I probe6.py

Produces the numbers in docs/superpowers/notes/relic-core-gemm-block-size-hazards.md, finding 1.
Case A is the V4.1 usage (block-32 activation scales) and is the silent wrong answer; case B feeds
block-128 scales and agrees; case C reads the same block-32 codes with a 128-wide expansion and
reproduces A exactly.
"""
import torch

torch.manual_seed(0)
DEV = torch.device("cuda:0")
from relic_core.kernels import ops as O
from relic_core.kernels.ops import Packed4BitWeightAlongK


def rscale(amax):
    return torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))


def ref_act_quant(x, bs):
    n = x.size(-1)
    flat = x.contiguous().view(-1, n).to(torch.float32)
    amax = flat.abs().view(-1, n // bs, bs).amax(dim=-1).clamp_min(1e-4)
    se = rscale(amax).to(torch.float8_e8m0fnu)
    y = torch.clamp(flat / se.to(torch.float32).repeat_interleave(bs, dim=-1),
                    -448.0, 448.0).to(torch.float8_e4m3fn)
    return y.view_as(x), se.view(*x.shape[:-1], n // bs)


def ubf16(v):
    return torch.pow(2.0, torch.floor(torch.log2(v.abs().clamp_min(1e-38))) - 7)


def cmp(name, a, b):
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    d = (a64 - b64).abs()
    amax = b64.abs().max()
    m = b64.abs().flatten() > amax * 1e-3
    u = (d.flatten() / ubf16(b64).flatten().clamp_min(1e-45))
    print(f"  {name}: max_rel_to_amax={(d.max()/amax).item():.3e} "
          f"max_ulp={u[m].max().item():.4f} mean={u[m].mean().item():.4f}")


M, N, K = 128, 512, 512
x = (torch.randn(M, K, device=DEV) * 2.0).to(torch.bfloat16)
w4 = torch.randn(N, K, device=DEV) * 0.02
codes = torch.clamp(torch.round(w4 / 0.02), -6, 6).abs().to(torch.uint8)
raw = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
b = raw.view(torch.int8).view(torch.float4_e2m1fn_x2)
pk = Packed4BitWeightAlongK(plain_shape=(N, K), layout_tensor=b)
ws32 = rscale(torch.rand(N, K // 32, device=DEV) * 0.04 + 0.001).to(torch.float8_e8m0fnu)
w_dq = O.soft_fp4_blockfp4_weight_dequant(pk, ws32, block_size=32, impl="auto").double()

q32, s32 = ref_act_quant(x, 32)
outA = O.fp4_gemm(q32, s32, b, ws32, torch.float32).to(torch.float64)
t32 = (q32.to(torch.float32).double()
       * s32.to(torch.float32).repeat_interleave(32, dim=-1).double()) @ w_dq.T
print("A) fp4_gemm with block-32 act scales (V4.1 usage):")
cmp("   fp4_gemm vs block-32 truth", outA.to(torch.bfloat16), t32.to(torch.bfloat16))

q128, s128 = ref_act_quant(x, 128)
outB = O.fp4_gemm(q128, s128, b, ws32, torch.float32).to(torch.float64)
t128 = (q128.to(torch.float32).double()
        * s128.to(torch.float32).repeat_interleave(128, dim=-1).double()) @ w_dq.T
print("B) fp4_gemm with block-128 act scales:")
cmp("   fp4_gemm vs block-128 truth", outB.to(torch.bfloat16), t128.to(torch.bfloat16))

a_32 = (q32.to(torch.float32).double()
        * s32.to(torch.float32).repeat_interleave(32, dim=-1).double())
a_as128 = (q32.to(torch.float32).double()
           * s32.to(torch.float32).repeat_interleave(128, dim=-1)[:, :K].double())
print("C) A vs the same act read with block-128 scale expansion:")
cmp("   literal-128 read vs correct-32 read",
    (a_as128 @ w_dq.T).to(torch.bfloat16), (a_32 @ w_dq.T).to(torch.bfloat16))
print("DONE")