"""Isolate the fp4_act_quant device-side assert on sm_75.

Read-only. Run with: PYTHONPATH=<relic-core> python -I probe3.py <case>
Cases: fp8_triton, fp4_act_quant, fp4_act_quant_fp32, fp4_gemm, triton_gemm.
"""
import sys
import torch

torch.manual_seed(0)
DEV = torch.device("cuda:0")
from relic_core.kernels import ops as O

which = sys.argv[1]


def ref_scale(n, k, bs=32):
    amax = torch.rand(n, k // bs, device=DEV) * 0.04 + 0.001
    s = torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))
    return s.to(torch.float8_e8m0fnu)

if which == "fp8_triton":
    x = (torch.randn(64, 256, device=DEV) * 3.0).to(torch.bfloat16)
    q, s = O.blockfp8_act_quant(x, block_size=32, impl="triton")
    torch.cuda.synchronize()
    print("fp8 triton act_quant OK", q.dtype, s.dtype)

elif which == "fp4_act_quant":
    x = (torch.randn(64, 256, device=DEV) * 3.0).to(torch.bfloat16)
    print("x dtype", x.dtype, "last dim", x.size(-1))
    pq, ps = O.fp4_act_quant(x, block_size=32)
    torch.cuda.synchronize()
    print("fp4_act_quant OK", pq.dtype, tuple(pq.shape), ps.dtype, tuple(ps.shape),
          "finite:", bool(torch.isfinite(pq.to(torch.float32)).all()))

elif which == "fp4_act_quant_fp32":
    # _quantize_fp4_codes indexes `levels` with .long(); that is the likely assert site
    x = torch.randn(64, 256, device=DEV) * 3.0
    pq, ps = O.fp4_act_quant(x, block_size=32)
    torch.cuda.synchronize()
    print("fp4_act_quant fp32 OK", pq.dtype, ps.dtype)

elif which == "fp4_gemm":
    M, N, K = 64, 512, 512
    x = (torch.randn(M, K, device=DEV) * 2.0).to(torch.bfloat16)
    q, s = O.act_quant(x, block_size=32, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
    w4 = torch.randn(N, K, device=DEV) * 0.02
    codes = torch.clamp(torch.round(w4 / 0.02), -6, 6).abs().to(torch.uint8)
    raw = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    b = raw.view(torch.int8).view(torch.float4_e2m1fn_x2)
    from relic_core.kernels.ops import Packed4BitWeightAlongK
    pk = Packed4BitWeightAlongK(plain_shape=(N, K), layout_tensor=b)
    out = O.fp4_gemm(q, s, b, ref_scale(N, K), torch.float32)
    torch.cuda.synchronize()
    print("fp4_gemm OK", out.dtype, tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))

elif which == "triton_gemm":
    M, N, K = 128, 512, 512
    xd = torch.randn(M, K, device=DEV) * 2.0
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.float8_e4m3fn)
    s = torch.ones(N, K // 128, device=DEV)
    out = O.soft_fp8_blockfp8_gemm(xd, w, s, impl="triton")
    torch.cuda.synchronize()
    print("triton gemm OK", out.dtype, tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))

print("DONE", which)