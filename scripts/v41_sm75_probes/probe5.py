"""Final: soft recipe vs relic-core's real soft GEMM kernels on sm_75.

Read-only. Run with: PYTHONPATH=<relic-core> python -I probe5.py
Produces the ulp figures quoted in the note (recipe 0.000 ulp, torch impl bitwise equal,
triton impl 592 ulp, fp4_gemm 49464 ulp).
"""
import torch

torch.manual_seed(0)
DEV = torch.device("cuda:0")
from relic_core.kernels import ops as O


def ref_round_scale(amax):
    return torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))


def ref_act_quant(x, bs=32):
    n = x.size(-1)
    z = x.contiguous()
    flat = z.view(-1, n).to(torch.float32)
    amax = flat.abs().view(-1, n // bs, bs).amax(dim=-1).clamp_min(1e-4)
    s_e8m0 = ref_round_scale(amax).to(torch.float8_e8m0fnu)
    sf = s_e8m0.to(torch.float32).repeat_interleave(bs, dim=-1)
    y = torch.clamp(flat / sf, -448.0, 448.0).to(torch.float8_e4m3fn)
    return y.view_as(z), s_e8m0.view(*z.shape[:-1], n // bs)


def ubf16(v):
    return torch.pow(2.0, torch.floor(torch.log2(v.abs().clamp_min(1e-38))) - 7)


def cmp(name, a_bf16, b):
    d = (a_bf16.to(torch.float64) - b.to(torch.float64)).abs()
    amax = b.to(torch.float64).abs().max()
    m = b.abs().flatten() > amax * 1e-3
    u = (d.flatten() / ubf16(b.to(torch.float64)).flatten().clamp_min(1e-45))
    print(f"  {name}: max_rel_to_amax={(d.max()/amax).item():.3e}  "
          f"max_ulp(|t|>1e-3amax)={u[m].max().item():.4f}  mean={u[m].mean().item():.4f}")


M, N, K = 128, 512, 512
x = (torch.randn(M, K, device=DEV) * 2.0).to(torch.bfloat16)
w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.float8_e4m3fn)
ws = ref_round_scale(torch.rand(N, K // 32, device=DEV) * 0.04 + 0.001).to(torch.float8_e8m0fnu)

q, s = ref_act_quant(x, 32)
a_ref = q.to(torch.float32) * s.to(torch.float32).repeat_interleave(32, dim=-1)
w_dq = O.soft_fp8_blockfp8_weight_dequant(w, ws, block_size=32, impl="auto")

# fp64 truth of the reference arithmetic
a64 = a_ref.to(torch.float64)
b64 = w_dq.to(torch.float64)
truth = torch.matmul(a64, b64.transpose(0, 1))

# the recipe: fp32 accumulate, cast bf16
recipe = torch.matmul(a_ref, w_dq.transpose(0, 1)).to(torch.bfloat16)
print("soft recipe vs fp64 truth of the same operands:")
cmp("recipe", recipe, truth.to(torch.bfloat16))

# relic-core's real _soft_gemm (torch impl) on identical operands
sg_torch = O._soft_gemm(a_ref, w_dq, "torch")
sg_triton = O._soft_gemm(a_ref, w_dq, "triton")
torch.cuda.synchronize()
print("relic-core _soft_gemm impls on identical operands:")
cmp("torch  vs recipe", recipe, sg_torch.to(torch.bfloat16))
cmp("triton vs recipe", recipe, sg_triton.to(torch.bfloat16))
print("  torch vs triton bitwise equal:",
      bool((sg_torch.to(torch.bfloat16) == sg_triton.to(torch.bfloat16)).all()))

# fp4_gemm against its fp64 truth
w4 = torch.randn(N, K, device=DEV) * 0.02
codes = torch.clamp(torch.round(w4 / 0.02), -6, 6).abs().to(torch.uint8)
raw = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
b = raw.view(torch.int8).view(torch.float4_e2m1fn_x2)
from relic_core.kernels.ops import Packed4BitWeightAlongK
pk = Packed4BitWeightAlongK(plain_shape=(N, K), layout_tensor=b)
out4 = O.fp4_gemm(q, s, b, ws, torch.float32)
torch.cuda.synchronize()
w4_dq = O.soft_fp4_blockfp4_weight_dequant(pk, ws, block_size=32, impl="auto")
t4 = torch.matmul(a64, w4_dq.to(torch.float64).transpose(0, 1))
print("fp4_gemm (fp8 act x fp4 weight) vs fp64 truth:")
cmp("fp4_gemm", out4.to(torch.bfloat16), t4.to(torch.bfloat16))
print("DONE")