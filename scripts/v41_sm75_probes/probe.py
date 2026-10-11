"""sm_75 feasibility probe: soft block-32 act_quant + fp8 gemm path.

Read-only. No model weights touched. Run with: PYTHONPATH=<relic-core> python -I probe.py
"""
import torch

torch.manual_seed(0)
DEV = torch.device("cuda:0")


def hdr(t):
    print("\n" + "=" * 70)
    print(t)
    print("=" * 70)


# ---------------------------------------------------------------- capability
hdr("1. capability fork resolution")
from relic_core.kernels import ops as O

print("device:", torch.cuda.get_device_name(0), "cap:", torch.cuda.get_device_capability(0))
print("_USE_TRITON      =", O._USE_TRITON)
print("_FP8_IMPL (env)  =", O._FP8_IMPL)
print("_FP4_IMPL (env)  =", O._FP4_IMPL)
print("_FP8_MAX         =", O._FP8_MAX)
for kind in ("fp8_quant", "fp8", "fp4"):
    print(f"_auto_impl({kind!r})            = {O._auto_impl(kind)!r}")
    print(f"_resolve_impl({kind!r}, 'auto') = {O._resolve_impl(kind, 'auto')!r}")
print("_choose_compute_dtype(cuda) =", O._choose_compute_dtype(DEV))
print("_to_output_dtype(cuda)      =", O._to_output_dtype(DEV))

# ------------------------------------------------------- 3. act_quant on card
hdr("3. act_quant on a real CUDA tensor (sm_75)")
M, K = 64, 256
x = (torch.randn(M, K, device=DEV) * 3.0).to(torch.bfloat16)
print("x:", x.dtype, tuple(x.shape), "finite:", bool(torch.isfinite(x).all()))

q, s = O.act_quant(x, block_size=32, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
torch.cuda.synchronize()
print("act_quant(block_size=32, scale_fmt='ue8m0', scale_dtype=e8m0) ->")
print("  q:", q.dtype, tuple(q.shape), "finite:", bool(torch.isfinite(q.to(torch.float32)).all()))
print("  s:", s.dtype, tuple(s.shape), "finite:", bool(torch.isfinite(s.to(torch.float32)).all()))
print("  s sample:", s.to(torch.float32).flatten()[:6].tolist())

q2, s2 = O.blockfp8_act_quant(x, block_size=32, impl="torch")
torch.cuda.synchronize()
print("blockfp8_act_quant(impl='torch')  ->", q2.dtype, tuple(q2.shape), s2.dtype, tuple(s2.shape))

try:
    q3, s3 = O.blockfp8_act_quant(x, block_size=32, impl="triton")
    torch.cuda.synchronize()
    print("blockfp8_act_quant(impl='triton') ->", q3.dtype, tuple(q3.shape), s3.dtype, tuple(s3.shape))
    print("  triton == torch q:", bool(torch.equal(q3, q2)), " s allclose:", bool(torch.allclose(s3, s2)))
except Exception as e:
    print("blockfp8_act_quant(impl='triton') FAILED:", type(e).__name__, str(e).strip().splitlines()[-1][:200])

# ------------------- reference act_quant, reimplemented in torch exactly
def ref_round_scale(amax):
    return torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))


def ref_act_quant(x, bs=32):
    n = x.size(-1)
    z = x.contiguous()
    flat = z.view(-1, n).to(torch.float32)
    amax = flat.abs().view(-1, n // bs, bs).amax(dim=-1).clamp_min(1e-4)
    s = ref_round_scale(amax)
    s_e8m0 = s.to(torch.float8_e8m0fnu)
    s_back = s_e8m0.to(torch.float32)
    sf = s_back.repeat_interleave(bs, dim=-1)
    y = torch.clamp(flat / sf, -448.0, 448.0).to(torch.float8_e4m3fn)
    return y.view_as(z), s_e8m0.view(*z.shape[:-1], n // bs)


def wscales(n, k, bs):
    amax = torch.rand(n, k // bs, device=DEV) * 0.04 + 0.001
    return ref_round_scale(amax).to(torch.float8_e8m0fnu)


def w_deq_f64(w_fp8, w_s, bs):
    n, k = w_fp8.shape
    wsf = w_s.to(torch.float32)
    wsf = torch.repeat_interleave(wsf, bs, dim=0)[:n]
    wsf = torch.repeat_interleave(wsf, bs, dim=1)[:, :k]
    return (w_fp8.to(torch.float32) * wsf).to(torch.float64)


def a_deq_f32(q, s, bs, k):
    sf = s.to(torch.float32).repeat_interleave(bs, dim=-1)
    return q.to(torch.float32) * sf


def truth_fp64(x_bf16, w_fp8, w_s, bs):
    q, s = ref_act_quant(x_bf16, bs)
    a = a_deq_f32(q, s, bs, w_fp8.shape[1]).to(torch.float64)
    b = w_deq_f64(w_fp8, w_s, bs)
    return torch.matmul(a, b.transpose(0, 1))


def report(name, got, tru):
    d = (got.to(torch.float64) - tru).abs()
    amax = tru.abs().max()
    exp = torch.floor(torch.log2(amax.clamp_min(1e-30)))
    ulp = torch.pow(2.0, exp - 7)
    r = d.max() / amax
    print(f"  {name}")
    print(f"    max_abs     = {d.max().item():.6e}")
    print(f"    amax(truth) = {amax.item():.6e}")
    print(f"    rel-to-amax = {r.item():.6e}")
    print(f"    bf16 ulp @amax = {(d.max()/ulp).item():.4f}")


# ---------------------------------------------- 2a. soft recipe vs fp64 truth
hdr("2a. soft block-32 recipe (fp32 accumulate -> bf16) vs fp64 (Turing)")
M, N, K = 256, 512, 512
x = (torch.randn(M, K, device=DEV) * 2.0).to(torch.bfloat16)
w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.float8_e4m3fn)
ws = wscales(N, K, 32)

q, s = ref_act_quant(x, 32)
a_ref = a_deq_f32(q, s, 32, K)
b_ref = (w.to(torch.float32) * w_deq_f64(w, ws, 32).to(torch.float32)) if False else None
# (b) bf16 path: dequantize weight to bf16 then matmul in fp32 (the port's recipe)
n, k = w.shape
wsf = torch.repeat_interleave(ws.to(torch.float32), 32, dim=0)[:n]
wsf = torch.repeat_interleave(wsf, 32, dim=1)[:, :k]
w_bf16 = (w.to(torch.float32) * wsf).to(torch.bfloat16)
soft = torch.matmul(a_ref, w_bf16.to(torch.float32).transpose(0, 1)).to(torch.bfloat16)
report("(a) bf16-dequant weight, fp32 accumulate, vs fp64:", soft, truth_fp64(x, w, ws, 32))

# also: weight dequant kept in fp32 (no bf16 rounding of the weight)
sq, ss = ref_act_quant(x, 32)
soft_f32w = torch.matmul(a_deq_f32(sq, ss, 32, K),
                         (w.to(torch.float32) * wsf).transpose(0, 1)).to(torch.bfloat16)
report("(b) fp32-dequant weight, fp32 accumulate, vs fp64:", soft_f32w, truth_fp64(x, w, ws, 32))

# ------------------------------------ 2b. relic-core public fp8_gemm (block 128)
hdr("2b. relic-core public fp8_gemm (its block-128 convention) on sm_75")
w128 = (torch.randn(N, K, device=DEV) * 0.02).to(torch.float8_e4m3fn)
ws128 = wscales(N, K, 128)
q128, s128 = ref_act_quant(x, 128)
out = O.fp8_gemm(q128, s128.to(torch.float32), w128, ws128, torch.float32)
torch.cuda.synchronize()
print("  fp8_gemm ->", out.dtype, tuple(out.shape), "finite:", bool(torch.isfinite(out).all()))
report("  fp8_gemm vs fp64 (block128):", out, truth_fp64(x, w128, ws128, 128))

# ----------------------------------- 2c. block-32 assembled soft gemm on Turing
hdr("2c. assembled block-32 fp8 soft gemm on sm_75")
q32, s32 = O.act_quant(x, block_size=32, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
a_dq32 = O._dequant_fp8_acts(q32, s32, 32)
w_dq32 = O.soft_fp8_blockfp8_weight_dequant(w, ws, block_size=32, impl="auto")
out32 = O._soft_gemm(a_dq32, w_dq32, "auto")
torch.cuda.synchronize()
print("  block-32 e8m0 act scales ->", out32.dtype, tuple(out32.shape),
      "finite:", bool(torch.isfinite(out32).all()))
report("  vs fp64:", out32, truth_fp64(x, w, ws, 32))

q32b, s32b = O.act_quant(x, block_size=32, scale_fmt=None, scale_dtype=torch.float32)
out32b = O._soft_gemm(O._dequant_fp8_acts(q32b, s32b, 32), w_dq32, "auto")
torch.cuda.synchronize()
print("  block-32 fp32 act scales ->", out32b.dtype, "finite:", bool(torch.isfinite(out32b).all()))
report("  vs fp64:", out32b, truth_fp64(x, w, ws, 32))

# ----------------------------------------------- 2d. fp4 weight path, block 32
hdr("2d. fp4 weight path (block 32) via the public fp4_gemm")
from relic_core.kernels.ops import Packed4BitWeightAlongK

w4 = torch.randn(N, K, device=DEV) * 0.02
codes = torch.clamp(torch.round(w4 / 0.02), -6, 6).abs().to(torch.uint8)
raw = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
pk = Packed4BitWeightAlongK(plain_shape=(N, K),
                            layout_tensor=raw.view(torch.int8).view(torch.float4_e2m1fn_x2))
w4s = wscales(N, K, 32)
try:
    out4 = O.fp4_gemm(q32, s32, pk.layout_tensor, w4s, torch.float32)
    torch.cuda.synchronize()
    print("  fp4_gemm ->", out4.dtype, tuple(out4.shape),
          "finite:", bool(torch.isfinite(out4).all()))
except Exception as e:
    print("  fp4_gemm FAILED:", type(e).__name__, str(e)[:300])

hdr("2e. fp4_act_quant (block 32, E8M0) on sm_75")
try:
    pq, ps = O.fp4_act_quant(x, block_size=32)
    torch.cuda.synchronize()
    print("  fp4_act_quant ->", pq.dtype, tuple(pq.shape), ps.dtype, tuple(ps.shape),
          "finite:", bool(torch.isfinite(pq.to(torch.float32)).all()))
except Exception as e:
    print("  fp4_act_quant FAILED:", type(e).__name__, str(e)[:300])

hdr("2f. triton soft gemm, forced, on sm_75")
try:
    outt = O.soft_fp8_blockfp8_gemm(a_dq32, w, ws128, impl="triton")
    torch.cuda.synchronize()
    print("  triton soft gemm ->", outt.dtype, "finite:", bool(torch.isfinite(outt).all()))
except Exception as e:
    print("  triton soft gemm FAILED:", type(e).__name__, str(e)[:400])

print("\nDONE")