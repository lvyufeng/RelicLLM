"""sm_75: refined per-element bf16-ulp accuracy of the soft block-32 recipe.

Read-only. Run with: PYTHONPATH=<relic-core> python -I probe2.py
Produces the per-element bf16-ulp table the note quotes: the recipe at two shapes, and the
fp16-weight-dequant penalty. This is the source of the 0.5 ulp and the ~600 ulp fp16 figures.
"""
import torch

torch.manual_seed(0)
DEV = torch.device("cuda:0")


def ref_round_scale(amax):
    return torch.pow(2.0, torch.ceil(torch.log2(amax / 448.0)))


def ref_act_quant(x, bs=32):
    n = x.size(-1)
    z = x.contiguous()
    flat = z.view(-1, n).to(torch.float32)
    amax = flat.abs().view(-1, n // bs, bs).amax(dim=-1).clamp_min(1e-4)
    s = ref_round_scale(amax)
    s_e8m0 = s.to(torch.float8_e8m0fnu)
    sf = s_e8m0.to(torch.float32).repeat_interleave(bs, dim=-1)
    y = torch.clamp(flat / sf, -448.0, 448.0).to(torch.float8_e4m3fn)
    return y.view_as(z), s_e8m0.view(*z.shape[:-1], n // bs)


def w_scale_f32(n, k, bs):
    amax = torch.rand(n, k // bs, device=DEV) * 0.04 + 0.001
    return ref_round_scale(amax).to(torch.float8_e8m0fnu).to(torch.float32)


def ulp_bf16(v):
    e = torch.floor(torch.log2(v.abs().clamp_min(1e-38)))
    return torch.pow(2.0, e - 7)


def report(name, got_bf16, tru64):
    d = (got_bf16.to(torch.float64) - tru64).abs()
    amax = tru64.abs().max()
    per_elem = (d / ulp_bf16(tru64).clamp_min(1e-45)).flatten()
    # only where the true value is large enough for a bf16 ulp to be meaningful
    mask = tru64.abs().flatten() > amax * 1e-3
    print(f"  {name}")
    print(f"    max_abs          = {d.max().item():.6e}")
    print(f"    amax(truth)      = {amax.item():.6e}")
    print(f"    rel-to-amax      = {(d.max()/amax).item():.6e}")
    print(f"    max |err| in bf16 ulp  (all)        = {per_elem.max().item():.4f}")
    print(f"    max |err| in bf16 ulp  (|t|>1e-3 amax) = {per_elem[mask].max().item():.4f}")
    print(f"    mean|err| in bf16 ulp  (|t|>1e-3 amax) = {per_elem[mask].mean().item():.4f}")


def expand(wsf, k, bs):
    n = wsf.size(0)
    return torch.repeat_interleave(wsf, bs, dim=1)[:, :k]


def soft(x, w, wsf, bs, weight_dtype):
    wsf = expand(wsf, w.size(1), bs)
    q, s = ref_act_quant(x, bs)
    a = q.to(torch.float32) * s.to(torch.float32).repeat_interleave(bs, dim=-1)
    wd = (w.to(torch.float32) * wsf).to(weight_dtype)
    out = torch.matmul(a, wd.to(torch.float32).transpose(0, 1))
    return out.to(torch.bfloat16)


def truth(x, w, wsf, bs):
    wsf = expand(wsf, w.size(1), bs)
    q, s = ref_act_quant(x, bs)
    a = (q.to(torch.float32) * s.to(torch.float32).repeat_interleave(bs, dim=-1)).to(torch.float64)
    b = (w.to(torch.float32) * wsf).to(torch.float64)
    return torch.matmul(a, b.transpose(0, 1))


for (M, N, K) in [(128, 512, 512), (64, 2048, 2048)]:
    print("=" * 70)
    print(f"shape M={M} N={N} K={K}")
    print("=" * 70)
    x = (torch.randn(M, K, device=DEV) * 2.0).to(torch.bfloat16)
    w = (torch.randn(N, K, device=DEV) * 0.02).to(torch.float8_e4m3fn)
    wsf = w_scale_f32(N, K, 32)
    t = truth(x, w, wsf, 32)
    report("(a) weight dequantized to bf16", soft(x, w, wsf, 32, torch.bfloat16), t)
    report("(b) weight dequantized to fp16", soft(x, w, wsf, 32, torch.float16), t)
    report("(c) weight kept fp32", soft(x, w, wsf, 32, torch.float32), t)

print("DONE")