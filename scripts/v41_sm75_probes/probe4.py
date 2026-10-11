"""Pin down where fp4_act_quant / fp4->f32 cast asserts on sm_75.

Read-only. Run with: PYTHONPATH=<relic-core> python -I probe4.py <case>
Cases: quant_only, cast_fp4, cast_fp4_cpu, fp4_roundtrip, fp8_cast.
"""
import sys
import torch

torch.manual_seed(0)
DEV = torch.device("cuda:0")
from relic_core.kernels import ops as O

which = sys.argv[1]

if which == "quant_only":
    x = (torch.randn(64, 256, device=DEV) * 3.0).to(torch.bfloat16)
    pq, ps = O.fp4_act_quant(x, block_size=32)
    torch.cuda.synchronize()
    # do NOT cast pq; only inspect metadata and the (fp8) scales
    print("fp4_act_quant returned: pq", pq.dtype, tuple(pq.shape),
          "ps", ps.dtype, tuple(ps.shape))
    print("pq contiguous:", pq.is_contiguous(), "numel:", pq.numel())
    print("ps finite (e8m0 cast is fp8->f32):",
          bool(torch.isfinite(ps.to(torch.float32)).all()))
    print("ps sample:", ps.to(torch.float32).flatten()[:8].tolist())
    print("pq bytes:", pq.view(torch.uint8).flatten()[:8].tolist())

elif which == "cast_fp4":
    # a bare fp4 tensor -> f32 cast, no quantizer involved
    raw = torch.arange(0, 128, dtype=torch.uint8, device=DEV)
    b = raw.view(torch.int8).view(torch.float4_e2m1fn_x2)
    try:
        f = b.to(torch.float32)
        torch.cuda.synchronize()
        print("fp4->f32 cast OK:", f.flatten()[:8].tolist())
    except Exception as e:
        print("fp4->f32 cast FAILED:", type(e).__name__, str(e).splitlines()[0][:200])

elif which == "cast_fp4_cpu":
    raw = torch.arange(0, 128, dtype=torch.uint8)
    b = raw.view(torch.int8).view(torch.float4_e2m1fn_x2)
    f = b.to(torch.float32)
    print("fp4->f32 cast on CPU OK:", f.flatten()[:8].tolist())

elif which == "fp4_roundtrip":
    # reconstruct the fp4 codes the way the quantizer stores them, then decode by hand
    x = (torch.randn(64, 256, device=DEV) * 3.0).to(torch.bfloat16)
    pq, ps = O.fp4_act_quant(x, block_size=32)
    raw = pq.view(torch.uint8)
    low = (raw & 0x0F).to(torch.long)
    high = ((raw >> 4) & 0x0F).to(torch.long)
    levels = O._fp4_levels(DEV)
    codes = torch.empty(raw.shape[0], raw.shape[1] * 2, dtype=torch.long, device=DEV)
    codes[:, 0::2] = low
    codes[:, 1::2] = high
    dec = levels[codes]
    sf = ps.to(torch.float32).repeat_interleave(32, dim=-1)
    dq = dec * sf
    print("hand-decoded fp4: finite:", bool(torch.isfinite(dq).all()),
          "max|x| =", dq.abs().max().item())
    print("vs original amax:", x.abs().max().item())

elif which == "fp8_cast":
    x = (torch.randn(64, 256, device=DEV) * 3.0).to(torch.bfloat16)
    q, s = O.act_quant(x, block_size=32, scale_fmt="ue8m0", scale_dtype=torch.float8_e8m0fnu)
    f = q.to(torch.float32)
    torch.cuda.synchronize()
    print("fp8->f32 cast OK, finite:", bool(torch.isfinite(f).all()))

print("DONE", which)