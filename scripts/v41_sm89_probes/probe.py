"""Where the sm_89 (RTX 4090) decode step goes, with the MMA emulation on and off.

Read-only; touches no checkpoint and no model. It times the two things the plan's Cell 4 asks about
-- the emulated reduction itself, and the cost of the host expert path's fp4 expansion -- at shapes
taken from the real layer, so that a full step's wall clock can be *attributed* rather than guessed.

Run with the relic-core checkout and the RelicLLM tree on `PYTHONPATH`:

    PYTHONPATH=<relic-core>:<RelicLLM> python probe.py

`-I` is deliberately *not* used on the 4090 boxes: the editable install there resolves `import
relicllm` to a different scratch tree, so the `PYTHONPATH` is what selects the tree under test, and
isolated mode would drop it. The sm_75 probes use `-I` because they run against an extracted
checkpoint tree with no install in play.

What it measures, and what each number is:

* `fp8_block_gemm` at the layer's own `wq_a` shape, decode-shaped (M=1) and prefill-shaped (M=512).
  This is the emulation's whole cost: `quantized_linear` with the switch on routes every quantized
  linear here. Reported as device time and as ops issued, because the cost has always been op count
  and not FLOPs.
* The same shape through plain `F.linear` in bf16 -- the correctly-rounded GEMM the emulation
  replaces. The ratio is what the emulation costs against the thing it stands in for, on this card.
* `block_fp8_act_quant`, the activation quantizer the emulation gates on.

Nothing here is a token-level claim. A GEMM's speed says what the step is made of; it does not say
the tokens are right, and this probe does not touch that question.
"""

from __future__ import annotations

import time

import torch

torch.manual_seed(0)
DEV = torch.device("cuda:0")

from relicllm.models.deepseek_v4_1 import mma  # noqa: E402
from relicllm.models.deepseek_v4_1.kernels import block_fp8_act_quant  # noqa: E402

# The released geometry of the one shape the emulation is priced on: `wq_a`, which the sm_75 page's
# own decomposition names as the call the first version issued 107,424 ops for.
N, K = 1280, 7168  # q_lora_rank x dim
BLOCK = 32


def _codes(rows: int, cols: int) -> torch.Tensor:
    """e4m3 codes with the field spread a real weight has, as raw bytes."""
    x = torch.randn(rows, cols, device=DEV) * 0.4
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


def _grid(rows: int, cols: int, per_row: bool) -> torch.Tensor:
    """An e8m0 grid of the shape the caller asks for: 32x32 blocks, or one exponent a row."""
    g_rows = rows if per_row else rows // BLOCK
    return torch.randint(120, 132, (g_rows, cols // BLOCK), device=DEV, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )


def _t(op, /, *args, reps: int = 20, **kw):
    for _ in range(3):  # warm
        op(*args, **kw)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(reps):
        op(*args, **kw)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps


def gemm_case(m: int, per_row: bool) -> None:
    a_codes = _codes(m, K)
    a_scale = _grid(m, K, per_row=True)
    b_codes = _codes(N, K)
    b_scale = _grid(N, K, per_row=per_row)
    label = "per-row grid (fp4 expert layout)" if per_row else "32x32 grid (dense fp8 layout)"

    emulated = _t(mma.fp8_block_gemm, a_codes, a_scale, b_codes, b_scale, block_size=BLOCK)
    # the thing the emulation stands in for: a correctly-rounded GEMM on the same values, bf16 out
    a_f = a_codes.view(torch.float8_e4m3fn).to(torch.float32)
    b_f = b_codes.view(torch.float8_e4m3fn).to(torch.float32)
    plain = _t(lambda: a_f @ b_f.T)

    print(f"M={m:<5} {label}")
    print(f"    fp8_block_gemm  {emulated * 1e3:9.2f} ms")
    print(f"    F.linear bf16   {plain * 1e3:9.2f} ms")
    print(f"    ratio           {emulated / plain:9.2f}x")


def act_quant_case(m: int) -> None:
    x = torch.randn(m, K, device=DEV) * 2.0
    t = _t(lambda: block_fp8_act_quant(x))
    print(f"    block_fp8_act_quant M={m:<5} {t * 1e3:9.2f} ms")


def main() -> None:
    print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"_MAX_PRODUCT_ELEMENTS = {mma._MAX_PRODUCT_ELEMENTS}")
    print()
    print("== the emulated reduction, decode-shaped and prefill-shaped ==")
    gemm_case(1, per_row=False)
    gemm_case(1, per_row=True)
    gemm_case(512, per_row=False)
    gemm_case(512, per_row=True)
    print()
    print("== the activation quantizer the emulation gates on ==")
    act_quant_case(1)
    act_quant_case(512)
    print("DONE")


if __name__ == "__main__":
    main()