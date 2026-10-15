"""Where the sm_89 (RTX 4090) decode step goes, with the MMA emulation priced against a plain GEMM.

Read-only; touches no checkpoint and no model. It prices the emulation's own reduction -- the cost the
plan's Cell 4 asks about -- at a shape taken from the real layer, so that a full step's wall clock can
be *attributed* rather than guessed. The `per_row` cases exercise the fp4 expert scale *layout*; the
probe does not time fp4 weight expansion.

Run with the relic-core checkout and the RelicLLM tree on `PYTHONPATH`:

    PYTHONPATH=<relic-core>:<RelicLLM> python probe.py

`-I` is deliberately *not* used on the 4090 boxes: the editable install there resolves `import
relicllm` to a different scratch tree, so the `PYTHONPATH` is what selects the tree under test, and
isolated mode would drop it. The sm_75 probes use `-I` because they run against an extracted
checkpoint tree with no install in play.

What it measures, and what each number is:

* `fp8_block_gemm` at the layer's own `wq_a` shape, decode-shaped (M=1) and prefill-shaped (M=512).
  This is the emulation's whole cost: `quantized_linear` with the switch on routes every quantized
  linear here. Reported as device wall-clock time; the cost has always been op count rather than
  FLOPs, which is what the ratio to a plain GEMM on the same operands exposes.
* The same operands through a plain bf16 `F.linear` -- the correctly-rounded GEMM the emulation
  stands in for. The weight and activation are dequantized to bf16 first, so the comparator is the
  dense bf16 tensor-core GEMM of that shape and not an fp32 matmul, which on Ada runs far slower and
  would understate the ratio.
* `block_fp8_act_quant`, the activation quantizer the emulation gates on.

Nothing here is a token-level claim. A GEMM's speed says what the step is made of; it does not say
the tokens are right, and this probe does not touch that question.
"""

from __future__ import annotations

import time

import torch
import torch.nn.functional as F

torch.manual_seed(0)
DEV = torch.device("cuda:0")

from relicllm.models.deepseek_v4_1 import mma  # noqa: E402
from relicllm.models.deepseek_v4_1.kernels import block_fp8_act_quant  # noqa: E402

# The released geometry of the one shape the emulation is priced on: `wq_a` is F8_E4M3 `[1280, 5120]`
# with an F8_E8M0 `[40, 160]` scale (docs/architecture/deepseek_v4_1_flash_design.md:130). `dim` is
# 5120, not the 7168 of some other family -- the grid `[40, 160]` is what fixes K, and this is the
# call `mma.py`'s cost note names as the one the first version issued 107,424 ops for.
N, K = 1280, 5120  # q_lora_rank x dim
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


def _dequant_bf16(codes: torch.Tensor, scale: torch.Tensor, block: int, per_row: bool) -> torch.Tensor:
    """Codes times their scale, widened to bf16 -- a stand-in for the runtime's dequantized weight.

    The two scale grids differ in their row axis: a per-row grid carries one exponent per row, a
    32x32 grid one per ``block`` rows. Both carry one exponent per ``block`` columns, so both
    broadcast over the column blocks with ``repeat_interleave``; only the row axis branches.
    """
    values = codes.view(torch.float8_e4m3fn).to(torch.float32)
    exponent = scale.to(torch.float32)
    if not per_row:
        exponent = exponent.repeat_interleave(block, dim=0)
    exponent = exponent.repeat_interleave(block, dim=-1)
    return (values * exponent).to(torch.bfloat16)


def gemm_case(m: int, per_row: bool) -> None:
    a_codes = _codes(m, K)
    a_scale = _grid(m, K, per_row=True)
    b_codes = _codes(N, K)
    b_scale = _grid(N, K, per_row=per_row)
    label = "per-row grid (fp4 expert layout)" if per_row else "32x32 grid (dense fp8 layout)"

    emulated = _t(mma.fp8_block_gemm, a_codes, a_scale, b_codes, b_scale, block_size=BLOCK)
    # the thing the emulation stands in for: the dense bf16 GEMM on the same (dequantized) operands
    a_bf16 = _dequant_bf16(a_codes, a_scale, BLOCK, per_row=True)
    b_bf16 = _dequant_bf16(b_codes, b_scale, BLOCK, per_row=per_row)
    plain = _t(lambda: F.linear(a_bf16, b_bf16))

    print(f"M={m:<5} {label}")
    print(f"    fp8_block_gemm  {emulated * 1e3:9.2f} ms")
    print(f"    F.linear bf16   {plain * 1e3:9.2f} ms")
    print(f"    ratio           {emulated / plain:9.2f}x")


def act_quant_case(m: int) -> None:
    # The runtime hands this a bf16 activation; feeding fp32 would price a cast the path never makes.
    x = (torch.randn(m, K, device=DEV) * 2.0).to(torch.bfloat16)
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
