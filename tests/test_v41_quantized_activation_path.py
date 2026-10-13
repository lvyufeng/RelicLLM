"""The quantized-activation path: the reference's `linear`, on the soft (no tensor core) path.

The port's divergence from the released V4.1 reference was two things, and two changes answer them.
First the activation: the reference quantizes it to fp8 before a block-32 GEMM, and the port fed an
unquantized bf16 activation to a plain `F.linear` -- that half is this file. Then the reduction: the
reference's GEMM is a tensor-core kernel whose `mma` does not add its products exactly, so the
arithmetic pinned here is the right activation summed correctly, and the emulation of the actual
reduction lives in `tests/test_v41_mma_reduction.py`. The weight half is not a difference: an fp8
(e4m3) or fp4 (e2m1) code carries at most three mantissa bits and a block scale is a power of two,
so `code * 2**k` is exact in bf16 and the loader's already-dequantized weights equal the reference's
quantized ones bit for bit. See `docs/superpowers/notes/quantized-activation-path.md`.

This file pins the primitive that restores the missing half -- `block_fp8_act_quant` and
`quantized_linear` in `relicllm/models/deepseek_v4_1/kernels.py` -- against the reference's stated
arithmetic rather than against a kernel, because the reference's kernel needs TileLang and a tensor
core and runs on neither host this suite runs on. Three properties:

* the block-32 quantized GEMM reproduces `dequant(act) @ W.T` and differs from a plain bf16
  `F.linear` by the fp8-activation error -- the whole of what the activation fix buys, and also the
  arithmetic a weight with no attached scale grid keeps, since the emulated reduction declines such
  a weight rather than invent a grid for it;
* the no-scales path is `F.linear` **exactly** -- not an approximation, the same call -- which is
  what lets `test_models_deepseek_v4_1_reference_parity.py`'s bf16 build keep the arithmetic it has;
* the activation quantization uses the reference's `1e-4` amax floor and power-of-two rounding.

Every test here is CPU-safe: the primitive is soft torch and does not need a card. The whole file
skips when torch is absent, the same way every sibling in `tests/` is guarded.
"""

from __future__ import annotations

import pytest

# The suite runs on boxes that may not carry torch. `importorskip` is the same guard the V4.1 sibling
# files use, and it has to come before the module's own import: a torch-less host must skip this file
# rather than fail to collect it.
torch = pytest.importorskip("torch")

from relicllm.models.deepseek_v4_1.kernels import (  # noqa: E402
    block_fp8_act_quant,
    quantized_linear,
)

# The reference's `kernel.py` globals: one fp8 scale per 32 activations, E4M3 as the code format,
# 448 as its largest finite. `scale_fmt="ue8m0"` is what turns the amax floor and the scale into the
# power-of-two form asserted below, and it is not optional -- the E8M0 scale the GEMM reads is that
# power of two.
FP8_BLOCK_SIZE = 32
FP8_MAX = 448.0
AMAX_FLOOR = 1e-4


def _longhand_block_fp8_act_quant(x: torch.Tensor, block_size: int = FP8_BLOCK_SIZE):
    """The reference's `act_quant` written out, independent of the primitive under test.

    Transcribed from `inference/kernel.py::act_quant_kernel`: `amax` over the block, floored at
    `1e-4`, the scale rounded up to a power of two, the clamped ratio cast to E4M3. The two lines
    that are not obviously the same operation as the primitive -- the floor and the rounding -- are
    the ones the tests below assert on a known input.
    """
    k = x.size(-1)
    flat = x.contiguous().view(-1, k).to(torch.float32)
    amax = flat.abs().view(-1, k // block_size, block_size).amax(dim=-1).clamp_min(AMAX_FLOOR)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / FP8_MAX)))
    scale_full = scale.repeat_interleave(block_size, dim=-1)
    quant = torch.clamp(flat / scale_full, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return (
        quant.view_as(x),
        scale.to(torch.float32).view(*x.shape[:-1], k // block_size),
    )


def _dequantize_activation(quant: torch.Tensor, scale: torch.Tensor, block_size: int = FP8_BLOCK_SIZE):
    """The reference's `_dequant_fp8_acts`: the code times its block's scale, in fp32."""
    return quant.to(torch.float32) * scale.repeat_interleave(block_size, dim=-1)


def test_quantized_linear_matches_dequant_act_times_weight() -> None:
    """The GEMM is `dequant(act) @ W.T` in fp32, and it is *not* a plain bf16 `F.linear`.

    The weight is a dense bf16 tensor, which is what the loader hands the module: the checkpoint's
    weight dequantization is exact in bf16, so the primitive needs no weight scale and the test
    needs no quantized weight fixture. The oracle is written out here rather than called, so a
    primitive that agreed with a helper it also defined could not pass this.

    The weight also carries no attached scale grid, and that is what fixes which path runs: the
    tensor-core emulation is on by default and recovers the weight's exact codes from the grid the
    file stores beside it (`quantized_linear`'s docstring carries the 224-octave argument for why
    the dense weight does not determine them), so a grid-less weight takes the *declined* path --
    one stderr line, then the correctly rounded accumulation asserted here. This test pins that
    fallback; the emulated rule has its own file, `tests/test_v41_mma_reduction.py`.
    """
    torch.manual_seed(0)
    m, n, k = 16, 48, 64
    x = torch.randn(m, k, dtype=torch.bfloat16)
    weight = torch.randn(n, k, dtype=torch.bfloat16) * (k**-0.5)

    got = quantized_linear(x, weight, quantized=True, block_size=FP8_BLOCK_SIZE)

    quant, scale = _longhand_block_fp8_act_quant(x)
    want = _dequantize_activation(quant, scale)
    want = torch.matmul(want, weight.to(torch.float32).transpose(0, 1)).to(x.dtype)

    assert torch.allclose(got, want, atol=1e-5, rtol=1e-4), (
        f"max abs diff {(got.float() - want.float()).abs().max().item()}"
    )

    # The point of the fix: this is not the arithmetic the port used to run. An fp8 activation is
    # three mantissa bits against bf16's eight, so on a unit-scale activation the two differ well
    # above the fp32 noise this comparison is carried at.
    plain = torch.nn.functional.linear(x, weight)
    gap = (got.float() - plain.float()).abs().max().item()
    assert gap > 1e-3, f"quantized and plain bf16 differ by only {gap}, too little to be fp8 quant"


def test_no_scales_path_is_exactly_f_linear() -> None:
    """`quantized=False` is `F.linear`, bit for bit, and this is the seam the bf16 build stays on.

    `test_models_deepseek_v4_1_reference_parity.py` builds `dtype="bf16", expert_dtype=None`, which
    turns every quantization boolean off; if this path were merely close to `F.linear` rather than
    identical, that comparison would move for a reason that has nothing to do with the reference.
    `torch.equal` and not `allclose`: the contract is the same call, not a good approximation of it.
    """
    torch.manual_seed(1)
    x = torch.randn(5, 32, dtype=torch.bfloat16)
    weight = torch.randn(7, 32, dtype=torch.bfloat16)

    assert torch.equal(quantized_linear(x, weight), torch.nn.functional.linear(x, weight))
    assert torch.equal(
        quantized_linear(x, weight, quantized=False),
        torch.nn.functional.linear(x, weight),
    )


def test_activation_scale_is_the_reference_power_of_two() -> None:
    """The scale is `2 ** ceil(log2(amax / 448))` for a known amax."""
    # One block of 32 whose largest magnitude is exactly 1.0 (the values are 1/32 .. 32/32).
    # amax / 448 is 2.232e-3, log2 of it is -8.807, so the scale rounds up to 2**-8.
    x = torch.arange(1, FP8_BLOCK_SIZE + 1, dtype=torch.float32) / FP8_BLOCK_SIZE
    x = x.unsqueeze(0)
    _, scale = block_fp8_act_quant(x, FP8_BLOCK_SIZE)
    assert scale.dtype == torch.float32
    assert scale.shape == (1, 1)
    assert scale.item() == 2.0**-8, scale.item()


def test_all_zero_block_gets_the_amax_floor_not_a_zero_scale() -> None:
    """The `1e-4` floor: an all-zero block still gets a nonzero scale, so the divide-back is legal.

    This is the reference's `T.max(amax_local[i], 1e-4)`, and it is not decoration -- a zero scale
    cannot be divided out on the way back, and the compressed-KV cache is read back every step. The
    floor puts `amax / 448` at `1e-4 / 448 = 2.232e-7`, whose `log2` is -22.09, so the scale is
    `2**-22`.
    """
    x = torch.zeros(2, FP8_BLOCK_SIZE)
    _, scale = block_fp8_act_quant(x, FP8_BLOCK_SIZE)
    assert (scale > 0).all(), scale
    assert torch.allclose(scale, torch.full_like(scale, 2.0**-22)), scale
    # And a sub-floor amax is treated as the floor: the same scale, not a smaller one.
    tiny = torch.full((1, FP8_BLOCK_SIZE), 1e-6)
    _, tiny_scale = block_fp8_act_quant(tiny, FP8_BLOCK_SIZE)
    assert tiny_scale.item() == 2.0**-22, tiny_scale.item()


def test_bf16_narrowing_of_a_block_dequant_is_lossless() -> None:
    """The premise the fix rests on: `dequant` is exact, so bf16 dense weights need no re-load.

    An fp8 e4m3 code carries three mantissa bits and an E8M0 block scale is a power of two, so
    `code * 2**k` is a bf16 value exactly. If this ever failed, the fix would need to keep the
    weights packed and re-quantize the activation against a weight scale, and the whole design
    would change -- so it is asserted rather than assumed.
    """
    from relicllm.models.deepseek_v4_1.kernels import dequant_fp8_weight

    torch.manual_seed(2)
    out_features, in_features = 64, 64
    codes = (torch.randint(0, 256, (out_features, in_features), dtype=torch.int32) & 0x7F).to(
        torch.uint8
    )
    # E4M3's only all-ones-mantissa pattern with the top exponent is NaN, and the mask above can
    # produce it (0x7F). Step it down to the largest finite code so the comparison is over numbers.
    codes = torch.where(codes == 0x7F, torch.full_like(codes, 0x7E), codes)
    weight = codes.view(torch.float8_e4m3fn)
    # E8M0 is a biased exponent byte with no sign and no mantissa: `2**(byte - 127)`.
    scale_bytes = torch.randint(100, 160, (out_features // 32, in_features // 32), dtype=torch.int32)
    scale = scale_bytes.to(torch.uint8).view(torch.float8_e8m0fnu)

    dense = dequant_fp8_weight(weight, scale, 32)
    narrowed = dense.to(torch.bfloat16).to(torch.float32)
    assert torch.equal(narrowed, dense), "the fp8 block dequant is not exact in bf16"