"""The DeepSeek-V4.1 op set, gathered behind one module.

The released V4.1 runtime (`inference/kernel.py`) writes its ops in TileLang and imports
`tilelang` at module scope; the stack needs `tilelang==0.1.8` plus `torch>=2.10.0`. Neither is
installed on this host and neither will be, and the sm_75 cards here have no FP4 tensor core for
the reference's own fallbacks. The reference therefore cannot execute here and there is **no
numeric oracle** for a V4.1 forward pass on this machine.

The response is not a second implementation. `relic_core/kernels/ops.py` already implements all six ops
the reference declares, and the paths this forward pass actually calls were checked against the
reference's arithmetic rather than assumed to match:

| reference op | `relic_core/kernels/ops.py` |
| --- | --- |
| `act_quant` | `act_quant` — `eps=1e-4` amax floor and `round_scale_to_pow2` are the reference's `T.max(amax, 1e-4)` and `fast_round_scale` |
| `fp4_act_quant` | `fp4_act_quant` — E8M0 scales, the branch the indexer takes |
| `fp8_gemm` | `fp8_gemm` — re-exported, but the linear layers here do not call it: it hardcodes V3's block of 128 and drops V4.1's `block_size=32`, so they go through `quantized_linear` instead |
| `fp4_gemm` | `fp4_gemm` — the same block of 128, and on block-32 scales it returns wrong numbers rather than failing; `docs/superpowers/notes/relic-core-gemm-block-size-hazards.md` measures both |
| `sparse_attn` | `sparse_attn` — held to a longhand softmax in `tests/test_models_deepseek_v4_1_kernels.py`, at `atol=rtol=2e-2` over bf16-stored inputs, plus the empty-row floor. That is the shape of the computation, not bit parity |
| `hc_split_sinkhorn` | `hc_split_sinkhorn` |

so this module re-exports them and adds the two things that are genuinely missing.

**One op is not a re-export: `fp4_act_quant_e4m3`.** The reference's `fp4_quant_kernel` branches on
its *scale dtype*, and `relic_core/kernels/ops.py` implements only the E8M0 branch. The E4M3 branch is what
the compressed-KV path calls (`fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)` in
`Attention._compress_kv`) and it is not a reformat of the E8M0 one — see the function below.

**The other addition is the inverse direction: `dequant_fp8_weight` and `dequant_fp4_weight`.**
The reference never needs them — it consumes the quantized weights in a quantized GEMM — but a
loader that fills a module tree written in ordinary PyTorch does, and the released checkpoint's
block scales are the only thing that says what its weights are. They are also what makes the
weights checkable without a tensor core: `dequant(weight, scale)` against a fixture is a statement
about the checkpoint, where a GEMM result on this host would be a statement about our own kernels.

Why a facade rather than importing `relic_core.kernels.ops` directly at each call site: the two
implementations agree today, and the tests in `tests/test_models_deepseek_v4_1_kernels.py` are what
makes that a checked property instead of an assumption. Two of those checks pin behaviour a
plausible rewrite would silently lose — the empty-top-k row contract and the Sinkhorn
normalization order — so they belong to V4.1, where the contract is stated, rather than being
implicit in V4-Flash's runtime.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from relic_core.kernels.ops import (
    Packed4BitWeightAlongK,
    act_quant,
    fp4_act_quant,
    fp4_gemm,
    fp8_gemm,
    hc_split_sinkhorn,
    soft_fp4_blockfp4_weight_dequant,
    soft_fp8_blockfp8_weight_dequant,
    sparse_attn,
)

__all__ = [
    "Packed4BitWeightAlongK",
    "act_quant",
    "block_fp8_act_quant",
    "dequant_fp4_weight",
    "dequant_fp8_weight",
    "fp4_act_quant",
    "fp4_act_quant_e4m3",
    "fp4_gemm",
    "fp8_gemm",
    "hc_split_sinkhorn",
    "quantized_linear",
    "sparse_attn",
]


# The largest magnitude the E2M1 codebook holds. Both FP4 scale conventions divide by it: the
# scale is whatever makes a block's amax land on this value, so `x / scale` fills [-6, 6].
_FP4_MAX = 6.0

# The eight E2M1 magnitudes, in code order. Code `i` is `+magnitudes[i]` for i < 8 and
# `-magnitudes[i - 8]` above, so the sign bit is the code's high bit and zero has two encodings.
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

# E4M3's smallest subnormal, 2**-9. The reference floors the *amax* at `6 * 2**-9`, which puts the
# derived scale at exactly `2**-9` -- the smallest value E4M3 can still represent as nonzero.
_E4M3_MIN_SUBNORMAL = 2.0**-9

# Both codebooks are host tuples turned into a device tensor per call, so the same eight or sixteen
# floats are re-copied over PCIe on every quantize. That is a pageable H2D -- a launch-time cost on
# the hot path and a hard failure inside a CUDA graph capture -- so the copy is kept per device.
_FP4_LEVEL_CACHE: dict[torch.device, torch.Tensor] = {}
_FP4_SIGNED_LEVEL_CACHE: dict[torch.device, torch.Tensor] = {}


def _levels_for(device: torch.device, cache: dict[torch.device, torch.Tensor], values) -> torch.Tensor:
    device = torch.device(device)
    levels = cache.get(device)
    if levels is None:
        levels = torch.tensor(values, dtype=torch.float32, device=device)
        cache[device] = levels
    return levels


def fp4_act_quant_e4m3(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """FP4 activation quantization with E4M3 scales -- the compressed-KV variant.

    `fp4_act_quant` rounds its scales to powers of two and stores them as E8M0, which is what the
    indexer path wants. The reference's `fp4_quant_kernel` branches on the scale dtype, and the
    branch that stores **E4M3** differs in two ways that are not cosmetic:

    * The scale is `e4m3(amax / 6)` -- a rounding of the *ratio*, not of the amax -- where E8M0
      stores `2**round(log2(amax / 6))`. So the two branches agree on the packed nibbles for most
      inputs and disagree on the scale for all of them.
    * The amax floor is `6 * 2**-9` rather than `6 * 2**-126`, which is high enough that an all-zero
      block still gets a **nonzero** scale. That is the point of the branch: a zero scale cannot be
      divided out on the way back, and the compressed-KV cache is read back on every step.

    The division uses the *rounded* E4M3 scale, matching the reference, which stores the cast value
    and then divides by it rather than by the exact ratio.

    `block_size` is 16 where `Attention._compress_kv` calls this and 32 for the indexer; both are
    accepted, and `inplace=True` quantizes and dequantizes back into `x`, as the reference's
    `inplace` does.
    """
    n = x.size(-1)
    if n == 0 or block_size <= 0 or n % block_size:
        # `n == 0` is called out separately because it is the one ragged width that passes both
        # modulus checks below and then fails later, inside the reshape, with a message that says
        # nothing about the caller's mistake.
        raise ValueError(f"last dim {n} is not a positive multiple of block_size {block_size}")
    if n % 2:
        raise ValueError(f"last dim {n} must be even to pack two FP4 codes per byte")

    flat = x.contiguous().view(-1, n).to(torch.float32)
    amax = flat.abs().view(-1, n // block_size, block_size).amax(dim=-1).clamp_min(_FP4_MAX * _E4M3_MIN_SUBNORMAL)
    scale = (amax / _FP4_MAX).to(torch.float8_e4m3fn).to(torch.float32)
    scale_full = scale.repeat_interleave(block_size, dim=-1)
    normalized = torch.clamp(flat / scale_full, -_FP4_MAX, _FP4_MAX)
    codes = _fp4_codes(normalized)

    if inplace:
        x.copy_((_fp4_values(codes) * scale_full).to(x.dtype).view_as(x))
        return x

    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8).view(*x.shape[:-1], n // 2)
    return packed, scale.to(torch.float8_e4m3fn).view(*x.shape[:-1], n // block_size)


def _fp4_codes(normalized: torch.Tensor) -> torch.Tensor:
    """Round values already clamped to [-6, 6] onto the E2M1 codebook, returning 4-bit codes.

    Ties resolve to the even code, which is what the reference's `T.Cast(FP4, ...)` gets from the
    hardware conversion; the reference never states a tie rule, so it is taken from the cast it
    relies on rather than chosen.
    """
    magnitude = normalized.abs()
    levels = _levels_for(normalized.device, _FP4_LEVEL_CACHE, _FP4_MAGNITUDES)
    upper = torch.searchsorted(levels, magnitude, right=False).clamp(1, len(_FP4_MAGNITUDES) - 1)
    lower = upper - 1
    # Round half to even: a tie goes to whichever of the two codes is even.
    to_upper = (magnitude - levels[lower]) > (levels[upper] - magnitude)
    to_upper |= ((magnitude - levels[lower]) == (levels[upper] - magnitude)) & (upper % 2 == 0)
    index = torch.where(to_upper, upper, lower)
    return (index + torch.where(normalized < 0, 8, 0)).to(torch.uint8)


def _fp4_values(codes: torch.Tensor) -> torch.Tensor:
    """The inverse of `_fp4_codes`, as float32."""
    levels = _levels_for(
        codes.device, _FP4_SIGNED_LEVEL_CACHE, [*_FP4_MAGNITUDES, *(-m for m in _FP4_MAGNITUDES)]
    )
    return levels[codes.long()]


def dequant_fp8_weight(weight: torch.Tensor, scale: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    """Expand an fp8 weight with E8M0 block scales back to float32.

    The released checkpoint stores every dense projection this way -- one scale per 32x32 block, so
    `scale` is `[ceil(out / 32), ceil(in / 32)]` and not a per-row or per-tensor scalar. `weight`
    keeps its `[out, in]` shape; only `scale` is a grid.
    """
    return soft_fp8_blockfp8_weight_dequant(weight, scale, block_size, impl="auto")


def dequant_fp4_weight(
    weight: torch.Tensor, scale: torch.Tensor, block_size: int = 32
) -> torch.Tensor:
    """Expand an fp4 weight, packed two-per-byte along K, back to float32.

    `weight` is `[out, in // 2]` of `I8` as the checkpoint stores it, low nibble first, and `scale`
    is `[out, in // block_size]` of E8M0 -- one scale per 32 elements *along K*, with no block
    structure on the output axis.
    """
    return soft_fp4_blockfp4_weight_dequant(
        Packed4BitWeightAlongK.convert_from(weight), scale, block_size, impl="auto"
    )


# The reference's `kernel.py` globals: one fp8 scale per 32 activations, E4M3 as the code format,
# 448 as its largest finite, and 1e-4 as the amax floor. `scale_fmt="ue8m0"` is not optional -- the
# E8M0 scale the GEMM reads is the power of two these three constants produce.
_FP8_BLOCK_SIZE = 32
_FP8_MAX = 448.0
_FP8_AMAX_FLOOR = 1e-4


def block_fp8_act_quant(
    x: torch.Tensor, block_size: int = _FP8_BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference's `act_quant(x, 32, "ue8m0", float8_e8m0fnu)` without a kernel.

    `relic_core.kernels.ops.act_quant` reproduces the same block-32 arithmetic, but it is a *kernel*
    binding whose output dtype and implementation are chosen by the host. The fix needs this to be
    pure torch: on a card with no fp8 tensor path (sm_75) the kernel binding still resolves to a
    soft path and would agree, but on a card that has one it may not, and the point of the primitive
    is to be the same arithmetic on every card. So the three steps are written out here:

    1. ``amax = x.abs().view(-1, K // block_size, block_size).amax(-1)``, then ``clamp_min(1e-4)``.
       The floor is the reference's ``T.max(amax_local[i], 1e-4)`` -- an all-zero block must still
       get a nonzero scale, because that scale is divided out on the way back.
    2. ``scale = 2 ** ceil(log2(amax / 448))``. The reference's ``fast_round_scale(amax, 1/448)`` is
       exactly this bit trick, so the result is the smallest power of two at or above ``amax / 448``,
       which is what makes the clamped ratio land inside the fp8 range without a second clamp.
    3. ``quant = clamp(x / scale, -448, 448).to(torch.float8_e4m3fn)``.

    Returns ``(quant, scale)`` where ``quant`` is ``float8_e4m3fn`` shaped like ``x`` and ``scale``
    is fp32 ``[..., K // block_size]``.
    """
    k = x.size(-1)
    if k == 0 or block_size <= 0 or k % block_size:
        raise ValueError(f"last dim {k} is not a positive multiple of block_size {block_size}")

    flat = x.contiguous().view(-1, k).to(torch.float32)
    amax = flat.abs().view(-1, k // block_size, block_size).amax(dim=-1)
    amax = amax.clamp_min(_FP8_AMAX_FLOOR)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax / _FP8_MAX)))
    scale_full = scale.repeat_interleave(block_size, dim=-1)
    quant = torch.clamp(flat / scale_full, -_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn)
    return quant.view_as(x), scale.to(torch.float32).view(*x.shape[:-1], k // block_size)


def quantized_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    quantized: bool = False,
    block_size: int = _FP8_BLOCK_SIZE,
) -> torch.Tensor:
    """The reference's `linear()` on the soft path, with the weight already dequantized.

    ``quantized=False`` is the no-scales path and it is exactly ``F.linear(x, weight)`` -- not an
    approximation of it, the same call. That is the seam the bf16 build stays on, and the seam
    ``test_models_deepseek_v4_1_reference_parity.py`` exercises with ``dtype="bf16",
    expert_dtype=None``.

    ``quantized=True`` is the reference's with-scales path. The ``weight`` handed in is the *dense*
    tensor the loader already produced, so there is no weight scale here and none is needed: an fp8
    e4m3 or fp4 e2m1 code carries at most three mantissa bits and a block scale is a power of two,
    so the loader's ``dequant`` is exact and ``weight`` equals the reference's ``dequant(weight,
    scale)`` bit for bit. What is missing from the port and restored here is the *activation*: it
    goes through :func:`block_fp8_act_quant`, and the product accumulates in fp32 and is narrowed
    once, to ``x.dtype``.

    The weight is upcast to fp32 explicitly even though it is already bf16, so the accumulator's
    width is a property of this function and not of what the caller happened to hand in; the upcast
    is free of information (bf16 is exact in fp32) and it fixes the reduction's width, which is what
    the reference's GEMM does.
    """
    if not quantized:
        return F.linear(x, weight)
    act, act_scale = block_fp8_act_quant(x, block_size)
    dequant_act = act.to(torch.float32) * act_scale.repeat_interleave(block_size, dim=-1)
    return torch.matmul(dequant_act, weight.to(torch.float32).transpose(0, 1)).to(x.dtype)
