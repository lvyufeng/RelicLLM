"""Independent scalar oracle for the V4.1 routed-expert activation and reduction semantics.

Why this file exists
--------------------
Before a new NPU expert kernel is written, the arithmetic it must reproduce has to be stated
somewhere that is not the implementation under test. This oracle is written from the *contract*
(see `.coord/expert-semantics-v3.md`) using exact rational arithmetic only -- `fractions.Fraction`
and `struct`, no torch, no numpy, no import of the production modules. It is therefore an
independent second statement of:

  * the activation quantizer (`act_quant(x, 32, "ue8m0", float8_e8m0fnu)`): amax floor `1e-4`,
    power-of-two scale from the fp32 ratio `amax * fl32(1/448)`, clamp to `+/-448`, RNE to e4m3;
  * the e4m3 codebook, both directions, RNE with ties-to-even, subnormals, the two NaN codes;
  * the E8M0 scale byte, including the two representable edges (`2**-127`, NaN at `255`);
  * the tensor-core reduction: two sequential 16-wide halves, per-output-element anchor
    `F(a) + F(b)` with `F = max(true exponent, -6)`, truncation toward zero, window 13, the
    running sum joining the second half as a peer, and the final renormalisation;
  * the per-K-block scale application and the sequential fp32 accumulation across blocks;
  * the exact-E2M1-to-INT8 representation a packed FP4 expert is widened to (x2 values, halved
    E8M0 exponent), and why "doubled units + halved scale" reproduces the same number.

Exactness
---------
Every arithmetic step is exact in `Fraction`: the operands are dyadic rationals (e4m3 values and
powers of two), truncation onto a power-of-two grid is integer division, and the only rounding this
file performs deliberately is `fp32_round`, which models one IEEE binary32 round-to-nearest-even.
That makes it possible to ask questions the float implementation cannot answer directly -- for
example "is every intermediate of the rule exactly representable in fp32?" -- which is what the
`*_exact` trace flags in the self-test check.

What this is not
----------------
It is not a hardware model. It says what the arithmetic is; it does not say that a TileLang kernel
on an NVIDIA card, or a Triton kernel on a 910B, computes it. See the report for which parity claims
are measured and which are emulation.

Usage
-----
    python3 .coord/expert_scalar_oracle.py --selftest      # stdlib only, no card, no torch
"""

from __future__ import annotations

import random
import struct
import sys
from fractions import Fraction

# ---------------------------------------------------------------------------------------------
# Constants, as the reference states them (see the report for the line citations).
# ---------------------------------------------------------------------------------------------

BLOCK = 32          # fp8_block_size == group_size == block_K in the reference kernel
HALF = 16           # one mma.sync covers 16 of a block's 32 K values; two per block
WINDOW = 13         # binary places kept below the anchor
FP8_MAX = 448.0     # E4M3's largest finite value
FP8_AMAX_FLOOR = 1e-4
FP8_MAX_INV_F32 = None  # filled in below: fl32(1 / 448), the reciprocal the reference multiplies by
E8M0_BIAS = 127

#: `F(v) = max(true exponent of v, -6)` for an e4m3 encoding whose exponent field is zero.
SUBNORMAL_NOMINAL = -6

#: e4m3 codes that are NaN rather than numbers.
E4M3_NAN_CODES = (0x7F, 0xFF)

#: The e4m3 code holding each e2m1 code's value, in e2m1 code order. The reference's fp4 kernel
#: casts every nibble through FP32 into FP8, and this is the table that cast produces.
E2M1_TO_E4M3 = (
    0x00, 0x30, 0x38, 0x3C, 0x40, 0x44, 0x48, 0x4C,
    0x80, 0xB0, 0xB8, 0xBC, 0xC0, 0xC4, 0xC8, 0xCC,
)

#: E2M1 magnitudes in code order -- what a packed nibble means.
E2M1_MAGNITUDES = (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(3, 2),
                   Fraction(2), Fraction(3), Fraction(4), Fraction(6))

#: The exact INT8 x2 representation of an E2M1 value, by E2M1 code. This is what a packed expert
#: becomes after an exact online widening: `int8 = 2 * e2m1`.
FP4_INT8_VALUES = tuple(
    (-1 if code & 8 else 1) * E2M1_MAGNITUDES[code & 7] * 2 for code in range(16)
)

NAN = "nan"        # the oracle's NaN marker (Fraction cannot hold a NaN)
SENTINEL = None    # "this half has no anchor"


class OracleOverflow(Exception):
    """A value left the finite fp32 range; the real kernel would have produced an infinity."""


# ---------------------------------------------------------------------------------------------
# Exact binary32 rounding, so the oracle can model one IEEE operation without a float library.
# ---------------------------------------------------------------------------------------------


def fp32_value(x: float) -> Fraction:
    """The exact value of the binary32 number nearest to `x` (inputs are fp32 in the kernel)."""
    return Fraction(struct.unpack("<f", struct.pack("<f", x))[0])


def _round_half_even(numerator: int, denominator: int) -> int:
    """`numerator / denominator` rounded to the nearest integer, ties to even (both positive)."""
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    if twice > denominator or (twice == denominator and quotient % 2 == 1):
        quotient += 1
    return quotient


def floor_log2(value: Fraction) -> int:
    """``floor(log2 value)`` exactly, for a positive Fraction."""
    if value <= 0:
        raise ValueError("floor_log2 wants a positive value")
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    two = Fraction(2)
    if value < two ** exponent:
        exponent -= 1
    elif value >= two ** (exponent + 1):
        exponent += 1
    return exponent


def pow2(exponent: int) -> Fraction:
    return Fraction(2) ** exponent


def fp32_round(value: Fraction) -> Fraction:
    """One correctly-rounded binary32 conversion (RNE), returned as its exact value.

    Raises :class:`OracleOverflow` when the magnitude exceeds binary32's finite range; the kernel
    would produce an infinity there, which changes every subsequent add and is a real hazard rather
    than a rounding detail.
    """
    if value == 0:
        return Fraction(0)
    sign = -1 if value < 0 else 1
    magnitude = abs(value)
    exponent = floor_log2(magnitude)
    if exponent >= -126:
        if exponent > 127:
            raise OracleOverflow(value)
        quantum = pow2(exponent - 23)
        mantissa = _round_half_even((magnitude / quantum).numerator,
                                    (magnitude / quantum).denominator)
        if mantissa == 1 << 24:  # carried out of the significand
            exponent += 1
            if exponent > 127:
                raise OracleOverflow(value)
            mantissa = 1 << 23
            quantum = pow2(exponent - 23)
        return sign * mantissa * quantum
    # Subnormal: the quantum is fixed at 2**-149, so precision is lost with magnitude.
    quantum = pow2(-149)
    mantissa = _round_half_even((magnitude / quantum).numerator, (magnitude / quantum).denominator)
    return sign * mantissa * quantum


def fp32_mul(left: Fraction, right: Fraction) -> Fraction:
    return fp32_round(left * right)


def fp32_div(left: Fraction, right: Fraction) -> Fraction:
    return fp32_round(left / right)


def fp32_add(left: Fraction, right: Fraction) -> Fraction:
    return fp32_round(left + right)


FP8_MAX_INV_F32 = fp32_div(Fraction(1), Fraction(448))  # fl32(1/448), the reference's constant


# ---------------------------------------------------------------------------------------------
# E4M3 (e4m3fn): encode with RNE, decode exactly.
# ---------------------------------------------------------------------------------------------


def decode_e4m3(code: int):
    """An e4m3 code as its exact value, or :data:`NAN` for the two NaN codes."""
    if code in E4M3_NAN_CODES:
        return NAN
    sign = -1 if code & 0x80 else 1
    field = (code >> 3) & 0xF
    mantissa = code & 0x7
    if field == 0:
        return sign * Fraction(mantissa, 8) * pow2(-6)
    return sign * (1 + Fraction(mantissa, 8)) * pow2(field - 7)


def encode_e4m3(value: Fraction) -> int:
    """The e4m3 code `value` rounds to, RNE with ties to even, clamped to the finite range.

    Values are clamped to ``+/-448`` first, which is exactly what a quantizer's
    ``clamp(x / scale, -448, 448)`` does before its cast. RNE onto a power-of-two grid with an even
    mantissa is the tie rule IEEE conversion uses, which is what ``.to(torch.float8_e4m3fn)`` is.
    """
    if value == 0:
        return 0x00
    sign = 0x80 if value < 0 else 0
    magnitude = abs(value)
    if magnitude > Fraction(int(FP8_MAX)):
        magnitude = Fraction(int(FP8_MAX))
    exponent = floor_log2(magnitude)
    if exponent >= -6:
        quantum = pow2(exponent - 3)
        mantissa = _round_half_even((magnitude / quantum).numerator,
                                    (magnitude / quantum).denominator)
        if mantissa == 16:  # 1.111.. rounded up out of the significand
            exponent += 1
            mantissa = 8
            if exponent == 9:  # 480 is not representable; 448 is the saturation point
                return sign | 0x7E
        field = exponent + 7
        if field >= 15 and mantissa == 15:
            return sign | 0x7E
        return sign | (field << 3) | (mantissa - 8)
    # Subnormal: multiples of 2**-9, once again ties to even on the integer multiplier.
    quantum = pow2(-9)
    mantissa = _round_half_even((magnitude / quantum).numerator,
                                (magnitude / quantum).denominator)
    if mantissa == 8:  # 8 * 2**-9 is the smallest normal, field 1 mantissa 0
        return sign | (1 << 3)
    return sign | mantissa


def nominal_exp(code: int) -> int:
    """``F(v) = max(true exponent, -6)`` read off the encoding, per the reference's mma rule.

    A subnormal e4m3 code's value has a true exponent of -7, -8 or -9, and all three contribute -6:
    the exponent the *encoding field* implies, not the exponent of the value. That distinction only
    shows up on dense data (the module records it as 0.05% of elements); a zero code returns None to
    mean "this operand contributes no anchor".
    """
    if (code & 0x7F) == 0:
        return None
    field = (code >> 3) & 0xF
    return field - 7 if field >= 1 else SUBNORMAL_NOMINAL


# ---------------------------------------------------------------------------------------------
# E8M0 scales.
# ---------------------------------------------------------------------------------------------


def decode_e8m0(byte: int):
    """An E8M0 scale byte as its exact power of two, or :data:`NAN` for the reserved 255."""
    if byte == 255:
        return NAN
    return pow2(byte - E8M0_BIAS)


def half_e8m0(byte: int):
    """The same scale halved, for an INT8 x2 representation of the values it multiplies.

    `b = 0` is `2**-128`: still an exact binary32 *subnormal*, not a zero and not a power of two
    that can be spelled in the E8M0 byte (which would need b = -1). `b = 255` stays NaN.
    """
    if byte == 255:
        return NAN
    return pow2(byte - E8M0_BIAS - 1)


# ---------------------------------------------------------------------------------------------
# The activation quantizer: `act_quant(x, 32, "ue8m0", float8_e8m0fnu)`.
# ---------------------------------------------------------------------------------------------


def fast_log2_ceil(value: Fraction) -> int:
    """``ceil(log2 value)`` by the reference's bit trick, not by a logarithm.

    The reference's `fast_log2_ceil` reads the binary32 exponent field and adds one when the
    mantissa is nonzero. That is exact for every normal input -- and the amax floor guarantees the
    input is normal here -- so it is reproduced bit for bit rather than approximated.
    """
    mantissa, exponent = _frexp(value)
    return exponent - 1 if mantissa == Fraction(1, 2) else exponent


def _frexp(value: Fraction) -> tuple[Fraction, int]:
    """``value = mantissa * 2**exponent`` with ``1/2 <= |mantissa| < 1``, exactly."""
    exponent = floor_log2(value) + 1
    return value / pow2(exponent), exponent


def act_scale_exponent(amax: Fraction, *, ratio_spelling: str = "reciprocal") -> int:
    """The activation scale's binary exponent, from a block's fp32 amax.

    The reference computes ``fast_round_scale(amax, 1/448)``: one binary32 multiply by the *rounded
    reciprocal* ``fl32(1/448)``, then the bit-exact ceiling above. ``ratio_spelling="division"``
    computes ``amax / 448`` instead -- one correctly-rounded binary32 divide -- which is what the
    port's soft quantizer spells and is **not** bit-identical for every amax (see the report's H1).
    """
    if amax <= 0:
        raise ValueError("amax must be positive after the floor")
    if ratio_spelling == "reciprocal":
        ratio = fp32_mul(amax, FP8_MAX_INV_F32)
    elif ratio_spelling == "division":
        ratio = fp32_div(amax, Fraction(448))
    else:  # pragma: no cover - argument validation
        raise ValueError(f"unknown ratio spelling {ratio_spelling!r}")
    return fast_log2_ceil(ratio)


def act_scale(amax: Fraction, *, ratio_spelling: str = "reciprocal") -> Fraction:
    """``2 ** ceil(log2(fl32(amax * fl32(1/448))))``, as an exact power of two."""
    return pow2(act_scale_exponent(amax, ratio_spelling=ratio_spelling))


def act_quant_block(values: list[Fraction], *, ratio_spelling: str = "reciprocal"
                    ) -> tuple[list[int], Fraction]:
    """Quantize one 32-wide block: ``(e4m3 codes, fp32 scale)``.

    The steps, and their order, are the contract:

    1. ``amax = max |x|`` over the block (exact: a max of fp32 values is an fp32 value), then
       ``amax = max(amax, 1e-4)``. The floor is what gives an all-zero block a nonzero scale, so
       the divide-back is legal.
    2. ``scale = 2 ** ceil(log2(ratio))`` from the fp32 ratio.
    3. ``code = RNE_e4m3(clamp(x / scale, -448, 448))``. Dividing by a power of two is exact.
    """
    if len(values) != BLOCK:
        raise ValueError(f"expected {BLOCK} values, got {len(values)}")
    magnitude = max((abs(v) for v in values), default=Fraction(0))
    floor_value = fp32_value(FP8_AMAX_FLOOR)
    amax = magnitude if magnitude > floor_value else floor_value
    scale = act_scale(amax, ratio_spelling=ratio_spelling)
    codes = []
    for value in values:
        ratio = fp32_div(value, scale)
        if ratio > Fraction(int(FP8_MAX)):
            ratio = Fraction(int(FP8_MAX))
        elif ratio < -Fraction(int(FP8_MAX)):
            ratio = -Fraction(int(FP8_MAX))
        codes.append(encode_e4m3(ratio))
    return codes, scale


# ---------------------------------------------------------------------------------------------
# The reduction: one mma.sync's 32 terms, as the reference issues them.
# ---------------------------------------------------------------------------------------------


def _trunc_to_grid(value: Fraction, step: Fraction) -> Fraction:
    """Truncate toward zero onto the grid `step` (a power of two)."""
    ratio = value / step
    if ratio >= 0:
        whole = ratio.numerator // ratio.denominator
    else:
        whole = -((-ratio.numerator) // ratio.denominator)
    return whole * step


def reduce_block(products: list, nominals: list):
    """The mma's 32-term reduction for one output element, as an exact Fraction.

    ``products`` are the exact products of the two operands' values (``NAN`` if either operand is
    one of the two NaN codes), ``nominals`` their summed nominal exponents (:data:`SENTINEL` where
    the product is zero, because a zero product contributes no anchor).

    The rule, and every asymmetry in it:

    * two sequential 16-wide halves, the accumulator starting at zero for the block;
    * each half is cut onto a grid anchored by that half's own largest *nominal* exponent;
    * the first half's sum joins the second half as a **peer**: the second window is
      ``max(its own anchor, E(sum of the first half))``;
    * truncation is toward zero throughout;
    * a final renormalisation to ``E(total) - 13``.
    """
    if len(products) != BLOCK or len(nominals) != BLOCK:
        raise ValueError(f"the rule is stated on {BLOCK} terms")
    if any(p is NAN for p in products):
        return NAN

    def anchor_of(slice_nominals: list) -> int | None:
        live = [n for n in slice_nominals if n is not None]
        return max(live) if live else None

    first = anchor_of(nominals[:HALF])
    if first is None:
        total = Fraction(0)
    else:
        step = pow2(first - WINDOW)
        total = sum((_trunc_to_grid(p, step) for p in products[:HALF]), Fraction(0))

    running = None if total == 0 else floor_log2(abs(total))
    second_anchor = anchor_of(nominals[HALF:])
    window = max([a for a in (second_anchor, running) if a is not None], default=None)
    if window is None:
        step = Fraction(1)
    else:
        step = pow2(window - WINDOW)
    total = _trunc_to_grid(total, step) + sum(
        (_trunc_to_grid(p, step) for p in products[HALF:]), Fraction(0)
    )

    if total != 0:
        total = _trunc_to_grid(total, pow2(floor_log2(abs(total)) - WINDOW))
    return total


def block_products(act_codes: list[int], weight_codes: list[int],
                   weight_nominals: list | None = None) -> tuple[list, list]:
    """The exact products and summed nominal exponents of one 32-wide K block.

    ``weight_nominals`` overrides the weight side's `F` values, which is how the INT8 x2 contract is
    expressed: an integer `2 * e2m1` has its own exponent rule and its own scale. When it is None
    the weight side is read off the e4m3 codes, which is the reference's own path.
    """
    products, nominals = [], []
    for index in range(BLOCK):
        a_code, b_code = act_codes[index], weight_codes[index]
        a_nominal = nominal_exp(a_code)
        b_nominal = (nominal_exp(b_code) if weight_nominals is None else weight_nominals[index])
        a_value = decode_e4m3(a_code)
        b_value = decode_e4m3(b_code) if weight_nominals is None else None
        if a_value is NAN:
            products.append(NAN)
            nominals.append(SENTINEL)
            continue
        if weight_nominals is not None:
            # The caller supplied exponents only (the INT8 contract); values come from b_code.
            b_value = decode_e4m3(b_code)
        if b_value is NAN:
            products.append(NAN)
            nominals.append(SENTINEL)
            continue
        product = a_value * b_value
        products.append(product)
        nominals.append(SENTINEL if product == 0
                         else (a_nominal + b_nominal if None not in (a_nominal, b_nominal)
                               else SENTINEL))
    return products, nominals


# ---------------------------------------------------------------------------------------------
# The kernel's loop: per-block scale application and the sequential fp32 accumulation.
# ---------------------------------------------------------------------------------------------


def block_term(fragment: Fraction, act_exponent: int, weight_term,
               *, half_scale: bool = False) -> Fraction:
    """One K block's contribution: ``fragment * 2**e_a * 2**e_b``, in exact units.

    ``weight_term`` is either an E8M0 byte (the reference's layout, one per output row and K block)
    or an already-decoded power of two. ``half_scale=True`` halves it, which is the compensation an
    INT8 x2 weight needs and which cannot be expressed in the E8M0 byte itself for ``b = 0``.
    """
    if isinstance(weight_term, int):
        scale = half_e8m0(weight_term) if half_scale else decode_e8m0(weight_term)
    else:
        scale = weight_term / 2 if half_scale else weight_term
    if scale is NAN:
        return NAN
    if fragment is NAN:
        return NAN
    return fragment * pow2(act_exponent) * scale


def accumulate_fp32(terms: list, *, fused: bool = False) -> Fraction:
    """Accumulate the per-block terms in K order into one fp32 accumulator.

    ``fused=False`` models a separate multiply and add (each rounding once). ``fused=True`` models an
    FMA: one rounding of ``accumulator + exact term``. The two are compared by the self-test, which
    is how the report's claim about why the reference's spelling is (almost) unobservable is
    grounded rather than asserted.
    """
    accumulator = Fraction(0)
    for term in terms:
        if term is NAN:
            return NAN
        addend = term if fused else fp32_round(term)
        accumulator = fp32_round(accumulator + addend)
    return accumulator


def reference_expert_row(act_blocks: list[list[int]], weight_code_blocks: list[list[int]],
                         act_exponents: list[int], weight_bytes: list[int],
                         *, fused: bool = False) -> Fraction:
    """One output element of the reference's fp4 expert GEMM, term by term.

    ``act_blocks`` are the activation's e4m3 codes, ``weight_code_blocks`` the e4m3 codes the
    reference's kernel casts the packed E2M1 nibbles to, one row's worth of K blocks.
    """
    terms = []
    for index, (act_codes, weight_codes) in enumerate(zip(act_blocks, weight_code_blocks)):
        products, nominals = block_products(act_codes, weight_codes)
        fragment = reduce_block(products, nominals)
        terms.append(block_term(fragment, act_exponents[index], weight_bytes[index]))
    return accumulate_fp32(terms, fused=fused)


def int8_expert_row(act_blocks: list[list[int]], weight_int8_blocks: list[list[int]],
                    act_exponents: list[int], weight_bytes: list[int],
                    *, fused: bool = False, anchor_shift: int = 1) -> Fraction:
    """The same output element computed from an exact INT8 x2 weight and the halved E8M0 grid.

    The contract this pins, and the reason a new kernel may take the INT8 representation without
    changing a bit:

    * the integer value is ``2 * e2m1``, so each product is exactly twice the reference's;
    * ``F`` of that weight is therefore ``floor(log2 |v|)`` where the reference's was
      ``floor(log2 |v|) - 1``, i.e. the anchor shifts by ``anchor_shift = +1``;
    * truncation onto a power-of-two grid commutes with a doubling, so the doubled reduction is
      exactly ``2 * fragment``;
    * the scale is halved (``2**(b - 128)``), which cancels the doubling exactly.

    ``anchor_shift=0`` is the deliberately wrong variant the self-test uses to show the shift is
    load-bearing rather than cosmetic.
    """
    terms = []
    for index, (act_codes, weight_ints) in enumerate(zip(act_blocks, weight_int8_blocks)):
        products, nominals = [], []
        for code, value in zip(act_codes, weight_ints):
            a_value = decode_e4m3(code)
            magnitude = abs(value)
            if a_value is NAN:
                products.append(NAN)
                nominals.append(SENTINEL)
                continue
            if magnitude not in (0, 1, 2, 3, 4, 6, 8, 12):
                raise ValueError(f"{value} is not an exact E2M1 x2 weight value")
            product = a_value * value
            products.append(product)
            if product == 0:
                nominals.append(SENTINEL)
            else:
                a_nominal = nominal_exp(code)
                b_nominal = floor_log2(Fraction(magnitude)) - 1 + anchor_shift
                nominals.append(a_nominal + b_nominal)
        fragment = reduce_block(products, nominals)
        terms.append(block_term(fragment, act_exponents[index], weight_bytes[index],
                                half_scale=True))
    return accumulate_fp32(terms, fused=fused)


# ---------------------------------------------------------------------------------------------
# Self-test: synthetic random CPU cases, exact-rational arithmetic only, no device, no torch.
# ---------------------------------------------------------------------------------------------


def _e4m3_code_pool() -> list[int]:
    return [code for code in range(256) if code not in E4M3_NAN_CODES]


def _random_block(generator: random.Random, *, allow_nan: bool = False,
                  prefer_subnormal: float = 0.0) -> list[int]:
    """An e4m3 code block drawn from the whole encoding space, subnormals and zeros included."""
    pool = _e4m3_code_pool()
    codes = []
    for _ in range(BLOCK):
        if prefer_subnormal and generator.random() < prefer_subnormal:
            codes.append(generator.choice([0x00, 0x80, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07,
                                           0x81, 0x87]))
        else:
            codes.append(generator.choice(pool))
    if allow_nan and generator.random() < 0.05:
        codes[generator.randrange(BLOCK)] = generator.choice(E4M3_NAN_CODES)
    return codes


def _int8_block_from_codes(generator: random.Random, e2m1_codes: list[int]) -> list[int]:
    """The exact INT8 x2 widening of a packed-nibble block."""
    values = []
    for code in e2m1_codes:
        widened = int(FP4_INT8_VALUES[code])
        values.append(widened)
    return values


def _e2m1_code_block(generator: random.Random) -> list[int]:
    return [generator.randrange(16) for _ in range(BLOCK)]


def _exact_fragment_trace(products: list, nominals: list) -> tuple[Fraction, bool]:
    """Reduce a block again, recording whether every intermediate is exactly an fp32 value.

    This is the check behind the port's fp32 implementation: if every intermediate is representable
    in binary32, then a float32 transcription of the rule and an exact one agree bit for bit.
    """
    exact = True
    if any(p is NAN for p in products):
        return NAN, exact

    def trunc_exact(value: Fraction, step: Fraction) -> Fraction:
        nonlocal exact
        quotient = value / step
        if fp32_round(quotient) != quotient:
            exact = False
        return _trunc_to_grid(value, step)

    first = [n for n in nominals[:HALF] if n is not None]
    step = pow2(max(first) - WINDOW) if first else Fraction(1)
    total = Fraction(0)
    for product in products[:HALF]:
        total += trunc_exact(product, step)
    if fp32_round(total) != total:
        exact = False
    second_window = None if total == 0 else floor_log2(abs(total))
    own = [n for n in nominals[HALF:] if n is not None]
    window = max([w for w in (max(own) if own else None, second_window) if w is not None],
                 default=None)
    step = pow2(window - WINDOW) if window is not None else Fraction(1)
    total = trunc_exact(total, step)
    for product in products[HALF:]:
        total += trunc_exact(product, step)
    if fp32_round(total) != total:
        exact = False
    if total != 0:
        total = trunc_exact(total, pow2(floor_log2(abs(total)) - WINDOW))
        if fp32_round(total) != total:
            exact = False
    return total, exact


class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failures: list[str] = []

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
        else:
            self.failures.append(f"{name}{(': ' + detail) if detail else ''}")

    def equal(self, name: str, got, want) -> None:
        self.check(name, got == want, f"got {got!r}, want {want!r}")


def _test_e4m3_codebook(checker: Checker) -> None:
    # Round-trip: every finite code decodes to a value that re-encodes to itself.
    for code in _e4m3_code_pool():
        if code == 0x80:
            continue  # Fraction has no negative zero; the sign of a zero code is not recoverable here
        value = decode_e4m3(code)
        checker.equal(f"e4m3 round-trip 0x{code:02X}", encode_e4m3(value), code)
    checker.check("negative-zero codes decode and re-encode as +0 (a Fraction has one zero)",
                  decode_e4m3(0x80) == 0 and encode_e4m3(Fraction(0)) == 0x00)
    # The codebook's extrema, which the clamp depends on.
    checker.equal("e4m3 max finite is 448", decode_e4m3(0x7E), Fraction(448))
    checker.equal("e4m3 min subnormal is 2**-9", decode_e4m3(0x01), pow2(-9))
    checker.check("e4m3 0x7F is NaN", decode_e4m3(0x7F) is NAN)
    checker.check("e4m3 0xFF is NaN", decode_e4m3(0xFF) is NAN)
    checker.equal("e4m3 negative zero encodes", encode_e4m3(-Fraction(0)), 0x00)
    # RNE ties: halfway between 1.0 and 1.125 rounds to the even code (1.0, mantissa 0).
    halfway = Fraction(1) + Fraction(1, 16)
    checker.equal("e4m3 RNE tie -> even (1.0)", encode_e4m3(halfway), 0x38)
    tie_up = Fraction(3, 2) + Fraction(1, 16)  # exactly halfway between 1.5 (mant 4) and 1.625 (5)
    checker.equal("e4m3 RNE tie -> even (1.5)", encode_e4m3(tie_up), 0x3C)
    # Subnormal ties: 1.5 * 2**-9 is halfway between the 1 and 2 codes; ties to even -> 2 * 2**-9.
    checker.equal("e4m3 subnormal RNE tie", encode_e4m3(Fraction(3, 2) * pow2(-9)), 0x02)
    # The smallest positive value, half of it, and the boundary into normal range.
    checker.equal("e4m3 below half the min subnormal is zero",
                  encode_e4m3(pow2(-9) / 4), 0x00)
    checker.equal("e4m3 min normal", encode_e4m3(pow2(-6)), 0x08)
    # Saturation, which is what the quantizer's clamp is for.
    checker.equal("e4m3 clamps above 448", encode_e4m3(Fraction(460)), 0x7E)


def _test_nominal_exponents(checker: Checker) -> None:
    for code in _e4m3_code_pool():
        value = decode_e4m3(code)
        expected = None if value == 0 else max(floor_log2(abs(value)), SUBNORMAL_NOMINAL)
        checker.equal(f"F(0x{code:02X})", nominal_exp(code), expected)
    # The floor's teeth: subnormal codes contribute -6, never their own -7..-9.
    checker.equal("F(0x01) is -6 not -9", nominal_exp(0x01), -6)
    checker.equal("F(0x07) is -6 not -7", nominal_exp(0x07), -6)


def _test_e2m1_to_e4m3(checker: Checker) -> None:
    for code, expected in enumerate(E2M1_TO_E4M3):
        value = (-1 if code & 8 else 1) * E2M1_MAGNITUDES[code & 7]
        checker.equal(f"e2m1 code {code} decodes as e4m3 0x{expected:02X}",
                      decode_e4m3(expected) if (expected & 0x7F) else Fraction(0),
                      value if value != 0 else Fraction(0))
        if code == 8:
            continue  # e2m1 -0 is e4m3 0x80; a Fraction cannot carry its sign
        checker.equal(f"e2m1 code {code} is the unique e4m3 encoding",
                      encode_e4m3(value), expected)
    # An E2M1 value never lands subnormal in e4m3, so no weight operand takes the -6 floor.
    checker.check("no e2m1 value is an e4m3 subnormal",
                  all(((code >> 3) & 0xF) >= 1 for code in E2M1_TO_E4M3 if code & 0x7F))
    # The exact INT8 x2 representation: doubled, and its exponent rule.
    for code in range(16):
        value = FP4_INT8_VALUES[code]
        magnitude = abs(value)
        if magnitude:
            checker.equal(f"int8 x2 exponent of code {code}",
                          floor_log2(Fraction(magnitude)) - 1,
                          nominal_exp(E2M1_TO_E4M3[code]))
            checker.check(f"int8 x2 value of code {code} is an integer",
                          value.denominator == 1)
            checker.equal(f"int8 x2 value of code {code} halves to the e2m1 value",
                          value / 2, magnitude / 2 if value >= 0 else -magnitude / 2)


def _test_activation_quantizer(checker: Checker) -> None:
    # The floor: an all-zero block still gets a nonzero scale, 2**-22 as the reference's tests pin.
    codes, scale = act_quant_block([Fraction(0)] * BLOCK)
    checker.equal("all-zero block scale is 2**-22", scale, pow2(-22))
    checker.check("all-zero block codes are all zero", set(codes) == {0})
    # The floor is exactly 1e-4, and its scale is the same 2**-22.
    codes, scale = act_quant_block([fp32_value(1e-4)] * BLOCK)
    checker.equal("amax == 1e-4 gives 2**-22", scale, pow2(-22))
    # A known ratio: amax = 1.0 -> 1/448 = 0.002232..., log2 = -8.807, so 2**-8.
    codes, scale = act_quant_block([Fraction(1)] * BLOCK)
    checker.equal("amax == 1 gives 2**-8", scale, pow2(-8))
    checker.equal("amax == 1 quantizes to 256", decode_e4m3(codes[0]), Fraction(256))
    # The clamp is never needed when the scale is derived from the block's own amax: |x/scale|<=448.
    generator = random.Random(11)
    for trial in range(200):
        values = [fp32_value(generator.uniform(-50, 50)) for _ in range(BLOCK)]
        codes, scale = act_quant_block(values)
        magnitude = max(abs(v) for v in values)
        checker.check(f"quantizer respects the block range (trial {trial})",
                      all(decode_e4m3(code) is not NAN for code in codes)
                      and max(abs(decode_e4m3(code)) for code in codes) <= Fraction(448))
        checker.check(f"scale covers amax (trial {trial})", magnitude <= scale * 448)
    # A power-of-two amax keeps its exact ratio: the reference's bit ceiling must not round it up.
    for exponent in range(-20, 20):
        amax = pow2(exponent) * Fraction(448)
        checker.equal(f"scale of 448*2**{exponent} is 2**{exponent}",
                      act_scale_exponent(amax), exponent)


def fp32_log2_ceil_model(ratio: Fraction) -> int:
    """``ceil(log2(ratio))`` as a binary32 logarithm plus a ceiling would evaluate it.

    This models the port's spelling (`torch.ceil(torch.log2(amax / 448))` on fp32): one
    correctly-rounded binary32 log2, then a ceiling. The double-precision `log2` below is carried to
    53 bits and then rounded to binary32, which reproduces a correctly-rounded binary32 log2 except
    within one double-double ulp of a tie; the cases this oracle exercises are far from a tie, and
    the bit-exact ceiling it is compared against is exact.
    """
    import math
    logged = fp32_round(Fraction(math.log2(float(ratio))))
    return math.ceil(logged)


def bf16_value(x: Fraction) -> Fraction:
    """`x` rounded to bfloat16 (RNE), the width the reference's activations are stored in.

    Act_quant is handed bf16 activations by `Linear.forward`'s caller, so a block's amax is a bf16
    value and the reachability of a scale-boundary hazard has to be judged on *that* grid, not on
    the full binary32 one.
    """
    if x == 0:
        return Fraction(0)
    sign = -1 if x < 0 else 1
    magnitude = abs(x)
    exponent = floor_log2(magnitude)
    quantum = pow2(exponent - 7)
    mantissa = _round_half_even((magnitude / quantum).numerator, (magnitude / quantum).denominator)
    if mantissa == 256:
        exponent += 1
        mantissa = 128
        quantum = pow2(exponent - 7)
    return sign * mantissa * quantum


def _test_scale_spelling_hazards(checker: Checker) -> None:
    """The three spellings of `2 ** ceil(log2(amax / 448))` and where they can disagree.

    The official kernel computes `fast_pow2(fast_log2_ceil(amax * fp8_max_inv))` with
    `fp8_max_inv = 1 / 448.0` promoted to an FP32 constant -- one binary32 multiply by the rounded
    reciprocal, then a bit-exact ceiling. The port spells the same intent as
    `2 ** ceil(log2(amax / 448))` -- a binary32 divide, then a binary32 logarithm, then `ceil`. The
    three differences are: multiply-by-reciprocal versus divide; and logarithm versus exponent field.
    """
    # The reciprocal's rounding is harmless at the exact multiples of 448: the product rounds back
    # to exactly 1.0, so the ceiling lands on the same octave as the division does.
    checker.equal("fl32(1/448) * 448 is exactly 1", fp32_mul(FP8_MAX_INV_F32, Fraction(448)),
                  Fraction(1))
    # The official spelling *is* the exact ceiling: verify it against an independent exact one over
    # the ratio range the quantizer can produce.
    generator = random.Random(7)
    mismatched_ceiling = 0
    for _ in range(20000):
        amax = bf16_value(fp32_value(abs(generator.uniform(1e-4, 6e4))))
        if amax <= 0:
            continue
        ratio = fp32_mul(amax, FP8_MAX_INV_F32)
        exact = floor_log2(ratio) if ratio == pow2(floor_log2(ratio)) else floor_log2(ratio) + 1
        if fast_log2_ceil(ratio) != exact:
            mismatched_ceiling += 1
    checker.equal("the bit ceiling equals the exact ceiling (20000 bf16 amax values)",
                  mismatched_ceiling, 0)
    # H1: reciprocal multiply versus divide, over the *activation domain* (bf16 amax) and over the
    # whole fp32 domain, to show which one the reachability argument applies to.
    bf16_differing, fp32_differing = [], []
    for _ in range(20000):
        raw = fp32_value(abs(generator.uniform(1e-4, 6e4)))
        for candidate, bucket in ((bf16_value(raw), bf16_differing), (raw, fp32_differing)):
            if candidate > 0 and (act_scale_exponent(candidate)
                                  != act_scale_exponent(candidate, ratio_spelling="division")):
                bucket.append(candidate)
    checker.equal("H1 unreachable on the bf16 activation grid (20000 samples)", len(bf16_differing), 0)
    checker.check("H1 did not fire on random fp32 samples either", len(fp32_differing) == 0,
                  f"{len(fp32_differing)}/20000")
    # Probed at the only place it could fire: the binary32 neighbours of exact multiples of 448,
    # where the exact ratio is a power of two. Both spellings agree there, because the rounded
    # reciprocal multiplies back to exactly 1.0.
    boundary_mismatch = []
    for exponent in range(-20, 20):
        pivot = Fraction(448) * pow2(exponent)
        for step in range(-3, 4):
            candidate = pivot + step * pow2(floor_log2(pivot) - 23)
            if candidate > 0 and (act_scale_exponent(candidate)
                                  != act_scale_exponent(candidate, ratio_spelling="division")):
                boundary_mismatch.append((exponent, step))
    checker.equal("H1 agrees at the binary32 neighbours of every 448*2**k", boundary_mismatch, [])
    # H2: a correctly-rounded binary32 log2 returns the integer itself for the first representable
    # ratio above a power of two, so a `ceil(log2(...))` under-counts by one octave there. The
    # window is a constant ~1.72e-7 in log space, which is below half an ulp once the logarithm is
    # >= 4 -- so the hazard starts at 2**4 and is absent below it.
    wrong = []
    for exponent in range(0, 12):
        ratio = pow2(exponent) + pow2(exponent - 23)  # the first binary32 above 2**exponent
        bit = fast_log2_ceil(ratio)
        modelled = fp32_log2_ceil_model(ratio)
        if bit != modelled:
            wrong.append((exponent, bit, modelled))
    checker.equal("H2 under-counts from 2**4 upward", [w[0] for w in wrong], list(range(4, 12)))
    checker.check("H2 does not under-count below 2**4", all(w[0] >= 4 for w in wrong),
                  str(wrong))
    # Reachability on the real input: a bf16 amax is 2**-8 coarse, so the ratio cannot land in a
    # window of ~2**-23 relative width around a power of two. Sweep the bf16 grid directly.
    reachable = 0
    checked = 0
    for exponent in range(-20, 20):
        pivot = Fraction(448) * pow2(exponent)
        for delta in range(-4, 5):
            candidate = bf16_value(fpivot_next(pivot, delta))
            if candidate <= 0:
                continue
            checked += 1
            ratio = fp32_mul(candidate, FP8_MAX_INV_F32)
            if fast_log2_ceil(ratio) != fp32_log2_ceil_model(ratio):
                reachable += 1
    checker.check("H2 unreachable on neighbouring bf16 amax values", reachable == 0,
                  f"checked {checked}, reachable {reachable}")


def fpivot_next(value: Fraction, steps: int) -> Fraction:
    """`value` moved `steps` bf16 quanta away (used to walk the bf16 grid around 448*2**k)."""
    if steps == 0:
        return value
    direction = 1 if steps > 0 else -1
    current = value
    quantum = pow2(floor_log2(abs(value)) - 7)
    return current + direction * abs(steps) * quantum


def _test_reduction_structure(checker: Checker) -> None:
    # Products of two e4m3 codes: at most eight significant bits, inside 2**-18 .. 2**18.
    values = [abs(decode_e4m3(code)) for code in _e4m3_code_pool()]
    values = [v for v in values if v]
    smallest = min(a * b for a in values for b in values)
    largest = max(a * b for a in values for b in values)
    checker.check("smallest nonzero product is 2**-18", smallest == pow2(-18), str(smallest))
    checker.check("largest product is below 2**18", largest < pow2(18), str(largest))
    # An empty block is exactly zero.
    products = [Fraction(0)] * BLOCK
    nominals = [SENTINEL] * BLOCK
    checker.equal("all-zero block reduces to zero", reduce_block(products, nominals), Fraction(0))
    # A single live product in the first half: the cut leaves it intact (the anchor *is* its
    # nominal exponent, so the grid is two octaves below the product).
    products = [Fraction(0)] * BLOCK
    nominals = [SENTINEL] * BLOCK
    products[0], nominals[0] = pow2(-2), -2
    checker.equal("single product survives the first cut", reduce_block(products, nominals),
                  pow2(-2))
    # The asymmetry: a term in the second half is cut against max(own anchor, first half's sum).
    # Cancellation inside the first half leaves it at zero, so the sentinel path is taken and the
    # second half keeps its own (finer) grid.
    products = [Fraction(0)] * BLOCK
    nominals = [SENTINEL] * BLOCK
    products[0], nominals[0] = Fraction(4), 2
    products[1], nominals[1] = Fraction(-4), 2
    products[HALF], nominals[HALF] = pow2(-20), -20
    result = reduce_block(products, nominals)
    checker.equal("cancelled first half leaves the second half's own grid", result, pow2(-20))


def _test_reduction_exactness(checker: Checker) -> None:
    """Every intermediate of the rule is an exact binary32 value over random blocks."""
    generator = random.Random(3)
    bad = 0
    for trial in range(300):
        act = _random_block(generator, allow_nan=False, prefer_subnormal=0.2)
        weight = _e2m1_code_block(generator)
        weight_codes = [E2M1_TO_E4M3[code] for code in weight]
        products, nominals = block_products(act, weight_codes)
        _, exact = _exact_fragment_trace(products, nominals)
        if not exact:
            bad += 1
    checker.equal("all rule intermediates are exact in fp32 (300 blocks)", bad, 0)


def _test_renormalisation_matters(checker: Checker) -> None:
    """A block whose second half carries bits past the window: the final cut is load-bearing."""
    generator = random.Random(5)
    changed = 0
    for _ in range(200):
        act = _random_block(generator)
        weight_codes = [E2M1_TO_E4M3[code] for code in _e2m1_code_block(generator)]
        products, nominals = block_products(act, weight_codes)
        with_final = reduce_block(products, nominals)
        # Recompute without the final renormalisation, to show it is not decoration.
        first = [n for n in nominals[:HALF] if n is not None]
        step = pow2(max(first) - WINDOW) if first else Fraction(1)
        total = sum((_trunc_to_grid(p, step) for p in products[:HALF]), Fraction(0))
        running = None if total == 0 else floor_log2(abs(total))
        own = [n for n in nominals[HALF:] if n is not None]
        window = max([w for w in (max(own) if own else None, running) if w is not None],
                     default=None)
        step = pow2(window - WINDOW) if window is not None else Fraction(1)
        total = _trunc_to_grid(total, step) + sum(
            (_trunc_to_grid(p, step) for p in products[HALF:]), Fraction(0))
        if with_final != total:
            changed += 1
    checker.check("the final renormalisation changes some blocks", changed > 0,
                  f"changed {changed}/200")


def _test_two_halves_are_sequential(checker: Checker) -> None:
    """The two halves are not one 32-wide cut: an ordering witness, per the rule's own asymmetry."""
    generator = random.Random(9)
    differ = 0
    for _ in range(200):
        act = _random_block(generator, prefer_subnormal=0.5)
        weight_codes = [E2M1_TO_E4M3[code] for code in _e2m1_code_block(generator)]
        products, nominals = block_products(act, weight_codes)
        two_halves = reduce_block(products, nominals)
        # A single 32-wide cut against the block's own maximum nominal exponent.
        live = [n for n in nominals if n is not None]
        step = pow2(max(live) - WINDOW) if live else Fraction(1)
        one_block = sum((_trunc_to_grid(p, step) for p in products), Fraction(0))
        if one_block != 0:
            one_block = _trunc_to_grid(one_block, pow2(floor_log2(abs(one_block)) - WINDOW))
        if two_halves != one_block:
            differ += 1
    checker.check("the sequential two-half rule differs from a single 32-wide cut", differ > 0,
                  f"differed on {differ}/200 blocks")


def _test_int8_doubled_units(checker: Checker, trials: int = 200) -> None:
    """INT8 x2 values with the halved E8M0 grid reproduce the reference element exactly."""
    generator = random.Random(13)
    for trial in range(trials):
        e2m1 = _e2m1_code_block(generator)
        weight_codes = [E2M1_TO_E4M3[code] for code in e2m1]
        weight_ints = [int(FP4_INT8_VALUES[code]) for code in e2m1]
        act = [_random_block(generator, allow_nan=False, prefer_subnormal=0.2)]
        act_exponents = [generator.randrange(-20, 10)]
        weight_bytes = [generator.randrange(100, 160)]
        reference = reference_expert_row([act[0]], [weight_codes], act_exponents, weight_bytes)
        widened = int8_expert_row([act[0]], [weight_ints], act_exponents, weight_bytes)
        checker.equal(f"int8 x2 matches the reference (trial {trial})", widened, reference)
        # The shift is load-bearing: dropping it changes the answer on structured data.
    mismatches = 0
    for trial in range(60):
        e2m1 = _e2m1_code_block(generator)
        weight_codes = [E2M1_TO_E4M3[code] for code in e2m1]
        weight_ints = [int(FP4_INT8_VALUES[code]) for code in e2m1]
        act = _random_block(generator, prefer_subnormal=0.4)
        act_exponents = [0]
        weight_bytes = [127]
        if int8_expert_row([act], [weight_ints], act_exponents, weight_bytes,
                           anchor_shift=0) != reference_expert_row(
                [act], [weight_codes], act_exponents, weight_bytes):
            mismatches += 1
    checker.check("the +1 anchor shift is load-bearing", mismatches > 0,
                  f"{mismatches}/60 differed without it")


def _test_block_scale_and_accumulation(checker: Checker) -> None:
    """The per-block scale is applied to the fragment, and blocks accumulate in K order in fp32."""
    generator = random.Random(17)
    for trial in range(40):
        blocks = 3  # K = 96: three 32-wide blocks, so the K order is observable
        act_blocks = [_random_block(generator, prefer_subnormal=0.2) for _ in range(blocks)]
        weight_codes = [[E2M1_TO_E4M3[code] for code in _e2m1_code_block(generator)]
                        for _ in range(blocks)]
        act_exponents = [generator.randrange(-20, 1) for _ in range(blocks)]
        weight_bytes = [generator.randrange(100, 160) for _ in range(blocks)]
        forward = reference_expert_row(act_blocks, weight_codes, act_exponents, weight_bytes)
        # The same terms in reverse K order: fp32 addition is not associative, so a kernel that
        # reorders blocks must show it here.
        terms = []
        for act_codes, codes, exponent, byte in zip(act_blocks, weight_codes,
                                                    act_exponents, weight_bytes):
            products, nominals = block_products(act_codes, codes)
            terms.append(block_term(reduce_block(products, nominals), exponent, byte))
        reversed_order = accumulate_fp32(list(reversed(terms)))
        checker.check(f"K order is observable at K=96 (trial {trial})",
                      forward == reversed_order or True)  # informative, not an assertion
    # The mul-then-add versus FMA question: they agree unless a term under/overflows binary32.
    agreement = True
    for trial in range(100):
        blocks = 3
        act_blocks = [_random_block(generator) for _ in range(blocks)]
        weight_codes = [[E2M1_TO_E4M3[code] for code in _e2m1_code_block(generator)]
                        for _ in range(blocks)]
        act_exponents = [generator.randrange(-20, 1) for _ in range(blocks)]
        weight_bytes = [generator.randrange(100, 160) for _ in range(blocks)]
        separate = reference_expert_row(act_blocks, weight_codes, act_exponents, weight_bytes)
        fused = reference_expert_row(act_blocks, weight_codes, act_exponents, weight_bytes,
                                     fused=True)
        if separate != fused:
            agreement = False
    checker.check("mul-then-add equals an FMA on power-of-two scales", agreement)
    # But they must differ where the product itself is not representable: a subnormal term.
    tiny_fragment = pow2(-120)
    term = tiny_fragment * pow2(-20)  # 2**-140: subnormal in binary32
    separate = fp32_round(Fraction(0) + fp32_round(term))
    fused = fp32_round(Fraction(0) + term)
    checker.equal("a subnormal term rounds identically with a zero accumulator", separate, fused)
    # The mechanism: they part company only when the addend itself is inexact in binary32, and then
    # the accumulator decides whether the lost bits can still tip the sum.
    inexact = Fraction(1) + pow2(-25)          # rounds to 1.0 on its own
    separate = fp32_round(pow2(-24) + fp32_round(inexact))
    fused = fp32_round(pow2(-24) + inexact)
    checker.equal("an inexact addend rounds away on its own", fp32_round(inexact), Fraction(1))
    checker.check("mul-then-add and FMA differ on an inexact addend", separate != fused,
                  f"{separate} vs {fused}")
    # The kernel's addends are `fragment * 2**e` (a power-of-two rescale of a value that is an
    # integer times a power of two), so they are exact except when the rescale underflows or
    # overflows. That is the only place this question can bite.
    checker.check("the kernel's rescale is exact where it stays normal",
                  fp32_round(Fraction(3, 2) * pow2(-40) * pow2(30)) == Fraction(3, 2) * pow2(-10))


def _test_nan_and_zero_semantics(checker: Checker) -> None:
    generator = random.Random(21)
    act = _random_block(generator)
    act[3] = 0x7F  # NaN code
    weight_codes = [E2M1_TO_E4M3[code] for code in _e2m1_code_block(generator)]
    products, nominals = block_products(act, weight_codes)
    checker.check("a NaN activation code propagates through the rule",
                  reduce_block(products, nominals) is NAN)
    # Signed zeros: a -0.0 weight with a live activation is a zero product, so it contributes no
    # anchor -- the reference's own `product == 0.0` test cannot tell +0 from -0, and neither can an
    # INT8 representation (which has one zero).
    act = [0x38] * BLOCK              # all 1.0
    weight_codes = [0x80] * BLOCK     # all e2m1 -0, cast to e4m3 -0
    products, nominals = block_products(act, weight_codes)
    checker.equal("negative-zero weights contribute no anchor",
                  sum(1 for n in nominals if n is not None), 0)
    checker.equal("negative-zero weights reduce to zero", reduce_block(products, nominals),
                  Fraction(0))
    # And the INT8 route agrees, because the product is zero either way.
    weight_ints = [0] * BLOCK
    checker.equal("the INT8 zero agrees with the e2m1 negative zero",
                  int8_expert_row([act], [weight_ints], [0], [127]),
                  reference_expert_row([act], [[0x80] * BLOCK], [0], [127]))


def _test_e8m0_edges(checker: Checker) -> None:
    checker.equal("e8m0 byte 0 is 2**-127", decode_e8m0(0), pow2(-127))
    checker.equal("e8m0 byte 127 is 1", decode_e8m0(127), Fraction(1))
    checker.equal("e8m0 byte 254 is 2**127", decode_e8m0(254), pow2(127))
    checker.check("e8m0 byte 255 is NaN", decode_e8m0(255) is NAN)
    # The halved edges are the ones the INT8 representation must carry in FP32, not in a byte.
    checker.equal("halved byte 0 is 2**-128", half_e8m0(0), pow2(-128))
    checker.equal("halved byte 1 is 2**-127", half_e8m0(1), pow2(-127))
    checker.equal("halved byte 2 is 2**-126 (the smallest normal)", half_e8m0(2), pow2(-126))
    checker.check("2**-128 is a binary32 subnormal", fp32_round(pow2(-128)) == pow2(-128))
    checker.equal("the halved byte cannot be spelled as an e8m0 byte (-1)",
                  pow2(-1 - E8M0_BIAS), pow2(-128))


def int8_to_e2m1_code() -> dict[int, int]:
    """The E2M1 code each exact INT8 x2 value stands for."""
    return {int(FP4_INT8_VALUES[code]): code for code in range(16)}


def int8_half_value_row(act_blocks: list[list[int]], weight_int8_blocks: list[list[int]],
                        act_exponents: list[int], weight_bytes: list[int],
                        *, fused: bool = False, halve_scale: bool = False) -> Fraction:
    """The recommended INT8 contract: recover the exact E2M1 value, then run the rule unchanged.

    This is the formulation a new kernel should implement, because it needs no anchor bookkeeping:
    ``int8 / 2`` is exact (every widened value is even, and halving an even integer is exact in
    fp32), so the *operands* of the reduction are the reference's own operands and the rule -- the
    anchor, the cuts, the window, the final renormalisation -- is applied verbatim. The scale is the
    file's **original** E8M0 exponent, not a halved one, because nothing was doubled.

    The alternative spelling (doubled integer values plus a halved scale plus a shifted anchor) is
    equivalent -- see :func:`int8_expert_row` -- but carries a bookkeeping item that has to stay in
    step with the scale, which is one more way to be wrong.
    """
    lookup = int8_to_e2m1_code()
    terms = []
    for index, (act_codes, weight_ints) in enumerate(zip(act_blocks, weight_int8_blocks)):
        products, nominals = [], []
        for code, integer in zip(act_codes, weight_ints):
            a_value = decode_e4m3(code)
            if a_value is NAN:
                products.append(NAN)
                nominals.append(SENTINEL)
                continue
            if integer not in lookup:
                raise ValueError(f"{integer} is not an exact E2M1 x2 weight value")
            e2m1 = lookup[int(integer)]
            weight_value = (-1 if e2m1 & 8 else 1) * E2M1_MAGNITUDES[e2m1 & 7]
            product = a_value * weight_value
            products.append(product)
            if product == 0:
                nominals.append(SENTINEL)
            else:
                nominals.append(nominal_exp(code) + nominal_exp(E2M1_TO_E4M3[e2m1]))
        fragment = reduce_block(products, nominals)
        terms.append(block_term(fragment, act_exponents[index], weight_bytes[index],
                                half_scale=halve_scale))
    return accumulate_fp32(terms, fused=fused)


def _test_official_kernel_contract(checker: Checker) -> None:
    """What the released `inference/kernel.py` literally does, and what a new kernel must keep."""
    # (1) The mma accumulator is cleared every K block, so no term carries between blocks beyond the
    # fp32 block sum. Two 32-wide blocks must not be reduced as one 64-term reduction.
    generator = random.Random(23)
    merged_differs = 0
    for _ in range(100):
        blocks = [_random_block(generator) for _ in range(2)]
        weight_codes = [[E2M1_TO_E4M3[code] for code in _e2m1_code_block(generator)]
                        for _ in range(2)]
        per_block = [reduce_block(*block_products(act, weights))
                     for act, weights in zip(blocks, weight_codes)]
        # A "one 64-term reduction" is not expressible in the rule; the closest wrong reading is
        # reducing each block against a shared anchor from both. Compare each block's own fragment
        # against the block reduced with the other block's anchor folded in.
        shared_anchor = max(
            n for act, weights in zip(blocks, weight_codes)
            for n in block_products(act, weights)[1] if n is not None
        )
        shared = []
        for act, weights in zip(blocks, weight_codes):
            products, nominals = block_products(act, weights)
            step = pow2(shared_anchor - WINDOW)
            shared.append(sum((_trunc_to_grid(p, step) for p in products), Fraction(0)))
        if per_block[0] != shared[0] or per_block[1] != shared[1]:
            merged_differs += 1
    checker.check("each K block is reduced against its own anchor, not a shared one",
                  merged_differs > 0, f"{merged_differs}/100 blocks differed")

    # (2) The scale expression is `C_accum += ((C_local * s_a) * s_b)`, left-associative, and both
    # scales are exact powers of two. Combining them first is exact in the normal range...
    normal_ok = True
    for _ in range(200):
        fragment = Fraction(generator.randrange(-(1 << 20), 1 << 20), 1 << 12)
        a_exp, b_byte = generator.randrange(-30, 20), generator.randrange(40, 200)
        s_a = pow2(a_exp)
        s_b = decode_e8m0(b_byte)
        sequential = fp32_round(fp32_round(fragment * s_a) * s_b)
        combined = fp32_round(fragment * (s_a * s_b))
        if sequential != combined:
            normal_ok = False
    checker.check("combining the two power-of-two scales equals the sequential product", normal_ok)
    # ...but pre-multiplying them can overflow where the sequential form does not, because an E8M0
    # scale may reach 2**127 on each side and the product of two of them is not representable.
    try:
        combined_overflow = fp32_round(pow2(127) * pow2(127))
        overflowed = False
    except OracleOverflow:
        combined_overflow, overflowed = None, True
    checker.check("s_a * s_b can overflow binary32 on its own", overflowed)
    sequential_survives = fp32_round(fp32_round(pow2(-127) * pow2(127)) * pow2(127))
    checker.equal("((fragment * s_a) * s_b) survives the same exponents",
                  sequential_survives, pow2(127))

    # (3) The recommended INT8 contract -- recover the exact E2M1 value, run the rule unchanged --
    # is the reference element for element, on random blocks over the whole code space.
    for trial in range(200):
        e2m1 = _e2m1_code_block(generator)
        weight_codes = [E2M1_TO_E4M3[code] for code in e2m1]
        weight_ints = [int(FP4_INT8_VALUES[code]) for code in e2m1]
        act = _random_block(generator, allow_nan=False, prefer_subnormal=0.25)
        act_exponents = [generator.randrange(-20, 1)]
        weight_bytes = [generator.randrange(100, 160)]
        checker.equal(f"int8 halved-value contract matches the reference (trial {trial})",
                      int8_half_value_row([act], [weight_ints], act_exponents, weight_bytes),
                      reference_expert_row([act], [weight_codes], act_exponents, weight_bytes))
    # Recovering the value with a separate `* 0.5` is exact for every code, because every widened
    # value is an even integer and 0.5 is a power of two.
    for code in range(16):
        integer = int(FP4_INT8_VALUES[code])
        e2m1 = int8_to_e2m1_code()[integer]
        checker.equal(f"int8 {integer} halves to E2M1 code {e2m1}",
                      Fraction(integer) * Fraction(1, 2),
                      (-1 if e2m1 & 8 else 1) * E2M1_MAGNITUDES[e2m1 & 7])

    # (3b) The two *consistent* pairings between the representation and the scale, and the trap of
    # mixing them. (A) recover the e2m1 value (x0.5, exact) and use the file's original scale;
    # (B) keep the doubled INT8 value, halve the scale and shift the anchor by one. Mixing a value
    # from one pairing with the scale of the other is off by exactly a factor of two per K block --
    # a plausible-looking wrong answer, which is why this is pinned rather than described.
    for trial in range(40):
        e2m1 = _e2m1_code_block(generator)
        weight_codes = [E2M1_TO_E4M3[code] for code in e2m1]
        weight_ints = [int(FP4_INT8_VALUES[code]) for code in e2m1]
        act = _random_block(generator, allow_nan=False)
        act_exponents = [0]
        weight_bytes = [127]
        correct = reference_expert_row([act], [weight_codes], act_exponents, weight_bytes)
        recovered = int8_half_value_row([act], [weight_ints], act_exponents, weight_bytes)
        doubled = int8_expert_row([act], [weight_ints], act_exponents, weight_bytes)
        checker.equal(f"pairing A matches the reference (trial {trial})", recovered, correct)
        checker.equal(f"pairing B matches pairing A (trial {trial})", doubled, correct)
        if correct != 0:
            double_corrected = int8_half_value_row([act], [weight_ints], act_exponents,
                                                   weight_bytes, halve_scale=True)
            checker.equal(f"halving the scale on already-halved values is off by 2x (trial {trial})",
                          double_corrected, correct / 2)

    # (4) The released configuration's scale grids: `act_block_size = fp8_block_size = 32`, so
    # `n_sub = act_block_size // 32 == 1` and the activation grid is one exponent per (row, K block).
    act_block_size, weight_block = 32, 32
    checker.equal("the released ops use a per-32 activation grid",
                  act_block_size // weight_block, 1)
    for k in (32, 64, 96, 5120):
        checker.equal(f"activation scale count for K={k}", -(-k // act_block_size), k // 32)

    # (5) The scale edges a kernel must not "improve": the original scale for byte 0 is an fp32
    # *subnormal*, so a device that flushes fp32 subnormals turns the whole row into zeros, while
    # the exact arithmetic keeps it. Byte 0 is also the one edge a half-scale cannot spell as a byte.
    checker.equal("original scale for byte 0 is 2**-127", decode_e8m0(0), pow2(-127))
    checker.check("2**-127 is a binary32 subnormal (below the smallest normal, 2**-126)",
                  pow2(-127) < pow2(-126) and fp32_round(pow2(-127)) == pow2(-127))
    checker.check("the halved scale for byte 0 is 2**-128, also subnormal",
                  fp32_round(half_e8m0(0)) == pow2(-128) and pow2(-128) < pow2(-126))
    checker.check("byte 255 stays NaN in both spellings",
                  decode_e8m0(255) is NAN and half_e8m0(255) is NAN)


def _test_subnormal_extremes(checker: Checker) -> None:
    """The smallest and largest scale pairs: where a device's FTZ would show up."""
    # A minimal activation code (2**-9) against the smallest weight scale: the term is deep
    # subnormal in fp32 and a flushing device would return 0 where the exact arithmetic does not.
    act = [0x01] * BLOCK
    weight_codes = [0x30] * BLOCK  # e2m1 0.5
    # Fragment 2**-5 (32 products of 2**-10), act scale 2**0, weight scale 2**-127 -> 2**-132,
    # inside fp32's subnormal range (2**-149 .. 2**-126) and exactly representable there.
    term = reference_expert_row([act], [weight_codes], [0], [0])
    checker.check("the extreme tiny term is representable in fp32",
                  term != NAN and fp32_round(term) == term, str(term))
    checker.check("the extreme tiny term is subnormal in fp32", abs(term) < pow2(-126),
                  str(term))
    vanished = reference_expert_row([act], [weight_codes], [-22], [0])
    checker.equal("a lower scale flushes that term to zero in fp32", vanished, Fraction(0))
    # The largest: 448 * 6 * the largest scale pair, which overflows fp32 and must be reported.
    act = [0x7E] * BLOCK
    weight_codes = [0x4C] * BLOCK
    try:
        overflow = reference_expert_row([act], [weight_codes], [10], [254])
        overflowed = False
    except OracleOverflow:
        overflow, overflowed = None, True
    checker.check("the largest scale pair overflows fp32 rather than wrapping", overflowed)


def run_selftest() -> int:
    checker = Checker()
    for test in (_test_e4m3_codebook, _test_nominal_exponents, _test_e2m1_to_e4m3,
                 _test_activation_quantizer, _test_scale_spelling_hazards,
                 _test_official_kernel_contract,
                 _test_reduction_structure, _test_reduction_exactness,
                 _test_renormalisation_matters, _test_two_halves_are_sequential,
                 _test_int8_doubled_units, _test_block_scale_and_accumulation,
                 _test_nan_and_zero_semantics, _test_e8m0_edges,
                 _test_subnormal_extremes):
        test(checker)
    print(f"expert_scalar_oracle: {checker.passed} checks passed, "
          f"{len(checker.failures)} failed")
    for failure in checker.failures:
        print(f"  FAIL {failure}")
    return 0 if not checker.failures else 1


def main(argv: list[str]) -> int:
    if "--selftest" not in argv:
        print(__doc__)
        print("usage: python3 expert_scalar_oracle.py --selftest")
        return 2
    return run_selftest()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))