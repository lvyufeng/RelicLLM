"""Triton Ascend kernel for the documented pair-rotation arithmetic.

The released runtime rotates consecutive last-axis ``(real, imag)`` pairs by
multiplying them as ``complex64`` against a ``cos + i sin`` table, i.e. source
``(a*c - b*s, a*s + b*c)``. A compiled complex multiply fuses one product per
component and rounds the other on its own, but **which** one is a property of the
reference build, not of complex multiplication: the Ascend host's aarch64 ATen
measured the second-product fusion for the real component and matched no single
documented spelling for the imaginary one, while an x86 build measured the
first-product form. This kernel therefore implements an explicit, documented
arithmetic -- :data:`relic_core.kernels.rotary.DOCUMENTED_POLICY` -- and does not
claim bit parity with a host whose reference contracts differently:

    real = fma(a, c, -round_f32(b * s))
    imag = fma(a, s, +round_f32(b * c))

where ``a``/``b`` are the FP32 pair and ``c``/``s`` the FP32 cosine/sine. This is
an **unbound experimental operator arithmetic**: the runtime's real-pair rotary
is what the model uses, and nothing here is wired into a forward pass.
``tests/test_ascend_rotary_math.py`` re-executes :func:`_rotate_lane` below
against an independent exact-rational statement of the two lines, and
``tests/test_ascend_rotary.py`` holds the device to the same exact model.

Two details are easy to get wrong and both are pinned by tests:

* the addend is a *pre-rounded* FP32 product, so ``b * s`` must not be fused into
  the ``fma`` as well -- the kernel computes it as its own value and passes it;
* the fused product is the first, not the second. Swapping the two is worth about
  one FP32 ulp on ordinary inputs, which survives a BF16 store on almost every
  element and still flips it on a large one.

The fused step is an explicit ``tl.fma``. It is *not* left to the compiler to
contract ``first * angle + addend``, because which product gets fused is the
whole contract: a compiler that fused the other one would silently produce a
different arithmetic than the one this file documents.

No complex arithmetic, no FP64 and no expanded table are used. Every lane reads
one element, its pair partner and one ``(cos, sin)`` pair, and writes one
element, so the whole op is a gather/update over the caller's own strides.
"""
from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def _rotate_lane(this, other, cos, sin, even):
    """Rotate one output lane of a ``(real, imag)`` pair.

    ``this`` is the lane's own input, ``other`` its pair partner, and ``even``
    selects the real (``True``) or imaginary (``False``) half. Written as one
    expression over both halves rather than two branches, because the lane layout
    -- not the branch -- is what the vectorised kernel iterates:

    * real lane:  ``a = this``,  ``b = other``, angle ``c``, other angle ``s``
    * imag lane:  ``a = other``, ``b = this``,  angle ``s``, other angle ``c``

    so in both cases ``out = fma(a, angle, sign * round_f32(b * other_angle))``
    with ``sign`` negative on the real half. ``-product`` is an exact negation,
    which is why it may be written that way instead of multiplying by ``-1``.
    """
    first = tl.where(even, this, other)
    second = tl.where(even, other, this)
    pair_angle = tl.where(even, cos, sin)
    other_angle = tl.where(even, sin, cos)
    # Rounded on its own, then handed to the fused step: this product is the
    # addend, never a second fused multiply.
    product = second * other_angle
    addend = tl.where(even, -product, product)
    return tl.fma(first, pair_angle, addend)


@triton.jit
def rotary_pairs_kernel(
    X, TABLE, ROWS, WIDTH, HALF_WIDTH, PROGRAMS, SEQUENCE, HEADS,
    STRIDE_BATCH, STRIDE_SEQUENCE, STRIDE_HEAD, STRIDE_WIDTH,
    K_TILE: tl.constexpr, K_STEPS: tl.constexpr, INVERSE: tl.constexpr,
    INDEX64: tl.constexpr,
):
    """Rotate ``ROWS`` ``(sequence, head)`` rows of ``WIDTH`` elements in place.

    ``TABLE`` is ``[SEQUENCE, WIDTH / 2, 2]`` FP32, laid out ``(cos, sin)`` along
    its last axis, so the pair's two angles are adjacent. ``ROWS`` is
    ``batch * SEQUENCE * HEADS`` flattened row-major, and one program walks rows
    ``pid, pid + PROGRAMS, ...``. ``K_TILE`` lanes cover at most ``WIDTH``
    elements per step, and ``STEPS`` may exceed one for a width the tile does not
    divide; the tail is masked.

    Every lane loads its own element *and* its partner before either is stored,
    so this is safe in place. A partner always belongs to the same tile: the
    lane's width offset flips exactly one low bit, and a power-of-two ``K_TILE``
    keeps consecutive steps on even boundaries.

    Strides are the caller's, so a head slice of a fused buffer rotates without a
    copy; a layout that could map two lanes onto one address is refused in the
    Python layer before this kernel is reached.
    """
    pid = tl.program_id(0)
    lane = tl.arange(0, K_TILE)
    for row in range(pid, ROWS, PROGRAMS):
        # Widen the row *before* it is decomposed. Casting the finished offset
        # would be too late: ``batch * STRIDE_BATCH`` can overflow int32 on its own
        # for a long sequence, and an overflowed product is not repaired by
        # widening the sum it already corrupted. ``wide_row`` is int64 exactly when
        # the Python layer decided any address can need it, so every product below
        # promotes instead of wrapping.
        wide_row = row.to(tl.int64) if INDEX64 else row
        head = wide_row % HEADS
        rest = wide_row // HEADS
        sequence = rest % SEQUENCE
        batch = rest // SEQUENCE
        base = batch * STRIDE_BATCH + sequence * STRIDE_SEQUENCE + head * STRIDE_HEAD
        for step in range(K_STEPS):
            column = step * K_TILE + lane
            active = column < WIDTH
            # Same rule for the column: widen before multiplying by the width
            # stride, not after the product has already wrapped.
            widened = column.to(tl.int64) if INDEX64 else column
            offset = base + widened * STRIDE_WIDTH
            # Flipping the low bit moves to the pair partner without an int64
            # xor: the adjustment is a small int and promotes with the offset.
            partner = offset + (1 - 2 * (column & 1)) * STRIDE_WIDTH
            pair = column >> 1
            table_index = sequence * (2 * HALF_WIDTH) + pair * 2
            cos = tl.load(TABLE + table_index, mask=active, other=0.0)
            sin = tl.load(TABLE + table_index + 1, mask=active, other=0.0)
            this = tl.load(X + offset, mask=active, other=0.0)
            other = tl.load(X + partner, mask=active, other=0.0)
            if INVERSE:
                # The reference rotates by the conjugate, i.e. the same multiply
                # with a negated sine. Negation is exact, so this is the only
                # change the reverse direction needs.
                sin = -sin
            value = _rotate_lane(
                this.to(tl.float32), other.to(tl.float32), cos, sin,
                (column & 1) == 0,
            )
            tl.store(X + offset, value.to(X.dtype.element_ty), mask=active)