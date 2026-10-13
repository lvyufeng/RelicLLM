"""The reference's fp8 GEMM written as arithmetic, instead of as a call to a tensor core.

The reference does not compute a dot product. `inference/kernel.py`'s `fp8_gemm_kernel` is a tilelang
kernel whose inner step is one `tl::mma_sync<e4m3, e4m3, fp32, 16, 8, 32>` per output element, and an
`mma.sync` does not add 32 products exactly. Everything after it *is* exact -- the K loop is
`C_local_accum[i, j] += C_local[i, j] * scale_a[i] * scale_b[j]` in fp32 and both scales are e8m0, so
multiplying by them is a power-of-two rescale that changes no bits -- which leaves the mma's own
32-term reduction as the only place the reference departs from a correctly rounded sum.

That departure is small and it is not noise. Measured on a real `wq_a` (M=9, N=1280, K=5120, the
checkpoint's own recorded operands and scales), the reference differs from the exact fp64 sum by a
median 4e-5 relative, which is enough to land on the other side of the bf16 rounding boundary for
**2.41%** of output elements. Those flips compound through the network and move the logits. They are
why a port that computes a correctly rounded GEMM cannot reproduce the reference's tokens: the port
was the *more accurate* implementation, and reproducing the reference means reproducing its rounding
rather than beating it.

The rule, per 32-wide K block:

1. Two sequential 16-wide halves, and the accumulator starts at zero for every K block -- nothing
   carries between blocks beyond the fp32 block sum.
2. Each half is cut onto a grid set by that half's anchor, truncating **toward zero**:
   ``anchor_h = max over k of (F(a_k) + F(b_k))`` with ``F(v) = max(true exponent of v, -6)``. ``F``
   is read off the e4m3 *encoding*, which is why the floor is exactly -6: a subnormal operand
   contributes -6, never its own -7..-9. A zero operand contributes nothing. The anchor is **per
   output element** -- not per tile, not per row.
3. The running sum joins the second half as a peer rather than as a container for it:
   ``anchor = max(anchor_h, E(C_in))`` where ``E`` is ``floor(log2 |.|)``, then
   ``C = TZ(C_in, g) + sum_k TZ(a_k b_k, g)`` with ``g = 2 ** (anchor - 13)``.
4. A final renormalisation, ``C = TZ(C, 2 ** (E(C) - 13))`` -- about fourteen significand bits, toward
   zero. This step moves outputs: omitting it breaks 43-170 elements of the reference's own trace.

Accuracy: bit-identical to the reference kernel on 1,142,784/1,142,784 elements over 32 dense
configurations, 382,976/382,976 on real layer-0 attention weights, and 11520/11520 end to end on the
real `wq_a` with real e8m0 scales -- where a correctly rounded sum of the same operands reaches only
11242/11520.

**Scope.** e4m3 x e4m3 only, which is what the reference's dense and attention linears use. The MoE
experts take an e4m3 activation against an **e2m1** weight (`expert_dtype == "fp4"`); e2m1 has a
different exponent field, so the nominal-exponent rule above does not transfer to it and nothing here
is claimed about that path.

**Cost.** This transcribes a tensor-core instruction in elementwise torch, so it is far slower than
the matmul it replaces, and it works in fp64 because the rule is stated on exact products. Its
intermediates are therefore eight bytes an element and it has to tile: the peak is one
``[rows, block, cols]`` product plus the slices it is built from, never a whole operand. That is the
price of bit parity on a card with no fp8 path at all, which is the point on sm_75; callers that want
throughput and do not need to reproduce the reference's rounding should not call it.
"""

from __future__ import annotations

import torch

#: The reference's own block width: `kernel.py`'s `block_K = group_size = fp8_block_size = 32`.
BLOCK = 32

#: One `mma.sync` handles 16 of those 32 K values, and the kernel issues two of them per K block.
HALF = 16

#: The reduction keeps this many binary places below the anchor.
WINDOW = 13.0

#: "This product has no nominal exponent." A zero operand contributes nothing to its half's anchor,
#: so its term is excluded rather than given the smallest exponent. The value is never used
#: arithmetically -- only compared, in `max`/`amax`, and tested to collapse the grid to a step of one.
_SENTINEL = -1.0e9

#: `_cut` evaluates ``2 ** (ref - WINDOW)``, which underflows to zero for a sentinel. Testing against
#: half the sentinel, rather than the sentinel, keeps the comparison exact in float64.
_SENTINEL_GUARD = _SENTINEL / 2

#: e4m3's largest finite value and the amax floor the reference clamps a block against before taking
#: its scale -- `kernel.py`'s `fp8_max` and `fp8_amax_floor`.
FP8_MAX = 448.0
FP8_AMAX_FLOOR = 1e-4

#: How many fp64 product elements one pass of :func:`fp8_block_gemm` may hold at once. The product is
#: ``[rows, BLOCK, cols]`` and it is the largest live tensor in the loop, so the row tile and the
#: column group are chosen against this. A multiple of :data:`BLOCK` keeps the weight scales aligned
#: to the walk.
#:
#: This is a **resident-memory** limit, not a tidiness one. The serving configuration runs four ranks
#: on one 24 GiB card with the prefix cache resident, and a first attempt at this loop that decoded
#: its operands whole died there: the engram's `wkv` is 147M elements, an fp64 copy of it is 1.17 GiB,
#: and the card had 0.67 GiB free. It matters most in *decode*, where one output row is the whole
#: first dimension -- there the product is `[1, BLOCK, N]` and N is what the budget cuts -- while a
#: long prefill cuts the row axis instead.
_MAX_PRODUCT_ELEMENTS = 1 << 21


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """Any 8-bit-encoded tensor as `uint8` bytes -- e4m3 codes or e8m0 scales alike.

    The rule is stated on encoding fields, not on values, so every entry point here works on bytes.
    """
    return t if t.dtype == torch.uint8 else t.view(torch.uint8)


def _pow2(exponent: torch.Tensor) -> torch.Tensor:
    """``2 ** exponent`` exactly, for an integer-valued float64 exponent tensor.

    **This is `torch.exp2` and nothing else.** Every other spelling was measured on the card and
    every other spelling is wrong: ``torch.pow(2.0, e)`` is ``exp2(e * log2(2))`` and ``log2(2.0)``
    is not bit-exact on CUDA, and ``torch.ldexp(1.0, e)`` -- which by its name and its host behaviour
    ought to be exact -- comes back one ulp low for 16 of the 254 e4m3 codes. Both errors are around
    1e-16 relative, which sounds harmless and is not, because a power of two here is a *grid*:
    `_cut` divides by it and multiplies back, and a grid that is a hair below the intended power of
    two moves the truncation boundary, so a term that should survive lands exactly on an integer and
    is cut away instead. ``exp2`` is the hardware instruction and is exact on both devices over the
    whole range this module can produce; the test file pins that rather than assuming it.
    """
    return torch.exp2(exponent.to(torch.float64))


def _exp_of(x: torch.Tensor) -> torch.Tensor:
    """``floor(log2 |x|)`` exactly, for a float64 tensor.

    ``frexp`` rather than ``log2``: ``log2`` is correctly rounded, so ``floor`` of a value that ought
    to be an exact integer can come out one below it, and that would silently move a grid by an
    octave. ``x == 0`` has no exponent; callers guard on the value rather than on this.
    """
    _, exponent = torch.frexp(x.abs().clamp_min(torch.finfo(torch.float64).tiny))
    return (exponent - 1).to(torch.float64)


def nominal_exp(codes: torch.Tensor) -> torch.Tensor:
    """``F(v) = max(true exponent of v, -6)``, read off the e4m3 encoding field.

    A subnormal e4m3 code has a true exponent of -7, -8 or -9, and all three contribute -6 -- the
    exponent the encoding field implies, not the exponent of the value. That distinction is the last
    0.05% of elements and it only shows up on dense data: synthetic corpora never separated it, while
    dropping the floor scores 7153/7168 wrong on a payload that is entirely subnormal.

    Zero is not made special here. Callers exclude zero *products* by value, because a zero operand
    contributes nothing rather than contributing -6.
    """
    field = (_bytes(codes).to(torch.int64) >> 3) & 0xF
    return torch.where(field >= 1, field - 7, torch.full_like(field, -6)).to(torch.float64)


def decode_e4m3(codes: torch.Tensor) -> torch.Tensor:
    """e4m3 codes as their exact float64 values.

    Decoded from the encoding by hand rather than cast: ``float8_e4m3fn`` has no dtype promotion, so
    ``.to(torch.float64)`` is unavailable on some backends and on gathered or sliced views, and this
    has to run on the sm_75 path where no fp8 arithmetic exists at all. Every code is exact, and
    ``0x7F``/``0xFF`` decode to NaN rather than to the 480 that the normal formula would give them.
    """
    raw = _bytes(codes).to(torch.int64)
    sign = torch.where((raw >> 7) & 1 == 1, -1.0, 1.0)
    field = (raw >> 3) & 0xF
    mantissa = raw & 0x7
    # Normals are 2**(field - 7) * (1 + m/8); the field == 0 encodings are subnormals, m/8 * 2**-6.
    # The significand has four significant bits and the scale is a power of two, so the product is
    # exactly representable and IEEE multiplication returns it exactly.
    significand = 1.0 + mantissa.to(torch.float64) / 8.0
    normal = significand * _pow2((field - 7).to(torch.float64))
    subnormal = (mantissa.to(torch.float64) / 8.0) * _pow2(torch.full_like(field, -6).to(torch.float64))
    value = torch.where(field >= 1, normal, subnormal)
    is_nan = (field == 0xF) & (mantissa == 0x7)
    value = torch.where(is_nan, torch.full_like(value, float("nan")), value)
    return sign * value


def e8m0_exponent(scale: torch.Tensor, *, verify: bool = True) -> torch.Tensor:
    """An e8m0 scale's exponent, exactly, as an int64 -- the inverse of the encoding.

    Through ``frexp`` rather than ``log2``: a scale is a power of two, and ``log2`` of an exact power
    of two can round down and give an exponent one too large, which would shift a whole block.
    Accepts either the ``float8_e8m0fnu`` bytes or the fp32 values the port's quantizers return.

    The fp32 form is the one that needs checking -- a tensor that is not a power of two has no e8m0
    exponent, and ``frexp`` will hand back a plausible-looking one anyway -- so ``verify`` is on by
    default and the check is a host ``bool`` over a device tensor, i.e. a synchronisation. **It must
    therefore be off on any path that runs inside a captured CUDA graph**, which is the port's
    default decode path (``backends/v41_backend.py`` ``decode_graphs=True``) and the reason the
    check exists at load time instead: ``loader.fill`` recovers and verifies every weight's codes
    once, under no capture, and the hot path never re-derives them. The byte form needs no check --
    an e8m0 byte *is* an exponent.
    """
    if scale.dtype == torch.float8_e8m0fnu:
        return _bytes(scale).to(torch.int64) - 127
    mantissa, exponent = torch.frexp(scale.to(torch.float64))
    if verify and not bool((mantissa.abs() == 0.5).all()):
        raise ValueError("an e8m0 scale must be a power of two")
    return (exponent - 1).to(torch.int64)


def _cut(x: torch.Tensor, ref: torch.Tensor, off: float = WINDOW) -> torch.Tensor:
    """Truncate ``x`` toward zero onto the grid ``2 ** (ref - off)``.

    Toward zero -- replacing ``torch.trunc`` with a nearest-rounding breaks 127-501 elements of the
    reference's own trace. The grid is a power of two, so the divide and the multiply back are exact
    and the truncation is the only step that loses anything. ``ref`` is per element, which is the
    whole point: the anchor is per output element, so the grid is too.
    """
    step = torch.where(ref < _SENTINEL_GUARD, torch.ones_like(ref), _pow2(ref - off))
    return torch.trunc(x / step) * step


def reduce_block(product: torch.Tensor, nominal: torch.Tensor) -> torch.Tensor:
    """The mma's 32-term reduction, for a ``[..., HALF * 2]`` block of products.

    ``product`` holds the exact products of the two operands' codes and ``nominal`` their nominal
    exponents summed, both float64. The two halves are not symmetric: the first establishes a window
    and the second is cut against ``max(its own anchor, the exponent of what the first half summed
    to)``, so the running sum is a peer of the second half's products. That asymmetry is what every
    tile-scoped model missed.
    """
    window = nominal[..., :HALF].amax(dim=-1)
    total = _cut(product[..., :HALF], window.unsqueeze(-1)).sum(dim=-1)

    # A sum of exactly zero has no exponent, and taking one from `tiny` would drag the second half's
    # window down to -1022 and flatten it. Carry the sentinel instead.
    running = torch.where(total != 0.0, _exp_of(total), torch.full_like(total, _SENTINEL))
    window = torch.maximum(nominal[..., HALF:].amax(dim=-1), running)
    total = _cut(total, window) + _cut(product[..., HALF:], window.unsqueeze(-1)).sum(dim=-1)

    # The final renormalisation, kept separate from the cuts above: the second half's sum can carry
    # bits past the window, and this is the step a probe cannot see unless told to look for it.
    return torch.where(total != 0.0, _cut(total, _exp_of(total)), total)


def fp8_block_gemm(a_codes, a_scale, b_codes, b_scale, block_size: int = BLOCK) -> torch.Tensor:
    """The reference's fp8 GEMM for ``[M, K] x [N, K]``, accumulated in fp32 over 32-wide K blocks.

    ``a_codes``/``b_codes`` are e4m3 codes, as ``float8_e4m3fn`` or as their raw bytes. Both scales
    are per 32-wide K block and are applied the way the kernel applies them, ``(fragment * s_a) *
    s_b`` after the reduction: ``a_scale`` is ``[M, K // block_size]``, one exponent per (row, K
    block), and ``b_scale`` is ``[N // block_size, K // block_size]``, the 32x32 grid the checkpoint
    stores beside a weight and the one the kernel indexes as ``scale_b[j]`` for output row ``j``.

    The two operands are never dequantized. The rule is stated on codes, and running it on codes keeps
    the per-block scales out of the exponents -- they are applied once, at the end of the block, which
    is what the kernel does and what keeps the two sides of the comparison in the same units.

    Both scales are read with ``e8m0_exponent(..., verify=False)``: that function's power-of-two
    check is a host ``bool`` over device data, i.e. a synchronisation, and this runs inside the
    captured decode body. Nothing is lost by skipping it here. A weight's grid is verified once, in
    ``codes_from_scale``, before any codes are attached to it -- a grid that is not a power of two
    does not describe the weight it was attached to and is refused there -- and an activation scale
    comes out of `block_fp8_act_quant` as ``2 ** ceil(log2(...))`` by construction.

    The loop walks K blocks outermost and decodes each operand **slice per block**, never whole. Both
    halves of that are load-bearing on a card: an fp64 copy of a weight is eight bytes an element and
    the engram's `wkv` is 147M of them, and the three loops are ordered so that no slice is decoded
    twice -- the column groups partition N and the row tiles partition M, so each operand element is
    decoded exactly once however the budget happens to cut the product. See
    :data:`_MAX_PRODUCT_ELEMENTS`.
    """
    a_bytes, b_bytes = _bytes(a_codes), _bytes(b_codes)
    m, k = a_bytes.shape[-2], a_bytes.shape[-1]
    n = b_bytes.shape[0]
    if k % block_size:
        raise ValueError(f"K = {k} is not a multiple of the {block_size} block size")
    if n % block_size:
        raise ValueError(f"N = {n} is not a multiple of the {block_size} block size")
    blocks = k // block_size
    if tuple(a_scale.shape) != (m, blocks):
        raise ValueError(f"activation scales {tuple(a_scale.shape)} are not [{m}, {blocks}]")
    if tuple(b_scale.shape) != (n // block_size, blocks):
        raise ValueError(
            f"weight scales {tuple(b_scale.shape)} are not [{n // block_size}, {blocks}]"
        )

    # The rule is stated on the codes and the scales are applied after, so the two acquire their
    # device independently of the operands: `e8m0_exponent` reads a host tensor's bytes the same way
    # it reads a device tensor's, and the caller's quantizers return whichever device they were
    # given. Moving them here keeps the inner loop free of a device question, and it is a device copy
    # and not a host one on the paths that matter -- a weight grid is hung on the weight's own device
    # by `attach_scale`.
    device = a_bytes.device
    a_exp = e8m0_exponent(a_scale, verify=False).to(device)
    b_exp = e8m0_exponent(b_scale, verify=False).to(device)

    # How much of the product to hold at once. Rows are cut first, because tiling them costs nothing
    # -- one output row already needs a `[1, BLOCK, N]` product, so a wide N is what has to come off
    # the column axis instead, and both cuts are partitions of independent output elements.
    rows, cols = m, n
    if m * block_size * n > _MAX_PRODUCT_ELEMENTS:
        if block_size * n <= _MAX_PRODUCT_ELEMENTS:
            rows = max(1, _MAX_PRODUCT_ELEMENTS // (block_size * n))
        else:
            rows = 1
            cols = (_MAX_PRODUCT_ELEMENTS // block_size) // block_size * block_size

    out = torch.zeros(m, n, dtype=torch.float32, device=device)
    for block in range(blocks):
        at = slice(block * block_size, (block + 1) * block_size)
        # `2 ** e` is exact in both formats, so the scales stay powers of two and the block sum stays
        # a plain rescale.
        a_scale_block = _pow2(a_exp[:, block].to(torch.float64)).to(torch.float32)
        for lo in range(0, n, cols):
            hi = min(lo + cols, n)
            # One weight scale per block_size-row group of the output, repeated down its own rows
            # before it meets the reduction's K axis -- `repeat_interleave` matches what the 32x32
            # grid means.
            b_scale_block = _pow2(b_exp[lo // block_size:hi // block_size, block].to(torch.float64))
            b_scale_block = b_scale_block.to(torch.float32).repeat_interleave(block_size)
            b_block = b_bytes[lo:hi, at]
            b_dec = decode_e4m3(b_block)
            b_nom = nominal_exp(b_block)
            for m0 in range(0, m, rows):
                m1 = min(m0 + rows, m)
                a_block = a_bytes[m0:m1, at]
                product = decode_e4m3(a_block).unsqueeze(1) * b_dec.unsqueeze(0)
                nominal = nominal_exp(a_block).unsqueeze(1) + b_nom.unsqueeze(0)
                # A zero operand contributes nothing to the anchor. Counting it as -6 instead is
                # provably unobservable -- every product is a multiple of 2**-18, and both candidate
                # grids are finer than that lattice -- but excluding it is what the anchor was
                # measured to do.
                nominal = torch.where(product == 0.0, torch.full_like(nominal, _SENTINEL), nominal)
                fragment = reduce_block(product, nominal).to(torch.float32)
                sa = a_scale_block[m0:m1].unsqueeze(-1)
                out[m0:m1, lo:hi] += fragment * sa * b_scale_block
    return out


#: Where a loader hangs a weight's literal e8m0 grid, for :func:`codes_from_scale` to find.
#:
#: An attribute on the parameter rather than a table keyed by ``id(weight)``: the scale has to travel
#: with the tensor the modules hand to ``quantized_linear``, and an id-keyed registry is the same
#: arrangement with an address that can be recycled by a tensor that has been freed.
SCALE_ATTR = "_relicllm_block_scale"

#: Where :func:`attach_codes` hangs the codes recovered from that grid, beside it.
CODES_ATTR = "_relicllm_block_codes"


def attach_scale(weight: torch.Tensor, scale: torch.Tensor) -> None:
    """Hang the checkpoint's own scale grid on the weight it scales. See :func:`codes_from_scale`.

    The grid is moved to the weight's device here, at load, rather than on every call. The loader
    attaches it to a parameter that is already where it belongs -- `fill` copies into the parameter
    in place, so nothing ever re-homes the weight -- but the grid arrives out of the file as a view
    of the shard mapping, which is **host** memory. A grid left on the host costs nothing while
    nothing reads it; the moment a GEMM does, it is a per-call host-to-device copy inside the decode
    body, allocating a fresh tensor each time where the recorded graph expects the one address it
    captured.

    A grid replaces whatever grid was there, and codes belong to **a** grid: a weight that arrives
    here already carrying them is one whose pair is about to be half-replaced, and the surviving half
    is a runnable wrong answer -- the exponents the rule reads would belong to a weight this one is
    not, and a plausible scale quantizes to plausible codes, so nothing downstream can notice. An
    absent pair cannot fail that way: it is the correctly rounded product, reported by
    ``quantized_linear``. So the codes go with the grid that produced them, which is why
    ``loader.fill`` attaches the two in this order -- a re-fill drops the old codes here and attaches
    the new ones a few lines later.
    """
    if attached_codes(weight) is not None:
        delattr(weight, CODES_ATTR)
    setattr(weight, SCALE_ATTR, scale.to(weight.device))


def attached_scale(weight: torch.Tensor) -> torch.Tensor | None:
    """The grid :func:`attach_scale` put on this weight, or ``None`` for one that never had any.

    ``None`` is the honest answer for a weight whose file stored no scale beside it -- a bf16 one, or
    a tensor a caller built by hand -- and callers treat it as "no codes are recoverable here"
    rather than as an error.
    """
    return getattr(weight, SCALE_ATTR, None)


def attach_codes(weight: torch.Tensor, codes: torch.Tensor) -> None:
    """Hang the e4m3 codes recovered from that grid on the weight, beside the grid itself.

    Recovery is not something the hot path can do. :func:`codes_from_scale` ends in a per-row-chunk
    ``torch.equal`` against the weight (a host ``bool`` over device data, i.e. a synchronisation) and
    reads the weight in full; the read is merely expensive, and the synchronisation is one of the
    calls CUDA forbids while a stream is being captured, which is what the port's default decode path
    does. So it happens exactly once, in ``loader.fill`` -- the only place a weight arrives with its
    grid -- and the hot path finds the result here.

    These are that grid's codes rather than the weight's, and :func:`attach_scale` is what keeps the
    two in step: it drops them when it replaces the grid they were recovered against.
    """
    setattr(weight, CODES_ATTR, codes)


def attached_codes(weight: torch.Tensor) -> torch.Tensor | None:
    """The codes :func:`attach_codes` put on this weight, or ``None`` for a weight without any.

    ``None`` is what ``quantized_linear`` reads as "no exact codes are recoverable here, run the
    correctly rounded GEMM instead and say so once". Two kinds of weight get it: one whose file
    stored no grid to recover from -- a bf16 weight, an fp4 routed expert's dense decode, a tensor a
    caller built by hand -- and one whose grid was refused because it did not describe the weight.
    The second kind is reported by name where it happens, at load.
    """
    return getattr(weight, CODES_ATTR, None)


def codes_from_scale(
    dense: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = BLOCK,
    *,
    verify: bool = True,
) -> torch.Tensor:
    """Recover a **dense** weight's e4m3 codes from the e8m0 grid the checkpoint stored beside it.

    This is the exact inverse of ``dequant_fp8_weight`` and it is what :func:`quantize_blocks` cannot
    be. ``224 * 2**E`` and ``448 * 2**(E - 1)`` are the same number, so a tile whose largest code is
    224 -- measured at 4.66% of the checkpoint's tiles -- has *two* power-of-two scales that re-expand
    to exactly the same dense values. The values alone do not say which one the file used, and
    ``quantize_blocks(verify=True)`` cannot tell the two apart, because it compares values: it
    accepted the recovered grid and the codes disagreed with the file's on 13 of 6400 `wq_a` tiles.
    That matters here and not in the dequantizer, because the rule reads the code's *exponent field*
    (``nominal_exp``), and the octave the codes were written against is part of what it reads.

    Handed the file's own grid there is nothing to decide: dividing by a power of two is exact in
    every format, and the cast that follows has an exact value to land on. ``verify`` re-checks that
    -- the cast back against the exact quotient, per row chunk -- so a weight that is *not* on this
    grid is refused rather than quantized against a scale that belongs to a different tensor.

    **This is a load-time call, not a hot-path one.** The ``verify`` comparison is a host ``bool``
    over device data, i.e. a synchronisation, and a sync inside the captured decode body is fatal to
    the ``decode_graphs`` serving path. ``loader.fill`` calls it once per weight, under no capture,
    and hangs the result on the weight with :func:`attach_codes`; the GEMM reads
    :func:`attached_codes`. The one other caller, the indexer's row-split band, derives its grid from
    a weight this already ran on and passes ``verify=False``, for the same reason.

    The loop is over 32-row chunks on purpose. A full-size reconstruction is an fp32 copy of the
    weight plus an fp32 copy of its expanded grid, and the all-at-once version of this is what asked
    for 1.17 GiB inside a TP4 rank that had 1.05 GiB left.
    """
    rows, cols = dense.shape[-2:]
    if rows % block_size or cols % block_size:
        raise ValueError(
            f"a {tuple(dense.shape)} weight is not a multiple of the {block_size} grid"
        )
    want = (rows // block_size, cols // block_size)
    if tuple(scale.shape) != want:
        raise ValueError(
            f"a {tuple(dense.shape)} weight wants a {want} scale grid and the one attached to it is "
            f"{tuple(scale.shape)}"
        )
    # The grid is read where the *weight* lives, not where the grid does. `loader.fill` attaches the
    # grid to a parameter that is already on its final device -- the tree is built there
    # (`Backbone(..., device=...)`) and filled in place (`parameter.copy_(value)`) -- but the grid
    # arrives as a view of the shard mapping, which is host memory, and `attach_scale` is the one
    # place that moves it across. A caller holding a host grid and a device weight, as a test or a
    # hand-built tensor may, would otherwise ask the card for a host step and get
    # "Expected all tensors to be on the same device" out of the next GEMM.
    exponent = e8m0_exponent(scale).to(dense.device)
    codes = torch.empty((rows, cols), dtype=torch.float8_e4m3fn, device=dense.device)
    for lo in range(0, rows, block_size):
        hi = min(lo + block_size, rows)
        step = _pow2(exponent[lo // block_size].to(torch.float64)).to(torch.float32)
        quotient = dense[lo:hi].to(torch.float32) / step.repeat_interleave(block_size)
        row = quotient.to(torch.float8_e4m3fn)
        if verify and not torch.equal(row.to(torch.float32), quotient):
            raise ValueError(
                f"the {block_size}x{block_size} block quantization of a {tuple(dense.shape)} weight "
                f"does not round-trip against the scale grid attached to it, so that grid belongs to "
                f"a different weight and this one's codes cannot be recovered"
            )
        codes[lo:hi] = row
    return codes


def quantize_blocks(
    x: torch.Tensor, block_size: int = BLOCK, *, verify: bool = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference's **weight** quantization: e8m0 scales on a square block grid.

    Returns ``(codes, scale)`` with ``codes`` in ``float8_e4m3fn`` shaped like ``x`` and ``scale``
    fp32 of shape ``[rows // block_size, cols // block_size]`` -- the layout the checkpoint stores
    beside a weight and the one :func:`fp8_block_gemm` takes.

    The reference picks the scale as the smallest power of two leaving ``max|w| / scale`` inside the
    e4m3 range, so that ratio lands in ``(224, 448]`` by construction. For a corpus this is the
    reference's own quantizer and a fixture builder. It is **not** a way back from a dequantized
    weight to the codes the reference used, which is what it was written for and what it cannot do:
    a tile maximizing at 224 admits two scales that re-expand to the same values, only one of which
    is the file's, and the choice is visible to the reduction rule. A caller holding the checkpoint's
    grid wants :func:`codes_from_scale`; a caller holding only the dense weight has no exact answer
    and should not pretend otherwise.

    ``verify`` re-expands the codes and raises if they do not reproduce ``x`` bit for bit. That is a
    check on the values, so it does not and cannot catch the ambiguity above.
    """
    rows, cols = x.shape[-2:]
    if rows % block_size or cols % block_size:
        raise ValueError(f"a {tuple(x.shape)} weight is not a multiple of the {block_size} grid")
    tiles = x.reshape(rows // block_size, block_size, cols // block_size, block_size)
    flat = tiles.permute(0, 2, 1, 3).reshape(rows // block_size, cols // block_size, -1)
    amax = flat.abs().amax(dim=-1).clamp_min(FP8_AMAX_FLOOR).to(torch.float64)
    # The reference's `ceil(log2(amax / 448))`, via `frexp` rather than `log2`. `log2` of an exact
    # power of two is not required to round to the integer itself on either device, and a scale that
    # came out an octave high would quantize the whole tile against the wrong grid and still look
    # plausible. `frexp` is exact, so the branch below is decided by bit inspection: a value whose
    # mantissa is exactly one half *is* a power of two, and `ceil` must not round it up.
    ratio = amax / FP8_MAX
    mantissa, binary_exponent = torch.frexp(ratio)
    at_power_of_two = mantissa.abs() == 0.5
    exponent = torch.where(at_power_of_two, binary_exponent - 1, binary_exponent)
    scale = _pow2(exponent.to(torch.float64))
    codes = (flat.to(torch.float64) / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    codes = codes.reshape(rows // block_size, cols // block_size, block_size, block_size)
    codes = codes.permute(0, 2, 1, 3).reshape(rows, cols)
    if verify:
        check = decode_e4m3(codes).to(torch.float32) * \
            scale.repeat_interleave(block_size, dim=-1).repeat_interleave(block_size, dim=0)
        if not torch.equal(check, x.to(torch.float32)):
            raise ValueError(
                f"the {block_size}x{block_size} block quantization of a {tuple(x.shape)} weight "
                f"does not round-trip; its scales are not the reference's power-of-two grid, so the "
                f"codes and their nominal exponents cannot be recovered from the dense weight"
            )
    return codes, scale.to(torch.float32)