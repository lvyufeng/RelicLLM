"""The routed experts' fp4 path: the e2m1 -> e4m3 cast, and the per-row grid the rule scales by.

A routed expert's checkpoint weight is e2m1 (`expert_dtype == "fp4"`), and the reference's routed
GEMM is the same e4m3 x e4m3 `mma` as every dense one -- `fp4_gemm_kernel` casts each four-bit value
to e4m3 *inside the kernel* before the `T.gemm`. So `mma`'s reduction rule describes this path too,
and the two things a routed call needs are the codes that cast produces and the grid the kernel
scales the reduction by: `scales_b[row, k]`, one exponent per **output row** and 32-wide K block,
where a dense fp8 weight's grid is one per 32x32 tile.

Both are pinned here, and both are the kind of claim that fails silently. A wrong nibble order, a
widened table with one sign flipped, a per-row grid read as if it were a block grid: each of them
still returns a plausible float, and each of them is exact on the payloads nobody happens to test.
So the table is compared against an independent transcription of the four-bit ladder *and* against
relic-core's own unpacker, the per-row layout is held against a scalar transcription of the kernel's
loop, and the store that attaches the pair is exercised through a fake checkpoint rather than
inspected by eye.

`quantized_linear` and `expert_forward` are transcribed a second time at the bottom for the same
reason the rule is: a seam between two modules is where an argument goes missing, and the two
transcriptions are the specification for what the routed forward is supposed to do with the pair.

CPU-only, like the rule it pins: the derivation is a gather and the GEMM is soft torch.
"""

from __future__ import annotations

import pytest

# The suite runs on boxes that may not carry torch; the guard has to come before the module's own
# import so a torch-less host skips this file rather than failing to collect it.
torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from relicllm.models.deepseek_v4_1 import mma  # noqa: E402
from relicllm.models.deepseek_v4_1.kernels import (  # noqa: E402
    _fp4_values,
    block_fp8_act_quant,
    dequant_fp4_weight,
    quantized_linear,
)
from relicllm.models.deepseek_v4_1.loader import (  # noqa: E402
    CheckpointRoutedExperts,
    scale_key,
)
from relicllm.models.deepseek_v4_1.modules import expert_forward  # noqa: E402

BLOCK = mma.BLOCK

#: The four-bit ladder in code order: e2m1's top bit is its sign, and codes 0..7 are the eight
#: magnitudes. Spelled out here rather than read off `kernels._FP4_MAGNITUDES` so that the table
#: below is checked against an independent statement of the encoding; the two are compared to each
#: other in `test_the_widening_table_is_the_reference_kernels_cast`, which is what makes "the port's
#: quantizer and the checkpoint's packer agree on code order" a checked fact.
#:
#: Nibble 8 is a **negative** zero (`-0.0`), not a positive one: the sign bit is set, and the cast
#: preserves it. The choice is unobservable in the arithmetic -- both zeros are the same operand --
#: but writing it down is what keeps the table and the ladder comparable byte for byte.
E2M1_LADDER = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _e2m1_codes(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    """e2m1 *codes* drawn from all sixteen, as bytes."""
    return torch.randint(0, 16, shape, generator=generator).to(torch.uint8)


def _e4m3_codes(shape: tuple[int, ...], generator: torch.Generator) -> torch.Tensor:
    """Random e4m3 codes, as bytes, with the two NaN encodings excluded.

    Codes and not values: the rule reads the encoding field, and `0x7F`/`0xFF` are NaN rather than
    the 480 the normal formula would give them. This is the activation side of the comparisons
    below, where a subnormal code is a legitimate operand rather than something to avoid.
    """
    raw = torch.randint(0, 256, shape, generator=generator, dtype=torch.int64)
    raw = torch.where((raw & 0x7F) == 0x7F, raw ^ 0x01, raw)
    return raw.to(torch.uint8)


def _pack(codes: torch.Tensor) -> torch.Tensor:
    """Two e2m1 codes per byte along the last axis, low nibble first -- the checkpoint's layout."""
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


def _row_grid(rows: int, groups: int, generator: torch.Generator, low: int = -14, high: int = -4):
    """A per-row e8m0 grid, `[rows, groups]`, of real power-of-two exponents."""
    exponent = torch.randint(low, high, (rows, groups), generator=generator).to(torch.int64)
    return (exponent + 127).to(torch.uint8).view(torch.float8_e8m0fnu)


def _expand(codes: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """The dense value a code matrix and a per-row grid describe, in fp32.

    The exponent is applied by a power-of-two multiply rather than through whatever the checkpoint's
    own dequantizer does, because the two are compared against each other below: writing it out here
    is what makes them independent.
    """
    exponent = mma.e8m0_exponent(grid).to(torch.float64)
    scale = mma._pow2(exponent).to(torch.float32)
    return mma.decode_e4m3(codes).to(torch.float32) * scale.repeat_interleave(BLOCK, dim=-1)


def test_the_widening_table_is_the_reference_kernels_cast() -> None:
    """`E2M1_TO_E4M3` is the cast the fp4 kernel performs, and that cast is exact.

    The table is a transcription, and a transcription is what a reader cannot check by looking: so it
    is held against an independent statement of the same encoding, and against torch's own
    `e2m1 value -> e4m3` cast, which is the arithmetic the reference's `T.Cast(FP8, T.Cast(FP32,
    ...))` performs. Then the port's *quantizer* is compared against the ladder, since a code order
    that agreed with the file but not with `kernels._fp4_values` would mean the port writes nibbles
    nobody can read back.

    The exactness claim is in the values: every widened code decodes to its ladder value bit for bit,
    so the widening rounds nothing and the routed reduction really does run on the operands the
    reference's mma is handed.
    """
    ladder = torch.tensor(E2M1_LADDER, dtype=torch.float32)
    table = torch.tensor(mma.E2M1_TO_E4M3, dtype=torch.uint8)

    assert torch.equal(ladder.to(torch.float8_e4m3fn).view(torch.uint8), table), (
        "the table is not what casting the e2m1 ladder to e4m3 gives"
    )

    decoded = mma.decode_e4m3(table.view(torch.float8_e4m3fn))
    assert torch.equal(decoded, ladder.to(torch.float64)), (
        f"the widening is not value-exact: {int((decoded != ladder.to(torch.float64)).sum())} of 16 "
        f"codes decode to something else"
    )

    assert torch.equal(_fp4_values(torch.arange(16)).to(torch.float64), ladder.to(torch.float64)), (
        "the quantizer's code order is not the ladder this table is indexed by, so the packer and "
        "the reader disagree about which nibble is which"
    )

    # And no code lands in e4m3's subnormal range, where `nominal_exp`'s -6 floor -- not the value's
    # own exponent -- is what the reduction would read. The reference's operands are e4m3 codes of
    # values its kernel just cast, so nothing there is subnormal either; a table that put one here
    # would move that block's anchor by whole octaves. The two zeros are excluded by their low seven
    # bits rather than by the byte, since `0x80` is a zero the byte comparison above calls nonzero.
    field = (table.to(torch.int64) >> 3) & 0xF
    assert bool((field[(table & 0x7F) != 0] >= 1).all()), (
        "a nonzero e2m1 value widened into an e4m3 subnormal, where the rule reads the -6 floor "
        "instead of the value's own exponent"
    )


def test_the_nibbles_widen_low_first_and_land_where_the_unpacker_puts_them() -> None:
    """The packed byte's low nibble is the even K, and the derivation says so in the file's order.

    This is the one convention with no second witness inside the derivation: byte `b` carries two
    codes, and reading them the other way round transposes every adjacent pair of the K axis -- a
    weight that is a permutation of the real one, which nothing about the shapes or the scales can
    reveal. So it is checked twice: against the interleave written out by hand, and against
    relic-core's own `dequant_fp4_weight`, which unpacks the same bytes for the dense decode. The
    second is what the design rests on -- a routed expert needs no repack, because the codes come out
    of the nibbles exactly -- so the two have to describe one weight.
    """
    generator = _gen(0)
    rows, cols = 8, 2 * BLOCK
    packed = _pack(_e2m1_codes((rows, 2 * cols), generator))
    codes = mma.e4m3_codes_from_e2m1(packed)

    table = torch.tensor(mma.E2M1_TO_E4M3, dtype=torch.uint8)
    expected = torch.empty(rows, 2 * cols, dtype=torch.uint8)
    expected[:, 0::2] = table[(packed & 0x0F).long()]
    expected[:, 1::2] = table[(packed >> 4).long()]
    assert torch.equal(codes.view(torch.uint8), expected), (
        "the widening does not interleave the low nibble to the even K and the high one to the odd K"
    )

    grid = _row_grid(rows, 2 * cols // BLOCK, _gen(1))
    dense = dequant_fp4_weight(packed.view(torch.int8), grid, BLOCK)
    reconstructed = _expand(codes, grid)
    assert torch.equal(dense, reconstructed), (
        f"{int((dense != reconstructed).sum())} of {dense.numel()} values differ between the file's "
        f"own unpacking of the packed weight and the derived codes times the row's scale, so the "
        f"derivation and the dense decode are not two descriptions of one weight"
    )

    if torch.cuda.is_available():
        on_card = mma.e4m3_codes_from_e2m1(packed.cuda())
        assert on_card.device.type == "cuda" and torch.equal(on_card.cpu(), codes), (
            "the widening table is not built on the device it is indexed on, so the card's codes "
            "would differ from the host's"
        )


def test_a_materialised_e4m3_repack_is_the_tensor_the_derivation_already_builds() -> None:
    """Widening each nibble through its *value* and widening it through the table are one tensor.

    A converted checkpoint could carry an e4m3 copy of every expert, and the question this answers is
    whether it would have to. It would not: a repack builds its codes out of the nibble's value and a
    cast, and every e2m1 value is exact in e4m3, so the repack's tensor is the derivation's tensor --
    byte for byte, and therefore the two GEMMs are the same bits and not merely the same to a
    tolerance. That is also why no expert has to be read twice: the packed bytes already say
    everything the reduction reads.

    The negative half is the reason the codes come from the *nibbles* rather than from the file's
    dense copy. Casting the scaled value back to e4m3 re-encodes it against a canonical exponent --
    `224 * 2**E` and `448 * 2**(E-1)` are one number with two codes -- and the rule reads the
    exponent field, so a code recovered after the scale is not the code the reference's kernel
    multiplied. Here a grid of `2**-127` underflows the cast to zero outright, which is the same loss
    at its clearest.
    """
    generator = _gen(6)
    rows, cols = 2 * BLOCK, 2 * BLOCK  # a routed expert's output dim is a multiple of the block
    nibbles = _e2m1_codes((rows, cols), generator)
    packed = _pack(nibbles)
    grid = _row_grid(rows, cols // BLOCK, generator)

    derived = mma.e4m3_codes_from_e2m1(packed)
    materialised = _fp4_values(nibbles).to(torch.float8_e4m3fn)
    assert torch.equal(derived, materialised), (
        f"{int((derived.view(torch.uint8) != materialised.view(torch.uint8)).sum())} of "
        f"{derived.numel()} codes differ between the nibble widening and a repack through the "
        f"nibble's own value"
    )

    # A routed-expert-shaped weight and an activation quantized the way this port quantizes one.
    activation = torch.randn(3, cols, generator=generator)
    act_codes, act_scale = block_fp8_act_quant(activation)
    assert torch.equal(
        mma.fp8_block_gemm(act_codes, act_scale, derived, grid, BLOCK),
        mma.fp8_block_gemm(act_codes, act_scale, materialised, grid, BLOCK),
    ), "the routed GEMM on derived codes is not the routed GEMM on a materialised e4m3 copy"

    underflowed = torch.zeros(rows, cols // BLOCK, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    rescaled = _expand(derived, underflowed).to(torch.float8_e4m3fn)
    assert not torch.equal(rescaled, derived), (
        "casting the scaled weight back to e4m3 recovered the codes here, so this fixture cannot "
        "show that a code read after the scale is not the code the kernel multiplied"
    )


def test_the_per_row_grid_is_one_the_block_grid_could_only_repeat() -> None:
    """The two accepted scale layouts are one rule: `[N, blocks]` is `[N // 32, blocks]` ungrouped.

    The routed experts are stored with one exponent per output row (`fp4_gemm_kernel` indexes
    `scales_b[row, k]`), and every dense fp8 weight with one per 32-row tile (`fp8_gemm_kernel`
    indexes `scales_b[row // group_size, k]`). A per-row grid whose rows are constant within each
    group is therefore the same description of the same GEMM, and both calls have to return the same
    bits -- not close, equal. What the second half shows is that the row layout is not a repackaging
    of the block one: a grid that varies *within* a group multiplies individual output rows by
    exponents no block grid can name, and a call that quietly grouped it would still be right on
    every grid that does not.
    """
    generator = _gen(2)
    m, n, k = 3, 64, 2 * BLOCK
    a_codes = _e4m3_codes((m, k), generator)
    b_codes = _e4m3_codes((n, k), generator)
    a_scale = _row_grid(m, k // BLOCK, generator)

    grouped_exponent = torch.randint(-8, -2, (n // BLOCK, k // BLOCK), generator=generator)
    grouped = (grouped_exponent + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    per_row = (grouped_exponent.repeat_interleave(BLOCK, dim=0) + 127).to(torch.uint8)
    per_row = per_row.view(torch.float8_e8m0fnu)

    assert torch.equal(
        mma.fp8_block_gemm(a_codes, a_scale, b_codes, grouped, BLOCK),
        mma.fp8_block_gemm(a_codes, a_scale, b_codes, per_row, BLOCK),
    ), "a per-row grid that repeats its block grid's exponent gave a different GEMM"

    varying = (torch.randint(-8, -2, (n, k // BLOCK), generator=generator) + 127).to(torch.uint8)
    varying = varying.view(torch.float8_e8m0fnu)
    assert not torch.equal(
        mma.fp8_block_gemm(a_codes, a_scale, b_codes, varying, BLOCK),
        mma.fp8_block_gemm(a_codes, a_scale, b_codes, grouped, BLOCK),
    ), (
        "a grid that differs between the rows of one group gave the same answer as a block grid, so "
        "this fixture cannot tell the per-row layout from the block one"
    )

    # A shape that is neither grid is still refused rather than read as whichever one it is not.
    with pytest.raises(ValueError, match="per-row grid an fp4 expert is"):
        mma.fp8_block_gemm(a_codes, a_scale, b_codes, varying[: BLOCK // 2], BLOCK)


def _row_scale_rule(a_codes, a_scale, b_codes, b_scale, block_size: int = BLOCK) -> torch.Tensor:
    """`fp4_gemm_kernel`'s loop with a per-row weight grid, written out element by element.

    The specification for the layout rather than a second opinion about the reduction: the same
    `mma.reduce_block` the module uses builds each 32-term block, and the surrounding loop -- blocks
    outermost, `out[i, j] += fragment * sa[i] * sw[j]` in fp32, truncating nothing between blocks --
    is transcribed from the kernel's `C_local_accum` update and from the fp32 accumulator it lands
    in. What it settles is which exponent output column `j` gets: `b_scale[j, block]`, with no
    grouping and no repetition down a tile.
    """
    m, k = a_codes.shape
    n = b_codes.shape[0]
    a_bytes, b_bytes = mma._bytes(a_codes), mma._bytes(b_codes)
    a_exp = mma.e8m0_exponent(a_scale, verify=False).to(torch.float64)
    b_exp = mma.e8m0_exponent(b_scale, verify=False).to(torch.float64)

    out = torch.zeros(m, n, dtype=torch.float32)
    for index in range(k // block_size):
        at = slice(index * block_size, (index + 1) * block_size)
        a_block, b_block = a_bytes[:, at], b_bytes[:, at]
        product = mma.decode_e4m3(a_block).unsqueeze(1) * mma.decode_e4m3(b_block).unsqueeze(0)
        nominal = mma.nominal_exp(a_block).unsqueeze(1) + mma.nominal_exp(b_block).unsqueeze(0)
        nominal = torch.where(product == 0.0, torch.full_like(nominal, mma._SENTINEL), nominal)
        fragment = mma.reduce_block(product, nominal).to(torch.float32)
        for i in range(m):
            for j in range(n):
                out[i, j] += (
                    fragment[i, j]
                    * mma._pow2(a_exp[i, index]).to(torch.float32)
                    * mma._pow2(b_exp[j, index]).to(torch.float32)
                )
    return out


def test_the_per_row_layout_is_the_kernels_loop_element_by_element() -> None:
    """`fp8_block_gemm` on a per-row grid is the kernel's own loop, output element by element.

    The scalar transcription above is the contract, and a vectorised rewrite of a loop is exactly
    where a grouping slips back in: indexing a per-row grid by `j // 32` is right for every grid
    constant within its group and wrong for a real expert's, where the 32 rows of a group were
    quantized against 32 different activations. So the two are compared with a grid drawn per row,
    over codes drawn from the whole encoding space -- subnormals included, since an activation block
    does contain them.
    """
    generator = _gen(3)
    m, n, k = 4, 32, 2 * BLOCK
    a_codes = _e4m3_codes((m, k), generator).view(torch.float8_e4m3fn)
    b_codes = _e4m3_codes((n, k), generator).view(torch.float8_e4m3fn)
    a_scale = _row_grid(m, k // BLOCK, generator)
    b_scale = _row_grid(n, k // BLOCK, generator)

    got = mma.fp8_block_gemm(a_codes, a_scale, b_codes, b_scale, BLOCK)
    want = _row_scale_rule(a_codes, a_scale, b_codes, b_scale)
    assert torch.equal(got, want), (
        f"{int((got != want).sum())} of {got.numel()} outputs differ from the kernel's loop "
        f"transcribed with the per-row grid"
    )


def _rule_linear(x: torch.Tensor, weight: torch.Tensor, block_size: int = BLOCK) -> torch.Tensor:
    """`quantized_linear(quantized=True)`'s emulated branch, written out from the pair.

    Nothing here is a re-derivation: it is the emulated branch's four steps -- quantize the
    activation at block 32, run the rule on the attached codes *and the attached grid*, cast back to
    the caller's width, restore the leading dimensions -- stated once more so that the store's
    attachment and the function that consumes it are checked against each other rather than against
    themselves.
    """
    flat = x.reshape(-1, x.shape[-1])
    act_codes, act_scale = block_fp8_act_quant(flat, block_size)
    out = mma.fp8_block_gemm(
        act_codes, act_scale, mma.attached_codes(weight), mma.attached_scale(weight), block_size
    )
    return out.to(x.dtype).reshape(*x.shape[:-1], weight.shape[0])


def _rule_forward(x, w1, w2, w3, block_size: int = BLOCK) -> torch.Tensor:
    """`expert_forward` on a routed expert, written out the same way: three rule linears and a SwiGLU.

    The clamps and the routed weight multiplier are left out because the call under test leaves them
    out too; a version that included them would be comparing two different functions.
    """
    gate = _rule_linear(x, w1, block_size).float()
    up = _rule_linear(x, w3, block_size).float()
    return _rule_linear((F.silu(gate) * up).to(x.dtype), w2, block_size)


class _FakeCheckpoint:
    """The four calls `CheckpointRoutedExperts` makes, over tensors a test built by hand.

    A real checkpoint is the loader suite's business (it writes one and reads it back); what this
    pins is the store's own behaviour -- which keys it reads, and whether the pair it attaches is on
    the tensor a caller gets back, on a cache hit as well as on a miss.
    """

    def __init__(self, dense: dict[str, torch.Tensor], packed: dict[str, torch.Tensor]) -> None:
        self._dense = dense
        self._packed = packed

    def weight(self, key: str, *, dtype=None, device=None):
        value = self._dense[key]
        return value if dtype is None else value.to(dtype)

    def packed(self, key: str) -> torch.Tensor:
        return self._packed[key]

    def is_packed_fp4(self, key: str) -> bool:
        return key in self._packed and self._packed[key].dtype == torch.int8


def _without_pair(weight: torch.Tensor) -> torch.Tensor:
    """The same numbers with the fp4 pair taken off, for the control below.

    `clone()` is not enough to say this: whether a clone carries the `__dict__` a tensor's attributes
    live in is a torch implementation detail, and a control that silently kept the pair would pass
    the comparison it exists to fail.
    """
    stripped = weight.clone()
    for attr in (mma.CODES_ATTR, mma.SCALE_ATTR):
        if hasattr(stripped, attr):
            delattr(stripped, attr)
    return stripped


def _fake_expert(seed: int):
    """One small expert's three projections, packed and dense, plus the grids beside them."""
    generator = _gen(seed)
    dim = inter = 2 * BLOCK
    packed, dense, grids = {}, {}, {}
    for which, rows, cols in (("w1", inter, dim), ("w2", dim, inter), ("w3", inter, dim)):
        weight_key = f"layers.0.ffn.experts.0.{which}.weight"
        codes = _e2m1_codes((rows, cols), generator)
        grid = _row_grid(rows, cols // BLOCK, generator)
        raw = _pack(codes).view(torch.int8)
        packed[weight_key] = raw
        packed[scale_key(weight_key)] = grid
        dense[weight_key] = dequant_fp4_weight(raw, grid, BLOCK).to(torch.bfloat16)
        grids[which] = grid
    return _FakeCheckpoint(dense, packed), packed, grids


def test_a_store_miss_hands_back_the_pair_the_rule_reads() -> None:
    """The store attaches the codes and the grid, and they are still there on a cache hit.

    This is the seam that fails silently in both directions. Without the attachment
    `quantized_linear` runs the correctly rounded product and says so once on stderr; with a *wrong*
    attachment it runs the rule on the wrong operands and says nothing at all. So the codes that come
    back are compared with the ones the packed bytes imply, the grid with the file's, and then the
    emulation is run the whole way through -- against the rule called by hand, and against the
    correctly rounded product, because a store that attached nothing would still pass the first
    comparison if the rule had stopped changing anything.
    """
    checkpoint, packed, grids = _fake_expert(4)
    store = CheckpointRoutedExperts(
        checkpoint, 0, n_experts=2, dim=2 * BLOCK, inter_dim=2 * BLOCK, cache_size=1, quantized=True
    )

    w1, w2, w3 = store.expert(0)
    for which, weight in (("w1", w1), ("w2", w2), ("w3", w3)):
        key = f"layers.0.ffn.experts.0.{which}.weight"
        assert torch.equal(mma.attached_codes(weight), mma.e4m3_codes_from_e2m1(packed[key])), (
            f"{which}: the codes on the weight are not the ones its packed nibbles imply"
        )
        assert torch.equal(
            mma.attached_scale(weight).view(torch.uint8), grids[which].view(torch.uint8)
        ), f"{which}: the grid on the weight is not the one stored beside it"

    again = store.expert(0)
    assert again[0] is w1 and mma.attached_codes(again[0]) is not None, (
        "a cache hit handed back a tensor without the fp4 pair, so every warm expert would fall "
        "back to the correctly rounded product"
    )

    torch.manual_seed(5)
    x = torch.randn(8, 2 * BLOCK, dtype=torch.bfloat16)
    got = quantized_linear(x, w1, quantized=True)
    assert torch.equal(got, _rule_linear(x, w1)), (
        "the store's weight did not reach the rule on the codes and the grid it attached"
    )

    rounded = quantized_linear(x, w1, quantized=True, emulate_reduction=False)
    assert not torch.equal(got, rounded), (
        "the emulated and the rounded routed GEMM coincide on this fixture, so it cannot show that "
        "the store's attachment is doing anything"
    )

    assert torch.equal(expert_forward(x, w1, w2, w3, quantized=True), _rule_forward(x, w1, w2, w3)), (
        "the routed forward is not the three rule linears and a SwiGLU the pair describes"
    )

    control = expert_forward(x, *(_without_pair(w) for w in (w1, w2, w3)), quantized=True)
    assert not torch.equal(expert_forward(x, w1, w2, w3, quantized=True), control), (
        "dropping the attached pair changed nothing, so the routed forward is not running the rule"
    )