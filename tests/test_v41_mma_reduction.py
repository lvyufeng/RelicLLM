"""The reference's tensor-core reduction, pinned against a scalar transcription of the same rule.

`quantized_linear` used to compute `dequant(act) @ W.T` in fp32 -- a correctly rounded product. The
reference does not: its fp8 GEMM is an `mma.sync`, and on a real `wq_a` that moves 2.41% of output
elements onto the other side of a bf16 boundary. Reproducing the reference means reproducing that
rounding, and `relicllm/models/deepseek_v4_1/mma.py` does.

A vectorised transcription of a rule is easy to get subtly wrong and hard to notice: every wrong
version still returns a plausible float. So the rule is written out a second time here, term by term
and index by index, with no tensor ops at all, and the two are compared on real e4m3 code blocks.
The scalar version is the specification; if the fast one and the slow one ever disagree, the fast one
is wrong.

The comparisons are exact (`==`, not `allclose`) and that is sound rather than lucky. Every product
of two e4m3 codes is a multiple of `2**-18` and is at most `448 * 448`; truncating onto any grid
leaves it a multiple of `2**-18`, and the grid is finer than that lattice whenever the anchor falls
below -5. So every partial sum is an exactly representable float64, every summation order gives the
same bits, and a tolerance here would only hide a defect.

Every test is CPU-safe except the pair at the end, under "Through a real capture": the rule is soft
torch with no card and no kernel binding, but the port's default decode path records its body into a
CUDA graph, and whether the emulated linear can be *recorded* is not a claim an eager run can make.
The file skips when torch is absent, the same guard the V4.1 siblings use; the two capture tests skip
without a card.
"""

from __future__ import annotations

import functools
import math
from dataclasses import replace

import pytest

# The suite runs on boxes that may not carry torch; the guard has to come before the module's own
# import so a torch-less host skips this file rather than failing to collect it.
torch = pytest.importorskip("torch")

from relicllm.models.deepseek_v4_1 import mma  # noqa: E402
from relicllm.models.deepseek_v4_1.kernels import (  # noqa: E402
    _reference_reduction_enabled,
    quantized_linear,
)
from relicllm.models.deepseek_v4_1.tp import ShardPlan  # noqa: E402

BLOCK = mma.BLOCK
HALF = mma.HALF
WINDOW = mma.WINDOW


def _scalar_rule(products: list[float], nominals: list[float | None]) -> float:
    """`mma.reduce_block` written out for one output element, with no tensor operations.

    This is the specification, not a re-derivation: two sequential 16-wide halves, each cut onto a
    grid set by `max(F(a)+F(b))` over that half, truncating toward zero, with the running sum joining
    the second half's anchor and a final renormalisation at the end.
    """

    def exp_of(value: float) -> int:
        _, exponent = math.frexp(abs(value))
        return exponent - 1

    def cut(value: float, ref: float | None) -> float:
        # `None` is the module's sentinel: a half with no live product has no anchor, and the grid
        # collapses to a step of one, which for a 2**-18 lattice of products is the identity.
        step = 1.0 if ref is None else 2.0 ** (ref - WINDOW)
        return math.trunc(value / step) * step

    def anchor(values: list[float | None]) -> float | None:
        live = [v for v in values if v is not None]
        return max(live) if live else None

    first = anchor(nominals[:HALF])
    total = sum(cut(p, first) for p in products[:HALF])

    running = None if total == 0.0 else exp_of(total)
    second = [n for n in (anchor(nominals[HALF:]), running) if n is not None]
    window = max(second) if second else None
    total = cut(total, window) + sum(cut(p, window) for p in products[HALF:])

    return total if total == 0.0 else cut(total, exp_of(total))


def _codes(seed: int, count: int) -> torch.Tensor:
    """Random e4m3 code *bytes*, excluding the two NaN encodings.

    Codes rather than values: the rule reads the encoding field, and `0x7F`/`0xFF` are NaN rather
    than the 480 the normal formula would give them.
    """
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randint(0, 256, (count,), generator=generator, dtype=torch.int64)
    raw = torch.where((raw & 0x7F) == 0x7F, raw ^ 0x01, raw)
    return raw.to(torch.uint8)


def _gen(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _nominal_exp_without_the_floor(codes: torch.Tensor) -> torch.Tensor:
    """`mma.nominal_exp` with the `-6` taken out: every field reads ``field - 7``, zeros included.

    This is the mutation and not a second opinion about the rule. A subnormal code's field is 0, so
    without the floor it reads -7 where the module reads -6 -- and the module reads -6 because that
    is the exponent the reference's `mma` carries into the reduction for such an operand, whatever
    the code's own value happens to be.
    """
    field = (mma._bytes(codes).to(torch.int64) >> 3) & 0xF
    return (field - 7).to(torch.float64)


def _expand(codes: torch.Tensor, scale: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """The dense weight a code grid and its scale grid describe, exactly.

    `scale` is either the file's e8m0 bytes or the exponents as a floating grid; both go through
    `_pow2`, because the scale has to be the exact power of two for `decode(code) * scale` to be the
    value the checkpoint dequantized to.
    """
    if scale.dtype == torch.float8_e8m0fnu:
        scale = mma._pow2(mma.e8m0_exponent(scale).to(torch.float64))
    grid = scale.to(torch.float32)
    return mma.decode_e4m3(codes).to(torch.float32) * grid.repeat_interleave(
        block, dim=-1
    ).repeat_interleave(block, dim=0)


def _block_inputs(
    a_raw: torch.Tensor, b_raw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A contraction block as the rule wants it: products and summed nominal exponents, float64.

    Both operands are `[rows, BLOCK]` so each output element gets its own 32-wide block, which is the
    shape the GEMM actually produces. No cross-product pairing: entry `k` of a block is `a[., k]`
    against `b[., k]`, exactly as the contraction reads it.

    A zero product takes the module's sentinel rather than the sum of its operands' exponents. The two
    readings are provably indistinguishable (see `mma.fp8_block_gemm`), but the block below is built
    the way the GEMM builds it, so the scalar rule is being compared against the same object.
    """
    product = mma.decode_e4m3(a_raw) * mma.decode_e4m3(b_raw)
    nominal = mma.nominal_exp(a_raw) + mma.nominal_exp(b_raw)
    nominal = torch.where(product == 0.0, torch.full_like(nominal, mma._SENTINEL), nominal)
    return product, nominal


def test_the_vectorised_rule_is_a_scalar_transcription_of_the_same_rule() -> None:
    """Random e4m3 blocks, one output element at a time, against the slow version, exactly.

    Two operand regimes, because the rule is asymmetric in a way one of them walks straight past. In
    the first both operands are drawn from the same distribution and the two halves are
    interchangeable; in the second the second half's operand is scaled far down or up, so the window
    the first half sets and the running sum that joins the second half's anchor are exercised
    separately -- including the case where the second half is entirely zeros and its anchor is the
    sentinel, which is the branch a rewrite is most likely to lose.
    """
    for seed, second_half_shift in [(0, 0), (1, 0), (2, 4), (3, 4), (4, -4), (5, -4)]:
        a_raw = _codes(seed, 6 * BLOCK).reshape(6, BLOCK)
        b_raw = _codes(seed + 100, 6 * BLOCK).reshape(6, BLOCK)
        if second_half_shift > 0:
            # Zero the second half outright: its anchor is the sentinel, and the block reduces to
            # the first half plus the running sum.
            b_raw[:, HALF:] = 0
        elif second_half_shift < 0:
            # Keep only the sign and the mantissa, clearing the exponent field: every code in the
            # second half becomes a subnormal, so its anchor is the -6 floor rather than a value.
            b_raw[:, HALF:] = _codes(seed + 200, 6 * HALF).reshape(6, HALF) & 0x87

        product, nominal = _block_inputs(a_raw, b_raw)
        got = mma.reduce_block(product, nominal)
        assert tuple(got.shape) == (6,)

        for row in range(6):
            values = [float(v) for v in product[row]]
            exponents = [
                None if float(v) < mma._SENTINEL_GUARD else float(v) for v in nominal[row]
            ]
            want = _scalar_rule(values, exponents)
            assert float(got[row]) == want, (
                f"seed {seed} row {row}: the vectorised rule gave {float(got[row])!r} where the "
                f"scalar transcription gives {want!r}"
            )


def test_the_rule_truncates_where_an_exact_sum_would_not() -> None:
    """The rule is not the correctly rounded sum, and that difference is the point of the module.

    One enormous product and one tiny one in the same half: the tiny one falls below the grid the
    enormous one sets and is cut away entirely, so the block reduces to the large product alone. A
    correctly rounded sum of the same operands carries the small term's bits. That is the whole reason
    this module exists -- scaled up to a full `wq_a` it is the 2.41% of elements where the port and
    the reference disagree.
    """
    a_raw = torch.full((1, BLOCK), 0x7E, dtype=torch.uint8)  # 448.0
    b_raw = torch.zeros((1, BLOCK), dtype=torch.uint8)
    b_raw[0, 0] = 0x7E  # 448.0, so `a[0, 0] * b[0, 0]` is 448 * 448
    product, nominal = _block_inputs(a_raw, b_raw)

    # A second live product 8 octaves down -- `_cut`'s grid for an anchor of 16 is 2**3, so this
    # truncates to zero while the large product survives untouched.
    product[0, 1] = 1.0
    nominal[0, 1] = 0.0
    got = mma.reduce_block(product, nominal)

    assert float(product[0, 0]) == 448.0 * 448.0
    assert float(got[0]) == float(product[0, 0]), (
        f"the small term should be cut away entirely, leaving the large product alone; got "
        f"{float(got[0])!r}, wanted {float(product[0, 0])!r}"
    )
    assert float(product[0].sum()) != float(got[0]), (
        "the exact sum and the rule agree here, so this block does not distinguish them"
    )


def test_the_weight_quantization_round_trips_and_a_weight_off_the_grid_is_refused() -> None:
    """`quantize_blocks` is the reference's quantizer, and it refuses what is not on its grid.

    For a weight the rule is running on, this is the fixture builder rather than the recovery: the
    codes come from the grid the file stored, because a dense weight does not determine them (see
    `test_the_stored_scale_is_what_recovers_the_codes`). What it still checks here is that the
    quantizer is the inverse of the dequantizer on the values, and that it declines rather than
    guesses at a weight whose values are not e4m3 codes times a power of two.
    """
    torch.manual_seed(0)
    weight = torch.randn(64, 64, dtype=torch.float32)
    codes, scale = mma.quantize_blocks(weight)
    dense = _expand(codes, scale)

    recovered, recovered_scale = mma.quantize_blocks(dense, verify=True)
    assert torch.equal(recovered, codes), "the codes did not survive the round trip"
    assert torch.equal(recovered_scale, scale), "the scale did not survive the round trip"
    assert torch.equal(_expand(recovered, recovered_scale), dense), (
        "the recovery is not exact, so the rule would run on exponents the reference never had"
    )

    # A weight that is not on that grid at all: arbitrary floats are not e4m3 codes times 2**k, so
    # the recovery has no right answer and must say so instead of inventing one.
    off_grid = torch.randn(32, 32, dtype=torch.float32) * 1e-3
    with pytest.raises(ValueError, match="does not round-trip"):
        mma.quantize_blocks(off_grid, verify=True)


def test_the_stored_scale_is_what_recovers_the_codes() -> None:
    """A dense weight does not determine its codes, and the file's own grid is what does.

    `224 * 2**E` and `448 * 2**(E-1)` are the same number, so a tile whose largest value is
    `224 * 2**E` -- the population a real tile lands in whenever its amax over the scale falls just
    above a power of two, 4.66% of this checkpoint's fp8 tiles -- is described equally well by two
    scales. Re-deriving the scale from the values picks the lower octave and doubles every code, and
    the re-expansion still matches the dense weight bit for bit, which is why a value-level `verify`
    cannot see it. The reduction rule reads the code's exponent field, so the two are not the same
    GEMM: `nominal_exp` of 224 is 7 and of 448 is 8.

    `codes_from_scale` is handed the grid the file stored, which is the only record of the octave,
    and recovers the codes exactly -- including on the tile that `quantize_blocks` gets wrong.
    """
    # One ambiguous tile (its largest code is 224) beside an ordinary one, in the same weight.
    file_scale = torch.tensor([[-11], [-9]], dtype=torch.int64)
    grid = (file_scale + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    codes = _codes(7, 2 * BLOCK * BLOCK).reshape(2, BLOCK, BLOCK)
    codes[0] &= 0x0F  # keep the first tile's largest code ours, or a random 448 would decide it
    codes[0, 0, 0] = 0x76  # 224.0: the largest e4m3 code that admits two scales
    assert float(mma.decode_e4m3(codes[0, 0, 0].reshape(1))) == 224.0
    codes = codes.reshape(2 * BLOCK, BLOCK).view(torch.float8_e4m3fn)
    dense = _expand(codes, grid)

    recovered, recovered_scale = mma.quantize_blocks(dense, verify=True)
    assert torch.equal(_expand(recovered, recovered_scale), dense), (
        "the recovery does not round-trip on values, so this fixture does not show what it claims"
    )
    assert not torch.equal(recovered, codes), (
        "the ambiguous tile recovered its codes after all, so this fixture no longer exercises the "
        "case the stored grid exists for"
    )
    assert torch.equal(recovered[BLOCK:], codes[BLOCK:]), (
        "the unambiguous tile should have recovered its codes; only the 224 tile is ambiguous"
    )
    recovered_exponent = mma.e8m0_exponent(recovered_scale)
    file_exponent = mma.e8m0_exponent(grid)
    assert int(recovered_exponent[0, 0]) == int(file_exponent[0, 0]) - 1, (
        "the ambiguous tile's recovered scale should be one octave low, with its codes doubled; it "
        "came back as 2**"
        f"{int(recovered_exponent[0, 0])} against the file's 2**{int(file_exponent[0, 0])}"
    )
    assert torch.equal(recovered_exponent[1:], file_exponent[1:]), (
        "the tile that is not ambiguous should have recovered the file's own octave"
    )

    # The stored grid has no such ambiguity, and it is checked rather than trusted: a grid that does
    # not belong to the weight it is attached to is refused instead of quantizing against a
    # neighbour's scales.
    exact = mma.codes_from_scale(dense, grid, BLOCK)
    assert torch.equal(exact, codes), (
        f"{(exact != codes).sum().item()} of {codes.numel()} codes do not match the file's, so the "
        f"stored grid did not recover them"
    )
    with pytest.raises(ValueError, match="belongs to a different weight"):
        mma.codes_from_scale(
            dense,
            (mma.e8m0_exponent(grid) - 6 + 127).to(torch.uint8).view(torch.float8_e8m0fnu),
            BLOCK,
        )
    with pytest.raises(ValueError, match="wants a"):
        mma.codes_from_scale(dense, grid[:1], BLOCK)


def test_a_replaced_grid_does_not_leave_the_grid_it_replaced_codes_behind() -> None:
    """The two attached tensors are a pair, and `attach_scale` is what keeps them one.

    `quantized_linear` runs the rule on `attached_codes` and `attached_scale` together, and the codes
    are a property of the grid rather than of the weight -- a dense weight does not determine them,
    which is the whole reason the file's grid is carried along at all. A second `attach_scale` over a
    weight that already carries codes therefore leaves a state that *runs* and is wrong: the
    exponents the rule reads belong to a weight this one is not, and no caller can tell, because a
    plausible scale and a plausible weight quantize to plausible codes. The alternative state is the
    one the hot path already reports -- a grid whose codes are missing -- so the pair is not a
    convention, it is enforced.
    """
    torch.manual_seed(4)
    raw = torch.randn(2 * BLOCK, 2 * BLOCK, dtype=torch.float32) * 0.1
    codes, scale = mma.quantize_blocks(raw)
    grid = (mma.e8m0_exponent(scale) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    weight = _expand(codes, grid).to(torch.bfloat16)

    # The pair, as `loader.fill` leaves it.
    mma.attach_scale(weight, grid)
    mma.attach_codes(weight, mma.codes_from_scale(weight, grid, BLOCK))
    assert mma.attached_codes(weight) is not None and mma.attached_scale(weight) is not None

    # The same weight under a grid one octave down, which is exactly what a weight of the same shape
    # and a neighbouring tile's grid looks like from here -- the case nothing downstream can see.
    other = (mma.e8m0_exponent(grid) - 1 + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    mma.attach_scale(weight, other)
    assert mma.attached_codes(weight) is None, (
        "attaching a second grid left the first grid's codes on the weight; the rule would read the "
        "old exponents against the new scale and return a number neither GEMM would"
    )
    assert torch.equal(mma.attached_scale(weight).view(torch.uint8), other.view(torch.uint8)), (
        "the grid that was attached is not the one that came back"
    )
    # And the way back is the way in: `loader.fill`'s own order, on the grid that describes this
    # weight, comes out with the file's codes again rather than with the neighbour's.
    mma.attach_scale(weight, grid)
    mma.attach_codes(weight, mma.codes_from_scale(weight, grid, BLOCK))
    assert torch.equal(mma.attached_codes(weight), codes), (
        "the recovery after a re-attach did not come back with the file's own codes"
    )


def test_a_sharded_grid_is_cut_exactly_the_way_its_weight_is() -> None:
    """`ShardPlan.local_scale` cuts the grid the way `local_value` cuts the weight it scales.

    A rank holds a quarter of `wq_b`'s rows, so it needs a quarter of its scale rows, and the cut has
    to land on whole tiles -- `wo_a`'s rows are `groups` blocks of `o_lora_rank` and the indexer's are
    `index_heads` blocks of `index_head_dim`, so a flat band across the whole grid is a *different*
    tensor from the grid of the band. The check below is the one that matters and it is not a shape
    check: the codes recovered from a rank's own dense weight, against its own grid, have to be the
    codes `local_value` cuts out of the file's. A cut that slipped by one tile fails it.
    """
    plan = ShardPlan(
        rank=0, world=4, heads=10, groups=1, index_heads=8,
        heads_global=40, groups_global=5, index_heads_global=32, row_split=False,
        head_dim=32, o_lora_rank=512, index_head_dim=128, moe_inter_dim=1024,
        reduce=None, gather=None,
    )
    for index, (name, rows, cols) in enumerate(
        [
            ("layers.3.attn.wq_b.weight", 40 * 32, 512),
            ("layers.3.attn.wo_b.weight", 512, 5 * 512),
            ("layers.3.attn.indexer.wq_b.weight", 32 * 128, 512),
            ("layers.3.attn.wo_a.weight", 5 * 512, 8 * 32),
            ("layers.3.ffn.shared_experts.w1.weight", 1024, 512),
            ("layers.3.ffn.shared_experts.w2.weight", 512, 1024),
        ]
    ):
        grid = (torch.randint(-14, -9, (rows // BLOCK, cols // BLOCK), generator=_gen(index))
                + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
        codes = _codes(index + 50, rows * cols).reshape(rows, cols)
        dense = _expand(codes, grid)
        for rank in range(plan.world):
            shard = replace(plan, rank=rank)
            local_grid = shard.local_scale(name, grid, BLOCK)
            local_codes = shard.local_value(name, codes.view(torch.uint8))
            got = mma.codes_from_scale(shard.local_value(name, dense), local_grid, BLOCK)
            assert torch.equal(got, local_codes.view(torch.float8_e4m3fn)), (
                f"{name} rank {rank}: the local grid does not describe the local weight, so that "
                f"rank's reduction would run on another rank's scales"
            )


def test_the_emulation_is_on_by_default_and_switches_off_cleanly(monkeypatch) -> None:
    """`quantized_linear` runs the rule by default, and every way of turning it off restores the GEMM.

    This is the guard on the whole change. With the emulation off the function has to be bit-for-bit
    the correctly rounded GEMM the port has always run, so a call site that is not chasing the
    reference's rounding -- the fp4 experts, and every test that pins the *activation quantization*
    rather than the reduction -- sees nothing moves. Both switches are checked, because they are
    different claims: `emulate_reduction=False` is what `expert_forward` passes, and the environment
    variable is what an operator reaches for without editing a call site.

    A weight carries its codes as an attribute (`mma.attach_codes`), recovered from the grid the file
    stored beside it by `loader.fill` and not here -- the recovery ends in a device sync, and the
    serving path captures its decode body into a CUDA graph. So the two ways of arriving without
    codes are checked too: the codes absent (which is the routed experts' case, and a bf16 weight's,
    and every tensor a caller built by hand), and a grid whose recovery was *refused* at load because
    it does not describe the weight -- `test_the_stored_scale_is_what_recovers_the_codes` pins that
    refusal where it now happens. Both have to fall back to the rounded product rather than run the
    rule on exponents the reference never had, which is the failure mode that would otherwise be
    silent: a wrong scale still produces plausible numbers.
    """
    torch.manual_seed(1)
    x = torch.randn(8, 64, dtype=torch.bfloat16)

    # A weight on the reference's own grid: quantize it, then expand the codes back out. The expand is
    # exact in bf16 -- e4m3 carries three mantissa bits and a scale is a power of two -- so this is a
    # weight a checkpoint could really hold, and not random floats the emulation would refuse.
    from relicllm.models.deepseek_v4_1.kernels import block_fp8_act_quant

    raw = torch.randn(64, 64, dtype=torch.float32) * 0.1
    codes, scale = mma.quantize_blocks(raw)
    grid = (mma.e8m0_exponent(scale) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    weight = _expand(codes, grid).to(torch.bfloat16)
    # Grid and codes, exactly as `loader.fill` leaves a weight: the grid as the file stored it, the
    # codes recovered from it once. Nothing on the hot path recovers anything.
    mma.attach_scale(weight, grid)
    mma.attach_codes(weight, mma.codes_from_scale(weight, grid, BLOCK))

    quant, act_scale = block_fp8_act_quant(x)
    rounded = quant.to(torch.float32) * act_scale.repeat_interleave(BLOCK, dim=-1)
    rounded = torch.matmul(rounded, weight.to(torch.float32).transpose(0, 1)).to(x.dtype)

    assert torch.equal(
        quantized_linear(x, weight, quantized=True, emulate_reduction=False), rounded
    ), "passing `emulate_reduction=False` did not restore the correctly rounded GEMM"

    monkeypatch.setenv("POCKETLLM_V41_MMA_EMULATE", "0")
    assert _reference_reduction_enabled() is False
    assert torch.equal(
        quantized_linear(x, weight, quantized=True), rounded
    ), "POCKETLLM_V41_MMA_EMULATE=0 did not restore the correctly rounded GEMM"

    # With the switch back on, the default runs the rule -- and on this much data the rule and the
    # correctly rounded product do not coincide, which is the claim the whole module rests on.
    monkeypatch.delenv("POCKETLLM_V41_MMA_EMULATE")
    assert _reference_reduction_enabled() is True
    emulated = quantized_linear(x, weight, quantized=True)
    assert emulated.shape == rounded.shape and emulated.dtype == rounded.dtype
    differs = int((emulated != rounded).sum())
    assert differs, (
        "the emulation returned the correctly rounded product bit for bit; either it is not running "
        "or the rule has stopped doing anything"
    )
    # And it is the same GEMM: the two differ by the rule's truncation and not by a different
    # product. A path that ran on exponents from another weight would pass `differs` too.
    assert torch.allclose(emulated.float(), rounded.float(), rtol=2e-2, atol=1e-2), (
        f"the emulation and the rounded product are not the same GEMM; the largest gap is "
        f"{float((emulated.float() - rounded.float()).abs().max()):g}"
    )

    # A weight carrying the grid but no codes declines, and *that it declines is the design*: the
    # recovery is a `torch.equal` over the whole weight (a device sync) and this call can be inside a
    # captured decode body. The grid being present is not enough for the hot path to act on; this is
    # the test that fails the day someone moves the recovery back in here for convenience.
    grid_only = _expand(codes, grid).to(torch.bfloat16)
    mma.attach_scale(grid_only, grid)  # the state `loader.fill` leaves a refused weight in
    assert torch.equal(grid_only, weight), "the fixture without codes is not the same weight"
    assert torch.equal(
        quantized_linear(x, grid_only, quantized=True), rounded
    ), (
        "a weight with a grid and no recovered codes must run the correctly rounded GEMM: "
        "`quantized_linear` recovers nothing, and a grid is not evidence that it can"
    )

    # And a weight with nothing attached at all -- a routed expert, a bf16 tensor, anything a caller
    # built rather than loaded.
    plain = _expand(codes, grid).to(torch.bfloat16)
    assert torch.equal(
        quantized_linear(x, plain, quantized=True), rounded
    ), "a weight with no attached codes should run the correctly rounded GEMM"


def test_the_wired_path_runs_the_rule_on_exactly_the_codes_it_is_given(monkeypatch) -> None:
    """`quantized_linear` with codes attached *is* `fp8_block_gemm` on those codes, bit for bit.

    The seam between the loader and the module is an attribute, and an attribute can be attached and
    never read: the failure looks like a working run whose linear is quietly the correctly rounded
    product, which is the difference this whole module exists to remove. So the wiring is checked
    from the outside, against the same call made directly -- the weight's values do not enter it.

    The codes are handed over as the file's own, not as something re-derived from the weight, and the
    two are not always the same: on a tile whose largest code is 224 they differ by an octave and
    their values agree exactly, which is why `codes_from_scale` exists and why this test would notice
    a `quantized_linear` that went back to guessing.

    The second half pins the *absence* of two calls rather than their effect. Both the code recovery
    and the power-of-two check end in a host `bool` over device data -- a synchronisation -- and the
    port's default decode path captures the decode body into a CUDA graph, where a synchronising call
    is an error. A test that only compares numbers cannot see that: on a host run the sync is
    invisible, and the failure it causes appears only on a card, only with `decode_graphs=True`, and
    at capture time, as a serving path that cannot start.
    """
    from relicllm.models.deepseek_v4_1.kernels import block_fp8_act_quant

    torch.manual_seed(4)
    rows, cols = 64, 2 * BLOCK
    codes, scale = mma.quantize_blocks(torch.randn(rows, cols, dtype=torch.float32))
    grid = (mma.e8m0_exponent(scale) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    weight = _expand(codes, grid).to(torch.bfloat16)
    mma.attach_scale(weight, grid)
    mma.attach_codes(weight, codes)

    x = torch.randn(8, cols, dtype=torch.bfloat16)
    got = quantized_linear(x, weight, quantized=True)

    act, act_scale = block_fp8_act_quant(x)
    want = mma.fp8_block_gemm(act, act_scale, codes, grid, BLOCK).to(x.dtype)
    assert got.shape == want.shape
    assert torch.equal(got, want), (
        f"{int((got != want).sum())} of {got.numel()} elements differ from the rule called directly "
        f"on the codes that were attached"
    )

    # Now the same call again, with the two synchronising entry points booby-trapped: the recovery
    # must not be reached at all, and the exponent read must not verify.
    seen: list[bool] = []
    real_exponent = mma.e8m0_exponent

    def watched_exponent(tensor: torch.Tensor, *, verify: bool = True) -> torch.Tensor:
        seen.append(verify)
        return real_exponent(tensor, verify=verify)

    def refuse(*args, **kwargs):
        raise AssertionError(
            "`quantized_linear` called `mma.codes_from_scale`: the recovery is a device sync and "
            "must happen once, at load (`loader.fill`), not inside a forward pass"
        )

    monkeypatch.setattr(mma, "codes_from_scale", refuse)
    monkeypatch.setattr(mma, "e8m0_exponent", watched_exponent)
    again = quantized_linear(x, weight, quantized=True)

    assert torch.equal(again, want), "the booby-trapped call returned a different GEMM"
    assert seen, "the GEMM never read an e8m0 exponent, so this test is not watching the right calls"
    assert not any(seen), (
        f"{sum(seen)} of {len(seen)} exponent reads were made with `verify=True`, whose power-of-two "
        f"check is a device synchronisation inside the captured decode body"
    )


def test_the_window_and_the_floor_are_the_constants_the_rule_was_measured_with(monkeypatch) -> None:
    """Both measured constants move a GEMM on this payload, and neither is decoration.

    The rule's *structure* is what a scalar transcription pins -- two halves, a per-element anchor,
    the running sum joining the second half, the final renormalisation. Its two numbers are a
    different kind of claim: thirteen binary places below the anchor, and a subnormal operand
    contributing the -6 its encoding field implies rather than its own -7..-9. Both came off the
    reference's own trace, and neither was pinned here -- with the floor dropped, or the window set
    to 12, every other test in this file still passes, because their fixtures are either all at
    ordinary magnitudes (nothing subnormal for the floor to move) or coarse enough that a grid one
    octave wider cuts exactly the same terms.

    This is the counter-fixture: a payload that is substantially subnormal, so the floor has
    something to move, at magnitudes where the window has something to cut.
    """
    gen = _gen(29)
    rows, out_rows, cols = 4, 32, 2 * BLOCK

    def scales(shape: tuple[int, ...]) -> torch.Tensor:
        return (
            (torch.randint(115, 130, shape, generator=gen)).to(torch.uint8).view(torch.float8_e8m0fnu)
        )

    # Activation codes drawn from 0..31, so a quarter of them carry exponent field 0 -- subnormals,
    # whose true exponents are -9..-6 while their field says -6 -- beside codes at ordinary sizes.
    a_raw = torch.randint(0, 32, (rows, cols), generator=gen).to(torch.uint8)
    b_raw = torch.randint(0, 256, (out_rows, cols), generator=gen).to(torch.uint8)
    b_raw = torch.where((b_raw & 0x7F) == 0x7F, b_raw ^ 0x01, b_raw)
    a_codes, b_codes = a_raw.view(torch.float8_e4m3fn), b_raw.view(torch.float8_e4m3fn)
    a_scale, b_scale = scales((rows, cols // BLOCK)), scales((out_rows // BLOCK, cols // BLOCK))

    def gemm() -> torch.Tensor:
        return mma.fp8_block_gemm(a_codes, a_scale, b_codes, b_scale, BLOCK)

    rule = gemm()

    # `_cut`'s `off` is a default argument and was bound when the function was defined, so setting
    # `mma.WINDOW` would change nothing at all -- rebinding `_cut` is the mutation, and it is the
    # same one a reader would make: `2 ** (anchor - 12)` instead of `2 ** (anchor - 13)`.
    with monkeypatch.context() as narrow:
        narrow.setattr(mma, "_cut", functools.partial(mma._cut, off=mma.WINDOW - 1))
        assert not torch.equal(gemm(), rule), (
            "a window one octave wider cut exactly the same terms, so this fixture cannot tell the "
            "rule's thirteen places from twelve"
        )

    with monkeypatch.context() as floorless:
        floorless.setattr(mma, "nominal_exp", _nominal_exp_without_the_floor)
        assert not torch.equal(gemm(), rule), (
            "dropping the -6 floor changed nothing on a payload whose activations are a quarter "
            "subnormal, so this fixture cannot tell the encoding field from the value's exponent"
        )


def test_every_power_of_two_the_rule_builds_is_exact() -> None:
    """`_pow2` is exact on both devices across the whole exponent range, against Python's own `ldexp`.

    This is not a style preference. A power of two here is a *grid*: `_cut` divides by it and
    multiplies back, so a grid a hair below the intended value moves the truncation boundary and
    silently drops terms that should have survived. The obvious spellings are all wrong on the card
    and right on the host, which is the worst possible combination -- a host-only test passes and the
    acceptance run is off by a lattice step. `torch.pow(2.0, e)` is `exp2(e * log2(2))`, and
    `torch.ldexp(1.0, e)` is one ulp low for 16 of the 254 e4m3 codes. Both were measured here, on
    this card, before this test was written.
    """
    exponents = torch.arange(-1074, 1024, dtype=torch.float64)
    truth = torch.tensor(
        [math.ldexp(1.0, int(e)) for e in exponents], dtype=torch.float64
    )

    got = mma._pow2(exponents)
    assert torch.equal(got, truth), (
        f"_pow2 is inexact on the host at "
        f"{int((got != truth).sum())} of {exponents.numel()} exponents"
    )

    if torch.cuda.is_available():
        got_gpu = mma._pow2(exponents.cuda()).cpu()
        assert torch.equal(got_gpu, truth), (
            f"_pow2 is inexact on the card at {int((got_gpu != truth).sum())} of "
            f"{exponents.numel()} exponents; the host agrees, which is exactly the failure mode "
            f"this test exists for"
        )


def test_the_gemm_follows_its_operands_device_whatever_its_scales_arrive_on() -> None:
    """The accumulator lives on the operands' device, and the scales find their way to it.

    The two scale tensors come from different places and are not required to share a device with the
    codes: activation scales from `block_fp8_act_quant`, weight scales from the loader. An
    accumulator allocated on the default device would be a CPU tensor added into from CUDA operands
    -- a loud failure, but one that only happens on the card and only on the path that matters. The
    case below hands the scales in on the host while the codes and their values are on the device,
    which is the arrangement a caller that quantized on the host would produce.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    rows, cols, out_rows = 3, 2 * BLOCK, 2 * BLOCK
    a_raw = _codes(11, rows * cols).reshape(rows, cols).to(device)
    b_raw = _codes(12, out_rows * cols).reshape(out_rows, cols).to(device)
    a_scale = torch.ones(rows, cols // BLOCK, dtype=torch.float32)  # deliberately left on the host
    b_scale = torch.ones(out_rows // BLOCK, cols // BLOCK, dtype=torch.float32)

    got = mma.fp8_block_gemm(a_raw, a_scale, b_raw, b_scale, BLOCK)
    assert got.device.type == device.type
    assert tuple(got.shape) == (rows, out_rows)

    # And the device does not change the arithmetic: the same call on the host agrees bit for bit.
    if device.type == "cuda":
        host = mma.fp8_block_gemm(
            a_raw.cpu(), a_scale, b_raw.cpu(), b_scale, BLOCK
        )
        assert torch.equal(got.cpu(), host), (
            "the GEMM's result depends on which device it ran on, so one of the two paths is not "
            "the reference's arithmetic"
        )


def test_the_code_recovery_reads_its_grid_where_the_weight_lives() -> None:
    """A weight on the card with its grid still on the host recovers its codes, rather than raising.

    That pairing is not hypothetical. The tree is built on its device (`Backbone(..., device=...)`)
    and filled in place (`parameter.copy_(value)`), while the grid comes out of the checkpoint file
    as a view of the shard mapping -- host memory. `mma.attach_scale` is what moves the grid across,
    so the loader's own path is device-and-device; but a caller that hands `codes_from_scale` a grid
    directly, as this test does and as the indexer's row-split band does, arrives at the mixed case,
    and a host step requested from a card tensor is a hard error at the next GEMM ("Expected all
    tensors to be on the same device"). `fp8_block_gemm` moves its scales onto its operands for the
    same reason, one step later in the same call.
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    rows, cols = 2 * BLOCK, 2 * BLOCK
    grid = (torch.randint(-14, -9, (2, 2), generator=_gen(3)) + 127).to(torch.uint8)
    grid = grid.view(torch.float8_e8m0fnu)
    codes = _codes(3, rows * cols).reshape(rows, cols).view(torch.float8_e4m3fn)
    dense = _expand(codes, grid)

    assert torch.equal(mma.codes_from_scale(dense, grid, BLOCK), codes), (
        "the grid does not recover the codes it was built from on the host, so this fixture says "
        "nothing about the card"
    )
    moved = mma.codes_from_scale(dense.to(device), grid, BLOCK)
    assert moved.device.type == device.type and torch.equal(moved.cpu(), codes), (
        "the codes recovered on the card differ from the host's for the same weight and grid, so "
        "the grid is being read where it was attached rather than where the weight lives"
    )


def test_the_decode_reproduces_the_encoding_it_claims_to() -> None:
    """The hand-written e4m3 decoder, against torch's own cast where torch can do one.

    The decoder exists because `float8_e4m3fn` has no dtype promotion on every backend and this has
    to run on sm_75, where no fp8 arithmetic exists at all. A decoder is exactly the kind of code
    that is right for every value anyone tries and wrong for the one they do not, so it is swept
    across the whole encoding space rather than sampled.
    """
    raw = torch.arange(256, dtype=torch.int64)
    raw = torch.where((raw & 0x7F) == 0x7F, raw ^ 0x01, raw)  # skip the two NaNs
    codes = raw.to(torch.uint8).view(torch.float8_e4m3fn)
    got = mma.decode_e4m3(codes)
    want = codes.to(torch.float64)
    assert torch.equal(got, want), (
        f"the decoder disagrees with torch's cast on "
        f"{int((got != want).sum())} of 254 codes"
    )

    # The two NaN encodings, which the sweep above skips: they must decode to NaN rather than to the
    # 480 the normal branch's formula would produce for them.
    nan_codes = torch.tensor([0x7F, 0xFF], dtype=torch.uint8).view(torch.float8_e4m3fn)
    assert bool(torch.isnan(mma.decode_e4m3(nan_codes)).all())


# --------------------------------------------------------------------------- #
# Through a real capture
# --------------------------------------------------------------------------- #

# The only tests in this file that need a card, and there is no host fallback for either: "this body
# can be recorded" is a claim a capture makes and an eager run cannot. Same marker as the V4.1
# siblings.
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="a capture needs a card")


def _capture(body) -> torch.cuda.CUDAGraph:
    """Record `body` after warming it on a side stream, in the shape `graphs._Step` records in."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(2):
            body()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    torch.cuda.synchronize()
    return graph


def _loadable_weight(device: torch.device) -> torch.Tensor:
    """A weight in the state `loader.fill` leaves one in: the file's grid, and its recovered codes.

    Built the way the sibling tests build theirs -- quantize random values onto the reference's own
    grid, then expand the codes back out, which is exact in bf16 -- and the codes are recovered
    through `mma.codes_from_scale` rather than handed over, in the loader's own order. Recovery is
    one of the two calls these tests are about, so a fixture that skipped it would be testing a
    weight no load produces.
    """
    dim = 2 * BLOCK
    raw = torch.randn(dim, dim, dtype=torch.float32) * 0.1
    codes, scale = mma.quantize_blocks(raw)
    grid = (mma.e8m0_exponent(scale) + 127).to(torch.uint8).view(torch.float8_e8m0fnu)
    weight = _expand(codes, grid).to(torch.bfloat16).to(device)

    mma.attach_scale(weight, grid)
    recovered = mma.codes_from_scale(weight, grid, BLOCK)
    assert torch.equal(recovered.cpu(), codes), (
        "the fixture's grid does not recover the codes it was built from, so this is not a weight a "
        "load could have produced and neither test below would mean what it says"
    )
    mma.attach_codes(weight, recovered)
    return weight


@CUDA
def test_the_emulated_linear_records_into_a_graph_and_replays_on_its_next_input() -> None:
    """The port's own decode environment: the emulated linear has to *record*.

    `decode_graphs=True` is the default serving path -- `graphs.py` records the decode step once and
    replays it -- and a capture forbids anything in the recorded body from reading a device value back
    to the host. `quantized_linear` broke that rule twice: `mma.codes_from_scale` on the weight it was
    handed, and `e8m0_exponent`'s power-of-two check. Both end in a host `bool` over device data, i.e.
    a synchronisation, and a synchronisation inside a capture does not raise where it happens -- it
    invalidates the capture, which is how a body can record cleanly on the host and die on the card.

    The sibling test above pins the weaker half of that: its booby-trapped entry points say the call
    no longer *reaches* the recovery. That is a statement about the source, and a host run cannot make
    the other one, because the sync it would have to catch is invisible there. Only a real capture
    separates the fix from the bug it fixes, which is why this needs a card.

    The replay is held against `mma.fp8_block_gemm` called directly on the attached codes -- bit for
    bit, not a tolerance -- and replayed again on a second input written into the same buffer. Both
    halves are load-bearing. A weight that arrives without its codes takes the rounded fallback, which
    is sync-free and records perfectly, so `differs` is what tells a reader the *rule* was in the body
    rather than nothing. And a recording holds addresses, so a body that had read its input off the
    host would replay the first input's answer forever; the second replay is the only thing here that
    would notice.
    """
    from relicllm.models.deepseek_v4_1.kernels import block_fp8_act_quant

    assert _reference_reduction_enabled(), (
        "the emulation is switched off (POCKETLLM_V41_MMA_EMULATE), so `quantized_linear` would take "
        "the rounded path here and this test would record nothing worth recording"
    )
    device = torch.device("cuda")
    width = 2 * BLOCK

    torch.manual_seed(1)
    first = torch.randn(8, width, dtype=torch.bfloat16).to(device)
    second = torch.randn(8, width, dtype=torch.bfloat16).to(device)
    weight = _loadable_weight(device)

    # The input buffer and the output the body writes into, both allocated before the recording: a
    # capture holds the addresses it was recorded against, so the input cannot be created inside it.
    move = torch.empty_like(first)
    out = torch.empty_like(first)

    def body() -> None:
        out.copy_(quantized_linear(move, weight, quantized=True))

    def rule(x: torch.Tensor) -> torch.Tensor:
        act_codes, act_scale = block_fp8_act_quant(x)
        got = mma.fp8_block_gemm(
            act_codes, act_scale, mma.attached_codes(weight), mma.attached_scale(weight), BLOCK
        )
        return got.to(x.dtype)

    move.copy_(first)
    graph = _capture(body)

    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, rule(first)), (
        f"{int((out != rule(first)).sum())} of {out.numel()} elements came out of the replay "
        f"differently from the rule run directly on the same input"
    )

    # Nothing above can fail if the emulation quietly declined: the rounded fallback also records, also
    # replays and also equals a call it would be compared against. The rule and the correctly rounded
    # product do not coincide on this much data, so this is what says the recorded body ran the rule.
    quant, quant_scale = block_fp8_act_quant(first)
    rounded = (
        (quant.to(torch.float32) * quant_scale.repeat_interleave(BLOCK, dim=-1))
        @ weight.to(torch.float32).t()
    ).to(first.dtype)
    assert not torch.equal(out, rounded), (
        "the replay is the correctly rounded product, so the recorded body took the fallback path "
        "and this test is not recording the rule at all"
    )

    move.copy_(second)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, rule(second)), (
        "the replay did not follow the buffer it was recorded against, so the body's input was read "
        "while recording rather than left to the replay"
    )


@CUDA
def test_the_recovery_that_moved_to_load_time_cannot_be_recorded_at_all() -> None:
    """The negative control: the call that moved out of the captured body is refused by a capture.

    Without it, the test above says "the emulated linear records" and a reader has to take the reason
    on faith -- that a `torch.equal` over device data is what a capture rejects, and not something
    incidental about the recovery. Here the same call is recorded twice. With its check off it is a
    loop of kernels and records; with the check on it does not. So the refusal is the host read, and
    the port recovers its codes in `loader.fill` for a measured reason rather than an asserted one.

    Off a capture the check is ordinary and passes, which is also here: it is what makes the refusal
    about the recording rather than about the operands. The refusal is reported by the capture
    machinery -- in the case `tests/test_xing4_0_decode_pos.py` documents, one step later at
    `capture_end`, as `cudaErrorStreamCaptureInvalidated` -- and the context stays usable, which is
    why this can share a process with the tests that do record. The two records are ordered so that
    the refused one comes last.
    """
    device = torch.device("cuda")
    weight = _loadable_weight(device)
    grid = mma.attached_scale(weight)

    assert torch.equal(
        mma.codes_from_scale(weight, grid, BLOCK).cpu(), mma.attached_codes(weight).cpu()
    ), "the recovery disagrees with the codes the loader would have attached, off a capture"

    _capture(lambda: mma.codes_from_scale(weight, grid, BLOCK, verify=False))

    with pytest.raises(RuntimeError, match="[Cc]aptur"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            mma.codes_from_scale(weight, grid, BLOCK)
    torch.cuda.synchronize()