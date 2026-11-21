"""The tile-major packed-expert store, checked on the CPU oracle path without a card.

`relicllm/models/deepseek_v4_1/ascend_tile_major_experts.py` keeps a rank's experts resident in HBM
in the checkpoint's **original packed encoding**, exactly as the row-major packed provider does, but
with the bytes permuted into sixteen-row tiles so a decode step's slice is one contiguous run. These
tests are about the store's half of that contract:

* the resident bytes are the file's bytes, only reordered -- the permutation is a bijection and
  inverting it recovers the shard's bytes exactly;
* the permutation is the operator's, not merely one this file happens to like: it is compared byte
  for byte against relic-core's own `pack_tile_major` on the same inputs;
* nothing is allocated before the checkpoint has been validated, and what is allocated is six
  ``uint8`` banks a layer -- no INT8-times-two bank, no dense BF16 bank, no second copy of the model;
* the arithmetic around the projections is the row-major provider's, unchanged, because this class
  inherits it: both stores are compared at **FP32 bits** for a decode step and for a multi-row
  prefill on the same fixture.

Three oracles, and none of them is the store under test:

* the frozen row-major provider `AscendPackedRoutedExperts` on the same fixture at the same rows --
  its operands are the same frozen byte mapping, so a wrong nibble, a halved grid where the original
  was wanted, or a transposed matrix makes the two disagree;
* relic-core's own tile operator `fp4_packed_tile_major_decode_gemv`, called through its CPU
  reference path -- so the M=1 case exercises the **real** operator end to end, not a stand-in;
* relic-core's `pack_tile_major`/`unpack_tile_major`, for the layout cross-check.

The fixture is `FakePackedCheckpoint` from `tests/test_v41_ascend_packed_experts.py` -- a fake
checkpoint exposing the four `V41Checkpoint` methods this provider reads, over hand-built bytes. It is
imported rather than copied so the two providers are provably tested against one checkpoint.

Nothing here runs on hardware, allocates HBM, or makes a performance claim. The bank is a hypothesis
until a whole-model bit-exact run says otherwise, and the tests say so where it matters.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("relic_core.kernels.block_gemm")
pytest.importorskip("relic_core.kernels.block_fp8_quant")
pytest.importorskip("relic_core.kernels.unpack")

from relic_core.kernels import tile_major_packed_gemv as core_tile  # noqa: E402
from relic_core.kernels.block_gemm import fp4_int8_block_gemm  # noqa: E402
from relic_core.kernels.unpack import fp4_int8_scale, unpack_fp4_to_int8  # noqa: E402

from relicllm.models.deepseek_v4_1 import ascend_tile_major_experts as tile_module  # noqa: E402
from relicllm.models.deepseek_v4_1.ascend_packed_experts import (  # noqa: E402
    AscendPackedRoutedExperts,
    PackedDecodeUnavailable,
    _allocate,
    packed_expert_bytes,
)
from relicllm.models.deepseek_v4_1.ascend_tile_major_experts import (  # noqa: E402
    AscendTileMajorRoutedExperts,
    TILE_ROWS,
    TileDecodeUnavailable,
    TileMatrix,
    pack_code_tiles,
    pack_scale_tiles,
    tile_matrix_bytes,
    unpack_code_tiles,
    unpack_scale_tiles,
)

from test_v41_ascend_packed_experts import FakePackedCheckpoint  # noqa: E402

BLOCK = 32
DIM = 64
INTER = 96
TOPK = 2
N_EXPERTS = 4
DTYPE = torch.bfloat16
SWIGLU_LIMIT = 10.0


# --------------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------------


def _reference_packed_gemv(a_codes, a_scale, b_packed, b_e8m0, *, backend="torch"):
    """The row-major provider's CPU stand-in, spelled here as that provider's tests spell it."""
    values = unpack_fp4_to_int8(b_packed.contiguous(), backend="torch")
    half_scale = fp4_int8_scale(b_e8m0.contiguous(), backend="torch")
    return fp4_int8_block_gemm(
        a_codes.contiguous(), a_scale.contiguous(), values, half_scale, backend="torch"
    )


def _packed_store(checkpoint, *, rank=0, world=1, preload=True, reserve_bytes=0):
    """The frozen row-major provider: the oracle this store's numbers have to match."""
    return AscendPackedRoutedExperts(
        checkpoint,
        0,
        n_experts=checkpoint.experts,
        dim=checkpoint.dim,
        inter_dim=checkpoint.inter,
        topk=TOPK,
        swiglu_limit=SWIGLU_LIMIT,
        device="cpu",
        dtype=DTYPE,
        rank=rank,
        world=world,
        backend="torch",
        budget_bytes=None,
        reserve_bytes=reserve_bytes,
        preload=preload,
        packed_gemv=_reference_packed_gemv,
    )


def _tile_store(checkpoint, *, rank=0, world=1, preload=True, reserve_bytes=0, tile_gemv=None,
                backend="torch"):
    """The store under test. `tile_gemv=None` resolves relic-core's own tile operator."""
    return AscendTileMajorRoutedExperts(
        checkpoint,
        0,
        n_experts=checkpoint.experts,
        dim=checkpoint.dim,
        inter_dim=checkpoint.inter,
        topk=TOPK,
        swiglu_limit=SWIGLU_LIMIT,
        device="cpu",
        dtype=DTYPE,
        rank=rank,
        world=world,
        backend=backend,
        reserve_bytes=reserve_bytes,
        preload=preload,
        tile_gemv=tile_gemv,
    )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to(torch.float32).contiguous().view(torch.int32)


def _bits_equal(first: torch.Tensor, second: torch.Tensor) -> bool:
    return torch.equal(_bits(first), _bits(second))


def _rows(store, count: int, seed: int = 11):
    """A valid `(x, weights, indices)` call: `count` rows, top-k routing over the layer's experts."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn((count, store.dim), generator=generator).to(DTYPE)
    values = torch.randn((count, store.topk), generator=generator)
    weights = values.softmax(dim=-1).to(torch.float32)
    pool = [expert for expert in range(store.n_experts) if expert in set(store.owned)]
    pool = pool or [0]
    indices = torch.tensor(
        [[pool[(row * TOPK + slot) % len(pool)] for slot in range(TOPK)] for row in range(count)],
        dtype=torch.int64,
    )
    return x, weights, indices


# --------------------------------------------------------------------------------------------
# The geometry, and the padding that is never paid
# --------------------------------------------------------------------------------------------


def test_the_released_geometry_needs_no_padding_and_pays_no_extra_byte():
    """Both extents of both projections are multiples of 32, hence of the 16-row tile."""
    # w1/w3 are [inter, dim]; w2 is [dim, inter]. Each extent is a K of some projection, so each is a
    # multiple of the 32-wide block -- and a multiple of 32 is a multiple of 16.
    for inter, dim in ((2304, 5120), (96, 64)):
        for rows, cols in ((inter, dim), (dim, inter), (inter, dim)):
            assert rows % TILE_ROWS == 0
            assert tile_matrix_bytes(rows, cols) == rows * (cols // 2) + rows * (cols // BLOCK)
    assert tile_matrix_bytes(2304, 5120) == 2304 * 2560 + 2304 * 160
    assert tile_matrix_bytes(5120, 2304) == 5120 * 1152 + 5120 * 72
    # One released expert is the same 18,800,640 bytes in either layout.
    assert sum(
        tile_matrix_bytes(*shape) for shape in ((2304, 5120), (5120, 2304), (2304, 5120))
    ) == packed_expert_bytes(5120, 2304)


def test_a_geometry_that_needs_a_padded_tail_is_charged_for_it_and_trimmed(monkeypatch):
    """Padding is unreachable from a validated checkpoint, so the padded path is exercised directly."""
    rows, cols = 24, 64  # 24 is not a multiple of 16: two padded tiles, 8 masked rows
    assert tile_module._padded_rows(rows) == 32
    padded_bytes = tile_module.tile_matrix_bytes(rows, cols)
    assert padded_bytes == (32 * (cols // 2)) + (32 * (cols // BLOCK))
    assert padded_bytes > rows * (cols // 2) + rows * (cols // BLOCK)

    codes = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
    scales = torch.randint(120, 136, (rows, cols // BLOCK), dtype=torch.uint8)
    tiles = pack_code_tiles(codes, rows, cols)
    scale_tiles = pack_scale_tiles(scales, rows, cols)
    assert tuple(tiles.shape) == (2, cols // BLOCK, 16, 16)
    assert tuple(scale_tiles.shape) == (2, cols // BLOCK, 16)
    # A padded bank is trimmed back to the logical rows exactly.
    assert torch.equal(unpack_code_tiles(tiles, rows, cols), codes)
    assert torch.equal(unpack_scale_tiles(scale_tiles, rows, cols), scales)
    # The masked tail is zero codes and the identity scale, as relic-core's own pack writes it.
    assert int(tiles[1, :, 8:, :].abs().sum()) == 0
    assert bool((scale_tiles[1, :, 8:] == 127).all())

    store = _tile_store(FakePackedCheckpoint(), preload=False)
    store._tile_shapes["w1"] = (rows, cols)
    store._tile_gemv_resolved = lambda *a, **k: None
    assert store._logical_rows(torch.empty((2, cols // BLOCK, 16, 16), dtype=torch.uint8)) == rows
    store._tile_shapes["w1"] = (32, cols)
    assert store._logical_rows(torch.empty((2, cols // BLOCK, 16, 16), dtype=torch.uint8)) is None


# --------------------------------------------------------------------------------------------
# The permutation: exact, and the operator's
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(96, 64), (64, 96), (2304, 5120), (5120, 2304), (24, 64)])
def test_pack_then_unpack_recovers_the_shard_bytes_exactly(shape):
    rows, cols = shape
    generator = torch.Generator().manual_seed(shape[0] * 31 + shape[1])
    codes = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator)
    scales = torch.randint(0, 256, (rows, cols // BLOCK), dtype=torch.uint8, generator=generator)
    tiles = pack_code_tiles(codes, rows, cols)
    scale_tiles = pack_scale_tiles(scales, rows, cols)
    assert tiles.is_contiguous() and scale_tiles.is_contiguous()
    assert torch.equal(unpack_code_tiles(tiles, rows, cols), codes), "the permutation is not injective"
    assert torch.equal(unpack_scale_tiles(scale_tiles, rows, cols), scales)
    # A permutation and not a transform: the bank holds the same byte *multiset* the file does plus
    # the zero tail, and the same is true of the scale bytes. A cast or an arithmetic step would break
    # this even where the round trip happened to survive.
    padded = tile_module._padded_rows(rows)
    assert tiles.numel() == padded * (cols // 2)
    assert torch.equal(tiles.flatten().sort().values, torch.cat(
        [codes.flatten(), torch.zeros((padded - rows) * (cols // 2), dtype=torch.uint8)]
    ).sort().values)
    assert scale_tiles.numel() == padded * (cols // BLOCK)
    expected_scales = scales.flatten()
    if padded != rows:
        expected_scales = torch.cat(
            [expected_scales, torch.full(((padded - rows) * (cols // BLOCK),), 127, dtype=torch.uint8)]
        )
    assert torch.equal(scale_tiles.flatten().sort().values, expected_scales.sort().values)


@pytest.mark.parametrize("shape", [(96, 64), (64, 96), (24, 64), (40, 128)])
def test_the_permutation_is_the_core_operators_byte_for_byte(shape):
    """Two independent statements of one layout: agreement is checked, not assumed."""
    rows, cols = shape
    generator = torch.Generator().manual_seed(sum(shape))
    codes = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8, generator=generator)
    scales = torch.randint(0, 256, (rows, cols // BLOCK), dtype=torch.uint8, generator=generator)

    mine_codes = pack_code_tiles(codes, rows, cols)
    mine_scales = pack_scale_tiles(scales, rows, cols)
    core_codes, core_scales = core_tile.pack_tile_major(codes, scales)

    assert torch.equal(mine_codes, core_codes), "this store's code bank is not the operator's layout"
    assert torch.equal(mine_scales, core_scales), "this store's scale bank is not the operator's"
    # And the operator's inverse agrees with this store's, both ways. relic-core's inverse trims a
    # padded bank only when it is told the logical row count, which is what `n_output` is for.
    assert torch.equal(unpack_code_tiles(core_codes, rows, cols), codes)
    back_codes, back_scales = core_tile.unpack_tile_major(mine_codes, mine_scales, n_output=rows)
    assert torch.equal(back_codes, codes) and torch.equal(back_scales, scales)


def test_the_tile_constants_are_the_core_operators_ones():
    assert TILE_ROWS == core_tile.NTILE_ROWS == 16
    assert BLOCK == core_tile.BLOCK
    assert tile_module._IDENTITY_SCALE == core_tile.IDENTITY_SCALE
    assert tile_module.TILE_DECODER_MODULE == "relic_core.kernels.tile_major_packed_gemv"
    assert tile_module.TILE_DECODER_NAME == "fp4_packed_tile_major_decode_gemv"
    assert callable(getattr(core_tile, tile_module.TILE_DECODER_NAME))


def test_malformed_shapes_and_dtypes_are_refused_by_name():
    codes = torch.zeros((96, 32), dtype=torch.uint8)
    scales = torch.zeros((96, 2), dtype=torch.uint8)
    with pytest.raises(ValueError, match="cols 65 is not a whole number"):
        pack_code_tiles(codes, 96, 65)
    # A container whose shape is not the logical one the caller states.
    with pytest.raises(ValueError, match="code bytes must be"):
        pack_code_tiles(torch.zeros((96, 16), dtype=torch.uint8), 96, 64)
    with pytest.raises(ValueError, match="scale bytes must be"):
        pack_scale_tiles(scales, 96, 128)
    with pytest.raises(TypeError, match="uint8"):
        pack_code_tiles(torch.zeros((96, 32), dtype=torch.int8), 96, 64)
    with pytest.raises(TypeError, match="uint8"):
        pack_scale_tiles(torch.zeros((96, 2), dtype=torch.int8), 96, 64)
    with pytest.raises(ValueError, match="rows must be a positive int"):
        pack_code_tiles(codes, 0, 64)
    with pytest.raises(ValueError, match="code tiles must be"):
        unpack_code_tiles(torch.zeros((6, 2, 16, 8), dtype=torch.uint8), 96, 64)


# --------------------------------------------------------------------------------------------
# Residency: the file's bytes, in one bank, nothing widened
# --------------------------------------------------------------------------------------------


def test_nothing_is_allocated_until_preload_and_the_bank_is_six_uint8_tensors():
    checkpoint = FakePackedCheckpoint(seed=3)
    store = _tile_store(checkpoint, preload=False)
    assert store.allocated is False
    assert store.resident_bytes == 0 and store.arenas == 0
    assert store.required_bytes == N_EXPERTS * packed_expert_bytes(DIM, INTER)

    held = store.preload()
    assert held == store.required_bytes == store.resident_bytes
    # Three projections, each one code bank and one scale bank: six tensors a layer, no second bank.
    assert store.arenas == 3
    assert store.cache_stats["tile_banks"] == 3
    resident = [
        tensor
        for arena in store._arenas.values()
        for tensor in (arena.codes, arena.scales)
    ]
    assert len(resident) == 6
    assert all(tensor.dtype == torch.uint8 for tensor in resident)
    assert all(tensor.shape[0] == len(store.owned) for tensor in resident)
    total = sum(int(tensor.numel()) for tensor in resident)
    assert total == store.required_bytes, "the bank holds resized bytes, not the file's count"
    # No widened form is resident anywhere: the INT8-times-two expert would be twice these bytes and
    # the dense BF16 one 3.75x, and neither exists in this store.
    assert store.resident_bytes * 2 < store.required_bytes * 4


def test_preload_refuses_a_checkpoint_it_cannot_represent_before_allocating(monkeypatch):
    """The base's validation is inherited and runs before `_allocate` is reached."""
    allocated = []
    real_allocate = _allocate

    def spy(shape, device):
        allocated.append(shape)
        return real_allocate(shape, device)

    monkeypatch.setattr(tile_module, "_allocate", spy)

    cases = {
        "grid": FakePackedCheckpoint(drop_scale=((0, 0, "w1"),)),
        "packed": FakePackedCheckpoint(not_packed=((0, 0, "w2"),)),
        "width": FakePackedCheckpoint(block=16),
        "codes": FakePackedCheckpoint(codes_shape=((0, 1, "w3"), (96, 3))),
    }
    markers = {
        "grid": "has no scale stored beside it",
        "packed": "is not stored as packed fp4",
        "width": "derives a 16-wide K block",
        "codes": "packed code bytes",
    }
    for name, checkpoint in cases.items():
        with pytest.raises(ValueError, match=markers[name]):
            _tile_store(checkpoint, preload=True)
        assert allocated == [], f"{name}: allocation happened before validation"

    # A budget below the bank is refused too, and still allocates nothing.
    with pytest.raises(ValueError, match="budget is"):
        AscendTileMajorRoutedExperts(
            FakePackedCheckpoint(), 0, n_experts=N_EXPERTS, dim=DIM, inter_dim=INTER, topk=TOPK,
            device="cpu", dtype=DTYPE, backend="torch", budget_bytes=1, reserve_bytes=0,
            preload=True,
        )
    assert allocated == []


def test_only_owned_experts_are_resident_and_unowned_ones_are_refused():
    checkpoint = FakePackedCheckpoint(seed=5)
    store = _tile_store(checkpoint, rank=1, world=2)
    assert store.owned == (1, 3)
    assert {arena.codes.shape[0] for arena in store._arenas.values()} == {2}
    assert store.required_bytes == 2 * packed_expert_bytes(DIM, INTER)
    assert torch.equal(
        unpack_code_tiles(store.expert(1).w1.codes, store.expert(1).w1.shape[0], store.expert(1).w1.shape[1]),
        checkpoint.stored_bytes(FakePackedCheckpoint._key(0, 1, "w1")),
    )
    for unowned in (0, 2):
        with pytest.raises(ValueError, match="is not owned by rank 1 of 2"):
            store.expert(unowned)
    # Asking before preload names the reason instead of loading on demand.
    cold = _tile_store(checkpoint, preload=False)
    with pytest.raises(RuntimeError, match="before the store preloaded"):
        cold.expert(0)


def test_expert_views_carry_the_logical_shape_and_no_copy():
    store = _tile_store(FakePackedCheckpoint(seed=6))
    expert = store.expert(0)
    assert isinstance(expert.w1, TileMatrix)
    assert expert.w1.shape == (INTER, DIM) and expert.w2.shape == (DIM, INTER)
    assert expert.w1.block32 == BLOCK == 32
    assert expert.w1.codesTile4D is expert.w1.codes
    assert expert.w1.scalesTile3D is expert.w1.scales
    # The view is a view: its storage is the arena's.
    assert expert.w1.codes.untyped_storage().data_ptr() == (
        store._arenas["w1"].codes.untyped_storage().data_ptr()
    )
    assert expert.nbytes == sum(m.nbytes for m in expert.matrices().values())


# --------------------------------------------------------------------------------------------
# Arithmetic: inherited, and bit-equal to the frozen row-major provider
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [1, 3])
def test_the_inherited_forward_matches_the_row_major_provider_bit_for_bit(rows):
    """One decode row (M=1) and a multi-row prefill (M>1), same fixture, FP32 bits."""
    checkpoint = FakePackedCheckpoint(seed=13)
    tile_store = _tile_store(checkpoint)
    packed_store = _packed_store(checkpoint)
    assert tile_store.required_bytes == packed_store.required_bytes

    x, weights, indices = _rows(tile_store, rows)
    expected = packed_store.forward(x, weights, indices)
    actual = tile_store.forward(x, weights, indices)
    assert actual.shape == expected.shape == (rows, DIM)
    assert _bits_equal(actual, expected), (
        f"the tile-major store disagrees with the row-major one on {rows} row(s)"
    )


def test_the_m1_hook_receives_the_tile_banks_and_they_are_the_file_bytes():
    """The provider's job is to hand the operator its own bank; a spy proves which bytes arrive."""
    checkpoint = FakePackedCheckpoint(seed=17)
    captured = {}

    def spy(a_codes, a_scale, b_codes, b_scales, *, backend="auto", **kwargs):
        # Cloned: the operands are views into the store's arenas, and a later assertion must see the
        # bytes that were handed over rather than whatever the arena holds when it runs.
        captured.setdefault("calls", []).append(
            (b_codes.clone(), b_scales.clone(), a_codes.clone(), a_scale.clone())
        )
        # Same contract as the operator: invert, then the frozen operators.
        rows = b_codes.shape[0] * TILE_ROWS
        cols = b_codes.shape[1] * BLOCK
        assert not kwargs, f"an unpadded bank must not be trimmed, got {kwargs}"
        codes = unpack_code_tiles(b_codes, rows, cols)
        scales = unpack_scale_tiles(b_scales, rows, cols)
        return fp4_int8_block_gemm(
            a_codes.contiguous(), a_scale.contiguous(),
            unpack_fp4_to_int8(codes, backend="torch"),
            fp4_int8_scale(scales, backend="torch"),
            backend="torch",
        )

    store = _tile_store(checkpoint, tile_gemv=spy)
    x, weights, indices = _rows(store, 1)
    actual = store.forward(x, weights, indices)
    assert captured["calls"], "the M=1 hook was never called"

    # Every captured bank is one expert's projection in this store's own arena, and its inverse is
    # exactly that shard's bytes -- no reordering of a reordering, no widening.
    expected_bytes = {
        (expert, which): checkpoint.stored_bytes(FakePackedCheckpoint._key(0, expert, which))
        for expert in store.owned
        for which in ("w1", "w2", "w3")
    }
    matched = set()
    for b_codes, b_scales, _a_codes, _a_scale in captured["calls"]:
        rows = b_codes.shape[0] * TILE_ROWS
        cols = b_codes.shape[1] * BLOCK
        recovered = unpack_code_tiles(b_codes, rows, cols)
        hits = [
            key for key, value in expected_bytes.items()
            if value.shape == (rows, cols // 2) and torch.equal(recovered, value)
        ]
        assert hits, "a tile bank did not invert to any stored projection"
        matched.update(hits)
        assert b_codes.is_contiguous() and b_scales.is_contiguous()
    assert len(matched) == 2 * 3, (
        f"two experts drew the row, so six projections should have been handed over; saw {matched}"
    )

    # ...and the numbers are the row-major provider's.
    packed_store = _packed_store(checkpoint)
    assert _bits_equal(actual, packed_store.forward(x, weights, indices))


def test_the_real_core_tile_operator_answers_the_decode_path():
    """`tile_gemv=None` resolves relic-core's own operator, and on CPU it runs its reference path."""
    checkpoint = FakePackedCheckpoint(seed=19)
    store = _tile_store(checkpoint, tile_gemv=None)
    assert store._tile_gemv_resolved is None, "the operator must not be resolved by construction"
    x, weights, indices = _rows(store, 1)
    actual = store.forward(x, weights, indices)
    assert store._tile_gemv_resolved is not None
    assert _bits_equal(actual, _packed_store(checkpoint).forward(x, weights, indices))


def test_a_prefill_of_one_row_still_takes_the_decode_operator():
    """The store decides per call, not per expert: a single row is a decode step."""
    checkpoint = FakePackedCheckpoint(seed=23)
    calls = []

    def spy(a_codes, a_scale, b_codes, b_scales, *, backend="auto", **kwargs):
        calls.append(b_codes.shape)
        codes = unpack_code_tiles(b_codes, b_codes.shape[0] * TILE_ROWS, b_codes.shape[1] * BLOCK)
        scales = unpack_scale_tiles(b_scales, b_codes.shape[0] * TILE_ROWS, b_codes.shape[1] * BLOCK)
        return fp4_int8_block_gemm(
            a_codes.contiguous(), a_scale.contiguous(),
            unpack_fp4_to_int8(codes, backend="torch"),
            fp4_int8_scale(scales, backend="torch"),
            backend="torch",
        )

    store = _tile_store(checkpoint, tile_gemv=spy)
    x, weights, indices = _rows(store, 1)
    store.forward(x, weights, indices)
    assert calls, "a one-row call must use the M=1 operator"
    # A multi-row call must not: it takes the inherited widening path instead.
    calls.clear()
    x, weights, indices = _rows(store, 4)
    store.forward(x, weights, indices)
    assert calls == [], "a multi-row call must not reach the M=1 operator"


def test_parallel_ranks_agree_with_the_row_major_provider_and_skip_unowned_experts():
    checkpoint = FakePackedCheckpoint(seed=29)
    for rank in range(2):
        tile_store = _tile_store(checkpoint, rank=rank, world=2)
        packed_store = _packed_store(checkpoint, rank=rank, world=2)
        x, weights, indices = _rows(tile_store, 3)
        assert _bits_equal(
            tile_store.forward(x, weights, indices), packed_store.forward(x, weights, indices)
        )
        assert tile_store.partial is True


def test_swiglu_order_and_the_shared_activation_quantization_are_inherited(monkeypatch):
    """One expert drawing the row takes exactly two quantizations: gate/up share one, w2 has its own.

    The bounded store calls the quantizer twice for gate and up; the packed provider documents the
    single shared call as its one deliberate departure, and this store inherits that. The assertion is
    on the count and on the shapes, so a second quantization of the same activation would fail it.
    """
    checkpoint = FakePackedCheckpoint(seed=31)
    # topk=1 so one row draws exactly one expert: the routing contract forbids a repeated id, and a
    # two-slot row would make the expected count "two per contributing expert" instead of two.
    store = AscendTileMajorRoutedExperts(
        checkpoint, 0, n_experts=N_EXPERTS, dim=DIM, inter_dim=INTER, topk=1,
        swiglu_limit=SWIGLU_LIMIT, device="cpu", dtype=DTYPE, backend="torch",
        reserve_bytes=0, preload=True,
    )
    calls = []

    from relic_core.kernels import block_fp8_quant

    real = block_fp8_quant.quantize_block_fp8

    def counted(x, **kwargs):
        calls.append(tuple(x.shape))
        return real(x, **kwargs)

    monkeypatch.setattr(block_fp8_quant, "quantize_block_fp8", counted)

    # One row, one slot, one owned expert.
    generator = torch.Generator().manual_seed(31)
    x = torch.randn((1, store.dim), generator=generator).to(DTYPE)
    weights = torch.tensor([[1.0]], dtype=torch.float32)
    indices = torch.tensor([[0]], dtype=torch.int64)
    store.forward(x, weights, indices)

    assert len(calls) == 2, f"expected two quantizations for a one-expert decode, got {calls}"
    # The gate/up activation is the input width; the w2 operand is the intermediate width.
    assert calls[0] == (1, store.dim)
    assert calls[1] == (1, store.inter_dim)


# --------------------------------------------------------------------------------------------
# Refusals and reporting
# --------------------------------------------------------------------------------------------


def test_a_missing_tile_operator_is_named_and_never_substituted(monkeypatch):
    import importlib

    def missing(name, *args, **kwargs):
        raise ImportError("simulated: relic-core predates the tile-major operator")

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(tile_module.TileDecodeUnavailable, match="not installed in the relic-core"):
        tile_module._import_tile_decode()

    # A module that exists but has no operator is also named, not silently accepted.
    class Stub:
        pass

    monkeypatch.setattr(importlib, "import_module", lambda name, *a, **k: Stub())
    with pytest.raises(tile_module.TileDecodeUnavailable, match="but no `fp4_packed_tile_major"):
        tile_module._import_tile_decode()

    # And the store surfaces it through the inherited forward rather than answering with the
    # row-major operator (which reads a different byte order).
    store = _tile_store(FakePackedCheckpoint(seed=37), preload=True)
    x, weights, indices = _rows(store, 1)
    with pytest.raises(tile_module.TileDecodeUnavailable):
        store.forward(x, weights, indices)


def test_the_row_major_unavailable_error_is_a_different_type():
    """A tile store must not be able to confuse the two operators' refusals."""
    assert not issubclass(TileDecodeUnavailable, PackedDecodeUnavailable)
    assert not issubclass(PackedDecodeUnavailable, TileDecodeUnavailable)


def test_a_non_callable_tile_gemm_is_refused_at_construction():
    with pytest.raises(TypeError, match="tile_gemv must be callable or None"):
        _tile_store(FakePackedCheckpoint(), tile_gemv=7)


def test_stats_report_the_strategy_and_the_same_byte_arithmetic():
    store = _tile_store(FakePackedCheckpoint(seed=41))
    stats = store.cache_stats
    assert stats["strategy"] == "tile-major"
    assert stats["retained_bytes"] == stats["required_bytes"] == store.required_bytes
    assert stats["entries"] == N_EXPERTS and stats["misses"] == 0
    assert store.strategy == "tile-major"
    shapes = store.tile_shapes
    assert shapes["w1"]["codes_shape"] == (6, 2, 16, 16)
    assert shapes["w2"]["codes_shape"] == (4, 3, 16, 16)
    assert shapes["w1"]["bytes"] == tile_matrix_bytes(INTER, DIM)
    expert = store.expert_stats
    assert expert["strategy"] == "tile-major"
    assert expert["required_bytes"] == N_EXPERTS * packed_expert_bytes(DIM, INTER)
    repr_text = repr(store)
    assert "AscendTileMajorRoutedExperts" in repr_text and "preloaded=True" in repr_text


def test_release_drops_the_bank_and_a_second_preload_is_idempotent():
    store = _tile_store(FakePackedCheckpoint(seed=43))
    held = store.resident_bytes
    assert store.preload() == held, "a second preload must not re-allocate"
    assert store.release() == N_EXPERTS
    assert store.allocated is False and store.resident_bytes == 0
    assert store.preload() == held


def test_the_prefill_operands_are_exactly_the_frozen_providers_operands():
    """`to_int8_x2` inverts the permutation, so the widening sees the file's bytes in file order.

    Compared tensor for tensor against the row-major provider's own conversion of the same shard: a
    single reordered nibble would make one of the two INT8 operands differ.
    """
    checkpoint = FakePackedCheckpoint(seed=59)
    tile_store = _tile_store(checkpoint)
    packed_store = _packed_store(checkpoint)
    for expert in tile_store.owned:
        tile_expert = tile_store.expert(expert)
        packed_expert = packed_store.expert(expert)
        for which in ("w1", "w2", "w3"):
            tile_operand = getattr(tile_expert, which).to_int8_x2(backend="torch")
            packed_operand = getattr(packed_expert, which).to_int8_x2(backend="torch")
            assert torch.equal(tile_operand.values, packed_operand.values), (
                f"expert {expert} {which}: the widened code values differ"
            )
            assert torch.equal(
                tile_operand.half_scale.view(torch.int32),
                packed_operand.half_scale.view(torch.int32),
            ), f"expert {expert} {which}: the half-scale grid differs"
            assert tile_operand.block == packed_operand.block == BLOCK


def test_the_old_provider_is_untouched_by_importing_this_module():
    """The default provider must stay the row-major one: this module only adds a specialization."""
    checkpoint = FakePackedCheckpoint(seed=47)
    store = _packed_store(checkpoint)
    assert not isinstance(store, AscendTileMajorRoutedExperts)
    assert isinstance(_tile_store(checkpoint), AscendPackedRoutedExperts)
    assert store.required_bytes == N_EXPERTS * packed_expert_bytes(DIM, INTER)


# --------------------------------------------------------------------------------------------
# On a leased card: the fused kernel, against the same fixture's CPU numbers
# --------------------------------------------------------------------------------------------

#: Set this to a leased device (``npu:0``) to exercise relic-core's fused tile-major kernel. The
#: default run skips it: a layout that has never been executed on the device is a hypothesis, and a
#: skip is not a pass.
NPU_DEVICE_ENV = "RELIC_ASCEND_TILE_MAJOR_DEVICE"


@pytest.mark.skipif(
    not os.environ.get(NPU_DEVICE_ENV),
    reason=f"set {NPU_DEVICE_ENV}=npu:0 on a leased card to run the fused case",
)
def test_the_fused_kernel_matches_the_cpu_store_on_a_leased_card():
    """The real operator on the device, at FP32 bits, against the same fixture's CPU arithmetic.

    What this establishes: the bank this store builds is the bank the fused kernel reads, and its
    arithmetic agrees with the accepted frozen operators. What it does not: tokens, a speedup, or a
    whole-model parity claim -- the layout's benefit is measured by the parent, not here.
    """
    pytest.importorskip("torch_npu")  # registers the npu device on torch
    device = torch.device(os.environ[NPU_DEVICE_ENV])
    checkpoint = FakePackedCheckpoint(seed=53)

    cpu_store = _tile_store(checkpoint, backend="torch")
    x, weights, indices = _rows(cpu_store, 1)
    expected = cpu_store.forward(x, weights, indices)

    device_store = AscendTileMajorRoutedExperts(
        checkpoint, 0, n_experts=N_EXPERTS, dim=DIM, inter_dim=INTER, topk=TOPK,
        swiglu_limit=SWIGLU_LIMIT, device=device, dtype=DTYPE, backend="triton",
        reserve_bytes=0, preload=True,
    )
    assert device_store.required_bytes == cpu_store.required_bytes
    assert all(
        tensor.device.type == "npu"
        for arena in device_store._arenas.values()
        for tensor in (arena.codes, arena.scales)
    )
    # The frozen operators take the device's tile bank and the same activation operands the CPU run
    # used: `forward` quantizes on the card, so the bytes it forms are the card's own.
    actual = device_store.forward(x.to(device), weights.to(device), indices.to(device))
    torch.npu.synchronize(device)
    assert actual.device.type == "npu"
    mismatches = int((_bits(actual.cpu()) != _bits(expected)).sum())
    assert mismatches == 0, (
        f"{mismatches} of {expected.numel()} FP32 values differ between the fused tile-major kernel "
        "and the frozen operators on the same fixture; this is a real finding, not a tolerance"
    )