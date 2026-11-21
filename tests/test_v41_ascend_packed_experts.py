"""The HBM-resident packed-expert store, checked on the CPU oracle path without a card.

`relicllm/models/deepseek_v4_1/ascend_packed_experts.py` keeps every expert a rank owns in the
checkpoint's **original packed encoding** -- E2M1 nibbles and raw E8M0 exponent bytes -- resident in
HBM, and executes it without ever storing a widened form. These tests are about the store's half of
that contract: that the resident bytes are the file's bytes, that nothing is allocated before the
checkpoint has been validated, that the expert-parallel deal is the cyclic one, and that the
arithmetic and ordering around the projections are the bounded store's.

Two oracles, and neither is the store:

* `AscendRoutedExperts` (the bounded widened store) on the same fixture, run at the same rows. Its
  operand mapping is the frozen `fp4_expert_int8`, so if this provider passed a different byte, a
  halved grid where the original was wanted, or a transposed matrix, the two disagree -- at FP32 bits,
  which is what the assertions compare;
* `_reference_packed_gemv` below, a same-contract stand-in for relic-core's M=1 operator built out of
  the accepted frozen ops (`unpack_fp4_to_int8` + `fp4_int8_scale` + `fp4_int8_block_gemm`). It is
  injected through the provider's explicit oracle seam, so the M=1 path's *plumbing* is exercised
  without a device and without pretending a kernel ran.

The fixture is a fake checkpoint exposing only the four `V41Checkpoint` methods this provider reads
(`packed`, `is_quantized`, `is_packed_fp4`, `block_size`), over hand-built bytes, exactly as the
bounded store's tests do. Nothing here runs on hardware: the one NPU case is opt-in and skipped unless
a device is named.

No performance claim is made anywhere in this file. Byte arithmetic is asserted (`18,800,640` bytes an
expert, `33.62 GiB` a card at the released geometry), not tokens per second.
"""

from __future__ import annotations

import importlib.util
import math
import random

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("relic_core.kernels.block_gemm")
pytest.importorskip("relic_core.kernels.block_fp8_quant")
pytest.importorskip("relic_core.kernels.unpack")

from relic_core.kernels.block_fp8_quant import quantize_block_fp8  # noqa: E402
from relic_core.kernels.block_gemm import fp4_int8_block_gemm  # noqa: E402
from relic_core.kernels.unpack import fp4_int8_scale, unpack_fp4_to_int8  # noqa: E402

from relicllm.models.deepseek_v4_1 import ascend_packed_experts  # noqa: E402
from relicllm.models.deepseek_v4_1.ascend_experts import (  # noqa: E402
    AscendExpertCache,
    AscendRoutedExperts,
    ResidentExpert,
    expert_forward_ascend,
    global_cache,
)
from relicllm.models.deepseek_v4_1.ascend_packed_experts import (  # noqa: E402
    AscendPackedRoutedExperts,
    PackedDecodeUnavailable,
    packed_expert_bytes,
    packed_expert_forward,
    packed_model_bytes,
    packed_store_bytes,
)
from relicllm.models.deepseek_v4_1.online_unpack import fp4_expert_int8  # noqa: E402

BLOCK = 32
DIM = 64
INTER = 96
TOPK = 2
N_EXPERTS = 4
DTYPE = torch.bfloat16
SWIGLU_LIMIT = 10.0

#: Scale bytes the fixtures draw from: ordinary octaves, so nothing overflows FP32 unless a test is
#: about exactly that (`2 ** (b - 127)` for b in [120, 136) is [2**-7, 2**9)).
_SCALE_BYTES = range(120, 136)

_ARENA_DTYPES = (torch.uint8,)


def _scale_key(key: str) -> str:
    """`...w1.weight` -> `...w1.scale`, spelled here so the fixture does not import the loader."""
    stem, _, leaf = key.rpartition(".")
    assert leaf == "weight"
    return f"{stem}.scale"


def _as_container(raw: torch.Tensor, *, typed: bool) -> torch.Tensor:
    """`raw` as the container the released file stores it in, without changing a byte."""
    if not typed:
        return raw
    dtype = getattr(torch, "float8_e8m0fnu", None)
    return raw if dtype is None else raw.view(dtype)


class FakePackedCheckpoint:
    """Hand-built original packed bytes behind the four methods the provider reads.

    The shapes are the released ones' *relationships* rather than their sizes: a projection is
    `[out, in/2]` of packed nibbles beside `[out, in/32]` of raw E8M0 bytes. Knobs make the failure
    cases explicit -- a missing grid, a grid of the wrong width, a projection of the wrong shape, a
    two-byte container -- so an assertion fails for the reason it names.
    """

    def __init__(
        self,
        *,
        dim: int = DIM,
        inter: int = INTER,
        experts: int = N_EXPERTS,
        layers: tuple[int, ...] = (0,),
        seed: int = 0,
        block: int = BLOCK,
        drop_scale: tuple[tuple[int, int, str], ...] = (),
        scale_shape: tuple[tuple[int, int, str], tuple[int, int]] | None = None,
        codes_shape: tuple[tuple[int, int, str], tuple[int, int]] | None = None,
        wide_container: tuple[tuple[int, int, str], ...] = (),
        not_packed: tuple[tuple[int, int, str], ...] = (),
        typed_container: bool = False,
    ) -> None:
        self.root = f"fixture://packed-{seed}"
        self.dim = dim
        self.inter = inter
        self.inter_dim = inter
        self.experts = experts
        self.layers = tuple(layers)
        self.block = block
        self._drop = set(drop_scale)
        self._scale_shape = scale_shape
        self._codes_shape = codes_shape
        self._wide = set(wide_container)
        self._not_packed = set(not_packed)
        # The released file holds packed fp4 as `int8` and its grids as the E8M0 container dtype.
        self._typed = typed_container
        self._tensors: dict[str, torch.Tensor] = {}
        generator = random.Random(seed)
        for layer in self.layers:
            for expert in range(experts):
                for which in ("w1", "w2", "w3"):
                    out, inn = (dim, inter) if which == "w2" else (inter, dim)
                    key = self._key(layer, expert, which)
                    shape = (out, inn // 2)
                    if codes_shape is not None and codes_shape[0] == (layer, expert, which):
                        shape = codes_shape[1]
                    count = math.prod(shape)
                    drawn = [generator.randrange(256) for _ in range(count)]
                    if (layer, expert, which) in self._wide:
                        self._tensors[key] = torch.tensor(drawn, dtype=torch.int16).reshape(shape)
                    else:
                        raw = torch.tensor(drawn, dtype=torch.uint8).reshape(shape)
                        # A *view* into the signed container the file uses; the bytes are identical.
                        self._tensors[key] = raw.view(torch.int8) if self._typed else raw
                    if (layer, expert, which) in self._drop:
                        continue
                    blocks = inn // block
                    grid_shape = (out, blocks)
                    if scale_shape is not None and scale_shape[0] == (layer, expert, which):
                        grid_shape = scale_shape[1]
                    count = math.prod(grid_shape)
                    drawn = [generator.choice(_SCALE_BYTES) for _ in range(count)]
                    raw = torch.tensor(drawn, dtype=torch.uint8).reshape(grid_shape)
                    self._tensors[_scale_key(key)] = _as_container(raw, typed=self._typed)

    @staticmethod
    def _key(layer: int, expert: int, which: str) -> str:
        return f"layers.{layer}.ffn.experts.{expert}.{which}.weight"

    # -- the four methods the provider reads ----------------------------------

    def packed(self, key: str) -> torch.Tensor:
        return self._tensors[key]

    def is_quantized(self, key: str) -> bool:
        try:
            sibling = _scale_key(key)
        except ValueError:
            return False
        return sibling in self._tensors

    def is_packed_fp4(self, key: str) -> bool:
        for layer in self.layers:
            for expert in range(self.experts):
                for which in ("w1", "w2", "w3"):
                    if self._key(layer, expert, which) == key:
                        return (layer, expert, which) not in self._not_packed
        return True

    def block_size(self, key: str) -> int:
        return self.block

    # -- what the tests read --------------------------------------------------

    def stored(self, key: str) -> torch.Tensor:
        """The stored tensor, in the container the file uses."""
        return self._tensors[key]

    def stored_bytes(self, key: str) -> torch.Tensor:
        return self._tensors[key].view(torch.uint8)


def _reference_packed_gemv(a_codes, a_scale, b_packed, b_e8m0, *, backend="torch"):
    """A same-contract stand-in for relic-core's M=1 operator, built from the frozen ops.

    It converts the packed weight exactly as the bounded store does -- INT8-times-two values beside
    the halved grid -- and calls the accepted block GEMM. The provider's job is to hand it the file's
    bytes in the file's layout, so agreement with the bounded store's own path is the statement this
    makes; it is *not* evidence that relic-core's fused kernel produces the same numbers, which is a
    device question this file does not answer.
    """
    values = unpack_fp4_to_int8(b_packed.contiguous(), backend="torch")
    half_scale = fp4_int8_scale(b_e8m0.contiguous(), backend="torch")
    return fp4_int8_block_gemm(
        a_codes.contiguous(), a_scale.contiguous(), values, half_scale, backend="torch"
    )


def _bounded_store(checkpoint, *, rank: int = 0, world: int = 1) -> AscendRoutedExperts:
    return AscendRoutedExperts(
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
        cache=AscendExpertCache(0),
        backend="torch",
    )


def _packed_store(
    checkpoint=None,
    *,
    rank: int = 0,
    world: int = 1,
    preload: bool = True,
    budget_bytes: int | None = None,
    reserve_bytes: int = 0,
    packed_gemv=_reference_packed_gemv,
    device: str = "cpu",
    backend: str = "torch",
) -> AscendPackedRoutedExperts:
    checkpoint = FakePackedCheckpoint() if checkpoint is None else checkpoint
    return AscendPackedRoutedExperts(
        checkpoint,
        0,
        n_experts=checkpoint.experts,
        dim=checkpoint.dim,
        inter_dim=checkpoint.inter,
        topk=TOPK,
        swiglu_limit=SWIGLU_LIMIT,
        device=device,
        dtype=DTYPE,
        rank=rank,
        world=world,
        backend=backend,
        budget_bytes=budget_bytes,
        reserve_bytes=reserve_bytes,
        preload=preload,
        packed_gemv=packed_gemv,
    )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """An FP32 tensor's bits, so a comparison is exact rather than a tolerance."""
    return tensor.detach().to(torch.float32).contiguous().view(torch.int32)


def _bits_equal(first: torch.Tensor, second: torch.Tensor) -> bool:
    return torch.equal(_bits(first), _bits(second))


# --------------------------------------------------------------------------------------------
# The byte arithmetic the residency plan is built on
# --------------------------------------------------------------------------------------------


def test_one_released_expert_is_18_800_640_resident_packed_bytes():
    assert packed_expert_bytes(5120, 2304) == 18_800_640
    # 17.93 MiB, against the widened store's 37.97 MiB INT8 form and the 67.5 MiB dense BF16 one.
    assert packed_expert_bytes(5120, 2304) == 5_898_240 * 3 + 368_640 * 3


def test_one_card_of_the_released_model_is_33_62_gib():
    store = packed_store_bytes(384, 5120, 2304, rank=0, world=8)
    assert store == 48 * 18_800_640
    total = packed_model_bytes(40, 384, 5120, 2304, rank=0, world=8)
    assert total == 40 * store
    assert total == 36_097_228_800
    assert total / 2**30 == pytest.approx(33.62, abs=0.005)


def test_the_deal_gives_each_rank_disjoint_experts_and_the_bytes_are_per_owned_expert():
    for world in (1, 2, 3):
        seen = []
        for rank in range(world):
            seen.extend(
                expert
                for expert in range(384)
                if expert % world == rank
            )
            assert packed_store_bytes(384, 5120, 2304, rank=rank, world=world) == len(
                [expert for expert in range(384) if expert % world == rank]
            ) * 18_800_640
        assert sorted(seen) == list(range(384)), "the cyclic deal must partition every expert once"


# --------------------------------------------------------------------------------------------
# Resident bytes are the file's bytes
# --------------------------------------------------------------------------------------------


def test_preload_keeps_every_stored_byte_unchanged_and_stores_only_uint8():
    checkpoint = FakePackedCheckpoint(seed=7)
    store = _packed_store(checkpoint, preload=False)
    assert store.preload() == store.required_bytes == N_EXPERTS * packed_expert_bytes(DIM, INTER)

    for expert in range(N_EXPERTS):
        packed = store.expert(expert)
        for which, matrix in packed.matrices().items():
            key = FakePackedCheckpoint._key(0, expert, which)
            expected = checkpoint.stored_bytes(key).reshape(matrix.codes.shape)
            assert matrix.codes.dtype in _ARENA_DTYPES
            assert matrix.scales.dtype in _ARENA_DTYPES
            assert torch.equal(matrix.codes.cpu(), expected), (
                f"{key}: the resident code bytes are not the file's bytes"
            )
            expected_scale = checkpoint.stored_bytes(_scale_key(key)).reshape(matrix.scales.shape)
            assert torch.equal(matrix.scales.cpu(), expected_scale), (
                f"{key}: the resident scale bytes are not the file's raw E8M0 bytes"
            )


def test_an_expert_is_a_view_of_the_arena_rather_than_a_copy():
    store = _packed_store(FakePackedCheckpoint(seed=8))
    first, second = store.expert(0), store.expert(1)
    # Two experts' views sit in one allocation of all owned experts, which a per-expert copy would not.
    assert (
        first.w1.codes.untyped_storage().data_ptr()
        == second.w1.codes.untyped_storage().data_ptr()
    )
    assert first.w1.codes.untyped_storage().size() == len(store.owned) * INTER * (DIM // 2)
    assert first.w2.scales.untyped_storage().size() == len(store.owned) * DIM * (INTER // BLOCK)
    assert first.w1.nbytes == (INTER * DIM // 2) + (INTER * (DIM // BLOCK))
    # The views are distinct slices of it.
    assert first.w1.codes.data_ptr() != second.w1.codes.data_ptr()


def test_the_released_container_dtypes_are_read_as_bytes_and_never_converted():
    """The file stores packed codes as `int8` and grids as the E8M0 container; both are payloads.

    A numeric cast of that E8M0 container would turn the exponent byte 127 into the integer 1, so
    this asserts the resident bytes are the container's *byte* values and that a cast would in fact
    have produced something different -- the failure mode the view exists to avoid.
    """
    checkpoint = FakePackedCheckpoint(seed=49, typed_container=True)
    store = _packed_store(checkpoint)
    codes_dtype = checkpoint.stored(FakePackedCheckpoint._key(0, 0, "w1")).dtype
    scale_key = _scale_key(FakePackedCheckpoint._key(0, 0, "w1"))
    scales_dtype = checkpoint.stored(scale_key).dtype
    assert codes_dtype == torch.int8
    assert getattr(torch, "float8_e8m0fnu", None) is None or scales_dtype == torch.float8_e8m0fnu

    for expert in range(N_EXPERTS):
        for which, matrix in store.expert(expert).matrices().items():
            key = FakePackedCheckpoint._key(0, expert, which)
            assert torch.equal(matrix.codes.cpu(), checkpoint.stored_bytes(key))
            assert torch.equal(matrix.scales.cpu(), checkpoint.stored_bytes(_scale_key(key)))

    # And the cast this avoided is measurably different: E8M0 byte 127 is the value 1.0.
    measured = store.expert(0).w1.scales.cpu()
    container = checkpoint.stored(scale_key)
    converted = container.float().to(torch.uint8)
    if bool((measured != converted).any()):
        differing = measured != converted
        assert torch.equal(measured[differing], torch.tensor(
            [int(value) for value in checkpoint.stored_bytes(scale_key)[differing]], dtype=torch.uint8
        ))
    else:  # pragma: no cover - only if a fixture drew bytes that happen to cast to themselves
        pytest.skip("this seed's scale bytes happen to cast to themselves")


def test_only_six_arena_tensors_are_allocated_for_a_whole_layer():
    store = _packed_store(FakePackedCheckpoint(seed=9))
    assert store.arenas == 3, "three projections, each one arena object holding two arrays"
    arrays = sum(
        2 for _ in range(store.arenas)
    )  # each arena object owns a code array and a scale array
    assert arrays == 6, "one layer must not build a per-expert tensor"
    # And the bytes are exactly the arithmetic, with no per-expert overhead.
    assert store.resident_bytes == store.required_bytes


# --------------------------------------------------------------------------------------------
# Nothing is allocated before the checkpoint is validated
# --------------------------------------------------------------------------------------------


def test_construction_allocates_nothing():
    store = _packed_store(FakePackedCheckpoint(seed=10), preload=False)
    assert store.allocated is False
    assert store.resident_bytes == 0
    assert store.resident_experts == 0
    assert store.required_bytes == N_EXPERTS * packed_expert_bytes(DIM, INTER)


def test_a_budget_below_the_residency_is_refused_by_name():
    required = N_EXPERTS * packed_expert_bytes(DIM, INTER)
    with pytest.raises(ValueError, match="residency is this provider's whole design"):
        _packed_store(budget_bytes=required - 1, preload=False)
    # Exactly enough is accepted.
    store = _packed_store(budget_bytes=required, preload=False)
    assert store.budget_bytes == required


def test_the_default_reserve_leaves_room_for_the_rest_of_the_run():
    """The residency is not the only thing on the card; the preflight floor says so explicitly."""
    assert ascend_packed_experts.DEFAULT_RESERVE_BYTES == 12 * 2**30
    checkpoint = FakePackedCheckpoint(seed=45)
    store = AscendPackedRoutedExperts(
        checkpoint,
        0,
        n_experts=checkpoint.experts,
        dim=checkpoint.dim,
        inter_dim=checkpoint.inter_dim,
        topk=TOPK,
        swiglu_limit=SWIGLU_LIMIT,
        device="cpu",
        dtype=DTYPE,
        backend="torch",
        packed_gemv=_reference_packed_gemv,
    )
    assert store.reserve_bytes == 12 * 2**30
    assert store.budget_bytes == store.required_bytes + 12 * 2**30
    # And an explicit reserve is honoured rather than being overridden by the default.
    explicit = _packed_store(checkpoint, preload=False, reserve_bytes=0)
    assert explicit.reserve_bytes == 0
    assert explicit.budget_bytes == explicit.required_bytes


@pytest.mark.parametrize(
    "checkpoint_kwargs, match",
    [
        ({"drop_scale": ((0, 0, "w1"),)}, "has no scale stored beside it"),
        ({"not_packed": ((0, 1, "w2"),)}, "is not stored as packed fp4"),
        (
            {"scale_shape": ((0, 0, "w1"), (INTER, 99))},
            "the original grid keeps one scale per output row",
        ),
        (
            {"codes_shape": ((0, 0, "w1"), (INTER, 99))},
            "packed code bytes",
        ),
        ({"wide_container": ((0, 0, "w1"),)}, "not an 8-bit original-code container"),
    ],
)
def test_validation_failures_happen_before_any_allocation(checkpoint_kwargs, match, monkeypatch):
    checkpoint = FakePackedCheckpoint(seed=11, **checkpoint_kwargs)
    store = _packed_store(checkpoint, preload=False)

    def refuse_allocation(shape, device):
        pytest.fail("an arena was allocated before the checkpoint was validated")

    monkeypatch.setattr(ascend_packed_experts, "_allocate", refuse_allocation)
    with pytest.raises((ValueError, TypeError), match=match):
        store.preload()
    assert store.allocated is False and store.resident_bytes == 0


def test_a_bad_block_width_is_refused_before_any_allocation(monkeypatch):
    class WrongBlock(FakePackedCheckpoint):
        def block_size(self, key: str) -> int:
            return 16

    store = _packed_store(WrongBlock(seed=12), preload=False)

    def refuse_allocation(shape, device):
        pytest.fail("an arena was allocated before the checkpoint was validated")

    monkeypatch.setattr(ascend_packed_experts, "_allocate", refuse_allocation)
    with pytest.raises(ValueError, match="K block"):
        store.preload()


def test_preload_and_release_report_the_bytes_held():
    store = _packed_store(FakePackedCheckpoint(seed=13), preload=False)
    assert store.release() == 0, "releasing a store that never loaded is a no-op"
    held = store.preload()
    assert held == store.resident_bytes == store.required_bytes
    assert store.allocated is True
    assert store.preload() == held, "preload is idempotent"
    assert store.release() == N_EXPERTS
    assert store.allocated is False and store.resident_bytes == 0
    assert store.clear_cache() == 0


# --------------------------------------------------------------------------------------------
# The cyclic expert-parallel deal
# --------------------------------------------------------------------------------------------


def test_each_rank_holds_and_serves_only_its_own_experts():
    checkpoint = FakePackedCheckpoint(experts=6, seed=14)
    stores = [_packed_store(checkpoint, rank=rank, world=3) for rank in range(3)]
    unions = [set(store.owned) for store in stores]
    assert set().union(*unions) == set(range(6))
    assert sum(len(owned) for owned in unions) == 6, "the deal must partition, not overlap"
    for rank, store in enumerate(stores):
        assert all(expert % 3 == rank for expert in store.owned)
        assert store.partial is True
        for expert in store.owned:
            assert store.expert(expert).w1.codes.shape == (INTER, DIM // 2)
        foreign = next(expert for expert in range(6) if expert % 3 != rank)
        with pytest.raises(ValueError, match="is not owned by rank"):
            store.expert(foreign)
    assert _packed_store(checkpoint, world=1).partial is False


def test_an_out_of_range_or_mistyped_expert_id_is_refused():
    store = _packed_store(FakePackedCheckpoint(seed=15))
    with pytest.raises(ValueError, match="outside"):
        store.expert(N_EXPERTS)
    with pytest.raises(TypeError, match="expert id is an int"):
        store.expert("w1")


# --------------------------------------------------------------------------------------------
# The arithmetic: the packed store against the bounded store, at FP32 bits
# --------------------------------------------------------------------------------------------


def test_one_row_matches_the_bounded_store_bit_for_bit():
    checkpoint = FakePackedCheckpoint(seed=16)
    bounded = _bounded_store(checkpoint)
    packed = _packed_store(checkpoint)
    x = torch.randn(1, DIM, generator=torch.Generator().manual_seed(17)).to(DTYPE)
    weights = torch.tensor([[0.5, 0.25]], dtype=torch.float32)
    indices = torch.tensor([[0, 2]], dtype=torch.int64)

    expected = bounded.forward(x, weights, indices)
    actual = packed.forward(x, weights, indices)
    assert _bits_equal(actual, expected), "the M=1 packed path disagrees with the bounded store"


def test_a_prefill_matches_the_bounded_store_bit_for_bit():
    checkpoint = FakePackedCheckpoint(seed=18)
    bounded = _bounded_store(checkpoint)
    packed = _packed_store(checkpoint)
    x = torch.randn(5, DIM, generator=torch.Generator().manual_seed(19)).to(DTYPE)
    weights = torch.tensor(
        [[0.5, 0.25], [0.75, -0.5], [1.0, 0.125], [0.25, 0.5], [-0.75, 0.375]],
        dtype=torch.float32,
    )
    indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 3]], dtype=torch.int64)

    expected = bounded.forward(x, weights, indices)
    actual = packed.forward(x, weights, indices)
    assert _bits_equal(actual, expected), "the prefill path disagrees with the bounded store"


def test_expert_parallel_partials_match_the_bounded_store_per_rank():
    checkpoint = FakePackedCheckpoint(experts=6, seed=20)
    x = torch.randn(3, DIM, generator=torch.Generator().manual_seed(21)).to(DTYPE)
    weights = torch.tensor([[0.5, 0.25], [0.75, -0.5], [1.0, 0.125]], dtype=torch.float32)
    indices = torch.tensor([[0, 3], [1, 4], [2, 5]], dtype=torch.int64)
    total_packed = torch.zeros(3, DIM, dtype=torch.float32)
    total_bounded = torch.zeros(3, DIM, dtype=torch.float32)
    for rank in range(3):
        bounded = _bounded_store(checkpoint, rank=rank, world=3)
        packed = _packed_store(checkpoint, rank=rank, world=3)
        expected = bounded.forward(x, weights, indices)
        actual = packed.forward(x, weights, indices)
        assert _bits_equal(actual, expected), f"rank {rank} disagrees with the bounded store"
        total_bounded += expected
        total_packed += actual
    assert _bits_equal(total_packed, total_bounded), "the ranks' partial sums compose differently"


def test_the_projection_level_path_agrees_for_one_expert():
    checkpoint = FakePackedCheckpoint(seed=22)
    packed_store = _packed_store(checkpoint)
    resident = packed_store.expert(0)
    bounded_resident = ResidentExpert(
        w1=fp4_expert_int8(
            checkpoint,
            FakePackedCheckpoint._key(0, 0, "w1"),
            _scale_key(FakePackedCheckpoint._key(0, 0, "w1")),
            device="cpu",
            backend="torch",
        ),
        w2=fp4_expert_int8(
            checkpoint,
            FakePackedCheckpoint._key(0, 0, "w2"),
            _scale_key(FakePackedCheckpoint._key(0, 0, "w2")),
            device="cpu",
            backend="torch",
        ),
        w3=fp4_expert_int8(
            checkpoint,
            FakePackedCheckpoint._key(0, 0, "w3"),
            _scale_key(FakePackedCheckpoint._key(0, 0, "w3")),
            device="cpu",
            backend="torch",
        ),
        block=BLOCK,
    )
    generator = torch.Generator().manual_seed(23)
    for rows, gemv in ((1, _reference_packed_gemv), (3, None)):
        x = torch.randn(rows, DIM, generator=generator).to(DTYPE)
        weight = torch.full((rows, 1), 0.5, dtype=torch.float32)
        expected = expert_forward_ascend(
            x, bounded_resident, swiglu_limit=SWIGLU_LIMIT, weights=weight, dtype=DTYPE,
            backend="torch",
        )
        actual = packed_expert_forward(
            x, resident, swiglu_limit=SWIGLU_LIMIT, weights=weight, dtype=DTYPE, backend="torch",
            packed_gemv=gemv,
        )
        assert _bits_equal(actual, expected), f"{rows} row(s) disagreed at the projection level"


def test_the_routing_weight_placement_is_visible_in_this_fixture():
    """The bit-equality tests would be vacuous on a degenerate fixture; this shows they are not.

    The reference multiplies the routing weight into the **intermediate** activation. Applying it to
    the expert's *output* instead is the other plausible reading, and it is a different number here:
    the assertion failing would mean this fixture cannot tell the two apart, which is what would make
    the bit comparisons elsewhere uninformative.
    """
    checkpoint = FakePackedCheckpoint(seed=24)
    store = _packed_store(checkpoint)
    x = torch.randn(4, DIM, generator=torch.Generator().manual_seed(25)).to(DTYPE) * 8.0
    # Not a power of two on purpose: a power-of-two routing weight scales the block amax and its
    # power-of-two scale together, so the quantizer would give the same codes and the two
    # placements would agree exactly -- a fixture with no teeth.
    weights = torch.tensor(
        [[0.3, 0.7], [1.3, -0.45], [0.11, 2.7], [-0.9, 0.65]], dtype=torch.float32
    )
    indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int64)
    reference = store.forward(x, weights, indices)
    assert float(reference.abs().sum()) > 0.0, "a degenerate fixture would make every comparison vacuous"

    # The same store, with the weight moved from the intermediate activation to the output. It is
    # built from the same resident bytes through the projection-level entry point, so the difference
    # is the placement and nothing else.
    late = torch.zeros_like(reference)
    buckets = store._routing(indices)
    for expert in sorted(buckets):
        rows, slots = buckets[expert]
        row_index = torch.tensor(rows, dtype=torch.int64)
        slot_index = torch.tensor(slots, dtype=torch.int64)
        gate_weight = weights.index_select(0, row_index).gather(1, slot_index.unsqueeze(1))
        unweighted = packed_expert_forward(
            x.index_select(0, row_index),
            store.expert(expert),
            swiglu_limit=SWIGLU_LIMIT,
            weights=None,
            dtype=DTYPE,
            backend="torch",
            packed_gemv=_reference_packed_gemv,
        )
        late.index_put_(
            (row_index,), late.index_select(0, row_index) + (unweighted * gate_weight).to(torch.float32)
        )
    assert not _bits_equal(late, reference), (
        "applying the routing weight to the output produced the same bits as applying it to the "
        "intermediate activation, so this fixture cannot distinguish the two"
    )


# --------------------------------------------------------------------------------------------
# Empty routing, and refusing to run before residency
# --------------------------------------------------------------------------------------------


def test_empty_routing_returns_an_empty_fp32_sum():
    store = _packed_store(FakePackedCheckpoint(seed=26))
    x = torch.zeros((0, DIM), dtype=DTYPE)
    indices = torch.zeros((0, TOPK), dtype=torch.int64)
    weights = torch.zeros((0, TOPK), dtype=torch.float32)
    actual = store.forward(x, weights, indices)
    assert actual.shape == (0, DIM) and actual.dtype == torch.float32


def test_a_forward_before_preload_is_refused_and_allocates_nothing():
    store = _packed_store(FakePackedCheckpoint(seed=27), preload=False)
    x = torch.randn(1, DIM, generator=torch.Generator().manual_seed(28)).to(DTYPE)
    indices = torch.tensor([[0, 1]], dtype=torch.int64)
    weights = torch.full((1, TOPK), 0.5, dtype=torch.float32)
    with pytest.raises(RuntimeError, match="does not load on demand"):
        store.forward(x, weights, indices)
    with pytest.raises(RuntimeError, match="does not load on demand"):
        store.expert(0)
    assert store.allocated is False


def test_the_routing_contract_is_the_bounded_store_s_one():
    store = _packed_store(FakePackedCheckpoint(seed=29))
    x = torch.randn(2, DIM, generator=torch.Generator().manual_seed(30)).to(DTYPE)
    weights = torch.full((2, TOPK), 0.5, dtype=torch.float32)
    with pytest.raises(ValueError, match="repeat"):
        store.forward(x, weights, torch.tensor([[0, 0], [1, 2]], dtype=torch.int64))
    with pytest.raises(ValueError, match="this layer has"):
        store.forward(x, weights, torch.tensor([[0, 99], [1, 2]], dtype=torch.int64))
    with pytest.raises(ValueError, match="topk="):
        store.forward(x, torch.full((2, 3), 0.5, dtype=torch.float32),
              torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64))


# --------------------------------------------------------------------------------------------
# The M=1 operator seam: used when present, refused by name when not, never substituted
# --------------------------------------------------------------------------------------------


def test_a_missing_m1_operator_is_a_named_refusal_and_not_the_prefill_path(monkeypatch):
    store = _packed_store(FakePackedCheckpoint(seed=31), packed_gemv=None)

    def unavailable():
        raise PackedDecodeUnavailable("no fused decode operator on this install")

    monkeypatch.setattr(ascend_packed_experts, "_import_packed_decode", unavailable)
    x = torch.randn(1, DIM, generator=torch.Generator().manual_seed(32)).to(DTYPE)
    indices = torch.tensor([[0, 1]], dtype=torch.int64)
    weights = torch.full((1, TOPK), 0.5, dtype=torch.float32)
    with pytest.raises(PackedDecodeUnavailable, match="no fused decode operator"):
        store.forward(x, weights, indices)

    # More than one row is the prefill path and must not consult the M=1 operator at all.
    several = torch.randn(3, DIM, generator=torch.Generator().manual_seed(33)).to(DTYPE)
    indices = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
    weights = torch.full((3, TOPK), 0.5, dtype=torch.float32)
    assert store.forward(several, weights, indices).shape == (3, DIM)


def test_the_m1_path_widens_no_weight_and_the_prefill_path_widens_one_projection_at_a_time(monkeypatch):
    store = _packed_store(FakePackedCheckpoint(seed=34))
    calls: list[tuple[int, int]] = []
    original = ascend_packed_experts.PackedMatrix.to_int8_x2

    def counting(self, *, backend="auto"):
        calls.append(self.shape)
        return original(self, backend=backend)

    monkeypatch.setattr(ascend_packed_experts.PackedMatrix, "to_int8_x2", counting)
    indices = torch.tensor([[0, 1]], dtype=torch.int64)
    weights = torch.full((1, TOPK), 0.5, dtype=torch.float32)
    one = torch.randn(1, DIM, generator=torch.Generator().manual_seed(35)).to(DTYPE)
    store.forward(one, weights, indices)
    assert calls == [], "the M=1 path must never build the INT8-times-two operands"

    several = torch.randn(4, DIM, generator=torch.Generator().manual_seed(36)).to(DTYPE)
    multi_indices = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]], dtype=torch.int64)
    multi_weights = torch.full((4, TOPK), 0.5, dtype=torch.float32)
    store.forward(several, multi_weights, multi_indices)
    # Four experts are drawn by that routing, each with three projections, one at a time.
    assert len(calls) == 12, f"expected one bounded widening per projection, got {calls}"
    assert all(shape in {(INTER, DIM), (DIM, INTER)} for shape in calls)


def test_the_real_core_operator_agrees_with_the_injected_oracle_at_one_row():
    """The seam is only useful if the real frozen operator takes the same operands the same way.

    relic-core's ``fp4_packed_decode_gemv`` is checked here on its CPU backend against both the
    bounded store and this file's injected stand-in. It is the operator the provider resolves by
    name, so this closes the gap between "the provider passes the right bytes" and "the operator
    reads them as expected" on the one backend that can run without a card. The NPU/fused backend is
    a device acceptance and is not claimed by this test.
    """
    decode_gemv = pytest.importorskip("relic_core.kernels.decode_block_gemv")
    operator = getattr(decode_gemv, "fp4_packed_decode_gemv", None)
    if operator is None:
        pytest.skip("the installed relic-core has no packed decode operator to compare")

    checkpoint = FakePackedCheckpoint(seed=46)
    bounded = _bounded_store(checkpoint)
    packed = _packed_store(checkpoint)
    x = torch.randn(1, DIM, generator=torch.Generator().manual_seed(47)).to(DTYPE)
    indices = torch.tensor([[0, 2]], dtype=torch.int64)
    weights = torch.tensor([[0.5, 0.25]], dtype=torch.float32)

    expected = bounded.forward(x, weights, indices)
    injected = packed.forward(x, weights, indices)
    assert _bits_equal(injected, expected)

    # The operator itself, called directly with the four operands the provider hands over, and
    # compared against the accepted INT8-times-two block GEMM on the same packed bytes.
    codes, scales = quantize_block_fp8(x, backend="torch")
    resident = packed.expert(0)
    measured = operator(codes, scales, resident.w1.codes, resident.w1.scales, backend="torch")
    widened = unpack_fp4_to_int8(resident.w1.codes, backend="torch")
    half = fp4_int8_scale(resident.w1.scales, backend="torch")
    accepted = fp4_int8_block_gemm(codes, scales, widened, half, backend="torch")
    assert _bits_equal(measured, accepted), (
        "relic-core's packed operator and the accepted INT8-times-two block GEMM disagree on the "
        "same packed bytes"
    )

    # And the operator refuses more than one row by name, which is why the provider routes a prefill
    # through the bounded widening instead of trying it here.
    two_rows = torch.randn(2, DIM, generator=torch.Generator().manual_seed(48)).to(DTYPE)
    two_codes, two_scales = quantize_block_fp8(two_rows, backend="torch")
    with pytest.raises(ValueError, match="exactly one activation row"):
        operator(two_codes, two_scales, resident.w1.codes, resident.w1.scales, backend="torch")


def test_gate_and_up_share_one_activation_quantization_measured_against_the_bounded_store(monkeypatch):
    import relic_core.kernels.block_fp8_quant as block_fp8_quant_module

    checkpoint = FakePackedCheckpoint(seed=37)
    packed = _packed_store(checkpoint)
    bounded = _bounded_store(checkpoint)
    original = block_fp8_quant_module.quantize_block_fp8
    packed_calls: list[int] = []
    bounded_calls: list[int] = []

    def count_into(sink):
        def wrapper(*arguments, **keywords):
            sink.append(1)
            return original(*arguments, **keywords)

        return wrapper

    one_row = torch.randn(1, DIM, generator=torch.Generator().manual_seed(38)).to(DTYPE)
    indices = torch.tensor([[0, 1]], dtype=torch.int64)
    weights = torch.full((1, TOPK), 0.5, dtype=torch.float32)

    monkeypatch.setattr(block_fp8_quant_module, "quantize_block_fp8", count_into(bounded_calls))
    expected = bounded.forward(one_row, weights, indices)
    assert len(bounded_calls) == 6, (
        "the bounded store quantizes gate, up and the w2 operand for each of two experts"
    )

    monkeypatch.setattr(block_fp8_quant_module, "quantize_block_fp8", count_into(packed_calls))
    actual = packed.forward(one_row, weights, indices)
    assert _bits_equal(actual, expected), (
        "sharing the gate/up activation bytes must not change a bit of the result"
    )
    assert len(packed_calls) == 4, (
        "one activation quantization plus one w2-operand quantization for each of two experts; the "
        f"observed count was {len(packed_calls)}"
    )

    # And the shared bytes are the bytes a second independent call produces: the quantizer is
    # deterministic, which is the whole justification for sharing.
    first = quantize_block_fp8(one_row, backend="torch")
    second = quantize_block_fp8(one_row, backend="torch")
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


# --------------------------------------------------------------------------------------------
# Reporting, and the cache this store deliberately does not use
# --------------------------------------------------------------------------------------------


def test_cache_stats_keep_the_widened_snapshot_keys_and_describe_residency():
    store = _packed_store(FakePackedCheckpoint(seed=39))
    stats = store.cache_stats
    for key in ("name", "budget_bytes", "retained_bytes", "entries", "hits", "misses", "evictions",
                "streamed", "hit_rate"):
        assert key in stats, f"the CLI reads {key!r} from the widened store's snapshot"
    assert stats["entries"] == N_EXPERTS
    assert stats["retained_bytes"] == store.required_bytes
    assert stats["misses"] == 0 and stats["evictions"] == 0 and stats["streamed"] == 0
    assert stats["hit_rate"] is None
    assert stats["preloaded"] is True
    assert stats["required_bytes"] == store.required_bytes


def test_expert_stats_and_repr_describe_the_store():
    store = _packed_store(FakePackedCheckpoint(seed=40), rank=1, world=2)
    stats = store.expert_stats
    assert stats["owned_experts"] == len(store.owned) == N_EXPERTS // 2
    assert stats["resident_experts"] == len(store.owned)
    assert stats["partial"] is True
    assert stats["packed_expert_bytes"] == packed_expert_bytes(DIM, INTER)
    assert stats["required_bytes"] == store.required_bytes
    text = repr(store)
    assert "AscendPackedRoutedExperts" in text and "preloaded=True" in text and "rank=1/2" in text


def test_the_process_wide_widened_cache_is_not_used():
    """Residency replaces the LRU; it must not also fill the shared widened cache."""
    store = _packed_store(FakePackedCheckpoint(seed=41))
    x = torch.randn(2, DIM, generator=torch.Generator().manual_seed(42)).to(DTYPE)
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int64)
    weights = torch.full((2, TOPK), 0.5, dtype=torch.float32)
    before = global_cache().snapshot()
    store.forward(x, weights, indices)
    after = global_cache().snapshot()
    assert after["entries"] == before["entries"], "the packed store put an entry in the shared LRU"
    assert after["retained_bytes"] == before["retained_bytes"]
    # The cache object the parent constructor requires is private and zero-budget on purpose.
    assert store.cache.budget_bytes == 0


# --------------------------------------------------------------------------------------------
# The device case, opt-in: a leased card is what runs this
# --------------------------------------------------------------------------------------------


def _device_case():
    name = __import__("os").environ.get("RELIC_ASCEND_PACKED_EXPERTS_DEVICE")
    if not name:
        pytest.skip("set RELIC_ASCEND_PACKED_EXPERTS_DEVICE=npu:0 on a leased card to run the device case")
    pytest.importorskip("torch_npu")  # registers the npu device on torch
    return torch.device(name)


def test_npu_residency_keeps_the_bytes_and_matches_the_host_oracle():
    """Opt-in. The lead runs this; the suite skips it. No speed claim is made here."""
    device = _device_case()
    checkpoint = FakePackedCheckpoint(seed=43)
    packed = _packed_store(checkpoint, device=str(device), backend="triton", packed_gemv=None)
    assert packed.resident_bytes == packed.required_bytes
    # Every resident byte is the file's byte, read back from the card.
    for expert in range(N_EXPERTS):
        seen = packed.expert(expert)
        for which, matrix in seen.matrices().items():
            key = FakePackedCheckpoint._key(0, expert, which)
            assert torch.equal(matrix.codes.cpu(), checkpoint.stored_bytes(key))
            assert torch.equal(
                matrix.scales.cpu(), checkpoint.stored_bytes(_scale_key(key))
            )
    # The decode path is a separate acceptance: it needs relic-core's fused operator and a card.
    if importlib.util.find_spec("relic_core.kernels.decode_block_gemv") is None:
        pytest.skip("relic-core has no packed decode operator to run on this device yet")
    bounded = _bounded_store(checkpoint)
    x = torch.randn(1, DIM, generator=torch.Generator().manual_seed(44)).to(DTYPE)
    indices = torch.tensor([[0, 1]], dtype=torch.int64)
    weights = torch.full((1, TOPK), 0.5, dtype=torch.float32)
    expected = bounded.forward(x, weights, indices)
    actual = packed.forward(x.to(device), weights.to(device), indices.to(device))
    torch.npu.synchronize(device)
    assert _bits_equal(actual.cpu(), expected), "the device M=1 path disagrees with the CPU oracle"