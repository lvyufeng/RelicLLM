"""The block table: what one page holds, what a pool holds, and what sharing costs.

`relicllm/runtime/kv_blocks.py` is the second artefact `docs/architecture/kv_declaration.md` named as
missing, after the declaration itself: a table that says *which* physical page holds a sequence's
positions, for each cache the model declared. It allocates bookkeeping and no tensors, so these tests
are the whole of its verification -- the geometry the four `KVKind`s resolve to, the pool's byte
accounting and its one-free-list-per-size rule, and the sharing invariant that is what removes
copy-on-write from the design.

Nothing here needs torch or a checkpoint. The DeepSeek-V4.1-Flash numbers are read from the same
checked-in config snapshot `tests/test_kv_spec.py` uses, so the page sizes are arithmetic over the
real declaration rather than a fixture's invention.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from relicllm.runtime.kv_blocks import (
    DEFAULT_BLOCK_SIZE,
    BlockGeometry,
    BlockPool,
    KVBlockAllocator,
    NoSpace,
)
from relicllm.runtime.kv_spec import (
    FullAttentionSpec,
    MLASpec,
    SlidingWindowSpec,
    StateSpec,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "triage" / "kv_geometry.json"

#: The architectures that publish a declaration, restated the way `tests/test_kv_spec.py` restates it
#: so a publisher that moves is a failure here rather than a silently skipped architecture.
PUBLISHERS = {
    "deepseek_v4": "relicllm.models.deepseek_v4.kv_spec",
    "deepseek_v4_1": "relicllm.models.deepseek_v4_1.kv_spec",
    "mimo": "relicllm.models.mimo_v2.kv_spec",
    "qwen4_exp": "relicllm.models.qwen4_exp.kv_spec",
    "xing4": "relicllm.models.xing4_0.kv_spec",
}


def _fixture_model(architecture: str) -> dict[str, Any]:
    for model in json.loads(FIXTURE.read_text())["models"]:
        if model["architecture"] == architecture:
            return model
    raise AssertionError(f"the fixture no longer carries {architecture}")


def _publisher_specs(architecture: str, config: Any) -> tuple[Any, ...]:
    import importlib

    return importlib.import_module(PUBLISHERS[architecture]).kv_spec(config)


def _gqa(layer_ids: tuple[int, ...] = (0,), head_dim: int = 8, name: str = "k") -> FullAttentionSpec:
    """A full-attention cache whose page at the default block size is exactly 512 bytes."""
    return FullAttentionSpec(
        name=name, layer_ids=layer_ids, dtype="bfloat16", num_kv_heads=1, head_dim=head_dim
    )


def _ring(sliding_window: int = 128, layer_ids: tuple[int, ...] = (0,)) -> SlidingWindowSpec:
    return SlidingWindowSpec(
        name="window_kv_cache",
        layer_ids=layer_ids,
        dtype="bfloat16",
        num_kv_heads=1,
        head_dim=8,
        v_head_dim=0,
        sliding_window=sliding_window,
    )


def _mla(compress_ratio: int, layer_ids: tuple[int, ...] = (0,)) -> MLASpec:
    return MLASpec(
        name="compress_kv_cache",
        layer_ids=layer_ids,
        dtype="bfloat16",
        num_kv_heads=1,
        head_dim=8,
        v_head_dim=0,
        compress_ratio=compress_ratio,
    )


def _state(layer_ids: tuple[int, ...] = (0,)) -> StateSpec:
    return StateSpec(
        name="kv_state",
        layer_ids=layer_ids,
        dtype="bfloat16",
        shapes=((16, 8),),
        dtypes=("bfloat16",),
    )


# --------------------------------------------------------------------------------------------
# Geometry: what one page of each kind holds
# --------------------------------------------------------------------------------------------


def test_a_full_attention_page_holds_one_block_size_of_positions() -> None:
    """The simple row, and the one every other row is read against."""
    geometry = BlockGeometry.of(_gqa(), DEFAULT_BLOCK_SIZE)

    assert geometry.positions_per_block == DEFAULT_BLOCK_SIZE
    assert geometry.max_blocks is None
    assert geometry.kind_is_growing, "a full-attention row is a function of position, so a prefix shares"
    assert geometry.page_bytes == DEFAULT_BLOCK_SIZE * 16 * 2
    assert [geometry.blocks_for(n) for n in (0, 1, 16, 17, 32)] == [0, 1, 1, 2, 2]


def test_a_ring_saturates_and_is_never_shared() -> None:
    """A ring holds its window however long the context runs -- the allocated/marginal split."""
    geometry = BlockGeometry.of(_ring(128), DEFAULT_BLOCK_SIZE)

    assert geometry.max_blocks == 8
    assert not geometry.kind_is_growing, (
        "a ring's slot for a position is position % window, so a continuation writes where a "
        "sharer is still reading"
    )
    assert [geometry.blocks_for(n) for n in (0, 1, 16, 128, 129, 10**9)] == [0, 1, 1, 8, 8, 8]


def test_a_ring_whose_window_is_not_a_block_multiple_still_holds_it() -> None:
    """The ceiling is `ceil(window / block_size)`, so the ring is padded up, never down."""
    geometry = BlockGeometry.of(_ring(100), DEFAULT_BLOCK_SIZE)

    assert geometry.max_blocks == 7
    assert geometry.max_blocks * geometry.positions_per_block >= 100


def test_a_state_is_one_page_for_the_sequence_and_zero_before_it_starts() -> None:
    """No token axis: the page is the whole state, and a sequence that computed nothing holds none."""
    geometry = BlockGeometry.of(_state(), DEFAULT_BLOCK_SIZE)

    assert geometry.positions_per_block == 0
    assert geometry.max_blocks == 1
    assert not geometry.kind_is_growing
    assert [geometry.blocks_for(n) for n in (0, 1, 10**9)] == [0, 1, 1]
    assert geometry.page_bytes == 16 * 8 * 2


@pytest.mark.parametrize("ratio", [1, 2, 3, 5, 7, 16, 64, 128])
def test_a_compressed_page_never_lays_out_fewer_latents_than_it_needs(ratio: int) -> None:
    """The page a compressed row buys has to cover every pooled group in it.

    `MLASpec.page_size_bytes` floors the latents at one per page and the geometry floors the coverage
    at one group per page, and the two have to be the same floor: a page the geometry thought covered
    15 positions but which holds one latent would drop the other four groups on the floor. Checked
    ratio by ratio, including ratios that divide the block size and ratios larger than it.
    """
    spec = _mla(ratio)
    geometry = BlockGeometry.of(spec, DEFAULT_BLOCK_SIZE)
    latents_per_page = max(1, DEFAULT_BLOCK_SIZE // ratio)

    assert geometry.page_bytes == spec.allocated_bytes(DEFAULT_BLOCK_SIZE)
    for positions in (1, 16, 17, 512, 4096):
        pages = geometry.blocks_for(positions)
        assert pages * latents_per_page >= -(-positions // ratio), (
            f"ratio {ratio} at {positions} positions: {pages} pages hold "
            f"{pages * latents_per_page} latents, the compressor produces {-(-positions // ratio)}"
        )


def test_the_default_block_size_leaves_no_cache_a_floored_remainder() -> None:
    """Why 16, checked against every cache the five publishers actually declare.

    A block size indivisible by a compress ratio gives a page fewer positions than blocks are cut at
    (ratio 3 against 16 covers 15), and one that is not a multiple of a ratio pads every page. The
    property the pool needs is that a page boundary is a block boundary for every cache at once, so
    this is asserted over the roster rather than over the constant's own arithmetic.
    """
    seen: set[tuple[str, int]] = set()
    for architecture in PUBLISHERS:
        model = _fixture_model(architecture)
        for spec in _publisher_specs(architecture, model["config"]):
            geometry = BlockGeometry.of(spec, DEFAULT_BLOCK_SIZE)
            seen.add((spec.kind.value, geometry.positions_per_block))
            if isinstance(spec, StateSpec):
                assert geometry.positions_per_block == 0
                continue
            assert geometry.positions_per_block % DEFAULT_BLOCK_SIZE == 0, (
                f"{architecture}'s {spec.name} covers {geometry.positions_per_block} positions a "
                f"page, which is not a whole number of {DEFAULT_BLOCK_SIZE}-position blocks"
            )

    assert seen == {
        ("full", DEFAULT_BLOCK_SIZE),
        ("sliding_window", DEFAULT_BLOCK_SIZE),
        ("mla", 16),
        ("mla", 128),
        ("state", 0),
    }, f"the roster's geometries changed shape: {sorted(seen)}"


@pytest.mark.parametrize("architecture", sorted(PUBLISHERS))
def test_the_pages_a_sequence_holds_always_cover_what_its_positions_cost(architecture: str) -> None:
    """The table's own invariant: it rounds a sequence up to a page, never down.

    Under-allocating is the one failure a block table cannot survive -- the writer would run off the
    end of the last page -- so each growing row is checked to hold every position it was grown to,
    at a block boundary and off it, in the page the positions fall in and never one more. The other
    half of the reading is the one that makes the table worth using: the bytes of those pages are the
    checkpoint-verified marginal cost of the positions in them, because a page is exactly a page's
    worth of positions and the marginal reading is what a page's bytes are cut from.

    A ring is not in this: it holds a fixed window however long the context runs, so its bytes are
    not a function of position at all. The ring test above and the V4.1 test below are where that is
    stated.
    """
    model = _fixture_model(architecture)
    specs = _publisher_specs(architecture, model["config"])
    allocator = KVBlockAllocator(specs, capacity_bytes=10**9)

    for positions in (1, 16, 17, 128, 129, 4096):
        for row, spec in enumerate(specs):
            geometry = allocator.geometry[row]
            if not geometry.kind_is_growing:
                continue
            pages = geometry.blocks_for(positions)
            covered = pages * geometry.positions_per_block

            assert covered >= positions, (
                f"{model['name']}'s {spec.name} at {positions} positions: {pages} pages cover "
                f"{covered}, which is fewer than were asked for"
            )
            assert covered - positions < geometry.positions_per_block, (
                "the positions fall in one page and the row stops at it -- a table pads, it does "
                "not buy a page nothing was written to"
            )
            assert pages * allocator.page_bytes(row) == covered * spec.bytes_per_token_whole_model(), (
                f"{model['name']}'s {spec.name} at {positions} positions holds "
                f"{pages * allocator.page_bytes(row)} bytes for {covered} positions the checkpoint "
                f"prices at {spec.bytes_per_token_whole_model()} a position: the page a row buys has "
                "to be the marginal cost of the positions in it, or the pool is sized wrong"
            )


# --------------------------------------------------------------------------------------------
# The pool: bytes, refcounts, and one free list per page size
# --------------------------------------------------------------------------------------------


def test_the_pool_is_denominated_in_bytes_because_pages_differ_in_size() -> None:
    """`N blocks remaining` would mean four amounts at once; `N bytes remaining` is one."""
    pool = BlockPool(4096)

    first = pool.acquire(512)
    second = pool.acquire(2048)

    assert pool.used_bytes == 512 + 2048, "a page of each size counted at its own size"
    assert pool.free_bytes == 4096 - 2560
    assert pool.blocks_in_use == 2
    assert pool.page_bytes(first) == 512 and pool.page_bytes(second) == 2048


def test_a_released_id_comes_back_only_to_a_page_of_its_own_size() -> None:
    """Ids are handed to the writer, so a reissue at a different size is a stale row pointing wrong."""
    pool = BlockPool(8192)
    small = pool.acquire(512)
    big = pool.acquire(2048)

    assert pool.release(small) is True
    reused = pool.acquire(512)
    assert reused == small, "the small page is idle, so its id is the one that comes back"
    assert pool.page_bytes(reused) == 512

    pool.release(reused)
    other = pool.acquire(2048)
    assert other != small, "an idle 512-byte page is not a 2048-byte page's free list"
    assert pool.page_bytes(other) == 2048 and pool.page_bytes(big) == 2048


def test_a_shared_page_is_counted_once_however_many_sequences_read_it() -> None:
    """The whole point of the table: a prefix reuse is a pointer copy, not a buffer copy."""
    pool = BlockPool(4096)
    block = pool.acquire(2048)

    pool.retain(block)
    assert pool.used_bytes == 2048, "one page of memory however many holders"
    assert pool.blocks_in_use == 1
    assert pool.refs(block) == 2 and pool.is_shared(block)

    assert pool.release(block) is False, "one holder still points at it"
    assert pool.used_bytes == 2048 and pool.free_bytes == 2048
    assert not pool.is_shared(block)

    assert pool.release(block) is True, "the last holder frees the page"
    assert pool.used_bytes == 0 and pool.blocks_in_use == 0 and pool.free_bytes == 4096


def test_a_double_free_is_refused_rather_than_corrupting_the_accounting() -> None:
    """A table that released twice would hand one id to two sequences, so this fails loudly."""
    pool = BlockPool(4096)
    block = pool.acquire(512)
    pool.release(block)

    with pytest.raises(ValueError):
        pool.release(block)
    with pytest.raises(KeyError):
        pool.retain(4096)
    with pytest.raises(ValueError):
        pool.acquire(0)


def test_a_request_that_does_not_fit_is_refused_all_or_nothing() -> None:
    """A half-grown row would leave pages the caller has to remember to release."""
    pool = BlockPool(4096)
    pool.acquire(2048)

    with pytest.raises(NoSpace):
        pool.acquire_many(1024, 3)
    assert pool.used_bytes == 2048, "the refused request took nothing"
    assert pool.blocks_in_use == 1
    assert pool.acquire_many(1024, 0) == []
    assert pool.acquire_many(1024, 2) != [], "and the same request with room does take pages"


def test_a_page_larger_than_the_whole_pool_is_refused_with_the_numbers() -> None:
    pool = BlockPool(1024)

    with pytest.raises(NoSpace) as refusal:
        pool.acquire(2048)

    assert "2048" in str(refusal.value) and "1024" in str(refusal.value)
    assert not pool.afford(2048) and pool.afford(1024)


# --------------------------------------------------------------------------------------------
# A sequence: growing, rolling back, and refusing to shrink
# --------------------------------------------------------------------------------------------


def test_a_sequence_buys_pages_at_the_block_boundary_and_not_in_between() -> None:
    """A decode step costs nothing until it crosses into a new page -- the block table's whole saving."""
    allocator = KVBlockAllocator([_gqa()], capacity_bytes=512 * 4)
    sequence = allocator.new_sequence()

    for positions, pages in ((0, 0), (1, 1), (16, 1), (17, 2), (32, 2), (33, 3)):
        sequence.grow(positions)
        assert sequence.positions == positions
        assert len(sequence.blocks(0)) == pages
        assert sequence.held_bytes == pages * 512
    assert allocator.pool.used_bytes == 3 * 512


def test_can_grow_answers_the_difference_so_a_scheduler_can_admit_before_committing() -> None:
    """The check and the commit read the same arithmetic, so the two cannot disagree."""
    allocator = KVBlockAllocator([_gqa()], capacity_bytes=512 * 2)
    sequence = allocator.new_sequence()
    sequence.grow(16)

    assert sequence.can_grow(16) and sequence.can_grow(32)
    assert not sequence.can_grow(33), "one more page than the pool has left"
    assert not sequence.can_grow(15), "a sequence does not shrink"

    sequence.grow(32)
    assert allocator.pool.free_bytes == 0
    assert not sequence.can_grow(33)


def test_a_refused_grow_leaves_the_pool_and_the_rows_exactly_as_it_found_them() -> None:
    """Row two cannot afford its page, so row one's -- already bought -- has to go back."""
    allocator = KVBlockAllocator(
        [_gqa(), _gqa(head_dim=16, name="k2")], capacity_bytes=512 + 1024 - 1
    )
    sequence = allocator.new_sequence()
    before = allocator.pool.free_bytes

    with pytest.raises(NoSpace):
        sequence.grow(16)

    assert allocator.pool.free_bytes == before, "the first row's page was rolled back"
    assert allocator.pool.used_bytes == 0
    assert allocator.pool.blocks_in_use == 0
    assert sequence.positions == 0
    assert sequence.blocks(0) == () and sequence.blocks(1) == ()


def test_a_sequence_refuses_to_shrink() -> None:
    allocator = KVBlockAllocator([_gqa()], capacity_bytes=512 * 4)
    sequence = allocator.new_sequence()
    sequence.grow(32)

    with pytest.raises(ValueError):
        sequence.grow(16)
    assert sequence.positions == 32 and len(sequence.blocks(0)) == 2


def test_required_bytes_is_the_whole_declaration_and_nothing_is_allocated_by_asking() -> None:
    """What a scheduler sizes a pool with, and the number a request is admitted against."""
    allocator = KVBlockAllocator(
        [_gqa(), _mla(2), _ring(128), _state()], capacity_bytes=10**6
    )

    assert allocator.required_bytes(0) == 0
    # One full-attention page (16 positions of 8 keys and 8 values at 2 bytes = 512), one compressed
    # page (ratio 2 pools two positions into a latent of the layer's whole 8 values: 16 // 2 latents
    # at 2 bytes = 128), the ring's first page (256 -- its value tensor is absorbed, and it buys pages
    # as it fills) and the state, which is one page from the sequence's first position (256).
    assert allocator.required_bytes(1) == 512 + 128 + 256 + 256
    assert allocator.required_bytes(16) == allocator.required_bytes(1)
    assert allocator.required_bytes(17) == 2 * 512 + 2 * 128 + 2 * 256 + 256

    sequence = allocator.new_sequence()
    assert sequence.required_bytes(17) == allocator.required_bytes(17)
    assert sequence.blocks_for(17) == (2, 2, 2, 1)
    assert allocator.pool.used_bytes == 0, "asking costs nothing"


# --------------------------------------------------------------------------------------------
# Sharing: whole pages, and only the rows a prefix is a function of
# --------------------------------------------------------------------------------------------


def test_sharing_stops_at_a_page_boundary_and_reports_what_it_actually_took() -> None:
    """A caller asking for 100 positions of a 16-position block size gets 96 and recomputes the rest."""
    allocator = KVBlockAllocator([_gqa()], capacity_bytes=512 * 16)
    producer = allocator.new_sequence()
    producer.grow(100)
    consumer = allocator.new_sequence()

    shared = consumer.share_prefix_from(producer, 100)

    assert shared == 96, "the partial page the producer is still filling is not shareable"
    assert consumer.positions == 96
    assert consumer.blocks(0) == producer.blocks(0)[:6]
    assert allocator.pool.refs(consumer.blocks(0)[0]) == 2
    assert allocator.pool.used_bytes == 7 * 512, (
        "six shared pages plus the producer's own partial seventh, each counted once"
    )


def test_a_continuation_writes_into_its_own_page_and_never_a_shared_one() -> None:
    """This is what removes copy-on-write: the shared pages are full, and a full page is done.

    A page is only ever written while it is filling, and the filling page is the sequence's last and
    private one -- so a sequence that adopts a prefix and keeps decoding never touches a page another
    sequence is reading. If this ever failed, the fix would be a copy-on-write path with a byte
    copier, which is the machinery the invariant exists to avoid.
    """
    allocator = KVBlockAllocator([_gqa()], capacity_bytes=512 * 16)
    producer = allocator.new_sequence()
    producer.grow(100)
    consumer = allocator.new_sequence()
    shared = consumer.share_prefix_from(producer, 100)

    consumer.grow(100)
    tail = consumer.blocks(0)[shared // 16 :]

    assert len(tail) == 1
    assert tail[0] not in producer.blocks(0), "the tail page is the consumer's own"
    assert allocator.pool.refs(tail[0]) == 1
    assert not allocator.pool.is_shared(tail[0])


def test_rings_and_states_are_allocated_privately_however_much_of_the_prefix_is_reused() -> None:
    """Sharing a ring would recycle the sharer's slots; a recurrent state is not a prefix at all."""
    allocator = KVBlockAllocator(
        [_gqa(), _ring(128), _state()], capacity_bytes=10**6, block_size=DEFAULT_BLOCK_SIZE
    )
    producer = allocator.new_sequence()
    producer.grow(64)
    consumer = allocator.new_sequence()

    shared = consumer.share_prefix_from(producer, 64)

    assert shared == 64, "a multiple of the growing row's page, so nothing was truncated"
    assert consumer.blocks(0) == producer.blocks(0) and allocator.pool.refs(consumer.blocks(0)[0]) == 2
    for row in (1, 2):
        assert consumer.blocks(row), "the ring and the state are allocated, not skipped"
        assert not set(consumer.blocks(row)) & set(producer.blocks(row))
        assert all(allocator.pool.refs(block) == 1 for block in consumer.blocks(row))
    assert consumer.positions == producer.positions == 64


def test_a_prefix_is_adopted_into_an_empty_sequence_only() -> None:
    allocator = KVBlockAllocator([_gqa()], capacity_bytes=512 * 16)
    producer = allocator.new_sequence()
    producer.grow(32)

    with pytest.raises(ValueError):
        producer.share_prefix_from(producer, 16)
    with pytest.raises(ValueError):
        producer.share_prefix_from(allocator.new_sequence(), 16)

    other = KVBlockAllocator([_gqa(), _ring()], capacity_bytes=512 * 16).new_sequence()
    with pytest.raises(ValueError):
        allocator.new_sequence().share_prefix_from(other, 16)


def test_a_shared_prefix_is_a_loan_that_freeing_gives_back_once_per_holder() -> None:
    allocator = KVBlockAllocator([_gqa()], capacity_bytes=512 * 16)
    producer = allocator.new_sequence()
    producer.grow(64)
    consumer = allocator.new_sequence()
    consumer.share_prefix_from(producer, 64)

    consumer.free()
    assert allocator.pool.used_bytes == 4 * 512, "the producer still holds its own pages"
    assert allocator.pool.refs(producer.blocks(0)[0]) == 1
    assert consumer.positions == 0 and consumer.blocks(0) == ()

    producer.free()
    assert allocator.pool.used_bytes == 0 and allocator.pool.blocks_in_use == 0


# --------------------------------------------------------------------------------------------
# The allocator: the declaration's rows, and how long a pool holds
# --------------------------------------------------------------------------------------------


def test_a_name_and_a_layer_resolve_a_row_because_a_name_alone_can_be_ambiguous() -> None:
    """V4.1 publishes `compress_kv_cache` twice -- ratio 2 and ratio 1 -- so the name is not a key."""
    model = _fixture_model("deepseek_v4_1")
    allocator = KVBlockAllocator(
        _publisher_specs("deepseek_v4_1", model["config"]), capacity_bytes=10**8
    )

    ratio2 = allocator.row("compress_kv_cache", layer=2)
    ratio1 = allocator.row("compress_kv_cache", layer=20)
    assert ratio2 != ratio1
    assert allocator.specs[ratio2].compress_ratio == 2
    assert allocator.specs[ratio1].compress_ratio == 1
    assert allocator.row("window_kv_cache") == allocator.row("window_kv_cache", layer=0)

    with pytest.raises(KeyError):
        allocator.row("compress_kv_cache")
    with pytest.raises(KeyError):
        allocator.row("no_such_cache")
    with pytest.raises(KeyError):
        allocator.row("compress_kv_cache", layer=3)


def test_the_v41_declaration_plugs_in_with_the_numbers_the_checkpoint_verified() -> None:
    """The real geometry, at the real block size: a block table sized for one 4096-position sequence.

    The ring is the number worth looking at. Forty layers hold a 128-slot window each -- 5,242,880
    bytes once full -- and the four growing rows buy 51,200 bytes every 16 positions. A ring buys its
    pages as it fills rather than reserving the window up front, so a 16-position sequence holds one
    of the eight; what matters is that its cost stops moving at 128 and a growing row's never does.
    """
    model = _fixture_model("deepseek_v4_1")
    specs = _publisher_specs("deepseek_v4_1", model["config"])
    allocator = KVBlockAllocator(specs, capacity_bytes=10**8)

    ring = allocator.row("window_kv_cache")
    assert allocator.geometry[ring].max_blocks == 8
    assert allocator.page_bytes(ring) == 655_360, "16384 bytes a page over forty layers"
    assert allocator.required_bytes(16) == 655_360 + 51_200, (
        "one ring page plus one growing page of each of the four cache kinds"
    )
    assert allocator.required_bytes(4096) == 5_242_880 + 256 * 51_200
    assert allocator.required_bytes(129) - allocator.required_bytes(128) == 51_200, (
        "past its window a ring costs nothing more, so only the growing rows moved"
    )
    assert (allocator.required_bytes(4096) - 5_242_880) // 4096 == (
        model["expected_bytes_per_token_whole_model"]
    ), "a growing page's bytes a position are the checkpoint-verified marginal cost"


def test_max_positions_is_exact_for_a_growing_declaration_and_none_when_nothing_grows() -> None:
    """Two answers a host prints: a number of positions, or "the limit is the context, not memory"."""
    growing = KVBlockAllocator([_gqa()], capacity_bytes=512 * 10)
    assert growing.max_positions() == 160
    assert growing.required_bytes(160) <= growing.pool.capacity_bytes
    assert growing.required_bytes(161) > growing.pool.capacity_bytes

    bounded = KVBlockAllocator([_ring(128), _state()], capacity_bytes=10**6)
    assert bounded.max_positions() is None, "no position count exhausts a pool of rings and states"
    assert bounded.required_bytes(128) == bounded.required_bytes(10**9), (
        "a bounded declaration does not cost nothing, it stops costing more: the list of caches is "
        "constant from the window on, which is why no position count exhausts the pool"
    )
    assert bounded.required_bytes(1) < bounded.required_bytes(128), (
        "and it is not constant before that -- a ring buys its pages as it fills"
    )


def test_a_pool_that_does_not_hold_the_rings_holds_no_positions_at_all() -> None:
    """The bounded rows are paid first and in full, so `max_positions` is 0 rather than `None`."""
    model = _fixture_model("deepseek_v4_1")
    specs = _publisher_specs("deepseek_v4_1", model["config"])
    rings_only = KVBlockAllocator(specs, capacity_bytes=655_360)
    assert rings_only.max_positions() == 0

    one_block = KVBlockAllocator(specs, capacity_bytes=655_360 + 51_200)
    assert one_block.max_positions() == 16


def test_the_block_table_is_readable_without_importing_torch() -> None:
    """The table is in the device plane beside the declaration, and the declaration's test says why.

    A host sizing a pool on a machine that never loads the model must not pay for a torch import, and
    a fresh process is the only way to check: torch is not in this one either, but the import could
    have happened at module scope of something this test imports indirectly.
    """
    program = "import sys, relicllm.runtime.kv_blocks; sys.exit(1 if 'torch' in sys.modules else 0)"
    completed = subprocess.run(
        [sys.executable, "-c", program], cwd=REPO_ROOT, capture_output=True, text=True
    )

    assert completed.returncode == 0, "the block table imported torch:\n" + completed.stderr
