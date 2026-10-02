"""Gate 0 in isolation: the card count, the ``(L, B)`` curve, and the host-memory precondition.

Every case is arithmetic over numbers chosen here, so this runs on any machine and pins the
*decisions* rather than the roster. The roster's own answers live in a separate fixture and need
checkpoints; these are the rules that would go wrong silently if somebody "simplified" the solver.

The three that matter:

* a **replicated** KV cache gets no help from another card, so a model whose cache alone exceeds one
  card does not fit at any count -- the definitive failure the static gate is allowed to give;
* the **context and batch are one budget**, so a curve exists and a single pair does not describe it;
* **host memory is a precondition**, so a bank larger than RAM spills back onto the cards and moves
  the card count rather than being noted and forgotten.
"""

from __future__ import annotations

import pytest

from relicllm.components.moe.placement import (
    HardwareProfile,
    context_batch_curve,
    estimate_even_shard,
    minimum_cards,
    usable_per_card,
)
from relicllm.triage.fit import hardware_profile_for

GIB = 1024 ** 3


def test_usable_per_card_tracks_the_reserve_it_was_given() -> None:
    """The reserve is a parameter because the two values in use differ by more than a model's margin."""
    box = hardware_profile_for(gpu_memory_gib=22.0)

    assert usable_per_card(box, reserve_fraction=0.15) == pytest.approx(18.7 * GIB, rel=1e-4)
    assert usable_per_card(box, reserve_fraction=0.10) == pytest.approx(19.8 * GIB, rel=1e-4)


def test_even_shard_rounds_up_so_it_never_understates_a_rank() -> None:
    assert estimate_even_shard(10, 4) == 3
    assert estimate_even_shard(0, 4) == 0
    assert estimate_even_shard(7, 0) == 7  # no cards is not a division by zero


def test_a_replicated_cache_cannot_be_fixed_with_more_cards() -> None:
    """The reason ``sharded=False`` exists, and the only definitive failure gate 0 gives.

    24 GiB of KV per card is over the 18.7 GiB budget however the weights divide, because a
    replicated cache does not divide at all. Under sharding the same numbers fit at two cards.
    """
    box = hardware_profile_for(gpu_count=4, gpu_memory_gib=22.0)
    kwargs = dict(weights_bytes=1 * GIB, context=32768, batch=1, hardware=box)

    assert minimum_cards(**kwargs, kv_bytes_per_token_per_rank_at_one_card=24 * GIB // 32768, sharded=False) is None
    assert minimum_cards(**kwargs, kv_bytes_per_token_per_rank_at_one_card=24 * GIB // 32768, sharded=True) == 2


def test_the_card_count_never_exceeds_what_the_box_has() -> None:
    box = hardware_profile_for(gpu_count=2, gpu_memory_gib=22.0)

    assert minimum_cards(
        weights_bytes=100 * GIB,
        kv_bytes_per_token_per_rank_at_one_card=0,
        context=1,
        batch=1,
        hardware=box,
    ) is None


def test_context_and_batch_are_one_budget_and_the_curve_shows_it() -> None:
    """Xing4's shape: 4 GiB of weights, 45 KiB of KV per token, one card.

    ``L * B`` is what the budget buys, so at batch 8 the same headroom reaches an eighth of the
    context. A tool that printed only "L_max" would be answering a question the caller did not ask.
    """
    box = hardware_profile_for(gpu_count=1, gpu_memory_gib=22.0)
    curve = context_batch_curve(
        weights_per_rank_bytes=4 * GIB,
        kv_bytes_per_token_per_rank=45000,
        hardware=box,
        contexts=(8192, 32768, 131072),
        batches=(1, 8),
    )

    at_8k = curve.at(8192)
    assert at_8k is not None
    assert at_8k.kv_bytes_per_rank == 8192 * 1 * 45000
    assert at_8k.headroom_bytes > 0

    # The same point at batch 8 pays eight times the KV, which is what makes the trade real.
    eight = [point for point in curve.points if point.context == 8192 and point.max_batch == 8][0]
    assert eight.kv_bytes_per_rank == 8 * at_8k.kv_bytes_per_rank
    assert eight.headroom_bytes < at_8k.headroom_bytes


def test_a_curve_that_does_not_fit_at_any_point_says_which_budget_bound() -> None:
    box = hardware_profile_for(gpu_count=1, gpu_memory_gib=22.0)

    weights_bound = context_batch_curve(
        weights_per_rank_bytes=20 * GIB, kv_bytes_per_token_per_rank=1, hardware=box, contexts=(2048,), batches=(1,)
    )
    kv_bound = context_batch_curve(
        weights_per_rank_bytes=1 * GIB,
        kv_bytes_per_token_per_rank=1024 * 64,
        hardware=box,
        contexts=(131072,),
        batches=(8,),
    )

    assert weights_bound.binds == "weights"
    assert kv_bound.binds == "kv"


def test_max_context_at_reads_the_curve_by_batch() -> None:
    box = hardware_profile_for(gpu_count=1, gpu_memory_gib=22.0)
    curve = context_batch_curve(
        weights_per_rank_bytes=4 * GIB,
        kv_bytes_per_token_per_rank=45000,
        hardware=box,
        contexts=(8192, 32768, 131072),
        batches=(1, 8),
    )

    assert curve.max_context_at(1) == 131072
    assert curve.max_context_at(8) == 32768
    assert curve.max_context_at(64) == 0


def test_a_runtime_with_no_tensor_parallelism_cannot_divide_its_weights() -> None:
    """``splits_weights=False``, which is Xing4's shape and was a real bug before it existed.

    A single-card runtime gets no help from cards 2-4, so 24 GiB of weights must be measured against
    one card's budget rather than a quarter of it. Without this flag the solver reports three cards
    for a runtime that refuses ``--tensor-parallel-size > 1``.
    """
    box = hardware_profile_for(gpu_count=4, gpu_memory_gib=22.0)
    kwargs = dict(weights_bytes=24 * GIB, context=1, batch=1, hardware=box, sharded=True)

    assert minimum_cards(**kwargs, kv_bytes_per_token_per_rank_at_one_card=0, splits_weights=True) == 2
    assert minimum_cards(**kwargs, kv_bytes_per_token_per_rank_at_one_card=0, splits_weights=False) is None


def test_the_offload_spill_is_measured_over_the_bank_not_the_whole_checkpoint() -> None:
    """A mis-sized spill double-counts the resident part, and the double count is not small.

    Xing4 is 4.32 GiB resident plus a 14.40 GiB bank. Spilling ``on_card - host`` instead of
    ``bank - host`` reports ``resident + on_card`` = 23.04 GiB of weights and calls the model
    impossible on a host with no memory to spare -- which is the right *verdict* from the wrong
    arithmetic, and would give the wrong verdict the moment the numbers moved by a gigabyte.
    """
    from relicllm.triage.fit import _host_bank_fits
    from relicllm.triage.weights import PrecisionOption

    option = PrecisionOption(
        label="test",
        path="/nowhere",
        format="safetensors",
        precision="I8",
        bytes_total=int(18.72 * GIB),
        on_card_bytes=int(18.72 * GIB),
        resident_on_card_bytes=int(4.32 * GIB),
        confidence="measured",
        evidence="test",
    )

    fits, spill = _host_bank_fits(None, option, hardware_profile_for(host_memory_gib=0.001), host_reserve_fraction=0.10)

    assert not fits
    assert spill == pytest.approx(int(14.40 * GIB), rel=0.01)
    # The resident part is not in the spill, so adding the two does not count it twice.
    assert int(4.32 * GIB) + spill == pytest.approx(option.on_card_bytes, rel=0.01)


def test_a_failed_fit_still_reports_the_width_its_runtime_would_run_at() -> None:
    """An ``IMPOSSIBLE`` verdict must not be explained with arithmetic from another runtime.

    The curve is priced at the widest card count the box offers whenever the solve fails, which is
    right for a runtime that shards and wrong for one that does not: Xing4's 18.36 GiB of resident
    weights came back as ``weights/rank 4.59 GiB`` beside a verdict that said they do not fit. The
    number the caller reads has to be the number the card would hold.
    """
    from relicllm.triage.fit import fit_precision
    from relicllm.triage.kv import SINGLE_CARD, Confidence, KvGeometry
    from relicllm.triage.weights import PrecisionOption

    geometry = KvGeometry(
        attention_kind="mla_latent",
        layers=(),
        values_per_token_per_layer=0,
        dtype_bytes=2,
        sharding=SINGLE_CARD,
        sharding_source="test",
        preallocated=True,
        confidence=Confidence.MEASURED,
    )
    option = PrecisionOption(
        label="test",
        path="/nowhere",
        format="safetensors",
        precision="I8",
        bytes_total=20 * GIB,
        on_card_bytes=20 * GIB,
        resident_on_card_bytes=20 * GIB,
        confidence="measured",
        evidence="test",
    )

    result = fit_precision(
        inventory=None,  # type: ignore[arg-type]  # the header is not read on this path
        option=option,
        geometry=geometry,
        hardware=hardware_profile_for(gpu_count=4, gpu_memory_gib=22.0, host_memory_gib=1007.0),
        context=8192,
        batch=1,
    )

    assert not result.fits
    assert result.min_cards is None
    assert result.binding_constraint == "weights"
    assert result.cards_at_width == 1
    assert result.weights_per_rank_bytes == 20 * GIB
    assert result.weights_per_rank_gib == pytest.approx(20.0, rel=1e-3)
    assert "no tensor-parallel path" in " ".join(result.notes)


def test_an_unstated_host_memory_is_not_a_default_of_zero() -> None:
    """``0`` means *unsaid*: the bank is assumed to fit and the report says it was assumed."""
    unknown = HardwareProfile(gpu_count=4, gpu_memory_gib=22.0)

    assert unknown.knows_host_memory is False
    assert unknown.host_bytes == 0
    assert HardwareProfile(host_memory_gib=1007.0).knows_host_memory is True
