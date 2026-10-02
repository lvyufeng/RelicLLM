"""Gate 0: does it fit, at which precision, on how many cards, holding how much context and batch.

Three inputs and one budget. The inputs are the checkpoint's resident weights (from the header),
what one token costs in KV at the width the runtime will actually run (from the allocation site),
and the box's cards and host memory. The budget is ``per-card bytes x (1 - reserve)``, and the
answer is the *curve* the three leave rather than a single pair of numbers.

**What this gate may and may not say.** A failure here is definitive: the arithmetic has no
workspace, no fragmentation and no bank-miss penalty to be wrong about, so anything that fails on
these numbers fails harder in reality. A *pass* is only provisional for the same reason -- the
measured Xing4 runs sit at 17.94 GiB of a 19.8 GiB budget with 0.65 to 1.86 GiB spare, and the
difference between that and a crash is exactly the workspace this calculation does not model. So a
pass routes to ``CANDIDATE`` and never to a tier.

**The host memory is a precondition, not a rounding term.** Ruling: a model whose routed experts and
gather tables live in host RAM is *placed*. But an offload plan needs somewhere to put the bank, and
the bank is often larger than the cards -- DeepSeek-V4.1 is 464.5 GiB offloadable against 88 GiB of
card, so if the host cannot hold it the excess comes back onto the cards and changes the card count.
Nothing here assumes 1007 GiB: ``HardwareProfile.host_memory_gib`` defaults to 0, which is read as
*unsaid*, and an unsaid host turns the offload into a note rather than a silent assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from relicllm.components.moe.placement import (
    ContextBatchCurve,
    HardwareProfile,
    context_batch_curve,
    estimate_even_shard,
    minimum_cards,
    usable_per_card,
)
from relicllm.triage.checkpoint import CheckpointInventory
from relicllm.triage.kv import (
    REPLICATED,
    SINGLE_CARD,
    KvGeometry,
    bytes_per_token_per_rank,
)
from relicllm.triage.weights import PrecisionOption

__all__ = [
    "DEFAULT_CONTEXTS",
    "DEFAULT_BATCHES",
    "FitResult",
    "fit_all",
    "fit_precision",
    "hardware_profile_for",
]


DEFAULT_CONTEXTS = (2048, 4096, 8192, 32768, 131072)
DEFAULT_BATCHES = (1, 2, 4, 8)


@dataclass(frozen=True)
class FitResult:
    """One precision's fit, with every number that produced it still attached."""

    label: str
    precision: str

    fits: bool
    min_cards: int | None
    """``None`` means it does not fit at any card count this box has -- the one definitive answer."""

    cards_at_width: int
    """The card count this fit was priced at, which is :attr:`min_cards` when there is one."""

    weights_per_rank_bytes: int
    kv_bytes_per_token_per_rank: int

    curve: ContextBatchCurve
    context: int
    batch: int

    binding_constraint: str
    """``weights`` | ``kv`` | ``host_bank`` | ``none`` -- which budget ran out first."""

    host_memory_gib: float
    host_bank_gib: float
    """Offloadable bytes that must be reachable in host memory for the card count to hold."""

    confidence: str
    basis: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)
    unknown: tuple[str, ...] = field(default_factory=tuple)

    @property
    def weights_per_rank_gib(self) -> float:
        return self.weights_per_rank_bytes / 1024 ** 3


def hardware_profile_for(
    *,
    gpu_count: int = 4,
    gpu_memory_gib: float = 22.0,
    host_memory_gib: float = 0.0,
    name: str = "consumer-gpu-box",
) -> HardwareProfile:
    """The box, with the card capacity the recorded runs actually used.

    ``22.0`` GiB and not ``22528 / 1024 = 22.0``: the cards report 22,528 MiB, and the whole recorded
    roster -- every threshold this package's gate 3 uses -- was measured against the amount of that
    a process can allocate after the driver takes its share. Using the raw MiB figure here would move
    every fit by about 0.4 GiB in the optimistic direction and quietly re-place Xing4, whose margin
    is smaller than that.
    """
    return HardwareProfile(
        gpu_count=gpu_count, gpu_memory_gib=gpu_memory_gib, host_memory_gib=host_memory_gib, name=name
    )


def _host_bank_fits(
    inventory: CheckpointInventory, option: PrecisionOption, hardware: HardwareProfile, *, host_reserve_fraction: float
) -> tuple[bool, int]:
    """Whether the offload bank fits in host memory, and what comes back onto the cards if not.

    Returns ``(fits, spill_bytes)``, where the spill is over the **bank**, not over the whole
    on-card total. The distinction matters and is easy to get wrong: ``resident + on_card``
    double-counts the resident part, and the double count is the difference between Xing4 fitting
    and Xing4 being called impossible. The spill is what a caller adds to the resident weights,
    because that is what it is -- bytes with nowhere else to be.
    """
    bank = option.on_card_bytes - option.resident_on_card_bytes
    if not hardware.knows_host_memory:
        return True, 0
    usable = int(hardware.host_bytes * (1.0 - float(host_reserve_fraction)))
    if bank <= usable:
        return True, 0
    return False, int(bank - usable)


def fit_precision(
    *,
    inventory: CheckpointInventory,
    option: PrecisionOption,
    geometry: KvGeometry,
    hardware: HardwareProfile,
    context: int = 8192,
    batch: int = 1,
    reserve_fraction: float = 0.15,
    host_reserve_fraction: float = 0.10,
    workspace_bytes: int = 0,
    contexts: tuple[int, ...] = DEFAULT_CONTEXTS,
    batches: tuple[int, ...] = DEFAULT_BATCHES,
) -> FitResult:
    """Price one precision of one checkpoint against one box.

    The card count is solved at ``(context, batch)`` and then the curve is reported at *that* card
    count, because the per-rank KV depends on how many ranks there are. Reporting a curve from a
    different width than the card count would be two answers to two questions.

    **There is no separate width parameter, and there used to be one that did nothing.** ``--tp-width``
    was accepted by the CLI, printed by the report, and read into a local that no line ever used: the
    width is the card count and always was. A knob that silently ignores its argument is worse than
    no knob, because it is the argument a reader trusts when the number looks wrong. Callers who want
    a particular width read :attr:`FitResult.cards_at_width`, and the probe command the report prints
    passes it to ``--tensor-parallel-size``.
    """
    notes: list[str] = []
    unknown: list[str] = []

    shards = geometry.sharding not in (REPLICATED, SINGLE_CARD)

    # What has to be resident every step, and what a host bank can hold.
    resident = option.resident_on_card_bytes
    bank = option.on_card_bytes - resident
    bank_fits, spill = _host_bank_fits(
        inventory, option, hardware, host_reserve_fraction=host_reserve_fraction
    )
    if not hardware.knows_host_memory and bank:
        unknown.append(
            f"{bank / 1024 ** 3:.1f} GiB of routed experts and tables need host memory and this "
            "profile does not state how much there is; the bank is assumed to fit"
        )
    if not bank_fits:
        notes.append(
            f"the offload bank is {bank / 1024 ** 3:.1f} GiB against "
            f"{hardware.host_memory_gib * (1 - host_reserve_fraction):.1f} GiB of usable host memory, "
            f"so {spill / 1024 ** 3:.1f} GiB comes back onto the cards"
        )

    weights_bytes = resident + spill
    # `SINGLE_CARD` is the "this runtime has no tensor-parallel path" signal, and it has to gate the
    # *weights* as well as the KV. Xing4 refuses `--tensor-parallel-size > 1`, so its fully resident
    # 18.72 GiB must fit on one card; dividing it across four and reporting three would be a fit for
    # a runtime that does not exist.
    splits = geometry.sharding != SINGLE_CARD
    cards = minimum_cards(
        weights_bytes=weights_bytes,
        kv_bytes_per_token_per_rank_at_one_card=bytes_per_token_per_rank(geometry, tp_width=1),
        context=context,
        batch=batch,
        hardware=hardware,
        reserve_fraction=reserve_fraction,
        workspace_bytes=workspace_bytes,
        sharded=shards,
        splits_weights=splits,
    )

    # A failed solve has no card count to price the curve at, so it is priced at the widest the box
    # offers -- but only where the runtime can use that width. A single-card runtime priced at
    # ``gpu_count`` prints a four-way shard of weights it will never split; that is how 18.36 GiB of
    # resident weights once came to be reported as 4.59 GiB next to an IMPOSSIBLE verdict.
    cards_at_width = cards or (int(hardware.gpu_count) if splits else 1)
    curve = context_batch_curve(
        weights_per_rank_bytes=estimate_even_shard(weights_bytes, cards_at_width),
        kv_bytes_per_token_per_rank=(
            bytes_per_token_per_rank(geometry, tp_width=cards_at_width) if shards else bytes_per_token_per_rank(geometry, tp_width=1)
        ),
        hardware=hardware,
        reserve_fraction=reserve_fraction,
        workspace_bytes=workspace_bytes,
        contexts=contexts,
        batches=batches,
    )

    if cards is None:
        # `curve.binds` rather than a second reading of `headroom_bytes`: it knows about the
        # workspace, and this branch is the one where mislabeling the binding is most misleading --
        # a fit that failed on a 12 GiB scratch allocation was reporting `kv`, and the note under it
        # then blamed a KV that was 0.13 GiB.
        binding = curve.binds
        if not bank_fits:
            binding = "host_bank"
        usable = usable_per_card(hardware, reserve_fraction=reserve_fraction)
        if binding == "workspace":
            # Per rank on both sides, which is how the budget is tested: the number in `weights/rank`
            # on the row above is the same one, and quoting the whole checkpoint here would read as
            # a card holding 28.7 GiB of a 18.7 GiB budget.
            shortfall = (
                f"{curve.weights_per_rank_bytes / 1024 ** 3:.1f} GiB of weights plus the "
                f"{workspace_bytes / 1024 ** 3:.1f} GiB workspace is over the "
                f"{usable / 1024 ** 3:.1f} GiB one card has, so no context fits at any length"
            )
        elif splits:
            shortfall = (
                f"the even shard of {weights_bytes / 1024 ** 3:.1f} GiB plus its KV exceeds "
                f"{usable / 1024 ** 3:.1f} GiB per card at every one of the {hardware.gpu_count} cards"
            )
        else:
            shortfall = (
                f"all {weights_bytes / 1024 ** 3:.1f} GiB has to be resident, this runtime has no "
                f"tensor-parallel path to divide it, and with its KV it is over the "
                f"{usable / 1024 ** 3:.1f} GiB one card has"
            )
        notes.append(f"at {context} tokens and batch {batch} {shortfall}")
    else:
        binding = curve.binds

    return FitResult(
        label=option.label,
        precision=option.precision,
        fits=cards is not None,
        min_cards=cards,
        cards_at_width=cards_at_width,
        weights_per_rank_bytes=curve.weights_per_rank_bytes,
        kv_bytes_per_token_per_rank=curve.kv_bytes_per_token_per_rank,
        curve=curve,
        context=int(context),
        batch=int(batch),
        binding_constraint=binding,
        host_memory_gib=float(hardware.host_memory_gib),
        host_bank_gib=bank / 1024 ** 3,
        confidence=str(option.confidence),
        basis=option.basis + (f"reserve fraction {reserve_fraction:g}",),
        notes=tuple(notes),
        unknown=tuple(unknown) + option.unknown,
    )


def fit_all(
    *,
    inventory: CheckpointInventory,
    options: tuple[PrecisionOption, ...],
    geometry: KvGeometry,
    hardware: HardwareProfile,
    **kwargs: object,
) -> tuple[FitResult, ...]:
    """Every precision of one checkpoint, in the order the ladder gave them.

    Ordered rather than filtered: a fit report whose best precision is dropped because a worse one
    passed is not a report. The tier is decided from all of them together -- "fits, but only at FP4
    on four cards" and "fits natively on one" are different answers and only the set says which.
    """
    return tuple(
        fit_precision(
            inventory=inventory,
            option=option,
            geometry=geometry,
            hardware=hardware,
            **kwargs,  # type: ignore[arg-type]
        )
        for option in options
    )
