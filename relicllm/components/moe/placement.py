from __future__ import annotations

from dataclasses import dataclass

from relicllm.components.moe.spec import PlacementDecision


@dataclass(frozen=True)
class HardwareProfile:
    gpu_count: int = 4
    gpu_memory_gib: float = 22.0

    host_memory_gib: float = 0.0
    """RAM the host can actually lend to an offload bank. ``0`` means *unsaid*, not *none*.

    The distinction is load-bearing and is why the default is zero rather than a plausible number.
    A model whose routed experts live in host memory is placed, not unplaced -- that is the whole
    point of the 137-458 GiB banks this runtime stages from. But an offload plan is only a plan if
    there is memory to hold it, and inventing a default here would let a fit pass on a machine that
    has nowhere to put the bank. Callers that know the number set it; callers that do not get the
    note that says so rather than a guess wearing a default's clothes.
    """

    name: str = "consumer-gpu-box"

    @property
    def total_gpu_bytes(self) -> int:
        return int(self.gpu_count * self.gpu_memory_gib * (1024 ** 3))

    @property
    def per_gpu_bytes(self) -> int:
        return int(self.gpu_memory_gib * (1024 ** 3))

    @property
    def host_bytes(self) -> int:
        return int(self.host_memory_gib * (1024 ** 3))

    @property
    def knows_host_memory(self) -> bool:
        return self.host_memory_gib > 0


def estimate_even_shard(total_bytes: int, gpu_count: int) -> int:
    if gpu_count <= 0:
        return int(total_bytes)
    return (int(total_bytes) + int(gpu_count) - 1) // int(gpu_count)


def usable_per_card(hardware: HardwareProfile, *, reserve_fraction: float = 0.15) -> int:
    """What one card can hold after the reserve.

    The reserve is a *parameter*, not a constant, and the difference between the two values in use
    decides real outcomes: 15% leaves 18.7 GiB of a 22 GiB card, 10% leaves 19.8 GiB, and Xing4's
    17.94 GiB of resident weights is the difference between 0.65 and 1.86 GiB of headroom -- about
    4,500 against 13,000 tokens of context at batch 1. A report must print which one it used.
    """
    return int(hardware.per_gpu_bytes * (1.0 - float(reserve_fraction)))


@dataclass(frozen=True)
class ContextBatchPoint:
    """One ``(context, batch)`` pair and the KV it costs on one rank."""

    context: int
    max_batch: int
    kv_bytes_per_rank: int
    headroom_bytes: int

    @property
    def total_tokens(self) -> int:
        return int(self.context) * int(self.max_batch)


@dataclass(frozen=True)
class ContextBatchCurve:
    """The context and batch a fixed budget can hold, which are one budget and not two numbers.

    A single ``(L_max, B_max)`` pair is a half-answer. The card holds
    ``weights + L * B * kv_bytes_per_token``, so raising the batch lowers the context and the two
    trade off exactly; Xing4 -- 17.94 GiB resident of a 22 GiB card -- can serve one person reading
    43k tokens or eight people chatting in 5.4k, and cannot do both. Which of those it should do is a
    product decision, so the tool reports the curve and lets the caller pick the point.
    """

    points: tuple[ContextBatchPoint, ...]
    weights_per_rank_bytes: int
    kv_bytes_per_token_per_rank: int
    usable_per_rank_bytes: int
    reserve_fraction: float
    workspace_bytes: int = 0

    def at(self, context: int) -> ContextBatchPoint | None:
        for point in self.points:
            if point.context == context:
                return point
        return None

    def max_context_at(self, batch: int) -> int:
        """The largest context on the curve that this batch has room for; 0 when none does."""
        fitting = [
            point.context for point in self.points
            if point.max_batch >= batch and point.headroom_bytes >= 0
        ]
        return max(fitting, default=0)

    @property
    def headroom_bytes(self) -> int:
        """What is left once weights and the smallest point's KV are paid -- can be negative."""
        return self.usable_per_rank_bytes - self.weights_per_rank_bytes

    @property
    def budget_bytes(self) -> int:
        """What the KV may spend: usable, less the weights, less the workspace.

        The same line every point in :attr:`points` is measured against, which is why
        :attr:`headroom_bytes` is *not* it -- that property answers "do the weights fit at all", a
        question the workspace has no part in.
        """
        return self.usable_per_rank_bytes - self.weights_per_rank_bytes - self.workspace_bytes

    @property
    def binds(self) -> str:
        """Which of the three budgets is tight: the weights alone, the workspace, or weights plus KV.

        ``workspace`` is its own answer rather than folded into ``kv``, because the two name
        different work: a KV-bound fit is asking for a shorter context, and a workspace-bound one is
        asking for a smaller scratch allocation and will not move at any context at all. It is
        reported when the workspace is what pushed the budget for KV negative -- i.e. when there is
        no context, however short, that could fit.
        """
        if self.headroom_bytes < 0:
            return "weights"
        if self.points and all(point.headroom_bytes < 0 for point in self.points):
            if self.budget_bytes < 0:
                return "workspace"
            return "kv"
        return "none"


def context_batch_curve(
    *,
    weights_per_rank_bytes: int,
    kv_bytes_per_token_per_rank: int,
    hardware: HardwareProfile,
    reserve_fraction: float = 0.15,
    workspace_bytes: int = 0,
    contexts: tuple[int, ...] = (2048, 4096, 8192, 32768, 131072),
    batches: tuple[int, ...] = (1, 2, 4, 8),
) -> ContextBatchCurve:
    """Solve ``weights + L * B * kv <= usable`` over a grid, rather than for one pair.

    ``kv_bytes_per_token_per_rank`` already carries the sharding rule: a replicated cache does not
    shrink as cards are added, so for those models more cards buy weights only. That is the whole
    reason the number is per *rank* here and not per model.
    """
    usable = usable_per_card(hardware, reserve_fraction=reserve_fraction)
    budget = usable - int(weights_per_rank_bytes) - int(workspace_bytes)
    points = tuple(
        ContextBatchPoint(
            context=int(context),
            max_batch=int(batch),
            kv_bytes_per_rank=int(context) * int(batch) * int(kv_bytes_per_token_per_rank),
            headroom_bytes=int(budget) - int(context) * int(batch) * int(kv_bytes_per_token_per_rank),
        )
        for context in contexts
        for batch in batches
    )
    return ContextBatchCurve(
        points=points,
        weights_per_rank_bytes=int(weights_per_rank_bytes),
        kv_bytes_per_token_per_rank=int(kv_bytes_per_token_per_rank),
        usable_per_rank_bytes=usable,
        reserve_fraction=float(reserve_fraction),
        workspace_bytes=int(workspace_bytes),
    )


def minimum_cards(
    *,
    weights_bytes: int,
    kv_bytes_per_token_per_rank_at_one_card: int,
    context: int,
    batch: int,
    hardware: HardwareProfile,
    reserve_fraction: float = 0.15,
    workspace_bytes: int = 0,
    sharded: bool = True,
    splits_weights: bool = True,
) -> int | None:
    """The fewest cards this fits on at one ``(context, batch)``, or ``None`` for "not at all".

    Weights split evenly (:func:`estimate_even_shard`) **when the runtime has more than one rank**;
    KV splits only if the runtime shards it. ``splits_weights=False`` is for a runtime with no
    tensor-parallel path at all -- Xing4 refuses ``--tensor-parallel-size > 1`` -- and it is not a
    detail: without it, Xing4's 18.72 GiB of fully resident weights divide across four cards and
    "fit" on a runtime that would run them on one.

    ``None`` is the definitive answer the static gate is allowed to give. Adding cards only ever
    lowers the weights term, so failing at ``gpu_count`` cards fails at every larger count too -- and
    the count is capped by what the box physically has.
    """
    if hardware.gpu_count <= 0:
        return None
    for cards in range(1, int(hardware.gpu_count) + 1):
        per_rank_weights = estimate_even_shard(weights_bytes, cards) if splits_weights else int(weights_bytes)
        kv = int(context) * int(batch) * int(kv_bytes_per_token_per_rank_at_one_card)
        per_rank_kv = -(-kv // cards) if sharded else kv
        if per_rank_weights + per_rank_kv + int(workspace_bytes) <= usable_per_card(
            hardware, reserve_fraction=reserve_fraction
        ):
            return cards
    return None


def lowbit_device_resident_decision(total_bytes: int, hardware: HardwareProfile, *, reserve_fraction: float = 0.15) -> PlacementDecision:
    per_gpu = estimate_even_shard(total_bytes, hardware.gpu_count)
    usable = usable_per_card(hardware, reserve_fraction=reserve_fraction)
    if per_gpu <= usable:
        return PlacementDecision(
            name="all_device_lowbit",
            status="candidate",
            reason=f"even sharding uses {per_gpu} bytes/GPU within reserved budget {usable} bytes/GPU",
            estimated_bytes=int(total_bytes),
            estimated_bytes_per_gpu=per_gpu,
        )
    return PlacementDecision(
        name="all_device_lowbit",
        status="deferred",
        reason=f"even sharding uses {per_gpu} bytes/GPU above reserved budget {usable} bytes/GPU",
        estimated_bytes=int(total_bytes),
        estimated_bytes_per_gpu=per_gpu,
    )


def heterogeneous_expert_decision(routed_bytes: int) -> PlacementDecision:
    return PlacementDecision(
        name="heterogeneous_routed_experts",
        status="candidate",
        reason="routed experts can be kept in CPU pinned/NUMA memory and staged by active routes when all-device placement is not practical",
        estimated_bytes=int(routed_bytes),
        estimated_bytes_per_gpu=None,
    )
