"""What `DeviceExpertMoE` is, decided on the host: a bank with no staging policy at all.

`HostExpertMoE` and `DeviceExpertMoE` answer the same `expert_weights` seam and run the same
`_dispatch_experts`, so every arithmetic difference between them would be a bug rather than a
design. The difference that *is* the design is placement, and it shows up in exactly two places:
where `rows()` gets its bytes, and what the staging counters read afterwards.

That second one is not decoration. `/metrics` reports `stage_bytes` / `stage_calls` /
`cache_hits` precisely so an operator can tell *residency* from *a cache that happens to hold
everything* -- and a run sized `--expert-cache 3072` at TP8 stages nothing after its first pass
over the set, so a low counter alone proves nothing. A backend on the resident path must publish
the counters and leave them at zero, and that is what the first test pins.

The rest are the seams that would silently break the mode: the dispatch must be the shared one
(so the accumulation order cannot drift), it must ask for exactly the experts the router named,
and it must refuse an expert that is not this rank's rather than reaching for a foreign row --
the round-robin placement is what `ShardedMoE` masks out, and a backend that answered anyway
would run another rank's expert on this rank's card and double-count it.

All of it is decidable on the host, so it runs here. The shard's own placement is exercised
without a card: what these tests need is an object that answers `owns()` and `rows()`, which is
the whole of the contract the backend has with it.
"""

from __future__ import annotations

import torch

from relicllm.models.qwen4_exp.moe import DeviceExpertMoE, _dispatch_experts


class _Shard:
    """This rank's expert rows, filled with the expert's own id so a read is identifiable.

    A stand-in for `HostExpertShard` under `device=...`: the same `owns` / `rows` surface, with
    `rows` answering device-placed tensors the way a resident shard's does.
    """

    def __init__(self, *, rank: int = 0, world_size: int = 4, num_experts: int = 16) -> None:
        self.rank = rank
        self.world_size = world_size
        self.num_experts = num_experts
        self.asked: list[tuple[int, int]] = []

    def owns(self, expert_id: int) -> bool:
        return expert_id % self.world_size == self.rank

    def rows(self, layer_idx: int, expert_id: int):
        self.asked.append((layer_idx, expert_id))
        # `gate_up [2*inter, hidden]` and `down [hidden, inter]` for hidden=8, inter=4 -- the
        # shapes `swiglu_expert` contracts on, so the dispatch runs rather than failing the matmul.
        return (
            torch.full((8, 8), float(expert_id), dtype=torch.float32),
            torch.full((8, 4), float(expert_id), dtype=torch.float32),
        )


def _draw(moe: DeviceExpertMoE, layer_idx: int, tokens: int = 2, top_k: int = 2):
    """A routing draw whose experts this rank owns, and the ids it named.

    Every id lands on this rank's round-robin set *and* inside `num_experts`, so the dispatch has
    nothing to mask out -- that masking is `ShardedMoE`'s job and is exercised elsewhere. `tokens
    * top_k` must not exceed the ids this rank owns, or the draw would name ids the dispatch is
    right to skip and a "read exactly these" check would be asserting the wrong set.
    """
    owned = moe.shard.num_experts // moe.shard.world_size
    assert tokens * top_k <= owned, "the draw names more experts than this rank owns"
    ids = torch.arange(tokens * top_k, dtype=torch.long)
    indices = (ids * moe.shard.world_size + moe.shard.rank).reshape(tokens, top_k)
    weights = torch.full((tokens, top_k), 1.0 / top_k, dtype=torch.float32)
    hidden = torch.ones(tokens, 8, dtype=torch.float32)
    return hidden, indices, weights


def test_the_staging_counters_are_published_and_stay_at_zero() -> None:
    """The acceptance criterion, as a property of the backend rather than of one run.

    Published *and* zero. A backend missing them would be indistinguishable from one that never
    staged at all, which is why the host path's three counters are kept here rather than dropped.
    """
    moe = DeviceExpertMoE(_Shard(), num_experts=16, device=torch.device("cpu"))

    assert (moe.stage_bytes, moe.stage_calls, moe.cache_hits) == (0, 0, 0)
    assert moe.cache_capacity == 0, "residency is not an LRU, and must not report as one"

    hidden, indices, weights = _draw(moe, layer_idx=0)
    moe(0, hidden, indices, weights)

    assert (moe.stage_bytes, moe.stage_calls, moe.cache_hits) == (0, 0, 0), (
        "a run through the resident path moved bytes through the staging path"
    )
    assert moe.cache_capacity == 0


def test_the_dispatch_is_the_shared_one_so_the_arithmetic_cannot_diverge() -> None:
    """The resident path answers the same numbers as a direct `_dispatch_experts` call.

    Both modes call the one dispatch, so an equal result here is not a coincidence to be
    re-derived per mode -- it is the seam being used as designed. The check that matters is that
    the backend *routes through* it rather than growing a loop of its own.
    """
    shard = _Shard()
    moe = DeviceExpertMoE(shard, num_experts=16, device=torch.device("cpu"))
    hidden, indices, weights = _draw(moe, layer_idx=3)
    # A second copy, because the first call's shard records what it was asked.
    reference_shard = _Shard()

    got = moe(3, hidden, indices, weights)
    want = _dispatch_experts(
        hidden,
        indices,
        weights,
        16,
        lambda expert_id: reference_shard.rows(3, expert_id),
    )

    assert torch.equal(got, want), "the resident mode changed the experts' arithmetic"
    assert shard.asked == reference_shard.asked, "the two paths read different expert rows"


def test_it_asks_for_exactly_the_experts_the_router_named() -> None:
    """No prefetch, no speculative read: the ids the draw named and nothing else.

    `HostExpertMoE` may read ahead -- that is what its staging pipeline is for. The resident path
    has nothing to hide behind, so an extra read is a bug rather than a warm-up.
    """
    # A shard wide enough that the draw names several *distinct* owned experts (64/4 = 16 owned),
    # which is what makes "exactly these and each once" a real check rather than one id in a loop.
    shard = _Shard(num_experts=64)
    moe = DeviceExpertMoE(shard, num_experts=64, device=torch.device("cpu"))
    hidden, indices, weights = _draw(moe, layer_idx=5, tokens=4, top_k=3)

    moe(5, hidden, indices, weights)

    named = sorted({int(e) for e in indices.reshape(-1)})
    assert [expert_id for _, expert_id in shard.asked] == named
    assert {layer for layer, _ in shard.asked} == {5}
    assert len(shard.asked) == len(named), "an expert was read twice in one draw"


def test_a_foreign_expert_is_refused_rather_than_read() -> None:
    """A rank reaching for another rank's row would run it here and double-count it downstream.

    `ShardedMoE` masks foreign ids out before the backend sees them, so this is the guard for the
    path that does not go through the wrapper -- a single-rank world, or a caller that assembles
    its own draw. It must name the placement rather than return a wrong tensor.
    """
    shard = _Shard(rank=1, world_size=4)
    moe = DeviceExpertMoE(shard, num_experts=16, device=torch.device("cpu"))

    foreign = next(e for e in range(16) if not shard.owns(e))
    try:
        moe.expert_weights(0, foreign)
    except ValueError as exc:
        assert str(foreign) in str(exc), "the error must name the expert it refused"
        assert f"% {shard.world_size}" in str(exc), "and the rule the placement follows"
    else:
        raise AssertionError("a foreign expert was answered instead of refused")
    assert shard.asked == [], "the refusal happened before any row was read"