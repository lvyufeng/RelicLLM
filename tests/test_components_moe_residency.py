"""The pool: which arena row a `(layer, expert)` key holds, and when it is worth staging into one.

`relicllm/components/moe/residency.py`'s `ResidentSet` is the half of the movement seam that outlives
a layer -- the arena a card keeps, the page-locked block a fill stages through, and the map that
decides which arena row a key occupies. The arena and the block need a card; the map does not, and
`pool_row` is the part of it that fails silently. A key that does not carry its layer answers one
layer's draw with another layer's bytes, at the right shape and out of a real expert, and it does so
*more* often the wider the pool is -- the bug a 512-token sweep caught only because the probe keeps
the logits.

So `pool_row` is driven here against a stub carrying only the four containers it reads, with the
shipped method bound to it unbound-style, so what is checked is the arithmetic that ships rather
than a re-spelling of it. `tests/test_models_deepseek_v4_1_tp.py` drives the same method through
`_resolve_row`, which is where it is checked against the arena rows a miss ends up naming; this file
is the contract on its own terms, and it runs on a host with no card at all.
"""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

from relicllm.components.moe.residency import ResidentSet


def _pool(*, held: dict[tuple[int, int], int], free: list[int], width: int) -> SimpleNamespace:
    """A `ResidentSet` carrying only what `pool_row` reads, with the shipped method bound to it."""
    one = SimpleNamespace(
        pool_rows=width,
        pool_staged=0,
        pool_evicted=0,
        # `free` is given in pop order, because `pool_free.pop()` takes the last.
        pool_free=[list(free)],
        pool_map=[dict(held)],
        pool_lru=[OrderedDict((key, None) for key in held)],
    )
    one.pool_row = lambda card, key: ResidentSet.pool_row(one, card, key)
    return one


def test_a_first_draw_is_admitted_and_a_second_one_is_answered() -> None:
    """The two answers that are not `None`: a free row to stage into, then the same row, no staging.

    `True` is the admission -- the row holds nothing yet and the caller must stage into it -- and
    `False` is the hit, which is the whole of what the pool buys: a draw answered by one dictionary
    probe and no H2D at all. The counters have to separate the two, because `pool_staged` is what
    says the pool is doing something while `drawn_rows - expert_rows` is what says it is worth it.
    """
    one = _pool(held={}, free=[5, 6, 7], width=3)

    assert one.pool_row(0, (4, 11)) == (7, True), (
        "a first draw was not handed a free row, so there is nowhere to stage its bytes"
    )
    assert one.pool_staged == 1 and one.pool_evicted == 0
    assert one.pool_free[0] == [5, 6], "the row handed out was not taken out of the free list"

    assert one.pool_row(0, (4, 11)) == (7, False), (
        "a second draw of a key the pool holds did not answer with the row it holds it in"
    )
    assert one.pool_staged == 1, "a hit was counted as a stage, or a hit took a second free row"
    assert one.pool_free[0] == [5, 6], "a hit consumed a free row"


def test_a_pool_of_no_rows_answers_nothing() -> None:
    """`pool_rows=0` is the off state and not a pool of width zero, and `None` is how it says so.

    A caller that gets `None` is on the path this class had before the pool existed -- the row's own
    miss rows, staged and never kept -- so the two have to be distinguishable: `(0, True)` would be
    an arena row the pool never owned.
    """
    one = _pool(held={}, free=[], width=0)
    assert one.pool_row(0, (4, 11)) is None, "a pool of no rows handed out an arena row"
    assert one.pool_staged == 0 and one.pool_evicted == 0


def test_the_key_carries_the_layer() -> None:
    """The rule the sweep bought: an expert id is only unique within its layer.

    The weights live at `layers.{layer}.ffn.experts.{expert}.{which}.weight`, so layer 5's expert 11
    and layer 6's expert 11 are different tensors that happen to share a number. Keyed on the id
    alone the pool answers the second with the first's bytes, silently, because the rows are the
    right shape and the bytes are a real expert's.
    """
    one = _pool(held={(4, 11): 7}, free=[5], width=2)

    assert one.pool_row(0, (4, 11)) == (7, False)
    assert one.pool_row(0, (5, 11)) == (5, True), (
        "another layer's expert under the same id answered this layer's draw, so the route names a "
        "row holding a different layer's tensor"
    )
    assert set(one.pool_map[0]) == {(4, 11), (5, 11)}, (
        "the two layers' experts were not kept apart, so one of them has no row of the pool"
    )


def test_a_full_pool_evicts_the_least_recently_used_row() -> None:
    """The eviction order, and the fact that a hit is a use: touch-then-fill has to displace the row
    that has gone longest without being drawn, or a key drawn every row can be evicted by one that
    is.

    The eviction is allowed at any size rather than prevented -- the row handed over is overwritten
    by an H2D the ring orders behind compute, so a kernel launched earlier cannot read it -- so what
    this pins is only *which* row goes and that the map stops claiming it.
    """
    one = _pool(held={(4, 1): 1, (4, 2): 2, (4, 3): 3}, free=[], width=3)

    # A hit on the key that was inserted *first*, so the order after it is 2, 3, 1 -- and a pool
    # that evicted the oldest insertion rather than the least recent use would take row 1 here.
    assert one.pool_row(0, (4, 1)) == (1, False)
    assert one.pool_row(0, (4, 4)) == (2, True), (
        "a full pool did not hand over the least recently used row, or a hit was not counted as a use"
    )
    assert one.pool_evicted == 1 and one.pool_staged == 1
    assert set(one.pool_map[0]) == {(4, 1), (4, 3), (4, 4)}, (
        "the evicted key survived the key that took its row"
    )
    assert list(one.pool_lru[0]) == [(4, 3), (4, 1), (4, 4)], (
        "the use order is not the least-recently-used one, so a key drawn every row can be evicted "
        "by one that is"
    )
