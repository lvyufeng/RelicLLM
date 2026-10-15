"""The staging ring: which slot a row's copy goes into, and when that slot may be reused.

`relicllm/components/moe/movement.py` is the path a routed expert's bytes take, and `StagingRing` is
the half of it that fails silently. A ring that reuses a slot before the DMA that filled it has
drained reads bytes that are not there yet; a ring that advances on a row which moved *nothing*
lands on a slot with no upload in front of it, and on a prefill the pool answers most rows without
moving anything, so that second failure is one a hit rate turns from luck into a regular loss. Both
are invisible in any counter and both are decidable on the host, which is why they are pinned here
and not on a card: the ring names two things -- its own events and `copy_stream` -- and this module
hands it both.

The ring is therefore driven for real, with only the CUDA entry points it names replaced by objects
that record what was asked of them. That is the same style the v41 tests use for the same reason,
and it is the reason `StagingRing` takes its events as an argument rather than allocating them.
"""

from __future__ import annotations

import contextlib

import torch

from relicllm.components.moe import movement
from relicllm.components.moe.movement import StagingRing


class _Event:
    """A slot's event, as `stage` and `load` leave it and `take` asks about it."""

    def __init__(self) -> None:
        self.recorded = 0
        self.waits = 0

    def record(self, stream: object) -> None:
        self.recorded += 1

    def synchronize(self) -> None:
        self.waits += 1


class _Stream:
    """The copy stream, seen only through what the ring does to it: order behind compute."""

    def __init__(self) -> None:
        self.waited: list[object] = []

    def wait_stream(self, other: object) -> None:
        self.waited.append(other)


def _harness(monkeypatch, *, slots: int, cards: int):
    """A ring on cards that are not there: recording events, and one recording copy stream a card."""
    events = [[_Event() for _ in range(slots)] for _ in range(cards)]
    streams: dict[object, _Stream] = {}
    monkeypatch.setattr(
        movement, "copy_stream", lambda device: streams.setdefault(device, _Stream())
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: "compute")
    devices = [torch.device("cpu") for _ in range(cards)]
    ring = StagingRing(slots=slots, devices=devices, events=events)
    return ring, events, streams


def _copies(pairs: int) -> list[list[tuple[torch.Tensor, torch.Tensor]]]:
    """`pairs` copies into one row of one card: a destination view and a source of its shape."""
    destination = torch.zeros((1, 8), dtype=torch.uint8)
    return [
        [
            (destination, torch.full((1, 8), i + 1, dtype=torch.uint8))
            for i in range(pairs)
        ]
    ]


def test_a_caller_that_names_its_slot_rotates_the_ring_once_a_row(monkeypatch) -> None:
    """`stage(copies, slot)`: the take is the caller's, and a row costs one rotation and not two.

    Both of the banks in this tree take a slot before they stage -- one to name its destination rows
    out of the slot's band, one to park a row's ids under the slot -- so `stage` and `take` are
    called by the same row one after the other. If the staging call took a slot of its own the row
    would rotate twice, and at the shipped two slots the *next* row's take would then hand out the
    slot this row had just filled and wait on the DMA it had just issued: the copy would be awaited
    the instant it was issued, which is the failure that reads as a mystery stall in a class wall and
    shows up in no counter at all.

    So the pair is pinned here: the copies land in the slot named and not another, the rotation
    advanced exactly once, and over three rows a slot's second use waits on the row in between.
    """
    ring, events, _ = _harness(monkeypatch, slots=2, cards=1)

    buffer = ring.take()
    assert buffer == 0 and ring._next == 1
    assert ring.stage(_copies(2), buffer) == 2, "the return is still the pairs moved"
    assert ring._next == 1, (
        "the staging call advanced the rotation after the caller had already taken a slot, so this "
        "row rotated twice and the next one is handed the slot it just filled"
    )
    assert ring._in_flight[0] == [True, False], (
        "the copies were flagged in flight on a slot the caller never named"
    )
    assert [event.recorded for event in events[0]] == [1, 0], (
        "the slot's event was recorded out of the slot the copies went into, so `take` will wait "
        "on one slot's DMA while the other's rows are the ones that moved"
    )

    # Three rows taken-then-staged, which is the shipped row loop. The rotation visits 0, 1, 0 and
    # the third row reuses slot 0, one whole working row after the copy into it -- so each slot's
    # second use is the first that waits, and no row waits on its own copy.
    ring, events, _ = _harness(monkeypatch, slots=2, cards=1)
    for _ in range(3):
        ring.stage(_copies(1), ring.take())
    assert ring._next == 1, (
        "three rows left the rotation at 1 (three takes modulo two) -- anywhere else is a second "
        "take inside a row"
    )
    assert [event.recorded for event in events[0]] == [2, 1], "the slots were not filled round-robin"
    assert [event.waits for event in events[0]] == [1, 0], (
        "slot 0 was reused by the third row with no wait on the copy the first row issued through it"
    )


def test_a_row_that_moves_nothing_neither_takes_a_slot_nor_waits(monkeypatch) -> None:
    """`stage`'s guard, which is the one property of the ring a hit rate can turn into a loss.

    `take` hands out the next slot *and waits on the DMA that last filled it*, so the guarantee that
    a slot being reused has had a whole row's copy issued since its last one only holds if every row
    that takes a slot also moves bytes with it. A pool answering 86.3% of a prefill's rows leaves a
    rotation advancing through rows that move nothing, and the slot it then hands out is the one the
    previous *working* row filled immediately before -- measured at 6.89 s of a 30.35 s class wall on
    that leg, against the ~2.5 s of DMA it covered.

    So an empty staging call has to leave the rotation and every event exactly where it found them,
    however many times it is asked. Unlike the rest of this file this is not about ordering -- the
    copies are ordered by the streams either way -- but about the ring's own state.
    """
    ring, events, _ = _harness(monkeypatch, slots=2, cards=1)

    for _ in range(8):
        assert ring.stage([[]]) == 0, "an empty staging call claimed to have moved pairs"

    assert ring._next == 0, (
        "the rotation advanced through rows that moved nothing, so the slot it will hand out next "
        "is one the previous working row filled with no upload in between to cover its copy"
    )
    assert [event.recorded for event in events[0]] == [0, 0], (
        "an empty staging call recorded a slot's event, which would make the next `take` wait on a "
        "copy that was never issued"
    )
    assert [event.waits for event in events[0]] == [0, 0], (
        "an empty staging call waited on a slot it never took"
    )


def test_the_rotation_advances_on_moving_rows_and_waits_a_whole_row_later(monkeypatch) -> None:
    """The rotation is round-robin over the rows that move, and a slot is owed exactly one of them.

    This is the guarantee the guard above buys, stated as the sequence it produces: two slots, a
    working row of one pair, three empty rows, then a working row of three. The rotation has to move
    twice and only twice, the two slots' events each get one record, and no wait is owed yet -- the
    third working row is the one that reuses slot 0, and it is the first that has to wait.
    """
    ring, events, _ = _harness(monkeypatch, slots=2, cards=1)

    assert ring.stage(_copies(1)) == 1, "the return is the pairs moved and not the rows staged"
    assert ring._next == 1

    for _ in range(3):
        ring.stage([[]])
    assert ring._next == 1, "the rotation moved over rows that staged nothing"

    assert ring.stage(_copies(3)) == 3
    assert ring._next == 0
    assert [event.recorded for event in events[0]] == [1, 1], (
        "a slot was filled twice, or a working row filled a slot it had not taken"
    )
    assert [event.waits for event in events[0]] == [0, 0], (
        "a working row waited on a slot that a whole earlier working row had already covered"
    )

    assert ring.stage(_copies(1)) == 1
    assert ring._next == 1
    assert [event.waits for event in events[0]] == [1, 0], (
        "the third working row did not wait on the slot it reused, so its copy can overwrite bytes "
        "the previous row's DMA is still reading"
    )


def test_a_row_stages_only_the_cards_it_was_dealt(monkeypatch) -> None:
    """One slot for every card, and one event per card: a card that moved nothing is not waited on.

    A row is one row and its cards are staged together, so the slot index is the same on all of
    them; but a deal hands a row's experts to some cards and not others, and the in-flight flag and
    the event are a pair *per card*. A ring that waited on every card's slot would make four
    independent links wait on each other's DMA, which is the serialization the split exists to avoid.
    """
    ring, events, _ = _harness(monkeypatch, slots=1, cards=2)

    ring.stage([_copies(1)[0], []])
    assert [cards[0].recorded for cards in events] == [1, 0], (
        "a card that held none of the row's misses recorded the slot's event anyway"
    )
    assert ring.take() == 0
    assert [cards[0].waits for cards in events] == [1, 0], (
        "the second card was made to wait on a copy it never issued"
    )

    # And the other way round, so the first card's empty slot is not simply never watched: with
    # card 1 in flight and card 0 empty, the next `take` waits on card 1 alone.
    ring, events, _ = _harness(monkeypatch, slots=1, cards=2)
    ring.stage([[], _copies(1)[0]])
    ring.take()
    assert [cards[0].waits for cards in events] == [0, 1]


def test_the_fill_is_one_transfer_a_tensor_behind_compute(monkeypatch) -> None:
    """`load`, the fill's half of the seam, which is the same ordering applied to a block of rows.

    The differences from `stage` are both facts about the fill and both are asserted here: the copies
    land (`copy_` is a real host copy on this stub) and there is one per tensor rather than one per
    miss row, and the event recorded is the one the *caller* passed in -- the set's, not a slot's --
    because what it orders is the next layer's host write against this layer's DMA.
    """
    ring, _, streams = _harness(monkeypatch, slots=1, cards=1)

    block = torch.zeros((4, 8), dtype=torch.uint8)
    source = torch.arange(32, dtype=torch.uint8).reshape(4, 8)
    resident = torch.zeros((6, 8), dtype=torch.uint8)
    fill_event = _Event()

    ring.load(0, [(resident[:3], source[:3])], fill_event)

    assert resident[:3].tolist() == source[:3].tolist(), (
        "the fill did not land, so the layer's resident rows would hold whatever they held before"
    )
    assert int(resident[3:].abs().sum()) == 0, "the fill wrote past the rows it was handed"
    assert fill_event.recorded == 1, (
        "the fill recorded no event, so the next layer's host write into the shared block is "
        "unordered against this layer's DMA reading it"
    )
    assert streams[torch.device("cpu")].waited == ["compute"], (
        "the fill rewrites arena rows the previous layer's kernels read and was not put behind them"
    )
