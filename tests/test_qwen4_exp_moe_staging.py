"""Which stream carries a Qwen4-Exp expert's bytes, and what is ordered against what.

`HostExpertMoE` stages the experts a draw needs out of this rank's host shard into freshly
allocated device tensors, on the copy stream `components/moe/movement.py` owns -- the same H2D
stream every other bank in this tree stages on -- and not on one of its own. That is this bank's
whole share of the movement layer, and it is deliberately not the arena path: a staged expert is a
new block that an LRU holds until it evicts it, not a row of a rotating arena, so a copy overwrites
nothing a kernel may still be reading and the ordering is one edge instead of two.

That difference is what these tests are about, because it is invisible in every counter. A miss is
copied on the movement layer's stream and records one event there; the compute stream waits on that
event. A hit stages nothing and the compute stream waits on nothing, which is correct rather than
missing -- the hit's block was put behind the same stream by the earlier draw's event, and the
compute stream is a FIFO. And nothing ever makes the copy stream wait on compute, which is the wait
the arena path does have and this one must not grow back: it would serialize every expert's copy
behind the kernel that read the last one.

All of it is decidable on the host, so it is pinned here in the style the movement tests use. The
CUDA entry points this path names -- `movement.copy_stream`, `torch.cuda.Stream`, `.stream`,
`.current_stream` and `.Event` -- are replaced by objects that record what is asked of them, and the
shipped `_prefetch` is driven. `_record_stream` is replaced as well: `Tensor.record_stream` is the
caching allocator's bookkeeping, it cannot be asked of a CPU block, and the call it stands for is a
stream edge on a card rather than a fact about this path.
"""

from __future__ import annotations

import contextlib

import torch

from relicllm.components.moe import movement
from relicllm.models.qwen4_exp.moe import HostExpertMoE


class _Event:
    """A staging event: `record` says which stream the copy it stands for was issued on."""

    def __init__(self) -> None:
        self.streams: list[object] = []

    def record(self, stream: object) -> None:
        self.streams.append(stream)


class _Stream:
    """The copy stream, seen through the one thing the arena path would ask of it."""

    def __init__(self) -> None:
        self.waited: list[object] = []

    def wait_stream(self, other: object) -> None:
        self.waited.append(other)


class _Compute:
    """The compute stream: what `.current_stream` names and what waits on the staging events."""

    def __init__(self) -> None:
        self.waited: list[_Event] = []

    def wait_event(self, event: _Event) -> None:
        self.waited.append(event)


class _Reader:
    """`expert_rows` as this bank reads it: a host pair an expert, filled with the expert's own id."""

    def __init__(self) -> None:
        self.asked: list[tuple[int, int]] = []

    def expert_rows(self, layer_idx: int, expert_id: int):
        self.asked.append((layer_idx, expert_id))
        return (
            torch.full((4, 8), float(expert_id), dtype=torch.float32),
            torch.full((8, 4), float(expert_id), dtype=torch.float32),
        )


def _harness(monkeypatch, *, cache_capacity: int = 0):
    """A `HostExpertMoE` on a machine with no card, and what the path asked its streams."""
    streams: dict[object, _Stream] = {}
    events: list[_Event] = []
    compute = _Compute()
    entered: list[object] = []

    @contextlib.contextmanager
    def opened(stream):
        entered.append(stream)
        yield

    def no_stream(*args, **kwargs):
        raise AssertionError("HostExpertMoE built a stream of its own")

    monkeypatch.setattr(
        movement, "copy_stream", lambda device: streams.setdefault(device, _Stream())
    )
    monkeypatch.setattr(torch.cuda, "Stream", no_stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: events.append(_Event()) or events[-1])
    monkeypatch.setattr(torch.cuda, "stream", opened)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: compute)
    monkeypatch.setattr(HostExpertMoE, "_record_stream", staticmethod(lambda weight, stream: None))

    reader = _Reader()
    device = torch.device("cpu")
    moe = HostExpertMoE(
        reader, 16, device, torch.float32, cache_capacity=cache_capacity
    )
    return moe, reader, streams, events, compute, entered, device


def test_a_miss_is_copied_on_the_movement_stream_and_only_the_compute_stream_waits(
    monkeypatch,
) -> None:
    """One stream a device from the movement layer, one event a miss on it, and one waiter."""
    moe, reader, streams, events, compute, entered, device = _harness(monkeypatch)

    staged = list(moe._prefetch(2, [3, 5, 7]))

    stream = streams[device]
    assert reader.asked == [(2, 3), (2, 5), (2, 7)]
    assert list(streams) == [device], "the stream is keyed by device and not by this instance"
    assert entered == [stream], "the copies were issued inside the copy stream's context"
    assert stream.waited == [], "a copy that waits on compute is the arena path's edge, not this one"
    assert [event.streams for event in events] == [[stream]] * 3
    assert compute.waited == events, "every miss is waited on, and in the order it was copied"

    gate_up, down = staged[0]
    assert torch.equal(gate_up, torch.full((4, 8), 3.0)), "a staged block holds its own expert"
    assert torch.equal(down, torch.full((8, 4), 3.0))
    assert moe.stage_calls == 3
    assert moe.stage_bytes == 3 * (4 * 8 + 8 * 4) * 4


def test_a_hit_stages_nothing_and_the_compute_stream_waits_for_nothing(monkeypatch) -> None:
    """The second draw of the same experts is the same blocks, copied once and waited on once."""
    moe, reader, streams, events, compute, entered, device = _harness(monkeypatch, cache_capacity=4)

    first = list(moe._prefetch(2, [3, 5]))
    after_misses = (len(events), list(compute.waited), moe.stage_calls, moe.stage_bytes)
    second = list(moe._prefetch(2, [3, 5]))

    assert reader.asked == [(2, 3), (2, 5)], "a hit does not go back to the host"
    assert (len(events), compute.waited, moe.stage_calls, moe.stage_bytes) == after_misses
    assert moe.cache_hits == 2
    assert [block[0].data_ptr() for block in second] == [block[0].data_ptr() for block in first]
    assert entered == [streams[device]] * 2
    assert compute.waited == events, "the hits' blocks are ordered by the misses' own events"
