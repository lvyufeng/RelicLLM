"""Where a routed expert's bytes wait, and how they get to the card.

Every runtime in this tree that serves an MoE model is, at its hot spot, moving routed experts from
host memory to the card. That movement is the tree's only real performance centre and the only place
where adding a model adds another copy of the same code: until this file existed the safetensors
side wrote it three times over, in `models/deepseek_v4_1/device_experts.py`,
`models/mimo_v2/device_experts.py` and `models/qwen4_exp/moe.py`, each with its own staging loop,
its own ring and its own pair of stream waits. `docs/architecture/expert_movement.md` is the seam
this file is the code side of, and the split it draws is the whole of the design: the *lifecycle* of
the bytes is shared, the *plan* is not.

What lives here is the path.

* **The arena.** `allocate_arena` is one `uint8` tensor per `(projection, kind)` a checkpoint
  declares, `rows` deep, on one device, sliced into a resident region in front and rotating staging
  slots behind it. One allocation rather than one a region, because the kernel is handed one tensor
  and indexes its first axis, so a resident row and a staging row have to be rows of the same tensor
  for one call to read both.
* **The slot ring.** `StagingRing` rotates round-robin over the slots, with a per-slot event and the
  rule that a slot is not reused until the DMA that last filled it has drained. It advances only on
  a row that actually moves bytes, and `StagingRing.stage` is where that guarantee is written down.
* **The copy stream and its two waits.** One H2D stream per device (`copy_stream`), and the two
  edges that make it correct: the copy stream waits on the compute stream before it overwrites rows
  a kernel may still be reading, and the compute stream waits on the copy stream before the kernel
  that reads the rows just written (`StagingRing.order_kernel`). Nothing on the host waits for
  either.
* **The page-locked staging block.** `PinnedBlock`, allocated on first use and never for a run that
  never fills.
* **The copy itself.** `StagingRing.stage` is the loop -- a row's sources into a row's arena, on the
  copy stream, recorded on the slot's event.

What stays with the model is the plan: which experts a draw needs (the deal), which arena row a key
occupies (the residency policy), what the kernel call is, and what a returned partial sum means.
Those are not incidental. `docs/architecture/one_scheduler_many_models.md` records a per-architecture
`ServingSpec` draft and rejects it as the adapter duplication this tree already suffers moved one
level up, and the three safetensors banks genuinely do not share an arithmetic: their deals, their
residency policies, their kernels and their sums all differ. A movement layer that also took the
arithmetic would be the same mistake in a smaller costume.

So the seam here is deliberately narrow, and it is narrow in one direction: this module names no
model, no checkpoint and no kernel, and every entry point is handed rows and byte ranges that a
model computed. `StagingRing` in particular never decides *which* row a row's miss stages into; it
decides that the copy into that row is ordered, that the slot holding it is not reused too early,
and that the rotation is not advanced by a row that moved nothing.
"""

from __future__ import annotations

from typing import Mapping, Sequence

import torch

__all__ = [
    "PinnedBlock",
    "StagingRing",
    "allocate_arena",
    "copy_stream",
]

#: One page of an expert's bytes: the destination view a copy writes and the source it reads. A
#: miss is six of them -- a projection and its scale -- because the kernel reads both halves and a
#: checkpoint stores them as two tensors.
Copy = tuple[torch.Tensor, torch.Tensor]

#: Reused across instances and across layers: a `torch.cuda.Stream` is a per-process resource, and a
#: 40-layer backbone would otherwise create 160 of them for the four cards to share four at a time.
_COPY_STREAMS: dict[int, torch.cuda.Stream] = {}


def copy_stream(device: torch.device) -> torch.cuda.Stream:
    """The one H2D stream for `device`, created on first use and never freed.

    One a *device* and not one a layer, because the ordering is a property of the pair of streams:
    the copy that fills a row and the kernel that reads it have to be on two streams that the same
    two waits order, and a fresh stream a layer would have nothing to be ordered against. This is
    the stream both sides of the seam name -- `stage` and `load` issue on it, `order_kernel` makes
    the compute stream wait on it -- and it is a function rather than a ring attribute so that a
    test can replace it with something that is not a card.
    """
    index = device.index if device.index is not None else torch.cuda.current_device()
    stream = _COPY_STREAMS.get(index)
    if stream is None:
        with torch.cuda.device(index):
            stream = torch.cuda.Stream(device=index)
        _COPY_STREAMS[index] = stream
    return stream


def allocate_arena(
    device: torch.device, rows: int, shapes: Mapping[tuple[str, str], Sequence[int]]
) -> dict[tuple[str, str], torch.Tensor]:
    """One device's arena: `rows` rows of every tensor the checkpoint declares for an expert.

    `shapes` is read from the checkpoint rather than derived from a geometry computed here, so a
    release laid out differently fails at the allocation the kernel's index arithmetic is read from
    rather than inside a kernel. The dtype is `uint8` for every key and not only for the packed
    codes, because an E8M0 scale has no CPU `copy_` out of a pyloaded view and the kernel reads both
    halves as bytes.
    """
    with torch.cuda.device(device):
        return {
            key: torch.empty((rows,) + tuple(shape), dtype=torch.uint8, device=device)
            for key, shape in shapes.items()
        }


class PinnedBlock:
    """The page-locked rows a fill stages *through*, one card's worth laid end to end, built lazily.

    A fill copies out of the bank into this block and then DMA-copies the block into the arena, so
    the block is not an implementation detail of the transfer: at `hot_rows` slots it is as much
    memory as the card's copy of the same rows, and sharing one without the other would halve the
    saving. Page-locked memory is not reclaimable by anything the host decides, and a decode-only
    run never fills, so `rows` is the first thing that pays for it and nothing else does.
    """

    def __init__(self, shapes: Mapping[tuple[str, str], Sequence[int]], rows: int) -> None:
        self._shapes = dict(shapes)
        self._rows = int(rows)
        self._block: dict[tuple[str, str], torch.Tensor] | None = None

    @property
    def rows(self) -> dict[tuple[str, str], torch.Tensor]:
        """The block, allocated on the first ask and kept after it."""
        if self._block is None:
            self._block = {
                key: torch.empty(
                    (self._rows,) + tuple(shape), dtype=torch.uint8, pin_memory=True
                )
                for key, shape in self._shapes.items()
            }
        return self._block

    @property
    def nbytes(self) -> int:
        """Page-locked bytes this block occupies, zero until the first ask."""
        if self._block is None:
            return 0
        return self._rows * sum(
            shape[0] * shape[1] for shape in self._shapes.values()
        )


class StagingRing:
    """The rotating staging slots of one instance, and the copies that fill them.

    `slots` slots and one event a *card* a slot: the transfer is per card, four links at once being
    the point of the split, but a row's slot is one slot for every card, because it is one row and
    its cards are staged together. The events are handed in rather than allocated here so that a
    test can drive this with no card in the machine -- which is the only way the rotation is checked
    at all, and the rotation is the half of this that fails silently.

    Two waits and no more, and both of them are about the *other* stream:

    * `stage` makes the copy stream wait on the compute stream, because the arena rows it is about
      to overwrite are the ones the previous `stage`'s kernel read.
    * `order_kernel` makes the compute stream wait on the copy stream, because the kernel's inputs
      are the rows the copies are still writing.

    Neither is implied by anything else: the two are on different streams, so nothing but these two
    edges orders them. `take`'s own wait is a third thing with a different job -- it bounds how far
    the host runs ahead of the card -- and it is the subject of `take`'s docstring.
    """

    def __init__(
        self,
        *,
        slots: int,
        devices: Sequence[torch.device],
        events: Sequence[Sequence[torch.cuda.Event]] | None = None,
    ) -> None:
        self.slots = max(1, int(slots))
        self.devices = list(devices)
        if events is None:
            events = [
                [torch.cuda.Event() for _ in range(self.slots)] for _ in self.devices
            ]
        self.events = [list(card) for card in events]
        if len(self.events) != len(self.devices) or any(
            len(card) != self.slots for card in self.events
        ):
            raise ValueError(
                f"{len(self.events)} cards of events for {len(self.devices)} devices and "
                f"{self.slots} slots"
            )
        # `False` means "no copy in flight through this slot from this card", which is the state a
        # fresh slot is in. The flag and the event are a pair: the event says the DMA was issued and
        # the flag says whether the wait on it is still owed.
        self._in_flight = [[False] * self.slots for _ in self.devices]
        self._next = 0

    def take(self) -> int:
        """The next staging slot, after the DMA that last used it has finished.

        This wait used to have a host buffer for its subject: the pinned arena's slot was about to
        be overwritten by the next row's staging and the DMA reading it was still in flight. There
        is no host buffer in the way any more, so what is left is the slot's event alone, and the
        arena rows a copy writes are ordered by the streams themselves -- `stage` puts the copy
        behind the compute stream and `order_kernel` puts the next kernel behind the copy.

        What the wait still does is bound how far the host runs ahead of the cards, and what it
        costs is measured. Recorded against `deepseek_v4_1`: with the copies gone from its `_stage`,
        a 512-token leg's rank-0 class wall is 13.40 s and **7.57 s of it -- 56.5%, 1.58 ms a call
        over 4807 calls -- is this one `synchronize`**, and the same wait on the chunked path (a
        4096-token chunk in the 32768..65536 range, the ranks that stage the most rows) is 1.79 ms a
        call, 15.8 s of a 36.01 s chunk, with 96% of the waits unsatisfied. That is the block and not
        a lever: an A/B that skips it entirely moves the chunk by **-0.35 s inside a 3.31 s
        chunk-to-chunk spread** (0.9903x). The host is the producer and this is its throttle, and
        with two slots there is nowhere to run ahead to, so removing the block moves the block and
        not the chunk. A phase tap cannot see this wait at all -- its own preamble calls
        `torch.cuda.synchronize()`, which satisfies the event before it is asked about.
        """
        slot = self._next
        self._next = (self._next + 1) % self.slots
        for card in range(len(self.devices)):
            if self._in_flight[card][slot]:
                self.events[card][slot].synchronize()
                self._in_flight[card][slot] = False
        return slot

    def stage(self, copies: Sequence[Sequence[Copy]], slot: int | None = None) -> int:
        """One row's staging: take a slot, order it behind compute, copy, record the slot's event.

        `copies` is per card a list of `(destination row, source)` pairs, in the order the model's
        own loop over its arena would ask for them -- so a card's misses at a fixed offset are one
        call, and the six tensors of one expert are six pairs. Nothing here knows which expert a
        pair came from, and that is the seam: the model names the bytes, this decides when they may
        move.

        **`slot` is for a caller that took the slot itself.** A caller has work to do with the slot
        before the copies can be built -- it names the destination rows out of the slot's band, or it
        parks the row's ids under the slot and the stage that fills them has to be handed the same
        one back -- and such a caller calls `take` and passes what it got here. Taking and staging
        are then one rotation between them, which is what they are: one `take` a row is the whole of
        what the rotation is advanced by, and a caller that took a slot and then let `stage` take a
        second would rotate twice a row and, at two slots, stage into exactly the slot the previous
        row had filled -- waiting on the DMA it had just issued. A caller with no use for the slot
        leaves it `None` and the take happens here.

        **A row with nothing to move returns before `take`, and that is a guarantee rather than an
        early exit for its own sake.** `take` hands out the next slot and waits on the DMA that last
        uploaded through it, so the guarantee that the slot being uploaded had a whole row's
        uploading since its copy was issued only holds if every row that takes a slot also does work
        with it. A residency policy can answer most of a prefill's rows without moving anything --
        recorded at **86.3% on a 512-token leg** -- and against that hit rate a rotation advanced
        through rows that move nothing lands on the slot the previous *working* row uploaded
        immediately, with no upload in between to cover the copy. Measured on that leg: **6.89 s of
        a 30.35 s class wall in this one wait**, against the ~2.5 s of DMA it covered. Rotating only
        on rows that move restores the guarantee by construction: consecutive working rows take
        consecutive slots, so a slot's copy was issued one working row before it is waited on,
        whatever the policy's hit rate is. A decode row misses by construction, which is why the
        step recorded 0.00 s here -- it was the one configuration where the wait always had a row's
        upload in front of it. This is a property of the ring and not of the policy, which is the
        reason the guard is here and not in the caller.

        The return is how many pairs were copied, which is the model's own count of what it asked
        for; a caller that wants rows or bytes has them where the deal is.
        """
        if not any(copies):
            return 0
        if slot is None:
            slot = self.take()
        moved = 0
        for card, pairs in enumerate(copies):
            if not pairs:
                continue
            device = self.devices[card]
            stream = copy_stream(device)
            with torch.cuda.device(device):
                # Behind this card's own compute stream, which is where its previous kernel was
                # launched: the copies rewrite arena rows that kernel read, and the two are on
                # different streams so nothing else would order them.
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for destination, source in pairs:
                        destination.copy_(source, non_blocking=True)
                    self.events[card][slot].record(stream)
            self._in_flight[card][slot] = True
            moved += len(pairs)
        return moved

    def order_kernel(self, card: int) -> None:
        """Put `card`'s compute stream behind its own copies, and behind no one else's.

        A card's kernel reads only its own arena rows, so four independent chains is what the
        hardware is; ordering every card behind every other card's copies would serialize links that
        were split precisely to run at once.
        """
        device = self.devices[card]
        torch.cuda.current_stream(device).wait_stream(copy_stream(device))

    def load(self, card: int, copies: Sequence[Copy], event: torch.cuda.Event) -> None:
        """A contiguous block of rows in one transfer, ordered behind compute, recorded on `event`.

        The fill's half of `stage`, and the differences are both facts about the fill. It is one
        copy a tensor for a whole layer's resident region rather than a loop of small ones, because
        the resident rows are contiguous in the arena and the block they come from is a plain
        `copy_` a tensor. And it records the event the *next* fill waits on rather than taking a slot
        of this ring, because the block it reads is shared by every layer of the model: what has to
        be ordered is the next layer's host write against this layer's DMA, and that event cannot
        live in a ring one layer owns.
        """
        device = self.devices[card]
        stream = copy_stream(device)
        with torch.cuda.device(device):
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for destination, source in copies:
                    destination.copy_(source, non_blocking=True)
                event.record(stream)
