"""The arena rows a card keeps, and the host block a fill stages them through.

`movement.py` is the *path* a routed expert's bytes take -- the allocation, the copy stream, the
slot ring and the two waits that order it. This module is the half of the same seam that outlives a
single layer: one arena a *card* and one page-locked block a *model*, plus the `(layer, expert)` pool
that decides which arena row a key keeps. Until this file existed that state was
`models/deepseek_v4_1/device_experts.py`'s `ResidentSet`, and `docs/architecture/expert_movement.md`
is the argument for it living here instead: none of it is a fact about a checkpoint, so a second
safetensors bank that sized and indexed its own would be the same code a second time.

What stays with the model is the *rule*. v41 chooses which experts deserve a resident row by counting
the layer's routing (`DeviceRoutedExperts._hot_rows`), and the pool below is only the row allocator
that rule and an ordinary draw share; mimo's least-frequently-used set is a different rule over the
same rows. So the class here is deliberately one class and not a policy protocol: there is exactly
one implementation in play until `models/mimo_v2/device_experts.py`'s `_Residents` is moved onto the
seam, and the second implementation is what will draw the boundary.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import torch

from relicllm.components.moe.movement import PinnedBlock, allocate_arena

__all__ = ["ResidentSet"]


class ResidentSet:
    """The arena and the staging block every layer's resident experts share, one of each a card.

    A resident set is worth having only at a size forty of them do not fit. At 64 slots a card it is
    1.2 GiB of device memory and as much again of page-locked host memory, so a layer holding its own
    would be 47 GiB of a 22 GiB card and 48 GiB of unswappable RAM. One set, refilled by whichever
    layer is running, is safe because the layers run one at a time and each `forward` drains every
    kernel it issued before it returns -- and because the device arena holds no state between layers,
    only bytes.

    The two halves are the same size and for the same reason. `_fill` copies out of the checkpoint
    into `pinned` and then DMA-copies `pinned` into `arena`, so the host block is not an
    implementation detail of the transfer: at `hot_rows` slots it is as much memory as the card's
    copy, and sharing one without the other would halve the saving.

    `events` is one a card for the fill's own DMA, and it is here rather than in the layer for the
    same reason: the next layer's host write into `pinned` has to be behind the previous layer's read
    of it, and with forty layers sharing one block that guarantee cannot live in forty instances.
    """

    def __init__(
        self,
        *,
        hot_rows: int,
        pool_rows: int = 0,
        rows_per_card: int,
        ranks: Sequence[int],
        devices: Sequence[torch.device],
        shapes: Mapping[tuple[str, str], Sequence[int]],
    ) -> None:
        self.hot_rows = int(hot_rows)
        self.pool_rows = max(0, int(pool_rows))
        self.rows_per_card = int(rows_per_card)
        self.arena_rows = self.rows_per_card + self.hot_rows + self.pool_rows
        self.ranks = list(ranks)
        self.devices = list(devices)
        self._shapes = dict(shapes)
        self.arena: list[dict[tuple[str, str], torch.Tensor]] = [
            allocate_arena(device, self.arena_rows, self._shapes) for device in self.devices
        ]
        self.events: list[torch.cuda.Event] = []
        for device in self.devices:
            with torch.cuda.device(device):
                self.events.append(torch.cuda.Event())
        # Built on the first fill and not here: a decode-only run never fills, and this many rows of
        # page-locked memory is not something to allocate for a set that will always be empty.
        self._pinned: PinnedBlock | None = None
        # The pool: which arena row a card keeps each expert in. The key is `(layer, expert)` and not
        # the expert id alone, and that is not a detail -- see `pool_row`, which is where the reason
        # is written down and where getting it wrong cost a whole sweep. It is on the set rather than
        # on the layer because the *arena* has to be one a card: an arena wide enough to hold a
        # layer's working set cannot be forty of them, so the map that says which arena row is whose
        # lives beside the rows it describes. `pool_free` is the rows never used yet and `pool_lru`
        # the ones in use, oldest first; a key is evicted only when both are empty. `pool_rows=0` is
        # the off state and the configuration every number above this was taken on.
        self.pool_map: list[dict[tuple[int, int], int]] = [{} for _ in self.ranks]
        self.pool_lru: list[OrderedDict[tuple[int, int], None]] = [
            OrderedDict() for _ in self.ranks
        ]
        self.pool_free: list[list[int]] = [
            list(range(self.rows_per_card + self.hot_rows, self.arena_rows))
            for _ in self.ranks
        ]
        # Counted, like the class's own counters: how many rows the pool took off the staging path
        # is what says whether it is doing anything, and `pool_evicted` is what says whether
        # `pool_rows` was the binding constraint rather than a size.
        self.pool_staged = 0
        self.pool_evicted = 0

    def pool_row(self, card: int, key: tuple[int, int]) -> tuple[int, bool] | None:
        """`(arena row, needs staging)` for `key` on `card`, or `None` when the pool is off.

        `None` is the off state, and a caller that gets it is on the path this class had before the
        pool existed -- the row's own miss rows, staged and never kept. `True` means the row has just
        been assigned and holds nothing yet, so the caller stages into it; `False` means it already
        holds this key's bytes and the draw costs no H2D at all.

        **`key` is `(layer, expert)`, and the layer half is load-bearing.** An expert id is not an
        identity: the weights live at `layers.{layer}.ffn.experts.{expert}.{which}.weight`, so layer
        5's expert 7 and layer 6's expert 7 are different tensors that happen to share a number. A
        pool keyed on the id alone answers a layer 6 draw with layer 5's bytes -- silently, since the
        rows are the right shape and the bytes are a real expert's -- and it does so *more* often the
        wider the pool is. That is what the first version of this did, and a 512-token sweep caught
        it only because the probe keeps the logits: every pooled run at every width moved the top 32
        by 4.7 to 6.8 of a max logit of 29.15, and the four runs at two widths agreed with each other
        to the digit, which is what a wrong answer looks like when nothing about it is random. It is
        worth stating as a rule rather than as an anecdote: a cache keyed on a number that is only
        unique within a scope must be keyed on the scope too.

        The row is stable for as long as the key is in the pool: a second draw of the same key later
        in the layer reads the row this returns, and nothing invalidates it but an eviction. That is
        the whole difference between this and `_fill` -- a fill row means "resident *this layer*" and
        is rewritten by the next layer's fill, while a pool row holds its bytes until the pool gives
        the row to somebody else, and a key evicted and re-drawn pays the staging again.

        **Eviction is safe at any size, and it is worth saying why rather than guarding it.** A row
        handed out to a different key is overwritten by an H2D that `StagingRing.stage` puts behind
        that card's compute stream, and `_issue` puts the next kernel behind the copy -- so the
        overwrite cannot be read by a kernel that was launched before it, however recently the
        evicted key was drawn. The arena the row lives in is what the ordering already protects; the
        pool changes which row a key is in and not when a byte is allowed to move. What a too-small
        pool costs is therefore only the fills it throws away, which is a hit rate and not a
        correctness property, and `pool_evicted` is the counter that says so. Note what this argument
        does *not* cover: it is about the bytes in a row and not about which bytes belong there,
        which is the other half of the problem and the reason the key carries a layer.
        """
        if not self.pool_rows:
            return None
        row = self.pool_map[card].get(key)
        if row is not None:
            self.pool_lru[card].move_to_end(key)
            return row, False
        if self.pool_free[card]:
            row = self.pool_free[card].pop()
        else:
            victim, _ = self.pool_lru[card].popitem(last=False)
            row = self.pool_map[card].pop(victim)
            self.pool_evicted += 1
        self.pool_map[card][key] = row
        self.pool_lru[card][key] = None
        self.pool_staged += 1
        return row, True

    @property
    def pinned(self) -> dict[tuple[str, str], torch.Tensor]:
        """The page-locked block the fill stages through, one card's worth laid end to end."""
        if self._pinned is None:
            self._pinned = PinnedBlock(self._shapes, len(self.ranks) * self.hot_rows)
        return self._pinned.rows

    @property
    def arena_bytes(self) -> int:
        """Packed fp4 bytes one card's arena occupies on the device."""
        return self.arena_rows * sum(shape[0] * shape[1] for shape in self._shapes.values())

    @property
    def pinned_bytes(self) -> int:
        """Page-locked host bytes the fill stages through, zero until the first fill asks."""
        if self._pinned is None:
            return 0
        return self._pinned.nbytes
