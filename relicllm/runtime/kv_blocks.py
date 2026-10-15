"""A block table for the KV declaration: the pages, not the writer.

``relicllm/runtime/kv_spec.py`` states what one position *costs* a cache -- ``values_per_token`` for
the marginal reading, ``page_size_bytes(block_size)`` for the allocated one -- and says in its own
closing words that "the declaration is the shape, not the writer". This module is the other half of
that sentence's first clause: the table that says *which* physical block holds a sequence's positions
for each cache the model declared. It allocates bookkeeping, never tensors, which is why it is
torch-free and sits in the device plane beside the declaration rather than in a runtime.

## Why a page is a page

A position is not a page. A ``FullAttentionSpec`` group grows one row a token, so a block of
``block_size`` positions holds exactly ``block_size`` of them. An ``MLASpec`` with
``compress_ratio = r`` pools ``r`` positions into one latent and a block holds ``max(1, bs // r)``
latents, so it covers ``max(1, bs // r) * r`` positions -- ``block_size`` whenever the ratio divides
it, and fewer when it does not, which is the case the arithmetic below resolves by flooring rather
than by assuming. A ``SlidingWindowSpec`` is a ring: it holds ``sliding_window`` slots however long
the context runs, so its row saturates. A ``StateSpec`` has no token axis at all and takes one page
for as long as the sequence lives. Four kinds, four demands, one table: :class:`BlockGeometry` is
where that is written down once, and it is the only place in this module that knows a ``KVKind``.

## Why the pool is denominated in bytes

Pages are not the same size as each other. On DeepSeek-V4.1-Flash a ``window_kv_cache`` page covers
all forty layers and a four-layer ``compress_kv_cache`` page is a fortieth of the model's other
numbers; a ``StateSpec`` page is a whole recurrent state. A pool of "N blocks" would therefore mean
four different amounts of memory depending on which cache asked. The pool is a byte budget with a
free list per page size instead, so "how much is left" is a memory fact and not a block-count fact,
and a page of any size is one allocation.

## Sharing, and the invariant that makes it safe without copy-on-write

Two sequences may point at the same physical page. That is the whole reason a block table is worth
having: it is what makes cross-request prefix reuse a pointer copy rather than a buffer copy, which
is the ceiling the whole-prompt anchor store in ``models/prefix_cache.py`` runs into today.

It is safe only because of a rule this module enforces rather than documents: **only a full page is
shared, and a page is never written after it is full.** The one page a sequence writes into is the
last, partially-filled one, and that page is private to it -- :meth:`SequenceBlocks.share_prefix_from`
takes only whole pages and reports how many positions it actually took, so a caller cannot
accidentally hand out a page that a continuation would then write into. That is vLLM's rule (a block
hash is published for a full block only); it is what removes copy-on-write from the design, and with
it the byte-copying machinery a copy-on-write path would need.

## What this is not

Not an allocator of KV tensors, and not a change to any runtime: no runtime builds a cache out of
these pages yet, and the padding a page layout needs is a kernel-side writer's business. What it is
is the second of the two artefacts ``docs/architecture/kv_declaration.md`` named as missing, so that
continuous batching and a block-granular prefix cache have a table to place a sequence in.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from relicllm.runtime.kv_spec import KVCacheSpec, KVKind, StateSpec

__all__ = [
    "BlockGeometry",
    "BlockPool",
    "DEFAULT_BLOCK_SIZE",
    "KVBlockAllocator",
    "NoSpace",
    "SequenceBlocks",
]

DEFAULT_BLOCK_SIZE = 16
"""The block size this tree allocates at, in positions.

The block size is a property of the pool and not of a cache, so it has to sit whole inside every
geometry the roster declares, and 16 is the value that does: the compress ratios the five publishers
produce are 1 and 2 (V4.1's source layers), 4 and 128 (V4), and 1 (Xing4.0's absorbed latent), and
every window is 128 slots (V4, V4.1, MiMo). Each of those either divides 16 or is a multiple of it, so
a page is a whole number of pooled groups and a page boundary is a block boundary -- which is the
property :meth:`SequenceBlocks.share_prefix_from` needs to stop on. A ratio that did neither would
give a page fewer positions than blocks are cut at (ratio 3 covers 15 of 16) and pad every page.
vLLM's default is also 16, which is the same arithmetic rather than a borrowed constant.
"""


class NoSpace(RuntimeError):
    """The pool cannot afford the pages a request asks for.

    Raised by :meth:`BlockPool.acquire`/:meth:`SequenceBlocks.grow` rather than returned, because
    every caller has to decide what to do and the two that matter -- refuse the request, or schedule
    it later -- both need the number that did not fit. The name is the pool's, not the device's: this
    module knows a byte budget and nothing about what backs it.
    """


@dataclass(frozen=True)
class BlockGeometry:
    """What one page of one cache holds, and what it costs.

    Built from a :class:`~relicllm.runtime.kv_spec.KVCacheSpec` by :meth:`BlockGeometry.of`, which is
    the only constructor a caller needs. The four numbers are the whole of this module's knowledge
    about an architecture: ``positions_per_block`` is ``0`` for a cache with no token axis,
    ``page_bytes`` is the group's cost for all the layers that share the buffer, and ``max_blocks``
    is the ring's ceiling or ``None`` when the row grows without bound.
    """

    name: str
    kind: KVKind
    positions_per_block: int
    page_bytes: int
    max_blocks: int | None = None

    @classmethod
    def of(cls, spec: KVCacheSpec, block_size: int) -> "BlockGeometry":
        """One cache's page, read off the declaration rather than recomputed.

        Every number here comes from a method of the spec -- ``allocated_bytes`` for the page and
        the spec's own ``compress_ratio``/``sliding_window`` for the coverage -- so a spec whose
        arithmetic is wrong is wrong in exactly one place, and the allocator does not get to disagree
        with the fit test about what a block costs.
        """
        page_bytes = spec.allocated_bytes(block_size)
        if isinstance(spec, StateSpec):
            # No token axis: the page is the whole state, and it is taken for the sequence's life.
            return cls(spec.name, spec.kind, 0, page_bytes, max_blocks=1)
        if spec.kind is KVKind.SLIDING_WINDOW:
            # A ring: `sliding_window` slots however long the context runs, so the row saturates.
            window = max(1, int(getattr(spec, "sliding_window", 0)) or 1)
            return cls(spec.name, spec.kind, block_size, page_bytes, max_blocks=_ceil_div(window, block_size))
        if spec.kind is KVKind.MLA:
            ratio = max(1, int(getattr(spec, "compress_ratio", 1)) or 1)
            # The same `max(1, block_size // ratio)` latents `MLASpec.page_size_bytes` lays out,
            # each covering `ratio` positions.
            return cls(spec.name, spec.kind, max(1, block_size // ratio) * ratio, page_bytes)
        return cls(spec.name, spec.kind, block_size, page_bytes)

    @property
    def kind_is_growing(self) -> bool:
        """Whether this row's pages are a function of position -- the rows a prefix can be shared on.

        A ring's pages are not: its slot for a position is ``position % sliding_window``, so a
        continuation writes into slots the sharer is still reading. A state's pages are not either:
        a recurrent state is the fold of a prefix, not a function of it, and cannot be entered
        half-way. Both are therefore private to a sequence however short the prefix, at a cost that
        is bounded and small.
        """
        return self.positions_per_block > 0 and self.max_blocks is None

    def blocks_for(self, positions: int) -> int:
        """How many pages this row holds for a sequence of ``positions`` positions.

        The one formula, and it is the row's own kind that decides which branch it takes: a state is
        one page from the sequence's first position onward (zero before that, because a sequence that
        has computed nothing has no state to keep), a ring is capped at its ceiling, and a growing row
        is ``ceil(positions / positions_per_block)``.
        """
        positions = max(0, int(positions))
        if self.positions_per_block == 0:
            return 1 if positions > 0 else 0
        blocks = _ceil_div(positions, self.positions_per_block)
        if self.max_blocks is not None:
            blocks = min(blocks, self.max_blocks)
        return blocks


class BlockPool:
    """Refcounted pages under a byte budget, one free list per page size.

    A block id is an index into this pool and is stable for the pool's life: releasing a page returns
    its id to the free list for its size, and a later acquire of that size reuses it. Ids are handed
    to the writer, so stability is the property that lets a table keep referring to a page across a
    reallocation elsewhere -- and it is also why a released id is never reissued to a *different*
    size, which would silently change what a stale row points at.

    Refcounting is per page and is what sharing costs: ``retain`` adds a holder, ``release`` removes
    one and frees the page at zero, and :attr:`used_bytes` counts a shared page once because it is
    one page of memory however many sequences read it.
    """

    __slots__ = ("_capacity", "_page_bytes", "_refs", "_idle", "_used", "_in_use")

    def __init__(self, capacity_bytes: int) -> None:
        self._capacity = int(capacity_bytes)
        if self._capacity < 0:
            raise ValueError(f"a pool cannot have a negative budget: {capacity_bytes}")
        self._page_bytes: list[int] = []
        self._refs: list[int] = []
        self._idle: dict[int, list[int]] = {}
        self._used = 0
        self._in_use = 0

    @property
    def capacity_bytes(self) -> int:
        return self._capacity

    @property
    def used_bytes(self) -> int:
        """Bytes held by pages someone is still pointing at, each page counted once."""
        return self._used

    @property
    def free_bytes(self) -> int:
        return self._capacity - self._used

    @property
    def blocks_in_use(self) -> int:
        """How many pages are held, shared ones counted once. A count of pages, not of holders."""
        return self._in_use

    def page_bytes(self, block: int) -> int:
        return self._page_bytes[block]

    def refs(self, block: int) -> int:
        return self._refs[block]

    def is_shared(self, block: int) -> bool:
        """Whether more than one sequence points at this page.

        The writer's question before it writes: a shared page is by construction a full one, so the
        answer is not "copy it first" but "you are writing where you should not be".
        """
        return self._refs[block] > 1

    def afford(self, page_bytes: int, count: int = 1) -> bool:
        """Whether ``count`` pages of ``page_bytes`` fit, without taking them."""
        page_bytes = int(page_bytes)
        count = int(count)
        if page_bytes <= 0 or count <= 0:
            return page_bytes >= 0 and count >= 0
        return self.free_bytes >= page_bytes * count

    def acquire(self, page_bytes: int) -> int:
        page_bytes = int(page_bytes)
        if page_bytes <= 0:
            raise ValueError(f"a page must have a positive size: {page_bytes}")
        if not self.afford(page_bytes):
            raise NoSpace(
                f"this pool has {self.free_bytes} bytes free and a page of this cache costs "
                f"{page_bytes}"
            )
        idle = self._idle.setdefault(page_bytes, [])
        if idle:
            block = idle.pop()
        else:
            block = len(self._page_bytes)
            self._page_bytes.append(page_bytes)
            self._refs.append(0)
        self._refs[block] = 1
        self._used += page_bytes
        self._in_use += 1
        return block

    def acquire_many(self, page_bytes: int, count: int) -> list[int]:
        """``count`` pages of one size, or none at all.

        All-or-nothing because the caller is growing a row: a half-grown row would leave pages in a
        table that the caller then has to remember to release on the failure path, which is the leak
        this method exists to make impossible.
        """
        count = int(count)
        if count <= 0:
            return []
        page_bytes = int(page_bytes)
        if not self.afford(page_bytes, count):
            raise NoSpace(
                f"this pool has {self.free_bytes} bytes free, and {count} pages of {page_bytes} "
                f"cost {page_bytes * count}"
            )
        return [self.acquire(page_bytes) for _ in range(count)]

    def retain(self, block: int) -> None:
        self._check(block)
        self._refs[block] += 1

    def release(self, block: int) -> bool:
        """Drop one holder, and answer whether the page went back to the free list.

        A release of a page nobody holds is a double free and raises rather than corrupting the
        accounting: a table that releases twice would otherwise hand the same id out to two
        sequences, and a KV corruption is a much worse way to find that out than an exception.
        """
        self._check(block)
        if self._refs[block] <= 0:
            raise ValueError(f"block {block} is not held by anyone, so it cannot be released")
        self._refs[block] -= 1
        if self._refs[block]:
            return False
        self._used -= self._page_bytes[block]
        self._in_use -= 1
        self._idle.setdefault(self._page_bytes[block], []).append(block)
        return True

    def _check(self, block: int) -> None:
        if not 0 <= block < len(self._page_bytes):
            raise KeyError(f"block {block} is not a block of this pool")


class SequenceBlocks:
    """One sequence's rows: which physical page holds each cache's positions.

    A row per spec, in the order the declaration lists them -- and the order is the key, not the
    name, because a declaration may name the same buffer twice with different geometry. It does:
    DeepSeek-V4.1-Flash publishes ``compress_kv_cache`` once for the ratio-2 source layers and once
    for the ratio-1 layer, which are two caches of two sizes and would collide in a table keyed by
    name. ``KVBlockAllocator.row`` is how a caller turns a name and a layer into that index.

    ``positions`` is the sequence's context length, not a sum over rows: a token that a ratio-2 cache
    pools with its neighbour is still one token of the sequence, and the rows disagree about how many
    pages it needs rather than about how long the sequence is.
    """

    __slots__ = ("_geometry", "_pool", "_rows", "_positions")

    def __init__(self, geometry: Sequence[BlockGeometry], pool: BlockPool) -> None:
        self._geometry = tuple(geometry)
        self._pool = pool
        self._rows: list[list[int]] = [[] for _ in self._geometry]
        self._positions = 0

    @property
    def positions(self) -> int:
        return self._positions

    @property
    def held_bytes(self) -> int:
        """Bytes this sequence's own pages occupy, shared pages counted once each."""
        return sum(self._pool.page_bytes(block) for row in self._rows for block in row)

    def blocks(self, row: int) -> tuple[int, ...]:
        """The physical pages of one row, in position order -- what a writer indexes into."""
        return tuple(self._rows[row])

    def blocks_for(self, positions: int) -> tuple[int, ...]:
        """The page count every row would hold at ``positions``, without allocating anything."""
        return tuple(geometry.blocks_for(positions) for geometry in self._geometry)

    def required_bytes(self, positions: int) -> int:
        """What this sequence would hold at ``positions``: the admission check before the commit."""
        return sum(
            geometry.blocks_for(positions) * geometry.page_bytes for geometry in self._geometry
        )

    def can_grow(self, positions: int) -> bool:
        """Whether growing to ``positions`` fits in what the pool has left.

        Asked before :meth:`grow` by a scheduler that has not committed the request yet, and answered
        from the same arithmetic so the two cannot disagree. It is the pages this sequence does not
        already hold that have to fit, which is why this is a check on the *difference*: a decode
        step buys a page once every ``positions_per_block`` tokens and nothing in between.
        """
        positions = max(0, int(positions))
        if positions < self._positions:
            return False
        return self._pool.free_bytes >= self._missing_bytes(positions)

    def grow(self, positions: int) -> None:
        """Grow the sequence to ``positions`` positions, buying whatever pages that needs.

        Positional rather than incremental because a sequence's rows are all functions of the one
        number, and an incremental ``advance(1)`` would spell the same thing once per token while
        making "how long is this sequence" a thing the caller has to track. A decode loop calls this
        with its running length; a prefill calls it once.

        Rows are grown one at a time, and a failure part-way through rolls the earlier rows back, so
        a refused grow leaves the pool exactly as it found it. The scheduler's admission check
        (:meth:`can_grow`) is what keeps that path rare rather than routine.

        Always this sequence's own pool, never a parameter: the pages are released by :meth:`free` and
        retained by :meth:`share_prefix_from` through that one pool, so a row bought from a second pool
        would be an id space nobody releases into -- a stale id handed to another sequence, which is
        KV corruption rather than a leak.
        """
        positions = max(0, int(positions))
        if positions < self._positions:
            raise ValueError(
                f"a sequence does not shrink: it holds {self._positions} positions and was asked "
                f"to grow to {positions}"
            )
        grew: list[tuple[int, int]] = []
        try:
            for index, geometry in enumerate(self._geometry):
                wanted = geometry.blocks_for(positions)
                held = len(self._rows[index])
                if wanted <= held:
                    continue
                self._rows[index].extend(
                    self._pool.acquire_many(geometry.page_bytes, wanted - held)
                )
                grew.append((index, wanted - held))
        except NoSpace:
            self._rollback(grew)
            raise
        self._positions = positions

    def share_prefix_from(self, other: "SequenceBlocks", positions: int) -> int:
        """Adopt ``other``'s pages for as much of its prefix as can be shared, and say how much.

        Returns the number of positions actually taken, which is what makes this safe rather than
        clever: sharing stops at a page boundary, so the tail a continuation writes into is the
        caller's own page. A caller asking for 100 positions of a 16-position block size gets 96
        back and recomputes the rest -- vLLM's rule for the same reason, and the reason a partial
        page is never published as shareable.

        Only growing rows share. A ring's pages are recycled by position modulo the window and a
        recurrent state's are a fold rather than a prefix, so both are allocated private to this
        sequence however much of the prefix is being reused.
        """
        if other is self:
            raise ValueError("a sequence cannot share a prefix with itself")
        if len(other._geometry) != len(self._geometry):
            raise ValueError("two sequences of one pool hold the same rows")
        if self._positions or any(self._rows):
            raise ValueError("a prefix is adopted into an empty sequence, not a running one")
        take = max(0, min(int(positions), other._positions))

        grew: list[tuple[int, int]] = []
        try:
            shared = take
            for index, geometry in enumerate(self._geometry):
                if geometry.kind_is_growing:
                    aligned = (take // geometry.positions_per_block) * geometry.positions_per_block
                    shared = min(shared, aligned)
            for index, geometry in enumerate(self._geometry):
                if geometry.kind_is_growing:
                    wanted = geometry.blocks_for(shared)
                    self._rows[index].extend(other._rows[index][:wanted])
                    for block in self._rows[index]:
                        self._pool.retain(block)
                    grew.append((index, wanted))
                else:
                    wanted = geometry.blocks_for(shared)
                    if wanted:
                        self._rows[index].extend(
                            self._pool.acquire_many(geometry.page_bytes, wanted)
                        )
                        grew.append((index, wanted))
        except NoSpace:
            self._rollback(grew)
            raise
        self._positions = shared
        return shared

    def free(self) -> None:
        """Release every page this sequence holds, shared ones by one holder only."""
        for row in self._rows:
            for block in row:
                self._pool.release(block)
            row.clear()
        self._positions = 0

    def _missing_bytes(self, positions: int) -> int:
        return sum(
            max(0, geometry.blocks_for(positions) - len(self._rows[index])) * geometry.page_bytes
            for index, geometry in enumerate(self._geometry)
        )

    def _rollback(self, grew: Iterable[tuple[int, int]]) -> None:
        for index, count in grew:
            row = self._rows[index]
            for block in row[-count:]:
                self._pool.release(block)
            del row[-count:]


class KVBlockAllocator:
    """One pool, one declaration, one block size: the table sequences are placed in.

    The allocator owns the bytes; :class:`SequenceBlocks` is what it hands out. Splitting them is
    what makes the sharing arithmetic readable -- a row knows its own geometry and asks the pool for
    pages, and nothing above has to hold both a sequence's map and the model's memory in its head.

    ``capacity_bytes`` is the adapter's number, not the model's, exactly as
    :class:`~relicllm.scheduler.core.KVCapacity` is: what a card can lend to KV after the weights,
    the expert banks and the graph have taken theirs is a fact about a load, and a declaration that
    guessed it would put a plausible-looking number where a measurement belongs.
    """

    __slots__ = ("_specs", "_geometry", "_block_size", "_pool")

    def __init__(
        self,
        specs: Iterable[KVCacheSpec],
        *,
        capacity_bytes: int,
        block_size: int = DEFAULT_BLOCK_SIZE,
    ) -> None:
        block_size = int(block_size)
        if block_size <= 0:
            raise ValueError(f"a block size is a positive position count: {block_size}")
        self._specs = tuple(specs)
        self._geometry = tuple(BlockGeometry.of(spec, block_size) for spec in self._specs)
        self._block_size = block_size
        self._pool = BlockPool(capacity_bytes)

    @property
    def pool(self) -> BlockPool:
        return self._pool

    @property
    def specs(self) -> tuple[KVCacheSpec, ...]:
        return self._specs

    @property
    def geometry(self) -> tuple[BlockGeometry, ...]:
        return self._geometry

    @property
    def block_size(self) -> int:
        return self._block_size

    def row(self, name: str, *, layer: int | None = None) -> int:
        """The row index of a cache, by name and optionally by the layer that reads it.

        A name alone is not a key: V4.1 declares ``compress_kv_cache`` twice. A layer disambiguates
        because a layer belongs to exactly one group per buffer name, so ``row("compress_kv_cache",
        layer=2)`` is the ratio-2 row; a name that matches more than one row with no layer to choose
        between them is refused rather than resolved by picking one.
        """
        matches = [
            index
            for index, spec in enumerate(self._specs)
            if spec.name == name and (layer is None or layer in spec.layer_ids)
        ]
        if not matches:
            raise KeyError(f"no cache named {name!r}{'' if layer is None else f' on layer {layer}'}")
        if len(matches) > 1:
            raise KeyError(
                f"{name!r} names {len(matches)} caches; name the layer that reads it "
                f"({', '.join(str(spec.layer_ids) for spec in (self._specs[i] for i in matches))})"
            )
        return matches[0]

    def page_bytes(self, row: int) -> int:
        return self._geometry[row].page_bytes

    def new_sequence(self) -> SequenceBlocks:
        """An empty table. It owns no pages until it is grown."""
        return SequenceBlocks(self._geometry, self._pool)

    def required_bytes(self, positions: int) -> int:
        """What one sequence of ``positions`` positions costs, for sizing a pool before a load."""
        return sum(geometry.blocks_for(positions) * geometry.page_bytes for geometry in self._geometry)

    def max_positions(self) -> int | None:
        """How long a single sequence this pool can hold, or ``None`` when nothing here grows.

        ``None`` is not a failure and not "unlimited": it is the honest answer for a declaration
        whose every row is bounded -- a ring and a recurrent state -- where no position count
        exhausts the pool and the limit is the context the model was built for rather than the
        memory. A caller that prints this prints that sentence.

        Exact rather than approximate. The demand is monotone in ``positions`` so it can be found by
        bisection, and the bracket is closed-form: bytes grow at least ``page_bytes /
        positions_per_block`` a position per growing row, so one position past the budget's worth of
        that rate is guaranteed not to fit.
        """
        if self.required_bytes(1) > self._pool.capacity_bytes:
            return 0
        rate = Fraction(0)
        for geometry in self._geometry:
            if geometry.kind_is_growing:
                rate += Fraction(geometry.page_bytes, geometry.positions_per_block)
        if rate == 0:
            return None
        low = 1
        high = int(self._pool.capacity_bytes / rate) + 1
        if self.required_bytes(high) <= self._pool.capacity_bytes:
            # Only reachable if the rate overestimates, which the inequality above rules out; kept as
            # a guard so a future geometry cannot turn this into an unterminated search.
            high *= 2
        while high - low > 1:
            middle = (low + high) // 2
            if self.required_bytes(middle) <= self._pool.capacity_bytes:
                low = middle
            else:
                high = middle
        return low


def _ceil_div(value: int, divisor: int) -> int:
    return -(-int(value) // int(divisor))
