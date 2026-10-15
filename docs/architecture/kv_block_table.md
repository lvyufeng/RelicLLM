# The KV block table

[The KV declaration](kv_declaration.md) answers what one position *costs* a cache: `values_per_token`
for the marginal reading, `page_size_bytes(block_size)` for the allocated one. It closed by naming
two artefacts still missing -- a paged allocator above it, and the block table that allocator would
need -- and left the block size open. This page is the second of those and the answer to the first
question it left.

`relicllm/runtime/kv_blocks.py` is the table. Like the declaration it sits in the device plane: it is
torch-free, it holds no tensors, and a host that wants to know how long a sequence fits can read it
on a machine that will never load the model.
`tests/test_kv_blocks.py::test_the_block_table_is_readable_without_importing_torch` is the check, run
in a subprocess because "this process did not import torch" is not evidence about a module's own
imports.

## A position is not a page

Four caches, four demands, and `BlockGeometry` is the one place that resolves them. It reads every
number off the spec -- `allocated_bytes` for the page, the spec's own `compress_ratio` and
`sliding_window` for the coverage -- so a spec whose arithmetic is wrong is wrong in one place and
the table cannot disagree with the fit test about what a block costs.

| cache | what one page covers | pages for one sequence | shareable |
| --- | --- | --- | --- |
| `FullAttentionSpec` | `block_size` positions (16) | `ceil(positions / 16)`, unbounded | yes |
| `MLASpec`, ratio *r* | `max(1, 16 // r) * r` positions | `ceil(positions / that)`, unbounded | yes |
| `SlidingWindowSpec` | 16 positions, over a window of 128 slots | saturates at `ceil(window / 16)` = 8 | no |
| `StateSpec` | no token axis at all | 1, from the sequence's first position | no |

The MLA row is the one that needs the arithmetic written down rather than assumed. A ratio *r* cache
pools *r* positions into one latent, so a 16-position block holds `max(1, 16 // r)` of those latents
and covers `max(1, 16 // r) * r` positions: 16 whenever the ratio divides the block size, and one
whole pooled group when the ratio is larger than it. DeepSeek-V4's ratio-128 row is that second case,
and it is why the coverage is floored at one group rather than computed as `16 // r` -- a page that
covered fewer positions than the single group it holds would let a writer index past its end.

## Why the block size is 16

The block size belongs to the pool, not to a cache, so it has to sit whole inside every geometry the
roster declares at once. The five publishers produce compress ratios of 1, 2, 4 and 128, and every
sliding window is 128 slots, so 16 works and 16 is not a coincidence: each ratio either divides it
(1, 2, 4) or is a multiple of it (128), and so is every window. That is the property
`SequenceBlocks.share_prefix_from` stops on -- a page boundary is a block boundary for every row at
once -- and `tests/test_kv_blocks.py::test_the_default_block_size_leaves_no_cache_a_floored_remainder`
asserts it over the roster's real declarations rather than over the constant's own arithmetic. vLLM
defaults to the same 16; that is the same arithmetic, not a borrowed constant.

A ratio that did neither would break the property quietly: ratio 3 against a block size of 16 covers
15 positions a page, so every page would carry a position's worth of slack and the lengths a prefix
could be resumed at would stop being the lengths pages end at. 16 is the smallest power of two that
the ratio list accepts, which is also why the pool's free lists are as few as they are.

## The pool is a byte budget

Pages are not the same size. On DeepSeek-V4.1-Flash a `window_kv_cache` page spans all forty layers
and is 655,360 bytes; a four-layer `compress_kv_cache` page is 24,576. A pool of "N blocks
remaining" would therefore mean four different amounts of memory depending on which cache asked, and
"how much is left" would stop being a memory fact. So the budget is in bytes, and a page of any size
is one allocation.

Three rules the pool enforces rather than documents:

- **One free list per page size.** A block id is an index into the pool and is stable for its life;
  releasing a page returns its id to the free list for its *size*, and a later acquire of that size
  reuses it. Ids are handed to the writer, so reissuing one at a different size would leave a stale
  row pointing at a page of another shape.
- **Refcounts, and a shared page counted once.** `used_bytes` is memory, not a holder count: two
  sequences reading one page is one page of memory. This is the accounting the whole design is for.
- **All or nothing, with rollback.** `acquire_many` takes every page a row asks for or none, and
  `SequenceBlocks.grow` rolls back the rows it already grew when a later row cannot afford its page.
  A half-grown row would leave pages in a table that the caller has to remember to release on a
  failure path, which is the leak the two methods exist to make impossible. A double free raises: a
  table that released twice would hand one id to two sequences, and a KV corruption is a much worse
  way to learn that than an exception.

`capacity_bytes` is the adapter's number, not the model's, exactly as `KVCapacity` is. What a card
can lend to KV after weights, expert banks and the graph have taken theirs is a fact about a load,
and a declaration that guessed it would put a plausible number where a measurement belongs.

## Whole pages only, and why that removes copy-on-write

Two sequences may point at the same physical page -- that is the reason a block table is worth
having. It is safe because of one rule: **only a full page is shared, and a page is never written
after it is full.** A sequence writes into its last, partially-filled page and into no other, and
that page is its own.

`share_prefix_from` is where the rule is enforced rather than described. It takes whole pages only,
and it returns *how many positions it actually took*, so a caller asking for 100 positions at a
block size of 16 gets 96 back and recomputes the remaining four. The tail a continuation writes into
is therefore the adopter's own page, and a shared page is by construction one nobody will write
again. vLLM publishes a block hash for a full block only, for the same reason.

The alternative is a copy-on-write path: copy the page, rewrite the row, keep the byte copier and the
bookkeeping that goes with it. The rule above is what makes that machinery unnecessary, which is why
`test_a_continuation_writes_into_its_own_page_and_never_a_shared_one` names the fix it would need if
it ever failed.

Only growing rows share. A ring's slot for a position is `position % sliding_window`, so a
continuation writes into slots a sharer is still reading; a recurrent state is the fold of a prefix
rather than a function of it and cannot be entered half-way. Both are allocated private to a sequence
however much of the prompt it was handed, and both are bounded and small -- which is the same
argument the declaration makes for a ring's marginal cost being zero.

## What one sequence costs on V4.1-Flash

The real declaration, at the real block size, from the checked-in config snapshot the fit test uses:

| row | layers | page | per 16 positions |
| --- | --- | --- | --- |
| `window_kv_cache` | 40 | 655,360 | 0 once the window is full |
| `compress_kv_cache`, ratio 2 | 3 | 24,576 | — |
| `compress_kv_cache`, ratio 1 | 1 | 16,384 | — |
| `k_cache`, ratio 2 | 3 | 6,144 | — |
| `k_cache`, ratio 1 | 1 | 4,096 | — |

The four growing rows buy 51,200 bytes every 16 positions, which is 3,200 bytes a position --
the checkpoint-verified marginal cost, not a second derivation of it. The ring saturates at
5,242,880 bytes (eight pages over forty layers) and stops costing anything at 128 positions, but it
buys its pages as it fills rather than reserving the window up front: a 16-position sequence holds
one of the eight. One 4096-position sequence costs 18,350,080 bytes, or 17.5 MiB.

`KVBlockAllocator.max_positions()` is the number a host prints for a pool: the longest single
sequence that budget holds, found by bisection with a closed-form bracket. `None` is not "unlimited"
and is not a failure -- it is the honest answer for a declaration whose every row is bounded, where
no position count exhausts the pool and the limit is the context the model was built for.

## The latent width this found

The first consumer of `page_size_bytes` found that `MLASpec.page_size_bytes` divided by the compress
ratio twice: `latents * values_per_token` instead of `latents * head_dim`. `values_per_token` is
already `head_dim / ratio`, so a compressed page came out at `1/ratio` of what the runtimes
allocate. The ground truth is what the two runtimes register -- `torch.zeros(max_batch_size,
max_seq_len // ratio, head_dim)` in `models/deepseek_v4/runtime.py` and
`models/deepseek_v4_1/attention.py` -- and a latent is `head_dim` values wide, not
`head_dim / ratio`. The marginal reading was right; only the page was wrong, and it had no reader
until now, which is why the defect sat there and why the fix lands with the table that reads it.

## What this is not

Not an allocator of KV tensors, and not a change to any runtime. No runtime builds a cache out of
these pages yet, the padding a page layout needs is a writer's business, and the kernels that would
write into a page are in relic-core. What has landed is the table a scheduler places a sequence in,
and the arithmetic under it.

The honest boundary is the prefix cache that exists today. `models/prefix_cache.py` keys a snapshot
by the prompt's tokens and walks stored anchors longest-first, and its unit is a whole-prefix anchor
with a 64-token granularity in the chain -- not vLLM's paged KV -- because both runtimes hold
pre-allocated contiguous buffers with a partial group that is not block-aligned. A block-granular
store is not a store on top of that layout; it is a cache that allocates from this table instead, and
that is a runtime-side change this page does not make. What the table does is turn "the caches would
have to be rewritten" from an assumption into a list of the rows that would have to change.

## What it does not decide

The writer, and whether the block table lands before or after the prefix-cache ports it would make
cheaper. The first is relic-core's and the second is the same open question the declaration closed
with; both are Phase 4's
[paged KV and continuous batching](../performance/old_hardware_roadmap.md).
