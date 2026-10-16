# V4.1 on sm_89: what the recorded numbers actually measure

**Status: the comparison is confounded, and that is measured rather than argued. Two numbers that are
read side by side in this tree are not comparable, and nobody has yet taken a like-for-like sm_75 vs
sm_89 reading of the decode step.**

This note is the accounting behind
[the sm_89 baseline row](https://github.com/lvyufeng/RelicLLM/pull/204). It supersedes the reading
that the 8× RTX 4090 is 2.6–2.9× slower than the 4× RTX 2080 Ti at V4.1 decode, because the two
numbers were not taken on the same decode path. It also sizes the thing that actually blocks the
4090, which is not the card.

## The two numbers, and the commits they were taken at

| Reading | Card | TPOT median | Recorded at | What the decode path was |
| --- | --- | ---: | --- | --- |
| `tests/fixtures/perf/v41_decode_graph_on.json` | 4× RTX 2080 Ti (`sm_75`) | **0.3704 s** | `4063fca`, 2026-10-03 | decode graph captured; **no mma emulation** |
| `v41_decode_graph_on_sm89.json` (PR #204) | 8× RTX 4090 (`sm_89`) | **1.0891 s** | 2026-10-15 | **mma emulation on; graph refused, so eager** |

Both were read back from the committed files and the commit graph:

- The emulation — and with it the graph refusal — entered the tree in `ca5a514`
  ("feat(v41): reproduce the reference fp8 GEMM's mma reduction", #170), **2026-10-13**.
- The 2080 Ti record's own `git_commit` is `4063fca`, **2026-10-03**, and
  `git merge-base --is-ancestor ca5a514 4063fca` is **false**. At that commit
  `_reference_reduction_enabled` did not exist, so nothing refused the recording and its
  `--backend-option decode_graphs=true` took effect.
- The 4090 record was taken **2026-10-15**, after `ca5a514`. Its argv also passes
  `decode_graphs=true`, and `V41Backend._resolve_decode_graphs` sets `_decode_graphs = False` when
  `_reference_reduction_enabled()`, printing `EMULATION_REFUSES_GRAPHS`. So its graph was refused and
  the step ran eager.

**So the 2.6–2.9× is 2080 Ti{graphed, no emulation} against 4090{eager, emulation}, and the card is
the least of the three differences.** Two consequences follow, and both are load-bearing:

1. **There is no valid sm_75 number for the current decode path either.** The emulation is on by
   default on every card, and the refusal is arch-independent, so a 2080 Ti run today is also eager.
   The 0.3704 s figure is a measurement of a path the tree no longer serves by default.
2. **The graph's recorded 2.49× / 3.3× was measured on the non-emulated path too.** Nothing in the
   tree shows what the decode graph is worth *with the emulation on*, because that configuration has
   been refused since the day the emulation landed.

## Why the emulation is not optional

`POCKETLLM_V41_MMA_EMULATE` is on by default, and
`relicllm/models/deepseek_v4_1/mma.py` says why in its own docstring: the reference's tilelang kernel
issues an `mma.sync`, and an `mma.sync` does not add 32 products exactly. Measured on the
checkpoint's own recorded `wq_a` operands, a correctly rounded GEMM lands on the other side of a bf16
boundary for **2.41%** of output elements, and those flips compound into different greedy tokens. A
port that computes the correctly rounded answer is *more* accurate and answers differently, which is a
failure here. So the emulation is the cost of running the reference's answer, not a debugging knob —
and "turn it off so the graph fits" is a change of behaviour, not a tuning flag.

## What one emulated decode GEMM costs

`scripts/v41_emulation_cost.py` counts ATen dispatches under `torch.profiler`, on CPU, so the number
is trackable per change rather than per GPU run. The split that matters is **views** (no kernel, and
during CUDA-graph capture no node) against **kernel-launching** ops (every one of which is a
candidate graph node, and on the eager path a host dispatch).

One call at `wq_a`'s real geometry, `K = 5120` so 160 K blocks:

| Shape | total ops | views | kernel-launching | K blocks |
| --- | ---: | ---: | ---: | ---: |
| decode, `M = 1` | 1167 | 526 | **641** | 160 |
| decode, `M = 9` (the recorded oracle shape) | 2155 | 680 | **1475** | 160 |
| prefill, `M = 13` | 2811 | 780 | **2031** | 160 |

The decode-`M=1` census, largest first: `copy_` 201, `add_` 160, `empty_strided` 56, `to` 44,
`_to_copy` 40, `empty_like` 17, `empty` 15, `sub` 11.

Three things are readable off that:

- **`add_` is exactly 160 — one per K block, and it is the floor.** The fp32 accumulation order
  across K is the one place the reference itself rounds, and `fp8_block_gemm`'s docstring is explicit
  that the K loop must stay outermost and accumulate in K order. This op is not removable without
  changing the answer.
- **The rest is plumbing: ~480 of the 641 are copies, dtype conversions and allocations.** `copy_`
  alone is 201. These produce no arithmetic; they exist because the operands are re-derived and
  re-cast per tile.
- **A decode step multiplies this by the emulated linears.** `quantized_linear` has 11 call sites in
  `relicllm/models/deepseek_v4_1/`, the per-layer ones among them — every fp8 weight the checkpoint
  stored a grid beside takes the emulated path, the shared experts included. Over 40 layers that is
  on the order of **10^5 kernel launches a token** at `M = 1`. (The per-step total is an estimate
  from the per-GEMM census rather than a measurement of a whole step: it needs the count of calls a
  layer actually issues, which is a property of the loaded checkpoint.) For comparison, the step this
  tree measured before the emulation —
  [the 2080 Ti probe of 2026-10-01](../../performance/deepseek_v4_1_flash_remaining_bottlenecks.md),
  `7,076` kernels and 11% device-busy — is one to two orders of magnitude smaller. The 7,076 figure
  describes the *non-emulated* step, and quoting it beside an emulated run is the same error this
  note exists to stop.

## The finding

**The 4090's decode is not slow because it is a 4090; it is slow because a correct decode now issues
~10^5 host dispatches that cannot be graphed.** The card is idle in both runs. Attributing the
difference to the architecture is what the confounded pair invites and what the measurement refuses.

This also re-ranks the work. The roadmap's Phase 2 is "take the host out of the loop", and the
emulation is now the largest single contributor to the host loop it did not have when the phase was
written. The lever is therefore not "port the runtime to more cards" — it is **make the reference-exact
path dispatch-bound no more**, on the card the tree already runs.

## What this changes in the plan

| Was | Is |
| --- | --- |
| The 4090 is 2.6× slower; bring up more cards to compare | The pair is confounded; take a like-for-like matrix first: {sm_75, sm_89} × {emulation on, off} × {graph on, off} |
| Turn the emulation off to fit the graph | Not available: it changes the tokens. The emulation has to get cheaper, or graphable, or both |
| Port to 3090 / V100 for more throughput | After the accounting: 3090 has no measured like-for-like number yet either, and V4.1 does not fit a V100's host bank at all |
| Optimization target: fewer kernels in the tree | Optimization target: the ~480 plumbing ops per GEMM, with `tests/test_v41_mma_reduction.py`'s 1.14M-element bit-exactness sweep as the guard |

The two directions the measurement points at, in order:

1. **Cut the per-GEMM plumbing.** ~75% of the kernel-launching ops are `copy_`/`to`/`_to_copy`/
   `empty*`. Decoding each operand once per tile, keeping one dtype through the reduction, and
   hoisting or reusing the temporaries are the candidates; the op count above is how a change is
   scored, and the bit-exactness sweep is how it is accepted.
2. **Then revisit the graph.** `EMULATION_REFUSES_GRAPHS` records ~897 MiB of driver graph memory a
   layer at the current node count against ~10 GiB free. If (1) cuts the nodes by the factor the
   census suggests, the recording becomes a memory question again rather than a refusal — and that is
   answerable by measurement on the box. **It has now been measured, and the answer is that the
   op-count cut is real but roughly three times short. See below.**

## What a capture actually costs, measured on a 4090

`scripts/v41_graph_memory.py` captures the emulated GEMM into real `CUDAGraph`s and reads
`torch.cuda.mem_get_info` around each one, because the cost is *driver* graph memory: the allocator's
own counters barely move and torch's graph pool stays under 20 MiB, so `memory_allocated` cannot see
it. Measured on ts-133 (8× RTX 4090, torch 2.9.1+cu128, one card), `M = 1`, `N = 1280`:

| Tree | K blocks | first capture | marginal | 
| --- | ---: | ---: | ---: |
| `a0abe9a` (master) | 160 | 164 MiB | — |
| this branch | 160 | 126 MiB | **106 MiB** |
| this branch | 80 | 88 MiB | 68 MiB |
| this branch | 40 | 66 MiB | 44 MiB |
| this branch | 20 | 44 MiB | 22 MiB |

Three readings, and each one matters:

- **The two optimizations cut the driver cost by 23%, 164 → 126 MiB a capture.** The op count fell
  36% and the memory fell 23%, so the two are related but not the same quantity — a node is not one
  `aten::` entry, and the copies and allocations the census counted are cheaper per node than a
  compute kernel.
- **The cost is sub-linear in K, and it is a node cost.** 20 → 160 K blocks is 8× the blocks and only
  2.8× the memory (22 → 106 MiB marginal), so a fixed per-graph component dominates at small shapes.
  Extrapolating upward, the marginal cost is roughly 0.6–0.7 MiB a K block at the acceptance shape.
- **The recorded ~897 MiB a layer reconciles with this.** 897 / 106 ≈ 8.5, which is the number of
  emulated GEMMs an attention layer has (the package has 11 `quantized_linear` call sites, not all
  live). So the refusal's number and this measurement are the same quantity, taken two ways.

**The conclusion for the plan is negative and it is the useful part:** the tree's own measured figure is
~897 MiB a layer, so a 40-layer emulated decode body is ~35 GiB against the ~10 GiB a 24 GiB card has
left once the 13.4 GiB of weights are loaded. A 23% cut does not change that verdict. Fitting the body
needs roughly **3–4× more**, i.e. ≤ ~33 MiB a capture at this shape — so the remaining work is not "a
few more percent of plumbing". (The per-layer 897 MiB and this per-GEMM 106 MiB are the same quantity
taken two ways — 897/106 is the several GEMMs a layer declares — but the per-*step* total is not simply
that product, because the layer's several GEMMs are not all `wq_a`-shaped; the wall-clock below is what
bounds that inference.)

## What the emulation costs in wall-clock, and the proof that it is dispatch

Same box, same tree, `torch.cuda` timing with the shape varied one axis at a time:

| shape (M, N, K) | emulated | a plain bf16 `F.linear`, same shape | ratio |
| --- | ---: | ---: | ---: |
| 1, 1280, 5120 | 18.8 ms | 0.123 ms | 152× |
| 1, 1280, 2560 | 12.8 ms | 0.120 ms | 107× |
| 1, 1280, 640 | 8.8 ms | 0.115 ms | 76× |
| 1, 5120, 1280 | 10.0 ms | 0.113 ms | 89× |
| **1, 128, 5120** | **18.1 ms** | 0.113 ms | 160× |
| 9, 1280, 5120 | 36.5 ms | 0.142 ms | 257× |

**The fifth row is the finding.** Cutting N from 1280 to 128 — a tenfold smaller output, a tenth of the
arithmetic and a tenth of the bytes — moves the time from 18.8 ms to 18.1 ms. The cost is a function of
K alone (about 5.5 ms fixed plus ~0.07 ms a K block), which is what a per-launch cost looks like and
what an arithmetic or bandwidth cost cannot. **The emulation is dispatch, all the way down**, which is
the same statement the CPU census makes with a different instrument, and it is why the lever is the
node count and not a faster kernel.

A cross-check falls out of the two numbers, and it is worth stating even though the served measurement
below had to be withdrawn: **the recorded 1089 ms a step is about 58 `wq_a`-equivalents at 18.8 ms
each**, not the several hundred a naive "every layer's every linear" would give. So either the model's
per-step inventory is far smaller than its linear count suggests — most of those GEMMs are much
smaller than `wq_a` — or the recorded figure was taken under conditions this note has not identified.
Which of the two it is cannot be settled without a completed served scenario, and that is the open
question this note leaves rather than guesses.

## What this leaves, honestly

The graph route is not closed by a further op trim of the same kind. Three ways forward, in the order
the evidence supports:

1. **Accept the eager path on sm_89 and attack the eager dispatch.** The emulation is correct and its
   cost is now countable two ways (the CPU census and this script), so the eager step is a target that
   can be scored. This is the only option with no unknown in it.
2. **A node count an order of magnitude smaller.** Which means the emulation stops being a sequence of
   torch ops and becomes a real kernel in `relic-core` — a change of kind, not of degree, and the one
   that would make the graph fit.
3. **A smaller capture unit.** A graph a layer leaves the same total, so this has to be smaller still,
   and the sub-linear fixed cost (~5.5 ms a capture) argues against many small graphs.

## The served A/B was wrong, and the cross-check that shows it

A served A/B was run on this box and its numbers are **void**. It is written down because the way it was
caught is the reusable part.

What was done: both trees served on ts-133 with the recorded argv verbatim, against the same staged
457.8 GiB bank, with the decode rate read off the server's own `generation_tokens_total` over a fixed
window. That produced 0.133 tok/s here and 0.100 tok/s for `a0abe9a` — 7.5 s and 10.0 s a token.

**That cannot be right, and this tree's own records say so:**

| reading | cards | decode rate |
| --- | ---: | ---: |
| `tests/fixtures/perf/v41_decode_graph_on.json` (TPOT median 0.3704 s) | 4× RTX 2080 Ti | **2.70 tok/s** |
| sm_89 baseline, PR #204 (TPOT median 1.0891 s) | 8× RTX 4090 | **0.92 tok/s** |
| this run, `a0abe9a` and this branch | 8× RTX 4090 | 0.10 / 0.13 tok/s |

An eight-card 4090 running **20× slower than four 2080 Ti** is not a card result, and no coverage
difference explains it: `7caa151` (#175, which turns the routed experts onto the mma rule) is an
**ancestor** of the recorded baseline's own `git_commit` (`1c9a9ecc`), so both runs emulate the same
linears; the argv matches flag for flag; and both served runs print the same residency lines
(`experts on cuda:0 x4, context 8192, 288 pooled rows a card, prefill 4096 tokens a forward`).

**The mistake is in what was measured, not in what was run.** `generation_tokens_total` was read while
`requests_total` was 1 — i.e. during the **warmup** request, the first one after startup. That request
pays for exactly the things a cold server has not yet done: populating the 288 pooled expert rows from
the host bank, the lazy kernel and module init, and an empty prefix cache. The recorded scenario sets
`num_warmups: 1` precisely to keep that out of the measurement; reading the counter inside it measures
the one request the scenario exists to discard.

So the served A/B is withdrawn. What stands from the box work is the **per-GEMM microbenchmark**, which
is a controlled comparison — warm, single call, same shape, measured object, no server in the loop:

| shape | `a0abe9a` | this branch |
| --- | ---: | ---: |
| M=1 N=1280 K=5120 | 26.8 ms | **18.8 ms** |
| M=1 N=1280 K=640 | 11.4 ms | **8.8 ms** |
| M=9 N=1280 K=5120 | 47.9 ms | **36.5 ms** |

The lesson is cheap to state and was expensive to learn: a rate read off live counters is only a
measurement of the configuration you think it is if the run is in the state you think it is in. A
completed `relicllm bench` scenario — which reports `tpot.median` over the measured requests and
discards the warmups — is the instrument for this, and it is what objective (4) actually asks for.

## Reproducing

```bash
# operator census: no card, no relic_core (mma.py imports torch alone)
python scripts/v41_emulation_cost.py
python scripts/v41_emulation_cost.py --json     # to diff two revisions

# driver graph memory: needs the card, and reads the tree it sits in
cd <checkout>; python scripts/v41_graph_memory.py --m 1 --n 1280 --k 5120 --repeats 40
```

Both were run against two revisions the same way: `git worktree add <dir> <rev>`, copy the script in,
and run it from that worktree — an installed `relicllm` would measure whichever tree is on `sys.path`
rather than the one under test. The wall-clock table above is an ad-hoc timing of the same call and is
not committed as a script (it needs the card and a plain `F.linear` baseline to compare against).