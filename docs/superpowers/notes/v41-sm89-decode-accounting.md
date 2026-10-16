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

**The conclusion for the plan is negative and it is the useful part:** a 40-layer emulated decode body
is ~8.5 captures a layer × 40 layers × 106 MiB ≈ **36 GiB**, against the ~10 GiB a 24 GiB card has
left once the 13.4 GiB of weights are loaded. A 23% cut does not change that verdict. Fitting the body
needs about **3–4× more**, i.e. ≤ ~33 MiB a capture at this shape — so the remaining work is not "a
few more percent of plumbing", and the honest options are:

1. a node count an order of magnitude smaller than the elementwise transcription can reach (which
   points at the emulation moving onto the device as a real kernel, in `relic-core`, rather than
   staying a sequence of torch ops), or
2. capturing only part of the body — a graph per layer leaves the same total, so this would have to be
   a much smaller unit — or
3. accepting the eager path on sm_89 and attacking the eager dispatch instead, which the CPU census
   and this script both now make measurable.

Option 3 is the one the current evidence supports: the emulation is correct and the eager step's cost
is now countable, while a full-body graph needs a change of kind rather than of degree.

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
rather than the one under test.