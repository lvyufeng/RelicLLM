# Qwen4-Exp performance on eight RTX 4090s

## Where this stands (8x RTX 4090, TP8, FP8, indexer budget 2048, batch 1)

The measured configuration is eight RTX 4090s, TP8 for the dense half with EP8 round-robin for the
routed experts, FP8 weights, the checkpoint's native `indexer_budget=2048`, and one request at a
time. Both execution modes produce identical tokens.

| measurement | value | section |
|---|---|---|
| 8192 prompt, 256 new tokens | **48.9-49.1 tok/s, 20.1-20.2 ms** median inter-token | "Re-verified after the speculative work" |
| 262015 prompt, 128 new tokens | **33.9 tok/s, 28.98 ms** median inter-token | "The score op as a GEMM" |
| 262080 prompt, 32 new tokens | 31.5-33.0 tok/s, 29.95 ms median inter-token | "256k re-measured" |
| graph versus eager | identical tokens at both contexts | "The graph and eager modes agree" |
| 8192 with every collective removed | **61.63 tok/s, 15.56 ms** | "Why the collectives cost what they cost" |
| 262080 with every collective removed | 25.55 ms | "Where the 256k step's extra 10 ms actually goes" |

**100 tok/s is not reached, and the measurements say why it cannot be reached by tuning.** The
tightest bound is the 8192 communicate-free ceiling: with all 96 all-reduces deleted, which is not a
configuration anyone can ship, the step is still 15.56 ms and 61.63 tok/s. A 100 tok/s step is
10.00 ms. So even a free fabric leaves 56% more arithmetic than the target allows, before any
context cost at 262080 is added -- and at that context the same arm measures 25.55 ms.

The cost decomposes, all measured rather than inferred:

| bucket | 262080 step | what would shrink it | status |
|---|---:|---|---|
| indexer scoring (context-dependent) | ~9.4 ms | proportional to the pooled-key count, so cheaper scoring | incremental indexer on (+16% at 256k); the score op was reshaped as a GEMM for a further 0.61 ms |
| fixed arithmetic and launch execution | ~15.6 ms | halving several thousand small kernels | a fusion programme |
| 96 collectives | 3.8 ms | nothing available | no device-to-device path on any of the 56 pairs |
| MTP | -- | a correct verify graph | blocked: the recordable expert path is 4.6 GiB too large with the head |

The speculative-decoding work is written up separately in
[MTP verify graph](qwen4_exp_mtp_verify_graph.md): the mechanism is correct at 8192 and 1.33 tokens a
step, and it cannot pay because its round cannot be captured. A KV hot-row cache was measured to be
the wrong lever for the context cost -- that cost follows the pool, not the number of selected
positions.

The three sections that follow are the handover, the record of which source changes are in play, and
the historical TP4 material. After those begins the measurement record proper, in the order it was
made, including the findings that were later corrected. Where a number in the measurement record
disagrees with the table above, the table is current and the section says so.

## Handover: what is done, what is not, and the three things to do next

**Done and verified.**

- Decode timing is a drained interval measured from the emit callback, with prefill reported
  separately; the earlier conditional step-sum timer is gone.
- The graph and eager modes produce identical tokens at 8192 and at 262080, which is the check that
  the QSA positions, the PLE history and the SSM recurrence survive a replay at an advanced position.
  It was missing from every earlier measurement in this document and is now recorded.
- `qwen4_exp_gated_delta_bf16_forward_states` returns the recurrent state after every position,
  verified bit-identical to the plain entry point and equal to the state a shorter scan reaches
  (`tests/test_qwen4_exp_gdn_states.py`, five tests, run on the card). That is the mechanism a
  partial-accept rewind needs.
- `pin_layers` weighs a layer on the host before moving it, so a budget that cannot hold a layer
  declines it instead of allocating it first and failing
  (`tests/test_qwen4_exp_moe_pin_budget.py`, three tests).
- The baseline is reproducible: 49.0 tok/s at 8192 and 33.6-34.3 tok/s at 262015 with identical
  token output, and the commands in "Reproduction commands" were executed verbatim while being
  written.

**Not achieved.** 100 tok/s. The measured bound is why: with every collective deleted -- not a
deployable configuration, just a ceiling -- the 8192 step is 15.56 ms and 61.63 tok/s, and at 262080
the same arm is 25.55 ms. A 100 tok/s step is 10.00 ms. The remaining cost is several thousand small
latency-bound kernels, which is a fusion programme.

**The three things to do next, in order of how much is known about them.**

1. **Fuse the elementwise work.** A third of a step's launches are one- and two-thousand-element
   kernels; the census in this document names them by count. This is the only item with the
   magnitude the target needs, and the least measured of the three.
2. **Make the verify round recordable, or stop calling MTP a candidate.** The mechanism is correct
   at 8192 and returns 1.33 tokens a step; it cannot pay because the round cannot be captured, and
   the capture fails because the one MoE backend that has no host read needs 4.6 GiB more than the
   card has once the head is aboard. `qwen4_exp_mtp_verify_graph.md` has the measurement and the
   ordered plan. The NVFP4 checkpoint that would free 6.2 GiB is the specific unblocker.
3. **Trim the indexer's top-k**, the largest single stage at 0.577 ms a layer (6.92 ms a step over
   twelve layers, against 4.6 for the score and 4.7 for its tail). Doing it exactly is not obvious:
   the obvious speedup, dropping the sort, returns a different *token set* and so is a quality
   change. This is the best-understood remaining item and the least disruptive to attempt.

**What not to retry**, because each was measured and did not work: NCCL knobs of every kind
(`NCCL_PROTO`, `MAX_NCHANNELS`, `P2P_DISABLE`), fp16 for the hyper-connection path (9.5% slower),
caching fp16 mirrors of dense weights, host-side KV at 8192 (equal), the argmax exchange (flat),
block-token cache (-2%), and a KV hot-row cache (the context cost follows the pool, not the selected
count, so a row cache cannot touch it).

## What changed during this work, and why the shipping numbers are untouched

A reader comparing this tree to the one that produced 23.08 tok/s at 262144 should know which source
changes are in play and which of them the headline numbers depend on. Every behavioural change added
here is behind an environment switch that defaults off, and the one unconditional fix is on a path
the shipping configuration does not execute.

| change | gate | on the shipping path? |
|---|---|---|
| `qwen4_exp_gated_delta_bf16_forward_states` (per-position state) | separate entry point | no -- only the verify path calls it |
| `moe.py`: `pin_layers` weighs a layer before moving it | none, unconditional | **no** -- see below |
| `moe.py`: row-at-a-time experts for a verify round | `POCKETLLM_QWEN4_MOE_VERIFY_ROWWISE=0` | no, defaults off |
| `runtime.py`: attempt a verify-round capture | `POCKETLLM_QWEN4_MTP_GRAPH=0` | no, defaults off |

The `pin_layers` reordering is the only ungated change, and it is off the shipping path: the
resident-fill path this configuration uses logs `resident experts 14.06 GiB ... after the fill`,
while `pin_layers` prints `pinned N/48 layers`. The latter message appears in no baseline run, so the
function is not executed there. It was checked rather than assumed, because a silent change to the
shipping path would invalidate every number above.

Nothing in this work changed a default: the baseline measured 49.0 tok/s at 8192 and 34.20 at 262015
after every edit, with token output matching the reference prefix each time.

## Historical: heterogeneous TP4 (superseded)

> **Status correction (2026-11-10):** The numbers below are historical observations, not a
> validated equivalent-quality baseline. A subsequent audit found wrong QSA indexer query RoPE
> rows, request state shared through the model's PLE object, and CPU PLE hash/table rows frozen
> by graph capture. Short matching text did not exclude these faults. The P0 work now separates
> query rows from the full pooled-key table, owns PLE state per request, refreshes graph-private
> pinned PLE rows before replay, and restores accepted state transactionally on release/failure.
> Standard `decode_s` is now a drained wall interval with setup reported separately and only
> the tokens after the prefill-first token counted as decode work. It is not comparable to the
> previous conditional step-sum timer. A local-shard near-tie heuristic is not a token-parity
> guarantee. Complete 262,144-context correctness and 100 tok/s have not yet been established.

## Native-budget correctness checkpoint (2026-11-10)

The controlled TP8 runs retained FP8 weights, an effective indexer budget of 2048,
BS=1, and the same native CUDA binary. A 2303-token teacher-forced prefix with
12 steps passed on all eight ranks: zero top-1 differences, bitwise-equal logits,
and KV/GDN/PLE-state agreement at `atol=rtol=1e-5`. The forced-handback variant
also passed (11 successful replays, one handback, four captures). These are small
numerical/lifecycle oracles, not task-quality or 256K-capacity benchmarks.

The initial 8192-token, 512-output ABBA run **failed** strict output reproduction.
Both graphs and the final eager request had the same output, but the first eager
request diverged at zero-based output index 119. Full graph wall rates were
7.224 and 7.498 tok/s; setup consumed 41.651 and 41.223 seconds. Their 17.572 and
18.975 setup-excluded rates are algebraic wall residuals, not GPU-only timings.
The cold/warm discrepancy therefore was not attributed to graph arithmetic alone.

A warmed, delayed-DMA microprobe then read old token 7 on all three immediate
pinned-CPU reads, changing to expected token 42 only after synchronization. Eager
token staging now waits for D2H completion before CPU history/hash operations;
owned graphs already bypass this transfer through their private CPU-history hook.
The corrected probe returned 42 immediately in all three trials. A subsequent
8192-token A/A/G/A control, retaining the original 8705-token arena and emitting
128 tokens per request, passed strict token equality and returned exit code zero.
Its graph request replayed all 127 decode steps, with 7.467 tok/s full wall and
10.280 seconds of setup. This verifies the previously divergent prefix; it does
not retroactively validate the failed 512-token ABBA run or score a task answer.

Two further optimizations remain opt-in pending device validation:

- `POCKETLLM_QWEN4_INCREMENTAL_INDEXER=1` caches completed-group mean/norm/RoPE
  values, leaving full-history score/top-k work and retrieval budget unchanged.
  The CPU comparison passed 227 combined cases and 68 enabled original QSA cases.
  At native 256K geometry its F32 pool adds 0.375 GiB/rank, excluding other memory.
- `POCKETLLM_QWEN4_GRAPH_REUSE=1` seeds the accepted last slot into slot zero and
  reuses fixed graph/input addresses rather than recapturing every small window.
  Its CPU tests cover parity, release at rebased slot zero, first-step handback,
  partial-copy rollback and arena bounds. The combined staging/reuse regression
  run passed 105 cases. Device oracles passed at 2303/8192 tokens (12/36 steps,
  1/2 captures, 2/8 rebases, bitwise-equal logits, `top1_mismatches=0`, states at
  `1e-5`), including a forced handback at 8192 tokens with 8 rebases.

### Recorded-step throughput, 8192 prompt / 512 outputs

One loaded TP8 process, a fresh request cache per sample, native budget 2048,
window 32, no profiler. All three samples emitted 512 tokens identical to the
first sample; each replayed all 511 decode steps. "Full wall" includes every
capture/rebuild inside the request; "excl. setup" is the algebraic residual and
is not a GPU-only time.

| Graph reuse | Incremental indexer | Capture s | Rebuild s | Rebases | Full-wall tok/s | Excl. setup tok/s | ITL p50 ms |
|---|---|---|---|---|---|---|---|
| off | off | 2.64 | 41.86 | 0 | 7.12 | 18.74 | 52.52 |
| on | off | 2.49 | 0.00 | 15 | 17.88 | 19.59 | 50.41 |
| on | on | 2.67 | 0.00 | 15 | 17.74 | 19.55 | 50.91 |

Reuse removes 41.86 s of recapture from a 512-token request and raises full-wall
throughput 2.5x, because the recorded step's arithmetic is unchanged: one capture
plus 0.10 s of slot seeding replaces sixteen captures. The incremental indexer
changed nothing measurable here (19.59 against 19.55 excluding setup), which is
expected while score/top-k still spans the history and the per-step recorded cost
is not normalization-bound. Neither result approaches 100 tok/s: the recorded step
itself is about 50 ms, so 100 tok/s requires that step to reach about 10 ms.

### What one recorded step spends, by kernel

CUPTI over eight replays of the same 8192-position recording (profiling inflates
absolute time, so the shares and counts are the evidence, not the totals):

| kernel | share | per step |
|---|---:|---:|
| `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` | 43.8% | 96 launches |
| `gather_rows_bf16_kernel` (host KV rows) | 20.3% | 24 launches |
| `qsa_scalar_bf16_kernel<3>` | 14.2% | 12 launches |
| `fp8_expert_act` + `fp8_expert_down` | 4.2% | 96 launches |

The step is therefore communication- and host-transfer-bound, not arithmetic-bound.
Two all-reduces a layer is the design: the expert partition is round-robin, so
`ShardedMoE` masks foreign routes and the per-layer sum that already followed the
MLP reduces the partials, and a 2560-wide row-parallel sum for attention. Both are
5 KiB latency-bound collectives, and on this box the eight cards span two NUMA
nodes joined by SYS.

That reading gives one immediately testable consequence: at 8192 positions the
arena is about 100 MiB, so keeping K/V on the card should remove the whole row
gather. It does, and it is reversible (host/device/host on one loaded model, 512
outputs identical in all three):

| KV arena | Full-wall tok/s | Excl. setup tok/s | ITL p50 ms |
|---|---|---|---|
| pinned host | 17.34 | 19.19 | 51.69 |
| device | 20.25 | 22.74 | 43.56 |
| pinned host | 17.85 | 19.50 | 50.99 |

Device KV is worth about 8 ms a step, matching the profiled gather share. It is a
capacity trade, not a free win: 262144 positions of K/V for the 12 QSA layers are
3.2 GiB a rank on the card against 3 GiB of resident experts, which is why the
host arena exists. The useful shape is a threshold — device up to the context that
fits, host above it — rather than one placement for every request. At 8192 the
ceiling this removes is small against the 96 collectives, which no placement
change touches.

### The recorded step does not grow with context

The row gather reads the same 2048 positions whatever the history is, and the
sparse-attention kernel consumes the same fixed selection, so the recorded cost
should be flat in context length. Measured at 32768 positions with everything else
held at the 8192 configuration (graph reuse on, window 32, 128 outputs):

| KV arena | Prefill tok/s | Prefill s | ITL p50 ms | Excl. setup tok/s |
|---|---|---|---|---|
| pinned host | 325.9 | 100.5 | 53.32 | 17.79 |
| device | 444.3 | 73.8 | 46.17 | 21.43 |

Both decode columns match their 8192 counterparts (19.19 and 22.74), which is the
flatness the kernel profile predicted. Prefill is the number that decides whether a
long context is reachable at all: 32768 positions took 73.8 s on device KV, well
under four times the 8192 figure, so indexing and attention during prefill are not
quadratic in a way that puts 262144 out of reach.

### The kernel profile overstated the collectives, and the check is cheap

Reading the CUPTI table as "43.8% is all-reduce" was wrong, and one run disproves it.
`POCKETLLM_QWEN4_SKIP_REDUCE=1` removes the row-parallel sums entirely -- the text is then
garbage, and the code says so, but the timing is the ceiling with ninety-six collectives
gone:

| 8192 positions, 256 outputs, device KV | ITL p50 | Excl. setup tok/s |
|---|---|---|
| all-reduces present | 43.56 ms | 22.74 |
| all-reduces removed | 39.46 ms | 22.26 |

So the ninety-six collectives cost about 4 ms a step, not the 17-33 ms the profile implied.
The same correction applies to the 11.2 ms `nccl:all_gather` the profile attributed to the
vocabulary gather: an exchange that ships two numbers a rank instead of a vocabulary slice a
rank measures 43.90 ms p50 against 43.56 ms, with identical output tokens. Both numbers are
profiler artefacts -- CUPTI charges a collective for the time it waits behind the preceding
work, and the recorded step is where that waiting happens.

The honest reading of the ceiling run is that about 39.5 ms of a step is arithmetic and
kernel execution, not communication. Scaled by the same profiler inflation that this
correction exposes, the sparse-attention kernel is the largest identified item at roughly a
quarter of it, with the rest spread over some eight thousand small launches a step across
forty-eight layers.

### The scalar QSA kernel was a default, not a requirement

The largest single item in the corrected picture was the sparse-attention kernel, and it was
being taken because of how the dispatch is written rather than because the card cannot do
better. `relic-core`'s `qwen4_exp_qsa.cu` carries two implementations: a tensor-core
`qsa_mma_bf16_kernel` built on `mma.sync.m16n8k8`, and a scalar one. The selector is

```c
const bool use_mma = selected_len >= kMmaKvTile && mma_env != nullptr &&
    std::strcmp(mma_env, "0") != 0;
```

so the tensor-core path needs `POCKETLLM_QWEN4_QSA_MMA` to be *set at all* -- unset means
scalar. Nothing about this model blocks it: `selected_len` is the 2048 budget, well past the
32-position tile, and the binary on this host contains both template sets.

One environment variable, same model, same 8192-position request, same device KV:

| QSA kernel | Excl. setup tok/s | ITL p50 | Full-wall tok/s |
|---|---|---|---|
| scalar (default) | 19.19 | 51.69 ms | 17.34 |
| tensor core | 31.09 | 30.49 ms | 23.48 |

That is 1.62x the decode rate and 41% off the inter-token time, with the 255 replayed steps
producing the same tokens as the scalar path. It is worth stating how this was missed
earlier: the kernel profile had listed `qsa_scalar_bf16_kernel` first by device time, and the
temptation was to conclude the kernel needed rewriting. It did not. The fast one already
existed and was gated behind an environment variable that defaulted to off.

`load_model` now sets that variable when the launch has not, so a run that says nothing gets
the tiled kernel; `POCKETLLM_QWEN4_QSA_MMA=0` stays a working opt-out. A launch with no
variable set measures 32.26 tok/s excluding setup at 30.45 ms p50 against 31.09 and 30.49 for
the explicit setting -- the same path, and the same tokens.

The same change applied to 262144 positions, where the sparse attention is called the same
number of times but against a much longer history:

| 262144 positions | Excl. setup tok/s | ITL p50 | Prefill |
|---|---|---|---|
| scalar QSA | 14.15 | 62.01 ms | 1087.7 s |
| tensor core | 17.35 | 47.41 ms | 1077.1 s |
| tensor core and both host fixes | 19.24 | 43.40 ms | 1077.8 s |

Inter-token time falls 30% across the three and the prefill is unchanged, which is the expected
split: prefill uses the long-sequence path, and these are decode-path and host-staging costs.
Each of the three runs replayed all 31 decode steps and produced the same tokens.

### A default that is correctly off

The same sweep turned up one more dispatch gate that looks like the QSA one and is not: the
fused hyper-connection kernels (`qwen4_exp_inject`, `qwen4_exp_hc_up_mix`, and siblings) are
gated on `POCKETLLM_QWEN4_HC_CUDA`, which defaults to off, with a `rows >= 128` floor. Decode
has one row, so it never qualifies. Forcing both open measures 37.23 tok/s and 25.07 ms
against 37.68 and 25.93 for the elementwise path -- an inter-token gain inside the noise and a
throughput loss. The threshold is doing its job: at a single row, launching the fused kernel
costs as much as the elementwise work it replaces. Left as it is.

### The host-side n-gram hash was doing tensor work for one integer

A recorded step still has host work between replays, and timing the two apart is what found it:
of 31.2 ms a step, 6.59 ms was `fill_history` and 5.82 ms was the replay enqueue itself. The
staging is the host computing the PLE n-gram row ids for all 48 layers -- and `row_ids` built a
full shifted copy of the history per n-gram order (a cummax, a concatenate, two gathers and a
select) to then read a *single* element out of each. At a decode step the history is exactly
`ngram_size` columns wide and one row is wanted, so the same answer is a few integer operations.

| | per layer | 48 layers, per step |
|---|---:|---:|
| tensor path | 99.30 us | 4.77 ms |
| integer path | 5.87 us | 0.28 ms |

The equality is the contract, and the arithmetic is int64 with wraparound: a product that
overflows flips sign, and the modulo takes the floor, so a "mathematically equal" big-integer
product selects a different row. The integer path re-signs each product and XOR to int64 exactly
as the tensor arithmetic does, and a test checks it against torch's own wrapped multiply rather
than against a second Python formula. 6064 cases -- every EOS placement in a 3-token window, over
an alphabet that includes the vocabulary ends, with batch sizes up to five -- agree bit for bit,
and any history wider than one window keeps the tensor path untouched.

Measured on the same 8192-position request:

| configuration | Excl. setup tok/s | ITL p50 |
|---|---|---|
| scalar QSA, tensor hash | 19.19 | 51.69 ms |
| tensor-core QSA, tensor hash | 32.26 | 30.45 ms |
| tensor-core QSA, integer hash | 34.53 | 28.51 ms |

The host gather that follows the hash had the same shape of problem one level down. It routed
four row ids shard by shard: two divides, a `unique`, a `nonzero` and a boolean-index write per
shard, which is eight PyTorch calls to place four rows. Integer `divmod` plus a slice a row does
the same routing in Python, measured at 8.67 us against 47.10 us. The vectorised form stays for
wide gathers, which is what a prefill asks for, and the two agree on every case in the tests --
shard boundaries, repeated ids, and the divisor padding past the last shard.

Timing the step apart from the replay is what made both visible, and it is worth keeping as the
first measurement of any future host work:

| | step wall | `fill_history` |
|---|---:|---:|
| before | 31.23 ms | 6.59 ms |
| after | 26.99 ms | 2.46 ms |

The dispatch and staging fixes together are 1.96x the decode rate and 50% off the inter-token
time, with identical tokens throughout:

| configuration | Excl. setup tok/s | ITL p50 |
|---|---|---|
| scalar QSA, tensor hash, wide gather | 19.19 | 51.69 ms |
| tensor-core QSA, tensor hash, wide gather | 32.26 | 30.45 ms |
| tensor-core QSA, integer hash, wide gather | 34.53 | 28.51 ms |
| tensor-core QSA, integer hash, small gather | 37.68 | 25.93 ms |

### What a step's launch count is made of

Time-ordered and count-ordered profiles disagree about the same step, and the count is the one
that predicts a fusion decision. At 8192 positions a replay launches about 7700 kernels a step,
and the top of the list by count is all PyTorch elementwise work on tensors of a few thousand
elements:

| launches/step | kernel |
|---|---|
| 473 | `unrolled_elementwise_kernel<direct_copy>` |
| 400 | `elementwise_kernel<128, 2, ...>` |
| 338 | `gemvx` (the projections, one matrix-vector each) |
| 316 | `vectorized_elementwise_kernel<4, bfloat16_copy>` |
| 314 | `elementwise_kernel<128, 4, ...>` |
| ~1050 | fourteen further elementwise and reduce variants, ~90-195 each |

Roughly a third of the step's launches are one- or two-thousand-element elementwise kernels, and
none of them is large enough to appear near the top of a time-ordered table. That is the shape of
the remaining cost: not one hot kernel, but a few thousand small ones, which is a fusion programme
rather than a setting. The hyper-connection path in this repository is already fused where a
fused op exists (`hc_split_pre_forward` and `hc_post_forward` carry it), so the elementwise kernels
are spread across the layer arithmetic rather than concentrated in one obviously-fusible place.

The aten census one might reach for here does not help: inside a replay the host makes no aten
calls at all, because the captured graph contains the kernels and not the Python that built them.
About 165 aten calls a step do appear, and they are the staging between replays -- which is the
same work the `fill_history` timing already measures.

### A 3% effect that one run could not see

The launch-count census suggested cutting the model's Python-level work: an eager decode step asks
for about 24000 aten calls, and the ordinary norm alone is eight of them (to float32, `pow`, `mean`,
`rsqrt`, two multiplies, cast back). A fused `qwen4_exp_grouped_rms_norm` was already in the
extension with exactly this model's semantics -- `gain = 1 + weight`, float32 accumulate, output in
the input dtype -- and it was unreachable, because the gate in front of it was `_hc_cuda_enabled`,
whose floor is 128 rows and whose default is off, so a one-row decode step could never qualify.

Routing both the grouped and the ungrouped case to it, one measurement said it did nothing: 36.97
tok/s against 37.53, and it was left off. **That conclusion was wrong, and the fault was the
measurement rather than the kernel.** Run three times per arm, interleaved inside one script so
that a change in the box cannot be mistaken for a change in the kernel:

| arm | tok/s, three rounds | median | ITL p50 |
|---|---|---:|---|
| torch path | 37.85, 37.56, 37.65 | 37.65 | 25.88 ms |
| fused norm | 39.00, 38.77, 38.77 | 38.77 | 25.13 ms |

All six runs produced the same tokens. The within-arm spread is 0.3 tok/s and the between-arm gap
is 1.1, so a 3% effect that a single run could not resolve is unambiguous here.

The mechanism is narrower than "fewer aten calls help a recorded step", which is still false for
the views: the eight calls were not all dispatches. `pow`, `mean`, `rsqrt` and two multiplies are
four to six real kernels, several times a layer, forty-eight layers -- those are device work
whether or not a graph absorbed the Python. `POCKETLLM_QWEN4_RMS_NORM_CUDA` therefore defaults on.

The general lesson is about how the rest of this document's negative results should be read: a
single throughput run resolves about 3-4%, so anything at or below that needs repetition before it
can be called either way. That lesson was then applied to the other finding sitting under the same
floor -- the hyper-connection fusion, which a single run had put at +0.5% -- and three interleaved
rounds priced it at 38.31 against 37.73 tok/s (25.47 against 25.89 ms p50), so it was a real 1.5%
dismissed for the same reason. It defaults on now too, with the row floor dropped from 128 to 1 so
a one-row decode step can reach it at all.

Both together, three more rounds with nothing else changed:

| 8192 positions, 256 outputs | Excl. setup tok/s | ITL p50 |
|---|---|---:|
| fused norm and hyper-connection, three rounds | 39.16, 39.41, 39.50 | 24.69 ms |

That is 4.7% over where the round started, and it also moves the collective ceiling reported
above: that 39.40 tok/s was measured with both of these still off, so the two are additive to the
budget rather than borrowed from it.

At 262144 positions the same two defaults measure 23.08 tok/s at 34.58 ms p50 against 22.90 and
34.63 -- inside the noise of a single run. That is the expected split rather than a puzzle: the
saving is a fixed ~1.2 ms of per-layer elementwise work, which is 5% of a 24.7 ms step and 3% of a
34.6 ms one, and at 262144 the indexer scoring dominates what is left.

That leaves the ~7900 kernels a step as all real device work, and the count as the thing to cut.

### The fixed-slot loop is where the kernel count is made, and it resists

The decode MoE runs a fixed ten-slot Python loop so a capture can hold it. Measured against the
layer count that is the bulk of the launches: ten slots against forty-eight layers, each slot
fetching an expert's two matrices, dequantizing them, running SwiGLU, and folding the route weight
into a sum.

Two things in that loop are not arithmetic and were removed. The ownership multiply was repeated
every slot although it is the same value every slot, and when the rows are the backend's own lines
it is a multiply by one that the caller has already applied -- so it is now done once, and skipped
in that case. Recorded: 37.73 tok/s excluding setup at 25.96 ms p50 against 37.53 and 26.10. That
is +0.5%, which is the noise floor, because nine scalar multiplies in a step of 7700 launches is
not a measurable share of it.

The larger-looking move is `out.addcmul_(contrib, weight)` in place of `contrib * weight` followed
by a separate add: one launch instead of two, ten times a layer, forty-eight layers. It was
rejected on numerics. A fused multiply-add rounds once where this form rounds the product and then
the sum, and in bf16 that is a different number -- 256 off a 50k value in the fixed-slot test, and
the existing tests caught it immediately. Two launches of the same arithmetic beat one launch of
different arithmetic, because every measurement on this branch is compared token for token against
the path it came from.

### Why a narrower weight format cannot buy the target

The obvious next lever after the launch-count census is to make the weights narrower -- NVFP4
over FP8 would halve the bytes a step must read. It is worth writing down why that cannot reach
the target, because the arithmetic is short and it rules out a whole family of proposals.

A decode step is not bandwidth-bound. Each rank owns 64 whole experts a layer (14.06 GiB across
the model) but computes only the one or two the router picked within its shard, so a step reads
about 0.28 ms of expert weights and about 0.6 ms of dense weights. Against a 25.9 ms step that is
under 1 ms of weight traffic out of 25.9 -- roughly 26x headroom. Halving the format halves a term
that was already small.

The device-time breakdown says the same thing from the other side, with the collectives corrected
to the 4 ms that removing them actually measured rather than the 48 ms the profiler attributes
to them:

| category | share of the real 25.9 ms step |
|---|---:|
| sparse attention (already tensor-core) | ~3.5 ms |
| elementwise, copy and reduce kernels | ~3.4 ms |
| GEMMs (`gemvx`, cutlass, cublas) | ~3.3 ms |
| FP8 expert kernels | ~2.3 ms |
| hyper-connection kernels | ~0.9 ms |
| all ninety-six collectives | ~4.5 ms (measured, not profiled) |

The FP8 expert kernels are 2.3 ms, so a perfect four-bit MoE would save at most about 1 ms of a
25.9 ms step -- roughly 4%, which is 40 tok/s and not 100. Even a hypothetical step with *every*
elementwise, copy and reduce kernel removed measures 40.6 tok/s, and those kernels cannot all go.
The step is not dominated by any category; it is dominated by the sum of about 7700 launches,
which is why the remaining work is fusing whole layers rather than narrowing anything.

It is also worth being precise about the card: the FP4 path in the extension is gated at sm_75
rather than at Blackwell, so an RTX 4090 can run it through an int8/dp4a form. That makes it
available, not profitable -- the format is a bandwidth answer to a launch-bound question.

The launch half of that was checked rather than assumed. `moe_single_token_fp4_forward` does
collapse the whole top-k MoE into two launches (`nvfp4_gate_up_kernel`, then `nvfp4_down_kernel`),
which looks like the fusion this step needs -- until it is compared with what runs today. The FP8
resident path already launches exactly two expert kernels a layer. So the four-bit variant would
be the same two launches carrying half the bytes: no launch reduction, and the bytes were never
the constraint.

### The indexer cost grows with the context, and the pool removes it

Long context was where the step time grew: 25.8 ms at 8192 positions against 43.4 ms at 262080.
The sparse attention itself is flat across that range -- it reads the budget's 2048 positions
whatever the context is -- so the growth had to be somewhere that touches every position. The
indexer is exactly that: to choose 2048 positions it scores the whole history, so without a
completed-prefix pool a step re-pools 262144 positions to answer about one.

`POCKETLLM_QWEN4_INCREMENTAL_INDEXER` keeps those pooled keys and appends as positions complete.
It was implemented and left off, and its long-context value had never been measured -- the earlier
check ran at about 2K positions, where there is almost nothing to save. Priced at three contexts,
each run replaying every decode step and matching tokens throughout:

| context | off | on | gain |
|---|---:|---:|---:|
| 8192 | 37.53 tok/s, 26.10 ms | 38.15 tok/s, 25.81 ms | +1.7% |
| 65536 | 30.96 tok/s, 29.74 ms | 32.80 tok/s, 28.48 ms | +6.0% |
| 262080 | 19.24 tok/s, 43.40 ms | 22.34 tok/s, 35.73 ms | +16.1% |

The gain grows with the context, which is what the mechanism predicts and what makes this worth
having as a default rather than a flag: 43.4 ms a token becomes 35.7 ms at 262144 positions. It is
now on unless `POCKETLLM_QWEN4_INCREMENTAL_INDEXER=0`, the opt-out is tested to choose the same
positions, and the pool costs 0.375 GiB a rank at 256K in float32.

### What the extra 10 ms at 262144 positions is made of

With the pool in place the step is 35.7 ms at 262144 against 25.8 ms at 8192, so about 10 ms still
grows with the context. Profiling the two replays and subtracting attributes it to three kernels
and nothing else:

| kernel | 262144 | 8192 | growth |
|---|---:|---:|---:|
| `unrolled_elementwise_kernel<direct_copy>` | 3.29 ms | 0.86 ms | +2.43 |
| `reduce_kernel<128, 4, ReduceOp>` | 2.48 ms | 0.23 ms | +2.25 |
| `vectorized_gather_kernel<16, long>` | 0.78 ms | 0.00 ms | +0.78 |
| `CatArrayBatchedCopy` | 0.67 ms | 0.00 ms | +0.67 |

Everything else is flat across a 32x change in context: the sparse attention kernel measures 3.55
against 3.53 ms, the two FP8 expert kernels are unchanged to two decimals, and `gemvx` moves by
0.06 ms. That is the shape the mechanism predicts -- the attention reads a fixed 2048 positions
whatever the context, and what scales is the part that has to consider every position in order to
choose them.

The four rows are the indexer's scoring pass over all 65536 pooled blocks: a reduction over them
(the scores), a full-size copy of the score tensor, and the gathers that turn the selected blocks
into positions. The pool from the previous section removed the *re-pooling* of completed positions;
it cannot remove the scoring, because the current query legitimately has to be compared against
every pooled key.

One of those four was a layout mistake rather than a cost. The scores were written with the head
axis in the middle -- `bqhd,bkd->bqhk` -- and then reduced over it, so each output walked four
elements a megabyte apart. Writing the axis last and reducing the contiguous one gives the same
four numbers in the same order (checked bitwise, in float32 and from bfloat16 inputs), and it
measured 2.48 ms down to a small fraction of that at 65536 blocks. `relu` became `relu_` for the
same reason: the intermediate is context-sized.

That took 262144 positions from 22.34 to 22.90 tok/s excluding setup and 35.73 to 34.63 ms p50,
with every step replayed and the tokens identical. At 8192 positions it is inside the noise, which
is the expected split -- there are 2048 blocks to reduce rather than 65536.

One nearby guess did not pay and was reverted. `block_tokens` -- the block-to-position table,
also the length of the context -- is rebuilt in every QSA layer every step, which looks like the
same waste; caching it one tensor a cache measured 22.45 tok/s against 22.90 and 35.77 ms against
34.63, so it was worse, not better, and it is gone. The likely reason is that a capture already
folds a pure constant of the geometry, so the rebuild was never costing what it looked like, and
holding the table outside the recording only moved it away from the memory the graph had laid out.
The lesson is the one this whole section keeps paying for: in a recorded step, the cost of an
expression is not readable from its source, and only the measurement settles it.

Cutting the remaining scoring further is an algorithmic change -- pre-filtering candidate blocks,
which would change which positions are selected and therefore the output -- and not a layout or
dispatch one.

### Why the collectives cost what they cost on this box

Ninety-six all-reduces of 2560 bfloat16 values -- 5 KiB each -- measured about 4.5 ms a step,
which is roughly 47 us a collective and would look like a badly tuned implementation. (Measured
again on the current tree: 25.81 ms p50 normally against 21.31 with `POCKETLLM_QWEN4_SKIP_REDUCE`,
which removes all ninety-six. A CUPTI profile of the same step attributes 60.26 ms to them, which
is thirteen times the measured cost -- it charges each collective for waiting on the other ranks,
so the profile can rank kernels but cannot price a collective.) It is not: none of
the eight RTX 4090s has peer access to any of the others. All 56 ordered device pairs report
`can_device_access_peer == False`, so every collective crosses PCIe into host memory and back
instead of being a device-to-device exchange.

That closes the usual answer to a latency-bound small all-reduce. The extension carries a custom
IPC all-reduce (`custom_allreduce_inplace` and its handle/open/close pair), and the DeepSeek-V4
runtime uses it, but it opens by asserting peer access between every pair of devices -- which is
precisely what this box does not have. There is no implementation change that removes the hop; the
cost is the topology.

Four and a half milliseconds of a 25.8 ms step is also not where the target is, which is worth
stating so the topology is not mistaken for the remaining problem.

That ceiling was re-measured on the current tree, because the figure above came from the 39.55
tok/s build and had gone stale. Four interleaved arms at 8192 positions:

| arm | tok/s | ITL p50 |
|---|---:|---:|
| normal | 48.78 | 19.988 ms |
| all collectives removed | 62.34 | 15.612 ms |
| normal | 49.19 | 20.163 ms |
| all collectives removed | 60.91 | 15.535 ms |

So the collectives cost **4.4-4.6 ms, 22.6% of the step**, and removing them gives **+25.8%** --
consistent with the old ratio, but the absolute ceiling is **61.63 tok/s at 15.56 ms**, not 39.40.
That is a much more useful number: even a free all-reduce over this fabric lands at 62 tok/s against
the 100 wanted, so the remaining 15.6 ms of arithmetic and launch execution is 38% more than the
target allows and is the whole of the rest of the problem.

The primitive itself was measured in isolation too, because 47 us a collective invited the question
whether the primitive or the contention was the cost. Eight ranks, a 2560-value bf16 buffer, 2000
back-to-back all-reduces: **243-264 us each**, against 47 us when the same collectives are separated
by compute in the step. The isolated loop is the contended case -- eight ranks issuing continuously
through host memory with no device-to-device path -- so the step's 47 us is already the better end of
this fabric, not a badly tuned primitive waiting to be replaced.

### Two re-tests the better method settled

Two findings had been decided by a single run, which the interleaved method now says is not enough.
Both were re-run three times per arm, alternating inside one script, at 8192 positions.

**Exchanging rank maxima instead of gathering the vocabulary.** Each rank takes its local argmax
and the ranks trade two numbers rather than a vocabulary slice, which is exact -- ties resolve to
the lowest global index, checked against the full gather in the tests -- and cuts a 600 KiB
all-gather to 16 bytes. Measured: 39.32 against 39.55 tok/s excluding setup, 24.65 against 24.69 ms
p50. No gain, confirming what the single run had said. The gather was never on the critical path;
the profile that claimed it was had charged the collective for waiting behind the recorded step,
which is the same accounting error as the all-reduce table.

**Hyper-connection arithmetic in fp16.** This one had a real mechanism on its side: the
hyper-connection path is elementwise on width-10240 tensors, and elementwise is bandwidth-bound
rather than tensor-core bound, so a narrower dtype should move half the bytes -- unlike the dense
projections, where fp16 and bf16 run at the same tensor-core rate and no gain was available.
Measured: 35.80 against 39.55 tok/s, 27.05 against 24.69 ms. **9.5% slower.** The elementwise ops
are not the whole path: entering fp16 needs `normed.to(torch.float16)` and leaving it needs a cast
back, and both are themselves width-10240 passes, so the narrowing pays two extra traversals to
save one. It stays off.

So the two gains found last round were not the start of a pattern. Both of them replaced several
*real kernels* with one fused launch; these two either remove nothing that was on the path or add
conversions. The distinguishing question is not "is there a narrower or cheaper form of this code"
but "how many kernels does the recorded step lose", and only the measurement answers it.

### The split-K default changes the text, and both kernels are right

Shipping split-K was justified with "tokens identical" and a `torch.equal` check on one decode
shape. Both of those were too strong, and re-measuring found it:

| arm at 8192 / 256 out | tok/s | ITL p50 |
|---|---:|---:|
| `POCKETLLM_QWEN4_QSA_SPLITK=1` | **48.65** | 20.14 ms |
| `POCKETLLM_QWEN4_QSA_SPLITK=0` | 41.86 | 23.36 ms |

The speed is real and reproducible, but **173 of 256 tokens differ**, first at position 76. The
"identical" that was recorded came from the bench's own check, which compares a run against the
other mode *inside that run* (eager against graph) -- it never compares one environment against
another, so it could not have detected this. Reading it as a cross-configuration result was a
mistake worth stating plainly, because it is the check the rest of this work leaned on.

The numbers say neither kernel is wrong. Twelve random 2049-row selections at 8449 keys, each
compared against a float64 reference built from the same selection:

| | value |
|---|---|
| worst `abs(mma - splitk)` | 0.000244 |
| worst relative | 2.2e-3 |
| error of MMA against float64 | 0.00024 - 0.00047 |
| error of split-K against float64 | **identical to MMA's, trial for trial** |

So split-K adds no error. It rounds differently from MMA in a few places, at bf16 output precision,
and greedy decoding turns one flipped near-tie at position 76 into a different continuation. That is
chaos, not a defect -- but it does mean the two paths are not interchangeable, which matters:

**It is a second reason speculation cannot work as things stand.** Plain decode is `seq_len == 1`
and takes split-K; a verify forward is `seq_len > 1` and takes MMA. The two disagree by 2.4e-4, so
the verify judges drafts against an answer the plain chain would not have given. Any working MTP
therefore has to put both paths on one kernel -- which is why the options are "run MTP with
`SPLITK=0` on both sides and give up the 6.8 tok/s" or "give split-K a multi-row form", not "leave
the default and fix the draft loop".

### The verify forward is consistent; the stack around it is not

Before blaming the draft loop, the verify step itself was tested. At 2048 positions, prefilled, one
token consumed, the same three drafts fed as one, two and three rows:

| rows in the forward | argmax of row 0 | max abs logit difference from the one-row forward |
|---|---:|---:|
| 1 | 561 | 0.0 |
| 2 | 561 | 0.766 |
| 3 | 561 | 0.672 |

Row 0 is the same computation either way. The logits move by a bf16-sized amount but the answer does
not, so **a K-row verify does judge drafts against what a one-row forward would have said**. The
verify is not where speculation breaks.

What the same probe found instead: prefilling those 2048 positions in one forward gives 561 for that
position, while `generate`'s `chunk_size=512` prefill over the same tokens gives 271. Same prompt,
same model, same position -- a different answer, because the prefill segmentation changed the
numbers.

So the draft loop diverges not because one of its parts is wrong but because **this stack contains
several numerically distinct paths that greedy decoding amplifies**: split-K against MMA (measured
above, 2.4e-4, both equally close to float64), chunked against single-shot prefill, and
`seq_len`-dependent kernels. Greedy decode has no tolerance: a 2.4e-4 difference at a near-tie
produces a different token, and every token after it follows. Speculation is only sound if the plain
chain and the verify take *one* path, and today nothing enforces that.

That makes the fix larger than the draft loop: it is a "one arithmetic path" property over the whole
decode stack -- the same attention kernel, the same prefill segmentation, the same `seq_len`
handling -- before an accept decision means anything. It is not a bug to patch in one place and it is
the reason this work stopped where it did rather than reporting a working MTP.

### The experts change arithmetic at two tokens, and that is the biggest single path difference

The row-invariance problem has a concrete home, and it is not the attention. `ShardedMoE`'s backends
branch on the token count:

```python
if hidden_states.shape[0] == 1 and self._resident:   # fused FP8 decode kernel
    batched = self._batched_from_bank(...)
if self.grouped_enabled:                             # >= 2 tokens: grouped bf16 kernel
    grouped = self._grouped_bf16(...)
```

One token runs a dequantisation-fused FP8 GEMV; two or more run a grouped bf16 GEMM. That is not a
different tiling of the same arithmetic, it is different arithmetic, and it is the largest single
source of the row-dependence measured above -- a verify row and the same row decoded alone do not go
through the same expert maths at all.

Two consequences were established by measurement rather than argument:

- **Doing the experts a row at a time on the verify path does not fix the round.** Wrapping the
  backend so a K-row verify calls the one-row path per row leaves the divergence exactly where it
  was -- the same 561 against 271 at position 2 -- so the experts are one of several differing
  paths, not the only one.
- **That wrapper is not a usable path anyway.** With it enabled, a 12-token, two-width draft probe
  did not finish inside a 50-minute timeout, twice. Serialising the expert path K times per round,
  on top of the per-expert loop, costs far more than the round could ever save.

So the three differing paths found are: the experts (decode kernel against grouped prefill kernel),
the sparse attention (split-K against MMA), and the prefill segmentation (which moves the final
logits by 1.9). A speculative verify has to agree with the plain chain on all of them, and today it
agrees on none. Making it agree is a rewrite of the decode-side kernels toward row-invariance, not a
configuration choice and not a loop fix.

### Where the 256k step's extra 10 ms actually goes

The same communicate-free ceiling was measured at 256k, which splits the step's context dependence
away from the fixed communication tax for the first time. Interleaved arms, 262080 positions:

| context | normal ITL p50 | collectives removed | communication |
|---|---:|---:|---:|
| 8192 | 20.05 ms | 15.56 ms | 4.49 ms |
| 262080 | 28.82 / 29.93 ms | 25.55 ms | 3.8 ms |

Two things fall out, and one of them corrects an earlier attribution:

- **Communication does not grow with context** -- 3.8 ms against 4.49 ms, the same fixed tax. That is
  expected: 96 collectives of 5 KiB each, and nothing in the payload depends on how much history
  precedes the token.
- **The context cost with communication held out is 9.99 ms** (25.55 - 15.56), which is 35% of a
  256k step. An earlier note in this document attributed only ~2.6 ms of real device work to context
  growth; that number came from comparing two profiles whose collective accounting was inflated, and
  it is **wrong**. The correct figure is ten milliseconds.

That matters because it is the one bucket that is neither topology-bound nor launch-shape-bound: it
is arithmetic that scales with how much history the sparse attention selects from. It is exactly the
target the objective's "incremental QSA / KV hot-row cache" names, and it is the only remaining
target of that size.

### 256k over 128 tokens: the sturdier headline number, quoted as a range

The 262080/32 measurement divides the whole decode wall by a 31-interval window, so a few slow steps
move it: median and rate disagreed (p50 29.95 ms implies ~33 tok/s, the arms reported 31.5-33.0). A
128-token arm is the cheapest fix -- four times the decode window, 127 intervals, no configuration
change -- and it needs the prompt shortened to 262015 so that prompt plus generation plus one stays
inside the native 262144.

Five arms of this shape were run across the session (two in one script, one in a later interleaved
A/B with two reverted variants, all on identical code for the shipping arm):

| arm | tok/s | ITL p50 |
|---|---:|---:|
| 1 | 34.09 | 28.628 ms |
| 2 | 34.31 | 28.571 ms |
| 3 | 33.30 | 29.451 ms |
| 4 | 33.65 | 29.528 ms |
| 5 | 33.52 | 29.144 ms |

so the defensible statement is **32.9-34.3 tok/s with a median inter-token latency of 28.6-29.6 ms**,
and the spread is not run-to-run noise on a fixed machine state -- arms 1-2 and arms 3-5 are hours
apart, and the whole run drifts by about 2% over that span. That is why the top of this document
quotes a range rather than 34.20: picking the best arm would be picking a clock state. The `tok/s`
and the median also disagree slightly within an arm, which is the same tail effect as before
(p99 35-42 ms against a 28.6 ms median).

Two things still hold across all five arms: the output tokens are identical arm to arm, and the
figure is far from 100 tok/s regardless of which end of the range is quoted.

A sixth and last pair was run back to back at the end of the session, both shapes in one machine
state, so the two numbers the summary rests on come from one moment rather than from runs hours
apart:

| shape | tok/s | ITL p50 | p95 | tokens |
|---|---:|---:|---:|---:|
| 8192 / 256 out | 48.87 | 20.174 ms | 22.441 ms | 256/256 |
| 262015 / 128 out | 32.94 | 29.590 ms | 35.370 ms | 128/128 |

Both landed at the **bottom** of the ranges quoted above, and the 262080 arm was below the floor
first written here (33.6) -- which is why the ranges now read 48.9-49.1 and 32.9-34.3. A range is
only as good as the arms behind it, and this pair widened it rather than confirming it.

### The graph and eager modes agree token for token, and that check was missing



Every measurement in this document until now reported `matches_first_eager` as null, and the reason
is worth recording because it was a hole in the verification rather than a property of the build:
`--order graph` never runs the eager path, so the comparison had nothing to compare against. The
field is only populated when eager runs first.

Run properly -- `--order eager,graph --host-kv-order 0,0`, the second value being required because
the per-mode toggles take one 0/1 per mode -- both contexts agree:

| context | mode | `matches_first_eager` | `matches_first_sample` |
|---|---|---:|---:|
| 8192 | eager | true | true |
| 8192 | graph | **true** | true |
| 262080 | eager | true | true |
| 262080 | graph | **true** | true |

Thirty-two tokens, identical in all four arms. This is what "the QSA positions and the PLE/SSM graph
state are right" has to mean at full scale, and it now has a direct measurement: a replayed step at
an advanced position produces the same tokens as an eager step at that position, across all 48
layers -- the twelve that select blocks from a position-dependent indexer, the thirty-six that carry
a recurrence, and the PLE history -- at the context where block selection is most sensitive.

### Where the indexer's context cost sits, stage by stage

Pricing the selection chain at the real 262080 shape (`k = 65536`, four heads, `topk = 512`), per
layer and over twelve QSA layers:

| stage | per layer | x12 layers |
|---|---:|---:|
| the score einsum | 0.383 ms | 4.6 ms |
| score tail: `relu_`, head reduction, causal mask | 0.392 ms | 4.7 ms |
| **`topk`** | **0.577 ms** | **6.92 ms** |
| whole chain | 1.366 ms | 16.4 ms |

The whole chain prices higher than the 9.99 ms the step actually pays for context, because in the
step this work overlaps with other kernels; the point of the table is the ratio between stages, and
it says the top-k is the largest, ahead of both the score and its tail. Two facts make that
surprising enough to record: the score buffer is 256 KB, so 0.577 ms is nowhere near bandwidth --
it is one kernel selecting from 65536 candidates, which is inherently serial inside a CTA -- and the
selected set is all that is consumed downstream, since the token runs get sorted later.

That last observation suggested dropping the sort, and the follow-up corrected the reasoning rather
than confirming it. In isolation `sorted=False` is 1.07x faster (0.5514 ms against 0.5152 ms) and
returns an **identical index set**, which looked like proof that the order is inert. It is not: in an
interleaved four-arm run at 262015, the sorted and unsorted arms produced **different tokens**.

| arm | tok/s | ITL p50 |
|---|---:|---:|
| sorted (shipping) | 33.65 | 29.528 ms |
| `sorted=False` | 33.52 | 29.144 ms |
| tokens | **different** | |

The order a gather happens in is not inert -- the masked `-1` slots and the staged runs downstream
depend on it -- so this is a **quality change, not an optimisation**, and the micro benchmark was
blind to it by construction: it compared index *sets* and the difference is in what the caller does
with the order. The switch was removed and the finding recorded at the call site.

Two corrections follow for earlier text in this document. An earlier claim that the change was
"slower end to end" compared an unsorted arm against a sorted arm run in a different script earlier
in the session -- two timestamps, not an A/B. With the arms interleaved the rates are within 0.4% of
each other on *different outputs*, so neither arm may be called faster. The first number was drift;
the second is a divergence.

### The score op was the wrong shape, and fixing the shape is exact

The stage pricing above left the score einsum at 0.383 ms a layer and blamed its size. Its size is
not the problem. At decode `q` has one row, so `einsum("bqhd,bkd->bqkh")` becomes a batch-of-four
GEMV reading a 65536-by-65536-byte key buffer: 33.5 MB at 0.383 ms is roughly 88 GB/s, an order of
magnitude under the card, because the kernel has four rows of parallelism.

The same arithmetic written as a GEMM with the keys on the M side has 65536 rows:

```python
_qf = q.float().reshape(batch_size, seq_len * self.n_heads, self.head_dim)
scores = torch.matmul(
    block_keys.float(), _qf.transpose(1, 2)
).reshape(batch_size, num_blocks_total, seq_len, self.n_heads).permute(0, 2, 1, 3)
```

It generalises over a prefill chunk -- `(B, K, D) @ (B, D, Q*H) -> (B, K, Q*H) -> (B, Q, K, H)` -- and
it is **not an approximation**:

| check | result |
|---|---|
| isolated values against the einsum form | `same_values: true`, bit-identical |
| 8192 tokens against the recorded reference prefix | identical |
| 262015 tokens, two arms | identical |

Measured:

| shape | einsum | GEMM | delta |
|---|---:|---:|---:|
| score op, per layer at 65536 keys | 0.264 ms | 0.186 ms | -30% |
| 8192 step, ITL p50 | 20.174 ms | 20.140 ms | -0.03 ms (K is only 2048 here) |
| 262015 step, ITL p50 | 29.590 ms | **28.981 ms** | **-0.609 ms, -2.1%** |
| 262015 rate | 32.94 tok/s | **33.94 tok/s** | +3.0% |

This is the one change in the recent work that survived both checks -- the isolated speedup *and*
end-to-end, on identical tokens. The distinction from the three rejections before it is worth
stating plainly: the pool-bf16 change traded away bit-exactness, `sorted=False` changed the selected
token set, and this changes neither. A formulation change that produces the same numbers is the only
kind that can be adopted on a micro benchmark's word.

### The scoring dtype: measured, real, and not reachable in the time available

The score op was priced in isolation, without a checkpoint, at the real 262080 shape
(`k = 65536` pooled keys, four heads, `d = 128`):

| form | per layer | note |
|---|---:|---|
| fp32, as shipped | 0.428 ms | both operands cast |
| bf16 | **0.264 ms** | runs, tensor cores accumulate fp32 |
| the fp32 cast alone | 0.069 ms | |

Across twelve QSA layers that is **~1.97 ms a step**, about 6.9% of the 28.6 ms 256k step, and the
two forms differ by a median 0.14% relative. So this is a real lever that a previous note in this
document dismissed too quickly after a failed experiment.

Why the experiment failed is now known, and it is not the einsum. `_incremental_keys` stores its
pool in **fp32** -- the docstring says so outright: "Float32 stores the exact values the old score
path casts to" -- so `block_keys` is Float while `q` is bfloat16, and a bf16 einsum raises
`expected scalar type BFloat16 but found Float`. The shipping `block_keys.float()` is therefore a
no-op and the real cost is reading a 33.5 MB fp32 buffer rather than 16.8 MB of bf16.

The fix is to store the pool in bf16, which makes the scoring dtype and the storage dtype one
decision. It was implemented, isolated, and then **abandoned for a reason worth recording: it is not
an optimisation, it is an exactness trade.**

Isolating the change three ways settled both the cause and the cost:

| change | `test_qwen4_exp_qsa_incremental.py` |
|---|---|
| pool storage dtype only | 93 passed |
| plus the writes following that dtype | 93 passed |
| plus a score dtype chosen from the *observed* `block_keys.dtype` | 25 failed |
| plus a score dtype chosen from the switch, uniformly | 86 failed with the switch on |

The third row is a genuine bug that the isolation caught: choosing the score dtype from
`block_keys.dtype` scores the incremental and non-incremental paths at different precision, because
the incremental path has a stored pool while the direct path returns the rotated groups in the model
dtype. That would have silently broken an equality the tests exist to hold. The fourth row is the
real answer: even with both paths switched together, the tests fail, because they assert `rtol=0,
atol=0` between "read the incremental pool" and "recompute from raw keys". A bf16 pool cannot be
bit-identical to a recomputation that rounds at a different point, so the 2 ms is bought by giving up
"the incremental path returns exactly what a recompute returns".

That is a design decision about a correctness property, not a tuning knob, and it needs token-quality
evidence rather than a timing run. It is therefore left undone with the measurement and the mechanism
recorded. Anyone picking it up should expect to renegotiate that test, not to satisfy it.

### The scoring dtype is not a lever either

The one cheap change the scoring offers is its input dtype. The score is spelled

```python
scores = torch.einsum("bqhd,bkd->bqkh", q.float(), block_keys.float())
```

and those casts materialise a context-sized fp32 copy of the keys every step: at 65536 blocks the
keys are 16.8 MB in bf16, so the cast reads 16.8 MB and writes 33.5 MB, and the wider read is another
33.5 MB -- about 67 MB a layer, 800 MB a step across twelve layers, roughly **0.8 ms of a 30 ms
step**.

It was wired behind a switch and measured at 262080 positions, interleaved, four arms. **The bf16
arms produced no measurement at all** -- only the two fp32 arms completed, at 30.030 and 30.061 ms p50
(0.03 ms apart, another tight reproduction of the 256k baseline). So the selection path downstream
does not accept the narrower scores as written.

The switch was removed rather than left in the tree: a configuration that crashes is a liability,
and the ceiling is 0.8 ms -- 2.7% of the step -- before the token changes that a differently-rounded
score would force re-verifying. It is recorded here so the idea is not re-derived, and so the reason
it was dropped is visible: too small to matter even if it worked.

### The context cost is the indexer's scoring, and a KV row cache is the wrong lever for it

The discriminator is whether that cost follows the number of selected positions (attention and
staging bound) or the size of the pool they are selected from (scoring bound). It was settled by
measuring one intermediate context with communication held out, which predicts different numbers:

| context | pool vs 8k | collectives removed, ITL p50 | context cost |
|---|---:|---:|---:|
| 8192 | 1x | 15.56 ms | 0 |
| 65536 | 8x | 17.845 / 17.915 ms | **2.32 ms** |
| 262080 | 32x | 25.55 ms | **9.99 ms** |

The 64k arms were two runs 0.07 ms apart. A cost proportional to the pool predicts
`9.99 x 8/32 = 2.50 ms` at 64k; measured 2.32 ms, inside the noise of a two-run spread that small.
A cost proportional to the selected count would have shown the same 9.99 ms at every context, since
the budget -- and therefore the attention and staging -- does not move with history.

So the ten milliseconds are the indexer scoring the pooled keys, and the useful consequence is
negative and worth stating plainly: **a KV hot-row cache cannot touch this.** It can only compress
the attention and staging term, which is fixed at `budget` positions and does not grow with context
at all. The lever that matches the measurement is cheaper scoring, and that is what the incremental
indexer already is -- it re-pools only the completed prefix instead of the whole history, and it is
on by default, measured at 19.24 -> 22.34 tok/s at 262080.



What the code does say, and this is documentation rather than a measurement, is the shape of the two
terms. `Indexer.__init__` sets `block_topk = budget // compress_ratio`, and the comment beside
`POCKETLLM_QWEN4_INCREMENTAL_INDEXER` states the split:

> the layer reads `budget` positions whatever the context is, so halving it halves the attention and
> the staging ... the indexer has to score every position in the context to pick the budget's worth

So a compacted KV window bounds the first term (fixed at the budget, which does not grow with
context) and cannot bound the second (proportional to the context). Which of them owns the 9.99 ms is
exactly what the sweep was meant to settle, and until it does, the useful statement is the negative
one: **halving the budget changes the attention and staging and leaves the scoring alone**, so if a
future measurement shows the step barely moving under that sweep, the row cache is the wrong lever
and cheaper scoring is the right one. Recording it this way keeps a documented mechanism separate
from a measured result, which is the distinction this document has been careful about throughout.

### The decode step is latency-bound, not bandwidth-bound -- and that is why it stops here

A round-number check asked whether the step is limited by memory bandwidth, because that decides
whether any scheduling work is left at all. It is not, and the measurement is worth recording
because it closes the last "maybe there is headroom" question in the step.

Each rank owns 64 of 512 experts, and a token routes to 10 of 512, so it touches
`10 x 64/512 = 1.25` local experts a layer on average. An expert is `gate_up` 2560x640x2 plus `down`
640x2560 in e4m3, about 4.9 MB, so a layer reads ~6.1 MB and the token reads ~295 MB across 48
layers. At the RTX 4090's 1008 GB/s that is **0.29 ms**; it measures **2.35 ms**, which is
**125 GB/s, or 12% of peak**.

The first hypothesis for the 8x gap was that masked foreign slots are computed anyway and their
weights read for nothing. That is wrong, and the kernel says so: `fp8_expert_act_fast_kernel` and
the down phase both start with `if (e < 0) return;`, and the header comment states that a foreign
expert is "skipped and its row is zero". So the masked slots cost a block launch, not a weight read.

The gap is therefore latency. At batch 1 with EP8 the per-rank work is 1.25 small GEMVs a layer, so
96 launches a step each move a few megabytes -- a few microseconds of bandwidth and roughly twenty
of launch and memory latency. Merging them is not available: the layers are sequentially dependent.
This is the same shape as the rest of the step (a third of the launches are one- and two-thousand
element kernels), and it is why the remaining work is a fusion programme and not a setting.

It also sets the target's price honestly. Reaching 100 tok/s means 10.00 ms a step against 20.14 --
a 2.01x cut -- and the step is a sum of many small latency-bound pieces across 48 sequentially
dependent layers. Neither the per-piece bytes nor the settings leave 2x on the table; batching would,
and batch 1 is the premise.

### 256k re-measured: the median is stable, the tail is not

Two interleaved rounds of the standard 256k benchmark (262080 prompt positions, 32 new tokens), run
after every change this session made:

| round | tok/s | ITL p50 | tokens | identical to reference | peak / free |
|---|---:|---:|---:|---|---|
| 1 | 31.81 | 29.131 ms | 32/32 | yes | 22.15 / 0.87 GiB |
| 2 | 31.38 | 29.941 ms | 32/32 | yes | 22.15 / 0.87 GiB |

Mean **31.60 tok/s, ITL p50 29.54 ms**.

That is above the 27.06 tok/s recorded earlier in this work, and the honest reading is that the
difference is **not** a code change: nothing in this session touched the default 256k decode path,
and the more stable statistic -- the median inter-token latency -- barely moved (29.54 ms now against
28.88 ms before). What moved is the tail, which is what `tokens/second` over a 32-token window is
most sensitive to. The plausible cause is the host's clock and thermal state at the time of
measurement (these runs are at 02:00-03:00 local, the earlier ones in the afternoon), not an
improvement.

So the defensible statement of the 256k result is a **range of 27-32 tok/s with a median inter-token
latency of about 29 ms**, measured over five runs across two sessions, with identical token output
and 0.87 GiB free of 23.52. Quoting the best of them as "the" number would be picking a run, and the
target is far enough away that the choice does not change any conclusion.

### Re-verified after the speculative work: the shipping configuration is unchanged

Everything this session added for the verify graph is behind a default-off switch, and the new kernel
is a separate entry point, so the decode path should be untouched. Measured rather than assumed, two
interleaved rounds of the standard 8192/256 benchmark after the last edit:

| round | tok/s | ITL p50 | tokens | identical to the reference sample |
|---|---:|---:|---:|---|
| 1 | 48.98 | 20.218 ms | 256/256 | yes |
| 2 | 49.06 | 20.070 ms | 256/256 | yes |

Mean **49.02 tok/s / 20.14 ms**, and the token prefix is `1596, 1144, 4087, 1156, 579, 1622, 13,
14235` -- the same reference prefix every earlier measurement used. No regression, and no change for
the better either: the speculative work moved knowledge, not this number.

### Why the verify step cannot be captured: the two constraints close on each other

The saved capture traceback names the primary error, which a summary line had been hiding:

```
File "moe.py", line 119, in _run_experts_fixed
    gate_up, down = expert_rows(int(rows[slot]))
CUDA error: operation not permitted when stream is capturing
  cudaErrorStreamCaptureUnsupported
```

That is a **host read of an expert row index inside the capture**, once per slot. The comment at that
line says so and calls it "the price of having it in Python" -- which is true, and means the backend
that contains it can never be recorded.

That identified the backend, and it is not the one the baseline runs. `builder.py` maps the flags:

| flags | backend | MoE arithmetic |
|---|---|---|
| `expert_fused_fp8=True` (the baseline) | `Fp8SlotExpertMoE` | fused kernel pair, no host read |
| `expert_resident=True` without it | `DeviceExpertMoE` | Python slot loop, one host read a slot |

Every MTP probe in this work ran the second one, because the probe read its fused-FP8 choice from
`PROBE_FUSED_FP8` and that defaults to `0`. So the MTP results (correct output, 1.33 tokens a step)
were measured on the non-fused path, and the capture failure belongs to that path rather than to
speculation. Worth stating plainly, because it means the earlier "capture cannot work" reading had
the wrong cause even though it had the right verdict.

Re-running with the fused path -- the one the baseline captures 263 replays of -- gives the actual
answer, and it is memory:

| configuration | fits? | capturable? |
|---|---|---|
| fused FP8 experts + MTP head | **no** -- out of memory at 8192 and again at 4096 | yes (the baseline) |
| non-fused resident experts + MTP head | yes -- 5.04 GiB free at 8192 | **no** (the host read above) |

The fits-with-the-head side was then pushed to see whether the gap was a tuning matter rather than a
wall -- a smaller arena and a much smaller capture window, at three contexts:

| context | graph window | outcome |
|---:|---:|---|
| 4096 | 8 | out of memory, 22.78 GiB allocated |
| 6144 | 8 | out of memory, 22.79 GiB allocated |
| 8192 | 8 | out of memory, 22.79 GiB allocated |

The demand is **the same at every context**, which is the informative part: the shortfall is not the
KV arena and not the capture window, it is a fixed cost of about 4.6 GiB that the head and its
buffers add to a trunk that already peaks at 18.17 GiB. Shrinking the things that scale with context
cannot touch it.

**The capturable expert path does not fit alongside the head, and the path that fits cannot be
captured.** That is a closed loop on a 24 GiB card, and it is the terminal reason the objective's
"correct MTP verify graph" is not reachable here: it is not a missing kernel, not a bookkeeping
error, and not the row-invariance work alone -- it is that the only recordable MoE is 5 GiB too
large once the head is aboard. Freeing that much is exactly what an NVFP4 checkpoint would do (the
measured 6.2 GiB), which is why that is the one lever that could reopen this.

### MTP is correct at 8192 and still cannot pay, because its round cannot be captured

Good news first, and it corrects an earlier reading: at **8192 positions** -- the operating point, as
opposed to the 2048 the first probes used -- the head loads, and a two-draft round is **token
identical to plain greedy**:

| | value |
|---|---|
| context | 8192 |
| head loads | yes (5.04 GiB free after) |
| drafts=2 output | `561, 3841, 13477, 37550, 33075, 888, 279, 15217` -- matches plain exactly |
| tokens per trunk step | **1.33** (8 tokens in 6 steps) |

That inverts the 2048-context conclusions ("multi-draft is broken", "every draft rejected"). Those
were artifacts of a short context whose answer is more sensitive to segmentation, not properties of
the mechanism. Correct bookkeeping plus the per-position state recording is enough for a right
answer.

What it is not enough for is speed. The baseline's 20.14 ms step is a **captured graph**, and a draft
round cannot be captured:

```
[graph_window] capture failed: CUDA error: operation failed due to a previous error during capture
[graph_window] ... 0 attempts, handed back to eager
```

So a round runs eagerly. The plain eager step measured about 150 ms a token against the captured
20.14 ms, and 1.33 tokens a round cannot make up an order of magnitude. Allowing the attempt (behind
`POCKETLLM_QWEN4_MTP_GRAPH=1`) and adding the per-position state the old comment named as one of the
two blockers was not enough -- the verify step still fails during capture and falls back.

That leaves the objective's own wording as the precise statement of what is missing: not MTP, which
now works, but the **verify graph**. Capturing it means the recorded step has to own the K-row write,
the state handback and the rewind, which is the row-invariance work described above arriving from the
other direction.

### The kernels are not row-invariant, and that is what blocks speculation

Chasing why prefill segmentation changes the answer ruled out the obvious suspect and found the real
one. Reading the same 2048 positions in different chunk sizes, then comparing the last position's
logits:

| prefill chunk | argmax | max abs logit delta | mean abs logit delta |
|---|---:|---:|---:|
| 2048 (one forward) | 5388 | 0.0 | -- |
| 1024 | 5388 | 1.951 | 0.246 |
| 512 | 5388 | 1.873 | 0.292 |
| 256 | 5388 | 1.938 | 0.238 |

The argmax survives, but the logits move by **up to 1.9, mean 0.25** -- four orders of magnitude more
than bf16 rounding, and nowhere near a rounding difference. Disabling the incremental indexer changes
these numbers by nothing at all (512-chunk: 1.873 / 0.292 either way), so the pool refresh at chunk
boundaries is not the cause.

The cause is that the kernels' arithmetic depends on how many rows they are given. The direct
measurement, from the verify probe: at one cache state, feeding the drafts as 1, 2 and 3 rows gives
row 0 the same argmax but logits differing by **0.766** and 0.672 -- the same row, the same position,
the same history, a different answer because the batch had more rows in it. A tensor-core attention
tile over Q groups the reduction differently for 1 row than for 3, and 48 layers compound it.

That is the real obstacle, and it is bigger than the draft loop: **a speculative verify forward can
only reproduce the plain chain if every kernel computes a given query row identically no matter how
many rows accompany it.** Nothing in this stack guarantees that today, for attention or for the other
Q-dependent kernels. Speculation's whole premise is "one pass over K positions answers what K single
positions would have answered", and the measurement says this engine does not satisfy it.

Fixing it means making the decode-side kernels row-invariant -- a per-row formulation of the sparse
attention and of anything else whose tiling depends on Q -- and paying whatever throughput that
costs. That is a rewrite of the attention path, not a patch, and it is where this work stopped. The
decode baseline itself is unaffected: a decode step is always one query row, so the 48.65 tok/s
figure is internally consistent and reproducible; what it cannot do is serve as the verify half of a
speculative round.

### MTP cannot currently be run at this configuration, and would not pay if it could

The objective names a correct MTP verify graph, and the 100 tok/s target cannot be reached by making
the step shorter -- 20.14 ms a token is 49.6 tok/s and the arithmetic that remains is at its floors.
A token multiplier is the only mechanism left that could cross 100, so MTP was measured rather than
assumed, for the first time in this work.

Two things had to be got past to measure anything, and both are findings.

**The MTP head does not fit beside a resident FP8 expert bank, and no reserve value reconciles them.**
`load_model(..., mtp=True)` with `expert_resident=True, expert_fused_fp8=True` -- the configuration
that produces the 48 tok/s -- runs out of memory at 23.49 of 23.52 GiB. The head is a full decoder
layer, not a small prediction module, so it needs real room. `expert_reserve_bytes` is the caller's
statement of what else must fit and is what the loader pins experts against, so it was swept:

| reserve | outcome |
|---|---|
| 3 GiB (default) | out of memory during the head's construction |
| 8 GiB | still out of memory |
| 12 GiB | refused by the loader: resident experts need 14.06 GiB, and 14.06 + 12 exceeds the card |

The two constraints close on each other: 23.52 GiB of card less 14.06 GiB of expert bank leaves
9.46 GiB for everything else, and the head wants more than the 8 GiB that was still available. There
is no setting that admits both, so **MTP and the fast expert configuration are mutually exclusive on
a 24 GiB 4090**, and any MTP work has to be judged against the much slower host-staged expert path
rather than against the 48 tok/s figure.

The head's cost was then measured directly rather than bracketed, by loading the same checkpoint
twice in one process and reading the allocator:

| stage | allocated | reserved | free |
|---|---:|---:|---:|
| start | 0.00 | 0.00 | 23.13 |
| after trunk load | **17.51** | 17.59 | 5.34 |
| after trunk freed | 0.00 | 0.00 | 22.93 |
| after head load attempt | **22.78** (failed) | 22.89 | 0.04 |

So the trunk is 17.51 GiB and loading with the head reached 22.78 GiB before running out, which puts
**the head at about 5.3 GiB** -- one decoder layer with its own routed experts, its dense weights and
attention buffers, not a small predictor module. Against the 262144 configuration, whose peak is
22.15 GiB, that would need 27.4 GiB of a 23.52 GiB card. The shortfall is about 4.4 GiB and it is a
measurement, not an estimate.

**A real bug in the eager draft path.** Running it twice in one process raised
"Inplace update to inference tensor outside InferenceMode" from `gather_logits`: it caches
`ctx.gather_parts` for reuse between steps, those were allocated inside `torch.inference_mode()` and
so became inference tensors, and the next call's `all_gather` may not write into them. Fixed by
allocating them with inference mode off, which is what `_QSAIndexerPool` already does for the same
reason.

With that fixed, acceptance was measured with the experts staged from host memory -- slower, but
acceptance and output are properties of the model, not of the speed:

| draft width | trunk steps for 32 tokens | tokens/step |
|---|---:|---:|
| 1 | 34 | 0.94 |
| 2 | 32 | 1.00 |
| 3 | 32 | 1.00 |

**Every draft was rejected -- and the cause was the recycle convention, not the head.** The head
itself is good: asked directly it predicts the backbone's next token 62.5-75% of the time and has it
in its top five 97-98% of the time, against a uniform chance of 1 in 248320.

The chain beyond the first link was the fault. `POCKETLLM_QWEN4_MTP_RECYCLE` chooses what the next
link is conditioned on -- `stream` recycles the four-stream stack the head's decoder layer ends on,
`collapsed` re-broadcasts the post-final-norm vector, which is what the reference implementation
does. It defaulted to `stream`. Traced at 2048 positions, three drafts a round:

| convention | traced chain | trunk steps for 32 tokens | tokens/step |
|---|---|---:|---:|
| `stream` | `drafts=[3841, 3841, 3841]` vs trunk `[3841, 13477, 3841]` | 32 | 1.00 |
| **`collapsed`** | `drafts=[3841, 4087, 13]` vs trunk `[3841, 13477, 310]` | **26** | **1.23** |

Under `stream` every link after the first repeats the first: feeding the head its own four-stream
output leaves it echoing the token it was just handed, which is precisely the failure the
`mtp_accuracy` docstring records having found once before in a different place. `collapsed` predicts,
and its output matches the plain decode where `stream`'s sixth token was 13477 against 37550.
`collapsed` is now the default.

**But the multi-draft path is not correct, and an earlier reading of this was backwards.** Comparing
each width against the *plain greedy output of the same prompt in the same process* -- which is the
only valid reference, and not the 8192 bench baseline an earlier round used -- gives:

| width | output | against plain greedy |
|---|---|---|
| -- | `5388, 13, 271, 248046, 248046, ...` | this is plain greedy |
| 1 | `5388, 13, 271, 248046, 248046, ...` | **identical** |
| 2 | `5388, 13, 561, 3841, 13477, ...` | diverges at position 2 |
| 3 | `5388, 13, 561, 3841, 13477, ...` | diverges at position 2 |

So single-draft speculation is exactly correct and every width above one accepts a token the
unspeculative chain would not have produced -- the opposite of what an earlier round concluded, when
the "wrong" output was in fact the plain reference and the plain output was mistaken for a defect.
A repetitive prompt makes plain greedy degenerate into repeating one token, and that degeneracy read
like a bug in whichever arm happened to agree with it.

The divergence is at position 2, one round in: the trace shows `drafts=[13, 13, 561]` against
`chosen=[13, 561, 271]`, where the trunk's own answer for the second position is 271 and the verify
forward returns 561 and 271 in rows 0 and 1. The verify rows are therefore not aligned with the
drafts they are meant to judge. That is a bug in the accept path, not in the head, which is why the
head's 75% next-token accuracy and the width-1 arm are both right while widths 2 and up are wrong.

Until that is fixed, MTP is *not* counted as progress toward 100 tok/s: width 1 is correct but buys
nothing (48 tokens in 50 trunk steps, slightly worse than plain), and the widths that do buy token
multiples produce text the model would not have generated.

**The cause is identified, and the obvious repair makes it worse.** A verify round runs K positions
and keeps an accepted prefix, so `_rewind_to` needs the recurrence state after *each* position.
`state_log` is the mechanism, but it is only filled in the torch branch:

```python
cuda_result = (cuda_gated_delta_rule(...)
               if self.cuda_rule_enabled and (seq_len > 1 or _gdn_decode_fused) else None)
if cuda_result is None:
    log = getattr(cache, "state_log", None)
    if log is not None and seq_len > 1:      # records a state a position
```

A verify round always has `seq_len > 1`, so the fused scan is always taken, so the log is always
empty, so the 36 gated-delta layers keep the state of all K positions while the sequence advances by
fewer. That is exactly consistent with the measurements: width 1 has `accepted_run == K` and nothing
to roll back, and it is the only correct width.

Giving the log priority over the fused scan is therefore the tempting fix, and it was tried:

| verify path | width 1 | widths 2, 3 |
|---|---|---|
| fused scan (current) | identical to plain greedy | diverge at position 2 |
| torch recording path | **diverges at position 3** | diverge at position 2 |

It is a regression -- width 1 got worse, not better. Forcing the torch branch changes the verify
forward's own arithmetic, so its rows stop agreeing with what plain decode answers even where there
is nothing to roll back. The two paths differ in more than state bookkeeping, so the log cannot be
obtained by switching this branch on; a correct multi-draft verify needs the rollback state recorded
*by the fused path*, which is a relic-core change and not a dispatch tweak. The revert is in place
and the reasoning is left at the call site so the next attempt does not repeat it.

The consequence for the target is direct: the step is 20.14 ms at 8192 and 29.77 ms at 262144, which
is 49.6 and 33.6 tok/s; a captured MTP verify at an acceptance of 1.0 changes neither. Reaching 100
tok/s would need acceptance above 2 *and* a working capture, and this measurement says the first
condition does not hold.

### What the card actually has left, measured at both lengths

The bench now reports the device memory a run really spent (`rank0_cuda` in its record), because
whether a feature can be added is decided by that number against 23.52 GiB and quoting it from
arithmetic invites an argument a measurement ends:

| context | peak allocated | peak reserved | free | total |
|---|---:|---:|---:|---:|
| 8192 | 18.17 GiB | 18.31 GiB | **5.08 GiB** | 23.52 GiB |
| 262080 | 22.15 GiB | 22.90 GiB | **0.87 GiB** | 23.52 GiB |

At 262144 positions there is **0.87 GiB free**. The MTP head is a whole extra decoder layer, and the
sweep above could not build it with 8 GiB available. The shortfall is more than an order of
magnitude, so this is not a tuning problem.

That closes the last route to the target. The objective asks for 100 tok/s at 256k or beyond; a
token multiplier is the only mechanism that could produce it from a ~29 ms step and reach 100, the
only multiplier available is MTP, and MTP needs an order of magnitude more memory than the 256144
configuration has left. **100 tok/s at 256k is therefore not reachable on eight 24 GiB 4090s with
this checkpoint**, independently of how much further the step is optimised.

### NCCL has nothing left to tune on this fabric

`NCCL_P2P_DISABLE=1` was the one setting that had ever looked promising, on the theory that
collectives were probing peer-to-peer support they would never find. Three interleaved rounds an
arm, tokens identical:

| arm | tok/s, three rounds | median | ITL p50 |
|---|---|---:|---:|
| baseline | 48.74, 48.73, 47.51 | 48.73 | 20.13 ms |
| `NCCL_P2P_DISABLE=1` | 48.66, 48.54, 47.97 | 48.54 | 20.13 ms |
| that plus `NCCL_MAX_NCHANNELS=1` | 45.35, 47.79, 48.25 | 47.79 | 20.40 ms |

No gain, so the 47 us a collective costs is the shared-memory path itself rather than a failed probe.
With this, and the earlier sweeps of protocol, algorithm and channel count, **every NCCL knob this
workload can reach has been measured and none of them helps**: 5 KiB has to cross PCIe into host
memory and back, because no pair of these eight cards can address the other's memory directly.

### A bit-exact constant that is worth keeping and worth nothing measurable

`-exp(A_log)` is a weight-derived constant in the gated-delta gate, and it was being recomputed in
each of the 36 layers on every token -- a `exp` launch and a negate of a few dozen values. Caching it
alongside the float32 widening is bit-identical by construction, since it is the same `exp` on the
same bits evaluated once, and a direct check confirms the cached tensor equals the uncached one
exactly. Measured, three rounds: 48.23 tok/s and 20.10 ms against 48.73 and 20.13 before.

That is a wash -- half a token a second down, three hundredths of a millisecond of inter-token time
up. It stays because it is correct and it removes work rather than adding a special case, and it is
recorded here as worth nothing measurable so that no one later counts it as a win. Removing 36
launches of a tiny kernel from a step with 5581 of them is not where this problem is.

### The census after those two changes

Re-run once the sparse attention and the gated-delta recurrence were both fused, because the shape
of a step is what decides where to look next, and it had moved:

| category | launches | ms/step | was |
|---|---:|---:|---:|
| collectives | 100 | 4.74 | 4.5 measured |
| GEMM/GEMV | 845 | 3.65 | 3.66 |
| elementwise/copy | 2899 | 3.27 | 3884 launches, 4.45 |
| FP8 experts | 96 | 2.35 | 2.33 |
| other | 887 | 1.76 | 2.26 |
| hyper-connection | 386 | 1.19 | 1.20 |
| RMSNorm | 136 | 0.54 | 0.54 |
| reductions | 170 | 0.32 | 314 launches, 0.66 |
| sparse attention (split-K) | 24 | **0.30** | 12 launches, 3.53 |
| gated-delta (fused) | 36 | **0.11** | not in the graph at all |

The step went from 6552 launches to 5581. What is left has no single item above 4.8 ms, and the two
inputs that would matter most -- collectives and the dense GEMVs -- are respectively a fact of the
fabric (no peer access) and already at bandwidth.

### [Corrected] Growing the context to 262144 was first read as 2.6 ms; the measured figure is 9.99 ms

> **This section's conclusion was wrong and is kept only as a record of the mistake.** It argued from
> a profiled device-time difference, discounted most of that difference as an all-reduce accounting
> artefact, and concluded the real context cost was ~2.6 ms with "no large context-proportional
> computation left to remove". A later measurement that removes the collectives outright, instead of
> discounting them on paper, gives **9.99 ms** of context cost at 262080 and shows it is
> **proportional to the pooled-key count** -- 2.32 ms at 65536 against 9.99 ms at 262080. Read the
> size of the context term from "Where the 256k step's extra 10 ms actually goes", not from here.
> The mistake is instructive: subtracting two profiles and then correcting one term by an estimated
> artefact is how a 10 ms cost became 2.6 ms, and the fix was to measure a ceiling rather than reason
> about one.

The 256144 step is 29.77 ms against 20.14 at 8192, so roughly 9.6 ms of the step is
context. Attributing it by profiling both and subtracting:

| | launches/step | profiled device ms |
|---|---:|---:|
| 8192 | 5581 | 18.23 |
| 262144 | 5809 | 52.55 |

The profiled difference is 34.3 ms, but **31.7 ms of it is the all-reduce kernel**, which is the
accounting error this document has recorded twice already: the profiler charges each collective for
waiting on the other ranks, and the wait is what grows with the step, not the collective. Discount
that and the real growth is about 2.6 ms:

| grower | delta ms | delta launches |
|---|---:|---:|
| `sgemm_32x128_tn` | 0.49 | 12 |
| `mbtopk` digit counting / cumsum / scan-by-key | 0.27 | 48 |
| `gatherTopK` | 0.04 | 12 |
| everything else | <0.1 each | |

That is the indexer's exact top-k over more blocks, and it is cheap: the block scores are already in
the pooled float32 buffer, and selecting 512 blocks out of 65536 is a handful of kernels. **There is
no large context-proportional computation left to remove at 262144.** The context shows up as wall
time through the collectives waiting longer on a slower step, which is the fabric again, not
arithmetic that could be reorganised.

**[Wrong, per the correction above.]** The claim that followed -- that the target cannot be bought by
making the long context cheaper because there is nothing large left to remove -- does not hold: ten
milliseconds of pool-proportional scoring is exactly something to remove, and it is the single
largest context-dependent term. What does survive is the inventory of the fixed step: 4.5 ms of collectives over a fabric with no peer access, 3.6 ms of dense
GEMV that is already at bandwidth, 3.3 ms spread over 2899 elementwise launches, and 2.4 ms of FP8
expert dequant.

### Host KV and device KV are interchangeable at 8192, and that is the answer

The objective names a KV hot-row cache, so it was worth re-pricing now that the step is shorter.
Three interleaved rounds an arm, tokens identical:

| KV placement | tok/s, three rounds | median | ITL p50 |
|---|---|---:|---:|
| device | 48.43, 48.23, 48.07 | 48.23 | 20.14 ms |
| host | 48.30, 48.52, 48.09 | 48.30 | 20.14 ms |

Equal to the last digit of the inter-token time. That is the expected answer rather than a
disappointment: the indexer reads 2051 selected rows a layer either way, so the rows it fetches are
the same bytes from the same gather whichever arena they sit in, and at 8192 positions the arena is
small enough that neither placement is short of bandwidth. Device KV stays, because 262144 needs it
to fit.

### The gated-delta layer was running its recurrence in torch at decode

The `aten` census of an eager step says `rsqrt` and `sum` appear about five times a layer, which no
fused norm explains. They come from `recurrent_gated_delta_rule`, the single-token branch of the
gated-delta layer: the fused scan `cuda_gated_delta_rule` was gated on `seq_len > 1`, so it served
prefill only and a decode step -- always one token -- fell through to the torch version. That
version does, per layer, two `l2norm` calls, five dtype conversions with transposes, and eight
elementwise passes over a float32 128x128 recurrent state, in each of the 36 gated-delta layers.

The kernel has no single-token restriction of its own; it is a recurrent scan. At one token it
agrees with the torch path to a cosine of 0.999993 on the output and 1.000000 on the state.

Four interleaved rounds at 8192 positions, tokens identical throughout:

| arm | tok/s, four rounds | median | ITL p50 |
|---|---|---:|---|
| torch recurrence | 45.24, 45.84, 45.55, 45.60 | 45.57 | 21.42 ms |
| **fused scan** | 47.89, 47.91, 48.06, 48.51 | **47.98** | **20.21 ms** |

Every fused run beat every torch run, so it is on by default (`POCKETLLM_QWEN4_GDN_DECODE_FUSED=0`
turns it off). At 262144 positions the same substitution moves p50 from 30.81 ms to 29.77 ms, a
fixed ~1 ms that is a smaller fraction of a longer step, as expected.

This is the third finding of the same shape -- a fused kernel in the extension, reachable only above
a threshold that decode never meets: the sparse attention had a `setenv` dispatch test, the sparse
attention's own kernel had a per-row grid, and this one had a `seq_len > 1` gate. Three guesses at
which code paths a decode step takes produced three wrong answers and three real gains.

### The expert kernel's block count is already the right one, and sharply so

One phase at a time, in case the shared number had been hiding something. `blocks_per_expert` drives
both the act and the down launch, so a change that helped one and hurt the other would look like a
plain regression; `POCKETLLM_QWEN4_FP8_EXPERT_DOWN_BLOCKS` separates the down phase so it can be
priced on its own. Three interleaved rounds an arm at 8192 positions:

| down blocks | tok/s (three rounds) | median | ITL p50 |
|---|---|---:|---|
| **4 (default)** | 44.95, 45.37, 45.73 | **45.37** | 21.55 ms |
| 8 | 33.05, 33.08, 32.97 | 33.05 | 29.69 ms |
| 16 | 39.82, 39.91, 39.54 | 39.82 | 24.63 ms |

The hypothesis was that the down phase is latency-bound: with ten slots and only ~1.25 of them
owned by this rank, a grid of (4, 10) leaves about five CTAs doing real work on a card with 128 SMs,
so more blocks should have been faster. It is not -- extra blocks cost more in re-reading the
activation than they recover. The default stays at 4 and the separate knob stays, because it is the
right shape for asking this question again on different hardware.

`POCKETLLM_QWEN4_FP8_EXPERT_BLOCKS` splits an expert's row range across blocks in both phases of the
fused kernel, and its default of 4 carries a comment saying the right value is a property of the
card. Worth pricing on a 4090 rather than trusting the comment, three interleaved rounds an arm:

| blocks per expert | tok/s (three rounds) | median | ITL p50 |
|---|---|---:|---|
| 2 | 12.71, 12.72, 12.67 | 12.71 | 78.20 ms |
| **4 (default)** | 39.58, 39.55, 39.61 | **39.58** | **24.71 ms** |
| 8 | 27.44, 27.33, 27.69 | 27.44 | 35.84 ms |

The default is not merely adequate, it is a sharp optimum, and the cost of getting it wrong is
three times the throughput in one direction and 31% in the other. The mechanism is the one the
comment names: 640 rows of `inter` across four blocks is 32 warps an expert, which fills the
scheduler without splitting the reduction past the point where the launch dominates. Two blocks
leaves the reduction longer than the work, eight adds launches that the smaller per-block range
cannot pay for. All twelve runs produced the same tokens.

MoE grouping was priced in the same sweep (`POCKETLLM_QWEN4_MOE_GROUPED=1` with the token floor
dropped to 1 so it would engage at all): 39.48 against 39.58 tok/s, 24.70 against 24.71 ms. Flat,
so the decode shape gains nothing from grouping and the default of off stays.

### Two more switches that are not levers

The same interleaved method closes two more candidates cheaply.

**A narrower compute dtype.** FP16 is not twice BF16 on this card: an RTX 4090 runs both at the
same tensor-core rate with an FP32 accumulate, and both occupy two bytes. `POCKETLLM_QWEN4_DENSE_FP16`
would additionally cast the *weight* on every call -- `_fp16_fp32_linear` does
`weight.to(torch.float16)` inside the linear, which is 676 materialisations of a constant a step --
so it is off for a reason and stays off.

**The GDN kernel's CTA tiling.** `POCKETLLM_QWEN4_GDN_GROUPS_PER_CTA` takes 1, 2, 4 or 8, and two
rounds each at 8192 positions put them at 39.75, 39.63, 39.52 and 39.66 tok/s excluding setup with
every inter-token time between 24.68 and 24.74 ms. Flat. The fused kernel is not short of
occupancy at this shape, so the default of 1 is kept.

### The largest single item is one kernel using two of the card's 128 SMs

With the kernel census finally complete -- 6541 launches a step across 148 kernel types, all of them
now named rather than truncated at a top-40 -- the largest single item that is not a collective is
the sparse-attention kernel: 12 launches, 3.53 ms, 294 us each. Everything else is either smaller
(gemvx 1.55 ms over 338 launches, the FP8 experts 2.36 ms over 96) or the flat tail.

294 us is not what that arithmetic costs. One launch is three query heads against 2048 selected
positions at head dim 256, which is about 6.3 MFLOP -- 21 GFLOP/s, or one part in eight thousand of
what the card does. The bytes are the same story: 2048 keys and values at 512 bytes each is 2 MiB,
which a 4090 moves in about 2 us. The kernel is 150 times its bandwidth floor.

The reason is in its launch geometry, and it is not subtle:

- `grid = (batch * query_len, kv_heads)` is **(1, 2)** for a batch-1 decode step. Two CTAs on a card
  with 128 SMs. Every other SM is idle for this kernel.
- `kMmaKvTile` is 32 and the budget is 2048, so each CTA walks **64 tiles in sequence**, with a
  `__syncthreads()` between them -- a chain of 64 dependent shared-memory phases per launch.
- `kMmaRows` is 16 and tensor parallelism leaves **3 query heads a rank**, so 13 of the 16 MMA rows
  compute padding.

The standard answer is flash-decoding: split the 64 tiles across many more CTAs, have each keep a
running max and sum, and combine the partials in a short second pass. Sixteen splits would put 32
CTAs to work and cut the serial chain from 64 tiles to 4, which on this shape is the difference
between 294 us and tens of microseconds. With 12 layers that is a ~3 ms a step item -- 12% of the
step -- and the largest single win still identifiable.

It is also not doable from this repository: the kernel is `relic-core`'s (`qwen4_exp_qsa.cu`), so
the production change belongs there and this repository gets the binding.

**The fix was prototyped and measured.** `tests/probe_qwen4_exp_qsa_splitk.py` builds a split-K
version through `torch.utils.cpp_extension` -- a separate short-lived extension in the torch cache,
so nothing in the installed building is touched -- and compares it against both the shipped op and a
plain torch reference of the same arithmetic. Sixteen splits, one CTA a split, each computing a
running max, a running sum and a partial output, combined by a second kernel:

| comparison | max abs diff | cosine per head |
|---|---|---|
| shipped vs torch reference | 4.8e-07 | 1.000, 1.000, 1.000 |
| **prototype vs torch reference** | **0.0** | 1.000, 1.000, 1.000 |
| prototype vs shipped | 4.8e-07 | 1.000, 1.000, 1.000 |

The prototype is bit-identical to the reference and within one bf16 rounding step of the shipped
kernel, because it accumulates in fp32 where the shipped one goes through fp16 MMA. Per call, eager,
one card: **0.958 ms shipped against 0.122 ms split, 7.84x.** Applied to the 3.53 ms the twelve
launches cost a step, that is 8k from 24.69 ms to about 21.6 ms (39.6 to about 46 tok/s), and 256144
from 34.58 ms to about 31.5 ms. Even if only half the ratio survives the graph's own overhead, 8k
still lands near 43 tok/s.

Two things the prototype had to get right, both worth recording for whoever writes the real one.
The scoring loop must keep every warp on the same trip count, because the block-wide barrier that
follows it cannot be reached by some warps and not others. And the softmax cannot be the obvious
one pass: a thread that has already stored `exp(score - max)` leaves a value of at most 1 in a slot
that another thread is still scanning for the maximum, and the first version of the prototype
measured cosine 0.66-0.80 against the reference for exactly that reason. Computing the maximum,
storing the exponentials and summing them are three phases with a barrier between each.

**And substituted into a real decode run, it pays.** A per-call ratio is not a throughput, so
`tests/probe_qwen4_exp_splitk_decode.py` swaps `load_ops` in the attention module for a proxy that
answers `qwen4_exp_qsa_bf16_forward` with the split-K build and forwards everything else to the
shipped extension, then runs the ordinary benchmark. At 8192 positions, 256 outputs, three
contiguous baseline runs against the substituted one:

| context | shipped kernel | split-K | change | tokens |
|---|---|---|---|---|
| 8192 | 39.16 tok/s, 24.70 ms | **43.61 tok/s, 21.47 ms** | +11.4%, -3.23 ms | identical |
| 262144 | 23.08 tok/s, 34.58 ms | **25.71 tok/s, 31.26 ms** | +11.4%, -3.32 ms | identical |

Both contexts gain the same 11.4% and shed the same ~3.3 ms, which is what a fixed per-step cost
looks like: the twelve launches are 3.53 ms at either length, so removing 88% of them removes the
same 3.1 ms whether the step is 24.7 ms or 34.6 ms. The 262144 run reports 31 of 31 graph replays and
`matches_first_sample: true`.

So the 7.84x on the kernel is +11.4% on the step rather than the +17% the proportion suggested, which
is the same lesson as everything else here: only a step measures a step.

The substitution also caught the mistake that a proportion would have hidden. The first run reported
46.44 tok/s and **all-zero tokens**: the proxy had taken the split-K path on the *prefill* call, which
passes 512 query rows each with its own selection, while the kernel implements one query row. It was
fast because it computed a 512th of the work. The shapes, once printed, are worth recording because
they were not what the kernel's own template arguments suggested:

| phase | query | key | selected |
|---|---|---|---|
| decode | `(1, 1, 3, 256)` | `(1, 1, 8449, 256)` | `(1, 1, 2051)` |
| prefill | `(1, 512, 3, 256)` | `(1, 1, 2048, 256)` | `(1, 512, 2048)` |

Decode is one query row against 2051 selected positions, one KV head a rank and three query heads --
exactly the shape the prototype was written for. The proxy now takes the split-K path only when
`query.size(1) == 1`, reports every distinct shape it is handed, and fell back on 192 prefill calls
while serving 24 decode ones.

**It is ported and measured on the production path.** The kernel now lives in
`relic-core/relic_core/csrc/qwen4_exp_qsa.cu` as `qwen4_exp_qsa_splitk_bf16_forward`, registered
beside the general op in `cuda_kernel.cpp` and bound in `attention.py`; the general op is untouched
and still serves prefill. The split-K entry point refuses `query.size(1) != 1` outright, so a
prefill call cannot reach it by accident. On the decode shape the two ops agree bit for bit:

| check | result |
|---|---|
| decode shape, split-K vs general | `torch.equal` **True** (max abs diff 0.0) |
| prefill shape to the split-K op | refused: "split-K QSA is decode-only" |

Three runs through the ordinary benchmark, no probe and no substitution:

| context | before | after | change | tokens |
|---|---|---|---|---|
| 8192 | 39.41 tok/s, 24.69 ms | **45.67 tok/s, 21.43 ms** | **+15.9%** | identical |
| 262144 | 23.08 tok/s, 34.58 ms | **27.06 tok/s, 28.88 ms** | **+17.3%** | identical |

The 8192 runs were 45.70 / 45.67 / 45.36, and the 262144 one reports 31 of 31 graph replays with
`matches_first_sample: true`. With the gated-delta fusion below also on, the same 8192 configuration
measures 48.66 (48.43, 48.89) and 262144 measures **27.06** at **28.88 ms** over three runs
(27.21, 26.68, 27.06) with identical tokens and `matches_first_sample` true in all of them -- so the
step has gone from 24.69 ms to 20.06 ms at 8192, and from 34.58 ms to 28.88 ms at 262144.

The long-context figure is quoted as a median of three runs rather than the single run it used to be,
because a single 262144 measurement moves about 2%: these three span 26.68 to 27.21, and the earlier
one-off of 26.33 sat below all of them. Both land near the 15% a fixed ~3.1 ms per step predicts, and the
production figure is a little better than the 11.4% the substitution probe measured, because this
path carries no proxy and no extra Python.

Nothing in the installed extension's other kernels changed: the rebuild recompiled
`qwen4_exp_qsa.cu` and `cuda_kernel.cpp` only and relinked, and the previous `.so` is backed up on
the box.

For completeness, the rest of the census, per step: collectives 96 launches, FP8 experts 96, gemvx
338, the four hyper-connection kernels 386, the fused norm 136, the sparse attention 12, and about
5400 launches spread over the remaining elementwise, copy and reduce variants at 36 to 97 each. No
item outside those is above 0.5 ms, which is why there is no next lever of this size anywhere but
the attention kernel.

### Transport tuning is not the lever

Four configurations were tried against the same 128-output, device-KV recording.
`NCCL_ALGO=Tree` cannot run at all: the vocabulary-sharded head needs
`all_gather` of `int8` ids, and NCCL has no Tree algorithm for that, which is the
same class of failure the 2080 Ti box reported and not a tuning opportunity. The
other three were neutral or negative, against the 22.74 tok/s device-KV reference:
`NCCL_PROTO=Simple` 19.39, `NCCL_P2P_DISABLE=1` 22.37, single-channel 22.23, with
the same ~43.9 ms p50. A ~250 µs latency floor per 5 KiB collective is what
ninety-six serialized collectives cost on eight cards joined by PCIe across two
NUMA nodes; changing the transport's protocol or channel count does not move it.

### What this does and does not establish

Established by measurement on eight RTX 4090s, FP8 weights, BS=1, native budget
2048: the graph recording is numerically equal to independent eager caches across
QSA and window boundaries (bitwise logits at 2303/8192 tokens, including a forced
handback and eight rebases); a cold/warm eager divergence at 8192 tokens was traced
to an unfenced CUDA-to-host token copy and fixed, with the corrected A/A/G/A control
matching on all 128 outputs; window reuse removes recapture and raises full-wall
throughput 2.5x; and the recorded step costs what the kernel profile says it costs.

Not established: 100 tok/s. The recorded step is 24.7 ms at 8192 positions and 34.6 ms at
262144 against the ~10 ms that target needs. Two of the three costs identified earlier are now
measured rather than assumed: the ninety-six collectives are worth about 4 ms (removed
entirely by `POCKETLLM_QWEN4_SKIP_REDUCE`), and the host-KV row gathers were a real
8 ms and are gone wherever the arena fits on the card. The third, the sparse-attention
kernel, was the real find and is worth 41% of the inter-token time at 8192 and 24% at 262144
-- it was not a kernel problem but a dispatch default. At 262144 positions the ITL is 62.0 ms, so the gap is 6x
there as well and for the same reason. Getting to 100 tok/s from here is not
tuning: it needs the collectives amortised over more work per step (a batched or
multi-token verify) or removed altogether (pipeline stages with one recording each,
which the current runtime refuses for `graph_window`), and a faster QSA kernel than
the scalar one. Both are structural changes, and neither was attempted here.

Long context is a capacity result, not a performance one: 262144 positions decode
at 34.6 ms a token, which is 23.08 tok/s excluding setup.

The speculative path is also not a shortcut yet. The MTP verifier has three
source-level rollback gaps that must close before any of its numbers mean anything:
zero acceptance restores only the head snapshot, leaving target GDN/PLE state from
the rejected verification; partial acceptance repairs PLE token ids but not the PLE
convolution state; and the supported multi-token CUDA GDN path returns before
`state_log` is populated, so a partial acceptance has no per-position recurrent
state to rewind to. An eager transactional verifier that restores the target
snapshot and replays only accepted inputs is the bounded first step, and its
verification rows must also be amortising the 96 collectives rather than replaying
96 calls per drafted token.

### 262144 positions, measured

The native limit does run, once the KV arena is on the card. Two host-KV attempts
failed first, and the reason matters more than the attempts: prefill with a
host-resident arena re-gathers the union of selected rows on every chunk, so the
row traffic over a whole prefill grows with the square of the context. That is a
prefill design limit, not a capacity one — which is why moving the arena onto the
card both fits and finishes.

One request, 8x RTX 4090, FP8, BS=1, native budget 2048, device KV (3.22 GiB a
rank), 262080 prompt tokens and 32 generated for 262144 total:

| Quantity | Value |
|---|---|
| Prefill | 1087.7 s (240.9 tok/s) |
| TTFT | 1090.6 s |
| Decode steps | 31/31 replayed, 0 handed back |
| Excluding setup | 14.15 tok/s |
| Full wall | 6.43 tok/s |
| ITL p50 / p95 / p99 | 62.0 / 72.8 / 84.8 ms |
| Capture | 2.63 s, one rebase at 0.006 s |

The process exits non-zero only because this harness marks a run with fewer than
128 outputs as too short to compare; the generated text is coherent and the
timings above are the whole request. `matches_first_sample` is true and
`matched_output_baseline` is false because a single sample has nothing to be
compared against — neither flag is a correctness verdict.

The number that matters for the target is the ITL: 62.0 ms at 262080 positions
against 43.6 ms at 8192 is a factor of 1.4, so decoding a quarter-million tokens
costs barely more per token than decoding eight thousand. Decode is not
context-bound at this budget; it is still the 96 collectives, and 100 tok/s needs
10 ms, which is 6x away at 256K and unchanged from the short-context finding.

**Date:** 2026-08-31

**Hardware:** 4 x RTX 2080 Ti, TP4 over PCIe Gen3
**Model:** Qwen3.8-Flash-Next, real checkpoint at
`/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next`

## Runtime architecture

The real model has 48 layers, hidden size 2560, 512 routed experts, and top-10
routing. One BF16 routed expert is 9.375 MiB, and the full routed-expert set is
about 225 GiB.

The routed experts are copied explicitly into host RAM during model startup:

- expert ownership is round-robin: `expert_id % world_size == rank`;
- every rank owns 128 experts per layer, or 56.25 GiB across 48 layers;
- the four disjoint rank shards together hold exactly one copy of the expert
  set;
- the shard is pinned by default, with an all-or-nothing pageable fallback;
- only active experts are copied to the rank's GPU and all MoE arithmetic runs
  on the GPU;
- the 95 GiB PLE table remains host-side.

This document uses *host-resident* only for the explicit 56.25 GiB/rank copy.
A warm mmap or page-cache hit is not treated as host residency. The latest warm
preload took about 53 seconds per rank; it is a one-time load cost and is not
included in steady-state throughput.

## Authoritative 8K TP4 prefill result

Measurement conditions:

- 8192 real tokenizer tokens;
- one 8192-token chunk;
- TP4;
- expert cache capacity 1024;
- pinned 56.25 GiB host expert shard per rank;
- first pass discarded as warmup;
- no synchronization-heavy profiler in the timed region;
- all rows below produced token 97792;
- preload excluded.

### Accuracy-first path

The accuracy-first candidate uses transient FP16 tensor-core dense projections
and the CUDA hyper-connection RMSNorm/injection path. Decode rows stay BF16
because all prefill precision gates require at least 128 rows.

| Path | Warm samples | Mean wall | Mean throughput | Peak/rank |
|---|---:|---:|---:|---:|
| BF16 baseline | 788.07 / 784.47 tok/s | 10.419 s | **786.27 tok/s** | 14.108 GiB |
| FP16 dense + CUDA HC | 951.04 / 947.77 tok/s | 8.629 s | **949.41 tok/s** | 13.839 GiB |
| BF16 baseline recheck | 770.43 / 767.59 tok/s | 10.653 s | **769.01 tok/s** | 14.108 GiB |

The candidate is a 20.7% gain over the first baseline mean and reaches a stable
about 950 tok/s, just below the 1000 tok/s target.

A 512-token prompt plus 32 forced-token decisions was checked on Chinese,
English, and code inputs. Chinese and code matched all greedy decisions. The
English arm changed decisions 4 and 23; the baseline margins were 0.7500 and
0.1875 respectively. Therefore this path is still opt-in rather than a strict
parity default.

The prefill precision switches do not affect single-token decode. With cache
1024 and a 512-token prefix, a separate 32-step forced-token A/B measured:

| Prompt | BF16 decode | Candidate decode |
|---|---:|---:|
| Chinese | 4.365 tok/s | 4.363 tok/s |
| English | 4.378 tok/s | 4.388 tok/s |
| Code | 4.367 tok/s | 4.370 tok/s |

This is within run noise and confirms the rows=1 BF16 fallback protects decode.

### Throughput-first path: target exceeded, not promoted

A more accurate FP16 GEMM form uses FP16 tensor-core inputs, FP32 accumulation,
and one BF16 output cast. Applying it to dense projections and the HC mix-down
and mix-up projections gives:

| Path | Warm samples | Mean wall | Mean throughput | Peak/rank |
|---|---:|---:|---:|---:|
| BF16 baseline | 784.68 / 780.69 / 777.29 tok/s | 10.491 s | **780.89 tok/s** | 14.108 GiB |
| FP32-output dense + HC down/up | 1063.42 / 1061.49 / 1059.48 tok/s | 7.718 s | **1061.46 tok/s** | 13.961 GiB |
| BF16 baseline recheck | 765.46 / 762.26 / 760.08 tok/s | 10.742 s | **762.60 tok/s** | 14.108 GiB |

This exceeds the 1000 tok/s goal by 6.1% and is 35.9% faster than the first
baseline mean. It is not promoted as a correctness-safe default:

- 512-token Chinese and code prompts matched all 33 decisions;
- the English prompt changed decisions 19 and 23;
- decision 19 was an exact baseline tie, but decision 23 had a baseline margin
  of 0.1875 and the candidate selected the other token with a 0.9375 margin;
- a 2048-token plus 8-decision matrix matched all decisions on all three
  prompts, so the mismatch is prompt- and margin-sensitive rather than a
  general runtime failure.

The correct status is therefore: **the performance target is reached by an
experimental precision mode, while the accuracy-first mode remains about
950 tok/s**.

The 8192-token chunk is also the correct operating point. A clean strict-BF16
sweep measured 154.70, 257.88, 409.14, 580.68, and 753.25 tok/s for chunks 512,
1024, 2048, 4096, and 8192 respectively. Smaller chunks do not improve
correctness and sharply increase repeated per-layer dispatch, communication,
and active-expert staging overhead.

## Remaining prefill attribution

A synchronization-heavy real TP4 profile is for attribution only. Under the
accuracy-first configuration, the main 8K phases were:

| Phase | Calls | Total | Per call |
|---|---:|---:|---:|
| Routed MoE | 48 | 7.777 s | 162.03 ms |
| Attention | 48 | 1.484 s | 30.92 ms |
| MLP hyper-connection | 48 | 0.863 s | 17.97 ms |
| Attention hyper-connection | 48 | 0.844 s | 17.58 ms |
| MLP all-reduce | 48 | 0.666 s | 13.88 ms |
| Attention all-reduce | 48 | 0.356 s | 7.41 ms |
| Router | 48 | 0.207 s | 4.31 ms |
| PLE | 1 | 0.149 s | 148.98 ms |

MoE attribution on rank 0 reported 4.189 seconds staging, 3.090 seconds
compute, 4652 active expert calls, and 42.59 GiB H2D. Turning off the existing
copy-stream prefetch reduced the no-profile path from about 950 tok/s to about
727 tok/s, so the overlap must remain enabled.

Attention sub-phases were:

| Sub-phase | Calls | Total | Per call |
|---|---:|---:|---:|
| Core | 48 | 0.808 s | 16.84 ms |
| Projection | 48 | 0.294 s | 6.12 ms |
| QSA indexer | 12 | 0.165 s | 13.73 ms |
| Output | 48 | 0.139 s | 2.89 ms |
| GatedDeltaNet convolution | 36 | 0.076 s | 2.12 ms |

The isolated 8K GatedDeltaNet layer measured about 20.2 ms with transient FP16
projections. Its recurrent core was 8.66 ms. Sweeping 1/2/4/8 work groups per
CTA did not improve the full layer, so CTA grouping is not the next lever.

A synchronization-free 512-token PyTorch trace also showed that all 96 NCCL
collectives ran on the default compute stream: NCCL occupied 327.46 ms, BF16
GEMMs 1.353 s, and pinned H2D copies 2.651 s, with zero timeline overlap between
NCCL and either GEMM or H2D. Moving NCCL to a separate stream reduced an
isolated 8192 x 2560 all-reduce from 7.54 to 7.28 ms, but an independent 0.58 ms
GEMM still did not overlap because both kernels saturate SM resources. The real
8K path gained only about 0.8%, so simple stream reassignment is not a path to
strict-parity 1000 tok/s. Projection slicing or a reduce-scatter formulation is
needed to create useful pipeline granularity.

## Numerical gates and rejected candidates

The parity harness generates one BF16 greedy chain per prompt, feeds that same
chain to every candidate, and reports the BF16 top-1/top-2 margin for every
mismatch. Exact ties are separated from real flips.

### Rejected or diagnostic precision paths

| Candidate | Kernel or 8K result | Parity verdict |
|---|---:|---|
| FP16 dense projections alone | fast | English decision 1 flips at BF16 margin 3.4375 |
| FP32-output dense projections alone | 885.48 tok/s | Only an exact-tie English decision changes; no non-tie flip in the 512+32 matrix |
| FP32-output dense + CUDA HC | 956.80 tok/s | English decision 23 flips at margin 0.1875; grouped RMSNorm is not bit-exact |
| FP32-output dense + HC mix-down only | 1013.28 tok/s | English decisions 0 and 1 flip at margins 4.3750 and 3.4375 |
| FP32-output dense + HC mix-up only | 1005.89 tok/s | English decision 0 flips at margin 4.3750 |
| FP32-output dense + HC injection only | 940.27 tok/s | English decision 23 flips at margin 0.1875 |
| FP16 dense + FP16 HC down/up | earlier 1043.35 tok/s was not reproducible | Chinese decision 5 flips at margin 0.9375 |
| FP32-output dense + HC down/up | 1061.46 tok/s | English decision 23 flips at margin 0.1875 |
| Full FP16 routed MoE | 2.65x kernel speedup | English decision 0 flips at margin 4.375 |
| FP16 gate / BF16 down MoE | 1.75x kernel speedup | real Chinese and English flips |
| BF16 gate / FP16 down MoE | 1.23x kernel speedup | English decision 23 flips at margin 0.1875 |

The routed-MoE FP16 experiments are kept opt-in and disabled by default. Their
small average activation error is not sufficient to preserve real generation.

### Exact-BF16 hyper-connection limits

Two scale-then-activation pairs now use dedicated CUDA kernels when
`POCKETLLM_QWEN4_HC_CUDA=1`: HC mix-down divide plus SiLU, and block-injection divide
plus `2 * sigmoid`. Direct tests at 1, 7, and 8192 rows are bit-identical to the
PyTorch BF16 expressions. At 8192 rows the isolated pairs improved from about
0.069 to 0.054 ms and from 0.083 to 0.049 ms respectively. Their aggregate
end-to-end ceiling is only about 5 ms per 8192-token prefill because each pair is
small relative to the model wall.

The larger CUDA grouped RMSNorm used by the same HC switch is not strictly
bit-exact to PyTorch: on real layer-0 weights at 8192 rows, 99.999541% of values
were equal and the maximum absolute difference was 0.0625. Therefore references
to the CUDA HC configuration mean an accuracy-tested near-parity path, not a
bit-exact implementation. The activation kernels themselves remain exact.

A steady-state TP4 run with two global 8K warmups is the usable end-to-end
measurement for the two activation fusions. With prompt=8192, chunk=8192,
TP4, and expert cache=1024, strict BF16 measured 769.12 / 765.31 tok/s
(767.21 mean), CUDA HC measured 801.09 / 799.17 tok/s (800.13 mean), and the
BF16 recheck measured 752.99 / 750.73 tok/s (751.86 mean). All arms produced
token 97792. This comparison includes the pre-existing non-exact CUDA grouped
RMSNorm and injection kernels; it does not isolate the roughly 5 ms activation
fusion gain. A later confirmatory run suffered an external machine-wide
slowdown from about 786 to 428 tok/s and is discarded rather than averaged into
these numbers. A 512-token Chinese prompt plus eight forced decode decisions
also kept every greedy decision unchanged; the full-logit delta reflects the
pre-existing grouped-RMSNorm approximation rather than rows=1 activation use.

An isolated strict-BF16 HC call at 8192 rows measured 25.363 ms. The two BF16
mix GEMMs consumed 7.071 and 7.610 ms; a kernel trace attributed 14.134 of
16.950 ms of CUDA time to `aten::mm`. On SM75 these GEMMs use
`magma_sgemmEx_kernel` at about 50% occupancy.

A separate optimistic compute-floor probe replaced every staged expert pair by
one GPU-resident pair, removing all 42.76 GiB of expert H2D while preserving the
same dispatch and GEMM shapes. At prompt=8192/chunk=8192/TP4, the real path
measured 779.93 tok/s (10.504 s) and the no-H2D floor measured 909.07 tok/s
(9.011 s). Perfect staging overlap therefore still cannot reach 1000 tok/s,
which requires 8.192 s; at least 0.819 s of compute and/or launch overhead must
also be removed. Elementwise or reduction fusion cannot supply that margin. The
remaining strict-path work must target the HC/attention/MoE compute structure,
not H2D overlap alone.

The weighted mean is already a 100%-occupancy BF16 multiply followed by a
100%-occupancy `MeanOps` reduction. Replacing `.mean(-2)` with
`.sum(-2) / hc_count` is bit-identical but slower. Hand-written left-to-right,
pairwise, and `einsum` reductions do not reproduce PyTorch's reduction order;
`einsum` is also much slower. These variants were rejected.

### Split-K and cached FP16 weight experiments

Splitting the K dimension into FP16 tensor-core partial GEMMs does not reproduce
the SM75 BF16 linear result. Equal-element fractions remained roughly
0.9984-0.9997 for every tested chunk size, while BF16 itself was not bit-equal
to an FP32 reference. Chunk sizes below 256 were also slower than BF16.

Caching FP16 mirrors of dense weights did not materially improve isolated GEMMs:
for example, the 8K GatedDeltaNet QKV projection was 2.360 ms with a transient
weight conversion and 2.365 ms with a cached FP16 weight. Permanent mirrors
would consume substantial VRAM without a measurable benefit, so weights remain
BF16 and conversion is transient.

## Confirmed architecture decisions

1. **Keep routed experts explicitly resident in host RAM.** Mechanical-disk mmap
   faults must never be part of steady-state inference.
2. **Keep per-rank round-robin ownership.** TP4 stages one disjoint 56.25 GiB
   shard per rank and never duplicates routed-expert work.
3. **Keep active-only H2D.** The full-local-layer grouped BF16 path increases
   H2D bytes by about 31.6% and regressed real 8K TP4 prefill to 577.13 tok/s.
   `POCKETLLM_QWEN4_MOE_GROUPED` stays disabled by default.
4. **Keep MoE copy-stream prefetch enabled.** It is worth about 950 versus
   727 tok/s in the current accuracy-first path.
5. **Keep prefill and decode precision paths separate.** Large prefill rows may
   use the experimental FP16 tensor-core modes; rows=1 remain BF16.
6. **Do not promote a candidate from token 0 alone.** Real forced-token matrices
   and baseline margins are required.

## Reproduction commands

### The final numbers in this document (8x RTX 4090, FP8, budget 2048)

These are the commands behind every figure quoted in the 8192 and 262080 sections above. They were
run verbatim while writing this section -- the 8192 block returned **49.05 tok/s / 20.222 ms** over
256 of 256 tokens in `mode=graph` with the same token prefix as every other run, which is the
documented figure inside the run-to-run spread.

They assume three things about the shell, since the blocks below show only the `POCKETLLM_*` part:
the conda environment that has `torchrun` on `PATH`, a `PYTHONPATH` naming this checkout and the
relic-core checkout that holds the built extension, and the repository root as the working directory
(the bench has no `conftest.py`, so the CWD is what puts `tests` on the path).

Both numbers come from the same benchmark, and the two flags that matter are `--order` and the
per-mode toggles. **`--order graph` alone never runs the eager path, so `matches_first_eager` comes
back null** -- it is populated only when `eager` runs first. The per-mode toggles (`--host-kv-order`
and friends) take one `0`/`1` per mode, and passing a single value for a two-mode order is a
`parser.error` exit rather than a default.

```bash
# 8192 prompt, 256 new tokens. This is the 48.99 tok/s / 20.05 ms figure.
POCKETLLM_QWEN4_HOST_KV=0 \
POCKETLLM_QWEN4_GRAPH_PAST_BUDGET=1 \
POCKETLLM_QWEN4_INDEXER_BUDGET=2048 \
POCKETLLM_QWEN4_SCHEDULE=by_layer \
POCKETLLM_GRAPH_RETURN_MARGIN=0 \
POCKETLLM_QWEN4_RUN_STAGING=0 \
POCKETLLM_QWEN4_GRAPH_REUSE=1 \
torchrun --nproc_per_node=8 --master_port=62500 \
  tests/bench_qwen4_exp_baseline.py --model /workspace/qwen38-fp8 \
  --context-tokens 8192 --max-new-tokens 256 --graph-window 32 --chunk-size 512 \
  --warmup-s 3 --order graph --host-kv-order 0 --label 8k
```

```bash
# 262015 prompt, 128 new tokens. This is the headline 33.9 tok/s / 28.98 ms figure.
# Prompt + generation + one must stay inside the native 262144, which is why the prompt is 262015
# rather than 262080: 262080 + 128 + 1 would be 262209 and the bench refuses it. Before the score op
# became a GEMM this same command measured 34.20 and 28.60 on one arm and 32.94 on another; the
# quoted 33.9 / 28.98 is the post-GEMM pair.
torchrun --nproc_per_node=8 --master_port=62520 \
  tests/bench_qwen4_exp_baseline.py --model /workspace/qwen38-fp8 \
  --context-tokens 262015 --max-new-tokens 128 --graph-window 16 --chunk-size 256 \
  --warmup-s 3 --order graph --host-kv-order 0 --label 256k-128
```

```bash
# 262080 prompt, 32 new tokens. The earlier 256k arms used this shape; 31.5-33.0 tok/s at 29.95 ms.
# It is tail-pessimistic -- 31 intervals -- and the 128-token arm above is the number to quote.
torchrun --nproc_per_node=8 --master_port=62522 \
  tests/bench_qwen4_exp_baseline.py --model /workspace/qwen38-fp8 \
  --context-tokens 262080 --max-new-tokens 32 --graph-window 16 --chunk-size 256 \
  --warmup-s 3 --order graph --host-kv-order 0 --label 256k-32
```

```bash
# The graph-versus-eager equality check. Both modes, so both toggles need two values.
torchrun --nproc_per_node=8 --master_port=62540 \
  tests/bench_qwen4_exp_baseline.py --model /workspace/qwen38-fp8 \
  --context-tokens 262080 --max-new-tokens 32 --graph-window 16 --chunk-size 256 \
  --warmup-s 2 --order eager,graph --host-kv-order 0,0 --label eg-256k
```

```bash
# The communicate-free ceiling, which is what prices the topology: add SKIP_REDUCE to the env block.
# Its output is meaningless as text -- the point is only the step time.
POCKETLLM_QWEN4_SKIP_REDUCE=1  # ... plus the environment block above
```

Two things about reading these numbers. First, `--order` sequences are interleaved on purpose; a
single arm cannot see a 2% effect, and the two-run spreads quoted throughout are 0.03-0.07 ms.
Second, the `tok/s` figure over a 32-token window is tail-sensitive while the median inter-token
latency is not -- when the two disagree, the median is the one to quote.

### Older 4x 2080 Ti commands, kept for provenance

These are from the pre-4090 measurements and are **not** the commands for the figures at the top of
this document. The `.scratch/` scripts they name no longer exist in the tree.

Accuracy-first 8K A/B:

```bash
PYTHONPATH=. \
POCKETLLM_QWEN4_DENSE_FP16=1 \
POCKETLLM_QWEN4_HC_CUDA=1 \
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun \
  --standalone --nproc_per_node=4 \
  .scratch/bench_qwen4_exp_tp_fp16_matrix.py \
  --prompt-tokens 8192 --chunk-size 8192 \
  --expert-cache 1024 --repeat 3 --global-warmup 2 \
  --arms baseline,dense_hc_cuda,baseline_recheck
```

Throughput-first 8K A/B:

```bash
PYTHONPATH=. \
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun \
  --standalone --nproc_per_node=4 \
  .scratch/bench_qwen4_exp_tp_fp16_matrix.py \
  --prompt-tokens 8192 --chunk-size 8192 \
  --expert-cache 1024 --repeat 4 --global-warmup 2 \
  --arms baseline,dense_fp32_hc_down_up_fp32,baseline_recheck
```

Forced-token margin matrix:

```bash
PYTHONPATH=. \
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun \
  --standalone --nproc_per_node=4 \
  .scratch/check_qwen4_exp_tp_long_matrix.py \
  --prompt-tokens 512 --decode-tokens 32 --expert-cache 0 \
  --prompts chinese,english,code \
  --arms dense_fp32_hc_down_up_fp32
```

## Theoretical ceilings of the heterogeneous scheme

All constants in this section were measured on this machine (4x RTX 2080 Ti,
TP4, PCIe Gen3) with `.scratch/probe_*.py`. Rows marked *projected* combine
measured constants arithmetically and have not been observed end to end.

### Hardware roofs

| Roof | Measured |
|---|---:|
| Pinned H2D, one GPU active | 10.49-10.82 GiB/s |
| Pinned H2D, all four GPUs concurrent | 42.42 GiB/s aggregate (10.61 GiB/s per GPU) |
| GEMM, BF16, 8192x4096x4096 | 7.50 TFLOP/s |
| GEMM, FP16, 8192x4096x4096 | 54.71 TFLOP/s |
| DRAM copy | 494.5 GiB/s effective |
| NCCL all_reduce 8192x2560 BF16 | 7.146 ms (5.47 GiB/s) |
| NCCL all_reduce 1x2560 BF16 | 0.092 ms |

Two of these matter more than the rest. **H2D does not contend across the four
GPUs**: each rank keeps its full ~10.6 GiB/s while all four copy at once, so
per-rank staging math is additive, not shared. And **BF16 GEMM runs 7.3x slower
than FP16** on SM75, because Turing has no BF16 tensor-core path — this single
ratio dominates the prefill ceiling.

### Prefill ceiling

Summing the measured per-kernel costs of one 8192-token TP4 pass (36
`linear_attention` + 12 `qwen_sparse_attention` layers, per-rank shapes):

| Arm | Floor | Ceiling |
|---|---:|---:|
| BF16 everywhere | 7.217 s | 1135 tok/s |
| FP16 dense, BF16 MoE (this PR's arm) | 4.123 s | 1987 tok/s |
| FP16 dense + FP16 MoE (projected) | 2.979 s | 2750 tok/s |
| + tensor-core QSA core, 3.44x (projected) | 2.126 s | 3853 tok/s |

Measured end to end for comparison: BF16 786.27 tok/s, FP16 dense + CUDA HC
949.41 tok/s, no-H2D compute floor 909.07 tok/s. The shipped arm therefore runs
at **48% of its own GEMM floor**; the missing 1.44-2.1x is launch overhead,
elementwise traffic, and Python dispatch, not arithmetic.

Phase attribution of the floor:

| Phase | BF16 | FP16 dense |
|---|---:|---:|
| routed MoE (x48) | 1.326 s / 18.4% | 1.326 s / 32.2% |
| QSA core (x12) | 1.202 s / 16.7% | 1.202 s / 29.2% |
| hyper-connection (x48) | 1.444 s / 20.0% | 0.234 s / 5.7% |
| GDN dense proj (x36) | 1.154 s / 16.0% | 0.158 s / 3.8% |
| NCCL all_reduce (x96) | 0.686 s / 9.5% | 0.686 s / 16.6% |
| shared + router (x48) | 0.657 s / 9.1% | 0.095 s / 2.3% |
| GDN recurrence (x36) | 0.290 s / 4.0% | 0.290 s / 7.0% |
| QSA dense proj (x12) | 0.378 s / 5.2% | 0.053 s / 1.3% |
| QSA indexer (x12) | 0.081 s / 1.1% | 0.081 s / 2.0% |

The hyper-connection is the largest single BF16 phase at 20%, which is why the
FP16 path bought 20.7% end to end. Once dense projections are FP16, the profile
inverts: routed MoE and the QSA core become 61% of the floor together, and both
are still BF16.

The QSA core sustains only 1.03 TFLOP/s, 14% of the BF16 GEMM roof, because it
runs as two `einsum` calls with an FP32 softmax between them. This is the same
shape of problem the `cpp_engine` Qwen path already solved with an SM75
`m16n8k8` kernel, where the analogous exact-GQA prefill gained 3.44x.

**H2D is not the prefill bound and cannot become one soon.** A cold 8192-token
chunk touches all 128 local experts across 48 layers = 56.25 GiB/rank = 5.31 s
at 10.6 GiB/s. Measured exposed staging was only 1.492 s of a 10.504 s wall, so
3.81 s already hides behind compute. Staging becomes the wall only if compute
drops below 5.31 s, i.e. past **1544 tok/s**. Between today's 949 tok/s and
1544 tok/s, every gain must come from compute; above 1544 tok/s the scheme turns
PCIe-bound and needs quantized experts to go further.

### Decode ceiling

Decode is bound by bytes, not arithmetic, and the gap is 3x.

Per token per rank: top-10 of 512 experts, round-robin ownership, 48 layers =
120 expert copies = 1125 MiB = 1.099 GiB -> **103.6 ms of H2D**. Measured decode
compute at 8192 context is 35.0 ms (dense 11.2 + full attention 8.8 + routed MoE
4.4 + NCCL 8.8 + lm_head 1.8), i.e. 28.6 tok/s if H2D were free.

| Scenario | ms/token | Ceiling |
|---|---:|---:|
| Serial H2D + compute | 138.6 | 7.2 tok/s |
| Perfect H2D/compute overlap | 103.6 | 9.6 tok/s |
| + 31% resident cache, uniform routing | 71.5 | 14.0 tok/s |
| INT8 experts | 51.8 | 19.3 tok/s |
| INT8 + 62% resident | 35.0 | 28.6 tok/s |
| INT4 experts | 35.0 | 28.6 tok/s |
| Any scheme with no expert H2D | 35.0 | 28.6 tok/s |

**28.6 tok/s is the hard decode ceiling of this model on this hardware**, and it
is a compute number, not a transfer number. Nothing done to the expert pipeline
can pass it; INT4 experts and fully-GPU-resident experts land on exactly the
same 35.0 ms because at that point H2D has fallen under the compute time. Past
28.6 tok/s the levers are NCCL (8.8 ms, 25% — 96 small all_reduces), the
`linear_attention` dense projections (11.2 ms, 32%), and full attention (8.8 ms,
25%).

Residency is set by the memory budget: 22.0 GiB per card minus 2.54 GiB dense
per rank minus ~2 GiB KV and activations leaves 17.46 GiB, or 1907 of the 6144
local experts (31%). That is what makes the 14.0 tok/s row reachable with BF16
experts and no quantization.

Quantized residency changes the picture qualitatively:

| Expert dtype | Total | Per rank | + dense | Fits 22 GiB |
|---|---:|---:|---:|---|
| BF16 | 229.7 GiB | 57.4 GiB | 60.0 GiB | no |
| INT8 | 112.5 GiB | 28.1 GiB | 30.7 GiB | no |
| Q4_K-ish (4.5 bit) | 63.3 GiB | 15.8 GiB | 18.4 GiB | **yes** |
| INT4 | 56.2 GiB | 14.1 GiB | 16.6 GiB | **yes** |

At 4 bits the entire routed-expert set fits in GPU memory across the four
ranks, which removes the heterogeneous H2D path from decode altogether and puts
decode at its 28.6 tok/s compute floor — a 3.0x lift over the 9.6 tok/s
overlap-only ceiling. This is the one structural change that moves decode by
more than a fraction.

**Speculative decoding does not help the bytes.** Expert selection is nearly
independent across positions: verifying 8 tokens touches 74.7 distinct experts
per layer instead of 8x10=80, only 6.6% fewer bytes per token. Batched verify
amortizes compute, which decode is not bound by, and leaves the actual bound
untouched.

### Ranked levers

Prefill, in order of measured headroom:

1. FP16 or INT8 routed-expert MoE — 32.2% of the FP16-dense floor, still BF16.
2. Tensor-core QSA core — 29.2% of that floor at 14% of the GEMM roof; the
   `cpp_engine` `m16n8k8` kernel is the existing template.
3. Close the 1.44-2.1x gap between the 4.123 s floor and the 8.629 s wall —
   kernel fusion and launch reduction, not new math.
4. HC projection TP slicing — element-wise validated, real all-gather cost
   still unmeasured.

Decode, in order:

1. 4-bit routed experts, fully resident — 9.6 to 28.6 tok/s, 3.0x, and the only
   lever that removes the bound rather than shrinking it.
2. Fewer, larger NCCL calls — 8.8 ms of the 35.0 ms compute floor.
3. `linear_attention` dense projections — 11.2 ms, the largest compute phase.

### Probes

```bash
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
  .scratch/probe_h2d_roof.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
  .scratch/probe_compute_roof.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
  .scratch/probe_attn_roof.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
  .scratch/probe_decode_layer.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/python .scratch/probe_gdn_and_budget.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/python .scratch/ceiling_model.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/python .scratch/ceiling_breakdown.py
```

A caveat on one number: an earlier PyTorch loop measured the GDN recurrence at
596 ms per 8192 rows. That is the naive Python scan, not the shipped kernel,
which measures 8.05 ms. Only the kernel number is used above.

## Open issue: the 6.0x decode gap on RTX 4090

**Date:** 2026-10-17
**Hardware:** 8 x RTX 4090 (sm_89), TP8, torch 2.11.0+cu130, against 8 x RTX 3090 (sm_86), TP8.
**Status:** explained — see *What it was* at the end of this section. The cause is launch
latency on this host, not the card and not the byte traffic.

This section exists so the next person does not re-run the eliminations. It records what has
been ruled out, with the evidence; what is still standing; the claims that were **withdrawn**;
and the one open confound. Read the retractions before trusting the table.

### What it was

Measured 2026-11-05 on ts-134, same host as the observation below, with the extension up to
date. An empty elementwise kernel streamed on GPU 0 costs **41.4–41.8 µs host-side per launch**,
against the **18 µs** the 3090 host (`ts-208`) was measured at in the table below. A bf16 M=1
matmul of the model's own expert shape (2560 → 1280) costs **113.6 µs**, of which 45 µs is the
arithmetic at 145 TFLOPS — the rest is launch. So the same 8,139 CUDA ops a decode step that cost
146 ms of launches on the 3090 cost **~0.92 s** here, which is the whole of the 6.0x.

The card is not the problem, and each of these was measured rather than assumed: persistence mode
on, 450 W cap, clocked to 2520 MHz under load, 917 GB/s device-to-device copy, 145 TFLOPS bf16.
The host is not saturated either: 192 vCPUs, load about 34 with eight ranks running. Expert
residency and staging are identical on both hosts (`64 experts/layer x 48 layers resident, 14.06
GiB`, `staging: 0 misses / 0 hits`).

**The consequence is that this host's eager decode has no headroom at all** — it is 1.08 tok/s,
slower than the 3090's 8.16, on a faster card — and that capturing the step is worth 44x here
against 4.1x there. See the graph section for those numbers.

### The observation

The same model, same checkpoint, same TP8, same 58-token prompt:

| | 8 x RTX 3090 | 8 x RTX 4090 |
|---|---|---|
| decode, steady state | **5.54 tok/s** | **0.9185 tok/s** |
| decode step | ~189 ms | ~1826 ms |

Both boxes were verified to run `world_size=8`, `nproc_per_node=8`, and 64 experts/layer per
rank. This is a real **6.0x**, and it is in the direction that a faster card should not be.

The 4090's steady state is 124 generated tokens stopped on a real end-of-sequence token
(`stopped: eos`), `--expert-resident`, `stage_bytes`/`stage_calls` 0/0, exit 0. The 3090's
number is on the staging path. That asymmetry matters and is addressed under *Open confound*.

### Eliminated, with the evidence

| Suspect | How it was eliminated |
|---|---|
| **The extension never loaded** | The `.so` was genuinely stale at first (built Oct 9, torch swapped Oct 17; `load_cuda_kernel()` returned `None`). After a rebuild the matched config returned `decode_s` **78.52 s -> 78.96 s**, a 0.56% difference over a 394 s wall. A torch fallback does not reproduce a fused-kernel timing to 1%. |
| **No fused kernel on the decode path** | The MoE extension is *not* on the decode path: `_grouped_bf16` is its only consumer and four gates close — `POCKETLLM_QWEN4_MOE_GROUPED` defaults `"0"` (`moe.py:222`), `rows < grouped_min_tokens` (128) fails at decode, `_dequantized_layer_tensors` returns `None` on an FP8 shard (`moe.py:284`), and world size 1. The attention QSA kernel *is* on the path, and was traced running: **2261 calls, 2261 `path=cuda`, 0 fallback** in a 24-token run — with the rate unchanged at 0.783 tok/s. |
| **Staging / PCIe** | Device-resident experts remove **100%** of the H2D (20.52 GiB, 4483 misses -> 0.00 GiB, 0 misses) and buy **+5.0% decode / +10.6% prefill**. On the 3090 the same mode gives **2.7x prefill**. The bytes were never the dominant term on sm_89. |
| **Different bytes per call** | `stage_bytes`/`stage_calls` is 4.6864-4.6897 MiB on every 4090 run against the 3090's 4.6881 — the same regime on both boxes, six runs. |
| **Silicon** | Clocks 2520-2805 MHz against a 3105 max; power 54-79 W against a 450 W cap; PCIe gen4 x16 full width; `SW Power Cap` and `HW Slowdown` both *Not Active* on all 8 cards; a bare GEMM reaches ~62 TFLOPS. Not throttled, not link-degraded. |
| **Torch version** | torch 2.11.0+cu130 is **13-15% slower** than 2.9.1+cu128 on this box — the hypothesis was backwards. It is a real regression, not a hidden fast path. |
| **Collectives** | An uncontended NCCL probe times the exact collective at 0.274 ms; 2 per layer x 48 = ~1.4% of a 4090 step. (The corresponding figure is ~22% of a *3090* step, because that step is 9.7x shorter — the absolute costs agree, the denominators do not. Neither share generalises.) |
| **Storage** | `/workspace` is local NVMe (`/dev/nvme1n1p1 ext4`) at 1.3 GB/s, 134x the 3090's 9.7 MB/s network path. |
| **NUMA / affinity** | 4 NUMA nodes (not 3), GPUs on nodes 0 and 2, every rank bound to its local 12 logical CPUs — which is the correct 24 physical cores per node over 4 GPUs. Not misfiring. |
| **Prefill-vs-decode division** | Both boxes use the same 58 prompt tokens, so `prefill_tps` divides by the same constant and is directly comparable. |

### Still standing

- **Both boxes run the same per-expert path.** `_grouped_bf16` is off by default, gated at
  `rows >= 128`, and returns `None` for an FP8 shard, so prefill and decode both run the
  per-expert loop, one expert at a time.
- **The 4090 does identical work at identical bytes and takes 6.0x longer.**
- **A candidate cost that no current counter sees.** `_expert_matrix` (`moe.py:56-69`, the
  dequantize call at `:67`) is invoked **inside** the per-expert loop (`moe.py:117-118` and
  `:148-149`) and dequantizes **both** expert weights, per expert, per layer, per step.
  `dequantize_block_fp8` (`quant.py:59`, `values = code.to(dtype)` at `:80`) materialises a
  **full dense bf16 copy** that is discarded after two GEMMs. One routed expert is 9.375 MiB
  in bf16 — exactly **2x** its 4.6875 MiB FP8 footprint, and `stage_bytes` counts only the
  FP8 H2D, so it is **blind to this entirely**. This is consistent with the resident result:
  residency changes where the codes live, not that they are dequantized per expert per step.

  **This is not yet shown to be the gap.** It is a real, unmeasured cost that predicts the
  observed uniformity across phases. It would only be *the* gap if the per-expert dequant is
  differentially expensive on sm_89; if it costs the same on both boxes, it is a cost but not
  the difference. The test is count and per-call time of the dequant on both boxes.

### Withdrawn claims

A record that keeps its retractions is worth more than one that looks clean. These were stated
during the investigation and are **not** evidence:

- **"The 4090 stages 65% more bytes"** (20.52 vs 12.43 GiB) — a **token-count artifact**:
  64 tokens against 32. Per-call volume is identical on both boxes. Withdrawn.
- **"~22,000 launches per step"** ceiling — not measured, withdrawn.
- **"Collectives are ~44% of a step"** — an artifact of `profiler.py`'s `scope()`, which
  synchronizes on both entry and exit, so every scope absorbs its predecessors' undrained work
  and reads high. The tell was two byte-identical `[1,1,10240]` reductions reporting 28.2% and
  3.5%. Withdrawn; see the measured ~1.4% above.
- **"Attention is 38.9% of a 4090 decode layer"** — same artifact. Re-measured with
  `attention.py`'s own self-timer (`POCKETLLM_QWEN4_ATTN_PHASE_PROFILE=1`, `time.perf_counter`
  between explicit syncs), attention is **~21.5%** (core 12.9 / projection 3.3 / output 2.4 /
  indexer 1.6 / conv 1.3). Withdrawn.
- **"Prefill is capped at ~7 tok/s, a low-arithmetic-intensity artifact"** — superseded:
  prefill is 19x more *efficient* per token than decode within a box (2.34 vs 45.38 ms/token),
  which is the healthy shape, but that does not explain a **cross-box** prefill gap at
  identical prompt length. The two comparisons are different and only the first was healthy.
- **"The 4090's load is 145 s vs the 3090's 21.7 s"** — `load_s` varied 82 / 120.8 / 145.0 s
  across three identical-config 4090 runs. **Neither box's load time should be quoted.**

### Open confound

There is **no apples-to-apples cross-box pair.** The 3090's figures are on **torch 2.6.0+cu124**
and its resident-mode run has not been taken at all; the 4090's are on **2.11.0+cu130**. Since
torch 2.11 is measurably 13-15% slower here, the two boxes are not on a common baseline. The
operator requires 2.11 on all legs. **Until the 3090 is re-baselined on 2.11 and run in
resident mode, the 6.0x is a real observation but not a clean experiment.**

### What would settle it

A per-expert marginal-cost instrument on the **real** path — cost as a function of expert count
at fixed rows, separating per-expert overhead from per-call overhead, with a **count and a time
per expert**. Two outcomes, both publishable:

- If the 4090 issues more launches per expert, the count explains it and the ~6.4 us flat
  per-launch law transfers.
- If the counts are **identical** and it is still 6x, the law does **not** transfer across
  architectures — "same op count, higher cost per launch on sm_89" — which is a different
  result and must be reported as such rather than forced onto the law.

Timing must not come from a `profiler.py` scope (see *Withdrawn claims*). The gate must be
printed into the log and asserted, so a stale `.so` aborts the run rather than silently
measuring a different graph.

## Steady-state single-request decode on RTX 3090: what the 150 ms step is made of

**Date:** 2026-10-19. **Box:** 8 x RTX 3090 (`ts-208`), TP8, expert-resident, FP8 checkpoint.
**Target this was measured against:** 50 tok/s on *one* request, which needs a 20 ms step.

The step is **146-151 ms**, so 6.6-6.8 tok/s. It is not the MoE, not the collectives, and not the
host's Python: it is **8,139 CUDA ops a step**, and at B=1 each of them is a GPU-blocking launch.

| | per decode step |
|---|---|
| CUDA ops (torch profiler) | **8,139** |
| `F.linear` calls alone | 911 |
| device time, summed over all ops | 104.3 ms |
| host `tolist()` / `.item()` syncs | 97 / 0, **1.6 ms total** |
| wall / GPU-observed | 146.2 / 146.2 ms (no host gap) |

The op count is the same for 8 ranks as for one, because every rank runs all 48 layers with no
tokens to parallelise. At ~18 us a launch (measured, `cudaLaunchKernel` 176 ms over 3,339 launches)
the launches alone are the step: 8,139 x 18 us = 146 ms.

**Eliminated this round, with the measurement that eliminated each:**

- *Routing read back to the host* (`unique.tolist()` twice an expert lists). 97 syncs a step and
  1.6 ms of the 146 -- worth removing for a capture, not worth a step. The device-side variant of
  the scan was written and removed on that number.
- *Batching.* A batch of 16/32/64 rows costs 251/277/309 ms a step against 151 at one, so the
  throughput path exists (`generate_batch`) but a request's own rate is unchanged. Not the target.
- *What the kernels are.* 783 D2D copies, 694 bfloat16 copy kernels, 326 sigmoid, 244 rsqrt, 220
  reductions -- the elementwise glue of PLE, hyper-connections, the router and the norm/activation
  pairs, each its own launch at [1, 2560], where a launch is ~64x the arithmetic.
- *The GDN kernel at decode.* It is gated to `seq_len > 1`; lifting the gate measured 6.64 -> 7.26
  tok/s but changed the emitted tokens (kernel vs the fp32 reference differ by 2e-4, enough to flip
  a greedy choice), so it was reverted rather than shipped silently.
- *The whole decode graph on one card.* 213.9 ms a step single-GPU against 146 on eight, so TP is
  already a win; the capture itself fails on the host reads the PLE and router make.

## Split reduce groups: 96 collectives at 10.5 ms, 2.2 ms in four strided groups

**Date:** 2026-10-19, 8 x RTX 3090, torch 2.6 / NCCL 2.21.5, `ts-208`.

A decode step's collective cost is **fixed latency per collective, not bytes**: 96 reduces of
`[1, 1, 2560]`, `[1, 1, 8192]` and `[1, 1, 22528]` each measure 121.5 / 116.2 / 125.0 us, and one
`[96, 1, 2560]` reduce of the same total bytes measures 449 us against 425 us for the 96 separate
ones. So the lever is the number of collectives *issued at once*, not their size.

Splitting the process group into **strided** sub-groups (`rank % k`) and issuing a subset each --
every rank creates the same group lists in the same order, and each reduces its own stride -- cuts
the wall time almost in proportion:

| groups | 96 reduces take | per collective |
|---|---|---|
| 1 | 10.5 ms | 109 us |
| 2 strided | 4.4 ms | 92 us |
| 4 strided | 2.2 ms | 92 us |

**What this does not do.** A TP all-reduce sums *across all eight ranks*; a sub-group sums within
itself. Splitting is not a way to reduce faster -- it is a measurement of what is left once the
per-collective latency is amortised, and the reason a true intranode implementation (one-shot or
two-shot over the PCIe topology, or a ring whose steps are smaller than a reduction latency) is the
term worth replacing. At four groups a step's 96 collectives would cost 2.2 ms.

**Not implemented in the runtime.** It needs an all-reduce that replaces the NCCL ring for this
topology, and a way to keep the layer dependencies intact; the probe above establishes the ceiling,
not a change.

## The all-reduces, measured

The same step, with events around each `all_reduce`: **73.1 ms of the ~150 ms step**, from **192
collectives** (two a layer) at **381 us each**. That is the single largest term after the launch
count, and it is a cross-rank latency that no graph can remove -- only fewer collectives can.

**Deferring them is not a local rewrite.** The hyper-connection streams are replicated, so a
reduced value is needed before the streams are *read* again, and the second reduce (the MLP block)
writes into those streams: the split `A = A_L + A_R` is consumed by the stream reads that follow.
So the first reduce must land before `mlp_hc` reads the streams -- one collective a layer is the
floor without moving the hyper-connection mix itself onto the device, which is a redesign rather
than a flag. A draft that deferred every reduce to the end of the step was written and reverted
unshipped: it would have produced wrong logits, silently.

**Two removals measured and rejected this round:**

- `torch.compile` on every layer's attention: **145.4 -> 145.2 ms a step.** The attention module is
  the largest single term (47 ms of compute) but compiling it fuses little, because the step's
  launches are spread over the whole graph and most of them are in the parts Triton will not take --
  the hyper-connection GEMMs, the router, the resident expert loop. There is no per-module win here.
- The resident path's per-expert weight *reads*: `rows()` builds two `FP8Tensor` views per expert per
  step, and those are 4 refcounted stores -- no launch. The step's CPU profile is 7,399 launches,
  4,637 `as_strided`, 1,550 `copy_`, 1,508 `reshape`: view bookkeeping, not something a faster read
  removes.

**What 20 ms a step requires, in order of how much of the step each removes:**

1. **One collective a layer, not two** (~36 ms), by making `mlp_hc` read the already-reduced
   streams and reducing only its block output.
2. **A step-wide CUDA graph** (up to ~100 ms of launches -> one replay), which needs the PLE n-gram
   hash and the router's selection on the device first.
3. **Fused glue kernels** (the 783 copies + 694 casts, ~15% of the device time) for the replay to
   be worth it.

The budget a step has to fit into is **20 ms**, and the terms measured above add to ~146: launches
~100, all-reduces 73, MoE 29. They overlap -- the collectives and the launches are not in series --
which is why the step is 146 and not 200, and why removing any one of them alone leaves the step
well above 20 ms.

None of these is a flag; each is a change to this model's decode path, and 2 needs 1 to be a single
stream of work a graph can hold.

## Where a decode actually stands on 8xRTX 3090

Measured on `ts-208` (8xRTX 3090, TP8, `/dev/shm` checkpoint), 128 tokens from a 53-token prompt,
`--expert-resident --expert-fused-fp8`:

```
decode  8.1 tok/s     122 ms a step
prefill 59 tok/s
```

That is the only configuration that answers correctly, and every number below is measured against
it.

### The step is bound by the host, not the card

A decode step issues **24,775 aten calls** -- about 516 a layer -- and dispatch alone is 3.72 us
each, so 92.7 ms of the 122 ms a step takes is the host queueing work. Device time is 42 ms. A
breakdown of the host side:

```
24,775 aten calls   92.7 ms   (mm 773, mul 1257, copy_ 1224, empty_strided 1107, to 1403 ...)
10,066 other        40.0 ms   (cudaLaunchKernel 6938 at 3.8 us)
```

There is no hot spot: the largest single entry, `cudaLaunchKernel`, is 26.5 ms, and 9,532 of the
calls are metadata-only (`as_strided`, `view`, `transpose`, `reshape`) that launch nothing at all.
Removing any one of them changes the step by a fraction of a percent.

### What each option was worth

| Attempt | Result |
|---|---|
| act kernel vectorised reads | device time halved; device was never the bound |
| `blocks_per_expert` 32 -> 48 | small gain, kept |
| NCCL protocol (LL / LL128 / Simple) | within 3%, noise |
| NCCL algorithm (Ring / Tree), channel count | no measurable change |
| dense FP16 linear path | **slower** than the default |
| `torch.compile` on the attention half | **53% slower**; on the whole model it fails -- triton 3.6 cannot bitcast e4m3 |
| RMSNorm gain precomputed | 0.4% |
| batching | see below: correct, and the only throughput lever there is |

### Batching: the one lever, and the caveat

The step is host-bound, so its cost barely moves with the number of rows: one row is 122 ms, eight
are 210 ms, twelve are 212 ms. Measured on identical rows, which decode the same tokens as a single
request does:

```
rows  1   122 ms/step    8.1 tok/s
rows  8   210 ms/step   38.1 tok/s
rows 12   212 ms/step   56.7 tok/s
```

That is throughput and not one request's latency -- each row still advances at about 4.8 tok/s when
there are eight of them -- but it is the only thing measured here that reaches 30 tok/s at all, and
it comes from the same fact that blocks the single-request target: the cards are idle while the
host queues work, so filling them with more rows is free.

The batched path used to crash outright: `_pinned_ids` allocated a one-row buffer and a two-row
batch failed writing into it. It follows the batch now.

**It is not yet correct, and the number above should be read with that.** Identical rows do not
all decode what a single sequence decodes: the answer goes off by one position at the sixth token,
and which rows are affected depends on the batch width -- with two rows both are wrong, with four
the first two agree and the last two do not. It is deterministic, and the top-1/top-2 margin at
that step is 5.0, so it is state rather than a near-tie in the logits. Making the id copy blocking
did not change it.

A batch does not take the fused path: `Fp8SlotExpertMoE`'s decode predicate is
`hidden_states.numel() == hidden`, true for one token and false for a batch, and the fast path
itself ends in `slot_rows.sum(0)`, which folds every row into one. So a batch is routed to
`_prefill` and grouped dispatch.

That route is not the cause, which was checked rather than assumed: the decoder hands the MoE
`flat = mixed.reshape(-1, hidden)`, so the dispatch's `token_idx = slot // top_k` indexes flattened
tokens and every row is kept apart. Where the divergence actually comes from is still open; it is
not the id copy (blocking it changed nothing) and not a near-tie in the logits (the margin is 5.0).

Until then, treat the batched throughput as an upper bound measured on a path that is not verified.

### The one thing that would get there, and why it does not

A captured decode step replays in **30 ms** -- 33 tok/s, which is the target -- so a CUDA graph is
the only mechanism that removes 90 ms of host queueing at once. Two shapes of it were built and
neither produced a correct token:

* **one graph per position.** A graph freezes the `past_len` it was recorded at, so a generation of
  N tokens needs N graphs, and every one of them records against the same mutable cache. Cloning
  the cache per position gets the first answer right and no others; the layers' recurrent state is
  advanced by the recording itself, so a replay applies it twice.
* **one step-major layout for the whole generation.** This is the right shape -- slot `p` holds
  position `p`, each step narrows to its own slot, the linear layers get one recurrent entry a
  position. It was built far enough to find real defects (the KV prefix was being copied along
  `head_dim` instead of the sequence axis; the recurrence was seeded from zero; a per-position view
  wrote through to the layout's own layers) and not far enough to be correct.

What the eager path needs if someone picks this up: the KV write and the selection index have to
become device-side, so one recording serves every position, and the n-gram window has to be a
per-position buffer filled between replays. `feature/qwen4exp-pipeline-parallel` carries a partial
`spec_decode.py` that starts from the same premise.

### What a single capture would need, concretely

The measured replay is 30 ms a step, so this is not a question of whether it would be fast enough.
It is a question of what has to change for one capture to serve every position, and that came down
to four things -- each of them identified by a failed attempt, none of them speculative:

1. **The KV write is a Python slice.** `cache.k[:, :, past_len : past_len + seq_len] = key` bakes the
   offset into the recording, so a graph recorded at position 53 writes position 53 forever. The
   write has to become `index_copy_`/`scatter_` against a device-side position tensor.
2. **The KV read is a Python slice too.** `key = cache.k[:, :, : cache.length]` fixes the key length
   in the graph, which is also what fixes the attention kernel's shapes. Reading a fixed-width arena
   and masking past `length` makes the shapes position-independent -- `cuda_qsa_attention` takes no
   `past_len` of its own, it infers everything from tensor shapes, so a constant-shape call is
   enough and no kernel change is needed.
3. **The recurrent state of the linear layers has to be indexed by position**, not advanced in
   place. Advancing in place is what made a replay apply the state update twice; a per-position
   array that each step reads and the next writes is the shape that works.
4. **The n-gram window is per position**, `ngram_size` ids ending in the token that step embeds. It
   cannot be computed during the recording, because the tokens do not exist yet, so it is a buffer
   filled between replays -- pinned, because a pageable crossing is refused inside a capture.

The head is vocabulary-sharded, so the argmax cannot be in the recording either: each position
records its logits and the choice is made on the gathered row outside, one sync a step.

### Where the capture actually stops

The four prerequisites above are in place and each is verified on its own: the KV write and read
take a device position, the linear state lives one entry a position, and the rope table can be
built before a recording. A whole step can now be captured and replayed in **30.4 ms** -- 32.9
tok/s, measured with the window fill, the replay, the token choice and the id copy all counted.

It decodes the wrong thing. The output is what says so, and nothing narrower does:

```
generate [1596, 1144, 310, 5707, 310, 1156, 328, 12675, ...]
capture  [1144, 310, 5707, 13, 271, 248069, 271, 248069, ...]
```

A layer-by-layer comparison was attempted and **did not measure what it claimed**: it read the
intermediate tensors of the capture itself rather than of a replay, so the numbers it produced were
of a pass that never ran. They are not recorded here as a finding. What the same harness did measure
reliably is the QSA kernel alone, which is bit-identical under a capture, so the attention kernel is
not the explanation.

The rest -- the slot index, the arena width, the n-gram window, TF32 -- was each changed and the
end-to-end output compared, and none of them changed it. That is a real exclusion, but a coarse one:
any of them could still matter in a way this comparison could not see.

What is left is that a replay and an eager step of the same input differ inside the operators, and
the error compounds until a token goes wrong. A working probe was eventually built -- layer 0's
returned tensor, eager against replayed, with no buffer in between -- and it measured:

```
diff max 0.352, mean 0.0038, 1276 of 10240 elements over 1e-2
```

The bad elements are spread evenly over the 10240 channels: split by residue over 4, 8 and 16 they
are 307/315/336/318, 160/159/152/152/147/156/184/166, 82/84/81/79/67/81/88/77. Two structural
readings that looked convincing in the moment -- "the last two hyper-connection streams are broken"
and "the eighth is a rank boundary" -- were both wrong, and the even spread is what killed them.

A large max with a small mean over the whole tensor is the shape of a different summation order
rather than of a wrong answer: the same arithmetic, accumulated in a different sequence. That is
not something a graph can be argued out of at this layer of the stack; matching it would mean
pinning the kernels' reduction order.

So the honest state of this one: eager decode is 8.1 tok/s and correct, a captured step replays in
30.4 ms and does not agree with it, and the disagreement is last-bits-wide and accumulating rather
than structural. Both numbers are measured. The 30 tok/s single-request target is not met.

Over a generation the difference flips an argmax where two tokens are close: the first five agree
with eager and the sixth does not. So the honest position is that the speed is reachable and the
agreement is not, and both of those are measurements rather than expectations.

### A bug that invalidated a great deal of measurement

The fused expert backend answered `nan` for every decode step for a long time, because `ShardedMoE`
masks another rank's expert with `num_experts` as a sentinel and `encode_route` tested ownership
with `index % world_size == rank`: for the sentinel that is true on rank 0, and `512 // 8` is row 64
of a 64-row table. The weight is zero, but zero times garbage is nan, and `argmax` over a nan logit
is index 0 -- so the model emitted `!` forever, and any throughput measured on that path was the
speed of a decoder that was not decoding.



Measured on `ts-208` (8xRTX 3090, TP8, `/dev/shm` checkpoint), 128 tokens from a 53-token prompt,
`--expert-resident --expert-fused-fp8`:

```
decode  8.00 tok/s    122 ms a step
prefill 59 tok/s
```

It is the only configuration that produces a correct answer, and everything below is measured
against it.

**The step is bound by the host, not the card.** A decode step issues about 7,000 kernels;
`cudaLaunchKernel` alone accounts for 6,938 of them, and the CPU profile is 148 ms against 86 ms of
device time. The largest single device term is the collective: 96 all-reduces (two a layer) of
5 KB each, 15.3 ms of a step.

**Two things this box makes worse than they need to be.** The eight cards sit on two PCIe switches
across NUMA with no NVLink between them, so `NCCL_PROTO=LL` cuts a 5 KB all-reduce from 91.5 to
74.5 us; `Simple` and `LL128` are both worse for this payload. And the resident experts hold 14 GiB
a card, which is what limits how much a captured step could ever be sized to.

**What was tried for the host cost, and what it cost to try.** A decode step can be captured as a
CUDA graph and replayed -- `--graph-window N`. One position replays correctly and is 3x faster.
Two do not, and the reason is structural rather than a missing fix:

* a graph records the position it was captured at, so a decode of N tokens needs N graphs;
* every graph is recorded against the same mutable KV cache, and captures write it;
* so the second capture's writes are what the first graph's replay reads back.

Restoring the cache between positions was tried in several forms -- snapshot and restore of the KV,
of each layer's `length`, of the linear layers' recurrent state and `initialized` flag, of the PLE
history, plus pre-growing the rotary table so the graphs' recorded addresses survive, plus blanking
the tail past each position. With one position the answer is right; with two it is not, and with
all graphs sharing a pool it is a segfault.

Cloning the cache per position is the closest this got: three positions then replay correctly
(`[1144, 1144, 1144]`, matching eager), which is what named the shared cache as the cause. It is
still not a decode. A clone seeded from the window's opening state is missing every position the
window has already decoded, and the answer repeats itself (`'We!!"! I am Qwen is Qwen is Qwen'`);
seeding each clone from the previous one instead makes it worse (`'WeWeWeWeWeWe...'`), because the
capture has to freeze the token at each position and the token is what the step before it has yet
to produce.

That is the shape of the problem: an autoregressive decode's step reads a history its own
predecessor writes, and a graph freezes the addresses it read. The versions that work in the field
put the position on the device -- `past_len` as a tensor rather than a Python int -- so one graph
serves every position. Here the rope and the causal mask already accept a tensor
(`attention.py:557`), but the KV cache is written and read by slicing to `past_len`
(`attention.py:502`, `:660`), and a slice needs an int. Moving that is a change to the cache's
addressing, not a flag.

`--graph-window` therefore stays at 0 and is marked experimental. The pieces it needed -- the
warmup that keeps cuBLAS's algorithm selection out of the capture, the cache save/restore, and a
logits selector because the head is vocab-sharded and argmax has to follow the gather -- are in
`graph_window.py` and are the part of it that is right.

**And one bug that invalidated a great deal of measurement.** The fused expert backend answered
`nan` for every decode step for a long time, because `ShardedMoE` masks another rank's expert with
`num_experts` as a sentinel and `encode_route` tested ownership with `index % world_size == rank`:
for the sentinel that is true on rank 0, and `512 // 8` is row 64 of a 64-row table. The weight is
zero, but zero times garbage is nan. `argmax` over a nan logit is index 0, so the model emitted `!`
forever -- and any throughput measured on that path was the speed of a decoder that was not
decoding. The fix is one predicate; the lesson is that a perf number needs a correctness check
beside it before it means anything.

## The captured decode, and what a capture costs

The three prerequisites above were built and measured on 8×RTX 3090 (`ts-208`, TP8, `/dev/shm`
checkpoint, 128 tokens from a 53-token prompt). Two numbers matter, and they are different numbers.

**The step itself: 36.5 tok/s (27.4 ms).** A decode step captured as a CUDA graph and replayed,
with the fused expert kernel, reads 15× faster than the 2.84 tok/s the fused path started at. The
terms, each measured on the device timeline:

| Change | Effect |
|---|---|
| `fp8_expert_act_kernel` splitting an expert's rows over the grid, not one block a slot | 2.84 → 7.03 |
| reading the hidden width off the plane shape instead of unpacking the layer for it | → 7.90 |
| the act phase addressing the packed plane directly (no unpack at all) | device time a step 168.9 → 74.9 ms |
| a lane reading a whole 16-byte plane chunk, so a warp's loads are contiguous | 25 → 28.3 GB/s of code reads |
| `blocks_per_expert` retuned for the captured step (32 → 48) | plateau, 24 is measurably worse |
| `NCCL_PROTO=LL` for the 5 KB all-reduces | 91.5 → 74.5 µs each, 33.2 → 27.1 ms a step |

That last one is this box, not the model: the eight cards sit on two PCIe switches across NUMA with
no NVLink between them, so a ring of 5 KB blocks is a latency problem, and the low-latency protocol
is the one that fits. On a machine with a real fabric it will not matter.

**The generation: about 2.6 tok/s for a 128-token request.** A graph records the position it was
captured at -- `past_len` picks a rope slice and a KV cache slice on the host -- so one capture
serves one position, and a decode of N tokens needs N captures. A capture costs **0.30 s a
position**: 124 ms of warmup (one full forward, because cuBLAS selects its algorithm on first sight
of a shape and doing that inside a capture voids it) and 176 ms of recording. Sixty-four positions
is what the resident experts leave room for, so a 128-token decode captures in rounds.

Three hundred milliseconds a position against the 13 ms a position it saves means the capture pays
for itself only past about **2700 generated tokens**. Shorter requests are slower on the captured
path than eager, and that is the whole of the gap between the two numbers above.

The capture cost was attacked and did not come down:

- **Sharing one pool and one stream across the window** removed an out-of-memory at 80 positions
  and returns the memory at `release()`, but leaves the 0.30 s.
- **Reusing windows across requests** works when the caller brings its own KV arena (the second
  request's `build()` returns in 0.000 s against 5.89 s) and fails in practice: a window is filed
  under a fixed size to be reusable, and spending a whole window on a tail that has fewer tokens
  left overruns the card.
- **`torch.compile` before the capture** does not survive this model's decode path.

What would collapse it is one graph serving every position, which needs `past_len` off the host:
the rope and the causal mask already take it as a tensor (`attention.py:557`), but the KV cache is
written and read by slicing to it (`attention.py:502`, `:660`), and a slice needs an int. That is a
change to the cache's addressing, not a flag.

So: the step is fast enough, the setup is not, and the setup is the open problem.

## Steady-state decode on 4x2080 Ti: 3.81 to 4.74 tok/s, and the τ that caps it

**Date:** 2026-11-03

**Hardware:** 4 x RTX 2080 Ti (21.48 GiB each), TP4, PCIe Gen3
**Model:** Qwen3.8-Flash-Next, FP8 checkpoint

The pages above stop at prefill for this box. This one is the single-request decode, and the point
of writing it down is that the ceiling is a number rather than a feeling.

### The step, measured three ways

A decode step of this model on this box is **210 ms** at the end of the work below. Three independent
measurements agree on what it is made of, and each one contradicted a plausible guess:

| probe | says | contradicts |
|---|---|---|
| CPU clock around the forward vs. a CUDA event pair over it | issue **273 ms**, GPU span **275 ms** | that the GPU is under-used |
| the same step with `all_reduce` stubbed to identity | removes **22 ms** of 210 | that the 96 collectives dominate |
| `torch.profiler` kernel totals | 2170 kernels, of which ~1300 are the FP8 dequantise's elementwise ops | that the experts' GEMV is the cost |

The profiler's `ncclDevKernel_AllReduce` row reads 111–175 ms, which is **its occupancy, not its
cost** — the kernel sits resident waiting for the slowest rank. The stub probe is the honest
number: 22 ms. Prefer a structural probe to a roofline read on a launcher-bound step.

What that leaves is a step whose wall clock is set by how long the host takes to issue it. At batch
1 the expert loop is ~50 small kernels a layer, and 48 layers do that 48 times.

### What was worth having

Two changes, both on `perf/qwen4exp-resident-layers` (#224 and its follow-up):

* **`--expert-resident-layers`** pins as many whole layers' experts as the free memory allows after
  the dense weights are loaded — **23 of 48** here — and gives a pinned layer's whole top-k to one
  bmm pair. The write-down is the checkpoint's own FP8 codes, not a bf16 expansion: 4.69 MiB an
  expert against 9.38.
* **`POCKETLLM_QWEN4_BATCH_STAGED=1`** gives the *staged* layers the same batch. It reads through
  the cache rather than around it; bypassing the cache to make room measured 3.69 against 4.60,
  because the hit rate is the whole reason a staged layer is worth having.

| configuration | 48 new tokens, alternate-run means |
|---|---:|
| baseline | 3.81 |
| `--expert-resident-layers --expert-cache 384` | 4.60 |
| `+ POCKETLLM_QWEN4_BATCH_STAGED=1`, reserve 5.5 GiB, cache 256 | **4.74** |

Every arm checked its output text, not only its rate. Cache capacity and the pinning draw on the
same memory: 448 or 512 entries OOM with the staged batches in place, and a 256-entry cache with
5.5 GiB reserved was the best of the combinations swept.

### Rejected, with the number that rejected it

| attempt | result |
|---|---:|
| routed experts computed on the CPU (`CpuExpertMoE`) | 0.13–0.97 |
| pinned layers widened to bf16 up front (halves the layers) | 4.30 |
| the whole stack widened in one `dequantize_block_fp8` (half the calls, `[k,...]` grids) | 4.44 |
| staged layers batched *around* the cache | 3.69 |
| `POCKETLLM_QWEN4_SCHEDULE=by_phase` | 6.9 — **and wrong**; see below |
| `NCCL_ALGO=Tree` / `PROTO=Simple` / `P2P_DISABLE=1` | 3.08–3.89 |

Two of these deserve their reason in words.

**`by_phase` is an invalid schedule, not an optimisation awaiting a PLE repair.**
Each layer consumes the preceding layer's reduced and injected output. Computing all layers'
HC/attention/MLP phases on a shared stale `h` breaks that dependency even without PLE; the old
path also reversed the router's `(weights, indices)` when calling MoE. The reported 6.9 tok/s
is the speed of a different, incorrect computation. The P0 runtime now refuses `by_phase`
explicitly rather than advertising it as a selectable performance path.

**`NCCL_ALGO=Tree` does not run here** — `ncclInvalidUsage` — on a box whose topology is
`GPU0-1 PHB / GPU2-3 NV2 / {0,1}<->{2,3} SYS`.

### The ceiling, as a number

Strata's formulation for a step of this shape is `τ = max_r W_r/B_r + Σ_serial`, the largest
weight-read term plus everything serialised behind it. On this box, at the end of the work:

```text
W_r = 0.55 GiB a rank a step   (the 25 layers that are not pinned)
B_r ≈ 6 GB/s                   (PCIe Gen3 x16, effective)
max_r W_r/B_r = 98 ms          against a 100 ms budget for 10 tok/s
```

**The staging term alone fills the budget.** No schedule reaches 10 tok/s on this box while those
experts have to cross PCIe each step, because there is nothing left for the arithmetic, the
dequantise, or the collectives. That is what caps this at 4.74 and not at what the kernel counts
suggest.

Strata reports the same shape of evidence for the same reason on its streaming-MoE models: a
register-fed FP4 kernel measured **0.98×** on DeepSeek V4 (whose decode is host-side expert compute
at a ~2% GPU matmul share), and it publishes a **1.066× ceiling** for Laguna S 2.1 before building
anything. A faster kernel on a relay-bound step buys nothing.

Two ways past it, both of which leave the FP8 premise this page is scoped to:

* **four-bit experts** — halves `W_r` to ~49 ms;
* **more VRAM so the experts stay resident** — 8 cards puts 14 GiB a rank, which fits, and is what
  would also make `--expert-fused-fp8` reachable (it needs the experts resident, and the FP8
  kernel needs sm_89+'s packed plane).

Neither is built. This box does 4.74 tok/s on 4x2080 Ti, and that number now has a reason.

### The three cards, side by side

| box | configuration | decode | status |
|---|---|---:|---|
| 4 x 2080 Ti, TP4 | `--expert-resident-layers --expert-cache 384` | **4.74** | measured here, 3 runs, output checked |
| 8 x 4090, TP8 | `--expert-resident --expert-fused-fp8 --graph-window 48` | **50.20** | re-measured 2026-11-03, `exit=0`, output checked |
| 8 x 3090, TP8 | expert-resident + fused + capture | 33.8 | from the session that produced this page; **not re-measured here** — the box reachable as `ts-208` today has neither the checkpoint nor a checkout |

The 4090 figure was re-run against this branch because the page above records **0.9185 tok/s** for
that box under the *old* configuration (resident, no fused kernel, no capture). The three flags are
what close the gap: without them the step runs a per-expert `_expert_matrix` that materialises
~4.39 GiB of bf16 a step, and the same box measures sub-1 tok/s. **The 6.0x "gap" in the section
above is a configuration artefact, not silicon** — 8 x 4090 at 50.20 against 8 x 3090 at 33.8 is
1.5x in the direction a faster card should be.

The 2080 Ti number is not comparable to either eight-card box: 21.48 GiB a card cannot hold
28.13 GiB of experts, so 25 of 48 layers stage every step, and `--expert-fused-fp8` needs those
experts resident. That is the whole of the difference, and it is why the τ above is 98 ms.

## NVFP4 experts whole-device on 4x2080 Ti: 5.82 tok/s, and the one scalar that hid the bug

`--expert-nvfp4` reads the routed experts from NVIDIA's ModelOpt NVFP4
checkpoint (`nvidia/Qwen3.8-Flash-Next-NVFP4`) instead of the FP8 one. A
rank's 48 layers are 14.9 GiB against 21.5 GiB of card, so every layer is
resident and a decode step stages nothing — the staging term that caps the FP8
path at 4.74 tok/s is not in the step at all.

| Path | Decode | Expert residency |
|---|---|---|
| FP8, pinned + batched staging | 4.74 tok/s | 256 of 1536 experts per layer |
| **NVFP4** | **5.82 tok/s** | **48/48 layers, 14.9 GiB** |

Three measurements on the same box, same prompt (`What is the capital of
France?`), output checked each time:

```
5.97 / 5.63 / 5.83   mean 5.81, then 5.96 and 5.82 after the last cleanup
```

### What the format is, and the two things that are easy to get wrong

Each projection stores E2M1 nibbles with an E4M3 scale every 16 elements, plus
a scalar `weight_scale_2`. Everything below was measured against the FP8
checkpoint of the same model, not assumed:

- **`weight_scale_2` is per expert.** At layer 0 the gate values run
  `2.65e-5 … 1.02e-4`, a factor of 3.9. Using the first expert's value for the
  bank leaves most experts scaled by a number several times too large or too
  small. It presents as a model that is *fluent and wrong* — pronouns and
  connectives in the right places, no content — so it reads like a numerical
  wobble rather than a missing index. Once the kernel indexes it by the route's
  expert, the same prompt answers correctly.

- **It belongs before the SwiGLU for gate/up, after the projection for down.**
  On one expert, against the FP8 expert's own output: split placement is 0.14
  relative, folding both onto the output is 0.97. `silu` sits between the two
  projections and is not linear, so the two are different functions.

- **The activation side is W4A16.** vLLM runs this format on sm75 through
  Marlin in exactly that regime, and it is the one with no activation-side
  error: `input_scale` never enters the forward pass. The checkpoint's own W4A4
  grid measures 0.86 on the same expert, because the recorded calibration range
  is `6 × input_scale ≈ 8.3e-3` while the residual stream at layer 0 runs
  1.6e-2 and reaches ~1 by layer 47. `POCKETLLM_NVFP4_A4=1` selects it for
  comparison.

### Where the time went

The first version that produced correct text ran at 1.86 tok/s, *slower* than
the FP8 path it replaces. Per layer, one MoE call:

| Change | ms/layer | tok/s |
|---|---|---|
| First correct version | 6.04 | 1.86 |
| Unrolled nibbles, two accumulator chains per projection | 3.24 | 2.45 |
| Split into gate/up and down launches, one block per column range | 1.00 | 3.16 |
| One warp per column, 32 lanes over the contiguous bytes | ~0.30 | 5.82 |

Two structural mistakes were behind the 6 ms. The kernel ran as `<<<1, 256>>>`
on a card with 68 SMs, and it split the two projections into separate launches
so each could use the grid — `down` reads every SwiGLU column, so a single
launch cannot split by column. Then the memory access: each thread owned a
column and strode by `hidden/2 = 1280` bytes, so every 8-byte read was its own
transaction. A column is contiguous, so one warp per column walking it 32 lanes
at a time turns the same traffic into coalesced loads.

For reference, the weights a layer actually streams are ~8.2 MB a rank, which
at the card's 616 GB/s is 13 µs. The kernel is now within about 20x of that
floor rather than the 460x it started at.

## Capturing the decode on 4x2080 Ti: 16.4 tok/s, and the two things that hid it

The step is launch-bound: 816 calls and a Python frame between each for
2 GMAC of work, which is 0.1% of the fp16 tensor-core peak. `--graph-window`
records it once and replays it, and on this box that turned 5.82 tok/s into
28 — with a reply that read `The capital of France is Paris, Paris is Paris,
Paris is Paris`.

Two of the three reasons were defects.

### The PLE window repeated the last token

`fill_history` was handed `prompt_ids + produced + [next_id]`. `produced`
already ends in the token the current step embeds — it is appended on the way
out of the previous iteration — so every id landed twice:

```text
[30, 271, 271]   where the sequence says [30, 271]
[271, 760, 760]  where the sequence says [271, 760]
```

The PLE term hashes that window and looks up a host-resident table. Every
n-gram it computed was one token too long, so the lookups that carry most of
what the model knows missed, and what came back was a fluent loop. This is
wrong on the eager path as well; the recording only made it easy to see,
because an answer that should have diverged was now repetitive enough to
notice.

### A rebuilt window restarted the linear layers

`--graph-window 2` spends two positions and rebuilds. Each rebuild seeded the
per-position state from `layer.conv_state`, which stopped being written the
moment the per-position slots were installed — so every rebuild restarted 36
of the model's 48 blocks from the state the sequence had at the *first*
capture. It now takes entry `steps` of the previous window, which is the entry
the last replay wrote.

Neither defect is visible in a state trace once fixed: KV, `index_k`,
`conv_state`, `recurrent_state` and the slot index all advance correctly,
window over window, and the handover is bit-exact.

### What is left is not a defect

With both fixed the reply was still wrong, and the reason is arithmetic rather
than bookkeeping. A recording and a loop that run the same kernels in a
different order agree everywhere except in the last bits. This model's decode
sits on top-2 logit gaps of **0.06 to 0.5**, and the difference between the two
orders is the same size. Greedy decoding resolves a near-tie by whichever of the
two is ahead, so those bits decide the sequence.

The page this model's capture machinery was written against says as much — the
first five tokens of a reply match the eager loop and the sixth does not. What
it also says is that the answer survives the drift. On this box it does not:
one flipped token puts the context somewhere the model answers with a loop.

So `replay` now declines to decide those steps. Each rank compares its own
shard's top-2 gap against `POCKETLLM_GRAPH_RETURN_MARGIN` (relative), the
proposals are reduced with a max, and a step that any rank considers too close
is handed back to the eager loop for that position alone. The reduction is not
optional: the eager fallback runs a different set of collectives, and one rank
taking it while another does not deadlocks both.

| `RETURN_MARGIN` | handed back (of 32) | tok/s | reply |
|---|---:|---:|---|
| off | 0 | 28.2 | `… is Paris, Paris is Paris, …` |
| 0.002 | 3 | 14.5 | `… is Paris, Paris is Paris, …` |
| **0.005** | **4** | **14.0–17.3** | **`… is Paris. What is the capital of France?`** |
| 0.02 | 8 | 12.0 | `… is Paris. What is the capital of France?` |

Four steps of thirty-two is the whole cost of the guarantee. Three runs at
0.005 gave 17.34, 16.37 and 12.69 tok/s with the same reply, against 5.82 for
the eager loop — 2.4x to 3.0x.

The margin is a property of this model on this card, not a constant. A model
whose decode does not sit on near-ties pays nothing for it; one that sits on
them pays the eager step for each. It is read from the environment for that
reason, and `0` restores the older behaviour exactly.

## The captured decode, across the cards

**Date:** 2026-11-05. Three boxes, three answers to the same question. Every row states its
quantisation and its card count, because the rows are not comparable across them: the 2080 Ti
run holds 4-bit experts and decodes them in software (sm_75 has no FP4 tensor cores), while the
3090 and 4090 runs are FP8 with a native fused kernel. What *is* comparable is the ratio within
a row.

| Box | experts | eager | graph | graph + handback |
|---|---|---:|---:|---:|
| 4 x RTX 2080 Ti (sm_75) | NVFP4, software decode | 5.82 | — | **15.5** |
| 8 x RTX 3090 (sm_86) | FP8, fused | 8.16 | **33.2** | 16.6 |
| 8 x RTX 4090 (sm_89) | FP8, fused | 1.08 | **46.7** | 3.80 |

Medians of three runs each, 32 new tokens, prompt `What is the capital of France?`. The 2080 Ti
is 4 cards because that box has 4; the other two are 8. Text was correct in every cell — the
answer is `The capital of France is Paris` in all of them.

### Reading the table

**The graph is the win on all three, and the size of it is the launch overhead made visible.**
The 4090 gains 44x, the 3090 4.1x, the 2080 Ti 2.7x. That ordering is not a property of the
cards: it is the inverse of each host's per-launch cost. The 4090 issues the same 8,139 launches
at ~41 µs each, the 3090 at ~18 µs, and the 2080 Ti's step was already bound by something other
than launches by the time the experts were resident. A capture deletes the launches, so the host
that pays the most for them gains the most.

**The 2080 Ti has no plain-graph number** and that is a real gap, not an omission: on that box
the plain graph produced a fluent loop (`The capital of France is Paris, Paris is Paris, Paris is
Paris`) and only the handback version answered correctly. Its 4-bit expert path sits on
top-2 logit gaps closer to the noise floor than either FP8 box does, so the near-tie problem
that is theoretical on the 3090 and 4090 is routine there. It is worth a measurement with a more
permissive margin to see how much of the 28 tok/s plain graph is recoverable while keeping the
answer.

**Handback is a heuristic, not a parity guarantee.** The following short 3090/4090 samples
matched their eager text, but neither those samples nor a local-shard top-2 gap prove
agreement for other tokens, histories, or global-vocabulary near-ties. The later P0 audit
also found state/PLE-input faults that a handback alone cannot repair. The historical
counts below record the cost of the tested heuristic, not a universal correctness bound.
What it cost in those samples was one eager step per near-tie: 4 of 32 positions on the 2080
Ti, 6 on the 3090, 8 on the 4090. On the 2080 Ti, where a replay is 40 ms and an eager step is
172 ms, that is 2.7x to 1.5x and still a win. On the 4090, where a replay is 21 ms and an eager
step is 920 ms, it is 47 tok/s to 3.8 — *below* the 3090's plain graph, and below what the 4090
does eager on a host that is not launch-bound.

**So the recommendation is per-box, not universal**: take the plain graph where the answer is
correct without help, and take the handback only where the near-ties actually reach the answer.

### A stale extension is a loud failure; a wrong window is a silent one

Two ways to measure the wrong thing, both found while taking these numbers.

The 4090 first reported 0.65 tok/s with a fluent loop. The extension on that host was at
`fcb8b07`, twelve commits before `b86fa05`, which is the commit that adds
`qwen4_exp_fp8_slots_swiglu` — the fused expert kernel. `--expert-fused-fp8` failed with
`RuntimeError: Fp8SlotExpertMoE needs the relic_core CUDA extension` rather than falling back,
which is the right behaviour, but the earlier runs without the flag had gone through the slow
path silently. After rebuilding, the same host gives 1.08 eager and 46.7 captured. **A 70x
difference between two runs of "the same" configuration is a deployment question before it is a
performance question.**

`--graph-window` has a quieter failure mode. The window is only built when

```python
if graph_window > 0 and ctx.device.type == "cuda" and max_new_tokens > graph_window:
```

so `--graph-window 48` against a 32-token generation builds nothing and the run is eager,
wearing a graph label, with no message. Three such runs were initially reported as graph
measurements at 8.2 tok/s — indistinguishable from the baseline, which is exactly what they
were. The guard now prints `no window was built (the generation is 32 tokens and the window is
48); these steps ran eagerly`, and the `replayed N/M` line is the check that a graph run actually
replayed: if N is 0, no capture happened.
