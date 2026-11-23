# Pure per-token B=1 Ascend decode: layout, graph and state follow-up

This record follows the merged N64/K128 scale-LUT consumer. It concerns one request,
batch one, **one complete autoregressive forward per next token**. It does not use MTP,
speculative decoding, request batching, or prefill throughput as a decode-rate substitute.

## Measured state

The merged selected provider was reproduced at **10.719303 tok/s** on one eight-card
Ascend910B3 machine with 64 GiB per card. Measurements use four excluded warmups and
64 synchronized decode forwards, context 24 through 88, including completion and host
argmax. Profiling and quality copies are separate excluded passes. The fixed-length
benchmark can continue after EOS and is one prompt/context regime, not a general
quality evaluation or a long-context distribution.

An opt-in layout preparation candidate replaces the sparse-attention gather/cast and
per-block transpose preparation with a fused gather into two contiguous FP32 KV operand
layouts. The reference already gathers and casts once outside its block loop; the
repeated transpose/slice preparation is the target. The candidate preserves:

- Original stored KV values and index order, without pruning, sorting or requantization.
- The existing QK and P.V `torch.bmm` calls and true last-block key width.
- The fixed 64-key online-softmax boundaries, FP32 denominator before narrowing,
  BF16 numerator before P.V, FP32 accumulators and final sink normalization.
- Original weight code/scale payload size and represented values.

This is a **layout-only mathematical transformation, not a bit-exactness guarantee**:
contiguous rather than strided operands can change backend matmul scheduling.

| Configuration | Rate (tok/s) | Latency (ms/token) |
| --- | ---: | ---: |
| Matched original attention-layout control | 10.655378 | 93.849324 |
| Contiguous KV layout candidate | 11.084782 | 90.213771 |
| Independent same-source candidate repeat | 11.118415 | 89.940876 |

The matched gain is **4.029931%**, saving **3.635553 ms/token**. The independent repeat is
not substituted for the matched improvement calculation.

## Explicit numerical acceptance

Strict full-distribution comparison initially rejected this candidate. All eight ranks
have identical output token IDs, original-reference eight-step tensors, native-provider
teacher eight-step tensors and final logits. In an excluded complete 64-step FP32 copy
pass, **63 steps are bit-identical, but step 12 / absolute context 36 differs**:

- 129,251 changed FP32 logit elements per rank.
- Maximum absolute difference: **0.003835439682006836**.

The user explicitly accepted **this observed deviation** after it was reported. The
original rejected receipt is preserved, with a separate direct-user acceptance receipt.
This does not turn the result into bit-exact equivalence to the previous provider,
equivalence to the original exact-default implementation, or a general model-quality
acceptance. It does not authorize new lossy weight quantization, different activation
grids, reduced Sinkhorn iterations, or narrower collective precision.

The independent candidate repeat matches **every saved FP32 tensor bit** on all eight
ranks, including the complete excluded 64-step distribution. That repeats the observed
error relative to the old control; it does not eliminate it.

## Admission, resources and failure history

The kernel uses an experimentally compiled 16-by-64 program tile, with separate complete
64-key blocks and exact-size tail layouts. A 32-by-128 prototype overflowed Ascend UB:
3,146,752 bits were required while 1,572,864 bits were available. The failed build is
retained rather than described as a working variant.

Nonnegative index bounds require an explicit caller attestation on the NPU path; a
metadata match alone is not proof. Real original indices were checked outside capture
and timing. Negative indices retain the reference's row-zero gather and invalid mask.
The public scratch admission uses a declared Python-visible tensor estimate; allocator
rounding, graph pools, BLAS workspace and Ascend local/UB memory are explicitly outside
that estimate. It is not a physical peak-memory cap.

The runtime binding is explicit and reversible for one model on one installer thread.
Other models, CPU/default providers, prefill and unsupported shapes retain the original
provider. Captured device execution does not increment Python selection counters. A
failed close fence retains bindings for retry rather than freeing referenced state.
No serving default has been switched.

Server-only checks:

- 62 core tests passed, including the actual compiled NPU layout case.
- 26 runtime ownership and lifecycle tests passed.
- 16 strict synthetic top-k cases and nine changing-input graph replays have identical
  same-NPU output bits, including invalid rows and true tails.
- 320 real attention calls per rank passed the excluded eight-step component gate.
- Those short gates did **not** catch the later context-36 difference; the complete
  64-step check was decisive and remains required for subsequent candidates.

No project tests, native builds or model measurements ran on the macOS development
machine. The original checkpoint file identities and shared CANN/CI setup were not
modified. The full checkpoint is not converted into a stored BF16/INT8 bank.

## Execution-boundary follow-up

A task-private prototype records the existing attention/router/HC half, native TP
expert body and shared-MoE/reduction/HC half into **one layer graph**, replacing three
separate replays without changing their arithmetic. Both comparison columns keep the
accepted contiguous KV layout enabled. Two fresh 64-step pairs measured:

| Accepted-layout comparison | Control tok/s | Inline-layer tok/s | Observed gain | ms/token saved |
| --- | ---: | ---: | ---: | ---: |
| First pair | 11.105647 | 11.169005 | 0.570507% | 0.510795 |
| Independent repeat pair | 11.086534 | 11.154427 | 0.612388% | 0.549009 |

Every saved original-reference eight-step, native-teacher eight-step, final and
complete excluded 64-step FP32 tensor matches the accepted layout control bit-for-bit
on all eight ranks, with matching token IDs. The prior accepted layout error is **not
additional numerical headroom**.

The server CPU fake-API suite passed 47 checks covering ownership, stream/device,
pre-write refusal, constructor rollback, partial-capture retention and release-fence
retry. Actual excluded device profiles confirm all forty ready layers replaying eight
times, 6,120 host replay counts across the full protocol, and both collective channels
sharing all 43 executed fused-graph model IDs. Source establishes group membership
(two expert-TP ranks and eight model ranks); exporter channel tags do not establish
membership or wire bandwidth. The native expert-kernel Model ID column is sentinel-only,
so that column does not independently map each native launch to a fused model.

This is an **approximately 0.6% observed gain in two pairs**, not a statistical confidence
claim or a major latency-floor reduction. The controller remains an explicit,
task-private experiment in the measured short-context regime. No serving default or PR
merge follows automatically. **50 tok/s is not reached**: it requires 20 ms/token,
versus approximately 90 ms/token on these configurations.

## Sparse softmax-state follow-up: candidate-only numerical qualification

The next opt-in kernel path keeps the accepted S1 full-block and true-tail KV layouts
and the native QK/P.V `torch.bmm` boundaries. It fuses score scaling, invalid masking,
running maximum, exponential preparation and the post-P.V accumulator update. The
FP32 denominator reduction and its multiply/add update remain in **the original Torch
expression over the exact key width**. Sink exponentiation and final division also
remain in Torch. Original weights, activation grids, TP collective precision and HC
iterations are unchanged.

An earlier S2 variant put the denominator reduction in Triton. It passed short device
cases but failed a real BF16 attention input before timing: one of 4,096 BF16 output
elements differed by 0.0009765625. The affected FP32 denominator update differed by
4.768371582e-7. Rounded-P parity did not isolate the reduction from raw exponential
rounding, so that failure is retained without claiming an isolated cause. The revised
S2b restores the ATen denominator tree and exposes the raw numerator for inspection.
On that saved failing input, the raw FP32 numerators, denominator, maximum, rescale,
rounded-P operand, reference-operand post-P.V update and final BF16 output match.

The Triton path explicitly admits **BF16 query and KV only**, rejecting FP32 and mixed
query/KV operands before layout allocation. The CPU oracle retains FP32 support. An
earlier general FP32 path failed a synthetic case; restricting the public NPU domain
is not accepting that error or granting extra numerical headroom.

Two independent candidate windows under the canonical excluded-oracle protocol measured:

| S1 + inline layer graphs + S2b | Rate (tok/s) | Latency (ms/token) |
| --- | ---: | ---: |
| First canonical-output candidate | 11.561396 | 86.494748 |
| Independent same-source candidate repeat | 11.548657 | 86.590159 |

Both candidates match **all saved FP32 tensors of the immutable accepted S1 result** on
all eight ranks: reference eight-step, native-teacher eight-step, final and complete
excluded 64-step copy pass. Output token IDs also match. The repeat matches the first
candidate bit-for-bit, with identical measured source bytes, library and stack, and
only the output-directory environment label changed. This qualifies an explicit BF16
experimental prototype for the measured B1/context-24-to-88 regime; it is not a general
model-quality, long-context, FP32-NPU or compiler-FMA theorem.

### Do not promote the apparent paired gain

The contemporaneous controls themselves drift from the immutable accepted S1 output
at the known context-36 step, returning to the original pre-S1 value. Changing the
excluded oracle protocol does not produce an output-stable control. The two earliest
S2b pairs match each other locally but not the canonical result; under the restored
protocol the candidates match the canonical result while their controls do not.

Consequently, the observed 3.08%, 3.38% and 3.89% paired rate differences **are not
accepted strict-output-paired speedups**. Report the independently repeated absolute
candidate rates above, with the control-provenance limitation, rather than silently
relabel a drifting control or reuse the prior user approval as a tolerance. The cause
of this context-specific backend variation is not established.

Actual excluded traces on every rank show 552 prepare and 552 post-P.V state-kernel
calls across four profiled forwards, 320 real BF16 attention gates and both collective
channels sharing 43 executed graph models. Kernel task durations are not additive with
layer/host wall-time scopes. A local multi-directory download initially mislabeled a
first-run selection receipt as repeat evidence; the exact repeat directory was fetched,
its different raw CSV identity verified and a superseding provenance correction added.
The earlier mislabeled record is retained, not rewritten as independent evidence.

The sanitized [follow-up evidence](evidence/ascend-pure-b1-20261122/follow-up/README.txt)
keeps numerical failures, matched-only comparisons, canonical audits, the independent
repeat and provenance corrections separate. No tensor/checkpoint payloads or process
ownership tokens are included.

## Where the remaining 72 ms actually is: three categories, measured

Profiling the lane by task counts rather than by kernel name changes the picture. Summing every
kernel's duration over the four profiled forwards gives **73.6 ms**, which is the step time itself, so
nothing overlaps; the step is the serial sum of its tasks. Grouping those tasks by how many AI cores
they occupy:

| Cores used | us/forward | share | tasks |
| ---: | ---: | ---: | ---: |
| 0 (communication) | 21218 | 28.8% | 161 |
| 24 (all of them) | 12284 | 16.7% | 138 |
| **1** | **11671** | **15.9%** | **4938** |
| 20 | 7911 | 10.7% | 454 |
| 40 | 5615 | 7.6% | 496 |

Three things follow, and each is a different kind of work.

### Communication is a quarter of the step and it is entirely idle

The communication bucket is all `hcom_allReduce`: about 130 collectives a forward, 18592 us in total.
Their durations are sharply bimodal:

| | us |
| --- | ---: |
| min / p25 / median / p75 | 27.8 / 29.9 / **35.8** / 38.8 |
| p90 | **505.8** |
| p99 | 985.2 |
| max | 5312.3 |

Nine tenths of the collectives cost about 36 us; the top tenth cost ten to a hundred and fifty times
that. If every collective cost the median the total would be 4668 us instead of 18592 us, so **14193
us a forward — 19.3% of the step — is excess above the median.**

This is not a warmup ramp. Splitting the capture into four forward windows leaves the tail in every one
of them (p90 513, 507, 506, 340 us; the single 5312 us outlier is in the first window only). It is also
not bandwidth: a 5120-element row across eight ranks is a hundred and sixty kilobytes, a few
microseconds of wire.

And nothing overlaps it. For every collective in the capture — slow and fast alike — the amount of
non-communication work running inside its window is **0.0 us**. The AI cores are entirely idle while
the machine waits at the barrier, for all 58627 us of collective time in the capture.

A collective that costs a hundred times the median while every core is idle is waiting for the slowest
rank. The expected cause is MoE routing imbalance: with expert-parallel assignment by id, the number of
selected experts a rank owns varies per token and per layer, so some layers leave one rank far behind.
That is a hypothesis this round did not test; what is established is the size of the prize and that the
wait is pure idleness, not throughput.

### The window kernel was slow for a reason worth recording

The single-core list put the window-index kernel at 17.11 us a call for a table of one hundred and
twenty-eight int32 values. The work is trivial, so the cost was in how it was written: the expression
`(lane + oldest) % window_size` was evaluated over **int64 lanes** with a **modulo per lane**. On this
device both are expensive — 64-bit lanes are emulated and integer division is a sequence, not an
instruction.

Both are avoidable without changing a single value. `lane` is below `WINDOW` and `oldest` is at most
`WINDOW - 1`, so the sum is below `2 * WINDOW` and the modulo has exactly one alternative: subtract
`WINDOW` when the sum reached it. Every value fits int32. The rewritten kernel is bit exact — 108
combinations of window, slot and row, zero mismatching elements — and measures:

| | us per call | us per forward |
| --- | ---: | ---: |
| int64 with modulo | 17.11 | 684.5 |
| int32 with a conditional subtract | **1.41** | **56.6** |

The kernel's own saving is 628 us a forward, but the **total device time fell 1898 us (73.6 to 71.7
ms)**, because the 64-bit lane work and the division were also stalling the dispatch around it. That
is a reminder that a kernel's own duration understates what it costs when it is the wrong kind of work
for the unit executing it.

Registration took three strict pairs against the narrow-head signature, because the change is bit exact
and must not move a single value:

| Pair | Control tok/s | Candidate tok/s | Gain | Token ids | Signature |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 12.417645 | 13.965585 | +12.4656% | identical on 8/8 ranks | reproduced |
| 2 | 12.482141 | 14.064584 | +12.6777% | identical on 8/8 ranks | reproduced |
| 3 | 12.450010 | 13.918896 | +11.7983% | identical on 8/8 ranks | reproduced |

Mean **+12.3139%**, median **+12.4656%**, best **14.064584 tok/s / 71.1006 ms per token** — against a
registered mean of +11.1036% before the kernel was rewritten, so the rewrite is worth about a point and
a fifth on the step.

### The inline one-layer NPUGraph is already the shipping configuration

An earlier revision of this page claimed the registered lane runs fully eager. **That was wrong and is
withdrawn.** The truth is stronger than "graphs are captured": the exact configuration the objective asks for
is what ships.

The per-rank records of the registered run (`pure-B1-fused-row-norm-candidate-profile-steps64-20261123T103426Z`,
the 14.287598 tok/s lane) say, **identically on all eight ranks**:

| field | value |
| --- | --- |
| `inline_TP_layer_graphs.layers` | **40** |
| `inline_TP_layer_graphs.captured` | **true** |
| `inline_TP_layer_graphs.variants` | 43 |
| `inline_TP_layer_graphs.driver_counts.replays` | 6120 |
| `independent_TP_graphs_retired_after_explicit_fence` | **40** |
| `graph_candidate_eager_FP32_bit_differences` | **[0, 0, 0, 0, 0, 0, 0, 0]** |
| `private_graph_capture_seconds_excluded` | 4.141067163087428 |

So the inline one-layer graph is in use, all forty layers are captured, the independent two-recording variant was
**retired** rather than left as a fallback, and the parity the objective demands — eight ranks, FP32
bit-identical against the eager column — is recorded as all zeros. Capture cost is excluded from the
measurement rather than folded into the step time.

**How the error happened.** I grepped the launcher for `NPUGraph` among other mixed-case spellings when the flag
is `RELIC_CUBE_NPUGRAPH`, all caps, set in the generated wrapper; the empty grep was read as absence. The
per-rank record states all of the above directly and would have answered it in one field. A second, smaller
error followed: the round-74 correction described the shipping behaviour as two recordings around an eager
middle, when the records show the inline variant is what runs and the independent variant was retired.

The consequence for where remaining time can come from is unchanged and now firmly founded: **this objective
item is delivered**, not pending. The 8.7% of profiled span that is not device-busy is not eager launch
overhead, and no capture work remains to be done.

### The expert projection measured: it is M=1 tile padding, not a slow kernel

The expert projection is 15.4% of the step and had only ever been bounded by reasoning. The bandwidth
driver that answers it had been in the evidence bundle since before this stretch of work, titled
**"written, not run"**. Run now, with the accepted v2-n64 library (`7dce3e3d...`) and CANN
(`bbbda1c0...`) verified first:

| variant | median | achieved |
| --- | --- | --- |
| 1 group, runtime bf16 | 366.3 us | 8.6 GB/s |
| 6 groups, runtime bf16 | 474.9 us | 39.8 GB/s |
| 6 groups, kernel fp32 | 474.0 us | 39.7 GB/s |

Five more experts cost only about 110 us, so the call is dominated by per-call cost rather than per-expert
bytes. Two cautions before reading it as headroom: the harness's same-volume read-only `sum` reference
reports 5.4 GB/s while a `copy_` over the same volume reports **448.3 GB/s** - only the copy is a
practical device reference; and the standalone numbers are host-bound, since the same kernel takes
474.9 us standalone against **101 us a call in the model profile**, a 4.7x gap. The implied 344.5 us
fixed cost is therefore not kernel-internal.

Sized in-graph instead: 18.8 MB in 101 us is **186.9 GB/s against a 448.3 GB/s practical reference, 2.40x
off**. A generous best case would recover about 7 ms of the 12.1 ms, and that best case is unreachable:
the kernel runs **M=1** - one activation row broadcast across expert groups - while the cube unit is built
for M=16 tiles, so most of every loaded tile is padding. That is a property of single-request decode, not
of the kernel, and it is the same reason a larger acceptance length is so much faster.

### A layout fix that is not bit-exact: the o-projection transpose

The engine table leaves one structural smell: `aclnnBatchMatMul_TransposeAiCore_Transpose`, 48 calls and
**1,171.6 us a forward**, dominated by a `1024,4096` to `4096,1024` shape about 40 times a forward — once
a layer. It comes from `attention.py:1688-1689`, which lays the o-projection weight out as
`[groups, o_lora_rank, d]` and then einsums over `d`, so the framework materialises a transpose of that
**constant** weight before every matmul.

Transposing a constant once per module looks free of consequence — a transpose changes no value — and that
reasoning was wrong. The claim was tested on the device before any lane change:

| variant | bit-identical to the current code | eager cost |
| --- | --- | --- |
| contiguous pre-transpose, `gdr` einsum | 3 of 6, then 5 of 8 — **not exact** | 212.1 us, *slower* |
| transposed view handed to matmul | **0 of 8** | 99.6 us, 40% cheaper |

Differences are 1 to 2 of 2048 elements, max abs up to **0.5** on bf16. The reduction order on this build
is therefore **not** layout-invariant: the faster variant changes results in every trial, and the variant
that sometimes agrees is slower than what runs today.

So the 1,171.6 us cannot be removed bit-exactly, and the change is not adopted. The faster variant is
recorded as a *new deviation class* — bf16 reduction-order deviation in the o-projection — with its
measured error, for the user to decide on. It is a different op from the row-normalisation and
narrow-head approvals, so those receipts do not cover it.

### The lane is device-bound, so host-side work is not a target

A tempting reading of the host-side table is that the step is full of fixable host overhead:
`aten::copy_` shows 95 calls and **4,277.7 us of host self time a forward** against only 63.1 us of
device time, and the most expensive copies are shape `1,1` by `1,1` — 70 to 177 us of host time with
**0.0 us of device time**, i.e. they never touch the accelerator.

That reading is wrong, and the measurement that settles it is the union of device task intervals: on the
registered lane, device 0 is busy **287,045 us of a 314,467 us span (91.3%)** over four forwards. The
busy total is 71.8 ms a forward against a timed step of 69.99 ms, so the profiler's instrumentation
inflates its own measurements and what looks like idle is mostly the measurement. The device is
wall-to-wall busy in the real step, so host-side time is overlapped and shortening it moves nothing.

A second hypothesis died the same way and is worth recording so it is not re-tried: the eight `pos.host`
reads in `attention.py` looked like a device-to-host scalar sync per layer. `decode_pos.py:194` shows
`Pos.host` is a property returning a plain Python int set in `__init__` — no `item()`, no
synchronisation.

The constraint this leaves: **every remaining gain must come from doing less device work or faster
device work.** Host-side restructuring has no headroom left.

### Before-and-after: the fusion removed exactly the operators it claimed

Profiling the lane before and after the indexer fusion checks the mechanism rather than trusting the
speedup. Cast, Sort and Greater each fall from nine a forward to one, and the survivor is site B at
`attention.py:1326`, the one deliberately left on torch:

| AI CPU operator | before, per forward | after, per forward |
| --- | --- | --- |
| Cast | 9 (1,173.1 us) | 1 (176.5 us) |
| Sort | 9 (352.8 us) | 1 (42.4 us) |
| Greater | 9 (334.1 us) | 1 (41.1 us) |
| total | 2,761.9 us, 31 tasks | 1,158.6 us, 7 tasks |

`COMMUNICATION` moved from 19,175.5 to 20,620.7 us between the two captures at an unchanged 130.2
tasks. That is the routing straggler wait varying run to run and it is larger than the AI CPU saving -
which is exactly why this lane registers on three pairs rather than on one profile.

### Moving the indexer's selection off the AI CPU: measured

`attention.py:1173` masks unreachable entries against a `tail` sentinel, sorts ascending and casts — a
Greater, a Sort and a Cast, three AI CPU operators per index layer, eight layers a forward. A single
kernel now computes the same selection as a rank: an entry's output slot is the number of reachable ids
below it, so the reader scatters rather than sorting. Nothing round-trips through the AI CPU.

Two details kept it honest. The reference sorts the **ids**, not the scores — the first kernel ranked by
score and ten standalone cases caught it before the lane did. And `attention.py:1326`, the 2048-wide
block selection, measured **0.66x** with the same kernel, so it was left on torch: a measured regression
is not bought to make the story tidier. The install rebinds `Indexer.forward` at run time and raises if
the anchor is missing or duplicated, rather than silently binding nothing.

Two pairs, both parity-clean on all eight ranks (`token_ids_equal_pair` and `token_ids_equal_canonical`
both true everywhere):

| pair | control | candidate | vs registered best |
| --- | --- | --- | --- |
| 20 | 12.4480 tok/s, 80.334 ms | **14.3768 tok/s, 69.557 ms** | +2.220% |
| 21 | 12.4729 tok/s, 80.174 ms | **14.2646 tok/s, 70.103 ms** | +1.422% |

| 22 | 12.4597 tok/s, 80.259 ms | **14.2214 tok/s, 70.317 ms** | +1.115% |

**Registered at three pairs: 14.287598 tok/s, 69.9922 ms a token, +1.586%** over the previous
14.064584 / 71.1006. All three pairs are positive and parity-clean on every rank, and the worst pair is
still above the previous best.

It registers as bit-exact and needs no new approval: the same int32 tensor in the same order comes out,
because the reference sorts the ids, the ids are unique, and every tie is between sentinels that all
collapse to the same padding value. A rendered comparison of the three pairs is in
`.coord/indexer-sort-fused/registration-three-pair.json`.

### The AI CPU bucket is the sparse indexer

The engine table above shows 31 operators a forward on the AI CPU for 2,762 us — no cube, no vector
unit, serialised against the stream. They are now traced to one line.

`RelicLLM/relicllm/models/deepseek_v4_1/attention.py:1326`, inside `Indexer._stream_prefix`, once per
tile per index layer:

```python
torch.where(block_values > -torch.inf, block_keys, -1).sort(dim=-1).values.int()
```

The `>`, the `sort` and the `.int()` are the Greater, the Sort and the Cast. The alternative
`select_candidate_blocks` at lines 815-844 applies the same rule over the whole axis, but its docstring
says it is not what runs.

The counts match the geometry. The checkpoint config lists **eight** index layers
(`index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]`), each selecting `index_topk = 512`
compressed positions, with candidate layer 20 narrowing to `candidate_topk_blocks = 2048` blocks of
`candidate_block_size = 8`. So this is an eight-way sparse indexer doing real selection work, and the
defect is only the engine it runs on.

The fix is bounded by a stated requirement rather than by taste: the sort exists to match the
reference's `topk(...).indices.sort()`, putting every `-1` after every real id (comment around line
1168). Reordering changes the downstream gather order and is not bit-exact, so the only legal fix is the
same selection, same order, same dtype, in a device kernel — the shape of change
`ascend_window_idxs_fused.py` already made for the window index build. The whole bucket is 3.5% of
device time, so the ceiling is around 2.8 ms a forward.

### The output head still runs in FP32 and reads 2.65 GB a step

The profile carries a host-side table, `operator_details.csv`, that names the operator behind the
kernels. It covers only about a quarter of device time (70,276 us of 294,384 us over four forwards), so
its internal percentages are not shares of the step — but its per-call durations agree with
`kernel_details.csv` where both see the same call, which is enough to pin the output head.

There is exactly **one** head-shaped matmul a forward:

| | |
| --- | --- |
| kernel | `aclnnMatmul_MatMulCommon_MatMulV2` |
| operands | `1,5120` by `129280,5120` to `1,129280` |
| calls a forward | 1 |
| duration | **2,346.8 us (3.3% of the step)** |
| declared dtype | FP32 by FP32 |
| a BF16 twin of the same shape | absent |

The arithmetic says what it is waiting on. FP32 weight traffic is 129,280 x 5,120 x 4 = **2.65 GB**, and
2.65 GB in 2.35 ms is 1.13 TB/s — the call is bandwidth-bound reading the weight, not compute-bound. The
checkpoint stores `head.weight` in BF16, which is 1.32 GB, so widening it to FP32 doubles the bytes read
every single token. `ascend_head_bf16.py` says as much in its own docstring: *`Head` widens
`head.weight` from the checkpoint's bf16 to fp32 and keeps it resident*.

The approved narrow-BF16-head deviation class exists precisely to avoid this traffic, and its receipt
validated all 8 x 129,280 = 1,034,240 logits. It is **not in effect in this capture**, which records `narrow_output_head: null` — but that capture is
stale. It was taken at 04:37 UTC, before the narrow head was wired into the profile module, and every
recent candidate trial of the registered lane records `{"installed": true, "weight_dtype": "bfloat16
resident"}`. So the 2.35 ms measured here is the *pre*-narrow-head cost: the gain was already banked in
the registered 14.064584 tok/s, and it is not available again.

The consequence is wider than one number. Because this capture predates the narrow head, every
attribution taken from it — the 39.4% vector core, the 3.7% AI CPU, the 6.26 ms of plumbing, the 73.6 ms
total device time — describes a lane about 2.3 ms a step slower than the registered one. Those figures
are kept because they explain why the head mattered, but they are **not** the current cost model and
must not be used to choose work. A fresh capture of the registered lane replaces them.

A caution about the same table: it also lists `c10d::barrier` at 5,313.6 us and `wait_event` at 5,146.0
us alongside two `HcclAllreduce` rows at 5,312.3 us each. Those are one 5.3 ms event seen from three
angles — the same single outlier found in the collective tail analysis — not three costs, and not a
per-forward cost.

### Grouping by accelerator core changes where the step is

The cost model above groups tasks by how many cores they occupy. Grouping the *same* capture by the
profiler's `Accelerator Core` column instead tells a different and more useful story, because it says
which engine is spending the step:

| core | tasks a forward | us a forward | share of the step |
| --- | --- | --- | --- |
| AI_VECTOR_CORE | 6,916 | 28,036 | 39.4% |
| COMMUNICATION | 130 | 18,592 | 26.2% |
| MIX_AIC | 200 | 14,499 | 20.4% |
| AI_CORE | 673 | 8,427 | 11.9% |
| AI_CPU | **31** | **2,626** | **3.7%** |
| MIX_AIV | 358 | 1,417 | 2.0% |

Two things fall out that the block-count view hid.

**Thirty-one operators a forward run on the AI CPU**, not on an AI core at all, for 2,626 us: 9 `Cast`
(1,025 us), 4 `Index` (937 us), 9 `Sort` (333 us), 9 `Greater` (331 us). An AI CPU operator occupies no
cube or vector unit and serialises against the stream, so this is 3.7% of the step bought with nothing
but a dispatch mistake — a top-k selection or a cast that a vector kernel should be doing.

**The vector core is not a long tail of unknown ops; it is plumbing.** 697 in-place copies that are
casts (1,892 us), 346 that are slices (1,103 us), 138 transposing copies (1,240 us), 84 scalar `Pow`
(1,060 us), 77 `Cat` (970 us), 578 `Mul` (873 us), 562 `Add` (877 us), 517 scalar `Add` (775 us). Those
five plumbing operators alone are **6.26 ms, 8.8% of the step**, and none of them is arithmetic the
model asked for: it is a batch-one decode moving one row of 5,120 values around a few thousand times.

So the next lever is not another fused kernel on top of the big matmuls. It is **the number of
framework-dispatched elementwise operators per layer** — the same class of problem the community stack
attacks with Model Runner V2 and PGO. The 84 `Pow` calls are the sharpest single example: `rsqrt`
lowers to `Pows(-0.5)`, the row-normalisation fusion patches `attention.RMSNorm.forward` and is
installed in this lane, and the remaining count matches `ascend_host_engram.py:475`, which evaluates
`torch.rsqrt` twice in one call — 40 calls a forward gives 80, plus 4 elsewhere. That attribution is a
hypothesis from the sources, not a targeted capture.

Capture and scripts: `.coord/accelerator-core-reattribution/`.

### The straggler wait is MoE routing imbalance, measured

The collective section above left the cause as a hypothesis: waiting a hundred times the median with
every core idle means waiting for the slowest rank. It is now measured, not inferred.

The lane runs expert parallelism four ways inside two-way tensor parallelism
(`expert_TP_EP = {expert_TP: 2, expert_EP: 4}`), a token selects **six** experts, and
`route_owned_kernel` states its ownership predicate literally: `owned = (ids % WORLD) == RANK`. So six
ids are dealt to four owners and the counts are small integers. A probe recorded the ids of 206 routed
dispatches on all eight ranks — buffered on device and fetched once, so no per-layer synchronisation
distorted anything — and the counts per owner are:

| | |
| --- | --- |
| spread (max - min owners) | 206 of 206 calls have at least 1; **141 of 206 (68.4%) at least 2**; max 5 |
| mean busiest owner | **2.772** of 6 |
| mean least busy owner | **0.388** of 6 |
| calls where one owner holds 3 or more | **125 of 206 (60.7%)** |
| mean idle owner-slots per dispatch | **5.087** |

Representative dispatches are `[2, 3, 1, 0]`, `[1, 0, 2, 3]`, `[0, 3, 2, 1]`: one owner does three
experts' work, another does none, and the remaining owners wait at the barrier. The busiest owner does
**seven times** the least busy one's work. That is the 19.3% of the step — not network, not bandwidth,
not a slow collective implementation.

Overlap was the obvious remaining fix, so it was measured before anything was built on it. Two
independent experiments say this device does not overlap streams at all. Two independent 4096-cubed
BF16 matmuls on two streams measured 1,078.4 us against 1,043.3 us sequential (ratio 1.034). And a
matmul on one stream against a 20 KB allReduce on another — different engines, cube against HCCL-SDMA,
so the compute-versus-compute result did not settle it — measured **481.5 us against 380.3 us
sequential**, which is the no-overlap sum (479.5 us) rather than the fully-overlapped maximum
(301.7 us). Two streams were 26.6% *slower* than one.

So the barrier wait cannot be hidden under work on another stream on this device. This closes the
stream-level route to the 19.3%, and it means the remaining gap to a tuned community stack is not an
overlap problem either.

Two consequences worth stating plainly. First, the fix cannot be a faster collective: the collective is
already fast when the ranks arrive together, and the wait is for compute that has not arrived.
`HCCL_OP_EXPANSION_MODE=AIV` improves a small collective's own latency and therefore does not address
this. Second, the fix cannot be rebalancing the routing: how many experts a token selects and which
owner holds them is the model's own behaviour and part of the protected contract. What is left is
overlap — issuing the collective on a stream that lets the waiting rank's next work proceed — which is
a change to the execution graph, and is the one structural lever this measurement points at.

Probe and receipt: `.coord/routing-imbalance/`.

### The single-core bucket is another sixth of the step

4938 tasks run on one core, averaging 2.36 us each. Most are at the launch floor (`Add` 1.48,
`Adds` 1.50, `Mul` 1.39 us), but a handful are not, and the worst of them was the window-index kernel
described above at 17.11 us for a 128-element table.

### The rest

The projection is the 24-core bucket almost by itself (12130 of 12284 us). Everything else is spread
across partial-core counts: 20 cores for 7911 us over 454 tasks, 40 cores for 5615 us over 496.

## Completion and remaining target

All model/test/native work stayed on the server. Final verification found 90 original
checkpoint files with stable metadata and no modification/change time inside this
phase; full checkpoint payload hashes were not recomputed. The selected native library
and shared CANN setup hashes are unchanged. All forty owned task receipts report
complete cleanup, with no remaining owned processes; the live fleet probe showed all
eight cards idle and no accelerator processes. The machine lease was released.

**50 tok/s remains unachieved.** The best accepted window measures 80.01 ms/token, while
50 tok/s requires 20 ms/token. No serving defaults or PRs were merged by this follow-up.
Further work must preserve the pure one-forward/one-token contract and target remaining
native operator and communication critical paths; dependent profiler scopes are not
independent latency floors and must not be summed.

## Exact original-activation preparation reuse

The model records its original FP4 weight codes and E8M0 scales verbatim and quantizes
activations on the original E4M3/E8M0 grid. One preparation result is only valid while
its source activation is unchanged, so the mechanism is a *scope*, not a cache: it holds
the original quantizer's `codes`/`scales` objects for one immutable batch-one attention
invocation, returns those same objects (including the existing fused-decode annotation)
to a second consumer, and clears on every exit. Nothing survives a forward or a graph
recording round, and the CPU path and prefill keep the original provider.

Three independent matched pairs at fixed accepted S1 layouts, one inline layer graph and
the S2b Torch-denominator state path measured:

| Pair | Control tok/s | Preparation reuse tok/s | Difference | Acceptance layer |
| --- | ---: | ---: | ---: | --- |
| First | 11.547189 | 11.606830 | +0.5165% | strict |
| Independent repeat | 11.560274 | 11.598147 | +0.3276% | approved endpoint |
| Independent third | 11.550968 | 11.614387 | +0.5490% | strict |

All three pairs are positive, mean +0.4644%, range +0.33% to +0.55%; this layer's best
window is 11.614387 tok/s at 86.1001 ms/token. The single-step kernel change described in
the last section then moves the whole step to 12.499052 tok/s, and that is the best number
this work has produced — still roughly 4.0x from 50 tok/s.

All eight ranks' reference-eight, teacher-eight, final and finite checks match the
immutable canonical result, and token IDs match. Actual excluded CANN traces show the
original preparation kernel dropping from 1,480 to 1,288 calls per four profiled
forwards — **48 real launches removed per forward on every rank** — while both native
expert-projection counts (480) and the accepted S2b state-kernel counts (552 each) stay
unchanged. The observed effect is a few tenths of a percent, consistent with the size of
the removed prep work rather than a structural change.

The same mechanism was then extended to the shared expert's gate and up projections,
which quantize one immutable activation twice. The extension is mechanically real — the
same trace drops from 1,288 to 1,128 calls, forty fewer launches per forward on every
rank — but it measured **no visible gain** (11.652520 to 11.631003, −0.18%, inside the
roughly half-percent run-to-run variation of one configuration). Its per-call site
validation issues two device-runtime queries on each of forty forwards, which plausibly
consumes the saving. It is recorded as **not adopted**: verification of a mechanism is
not a measured improvement, and the guard may not be relaxed without an explicit
session that keeps the same fail-closed semantics.

### The single allocator-sensitive step, and the two-value criterion

Across this whole session one step of the excluded 64-step copy pass — context 36 —
showed exactly **two** discrete FP32 states: the accepted contiguous-layout value
(`852d957a…`) and the pre-layout original value (`4824df27…`), differing by 129,251
elements with maximum absolute difference `0.003835439682006836`. Control columns have
landed on either state in different runs, so this is a process/allocation-state
artifact rather than a property of any one optimization. The mechanism is not proven and
is not claimed.

A direct user decision therefore approved a closed criterion: all other sixty-three
steps, both eight-step tensor families, the final logits, finiteness and the token IDs
must match the canonical result on all eight ranks, and that one step must equal
*exactly one* of those two named hashes with exactly that element count and maximum
absolute difference. Any third value, any additional changed step or any other magnitude
is rejected. This is a two-element set, not a tolerance: it does not retire the earlier
strict rejection, does not authorize new lossy quantization, a different activation grid,
fewer Sinkhorn iterations, narrower collective wire, a fused Triton replacement for the
Torch FP32 denominator tree, MTP, speculation or request batching.

The comparison tool reports the strict and the endpoint verdicts separately, with the
endpoint layer unavailable unless the recorded approval receipt matches those exact
values, and it is covered by synthetic decision tests that reject two columns merely
agreeing with each other on a third value, an extra changed step, a shared error in
another tensor, a wrong magnitude, or a missing approval.

## Where the remaining 86 ms actually goes, and why 50 tok/s is not a hardware wall

The accepted step is **not** near this card's memory limit. That is one resource, not a proof
that no hardware limit of any kind applies. Each card must read about 1.53 GB per
token (about 564 MB of routed expert weights at FP4 and about 970 MB of dense weights at
FP8), and the same card copies the same byte volume at **428.9 GB/s** (1232 GB/s at
256 MiB). That puts the memory floor at roughly **1.2–3.6 ms per token**, so the measured
86.1 ms is about **45x above the floor**, and 50 tok/s would need only about 10% of the
card's practical bandwidth.

The excluded rank-0 profile of four forwards shows where the time is charged. The sum of
every device kernel duration is 79.39 ms per forward against an 80.01 ms step, which is
numerically equal to the wall clock; that equality is *consistent with* a step that does not
overlap its work, but it is not a proof of it — overlap and host-side gaps can offset each
other. Per forward the step issues **10,877 kernels**, and the largest device-time groups on
the current best stack are:

| Group | ms per forward | calls per forward |
| --- | ---: | ---: |
| collectives (`hcom_allReduce_`) | 17.05 | 128 |
| expert projection kernel | 12.09 | 120 |
| `MatMulV2` | 6.46 | 348 |
| `Cast` | 4.12 | 1,615 |
| fused FP8 activation decode | 3.44 | 322 |
| `Mul` | 2.57 | 1,547 |
| `Transpose` | 2.55 | 201 |
| `MatMulV3` | 2.39 | 80 |

The remaining groups are almost all elementwise and layout glue: 662 `Slice`, 1,287 `Add`,
176 `StridedSlice`, 250 `Pows`, 226 `ReduceMean`/`ReduceSum`, 97 `Range`, 201 `ConcatD`,
plus 320 of the 1,615 casts. The glue population as a whole is therefore on the order of a
quarter of the step.

### The step costs about 5.5 times its own Cube time, and nothing overlaps

The same capture answers the question the buckets above only hint at. Taking the union of every
non-wait task interval gives **62.7 ms per forward**, which is exactly the sum of those tasks'
reported durations — so **no device work runs concurrently with any other device work**. The
step is one serialised chain, and its shape is:

| Component | ms per forward | tasks per forward |
| --- | ---: | ---: |
| small vector operators | **34.50** | 9,319 |
| collectives | 17.05 | 128 |
| Cube work (matmul and expert projection) | **14.48** | 200 |
| `AI_CORE` | 8.45 | 672 |
| `AI_CPU`, `MIX_AIV`, memcpy, SDMA | 5.28 | 1,210 |

The small-operator population is **overhead, not computation**: its mean task costs 3.70 µs
while moving kilobytes, which the same card copies at 428.9 GB/s in well under a microsecond,
and 82% of its calls are 5 µs or less. Fusing those chains would remove their cost without
changing a single arithmetic operation.

So the step costs **about 5.5 times its own Cube time**. Twenty milliseconds per token is
therefore the same target as bringing the step down to roughly its Cube work — with the
small-operator population fused, the collective wall time overlapped or reduced, and the
remaining Cube work shrunk — rather than a target that needs a faster arithmetic unit. That is
also why the earlier memory-floor comparison was too crude a summary: the 1.2–3.6 ms floor says
the card has bandwidth headroom, but it does not say what is actually spending the 80 ms, and
what is spending it is per-task cost and serialisation.

### Overlap does not exist on this device, which decides the shape of the work

The step's task intervals tile the wall clock with no gaps and no overlap, so the obvious next
question is whether that serialisation is a property of the model's dependency graph — something
that could be restructured — or of the device. The experiment is small: two independent
4096x4096 BF16 matmuls, captured into one NPU graph. Sequentially the replay costs **1043.3 µs**;
with the two matmuls on separate streams inside the same graph and joined by events it costs
**1078.4 µs**, a ratio of **1.034**. **No overlap.**

That closes the avenue rather than opening it, and it retro-explains two earlier observations:
the union of the task intervals equalling their sum, and making HCCL 4.9x faster moving the step
by only 0.86%. Two chains that share nothing still serialise, so no reordering of the model can
create concurrency here.

The consequence is unambiguous: **the step time is the sum of its task times.** Reducing it means
fewer or coarser device tasks — fusion — and nothing else. Stream parallelism, faster links and
better scheduling are all closed on this platform.

### What the fusion bound says about the target

Adding up the operands of every operator whose shapes and dtypes are parseable gives **7.49 GB moved
per forward** over 38,179 operators costing 36.44 ms. At this card's measured 428.9 GB/s that volume
would take **17.47 ms**, so the upper bound on what fusing buys is about **19 ms** — and that bound is
optimistic, since it assumes a perfect kernel and no dependency stall.

The composition matters more than the total. About **2.65 GB is the vocabulary projection alone**, a
single `FLOAT` matmul of `1,5120 x 129280,5120`, and about 2.4 GB is the rest of the dense matmuls. The
resident weights are only about **1.53 GB** per card per token, roughly 3.6 ms, so most of the
seven and a half gigabytes is intermediate traffic that a fused schedule would keep near the compute
units instead of writing and re-reading.

In those terms **50 tok/s is about 5.5 times the weight-only floor** and needs no faster arithmetic: it
needs the step to move roughly its weights once and to stop paying per-task cost for the intermediate
traffic. It is not reachable by tuning the current schedule, which pays 34.5 ms of per-task cost plus
17.05 ms of collective wall time on top of its 14.5 ms of Cube work.

### The collective term is a tail, not a per-call cost

A first reading of the aggregate — 128 collective calls at an average of 133 µs — suggests a
fixed per-call overhead that could be attacked by making the collective cheaper. **That
reading is wrong, and the distribution says so.** The calls split into two families:

| Family | calls per forward | median | maximum | total |
| --- | ---: | ---: | ---: | ---: |
| op 612 | 90 | **35.9 µs** | 2,164.7 µs | **15.92 ms** |
| op 909 | 40 | 29.3 µs | 38.8 µs | 1.20 ms |

The slow family's *median* is close to the 40.9 µs an isolated in-graph eight-rank
all-reduce of the same shape measured, and the fast family is uniform. What creates the
15.92 ms is a tail: roughly twenty-eight calls per forward run at 400 to 2,165 µs and supply
about 12.7 ms of it. The recorded wait column has a median of 2.0 µs, so those calls are not
waiting on their producers, and the earlier experiment that made HCCL itself 4.9x faster
moved the whole step by only 0.86%. Contention with concurrently running compute is a
plausible cause of the tail, but it was not measured here and is recorded as a hypothesis, not
a finding.

The practical consequence is that this term cannot be tuned away: it needs fewer collectives
or overlap with compute, which follows from the two-way tensor-parallel sharding rather than
from any knob. The same is true of the glue: 1,615 casts and 1,547 multiplies per forward are
a property of how the forward composes its operators, not of how fast any one of them is.

Two measurements keep the diagnosis honest:

- The collective bucket is **dependency wait, not wire time**. Setting
  `HCCL_OP_EXPANSION_MODE=AIV` makes a captured eight-rank FP32 all-reduce of `[1, 5120]`
  run in 8.3 µs instead of 40.9 µs (4.9x), yet the whole step improved only 0.86%
  (11.6438 to 11.7436 tok/s) and the outputs stopped being bit-identical (129,280 changed
  final-logit elements per rank), so it was rejected. Faster collectives do not help a
  chain that is waiting for its producers.
- The largest compute term is **scalar-bound, not bandwidth-bound**. The expert projection
  kernel reaches 39.8 GB/s where the card copies at 428.9 GB/s, and its hardware counters
  say why: `aiv_scalar_ratio` 0.93, `aiv_vec_ratio` 0.72, `aic_mac_ratio` 0.03. The Cube
  multiply unit is nearly idle while the AIV scalar unit is saturated, because the decode
  index and address arithmetic runs as per-element scalar code.

### A first verified kernel improvement on that term

The per-pair scale-offset expansion had a fixed pattern (each of the N group offsets
repeats eight times, and that expansion fills the group twice) but used a general gather,
whose per-element offsets are exactly the scalar work above. Replacing it with the vector
broadcast instruction reproduces the offsets exactly and measures:

| Case | Accepted kernel | Broadcast variant | Change |
| --- | ---: | ---: | ---: |
| one selected expert | 258.3 µs | 242.9 µs | −5.96% |
| six selected experts (the real width) | 670.8 µs | 563.1 µs | −16.06% |

Outputs are **bit identical**: 0 of 1152 and 0 of 6912 elements changed, maximum absolute
difference 0.0.

Two details were found in review and both matter for landing this change. The broadcast
destination repeat stride is a constant **eight 32-byte blocks** — one repeat always writes
eight blocks totalling 64 32-bit words regardless of the tile width, so it is a block stride of
8 and a word stride of 64 — so deriving it from the tile
count (`N/8`) is only accidentally right at N64 and would pack the N32 geometry wrongly; the
source now uses the constant. The implementation header also shows that only
`dstBlkStride`, `dstRepStride` and `repeatTime` reach the `vbrcb` instruction, so the source
advance is implicitly consecutive and no source stride is configured. With those in place
the final source (sha256 `caca541d1504c52699f0eb2aaf785ffca8bc7dae0a79958bae80480097791b52`,
33-line diff) rebuilds to exactly the N64 library the three pairs measured. Its N32 build is a
different binary by construction, and that binary reproduces the accepted N32 library's
**outputs** bit for bit (0 of 1152 and 0 of 6912 elements, maximum absolute difference 0.0);
no claim is made that the two N32 binaries are the same file. The variant is built as its own library
(`46aca3d9e67cdf8204382d8eb37e8bf3ae603150242b0bd099cc2be8f16c9117`) and the native build
was shown to be reproducible: rebuilding the accepted source with
`-DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu -DSOC_VERSION=Ascend910B3
-DRELIC_LUT_OUTPUT_ROWS=64` and `--parallel 2` reproduced the accepted library byte for
byte. Release mode matters — without it the CANN preprocess/merge step fails with a bare
`unknown file type`.

The variant was then wired in as the only change against the accepted stack, and three
independent matched pairs measured the whole step:

| Pair | Accepted library | Broadcast variant | Change | Saved | Output gate |
| --- | ---: | ---: | ---: | ---: | --- |
| First | 11.605510 | 12.499052 | +7.6993% | 6.160 ms | strict |
| Independent repeat | 11.632330 | 12.447992 | +7.0120% | 5.633 ms | strict |
| Independent third | 11.623546 | 12.451103 | +7.1197% | 5.718 ms | strict |

All three are **strict**: every reference-eight, teacher-eight, final and complete
sixty-four-step FP32 tensor plus the token ids match the immutable canonical result on all
eight ranks, on the first acceptance layer with no endpoint allowance needed. Mean gain
+7.28%; the best accepted window becomes **12.499052 tok/s at 80.0061 ms/token**, down
from 86.17 ms.

The measured gain is larger than a proportional reading of the kernel's profiled share
would suggest: the kernel was 14.83 ms of the control stack's 86.99 ms, times 16% is about
2.7% of that step's time, while the pair
measured a 7.15% time reduction (86.166 to 80.006 ms), i.e. 7.7% more tokens per second. The
two are quoted in the units they were measured in and are not interchangeable. **Why is not isolated and is not claimed.** A kernel-level number
and an in-model number are taken under different conditions, and a changed decode loop can
also influence register pressure, scheduling and the code the compiler emits for the rest of
the kernel, so the two figures are not required to agree and neither explains the other.
Membership of the critical path does not by itself amplify a kernel's own absolute saving.

On the provider question: each side's preconditions record `RELIC_TILE_CONSUMER_LIBRARY`
and `RELIC_TILE_CONSUMER_SHA256`, and the consumer binding hashes the file it was pointed at
and only then hands that same path to the dynamic loader, so the verified bytes are the bytes
loaded. The comparator additionally re-hashes both files after the runs and refuses a pair
whose recorded identities are not distinct. That is strong evidence the two sides ran
different named binaries; it is not an independent trace of what the NPU executed.

### The output head is the second largest item, and it is a deliberate fidelity choice

The step's second largest single kernel is not in the layer stack at all. The profile's one
`MatMulV2` FP32 call per forward costs **2336 us**; its neighbours identify it unambiguously as
`logits = self.head(self.norm(h))` — the fused row norm, then the projection, then `ArgMaxV2` and
`ConcatD` for the greedy token.

`Head` keeps its weight in **fp32 although the checkpoint stores bf16**, and the code says why: the
logits come out of it and the sampler reads them at full width. The consequence is that a single-row,
memory-bound projection reads **2.648 GB instead of 1.324 GB per token** — doubling the traffic to
carry no additional information, because every bf16 weight value is already exact in fp32.

Measured on the device at the real geometry (129280 x 5120):

| Head | per call | logits dtype | worst logit error |
| --- | ---: | --- | ---: |
| fp32 weight (current) | 2390.7 us | float32 | — |
| bf16 weight | **1104.0 us** | **bfloat16** | 0.95 absolute |

So the bf16 head is 2.17x faster, worth about 1.29 ms a token (1.8% of the step), and it **cannot be
taken for free**: this platform returns bf16 logits from a bf16 matmul, and `npu_quant_matmul` — the
one API that accepts an `output_dtype` — rejects bf16 operands outright here, so there is no
bf16-in/fp32-out path. Narrowing the logits is a change to the values the sampler reads, which is
outside the row-normalization deviation class and outside what this programme may adopt on its own.
It is recorded as a decision for the user, not as a candidate.

#### The bf16 head measured at the step level

A single strict pair was run with the bf16 head in the candidate column, purely to price the decision:

| | value |
| --- | ---: |
| control | 12.4077 tok/s |
| candidate | **13.9037 tok/s** |
| gain | **+12.057%, 8.672 ms** |
| over the accepted three-fusion lane (13.5948) | about +2.2% |
| token ids, all eight ranks, all 64 steps | **identical** |

The token ids matching is a real and useful observation, but the tensor deviation is not small: the
reference tensor differs on 1,034,216 of 1,034,240 values at up to 0.0623, the teacher-forced tensor
on 1,034,219 values, and the final tensor on 129,277 of 129,280 at up to 0.0603 — on **every one of the
64 steps**. Against an approved class of at most 130,000 values, 1e-4 and a single step, that is a
different kind of change: a broad reduction in the width the sampler reads.

**The user authorised adopting it** after being shown those numbers, so it is now part of the lane:

* the head keeps the width the checkpoint already stores, instead of being widened to fp32;
* the weight payload on disk is untouched — the values used are exactly the published ones;
* the sampler now reads bf16 logits, which is the cost that was accepted, and it is recorded as its
  own deviation class in `narrow-output-head-user-approval-20261123.json`.

This needed its own decision rather than an extension of the row-normalization class, and the receipt
says so explicitly: eight million changed values across all 64 steps is a different kind of change from
130,000 on one step, even though the token ids happen to be unchanged.

Registration took three independent strict pairs, all against the same recorded signature:

| Pair | Control tok/s | Candidate tok/s | Gain | Token ids | Signature |
| ---: | ---: | ---: | ---: | --- | --- |
| 1 | 12.434331 | 13.854150 | +11.4185% | identical on 8/8 ranks | reproduced |
| 2 | 12.439337 | 13.806608 | +10.9915% | identical on 8/8 ranks | reproduced |
| 3 | 12.515195 | 13.879445 | +10.9008% | identical on 8/8 ranks | reproduced |

Mean **+11.1036%**, median **+10.9915%**, best **13.879445 tok/s / 72.0490 ms per token**. The
signature held exactly in every pair: reference 1,034,216 changed values at 0.062255859375,
teacher-forced 1,034,219 at 0.0624847412109375, final 129,277 at 0.06025886535644531.

### Where the lane stands after four changes

| Change | Class | Effect |
| --- | --- | --- |
| fused row normalization | approved deviation class | +2.518% mean |
| fused rotary pairs | bit exact | +8.254% mean |
| fused window index build | bit exact | engages twice a step; inside the variation |
| narrow bf16 output head | user-authorised class | +11.104% mean over the brcb control |

The last figure is measured against the same brcb control, so it includes the three earlier changes:
the lane now runs **13.879445 tok/s at 72.0490 ms a token**, against 12.499 tok/s / 80.006 ms at the
start of this series.

### A gather that looked like an int64 problem and is not

The four `Index` tasks cost 936.5 us a forward, each preceded by `IndexCheck_int64`, which suggests
int64 addressing. It is not: an int64 gather and an int32 gather of the same shapes measure within one
percent of each other (ratio 1.00 to 1.01, identical values). The cost is the gather itself. Closed.

### What the scale-LUT kernel is actually bound by, and why it is not worth rewriting

The scale-LUT expert projection is the largest single device item on the step, 120 calls and about
12.1 ms a forward. The profiler's per-kernel columns split it by core: the vector side is busy on
99.0 us of a 101.1 us task while the Cube side is busy on 28.4 us, and `aiv_scalar` alone is 58.3 us
of that. The kernel is vector-scalar bound, not matrix bound.

The obvious suspect is the two general gathers in the decode loop, which pay per-element offset work
on the scalar unit. To price them rather than guess, a throwaway library was built with the large
gather on `BYTES` replaced by an equal-size non-gather op, and both libraries were run through
`compare_lut_kernels_any_rows.py` at the released geometry (1152 rows, K 5120, six experts, TP2):

| Groups per call | With the gather | Without it | Difference |
| ---: | ---: | ---: | ---: |
| 1 | 251.1 us | 241.6 us | 9.5 us |
| 6 | 568.5 us | 456.2 us | 112.3 us |

Marginal cost per group falls from 63.5 us to 42.3 us, so the gather is about a third of the
marginal per-group cost and roughly a fifth of the kernel. The probe differs numerically (it has to,
it is not the decode), and a differing candidate is rejected outright here — its only purpose was to
price the gather.

That bounds the prize: even replacing the gather with *free* arithmetic cannot return more than about
2.4 ms of the 73.2 ms step, and a bit-exact arithmetic decode would return substantially less for a
multi-hour native rewrite. `block_dim` is already the hardware maximum of 24, so there is no unused
parallelism either. This avenue is closed on the evidence, not on a hunch.

### A fusion that was bit exact and still lost: the hyper-connection affine front end

The gate builds its three coefficient groups with twelve small dispatches — three multiplies, three
adds and the scalar and slice reads feeding them — before sigmoid, softmax and the Sinkhorn kernel run
untouched. Those twelve are pure elementwise multiply-then-add, so a single kernel reproducing them
was expected to be free.

It was bit exact. Four trials at one and five rows, all three outputs, zero mismatching elements, with
no fallback in the path. A first attempt was a false pass because a compile error silently fell back
to the original provider; the retest disabled the fallback and drove the kernel directly, which is the
only reason the difference was caught.

And it was slower: **12.4791 against 11.4401 tok/s, 8.3% worse**, with 2,252 fused calls and zero
fallbacks in the timed run. One program over twenty-four lanes carries fixed device cost — dispatch and
a single-block execution — that exceeds twelve tiny vector ops which the captured graph issues
back-to-back. Reducing operator count is not the same as reducing time on a device whose graph already
batches small work. The change was reverted and the lane returns to its accepted state.

### The third change: a single-kernel window index build, and what it actually buys

`get_window_topk_idxs` builds a 128-entry int32 table (four hundred bytes) with eight launches: an
`arange`, an add, a modulo, a compare, a select, a cast, a broadcast over the batch axis and a
materializing copy. On the device path the position arrives as a tensor, so the build is inside the
capture and cannot be cached in Python. One kernel reproduces it exactly — integer arithmetic in the
same order and widths, cast after the select, and no copy because the kernel already writes a real row.

Bit parity was checked at twelve positions including the ring wrap boundaries (127, 128, 129): **0
mismatches**, same int32 dtype, same `[1, 1, 128]` shape.

Its contribution, however, is nearly nothing, and this is worth recording as such. In the timed run the
fused build took **126 calls and 1,000 fallbacks**: most calls take a path where the position is a host
integer, so the fused form engages about **twice a step**. Twenty-five microseconds saved twice is
0.07% — inside the variation.

The combined lane was then measured in three independent strict pairs:

| Pair | Control tok/s | Candidate tok/s | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 12.513459 | 13.632089 | +8.939% |
| 2 | 12.439789 | 13.658971 | +9.801% |
| 3 | 12.437151 | 13.650268 | +9.754% |

Mean **+9.498%**, median **+9.754%**, best **13.658971 tok/s / 73.2120 ms**. Every pair keeps the
reference, teacher-forced and final tensors identical, and the only difference is still the approved
row-normalization one (step 10, 127,476 of 8,273,920 values, 9.34600830078125e-05), so neither the
rotary nor the window change added any deviation.

The previous lane measured +8.254% and this one +9.498%, but the window build can only account for
about a tenth of a point, so **that one-point shift is not attributed to it** — it is run-to-run
variation between capture sessions, and the honest reading is that the lane sits between eight and ten
percent over the brcb control.

### The second fusion: rotary pairs, bit exact, +8.25% over the norm-fused base

The next cluster was the rotary embedding, `apply_rotary_emb_pairs`. It is thirteen operators per
call — two table slices, two casts, two strided views of the activation, two more casts, four
multiplies, a subtract, an add, a stack, a flatten and a copy — and the decode path calls it about
fifty-four times a step.

Its arithmetic is four products and two sums per pair, and it is already the real `(cos, sin)` form
rather than a complex multiply. A kernel performing the same four products and two sums **in the same
order**, writing into the activation's own storage so the copy disappears, reproduces it bit for bit:

| Where it was checked | Elements | Differences |
| --- | ---: | ---: |
| query, 8 heads of 512 | 4,096 | **0** |
| key, 8 heads of 128 | 1,024 | **0** |
| inverse on the attention output | 4,096 | **0** |
| the strided `q[..., -rope:]` slice | 5,120 | **0** |

So this one needs no numerical approval at all. Three independent strict pairs, 64 steps, all eight
ranks:

| Pair | Control tok/s | Candidate tok/s | Gain |
| ---: | ---: | ---: | ---: |
| 1 | 12.490423 | 13.478655 | +7.912% |
| 2 | 12.459577 | 13.514229 | +8.465% |
| 3 | 12.488831 | 13.536170 | +8.386% |

Mean **+8.254%**, median **+8.386%**. In all three pairs the reference tensors, the teacher-forced
tensors and the final logits are identical, and the only difference anywhere is the same one the row
normalization produces — step 10, 127,476 of 8,273,920 values, `9.34600830078125e-05` — so the rotary
change contributed **exactly zero** to the deviation.

New best pure per-token B=1 measurement: **13.536170 tok/s / 73.8761 ms per token**, against
12.499052 / 80.0061 before the two fusions and 12.775019 / 78.2817 after the first.

### The first accepted fusion: +2.52% under a user-approved deviation class

The bit-exactness wall below was then taken to the user as a scoped decision, with the numbers that
the wall produces. The user authorised **this class**: a fused kernel may be accepted when all eight
ranks keep the original reference tensors, the teacher-forced tensors, the final FP32 logits, the
timed token IDs and finiteness identical, and every difference is confined to **at most one of the
sixty-four steps, at most about 130,000 of that step's 8,273,920 FP32 values, and at most 1e-4 in
absolute value**.

The row normalization the decode path calls about 75 times a step was then fused into two kernels
with rstd left in the framework. Three independent strict pairs, 64 steps each, all eight ranks:

| Pair | Control tok/s | Candidate tok/s | Gain | Difference |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 12.468812 | 12.774372 | +2.451% | step 10, 127,476 values, max 9.35e-05 |
| 2 | 12.470666 | 12.747763 | +2.222% | step 10, 127,476 values, max 9.35e-05 |
| 3 | 12.417360 | 12.775019 | +2.880% | step 10, 127,476 values, max 9.35e-05 |

Mean **+2.518%**, median **+2.451%**, and the difference is **bit for bit the same in all three
pairs** — the same step, the same count, the same magnitude. That reproducibility is what makes the
class a class rather than a tolerance: it is a deterministic consequence of the framework's row
reduction associating the terms differently from the kernel's, and the affected step is the one whose
activations carry the outliers that separate the two associations.

New best pure per-token B=1 measurement: **12.775019 tok/s / 78.2817 ms per token**, against the
previous accepted **12.499052 tok/s / 80.0061 ms**. Serving defaults are unchanged and no claim of
50 tok/s is made.

### Bit-exact fusion is available but slower, which is the real wall

Because the step is the sum of its device tasks and nothing overlaps, the only way down is fewer and
coarser tasks. That raises one question: can a coarser task reproduce the framework's bits? The
normalization the decode path repeats many times a layer was used to answer it.

| What was tried | Accuracy | Cost vs eager |
| --- | --- | ---: |
| one kernel, reductions and cast inside | 3 of 819,200 BF16 elements differ | **0.57x** (195 vs 340 us) |
| two kernels, rstd in the framework | **0 of 819,200** | 1.41x (478 vs 340 us) |

So the exact form is *slower* — the second kernel re-reads the activation and writes an FP32 result
the eager chain kept in registers — and the fast form deviates by about one BF16 element in 200,000.
The same pattern held for the hyper-connection gate: a fused kernel was 3.2x faster per call and
differed in 950 of 1,280 comb elements.

Parts of the arithmetic *are* reproducible exactly, and they are worth recording because they bound
what is possible: a single-block row sum of squares (tile-wise accumulation is not), the mean as the
sum **multiplied by the reciprocal** of the length rather than divided by it, `sigmoid` as
`1/(1+exp(-x))`, `exp`, and softmax over four terms with a **sequential** denominator and true
division. Not reproducible: the four-by-four Sinkhorn reduction (four candidate associations each
differed in over 900 of 1,280 elements) and `rsqrt` inside the kernel (7 of 40 rows differed, enough
to change the final rounding).

The consequence is not an engineering step but a decision. Reaching 20 ms by fusion needs a scoped
numerical deviation for the affected patterns, and that deviation is already measured: about one
BF16 element in two hundred thousand buys roughly 1.7x on the normalization pattern. No such
authorisation exists today, and this work does not assume one.

### What that means for the target

Fixing what is provably broken — the expert decode path, the glue operators, the small-op
population, the dense GEMV — is a *heuristic* projection of roughly **22–25 tok/s, about
2x**, made by attributing each bucket's measured time and assuming a plausible reduction. It
is not a proven bound on tuning this kernel set, and it does not account for interactions
between the buckets. Reaching 20 ms per token needs
about another 2x on top, which means a fused decode path that reads each resident weight
once per token and never materializes the intermediate elementwise chain: a kernel
architecture change, not further tuning. Until such a path is measured under the full
eight-rank gate, **no 50 tok/s claim exists** and none of the numbers above is a serving
throughput.

## Evidence bundle for rounds 49 to 67

`docs/performance/evidence/ascend-pure-b1-20261122/follow-up/rounds-49-67/` collects this stretch of
work with a sha256 manifest and two verifier scripts: the routing-imbalance measurement, the
negative results for stream overlap and host-side overhead, the cost model by engine, the
indexer-selection fusion and its three-pair registration, and the community deployment survey that
bounds the target. Every number quoted there was checked against its receipt programmatically.

## Evidence bundle for rounds 68 to 70

`docs/performance/evidence/ascend-pure-b1-20261122/follow-up/rounds-68-70/` records the three measured
negative results that followed the indexer fusion — the o-projection transpose, the scale-LUT expert
projection run for the first time, and the device-bound measurement — plus the redundant-quantisation
lead with the buffer-reuse hazard that makes a quick fix unsafe. The registered result did not change:
none of that work was adopted. A sha256 manifest covers every file.
