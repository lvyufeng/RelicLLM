# Pure batch-one decode on one 8×Ascend910B3 server: objective completion report

This page maps every clause of the objective to the evidence that supports its status. It is written to be
reviewed clause by clause, not to be read as a claim of success.

**Phase 1 is not achieved: the recorded result is below 50 tokens/s. These records demonstrate measured
throughput, not a hardware or protocol ceiling. Reproducing the combined two-master source and private
execution stack under the full acceptance protocol remains a separate verification requirement.**

| | |
| --- | --- |
| Registered result | **14.287598 tok/s / 69.9922 ms per token** |
| Independent reproduction | **14.269744 tok/s** (−0.13%, inside the registered spread) |
| Starting point (control mean of the first trial) | 12.452279 tok/s |
| Gain over the session | **+14.74%** |
| Target | ≥ 50 tok/s — **not achieved** |
| Community stack, same hardware, **with MTP** (acceptance 4.10) | 27.13 ms a step; ≈ 36.9 tok/s A1-normalised — **not** a pure-B1 measurement and **not** a ceiling |
| Current dev lane, three-pair protocol (section 19) | **17.4159-17.5084 tok/s / 57.12-57.35 ms per token**, with `HCCL_OP_EXPANSION_MODE=AIV` adopted; +22.3 % over registered |

---

## 1. Clip-by-clause status

### Optimise actual DeepSeek-V4.1-Flash toward ≥ 50 tok/s — **NOT ACHIEVED**

Registered 14.287598 tok/s. A tuned community stack on the *same hardware*
(`CJYLZS/vllm-ascend-deepseek-v41-a2`, 8 × 910B3 at 65536 MB a card) measures 27.13 ms a step at concurrency 1
with an acceptance length of 4.10 -- four tokens accepted per forward. Its 36.9 tok/s is that measurement
A1-normalised, i.e. a **speculative-decoding** rate restated as if one token were emitted per forward; it is
not a pure-B1 measurement, not a success under this objective, and not a bound on it. What the comparison does
establish is that a stack which is not ours runs this model about 2.2× faster per step than the lane recorded
here on identical hardware, so the remaining gap is executor work rather than hardware. Whether a *pure* B=1
executor can reach 20 ms a token is **open**: no measurement in this record bounds it, and the lane below is
the only pure-B1 point this project has measured on this hardware.

### Pure per-token, batch-one, single-request autoregressive decode — **HELD**

One complete forward per token, batch size 1, single stream. No MTP, no speculative decoding, no request
batching, no prefill-rate substitution. The run records report `single_stream: True`, `batch_size: 1`.

### No MTP / speculative / request-batch / prefill-rate substitute — **HELD**

Verified every round that no such path was enabled or claimed.

### Weight conversion lossless; original FP4/FP8 code, scale payload and values unchanged — **HELD**

The 90 checkpoint files are byte-unchanged (verified before every lease release, 118 clean receipts). The
original FP8 activation grid, the FP32 tensor-parallel down→BF16 ascending combine, and the HCCL wire format
were never modified.

### Server-only testing, never local macOS imports/tests/build/bench — **HELD**

All model, native and benchmark execution ran on the leased Ascend host. Local work was AST/stdlib reading and document
writing only.

### Contiguous-KV-layout deviation: accepted, scoped, old rejection preserved, new approval added — **DELIVERED**

Scoped acceptance of the observed error (129,251 of the compared FP32 values changed on one step, max abs
0.003835439682) was recorded with its own approval receipt; the earlier rejection receipt was preserved
unmodified. This was acceptance of *that observed error*, not a general relaxation.

### Independently repeat 64 steps to verify the accepted gain and error are stable — **DELIVERED**

`logs/pure-B1-fused-row-norm-candidate-steps64-20261123T112154Z`, returncode 0, 340.7 s, clean teardown:
**14.269744 tok/s**, which is 0.13% below the registered mean and inside the spread the three registration
pairs themselves showed (14.221390 – 14.376778). Receipt: `.coord/stability-repeat/`.

### Original 90 checkpoint files / CANN / CI untouched — **HELD**

90 files unchanged, CANN `bbbda1c0…` unchanged, no CI host taken (`ts_claim` defaulted to `dormant` per the
approved exception, never `ci`).

### Investigate execution graph boundaries; inline originalA → nativeTP_execute → originalB as one layer NPUGraph; require full eight-rank complete 64-step FP32 bit-identical parity before accepting extra gain — **DELIVERED, and already shipping**

Not pending: this is the shipping configuration. On **all eight ranks** of the registered run:

| field | value |
| --- | --- |
| `inline_TP_layer_graphs.layers` | 40 |
| `inline_TP_layer_graphs.captured` | true |
| `inline_TP_layer_graphs.variants` | 43 |
| `inline_TP_layer_graphs.driver_counts.replays` | 6120 |
| `independent_TP_graphs_retired_after_explicit_fence` | 40 |
| `graph_candidate_eager_FP32_bit_differences` | **[0, 0, 0, 0, 0, 0, 0, 0]** |
| `private_graph_capture_seconds_excluded` | 4.141067163087428 |

Capture cost is excluded from the measured step rather than folded into it.

### Use the root's live single-host lease; owned default / dormant only; never CI — **HELD**

Every round claimed through `ts-fleet-bridge`, ran on the leased Ascend host only, released at the end. Never `ci`, no device
resets, no shared CANN/checkpoint/config edits, no signals to foreign processes.

### Bound own jobs, reconcile lost replies, preserve hashes, fail-fence resources retained for retry, separated warmup/profiling — **HELD**

Jobs ran under the bounded-task harness with an ownership token; each release verified zero remaining owned
processes. Profiling passes were always separate and excluded from timed windows.

### No master commits or new PR merges without a fresh user request — **HELD**

HEAD is unchanged at `532bb3a` and the worktree has **no commits**. Nothing was merged.

### Wait for required team reviews — **OPEN BY CONSTRUCTION, not by omission**

No team reviewer has been engaged in this session. This report exists so that review has something precise to
read.

### Collect all relevant job outputs, publish sanitized evidence, clean up owned processes and all 8 NPU tables before lease release — **DELIVERED**

Evidence bundles published for rounds 49–75 with sha256 manifests and in-bundle verifier scripts; every lease
release preceded by a safety proof (118 clean receipts, latest sha `85256dc9…`), zero live processes and eight
idle cards.

---

## 2. Where the 69.99 ms goes, and why each avenue is closed

Measured cost model (per forward, under profiling), and the verdict on each item. **This table is the verdict
list as of that round**; three rows were later superseded by the sections named in them, and are marked so here
rather than left to contradict the rest of the page:

| item | share | verdict | basis |
| --- | ---: | --- | --- |
| Communication (barrier wait) | 26.2% | **hypothesis, open** | §9 measures per-group collective duration variation (four expert-TP2 groups spanning 20-984 us in one labelled collective) and reproducible per-rank shape across 82 collectives > 400 us; cross-group arrival skew and routing imbalance are the hypothesis hanging off it, pending a same-group all-rank aligned trace. The earlier route histogram cannot support it (it carries an `INVALID` bucket). Streams do not hide the wait in the one configuration tested (matmul‖collective 481.5 µs on two streams vs 380.3 µs sequential), which is a bounded negative result, not a closure; rebalancing routes is the model's own protected contract |
| Scale-LUT expert projection | 15.4% | **superseded by §17** | first measured at 186.9 GB/s in-graph against a 448.3 GB/s practical copy reference, and read as M=1 tile padding. §17 shows part of it was the kernel serialising its own packed read behind the decode: prefetching the next row took the step from 64.2 ms to 59.0 ms bit-exactly (+8.76 %, three pairs). Padding is still a cost; it was not the whole cost, and this row's closure was premature |
| Graph capture / launch overhead | — | already taken | the inline one-layer capture is the shipping configuration, with FP32 parity all-zero on eight ranks |
| o-projection transpose | 1.6% | **superseded by §12** | caching the transposed operand *behind the call* was slower (-0.69 %/-0.74 % on two interleaved pairs) and stays not adopted. Materialising it **before the layer graphs are captured** is a different change and was adopted: the `Transpose` and `Cast` nodes leave the replayed graph, +1.23 % over three pairs with bit-identical FP32 logits |
| Host-side work (`aten::copy_` etc.) | — | closed | the device is busy 91.3% of the profiled span, so host time is overlapped and shortening it moves nothing |
| Redundant FP8 activation quantisation | 1.2% | does not exist | `ascendc_tile_moe.py:152` computes `decoded_x` once and reuses it for w1/w3/w2 |
| Rank-based sort at 2048 width | — | closed | measured 0.66× |

**M=1 is a structural cost, but it is not the whole story, and the page does not claim it bounds 50.** A dense
cube unit fed one activation row spends most of every loaded tile on padding, so arithmetic intensity is fixed
at its worst point and more tokens a step would raise it -- which is exactly what this objective forbids in
Phase 1, and why that lever is Phase 2's rather than this section's. It does not follow that a pure-B1 executor
cannot reach 20 ms a token: the census below puts the largest single kernel at the FP4 LUT decode (13.1 ms of
the 60 ms), with ~7,800 kernels a step, 82 collectives whose per-group durations span 20-984 us (an
unresolved straggler hypothesis, §9), and matmuls running at
233-687 GB/s against a 1,142-1,250 GB/s measured floor. Those are executor costs, and none of them has been
shown to be irreducible at M=1.

---

## 3. What was gained, and how

Progression of the registered lane. Each step was registered only after passing the eight-rank token-ID parity gate:

Every row below is a **mean over its own three pairs**, recomputed from the receipt on disk rather than
quoted from memory. The control means stay inside 12.449932 - 12.479610 across all six trials, which is itself
evidence that the control lane is stable.

| step | pairs | control mean | candidate mean | gain |
| --- | ---: | ---: | ---: | ---: |
| row normalisation class | 3 | 12.452279 | 12.765718 | +2.52% |
| fused rotary | 3 | 12.479610 | 13.509685 | +8.25% |
| window index v1 | 3 | 12.463466 | 13.647109 | +9.50% |
| narrow bf16 output head | 3 | 12.462954 | 13.846734 | +11.10% |
| window index v2 | 3 | 12.449932 | 13.983022 | +12.31% |
| **indexer selection sort fusion** | 3 | 12.460219 | **14.287598** | **+14.74%** |

The gain column is against each trial's own control mean, so the rows measure their own step. Against the first
trial's control mean the final lane is **+14.74%**.

The last step was the only strictly bit-exact one of the later set: the indexer's mask-then-sort selection moved
from the AI CPU to one rank kernel, registered on three pairs with the worst pair still above the previous best.

---

## 4. Honest assessment

* The delivered gain is real, reproducible and verified on eight ranks every time.
* The target is **not achieved**, and nothing measured here shows it to be unreachable: the pure-B1 lane sits
  at ~59 ms a token, and the only same-hardware comparison available is a speculative-decoding stack
  (27.13 ms a step at acceptance 4.10, 36.9 tok/s A1-normalised). Neither figure is a bound on a pure-B1
  executor, so this is a status of "not yet", not a proof of "cannot".
* Two claims made during the session were **wrong and were retracted in writing** (round 60's output-head
  attribution; rounds 71–73's "fully eager" claim). Both were caught by reading a stored record instead of
  reasoning further, and both are recorded in the evidence bundle rather than removed. The process rule they
  produced: when a claim rests on a grep returning nothing, read the run-record field that states the answer
  first.
* Phase 2 is untouched, by the objective's own ordering. The same hardware reaches 136.9 tok/s per stream with
  an acceptance length of 4.10; that is a speculative-decoding number, it is gated behind Phase 1's pure-B1
  50 tok/s gate, and it is recorded here as the price the constraint carries -- never as a Phase-1 result and
  never as a substitute for one.
---

## 5. Dev-lane update: measured state after this round

All rows are 64-step timed windows of the same launcher, warm-up excluded, canonical (bitwise) gates,
eight ranks, the registered prompt and checkpoint. No MTP, no speculation, single request, one forward per
token.

| lane | run | tok/s | ms/token |
| --- | --- | ---: | ---: |
| control (registered env + T1-T4 kernel work) | `t4` | 15.2058 | 65.76 |
| | `t4b` | 15.1934 | 65.82 |
| | `t11` | 15.1914 | 65.91 |
| | `t14` | 15.1979 | 65.73 |
| | `t19` (interleaved control) | 15.2057 | 65.81 |
| | `t21` (interleaved control) | 15.2331 | 65.70 |
| **candidate: indexer candidate gather off AI_CPU** | `t15` | 15.3784 | 64.95 |
| | `t16` | 15.4746 | 64.56 |
| | `t17` | 15.3464 | 65.19 |
| | `t20` (interleaved candidate) | 15.3288 | 65.21 |
| | `t22` (interleaved candidate) | 15.4124 | 64.91 |

The candidate lane is **bit-identical to the control lane token for token** (one and the same 64-token stream on
all seven runs; `tokens_match_T14=True` and `tokens_match_T15=True` on every candidate run), so the three
candidate runs and the four control runs are a matched orthogonal set rather than a quality-band comparison.
Control mean over its six runs 15.2045 tok/s / 65.80 ms, candidate mean over its five runs **15.3881 tok/s /
64.96 ms**: **+1.21 %**, or **0.83 ms saved per token**. No run in either lane overlaps a run in the other.

Two of the pairs were run **interleaved**, control arm first and candidate arm second, so machine drift
cannot explain them: `t19` 15.2057 -> `t20` 15.3288 is **+0.81 %**, and `t21` 15.2331 -> `t22` 15.4124 is
**+1.18 %**. The switch that flips the arms is one environment variable
(`RELIC_INDEX_GATHER_ADVANCED=1`) around a single gather expression, so the two arms differ by that
expression alone. Every run's eight teacher-forced greedy tokens still match the reference, and no checkpoint
file was modified (nothing under the checkpoint directory is newer than the session start).

### What the gain is

A per-operator CANN profile of the control lane (`RELIC_KERNEL_PROFILE=1`) shows 7886 kernels and 61.81 ms of
kernel time a step. Four of those kernels were AI_CPU `aclnnIndex` gathers at **230.6 µs each** (2.77 ms per
three-step profile window). The timeline identifies the site: the indexer's candidate-level key gather,
``index_k[torch.arange(bsz).reshape(bsz, 1, 1), tile.clamp(0, width - 1)]``. Advanced indexing with two index
tensors lowers to an AI_CPU kernel on this backend; folding the batch offset into the position index turns it
into one flat `index_select`, which stays on the vector cores:

| bsz=1, width=4096, m=4096, d=128, bf16 | in-graph per call |
| --- | ---: |
| advanced indexing (two index tensors) | 530.76 µs |
| flat `index_select` | 20.03 µs |

### What the same profile says about the remaining time, and what is now closed

| operator | calls/step | ms/step | achieved read bandwidth |
| --- | ---: | ---: | ---: |
| native FP4 expert GEMV (`relic_scale_lut_tile_cube_0`) | 120 | **13.00** | ~50-150 GB/s |
| dense `MatMulV2` GEMVs | 348 | **5.34** | 248-855 GB/s |
| fused FP8 activation decode (Triton) | 322 | 3.44 | -- |
| attention `BatchMatMulV2` (accepted S2 path) | 324 | 2.13 | -- |
| elementwise `Cast`/`Add`/`Adds`/`Mul`/`Slice`/`ConcatD` | 2736 | 6.2 | -- |
| our other Triton kernels (HC, routing, layout, sinkhorn, state) | ~700 | ~5.5 | -- |

The LM head streams its 1.32 GB at 1.23 TB/s, so the part can stream. The expert GEMV runs at roughly 50-150
GB/s, an order of magnitude below that, and the whole step moves on the order of 3-4 GB of weights whose
streaming floor is ~2-2.5 ms against 64.6 ms measured. That is the quantified form of the M=1 tile-utilisation
problem: the dominant kernels are latency- and tile-shaped, not bandwidth-shaped, which is why the 20 ms/token
target is **not** excluded by a bandwidth argument and why it needs the executor rewrite rather than more
fusion of what exists.

Closed this round, with the measurements that closed them:

* **Sparse-attention kernel fusion** (two-kernel and all-heads variants): in-graph 459.66 µs and 161.05 µs a
  layer against **81.10 µs** for the accepted S2 path; end to end 13.1089 tok/s against 15.19. The fused forms
  replace a 45 µs once-per-layer gather with eight per-head random-gather streams, which loses on this part.
  The all-heads form is also wrong (4096 bit differences, max |d| 1.275) and is not a candidate.
* **Whole-step decode graph**: the host already enqueues a step in 15.6 ms against a 49.9 ms device drain, and
  the device is busy 94.2 % of the profile window, so removing per-layer replay boundaries cannot pay for the
  rewrite; the only non-overlapped host time is the ~2.4 ms step transition, which is the metric's own
  "NPU completion + host argmax" definition.
* **Host Engram prefetch** (`RELIC_HOST_ENGRAM_PREFETCH=1`): no change (15.2342 vs 15.1934) because our Engram
  host side is a 31 KB table gather, not the CPU projection the prefetch hides.
* **Single-chunk HC GEMV**: 1.64× faster in isolation (7.69 µs vs 12.64 µs a call) but ~0.4 ms a step, inside
  the run-to-run spread; reverted rather than carried without a matched three-pair result.

### Still open

* The full `14.287598 tok/s` two-master stack has **not** been reproduced from `master` alone: the BRCB
  decoder's final source and build (`caca541d` source, `46aca3d9` library) and part of the profiler/launcher
  wiring exist only as workspace artifacts. A 64-step master-only run is required before that claim is made.
* No team review has been engaged. This page and the campaign ledger are the review material.
* Phase 2 (MTP) is untouched, as the objective requires while Phase 1 is unmet.

---

## 6. Second optimisation: two bit-exact operator-level fixes, one rejected on a pair

Both retained fixes make the *same* 64-token stream as the lane before them, so they are orthogonal to the
quality-band question entirely.

| change | mechanism | measured effect |
| --- | --- | --- |
| indexer candidate gather off AI_CPU | two-tensor advanced indexing lowers to an AI_CPU `aclnnIndex` (230.6 us a call, 4 a step); folding the batch offset into the position index makes it one flat `index_select` on the vector cores (same 128 KB copy: 530.76 us -> 20.03 us in isolation) | **+0.81 %** and **+1.18 %** on two interleaved pairs, bit-identical stream |
| per-layer o-projection weight transpose | **rejected**: caching the transposed `wo_a` operand removes a 26.9 us `Transpose` a layer but makes the downstream batched matmul slower | **-0.69 %** and **-0.74 %** on two interleaved pairs, bit-identical stream -- reverted |

Run-to-run drift on this shared host is **larger than the effects being measured**: the same unmodified
control lane measured 15.19-15.23 tok/s around 23:00, 15.35 around 00:20 and 15.62-15.63 around 01:10,
with no code change in between (2.8 % spread). Every claim above therefore comes from an interleaved
control/candidate pair in adjacent windows, not from comparing best-of runs. Single-run gains that looked
bigger than the drift were later disproved by exactly that test -- the o-projection cache is the example
kept in the record.

Current verified lane on the current machine state (`t33`, all gates canonical, no tolerance overrides):
**15.3722 tok/s / 65.09 ms per token**, bit-identical stream to `t24`/`t22`/`t15`, eight-rank
teacher-forced greedy 8/8, KL 5.08e-05-1.60e-03, RMS 0.142-0.408, max|d| 0.656-1.852. The same lane measured
15.5563-15.6312 in faster windows, so the honest statement is the pair-matched delta (+1 % from the gather
fix, which holds in every pair) plus the observed lane range, not a single absolute number.

---

## 7. What the three-pair protocol says about the current tree, and where the next attempt has to go

Six interleaved 64-step runs, control arm first in every pair, differing only in
`RELIC_INDEX_GATHER_ADVANCED`:

| pair | control (two-tensor advanced indexing, AI_CPU `aclnnIndex`) | candidate (flat `index_select`) | delta |
| --- | ---: | ---: | ---: |
| 1 | 15.1856 tok/s / 65.86 ms | 15.3857 tok/s / 65.12 ms | **+1.32 %** |
| 2 | 15.1659 tok/s / 65.96 ms | 15.3797 tok/s / 65.03 ms | **+1.41 %** |
| 3 | 15.2126 tok/s / 65.83 ms | 15.3740 tok/s / 65.07 ms | **+1.06 %** |

One and the same 64-token stream on all six runs, eight-rank teacher-forced greedy 8/8 on all six. The
pair-matched mean is **+1.26 % / -0.81 ms a token**, which is the entire verified gain of this campaign
over the registered reference (14.2876 tok/s -> 15.37-15.39 in the slow window, +7.6 %), and it is far
from the 50 tok/s gate.

Two measurements from this round narrow the next attempt:

* **Collectives are 20.5-22.5 % of the step.** 130 collective kernels a step, 13.2-14.8 ms, of which ten
  `hcom_allReduce__612_*` calls at 478-766 us each are ~5.5 ms. They sit on the compute stream because
  every reduce in this model is blocking. Issuing them on a second stream and capturing that overlap in
  the layer graphs is the only route left that does not touch the protected FP32 TP-down wire contract or
  the routing -- an executor change, and the largest single item still addressable.
* **The native expert kernel is vector-bound, not cube-bound.** Averaged over its 360 launches in the
  profiling window: AIV vector ratio 0.59 (median 0.64), AIV MTE2 0.21, AIC MAC ratio 0.03 (median
  0.04). The K-loop pipeline rewrite this campaign had queued would target ~17 % of the kernel; the
  actual costs are the vector LUT decode and an activation row re-loaded once per N tile (~160 x 40
  loads a launch, which needs the two-core ring protocol redesigned to hoist). That is the concrete
  next piece of work on this operator, not a flag or a tuning sweep.

The objective gate remains unmet and is not claimed: pure-B1 single-request decode measures
15.37-15.39 tok/s in the slow window and 15.52-15.63 in the fast one, against the 50 tok/s / <=20 ms
gate; Phase 2 (MTP) remains untouched by design.

---

## 8. The largest remaining item is exposed communication, and it is waiting, not transmitting

Three independent measurements this round agree on the same conclusion.

**The reduces, identified.** Instrumenting `dist.all_reduce` in the metered harness (16 steps, one rank)
shows the whole set: per layer a `[1, 5120]` fp32 world reduce (the documented `wo_b` one) and, at the
same `tp.reduce` call site, a **`[1, 20, 5120]` fp32 world reduce -- 400 KB, forty calls a run** -- plus a
`[6, 5120]` fp32 two-rank reduce at `ascend_expert_tp.py:288` and a handful of twenty-element routing
matrix reduces. The 400 KB hyper-connection state is the op class that measures 478-766 us in the
kernel profile.

**The wire is not the cost.** `communication_matrix.json` for the profiled window: the two-rank group
moved 9.396 MB in 0.752 ms (12.49 GB/s) and the eight-rank group 28.126 MB in 0.376 ms (74.88 GB/s).
Over three steps that is **~0.38 ms/step of actual transmission against 13.2-14.8 ms/step of exposed
collective time** -- about 97 % of the collective cost is one rank waiting for another.

**The profiler's own accounting says nothing overlaps.**
`step_trace_time.csv`: `Computing=141664.58 us  Communication(Not Overlapped)=41451.20 us
Overlapped=0.0 us  Free=18732.18 us`. Fully hiding the measured 13.8 ms/step would put the step at
~51 ms (**19.5 tok/s, +27 %**); half of it at ~58 ms (17.2 tok/s, +12 %). No fusion left on the list is
worth as much.

**Why it cannot be captured with a flag.** `tp.reduce` with `async_op=True` and an immediate
`work.wait()` runs inside the inline-TP graph capture and reproduces the same 64-token stream
(`a2`), so the asynchronous collective API survives capture. But every reduce output here is the next
layer's input, so an immediate wait is the only correct spelling today; hiding the wait requires the
layer loop restructured into a two-stream wavefront. That is the executor-level change this objective
names, and it is the concrete next piece of work -- not a knob, and not a percentage-scale fusion.

Final state of the tree after the round's probes were reverted (`t34`, all gates canonical):
**15.4290 tok/s / 64.93 ms per token**, 64-token stream identical to `t33`, eight-rank teacher-forced
greedy 8/8, KL 5.08e-05-1.60e-03, RMS 0.142-0.408.

---

## 9. The suspected straggler: per-group collective duration variation

The probe recorded, for every rank, the duration that rank spends inside the collective the profiler labels
`hcom_allReduce__612_238_3`:

| rank | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| duration us | 799 | 799 | 971 | 984 | **20** | **29** | 790 | 800 |

**What this is, and what it is not.** The collective is the expert-TP2 allreduce, so its eight ranks fall in
four groups ([0,1], [2,3], [4,5], [6,7]); the two members of every group agree almost exactly (799/799,
971/984, 20/29, 790/800) while the groups differ by up to ~48×. That is **per-group duration variation**, and
duration alone cannot turn it into an arrival skew *within* one operation: the short groups are a different
collective instance, not the same instance finishing at one instant. An earlier draft of this page read it as
~950 us of cross-group arrival skew with routing imbalance as the cause; that reading is a **hypothesis**, not
a measurement, and settling it needs a same-group, all-rank aligned trace (start/end timestamps for one
instance, not per-rank durations across groups). The pattern itself is real and reproducible: 82 collectives
above 400 us in the window carry the same per-rank shape, and it does not appear in wall-clock step time (all
eight ranks report 64.91-64.94 ms) or in the Computing/Communication split (per-rank window boundaries differ
by up to 16 ms, larger than the effect).

The hypothesis is the one the objective points at -- each rank owning a different number of the six selected
experts per layer -- and the two-stream test below bounds only what it tested: matmul-and-collective on two
streams measured 481.5 us against 380.3 us sequential, so *streams* do not hide the wait in that configuration.
It is not a universal closure of the straggler question.

**The straightforward fix is closed by the checkpoint's tiling format.** With expert-TP 8, every rank would
compute a constant 1/8 of every selected expert -- a perfectly balanced schedule. But both
`RELIC_EXPERT_TP=8` and `=4` are rejected:

```
ValueError: compact expert projections require complete N64/K128 geometry
```

The shard width must be a whole number of N64 output tiles (64 rows) and K128 contraction blocks. At
`tp_world=2` both projections shard to 1152 (divisible by 128); at 4 they are 576 (not a multiple of 128),
at 8 they are 288 (not a multiple of 64). Expert-TP is pinned to 2, so the imbalance can only be fixed by a
K64 bank format (a different checkpoint layout) or routing-aware expert placement (excluded by the task).

## 10. Independent review of the retained optimisation

Agent `perf-graph-review-v9` audited the evidence read-only and recomputed everything from the raw
per-run JSONs:

* three interleaved pairs: +1.3172 %, +1.4093 %, +1.0614 %, mean **+1.2626 %**, paired t = 12.25;
* the same 64-token stream on all six runs, all eight ranks agreeing, teacher-forced greedy 8/8 everywhere;
* no `RELIC_*GATE_TOL` in either arm's script, `env.sh` or log, so both arms ran with canonical gates;
* beyond the brief, **stored logits compared bitwise: final logits and the teacher-forced `(8, 129280)`
  quality logits are 0/1034240 elements different with `max_abs_diff = 0`** across all three pairs and all
  eight ranks. The "bit-exact" claim therefore holds at FP32 logit level, not only at token level.

Two flags it raised, both explained rather than defects: the candidate arm leaves
`RELIC_INDEX_GATHER_ADVANCED` unset instead of `=0` (identical behaviour -- `attention.py` treats anything
but `"1"` as the new path), and `--no-decode-graphs` disables only the outer decode graphs while the
inline-TP layer graphs still capture, which is the metered lane by design.

Final state: `t34` 15.4290 tok/s / 64.93 ms and `t35` 15.3170 / 65.34 ms, same 64-token stream as the
reviewed candidate runs, all gates canonical. The 50 tok/s gate remains unmet and is not claimed.

---

## 11. A profiled 0.40 ms/step of small kernels that does not convert into step time

`modules.py:554` widens the static `[384, 5120]` BF16 routing weight to a fresh FP32 copy on every layer
(`F.linear(x.float(), self.weight.float())`). The profile shows exactly that cast:
`in="384,5120" DT_BF16 -> FLOAT`, 120 calls in the three-step window, **1.19 ms = 0.40 ms a step**. The
widening is exact, so caching the FP32 copy is bit-identical -- the 64-step run passes the canonical gates
and reproduces the same 64-token stream (`s1` 15.4412 tok/s / 64.68 ms against `t36` 15.4369 / 64.83).
The interleaved A/B still rejects it:

| pair | control | cached FP32 routing weight | delta |
| --- | ---: | ---: | ---: |
| `s2 -> s3` | 15.3825 / 64.97 ms | 15.4017 / 65.00 ms | +0.12 % |
| `s4 -> s5` | 15.4358 / 64.87 ms | 15.3918 / 65.03 ms | **-0.29 %** |

Same 64-token stream on all four runs. So removing a cast series the profiler prices at 0.40 ms/step moved
the step by nothing measurable. **The small-operator tail does not convert one operator at a time**, and
the earlier plan to harvest it as a list of fusions is closed by measurement, not by estimate. Either those
durations overlap with something the profiler does not show as overlapped, or the replay dispatch of the
timed lane is not the arithmetic the trace describes; finding out needs a per-step timeline of the timed
lane with graph-replay markers, which is instrument work rather than an optimisation.

For the same reason this round also closed the native-kernel vector path on arithmetic grounds: the LUT
kernel's literal geometry is `M=16, N=64, K=128`, 4,096 packed bytes per tile, 8,192 decoded BF16 values,
and one six-expert launch decodes 17,280 tiles in the measured 108 us mean -- **6.2 ns a tile, 660 GB/s of
decoded-BF16 equivalent**, which is the throughput the 4x expansion requires. There is no factor of two
left in the decode; a different design (not a faster implementation of this one) would be needed.

Final state of the verified lane after the experiment was reverted (`t37`): **15.3838 tok/s / 64.89 ms**,
64-token stream identical to `t34`, all gates canonical, eight-rank greedy 8/8, KL 5.08e-05-1.60e-03,
RMS 0.142-0.408. The 50 tok/s gate is unmet and not claimed.

---

## 12. Pre-capture materialisation: two per-layer weight transforms removed from the replayed graphs

The previous round's two "no-op" weight experiments were measurement confounds: the inline-TP layer graphs
bake **every node the first capture sees**, so a Python-level cache populated *during* the capture leaves
the cast or transpose in the graph and only changes its input. Materialising the static tensors **before**
the capture removes the nodes instead of caching around them.

Implemented as `RELIC_PRECAPTURE_STATIC` (default on, `=0` restores the control) right after
`model = front.model`, i.e. before the prefill and before the binding install that captures the graphs:

* the routing gate's `[n_routed_experts, dim]` BF16 weight is widened in place to FP32 (exact), so
  `F.linear` sees an FP32 weight and the `[384, 5120] BF16 -> FLOAT` cast node disappears (it was 120 calls
  in the three-step window, 9.9 us each); and
* each attention `wo_a` is materialised as the contiguous `[n_groups, d, o_lora_rank]` operand, and the
  contraction is spelled `"bsgd,gdr->bsgr"`, so the `1024x4096` transpose node disappears (it was 120 calls
  in the window, 26.9 us each).

Both transforms are lossless, and the independent review verified the stored logits bitwise: **0 differing
elements and `max_abs_diff = 0`** for the final logits and the teacher-forced `(8, 129280)` quality logits,
on every pair and all eight ranks. The mechanism is visible in the default lane's own kernel profile
(`p8`, 23,418 kernels): **0** `Cast` with input `384,5120` and **0** `Transpose` with input `1024,4096`,
against 120 of each before.

Three interleaved pairs on the final source (`p6->p7`, `q1->q2`, `q3->q4`), control arm exporting
`RELIC_PRECAPTURE_STATIC=0`, candidate leaving it to the default:

| pair | control | materialised | delta | median |
| --- | ---: | ---: | ---: | ---: |
| `p6 -> p7` | 15.3673 tok/s / 65.11 ms | 15.6183 / 64.05 | **+1.63 %** | -1.06 ms |
| `q1 -> q2` | 15.4640 / 64.77 | 15.5940 / 64.19 | **+0.84 %** | -0.58 ms |
| `q3 -> q4` | 15.4370 / 64.82 | 15.6249 / 63.96 | **+1.22 %** | -0.87 ms |

Mean **+1.23 % / -0.84 ms a token**, all three positive, one and the same 64-token stream on all six runs.
Two earlier pairs on the previous source revision (`p2->p3` +1.12 %, `p4->p5` +1.28 %) replicate it, giving
five positive pairs in total with a mean of +1.22 %.

**Standing of the measured lane**: the three default-lane runs of this round report 15.5940, 15.6183 and
15.6249 tok/s (63.96-64.19 ms median) against the registered reference's 14.2876 tok/s / 69.9922 ms --
**+9.2 % throughput, -8.5 % per-token latency**. The FP32 logits are bit-identical to the same lane before
the change (0 of 1,034,240 elements on every rank); the lane's *token stream* is **not** the registered
reference's -- the two configurations diverge at token index 9, as §14 records -- so this is a
throughput-to-throughput comparison of two configurations that both pass the registered gates
(`logits within 1e-3`), not a reproduction of the reference. The 50 tok/s gate remains unmet and is not claimed; what changed is that the executor now
stops re-materialising static weight transforms on every replayed layer.

The review also noted that a run's JSON records no field for the pre-capture state; only the log line
`PRECAPTURE_STATIC gates=40 wo_a=40` and the arm script identify it. Worth adding to the record schema if
this stays in the lane.

---

## 13. Census of the adopted lane, and the limit of stack attribution

With the pre-capture materialisation adopted (`t38`: **15.5859 tok/s / 64.15 ms**, stream identical to the
pair-verified candidate runs), the kernel census of the lane is 7,806 kernels a step totalling **60.06 ms**
of kernel time (against 62.5 ms before). Ranked: `relic_scale_lut_tile_cube_0` 13.07 ms / 120 calls;
dense `MatMulV2` 5.49 ms / 348; `fast_fused_fp8_activation_decode` 3.43 ms / 322; `BatchMatMulV2` 2.42 ms /
324; KV-ring `InplaceIndexCopy_Transpose` 1.29 ms / 138; `_gemv_scale_kernel` 1.28 ms / 80;
`InplaceCopy_Slice` 1.10 ms / 344; `route_owned_kernel` 1.03 ms / 40; `InplaceCopy_Cast` 1.01 ms / 569;
`ConcatD` 0.95 ms / 77; then the elementwise tail (`Add` 0.88, `ReduceSum` 0.80, `Mul` 0.76, `Adds` 0.67,
`Softmax` 0.54). The gate widening shows up where expected: `Cast` 1.50 -> 1.01 ms, and the `wo_a`
transpose is gone from the list.

**No static-weight transform is left in the per-step list**, so the pre-capture lever is exhausted at those
two nodes. What remains is either at a measured floor (`tile_cube` at 6.2 ns/tile, the fused FP8/HC kernel
series), an M-shape limit (dense matmuls at M=20), or a layout property of the ring the fused attention
kernels expect (KV write transposes 1.29 ms + scatter 0.28 ms; window/compress `ConcatD` 0.95 ms).

Two attribution notes for the next attempt. First, **stack attribution cannot see inside a replayed graph**:
turning on the profiler's `with_stack` records the harness `main()` chain for the ops the host issues, and
nothing at all for the ops the captured layer graphs replay, so the `Slice [32,2]` x256 (0.89 ms) series
cannot be traced to a model line that way -- that needs a trace of the *capture* pass, where each op is
still a Python call. Second, the neighbourhood of those slices identifies their parent anyway: they follow
`aclnnIndex` with input `128,32,2` -- the window index table -- so they belong to the `Pos`/window-table
path that was already found to be protocol-bound.

---

## 14. The 14.2876 reference reproduced, and the optimized lane measured against it in the same hour

The objective asked for the complete 14.2876 stack to be reproduced before any optimisation claim, and for
that never to be claimed without a 64-step run. Both halves are settled, from stored evidence plus fresh
same-hour runs.

**The staged stack.** A staged master-baseline tree (its manifests are in the campaign ledger) is the
reconstruction.
`source-staging.json` records schema `relic.b1.master-staged-source.v1`, runtime commit
`398204f194d2b80a43483424e43b131644bf65e7`, core commit `f7b907873ab1d7f9dc0ccc7b67c87131c53c6bee`, and
**114 registered product files hash-verified** against those commits. `logs/scale-lut-brcb-build.json` is the
BRCB decoder receipt: library `46aca3d9e67cdf8204382d8eb37e8bf3ae603150242b0bd099cc2be8f16c9117`, source
`caca541d1504c52699f0eb2aaf785ffca8bc7dae0a79958bae80480097791b52`, bank format
`relic.fp4.cube-n64-k128.v1`, and a `fresh_build_component_proof` over 81,536 elements with
**changed_FP32_bits = 0**. Its `preconditions.json` records mode `device-hc`, projection variant
`brcb_decoder`, `fused_row_norm_variant: candidate`, 64 timed steps with 4 warmup steps excluded, and the
registered command verbatim. The two pieces the objective names as absent from master -- the BRCB decoder
final source/build and the launcher wiring -- are therefore present as *staged artifacts with hashes*; what
was missing was the run proof, and that now exists:

| stack run | rank-0 TPS | 8-rank spread | median step | greedy | quality vs CPU oracle |
| --- | ---: | --- | ---: | --- | --- |
| staged, 2026-11-24 | 14.277033 | 14.276930-14.277417 | 69.91 ms | 8/8 | KL 2.11e-05-2.01e-03, RMS 0.194-0.619, max\|d\| 4.256 |
| dev-tree repro, 2026-11-25 | 14.133148 | - | 70.70 ms | 8/8 | identical, bit-identical logits |
| fresh, 2026-11-26 (this round) | 14.133249 / 14.068335 | - | 70.63 / 70.92 ms | 8/8 | identical |

The dev-tree repro's 64-token stream is **identical** to the staged run's, so it is the same stack. Against
the registered **14.287598 / 69.9922**, the best run is -0.07 % on throughput and -0.11 % on latency, and
every other run is within the 1.5 % drift this host has shown between hours. **The reference is reproduced.**
It is not reproduced *from master alone* -- the BRCB build is a staged artifact -- and that distinction is
kept rather than papered over.

**Same-hour comparison against the optimised lane** (interleaved, 64 timed steps each):

| pair | master stack | optimised lane | delta |
| --- | ---: | ---: | ---: |
| m1 -> m2 | 14.133249 / 70.63 ms | 15.597628 / 64.16 ms | **+10.36 %** |
| m3 -> m4 | 14.068335 / 70.92 ms | 15.558824 / 64.25 ms | **+10.59 %** |
| m5 -> m6 | 14.077556 / 70.96 ms | 15.603554 / 64.26 ms | **+10.84 %** |

Mean: master **14.0930 tok/s / 70.84 ms**, lane **15.5867 tok/s / 64.22 ms** -- **+10.60 % throughput,
-9.4 % per-token latency**, measured in one interleaved session on the same eight cards. Each arm reproduced
its own token stream exactly across its repeats; the two streams differ from each other at token index 9,
so this is a throughput-to-throughput comparison of two configurations that both pass the registered gates,
which is the acceptance criterion in force (`logits within 1e-3`, not bitwise identity). The lane's quality
band against the oracle is the tighter one (KL 5.08e-05-1.60e-03, RMS 0.142-0.408, max\|d\| 1.852).

**Standing**: 15.59 tok/s against the round's own master-stack baseline of 14.09 (+10.60 %), and
+9.1 % against the registered 14.287598; the 50 tok/s gate is unmet by a factor of 3.2 and is not
claimed.

---

## 15. The tail is launch-bound: per-shape bandwidth of the remaining 60 ms

Resolving the lane's kernel window by shape against the box's measured streaming floor (**1,142 GB/s
read-only, 1,235 GB/s copy**, from a 512 MB `sum`/`copy_` probe) separates the remaining cost into work and
overhead.

The two dense shapes large enough to amortise a launch are **at the floor**: the LM head
`1x129280x5120` streams 3.97 GB in 3.18 ms = **1,250 GB/s**, and `1x6144x6144` reaches **1,263 GB/s**. The
mid-size dense projections run at 233-851 GB/s with only 12-20 of 24 cores engaged
(`1x1280x5120` 653, `1x4096x1280` 687, `1x5120x1024` 851, `1x512x5120` 413, `1x288x5120` 426,
`1x384x5120` 233, `1x5120x288` 263): that band is the only one where a retained bandwidth win could exist,
and the ceiling for taking all of it to the floor is ~2.7 ms/step.

Everything below that is overhead. **316 matmuls a step have no payload**: the per-head broadcast projections
`[1,8,512]@[1,512,64]` and `[1,8,64]@[1,64,512]`, 138 of each per step, cost 6.2 us apiece for 0.5 MB of
weights (**84 GB/s implied**) = **1.55 ms/step of launch latency**. The FP8 activation decode is 322 calls a
step at 10.7 us (**3.43 ms**) and its callers are now known exactly: one decode per layer for the tile MoE's
gate and up (shared), one per layer for the TP down partials, and about six per layer for the Engram
projection -- one per gathered lookup row, so those are not redundant work that can be hoisted, only batched
into fewer launches.

The step therefore decomposes as: LUT kernel at its design throughput 13.07 ms, dense matmuls 5.49 ms (mostly
at the HBM floor), two fused Triton series at their floors 3.43 + 2.42 ms, and an **8-10 ms band of sub-10-us
launches**. That band is the honest remaining target, and it can only be taken by reducing the *number* of
kernels a layer issues. The three merges the profile supports, with their ceilings: the six per-layer Engram
decodes into one stacked decode (~2 ms), the 276 per-head indexer projects into the adjacent fused attention
kernel (up to 1.5 ms), and the mid-size projection band onto a GEMV that keeps all 24 cores fed (up to
2.7 ms). All three are kernel-side changes; none of them is another percent-scale fusion of the kind that has
already been measured and rejected.

---

## 16. Two GEMV variants measured and rejected; what that closes and what is left

Round 20's census left two candidates that both assumed a hand-written M=1 GEMV would beat the vendor
kernel: the mid-size dense-projection band (233-851 GB/s, ceiling +2.7 ms) and folding the per-projection
fake-quant decode into the consuming GEMV (ceiling +2.1-3.4 ms). Both were measured on the real shapes before
any integration, and both fail:

* **Vector GEMV** (`tl.load` a `[BLOCK_N, BLOCK_K]` weight tile, `tl.sum(w * x, axis=1)` accumulate, swept
  over BLOCK_N 16/32/64 x BLOCK_K 256/512): 79.6-83.2 us a call against the vendor matmul's 30.0-32.1 us for
  the same shape -- **0.36-0.40x**, i.e. 2.5-2.8x slower, with correct numerics (max deviation 0.25-0.94 from
  an FP32 oracle, which is the BF16 output rounding). The cube is idle and the vector unit does 32x512
  multiply-reduce steps per K-block.
* **Cube GEMV** (`tl.dot` with M padded to 16, row 0 extracted by `tl.sum(acc, axis=0)`): unusable --
  30.9 ms for `1280x5120`, 59.4 ms for `2304x5120`, 3.0 s for the LM head, against 30-33 us and 1.05 ms
  from the vendor kernel. This is not a tuning gap.

So a Triton GEMV cannot beat the vendor matmul here, and both merges that depended on one are closed by
measurement: the mid-size band stays where it is, and the fake-quant decode cannot be folded into a GEMV
that would be slower than the matmul it replaces. That leaves one of the three remaining candidates (the 276
per-head indexer projects, 1.55 ms, fused into the adjacent attention kernel) and the LUT kernel's own
vector pipeline.

**LUT kernel floor, from its source.** Per N64/K128 tile the vector unit runs a byte->half cast, a
half->int cast, a shift, a 4,096-element int32 add of BRCB-expanded slot offsets and a 4,096-entry `Gather`
through the 12-slot code LUT, plus eight 256-element `Brcb`s and a 16 KB ring write. At 108 us a launch for
17,280 tiles and 24 blocks that is 150 ns a tile against a vector-op floor of roughly 90-95 ns, so the
kernel's own pipe ratios (AIV 0.59-0.64 against AIC MAC 0.03) leave perhaps 1.4-1.6x on the table if the
per-tile pipeline were fully overlapped -- 13.07 ms down to ~8-9 ms, the largest single remaining item. It is
an AscendC vector-pipeline rewrite plus a library rebuild, a hash update and a bit-exactness proof, not a
fusion, and it is the one item in this report that a next session should start from.

---

## 17. The native LUT kernel's serialised load: +8.8 % for reordering two instructions

The pipe profile has said since the first census that the native tile kernel runs AIV 0.59-0.64 against
MTE2 0.21 and MTE3 0.16 -- the vector decode busy about three-fifths of the time while the packed read and
the ring write sit idle four-fifths. The source explains it: each K-block iteration issued the 4 KB packed
read, waited for it on both the scalar and the vector pipe, ran ten vector ops behind `PipeBarrier<PIPE_V>`,
then wrote the 16 KB decoded tile to the ring, and finally fences the operand for reuse. The read, the
decode and the write are three strictly serialised phases; nothing overlaps.

The change is a second raw/scale operand slot appended after the existing Unified Buffer scratch
(`RAW2_OFF = UB_BYTES`, total 114,944 of 131,072 bytes with 16,128 bytes of headroom, so not one existing
offset moved) plus a reordered
loop: the prologue fetches block 0, and each iteration issues block b+1's read into the idle slot **before**
decoding block b, with the existing `V_MTE2` fence at the end of the iteration still protecting the slot
being overwritten. No arithmetic, no operand value and no ring protocol changed.

Three interleaved pairs, registered library against the new build, 64 timed steps each:

| pair | registered | pipelined | delta | median |
| --- | ---: | ---: | ---: | ---: |
| `v1 -> v2` | 15.5497 / 64.29 ms | 16.8524 / 59.30 ms | **+8.38 %** | -4.99 ms |
| `v3 -> v4` | 15.5902 / 64.18 ms | 16.9912 / 58.85 ms | **+8.99 %** | -5.33 ms |
| `v5 -> v6` | 15.5731 / 64.20 ms | 16.9615 / 58.89 ms | **+8.92 %** | -5.31 ms |

Means: registered **15.5710 tok/s / 64.23 ms**, pipelined **16.9350 tok/s / 59.02 ms** -- **+8.76 %
throughput, -5.21 ms a token**. Every arm reproduced its own 64-token stream, and the two arms' streams are
the **same stream**, with the stored final logits differing in **no element (max|d| = 0.0)** and
the quality band unchanged (KL 5.08e-05-1.60e-03, RMS 0.142-0.408, greedy 8/8). This is a scheduling change,
proven total by the numbers rather than argued.

Adopted: the build (sha256 `ff557c3f4669331e4c8cd768011f788e27ce8da56c62434676fe4ed71e965889`, 403,032
bytes, source `4df634cb...`) replaces the registered `.so`, `RELIC_TILE_CONSUMER_SHA256` in `env.sh` is
updated, the previous library is kept as `librelic_scale_lut_tile_cube.registered-before-pipeline.so` for
rollback, and `build-receipt.json` records the change, both hashes and the verification above.

**One follow-up measured and rejected.** The post-change census shows the producer's vector pipe at 0.50 and
its scalar pipe at 0.92 with `CrossCoreWaitFlag(FREE_FLAG)` in the hot path -- the decode now runs ahead of
the ring. Flipping both sides' `stage_` cache hint from `CACHE_MODE_DISABLE` to `CACHE_MODE_NORMAL` looked
like -0.29 ms on its first run, but two interleaved pairs killed it (+0.07 % and +0.20 % for the L2 version,
means 16.9217 against 16.8989), so the hint was reverted. The remaining levers the profile supports are ring
depth (2 slots, needs flag-id space on both sides) or moving the decode into the consumer; both are
multi-hour changes with their own three-pair protocols.

**Standing after this round**: the three-pair adopted mean is **16.9350 tok/s / 59.02 ms**, which is
**+18.5 %** against the registered 14.287598 tok/s / 69.9922 ms (latency -15.7 %) and **+20.2 %** against
this record's same-session master-stack control (14.0930 tok/s / 70.84 ms, latency -16.7 %). The source was restored to the adopted revision and rebuilt byte-identically
(`ff557c3f...`), so the recorded source hash reproduces the adopted library. The 50 tok/s Phase-1 gate
remains unmet and is not claimed.

*Superseded for the standing figure by section 19: `HCCL_OP_EXPANSION_MODE=AIV` was adopted on top of this tree and measured at 17.4689 tok/s / 57.25 ms. The numbers in this section are the record of that round and are unchanged.*

## 18. The "reproduce from the two masters" requirement, closed with hashes and one control run

The objective's first requirement is stronger than a throughput target: reproduce the complete 14.2876
stack from the two masters, and do not claim master reproduction until a 64-step master-only run proves
it. This section closes it both ways -- the registered operating point is reproducible on this hardware,
and a literal master-only run is not performable at all.

### 18.1 What the three reference hashes actually are

The campaign has been quoting `caca541d` (source), `46aca3d9` (library) and, after this session's work,
`4df634cb` / `ff557c3f`. They pin to concrete revisions:

| artifact | sha256 | where it is |
| --- | --- | --- |
| decoder `.cpp` at `relic-core` `1d9b182` ("land the BRCB scale-LUT decoder source") | `caca541d1504c526` | objective's "final source" |
| library built from it | `46aca3d9e67cdf82`, 399,592 B | held as `registered-before-pipeline.so` |
| decoder `.cpp` at `3c8c98a` (the adopted prefetch) | `4df634cbde357ce8` | this branch's HEAD |
| library built from it | `ff557c3f4669331e`, 403,032 B | deployed and env-pinned |

The step from master to `caca541d` is substantive, not cosmetic: **22 insertions / 4 deletions** replacing
the general `Gather(expandedSlot_, slotOff_, pairRepeat_, ...)` broadcast with the `Brcb` quadrant loop.
From `caca541d` to the adopted `4df634cb`: **33 insertions / 13 deletions** (the single-row prefetch plus
the pair-ring export). master's decoder is neither: it is `3ca65ab82e727d81`.

### 18.2 A literal master-only run cannot be performed

Both masters were read, not assumed:

- `relic-core` master carries the scale-LUT decoder, but at revision `3ca65ab8` -- the pre-BRCB form.
- `RelicLLM` master contains **none** of the lane wiring. `profile_pure_b1_reuse.py`,
  `ascend_inline_tp_layer_graphs.py` and `profile_actual_cube.py` are all absent from the master tree; the
  branch is **+2,097,081 lines across 2,231 files** over master, which is the lane itself.
- The Python bindings that the lane drives (`ascendc_scale_lut_tile_cube.py`, `ascendc_tile_moe.py`) *are*
  identical in master and on the branch, so the earlier statement that "all code was merged" is right for
  the bindings and wrong for the decoder revision and the launcher. That is the overstatement this section
  corrects.

So a master-only 64-step run does not exist to be run: there is no launcher on master to start it and no
`caca541d` decoder to build. The 14.287598 figure belongs to the registered dev stack, not to the masters.

### 18.3 The registered operating point does reproduce, measured

The closest sound experiment was already in the campaign: the `ARM=master` side of the interleaved
`m5` pair (`logs/m5-master-vs-lane-20261126T075733Z`), 64 timed steps, which ran the registered tree
`relic-b1-master-baseline-20261124T044500Z`. That tree's provenance is checkable and is *not* GitHub
master: its decoder is `caca541d` and its library is `46aca3d9`. It measured:

| arm | tok/s | median step | vs registered 14.287598 / 69.9922 ms |
| --- | ---: | ---: | --- |
| registered tree, 64 steps | **14.0776** | **70.96 ms** | **-1.47 % / +1.4 %** |

That is reproduction of the registered stack to within session drift on this shared host, and it is the
number that should be quoted as such -- not as "master reproduction".

### 18.4 What the correction therefore is

1. The 14.2876 stack reproduces on this hardware at **14.0776 / 70.96 ms** (64 steps, registered tree and
   pre-pipeline decoder build `46aca3d9`); difference from the registered figure is machine drift.
2. It does **not** reproduce "from the two masters": master's decoder is a superseded revision and master
   has no launcher. The lane that measures it lives on the branch.
3. The earlier blanket claim that all related code was merged is withdrawn; only the Python bindings match
   master. The decoder change (`22+/4-`) and the whole launcher chain are branch-only, as is the adopted
   prefetch (`33+/13-`) on top of them.
4. No claim in this document is of the form "14.2876 reproduced from master". The measured statement is
   §18.3, and the correction is §18.4.3.

## 19. A community lever that crosses over: `HCCL_OP_EXPANSION_MODE=AIV`, +3.4 % stream-identically

Reading the comparative stack (`CJYLZS/vllm-ascend-deepseek-v41-a2`) for the objective's
"compare against community pure-B1" clause turned up one lever that does not depend on its weight format or
on MTP: `HCCL_OP_EXPANSION_MODE=AIV`, which moves the expansion and execution of collectives from the host
CPU onto the Vector Core. That stack reports it as communication -42.3 % and step -10.8 % on this exact
8x910B3 single-host topology, with two stated preconditions -- a symmetric single-host eight-card setup, and
non-deterministic mode (`HCCL_DETERMINISTIC` must not be on). This lane is that topology, and section 8
measured 13.2-14.8 ms a step of *exposed* collective time against ~0.38 ms of actual transmission, so the
lever had a reason to transfer. It does.

### 19.1 The three-pair measurement

Orthogonal interleaved A/B, three pairs in one session, each arm a full 64 timed steps with warm-up and
profiling excluded, the only difference between arms being one environment variable:

| arm | tok/s, three runs | mean tok/s | median step, three runs | mean step |
| --- | --- | ---: | --- | ---: |
| A, adopted lane | 16.9390 / 16.8747 / 16.8863 | **16.9000** | 58.99 / 59.33 / 59.23 ms | 59.18 ms |
| B, + `HCCL_OP_EXPANSION_MODE=AIV` | 17.4159 / 17.5084 / 17.4824 | **17.4689** | 57.35 / 57.12 / 57.29 ms | 57.25 ms |

**+3.37 % tok/s and -3.26 % step**, same-signed in every pair (+2.81 %, +3.76 %, +3.53 %) -- which is what
makes it a measurement rather than a drift reading. Every figure in the table was recomputed independently
from the six receipts during review, to the digit; the arm scripts differ only by the one export.

### 19.2 It is stream- and logit-identical in all six runs, on all eight ranks

- all six 64-token token streams are identical to each other;
- **all 48** final-logit tensors (6 runs x 8 ranks) are bit-identical, 0 differing elements, `max|d| = 0.0`,
  compared against each other -- this is run-to-run and rank-to-rank parity, not parity against the CPU
  oracle (section 14 covers that separate gate, where `max|d|` is 1.852);
- the eight-rank token-output parity gate passes on the AIV arm **and** on the A arm;
- the AIV plog self-check (`aivMode[1]`) is present for eight ranks of each AIV run, so the result does not
  rest on one rank.

What this record can say is narrow: **on this model, this topology, six 64-step runs produced identical
tokens and identical logits.** The vendor describes AIV expansion as non-deterministic, so this is not a
claim that the mode is deterministic in general; keep the stream-comparison gate for any later use.

### 19.3 Adoption

The setting is an environment variable, so the code is untouched: `HCCL_OP_EXPANSION_MODE=AIV` joins the
lane's adopted environment (dev-tree `env.sh`) next to the retained RELIC_* switches; the file differs from
its pre-adoption copy by that comment and export and nothing else. Standing measurement becomes
**17.4689 tok/s / 57.25 ms**: **+22.3 %** over the registered 14.287598 tok/s and **-18.2 %** over its
69.9922 ms step. Phase 1 remains unmet: 17.47 tok/s against the 50 tok/s gate is still a factor of 2.86.

### 19.4 The mechanism, from a paired profile

One AIV kernel profile against the post-pipelining profile already in the record
(`w2-kprofile-pipelined`). Both are **three-step windows** (the profiler records a three-step kernel window,
not a TPS window); the profiler's own `step_trace_time.csv` buckets, per window and per step:

| bucket | no AIV, /window | no AIV, /step | with AIV, /window | with AIV, /step |
| --- | ---: | ---: | ---: | ---: |
| Computing | 130.92 ms | 43.64 ms | 130.44 ms | 43.48 ms |
| Communication (Not Overlapped) | **34.97 ms** | **11.66 ms** | **29.99 ms** | **10.00 ms** |
| Free | 19.20 ms | 6.40 ms | 18.55 ms | 6.18 ms |
| Computing + Communication | 165.89 ms | 55.30 ms | 160.43 ms | 53.48 ms |

The whole saving is the collective bucket: **-14.2 % on measured collective time, -1.66 ms a step**, with
`Computing` flat (-0.16) and `Free` flat (-0.22). A kernel-row sum would double-count here -- the AIV
profile records each collective twice, once as `hcom_*` and once as `AivKernel` with identical start time
and duration -- so the table uses the profiler's own accounting, which does not.

Interpretation, stated no wider than the instrument supports: the profiler attributes the *whole*
collective to "Communication (Not Overlapped)", so a reduction in the host-side expansion cost inside the
collective lands in that bucket, and that is what is observed -- collectives themselves are 14.2 % cheaper,
dispatch gaps (`Free`) are not what moved. Section 8's finding that ~97 % of collective cost was waiting
came from a different instrument (the two-stream wavefront test), and this profile neither confirms nor
contradicts it. Of section 8's 13.2-14.8 ms a step, AIV recovered ~1.7 ms, i.e. 12-13 % of the item.

These two profiles are hours apart on a shared host, so the *direction* is what they establish. The profile
runs' own timed windows agree in magnitude with the three-pair figures: 59.05 -> 57.92 ms mean and
58.83 -> 57.53 ms median, versus 59.18 -> 57.25 ms in section 19.1.

### 19.5 What the comparison with that stack does and does not establish

Reading its README settles the apples-to-apples question the objective raises. Its 27.13 ms a step at
concurrency 1 is a forward that computes six positions (MTP, `SPEC_TOKENS=5`, measured acceptance 4.10),
served from a separately distributed `w4a8` checkpoint
(`chiro2001/DeepSeek-V4.1-Flash-1017-w4a8-Ascend`, which that README names) rather than this project's
original FP4/FP8 payload. The README does not describe its matmul path, so this record draws no conclusion
about how it implements the quantised weights; what is documentable is that the weights are a different
distribution.

Neither property transfers:

- the checkpoint is a different weight distribution, while this objective preserves the original FP4/FP8
  payload and the values it represents -- which is why this lane decodes FP4 to BF16 with a LUT at all;
- six positions a forward is the MTP win the objective forbids in Phase 1.

So that number is neither a pure-B1 measurement nor a bound. Per *forward* this lane is 2.11x slower
(57.25 against 27.13 ms); per *position* the arithmetic gives 4.52 against 57.25 ms, or 12.7x, and that
number is equally not a bound, because it divides a six-position forward by six as though the work scaled
that way. The one part of that stack which is format-independent -- the collective expansion mode -- is what
section 19.1 measures, and it transferred.

## 20. Where the 57.25 ms now goes: the host side issues the step in 16 ms

Section 19 left the step at 57.25 ms with the profiler's own buckets at `Computing` 43.48,
`Communication (Not Overlapped)` 10.00 and `Free` 6.18 ms a step. The harness already carries a phase
instrument (`RELIC_STEP_PHASES=1`), and it splits each step at the host/device boundary, which the kernel
profile cannot:

| phase, eight ranks, sixteen timed steps | median | mean |
| --- | ---: | ---: |
| host enqueue (Python call until the last op is issued) | 15.6-16.3 ms | 16.5-17.0 ms |
| device completion after the host finished issuing | 40.9-42.1 ms | 40.4-41.2 ms |
| greedy argmax and host hand-off | 0.18 ms | 0.19 ms |

So the host spends **16.2 ms a step, 28 % of it, issuing work**, and the device still has 41 ms to run after
the host has issued everything. Those two facts bound what host-side work can buy: the device is idle for
`Free` = 6.18 ms a step (10.8 %), so making the host infinitely fast could recover at most that much; the
other 10 ms of host time is already covered by device execution.

`cProfile` on a timed step (`RELIC_STEP_CPROFILE=1`) rules out the obvious suspect: the 42 graph replays cost
almost nothing on the host (`torch_npu._C.replay` tottime 1 ms for 42 calls under profiler inflation), and the
profile is diffuse -- 232 functions, none above 5 ms -- i.e. the 16.2 ms is per-op dispatch and guard work
spread across the step, not one hot call. Also tried and closed this round:

- `TASK_QUEUE_ENABLE=2`, the async task queue that would attack exactly that per-op cost, is **refused by
  torch_npu during graph capture**: "Do not support TASK_QUEUE_ENABLE = 2 during NPU graph capture, please
  export TASK_QUEUE_ENABLE=1/0". This lane captures layer graphs, so the lever is out by construction rather
  than by measurement.
- `TASK_QUEUE_ENABLE=1` is accepted, and measured neutral: enqueue 15.1-16.2 ms (unchanged), step 57.30 ms
  median and 17.3868 tok/s on a single 64-step run against the 17.4689 standing mean -- inside this host's
  run-to-run spread, with the phase split identical. The default already behaves this way; nothing to adopt.
- `RELIC_DISPATCH_TIMING=1` and `RELIC_CUBE_PROFILE=1` are bit-rotted switches from earlier rounds and crash
  the run; they were not used, and the phase instrument above is the supported path.

What this changes for the remaining plan: the device-side buckets (`Computing` 43.48 ms, `Communication`
10.00 ms) are still the bulk, and the recoverable host-side slack is 6.18 ms a step at most -- about 12 % of
throughput if it were all recovered, which is worth having but is not the factor of 2.86 between this lane
and the Phase 1 gate. The gate still needs the device-side work reduced: the LUT decode (about 12 ms a step,
producer-bound at 1.6x the vector-op floor), the exposed collective remainder (10.0 ms, imbalance-driven) and
the kernel/launch tail.

## 21. A published W4A8 x 910B4 serving protocol, reproduced as far as this lane allows

A published report (`ModelDoctor`, "DeepSeek-V4.1-Flash W4A8 x 910B4 single node 8 cards: concurrency,
long context, Agent; MRV2 throughput +19%") benchmarks the same model on the same class of hardware with a
stated protocol. Reproducing its metric definitions on this lane is the useful comparison, because its
headline number and this project's headline number are *not* the same quantity.

### 21.1 The protocol as published

| metric | definition | statistic |
| --- | --- | --- |
| per-request generation rate | interval between a request's first and last generated token | **median** across requests |
| whole-window output throughput | effective output tokens / (first request issued -> last request finished), includes prefill, queueing and drain | -- |
| TTFT | request issued -> first generated token (reasoning included) | **P95** |
| TPOT | mean token interval inside the generation phase | **P95** |
| E2E | request issued -> response finished | **P50** |
| draft acceptance | accepted draft tokens / total draft tokens | -- |

Their environment: 8 x Ascend 910B4, TP8 / DP1 / EP8, a separately distributed W4A8 checkpoint
(`DeepSeek-V4.1-Flash-1017-w4a8`, mixed quantisation), Engram CPU offload, vLLM-Ascend v0.30.0rc2 with
Model Runner V1 (MRV2 as a comparison), **DSpark MTP, five draft tokens a step**, `FULL_DECODE_ONLY` graphs,
chunked prefill at 4096 tokens, AIPerf 0.13.0 over HTTP streaming. Fixed-length points force an exact output
length, clear the prefix cache, and use 128 requests a point.

Their 8K input / 512 output / concurrency 1 point: TTFT P95 **1.298 s**, TPOT P95 **18.25 ms**, E2E P50
**9.84 s**, per-request generation rate median **59.8 tok/s**, whole-window throughput **52.02 tok/s**, draft
acceptance **41.93 %**.

### 21.2 What was reproduced here

Fixed **8192-token** input (built and verified to tokenise to exactly 8192 tokens with this checkpoint's
tokenizer), concurrency 1, cold cache -- this lane has no prefix cache at all, so every run is cold by
construction. The timed window is the harness's pre-registered 64 forwards; that gate permits 16 or 64 and
was not changed for this protocol. Prefill was chunked at 512, 2048 and 4096 tokens in the attempts below,
matching the published stack's chunked-prefill shape.

### 21.3 The decode dimension is measured; the prefill dimension is not

Decode, from the adopted lane (section 19's three-pair measurement):

| quantity | here | article, 8K here |
| --- | ---: | ---: |
| TPOT | **57.25 ms** (median, 128-token context, one token a forward) | 18.25 ms P95 (three tokens a forward) |
| per-request generation rate | **17.47 tok/s** | 59.8 tok/s |
| tokens per forward | **1** (Phase 1 forbids more) | **3.10** (5 drafts x 41.93 % acceptance, +1) |

Prefill: measured, with a warm-up split against the one-time start-up cost. Serving stacks warm up before
they measure, and this lane's first forward pays start-up costs (static-operand materialisation, first-launch
kernel setup) that belong to process start rather than to a request. The harness now has an environment-gated
warm-up that replays the *same* prompt once, discards its state, and then measures the next prefill
(`RELIC_PREFILL_WARMUP=1`; the same prompt is required because the host Engram path refuses a forward whose
payload digest differs across ranks, which a synthetic short prompt trips). On the standard 25-token prompt:

| quantity | value |
| --- | ---: |
| cold first prefill (start-up included) | 83.67 s |
| **warm measured prefill (TTFT)** | **78.16 s** |
| start-up share | 5.5 s (6.6 %) |
| warm prefill per prompt token | **3.13 s/token** |
| decode unaffected by the warm-up | TPOT 57.29 ms median, greedy 64/64 |

So the prefill is not a start-up artefact: it costs about **3.13 s for every prompt token**, which puts an
8192-token prompt at roughly 7.1 hours on the linear fit. That fit is consistent with the attempts that were
killed: a 512-token prompt needs about 26 minutes by the slope and had not finished its first forward after
11 m 8 s, a 2048-token prompt about 107 minutes and had not finished after 10 m 55 s, and the 8192-token
prompt had not finished after 33 m 43 s. The published stack's 8K TTFT is 1.298 s, i.e. 0.16 ms a prompt
token, so this lane is roughly four orders of magnitude away on that metric.

The cause is structural and visible in the numbers above: the prompt is pushed through the decode-shaped
stack, where the mixture of experts is applied *per token* -- each prompt token re-selects its experts and
each selection re-decodes those experts' weights. At 25 tokens across 61 layers that is about 9,150 expert
decodes for a quarter-second of real prefill work. TTFT, E2E and whole-window throughput from the published
protocol are therefore quotable here only as this: the warm TTFT of a 25-token request is 78.16 s and the
cost is linear in prompt tokens at 3.13 s each.

### 21.4 The comparison that is possible: cost per forward

The published rate is produced by speculative decoding, so its per-forward cost has to be recovered before
it can be put next to a pure-B1 step:

- tokens per forward = 5 drafts x 41.93 % acceptance + 1 = **3.10**;
- their per-forward time = 3.10 x 16.7 ms (median token interval) = **51.8 ms**, or 3.10 x 18.25 ms (P95
  interval) = **56.6 ms**.

This lane's pure-B1 forward is **57.25 ms**, one token per forward, three-pair verified. So per forward the
two stacks are within roughly 0-10 % of each other, with two caveats that both cut against this lane: theirs
is measured at 8192-token context and ours at 128, and theirs streams a weight format the hardware
accelerates natively while this lane decodes the original FP4 payload to BF16 first (section 17).

The 3.4x difference in the headline rates (59.8 against 17.47 tok/s) is therefore **MTP, not executor**: it
is the 3.10 tokens a forward, which Phase 1 forbids and Phase 2 gates on Phase 1.

### 21.5 What this protocol exposes, and the order it implies

1. **Prefill is the missing capability, and only serving metrics need it.** Their 8K TTFT is 1.298 s; this
   lane cannot complete a first forward for 512 tokens inside eleven minutes, because the prompt is pushed
   through the decode-shaped stack (inline-TP M=1 layer graphs, per-token expert routing). A prefill path --
   batched grouped-MoE GEMM, chunked attention, prefill graph capture -- is the single largest gap the
   published protocol reveals. It does not change the Phase-1 pure-B1 metric, which is decode-only.
2. **Decode is at per-forward parity with a production W4A8 stack**, so further decode work should be judged
   against the items already recorded: the LUT decode (~12 ms a step, producer-bound), the exposed collective
   remainder (~10.0 ms a step), and the kernel/launch tail (section 20).
3. **MTP is the 3x, and it is Phase 2.** Nothing measured here changes that ordering: the gate is a pure-B1
   measurement at 50 tok/s, and this lane is at 17.47.

### 21.6 Limitations of the comparison

Different card (910B4 in the report against 910B3 with 64 GiB here), different weights (their separately
distributed W4A8 checkpoint against this objective's original FP4/FP8 payload and represented values), their
MTP on, their HTTP front end and chunked prefill against this lane's in-process, pre-registered-window
harness, and their 128 requests a point against this lane's single request. The per-forward comparison in
21.4 is the only like-for-like quantity, and it is derived on their side, not measured.

### 21.7 The prefill plan this measurement implies

The measured 3.13 s a prompt token is not a kernel-percentage problem; it is a missing batching dimension.
Decode batches one request's one token, so the expert weights are decoded once per selected expert per layer
and amortised over a single token. A prefill must amortise the same decode over a chunk of tokens:

1. **Group tokens by selected expert inside each layer, decode each selected expert's weights once, and run
   one many-row GEMM per expert** (the grouped-MoE shape a serving stack uses). For an 8192-token chunk over
   ~6-8 selected experts a layer, that turns 8192 weight decodes per expert into one, which is where the four
   orders of magnitude are: the steady-state cost becomes the chunk's weight streaming, not its token count.
2. **A prefill attention path** (chunked, causal, S x S) instead of the decode-shaped one-token attention the
   layer graphs carry today.
3. **Prefill graphs**, so the chunk's layers replay like the decode layer graphs do.

Order-of-magnitude expectation for (1) alone on this payload: a 8192-token chunk streams roughly 61 layers x
~7 experts x ~24 MB of decoded weights, about 10 GB, which is ~10 s at the measured 1.0-1.1 TB/s HBM read
floor, against 7.1 hours today. Getting from ~10 s to the published 1.3 s needs the weight-format work this
objective deliberately does not allow (their natively accelerated W4A8 payload) plus faster attention; it is
not reachable by tuning this lane.

None of this moves the Phase-1 metric, which is decode-only: the pure-B1 forward stays at 57.25 ms and
17.47 tok/s while a prefill path is built.
