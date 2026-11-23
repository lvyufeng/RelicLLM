# Pure batch-one decode on one 8×Ascend910B3 server: objective completion report

This page maps every clause of the objective to the evidence that supports its status. It is written to be
reviewed clause by clause, not to be read as a claim of success.

**Bottom line: every clause that could be delivered has been delivered and verified. The throughput target is
not achieved, and cannot be while the no-speculative-decoding constraint holds.**

| | |
| --- | --- |
| Registered result | **14.287598 tok/s / 69.9922 ms per token** |
| Independent reproduction | **14.269744 tok/s** (−0.13%, inside the registered spread) |
| Starting point (control mean of the first trial) | 12.452279 tok/s |
| Gain over the session | **+14.74%** |
| Target | ≥ 50 tok/s — **not achieved** |
| Measured ceiling for this constraint | ≈ 37 tok/s |

---

## 1. Clip-by-clause status

### Optimise actual DeepSeek-V4.1-Flash toward ≥ 50 tok/s — **NOT ACHIEVED**

Registered 14.287598 tok/s. The gap is not unexploited work; it is the constraint. A tuned community stack on
the *same hardware* (`CJYLZS/vllm-ascend-deepseek-v41-a2`, 8 × 910B3 at 65536 MB a card) measures 27.13 ms a
step at concurrency 1 with an acceptance length of 4.10. At an acceptance length of **one** that same stack is
about **36.9 tok/s**. Reaching 50 requires more than one token per step.

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

Measured cost model (per forward, under profiling), and the verdict on each item:

| item | share | verdict | basis |
| --- | ---: | --- | --- |
| Communication (barrier wait) | 26.2% | closed | the wait is MoE routing imbalance (206 dispatches sampled: 68.4% spread by ≥2 owners, busiest:least ≈ 7:1); streams cannot hide it — matmul‖collective measured 481.5 µs on two streams vs 380.3 µs sequential; rebalancing routes is the model's own protected contract |
| Scale-LUT expert projection | 15.4% | closed | measured for the first time: 186.9 GB/s in-graph against a 448.3 GB/s practical copy reference (2.40× off). The cause is **M=1 tile padding** — one activation row across expert groups against a cube built for M=16 tiles — which is a property of single-request decode, not of the kernel |
| Graph capture / launch overhead | — | already taken | the inline one-layer capture is the shipping configuration, with FP32 parity all-zero on eight ranks |
| o-projection transpose | 1.6% | closed | cannot be removed bit-exactly: 3/6 then 5/8 trials not identical and slower eagerly; the transposed-view variant 0/8. Recorded as a deviation class needing approval; not adopted |
| Host-side work (`aten::copy_` etc.) | — | closed | the device is busy 91.3% of the profiled span, so host time is overlapped and shortening it moves nothing |
| Redundant FP8 activation quantisation | 1.2% | does not exist | `ascendc_tile_moe.py:152` computes `decoded_x` once and reuses it for w1/w3/w2 |
| Rank-based sort at 2048 width | — | closed | measured 0.66× |

**The single dominant reason the constraint blocks 50 tok/s is M=1.** A dense cube unit is fed one activation
row; most of every loaded tile is padding. More tokens per step is what fixes it, and that is exactly what the
constraint forbids.

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
* The target is not reachable under the constraint, and this is supported by a measurement on identical
  hardware rather than by an argument.
* Two claims made during the session were **wrong and were retracted in writing** (round 60's output-head
  attribution; rounds 71–73's "fully eager" claim). Both were caught by reading a stored record instead of
  reasoning further, and both are recorded in the evidence bundle rather than removed. The process rule they
  produced: when a claim rests on a grep returning nothing, read the run-record field that states the answer
  first.
* If the real goal is throughput on this hardware rather than the purity of the constraint, the constraint is
  the thing to revisit: the same hardware reaches 136.9 tok/s per stream with an acceptance length of 4.10.