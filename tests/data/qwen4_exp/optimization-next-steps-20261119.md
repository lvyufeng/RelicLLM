# Qwen3.8-Flash-Next: remaining optimization opportunities — 2026-11-19

## Scope and decision

This is a **read-only analysis of the current code and already saved measurements**. No new GPU measurements, model runs, source changes, or fleet leases were performed for this analysis. Rates below are either explicitly identified prior measurements or fixed-acceptance budget scenarios, never promised improvements.

There are two different objectives:

1. **Higher single-request steady decode TPS:** spend most effort on the exact main verifier, not setup that graph reuse already removed.
2. **Lower time to first token for repeated long prefixes:** integrate exact prefix-state reuse. This has much larger user-visible headroom than shaving another small elementwise kernel, but does not improve steady decode TPS.

The validated baseline remains TP8 on owned 8×RTX 4090, native `indexer_budget=2048`, 261632 input + 512 output = 262144 total context, NVFP4 experts, BF16 dense/KV, FP32 indexer/recurrent state, strict verifier arithmetic, and all original greedy IDs/callbacks/ranks intact.

## 1. Where the warm decode time goes

Source: [native cold/warm evidence](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-native512-warm-20261118.jsonl>), second request. The decode numerator is **511**, because the first of 512 emitted tokens comes from prefill.

| Component | Seconds | Share of 8.811469 s decode | Scope |
|---|---:|---:|---|
| Verifier device interval | 7.180652 | 81.49% | Already includes its kernels, copies, collectives and scheduling gaps |
| Verifier PLE staging | 0.292926 | 3.32% | **Inside** `verify_s`, not additive to it |
| Remaining verifier wrapper time | 0.017044 | 0.19% | `verify_s - device - staging` |
| MTP head | 1.087123 | 12.34% | Catch-up, two chain links, associated vocab selection |
| Rewind | 0.045099 | 0.51% | Derived from per-round timing × 233 rounds |
| Exact native-boundary tail | 0.088074 | 1.00% | Five one-row tail replays |
| Other decode work | 0.100551 | 1.14% | Host decisions, delivery and remaining interval |

`verify_s = 7.490623 s`, or **85.01%** of decode. Steady rate is **57.992599 tok/s**. Setup is only **8.775 ms** on this compatible warm request, so it is no longer a meaningful steady-TPS target.

The saved [instrumented native trace](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-native-trace-20261118.json>) has 9978 kernels/replay and about 29.947 ms summed kernel duration, with 7.173 ms in the collapsed GEMV family. Its 86.105 ms replay span is profiler-perturbed, **not** a hardware lower bound. Kernel durations identify candidates; they are not a substitute for untraced round timing.

### Fixed-output scenarios, not new benchmarks

| Hypothetical saving at unchanged acceptance | Projected steady TPS |
|---|---:|
| Current measured request | 57.99 |
| Halve all head time | 61.81 |
| Remove all head time, impossible ceiling | 66.15 |
| Remove all PLE staging, impossible ceiling | 59.99 |
| Remove both head and staging, impossible ceiling | 68.76 |
| Verifier device interval −20% | 69.28 |
| Verifier device interval −30% | 76.76 |
| Verifier device −20% and head −50% | 74.80 |
| Verifier device −30% and head −50% | 83.58 |

100 TPS requires 8.811469 → **5.11 seconds**, a **42.01% total decode reduction** at this output/acceptance profile. If only the verifier device interval changes, it must fall by **51.55%**; even halving the head still requires **43.98%** off that device interval. Small head/staging tweaks cannot establish 100 TPS.

## 2. Ranked decode candidates

### A. Exact main-verifier projection/layout fusion — highest steady-TPS headroom, hardest arithmetic contract

Call site: [row-invariant linear](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/layers.py#L88-L105>). Small BF16 batches deliberately issue independent M=1 `F.linear` calls, followed by concatenation. This protects the decode arithmetic but keeps many GEMV, copy and layout operations in the replay.

The target is an **independently named, decode-shaped grouped operator** or fusion around these calls that preserves each row's FP32 accumulation tree and BF16 rounding. Start with actual hot projection shapes and contiguous/strided layout copies, not an indiscriminate Tensor-Core GEMM replacement.

Important corrections:

- The [old BF16 linear micro evidence](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/r360_bf16_linear_micro.jsonl>) is not an allowlist proof. Some initially equal `bmm` shapes fail when inputs mutate: `[4,10240,320]`, `[4,2560,512]`, and `[4,80,2560]` all contain mutation failures. The recent 2560→2560 bmm micro also differed. A single random tensor equality is inadequate.
- It is not valid to infer four DRAM scans from the total model weight size. The code processes the four rows of **one projection immediately together**; a matrix that fits in L2 may already be reused. Many dense matrices are far below the 4090's L2 size. Measure the hot matrix's actual memory traffic and cache hit rate; do not divide whole-model bytes by a profiled family duration to claim measured bandwidth.
- CUDA Graphs remove repeated host launch calls, not the device-side cost of thousands of dependent kernels. Both launch granularity and arithmetic/memory work remain candidates until measured.

**Smallest useful experiment:** collect actual shape/call-site frequency; replay an isolated candidate over recorded real inputs, cancellation-heavy vectors, near-BF16-rounding ties and repeated mutations. Compare every output bit to the existing M=1 path and measure an untraced, sufficiently long CUDA interval. Retain the old path for every unqualified shape. Then require all-layer/all-row state parity and native 512 IDs/callbacks/8 ranks.

**Benefit:** unknown for a specific operator. The 20–30% verifier-device scenarios above show why this family is worth pursuing, not that such a saving has been achieved. Do not enable an ordinary bmm or a global `torch.compile` arithmetic rewrite on the basis of a fast micro alone.

### B. NVFP4 launch utilization and single-writer epilogue — low-cost targeted kernel work

Source: [NVFP4 launch geometry](</Users/lvyufeng/Projects/relic-core/relic_core/csrc/qwen4_exp_nvfp4_exact.cu#L319-L348>) and [down kernel](</Users/lvyufeng/Projects/relic-core/relic_core/csrc/qwen4_exp_nvfp4_exact.cu#L175-L237>).

Two concrete opportunities:

1. The launch partitions four output columns per CTA using `warps_per_block=4`, but launches **256 threads = eight warps**. With the measured divisible 640/2560 widths, only four warps have output columns. A 128-thread independently tested variant keeps the same active warp/lane arithmetic while removing inactive warp allocation. Shared-input loading and occupancy still require timing; benefit is not guaranteed.
2. Each down output `(row,column)` has a single writer: `grid.y=1`, z selects one row, x selects a disjoint column interval, and only lane zero of its unique warp writes. The route loop is already ordered. Therefore it is wrong to call this output an inherently unordered cross-warp atomic reduction. Register accumulation and one final store may remove repeated `atomicAdd` operations and the initial output-zero kernel.

**Exactness caveat:** global float atomic-add FTZ/signed-zero/NaN behavior can differ from a normal register add, and a compiler may contract a new multiply-add. Keep the existing route order and product boundaries; explicitly match atomic rounding/FTZ semantics, including the zero/no-local-route case. Preserve or prove unnecessary every initialization of the activation buffer. Old-vs-new corner and graph replay tests gate the variant.

**Benefit:** likely a small local gain, not a 100-TPS path. It is cheap enough to micro-test before a larger GEMV rewrite, and can also matter during prefill.

### C. MTP catch-up by kept-prefix width — genuine redundant work, bounded whole-request gain

Source: [head catch-up and chain](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/verify_graph.py#L788-L807>).

The current draft replay forwards **all four verifier rows**, selects row `kept-1`, then forwards **two**, not three, single-row chain links for K=3. It performs three head forwards/vocab selections per replay, not seven collectives.

The measured replay histogram is `[0,77,79,35,41]` over 232 replays: mean kept width **2.172414**. Catch-up alone could avoid about **45.69%** of its rows. If head cost were proportional to all processed query rows, total work would fall from `4+2=6` to `2.172+2=4.172`, saving **30.46% of head time**. That illustrative linear-cost scenario yields only **60.26 TPS**, not a 46% decoder gain. Fixed overhead and serial collectives can erase part of it.

Record kept=1..4 variants that share the stable head arena, or first run an eager width-specific oracle. Verify identical seeds/proposals and all head cache visibility/pooling behavior. The head is position-addressed rather than recurrent, but changing row count can still change BF16 dense, norm, score or collective arithmetic. Existing safe baseline overwrite/causal-mask behavior must remain intact.

Memory/capture growth belongs in the session key and ownership accounting. Build all variants before timing, and compare reused warm requests; do not charge four new captures to a purported decode win without reporting them.

### D. Exact query-strided norm/layout fusion — small but well-scoped

Source: [packed norm handling](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/layers.py#L313-L357>). The corrected strict path intentionally evaluates original per-query slices so decode's contiguous-versus-gapped dispatch does not change.

A stride-aware query-grid native operator could combine these launches/copies while retaining the old one-query arithmetic. Start with the indexer query case whose per-query slices are contiguous and whose existing fused kernel is the oracle. Do not treat gapped QSA heads as equivalent: their fallback torch mean has a geometry-dependent reduction tree. A combined torch mean is not automatically exact just because each mathematical row reduces separately.

This is lower headroom than dense projection work. The previous gated-norm epilogue is already numerically qualified but its +0.24% comparable native result was noise-sized; leave it off and extend a genuinely broader boundary rather than relabel the same change as a new optimization.

### E. Draft-depth selection — first qualify three-row collectives, then price K2

Measured K3 prefix survival is `[156,76,41]/233`. Holding that distribution fixed gives expected advances of **2.171674** for K3 versus **1.995708** for K2. K2 only wins if its **entire round cost is at least 8.10% lower**.

The obstacle is explicit: [rank-chunk reduction](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/runtime.py#L401-L420>) allows row spans `(4,5,6,8)`, not three. K2 currently uses three separate row collectives at each reduction rather than K3's one packed collective. For 48 layers × two dependent reductions this changes the verifier from 96 collective calls to **288**. Assuming that all costs scale 4→3 is therefore wrong; K2 can be slower.

Next test: a three-row cancellation-heavy reduction/layout oracle, then matched-prefix K2/K3 untraced round timing and full native output validation. Do not infer greedy invariance solely from the speculative algorithm: numerical row geometry is why strict qualification exists. K4 has native short-output evidence but still needs matched sustained-output cost/acceptance evidence; old 128-output runs are not a controlled extrapolation to this 512-output workload.

## 3. A separate algorithmic lever: improve proposal quality without changing the target

Latency is not the only lever. The K3 verifier currently advances about **2.17 tokens per round**;
its unconditional first/second/third proposal survival is 66.95% / 32.62% / 17.60%. A better private
MTP draft predictor can increase accepted tokens while the original target model still verifies
and supplies every committed greedy answer.

Candidates are calibration of the **private NVFP4 head's** quantization scales, or later a small
head-only distillation/adapter using teacher trunk streams. Preserve convention B
`head(h[i], t[i+1]) -> t[i+2]` and collapsed recycling, and do not retrain/quantize the target backbone
as part of this comparison. Training cost and achievable acceptance are unmeasured; the cheaper
first experiment is offline proposal-quality evaluation on recorded teacher states before any
serving change. Invalidate recordings/prefix checkpoints whenever the draft head identity changes.

At unchanged round cost, moving mean advances 2.17 -> **2.50** would budget roughly **66.8 TPS**,
and 2.17 -> **2.70** roughly **72.1 TPS**. These are acceptance scenarios, not new measurements or
promised fine-tuning results. Approaching 100 TPS at today's K3 round cost would need about
**3.7 accepted/bonus advances per round**, close to K3's maximum of four, so proposal tuning alone is
not an established 100-TPS solution. Validate original greedy IDs/ranks and measure head cost as
well as survival; a more accurate but slower draft model can lose.

## 4. TTFT and prefill candidates

### F. Exact resident prefix checkpoint reuse — largest gain for repeated long prompts

This is different from graph reuse: the current compatible warm request still resets rolling state and redoes **489.854 seconds of prefill**.

There is already exact native prefix-restore evidence in the [controlled native A/B](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-native512-norm-20261118.jsonl>):

- Full source prefill: **485.677 seconds**.
- Three complete state restores: **53.698–53.843 seconds**; mean **53.776 seconds**.
- All 512 original greedy IDs/callbacks/ranks matched; all 13 indexer prefixes rebound, zero recomputation.
- The one-time full state freeze cost **74.404 seconds**, and increased the source request's measured TTFT. It is not free.

The present host-side full-bundle import is therefore not a subsecond cache. Nevertheless, it already demonstrates roughly **9× lower prefix preparation** on repeated identical inputs, not in the production cache path.

The next engineering step is a same-process, resident checkpoint that retains an immutable prompt KV/indexer prefix and restores only GDN/PLE rolling state, lengths, history, pooled-prefix counts and the first-token/private-head seed. Decode writes only a suffix; prove that contract with partial blocks and speculative head writes before eliminating prefix copies. Avoid repeatedly hashing/copying duplicate unchanged multi-GiB arenas. Integrate with the graph session only after tensor addresses and table identity remain valid.

Key the cache by exact tokens/prefix length, model/checkpoint/quantized head identity, math/position configuration and state schema. Never reuse after a model/head precision transition or for an unrelated prompt. Charge cold freeze, hit restoration, eviction and final close, and impose a memory budget.

**Benefit:** lower repeated-prefix TTFT, not higher single-request decode TPS. Resident hit latency is unmeasured. For unrelated prompts this provides no benefit; longer-prefix extension requires its own continuation-state proof. This is exact prompt-state reuse, not the previously rejected decoder hot-KV/block-token approximation.

### G. Prefill GDN column-striping and expert reuse — profile first, keep the recurrence

The full native prefill still takes about 490 seconds, and its current component shares are not established by the decode trace.

A concrete GDN candidate is [column-parallel recurrence geometry](</Users/lvyufeng/Projects/relic-core/relic_core/csrc/qwen4_exp_gated_delta.cu#L88-L109>) and [launch grid](</Users/lvyufeng/Projects/relic-core/relic_core/csrc/qwen4_exp_gated_delta.cu#L278-L300>). With 128 columns, four columns/subgroup, two 16-lane subgroups/warp, default groups-per-CTA=1 and six local value heads, it launches **96 CTAs** for the scan. A 512-token prefill runs the token loop inside those CTAs.

Test smaller independent column stripes (two/one columns) to expose 192/384 CTAs and lower per-thread register state, while retaining each column's exact token order, FMA operations and 16-lane reduction. This differs from replacing the recurrent scan with an associative chunked algorithm, which can change arithmetic. Decoder/with-states logs must remain bit-identical; use a separately selected prefill variant and preserve older-architecture paths.

For MoE, the current NVFP4 path already takes the whole chunk in one kernel pair: [rows dispatch](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/moe.py#L1463-L1526>). There is no 512× per-token Python launch loop to remove in the qualified mode. Potential improvement is grouping/reusing **expert weight tiles across token routes**, not merely adding batching that already exists. Preserve each token's inner dot and routed contribution order; an ordinary grouped Tensor-Core GEMM may change the reference.

First measure warmed prefill chunks by component at representative early and near-native positions: GDN recurrence, MoE, dense projections, QSA/indexer and PLE. Only then decide whether column stripes or routed weight-tile reuse is worth implementation.

### H. Prefill indexer memory traffic — possible TTFT gain, high exact-topk risk

Source: [score materialization and ordered selection](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/attention.py#L1403-L1439>). At 65536 blocks, a chunk of 512 queries × four index heads materializes **512 MiB** of FP32 scores before reducing heads. Chunks 1024/2048 would materialize 1/2 GiB, not 256/512 MiB.

A score-epilogue fusion can retain the existing GEMM outputs and fuse ReLU, exact four-head sum, scale and visibility mask, reducing memory passes without first changing selection arithmetic. A more ambitious blockwise running top-k can avoid the large materialization, but ordinary block top-k + merge does not guarantee the same tied index ordering as torch's sorted top-k. Changing GEMM tile geometry can also change scores. Require score bits plus **ordered indices**, invalid slots, final selections and greedy outputs—not only the selected set.

This primarily targets prefill/TTFT and memory peaks. The full 490-second prefill does not establish indexer dominance; no speed percentage is justified before component timing.

## 5. Small host work and exclusions

PLE staging is only 3.32% of warm decode. A native CPU batch hash/gather could reduce NumPy/Python overhead at [stager](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/verify_graph.py#L582-L634>), without changing integer hash or pinned destination addresses. The impossible all-staging-free bound is only 59.99 TPS.

Do not propose hiding all next-round staging under the current verifier/head by precomputing four kept alternatives: **the next feed also needs the next head's draft tokens**, which are not known then. Only stable history/known accepted-prefix rows can be prefetched; asynchronous staging needs distinct pinned buffers and completion/lifetime fences. Current graph buffer addresses are an ownership contract.

Do not re-open as primary candidates:

- ordinary bmm/FP16 dense mirrors or broad compile rewrites without exact arithmetic gates;
- merging attention and MLP all-reduces: [the real dependency](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/model.py#L449-L477>) prevents it;
- previously unhelpful NCCL knobs/argmax exchange, BF16 indexer pools or unsorted top-k;
- unlimited prefix/KV caches or unverified TP4/TP16 memory/topology claims;
- using the two 4090 hosts as if they automatically accelerate one decode. Independent replicas/PD phase separation target aggregate workload, not an already implemented 16-GPU low-latency decoder.

## Recommended work order

1. **Two tracks:** exact main-verifier hot projection/layout work for TPS, and resident prefix checkpoints for repeated-prompt TTFT.
2. In parallel with the first track's shape census, run the cheap **NVFP4 128-thread utilization** and single-writer epilogue oracles. Reject locally if there is no repeatable gain.
3. Record **prefill component timings**; investigate the GDN column-stripe / expert-tile paths only where cost is real.
4. Evaluate **private-head proposal calibration** on saved teacher states before investing in training; it targets acceptance, not just head runtime.
5. After those baselines, test kept-width MTP graphs and qualify K2's three-row collective economics. Stage-only micro-fixes are last.

Every enabled change must retain the original complete native 512-token greedy reference, all callbacks/ranks, and rolling state/indexer selections. Final A/B runs keep the same initialized NVFP4 head, arena width, prefix and output budget. Prefix restoration is acceptable for a decode-only comparison but must not masquerade as a faster cold prefill measurement.
