# Remaining optimization roadmap for Qwen3.8-Flash-Next on 8x4090

Analysis dated 2026-11-18. This is a source/evidence review, **not a new GPU benchmark**. It preserves
NVFP4 routed weights, BF16 dense/KV, FP32 recurrent/indexer state, native indexer budget 2048 and
complete greedy-output parity. The two hosts were not leased or modified for this analysis.

## Executive recommendation

There are still meaningful optimizations. For **single-request decode**, prioritize the main
verifier's arithmetic-preserving fusion and copy reduction, then measure adaptive proposal depth
and less padded head catch-up. For **TTFT**, persistent graph arenas and repeated-prefix caching
are more direct than another quantization change. For **prefill-dominated aggregate throughput**,
keep two independent TP8 replicas unless a measured isolation/batching requirement justifies PD.

Do not conflate these metrics. Removing graph setup improves cold requests but does not increase
steady-state decode TPS. Prefix caching improves repeated prompts but does not accelerate a new
prompt. PD does not automatically give one decode request sixteen GPUs.

## 1. Use the corrected long-output budget

The qualified case is 261632 input + 512 output = 262144 total positions. It delivered every token,
callback and rank identically to the previous plain greedy reference. The rate is 55.702 tok/s
excluding model load, prefill, graph setup and cleanup; 53.030 tok/s including request cleanup.

| Component | Seconds | Fraction of 9.174 s decode | Interpretation |
|---|---:|---:|---|
| Verifier wall time | 7.799 | 85.0% | Includes staging and GPU work |
| Verifier device interval | 7.130 | 77.7% | Includes device/collective waiting, not just compute |
| Staging | 0.650 | 7.1% | Already inside verifier wall time; do not add twice |
| MTP head | 1.119 | 12.2% | Catch-up plus proposal chain |
| Rewind | 0.050 | 0.5% | Small, already optimized |
| Five captured one-query tail steps | 0.094 | 1.0% | Small, already captured |
| Post-delivery cleanup | 0.462 | Outside decode | Must be charged to request throughput |
| Graph setup | 8.225 | Outside decode | Main opportunity for warm-request reuse |

Earlier 128-output native testing reached 72.679 tok/s with 2.702 tokens/trunk step. The corrected
512-output case has 2.147 tokens/trunk step. Its average interval per verify round is similar, not
an established regression from the norm correction. **The different workload and proposal survival
explain much of the TPS difference.** Do not compare the two rates as a controlled optimization A/B.

Sources: [corrected native 512-output evidence](r362_norm_fixed_native512.jsonl),
[earlier native 128-output evidence](r361_exact_startup_native256k.jsonl).

### Amdahl scenarios: estimates, not promised speedups

Hold the same 511 decode-produced tokens and proposal decisions fixed:

| Hypothetical change | Derived TPS | Meaning |
|---|---:|---|
| Halve head time | 59.32 | Modest single-request effect |
| Remove all head time | 63.44 | Impossible idealized bound, not an achievable design |
| Remove all staging time | 59.95 | Idealized bound; real dependencies prevent simply hiding it all |
| Reduce verifier device interval by 20% | 65.95 | The main verifier is the large lever |
| Reduce verifier device interval by 30% | 72.64 | Useful intermediate target, not a measurement |
| Reduce verifier device interval by 30% and halve head | 78.92 | Still short of 100 with unchanged proposal survival |

These scenarios are not additive independent benchmarks. At fixed acceptance, 100 tok/s requires
9.174 s -> 5.110 s, **44.3% less decode time**. Neither the measured 938 GB/s device-copy bandwidth nor
a perturbed 8K trace establishes a universal TPS ceiling. There is no basis to label 60–70 tok/s a
hard engineering maximum either.

## 2. First choice for decode: preserve arithmetic, fuse surrounding work

The existing RMSNorm, HC activation/injection/up-mix, QSA split-K and NVFP4 operand-decoder fusions
are already enabled. Re-enabling those flags is not a new optimization. The next fusion must remove
work **between** those existing kernels while retaining the verified decode arithmetic.

Concrete targets:

1. **GDN gated RMSNorm**: its float32 normalization, cast to the input dtype, weight multiplication,
   gate activation, final multiply and cast are still separate torch operations in
   [RMSNormGated](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/layers.py#L384-L391>).
   Fuse them while retaining every BF16 rounding boundary. Removing intermediate BF16 casts or
   allowing an extra fused multiply-add is not bit-exact merely because the formula is the same.
2. **Packed/strided Q normalization and RoPE**: the corrected verifier intentionally reproduces
   one-query strides and reduction geometry. An exact strided kernel can remove copies and
   elementwise passes, but must reproduce that actual fallback/fused path, not silently select a
   different contiguous norm. This was the previous long-output correctness bug.
3. **Row-shaped projection output placement**: strict projections record multiple M=1 calls and
   concatenate their results in [row_invariant_linear](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/layers.py#L88-L105>).
   Retain each M=1 cuBLAS arithmetic/algorithm and place its output into stable request buffers where
   possible; then let consumers read that layout directly. Do not replace this with ordinary M=K
   GEMM: that already changed greedy tokens.
4. **GDN scalar preprocessing / HC epilogues / residual boundaries**: combine compatible
   pointwise operations and avoid redundant materialization. The two all-reduces themselves have
   real dependencies; an attention reduce cannot be merged with the MLP reduce that depends on it.

A current native-context kernel census is the first implementation gate. The existing approximately
9386-kernel trace was at 8K, before the final packed-norm configuration, and instrumented timing is
perturbed. It supports a fragmentation hypothesis but does not tell us how many milliseconds a
particular new fusion will save at 256K. Measure current per-category time and slow-rank waits, then
implement one bounded fusion and compare unchanged full outputs.

**Priority:** high. **Numerical risk:** medium/high unless intermediate casts and reduction order
are explicitly preserved. **Expected gain:** not yet measured; use the Amdahl scenarios as budgets.

## 3. Proposal efficiency: depth should follow marginal benefit, not “more is better”

The corrected K=3 run proposed each position 233 times. Prefix-survival counts are 156, 76 and 41:

- First draft survives: 66.95%.
- Both first and second survive: 32.62%.
- All three survive: 17.60%.

These are prefix-survival probabilities, **not** three independent draft accuracies. Holding the
proposal distribution fixed gives expected advancement per round of 1.670, 1.996 and 2.172 positions
for K=1, K=2 and K=3 respectively, before request-tail effects.

A useful break-even test is:

```
rate(K) = expected advancement(K) / measured round cost(K)
```

Under that fixed-distribution assumption, K=2 beats K=3 if its measured round is at least **8.1%
faster**; K=1 needs at least **23.1%** lower cost. This makes an adaptive K2/K3 experiment defensible,
not automatically profitable. Proposal/cache behavior can also change with K, so actual survival
must be measured rather than inherited from this calculation.

Current constraints:

- The graph loop fixes verifier width; changing it currently releases bindings or rejects a
  recorded draft mismatch. Adaptive widths need coordinated graph/cache ownership.
- TP8 rank-owner packing is qualified for rows 4/5/6/8, **not rows 2/3**. K1/K2 may therefore use
  several serial BF16 collectives. A smaller verifier can paradoxically become slower.
- K4's earlier short-output rate being similar to K3 does not prove every smaller/larger depth is
  unhelpful on the corrected long workload. Nor does it qualify a new depth for production.

**Priority:** medium, after cheap cost microbenchmarks and exact layout qualification. Start with
fixed-depth A/B, then a conservative per-request or round-boundary selector. Do not simultaneously
change the target model or precision.

### Head catch-up by accepted prefix

[DraftGraph._run](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/verify_graph.py#L717-L736>)
currently processes all four verification rows, then selects the accepted-prefix endpoint. The
measured mean kept prefix was only **2.172 of 4 rows** over 232 head replays.

Recording separate catch-up variants for kept=1/2/3/4 can avoid padded computation and unnecessary
future cache writes. This is a real candidate, but 45.7% fewer logical catch-up rows does **not** mean
45.7% less head time: the chain is still needed and rows 2/3 may lose packed-collective efficiency.
Measure each variant. Similar trunk costs at M=1 and M=4 cannot establish that head rows are free.
Even eliminating the entire current head budget only reaches the 63.44 tok/s bound above at fixed
acceptance.

### Better proposals without changing the target answer

Head-only distillation/calibration against the current NVFP4 target, or a lightweight additional
draft model, could improve acceptance. A correct verifier still decides every emitted token, so a
proposal-only model can change without changing the final greedy answer. This is a larger
training/integration project, not a free precision switch. Earlier FP8-host-head A/B had both slower
head time and no better acceptance than NVFP4; it does not justify switching back blindly, nor does
it isolate every possible FP8 implementation.

Tree/multi-candidate speculation is not mathematically impossible, but the 36 GDN layers require
branch-local recurrent/conv state and exact commit/rewind. A Transformer-only tree mask cannot be
reused unchanged. Treat this as a high-cost research option, not the next patch.

## 4. Warm serving: persistent graph arenas and prefix state

### Keep recordings and request buffers alive

Current lookahead allocates a request cache, records verifier/head/tail/startup graphs, and releases
them at request end. Persistent width buckets plus resettable cache/state ownership could avoid the
measured **8.225 s setup** on compatible subsequent requests and much of the 0.462 s destruction.

For this example the interval after prefill is about 9.174 + 8.225 + 0.462 = **17.861 s**. Removing
setup alone would reduce it by about **46%**, without changing the 55.702 steady decode rate. This is
a large serving-latency opportunity, not a new TPS claim.

Require two consecutive real requests with different token histories, correct resets of GDN/PLE/KV
lengths/indexer validity, stable pinned owners and local graph pointers, plus cancellation/EOS tests.
Native-tail binding retirement means retaining Python graph objects alone is not sufficient.

### Cache repeated prefixes, not just Transformer KV

The PD serializer already transports QSA, GDN, PLE, indexer and private-head seeds exactly. It can be
used as a basis for host-resident repeated-prefix snapshots. Exact hits can skip most of the
approximately 500 s prefill; partial hits need a new suffix-prefill entry that retains the original
chunk/arithmetic boundary contract. The head's convention-B pairing crosses the prefix boundary:
its last cached trunk stream must be retained so that position can be re-paired with the new suffix's
first token. The current full-prompt handoff does not by itself implement that arbitrary-suffix seam.

Key on model/weights, kernel/code/flag fingerprints, exact token ids and compatible TP/cache
geometry. Store all recurrent/PLE state, not just k/v. Apply admission/eviction limits.

This helps repeated system prompts, documents and conversations, **not a unique cold prompt**.
Restoring multi-GB state and building recordings take real time. The demonstrated 8K import alone
took about 1.5 s; no evidence supports promising subsecond native-256K TTFT. Persistent local GPU
state may be faster, but its memory cost limits concurrency.

**Priority:** high if warm requests or repeated prefixes are common; otherwise lower than verifier
fusion. **Steady decode gain:** none directly.

## 5. Prefill: improve data reuse and occupied parallelism, with a profile first

Native prefill takes about 479 s plain / 512 s with MTP. Historical 8K and 256K samples both produce
roughly 500 tokens/s, which suggests a large context-independent cost. It does **not** prove the
indexer is irrelevant or that NCCL is the current dominant component: configurations, warming and
kernel paths differed, and there is no current native chunk-level attribution.

### Exact expert scheduling and NVFP4 cleanup

The NVFP4 many-row kernel uses the same row/route scalar-dot arithmetic for a full 512-token chunk.
Repeated expert visits can potentially reuse the same weight tiles:

- Cluster whole input rows by expert locality while keeping each row's original route order and
  restoring output order; or introduce a route-slot intermediate with deterministic final fold.
  The latter must preserve route accumulation order, not use arbitrary atomics/scatter.
- Investigate register accumulation instead of global `atomicAdd` in
  [the NVFP4 down kernel](</Users/lvyufeng/Projects/relic-core/relic_core/csrc/qwen4_exp_nvfp4_exact.cu#L175-L237>)
  if each output column has one writer. Preserve route order, rounding, FTZ and signed-zero behavior;
  compare with the unchanged old kernel. Decode's MoE-only budget is relatively small; prefill may
  provide more leverage.
- A grouped Tensor Core W4A16 implementation is a higher-performance alternative, but not an
  automatic bit-exact substitution: current reconstructed float weights and scalar FMA order
  differ from BF16 dequantization/MMA. RTX4090 has no native NVFP4 Tensor Core execution like
  Blackwell. Use a separate quality-changing track only if allowed, not a lossless speed claim.

### Chunk sweep is a diagnostic, not a free production improvement

Compare 512/1024/2048-token chunks in a controlled prefill benchmark and record maximum allocation.
At 262144 positions, the **full four-head FP32 score tensor** is approximately:

| Query chunk | Four-head score tensor alone |
|---|---:|
| 512 | 512 MiB |
| 1024 | 1 GiB |
| 2048 | 2 GiB |

ReLU intermediates, head-reduced scores, attention, MoE scratch and cache coexist. These figures are
not total workspace, and an 8K scan does not prove native memory fit. A different prefill chunk size
can change numerical results; passing the original greedy reference is required for the exact
track. Do not “solve” that by redefining the reference.

## 6. Two-host options and boundaries

1. **Two complete TP8 replicas** remain the simple aggregate-throughput baseline for long prompts
   and short outputs. Add routing, admission control and prefix affinity before a complex split.
2. **PD streaming/deduplication** is necessary if PD is selected. Real 8K distinct state is 0.375 GB
   versus 1.317 GB of rank-local copies, a 3.51x payload reduction. Native distinct state is an
   approximately 8.4 GB geometry projection. Streaming can overlap current prefill, but current
   1GbE file payload saturates around 2400 prefill tokens/s; faster prefill can make it the bottleneck.
   The 25GbE/IB ports were DOWN in the previous inspection, not automatically usable capacity.
3. **TP4 / smaller TP** needs a fresh exact byte census and actual allocation test. Routed weights
   are roughly 15.7–15.8 GiB/card at TP4, not 17 GiB due to mixing GB/GiB; some dense tensors remain
   replicated while others grow. The present owned NVFP4 head is much smaller than the obsolete
   multi-GB all-expert FP8 head. Neither “definitely fits native context” nor “cannot fit even 8K” is
   established. It also changes collective grouping and numerical order. Keep this behind the
   simpler software opportunities.
4. **Do not conflate PP with TP16.** Attention/MLP all-reduces are TP operations; PP sends stage
   activations, not 96 cross-host all-reduces per token. Nevertheless current captured MTP refuses
   PP, and the existing context only models pure TP or pure PP, not a ready TP8xPP2 graph scheduler.
   A hybrid pipeline is a substantial architecture project. Cross-host TP16 on the observed 1GbE
   link is not an attractive shortcut.

## 7. Bounded implementation order

1. Profile the **current corrected** native verifier and one current prefill chunk. Keep throughput
   runs uninstrumented, full-token and repeatable; distinguish peer waiting from useful NCCL work.
2. Implement one exact gated/strided-normalization or output-copy fusion, not a broad compiler
   replacement. Require per-row/rolling-state equality and native 512-output parity.
3. In parallel, build persistent request/graph ownership if warm serving matters. Measure cold and
   warm TTFT separately, with graph setup and cleanup charged honestly.
4. Microbenchmark head-by-kept variants and qualified K2/K3 round costs. Keep only variants that
   exceed their measured break-even point without changing target outputs.
5. Optimize prefill expert reuse / repeated-prefix snapshots based on the workload, and add PD only
   if its measured service-level benefit beats the two-replica baseline.

Do not repeat already rejected NCCL knobs, HC/dense FP16 mirrors, host-KV, argmax exchange,
block-token/KV-hot caching, BF16 indexer pools or unsorted top-k. Do not merge attention and MLP
collectives across a real dependency. Whole-graph compilation, different reduction layouts and
Tensor Core substitutions need the same numerical proof as handwritten kernels.

**Bottom line:** prioritize the 77.7% device-verifier interval for steady TPS; prioritize the 8.225 s
recording lifecycle and approximately 500 s repeated prefill for request latency. There is useful
space beyond the current result, but **no new TPS improvement is measured by this analysis**.
