# Exact original-checkpoint decode optimization on Ascend

This continues [original-checkpoint inference](ascend_original_inference.md). The target remains
**50 sustained tokens/s for one request, batch one**, without checkpoint or activation
requantization. The original milestones below keep complete Engram computation on CPU. A later
explicit human request on November 20 relaxed that placement constraint to **original tables on
host**, allowing a separate opt-in NPU WKV/gating experiment. Native accumulation experiments and
this new placement are not promoted to the exact production default. The target has **not been
reached** by the measurements below.

## Published evidence and measurement identities

The public evidence is a **sanitized view of the original captures**, not a new experiment. Home
and private workspace/checkpoint paths, fleet labels, workstation names and non-standard network
identifiers are replaced by neutral labels. Loopback, wildcard binds and the standard container
bridge are retained. Filenames and line endings are unchanged; anchored matching avoids treating a
date in an artifact filename as a host identifier.

Frozen manifest sizes and SHA-256 values continue to identify the **original measured bytes**. They
are not recomputed against a sanitized source, log or receipt. The separate
[publication hash map](<evidence/ascend-publication-sanitization-20261121.json>) records the original
and current published identities for every changed file, including any sanitized manifest itself.
A published file without a mapping entry is unchanged. Recorded tensor/checkpoint hashes, token IDs,
timings, numerical arrays and acceptance counts are preserved.

The [publication checker](https://github.com/lvyufeng/RelicLLM/blob/master/scripts/check_ascend_publication.py)
verifies current published hashes and resolves every retained manifest entry through that mapping.
It performs file/metadata checks only, not model execution or accelerator acceptance. Neutral paths
and host labels in retained driver snapshots must be replaced with newly authorized local values
before reuse; the snapshots are not portable launch configurations.

The pre-merge scratch-geometry and graph-replay admission fixes have not been rerun on hardware.
Existing acceptance still applies to its recorded sources and scopes, not automatically to every
later guard or test addition. Original Git/PR history is not rewritten by this publication cleanup;
removing historical copies would require a separate, explicitly authorized history operation.

## Reproducible scope

- Released DeepSeek-V4.1-Flash checkpoint, complete 40-layer text backbone, dense/attention TP8.
  Original milestones use expert EP8; the later explicit expert TP2/EP4 experiment is scoped below.
- Eight Ascend 910B3 cards, 64 GiB/card; CANN 9.2.0; Torch 2.10.0+cpu and torch_npu 2.10.0.post7.
- Original checkpoint bytes remain read-only. Original E4M3 activation quantization and E8M0 grids,
  exact doubled-INT8 interpretation of E2M1 weights, and the accepted two-half/window-13/carry/
  renormalization/sequential-K32 reduction are preserved.
- In the original exact-path milestones, Engram tables, hashing, hash IDs, lookup, projection,
  normalization and gate remain on CPU; only current activations and exact narrowed result bytes
  cross the NPU boundary. The latest host-table/NPU-compute comparison is separately scoped below.
- The exact-path milestone below includes no native-Cube substitute, W8A8/W4A8 conversion,
  speculative-token accounting, batch-throughput substitution or serving capability change.
  Later sections separately record the human-authorized changed-accumulation experiments.

The prompt is:

```text
List the integers from 1 through 100 in order, separated by commas. The list is:
```

It is 20 tokens. Four excluded forwards precede the **16 timed autoregressive forwards**, covering
context positions **24 through 39** and leaving position 40. Each timed forward includes NPU
completion and the host greedy argmax. Prefill, JIT warmup, nested profiling and logit copies for
verification are excluded. This fixed-length benchmark is allowed to continue after EOS, but these
runs observed **no EOS**. The three-token Paris sample in the preceding inference page is not this
benchmark and cannot supply its TPS denominator.

## Whole-model measurements

| Configuration | Timed forwards | Slowest-rank wall | Sustained single-stream TPS |
| --- | ---: | ---: | ---: |
| Initial streamed expert/cache path | 16 | 173.927 s | 0.091993 |
| Original packed experts resident, compiled CPU Engram, exact rank-zero host result broadcast | 16 | 98.737 s | 0.162046 |
| Same residency plus raw K-lane CPU SIMD, packed-byte prefetch and exact FP8 floating-significand fragments | 16 | 85.355 s | 0.187453 |
| Same optimized operators with the private explicit NPU A/routed/B graph adapter | 16 | 82.287 s | 0.194441 |

The latest eager run is **2.04x** the initial control. The sustained graph run is approximately
**2.11x**, still roughly **257x short** of 50 TPS.
The initial profiler mistakenly performed the model's unused Gumbel sample; the optimized driver
resets state and sets `model.temperature=0` like normal greedy generation. This timing caveat is
retained: the ratio is against that initial measured control, not a clean causal ablation of only
one optimization.

For both optimized eager configurations and the sustained graph run, all eight ranks produced
the same 16 greedy token IDs as the control, and **every final finite FP32 logit bit matched**
(129,280 logits/rank). That is scoped
same-checkpoint/same-prompt/same-context implementation parity, not official TileLang reference
parity, model-quality evaluation or universal cross-device equivalence.

## What the profiling actually identified

The first two synchronized, nested instrumented forwards showed approximately:

- 1.8 s/token at the complete CPU Engram boundary;
- 1.17 s/token in original FP8 NPU GEMMs;
- 1.05 s/token in FP4 NPU GEMMs;
- 1.3 s/token in FP4 unpack;
- about 5.25 s/token inside all-reduce calls, **including rank wait/imbalance**.

The original 512 MiB global expanded-expert LRU held only **13 experts**. Its measured timed-window
hit count was **zero**. EP8 owns 1,920 experts/rank: their expanded INT8-plus-FP32-half-scale form
would need **71.19 GiB/card**, which does not fit. The original packed form needs **33.62 GiB/card**
(36,097,228,800 bytes), and every owned expert now resides in that one packed bank. Admission sums
all 40 layers **after dense loading**, requires an additional 12 GiB reserve, validates all raw
payloads before allocating the first arena, and allocates six code/scale arenas per layer.

In the latest eager nested profile, per token on rank zero:

| Scope | Seconds/token | Important qualification |
| --- | ---: | --- |
| Complete CPU Engram boundary | 0.0422 | Both Engram layers; CPU projection/lookup/gate unchanged |
| Compiled raw CPU projections | 0.0330 | Included inside the boundary above |
| Exact NPU FP8 floating-significand GEMVs | 0.7152 | 288 calls/token; same sequential-K rule |
| Resident packed FP4 projections | 1.4033 | No streamed expert read or full unpack in decode |
| All-reduce calls | 2.8817 | Includes waiting for other ranks, not a measured link floor |

These are inclusive nested instrumented measurements and must not be added as independent walls
or used in the sustained-TPS denominator.

## Proven levers and negative results

### Host Engram

The compiled scalar CPU kernel reduced one real `[25600,6144]` Engram projection from approximately
1.08 s to 25 ms at 64 threads. A raw-byte K-lane NEON implementation then measured **13.82 ms at
96 threads**, with all 25,600 output FP32 bits matching. It reads the original byte rows without a
weight prepack. Larger thread counts were not a free win: the subsequent 64/96/128/160/192 sweep
measured approximately 13.4/13.7/15.6/16.3/17.7 ms on this host.

The 4-byte prepacked CPU variant did not clearly beat scalar at 64/96 threads; the 2-byte compact
integer variant also lost to raw SIMD. Both remain unbound comparison artifacts. Apple-local
measurements and linear extrapolation to 192 host CPUs are **not** target-host performance evidence.

Eight ranks originally repeated the same host Engram work. Before rank-zero-only execution, the
helper runs the old CPU body on every rank and proves current activation/hash/mask/result byte
identity (bounded raw-byte comparison or whole-payload digest). Steady-state rank zero computes
once and HCCL broadcasts an **int8 byte alias** of the BF16 result; no floating arithmetic is applied
on the wire. Per-module proof state and span-sized buffers are retained. The real eight-rank HCCL
probe and full-model runs passed. Deduplication removes repeated work, not a presumed 8x wall factor.

### NPU exact projection kernels

- The first independent-K-fragment two-pass prototype passed **93 NPU tests**, but packed FP4 was
  approximately **49.6 ms versus 12.7 ms** over expanded INT8 operands. It was not adopted as a fast
  default.
- Constant nibble shifts and grouped fragments reduced packed w1 to about **18.6 ms**; 128 tile
  rows failed the actual 192 KiB unified-buffer budget, so the public grouped limit is 64 rows.
- Wider original-byte prefetch measured **16.47 ms for w1** and **16.42 ms for w2**, all output bits
  exact. Its dense FP8 variant mismatched **71 finite outputs** at `[1280,5120]`; that variant and
  the unvalidated dense-INT8 variant now refuse by name. No tolerance was introduced.
- Integer-significand fragments preserved the accepted result but took **62.9 ms grouped / 85.4 ms
  serial** for w1, slower than the packed FP32 path. They remain proof/comparison kernels, unbound.
- Exact floating significands removed the per-lane integer multiply and varying shifts. Packed
  FP4 still lost (**30.6 ms grouped**), so that branch is unbound; dense FP8 wq_a measured
  **4.37 ms versus 6.24 ms**, then passed the full-model final-logit comparison across all actual
  dense geometries reached by this benchmark.

The floating-significand product has at most eight significant bits and every window factor is
an exactly representable power of two; the change does not relax arithmetic. Normal-to-normal
E8M0 rescaling is guarded, applied twice in the original order, with the generic scale helper for
subnormal/overflow/Inf/NaN/byte-255 cases. K blocks are still accumulated sequentially.

### Graph boundaries

HCCL's isolated 88-collective probe measured **14.85 ms eager versus 3.55 ms captured** with exact
nonzero sums. This is not a full-model graph or a TPS result. Capturing an individual exact GEMV
only saved about **0.2 ms**: expanded FP4 12.62 to 12.45 ms, packed FP4 18.18 to 17.93 ms, and FP8
6.27 to 6.09 ms. Launch-provider validation itself was approximately 18 microseconds.

The NPU adapter reuses the existing A/routed/B split. Complete host Engram and Python expert
routing stay eager. A real original-checkpoint eight-step paired probe passed **every FP32 logit
bit at every position** on all ranks, including compressor phase changes. Rank zero's median
forward was approximately 5.00 s eager versus 4.71 s captured. Per-step verification copies make
that paired probe **not** the sustained graph-TPS benchmark. A separate no-per-step-logit-copy
benchmark then measured **16 forwards in 82.287 s, 0.194441 sustained single-stream TPS**, with the
same context 24..40, all 16 tokens and final FP32 logit bits matching the original control. Graph
capture and four warmups were excluded; NPU completion, position updates and host argmax were
included. This is only **3.7% faster** than the latest eager run, not a bridge to 50 TPS. The model
CLI and serving graph guards remain unchanged; these probes use a private explicit adapter
constructor.

## Explicit opt-in bindings

All production selection remains opt-in. The latest verified eager configuration uses:

```bash
export POCKETLLM_V41_MMA_EMULATE=1
export DEEPSEEK_V41_ASCEND_PACKED_RESIDENT=1
export DEEPSEEK_V41_ASCEND_PACKED_GEMV=prefetch
export DEEPSEEK_V41_ASCEND_GROUP_K=4
export DEEPSEEK_V41_ASCEND_ROWS_N=16
export DEEPSEEK_V41_ASCEND_DECODE_GEMV=1
export DEEPSEEK_V41_ASCEND_FP8_GEMV_IMPL=float_significand
export DEEPSEEK_V41_CPU_EXACT_GEMV=1
export DEEPSEEK_V41_CPU_GEMV_IMPL=raw_simd
export DEEPSEEK_V41_CPU_GEMV_THREADS=96
export DEEPSEEK_V41_ENGRAM_RANK0=1
export RELIC_CPU_RAW_SIMD_GEMV_CACHE=/path/to/private/writable/cache
```

The cache path is a local deployment setting, not a universal host path. A missing compiler,
invalid metadata, unsupported backend or failed kernel raises by name rather than silently
switching reduction or precision. CPU/CUDA default paths and OpenAI-serving capability are not
changed by these flags.

## User-authorized native accumulation experiment (November 19)

The owner explicitly permitted **testing different accumulation rules**, not replacing the exact
production default. A separate private driver retains the original checkpoint, original FP8
activation quantizer, exact FP4 INT8-times-two/half-scale operand pair, and complete CPU Engram.
The SDK mixed-weight Cube operator applies the original weight scales before its natural full-K
accumulation. BF16 input dense projections use exact original decoded operands; wider-output
projections retain the original path to avoid an additional BF16 final-store rounding.

The first completed full-model run measured **16 forwards in 73.677 s, 0.217164 TPS** on the slowest
of eight ranks, context 24..40, with four warmups and quality copies excluded. All ranks agreed on
the candidate tokens and final FP32 logits. This remains about **230x short of 50 TPS**.

This candidate is **not numerically equivalent** to the original path. In eight teacher-forced
positions on each rank, all 64 candidate argmaxes matched the same-context exact control, but the
largest FP32 logit difference was **2.03074**, the largest relative L2 difference **0.16701**, and
the largest reference-to-candidate KL **0.00177742**. Free generation later diverged: only six of
16 same-position tokens matched the exact run. One easy counting prompt is not a model-quality
benchmark, and matching teacher-forced argmaxes is not permission to call the outputs identical.
Nothing in the CLI, serving guard or default exact dispatch was replaced.

Operator-level evidence also includes negative results: a fused certified-BF16 lookup Cube path
compiled and produced finite original-weight outputs but took approximately **106 ms for w1 /
102 ms for w2**, worse than the exact kernels. It remains unbound. The simpler block32 Cube path
passed its constant smoke, while several other tile/container combinations exceeded the actual
192 KiB unified-buffer budget. No compiler-budget bypass was applied.

## Further exact scheduling and unpack improvements

Persistent CPU workers preserve every original reduction operation. On the real `[25600,6144]`
Engram projections at 96 threads they measured **6.434 ms / 6.678 ms**, versus **13.248 ms /
12.953 ms** with per-call thread creation. All 25,600 FP32 output bits matched on both layers;
49 target-host tests and 49 portable tests passed. These are projection times, not model TPS.

Static FP4 nibble decoding also preserved every widened byte. The original 11,796,480-byte w1
output took approximately **15.18 ms** with variable shifts, **2.78 ms** with a static ladder, and
**2.50 ms** with a single little-endian INT8-pair lookup/store per packed byte. All 256 packed-byte
values and the full original matrix were checked. A larger static tile overflowed the real unified
buffer and was not adopted.

A private host-prefetch manager prepares current-token lookup/WKV on CPU before the backbone's
first layer. This is causally valid because WKV depends on the already-computed CPU hash IDs and
fixed CPU weights, not the current hidden state; the unchanged original Engram body computes the
gate after its hidden-state activation arrives on CPU. Thirteen portable tests pin original result
bits, masks, stale hashes, bounded futures and error/cleanup behavior. The combined private run
(native accumulation, exact paired widening, persistent CPU workers and current-token host
prefetch) then measured **16 forwards in 21.495 s, 0.744370 TPS**. All eight ranks' same-context
exact controls, native teacher-forced logits and native final logits matched the preceding
0.217164-TPS candidate **bit for bit**. Thus these scheduling/unpack changes improved that
candidate by about **3.43x without adding numerical differences**; the candidate still differs
from the exact original reduction and remains roughly **67x short of 50 TPS**.

## Private graphs, bounded operand cache and HC reductions

Capturing the NPU portions of that candidate, with host Engram and expert-ID dispatch still eager,
then measured **16 forwards in 14.444 s, 1.107696 TPS** on the slowest rank. Each of the eight
teacher-forced candidate positions and the final FP32 logits matched same-mode eager execution
bit for bit on all eight ranks. Capture time was excluded, not amortized into the decode window.

A further isolated run added a **4 GiB/card LRU of exact INT8-times-two operands** beside the
original packed bank, plus a fused FP32 HC normalization loop. The cache does not requantize or
retain a second full expert bank. Original FP32 HC weights, sigmoid, softmax, epsilon and iteration
count remain unchanged, but the fused normalization has native FP32 reduction/division behavior:
12/13 of the 16 combination coefficients differed by at most `1.19209e-7` / `5.96046e-8` in the
original-weight operator probes. It is therefore not described as an exact scheduling change.

This combined numerical experiment measured **16 forwards in 9.779 s, 1.636090 TPS**, context
24..40, batch one, complete CPU Engram, NPU completion and host argmax included. All ranks agreed
on candidate tokens and final logits, and the private graphs matched their own eager candidate.
All 64 same-context teacher-forced argmaxes matched the exact control, but the largest logit
absolute difference was **2.173757**, relative L2 **0.156518**, and KL **0.00133455**. Free generation
differed from both the exact path and the preceding native candidate. These are small diagnostic
runs, not a quality benchmark or production adoption recommendation.

The explicitly timed window is contiguous and includes cache misses; its later steps were slower
than its first steps. Neither the best step nor a cache-hit-only subset was substituted for the
sustained result. New vector widening and fused exact activation decoding are being tested next.

## Exact vector widening and activation fusion

The next vector FP4 decoder removed every global lookup from the widening arithmetic. Both full
original w1 and w2 matrices matched all 11,796,480 INT8 output bytes, and the selected tile sizes
passed **32 target-host tests** (the unmeasured 4096 tile was explicitly deselected). At tile 2048,
w1/w2 widening measured **0.387 ms / 0.390 ms**, versus **2.480 ms / 2.468 ms** for the global
paired lookup. The local-LUT alternative was correct but slower, so arithmetic is the private
experiment's selected variant.

A fused activation operator consumes the original FP32 amax/floor/IEEE scale-ceiling/E4M3 RNE
rules and immediately decodes the resulting original code/grid pair to BF16. It is not an INT8
activation or a new quantizer. Seven target-host batteries checked BF16 output bits, E4M3 code
bytes and E8M0 scale bytes against the original two-launch form, including real row widths,
prefill rows, signed zeros, tiny magnitudes and nonfinite blocks; every difference count was zero.
With original byte buffers retained for the provider seam, one `[1,5120]` row measured **0.349 ms**
versus **0.609 ms** for two launches. The binding leaves every CPU Engram quantizer and decoder
on the original CPU path and restores the global NPU provider seams on close.

Combined with the preceding native/HC/cache experiment, these **bit-preserving** changes measured
**16 forwards in 6.968 s, 2.296077 TPS** on the slowest of eight ranks, context 24..40, batch one.
All eight ranks' exact teacher controls, native candidate teacher logits and native final logits
matched the preceding 1.636090-TPS candidate **bit for bit**. The inherited native-accumulation/HC
numerical differences remain; these new unpack/activation changes did not add differences.

These native-candidate timings are **fixed-length forward workloads**, not normal EOS-stopped
service responses. The candidate emitted EOS at timed position 8 (the ninth token), and the private
driver continued through its 16-forward quota. This is stated rather than pricing only the fastest
prefix or silently presenting post-EOS continuation as useful generated text. Normal serving
termination and capability guards remain unchanged.

## Recorded expert bodies, static-shift activation and M1 routing

Three further **bit-preserving** changes were measured on the full checkpoint:

* **Recorded routed-expert bodies.** Each resident expert's original
  `packed_expert_forward` body is recorded once into one shared graph pool and replayed from staging
  buffers outside the pool. 1,920 graphs per rank (40 layers x 48 owned experts) captured in 21.2 s,
  excluded from the timed window. Host expert-ID routing stays eager; a call that does not match a
  recorded expert triple runs the unchanged eager body instead of a stale replay.
* **No variable bit shifts in the fused activation pass.** The released E4M3 subnormal encode is the
  exact power-of-two scale `|value| * 512`, and the released `1e-4` amax floor with `fp32(1/448)`
  makes the smallest producible E8M0 byte 105, so every produced nonzero value is a BF16 normal and
  the decode becomes a constant-bit ladder. Seven target-host batteries reported zero E4M3 byte,
  E8M0 byte and BF16 bit differences, including signed zeros, subnormals and nonfinite blocks.
  One `[1,5120]` row measured **0.299 ms** against **0.635 ms** for two launches.
* **M1 routed dispatch.** With one row, the constant row index and its `index_select`/`gather`/
  `index_put` copies are replaced by a one-column slice, the strictly ascending expert-id FP32
  summation unchanged. The real layer-0 store matched 0 result bits and measured 12.42 ms against
  13.74 ms.

The combined candidate measured **16 forwards in 3.242 s, 4.934599 TPS** on the slowest of eight
ranks, context 24..40, batch one, complete CPU Engram, NPU completion and host argmax included. All
eight ranks agreed on tokens and final FP32 logits, and every exact control, teacher-forced candidate
and final logit matched the preceding 2.296-TPS candidate **bit for bit** — so these three changes
added no numerical difference. The inherited native-accumulation/HC differences are unchanged.

Honest limits of this number: the workload is fixed-length forward work, the candidate emitted EOS at
timed position 8 and the driver continued through its 16-forward quota, and the graph pool's real
footprint is **not** measured — `memory_allocated`/`memory_reserved` do not expose NPUGraph pool
memory, so the round stayed inside its 512 MiB refusal bound without that bound being a measurement
of what the graphs occupy.

## Where the remaining time is, measured rather than assumed

The 16-forward window takes 3.242 s, i.e. **202.6 ms per token** across 40 serial layers (~5.1 ms per
layer). Three probes bound the candidates, and one short-window ablation splits the token directly.
The ablation deliberately breaks the numbers to attribute cost; each variant is labelled as a timing
ablation and none is a candidate. Eight forwards, three warmups, all eight ranks agreeing:

| Variant | TPS | ms/token | Cost it removes |
|---|---|---|---|
| same configuration, shorter window | 5.081 | 196.8 | — (baseline) |
| no per-layer device-to-host routing read | 5.659 | 176.7 | 40 routing drains: **20.1 ms (10.2%)** |
| no routed-expert work at all | 10.705 | 93.4 | expert bodies, replays and the routing read: **103.4 ms (52.5%)** |

So **the routed experts are half the token**, and the recorded bodies are individually the largest
single item; the per-layer routing drain is a further tenth.

* **Collectives are not the bottleneck.** A transport probe timed the model's own message — one
  `[1, 5120]` FP32 all-reduce across all eight ranks — outward and inside a captured graph. All eight
  ranks returned exactly the expected sum `36.0`, at **0.0939 ms captured** against **0.4149 ms
  eager**. The layer already carries its reductions inside the recordings, so even at two collectives
  per layer this is about **7 ms of a 203 ms token (≈3.6%)**.
* **Host Engram is small.** The instrumented eager boundary measured **7.8 ms of a 203 ms token**,
  complete on the CPU.
* **The bounded operand cache loses to the recorded bodies.** Running the same candidate with a 4 GiB
  per-expert decoded-operand cache and *without* the recorded expert bodies measured **3.169 TPS**
  (315.6 ms/token) against **4.935 TPS** with the recordings. Widening an expert once and keeping it
  is worth less than recording the body, because the cache cannot hold 1,920 experts (about 69 GB of
  INT8 operands against 64 GiB per card) and misses pay the full widening anyway.

### Why the 4-bit weights cannot skip the widening

The obvious way to make the expert bodies cheaper is to hand the original 4-bit payload straight to a
hardware W4A16 matmul. That is not possible with this checkpoint's format, and the reason is
arithmetic rather than a missing operator: the released E2M1 code-to-value map is
`[0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12]` in units of the group's half scale, and it is **not**
two's-complement int4. Hardware int4 ranges over -8..7 and cannot represent the value 1.5 that code 1
denotes, so the widening to INT8 (with the value scaled by two) is what makes the original payload
exact. A probe confirmed the installed build's weight-quant matmul does expose a `weight_dtype`
selector, but the encoding mismatch means it cannot consume these codes; the widening is inherent to
the original format.

## The widening tile was leaving a third of its throughput on the table

The released arithmetic widening was timed across its documented tiles on both expert geometries. It
is not arithmetic-bound at the old default: at tile 512 it costs the same as a copy kernel that does
no decoding at all (0.5428 ms against 0.5426 ms), so the per-program overhead was the whole cost. The
larger tile removes it, with full byte identity against the released widening at every tile:

| tile | w1/w3 (2304x2560) | w2 (5120x1152) |
|---|---|---|
| 1024 | 0.4724 ms | 0.4578 ms |
| 2048 | 0.3854 ms | 0.3835 ms |
| 4096 | **0.3478 ms** | **0.3418 ms** |

4096 is now the library default for the arithmetic mode, and 8192 aborts the Ascend Triton compiler,
which makes 4096 the ceiling rather than an unmeasured option. Carried into the full model this moved
the verified candidate from **4.934599 to 5.151722 TPS** (194.11 ms per token, the optimistic end
of a 5.03 to 5.16 spread; see the honest range below) with the final FP32
logits of all eight ranks **bit-identical** to the previous candidate and zero teacher/graph
differences, so the change added no numerical difference.

## What one recorded expert body is made of, and why it cannot get much cheaper

Timed per stage with the repeats inside one timed region, so a synchronize is amortized instead of
charged to each stage. The matmul is a small part of the body:

| stage (per projection) | ms | share of the projection's device work |
|---|---|---|
| FP4 code widening to INT8 | 0.212 | 38% |
| activation decode to BF16 | 0.247 | 45% |
| transpose to the operator's `[K, N]` | 0.051 | 9% |
| native weight-quant matmul | 0.043 | 8% |
| **sum** | **0.552** | |

Three projections make 1.66 ms of the body's measured ~2.4 ms; the remainder is the activation
quantizations, the SwiGLU clamp and gate, and the routing weight. Two consequences:

* **Making the matmul faster is nearly pointless** — it is under a tenth of the projection. The cost
  is data movement and per-kernel dispatch at M=1, not arithmetic.
* **Two avenues were measured and closed.** Handing the original 4-bit payload to a hardware W4A16
  matmul is arithmetically impossible for this checkpoint (the E2M1 code-to-value map is not
  two's-complement int4). And the widening is not element-width bound: a two-bytes-per-lane rewrite
  that loads one 16-bit word and stores one 32-bit word measured 0.2493 ms against the released
  0.2770 ms — about 10%, with byte identity — while the four-byte form aborts the Ascend Triton
  compiler. Both the released kernel and a copy that decodes nothing sit at 64-84 GB/s, roughly a
  tenth of the card's bandwidth, so that is where this Triton path caps out.

## One replay at a time: what the per-layer chain is actually made of

The ablation says removing all routed-expert work leaves 93.4 ms, but it does not say where that goes.
Timing a single recording in isolation does, by replaying it back to back inside one timed region and
synchronizing once (state is reset afterwards, and the run's final logits stayed bit-identical):

| recording | one replay | times per token | per token |
|---|---|---|---|
| layer A — everything through the gate | **1.386 ms** | 40 | **55.4 ms** |
| layer B — shared experts, reduce, fold | 0.144 ms | 40 | 5.8 ms |
| one routed expert body | 0.832 ms | ~30 replays | 25.0 ms |

Three things follow, and they close the question of where the last order of magnitude is:

* **The attention half of a layer is the single largest item in the token** — 28% of it — and it is
  inside a recording, where no Python runs. It is device work: the indexer's scoring and selection,
  the hyper-connection mixes, the KV bookkeeping and the sparse attention itself. The structurally
  similar B recording, which also carries an M=1 dense feed-forward, costs ten times less, so the
  cost is specific to what A contains rather than to replay or launch overhead.
* **Replay overhead is not the problem.** 80 layer replays plus 30 expert replays account for about
  86 ms, and merging A into B would save at most the fixed launch part of that, which the 0.144 ms B
  replay shows to be small.
* **The ablations' expert saving (103 ms) exceeds the 25 ms of expert replays plus the 20 ms of
  routing reads**, so most of that saving is the serialization the dispatch introduces: each expert's
  replay cannot start until the previous one's staging copy lands, which the isolated back-to-back
  timing cannot see.

## The host blocks on the device, it does not work

Wrapping the routed store's two host-side halves with `perf_counter` and adding no synchronisation
gives the split directly, and it corrects the reading a smaller ablation could suggest:

| host measure | per token |
|---|---|
| wall clock, whole forward | 190.9 ms |
| time inside the routed store's `forward` | 202.0 ms (inclusive of the read) |
| of which blocked in the device-to-host routing read | **169.5 ms** |
| of which other host work (non-blocking) | **32.5 ms** |

The routing read blocks the host for 169.5 ms of a ~194 ms token — 87% of it — and yet the ablation
that removed the read entirely saved only **20.1 ms**. Both numbers are right, and together they say
something important: **the host is waiting, not working.** The block is where the device's latency
becomes visible, and the work it waits for happens whether or not the read is there. The host's own
non-blocking work is 32.5 ms per token, and only about 20 ms of the wait is genuinely recoverable
without changing device work.

So the earlier framing of that 20 ms as "the sync cost" was too generous to the host: the transfer is
not expensive, the device is simply the bottleneck, and the host is blocked on it for most of the
token. Any further gain has to remove device work or device dependencies, not host code.

A caveat that applies to the replay table above: those timings replay a recording back to back, so
they measure pipelined throughput and are a **lower bound** on what the same recording costs when its
result is needed by the next dependent step. The expert ablation's 103 ms is the in-situ figure and is
larger than the 25 ms the isolated replays suggest for exactly that reason.

## The launch flood is the numerical contract, not a bug

Asking *which* call site issues the thousands of small kernels gives the clearest answer on this page.
Grouping the profiler's ATen table by operator and input shape over an eager decode step:

| shape converted | what it is | conversions per token |
|---|---|---|
| `[1,1,5120]` | hyper-connection stream | 328 |
| `[1,5120]` | sublayer output | 297 |
| `[1,1,8,64]` | 64 attention heads / TP8 x head_dim 64 | 276 |
| `[1,32]` | routing weights | 264 |
| `[1,2304]` | MoE intermediate | 224 |
| `[1,288]` | MoE intermediate / TP8 | 200 |
| `[1,1,4,5120]` | hyper-connection four-copy stream | 167 |
| `[1,1,8,32]`, `[1,1,32]` | indexer per-head tensors | 248 |

That is about **5,700 dtype conversions and 3,000 copies per token — roughly 142 per layer**, spread
across attention, the MoE block and the hyper-connections. There is no single redundant call site to
clean up, and the casts are not sloppiness: the reference accumulates in FP32 and narrows deliberately,
which `attention.py` documents at its norm and rotary sites. Removing them would change resolution
bits.

Not all of that count is device work, and the difference matters. A dispatch that only rewrites
strides — `permute`, `view`, `unsqueeze`, `slice` — costs host time and emits **no kernel at all**.
One `sparse_attn` call issued 110 dispatches of which 59 were exactly that, and hoisting the operand
layouts out of its block loop (bit-identically, against both the previous form and the recorded
released-kernel fixture) cut the call to 70 and the token by 1,600 dispatches — and moved TPS not at
all, 5.1052 against 5.09 to 5.14 for the unchanged tree. So the host dispatch count is not this
model's limiter either.

## Corrections to attribution and feasibility claims (November 20)

The earlier conclusion that fusion or device-owned expert selection must change the ascending
expert sum was unsupported and is withdrawn. Device kernels can sort the selected global IDs and
retain the original FP32 addition order. Complete CPU Engram does not require MoE routing IDs to
be copied back to CPU on every layer.

The corrected inclusive dispatch measurement is 172.35 ms/token, with 122.34 ms inside routing
calls and 21.12 ms in the rest of expert dispatch, over exactly 40 layers. **Time inside a blocking
call is not time caused by that call.** The bypass experiments suggested about 10–12% in their
modified runs, not a recoverable 71% of the original workload. They change routing/workload and
cannot establish an exact causal decomposition or a hardware ceiling. The earlier 1.25x warmup
accounting error is retained in the historical receipts, not used as a current measurement.

The 93.4 ms expert-off ablation is also not a lower bound for another implementation. It deletes
model work and changes the execution graph. Neither it nor an eager profiler establishes that
50 TPS is impossible. The previous hard-ceiling claims, the supposed closed 67 ms accounting,
and the assertion that a faithful fused attention is impossible are withdrawn. Attention fusion
remains unimplemented and unvalidated here; a different CUDA implementation is not proof against
all possible exact implementations.

Two other conclusions were unsupported:

- The transpose-view comparison timed the SDK operation after materialization. Equal operation-only
  latency does not prove an internal copy or rule out end-to-end preparation savings.
- 17.7 MB divided by 15.1453 **milliseconds** is approximately **1.17 GB/s**, not 1168 GB/s. That
  probe also targeted the old unpack implementation, not the active vector widening substitution.
  It cannot establish that the active kernel saturates HBM or that this optimization space is closed.

## Actual fused operators and device selection

The new opt-in implementation performs cyclic ownership selection, duplicate/range validation,
ascending-global-ID ordering and routing-weight association entirely on device. A sticky device
error poisons the final result for invalid routes; the finite-logit serving guard is not bypassed.
Dynamic graph replays with 0, 1, 3 and 6 owned selections passed, including input permutations.
The Torch-dispatch audit observed **zero MoE routing D2H copies** in that replaced decode body.
This does **not** mean zero total model D2H: complete CPU Engram and host argmax still require their
own boundary transfers. Prefill remains the original path.

The first design prepared selected BF16 matrices for native grouped Cube. It was numerically
valid on the tested fixtures, but full-model performance was **0.296793 TPS**, so it is rejected as
an optimization. A stage profile measured 32.53 ms in the preparation kernel versus 39.1 us in the
native grouped matmul; this is a measurement of that design, not a universal bandwidth limit.

The revised stage emits the **original INT8-times-two values and original BF16 half-scales** into
one shared bounded temporary. It does not rewrite or requantize the packed checkpoint. Gate and up
share one wider native Cube call because all selected experts consume the same activation row;
down retains the distinct intermediate activation rows. The released geometry uses **150,466,668
bytes** of shared stage scratch, not a second resident expert bank. Physical graph-pool memory is
not inferred from the allocator delta.

There are actual fused clamp/widen and weighted FP8 encode/decode device kernels. The latter keeps
an explicit BF16 RNE boundary before the original block-32 amax, E8M0 scale and E4M3 code selection.
The original FP32 SiLU and multiplication are retained: a native SwiGLU pilot passed small fixtures
and 64 teacher-logit comparisons but its independent 16-token sequence differed, so it is **not**
accepted as proof of full-context numerical neutrality. Its throughput, 5.670944 TPS against a
matched 5.723332 TPS native baseline, did not establish a speedup either.

The revised original-SiLU/I8-stage path completed all eight ranks: **3.821369 TPS** over
16 synchronized forwards after four excluded warmups, context 24..40. It observed zero replaced
MoE routing readbacks and kept complete CPU Engram, but was slower than the matched **5.723332 TPS**
baseline. It remains experimental and is not recommended as a performance optimization. Seven
original-activation byte probes, including two million-scale randomized cases, matched FP8 codes,
E8M0 scales and decoded BF16 bits. This is scoped validation, not universal model-quality parity.
No microbenchmark is extrapolated to token throughput. The target remains **unachieved**, the exact
default and serving guards remain unchanged, and neither rejected pilot is recommended for adoption.

## vllm-ascend-informed original-byte tile consumer (November 20)

Source inspection is pinned to
[`a38556f69529595e70598d0d6203009f5ac3293e`](https://github.com/vllm-project/vllm-ascend/tree/a38556f69529595e70598d0d6203009f5ac3293e).
The useful design is device-owned group scheduling plus bounded Vector/Cube operand preparation,
not an interchangeable `int4` codec:

- The [A2 device adaptor](https://github.com/vllm-project/vllm-ascend/blob/a38556f69529595e70598d0d6203009f5ac3293e/vllm_ascend/device/device_op.py#L418-L460)
  explicitly rejects MXFP MoE. Its integer GMM/SwiGLU/dynamic-INT8 quantizer is not the original
  E2M1/E8M0 + E4M3/E8M0 contract. Native A5 MXFP kernels cannot simply be selected on 910B.
- The [token dispatcher](https://github.com/vllm-project/vllm-ascend/blob/a38556f69529595e70598d0d6203009f5ac3293e/vllm_ascend/ops/fused_moe/token_dispatcher.py#L423-L457)
  carries device group counts and permutation metadata into GMM. Our cyclic EP8 deal is not its
  contiguous active-expert range: the existing device selector retains cyclic ownership and sorts
  global expert IDs without a routing readback.
- The [MXFP4 loader](https://github.com/vllm-project/vllm-ascend/blob/a38556f69529595e70598d0d6203009f5ac3293e/vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py#L328-L375)
  explicitly transforms packed and scale layouts and performs an actual NZ cast. An ND reshape is
  not NZ. Our resident representation uses the existing invertible 16-row byte permutation,
  preserving every original code and scale byte and replacing, not duplicating, the row-major bank.
- The [A2 BF16 consumer](https://github.com/vllm-project/vllm-ascend/blob/a38556f69529595e70598d0d6203009f5ac3293e/csrc/mc2/dispatch_ffn_combine_bf16/op_kernel/dispatch_ffn_combine_bf16_kernel.hpp#L377-L453)
  loads device group counts, skips empty groups, and coordinates tile lifetimes. Several upstream
  vendor headers use the CANN Open Software License, not blanket Apache licensing; the new kernel
  is independently written rather than copied.

### Implemented 910B path

The first same-kernel Triton local-unpack/Cube prototype passed small operand and dynamic-routing
checks without expanded global weight scratch, but six-active-slot projection replays cost
**84.24–104.00 ms**. It was not promoted to a full-model replacement.

The independent AscendC implementation explicitly schedules one AIV producer per AIC consumer.
On the installed 910B CANN 9.2 SDK, UB-to-L1 through that API uses a Matmul software service; the
A5 direct-handoff assumption failed its small gate. The validated A2 implementation therefore uses
`original packed bytes -> AIV local decode -> small double-buffered GM ring -> AIC L1/L0 -> Cube`.
The ring has **two 16-by-128 BF16 tiles per worker**, **196,608 bytes total at 24 workers**. It does
not hold an entire selected matrix. K128 tiles retain four independent original K32 scale rows;
there is no regrouping of the stored scale grid. Each tile decodes the exact doubled-INT8/half-scale
value before BF16 Cube consumption, with the mandatory E8M0 range certificate.

The original FP8 activation encoding/decoding and original FP32 SiLU are still used. Selected
expert contributions narrow to BF16 and combine in ascending global-expert FP32 order. Native K
accumulation is the previously authorized numerical experiment, not the exact default.

Validation of the final library (`0cd9111e373859c5b8d4ec09089a952467f26b6c88a3fb387842b11c0725b287`):

- Every packed byte/nibble at every certified E8M0 scale: **128,512 BF16 values, zero differing bits**.
- **128 dynamic projection graph replays**, owned counts 0/1/3/6, zero differing fixture bits;
  invalid source IDs poison output and empty slots are overwritten with zero.
- Whole MoE composition: **32 graph replays**, original quantization/activation and ordered combine,
  zero fixture FP32 output differences and zero routing D2H under the dispatch audit.
- Six-active-slot synchronized projection replay: **2.754 ms** for `[N=2304,K=5120]`, **2.826 ms**
  for `[N=5120,K=2304]`, 16 replays each. These are projection timings, **not model TPS**.
- Core layout/native-binding CPU tests: **21 passed**. Runtime original/tile routing tests:
  **11 passed**. The root also ran **186 passing surrounding regression cases with four
  hardware-gated skips**, plus **14 passing nonzero whole-MoE/scratch-guard cases**. These are
  targeted runs, not a claim that the entire repository suite passed.

### Full-checkpoint result: improvement over staging, still a regression against the baseline

The full 40-layer TP8/EP8 model ran four excluded warmups followed by **16 timed, synchronized
single-stream batch-one forwards**, context 24 through 40, complete CPU Engram and host argmax
included. All eight rank receipts were validated:

- Tile consumer: **4.777145 TPS**, slowest-rank wall **3.349281 s**.
- Earlier same-scenario original-I8 whole-matrix staging experiment: **3.821369 TPS**.
- Earlier best-known same-scenario native-Cube baseline: **5.723332 TPS**.

These are separate recorded runs, not a same-hour causal ablation. The tile run's 16 token IDs
match both earlier native runs; all eight final outputs are finite. Graph/eager parity is zero at
each of the eight teacher positions on all ranks. The **64/64 teacher greedy matches** against the
original exact path do not establish bit-neutrality: changed native accumulation still gives maximum
logit absolute difference **2.173757** and reference-to-candidate KL **0.001334548**.

Explicit shared workspace is **430,188 bytes**, including the 192 KiB ring and projection outputs,
versus the preceding **150,466,668-byte** I8 staging workspace. The original resident byte count is
still **36,097,228,800 bytes/rank**. Physical graph-pool allocation is not measured and is not
included in this explicit scratch count. Zero **routing** D2H does not mean zero total model D2H:
complete CPU Engram and host greedy selection retain their required boundaries.

**The 50 TPS objective is not reached.** This remains an explicit experiment, not a replacement for
the faster native baseline or the exact production default. After the bounded run, all eight NPU
process tables were empty, the 90 checkpoint stat identities and shared CANN settings were
unchanged, and the fleet lease was released.

Evidence, per-rank receipts, measured source snapshots and hashes are retained in the
[tile-consumer manifest](evidence/ascend-decode-tile-consumer-20261120/manifest.json) and
[validated comparison](evidence/ascend-decode-tile-consumer-20261120/comparison.json). After freezing
that run, a host-only metadata guard was added to reject integers that would truncate in the native
uint32 C ABI. The measured device binary and arithmetic are unchanged; no post-guard model TPS
measurement is claimed.

## Current tile-path bottlenecks: matched all-rank profiling (November 20)

This diagnosis uses two fresh sequential runs on the same leased 910B3/CANN 9.2 host: the current
tile consumer and the native baseline. Neither run changes device arithmetic for profiling. Both
keep original checkpoint bytes, FP8/E8M0 activations, complete CPU Engram and existing serving
guards. The **16 timed forwards after four excluded warmups** cover context 24..40, including NPU
completion and host argmax. All eight ranks produce the same tokens in both runs. The excluded
route telemetry also confirms identical expert choices on every layer/rank in the diagnostic pass.

| Uninstrumented configuration | Slowest-rank wall, 16 forwards | Mean wall per forward | TPS |
| --- | ---: | ---: | ---: |
| Current native tile consumer | 3.342140 s | 208.884 ms | 4.787352 |
| Native baseline | 2.803461 s | 175.216 ms | 5.707230 |

The current consumer therefore regresses by **33.667 ms/forward** in this matched workload. The
comparison is same-host and same-route, but sequential trials, not a randomized statistical study.

### Diagnostic passes and scope

After the TPS window, the driver restores a **valid context-24 KV and hash snapshot plus its next
input token**, not a context-20 snapshot with a mismatched start position. It separately records:

1. **Eight forwards with NPU events** around graph A, routed experts, graph B and complete CPU
   Engram; device routing is cloned D2D outside the scoped region and copied back only after the
   diagnostic pass. This telemetry is not a hot routing D2H or part of the TPS denominator.
2. **Four forwards with the CANN CPU/NPU profiler on all eight ranks**, avoiding a single profiled
   rank's setup skew. CPU forward markers are checked against kernel timestamp domains before any
   attribution. Profiler overhead still exists and these are not TPS measurements.

Both diagnostic passes reproduce the uninstrumented window's tokens. Event/profiler times are
separate views, not additive latency components. Event instrumentation perturbs the host-routed
baseline more than the device-routed tile configuration.

### Primary bottleneck: original expert tile consumption, amplified by EP skew

The eight-forward event pass selects the **rank with the longest routed scope on each layer**,
then reports that same rank's A and B scopes. It deliberately does not independently maximize B,
which would count another rank's waiting time again:

| Scope on the selected rank per layer | Current tile path, summed per forward | Baseline diagnostic pass |
| --- | ---: | ---: |
| Graph A: attention + router + HC | 51.684 ms | 52.612 ms |
| Routed experts | 139.121 ms | 112.950 ms |
| Graph B: shared expert + reduce + HC | 5.551 ms | 5.552 ms |

These are **stream/dependency intervals**, not pure compute or an exact decomposition of the
16-forward wall. Nevertheless, the CANN device trace independently identifies the consumer as the
largest computational change. On rank zero:

- The current native tile kernel runs **120 times/forward**, totaling **50.725 ms**.
- The baseline instead spends **12.899 ms** in vector FP4 widening, **5.598 ms** in weight-layout
  transposes, **3.794 ms** in native weight-quant matmul and **1.588 ms** in half-scale preparation.
- Total non-collective kernel duration sums increase from **83.628 to 112.440 ms/forward** on rank
  zero. Across ranks the ranges are **80.635..85.132 ms** for baseline and **107.811..115.885 ms**
  for tile. These sums are profiler execution statistics, not global duty cycles.

Rank-zero mean work is not the model's critical expert work. Topk6/EP8 averages **0.75 expert per
rank per layer**, but the **maximum** rank owns **2.1 experts per layer on average**. Among the
320 observed layer/forward cases, the maximum counts are one/two/three/four experts in
**36/220/60/4 cases**. This is a **2.8x maximum-to-mean ownership ratio**; it is not a speedup
promise for rebalancing. The next layer cannot proceed until the busy rank finishes.

The four-forward trace verifies every native projection triplet against the observed owned counts.
Summing each layer's slowest rank's three native projections gives **134.485 ms/forward**, divided
almost equally between gate/up/down (**44.658/44.625/45.203 ms**). Every such critical rank has the
maximum owned expert count. Three-projection time per layer is approximately **0.105/1.672/3.176/
4.711/6.209 ms** at zero/one/two/three/four local experts. Empty-source launch work is real but small:
rank zero pays **1.993 ms/forward** for empty tile calls, not the dominant 48.732 ms of active ones.

### Why the fused consumer remains expensive

The native tile kernel's duration-weighted CANN ratios on rank zero show **AIC MAC 3.7%** and
**AIV scalar/control 95.4%**, with AIV vector 61.1%. Pipeline ratios can overlap and the scalar
figure includes synchronization waits; they are not wall percentages or independently recoverable
costs. They do show that this is not a saturated-Cube arithmetic workload. The implementation has
per-Ktile producer/consumer flags, single-slot input reuse fences, repeated LUT/index initialization
at each projection launch and a global-memory handoff of decoded BF16 tiles.

The **192 KiB ring is a capacity bound, not a traffic bound**. From the actual original matrix
geometry and six-expert routes:

- Original packed code/scale reads average **564,019,200 bytes/rank/forward** (about 0.564 GB).
- BF16 ring write **plus** read averages **4,246,732,800 logical bytes/rank/forward** (about 4.25 GB).

These are **geometry-derived logical accesses**, not hardware DRAM counters: caches may change
actual off-chip traffic. They do not establish a bandwidth ceiling. They explain why reducing
scratch capacity from 150 MB to 430 KB alone did not remove expanded-weight data movement.

The next consumer investigations should therefore target a **910B-supported handoff that avoids
BF16 GM write/read**, reduced per-tile scalar/event serialization and initialization, and combined
gate/up scheduling while preserving the original quantization and BF16 boundaries. Any SDK-based
alternative must be checked for hidden staging, not assumed to be a zero-traffic replacement.

### Secondary region: attention/router/HC small-kernel chains

Graph A stays close to **52 ms/forward** in both diagnostic runs. In the current rank-zero device
trace, ordinary MatMul kernels together cost **8.700 ms**, while common small families include
multiply **5.116 ms**, casts **4.035 ms**, FP8 activation encode/decode **3.908 ms**, BatchMatMul
transposes **3.382 ms** and ReduceSum **2.639 ms**. The full graph emits thousands of small
vector/reduction/movement kernels (13,183.5 non-collective calls/forward in the four-forward trace).
Those families span A and B, so their global totals must not be presented as an attention-only
breakdown. They identify the next fusion/layout opportunities after expert consumption rather than
a reason to tune GEMM peak FLOPS alone.

### What the waiting clocks do not establish

On rank zero, the current graph B event scope totals **87.834 ms/forward**, whereas B on each
layer's busiest routed rank totals **5.551 ms**. Other ranks have less expert work and reach the
all-reduce earlier, so their B/HCCL durations expose waits for the busy ranks. Calling the roughly
83 ms difference an independent network cost would double-count the MoE critical path.

The complete CPU Engram stream boundary is **8.729 ms/forward**, compared with **9.186 ms** on
baseline. Rank-zero exposed prefetch wait is **4.672 ms** (baseline **4.539 ms**), included within
its boundaries rather than added separately. Two CPU lookup/projection workers average
**11.937 ms each**, much of which overlaps earlier NPU work. The host-inclusive Engram method
clock reads 59.481 ms on the tile path because it also waits for queued device work; it is not
59 ms of CPU computation or removable D2H. Engram remains on CPU and is not the primary new
bottleneck.

The current priorities are **expert consumer efficiency and per-layer EP critical-rank work first,
then graph-A fusion/layout**, not another hot routing readback optimization (the replaced tile
routing has none). No new kernel or production-default change is made by this diagnosis.

All-rank scopes, model receipts, pipeline counters, source hashes, comparison and safety receipts
are retained in the [bottleneck manifest](evidence/ascend-decode-bottlenecks-20261120/manifest.json)
and [validated comparison](evidence/ascend-decode-bottlenecks-20261120/current-bottleneck-comparison.json).
Complete compressed device traces are retained in the private/local run archives, with checksums in
the manifest rather than duplicating roughly 1 GB of raw traces into the documentation site. All
90 checkpoint stat identities and CANN settings were unchanged, all eight NPU process tables were
empty, and the fresh fleet lease was released after both trials.

### Expert TP is the next partitioning experiment for this workload

The current dense/attention path already uses TP8; the proposed change is **routed expert weights
from EP8 to TP8**, not enabling TP for the first time. For batch-one latency, splitting each selected
expert's gate/up output channels and down input channels across all ranks can avoid the measured
2.1-maximum-versus-0.75-mean expert ownership skew. A TP8 expert partition retains approximately
the same original packed weight bytes per rank as EP8: all experts at one-eighth width instead of
one-eighth of the complete experts. This is a load-time slice of original bytes/scales, not a
checkpoint requantization or rewrite.

It is not an already measured speedup. TP adds partial-output reductions, changes tile geometry
and can increase synchronization or replicated small-operator work. Gate/up/SwiGLU are locally
computable without an intermediate all-gather, but down partials must be reduced at the right
numerical boundary. Summing already narrowed BF16 partials is not equivalent to narrowing once
after the full expert reduction. The exact default cannot be silently replaced; the previously
separate changed-accumulation experiment needs fresh quality checks.

Released inter-dimension 2304 gives **288 channels/rank at TP8**: this preserves original K32
blocks (nine blocks) but does **not** fit the current fixed four-K32-block down consumer without a
K-tail implementation or a separately validated K32 variant. That is a concrete compatibility check,
not a reason to alter the quantization grid. The next useful partitioning comparison is **EP8 vs
expert TP8 vs TP4/EP2**, holding prompt, context, original quantization and complete CPU Engram
constant. Large-batch throughput may favor EP, so any result must remain scoped to batch-one
single-stream latency. No expert-TP benchmark was run as part of this diagnosis.

### Hybrid expert TP/EP: static load estimate, not a measured speedup

For eight physical ranks, an expert tensor-parallel width T and expert-parallel group count E
satisfy **T x E = 8**. EP assigns whole expert identities to a group; TP slices each of that group's
experts across its T ranks. Dense/attention TP8 can remain unchanged. This is distinct from the
current dense-TP8/expert-EP8 arrangement, where each routed expert itself is not tensor-sharded.

Reassigning the **same 320 measured topk6 routes** using `expert_id % E`, and dividing the maximum
EP-group expert count by T, gives the following ideal arithmetic load estimate:

| Expert TP / EP | Mean maximum selected experts per EP group | Mean maximum full-expert-equivalent work per rank | Local intermediate width | Current K128 down-consumer aligned? |
| --- | ---: | ---: | ---: | --- |
| TP1 / EP8 | 2.1000 | 2.1000 | 2304 | Yes |
| TP2 / EP4 | 2.7594 | 1.3797 | 1152 | Yes |
| TP4 / EP2 | 3.8781 | 0.9695 | 576 | No: 64-element tail |
| TP8 / EP1 | 6.0000 | 0.7500 | 288 | No: 32-element tail |

Every arrangement has the same **0.75 full-expert-equivalent mean work per rank**, and the same
original packed expert bytes per rank if weights are partitioned rather than replicated. The table
only estimates static arithmetic balance from current routes. It does not model communication,
launch overhead, altered tile efficiency, future route changes from different accumulation, or TPS.
TP4/EP2 brings the ideal maximum-to-mean work ratio down from 2.80 to about **1.29**; this makes it
an attractive compromise to test, not an established fastest configuration. TP2/EP4 is the simpler
initial prototype because its 1152-channel down K fits the existing K128 consumer.

Each group's gate/up output channels are sliced, then SiLU and original K32 activation quantization
remain local. The row-parallel down projection produces **FP32 expert partials**, which must be
reduced across TP at the correct per-expert boundary before BF16 narrowing and ordered expert
combination. EP then combines disjoint expert contributions. No intermediate gate/up all-gather is
required for this partition. For this one replicated token, a large-batch all-to-all token dispatcher
is not necessary, but output reductions and any boundary-preserving synchronization still are.
Combining FP32 partials across all experts before their original per-expert BF16 narrowing would be
a separate numerical change, not a free collective fusion.

Hybrid partitioning targets **critical-rank skew**, not the consumer's original unpack/control
inefficiency. Average logical expanded-BF16 ring traffic remains approximately 4.25 GB/rank/forward
at fixed total expert work; TP does not automatically remove it. A useful next comparison is
EP8, TP2/EP4 and TP4/EP2 (plus TP8 as the fully balanced control), keeping original bytes, complete
CPU Engram, the prompt and quality checks fixed. No hybrid configuration has been implemented or
benchmarked in this diagnosis.

## Lessons from chiro2001's Ascend deployment repository (November 20)

A read-only source review pins
[`chiro2001/deepseek-v4.1-flash-ascend910B` at `ab88ff84`](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/tree/ab88ff84efa7970875f3f8cc38dfcb06c491b154).
Twenty-five inspected source artifacts were checked byte-for-byte against their pinned Git blobs.
No upstream scripts or binaries were run, no accelerator was claimed, and this review changes no
runtime, kernel, quantizer or serving default. The clone's case-insensitive `PROGRESS.md`/`progress.md`
collision is outside the inspected artifacts and was not used as evidence.

### The performance and numerical contracts are different

The upstream [quantization recipe](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/quant/README.md)
rebuilds official weights as **W4A8_DYNAMIC**, applies QuaRot, and separately requantizes Engram
FP8 tables to INT8. This is not our exact widening of original E2M1 codes to I8x2 with original K32
half-scales, nor our original E4M3/E8M0 activation contract. Its fast native W4A8 path cannot be
substituted by reinterpreting the original FP4 nibbles as integer INT4.

Its headline [README results](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/README.md)
include A3/910C and speculative DSpark runs, with `tok/s = accepted_length * 1000 / ms_per_round`.
Some A2-shaped results are explicitly A3 simulations, not 910B measurements. The repository also
records a non-speculative A=1 result of **19.289956 ms / 51.840451 tok/s**, but that is still a
re-quantized A3 W4A8 model, not a matched original-checkpoint 910B baseline. Its reports are source
material, not measurements independently reproduced by this review.

Host-resident Engram also means something narrower there: the
[deployed model](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/patches/files/model.py#L1484-L1505)
still performs WKV projection and gating in the device forward. This did not meet the complete-CPU
Engram requirement in effect at the time of the source review. The later November 20 request
explicitly permits NPU WKV/gating while retaining the original tables on host. That placement
pattern can now be tested, but neither upstream device indexing nor its INT8 table conversion is
adopted; hashing and original FP8 table lookup remain on CPU.

### Techniques worth evaluating on our current bottlenecks

| Priority | Transferable technique | Scope and required validation |
| --- | --- | --- |
| First | **Fused HC pre/post**, including projection/normalization/Sinkhorn coefficient preparation and residual folding | The upstream model calls `npu_hc_pre_v2` and `npu_hc_post`, whereas our current binding still has separate HC preparation and folding chains. This targets part of the approximately 52 ms graph-A region, not all 52 ms. Preserve the original Sinkhorn iteration count, epsilon, FP32 operations and BF16 boundaries; native/reimplemented outputs need fresh real-consumer and full-model checks. |
| First | **Reduce redundant AIC/AIV synchronization and launch initialization** | The upstream `gmm1_armF` documents fewer SyncAll/Reset rounds. Our current consumer shows low AIC MAC activity and heavy AIV control activity, so the scheduling lesson is relevant. Its delivered object is an A3 W4A8 artifact, not an E2M1/910B kernel to load directly. The ring's reuse/visibility fences that fixed replay corruption must not simply be removed. |
| Next | **Shared activation preparation and dependency-aware Q/KV Vector-Cube overlap** | Upstream shares input quantization only after checking quantization/communication compatibility, and overlaps Q normalization with KV Cube work, then Q Cube work with KV norm/RoPE/cache writes. Our current Q and window-KV branches are sequential. Reuse original FP8 codes/scales, preserve the V4.1 tail and cache writes, and check multistream graph/replay ownership. |
| Conditional | **Batch common integer metadata preparation and flatten index gathers** | Upstream uses one-launch descriptor-based slot preparation and `index_select` to replace broadcast/gather chains. Our ring/cache representation differs from its paged vLLM metadata: transfer the pattern only where the actual current profile identifies the same repeated work. Cache immutable layout, not dynamic positions, counts or token values. |

Relevant implementations are the
[HC entry points](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/patches/files/model.py#L810-L833),
[armF artifact and synchronization description](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/kernels/gmm1_armF/README.md),
[Q/KV scheduling](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/patches/files/dsa_v1.py#L1820-L1923),
[slot batching](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/experiments/operator_stack/tp8_slot_batches.py), and
[RoPE index selection](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/patches/vllm-ascend/0003-perf-rope-fuse-cos-sin-table-index-selection.patch).
No transferred technique has been implemented or timed in RelicLLM by this review.

### Important negative results and non-transferable shortcuts

The repository retains later verdicts, not just attractive initial microbenchmarks:

- **Q_a/KV matmul merging is not the same as sharing input quantization.** Its
  [formal QKV report](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/experiments/operator_stack/reports/a321-v41-qkv-audit-pass-20261011-v35.md)
  passes all-rank real consumers but loses all 12 end-to-end pairs. The early-KV-tail variant also
  loses all 12. Do not add their isolated microseconds to a projected model improvement.
- **Sparse MSD and prefix-list changes are not accepted fast paths.** The
  [candidate audit](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/experiments/operator_stack/evidence/operator_revalidation_20261013/history_candidates.md)
  records nonuniform M1 timing, slower K5 cases and missing actual-consumer coverage. These also
  retain a different INT4/INT8 W4A8 numerical pipeline and cannot replace our original FP4 consumer.
- **The three-output RMSNorm/INT8 fusion package is not a proven gain.** Its
  [root-cause report](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/docs/LNORM-FUSE-RCA-20261008.md)
  retracts the reported improvement because the fused operator never actually executed. Fake/Meta
  registration and real-consumer execution counters are useful engineering lessons, not performance
  evidence. Its INT8 quantizer is also incompatible with our FP8 activation contract.
- **The QLI no-candidate switch is not only an empty-candidate skip.** Its
  [patch](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/patches/vllm-ascend/0004-perf-indexer-fast-path-when-QLI-has-no-candidate.patch)
  forces mode 3 and disables candidate-source/filter behavior when enabled. That does not establish
  equivalence to our hierarchical sparse indexer on arbitrary long contexts; it is not copied.

Finally, the main `TP8+EP8` arrangement is **dense TP8 and 48 complete local experts per rank**.
The [formal expert-boundary adapter](https://github.com/chiro2001/deepseek-v4.1-flash-ascend910B/blob/ab88ff84efa7970875f3f8cc38dfcb06c491b154/experiments/operator_stack/formal_w4a8_prefix.py#L58-L85)
checks exactly that ownership. Its AllGather switch chooses a small-token communication regime;
it does not implement the expert TP4/EP2 partition discussed above. Our original cyclic device
routing already avoids the generic expert-map lookup/host dispatch, so copying its contiguous-range
mask or AllGather flag is neither a correct ownership rewrite nor a demonstrated new gain.

**Result:** borrow fused-HC coverage, producer/consumer scheduling, compatible FP8 input reuse and
measurement discipline. Keep expert TP/EP partitioning as a separate experiment. Do not replace
original quantization or promote failed microbenchmarks as proven improvements. Complete-CPU
Engram remains the exact default; the subsequently authorized host-table/NPU-compute binding is
explicit and reversible, with its own numerical and full-model evidence below.

## Host-table NPU Engram and fused HC folds (later November 20 authorization)

The subsequent direct request permits **original Engram tables on host**, rather than requiring
lookup, WKV and gating all to execute on CPU. This section records a new **explicit experiment**;
all earlier complete-CPU results retain their original scope. It uses the existing native
full-K dense/expert candidate, dense TP8 and complete-expert EP8. It is not the exact production
path and does not implement expert TP2/EP4 or TP4/EP2.

### Implemented placement and lifecycle

- Original checkpoint-backed table objects, hashing, lookup and original parameter/code/scale
  sources stay on CPU. Each Engram transfers only the current **6,144 BF16 lookup elements**
  (12 KiB) and a mask to NPU. Across two Engram layers this is 24 KiB/rank/forward plus two mask
  bytes, a geometry count rather than measured PCIe bandwidth. The binding never copies hidden
  state D2H; host argmax and the selected baseline's expert dispatch still have their own boundaries.
- WKV is **output-channel TP8**, with a contiguous original BF16 `[6144,3200]` weight slice per
  Engram/rank. Load-time CPU checks certify each slice against original FP8 codes and its original
  scale grid; TP boundaries preserve the original 32-row blocks. Activation quantization remains
  original E4M3/E8M0 block32. No INT8 activation or table requantization is introduced.
- Raw BF16 byte aliases are all-gathered into complete KV before the original FP32 gate sequence.
  Native full-K accumulation and the different output-channel matrix tiling are explicitly
  **changed-reduction experiments**, not bit-neutral substitutes for the original CPU WKV.
- `torch_npu` 2.10's `copysign` unexpectedly falls back to CPU, which broke graph capture. Exact
  FP32 sign-bit injection now replaces that unsupported operation. Signed zero, infinities and
  NaN payloads are checked independently; no value is clipped or silently sanitized.
- Each module has persistent hidden/lookup/mask/gather inputs and its own captured result. Pinned
  input reuse waits for the previous H2D completion event; one execution-stream owner is enforced.
  Returned graph outputs are cloned on device so later replays cannot overwrite callers' results.
  The original model/module forward descriptors and any existing overrides are restored on close,
  including after lookup failures. Metadata and available tensor versions guard source ownership;
  versionless inference sources must remain caller-owned immutable inputs.
- The explicit resident-weight/channel/input budget is **79,015,938 bytes/rank** (about 75.36 MiB).
  Graph-pool temporaries and allocator caches are explicitly excluded, not claimed to fit that
  exact byte count. Prefill and wider inputs retain the original CPU Engram fallback.

An existing CPU WKV-prefetch owner must be drained and closed before installation. Original
prefill and exact-reference collection in the benchmark run before this placement is installed.
Importing the helpers changes no default, and these helpers alone do not select the native
projection/routing/graph stack used for the measurements:

```python
from relicllm.models.deepseek_v4_1.ascend_device_engram import HostTableDeviceEngram
from relicllm.models.deepseek_v4_1.ascend_hc_decode import FusedHCDecodeFold

if host_prefetch is not None:
    host_prefetch.close()
# An existing TP process group and the current NPU device are required.
with HostTableDeviceEngram(front.model, device=device, capture_graphs=True):
    with FusedHCDecodeFold(front.model, backend="triton"):
        run_existing_batch_one_decode_loop()
```

### HC folding, not a fused coefficient pipeline

Two new core kernels fuse the HC collapse and residual fold, retaining original FP32 coefficient
values, multiply-before-sum rounding and the existing BF16 output boundary. Projection, RMS
normalization, affine transforms, sigmoid/softmax and Sinkhorn preparation are unchanged. This is
**not** a claim to have implemented the upstream complete `npu_hc_pre_v2` pipeline.

The generic Triton four-input reduction tree failed a real-device post-fold comparison by one
BF16 value. Changing only post to a sequential sum then exposed a pre-fold difference at a later
case. Explicit sequential FP32 sums, disabled FP fusion and the original reduction's positive-zero
result passed **64 changing-input graph replays** on the actual 5120-wide device geometry. A paired,
synchronized single-card graph probe measured one pre+post pair at **62.783 us versus 94.661 us**;
this isolated probe is not model TPS. Per-block coverage in the full model confirms both kernels
were selected in all 40 blocks, not merely imported or registered.

### Whole-model measurements and the generation-drift caveat

All runs use the same checkpoint, host, prompt, batch one and eight ranks. Four excluded warmups
precede the timed generation. NPU completion and host greedy selection are included; prefill,
compile/capture and quality copies are excluded. Walls below are the **slowest-rank** values.

| Configuration | Timed forwards | Context | Wall | Sustained single-request TPS |
| --- | ---: | --- | ---: | ---: |
| Existing native candidate, complete CPU Engram | 16 | 24..40 | 2.816559 s | 5.680690 |
| Original host tables, NPU WKV/gating | 16 | 24..40 | 2.662798 s | 6.008716 |
| Same NPU Engram plus fused HC folds | 16 | 24..40 | 2.613625 s | 6.121766 |
| Same candidate, longer sustained window | 64 | 24..88 | 10.264791 s | 6.234906 |

The host-to-device Engram comparison changes the generated trajectory: the first ten timed output
tokens agree, then output index 10 differs and later inputs diverge. Its **1.057744 observed TPS
ratio is not an isolated same-stream speedup**; similarly, the host-to-device-plus-HC ratio must
not be advertised as a matched combined gain. Earlier conversational attribution of 5.8% as a
strict A/B improvement is withdrawn.

The **HC-only pair** does use the same 16 autoregressive tokens. It measures **1.018814x** TPS,
or **3.073 ms/forward** less observed wall. On all eight ranks, its eight teacher-forced FP32
logit tensors and its final finite FP32 logits match the pre-HC candidate **bit for bit**. The
64-step repetition also reproduces the same first 16 tokens, but its longer context is not a
matched 16-step comparison and supplies no additional percentage-gain claim.

The NPU Engram operator probe checks two actual checkpoint layers, 32 changing hash/hidden/mask
cases and all eight ranks. Captured projections match eager **TP-shaped** projections exactly;
gating and exported-output ownership also pass. The separate full-matrix comparison exposes
native tiling differences, including more than one BF16 ULP near cancellation, and records them
rather than relaxing the eager-TP/graph parity requirement. Full-model teacher forcing gives 8/8
greedy matches against the original exact reference in each run, but native Engram's logit bits
differ and its free-generation divergence is real. This small window is **not general model-quality
acceptance**, so no exact default or serving guard is replaced.

The portable Engram/HC/original-activation/graph-seam regression records **304 passed, 6
hardware-gated skips**. Independent leased-device probes and model trials provide the actual NPU
evidence; skips are not counted as hardware passes. The final safety receipt confirms no private
live jobs, eight empty NPU process tables, unchanged stat identities for all 90 checkpoint files,
and untouched shared CANN/CI configuration; the lease was explicitly released.

**Remaining work at that milestone:** expert TP2/EP4 and TP4/EP2 had not been implemented or measured. The original
expert consumer's decoding/control cost and the unfused attention/HC coefficient chains remain
separate optimization targets. **50 TPS is still not reached.** The
[all-rank comparison](<evidence/ascend-host-engram-hc-20261120/Engram-HC-model-comparison.json>) retains
matched and unmatched scopes rather than combining their clocks into a projected ceiling.

## Consumer grain and attention preparation reuse (next November 20 iteration)

The next continuation tested two separable hypotheses, keeping the original host tables, explicit
NPU Engram and fused HC folds equal. It did not change expert EP8, the original activation
quantizer, the per-row K32 scales or any serving default. The online consumer keeps its N16 tile
layout and every ring/cache/Vector/MTE1 ownership fence.

### Larger native K tile: a useful online-consumer improvement

A separately built **K256** consumer consumes eight original K32 blocks per Cube tile instead of
four. Each original scale is still applied independently before Cube accumulation; no scale is
averaged or replaced. The ring is **393,216 bytes** at 24 workers and the shared complete-MoE
workspace is **626,796 bytes**, not a globally expanded expert weight matrix. The default build
still selects K32 and has no activation preloading.

Independent tests check all 256 packed E2M1 byte values, certified E8M0 bytes **2..252** and all
**256 K positions**, giving **1,028,096 exact BF16 comparisons per K256 variant**. Changing input
and selected IDs, empty-slot overwrites, invalid-ID poisoning and an over-capacity activation-L1
fallback also pass. Same-input paired graph probes of actual checkpoint W1/W3/W2 show an
approximately **9–16%** projection improvement for one through six active slots. Empty-slot K256
initialization is slightly slower, so a projection-only win is not extrapolated into model TPS.

### Activation-L1 reuse: tested, not selected

A separate build can preload immutable activation rows in 512 KiB L1, accounting for the M16 NZ
padding and weight double buffer. Oversized geometry falls back to the original A-tile load. This
keeps the bounded GM weight ring and all existing reuse fences; it does not introduce another
activation buffer in GM. Current real-weight projection timings show no meaningful benefit.
Consequently the full-model K256 run uses **preloading disabled**. The evidence retains this
negative result rather than attributing the K256 gain to L1 reuse.

### Shared original FP8 preparation in attention: numerical success, negligible TPS

Q-A and window-KV projections both read the same immutable attention input; compressed/indexer
paths can also read that input or the same normalized QR. The explicit runtime controller reuses
original **E4M3 codes, E8M0 scale bytes and the existing fused BF16 decode owner**, but only inside
one batch-one attention invocation. It retains storage identities and available tensor versions,
is thread-owned, caps live preparations at eight, and clears after every forward or capture.
CPU/prefill/out-buffer calls retain their original provider. Versionless inference operands require
explicit caller immutability; no cross-token activation cache is introduced.

An actual Q-A/KV graph probe passes **32 changing-input replays** with zero BF16 projection bit
differences. Its synchronized pair measures **89.699 us versus 80.422 us**. The full-model controller
covers all 40 attention modules, reports **978 preparation reuses** across eager/capture activity,
and peaks at four live preparations. Those counters include capture, **not a device-replay count**.
Despite actual dispatch coverage, the whole-model timing change is only approximately **0.1%**,
which is insufficient evidence of an effective acceleration. It remains an explicit reversible
candidate, not a new default.

### Full-model controls

These are fresh, separate model runs on the same eight-card host, each with four excluded warmups
and **16 synchronized batch-one single-request autoregressive forwards at context 24..40**. NPU
completion and host argmax are included. No profiler or verification-copy time is counted.

| Configuration | Slowest-rank wall | Sustained single-request TPS |
| --- | ---: | ---: |
| K128 device-selected online tile consumer | 3.153782 s | 5.073273 |
| K256 device-selected online tile consumer | 2.781112 s | 5.753094 |
| Existing native on-chip-antiquantization/expert-body graph control | 2.601831 s | 6.149515 |
| Same native control plus attention preparation reuse | 2.599201 s | 6.155739 |

All eight rank teacher-forced and final FP32 tensor artifacts were compared, not just rank-zero
JSON fields. Both the **K128/K256 pair** and the **native/reuse pair** use the same timed token
stream and have zero teacher/final FP32 bit differences within the pair. The K256 pair improves
TPS **1.134000x**, with **23.292 ms/forward** less observed wall. It still loses to the faster
native control, so it is **not** selected as that control's replacement. The native/reuse pair's
**1.001012x** ratio and **0.164 ms/forward** wall change are not a reliable performance milestone.
The earlier 64-forward 6.234906 TPS window is unchanged; these are different 16-forward trials and
cannot replace its denominator or create a new combined gain.

The native numerical experiment still differs from the original exact production reduction;
scoped equality to another native candidate does not establish general model quality or exact
window-13 equivalence. Original checkpoint bytes and quantization remain unchanged. The regression
records **359 passed and 6 hardware-gated skips**; real NPU acceptance is supplied by the separate
leased-device probes and model trials. Final safety checks again verify empty NPU process tables,
no private live jobs, all 90 checkpoint stat identities and shared CANN/CI configuration unchanged;
the new lease was released.

### The next priorities

- **AIV producer/control fusion:** the current decode still performs multiple cast, gather,
  FP32 scale-multiply, BF16 store and barrier stages per tile, with per-launch lookup setup even
  for empty ownership. K256 amortizes this work but does not remove it. Test a faster exact byte/
  scale decoder and a safe no-owned-expert initialization path; preserve the proven ring fences.
- **Expert TP2/EP4, then TP4/EP2:** address the measured expert skew independently of producer
  optimization. TP down partials must be combined in FP32 before the original per-expert BF16
  boundary. The current uniform K256 path does not divide TP2's local K1152; use K128 there or
  implement an original-K32-aligned tail before benchmarking. Neither hybrid had been implemented
  at that point; the later TP2/EP4 implementation is documented below.
- **HC coefficient and attention dependency fusion:** collapse/residual folding is fused, but the
  FP32 HC projection/normalization/affine/Sinkhorn preparation and Q/KV norm/RoPE/cache chains are
  still separate. The negligible preparation-reuse result argues against treating quantizer
  deduplication alone as the main attention optimization. Preserve the FP32 HC weight precision,
  the original scale/iteration rules and every cache-write dependency.

No projected combination of these items is presented as achieved TPS. **50 TPS is not reached.**
The [complete new comparison](<evidence/ascend-consumer-attention-20261120/consumer-attention-model-comparison.json>)
retains the same-stream checks and the negative performance results.

## Expert TP2/EP4: implemented and measured (late November 20)

The next direct implementation targets the expert critical-rank skew rather than another quantizer
micro-optimization. The [runtime controller](<evidence/ascend-expert-tp-20261120/end-of-round-sources/RelicLLM/relicllm/models/deepseek_v4_1/ascend_expert_tp.py>)
reshards original packed residency into **four EP groups, two TP members per group**. Each rank
holds 96 expert identities with half the original intermediate channels, instead of 48 complete
experts. Its packed weight/scale byte count remains **36,097,228,800 bytes (33.62 GiB)** per rank
across all 40 layers. No second whole-expert INT8/BF16 bank, checkpoint rewrite or activation
requantization is introduced. Dense, attention and the shared expert remain TP8.

### The implemented boundary

- W1/W3 slice original **output rows**. W2 slices the matching original **input columns**, including
  their unmodified low-nibble-first E2M1 bytes and per-row K32 E8M0 scale columns.
- Gate/up keep their original BF16 store; the original FP32 clamp/SiLU/routing product and block32
  FP8 quantizer run on each local intermediate slice. Local width **1152** retains 36 complete K32
  blocks and nine K128 tiles. K256 is not used to force an incompatible down geometry.
- The [core composition](<evidence/ascend-expert-tp-20261120/end-of-round-sources/relic-core/relic_core/kernels/expert_tp_moe.py>)
  requests the consumer's existing **FP32 down accumulator**. A subgroup HCCL sum completes each
  expert's down projection **before** the one original per-expert BF16 narrowing. Reducing
  separately narrowed BF16 TP partials would be a different, avoidable precision loss.
- Only each TP pair's leader emits the ordered EP routed sum into the existing global MoE
  reduction. The other TP member still computes and joins the subgroup collective, but emits zero
  routed contribution; its original shared-expert partial remains in the global sum. This avoids
  accidentally counting every routed expert twice.
- Device selection has **zero routing D2H calls**, one shared **374,892-byte** local workspace,
  the unchanged **196,608-byte** two-slot weight ring and 40 expert graphs per rank. No proven
  cache/Vector/MTE1 ring fence is removed. Graph allocator admission is not a physical-memory claim.

This is an explicit **decode-only** experiment. Install it on every rank after original prefill,
with the model's physical NPU current and immutable banks for the lifetime of capture/replay:

```python
import torch

from relic_core.kernels.ascendc_online_tile_cube import AscendCTileCubeConsumer
from relicllm.models.deepseek_v4_1.ascend_expert_tp import ExpertTPDispatch

# Use a separately validated K128 library and its exact checksum.
consumer = AscendCTileCubeConsumer(library_path, sha256=library_sha256)
with torch.inference_mode(), ExpertTPDispatch(model, consumer=consumer, tp_world=2):
    # Run private batch-one decode forwards on the same serialized NPU stream.
    ...
```

Close existing host expert-body/device-routing owners before resharding. Close restores the
original residency by default; a private process exiting after its trial may explicitly avoid
reloading it. Multi-row prefill while the TP owner is active is refused rather than silently
substituting another arithmetic path. KV/hash state is not a reversible numerical snapshot; reset
and run original prefill before switching a live request back to another provider. This is not a
new loader default or a serving-capability claim.

### Complete-model generation, with the divergence retained

Fresh complete-checkpoint runs include NPU completion and host greedy selection, exclude four
warmups/prefill/capture/verification-copy time, and use one request at batch one:

- **K128 EP8 control:** 16 autoregressive forwards, context24..40, **3.159921 s / 5.063417 TPS**.
- **K128 TP2/EP4:** 16 autoregressive forwards, **2.463771 s / 6.494111 TPS**.
- **TP2/EP4 sustained repeat:** 64 autoregressive forwards, context24..88,
  **9.928368 s / 6.446176 TPS**. Its first 16 tokens and teacher-forced FP32 tensors match the
  short TP2 run on every physical rank. These are fixed-length decode windows, not successful
  completion/serving or model-quality claims.

The EP8/TP2 streams differ at **zero-based timed output index10**. Both short runs have eight
teacher-forced greedy matches against the original exact reference, but their FP32 logits differ.
Consequently the observed **1.282555x generation ratio is not a matched-continuation causal gain**.
The earlier 6.234906 TPS native 64-forward milestone remains its own frozen experiment, not a
same-stream control for this repeat. No ratios from separate milestones are multiplied together.

### Identical-token-input controls: the actual latency improvement

Because the continuations differ, three further complete-model trials feed the **same 16 token
inputs**, with the same original prompt/prefill and context24..40. NPU completion and host argmax
still execute and are timed, but the argmax output is not fed back. They therefore measure forward
latency, **not autoregressive generation TPS**:

- K128 EP8: **197.011 ms/forward**.
- Existing native on-chip-antiquantization/expert-body graph control: **162.631 ms/forward**.
- K128 TP2/EP4: **153.537 ms/forward**.

TP2/EP4 saves **43.474 ms/forward** against its same-consumer EP8 control and **9.094 ms/forward**
against the faster native control, an observed approximately **5.6% native-control latency reduction**
(1.059227x forward-rate ratio). This is a sequential same-host comparison, not randomized
statistical significance. All eight ranks' complete teacher/final tensor artifacts were checked,
not just rank-zero scalars; their original exact reference phases also agree across these controls.

### Acceptance and remaining scope

Two independent actual-checkpoint layer0/expert0..7 probes each run **32 changing-input/route
replays on all eight NPUs**. Same-TP eager/capture FP32 output bits agree exactly, empty ownership
cannot expose old slot values, and every rank receives the same global output. Differences versus
complete-expert EP8 are retained. Portable tests exercise byte-slice reconstruction, unchanged
resident-byte budgets, FP32-before-BF16 reduction, single-leader emission, ownership refusal,
rollback/restoration and private subgroup retirement: **399 passed, 6 hardware-gated skips**.
The leased-device probes/model runs, not the skipped fixtures, supply NPU acceptance.

Native down-K partitioning and the new grouped global FP32 addition order are not the exact
production reduction. This small teacher window and prompt do **not** establish general quality
acceptance. Original quantization and exact production/serving guards stay unchanged. The next
optimization work remains original-byte AIV preparation fusion, mixed-grain/tail-aware consumers
and attention/HC dependency-chain fusion; TP4/EP2 has no accelerator benchmark in this iteration.
**50 TPS is not reached.**

A lost SSH reply did not cause a native-control rerun: the owned helper had already persisted
returncode0 and complete process cleanup before printing. That original receipt and all eight
model artifacts were reconciled, while the missing outer summary remains missing. The final safety
check again confirms eight idle NPU process tables, all 90 checkpoint stat identities, shared CANN
and the resident CI identity unchanged; the lease was released. The
[complete retained comparison](<evidence/ascend-expert-tp-20261120/generation-audit/expert-TP2-EP4-model-comparison.json>)
and [matched-input latency audit](<evidence/ascend-expert-tp-20261120/matched-input-audit/expert-TP2-EP4-matched-input-latency.json>)
keep quality, generated TPS, controlled latency, source versions and transport recovery separate.

## Real-output floor and wider online consumer (November 21)

This follows the explicitly authorized TP2/EP4 route, using the same original checkpoint and
host-table/NPU-Engram placement. It separates three questions: what remains when real FFN results
are available for replay, whether direct original-bit decoding is actually faster, and whether a
wider native output tile helps without changing any stored weight or activation quantizer.
**The final accepted experiment reaches 7.872578 TPS, not 50 TPS.**

### A conditioned non-expert floor, not a zero-output benchmark

The private driver records the actual routed outputs, complete FFN outputs, FFN inputs, expert IDs,
routing weights and full FP32 logits for 16 forwards at context24..40. Before each replay it restores
the complete KV/compression snapshot and Engram hash state. Each of the 40 layers receives its own
recorded real output; no expert result is replaced by zero. Logit bits, intermediate inputs/routes,
final cache/compression bytes and hash bytes are verified on all eight ranks before timing.

| Same-input path | Slowest-rank latency per forward | Scope |
| --- | ---: | --- |
| Actual original-byte TP2 baseline | 153.895 ms | Complete model |
| Replay actual routed expert output | 57.802 ms | Shared expert and global MoE reduction remain |
| Replay actual complete FFN output | 53.388 ms | HC post, attention, cache/Engram/head/host path remain |

D2D output staging and the host argmax are included. These are **conditioned diagnostic forward
latencies, not autoregressive generation TPS**, and not an additive device-time decomposition.
The 53.388 ms remaining path exceeds the 20 ms/token target by itself. Expert optimization alone
cannot make this measured implementation reach 50 TPS; attention/HC/cache/launch dependencies also
need substantial work. This does not prove an immutable hardware ceiling for a redesigned path.

### Direct BF16 bits and a no-ring AIV: correct is not necessarily fast

The independent AscendC direct-bit variant constructs original E2M1/E8M0 BF16 values using modular
int16 exponent offsets, zero masks and invalid-scale poison. It removes floating weight scaling,
but retains NZ rearrangement and every ring/visibility/reuse fence. Exhaustive all-K-position
checks cover **514,048 K128** and **1,028,096 K256** BF16 values with zero bit differences.
The independent no-ring AIV consumer passes the 1,028,096-value codec gate after its first N16/K256
compilation exceeds the physical 192 KiB UB and the output-row tile is reduced to N8.

Real original layer0/expert0..7 TP2 projections tell a different performance story:

- Direct-bit **K128** W1/W3/down are slower than the existing K128 floating decoder. A full 16-step
  same-stream comparison preserves every all-rank teacher/final FP32 bit but falls from
  **6.475377 to 6.088056 TPS**. It is not selected as a faster default.
- Direct-bit **K256** improves W1/W3 modestly, but TP2's local down K1152 does not divide K256. No
  uniform K256 TP2 full-model gain is claimed.
- The pure **N8 AIV** avoids the expanded-weight GM handoff but takes roughly **32.6/32.6/36.2 ms**
  for W1/W3/down with three active local experts, against about **0.75/0.75/0.81 ms** on Cube.
  One down BF16 value differs under its separately authorized FP32 reduction. The already-slower
  real-weight candidate is not run through a needlessly expensive full-model benchmark or promoted.

Dynamic captured replay includes changing activation, bank data, expert IDs, zero/empty ownership,
per-slot inputs, original K tails and poison cases. Native prepared/bit consumers have no observed
routing D2H in these probes. All candidates still retain original checkpoint codes/scales and the
original block32 FP8 activation encode/decode; none is INT4/W4A8 or a re-quantized checkpoint.

### Hoist immutable metadata, then consume two original N16 banks per Cube tile

A fresh eight-rank CANN timeline again finds heavy Vector/scalar/control activity and low AIC MAC
activity in the mixed consumer. One avoidable cost is rebuilding small lookup, scale-repeat and NZ
index tables in every AIV launch. The new explicit prepared-table ABI copies those immutable
metadata tables from a per-device buffer, prepared before capture. It contains **no checkpoint
weights or dynamic routing values**. Legacy libraries keep their original 15-argument launch ABI;
new prepared launches use a checked separate entry point. No previous binary is overwritten.

The prepared **N16/K128 floating decoder** uses **14,336 metadata bytes/device**, retains the same
196,608-byte double-slot ring, and saves about 0.04 ms per real projection. The 64-step model reaches
**6.669336 TPS**, versus **6.437617 TPS** for a freshly measured original K128 control: an observed
same-stream **1.035994x** rate ratio.

The final **N32/K128 prepared consumer** reads two adjacent original physical N16 banks without
repacking. It gathers them into the proper K16-major/N16-minor operand order and consumes a wider
Cube output tile. Source checkpoint tiles remain N16, scale blocks remain K32, and K accumulation,
BF16 stores and the TP2 FP32-down-before-BF16 boundary are unchanged. The failed first L0B ordering
and its corrected second independent build remain separate negative/positive evidence.

- **26,624 bytes/device** of immutable metadata; **393,216 bytes/rank** of bounded double-slot ring;
  **571,500 bytes/rank** of shared MoE workspace. These are explicit tensor sizes, not full graph-pool
  or measured DRAM-traffic claims. There is no full expanded-weight matrix allocation.
- **516,096 BF16 codec comparisons**, including 16 duplicate alignment rows; all 4016 original
  byte/scale corpus rows are retained. Real W1/W3/down **BF16 and FP32 accumulator bits match** the
  existing K128 consumer.
- With three active local experts, real W1/W3/down latency changes from
  **0.746492/0.745699/0.806894 ms** to **0.522838/0.524917/0.610869 ms**. These isolated synchronized
  projection timings are not model TPS.
- Original layer0/expert0..7 probes run **32 changing-input/route replays on all eight cards**.
  Captured/eager FP32 output bits agree, empty/nonleader cases cannot emit old slot data, and global
  outputs agree across ranks. Differences against original complete-expert EP8 remain scoped as
  the existing authorized TP reduction experiment.

### Complete-model acceptance, including sustained decode

| Original-byte TP2 consumer | Forwards | Context | Slowest-rank wall | Single-request batch-one TPS |
| --- | ---: | --- | ---: | ---: |
| Original N16/K128 control | 16 | 24..40 | 2.470898 s | 6.475377 |
| Prepared N16/K128 | 16 | 24..40 | 2.384895 s | 6.708890 |
| Prepared N32/K128 | 16 | 24..40 | 2.025186 s | 7.900509 |
| Fresh original N16/K128 control | 64 | 24..88 | 9.941566 s | 6.437617 |
| Prepared N16/K128 | 64 | 24..88 | 9.596158 s | 6.669336 |
| Prepared N32/K128 | 64 | 24..88 | 8.129484 s | **7.872578** |

All eight complete teacher/final FP32 tensor artifacts are compared, with remote path/size/SHA256
identities retained. N32 and prepared N16 match the control's full teacher/final bits and **the same
64 autoregressive token IDs**. Each short/long pair has equal teacher bits and the same first 16
outputs. Forty isolated complete-FFN graphs per rank also preserve each recorded real layer's BF16
output exactly before their separate timings are retained.

The N32 64-forward rate ratio is **1.222902x**, an observed **22.3% throughput increase** on the
same stream. Latency changes from **155.336976 to 127.023191 ms/token**, saving **28.313785 ms/token**.
The separately measured 16-step ratio is 1.220085x. These sequential same-host windows are not a
randomized significance claim or a general quality evaluation. They are substantially stronger
than extrapolating isolated projection timings to whole-model TPS.

### HC coefficients and one collective: retain the unsuccessful experiments

Two independent larger fusions are implemented without changing the production defaults:

1. **HC FP32 coefficient projection plus RMS in one Vector kernel.** Original full FP32 HC weights
   are never narrowed; original affine/sigmoid/softmax/Sinkhorn providers and their iterations/epsilon
   remain separate and unchanged. Native projection/RMS reduction order is explicit. Actual layer0,
   layer20 and layer39 attn/FFN coefficient probes keep eager/capture bits equal; their HC output
   maximum absolute differences are at most about 3e-7 in this probe. Full-model same-input latency
   saves only **1.155464 ms/forward** (153.601697 to 152.446233 ms), and the generated sequence differs
   at zero-based output index10. Eight exact-reference teacher choices agree, but that is not general
   quality acceptance or a same-generated-stream causal speed claim.
2. **One global FFN SUM over six independent expert FP32 slots plus one shared slot.** Compact local
   EP slots are aligned by original global expert IDs on device. Physical-rank FP32 partials are summed
   before **each expert's own BF16 narrowing**, then experts combine in ascending order. It never
   sums all expert FP32 rows before narrowing and never duplicates TP-leader contributions. Attention
   TP is untouched. Synthetic dyadic/cancellation checks run 32 changing all-rank replays, and the
   complete model matches the existing TP2 teacher/final bits and output stream here. But same-input
   latency saves just **0.200347 ms/forward**, and its fresh 64-step rate is **6.440064 TPS**, effectively
   the same as 6.437617 TPS control. Removing a collective does not demonstrate useful model speedup.

The latest TP2 profile preserves all-rank event scopes, excluded route telemetry, separate
four-forward CPU/NPU markers and raw artifact hashes. HCCL intervals include rank waiting and
profiler inflation; they are **not removable link-only cost**, and scope/kernel sums are not an
additive wall-time budget. Heavy cast/transpose/index/elementwise attention/HC chains and the
conditioned non-expert floor remain valid next targets, not measured combined speedups.

Portable regression coverage for this iteration is **618 passed, 33 hardware-gated skips**. The
leased-device codec, route/capture and full-model runs supply actual NPU acceptance; skips do not.
Runtime tests cover immutable original HC storage, foreign-owner refusal, installation rollback,
release-before-close, synchronization-failure cleanup and original source descriptors. Both worktree
diff checks remain clean. Kernels are in relic-core; RelicLLM owns only explicit placement/binding.

All new paths are decode-only numerical/performance experiments, not serving capabilities. The
original FP4/E8M0 bytes, FP8 activation boundaries, host Engram tables and production/serving guards
remain unchanged. The final safety receipt verifies eight idle NPU process tables, all **90 original
checkpoint stat identities**, unchanged shared CANN/CI identity and no live owned job across 264
prior receipts; the lease is released. **50 TPS is not reached; the sustained result still needs about
6.35x improvement.**
The [complete all-rank consumer audit](<evidence/ascend-decode-floor-n32-20261121/trials/compare-final-original-consumers-20261121T060626Z/final-original-consumer-model-comparison.json>)
keeps generated TPS, reference quality, fixed-input controls and source versions separate.

## Evidence milestones

Seventeen separate bundles, because each earlier one is frozen:

* [ascend-decode-optimization-20261118](evidence/ascend-decode-optimization-20261118/manifest.json) —
  145 retained files (4,326,023 bytes): the exact-path milestone, all-rank comparison reports,
  compiler failures and the idle safety check before its lease was released. This one is not
  rewritten by later work.
* [ascend-decode-round2-20261119](evidence/ascend-decode-round2-20261119/manifest.json) —
  158 retained files (5,469,208 bytes): the six verified candidate receipts from 0.217164 to
  4.934599 TPS, the operator microbenchmarks behind each exact change, the eight-rank collective
  floor, the new probe and driver sources, and the new-lease safety receipt.
* [ascend-decode-round3-20261119](evidence/ascend-decode-round3-20261119/manifest.json) —
  20 retained files (1,677,744 bytes): the timing ablation that splits a token across all eight ranks,
  the run showing the bounded operand cache (3.169 TPS) loses to the recorded bodies (4.935 TPS), and
  the probe that closes the hardware 4-bit avenue for this checkpoint's E2M1 codes.
* [ascend-decode-round4-20261119](evidence/ascend-decode-round4-20261119/manifest.json) —
  22 retained files (1,760,146 bytes): the widening-tile sweep on both expert geometries, the
  5.151722 TPS candidate with its all-rank bit identity, and the modified expert-off ablation, which is not a hardware or implementation ceiling.
* [ascend-decode-round5-20261119](evidence/ascend-decode-round5-20261119/manifest.json) —
  16 retained files (899,124 bytes): the per-stage split of one recorded expert body, the
  two-bytes-per-lane widening probe, and the receipts behind the two closed avenues.
* [ascend-decode-round6-20261119](evidence/ascend-decode-round6-20261119/manifest.json) —
  11 retained files (1,017,022 bytes): the one-replay-at-a-time attribution (layer A 1.386 ms against
  layer B 0.144 ms), the expert-body replay, and the eight-rank bit identity of the run that measured
  it.
* [ascend-decode-round7-20261119](evidence/ascend-decode-round7-20261119/manifest.json) —
  3 retained files (249,983 bytes): the host-side dispatch split showing the host blocked for
  169.5 ms of a 194 ms token while its own work is 32.5 ms, which reframes the routing read as a
  wait on the device rather than a transfer cost.
* [ascend-decode-round8-20261119](evidence/ascend-decode-round8-20261119/manifest.json) —
  4 retained files (114,957 bytes): the pricing that puts a plain M=1 matmul of layer A's own
  attention weights at 0.1334 ms against the recorded layer's 1.3863 ms, so the matmuls are under a
  tenth of the layer and there is no row-dimension penalty on these shapes.
* [ascend-decode-round9-20261119](evidence/ascend-decode-round9-20261119/manifest.json) —
  5 retained files (12,246 bytes): the dense GEMV tiling sweep showing its one remaining knob is worth
  1.02x and is bit-identical everywhere, and that the exact control path pays 595 ms per token for its
  five dense projections alone at 3.27 GB/s.
* [ascend-decode-round10-20261120](evidence/ascend-decode-round10-20261120/manifest.json) —
  10 retained files (12,562,219 bytes): the rounds that stopped producing gains and started producing
  limits. The CANN profile's operator table and the same table grouped by input shape, which is how
  the per-token dtype conversions were attributed; the tile-major residency accepted as exact on the
  card and measured 3.05x in isolation; two independent candidate re-verifications whose per-step
  quality entries are computed against the exact reference reduction in the same process; and the two
  tests that pin the hand-hoisted attention dispatch count and the loader's tile-major switch. It also
  records the null results -- allocator policy, an eight-fold thread range, metadata-only dispatch
  removal, and tile-major in the whole model -- because a claim that a knob is dead is a measurement
  too, and without a receipt a later reader has to re-run it.

* [ascend-decode-round11-20261120](evidence/ascend-decode-round11-20261120/manifest.json) —
  6 retained files (24,508,389 bytes): the two kernel profiles either side of the round-27 change, the
  probe that ranks a profile by device time, and the test pinning the rewrite's bit identity. The
  first profile is the one that put a *copy* kernel at 19.1% of the token and so found the gain; the
  second shows that kernel at zero. Its manifest also records what the ranking leaves unexplained:
  non-collective kernel execution falls to 81.6 ms of a 178.6 ms token, so 97 ms is HCCL collectives
  plus the forty per-layer routing readbacks, and those readbacks drain the device pipeline by
  construction because the host has to see the indices to choose which expert bodies to replay.

* [ascend-decode-tile-consumer-20261120](evidence/ascend-decode-tile-consumer-20261120/manifest.json) —
  25 retained files (920,024 bytes): the pinned upstream source findings, independently written
  AscendC kernel and binding snapshots, exhaustive codec and dynamic replay proofs, all eight
  full-model rank receipts and the released-lease safety check. The 4.777145 TPS result improves
  on whole-matrix staging but remains slower than the earlier 5.723332 TPS native baseline.

* [ascend-decode-bottlenecks-20261120](evidence/ascend-decode-bottlenecks-20261120/manifest.json) —
  63 retained files (13,415,577 bytes): matched fresh tile/native runs, all eight rank model and event
  receipts, all-rank CANN operator/pipeline summaries, validated critical-rank attribution, probe
  sources and beginning/ending safety checks. Full compressed traces stay in the checksum-pinned
  private/local archives. It diagnoses 4.787352 TPS tile versus 5.707230 TPS native, not a new speed
  milestone or a measured expert-TP configuration.

* [ascend-host-engram-hc-20261120](<evidence/ascend-host-engram-hc-20261120/manifest.json>) —
  71 checksum-listed files (779,193 bytes) plus the manifest: all 32 rank model extracts across four
  runs, full teacher/final-tensor checksums and an all-rank CPU comparison, actual-checkpoint
  Engram replay probes, HC reduction failures and the accepted sequential-fold probe, final timed
  source snapshots, portable tests and beginning/ending safety receipts. It records 6.234906 TPS
  over 64 forwards, a matched HC-only 1.018814x pair, and the **unmatched** native-Engram generation
  change. The explicitly released lease and unchanged checkpoint/quantization/defaults are retained.

* [ascend-consumer-attention-20261120](<evidence/ascend-consumer-attention-20261120/manifest.json>) —
  78 checksum-listed files (858,168 bytes) plus the manifest: all 32 model rank extracts, complete
  teacher/final artifact checksums and CPU comparisons, actual-weight K/L1 timing probes,
  exhaustive all-position K256 byte/scale checks, attention reuse capture/coverage, exact timed
  source snapshots separated from later metadata hardening, portable regressions and fresh lease
  safety receipts. It proves the matched online-consumer 1.134000x gain, while retaining the
  approximately 0.1% attention result and non-improving L1 candidate as negative performance findings.

* [ascend-expert-tp-20261120](<evidence/ascend-expert-tp-20261120/manifest.json>) —
  164 checksum-listed files (2,372,500 bytes) plus the manifest: all 48 model rank extracts across
  generated and identical-input controls, complete teacher/final artifact hashes and CPU audits,
  two real-expert eight-rank replay probes, exact timed sources separated from final admission/
  lifecycle hardening, 399 portable passes and released-host safety receipts. It records 6.446176
  TPS over 64 autoregressive forwards and 9.094 ms less same-input forward wall than the native
  control, without calling forced-input timing generation TPS or claiming the changed reductions
  preserve original-exact quality. The lost SSH reply and unreplayed persisted ownership receipt
  remain explicit negative transport evidence.

* [ascend-decode-floor-n32-20261121](<evidence/ascend-decode-floor-n32-20261121/manifest.json>) —
  **875 files, 18,623,096 bytes** excluding the manifest: conditioned real-output floor checks,
  direct-bit/no-ring negative results, immutable-table and N32 native build snapshots, exhaustive
  codecs, all-rank real-expert replays, complete 16/64-step model audits and original source versions.
  It also retains the unsuccessful HC coefficient/one-collective controls, compact all-rank current
  CANN attribution with raw artifact hashes, 618 portable passes / 33 hardware-gated skips, ending
  checkpoint/idle safety checks and the confirmed lease release. The same-stream N32 sustained result
  is 7.872578 TPS / 1.222902x, not a 50 TPS or original-exact/default quality claim.

All bundles carry hashes for every file. The comparison JSONs verify all eight rank artifacts.
Earlier final-logit tensors remain in the working tree rather than being duplicated into bundles.
For the new Engram/HC bundle, the completed full teacher/final tensors were compared on the host and
are retained through exact remote-path/byte-count/SHA256 identities; failed optional local tensor
transfers are excluded from the evidence. The expert-TP bundle likewise retains remote identities
for the full teacher/final tensors checked on all eight ranks, not incomplete local tensor copies.
No bundle treats one rank's tensor as proof of every rank.
