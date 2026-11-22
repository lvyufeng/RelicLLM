# Lossless compact FP4 weights in Ascend Cube order

This is an opt-in extension of the merged original-format decode experiment. The checkpoint's
represented values, E2M1 nibbles, E8M0 K32 scale bytes and FP8 activation grid do not change. The
physical byte order does change, explicitly, and is not silently substituted into the original
loader or consumer.

## Measured result (2026-11-21)

The [scale-LUT follow-up](<#scale-lut-follow-up-2026-11-22>) below adds an explicitly certified
N32/N64 K128 consumer. The original measurement remains a separately frozen milestone.

On one freshly leased eight-card Ascend910B3 64-GiB/card server, using the same full 40-layer model,
TP2/EP4 expert placement, native provider, host-resident original Engram tables and NPU WKV/HC
bindings:

| Matched configuration | Critical-rank wall, 64 outputs | ms/output token | Single-request decode TPS |
| --- | ---: | ---: | ---: |
| Original N16 source layout, N32/K128 consumer | 8.144478 s | 127.257467 | 7.858085 |
| Compact Cube-order source, unsigned byte indices, double UB slots | 7.458074 s | 116.532406 | 8.581304 |

Each run excludes four warmups, times 64 synchronized autoregressive forwards at context 24→88,
and includes host token selection. The benchmark is explicitly fixed-length and continues after
observed EOS IDs at zero-based steps 8 and 13; these are 64 benchmark outputs, not a normal
EOS-terminated 64-token completion. The improvement is **1.092035x TPS (+9.2035%)**, or **10.725061 ms
less latency/output token**. A single paired sustained window is not a distribution across diverse
prompts or context lengths. This is neither prefill throughput nor batched aggregate throughput.

The two runs produce the same 64 generated token IDs. An independent 16-forward compact repeat
at context 24→40 achieved **8.609725 TPS / 116.147730 ms per token**, with the same first 16 output
IDs and byte count on every rank; it is a repeat, not a second matched control pair. Its
[repeat receipt](<evidence/ascend-compact-weight-format-20261121/repeat-16/summary.json>) is separate
from the frozen 64-step comparison. In that **64-step comparison**, all eight ranks' original-reference,
candidate-teacher and complete final FP32 logit tensors compare with **zero differing bits**.
The full-model result is therefore an accepted *same-provider delta*. Native Cube/HC/Engram
accumulation remains the separately authorized experiment; it is not bit-neutral to the original
exact production path and not general quality acceptance. **The 50-TPS objective is not reached.**

The [full comparison](<evidence/ascend-compact-weight-format-20261121/full-model/comparison.json>) and
[manifest](<evidence/ascend-compact-weight-format-20261121/manifest.json>) retain all-rank measurements,
original tensor/source hashes, sanitized source views and regression records.

## What changes before decode

The physical format is `relic.fp4.cube-n32-k128.v1`:

```python
codes_cube = codes.view(N // 32, 2, 16, K // 128, 8, 8).permute(0, 3, 4, 1, 2, 5).contiguous()
scales_cube = scales.view(N // 32, 32, K // 128, 4).permute(0, 2, 3, 1).contiguous()
```

The code shape is `[N/32,K/128,8,2,16,8]`, directly matching the Cube B operand's K16-major /
N16-minor order; scale shape is `[N/32,K/128,4,32]`, retaining independent K32 scales. It requires
complete N32/K128 geometry and refuses padding rather than silently growing the checkpoint.
The code inverse uses the same permutation; the scale inverse is `permute(0,3,1,2)`.

The byte budget is exactly:

\[
B=NK/2+NK/32=17NK/32.
\]

Two nibbles stay in their original byte. No decoding, new rounding, requantization, doubled-INT8
bank or whole BF16 matrix is stored. Including signed-zero encodings, every original code and scale
byte is recoverable. Routed expert residency stays **36,097,228,800 bytes/card = 33.618164 GiB**.

The trial converts one CPU projection at load time and replaces the old resident bank layer by
layer. A private operator probe also writes and rereads one TP2 half-expert's three compact
projections; its payload remains 9,400,320 bytes. **No full 510-GB converted checkpoint is claimed
as delivered.** Retaining an original checkpoint plus a sidecar also increases combined disk use,
even though the sidecar's weight payload is the same compact size.

## What the hot kernel no longer does

1. The converted byte stream already has the Cube NZ order, removing the post-decode weight
   `Gather` and its table copy. Immutable per-device table bytes drop from 26,624 to **18,432**.
2. A2's supported exact `uint8→half→int32` cast chain indexes bytes 0…255 directly, removing the
   signed-byte `+128` correction and barriers for both codes and scales. Prepared tables declare
   the corresponding byte order; no FP4 value or scale is relabelled.
3. The decoded BF16 tile uses two UB slots in the otherwise unused NZ-index region. Existing
   READY/FREE cross-core fences prove retirement before slot reuse, removing the single-slot
   `MTE3→Vector` wait without increasing the Vector scratch extent.

The bounded GM ring remains necessary for this validated A2 handoff. The format does not pretend
that A5 direct UB→L1 transfer or native FP4/FP8 arithmetic is available on 910B. It also does not
claim FP4 equals signed INT4: E2M1×2 includes ±12 and cannot be represented by one affine signed
INT4 codebook scale. The user explicitly chose **lossless conversion only**.

## Explicit admission

`ExpertTPDispatch(..., bank_format="original-n16")` keeps the default unchanged. A caller may ask
for the new format only with the exact matching compiled consumer, N32/K128 geometry, original
CPU code/scale sources, NPU residency and the original E8M0 2…252 operand certificate. Constructor
and installation reject mismatches before freeing a bank or creating TP groups.

The common MoE validator still receives same-sized contiguous compatibility views; those tensor
extents are opaque physical bytes under the compact contract, not old N16 semantics. A legacy
consumer cannot be used for the compact bank. Neither the CPU oracle nor serving silently falls
back to a consumer that would read the bytes in the wrong order.

## Validation and limitations

- Each native candidate passed **516,096 decoded BF16 value checks**, covering all 256 packed
  byte values, certified scale codes 2…252 and all 128 activation positions.
- **96 changing-input/route graph replays** per gate: shared/unshared activations, active counts
  0/1/3/6, two K geometries, invalid-source and invalid-scale poisoning, same-Cube FP32/BF16 bits.
- Actual checkpoint W1/W3/W2 TP2 projection outputs and FP32 accumulators: zero bit differences.
  In the v2 active-three-slot projection pair, mean latency was 0.534656→0.447271 ms,
  0.532950→0.445024 ms and 0.623311→0.526040 ms. These are projection timings, not model TPS.
- On the leased Linux server: **175 core cases passed, one CUDA-only case skipped**; **49 runtime
  cases passed**. These are targeted suites, not a full-repository pass.
- Independent static review checked the byte/scale coordinate maps, UB slot extents, native ABI,
  default compatibility, poison/sentinel handling and cross-core reuse fences.
- No tests, model runs, benchmarks or builds were executed on the macOS development machine.
- Earlier unsuccessful missing-dependency, fresh-cache compilation and test-expectation attempts
  did not enter timing claims. The source and checkpoint were not changed to conceal them.
- After the accepted pair, all eight NPU process tables were empty and the original 90-file
  checkpoint stat identities and shared CANN configuration were unchanged. Original weight hashes
  remain the prior verified identities; no new full-checkpoint checksum is claimed.

The next bottleneck remains the combined projection/attention/collective path. More offline
format work must still be judged against complete same-provider logits and sustained decode,
not inferred by subtracting isolated kernel timings from the earlier 53-ms conditioned replay floor.

## Scale-LUT follow-up (2026-11-22)

This change keeps only the measured N32/N64 **K128** scale-LUT path. Paired W1/W3, K256, and
its depth-one/depth-two streaming implementations remain on the preserved experimental branches;
they are not runtime options in this PR.

### End-to-end result

| Same-provider B=1 configuration | Matched control TPS | Candidate TPS | Candidate ms/token |
| --- | ---: | ---: | ---: |
| N32/K128 scale-aware LUT | 8.562436 | 9.566600 | 104.530348 |
| **N64/K128 scale-aware LUT** | **8.562436** | **10.680570** | **93.627963** |

The N64 change is **+24.7375% TPS**, saving **23.161225 ms/token**. Two later 64-step N64
windows measured **10.693568** and **10.694418 TPS**. A separate matched 16-step repeat gives
**8.586612 → 10.752446 TPS (+25.2234%)** and is not substituted for the sustained benchmark.

All sustained windows exclude four warmups and time 64 synchronized autoregressive forwards,
context 24→88, including host argmax. The shorter repeat uses context 24→40. These are
single-request, batch-one decode rates, **not** prefill or batched aggregate throughput, and the
fixed-length benchmark deliberately continues after EOS. This is one prompt/context regime, not
a distribution across diverse prompts or long contexts.

The [N64 comparison](<evidence/ascend-scale-lut-20261122/comparisons/N64-LUT.json>),
[N32 comparison](<evidence/ascend-scale-lut-20261122/comparisons/N32-LUT.json>) and
[short-repeat comparison](<evidence/ascend-scale-lut-20261122/comparisons/N64-repeat16.json>)
check output IDs and all eight ranks' three complete saved FP32 tensors: original-reference,
candidate-teacher and final. Every comparison has zero differing bits. This does **not** claim
every timed step's full logit distribution was saved, or equivalence to the original exact-default
implementation or a general model-quality evaluation. The separately authorized native
Cube/HC/Engram experimental provider remains the control; no new lossy quantization is added.

### Why the decoder can be smaller

The complete scan inspects **46,080 original scale matrices**, **16,986,931,200 E8M0 bytes**,
without expanding whole weight matrices. All literal scale codes lie in **117…127**. The
[global/matrix summary](<evidence/ascend-scale-lut-20261122/scale-histogram-summary.json>) is
complete; its separately labelled tile-local histogram samples 64 tiles per matrix.

A universal **12 × 256 uint32 table = 12,288 bytes** maps one packed code byte plus one original
scale byte to two exact BF16 operands. Eleven slots retain every observed scale, including rare
outliers; slot 11 is `0x7fc07fc0` poison. The total scale-byte map sends unsupported values to
that poison slot, rather than reinterpreting them or dropping outliers. This is not a per-expert
dictionary or an expanded weight bank.

The A2 kernel keeps the proven true-unsigned `uint8→half→int32` index casts. It expands the
scale slot offsets in Cube order, adds packed-byte indices, and gathers both BF16 operands
straight into the bounded decoded tile. The separate FP32 weight multiply and BF16 narrowing
are removed. Operand zero-folding follows the already accepted native consumer.

### N64 physical format and unchanged payload

The additional source tag is `relic.fp4.cube-n64-k128.v1`:

```python
codes_cube = codes.view(N // 64, 4, 16, K // 128, 8, 8).permute(0, 3, 4, 1, 2, 5).contiguous()
scales_cube = scales.view(N // 64, 64, K // 128, 4).permute(0, 2, 3, 1).contiguous()
```

The inverse recovers every original code and scale byte, including signed-zero nibble encodings.
N64/K128 geometry is required, not inferred from a same-sized tensor. TP2 W1/W3 is 1152×5120
and W2 is 5120×1152, each requiring **no padding**. N64 halves output-tile tasks, not weight
traffic or matrix FLOPs. Payload stays `NK/2 + NK/32`, or exactly **36,097,228,800 bytes/card
= 33.618164 GiB** for routed experts. No full persistent converted checkpoint is delivered.

| Fixed K128 consumer | Vector UB bytes | BF16 GM ring bytes, 24 workers | Prepared-table bytes/device |
| --- | ---: | ---: | ---: |
| N32 LUT | 61,952 | 393,216 | 21,504 |
| N64 LUT | 110,592 | 786,432 | 29,696 |

The bounded GM ring is the validated A2 handoff; no A5 direct UB→L1 or native FP4/FP8
arithmetic is assumed. A1 has one slot and is loaded from its base. Shared/unshared activations
select only the GM row address; output-buffer READY/FREE fences guard slot reuse.

### Explicit admission and ownership

The default `bank_format="original-n16"` is unchanged. N32 keeps both its existing compact
consumer and the new optional LUT consumer. N64 requires its exact-format scale-LUT consumer,
`group_k=4`, `output_rows=64`, the original NPU/TP placement and complete N64/K128 geometry.
Both formats use an exact library SHA-256 and queried ABI; same-sized tables or scratch do not
identify a source format. Each output-row variant is built as a separate native library.

All original CPU scale bytes are checked against 117…127 **before** conversion/copy for a LUT
bank. The manager marks its scale certificate true only after every bank installs successfully.
Both BF16 and scale-window certificates are explicit booleans. Ordinary consumers receive no
new LUT keyword; neither serving nor the CPU oracle silently chooses the new implementation.
Tables are prepared before graph capture, and input/output/staging storage must not alias.

If a compact owner's NPU completion fence fails during close, graphs, banks, workspace and TP
group stay owned for a cleanup retry. Retirement occurs only after the fence succeeds; the
original noncompact default close policy is not changed.

### Validation, rejected alternatives and evidence

The original Ascend validation covers all 256 packed-byte codes, all eleven scale codes, 24,576
basis BF16 values per K128 candidate, 96 changing-input/route graph replays, and nine real
W1/W3/W2 TP2 projections across layers 0/20/39. Complete FP32 accumulators and BF16 outputs
match, with empty slots, invalid IDs and unsupported-scale poisoning separately gated. These
[operator records](<evidence/ascend-scale-lut-20261122/operator-gates.json>) are numerical
acceptance, not model-rate extrapolations.

For completeness, the [rejected-alternatives record](<evidence/ascend-scale-lut-20261122/rejected-alternatives.json>)
retains measured decisions without shipping those implementations: paired W1/W3 changes TPS
by only +0.4674%; N32/K256 reaches 10.250180 TPS; N64/K256 depth one falls to 9.849394 TPS;
depth two recovers to 10.645824 TPS but does not beat its matched 10.694418 N64/K128 control.
Those results are not multiplied into the selected speedup.

The [compact evidence manifest](<evidence/ascend-scale-lut-20261122/manifest.json>) distinguishes
historical full-model validation from the newly trimmed PR test suite. It retains all-rank tensor
hashes, [rank timing summaries](<evidence/ascend-scale-lut-20261122/rank-timings.json>), native
source/library hashes and a deterministic [raw selected-receipt archive](<evidence/ascend-scale-lut-20261122/raw-selected-receipts.tar.gz>).
The full original experimental bundle and commits remain preserved rather than rewritten.
Checkpoint weights and full tensor payloads are not copied into the PR; the latter were compared
on the Ascend server and remain represented here by their hashes.

Tests and builds run only on Linux, never on the macOS development machine. The historical
Ascend model measurement is not rerun merely to trim unused optional branches. Fresh portable
CPU/mock tests validate the slim integration without claiming a new Ascend performance result.
Original checkpoint identities and shared CI/CANN configuration were unchanged, and the
historical [lease-release receipt](<evidence/ascend-scale-lut-20261122/safety/release.json>) is retained.

**50 TPS is not reached.** It still requires 20 ms/token, versus approximately 93.5 ms/token on
the selected path. Attention/cache/HC/head and collective work remain the next useful focus,
not an unmeasured claim that a larger FP4 tile alone meets the target.
