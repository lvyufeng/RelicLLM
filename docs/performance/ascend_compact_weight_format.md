# Lossless compact FP4 weights in Ascend Cube order

This is an opt-in extension of the merged original-format decode experiment. The checkpoint's
represented values, E2M1 nibbles, E8M0 K32 scale bytes and FP8 activation grid do not change. The
physical byte order does change, explicitly, and is not silently substituted into the original
loader or consumer.

## Measured result

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
