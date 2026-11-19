# Exact HC4 up-mix candidate: operator win, end-to-end stop-loss

Round of 2026-11-19. Scope: the hyper-connection mixer's up projection, gate and stream fold on the
8x RTX 4090 TP8 host. This record answers one question: how much decode throughput does shaving a
genuinely hot kernel actually buy at the current operating point?

## What was built

`qwen4_exp_hc_up_mix_exact` is a new, independent CUDA entry in relic-core
(`relic_core/csrc/qwen4_exp_hc_mix_exact.cu`). The qualified kernel runs one warp per output element
and walks the four streams in sequence, so each stream re-reads the same activation row and each
stream's dot product is a single dependent FMA chain. The candidate loads that activation once per
`k` and keeps four independent accumulators.

The arithmetic is deliberately the qualified kernel's, not an approximation: the same lane-strided
`k` order (`k = lane, lane + 32, ...`), the same words fed to the same FMA, the same `shfl_down`
16..1 tree, the same `1 / (1 + expf(-x))` gate, the same stream-ordered fold starting from `+0.0f`,
and one BF16 cast of `folded / hc`. Only the loads are shared, so raw-bit equality is the expected
outcome rather than a hope, and the test asserts it word-wise instead of by tolerance.

It stays off by default (`POCKETLLM_QWEN4_HC_UP_MIX_EXACT=0`). When the switch is on and the entry is
missing, `layers.py` raises instead of silently falling back.

## Gate 1 - the operator is bit-identical

13 passed, 0 skipped, 0 failed on the leased host. Reference: the qualified module
`d0b85ea554fcaa0ad87751783e92dda5e933b09300ad0a4376ad95a1ddc6f1c3`. Candidate: the isolated build
`f2280057436d07464a132743d5b79322584b47d10cb6b1759aa95b6747aab8af`, installed as the *only* extra
entry (`replaced_ops: []`), so every other operator in the process keeps its qualified identity.

| Check | Result |
|---|---|
| Raw BF16 words, rows 1 / 4 / 512 | identical |
| Thread widths 64 / 128 / 256 / 512 | all identical to the oracle |
| One mutated weight word | both sides move identically; mutation observable |
| CUDA-graph capture then replay over mutated inputs | identical after both replays |
| Wrong row count, width, hc, hidden, dtype, non-contiguous, bad threads, CPU | refused before any launch |
| Source contract (single activation load, shuffle tree, sigmoid, fold order, scope guard) | holds |

## Gate 2 - the operator is faster (graph-captured device time)

A per-call Python loop is not a kernel measurement here: the host floor is about 47 us a call, which
is several times either kernel, so it hides the effect (that arm is kept as `--bench-host` only as a
counter-example). The runtime always replays these launches inside a CUDA graph, so the honest
measurement captures 200 launches in one graph and divides, alternating the arms so neither arm sees
a different clock state.

| Rows | Qualified | Candidate | Ratio | Candidate bandwidth |
|---|---|---|---|---|
| 4 (decode verify) | 10.68 us | 8.41 us | 0.788 | 779 GB/s |
| 1 | 4.30 us | 3.35 us | 0.781 | - |
| 512 | 1001.3 us | 772.8 us | 0.772 | - |

**-21.2% device time at the decode shape.** At 8.41 us for the 6.55 MB of mixer weights it reads,
the candidate is close to this card's bandwidth limit; the kernel has little left in it.

## Gate 3 - composition, on the native request

Native 261632 input + 512 output, budget 2048, BF16 dense/KV, verified against the frozen original
greedy ids (`reference_sha256 c16853f71d571cd76f310ff9bfe1e9121fb31e1f5f65cdce03f8ccfb98c621e0`),
three interleaved off/on pairs with CPU prefix restore for the later arms.

| Trial | Variant | Valid | decode tok/s |
|---|---|---|---|
| 0 | off (full prefill) | yes | 52.483 |
| 1 | on | yes | 55.771 |
| 2 | off | yes | 56.649 |
| 3 | on | yes | 56.299 |
| 4 | off | yes | 57.266 |
| 5 | on | yes | 57.358 |

Every arm reproduced all 512 original greedy ids with equal callbacks and equal ids on all eight
ranks, at 233 graph rounds and 5 tail steps - so enabling the candidate changes no token and no
acceptance decision.

Warm pairs: -0.62% and +0.16%, mean
-0.23%. The first pair is excluded: trial 0 pays the full
prefill and starts from a cold device. The spread across the three `off` arms is
9.11%, which is the scale of the noise that arm order and device state
impose here.

## Why the end-to-end number could not move

| Quantity | Value |
|---|---|
| Decode rounds in the request | 233 |
| Generated tokens per round | 2.193 |
| Round wall at the baseline arm | 38.30 ms |
| Mixer launches per round (2 a layer, 48 layers) | 96 |
| Kernel time per round | 1.025 ms |
| **Share of the round** | **2.68%** |
| Expected gain from -21.2% | **+0.57%** |
| Observed warm mean | -0.23% |
| Gain needed for 100 tok/s | **+74.6%** |

The kernel is 2.7% of a round, so cutting it by a fifth is worth about half a percent - inside the
noise this host produces even with interleaved pairs. **Decision: keep the candidate available and
off by default; do not claim it as a throughput step.**

## What this establishes for the 100 tok/s target

The same arithmetic applies to every remaining single kernel. The measured per-family busy time in
this operating point puts the mixer at ~1.0 ms of a ~38 ms round; the store combination (an
architectural change that removed a whole extra pass) was worth +2.2%, and the cat-free projection
+1.9%. A 100 tok/s target needs **+75%**. No kernel left in the
trace carries that much: the remaining budget sits in thousands of small copies and elementwise ops
(~12 ms of traced busy, spread over ~10k launches) and in the two collectives a layer. Reaching the
target therefore requires one of:

1. **Higher accepted tokens per round.** 2.19 today against a cap of 4
   at K=3; +75% would need ~3.7 tokens a round, i.e. near-total
   draft acceptance, which the private head does not currently deliver.
2. **Fewer, larger model passes.** Dense/KV quantization changes the numerical contract and cannot be
   reported against the original greedy reference.
3. **A cheaper communication topology.** Two collectives a layer, at 4090 PCIe latency, is the
   structural difference from the layer-split engines; it needs a new runtime path, not a kernel.

Kernel micro-optimisation at this operating point is exhausted as a route to the target.

## Reproduction

```bash
# operator: build the isolated candidate, then compare against the qualified module
python tests/build_hc_exact_micro_extension.py                     # prints the .so path
Q4_HC_EXACT_SO=<that .so> python -m pytest tests/test_qwen4_exp_hc_up_mix_exact.py -q
Q4_HC_EXACT_SO=<that .so> python tests/test_qwen4_exp_hc_up_mix_exact.py --bench

# composition: native request, three interleaved pairs
torchrun --nproc_per_node=8 --master_port=63281 tests/probe_qwen4_exp_optimization.py \
  --model "$CHECKPOINT" --context-tokens 261632 --max-new-tokens 512 --chunk-size 512 \
  --draft-tokens 3 --switch hc_up_mix --variants off,on,off,on,off,on --reuse-prefill \
  --hc-exact-overlay-so <candidate.so> \
  --reference-jsonl tests/r362_norm_fixed_native512.jsonl --output-jsonl /tmp/hc-native.jsonl
```

## Artifacts

- `optimization-exact-hc-up-mix-native512-20261119.jsonl` - the six native arms, verbatim.
- `optimization-exact-hc-up-mix-summary-20261119.json` - pairs, ceiling arithmetic, identities.
- `hc-up-mix-exact-device-4090-20261119.json` - graph-captured operator timings, both binaries hashed.

## Limits

The candidate was measured on one owned 8x4090 host with the qualified binary as oracle. It is not
HTTP-qualified, and the composition run used CPU prefix restore for the later arms rather than a
fresh prefill for each. The end-to-end numbers are decode-only: prefill, graph setup and cleanup are
excluded, as in every earlier record.
