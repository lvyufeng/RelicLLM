# Qwen3.8-Flash-Next optimization implementation — 2026-11-18

## Decision

**The verified gain is compatible-request graph/arena reuse, not a claimed 100 tok/s kernel breakthrough.** The new GDN epilogue is numerically qualified but stays off by default because its measured steady-state gain was not distinguishable from run variability.

All model measurements used owned 8×RTX 4090 hosts, pure TP8, batch 1, NVFP4 routed experts, BF16 dense/KV tensors, FP32 recurrent/indexer state, native `indexer_budget=2048`, and the qualified row-invariant verifier. No checkpoint weights, context budget, routing budget, or output length were reduced.

## 1. Two complete native-context requests

Source: [native warm-request evidence](</Users/<user>/Projects/RelicLLM/tests/data/qwen4_exp/optimization-native512-warm-20261118.jsonl>). Each request had **261632 input + 512 output = 262144 total tokens**. Both performed their entire prefill; there was no PD import or prefix-cache shortcut. All 512 output IDs matched the original plain-greedy reference, rank-0 callbacks matched those IDs, and all eight ranks agreed.

| Measured field | First request: new recordings | Second request: reused recordings |
|---|---:|---:|
| Full prefill, seconds | 497.897725 | 489.854031 |
| Graph build, seconds | 5.743099 | **0** |
| Graph setup/alignment, milliseconds | 5774.829 | **8.775** |
| Reported rolling-state reset, milliseconds | 0 | 4.302 |
| Decode rate excluding setup/final session release, tok/s | 54.495418 | **57.992599** |
| Per-request cleanup, milliseconds | 1.423 | 1.291 |
| Verifier replays | 233 | 233 |
| Draft replays | 232 | 232 |
| Startup replays | 1 | 1 |
| Exact one-row native-boundary tail replays | 5 | 5 |
| Inner-owner boundary suspensions | 2 | 2 |

**Setup fell by 99.848%, saving 5.766 seconds for the compatible second request.** Retained-buffer identity checks passed and the pool recorded one adoption, one hit, one resume, and no boundary/address invalidations. The native tail now reversibly suspends inner bindings rather than permanently disarming otherwise reusable recordings.

Accounting matters:

- Final session close cost **0.532784 seconds**, recorded separately, not erased.
- Charging that entire final close to the second request gives **54.602061 tok/s**, including its setup, reported reset, and per-request cleanup, but excluding prefill.
- Across these two requests, charging all setup/reset/per-request cleanup/final close gives **41.694190 tok/s**, excluding the two prefills.
- The approximately 58 tok/s figure is a **steady decode measurement**, not an end-to-end request rate or a demonstrated hardware ceiling.
- Prefill still takes roughly 490–498 seconds. This is not a subsecond-TTFT or prompt-cache claim. Two requests are not sufficient to establish a repeatable 6.4% steady-TPS improvement; the eliminated recapture cost is the robust gain.

The separate [8K warm test](</Users/<user>/Projects/RelicLLM/tests/data/qwen4_exp/optimization-warm8k-20261118.jsonl>) qualified three full 128-output requests, including a different middle-of-prefix token history. Both compatible follow-ups had zero recapture and matched their respective plain references on every rank.

## 2. Actual OpenAI SSE lifecycle qualification

Source: [HTTP evidence](</Users/<user>/Projects/RelicLLM/tests/data/qwen4_exp/optimization-http-session-20261118.jsonl>). This used the standard OpenAI handler, backend scheduler, and backend worker broadcast transport—not an HTTP-looking mock around direct model calls. The decoded/re-tokenized HTTP prompt had **7893 input tokens**; this must not be called an exact 8192-token HTTP input.

| HTTP request | Internally produced tokens | Graph setup, ms | Reused | Result |
|---|---:|---:|---|---|
| First full request | 128 | 5424.466 | No | All IDs/callbacks/ranks/text correct |
| Compatible full request | 128 | 9.352 | Yes | No recapture; all checks correct |
| Client disconnect after first text event | 4 | 10.128 | Yes | All ranks cancel; exact greedy prefix; bundle discarded |
| Request after cancellation | 128 | 5234.653 | No | Fresh capture; complete original greedy output |

The four-request summary passed. Shutdown ended with zero active MTP requests and a closed backend.

The [initial HTTP failure log](</Users/<user>/Projects/RelicLLM/tests/data/qwen4_exp/optimization-http-initial-failure-20261118_INVALID.log>) is retained as invalid evidence: two normal requests passed, then disconnect exposed a bounded-queue deadlock. Waiting for a producer before notifying cancellation allowed rank 0 to park inside `put()` while peers waited in a collective. The fix notifies reader closure before joining, uses cancel-aware bounded writes for MTP, and guarantees terminal delivery without blocking a vanished reader. Callback exceptions still travel through the existing collective-safe cancellation protocol.

## 3. Exact GDN normalization epilogue

Implementation: [native CUDA operator](</Users/<user>/Projects/relic-core/relic_core/csrc/qwen4_exp_gated_norm_exact.cu>) and [model-side dispatch](</Users/<user>/Projects/RelicLLM/relicllm/models/qwen4_exp/layers.py>).

The exact entry keeps torch's FP32 statistic/reduction tree and sigmoid/SiLU activation. It fuses the proven multiply/cast epilogue while preserving all three BF16 rounding boundaries: normalized value, BF16 weight product, and final output. Explicit PTX `mul.rn.f32` and integer RNE prevent the project's global fast-math flags from changing those boundaries. CUDA NaN canonicalization is architecture-specific; real-device subnormal/overflow/NaN tests were added, and the CPU sweep's negative sign-bit coverage was corrected. The separately named activation-fused experiment is not used by the runtime.

Evidence:

- [CUDA kernel regressions](</Users/<user>/Projects/RelicLLM/tests/data/qwen4_exp/optimization-cuda-kernel-tests-20261118.log>): **270 passed, zero skipped**, including adjacent NVFP4/QSA tests.
- [Real activation oracle](</Users/<user>/Projects/RelicLLM/tests/data/qwen4_exp/optimization-real-norm-oracle8k-20261118.jsonl>): **648 calls per rank, zero differing elements on every rank**.
- [Native 512-output A/B](</Users/<user>/Projects/RelicLLM/tests/data/qwen4_exp/optimization-native512-norm-20261118.jsonl>): all four arms matched all original greedy IDs and callbacks/ranks, with the same 233 verification rounds.

| Native arm | Prefix preparation | Exact epilogue | Steady tok/s |
|---|---|---|---:|
| 0 | Full prefill | Off | 53.800742 |
| 1 | Same frozen prefix restored | On | 56.180539 |
| 2 | Same frozen prefix restored | Off | 57.275644 |
| 3 | Same frozen prefix restored | On | 57.412454 |

The first arm's preparation differs from subsequent arms, so it is not a valid isolated speed comparison. The last comparable off/on pair differs by **0.24%**, too little to claim a robust throughput win. The [controlled 8K A/B](</Users/<user>/Projects/RelicLLM/tests/data/qwen4_exp/optimization-norm8k-20261118.jsonl>) also overlaps in rate. Consequently `POCKETLLM_QWEN4_GATED_NORM_EXACT` remains **0 by default**.

The [initial uncontrolled norm run](</Users/<user>/Projects/RelicLLM/tests/data/qwen4_exp/optimization-norm-initial-uncontrolled-20261118_INVALID.jsonl>) is not a speed result: its first head prefill used FP8, later prefills used the one-way NVFP4 head replacement, and one arm differed at token 67. Controlled tests pre-quantize the private head before references/arms to remove this confound.

## 4. Ownership, reset, failure, and compatibility

[Graph sessions](</Users/<user>/Projects/RelicLLM/relicllm/models/qwen4_exp/graph_session.py>) own both request arenas and all dependent recordings/pinned staging buffers. Compatible checkout resets GDN/PLE state, lengths/history and indexer counts in place; unused multi-GiB position-addressed arenas are not zeroed. Bindings suspend in reverse construction order and resume forward. RoPE tables retain their allocation, and tensor fingerprints include pointers, shapes, **strides**, dtypes and devices.

[Runtime integration](</Users/<user>/Projects/RelicLLM/relicllm/models/qwen4_exp/runtime.py>) exchanges TP readiness votes before mixed-rank recapture/reuse and validates resumption collectively before replay/alignment. Moved buffers fail the whole group at a safe seam. Failed prefills/adoptions, cancellation and suspended-owner release are covered; per-request counters are offsets of lifetime counters, not reset totals.

[Serving integration](</Users/<user>/Projects/RelicLLM/relicllm/backends/qwen4_exp_backend.py>) uses an opt-in pool, charges setup/cleanup, skips session checkout for one-token or insufficient-verifier-room tails, and quiesces active MTP before freeing graphs or broadcasting shutdown. Different widths, draft depths, model identities or math flags do not silently reuse incompatible recordings.

A speculative BF16 `bmm` replacement for serial M=1 linear calls was also rejected: the CUDA micro-oracle found differing output bits, including 2/10240 elements for a 4-row 2560→2560 projection. Mathematically equivalent batching is not a valid exact-verifier replacement.

## 5. How to select the verified implementation

Keep the already qualified NVFP4, native indexer, row-invariant reduction, split-K, startup and tail flags. Add:

```bash
export POCKETLLM_QWEN4_MTP_SESSION=1
export POCKETLLM_QWEN4_GATED_NORM_EXACT=0
export POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT=1
export POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT=1
# Serve from the updated runtime, with the already qualified NVFP4 flags:
python -m relicllm serve --model <checkpoint> --backend qwen4_exp \
  --device cuda --tensor-parallel-size 8 --max-model-len 262144 \
  --backend-option expert_resident=true --backend-option prefill_chunk=512 \
  --speculative-method mtp --speculative-tokens 3
```

The pool currently uses exact arena-width buckets and capacity one. Mixed request lengths/budgets may therefore recapture; no arbitrary-length reuse claim is made. This work qualifies isolated test servers, not an already-running production endpoint.

Validation: [CUDA-host runtime suite](</Users/<user>/Projects/RelicLLM/tests/data/qwen4_exp/optimization-cuda-runtime-tests-20261118.log>) **1226 passed, 34 skipped**, with the checkpoint-dependent/inapplicable skips kept distinct from passes. Latest targeted model, lifecycle, queue and shared-backend contracts passed **181 CPU tests**. Python compilation and scoped diff checks passed. The broad local Mac suite additionally exposed three existing platform-specific routing/pinned-memory failures; the complete target-CUDA suite passed. Existing uncommitted work was preserved; no commits were made.

**Not achieved: 100 tok/s, a new native-context hardware ceiling, cross-host native PD streaming, or a validated batched GEMV kernel.** The next substantial steady-state target remains exact main-verifier GEMV/fusion work; graph reuse has removed a separate request-lifecycle cost.
