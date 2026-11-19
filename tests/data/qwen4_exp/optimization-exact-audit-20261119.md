# Exact arithmetic and measurement audit — 2026-11-19

## Standing

This is the integrated audit, corrected against the actual device logs and disjoint timing budgets.
The independent reviewer performed only CPU tests. The root subsequently checked the device
artifacts, repaired the accepted surface and verified the final runtime on CUDA. A skip is not a
pass. These are `sm_89` results, not qualification of `sm_75`.

The original findings and two rejected experiments remain visible; none was converted into a
successful throughput result.

## 1. Preserve-rank `linear.out` is rejected

The original device run collected 269 tests: **22 failed, 247 passed**. The actual failed set is:

- 20 `test_gpu_bits_match_the_recorded_reference_for_each_real_shape` cases: preserved-rank staging,
  the two rank-three/four leading geometries, and ten of the eleven tested projections.
  Hyper-connection `10240 → 320` was the exception, not a failure.
- `test_gpu_adversarial_vectors_stay_bit_identical[leading1-same]`.
- `test_gpu_result_survives_a_captured_replay[same]`.

The [original device log](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-linear-out-initial_REJECTED-20261119.log>)
contains the node ids. Both the adversarial and capture cases **failed**. The preliminary review's
claim that they passed, and therefore that identical operands both failed and passed in this run,
was an incorrect inference from the total count. There is no measured nondeterminism claim here.

Two-dimensional staging passed all of its original device gates, including bias. A separate
[real-shape micro](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-linear-out-2d-micro-20261119.jsonl>) checked
17 bias-free projection geometries, eight operand mutations and captured replay: every compared
BF16 word matched. The model call sites are bias-free; biased and bias-free paths are now both
covered by the formal device suite.

### Implemented guard

[The helper](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/linear_out.py>) returns
`None` before inspecting or dispatching ATen if `x.ndim > 2` without
`POCKETLLM_QWEN4_LINEAR_OUT_2D=1`. The caller falls back to the unchanged per-row reference loop.
At rank two both staging requests are literally `(1, K)`, so the helper can serve either.

[The model binding](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/layers.py>) additionally
uses `linear_out_enabled()`: the optimization, strict-row invariant and two-dimensional staging
switches must all equal `"1"`. The helper and binding now use the same literal parser;
`true` and `yes` do not silently select a different surface.

[The suite](</Users/lvyufeng/Projects/RelicLLM/tests/test_qwen4_exp_linear_out.py>) asserts refusal
with zero dispatch, rather than deleting failing cases, using tolerances, or adding expected
failures. It retains a direct-ATen CPU diagnostic outside the production helper. The formal gate
covers bias and no-bias, rank-two and higher-rank inputs staged as matrices, adversarial operands,
row offsets and graph capture/replay. Before CUDA execution the updated module reported
**421 CPU passes and 97 device skips**.

## 2. The throughput delta reconciles

In the final 8K pair, both arms delivered the same 128 ids, callbacks and all eight ranks, with
46 verifier rounds and accepted-prefix histogram `[8, 9, 15, 14]`. The immutable launcher explicitly
exported `POCKETLLM_QWEN4_LINEAR_OUT_2D=1`; it did not measure the rejected staging mode.

The [disjoint budget](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-linear-8k-budget-20261119.json>)
accounts for the measured **76.5007 → 79.6871 tok/s** increase:

| Saving in the final pair | Seconds |
|---|---:|
| Decode total | 0.0663818 |
| Verifier total | 0.0296213 |
| Private head | 0.0368091 |
| Rewind | -0.0005779 |
| Other | 0.0005293 |

The rows sum to the decode saving. Verifier device time fell by 0.0298108 seconds; staging is
**inside** verifier time and is not added a second time. The preliminary review omitted the head
saving. The +4.16% total-rate observation is not a +4.16% verifier-device-only claim, nor a guarantee
of head speedup for another context or proposal schedule.

At native context, the [four-arm artifact](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-linear-native512-20261119.jsonl>)
matched every one of the original 512 greedy ids, callbacks and all eight ranks. Each arm used
233 rounds and histogram `[77, 80, 35, 41]`. The final pair was **57.6392 → 58.7341 tok/s** excluding
prefill/setup/cleanup. This is a measured +1.90%, not 100 tok/s and not a new theoretical ceiling.
The later arms restored a CPU prefix; their 52–55 second restore times are not fresh prefill or a
production prefix-cache latency claim.

## 3. Probe and graph ownership

[The optimization probe](</Users/lvyufeng/Projects/RelicLLM/tests/probe_qwen4_exp_optimization.py>)
now checks the actual `aten.linear.out` overload and explicitly selects/records the matrix staging
for a linear-out A/B. It fingerprints the helper, probe, reference and native binary; each arm
records its complete `POCKETLLM_*` environment. The private head is NVFP4-quantized **before** any
reference or arm, avoiding the previous one-way FP8-to-NVFP4 proposal confound.

[The session key](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/graph_session.py>) hashes
every `POCKETLLM_*` pair, so changing the optimization, staging or launch threshold cannot replay an
old captured dispatch. An explicit operator overlay also exports its SHA256 into that namespace.
Non-`POCKETLLM_*` CUDA/BLAS math selectors must remain fixed while a session is active; this round
introduces no TF32 relaxation or low-precision dense/KV path.

The final guarded runtime's [CUDA regression](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-runtime-cuda-20261119.log>)
reported **1828 passed, 34 skipped**, with only the two pre-existing unknown `slow` marker warnings.
Checkpoint-dependent skips are not qualification of an unrun end-to-end case.

## 4. NVFP4 launch and store invariants

[The kernel](</Users/lvyufeng/Projects/relic-core/relic_core/csrc/qwen4_exp_nvfp4_exact.cu>) keeps the
original 256-thread entry and independent 128-thread compact/store names. The grid still budgets
four columns per CTA. Warps 0–3 own the same columns in both launch geometries; surplus oracle warps
perform no column arithmetic. Shared input staging covers the same words before a block barrier.

The store entry preserves every route's dot/FMA tree, keeps `(acc * route_weight) * scale` as two
explicit `mul.rn.ftz.f32` operations, and accumulates routes in their original order using
`add.rn.ftz.f32`. The old global atomic was already single-writer for an output element; this is not
an unordered reduction being made deterministic.

Every output column is stored exactly once, including `+0.0f` for no owned route. Gate and down
use the same owned-route predicate; down skips before reading a foreign route's activation.
[The source-contract tests](</Users/lvyufeng/Projects/relic-core/tests/test_nvfp4_exact_store.py>) now
assert both predicates agree, in addition to pointer/read ordering and complete write coverage.

The [corrected device run](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-store-cuda-20261119.log>)
reported **437 passed, no skips**, including raw-word comparison of NaNs/signed zero, mutable
capture/replay and the subnormal-cancellation diagnostic. This qualifies the tested `sm_89` form;
other architectures still need the oracle.

Compact is not universally faster. The TP8-distributed-route micro was neutral at four rows,
slower at one row and faster at 512 rows. Its default minimum is now 128: that leaves small strict
row spans untouched. **128 is a dispatch policy, not a measured crossover**; no intermediate row
sweep was done. Store's kernel micro is also not a model-level TPS claim.

## 5. Native qualification is host-scoped

The first full-native store experiment on the build host failed its **OFF** reference before the
candidate was enabled. The [rebuilt-binary artifact](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-store-native-crosshost_REJECTED-20261119.jsonl>)
and [unchanged-binary control](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-original-native-crosshost_REJECTED-20261119.jsonl>)
both first differ from the other host's stored reference at output index 78. Both use the same
Python sources/math flags and deliver internally consistent rank/callback outputs. The unchanged
binary therefore does not implicate the new store epilogue; nor does kernel-level equality qualify
a cross-host native model result. The underlying host/reference difference is not assigned a cause
without further evidence.

These runs are **invalid throughput evidence**. Native promotion is re-gated on the already
reference-qualified host. [The additive overlay installer](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/nvfp4_overlay.py>)
keeps every baseline operation, including quantization/GDN/QSA, from the qualified module and exposes
only the two independently named new NVFP4 operations. It records both binary hashes, rolls back on
failure, and restores attributes after model/graph quiescence. Its original **31 CPU tests** are
contract checks; later retryable-close and immutable-load guards expanded the focused module to
**33 passing CPU tests**. A subsequent independent-module CUDA run compared the new operators with the actually
loaded original binary: **216 passed, no skips**. It separately recorded both binary hashes and
operator timing; kernel equality still does not substitute for the native model gate.

## 6. Completed combination and final decision

The [four-arm combination](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-combined-native512-20261119.jsonl>)
passed all original 512 greedy ids, callbacks and all eight ranks. Every arm retained 233 rounds and
histogram `[77, 80, 35, 41]`. The final pair was **58.5222 → 59.8137 decode tok/s (+2.21%)**, with
later arms using CPU-restored prefixes. This is not a fresh-prefill, warm graph-session or HTTP
result. Both native binary hashes, helper sources, per-arm flags and the 511-token decode denominator
are preserved in the [checked summary](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-combined-native-summary-20261119.json>).

Store alone passed four complete full-prefill requests but only improved the final model pair by
0.77% in decode and 0.26% in prefill, despite its approximately 26% operator micro gain. It and the
combination stay opt-in; no automatic rollout or additive-gain prediction is justified. All managed
GPU work finished and the two owned CUDA host leases were released.

## Conclusion

The accepted linear-out form has a real native-context improvement and an unchanged-greedy proof.
The combined optional mode also passes the unchanged native sequence. The rejected higher-rank form
remains asserted as a refusal. Launch/operator micros, native model proof, graph ownership, prefill
restore and request cleanup are separate claims. No 100-tok/s result, 72-tok/s ceiling, native
cross-host parity or `sm_75` qualification is asserted.
