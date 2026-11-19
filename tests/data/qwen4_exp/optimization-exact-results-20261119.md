# Exact Qwen NVFP4 optimization results — 2026-11-19

## Result and scope

The combined exact decoder measured **59.8137 decode tok/s** in the final native A/B pair
(**58.5222 → 59.8137**, +2.21%). Cat-free linear alone separately qualified warm graph reuse at
**59.0144 tok/s**. These are different runs and lifecycle scopes, not additive gains.
The target of 100 tok/s is **not achieved**. No 72-tok/s upper bound or RTX 2080 Ti qualification is
claimed.

All native proofs use **261632 input + 512 output = 262144 total tokens**, `indexer_budget=2048`,
pure TP8 on eight RTX 4090 cards, NVFP4 expert weights, BF16 dense/KV tensors and FP32 indexer state.
The first generated token belongs to prefill: decode throughput uses **511 decode tokens**.

The runtime/kernel changes are opt-in. No repository commit was created and the pre-existing working
changes were preserved.

## 1. Cat-free strict-row projection: accepted

The strict verifier already calls `F.linear` independently at M=1 for each small BF16 row span.
[The new helper](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/linear_out.py>) writes
those rows into one output allocation through `aten.linear.out`, avoiding the final concatenation.
It retains M=1 staging, dtype and leading output shape; it is not a bmm, FP16 mirror or TF32 mode.

Higher-rank preserve-shape `linear.out` was rejected by actual CUDA tests. The production helper
refuses that form before dispatch and the [binding](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/layers.py>)
requires the explicit two-dimensional staging flag. Bias and no-bias, mutable operands, output
storage offsets and graph replay are covered by [the tests](</Users/lvyufeng/Projects/RelicLLM/tests/test_qwen4_exp_linear_out.py>).

### Native four-arm A/B

| Arm | Decode tok/s | Including request cleanup tok/s | Prefix preparation |
|---|---:|---:|---|
| Off 0 | 54.5131 | 53.6799 | Full prefill, 496.297 s |
| On 1 | 58.0435 | 56.7226 | CPU restore, 55.146 s |
| Off 2 | 57.6392 | 56.1694 | CPU restore, 53.219 s |
| On 3 | 58.7341 | 57.8469 | CPU restore, 52.715 s |

Every arm matched the original 512 greedy ids, all callbacks and all eight ranks. Every arm used
233 verifier rounds and accepted-prefix histogram `[77, 80, 35, 41]`. The final same-host pair gives
**+1.90%** decode rate. The cold first arm is not used to inflate the gain.

Source: [all native ids and rank/callback records](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-linear-native512-20261119.jsonl>).
Restored-prefix arms qualify decode arithmetic; their restore times are not fresh prefill speed or
production prefix-cache latency.

### Final guarded warm session

| Item | First request | Second request |
|---|---:|---:|
| Full prefill | 495.613 s | 493.603 s |
| Graph build | 5.963991 s | 0 |
| Total graph setup | 5.984327 s | 0.008146 s |
| Decode rate | 55.3605 tok/s | **59.0144 tok/s** |
| Request cleanup included | 55.3524 tok/s | 59.0061 tok/s |
| Original greedy ids | 512/512 | 512/512 |

Both requests performed a full prefill; there was **no prefix reuse**. The pool had one adoption,
one hit, no address/boundary invalidations, and all rank/callback ids remained exact. The final
session close cost **0.543735 s**, separately reported rather than hidden in the warm-request rate.

Source: [warm-session lifecycle and all output ids](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-linear-native-warm-20261119.jsonl>).

### How to opt in

Keep the previously qualified native TP8 settings. Add:

```bash
export POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT=1
export POCKETLLM_QWEN4_LINEAR_OUT=1
export POCKETLLM_QWEN4_LINEAR_OUT_2D=1
# For the backend's request graph ownership pool, separately opt in:
export POCKETLLM_QWEN4_MTP_SESSION=1
```

[The reusable baseline environment](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-env-20261119.sh>)
keeps experimental dispatches off until a probe explicitly selects an arm. `rank_chunk` reduction
is hardware-scoped; qualify a new host instead of assuming the stored native reference transfers.

## 2. NVFP4 compact launch and ordered store: small model gain

[The independent kernel entries](</Users/lvyufeng/Projects/relic-core/relic_core/csrc/qwen4_exp_nvfp4_exact.cu>)
retain the original 256-thread implementation. The compact twin uses the same grid, four active
column-owning warps, dot/FMA order, route accumulation and zeroed buffers, but launches 128 threads.

The store twin uses that launch and a single ordered register accumulation per output column.
Two explicit `mul.rn.ftz.f32` operations retain the original product order and `add.rn.ftz.f32`
retains the global float atomic's tested semantics. Every output column is stored, including +0 for
no owned route. The gate/store ownership predicates are checked before removing activation/output
zero fills. Neither the expert format nor activation precision is changed.

### Independent operator evidence

- [Compact CUDA run](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-compact-cuda-20261119.log>): **355 passed**, no skips.
- [Store CUDA run](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-store-cuda-20261119.log>): **437 passed**, no skips.
- [Actually unchanged binary versus additive overlay](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-overlay-cuda-20261119.log>): **216 passed**, no skips, plus captured rotating-bank timing.

The last run keeps every baseline operation from the original module and exposes only the two new
NVFP4 entries from the experimental shared object. Both binary hashes are recorded. The kernel micro
uses eight banks totaling 1.416 GB and seven samples; it is not measured HBM traffic or model TPS.

| TP8-distributed route micro | Store / legacy median operator time |
|---|---:|
| 1 row | 0.9927 |
| 4 rows | 0.9606 |
| 512 rows | 0.7413 |

Source: [independent-binary micro samples](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-overlay-micro-20261119.jsonl>).
Compact alone slowed one-row decode and did not improve the real four-row TP8 micro; its default
minimum is now 128 rows. That threshold is a policy between measured small and large spans, not a
measured crossover.

### Four full native requests, no prefix restore

| Store arm | Full prefill (s) | Decode tok/s | Request cleanup included tok/s |
|---|---:|---:|---:|
| Off 0 | 505.849 | 54.4759 | 53.5864 |
| On 1 | 486.598 | 59.5431 | 58.6966 |
| Off 2 | 486.908 | 59.1071 | 58.0249 |
| On 3 | 485.637 | 59.5624 | 58.5857 |

All four arms matched every original greedy id, callback and rank; all used 233 rounds and histogram
`[77, 80, 35, 41]`. The stable final pair saves **1.271 s of prefill (0.26%)** and gives **+0.77%**
decode rate. The first-arm warm-up difference is not a 4% prefill optimization. The approximately
26% operator saving does **not** become a 26% model saving.

Source: [full-native store proof](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-store-native512-20261119.jsonl>)
and [computed final-pair summary](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-store-native-summary-20261119.json>).
Store and compact remain default-off; the sub-one-percent stable model delta does not justify an
automatic rollout. The additive [overlay installer](</Users/lvyufeng/Projects/RelicLLM/relicllm/models/qwen4_exp/nvfp4_overlay.py>)
and the probe's explicit `--nvfp4-overlay-so` retain the qualified base module rather than silently
replacing quantization, GDN or QSA bindings.

### Combined cat-free projection and ordered store

The final combination was tested on the same native request against the original greedy reference:

| Combined arm | Decode tok/s | Request cleanup included tok/s | Prefix preparation |
|---|---:|---:|---|
| Off 0 | 54.2400 | 53.6258 | Full prefill, 491.582 s |
| On 1 | 59.6958 | 58.8006 | CPU restore, 54.837 s |
| Off 2 | 58.5222 | 57.6930 | CPU restore, 52.576 s |
| On 3 | **59.8137** | **59.1566** | CPU restore, 48.861 s |

Every arm matched all 512 original greedy ids, rank-0 callbacks and all eight ranks. All four used
233 rounds and the same accepted-prefix histogram `[77, 80, 35, 41]`. The final pair gives **+2.21%**
decode rate, or +2.54% including request cleanup. The first cold arm is excluded from that comparison.

Source: [combined output, flags and all rank/callback records](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-combined-native512-20261119.jsonl>)
and [independently checked summary](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-combined-native-summary-20261119.json>).
Later arms restore a CPU prefix; they are not a measurement of combination prefill acceleration.
This run did not use a warm graph session and is not an HTTP-serving qualification. It does not
establish that the individual linear/store gains add, since they were measured in different runs.

The combination is a reproducible **optional** mode, not the new default:

```bash
# Retain the qualified native environment and the original base extension.
# Set the actual per-host checkpoint and measured additive .so paths explicitly.
source tests/data/qwen4_exp/optimization-exact-env-20261119.sh
torchrun --nproc_per_node=8 --master_port=63181 tests/probe_qwen4_exp_optimization.py \
  --model "$CHECKPOINT" --context-tokens 261632 --max-new-tokens 512 \
  --chunk-size 512 --draft-tokens 3 --switch combined --variants off,on,off,on --reuse-prefill \
  --nvfp4-overlay-so "$OVERLAY_SO" \
  --reference-jsonl tests/data/qwen4_exp/r362_norm_fixed_native512.jsonl \
  --output-jsonl /tmp/qwen4-combined-new-run.jsonl
```

The measured base binary SHA256 is `d0b85ea554fcaa0ad87751783e92dda5e933b09300ad0a4376ad95a1ddc6f1c3`;
the additive operator binary is `1e30599d0be2cbaec257e6b379c3ffc43a978589d275af065d91af13fa847254`.
A different build or host must repeat qualification; do not assume native greedy parity transfers.

## 3. Hotspot census and rejected evidence

[Full kernel names and family counts](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-kernel-census-20261119.json>)
replace the old lossy `std::enable_if` bucket; [real projection geometries](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-linear-shapes-20261119.json>)
record shape, dtype, stride and callsite without changing arithmetic. Profiler spans are explicitly
instrumented and are not an achievable floor. Four rows of one projection are adjacent and may
reuse L2: multiplying the whole model's dense bytes by four is not measured DRAM traffic.

The build host's unchanged and rebuilt OFF controls both diverged from the other host's stored
native reference at output index 78. Those [unchanged-control](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-original-native-crosshost_REJECTED-20261119.jsonl>)
and [rebuilt-control](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-store-native-crosshost_REJECTED-20261119.jsonl>)
runs remain rejected. They do not establish a store-kernel defect, a speed result or native
cross-host parity. Model promotion was re-gated on the already reference-qualified host.

## 4. Verification and remaining work

[Final CUDA runtime regression](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-runtime-cuda-20261119.log>):
**1828 passed, 34 skipped**; the two unknown `slow` marker warnings predate this round.

[Final Mac/CPU regression](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-runtime-cpu-20261119.log>):
**1744 passed, 146 skipped**, with the same three pre-existing platform failures: two automatic
backend selections refuse CPU, and the pinned shard clone test assumes CUDA. No failure was added
to a baseline file. Later overlay teardown and immutable-load guards have **33 CPU contract passes**;
these are not additional GPU claims.

[The corrected audit](</Users/lvyufeng/Projects/RelicLLM/tests/data/qwen4_exp/optimization-exact-audit-20261119.md>)
records the rejected preserved-rank mode, timing reconciliation, identity/lifetime rules and host
scope. The next material gains still have to come from the verifier's dense/communication budget
or a qualified private-head change. No K2, bmm allowlist, approximate top-k, reduced indexer budget,
or unmeasured proposal-quality gain is enabled by this round.
