# The MTP verify graph: what it needs, and what was measured

This is a **historical FP8-only design record**, not the current NVFP4 result. At the time of this
investigation, speculative decoding was correct but not faster because its verify step could not be
captured. That obstacle was subsequently resolved; see the [current NVFP4 status](#current-nvfp4-status-2026-11-18)
at the end. The older arithmetic, head-footprint and throughput measurements below must not be read
as the current captured decoder's state.

Everything below is measured on 8×RTX 4090 (24 GiB each, `sm_89`, no device-to-device peer access),
FP8 weights, native `indexer_budget=2048`, batch 1.

## Where the mechanism stands

| property | value | how |
|---|---|---|
| MTP head loads at 8192 positions | yes, 5.04 GiB free afterwards | measured |
| two-draft round reproduces plain greedy | yes, token for token | measured at 8192 |
| tokens per trunk step | 1.33 (8 tokens in 6 steps) | measured at 8192 |
| a round captured into a graph | **no** | capture raises, falls back to eager |
| plain captured step | 20.14 ms | measured, 8×4090 |
| plain eager step | ~150 ms | measured |

The mechanism works. The 7× gap between a captured step and an eager one is the whole problem: a
round that returns 1.33 tokens has to cost less than 1.33 captured steps to win, and eager costs
about 7.

## What a verify round actually asks of the recorder

A round is: run the head on the last accepted token to draft `K` tokens, then run the trunk once over
those `K` positions, then accept the longest agreeing prefix and rewind the caches to the accepted
length. Recording that step means the graph owns four things it does not own today:

1. **A `K`-row write.** The step writes `K` positions of KV, indexer pool and recurrent state, where
   the plain step writes one.
2. **The per-position recurrent state.** A partial accept rewinds to position `j < K`, so the states
   after every position must be available without replaying. *This now exists*:
   `qwen4_exp_gated_delta_bf16_forward_states` returns the state after each position, and
   `tests/test_qwen4_exp_gdn_states.py` holds the two properties that make it usable -- recording
   does not change the output or the final state (bit-identical to the plain entry point), and the
   state recorded at position `j` is what a scan of `j+1` rows from the same start reaches. That is
   what makes a rewind exact rather than approximate. Five tests, run on the card.
3. **A rewind.** Copying the chosen state back into the cache slot the next step reads.
4. **No host reads.** The accept decision is a host decision, so it happens outside the recording;
   anything the recording itself reads back from the device forbids the capture.

Item 2 was the one the code named as a blocker, and it is done. The capture still fails, for a
different and more basic reason.

## The three measured obstacles

### 1. The MoE changes arithmetic at two tokens

`builder.py` picks the expert backend from the flags, and the two backends do not compute the same
function:

| flags | backend | arithmetic |
|---|---|---|
| `expert_fused_fp8=True` (the shipping configuration) | `Fp8SlotExpertMoE` | fused FP8 kernel pair, no host read |
| `expert_resident=True` alone | `DeviceExpertMoE` | Python slot loop, one host read per slot |

The second backend reads an expert row index back to the host once per slot
(`int(rows[slot])` in `_run_experts_fixed`), which is `cudaErrorStreamCaptureUnsupported` inside a
capture. So it can never be recorded. The first cannot be made to fit alongside the head — see
obstacle 3.

That the two backends differ numerically is also why a verify row and the same row decoded alone do
not agree: measured on one cache state, the same row at the same position differs by up to **0.766**
in the logits depending only on how many rows travelled with it.

### 2. The capture window is not the issue; the shortfall is fixed

The natural first guesses -- a smaller graph window, a shorter context -- were both measured and both
do nothing:

| context | graph window | outcome |
|---:|---:|---|
| 4096 | 8 | out of memory, 22.78 GiB allocated |
| 6144 | 8 | out of memory, 22.79 GiB allocated |
| 8192 | 8 | out of memory, 22.79 GiB allocated |

The demand is the same at every context, so the gap is a fixed cost of about 4.6 GiB that the head
and its buffers add to a trunk already peaking at 18.17 GiB. Shrinking anything that scales with
context cannot touch it.

### 3. The head's pin is all-or-nothing against free memory at build time

`_head_pin_budget` returns `free - reserve`, sampled after the trunk's bank is filled and *before* the
long-context caches exist. With a small reserve the head therefore pins about 5 GiB it cannot afford,
and the caches later run out. Setting the reserve high enough to deny the pin is the right idea and
now does what its comment claims -- `pin_layers` weighs a layer on the host before moving it (fixed;
`tests/test_qwen4_exp_moe_pin_budget.py`) -- but the head's build then fails earlier, inside
`mtp_expert_rows`, which needs transient device memory of its own.

## The ordered plan this implies

Each step is a prerequisite for the next, and each is separately verifiable.

1. **Make one MoE arithmetic serve both row counts.** Take the fused FP8 kernel path and extend it to
   `K` rows, so `shape[0] == 1` is a case of the general path rather than a different one. This is
   the only way a verify round can agree with the decode chain; it is also what makes the recorded
   step free of host reads for any `K`.
2. **Give the head a footprint that fits beside the trunks.** 4.6 GiB fixed. The measured lever is an
   NVFP4 checkpoint, which frees 6.2 GiB of expert bank (per-weight 0.5625 bytes against 1.0078) and
   would cover it. Absent that, the head has to stream its experts per round rather than pin, which
   step 3's fix now lets it decline cleanly instead of failing.
3. **Record the round.** With 1 and 2 in place the recorded step has no host read and fits; the
   per-position state it rewinds with already exists.

The acceptance test for the whole thing is not a speed number: it is that a two- and a three-draft
round reproduce the plain chain token for token **with the round captured**, and that
`tokens/step × 1/step_time` beats 49.02 tok/s. Anything less is a faster wrong answer.

## What is *not* the problem

Three candidate explanations were measured and eliminated, and they are recorded so they are not
re-investigated:

- **The draft loop's bookkeeping.** Accept, rewind and cache snapshot/restore are correct: at 8192 a
  two-draft round is token identical to plain greedy.
- **The indexer's incremental pool.** Disabling it changes the chunking-induced logit shift by
  nothing at all (1.873 max / 0.292 mean either way).
- **The prefill chunk size is not what breaks a round.** Chunking does move the final logits (up to
  1.9, mean 0.25 -- four orders of magnitude more than bf16 rounding), but the round's disagreement
  survives with all contexts and windows held equal, so it is the per-row arithmetic above, not the
  segmentation.

## Historical FP8 status

Correct and uncaptured: verified at 8192, 1.33 tokens a step. Captured was not achieved in this earlier
FP8 investigation; obstacles 1 and 3 needed kernel work rather than configuration.

## Current NVFP4 status (2026-11-18)

Captured TP8 decoding now passes the complete **261632 input + 512 output = 262144 native total**
request, with native `indexer_budget=2048`. All 512 IDs match the original plain-greedy reference,
rank-0 callbacks match the IDs, and all eight ranks agree. The native-boundary suffix consists of
five exact one-row tail graph replays; neither input nor requested output is shortened.

Compatible requests can opt into persistent graph/arena ownership with
`POCKETLLM_QWEN4_MTP_SESSION=1` (default remains off), beside the qualified strict-row, split-K,
startup/tail and NVFP4 flags. On two complete real native requests, graph build was 5.743 seconds on
the first and zero on the second; setup/alignment fell from **5.775 seconds to 8.775 milliseconds**.
Both requests still executed their full prefills (497.9/489.9 seconds). The second request measured
57.993 tok/s in steady decode; final session release cost 0.533 seconds separately, or 54.602 tok/s
when that entire release is charged to the second request along with setup/reset/request cleanup.
This is graph reuse, not prompt caching or subsecond TTFT.

The actual OpenAI SSE backend also passed a full request, a warm full request, client-disconnect
cancellation, and a fresh full request after cancellation. All complete responses and cancelled
prefixes agreed across ranks. A cancel-aware event queue prevents a disappeared reader from
stranding rank 0 inside a bounded token callback.

The additive exact GDN normalization epilogue is CUDA-qualified (270 kernel/regression tests, zero
skips; real activations bit-identical on all eight ranks), but remains off by default with
`POCKETLLM_QWEN4_GATED_NORM_EXACT=0`: the last comparable native off/on pair differed by only 0.24%,
not a robust speed improvement. The current measured benefit is removed per-request recapture cost.
**100 tok/s remains unachieved**, and no native-context hardware ceiling has been established.

## Exact projection and launch follow-up (2026-11-19)

The optional strict-row projection now has a cat-free `aten.linear.out` form. It still issues M=1
BF16 calls, but writes each row into one output allocation. The higher-rank preserved-shape out
form was rejected by device tests; the model binding requires
`POCKETLLM_QWEN4_LINEAR_OUT=1` and `POCKETLLM_QWEN4_LINEAR_OUT_2D=1` beside the existing strict-row
flag. Unqualified operands keep the unchanged reference loop. No bmm, FP16 mirror, TF32 or indexer
budget relaxation is introduced.

Four native requests retained all 512 original greedy IDs, callbacks and all eight ranks, with
233 verifier rounds and unchanged acceptance histogram `[77, 80, 35, 41]`. The final linear-only A/B
was **57.639 → 58.734 tok/s** (+1.90%). A separate guarded warm-session proof performed two complete
prefills and measured **59.014 tok/s**, zero second-request graph build and **8.146 ms** setup.
Final session close was **0.543735 s**, separately reported. This does not imply subsecond TTFT.

Independent NVFP4 compact and ordered-store entries preserve the original expert operand/dot tree.
The 128-thread compact launch was not faster at the real four-row TP8 route geometry and slowed
one-row decode, so it stays optional with a large-row threshold. Ordered store saved about 26% in
a rotating-bank 512-row operator micro, but the stable full-native model pair improved prefill by
only **0.26%** and decode by **0.77%**. The operator micro is not a model throughput prediction.

The combined strict-row projection plus ordered-store A/B measured **58.522 → 59.814 tok/s**
(+2.21%), again matching the complete original greedy sequence and all rank/callback IDs. Later A/B
arms used a CPU-restored prefix and did not use a warm graph session. The combined result is not
an HTTP-serving qualification or proof that separate gains add. All variants remain default-off.

The additive operator experiment kept every other binding from the qualified original native
binary. Both unchanged and rebuilt OFF controls on the other host differed from the stored native
reference at token 78; those cross-host runs were rejected rather than counted as speed evidence.
The final model gates were run on the already reference-qualified host.

The final CUDA runtime regression reported **1828 passed, 34 skipped**; skips are not passes.
The full native token/rank/callback comparisons and reproduction commands were retained with these
measurements. These are `sm_89` results, not `sm_75` qualification. **100 tok/s is still unachieved.**


## Exact HC4 up-mix candidate (2026-11-19)

A new opt-in entry, `qwen4_exp_hc_up_mix_exact`, shares one activation load across the four
hyper-connection stream dot products instead of re-reading it per stream. It is bit-identical to the
qualified `qwen4_exp_hc_up_mix` on rows 1/4/512 and all four thread widths, survives graph
capture/replay, and is **-21.2% device time** at the decode shape (10.68 -> 8.41 us, graph-captured,
host dispatch excluded).

It is **off by default and not a throughput step**: 96 launches a round make the kernel
2.68% of a 38.30 ms round, so the ceiling is
+0.57% TPS. On the native 261632+512 request, three interleaved off/on
pairs gave -0.62% and +0.16% (mean
-0.23%) against an `off`-arm spread of
9.11%, i.e. no resolvable change. All six arms reproduced the frozen
original greedy ids with all eight ranks and callbacks equal, at 233 rounds and 5 tail steps.

The same arithmetic bounds the rest of the remaining single-kernel work: 100 tok/s needs
+75% from this baseline, which no remaining kernel carries. The
record is `tests/data/qwen4_exp/optimization-exact-hc-up-mix-resul<host-3090>61119.md`.
