# Evidence: rounds 71 to 75

This section exists mostly because **two of these rounds contained wrong claims**, and the record of how they
were caught is worth more than the claims were. The registered result did not change: **14.287598 tok/s /
69.9922 ms**, and nothing here was adopted.

Every file has its sha256 in `summary.json` under `manifest`.

## The graph-boundary question, answered

The objective asks for one item here: *inline originalA → nativeTP_execute → originalB as one layer NPUGraph,
with the same kernel/layout/provider contracts, compared against the layout control, and full eight-rank
complete 64-step FP32 bit-identical parity before any extra gain is accepted.*

**It is delivered in the shipping configuration**, not pending. The per-rank records of the registered run
(`pure-B1-fused-row-norm-candidate-profile-steps64-20261123T103426Z`) state, **identically on all eight ranks**:

| field | value |
| --- | --- |
| `inline_TP_layer_graphs.layers` | 40 |
| `inline_TP_layer_graphs.captured` | true |
| `inline_TP_layer_graphs.variants` | 43 |
| `inline_TP_layer_graphs.driver_counts.replays` | 6120 |
| `independent_TP_graphs_retired_after_explicit_fence` | 40 |
| `graph_candidate_eager_FP32_bit_differences` | **[0, 0, 0, 0, 0, 0, 0, 0]** |
| `private_graph_capture_seconds_excluded` | 4.141067163087428 |

All forty layers are captured, the independent two-recording variant was **retired** rather than left as an
unclear fallback, and the parity the objective demands is recorded as all zeros. Capture cost is excluded from
the measured step rather than folded into it.

Consequence: **no capture work remains.** The 8.7% of profiled span that is not device-busy is not eager launch
overhead, which agrees with the device-bound measurement (91.3% busy).

## The wrong claims, and how they were caught

**Round 60 — retracted earlier.** Claimed the output-head matmul was worth 1.27 ms not yet banked. The capture
the claim rested on predated the narrow head, so the saving was already in the lane. Caught by reading the run
record, which stated `narrow_output_head` installed and `bfloat16` while the capture predated it.

**Rounds 71–73 — retracted here.** Claimed the registered lane runs fully eager, that NPU graph capture is
implemented but used by nothing, that `NpuDecodeGraphs` is never instantiated, and that remaining device idle
was plausibly eager launch overhead worth attacking. All false; see the table above.

*Root cause:* a **case-sensitive grep**. I searched the launcher for `NPUGraph`, `npu.graph`, `graph_capture`,
`decode_graphs`, `make_graphed`. The flag is spelled **`RELIC_CUBE_NPUGRAPH`**, all caps, and it is set in the
*generated wrapper*, not in that script. The empty result was read as absence.

*Secondary error:* the round-74 retraction withdrew "fully eager" but then described the shipping behaviour as
two recordings around an eager middle — when the records show the inline variant is what runs, so that middle is
**inside** the recording.

*How it was caught:* by reading the per-rank run records, which state `graphs = True` and carry the inline
capture, variant count, retirement count and the per-step parity list directly. One field answered what three
rounds of source reading had got wrong.

## The rule this section argues for

Both wrong claims were caught the same way — **by going to a stored record instead of reasoning further**. When a
claim rests on a grep returning nothing, read the run-record field that states the answer before building
anything on it. That is cheaper than a round, and it was available both times.

## Other findings in this stretch, both closings

**Redundant FP8 activation quantisation (the expected 1.2%): does not exist.** `quantize_decode_fp8_fast` has
two call sites; the one that matters (`ascendc_tile_moe.py:152`) computes `decoded_x` **once** and reuses it for
`w1`, `w3` and `w2`. The duplicate call in `modules.py:203-204` is on the soft/reference path this lane does not
run.

**The o-projection transpose (1,171.6 µs) cannot be removed bit-exactly.** Pre-transposing the constant weight
looked free of consequence, since a transpose changes no value; measured, it was not bit-identical in 3 of 6 then
5 of 8 trials and was *slower* eagerly, and the transposed-view variant was 0 of 8. Differences are 1–2 of 2048
elements, max abs up to 0.5 on bf16. The reduction order on this build is not layout-invariant. Recorded as a
deviation class needing explicit approval; **not adopted.**