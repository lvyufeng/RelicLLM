# Evidence: rounds 76 to 78

This section closes the work out. It contains the independent stability repeat the objective asks for, and the
clause-by-clause completion report. **No optimisation was adopted in these rounds and the registered result did
not change: 14.287598 tok/s / 69.9922 ms per token.**

Every file has its sha256 in `summary.json` under `manifest`.

## The stability repeat

The objective requires an independent repeat to verify the accepted gain is stable. The registered lane was
rerun once, unchanged, in the shipping configuration
(`logs/pure-B1-fused-row-norm-candidate-steps64-20261123T112154Z`, returncode 0, 340.7 s, teardown reported zero
live owned processes).

| rank | sustained single-stream TPS |
| ---: | ---: |
| 0 | 14.269744111 |
| 1 | 14.269698004 |
| 2 | 14.269730267 |
| 3 | 14.269887508 |
| 4 | 14.269703033 |
| 5 | 14.269826899 |
| 6 | 14.269793264 |
| 7 | 14.269760082 |

* **mean across the eight ranks: 14.269767896 tok/s**
* **rank spread: 0.001328%** — the eight independent timings agree to two parts in 10⁵, which is itself evidence
  the measurement is clean
* deviation from the registered mean (14.287598): **−0.125%**
* inside the spread the three registration pairs showed (14.221390 – 14.376778): **yes**

So the registration is not a lucky pair. The same configuration reproduced, from a fresh process, within a
fraction of the noise the registration itself exhibited.

The run also re-observed the configuration identically on **every rank**: `graphs = True`,
`inline_TP_layer_graphs` captured with 40 layers, `independent_TP_graphs_retired_after_explicit_fence = 40`, and
`graph_candidate_eager_FP32_bit_differences` all zeros. That is an independent second confirmation of the
finding that the inline one-layer NPUGraph is the shipping configuration.

## The completion report, and the three errors it caught in its own draft

`objective-closeout.md` maps every clause of the objective to the evidence for its status. Writing it caught
three errors in the draft:

1. The progression table mixed `best_candidate_TPS` values for the early steps with a **mean** for the last one,
   and presented them in a single column as if comparable.
2. The starting point was given as 12.499052 — a single pair's value. The control mean of the first trial is
   **12.452279**.
3. The overall gain was stated as +14.3%, when the receipt-derived figure is **+14.74%**.

The table in the published report is now recomputed from the receipts on disk, every row a mean over its own
three pairs, and **all twelve cells were re-verified programmatically** against the receipt files. The six
trials' control means all land inside **12.449932 – 12.479610**, which is separate evidence that the control lane
is stable.

## What the report concludes

| objective clause | status |
| --- | --- |
| Optimise toward ≥ 50 tok/s | **not achieved** — registered 14.287598 |
| Pure per-token, batch-one, single-request | held |
| No MTP / speculative / request-batch / prefill substitute | held |
| Weights lossless; FP4/FP8 codes, scales, values unchanged | held, 90 checkpoint files byte-unchanged |
| Server-only testing | held |
| Contiguous-KV-layout deviation scoped, old rejection preserved | delivered |
| Independent repeat to verify stability | **delivered — this section** |
| Original 90 files / CANN / CI untouched | held |
| Inline one-layer NPUGraph + 8-rank FP32 bit-identical parity | **delivered, and already shipping** |
| Bounded jobs, ownership tokens, separated profiling | held |
| No master commits / PR merges | held — HEAD `532bb3a`, no commits |
| Publish sanitized evidence, clean up, release | delivered — 118 clean receipts |
| ≥ 50 tok/s measured pure batch-one | **not achieved** |

The reason the target is unreachable is not unexploited work. A tuned community stack on the *same hardware*
measures 27.13 ms a step at concurrency 1 with an acceptance length of 4.10; at an acceptance length of one that
is about **36.9 tok/s**. This lane is at 14.29, roughly 39% of that ceiling. Reaching 50 requires more than one
token per step, which the objective forbids.

The dominant single cause inside the constraint is **M=1**: the cube unit is fed one activation row, so most of
every loaded tile is padding. That is a property of single-request decode, not of any kernel implementation, and
it is recorded in the cost-model section of the main measurement page along with the evidence closing each other
avenue.