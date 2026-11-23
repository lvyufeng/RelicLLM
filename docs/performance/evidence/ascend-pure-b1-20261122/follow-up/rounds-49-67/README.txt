# Evidence: rounds 49 to 67

This section covers the later rounds of the pure batch-one decode work on one 8 × Ascend 910B3 server
(64 GiB a card). The parent `follow-up/` directory records an earlier state and is **not** rewritten;
anything already stated there is not restated here.

Every file in this directory has its sha256 in `summary.json` under `manifest`, so a reviewer can check
the copies against the receipts in the working tree.

## The registered result

**14.287598 tok/s, 69.9922 ms a token**, registered on three pairs, against the previous 14.064584 /
71.1006 — **+1.586%**.

| pair | control | candidate | vs previous best |
| --- | --- | --- | --- |
| 20 | 12.4480 tok/s, 80.334 ms | 14.3768 tok/s, 69.557 ms | +2.220% |
| 21 | 12.4729 tok/s, 80.174 ms | 14.2646 tok/s, 70.103 ms | +1.422% |
| 22 | 12.4597 tok/s, 80.259 ms | 14.2214 tok/s, 70.317 ms | +1.115% |

All three pairs are positive, and the **worst** pair is still above the previous best — which is what
makes it a result rather than a lucky run. `token_ids_equal_pair` and `token_ids_equal_canonical` are
true on all eight ranks in every pair.

The change is a **bit-exact device relocation**: the sparse indexer's mask-then-sort selection at
`attention.py:1173` ran as a Greater, a Sort and a Cast on the AI CPU for eight index layers a forward;
one kernel now computes the same selection as a rank and the reader scatters instead of sorting. It
registers as bit-exact and needs **no new approval**: the reference sorts the ids, the ids are unique,
and every tie is between sentinels that all collapse to the same padding value, so the output is
uniquely determined.

`attention.py` was never edited — the fusion is a runtime rebind of `Indexer.forward`, and its md5 is
`87727e6cf0e32010d90ba5cc0ae134ac` before and after.

Prior approvals remain in force and are unchanged: the row-normalisation class and the narrow bf16
output head, both listed in `summary.json` with their receipts and error bounds.

## Measurement that decided where the work went

**The straggler wait is MoE routing imbalance, measured — not inferred.** The lane runs expert
parallelism four ways inside two-way tensor parallelism, a token selects six experts, and the ownership
predicate is stated literally in `route_owned_kernel` as `owned = (ids % WORLD) == RANK`. Sampling 206
dispatches on all eight ranks:

* **206 of 206** dispatches have a spread of at least one owner; **68.4%** at least two; max 5
* mean busiest owner **2.772** experts, mean least busy **0.388** — about a sevenfold ratio
* **60.7%** of dispatches have one owner holding three or more

That is the 19.3% of the step spent waiting at collective barriers with every core idle.

**The device does not overlap streams, so the wait cannot be hidden.** Two independent cases:
compute against compute (1,078.4 µs on two streams against 1,043.3 µs sequential) and compute against
collective — the one that mattered, since a collective runs on the HCCL-SDMA engine rather than the cube
units — which came out at **481.5 µs against 380.3 µs sequential**, i.e. the no-overlap sum (479.5), not
the overlapped maximum (301.7). Two streams were 26.6% *slower* than one.

**The lane is device-bound and the device is wall-to-wall busy.** The union of device task intervals is
**287,045 µs of a 314,467 µs span (91.3%)**. So `aten::copy_` showing 95 calls and 4,277.7 µs of *host*
self time a forward against 63.1 µs of device time is overlapped and not recoverable by host
restructuring. Every remaining gain must be less or faster **device** work.

## Closed as negative, with the reason

| item | why it is closed |
| --- | --- |
| `HCCL_OP_EXPANSION_MODE=AIV` | improves a small collective's own latency; the wait is for compute that has not arrived. Attempted twice, never successfully measured — recorded rather than hidden |
| rebalancing MoE routing | which owner holds which expert is the model's own routing, part of the protected contract |
| stream-level overlap | measured unavailable on this device, for compute and for collectives alike |
| `pos.host` device-to-host sync | `decode_pos.py:194` shows it is a plain Python int property, no synchronisation |
| host-side `aten::copy_` overhead | 4,277.7 µs of host time a forward, but the device is 91.3% busy, so it is overlapped |
| rank-based sort for the 2048-wide block selection | measured **0.66×** — slower than the four operators it would remove, so it was left on torch |

## Where this leaves the target

The objective asks for ≥ 50 tok/s pure per-token batch-one. Measured against community deliveries on the
same hardware: `CJYLZS/vllm-ascend-deepseek-v41-a2` runs 8 × 910B3 at 65536 MB a card and measures
**27.13 ms a step at concurrency 1** with an acceptance length of 4.10, i.e. 136.9 tok/s per stream.

At an acceptance length of one that same tuned stack is about **36.9 tok/s**, so 50 tok/s pure batch-one
is outside what this hardware gives even with a reference-quality stack. Reaching 50+ requires the model
to emit more than one token a step, which the objective excludes. The section therefore records the
ceiling honestly rather than implying the gap is entirely unexploited work.

Within the constraint, what remains open is device work only: the scale-LUT cube kernel at 15.4% of the
step (already bounded at under 2.4 ms of upside, with `block_dim` at its 24 maximum and the gather about
20% of the kernel), the plumbing copies at roughly 4%, and communication at 26.2% which is closed on both
routes.