# Pure-B1 accepted S1 + provisional S2b follow-up evidence

Sanitized metadata-only follow-up to the immutable S1 bundle in the parent directory.
No checkpoint, logits or input-tensor payloads are copied, and no project code was
imported, built or executed to assemble it: stdlib bytes/JSON and hashes only.

## What is accepted

The accepted S1 kernel (`ascend_sparse_attention_layout.py`, immutable hash recorded in
`core-source-manifest.json`) plus the private inline TP layer-graph fuse
(`<coord>/ascend_inline_tp_layer_graphs.py`, sha256 `9ddd1f4c...b641`, not copied into
this bundle; measured receipts in `accepted-s1-actual-selection.json`). S1's user
acceptance was measured at context 36 only; it is not general numerical headroom.

## What is provisional (NOT promoted)

The S2b state path matches its matched control bit-for-bit on all eight ranks and is
faster in the pair measured (+3.383% first, +3.079% repeat), but the cross-canonical
audits show that both the new control and the candidate reproduce only the original
pre-S1 identity 4824df at context 36, not the immutable accepted S1 canonical identity
852d95. `follow-up-decision.json` therefore records:

* `numerical_acceptance_global: false`
* `canonical_acceptance: false`
* current decision: **not accepted / not promoted**
* pending: new canonical controls restoring the old excluded-oracle allocation pair.

## Gate graph (metadata)

    accepted & immutable:  S1 layout kernel + inline TP (ctx36 measured acceptance)
                                |
    provisional pair only:  S2b state + matched control, bits 0/8 ranks
                                |  +3.383% / +3.079%
                                v
    canonical audit:        ctx36 -> 4824df (control AND candidate)
                            != accepted canonical 852d95
                                |
                                v
    promotion:              blocked -> numerical_acceptance_global false
                            (waiting for new canonical control/candidate run)

## Limitations

* B1 only: one complete forward per next token, 64 timed steps, context 24..88,
  completion + host argmax, 4 warmups excluded, no MTP/speculation/batch.
* S2b NPU domain is BF16 q/kv only; FP32 and mixed-dtype pairs are refused by name.
* The old S2 FP32 synthetic rejection and the old real-BF16 rejection are preserved
  separately; neither becomes general error headroom.
* Environment-change labels are sanitized; measured sha256 values are retained verbatim.
* Inline private provenance: source sha256 `9ddd1f4c...b641`, tests sha256 `950ee0e4...3bb5`;
  CPU fixture counters account for the capture-pass replay (metadata only).
* No tests, native builds or benchmark runs were performed on this Mac host.

## Files

`manifest.json` (raw source identities + derived hashes), `follow-up-decision.json`,
`provisional-s2b-comparison-1.json`, `provisional-s2b-comparison-repeat.json`,
`canonical-drift-audit.json`, `s2-rejected-gates.json`, `s2b-gate-receipts.json`,
`inline-tp-comparison-pair1.json`, `inline-tp-comparison-repeat.json`,
`accepted-s1-actual-selection.json`, `core-source-manifest.json`.

## Canonical protocol extension (provisional, pending repeat)

A new canonical-protocol run separates the control from the candidate:

* **Control 11.1284 TPS**: still the original pre-S1 identity `4824df` at context 36,
  not the immutable accepted canonical `852d95`; a drifting control cannot authorize
  a matched-pair gain.
* **Candidate 11.5614 TPS**: reproduces canonical `852d95` on all eight ranks —
  reference-8, teacher-forced-8, final and full 64-step tensors bit-identical, token
  ids exact (0 differences), with the actual fused state kernels selected (320 BF16
  attention calls per rank checked).
* The matched pair is therefore rejected (`all_eight_ranks_bit_identical: false`,
  `numerical_acceptance: false`); the +3.8906% is **not** a strict-output-pair
  approved gain.

`canonical-protocol-record.json` carries the derived sanitized evidence and
`follow-up-decision-v2.json` records the separated decision: candidate-only
canonical admission is **conditional** on the independent canonical-candidate repeat
(root run 71cffe) also measuring 0 differences (review-109 condition). No new error
waiver, no global numerical acceptance, and the earlier provisional/global-false
records are preserved unmodified. S1 acceptance at context 36 remains limited to its
measured scope.

## Final independent canonical-candidate repeat — candidate-only qualified

Review-109's final conditional criteria are all satisfied after the independent
canonical-candidate repeat on the same frozen sources:

* First canonical candidate: **11.561395608 TPS**; independent repeat:
  **11.548656515 TPS** (86.590 ms/token). The measured candidate range is
  **11.548657–11.561396 TPS**; absolute values are measurements, not a promise.
* Repeat: all eight ranks reproduce canonical `852d95` for reference-8,
  teacher-forced-8, final and full 64-step tensors with bit difference 0; tokens,
  layout/inline/state selections all true and no changed contexts. Only the output
  environment label changed between the two runs.
* Actual kernels: 552 prepare + 552 post-PV launches across eight ranks, 43 graphs,
  320 BF16 attention calls per rank; summary reports `owned_clean: 0`. Frozen
  sources: helper `bfc547…`, kernel `55f57a…`, tests `b3f15…`; native `.so` and CANN
  labels unchanged.
* The new control still reproduces the original pre-S1 `4824df` at context 36, so the
  aggregate canonical audit is deliberately `false` and the strict-output paired
  gains (+3.38%/+3.89%) remain **provisional** — they are not promoted as an
  approved strict pair.

`follow-up-decision-v3.json` therefore records: candidate-only numerical
**qualified**, explicitly BF16 q/kv-only B1 prototype, context 24..88; numerical
global and general-quality headroom **false**; no new error waiver, no serving
default/PR merge change, no 50 TPS claim, no additional NPU trials. The earlier v1/v2
records, old test-109 receipt and S2 rejections are preserved unmodified.

## Provenance correction (v4) — verified repeat 085516

The earlier v3 repeat record's local selector and summary were sourced from the
**first duplicate-basename copy**: the root wild rsync flattened both candidate runs
(`084544` and `085516`) into one `.coord/S2b-canonical-candidate-repeat` folder, and
duplicate basenames let the first win. The v3 dep trace is therefore mislabeled; its
files are preserved unmodified as history and superseded by v4.

The verified repeat lives in `.coord/S2b-canonical-repeat-verified-085516`
(output run labeled `085516`, started 08:55:16Z, returncode 0, cleanup complete with
0 live owned processes), while the root-level repeat comparison and direct canonical
audit are the correct 085516 receipts produced with explicit server arguments. The
numerical proof is unaffected: first canonical candidate 11.561395608 TPS vs verified
repeat 11.548656515 TPS (86.590 ms/token, -0.1102% vs first), all eight ranks
bit-identical, repeat audit candidate rows 8/8 canonical `852d95` with no changed
contexts; the paired audit aggregate remains false only because it still carries the
non-canonical control rows (pre-S1 `4824df`). Actual kernels in the verified output:
552 prepare + 552 post-PV calls per rank on all eight ranks, 43 graphs, 320 BF16
attention calls per rank.

**Review status correction:** review-109 has **not** issued a final admission. The
candidate-only numerical result stays **conditional** until that final review lands
after this provenance fix. Paired gains remain provisional (control still drifts),
and there is no new error waiver, default/merge change or 50 TPS claim.
`canonical-repeat-verified-085516-record.json` and `follow-up-decision-v4.json` hold
the corrected provenance; the wrong old folder is kept unchanged as history.

### Verified-folder provenance and release safety (final v4 correction)

The v4 selector/summary provenance now comes **only** from
`.coord/S2b-canonical-repeat-verified-085516`: its rank-0 raw kernel CSV is
`027141340dd18cf3c8268c743a0f600b79be46c777a93d992c04a679762f8b34` (the first-copy
CSV `346dfe85…` is superseded), state-kernel durations are 1584.5 us prepare /
1986.0 us post-PV, and the preconditions carry the `085516` output marker with
`S2_softmax_state_variant: candidate` and accepted S1 layout+inline fixed. The
earlier v3 public record is preserved as mislabeled history and annotated as
superseded on these exact fields by v4.

A small sanitized `final-release-safety-summary.json` records the release proof as
counts and hashes only: 90 checkpoint files with stable metadata (no mtime/ctime
inside the task phase; payload hashes explicitly not recomputed), 40 owned tasks all
cleanup-complete with 0 remaining owned processes, unchanged CANN setup
(`bbbda1c0…`) and native library (`7dce3e3d…`), and no device reset/shared-CI
changes/deletions. Device-table proof remains a separate live fleet probe.

Review-109 has **not** issued its final admission: exact proofreading is in progress,
the candidate-only criteria are all met, and the numerical admission stays
conditional until that final message. No new error waiver, no default/merge change
and no 50 TPS claim.

## History index honesty (v7)

Version history is indexed truthfully in `manifest.json`: original raw source receipts
(measured hashes/bytes) are unchanged, while README/manifest are updated by design at
each version and two derived correction records (`canonical-repeat-verified-085516-record.json`,
`follow-up-decision-v4.json`) were expanded by design in v5 — no previous data was
retconned and nothing was deleted. The manifest's `canonical_protocol_extension` block
is a **historical v2** status (pending repeat) and is superseded by the final admission.

## FINAL ADMISSION (v6) — opt-in experimental retention

Review-109 has now issued its **final** admission (task109 rev3): no code/lifetime
P0/P1, approve to **RETAIN** the explicit BF16 q/kv S2b path for B1 context 24..88 as
**opt-in experimental — not a default production path**. The exact `085516`
selector and summary were verified, and the canonical first/repeat runs are
bit-identical across all eight ranks' full four tensor families with finite outputs
and 0 token-id differences. Absolute measured candidate TPS are reportable
(11.5614 first / 11.5487 repeat).

Not approved, unchanged: strict-output paired causal performance (+3.38%/+3.89%)
because the controls drift; FP32 NPU path; FMA dependence; general quality headroom;
serving default/PR merge; 50 TPS; any new error waiver.

`final-admission-v6.json` is the newest record; the v4/v5 conditional records remain
historical; the two derived v4 correction records were expanded by design in v5 with no previous data retconned and no deletions. Release safety is recorded in
`final-release-safety-summary.json`: arbiter lease released, live probe free on all
eight ranks (0% util, 0 procs), 90 checkpoint files metadata-clean, 40 owned tasks
clean, toolchain hashes unchanged.

## Token-label clarification (v7.1)

The historical `final-admission-v6.json` stores
`verified_proof.token_ids_equal = 0`, which is a **token-id difference count**, not a
boolean. The canonical audit's eight candidate rows all report
`tokens_equal_canonical: true`, so the correct reading is
**`token_id_differences: 0`, `token_ids_equal: true`**. See
`token-label-clarification-v7.json`; the v6 file and every measured hash are
preserved unchanged, and numeric admission/review scope are unaffected.

## v9 — accepted preparation reuse, two-endpoint criterion, and a negative result

**Accepted exact attention preparation reuse.** Pair 1 measured control
11.547188662864219 → candidate 11.606829728700983 TPS (+0.5164986%), acceptance_layer
`strict`; the independent repeat measured 11.560274208616125 → 11.59814733280801
(+0.327614%), acceptance_layer `approved_endpoint`. Both pairs are canonical across
all eight ranks (reference-eight, teacher-eight, final, finiteness, token IDs), and
the actual original FP8 preparation kernel dropped 1480 → 1288 calls per four
excluded traces = 48 launches removed per forward on every rank, while the native
expert projections (480) and both S2b state-kernel counts (552) stayed unchanged. The
mechanism is a scope, not a cache: it returns the original quantizer's own
codes/scales objects including the fused-decode annotation, clears on every exit, and
CPU/prefill keep the original provider.

**Direct-user two-endpoint criterion.** At context 36 / timed step 12 of the excluded
copy pass exactly two states are accepted (`852d957a…` / `4824df27…`), differing in
129251 elements with max-abs 0.003835439682006836; the other 63 steps, both
eight-step families, the final state, finiteness and token IDs must match canonical on
all eight ranks, and the named step must equal exactly one of the two hashes with
exactly that count and max-abs. It is a closed two-value set, **not** a tolerance: a
third value, an extra changed step or another magnitude is rejected, the old strict
rejection is retained, and none of the lossy/FP16/grid/Sinkhorn/MTP/default/merge/50
permissions were granted. Approval receipt sha256 `60ec1eca…f946c37214`.

**Shared-expert preparation scope — negative result.** The mechanism was verified
(1288 → 1128 = 40 fewer launches per forward on all eight ranks), but the measured
control 11.652519901152331 → candidate 11.631003375325955 TPS is −0.1847%, inside the
roughly half-percent same-configuration run-to-run variation. It is **not adopted**
and carries no performance claim; the plausible cause is that per-call site guards
(two device-runtime queries on each of forty forwards) consumed the saved launches.
Guard relaxation would need an explicit fail-closed session and was not done. Lesson:
preparation-kernel-count reduction is not by itself a valid lever.

**Decision logic and closed negative findings.** The comparison source
(`compare_pure_b1_reuse.py`) and its ten server-side tests reject columns agreeing on
a third value, an extra changed step, a shared other-tensor error, a wrong magnitude
at the approved step and a missing approval. Separately, no provably redundant
per-layer copy exists in this stack: every executed copy/cat site is a semantic
lifetime, layout or dtype requirement, and the aggregate Memcpy category includes
non-copy rows and is a budget rather than a critical path. No copy-removal candidate
is proposed.

`v9-accepted-preparation-reuse.json` carries the derived sanitized record; all earlier
records remain unchanged.

## v10 — accepted expert-projection kernel variant (vector-broadcast scale offsets)

**Mechanism.** The expert projection kernel expands each per-pair scale offset with a
fixed pattern — each of the N group offsets repeats eight times, and the resulting
N*8 words fill the group twice — but it used a general gather whose per-element
offsets were executed on the AIV scalar unit. It now emits the same offsets with the
vector broadcast instruction. The prepared table ABI, bank format, activation grid,
Cube MMA and FP32 accumulation order are unchanged.

**Component measurement** (same geometry, six synthetic byte-exact N64 projections at
n=1152/K=5120): one expert 258.3 → 242.9 us (−5.96%), six experts 670.8 → 563.1 us
(−16.06%), with 0 of 1152 and 0 of 6912 changed elements and max-abs 0.0 — bit
identical. Variant library sha256 `46aca3d9…f16c9117` (399,592 B), variant source
`c45bd134…64b6e1764e96` (24,927 B), accepted source `3ca65ab8…61dd188` (24,114 B),
27-line unified diff archived; accepted library `7dce3e3d…f9072c`.

**Build reproducibility.** Rebuilding the accepted source with
`-DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu -DSOC_VERSION=Ascend910B3
-DRELIC_LUT_OUTPUT_ROWS=64` and `cmake --build … --parallel 2` reproduced the
accepted library byte for byte. Without `-DCMAKE_BUILD_TYPE=Release` the CANN
preprocess/merge step fails with a bare `unknown file type` — the concrete blocker
that was resolved. CANN `set_env.sh` sha256 `bbbda1c0…3c3b12`.

**End-to-end matched pairs (three, all STRICT).** 11.605510 → 12.499052 tok/s
(+7.6993%, 6.160 ms saved); 11.632330 → 12.447992 (+7.0120%, 5.633 ms); 11.623546 →
12.451103 (+7.1197%, 5.718 ms); mean +7.2770%. Every pair is strict: reference-eight,
teacher-eight, final and complete 64-step FP32 tensors plus token IDs identical to the
immutable canonical result on all eight ranks; no endpoint allowance was consumed. The
protocol is 64 timed steps, 4 excluded warmups, context 24–88, batch 1, one full
forward per token, no draft/speculation/batching, and the complete copy pass excluded
from timing. Best accepted window 12.499052 tok/s at 80.0061 ms/token; 50 tok/s still
needs 20 ms, so the target is **not** achieved.

**Control integrity.** Each side's preconditions recorded the exact
`RELIC_TILE_CONSUMER_LIBRARY` / `RELIC_TILE_CONSUMER_SHA256` it ran (control
`7dce3e3d…`, candidate `46aca3d9…`), the comparator verified those recorded hashes
against the actual files, the environment delta is exactly the two library variables
plus the output directory, and both sides ran bank format
`relic.fp4.cube-n64-k128.v1` with attention preparation reuse enabled.

**Honest scope.** The variant library is task-private (built at a private path); it is
**not** a serving default, **not** merged, and landing it is a relic-core change that
requires its own review and a fresh user request before any pull request. No broader
numerical tolerance was granted — the change is bit identical, so no acceptance-layer
allowance was consumed.

**Future work.** The second term in the same kernel, the per-pair code-table gather,
remains: the projection still runs at about 39.8 GB/s against 428.9 GB/s for an
equal-volume copy, and its counters remain scalar-heavy. No claim is made that this
remaining term can be removed.

`v10-accepted-lut-brcb-variant.json` carries the derived sanitized record; all earlier
records remain unchanged.

## v11 — provenance correction for the accepted LUT-BRCB kernel variant

`v10-accepted-lut-brcb-variant.json` recorded the variant source as `c45bd134…`.
That source was correct for N64 but derived the broadcast destination repeat stride
from the tile count, which is only accidentally right at N64 and packs the N32
geometry wrongly; review caught it. A v11 record supersedes v10 **for provenance
only** — v10 is preserved byte-identically and every measurement in it remains
valid.

**Final provenance.** Variant source `caca541d…791b52` (25,488 B), 33-line diff
`bcfe4369…7131a451`, accepted source `3ca65ab8…61dd188` (24,114 B). The corrected
form is `BrcbRepeatParams broadcast(1, 8)`: one broadcast repeat always writes eight
32-byte destination blocks (64 32-bit words) whatever the tile width, so the
destination repeat stride is the constant eight blocks; `N/8` is only accidentally
right at N64 and wrong at N32. No source stride is configured because `BrcbImpl`
calls `vbrcb(dst, src0, dstBlkStride, dstRepStride, repeatTime)` — the source
advance is implicitly consecutive eight words per repeat — and an earlier
`src0RepStride = 1` assignment was a no-op that was removed. Captured API excerpts
and the index receipt are recorded.

**No-op at the measured width.** The final source rebuilt for N64 produces
`46aca3d9…f16c9117`, byte-identical to the library all three measured pairs used;
rebuilt for N32 it produces `cf9f5c6c…f5cc224f80`.

**N32 native parity gate now passes.** The new N32 binary compared against the
accepted N32 library `3b330d3f…` shows 0 of 1152 and 0 of 6912 changed elements,
max-abs 0.0; medians 277.0 → 268.8 us (one expert) and 834.1 → 772.3 us (six
experts). These are component numbers only — no N32 model rate is claimed.

**Build provenance correction.** The v2 build-provenance receipt (`71e48b4c…`)
states plainly that the first build receipt was assembled by copying the baseline
receipt and swapping only library path and sha, so its source hashes, source
snapshot and configure/build commands described the baseline build; the three
measured pairs are unaffected because the launcher consumed only that receipt's
library path and sha and verifies the pair against the file. Per-site hashes are
provided for source, binary, cmake.log and build.log.

**Unchanged.** The three end-to-end pairs remain +7.6993%, +7.0120% and +7.1197%
(mean +7.2770%), all strict; best 12.499052 tok/s at 80.0061 ms/token; 50 tok/s is
not achieved. The library is task-private, not merged, not a serving default, and
landing needs its own relic-core review and a fresh user request. Noncanonical
custom prepared tables are not claimed equivalent. There is no independent trace of
what the NPU executed beyond post-run file hashing plus the binding's hash-then-load
check. Preconditions for all six measured runs are archived so a reviewer can read
the selected library identity without host access.

`v11-brcb-provenance-correction.json` carries the corrected record; all earlier
records remain unchanged.
