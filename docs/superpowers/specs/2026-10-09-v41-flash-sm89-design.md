# DeepSeek-V4.1-Flash on sm_89 (8xRTX 4090): bring-up and cross-arch parity

Status: design, awaiting approval.

## Why

The V4.1-Flash runtime (`relicllm/models/deepseek_v4_1/`) is validated only on
**4xRTX 2080 Ti (sm_75)**, where its reference arithmetic runs on the `torch` soft path because
Turing has no FP8/FP4 tensor core. The two 4090 boxes (ts-133, ts-134) are 8x24 GiB sm_89, with
FP8 tensor cores, a 72 MB L2 and twice the host RAM of the 2080 Ti box. Nothing in the V4.1 path
has ever been executed on sm_89.

The first question is not performance; it is **whether the runtime is even correct on a second
architecture**. On sm_89 `relic_core.kernels.ops._auto_impl` resolves `fp8` and `fp4` to **triton**
(verified on ts-134: `_resolve_impl("fp8","auto") == "triton"`, `_resolve_impl("fp4","auto") ==
"triton"`), where on sm_75 both resolve to **torch**. A triton/torch disagreement would be a silent
numeric divergence, not a crash. This spec is the bring-up-and-parity pass that answers that
question and makes the answer a reproducible check. Optimization on the 4090s is explicitly out of
scope.

## Goal and acceptance

Run V4.1-Flash on **ts-134 at TP4** — the same tensor-parallel width the sm_75 reference was taken
at, so that the architecture is the only variable — and accept it in two legs:

- **Bit-identical leg.** With `DEEPSEEK_FP8_IMPL=torch` and `DEEPSEEK_FP4_IMPL=torch` (forcing the
  soft path on sm_89), running the `v41` golden fixture's own `argv` must produce **exactly** the
  `expected.token_ids` recorded in `tests/fixtures/golden/v41.json` on sm_75
  (`[455, 22471, 45117, 16, 455, 22471, 45117, 16, 1162, 344, 260, 1950, 16, 1162, 344, 260]`).
  This proves the port is free of layout/eval-path errors independent of any kernel difference.
- **Tolerance leg.** With the implementation left at its native `auto` (triton on sm_89), the same
  request must (a) reproduce the same greedy tokens, and (b) agree with the sm_75 reference logits
  within a stated tolerance. Because greedy tokens alone are a coarse signal, the tolerance leg is
  measured against a **numeric product** (see "The tolerance leg"), not against tokens.

Both legs are driven by **test-injected env**, not by editing the fixture: the fixture's stored
`env` holds only `DEEPSEEK_V41_RESIDENT_EXPERTS`, and the check adds `DEEPSEEK_FP8_IMPL=torch
DEEPSEEK_FP4_IMPL=torch` for the bit-identical leg, omitting them for the tolerance leg. Same
fixture file, two runs.

A finding either way is a valid outcome: if triton and torch agree bit-for-bit, the tolerance leg
collapses into the bit-identical leg; if they do not, the size of the gap is the deliverable.

## Non-goals

- Any performance work, TP>4, expert residency on the card, EP, or batching. The 8-card advantage
  (192 GiB VRAM, FP8 tensor cores, 72 MB L2) is the subject of a **separate** spec.
- ts-133. It has no RelicLLM/relic-core checkout and no conda environment for this work; only
  ts-134 is prepared. ts-133 is left alone.
- Making V4.1 *fast* on sm_89, or changing any kernel.

## Preconditions

1. **Checkpoint.** 475 GiB, 48 shards, `deepseek-ai/DeepSeek-V4.1-Flash` on ModelScope. Not present
   on ts-134. This is the critical path and is started first, in the background. It must land at
   **the path the fixture's `argv` names** — `/mnt/data3/DeepSeek-V4.1-Flash`, which
   `tests/fixtures/golden/v41.json` hardcodes in both `checkpoint` and `argv`. If it lands anywhere
   else the fixture would have to be edited, which changes the thing being tested, so the recorder
   downloads to that path (symlink or download target) rather than editing the fixture. ts-134 has
   no `/mnt/data3` today; creating it, or pointing it at the download, is a precondition step.
2. **Sources.** ts-134's `/home/mseco/relic/RelicLLM` and `/home/mseco/relic/relic-core` are on
   non-master branches. Sync both to merged master (RelicLLM `169f94a`, relic-core `b84ff02`).
3. **Extension.** ts-134 already builds `cuda_kernel` for sm_89 (`(8,9)`, `relicllm` env, torch
   2.9.1+cu128) and `relicllm.models.deepseek_v4_1` imports. Confirm the build carries the fp8
   device-guard (`soft_fp8_blockfp8_*` checking `is_cuda`); rebuild if not. The arch list must be
   `7.5;8.9` (SASS, no PTX) per the project's dual-arch rule.
4. **Environment.** `/home/mseco/miniconda3/envs/relicllm`; `modelscope` installed there.
5. **sm_75 oracle access for the numeric record.** The `logits_check` values are recorded *once*
   on the sm_75 box (see "Producing the sm_75 numeric record"); that recording session needs the
   sm_75 box, its checkpoint (`/mnt/data3/DeepSeek-V4.1-Flash`, present) and a free card. It is a
   distinct step from the ts-134 run and is listed as such below.
6. **`/dev/shm` for the resident expert bank.** The `v41` fixture declares
   `requires.dev_shm_bytes = 504658657280` (~470 GiB). The golden runner treats a short `/dev/shm`
   as a **skip, not a failure**, so a short tmpfs would make the acceptance gate pass vacuously.
   Verified on ts-134 today: `/dev/shm` is a 1008 GiB tmpfs and empty. Re-confirm before the run,
   since another job sharing the box could consume it.

## Design

### Where the work lives

Almost none of this is new runtime code. The V4.1 runtime is architecture-agnostic PyTorch; the two
legs differ only in the env knobs `relic_core` already reads at import
(`DEEPSEEK_FP8_IMPL`, `DEEPSEEK_FP4_IMPL`). The new artifacts are **test** artifacts:

1. A **cross-architecture golden check** that runs the existing `v41` fixture on whatever card it
   is handed and asserts the recorded `token_ids` under the forced-torch leg.
2. A **numeric tolerance product** in the fixture, so the native-triton leg has something finer
   than tokens to compare.

### The bit-identical leg

Runs the `v41` fixture's `argv` with `DEEPSEEK_FP8_IMPL=torch DEEPSEEK_FP4_IMPL=torch` injected,
through the existing `serve` entry point, and asserts `token_ids == expected.token_ids`. The
existing golden runner (`tests/golden_fixtures.py`, `tests/test_served_path_golden.py`) already
launches a fixture's argv and compares; this leg is that machinery with the two env vars added. The
subsequent *tokens* on ts-134 are compared against the record already in `v41.json`; the token
oracle is not re-run. (The *numeric* record for the tolerance leg does need one fresh sm_75 pass —
that is a separate precondition and step, see below; this paragraph is about the token leg only.)

### The tolerance leg

`token_ids` is too coarse: a triton/torch divergence in the last few bits can leave a greedy token
unchanged, so "same tokens" does not measure the gap. The tolerance leg therefore needs a numeric
product recorded from sm_75. The minimal such product is a **fixed-input logits checksum at a
stated position**: a small, deterministic summary (e.g. the greedy top-k logits of the first
generated step for a pinned prompt) rather than a full logits dump, so the fixture stays small.

This extends the golden fixture schema with an optional numeric field. The schema change is
deliberately minimal and additive:

```json
"expected": {
  "prompt_tokens": 9,
  "token_ids": [ ... ],
  "text": "...",
  "logits_check": {
    "step": 0,
    "top_k": 8,
    "token_ids": [ ... ],
    "values": [ ... ],
    "atol": 0.0,
    "rtol": 1e-5
  }
}
```

`logits_check.token_ids` are the vocab ids of the `top_k` positions and are positionally paired
with `values` — both arrays are the same length and the same order. They are *not* the same as
`expected.token_ids` (the generated sequence): one is the argmax candidates at step 0, the other
the decoded output. The token ids are recorded so that a divergence can be attributed — same ids
with different logits is a numeric gap, different ids is a wrong argmax.

`logits_check` is optional — fixtures without it are unchanged. `atol`/`rtol` are recorded with
the values so the tolerance is part of the record and not a constant in the test. On api/cli
changes this follows the existing golden-fixture provenance rules (`card`, `commit`, `taken_at`).

**What `values` are, precisely** (a planner must not have to guess, since ordering is what makes
two architectures comparable): the **raw pre-softmax logits** of the greedy top-k positions of the
first generated step, **sorted descending by value**, with the **token ids** of those positions
recorded alongside them in the same order. The record names the step index and the prompt token
count so a mismatch in what was compared is visible. Not post-softmax, not the full vocab.

### Producing the sm_75 numeric record

The `logits_check` values are recorded **once on the sm_75 box** (the existing oracle) by the
recorder, then committed with `card: "cuda 7.5 (Turing / RTX 2080 Ti)"`. ts-134 then compares
against them. No second fixture, no second checkpoint path — consistent with the project's
one-fixture-per-entry-point convention. The sm_75 recording writes **placeholder** `atol`/`rtol`
(0.0); they are re-frozen into the fixture after the first ts-134 measurement, so nobody treats the
first-written tolerance as authoritative.

## Components

| Unit | Purpose | Depends on |
| --- | --- | --- |
| `v41` golden fixture (+ optional `logits_check`) | the oracle record | sm_75 box, checkpoint |
| **child result protocol extension** (`tests/golden_fixtures.py` `Outcome` + `to_payload`/`from_payload` + `CHILD_RESULT_ENV` payload) | carries `logits_check` out of the served child alongside `token_ids` | `generate.py` `on_token` (already yields the producing logits) |
| **logits recorder** in the child (`serve` side) | captures the first step's logits via `on_token` into the payload | the protocol extension |
| cross-arch golden check (in `tests/`) | runs fixture argv under injected forced-torch env on any card, asserts tokens (bit-identical leg); reruns under native `auto` and compares `logits_check` within tolerance (tolerance leg) | golden runner, the two units above, `relic_core` env knobs |
| schema test (`tests/test_golden_fixture_schema.py`) | validates `logits_check` shape when present | fixture schema |

Each is separately testable: the schema test needs no GPU; the child-protocol extension is unit
testable in-process; the cross-arch check needs a card and the fixture's checkpoint.

**Why the protocol extension is required:** `Outcome` today carries only
`token_ids`/`text`/`prompt_tokens`/`elapsed_seconds`, and `_run_python` never extracts logits. The
tolerance leg's acceptance criterion (2b) therefore has no owning unit without this extension —
it is the missing piece the tolerance leg hangs on.

## Testing / verification

1. Schema test passes for fixtures with and without `logits_check`.
2. On ts-134, the bit-identical leg reproduces the sm_75 `token_ids` exactly (the acceptance gate).
3. On ts-134, the tolerance leg reports the triton-vs-torch gap; if zero, it is recorded as zero.
4. The sm_75 box still passes its own full suite (no regression from the schema/env additions).

## Risks

- **475 GiB download** is the critical path; bandwidth-bound and outside our control. Started first.
- **ts-134 GPUs may be busy** with another user's job (has happened before). The run needs 4 free.
- **triton/torch may differ enough to change greedy tokens**, in which case the bit-identical leg
  still isolates the port and the tolerance leg carries the quantitative result.
- **sm_89 extension arch list** must be `7.5;8.9`, no PTX-only; otherwise a rebuild is required.

## Open questions (resolved during implementation)

- Exact `atol`/`rtol` — set from the first measured gap, then frozen into the fixture.
- Whether the logits record is per-step or first-step-only — start first-step-only (smallest signal
  that is still numeric).