# The test suite

`tests/` holds two different things and pytest collects only one of them.

- **`test_*.py`** — the suite. Run it from the repository root, because there is no `conftest.py`
  and no pytest configuration: modules import from the root and nothing puts it on `sys.path`.
- **`bench_*.py`, `probe_*.py`, `profile_*.py`, `debug_*.py`, `summarize_*.py`** — measurement
  scripts. pytest does not collect them, and each one prints the command that produced its numbers
  in its own docstring. `tests/fixtures/` and `tests/data/` are their inputs.

  The vLLM-convention serving benchmark is no longer one of these: its implementation moved into the
  package (`relicllm/bench/`) and is driven by **`relicllm bench serve`**, which launches a server,
  measures it and writes a metadata envelope — see [serving latency
  metrics](../docs/guides/latency_metrics.md#invocation). `tests/bench_serving.py` remains as a shim
  that re-exports the client and forwards `main()`, so `python tests/bench_serving.py …` still runs;
  new work should call `relicllm bench`.

```bash
python -m pytest tests/ -q          # the whole suite
python -m pytest tests/test_x.py -q # one module
```

There is no `conftest.py` by design rather than by accident: a root `conftest.py` would change how
every module is imported, and the suite's modules are run as scripts often enough (see the skip
policy below) that the two entry points have to agree. Shared fixtures are defined in the module
that uses them.

## The baseline, and why it is a set

**The suite is green on `master` today, and the file is what keeps that a claim rather than a
memory.** The failures that are known are recorded, one per line by node id, in
[`baseline_failures.txt`](baseline_failures.txt), and
[`scripts/check_test_baseline.py`](../scripts/check_test_baseline.py) is what diffs a run against it:

```bash
python scripts/check_test_baseline.py                 # run the suite, diff, exit 1 on anything new
python scripts/check_test_baseline.py --observed FILE # diff a run recorded earlier
python scripts/check_test_baseline.py --update        # rewrite the baseline from a fresh run
python scripts/check_test_baseline.py --fail-on-fixed # also fail when an entry stops failing
```

It is a **set of node ids and not a count**, and that distinction is the whole point. Three tests
fixed and one broken is a *net improvement* in a count and a regression in the tree, and reporting
that as "-2" hides the one that matters. It is also symmetric in the other direction: a test that
used to fail and now passes silently is reported on its own line, because a fixed test and a test
that has started skipping itself look identical from one run.

`--fail-on-fixed` is off by default because those two really are indistinguishable from here: this
suite skips itself for want of a card, a checkpoint or a built extension, and a failure recorded on
this host may legitimately become a skip on another. Deciding otherwise is a decision about the
host, so it is a flag rather than the default.

The recorder itself, [`scripts/_baseline_recorder.py`](../scripts/_baseline_recorder.py), is loaded
with `-p` by that script and never by the suite, so a plain `pytest` run is unaffected by any of
this.

## Skips are not passes

A module that needs a GPU, a real checkpoint or a built extension skips itself — `pytest.skip`,
`pytest.importorskip` or a fixture that skips — and **a skip is not a pass**. It says the test could
not run here, which is a different claim from the test having run and succeeded.

That is why skips are absent from the baseline. An entry in `baseline_failures.txt` is a claim that
the test *runs on this host and fails*; a test that cannot run here belongs nowhere in that file, and
a skip that was reported as a new failure would make the check useless on any machine but the one it
was recorded on.

## What is not in this suite

**The kernels' own tests are not here.** They belong with the kernels, which live in
[relic-core](https://github.com/lvyufeng/relic-core) — its `tests/` carries the kernel-vs-reference
checks and its own baseline. The two suites are disjoint by construction: a test lives here if it
imports anything from `relicllm` beyond `relic_core.kernels.*`, and there if that is all it
imports. A kernel test moved here would drag a model, a loader or a checkpoint into a repository
that has none.

**The native-build tests are not here either.** `test_native_build_preflight.py` and
`test_sdist_native_sources.py` covered the C++ engine's build surface — `cpp_engine/CMakeLists.txt`,
`MANIFEST.in`, the prerequisites `setup.py` refused to build without. That engine was retired with
the split: RelicLLM's `setup.py` is pure Python and has no native build to preflight, and the
kernels' own build (now `relic-core/setup.py`) is checked by relic-core. The two modules were
removed rather than left failing against a tree that no longer exists.

The skipped set is a coverage claim, so it is worth reading before trusting a green run:

```bash
python -m pytest tests/ -q -rs        # show why each test was skipped
```

## The no-checkpoint tests and the served-path tests

Most of the suite is hermetic: it tests a kernel against a reference, a parser against a fixture, or
an invariant against the source tree, and it runs in seconds. Those are the tests to keep fast.

A smaller set needs a **real checkpoint** and is the only place an end-to-end claim is checked — a
prompt in, token ids out, through a real entry point. Those tests find their checkpoint through the
environment and skip when it is absent, so the suite is runnable on a machine with no weights at
all. Each golden fixture records the checkpoint, the flags and the prompt it was taken under, in
[`fixtures/golden/`](fixtures/golden/), together with the token ids that came out; see
[the golden fixtures section](#golden-fixtures) below.

## What CI runs

**Nothing here.** `.github/workflows/publish-pypi.yml` builds and uploads a release, and
`.github/workflows/pages.yml` builds the documentation site with `mkdocs build --strict`. No
workflow runs pytest, so `python scripts/check_test_baseline.py` is a manual step and the baseline
is a record rather than a gate. Anything that has to be enforced therefore has to be enforced by a
check that runs somewhere — which, in this repository, means the `mkdocs build --strict` link check
for documentation and the `check_layering` CMake target for the C++ layering.

## Golden fixtures

A golden fixture is one real request through one real entry point, recorded with everything needed
to reproduce it:

| Field | What it is |
| --- | --- |
| `entry` | the entry point: `v41`, `mimo`, `xing4` or `torch` |
| `checkpoint` | the checkpoint directory or GGUF, as an absolute path |
| `env` | the environment variables the run needs (`POCKETLLM_CPP_BATCHED_DECODE`, `CUDA_VISIBLE_DEVICES`, …) |
| `argv` | the command line, **as an operator types it**, minus `--prompt` |
| `prompt` | the prompt **as text** |
| `path` | which route `prompt` is fed through: `raw` as a completion prefix, `chat` through the checkpoint's chat template. Absent means `raw`, which is what every fixture recorded before the field drove |
| `sampling` | greedy is `temperature: 0.0`; a fixture that samples is a fixture that fails one run in ten |
| `requires` | a resource the run needs from this host, currently `dev_shm_bytes` |
| `expected.prompt_tokens` | how many tokens the prompt rendered to for this checkpoint's tokenizer |
| `expected.token_ids` | what came out (empty for `torch`, whose runtime does not report ids) |
| `expected.text` | what came out |
| `expected.logits_check` | optional first-decoded-step top-k logits, as paired `token_ids`/`values` with the recorded `atol`/`rtol` the values are compared within; only `v41` records one, and only when the run that produced it had `POCKETLLM_V41_LOGITS_CHECK` set |
| `commit`, `taken_at` | the revision and the day it was recorded |
| `card` | the silicon it was recorded on, as free text (`cuda 8.9 (Ada / RTX 4090)`), or `""` when nothing was read |

Four things about that list are load-bearing, and the third is about how a fixture *runs* rather
than what it records.

**`argv` is a command line and the Python entries are driven through the CLI parser**, so
`build_parser()` → `_args()` → `EngineArgs` is exercised on every run of the fixture. A flag that
stops being read, or a default that moves, changes the tokens the fixture produces. Driving
`EngineArgs` directly would leave untested the one layer a serving refactor is about.

**`expected.prompt_tokens` is compared before the answer**, because the tokenizer and the chat
template are half of what a fixture pins: the same text rendering to a different number of tokens
means the model was asked a different question, and comparing only the answer would blame the model
for it.

**Each fixture runs in its own process.** The test spawns `tests/golden_fixtures.py --entry <name>`
and compares what the child hands back, rather than running five engines inside pytest's interpreter.
That is not tidiness — five engines in one process is a configuration nothing else in this repository
uses, and it fails in ways that have nothing to do with the fixtures:

| Order | What happens | Alone |
| --- | --- | --- |
| `mimo` then `xing4` | `CUDA error: resource already mapped`, inside the load | passes |
| `mimo` then `torch` | `WORLD_SIZE` read as 1 against a four-way sharded model → `q8_0 block shape mismatch` | passes |

The rule is one process per rank per GPU, and the operator's own entry point is `relicllm serve`,
started once per configuration. So the child is the unit of execution and the comparison stays in
the parent, where a failure can name the fixture.

**`requires` is how a fixture states what it needs from the host**, and it is why a missing resource
is a skip and not a failure. The two bank-pinning entries need a lot of `/dev/shm` — 458 GiB for
`v41` and 150 for `mimo` — and **they do not fit together in the 504 GiB tmpfs this box has**:

```bash
rm -rf /dev/shm/pocketllm_*_experts*     # drop both banks, then run whichever you need
```

Without the field the second one would die partway through a fill with an error that reads like a
model bug. With it, the fixture that cannot fit skips and names the size it wanted. The banks are
worth knowing about for a second reason: they persist, so a *warm* `v41` run is 3 minutes against 41
cold, and `mimo` is 75 seconds against 12 minutes.

**`card` is the one field that records and does nothing else.** It answers a question `commit` and
`taken_at` cannot — *which silicon produced this answer* — because `relic-core` forks on compute
capability (below `sm_80` an FP8 matmul takes its Torch path, and the compute dtype is fp16), so two
cards can agree on the revision and the day and still disagree on the tokens. That makes it worth
writing down beside an answer that might one day differ.

It is deliberately **not** a requirement and not a skip reason. A fixture whose recorded card is not
the card reading it has not changed its answer; it was taken somewhere else, and reporting that as an
unrunnable host would be a different claim about a different thing. Nothing compares the string and
`unwritable_reason` never consults it, which is also why the four fixtures recorded before the field
existed are left empty rather than back-filled with what the tree *would have been* run on: an
inference is not a measurement, and a field whose whole purpose is provenance is worthless if it
carries guesses.

The two consequences of that: a full `POCKETLLM_GOLDEN=1` run covers four of the five fixtures and
skips whichever of the pair the resident bank crowds out, and *which* one is skipped depends on what
was already in `/dev/shm` when the run started rather than on the order in the file. Clear the banks
before a run that is meant to cover both, and expect the first one to pay the cold cost.

A fixture is a smoke test, not a benchmark: it answers "does this checkpoint still produce this
answer through this entry point", which is the question a refactor of the request lifecycle must not
break. It is deliberately not a rate measurement — a rate belongs with the command that produced it
and the host it was measured on, and it moves when either of those does.

Recording one is a deliberate act rather than a side effect, because re-recording is how a real
regression gets quietly blessed as correct:

```bash
python scripts/record_golden_fixture.py --entry xing4 \
    --checkpoint /mnt/data2/Xing4.0-29B-A4B-GGUF/xing4_0-29b-IQ4_NL.gguf \
    --max-tokens 16 -- --max-model-len 4096
```

### Running them is opt-in

The fixtures load real checkpoints, and their cost spans three orders of magnitude: the `xing4` one
answers in 18 seconds, while `v41` pins a 457.8 GiB resident expert bank and takes forty minutes on a
cold segment. So the *runs* are gated:

```bash
python -m pytest tests/test_served_path_golden.py -q                    # 5 skips, one a fixture
POCKETLLM_GOLDEN=1 python -m pytest tests/test_served_path_golden.py -q # now they run
```

The cross-architecture acceptance for `v41` is those same fixtures run twice, in that order, in
separate shells:

```bash
# bit-identical leg: select the v41 comparison node, not the whole module
POCKETLLM_GOLDEN=1 POCKETLLM_GOLDEN_CHECKPOINT=<this host's checkpoint> \
    DEEPSEEK_FP8_IMPL=torch DEEPSEEK_FP4_IMPL=torch \
    python -m pytest "tests/test_served_path_golden.py::test_served_path_matches_the_recorded_answer[v41]" -q -rs

# tolerance leg: native implementation, logits compared within the recorded tolerance
POCKETLLM_GOLDEN=1 POCKETLLM_GOLDEN_CHECKPOINT=<this host's checkpoint> \
    python -m pytest "tests/test_served_path_golden.py::test_the_v41_logits_agree_with_the_recorded_ones" -q -rs
```

The first forces the soft path so `sm_89` is compared against `sm_75` with no kernel difference;
the second leaves the implementation native. Two things about running them:

- **Select the node, do not run the whole module.** The bit-identical shell exports the forced-torch
  vars, and the tolerance test asserts those two are *unset* by design — collecting it here fails
  the leg with an assertion that reads as the comparison breaking. `-k v41` does not select it
  either: the tolerance test's name also contains "v41". The node id is the only selection that
  takes exactly one.
- **Carry `POCKETLLM_GOLDEN_CHECKPOINT`.** The acceptance host holds the checkpoint at another
  path, so without it both legs skip for a checkpoint that is sitting on the disk under another
  name. The paragraph below is what explains the variable; the commands show it so a reader does
  not have to infer the connection.

Both legs run in separate shells — the first exports vars the second refuses. `-rs` is there so a
skip names itself instead of passing quietly.

A fixture records the absolute path its oracle was taken at, and a host that holds the same
checkpoint elsewhere sets `POCKETLLM_GOLDEN_CHECKPOINT` to run the same fixture without editing it.
The variable moves both the recorded `checkpoint` field — which `unwritable_reason` consults before
deciding to skip — and the `argv` item that *is* that path (exact match, not substring: a path that
merely contains the recorded one is a different file and is left alone). A run without it uses the
recorded path.

Everything else in that module always runs, and it is the part that matters for coverage: the
fixture set is complete, every file parses, and every one carries an answer. What the gate defers is
the end-to-end execution, not the claim that the fixture exists.

The file is a statement about the repository, so it is checked in even when the checkpoint lives only
on one machine; a run skips when the path is absent, and `-rs` prints the path it wanted.

### The reference oracle, and why it is a separate file

A fixture under `fixtures/golden/` is a record of what *this repository* produced, so a test that
runs it and compares establishes only that the port is **stable** -- the same input still gives the
same answer. It cannot say whether that answer is **right**, because the fixture and the run share
one implementation. A port whose bias became its own standard would pass that check forever.

`fixtures/reference/<entry>.json` is the other half: what an **upstream** implementation produced for
the same prompt. It is written by `scripts/record_reference_v41.py`, which runs the checkpoint's own
`inference/model.py`, **imported unmodified by path** and never edited -- so the arithmetic under it
is DeepSeek's, not relicllm's. The record is loaded by `load_reference()`, which returns a plain dict
rather than a `GoldenFixture`: a reference carries no `argv` that runs *here* and must not be
loadable as one of this repository's served-path answers.

The one deviation in the recorder is memory placement, and it is forced rather than chosen -- a stock
forward needs ~62.5 GiB per rank and a 24 GiB card cannot hold it, so the two sparse tables stay in
host RAM and their active slice is streamed in per step. That patch replaces `MoE.forward` and
`ParallelEngramEmbedding.forward` with a version that differs from the released one **only by the
`.to(device)` moves**; the arithmetic, the clamps, the routing and the reduction order are the
checkpoint's. The two `forward` bodies are there to diff against `model.py`.

The oracle seeds a **route's** ids, and which route is part of the record. `v41.json` is the **raw**
route: the nine ids the prompt is written as, which is what the served-path fixture feeds. `v41_chat.json`
is the **chat** route: the thirteen ids the checkpoint's own `encode_messages` renders for the same
words, which is what a client of the served chat route sends. The same words are a different token
sequence on each, so an oracle for one route is not an oracle for the other, and each record names
its own in `path` (absent means `raw`, which is what a record written before the field existed drove).
The file suffix follows the same split: `<entry>.json` for raw, `<entry>_<route>.json` for anything
else. `logits_check` records the reference's step-0 top-k for the same reason the fixture does.

Two legs compare against them, and they are gated the same way as the rest:
`test_the_v41_tokens_agree_with_the_official_reference` (raw) and
`test_the_v41_chat_route_agrees_with_the_official_reference` (chat). They are the tests here that
compare the port to the *model* rather than to itself. The **chat** leg is the one that can fail with a
port defect rather than a tolerance drift: its step-0 margin is 14 nat, far above any arithmetic
difference the port carries.
The **raw** leg is a fidelity yardstick, not a defect oracle — the reference decides its step 2 by
0.16 nat, and the port's own residual on that row is ~1.0 nat, so a divergence there is as likely to
be the two implementations' kernels differing as a port defect (the reference's own code picks the
port's index-2 token once its tilelang GEMMs are replaced). Read the margin at the diverging step
before attributing a raw-leg failure to a defect. To
record or refresh an oracle, on a box holding the converted checkpoint and the reference's own env:

```bash
python scripts/record_reference_v41.py \
    --ckpt <converted TP checkpoint> --inference <checkpoint>/inference \
    --out tests/fixtures/reference/v41.json

python scripts/record_reference_v41.py \
    --ckpt <converted TP checkpoint> --inference <checkpoint>/inference \
    --path chat --out tests/fixtures/reference/v41_chat.json
```

A human reads the printed tokens and step-0 top-k before the record is committed, exactly as
`record_golden_fixture.py` expects for its own answer.

### `reproduction`: the configuration the oracle is satisfiable under

The leg above does **not** run the fixture's own `argv`. A reference is a claim about the model, and a
port can only satisfy it from a configuration whose arithmetic *is* the model's; the served-path
fixture's argv is what this repository was recorded producing on one card, which is a different
statement and, for v41, a different computation. So a reference may carry a `reproduction` object
naming the port configuration that reproduces it:

```json
"reproduction": {
  "argv": ["serve", "--backend", "v41", "--model", "/mnt/data3/DeepSeek-V4.1-Flash", ...],
  "env": {},
  "requires": {}
}
```

`reproduction_fixture()` in `tests/golden_fixtures.py` reads it, and the leg runs that configuration
instead. The block is **optional and the identity when absent** -- every other entry point has none --
and it must keep the fixture's `checkpoint` as an item of `argv`, because the harness rewrites the
recorded path by exact match to wherever this host keeps the bytes. `env` and `requires` **replace**
the fixture's rather than extending them; for v41 the empty `env` is load-bearing, because the
fixture's `DEEPSEEK_V41_RESIDENT_EXPERTS=1` pins a 457.8 GiB bank that the host path never reads.

For v41 the configuration is `--tensor-parallel-size 1` with no `--device-ids`, which puts the dense
tree and the routed experts on the host and runs the MoE through `expert_forward`/`quantized_linear`
-- the fp8 E4M3 block-32 activation the reference's own `linear()` quantizes. The fixture's world=4
`argv` routes the same experts through `DeviceRoutedExperts`, whose activation is int8 with one scale
per row; that is a different quantization by construction, and no re-recording makes it agree.

Because the recorder writes the whole file, it would erase a hand-added `reproduction` block on a
refresh; it carries an existing one across for that reason.

A reference whose route has **no** served-path fixture is run on the configuration its own block
names, with `fixture_from_reference()` building the leg from the record alone: the chat route is the
case, because `fixtures/golden/` holds one fixture per entry point and a second answer for the same
entry point is a second question nobody recorded. The fixture that helper builds carries **no
`expected`** -- the reference's `tokens` are what the run is compared against, and copying them into
the fixture would make the leg compare the oracle to itself -- and no `commit`/`card` either, because
those fields mean "this repository produced this answer, here" and this leg has produced no answer of
its own.

### Multi-prompt references: a *set* of answers, and what to do when the reference is unsure

One prompt is a question a broken port can answer by luck. `tests/fixtures/reference/v41_chat_multi.json`
is the reference's answers to **eight** chat-templated prompts, recorded in one loaded process
against the same unmodified `inference/model.py`: a control the recorder repeats first and last, a
twenty-token count, arithmetic, a repetition instruction, an open-ended question, a code expression
and a Chinese sentence. The leg is
`test_the_v41_chat_route_agrees_with_the_official_reference_on_each_prompt`, one prompt a child
process, collected by `_recorded_multi_prompts()`.

The record is the single-prompt shape with two differences. `prompts` is a list of
`{label, prompt, prompt_tokens, max_new, tokens, step_gaps}` rather than one `prompt`/`tokens` pair,
and each prompt carries no `expected` of its own -- as with `fixture_from_reference`, the reference's
`tokens` *are* the answer, and copying them into a fixture would compare the oracle to itself. The
loader `load_multi_reference()` reads `<entry>_<route>_multi.json` and validates it through
`multi_reference_prompts()`, which raises rather than returning an empty set: a malformed criterion
is a broken one, so it aborts collection of the module instead of silently running fewer legs.
`test_the_multi_prompt_reference_is_committed` is the checkpoint-free guard that the file exists at
all, and `MULTI_PROMPT_LABELS` pins the label *sequence* so a re-record that dropped or reordered a
prompt fails rather than shrinking the gate.

**`step_gaps` is why the record carries more than answers.** It is the reference's own top1-top2
logit gap at *every* generated step, read from the recorder's `all_logits`. A greedy answer is only
as decisive as its margin: `explain-blue` is a prompt the reference decided by **0.1554 nat** at step
12 of 24, and on that step a port that picks a different word is being asked to reproduce one
kernel's rounding rather than the model's answer. So the leg judges the gap **at the step the port
actually diverged** -- not the sequence minimum, which would launder a wrong first token into a coin
flip. At or above the record's `confident_nats` line (1.0) the reference decided by more than any
run-to-run float spread it carries, so the divergence is a **port defect** and the leg fails; below
it the divergence is a **measurement** and the leg calls `pytest.xfail()` *after* the run, so a
divergence reports as an expected failure while an agreement still shows as a pass.

**A disagreement about length is never a coin flip, in either direction.** When the two runs stop at
different steps there is no reference margin that speaks to where they parted -- the reference's
decision there was to emit nothing more, or to emit a token the other run never reached. So a port
that stops short, or one that generates past the reference, **fails** rather than xfails; otherwise a
port that ignores EOS would read as green. `test_the_multi_prompt_leg_fails_a_confident_divergence_and_records_a_coin_flip`
exercises both directions of the branch, and the length cases, against a synthetic record with the
model stubbed out, so the decision is pinned without a checkpoint.

**The `confident_nats` line is only meaningful because the criterion config is bitwise stable.** The
line draws the boundary at this port's own run-to-run spread, so it is worth knowing what that spread
is. On the criterion (`TP1` host, `decode_graphs=false`) configuration the recorded control appears
twice in one process, six prompts apart, with **bitwise-identical** step-0 rows -- so the spread there
is zero, and a 1.0-nat line is conservative rather than tight. (A larger 0.4-1.5 nat in-process
spread has been seen on the `TP4` device config with decode graphs, so the line is stated per-record
rather than hard-coded: a criterion whose config turns out to be less stable carries its own
`confident_nats`.)
