"""Golden fixtures: the served path, end to end, per entry point.

Three tests, and they answer different questions.

- `test_every_entry_point_has_a_fixture` is a statement about the repository and needs no weights,
  no card and no engine: every entry point `tests/README.md` names has a recorded fixture. It is the
  test that makes a missing entry point visible, because a fixture that does not exist skips rather
  than fails and a suite of skips reads as green.
- `test_every_entry_point_that_is_recorded_parses` says the recorded files are loadable and carry an
  answer. Also free.
- `test_served_path_matches_the_recorded_answer` runs each fixture and compares what comes back. It
  is **opt-in**, gated on `POCKETLLM_GOLDEN=1`, and it skips when the checkpoint is not on this
  machine -- the fixture names a path, and the only honest outcome without it is a skip.

Each fixture runs in its own child process, for the reason `run_isolated` documents: five engines in
one interpreter is a configuration nothing else here uses, and `xing4` after `mimo` and `torch`
after `mimo` both fail in it for reasons that are not about the checkpoint. The comparison stays
here, so the failure still names the fixture.

Why that gate rather than running the fixture always: these are the only tests here that load real
weights, and the fixtures span three orders of magnitude in cost. `xing4` answers in 18 seconds; the
`v41` fixture pins a 457.8 GiB resident expert bank and takes forty minutes on a cold segment. A
suite that a person runs before a commit cannot include the second one by default, and a gate with a
documented reason is better than either a silently cheap subset or a suite nobody runs.

What a mismatch means: this checkpoint, through this entry point, with these flags, no longer
produces these tokens. That is the claim a request-lifecycle refactor has to keep, and it is the one
claim no kernel-local test can make.

The `v41_*` tests below are the other question -- not "is the port stable" but "is it *right*" -- and
they compare against `fixtures/reference/`, which is DeepSeek's own code rather than this repository's.
There are two of them because there are two oracles: the raw route (`v41.json`, nine prompt ids) and
the chat route (`v41_chat.json`, the thirteen ids the checkpoint's own renderer produces for the same
words). They are documented in `tests/README.md`; the short version is that a reference names the port
configuration that satisfies it and the leg runs that configuration rather than the fixture's.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib

import pytest

from tests.golden_fixtures import (
    ENTRY_POINTS,
    GOLDEN_GATE_ENV,
    PATH_CHAT,
    GoldenFixture,
    Outcome,
    _with_host_checkpoint,
    fixture_from_reference,
    fixture_path,
    load_fixture,
    load_reference,
    logits_check_mismatch,
    reference_path,
    reproduction_fixture,
    run_isolated,
)


def _recorded() -> list[tuple[str, GoldenFixture]]:
    return [(entry, fixture) for entry in ENTRY_POINTS if (fixture := load_fixture(entry)) is not None]


def _served_skip_reason(fixture: GoldenFixture) -> str | None:
    """Why this fixture cannot run here, or `None` when it can.

    Resolving the host's checkpoint first is what keeps this decision and `run_isolated`'s in
    agreement: if the skip were decided on the recorded path while the launcher opened the host's, a
    host holding the bytes at another path would skip, and the acceptance gate would pass vacuously
    -- the failure this whole harness exists to close. Split out so the wiring is testable without a
    checkpoint on disk.
    """
    return _with_host_checkpoint(fixture).unwritable_reason()


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_every_entry_point_has_a_fixture(entry: str) -> None:
    path = fixture_path(entry)
    assert path.is_file(), (
        f"no golden fixture for the {entry!r} entry point. Record one with\n"
        f"    python scripts/record_golden_fixture.py --entry {entry} --checkpoint <path>\n"
        f"which writes {path.relative_to(path.parents[2])}. A path that does not exist is not a "
        f"reason to leave it out: the fixture records where the checkpoint lives and the test skips "
        f"when it is absent, so the file is a statement about this repository rather than about the "
        f"machine reading it."
    )


def test_every_recorded_fixture_belongs_to_a_known_entry_point() -> None:
    """A file in `fixtures/golden/` that is not an entry point is a fixture nobody will ever run.

    The completeness test above goes the other way -- an entry point with no file -- so the two
    together pin the set, and this one is what notices `fixtures/golden/troch.json`, which would be
    a typo'd fixture that skipped nothing and checked nothing.
    """
    from tests.golden_fixtures import FIXTURE_DIR, load_fixtures

    on_disk = {path.stem for path in FIXTURE_DIR.glob("*.json")}
    orphans = sorted(on_disk - set(ENTRY_POINTS))

    assert not orphans, (
        f"{', '.join(orphans)} in {FIXTURE_DIR.relative_to(FIXTURE_DIR.parents[1])} name no entry "
        f"point, so nothing loads them. Either the entry point is missing from ENTRY_POINTS or the "
        f"file is a typo"
    )
    assert set(load_fixtures()) == on_disk


@pytest.mark.parametrize("entry,fixture", _recorded(), ids=[entry for entry, _ in _recorded()])
def test_served_path_matches_the_recorded_answer(entry: str, fixture: GoldenFixture) -> None:
    if not os.environ.get(GOLDEN_GATE_ENV):
        pytest.skip(
            f"the served-path fixtures are opt-in: set {GOLDEN_GATE_ENV}=1 to run them. They load "
            f"real checkpoints and the cost ranges from seconds to the better part of an hour"
        )
    reason = _served_skip_reason(fixture)
    if reason is not None:
        pytest.skip(reason)

    outcome = run_isolated(fixture)

    # The prompt's own token count first, because it is the half of a fixture a moved *tokenizer*
    # shows up in, and a comparison of the answer alone would blame the model for it.
    if fixture.expected_prompt_tokens is not None:
        assert outcome.prompt_tokens == fixture.expected_prompt_tokens, (
            f"{entry}: the prompt rendered to {outcome.prompt_tokens} tokens, not "
            f"{fixture.expected_prompt_tokens}. The text is unchanged, so this is the tokenizer or "
            f"the chat template -- both of which change what the model is asked"
        )

    compared = False
    # `is not None` and not truthiness: an empty `token_ids` is a recorded *list*, and the point of
    # recording it is to assert the answer is still empty -- that the entry point has *stopped*
    # producing ids, which is a regression a positive list could not show. A truthiness test reads
    # `token_ids: []` as "the fixture records none", skips the comparison, and lets an entry point
    # that silently stopped returning ids pass on its text alone.
    if fixture.expected_token_ids is not None:
        assert outcome.token_ids is not None, (
            f"the {entry} fixture records token ids but this entry point produced none; either the "
            f"entry point stopped returning ids or the fixture was recorded under a different kind"
        )
        assert list(outcome.token_ids) == list(fixture.expected_token_ids), (
            f"{entry}: token ids moved. Recorded {list(fixture.expected_token_ids)}, got "
            f"{list(outcome.token_ids)}. The fixture was taken at {fixture.commit or 'an unknown'} "
            f"with {' '.join(fixture.argv)}; a mismatch is either a regression or a fixture that "
            f"needs re-recording, and `git log -1 --format=%H` is what tells them apart"
        )
        compared = True
    if fixture.expected_text is not None:
        assert outcome.text is not None, f"the {entry} fixture records text but none came back"
        assert outcome.text == fixture.expected_text, (
            f"{entry}: served text moved. Recorded {fixture.expected_text!r}, got {outcome.text!r}"
        )
        compared = True
    assert compared, (
        f"the {entry} fixture records neither token ids nor text, so it cannot fail and is not a "
        f"fixture"
    )


def _run_under(fixture: GoldenFixture, extra_env: dict[str, str]) -> Outcome:
    """Run one fixture in a child with extra env injected, skipping for the same host reasons.

    The gate check here duplicates the caller's on purpose: this is the helper the one caller uses,
    but it is written to stand alone, and a reader who "fixes" the duplication would make it depend
    on a caller it does not name. The skip reason goes through `_served_skip_reason` rather than
    `unwritable_reason` for the reason that helper documents -- a skip decided on the recorded path
    while the child opens the host's passes vacuously on the host the override exists for.
    """
    if not os.environ.get(GOLDEN_GATE_ENV):
        pytest.skip(f"the served-path fixtures are opt-in: set {GOLDEN_GATE_ENV}=1 to run them.")
    reason = _served_skip_reason(fixture)
    if reason is not None:
        pytest.skip(reason)
    return run_isolated(fixture, extra_env=extra_env)


def test_the_v41_logits_agree_with_the_recorded_ones():
    """The tolerance leg: the fixture's own argv, native implementation, judged on logits.

    Skips unless the fixture records a `logits_check` -- only v41 does, and only after the sm_75
    recording pass. When triton and torch agree bit-for-bit this passes with a zero gap; the gap
    is what the assertion message reports when they do not.

    This leg is only meaningful with the implementation left native, so a shell that already
    exports `DEEPSEEK_FP8_IMPL`/`DEEPSEEK_FP4_IMPL` from the bit-identical leg would silently make
    this a second copy of it. The guard below fires only once the leg would actually run, so a host
    that exports those for unrelated reasons does not fail an ordinary `pytest tests/` run.
    """
    fixture = load_fixture("v41")
    if not os.environ.get(GOLDEN_GATE_ENV):
        pytest.skip(f"the served-path fixtures are opt-in: set {GOLDEN_GATE_ENV}=1 to run them.")
    if fixture is None or fixture.expected_logits_check is None:
        pytest.skip("v41 records no logits_check; run the sm_75 recording pass first")

    forced = [key for key in ("DEEPSEEK_FP8_IMPL", "DEEPSEEK_FP4_IMPL") if key in os.environ]
    assert not forced, (
        f"{', '.join(forced)} is set, which forces the soft path; the tolerance leg must run with "
        f"the implementation native. Run the two legs in separate shells."
    )

    outcome = _run_under(
        fixture,
        {"POCKETLLM_V41_LOGITS_CHECK": str(fixture.expected_logits_check["top_k"])},
    )
    assert outcome.logits_check is not None, (
        "the run was asked for a logits record and produced none"
    )
    mismatch = logits_check_mismatch(outcome.logits_check, fixture.expected_logits_check)
    assert mismatch is None, mismatch


def test_the_v41_tokens_agree_with_the_official_reference():
    """The upstream leg: this card's greedy tokens against DeepSeek's own reference.

    The oracle is recorded by `scripts/record_reference_v41.py` from the checkpoint's **unmodified**
    `inference/model.py` (host-offloaded to fit a 24 GiB card), not from a relicllm run -- so this is
    the one check here that compares the port to the model rather than to itself. The served-path
    fixture's own `token_ids` are a record of what *sm_75* produced, which is a different question:
    a fixture that matches itself proves the port is stable, not that it is right.

    **The configuration this leg runs is the reference's, not the fixture's.** A reference is a claim
    about the model, and only a port configuration whose arithmetic *is* the model's can satisfy it;
    the fixture's argv is a device-path configuration that was recorded on the card for a different
    purpose and, for v41, quantizes the expert activations differently by construction.
    `reproduction_fixture` reads the configuration the reference names for itself and is the identity
    for a reference that names none, so this is a no-op for every entry point but v41.

    The comparison is on greedy **tokens** only. The reference's step-0 logits come from an fp32 head
    on the reference's own arithmetic; a per-value tolerance across two implementations is a weaker
    statement than "the argmax sequence agrees", and the token sequence is what a served answer is.
    """
    fixture = load_fixture("v41")
    reference = load_reference("v41")
    if not os.environ.get(GOLDEN_GATE_ENV):
        pytest.skip(f"the served-path fixtures are opt-in: set {GOLDEN_GATE_ENV}=1 to run them.")
    if fixture is None or reference is None:
        pytest.skip("v41 has no recorded official reference; run scripts/record_reference_v41.py")

    # The oracle is the raw route: `scripts/record_reference_v41.py` seeds the nine raw ids, not a
    # chat render. A fixture claiming the other route answers a different question, and comparing
    # the two would fail as "a port defect" while the port is fine. The route is only checkable
    # here, because the oracle is what decides which route the fixture is allowed to claim.
    assert fixture.path == "raw", (
        f"the v41 fixture records the {fixture.path!r} route, but the official reference was "
        f"recorded on the raw ids; the two answers are not to the same question"
    )

    expected_tokens = list(reference["tokens"])
    outcome = _run_under(reproduction_fixture(fixture, reference), {})

    assert outcome.token_ids is not None, "the run produced no token ids"
    assert list(outcome.token_ids) == expected_tokens, (
        f"v41: the greedy tokens diverge from the official reference.\n"
        f"  reference: {expected_tokens}\n"
        f"  this card: {list(outcome.token_ids)}\n"
        f"The reference is {reference['source']} at commit {reference.get('commit') or 'unknown'}; "
        f"the run is {fixture.card or 'an unrecorded card'}. A divergence here is a port defect, "
        f"not a kernel tolerance."
    )


def test_the_v41_chat_route_agrees_with_the_official_reference():
    """The second oracle: the same prompt through the served chat route, against DeepSeek's code.

    The sibling test above compares the **raw** route -- the nine prompt ids as written -- against
    `fixtures/reference/v41.json`. This one compares the **chat** route, where the port renders the
    prompt through the checkpoint's own `encode_messages` and sends thirteen ids, against
    `fixtures/reference/v41_chat.json`, recorded from the same unmodified `inference/model.py` seeded
    with those thirteen. Two routes, two questions: the same words are a different token sequence on
    each, so one oracle cannot judge the other's answer, and the served path has to be right on both.

    **The leg is built from the reference, not from a served-path fixture.** There is no
    `fixtures/golden/v41_chat.json`, and there should not be one invented to hold a second answer for
    the same entry point: `fixture_from_reference` reads the configuration the reference names and
    the route it was recorded on, and `sampling` is the greedy default both recorders use. What is
    compared is the reference's own `tokens`, so nothing here compares the port to itself.

    The one thing this test does *not* establish by itself is that the port and the reference were
    shown the same thirteen ids -- it pins the prompt's *count*, because a run that took the raw
    route would answer a different question while looking like a divergence from this one, and two
    different thirteen-id prompts compare just as well as one. The ids themselves are the subject of
    `test_the_v41_chat_route_sends_the_thirteen_ids_the_reference_was_seeded_with` below, which
    renders the prompt through the checkpoint's own encoder and tokenizer.
    """
    reference = load_reference("v41", PATH_CHAT)
    if reference is None:
        pytest.skip("v41 has no recorded official reference for the chat route; see tests/README.md")

    fixture = fixture_from_reference(reference, entry="v41")
    # The route this leg runs is the record's, and the record is the thing being satisfied -- the
    # sibling test pins the same field for the same reason. `load_reference` has already refused a
    # file whose own `path` disagrees with the route it was loaded under, so this is the assertion
    # that the leg built from it still drives that route.
    assert fixture.path == PATH_CHAT, (
        f"the v41 chat leg was built on the {fixture.path!r} route, but the record it is compared "
        f"against answers the {PATH_CHAT!r} route"
    )
    expected_tokens = list(reference["tokens"])
    outcome = _run_under(fixture, {})

    assert outcome.token_ids is not None, "the run produced no token ids"
    expected_prompt_tokens = (reference.get("logits_check") or {}).get("prompt_tokens")
    if expected_prompt_tokens is not None and outcome.prompt_tokens is not None:
        assert outcome.prompt_tokens == expected_prompt_tokens, (
            f"v41 chat: the port sent {outcome.prompt_tokens} prompt tokens and the reference was "
            f"seeded with {expected_prompt_tokens}; the two are answering different questions, so "
            f"the token comparison below would not mean what it says"
        )
    assert list(outcome.token_ids) == expected_tokens, (
        f"v41 chat: the greedy tokens diverge from the official reference.\n"
        f"  reference: {expected_tokens}\n"
        f"  this card: {list(outcome.token_ids)}\n"
        f"The reference is {reference['source']} at commit {reference.get('commit') or 'unknown'}, "
        f"seeded with the checkpoint's own chat render and run on the configuration its "
        f"`reproduction` block names. A divergence here is a port defect, not a kernel tolerance."
    )


def test_the_v41_chat_route_sends_the_thirteen_ids_the_reference_was_seeded_with():
    """The other half of the chat criterion: the ids, not their count.

    `test_the_v41_chat_route_agrees_with_the_official_reference` holds the port to the reference's
    answer and pins the prompt's *count*. A count is not a prompt: any thirteen ids pass it, and the
    ids are the whole reason the two routes are two questions. This runs the two steps the served
    chat route runs before it has anything to answer with -- `prepare_messages` and the checkpoint's
    own `encode_messages`, then the checkpoint's tokenizer with `add_special_tokens=False` -- and
    holds the result against the record's `raw_token_ids`, the sequence the oracle was actually fed.
    A port that rendered the prompt differently, or tokenized it with a second beginning-of-sentence
    token, would answer `[69865, 267, 45117, 1]`-shaped questions that this test is the only one to
    attribute.

    It needs the checkpoint's encoder and tokenizer but no weights and no card, so it is the cheap
    half of the gate: same `POCKETLLM_GOLDEN` gate, seconds rather than an hour.
    """
    if not os.environ.get(GOLDEN_GATE_ENV):
        pytest.skip(f"the served-path fixtures are opt-in: set {GOLDEN_GATE_ENV}=1 to run them.")
    reference = load_reference("v41", PATH_CHAT)
    if reference is None:
        pytest.skip("v41 has no recorded official reference for the chat route; see tests/README.md")
    recorded_ids = list(reference.get("raw_token_ids") or [])
    assert recorded_ids, (
        "the v41 chat reference carries no `raw_token_ids`, so the prompt it was seeded with is "
        "not recorded and this leg has nothing to hold the renderer to"
    )
    fixture = _with_host_checkpoint(fixture_from_reference(reference, entry="v41"))
    reason = fixture.unwritable_reason()
    if reason is not None:
        pytest.skip(reason)

    from relicllm.encoding.deepseek_v4_1 import encode_messages
    from relicllm.protocol.chat import prepare_messages

    pytest.importorskip("transformers", reason="the tokenizer is a transformers tokenizer")
    from transformers import AutoTokenizer

    messages = prepare_messages(
        {"messages": [{"role": "user", "content": str(reference.get("prompt") or "")}]}
    )
    text = encode_messages(fixture.checkpoint, messages)
    tokenizer = AutoTokenizer.from_pretrained(fixture.checkpoint)
    # The rendered prompt opens with its own beginning-of-sentence token, so the tokenizer must not
    # add a second one -- the same call the served route makes (`v41_backend._encode_chat`).
    ids = [int(token) for token in tokenizer(text, add_special_tokens=False)["input_ids"]]

    assert ids == recorded_ids, (
        f"v41 chat: the port renders and tokenizes this prompt to a different sequence than the "
        f"official reference was seeded with.\n"
        f"  reference: {recorded_ids}\n"
        f"  this port: {ids}\n"
        f"rendered text: {text!r}\n"
        f"The token comparison in `test_the_v41_chat_route_agrees_with_the_official_reference` "
        f"counts these ids and would pass on a different thirteen; this is the prompt itself."
    )


def test_a_reference_route_that_is_not_one_of_the_two_paths_is_refused():
    """A route the loader does not know must not come back as `no oracle yet`.

    The route picks the file, so an unrecognised one names a file nobody recorded: read as a missing
    oracle it turns a criterion into a skip, and a skipped acceptance leg reads as a pass.
    `_read_path` refuses an unknown `path` inside a record for the same reason -- one refusal at each
    boundary, because the name and the field arrive from different places.
    """
    with pytest.raises(ValueError, match="reference route must be one of"):
        reference_path("v41", "chatt")


def test_a_reference_record_is_refused_when_its_route_disagrees_with_the_name_it_is_read_under(
    tmp_path, monkeypatch
):
    """The file name is the caller's claim; the record's `path` is the recorder's.

    They can be made to disagree -- `record_reference_v41.py --out` takes any name it is given -- and
    a record that ran one route while sitting at the other's name would be compared as if it answered
    the route its name claims, against a prompt the run was never given. Refused rather than
    defaulted: the two routes are two questions with two different answers.
    """
    from tests import golden_fixtures

    monkeypatch.setattr(golden_fixtures, "REFERENCE_DIR", tmp_path)
    record = tmp_path / "v41_chat.json"
    record.write_text(json.dumps({"path": "raw", "tokens": [1]}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="records the 'raw' route"):
        load_reference("v41", PATH_CHAT)

    # The positive control: the same file, recording the route its name claims, loads. Without it
    # this test would pass on a loader that refused every record, including the real ones.
    record.write_text(json.dumps({"path": PATH_CHAT, "tokens": [1]}) + "\n", encoding="utf-8")
    assert load_reference("v41", PATH_CHAT) == {"path": PATH_CHAT, "tokens": [1]}


def _reference_recorder():
    """`scripts/record_reference_v41.py` loaded by path.

    By path rather than by `import scripts.record_reference_v41`: `scripts/` is not a package, and a
    namespace-package import would depend on the working directory being the repository root, which
    is not something this suite's invocation promises. The module is import-safe -- everything it
    needs at import time is stdlib, and the reference run itself is behind `main()`.
    """
    path = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "record_reference_v41.py"
    spec = importlib.util.spec_from_file_location("_record_reference_v41", path)
    assert spec is not None and spec.loader is not None, f"{path} is not importable"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_chat_reference_carries_exactly_what_its_recorder_writes():
    """A re-record has to reproduce the record, not quietly edit it.

    Every field of `v41_chat.json` except the answer the reference run produced -- `tokens`,
    `logits_check`, and the `commit`/`taken_at` stamp -- is derived by
    `scripts/record_reference_v41.py`, and the `reproduction` block is carried across from the file
    being overwritten. The derivation is checked here against the script rather than against a copy
    of its text, so a refresh that produced a different file would be a failing test instead of a
    diff nobody reads; the route sentence in `notes` is a field of the record for the same reason,
    rather than a sentence somebody appended to it once and a refresh would drop.
    """
    reference = load_reference("v41", PATH_CHAT)
    if reference is None:
        pytest.skip("v41 has no recorded official reference for the chat route; see tests/README.md")
    recorder = _reference_recorder()

    assert reference["entry"] == "v41"
    assert reference["source"] == recorder.SOURCE
    assert reference["prompt"] == recorder.DEFAULT_PROMPT
    assert list(reference["raw_token_ids"]) == list(recorder.CHAT_TOKENS), (
        "the record's seeded ids are not the ones its own route seeds"
    )
    assert reference["path"] == PATH_CHAT
    assert reference["notes"] == recorder.BASE_NOTES + recorder.ROUTE_NOTES[PATH_CHAT]
    assert "reproduction" in reference, (
        "the chat leg runs the configuration this block names; a record without one is refused by "
        "`fixture_from_reference` and the criterion cannot run at all"
    )
