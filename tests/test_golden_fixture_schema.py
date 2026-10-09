"""The golden-fixture harness itself: the schema, the flag reader, the child protocol, the skip rule.

The end-to-end tests in `test_served_path_golden.py` all skip without a checkpoint, so without these
the harness would be code that runs nowhere on a machine with no weights -- which is most machines.
What they cover is the part that decides *what* an end-to-end run means: how a fixture is read and
written, how a child hands an outcome back, and when a fixture is skipped rather than run.

The skip rule is the one worth being explicit about, and it has exactly two reasons, both about the
machine rather than about the answer: the checkpoint is not here, and a resource the fixture states
it needs is not free. Both are asserted below so that nothing else can quietly join that list: a
fixture that skipped because its *answer* moved would be a regression wearing a skip.

The `card` a fixture records is deliberately *not* a third reason, and there is a test below that
holds it to being recorded-and-never-read: the field answers which silicon produced the answer, and
a fixture whose card differs from the one reading it has not moved its answer -- it was simply taken
somewhere else. Skipping on it would report that as an unrunnable host, which is a different claim.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest
from tests.golden_fixtures import (
    CHECKPOINT_ENV,
    CHILD_FIXTURE_ENV,
    CHILD_RESULT_ENV,
    DEFAULT_SAMPLING,
    ENTRY_POINTS,
    GOLDEN_MODULE,
    REPO_ROOT,
    GoldenFixture,
    Outcome,
    _with_repo_root_on_the_path,
    captured_card,
    fixture_path,
    load_fixture,
    load_fixtures,
    run_isolated,
)

PAYLOAD = {
    "entry": "xing4",
    # Synthetic, and deliberately a path that does not exist: this module must not depend on which
    # checkpoints this machine happens to have, and one test below asserts the skip rule on it.
    "checkpoint": "/mnt/nope/xing4_0-29b-IQ4_NL.gguf",
    "argv": ["serve", "--backend", "xing4", "--model", "/mnt/nope/xing4_0-29b-IQ4_NL.gguf"],
    "env": {"CUDA_VISIBLE_DEVICES": "2"},
    "prompt": "Reply with the single line: golden fixture.",
    "sampling": {"temperature": 0.0, "max_tokens": 16},
    "expected": {"prompt_tokens": 9, "token_ids": [35, 48972], "text": "\ngolden fixture"},
    "commit": "abc1234",
    "taken_at": "2026-09-26",
    "notes": "why this checkpoint",
}


def test_a_fixture_round_trips_through_json() -> None:
    fixture = GoldenFixture.from_json(PAYLOAD)

    assert GoldenFixture.from_json(json.loads(json.dumps(fixture.to_json()))) == fixture


def test_the_card_is_recorded_and_a_fixture_without_one_is_not_a_skip() -> None:
    """The field is provenance, not a requirement, and the two are kept apart on purpose.

    Every fixture recorded before the field existed has no card, and reading one has to stay a
    non-event: an empty `card` is a gap in what the file knows, not a host that cannot run. If this
    ever grew into a skip reason it would report "different silicon" as "cannot run here", which is
    the wrong claim about the wrong thing.
    """
    without = GoldenFixture.from_json(PAYLOAD)
    assert without.card == ""

    with_card = GoldenFixture.from_json({**PAYLOAD, "card": "cuda 8.9 (Ada / RTX 4090)"})
    assert with_card.card == "cuda 8.9 (Ada / RTX 4090)"
    # Same checkpoint, same reason, card or no card: the field never reaches the skip rule.
    assert with_card.unwritable_reason() == without.unwritable_reason()


def test_a_card_is_only_named_when_it_was_read_and_recognised() -> None:
    """An unread or unrecognised card is `""`, not the floor the probe falls back to.

    `probe_card_capability` answers Turing (``7.5``) on a host it cannot read a device on, and that
    value is right about *capability* and wrong about *identity* -- recording it would pin a lie.
    The two cases below are the ones a recording host actually hits, and both either name the card
    or say nothing.
    """
    assert captured_card(platform="cuda", capability=(8, 9)) == "cuda 8.9 (Ada / RTX 4090)"
    # A recognised but older card is still named -- the descriptor knows it.
    assert captured_card(platform="cuda", capability=(7, 5)) == "cuda 7.5 (Turing / RTX 2080 Ti)"
    # A capability the descriptor does not recognise: read, but not believed, so nothing is named.
    assert captured_card(platform="cuda", capability=(6, 1)) == ""


def test_the_card_label_names_the_platform_it_was_read_on() -> None:
    """A non-CUDA host is named as hardware, never as a compute capability it does not have."""
    assert captured_card(platform="ascend") == "ascend 910B"
    assert captured_card(platform="cpu") == ""


def test_the_expected_side_is_read_by_name_and_not_by_position() -> None:
    """The three fields are separate claims and each has its own accessor."""
    fixture = GoldenFixture.from_json(PAYLOAD)

    assert fixture.expected_token_ids == (35, 48972)
    assert fixture.expected_text == "\ngolden fixture"
    assert fixture.expected_prompt_tokens == 9


def test_an_empty_recorded_answer_is_not_the_same_as_no_recorded_answer() -> None:
    """Three states, not two: ids, an empty answer held to be empty, and a field that was never read.

    `token_ids: []` is a claim -- this entry point returns no ids -- and the thing that makes it worth
    recording is that the suite still holds the entry point to it. Folding it into "records none"
    with an `or ()` is how a fixture that records no ids becomes a fixture that checks nothing.
    """
    empty = GoldenFixture.from_json({**PAYLOAD, "expected": {**PAYLOAD["expected"], "token_ids": []}})
    absent_payload = {key: value for key, value in PAYLOAD["expected"].items() if key != "token_ids"}
    absent = GoldenFixture.from_json({**PAYLOAD, "expected": absent_payload})

    assert empty.expected_token_ids == ()
    assert absent.expected_token_ids is None


def test_a_fixture_without_the_required_fields_is_refused() -> None:
    """Every one of these is load-bearing, so a truncated fixture fails at load, not at compare."""
    for missing in ("entry", "checkpoint", "argv", "prompt", "expected"):
        payload = {k: v for k, v in PAYLOAD.items() if k != missing}
        with pytest.raises(ValueError, match=missing):
            GoldenFixture.from_json(payload)


def test_a_fixture_is_skipped_only_for_a_missing_checkpoint() -> None:
    fixture = GoldenFixture.from_json(PAYLOAD)

    reason = fixture.unwritable_reason()

    assert reason is not None and "checkpoint not present" in reason


def test_a_fixture_whose_checkpoint_exists_is_not_skipped(tmp_path: pathlib.Path) -> None:
    present = tmp_path / "checkpoint"
    present.mkdir()
    fixture = GoldenFixture.from_json({**PAYLOAD, "checkpoint": str(present)})

    # No GPU, no built extension, no engine -- and none of that is a reason to skip. Those belong to
    # the entry point, which is expected to raise, because an answer that moved has to be reported.
    assert fixture.unwritable_reason() is None


def test_the_entry_point_set_is_the_one_the_readme_promises() -> None:
    """A new backend cannot be added without either recording a fixture or failing this test."""
    assert ENTRY_POINTS == ("v41", "mimo", "xing4", "torch")


def test_the_default_sampling_is_greedy() -> None:
    """A fixture that samples is a fixture that fails one run in ten for no reason."""
    assert DEFAULT_SAMPLING["temperature"] == 0.0


def test_a_recorded_fixture_names_its_own_entry_point() -> None:
    """The file name and the `entry` field have to agree, or the completeness test lies.

    `load_fixture("mimo")` reads `mimo.json`; a file whose contents say `xing4` would satisfy the
    completeness check for a backend that has no fixture at all.
    """
    for entry in ENTRY_POINTS:
        fixture = load_fixture(entry)
        if fixture is None:
            continue
        assert fixture.entry == entry, f"{fixture_path(entry)} records entry={fixture.entry!r}"
        assert fixture_path(entry).name == f"{entry}.json"


def test_every_loaded_fixture_can_be_compared() -> None:
    """A fixture with no recorded answer cannot fail, and a thing that cannot fail is not a test."""
    for entry, fixture in load_fixtures().items():
        has_answer = fixture.expected_token_ids is not None or fixture.expected_text is not None
        assert has_answer, f"{entry} records neither token ids nor text"


def test_the_outcome_survives_the_trip_through_the_child() -> None:
    """The parent reads the child's answer off disk, so every field has to round trip exactly.

    `None` is the interesting case: an entry point that produces no token ids is a fact about the
    entry point, not a missing field, and a round trip that turned it into `[]` would make `torch`'s
    fixture look like it had been compared when it had not.
    """
    with_record = Outcome(
        token_ids=[1, 2],
        text="hi",
        prompt_tokens=3,
        logits_check={"step": 0, "prompt_tokens": 3, "top_k": 2,
                      "token_ids": [5, 6], "values": [1.5, 0.5]},
    )
    for outcome in (
        Outcome(token_ids=[35, 48972], text="\ngolden fixture", prompt_tokens=9, elapsed_seconds=1.5),
        Outcome(text="golden fixture.", prompt_tokens=21, elapsed_seconds=16.0),
        with_record,
    ):
        assert Outcome.from_json(json.loads(json.dumps(outcome.to_json()))) == outcome

    # The record is read by name by the tolerance leg, so its shape is asserted directly and not
    # only through dataclass equality.
    back = Outcome.from_json(json.loads(json.dumps(with_record.to_json())))
    assert back.logits_check == {"step": 0, "prompt_tokens": 3, "top_k": 2,
                                 "token_ids": [5, 6], "values": [1.5, 0.5]}


def test_a_payload_recorded_before_the_logits_field_still_loads() -> None:
    """Every fixture committed before this field exists has no `logits_check` key at all.

    `from_json` reads it with `.get()`, so an old payload loads with the record absent rather than
    raising -- which is what keeps the committed fixtures readable.
    """
    older = {"token_ids": [35, 48972], "text": "\ngolden fixture", "prompt_tokens": 9,
             "elapsed_seconds": 1.5}

    assert Outcome.from_json(older).logits_check is None


def test_the_child_module_is_launchable() -> None:
    """`run_isolated` spawns this file by path, so it has to work as a program and not only a module.

    It is checked here because the alternative is finding out during a fixture run, where the failure
    arrives as a non-zero exit from a forty-minute process.
    """
    completed = subprocess.run(
        [sys.executable, str(GOLDEN_MODULE), "--help"], capture_output=True, text=True
    )

    assert completed.returncode == 0, completed.stderr
    assert "--entry" in completed.stdout


def test_the_child_refuses_an_entry_point_with_no_fixture() -> None:
    completed = subprocess.run(
        [sys.executable, str(GOLDEN_MODULE), "--entry", "definitely-not-an-entry"],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "invalid choice" in completed.stderr


def test_the_child_runs_the_fixture_it_is_handed_and_not_the_one_on_disk(tmp_path) -> None:
    """The fixture travels to the child, so a recorder runs what it built and not the committed one.

    The child used to re-read `fixtures/golden/<entry>.json`, which is right for the *comparison* --
    there the file is the thing under test -- and wrong for the recorder, which runs a fixture it has
    not written yet. The two share an entry name, so the mismatch is silent: the recorded answer comes
    from the committed checkpoint while the file written describes the one just named. Handing the
    fixture over is what makes the recorder's `--checkpoint` mean something.
    """
    handed = tmp_path / "handed.json"
    handed.write_text(json.dumps({**PAYLOAD, "entry": "torch"}), encoding="utf-8")
    env = {**os.environ, CHILD_FIXTURE_ENV: str(handed), CHILD_RESULT_ENV: str(tmp_path / "out.json")}
    completed = subprocess.run(
        [sys.executable, str(GOLDEN_MODULE), "--entry", "xing4", "--help"],
        capture_output=True,
        text=True,
        env=env,
    )
    # `--help` exits before the fixture is read, so this only proves the option parses; the mismatch
    # check is what the next run exercises.
    assert completed.returncode == 0, completed.stderr

    completed = subprocess.run(
        [sys.executable, str(GOLDEN_MODULE), "--entry", "xing4"],
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode != 0
    assert "is for 'torch', not 'xing4'" in completed.stderr, completed.stderr


def test_the_child_needs_somewhere_to_write_the_outcome(tmp_path) -> None:
    """Without the env var there is nowhere for the answer to go, so it stops before loading weights.

    Required rather than defaulted: a child that defaulted to stdout would print its JSON into a
    stream the entry point is also writing to, and the parent would parse whatever came first.
    """
    handed = tmp_path / "handed.json"
    handed.write_text(json.dumps(PAYLOAD), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if k != CHILD_RESULT_ENV}
    env[CHILD_FIXTURE_ENV] = str(handed)
    completed = subprocess.run(
        [sys.executable, str(GOLDEN_MODULE), "--entry", "xing4"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    assert CHILD_RESULT_ENV in completed.stderr


def test_the_child_needs_a_fixture_to_run(tmp_path) -> None:
    """Required rather than defaulted, for the reason `run_isolated` gives: a child that fell back to
    the file on disk is exactly the recorder bug this replaced."""
    env = {k: v for k, v in os.environ.items() if k != CHILD_FIXTURE_ENV}
    env[CHILD_RESULT_ENV] = str(tmp_path / "out.json")
    completed = subprocess.run(
        [sys.executable, str(GOLDEN_MODULE), "--entry", "xing4"],
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode != 0
    assert CHILD_FIXTURE_ENV in completed.stderr
    assert not (tmp_path / "out.json").exists(), "nothing should have run, so nothing was written"


def test_a_fixture_that_cannot_run_here_raises_rather_than_spawning_a_child(monkeypatch) -> None:
    """The parent checks the host first, so an unrunnable fixture costs no process and no model load.

    `run_isolated` is what the suite calls; a fixture whose checkpoint is absent must raise there --
    the test that calls it skips earlier, and this is the belt to that pair of braces.
    """
    # The override is how a host runs a fixture it keeps elsewhere, and an on-box run exports it. It
    # would rewrite this deliberately-absent path to a real checkpoint for *every* fixture, so the
    # raise would not happen -- leaving the meaning of this test depending on the shell. It asserts
    # the skip rule, so it decides its own environment.
    monkeypatch.delenv(CHECKPOINT_ENV, raising=False)
    fixture = GoldenFixture.from_json(PAYLOAD)

    with pytest.raises(RuntimeError, match="checkpoint not present"):
        run_isolated(fixture)


def test_the_child_is_given_the_repository_root_on_its_import_path(monkeypatch) -> None:
    """The suite finds `relicllm` because the root is the working directory; the child does not.

    The child is launched by path, so the interpreter puts `tests/` on `sys.path` instead -- which
    made every Python fixture fail with `ModuleNotFoundError: No module named 'relicllm'` the first
    time the runs were isolated. It goes in the environment rather than the child's `sys.path`
    because the entry points start their own ranks, and those inherit only the environment.
    """
    monkeypatch.delenv("PYTHONPATH", raising=False)
    assert _with_repo_root_on_the_path() == str(REPO_ROOT)

    # An existing path is kept, behind the root: the root has to win, or a stale install shadows it.
    monkeypatch.setenv("PYTHONPATH", "/somewhere/else")
    assert _with_repo_root_on_the_path().split(os.pathsep) == [str(REPO_ROOT), "/somewhere/else"]


def test_extra_env_reaches_the_child(monkeypatch):
    """`run_isolated(extra_env=...)` is how the two parity legs differ.

    The child is never actually spawned -- `Popen` is stubbed -- so this proves the env dict
    `run_isolated` assembles, which is the whole mechanism. `unwritable_reason` is stubbed to
    `None` because the point is the env, not the checkpoint, and no fixture on this host has one.
    """
    import inspect

    from tests import golden_fixtures

    assert "extra_env" in inspect.signature(run_isolated).parameters

    captured: dict[str, object] = {}

    class _FakeProcess:
        stdout = iter(())

        def wait(self):
            return 0

    def _fake_popen(argv, **kwargs):
        captured.update(kwargs["env"])
        return _FakeProcess()

    monkeypatch.setattr(golden_fixtures.subprocess, "Popen", _fake_popen)
    fixture = GoldenFixture.from_json(PAYLOAD)
    monkeypatch.setattr(type(fixture), "unwritable_reason", lambda self: None)

    with pytest.raises(RuntimeError):  # no result file, so run_isolated ends in its own check
        run_isolated(fixture, extra_env={"DEEPSEEK_FP8_IMPL": "torch"})

    assert captured["DEEPSEEK_FP8_IMPL"] == "torch"
    # The child-protocol keys survive an injection and are not overridable by it.
    assert captured[CHILD_RESULT_ENV]
    assert captured[CHILD_FIXTURE_ENV]
    assert captured["PYTHONPATH"] == _with_repo_root_on_the_path()


def test_the_checkpoint_can_come_from_this_host(monkeypatch):
    """`POCKETLLM_GOLDEN_CHECKPOINT` substitutes the recorded path, wherever it appears.

    Two substitutions, and both matter: the `checkpoint` field is what `unwritable_reason` consults
    before deciding to skip, and the `argv` entry is what the engine opens. A harness that moved
    only one of them would either skip on a host that has the bytes, or launch against a path that
    is not there.
    """
    from tests import golden_fixtures

    monkeypatch.setenv(golden_fixtures.CHECKPOINT_ENV, "/elsewhere/v41")
    moved = golden_fixtures.GoldenFixture.from_json(PAYLOAD)

    resolved = golden_fixtures._with_host_checkpoint(moved)

    assert resolved.checkpoint == "/elsewhere/v41"
    # The whole command line, so an implementation that rewrote *every* argv item would fail here:
    # only the `--model` value is the checkpoint, and `serve --backend xing4` has to survive.
    assert resolved.argv == (
        "serve",
        "--backend",
        "xing4",
        "--model",
        "/elsewhere/v41",
    )
    assert PAYLOAD["checkpoint"] not in resolved.argv
    assert PAYLOAD["checkpoint"] in moved.argv  # the fixture handed in is not mutated


def test_a_host_without_the_override_gets_the_recorded_path(monkeypatch):
    """Unset, the helper is the identity: the recorded path is the default, untouched.

    The object is returned as-is rather than a copy with equal fields, which is the contract the
    helper's docstring gives and what makes the ordinary case behave exactly as it did before the
    override existed.
    """
    from tests import golden_fixtures

    monkeypatch.delenv(golden_fixtures.CHECKPOINT_ENV, raising=False)
    fixture = golden_fixtures.GoldenFixture.from_json(PAYLOAD)
    resolved = golden_fixtures._with_host_checkpoint(fixture)

    assert resolved is fixture
    assert resolved.checkpoint == PAYLOAD["checkpoint"]
    assert resolved.argv == tuple(PAYLOAD["argv"])


def test_the_served_path_skip_is_decided_on_the_host_checkpoint(monkeypatch, tmp_path) -> None:
    """The skip the served-path test takes and the path `run_isolated` opens must be the same one.

    `test_served_path_golden.py` decides its own skip from `unwritable_reason` *before* it calls
    `run_isolated`, so resolving only inside `run_isolated` leaves the skip reading the recorded
    path: on the host this override exists for -- checkpoint kept elsewhere, override exported --
    every fixture would skip on a path it is not going to open, and the acceptance gate would pass
    vacuously. That is the failure the override closes, so it is pinned here against the production
    helper rather than a copy of it.
    """
    from tests import golden_fixtures
    from tests.test_served_path_golden import _served_skip_reason

    # A checkpoint that exists here, standing in for the bytes the recording host kept at `/mnt`.
    monkeypatch.setenv(golden_fixtures.CHECKPOINT_ENV, str(tmp_path))

    # The recorded checkpoint is deliberately absent, so a skip decided on it is a skip; the host
    # checkpoint is a real directory, so the same decision on the resolved fixture is not. That
    # difference is the whole point of resolving before deciding.
    reason = _served_skip_reason(golden_fixtures.GoldenFixture.from_json(PAYLOAD))
    assert reason is None, (
        "the served-path skip is still reading the recorded path; resolve the fixture with "
        "_with_host_checkpoint before deciding to skip"
    )
