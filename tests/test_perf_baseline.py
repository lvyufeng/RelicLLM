"""The performance gate itself: what it calls a regression, and whether its own baseline is sane.

``scripts/check_perf_baseline.py`` is the instrument P0.2 exists for, and the mistake it is written
against is a gate that is green because it compared nothing. These tests drive ``observations``,
``compare`` and ``_runtime_of`` directly, on records short enough to read, and pin the four
behaviours the design turns on:

- **both directions are reported**, and a metric is a regression only if it moved the way *that*
  metric does not want -- a larger ``output_throughput`` is not a regression, a larger ``e2el`` is;
- **a key the baseline holds and the run did not produce is reported, not passed** -- a skip has to
  look different from a pass, which is the same rule ``scripts/check_test_baseline.py`` states for a
  skip;
- **the runtime and the card come from the record**, from the launch argv and the ``cuda`` block, so
  neither a renamed file nor a second box changes what a number is about;
- **``None`` and ``0.0`` are different answers**: a metric a run did not report is not a claim it was
  zero.

The card tests are what keeps an 8x RTX 4090 reading from being diffed against a 4x RTX 2080 Ti
baseline and reported as a 2.6x regression, which is what the keys did before they carried the card.
A record with no ``cuda`` block, and a baseline entry with no card, keep the comparison they have
always had.

The committed-file tests read the baseline itself, because a baseline nothing reads is a file that can
be replaced by a wrong one without anything going red, which is the failure mode P0.2 is about.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CHECKER = REPO_ROOT / "scripts" / "check_perf_baseline.py"
BASELINE = REPO_ROOT / "tests" / "fixtures" / "perf" / "baseline.json"
SM75_RECORD = REPO_ROOT / "tests" / "fixtures" / "perf" / "v41_decode_graph_on.json"

#: The device names the two boxes report, as ``relicllm bench serve`` spells them in ``cuda.devices``.
CARD_SM75 = "NVIDIA GeForce RTX 2080 Ti"
CARD_SM89 = "NVIDIA GeForce RTX 4090"


def _load_checker():
    """Import scripts/check_perf_baseline.py by path; scripts/ is not a package."""
    spec = importlib.util.spec_from_file_location("check_perf_baseline_under_test", CHECKER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _record(runtime: str = "v41", card: str | None = None, **metrics) -> dict:
    """A bench record with one scenario, in the shape ``relicllm bench serve`` writes.

    ``card`` is the device name the envelope's probe reports; ``None`` writes no ``cuda`` block at
    all, which is what a box with neither ``torch`` nor ``nvidia-smi`` produces.
    """
    record = {
        "launch": {"argv": ["python", "-m", "relicllm", "serve", "--backend", runtime]},
        "scenarios": {"decode": {"metrics": {"tpot": {"median": 100.0}, **metrics}}},
    }
    if card is not None:
        record["cuda"] = {
            "count": 4,
            "devices": [card] * 4,
            "torch": "2.9.1+cu128",
            "cuda_runtime": "12.8",
        }
    return record


def _write(path, record: dict):
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_a_metric_past_the_threshold_is_reported_in_the_direction_it_moved() -> None:
    baseline = {"v41/decode/tpot.median": 100.0, "v41/decode/output_throughput": 2.0}

    worse = checker.compare(
        baseline, {"v41/decode/tpot.median": 130.0, "v41/decode/output_throughput": 2.0}
    )
    assert [delta.key for delta in worse.regressions] == ["v41/decode/tpot.median"]
    assert not worse.ok

    better = checker.compare(
        baseline, {"v41/decode/tpot.median": 130.0, "v41/decode/output_throughput": 3.0}
    )
    # The same ratio moves one metric the wrong way and the other the right way, which is why the
    # direction is a property of the metric and not of the delta.
    assert [delta.key for delta in better.regressions] == ["v41/decode/tpot.median"]
    assert [delta.key for delta in better.improvements] == ["v41/decode/output_throughput"]


def test_a_larger_throughput_is_not_a_regression_and_a_smaller_one_is() -> None:
    baseline = {"v41/decode/output_throughput": 2.0}

    up = checker.compare(baseline, {"v41/decode/output_throughput": 3.0})
    assert up.ok and len(up.improvements) == 1

    down = checker.compare(baseline, {"v41/decode/output_throughput": 1.0})
    assert not down.ok and len(down.regressions) == 1


def test_a_metric_inside_the_threshold_is_unchanged() -> None:
    baseline = {"v41/decode/tpot.median": 100.0}

    # 110/100 is 10%, under the 15% default; two identical runs are never identical, and a gate that
    # asked for equality would be red on every run.
    assert checker.compare(baseline, {"v41/decode/tpot.median": 110.0}).unchanged


def test_a_key_the_run_did_not_produce_is_reported_and_does_not_fail_on_its_own() -> None:
    baseline = {"v41/decode/tpot.median": 100.0, "mimo/decode/tpot.median": 100.0}

    report = checker.compare(baseline, {"v41/decode/tpot.median": 100.0})

    assert report.missing == ("mimo/decode/tpot.median",)
    # A runtime skipped for want of a checkpoint is ordinary; "the gate is green" and "the gate
    # compared nothing" being distinguishable is what the missing list is for.
    assert report.ok


def test_a_run_that_measured_nothing_is_not_a_pass(tmp_path) -> None:
    """The failure mode the whole script is written against: green because it compared nothing."""
    empty = tmp_path / "empty.json"
    target = tmp_path / "baseline.json"
    empty.write_text(json.dumps({"launch": {"argv": []}, "scenarios": {}}), encoding="utf-8")

    # No metric read means nothing to record, and recording nothing would leave the next run with an
    # empty baseline that reports every key as compared and every comparison as green.
    assert checker.observations(json.loads(empty.read_text())) == {}
    with pytest.raises(SystemExit, match="carries no comparable metric"):
        checker.main(["--update", "--observed", str(empty), "--baseline", str(target)])
    assert not target.exists()


def test_the_runtime_is_read_off_the_recorded_argv_in_both_spellings() -> None:
    assert checker._runtime_of({"launch": {"argv": ["x", "--backend", "mimo"]}}) == "mimo"
    assert checker._runtime_of({"launch": {"argv": ["x", "--backend=xing4"]}}) == "xing4"
    # A run against an already-running server has no --backend to read, and is not guessed at.
    assert checker._runtime_of({"launch": {"argv": ["x", "--model", "/ckpt"]}}) == "unknown"


def test_the_launch_shape_is_recorded_per_runtime_and_reads_both_spellings() -> None:
    """The width a number was measured at travels with it; it cannot be read off the metric key.

    ``tpot.median`` is the same key for a 4-card aggregate and a single-card reading, so a baseline
    that recorded only the value would compare the two as if they were one measurement.
    """
    four = {
        "launch": {"argv": ["x", "--backend", "v41", "--tensor-parallel-size", "4", "--max-model-len", "32768"]},
        "scenarios": {},
    }
    one = {"launch": {"argv": ["x", "--backend=xing4", "--tensor-parallel-size=1"]}, "scenarios": {}}

    assert checker.shapes_of([four, one]) == {
        "v41": "tensor-parallel-size=4 max-model-len=32768",
        "xing4": "tensor-parallel-size=1",
    }


def test_the_card_a_record_was_measured_on_reaches_the_key() -> None:
    """The key says which box the number belongs to; the shape block is keyed the same way.

    Without the card, an 8x RTX 4090 reading and a 4x RTX 2080 Ti reading are both
    ``v41/decode/tpot.median``, which is what made the gate report the first as the second moved.
    """
    on_a_2080 = _record(card=CARD_SM75)
    assert checker.observations(on_a_2080) == {f"v41@{CARD_SM75}/decode/tpot.median": 100.0}
    assert checker.shapes_of([on_a_2080]) == {f"v41@{CARD_SM75}": "unknown"}

    # Absent is absent: no cuda block keeps the unqualified key this script has always written, and so
    # does a cuda block that named no device.
    assert checker._card_of({"cuda": {"count": 8, "torch": "2.9.1+cu128"}}) is None
    assert checker._card_of({"cuda": {"devices": []}}) is None
    assert set(checker.observations(_record())) == {"v41/decode/tpot.median"}

    # A box that is not uniform names both devices rather than half of them.
    mixed = {"cuda": {"devices": [CARD_SM89, CARD_SM75, CARD_SM89]}}
    assert checker._card_of(mixed) == f"{CARD_SM89} + {CARD_SM75}"


def test_an_sm_89_record_does_not_compare_against_an_sm_75_key(tmp_path, capsys) -> None:
    """The defect this exists for: TPOT 1.0891 s on a 4090 against a 2080 Ti record's 0.4129 s.

    Card-blind, that is a 2.6x regression and every cell of the run reads the same way. With the card
    in the key the two never meet, and the run is reported as one that could not be compared rather
    than as a regression -- the same answer the gate gives a runtime the baseline does not cover.
    """
    sm75 = _record(card=CARD_SM75, tpot={"median": 0.4129})
    sm89 = _record(card=CARD_SM89, tpot={"median": 1.0891})
    baseline = checker.observations(sm75)

    card_blind = checker.compare(
        {"v41/decode/tpot.median": 0.4129}, {"v41/decode/tpot.median": 1.0891}
    )
    assert [delta.key for delta in card_blind.regressions] == ["v41/decode/tpot.median"]

    report = checker.compare(baseline, checker.observations(sm89))
    assert report.regressions == ()
    assert report.missing == (f"v41@{CARD_SM75}/decode/tpot.median",)

    target = _write(tmp_path / "baseline.json", {"metrics": baseline})
    measured = _write(tmp_path / "sm89.json", sm89)
    assert checker.main(["--observed", str(measured), "--baseline", str(target)]) == 1
    out = capsys.readouterr().out
    assert "REGRESSIONS" not in out
    assert CARD_SM89 in out and CARD_SM75 in out
    assert "nothing was compared" in out


def test_two_cards_of_one_runtime_are_recorded_side_by_side(tmp_path) -> None:
    """A second card's numbers are keys of their own, so one file holds both boxes.

    This is the shape the sm_89 work needs: the 4090 record is added beside the 2080 Ti one, and each
    record then compares against its own numbers and neither moves the other's metric.
    """
    sm75 = _write(tmp_path / "sm75.json", _record(card=CARD_SM75, tpot={"median": 0.4129}))
    sm89 = _write(tmp_path / "sm89.json", _record(card=CARD_SM89, tpot={"median": 1.0891}))
    target = tmp_path / "baseline.json"

    assert checker.main(
        ["--update", "--observed", str(sm75), "--observed", str(sm89), "--baseline", str(target)]
    ) == 0
    document = json.loads(target.read_text(encoding="utf-8"))
    assert set(document["metrics"]) == {
        f"v41@{CARD_SM75}/decode/tpot.median",
        f"v41@{CARD_SM89}/decode/tpot.median",
    }
    assert set(document["runs"]) == {f"v41@{CARD_SM75}", f"v41@{CARD_SM89}"}

    for record in (sm75, sm89):
        assert checker.main(["--observed", str(record), "--baseline", str(target)]) == 0


def test_a_baseline_entry_that_names_no_card_keeps_the_comparison_it_has_always_had() -> None:
    """A baseline recorded before the card was read, or a run from a box that could not read one.

    Neither side gains a failure from the field being absent, which is the hard requirement: a card on
    one side only is not a mismatch, and the two keys are compared exactly as they were before the
    card existed.
    """
    on_a_4090 = checker.observations(_record(card=CARD_SM89))
    assert checker.compare({"v41/decode/tpot.median": 100.0}, on_a_4090).unchanged

    unattributed = checker.observations(_record())
    assert checker.compare({f"v41@{CARD_SM75}/decode/tpot.median": 100.0}, unattributed).unchanged


def test_update_says_which_card_it_no_longer_holds(tmp_path, capsys) -> None:
    """``--update`` replaces the file, and with a second card in it that is worth saying out loud."""
    target = tmp_path / "baseline.json"
    sm75 = _write(tmp_path / "sm75.json", _record(card=CARD_SM75))
    sm89 = _write(tmp_path / "sm89.json", _record(card=CARD_SM89))
    assert checker.main(["--update", "--observed", str(sm75), "--baseline", str(target)]) == 0

    assert checker.main(["--update", "--observed", str(sm89), "--baseline", str(target)]) == 0
    out = capsys.readouterr().out
    assert f"v41@{CARD_SM75}" in out
    assert set(json.loads(target.read_text(encoding="utf-8"))["metrics"]) == {
        f"v41@{CARD_SM89}/decode/tpot.median"
    }


def test_a_record_for_a_runtime_the_baseline_does_not_cover_is_skipped_not_compared(tmp_path, capsys) -> None:
    """Only a runtime the baseline covers may move a metric, and a run that compared nothing is not
    a pass: a record against a server somebody else started carries no ``--backend`` to read, so it
    is reported as skipped and the gate exits non-zero rather than green."""
    target = tmp_path / "baseline.json"
    target.write_text(json.dumps({"metrics": {"v41/decode/tpot.median": 100.0}}), encoding="utf-8")
    unattributed = tmp_path / "unattributed.json"
    unattributed.write_text(
        json.dumps({"launch": None, "scenarios": {"decode": {"metrics": {"tpot": {"median": 1.0}}}}}),
        encoding="utf-8",
    )

    assert checker.main(["--observed", str(unattributed), "--baseline", str(target)]) == 1
    out = capsys.readouterr().out
    assert "skipped" in out
    assert "nothing was compared" in out


def test_two_records_are_merged_into_one_baseline(tmp_path) -> None:
    """``--observed`` is repeatable because the baseline covers several runtimes and a run has one.

    A single-record ``--update`` could not regenerate the committed file at all: it would hold one
    runtime and the next comparison would report the others as "not measured this run".
    """
    target = tmp_path / "baseline.json"
    first = _record("v41", e2el={"median": 11.0})
    second = _record("mimo", e2el={"median": 22.0})
    paths = []
    for index, record in enumerate((first, second)):
        path = tmp_path / f"run{index}.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        paths += ["--observed", str(path)]

    assert checker.main(["--update", *paths, "--baseline", str(target)]) == 0
    document = json.loads(target.read_text(encoding="utf-8"))

    assert "v41/decode/e2el.median" in document["metrics"]
    assert "mimo/decode/e2el.median" in document["metrics"]
    assert set(document["runs"]) == {"v41", "mimo"}


def test_an_absent_metric_is_not_a_measured_zero() -> None:
    """``None`` is "not reported"; ``0.0`` is a claim the server served nothing."""
    record = {"launch": {"argv": ["x"]}, "scenarios": {"decode": {"metrics": {"output_throughput": None}}}}

    assert checker.observations(record) == {}
    assert checker.observations(
        {"launch": {"argv": ["x"]}, "scenarios": {"decode": {"metrics": {"output_throughput": 0.0}}}}
    ) == {"unknown/decode/output_throughput": 0.0}


@pytest.mark.skipif(not BASELINE.exists(), reason="no committed performance baseline")
def test_the_committed_baseline_has_the_shape_the_gate_reads() -> None:
    """The file P0.2 commits is the file the gate can read -- and it covers what it claims to.

    The header travelled into the JSON because a baseline whose commit and box live in a docstring go
    stale independently of the numbers; this checks the envelope, and that the set of runtimes the
    gate would find a comparison for is not empty. It also checks that every key names the card its
    numbers came from: a file that names none is a file that diffs a 4090 against a 2080 Ti again.
    It does **not** check the values, which only a run on this host can judge.
    """
    document = json.loads(BASELINE.read_text(encoding="utf-8"))

    assert 0.0 < float(document["threshold"]) < 1.0
    assert document["note"]
    metrics = document["metrics"]
    assert metrics, "a baseline with no metrics compares nothing"
    # The numbers are not comparable without the width they were measured at, and this runtime set
    # does not share one -- xing4 serves a single card while the rest run TP4 here. The shape is
    # recorded rather than inferred because `tpot.median` is one key at either width. The block is
    # keyed the way the metrics are, so one entry is one runtime on one card.
    runs = document["runs"]
    assert set(runs) == {checker._token_of_key(key) for key in metrics}
    assert any("tensor-parallel-size=1" in shape for shape in runs.values())
    assert any("tensor-parallel-size=4" in shape for shape in runs.values())

    # Every key parses into the quadruple the comparison is keyed by, and every runtime named is one
    # `RUNTIMES` declares -- a typo in a key is a key the gate would silently never compare.
    from relicllm.backends.capabilities import RUNTIMES

    runtimes = set()
    for key in metrics:
        token, scenario, metric = checker._key_parts(key)
        runtime, card = checker._split_token(token)
        assert card, f"{key} names no card"
        assert metric in checker._METRICS, metric
        assert scenario, "the baseline names no scenario"
        runtimes.add(runtime)
    assert runtimes <= set(RUNTIMES), runtimes - set(RUNTIMES)


@pytest.mark.skipif(
    not (BASELINE.exists() and SM75_RECORD.exists()),
    reason="no committed baseline or no committed sm_75 record",
)
def test_an_sm_75_record_still_compares_against_the_committed_baseline(capsys) -> None:
    """The card the baseline records is the card its numbers came from, so nothing changed here.

    ``v41_decode_graph_on.json`` is a 4x RTX 2080 Ti record, measured on the box the baseline was
    measured on: every ``v41`` key it produces is a key the baseline holds, and the gate exits 0 with
    no regression line. The runtimes this record does not cover are reported as not measured, which is
    not a failure -- the same answer the file gave before the card was in the key.
    """
    record = json.loads(SM75_RECORD.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    observed = checker.observations(record)

    assert checker._card_of(record) == CARD_SM75
    v41_keys = {key for key in baseline["metrics"] if key.startswith("v41@")}
    assert v41_keys and v41_keys <= set(observed)

    assert checker.main(["--observed", str(SM75_RECORD), "--baseline", str(BASELINE)]) == 0
    out = capsys.readouterr().out
    assert "REGRESSIONS" not in out
    assert "skipped" not in out

def test_a_baseline_of_zero_is_reported_rather_than_divided_by() -> None:
    """A zero is a claim the server served nothing, and a ratio off it is not a comparison."""
    report = checker.compare({"v41/decode/output_throughput": 0.0}, {"v41/decode/output_throughput": 3.0})

    assert report.ok
    assert report.unenforced[0].key == "v41/decode/output_throughput"
    assert report.unenforced[0].ratio == float("inf")
