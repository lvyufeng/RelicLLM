"""The first-step logits record: the summary the backend captures and the comparison the
tolerance leg runs. Both are pure functions over CPU tensors, so they need no card."""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from relicllm.backends.v41_backend import _logits_check


def test_the_summary_is_the_top_k_positions_sorted_descending():
    row = torch.tensor([0.5, 3.0, -1.0, 2.0, 9.0, 1.0], dtype=torch.float32)
    record = _logits_check(row, top_k=3)
    assert record["step"] == 0
    assert record["top_k"] == 3
    # ids of the three largest, in descending value order: 9.0(idx4), 3.0(idx1), 2.0(idx3)
    assert record["token_ids"] == [4, 1, 3]
    assert record["values"] == [9.0, 3.0, 2.0]


def test_the_ids_and_values_are_positionally_paired_and_the_same_length():
    row = torch.randn(64)
    record = _logits_check(row, top_k=8)
    assert len(record["token_ids"]) == len(record["values"]) == 8
    # The k-th id names the position whose logit is the k-th value.
    for token_id, value in zip(record["token_ids"], record["values"]):
        assert torch.isclose(row[token_id], torch.tensor(value), atol=0, rtol=0)


def test_a_top_k_larger_than_the_row_is_clamped_not_an_error():
    row = torch.tensor([1.0, 2.0])
    record = _logits_check(row, top_k=8)
    assert record["top_k"] == 2
    assert record["token_ids"] == [1, 0]


def test_the_record_is_json_native_so_it_can_leave_the_child():
    import json

    row = torch.randn(16)
    record = _logits_check(row, top_k=4)
    assert json.loads(json.dumps(record)) == record


def test_the_capture_is_off_unless_the_env_knob_is_set(monkeypatch):
    from relicllm.backends.v41_backend import _logits_check_top_k

    monkeypatch.delenv("POCKETLLM_V41_LOGITS_CHECK", raising=False)
    assert _logits_check_top_k() == 0
    monkeypatch.setenv("POCKETLLM_V41_LOGITS_CHECK", "8")
    assert _logits_check_top_k() == 8


def test_a_bad_env_value_is_off_rather_than_a_crash(monkeypatch):
    from relicllm.backends.v41_backend import _logits_check_top_k

    monkeypatch.setenv("POCKETLLM_V41_LOGITS_CHECK", "yes")
    assert _logits_check_top_k() == 0


def test_a_record_within_tolerance_matches():
    from tests.golden_fixtures import logits_check_mismatch

    recorded = {"step": 0, "top_k": 2, "token_ids": [5, 6], "values": [1.0, 0.5],
                "atol": 1e-3, "rtol": 1e-5}
    observed = {"step": 0, "top_k": 2, "token_ids": [5, 6], "values": [1.0009, 0.5]}
    assert logits_check_mismatch(observed, recorded) is None


def test_a_record_outside_tolerance_reports_the_gap():
    from tests.golden_fixtures import logits_check_mismatch

    recorded = {"step": 0, "top_k": 2, "token_ids": [5, 6], "values": [1.0, 0.5],
                "atol": 1e-3, "rtol": 1e-5}
    observed = {"step": 0, "top_k": 2, "token_ids": [5, 6], "values": [1.5, 0.5]}
    message = logits_check_mismatch(observed, recorded)
    assert message is not None
    assert "0.5" in message or "1.5" in message


def test_different_argmax_ids_are_reported_as_an_ordering_divergence():
    from tests.golden_fixtures import logits_check_mismatch

    recorded = {"step": 0, "top_k": 2, "token_ids": [5, 6], "values": [1.0, 0.5],
                "atol": 1e-3, "rtol": 1e-5}
    observed = {"step": 0, "top_k": 2, "token_ids": [6, 5], "values": [1.0, 0.5]}
    message = logits_check_mismatch(observed, recorded)
    assert message is not None
    assert "argmax" in message or "order" in message


def test_a_shorter_observed_record_is_refused_rather_than_compared_by_prefix():
    """`zip` truncates, so two unequal lists would compare as agreeing on the shorter one.

    The schema refuses a recorded record whose `token_ids` and `values` disagree in length, but
    `observed` comes from a live run and nothing checks it -- so the comparison helper is the only
    place this can be caught.
    """
    from tests.golden_fixtures import logits_check_mismatch

    recorded = {"step": 0, "top_k": 3, "token_ids": [5, 6, 7], "values": [1.0, 0.5, 0.25],
                "atol": 1e-3, "rtol": 1e-5}
    observed = {"step": 0, "top_k": 3, "token_ids": [5, 6, 7], "values": [1.0, 0.5]}

    message = logits_check_mismatch(observed, recorded)
    assert message is not None
    assert "different number of logits" in message
