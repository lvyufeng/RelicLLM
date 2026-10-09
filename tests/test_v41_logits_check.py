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


def test_an_outcome_carries_a_logits_record_through_json():
    import json

    from tests.golden_fixtures import Outcome

    outcome = Outcome(
        token_ids=[1, 2],
        text="hi",
        prompt_tokens=3,
        logits_check={"step": 0, "prompt_tokens": 3, "top_k": 2,
                      "token_ids": [5, 6], "values": [1.5, 0.5]},
    )
    back = Outcome.from_json(json.loads(json.dumps(outcome.to_json())))
    assert back == outcome
    assert back.logits_check == {"step": 0, "prompt_tokens": 3, "top_k": 2,
                                 "token_ids": [5, 6], "values": [1.5, 0.5]}


def test_an_outcome_without_a_record_round_trips_as_none():
    import json

    from tests.golden_fixtures import Outcome

    outcome = Outcome(token_ids=[1], text="x", prompt_tokens=1)
    back = Outcome.from_json(json.loads(json.dumps(outcome.to_json())))
    assert back == outcome
    assert back.logits_check is None