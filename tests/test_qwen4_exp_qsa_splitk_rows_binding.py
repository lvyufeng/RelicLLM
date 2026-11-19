"""The small-query split-K binding is explicit and retains the serial exactness fallback."""
from types import SimpleNamespace

import pytest
import torch

from relicllm.models.qwen4_exp.attention import _qsa_forward_rows


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 9])
@pytest.mark.parametrize("new_available", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_new_entry_uses_one_call_only_in_approved_small_verify_dispatch(monkeypatch, rows, new_available, enabled):
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_SPLITK", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_SPLITK_ROWS", "1" if enabled else "0")
    q = torch.randn(1, rows, 3, 256, dtype=torch.bfloat16)
    k = torch.zeros(1, 1, 16, 256, dtype=torch.bfloat16)
    v = torch.ones_like(k)
    selected = torch.zeros(1, rows, 2051, dtype=torch.int32)
    calls = []

    def invoke(name):
        def operation(query, key, value, selection, scale):
            assert key is k and value is v and scale == 0.0625
            assert query.is_contiguous() and selection.is_contiguous()
            calls.append((name, query.shape[1]))
            return query + 1
        return operation

    ops = SimpleNamespace(qwen4_exp_qsa_bf16_forward=invoke("general"),
                          qwen4_exp_qsa_splitk_bf16_forward=invoke("one"))
    if new_available:
        ops.qwen4_exp_qsa_splitk_rows_bf16_forward = invoke("rows")
    out = _qsa_forward_rows(ops, q, k, v, selected, 0.0625)
    assert torch.equal(out, q + 1)
    active = 2 <= rows <= 8
    if active and enabled and new_available:
        assert calls == [("rows", rows)]
    elif active:
        assert calls == [("one", 1)] * rows
    elif rows == 1:
        assert calls == [("one", 1)]
    else:
        assert calls == [("general", rows)]


def test_splitk_disabled_or_unsupported_selection_keeps_general_per_row(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_SPLITK_ROWS", "1")
    calls = []

    def general(q, k, v, selected, scale):
        calls.append(q.shape[1])
        return q

    def forbidden(*_args):
        raise AssertionError("unapproved new split-K entry was dispatched")

    ops = SimpleNamespace(qwen4_exp_qsa_bf16_forward=general,
                          qwen4_exp_qsa_splitk_bf16_forward=forbidden,
                          qwen4_exp_qsa_splitk_rows_bf16_forward=forbidden)
    q = torch.zeros(1, 4, 3, 256, dtype=torch.bfloat16)
    kv = torch.zeros(1, 1, 16, 256, dtype=torch.bfloat16)
    for flag, width in [("0", 2051), ("1", 4097)]:
        monkeypatch.setenv("POCKETLLM_QWEN4_QSA_SPLITK", flag)
        calls.clear()
        selected = torch.zeros(1, 4, width, dtype=torch.int32)
        assert torch.equal(_qsa_forward_rows(ops, q, kv, kv, selected, 0.1), q)
        assert calls == [1, 1, 1, 1]
