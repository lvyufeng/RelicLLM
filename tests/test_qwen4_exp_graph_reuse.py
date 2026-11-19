"""CPU invariants for rebasing one recording onto successive accepted windows."""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from test_qwen4_exp_graph_lifecycle import (
    _Model, _accepted, _cache, _cpu_capture, _holder, _states,
)


@pytest.fixture(autouse=True)
def _reuse(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_GRAPH_REUSE", "1")


def _step(holder, history, position, token):
    holder.fill_history(history + [token])
    selected = holder.replay(position, torch.tensor([[token]], dtype=torch.long, device="cpu"))
    history.append(token)
    return selected


def _prefix(holder):
    history = [11, 12]
    _step(holder, history, 3, 3)
    _step(holder, history, 4, 4)
    return history, [value.clone() for value in _accepted(holder, holder.cache)]


def test_reused_windows_match_eager_and_keep_capture_addresses():
    model, cache = _Model(), _cache()
    reference_model, reference = _Model(), _cache()
    holder = _holder(model, cache, steps=2)
    recording = holder.graph
    addresses = [holder.inputs.data_ptr(), holder.outputs.data_ptr(), holder.history.data_ptr()]
    stacks = [layout[name].stack.data_ptr() for layout, (_, fields) in zip(holder._layouts, holder._bindings)
              for name in fields]
    history = [11, 12]
    for position, token in enumerate((3, 4, 0, 5, 1, 2, 7, 9), start=3):
        expected = reference_model.forward(torch.tensor([[token]], device="cpu"), cache=reference,
                                           past_len=position)
        assert _step(holder, history, position, token) is not None
        torch.testing.assert_close(holder.outputs, expected[:, -1])
        for actual, wanted in zip(_accepted(holder, cache), _states(reference)):
            torch.testing.assert_close(actual, wanted)
        assert holder.graph is recording
        assert addresses == [holder.inputs.data_ptr(), holder.outputs.data_ptr(), holder.history.data_ptr()]
        assert stacks == [layout[name].stack.data_ptr() for layout, (_, fields) in zip(holder._layouts, holder._bindings)
                          for name in fields]
    assert recording.calls == 8 and holder.rebase_count == 3
    assert len(model.setup_reads) == 2
    holder.release()
    for actual, wanted in zip(_states(cache), _states(reference)):
        torch.testing.assert_close(actual, wanted)
    assert cache.length == reference.length == 11
    torch.testing.assert_close(cache.ple_token_history, reference.ple_token_history)


def test_release_immediately_after_rebase_restores_slot_zero_not_original_prefix():
    holder = _holder(steps=2)
    history, accepted = _prefix(holder)
    assert holder._rebase_window(5)
    assert holder._committed_slot == 0 and holder._committed_any
    holder.release()
    assert holder.cache.length == 5
    for value, wanted in zip(_states(holder.cache), accepted):
        torch.testing.assert_close(value, wanted)
    torch.testing.assert_close(holder.cache.ple_token_history, torch.tensor([history[-2:]]))


def test_first_rebased_handback_restores_prior_window_not_unaccepted_slot():
    holder = _holder(steps=2)
    history, accepted = _prefix(holder)
    holder._should_handback = lambda: True
    assert _step(holder, history, 5, 9) is None
    assert holder.rebase_count == 1 and holder.cache.length == 5
    for value, wanted in zip(_states(holder.cache), accepted):
        torch.testing.assert_close(value, wanted)
    torch.testing.assert_close(holder.cache.ple_token_history, torch.tensor([[3, 4]]))


def test_partial_rebase_copy_failure_restores_untouched_accepted_source(monkeypatch):
    holder = _holder(steps=2)
    history, accepted = _prefix(holder)

    def fail():
        entry = holder._layouts[0]["conv_state"]
        entry.stack[0].copy_(entry.stack[holder._committed_slot])
        raise RuntimeError("partial rebase copy")

    monkeypatch.setattr(holder, "_seed_rebased_state", fail)
    with pytest.raises(RuntimeError, match="partial rebase copy"):
        _step(holder, history, 5, 9)
    assert holder._released and holder.cache.length == 5
    for value, wanted in zip(_states(holder.cache), accepted):
        torch.testing.assert_close(value, wanted)


def test_reuse_stops_at_actual_arena_capacity():
    holder = _holder(steps=2)
    history = [11, 12]
    for position in range(3, 12):
        assert _step(holder, history, position, 3) is not None
    accepted = [value.clone() for value in _accepted(holder, holder.cache)]
    assert _step(holder, history, 12, 3) is None
    assert holder.cache.length == 12 and holder.rebase_count == 4
    for value, wanted in zip(_states(holder.cache), accepted):
        torch.testing.assert_close(value, wanted)


def test_skipped_boundary_is_not_rebased():
    holder = _holder(steps=2)
    history, accepted = _prefix(holder)
    assert _step(holder, history, 6, 3) is None
    assert holder.rebase_count == 0 and holder.cache.length == 5
    for value, wanted in zip(_states(holder.cache), accepted):
        torch.testing.assert_close(value, wanted)


def test_disabled_reuse_retains_original_exhaustion_policy(monkeypatch):
    monkeypatch.delenv("POCKETLLM_QWEN4_GRAPH_REUSE")
    holder = _holder(steps=2)
    history, accepted = _prefix(holder)
    assert _step(holder, history, 5, 3) is None
    assert not holder.reuse_windows and holder.rebase_count == 0
    for value, wanted in zip(_states(holder.cache), accepted):
        torch.testing.assert_close(value, wanted)
