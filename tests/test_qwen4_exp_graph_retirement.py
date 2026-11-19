"""Metadata-only retirement of a captured graph's cache bindings; no CUDA work is asserted here.

The native-boundary handoff releases the K-row verifier and the draft chain while the one-row tail
recording still replays over the same rolling-state buffers. `retire_bindings()` is the half of
teardown the eager trunk needs (the cache stops pointing at the recording) without the half that
would free graph pools, rewind logs and pinned staging out from under an in-flight replay. These
CPU fakes pin that split; the device timing and full-token parity remain a separate GPU measurement.
"""
from types import SimpleNamespace

import pytest
import torch

from relicllm.models.qwen4_exp import verify_graph as graphs
from relicllm.models.qwen4_exp.verify_graph import DraftGraph, VerifyGraph, _MISSING


class _CountingTarget:
    """A remembered target that records every attribute write and delete."""

    def __init__(self, **values):
        self.__dict__.update(values)
        self.writes = []

    def __setattr__(self, name, value):
        if name != "writes":
            self.__dict__.setdefault("writes", []).append(("set", name, value))
        object.__setattr__(self, name, value)

    def __delattr__(self, name):
        self.__dict__["writes"].append(("del", name, None))
        object.__delattr__(self, name)


def _verify_holder(*, retired=False, device="cpu"):
    layer = SimpleNamespace(state_log=[torch.zeros(1)], kv_width=64)
    state = SimpleNamespace(state_log=[torch.zeros(1)])
    cache = SimpleNamespace(layers=[layer], ple_states={2: state})
    table = SimpleNamespace(_rows_buffer=torch.ones(3))
    original = table._rows_buffer
    private = torch.zeros(12)
    ple = SimpleNamespace(ngram_table=table)
    holder = VerifyGraph.__new__(VerifyGraph)
    holder.device = torch.device(device)
    holder.cache = cache
    holder.model = SimpleNamespace(config=SimpleNamespace(ngram_size=3))
    holder.graph = object()
    holder._rewinds = {1: object(), 2: object()}
    holder._gdn_logs = [(layer, list(layer.state_log))]
    holder._ple_logs = [(state, list(state.state_log))]
    holder._ple_rows = [(ple, table, private, (1, 4, 8))]
    holder._stager = object()
    holder._saved_rows_buffers = [(table, original)]
    holder._logs_open = True
    holder._bindings_retired = retired
    holder._released = False
    holder.replays = 0
    target = _CountingTarget(kv_width=64)
    holder._target = target
    holder._saved_attrs = [(cache, "kv_width", _MISSING), (target, "kv_width", 64)]
    cache.kv_width = 2
    target.kv_width = 2
    return holder, cache, target, table, original, private, layer, state


def test_retire_restores_bindings_but_keeps_every_recording_owned_buffer(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    holder, cache, target, table, original, private, layer, state = _verify_holder()
    graph, rewinds, gdn, ple, rows, stager = (
        holder.graph, holder._rewinds, holder._gdn_logs, holder._ple_logs, holder._ple_rows,
        holder._stager,
    )

    holder.retire_bindings()

    # The cache no longer points at the recording's private staging: eager decode is safe again.
    assert table._rows_buffer is original
    assert not hasattr(cache, "kv_width") and target.kv_width == 64
    assert layer.state_log is None and state.state_log is None
    assert not holder._logs_open
    # Everything the recording still replays through stays referenced by the object.
    assert holder.graph is graph
    assert holder._rewinds is rewinds and holder._gdn_logs is gdn and holder._ple_logs is ple
    assert holder._ple_rows is rows and rows[0][2] is private
    assert holder._stager is stager
    assert holder._saved_rows_buffers == [] and holder._saved_attrs == []
    assert not calls, "retirement must not flush the caching allocator"


def test_retire_is_idempotent_and_never_rewrites_a_restored_attribute(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    holder, cache, target, _table, _original, _private, _layer, _state = _verify_holder()
    target.writes.clear()
    holder.retire_bindings()
    first = list(target.writes)
    cache.kv_width = 9
    target.writes.clear()

    holder.retire_bindings()

    assert target.writes == [], "a second retirement must not restore old values over new ones"
    assert first == [("set", "kv_width", 64)], "the first retirement restores once"
    assert cache.kv_width == 9 and target.kv_width == 64


def test_reverse_order_leaves_the_value_held_before_the_first_remember():
    """A target remembered twice must end at the original, not the intermediate installation."""
    holder = VerifyGraph.__new__(VerifyGraph)
    holder.device = torch.device("cpu")
    holder.cache = SimpleNamespace(layers=[], ple_states={})
    holder._logs_open = False
    holder._saved_rows_buffers = []
    target = _CountingTarget(flag="original")
    holder._saved_attrs = [(target, "flag", "original"), (target, "flag", "middle")]
    target.flag = "installed"
    target.writes.clear()

    holder.retire_bindings()

    assert target.flag == "original"
    assert [value for kind, name, value in target.writes if kind == "set"] == ["middle", "original"]


@pytest.mark.parametrize("kept", [1, 2, 4])
def test_retired_bindings_refuse_replay_and_rewind_before_touching_staged_buffers(kept):
    holder, *_rest = _verify_holder(retired=True)
    with pytest.raises(RuntimeError, match="bindings were retired"):
        holder.replay(8, [1, 2, 3, 4])
    with pytest.raises(RuntimeError, match="bindings were retired"):
        holder.rewind(8, kept, [1, 2, 3, 4])


def test_release_after_retirement_clears_state_once_without_rebinding(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))
    holder, cache, target, table, original, _private, _layer, _state = _verify_holder()
    holder.retire_bindings()
    cache.kv_width = 7
    target.kv_width = 7

    holder.release()

    assert holder.graph is None and holder._rewinds == {} and holder._stager is None
    assert holder._gdn_logs == [] and holder._ple_logs == [] and holder._ple_rows == []
    assert holder._released
    assert cache.kv_width == 7 and target.kv_width == 7
    assert table._rows_buffer is original
    assert not calls, "a CPU-device holder must not touch the CUDA allocator"
    # Releasing twice is still safe.
    holder.release()
    assert cache.kv_width == 7


def test_release_without_prior_retirement_still_restores_in_constructor_failure_path():
    """The constructor's `except: self.release()` must restore bindings exactly as before."""
    holder, cache, target, table, original, _private, layer, state = _verify_holder()
    cache.layers[0].kv_width = 2

    holder.release()

    assert table._rows_buffer is original
    assert layer.state_log is None and state.state_log is None and not hasattr(cache, "kv_width")
    assert target.kv_width == 64
    assert holder._released and holder.graph is None


def _draft_holder(*, retired=False, device="cpu"):
    layer = SimpleNamespace(kv_width=64)
    cache = SimpleNamespace(layers=[layer])
    moe = SimpleNamespace(rowwise=False)
    holder = DraftGraph.__new__(DraftGraph)
    holder.device = torch.device(device)
    holder.cache = cache
    holder.graph = object()
    holder.out = torch.zeros(3, dtype=torch.long)
    holder.replays = 0
    holder.width = 128
    holder.drafts = 3
    holder.rows = 4
    holder._bindings_retired = retired
    holder._released = False
    holder._saved_attrs = [(moe, "rowwise", False), (layer, "kv_width", 64)]
    layer.kv_width = 128
    moe.rowwise = True
    return holder, cache, layer, moe


def test_draft_retirement_restores_head_cache_and_keeps_its_recording(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    holder, cache, layer, moe = _draft_holder()
    graph, out = holder.graph, holder.out

    holder.retire_bindings()

    assert layer.kv_width == 64 and cache.layers[0].kv_width == 64
    assert moe.rowwise is False
    assert holder.graph is graph and holder.out is out
    assert holder._saved_attrs == []


def test_retired_draft_refuses_replay_and_release_keeps_new_bindings(monkeypatch):
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    holder, _cache, layer, _moe = _draft_holder()
    holder.retire_bindings()
    with pytest.raises(RuntimeError, match="bindings were retired"):
        holder.replay(8, 2)
    layer.kv_width = 99

    holder.release()

    assert holder.graph is None and holder._released
    assert layer.kv_width == 99, "release must not restore bindings a second time"


def test_missing_flags_on_a_partial_holder_stay_safe():
    """Constructor-failure and `__new__` holders may lack the bookkeeping attributes."""
    holder = VerifyGraph.__new__(VerifyGraph)
    holder.device = torch.device("cpu")
    holder.cache = SimpleNamespace(layers=[], ple_states={})
    holder._saved_attrs = []
    holder._saved_rows_buffers = []
    holder._gdn_logs = []
    holder._ple_logs = []
    holder._ple_rows = []
    holder._stager = None
    holder._rewinds = {}
    holder.graph = None

    holder.retire_bindings()
    holder.release()

    assert holder._bindings_retired and holder._released