"""CPU lifecycle contracts for the captured verifier; no CUDA performance assertions."""
from types import SimpleNamespace

import pytest
import torch

from relicllm.models.qwen4_exp.verify_graph import VerifyGraph, _MISSING


def test_release_clears_capture_logs_and_restores_installed_staging():
    layer = SimpleNamespace(state_log=[object()], kv_width=100)
    state = SimpleNamespace(state_log=[object()])
    cache = SimpleNamespace(layers=[layer], ple_states={2: state}, ple_history_at=lambda _ids: None)
    original = torch.zeros(1)
    table = SimpleNamespace(_rows_buffer=torch.ones(1))
    graph = VerifyGraph.__new__(VerifyGraph)
    graph.device = torch.device("cpu")
    graph.cache = cache
    graph.graph = object()
    graph._logs_open = True
    graph._rewinds = {1: object()}
    graph._gdn_logs = [(layer, layer.state_log)]
    graph._ple_logs = [(state, state.state_log)]
    graph._ple_rows = [object()]
    graph._stager = object()
    graph._saved_rows_buffers = [(table, original)]
    graph._saved_attrs = [(layer, "kv_width", _MISSING), (cache, "ple_history_at", _MISSING)]

    graph.release()
    assert graph.graph is None and not graph._rewinds
    assert layer.state_log is None and state.state_log is None
    assert graph._stager is None
    assert not hasattr(layer, "kv_width")
    assert not hasattr(cache, "ple_history_at")
    assert table._rows_buffer is original
    # A cleanup block can run after a constructor's own error cleanup.
    graph.release()
    assert table._rows_buffer is original


def test_constructor_failure_closes_logs_and_restores_recurrence(monkeypatch):
    layer = SimpleNamespace(
        state_log=None,
        k=torch.zeros(1, 1, 8, 2),
        conv_state=torch.tensor([1.0]),
        recurrent_state=torch.tensor([2.0]),
        length=3,
        initialized=True,
    )
    state = SimpleNamespace(state_log=None, conv_state=torch.tensor([3.0]))
    cache = SimpleNamespace(layers=[layer], ple_states={2: state}, length=3,
                            ple_token_history=torch.tensor([[4, 5]]))
    model = SimpleNamespace(pp_size=1, lm_head=torch.ones(1), config=SimpleNamespace(ngram_size=3))
    ctx = SimpleNamespace(device=torch.device("cpu"))

    def prepare(self):
        self._remember(self.cache, "ple_history_at")
        self.cache.ple_history_at = lambda _ids: None

    def capture(self):
        self._begin_logs()
        layer.conv_state.add_(10)
        state.conv_state.add_(10)
        raise RuntimeError("injected capture failure")

    monkeypatch.setattr(VerifyGraph, "_prepare", prepare)
    monkeypatch.setattr(VerifyGraph, "_capture", capture)
    with pytest.raises(RuntimeError, match="injected capture failure"):
        VerifyGraph(model=model, ctx=ctx, cache=cache, start=3, rows=2)
    assert layer.state_log is None and state.state_log is None
    assert not hasattr(cache, "ple_history_at")
    assert layer.conv_state.tolist() == [1.0]
    assert layer.recurrent_state.tolist() == [2.0]
    assert state.conv_state.tolist() == [3.0]
    assert cache.length == layer.length == 3


@pytest.mark.parametrize("rows", [2, 4, 5])
@pytest.mark.parametrize("ngram", [2, 3, 4])
def test_rewind_ple_prefix_matches_append_without_tensor_construction(rows, ngram, monkeypatch):
    feed = list(range(20, 20 + rows))
    prefix = list(range(1, ngram))
    for kept in range(1, rows + 1):
        graph = VerifyGraph.__new__(VerifyGraph)
        graph.ngram = ngram
        graph.rows = rows
        graph._rewinds = {}
        graph._gdn_logs = []
        graph._ple_logs = []
        graph.history = torch.tensor([prefix + feed], dtype=torch.long)
        graph._history_np = graph.history.numpy()
        committed = torch.tensor([prefix], dtype=torch.long)
        graph.cache = SimpleNamespace(layers=[], length=10, ple_token_history=committed)
        expected = (prefix + feed[:kept])[-(ngram - 1):]
        with monkeypatch.context() as context:
            def forbidden(*_args, **_kwargs):
                raise AssertionError("rewind constructed a host tensor")
            context.setattr(torch, "tensor", forbidden)
            context.setattr(torch, "cat", forbidden)
            graph.rewind(10, kept, feed)
        assert graph.cache.length == 10 + kept
        assert graph.cache.ple_token_history is committed
        assert committed.tolist() == [expected]
