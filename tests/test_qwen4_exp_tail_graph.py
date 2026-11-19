"""Native-boundary one-row graph ownership and host control-flow contracts.

These CPU fakes prove staging independence, reverse teardown, complete output and timing scope.
Actual CUDA replay/full-checkpoint greedy identity remains a separate required GPU measurement.
"""
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from relicllm.models.qwen4_exp import runtime
from relicllm.models.qwen4_exp import verify_graph as graphs
from relicllm.models.qwen4_exp.model import Qwen4ExpCache, PLEState
from test_qwen4_exp_lookahead_contract import _install, _run, _Verify
from test_qwen4_exp_verify_stage import _config, _plan


FLAG = "POCKETLLM_QWEN4_MTP_TAIL_GRAPH"


@pytest.mark.parametrize("budget", [2, 4, 7, 8, 12, 17, 24])
def test_exact_native_tail_runs_prebuilt_recording_without_eager_calls(monkeypatch, budget):
    model, ctx, clock = _install(monkeypatch, captured=True, native_limit=2 + budget)
    monkeypatch.setenv(FLAG, "1")
    built, released = [], []

    class Recording(_Verify):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            built.append(self)

        def release(self):
            released.append(self.rows)
            super().release()

    monkeypatch.setattr(graphs, "VerifyGraph", Recording)
    output, stats = _run(model, ctx, max_new=budget, captured=True)
    assert output == list(range(10, 10 + budget))
    assert stats["logical_total_tokens"] == stats["arena_tokens"] == 2 + budget
    assert stats["tail_graph_steps"] > 0 and stats["tail_graph_built"]
    assert stats["plain_tail_steps"] == stats["mtp_eager_steps"] == 0
    assert stats["trunk_steps"] == stats["graph_rounds"] + stats["tail_graph_steps"]
    assert model.calls == [(0, 2)], "decode must not call eager trunk or compile a new tail graph"
    assert [r.rows for r in built] == ([1, 4] if budget >= 8 else [1])
    assert released == ([4, 1] if budget >= 8 else [1])
    assert all(r.released for r in built)
    assert stats["graph_build_s"] == (11.0 if budget >= 8 else 4.0)
    assert stats["ttft_s"] >= stats["prefill_s"] + stats["graph_build_s"]


def test_disabled_tail_flag_keeps_safe_eager_default(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True, native_limit=19)
    monkeypatch.setenv(FLAG, "0")
    output, stats = _run(model, ctx, max_new=17, captured=True)
    assert output == list(range(10, 27))
    assert stats["plain_tail_steps"] > 0 and stats["tail_graph_steps"] == 0
    assert not stats["tail_graph_built"]


def test_failed_main_constructor_releases_underlying_tail(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True, native_limit=26)
    monkeypatch.setenv(FLAG, "1")
    built = []

    class Recording(_Verify):
        def __init__(self, **kwargs):
            if kwargs["rows"] > 1:
                raise RuntimeError("injected main capture failure")
            super().__init__(**kwargs)
            built.append(self)

    monkeypatch.setattr(graphs, "VerifyGraph", Recording)
    with pytest.raises(RuntimeError, match="main capture"):
        _run(model, ctx, max_new=24, captured=True)
    assert len(built) == 1 and built[0].released


def test_callback_failure_releases_main_then_tail(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True, native_limit=26)
    monkeypatch.setenv(FLAG, "1")
    released = []

    class Recording(_Verify):
        def release(self):
            released.append(self.rows)
            super().release()

    monkeypatch.setattr(graphs, "VerifyGraph", Recording)

    def fail(_token):
        raise RuntimeError("injected callback failure")

    with pytest.raises(RuntimeError, match="callback"):
        _run(model, ctx, max_new=24, captured=True, callback=fail)
    assert released == [4, 1]


def _real_holders(monkeypatch):
    config = _config()
    plan = _plan(config, layers=1, rows=4, dtype=torch.bfloat16)
    ple, table, _buffer, _shape = plan[0]
    ple.ngram_table = table
    original_buffer = torch.zeros(1, table.head_dim, dtype=table._gather_dtype)
    table._rows_buffer = original_buffer
    layer = SimpleNamespace(
        k=torch.zeros(1, 1, 32, 2), v=torch.zeros(1, 1, 32, 2),
        index_k=torch.zeros(1, 32, 2), length=8, state_log=None,
        conv_state=torch.tensor([1.0]), recurrent_state=torch.tensor([2.0]), initialized=True,
    )
    cache = Qwen4ExpCache(
        layers=[layer], length=8, ple_token_history=torch.tensor([[4, 5]]),
        cos=torch.ones(1, 32, 2), sin=torch.zeros(1, 32, 2),
        ple_states={0: PLEState(conv_state=torch.tensor([3.0]))},
    )
    model = SimpleNamespace(config=config, dtype=torch.bfloat16, pp_size=1, lm_head=torch.ones(1),
                            layers=[SimpleNamespace(ple=ple)])
    ctx = SimpleNamespace(device=torch.device("cpu"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)

    def capture(self):
        # Model warmup/capture changes rolling values, which constructors must restore in place.
        layer.conv_state.add_(10)
        layer.recurrent_state.add_(10)
        cache.ple_states[0].conv_state.add_(10)
        self.graph = object()
        self._rewinds = {}

    monkeypatch.setattr(graphs.VerifyGraph, "_capture", capture)
    return model, ctx, cache, table, original_buffer


def test_private_ple_buffers_and_live_state_addresses_restore_in_reverse(monkeypatch):
    model, ctx, cache, table, original = _real_holders(monkeypatch)
    conv_ptr = cache.layers[0].conv_state.data_ptr()
    recurrent_ptr = cache.layers[0].recurrent_state.data_ptr()
    ple_ptr = cache.ple_states[0].conv_state.data_ptr()
    tail = graphs.VerifyGraph(model=model, ctx=ctx, cache=cache, start=8, rows=1)
    tail_callback = cache.ple_history_at
    tail_buffer = table._rows_buffer
    main = graphs.VerifyGraph(model=model, ctx=ctx, cache=cache, start=8, rows=4)
    assert table._rows_buffer is not tail_buffer
    assert tail_buffer.shape[0] * 4 == table._rows_buffer.shape[0]
    assert tail.history.data_ptr() != main.history.data_ptr()
    assert cache.layers[0].conv_state.data_ptr() == conv_ptr
    assert cache.layers[0].recurrent_state.data_ptr() == recurrent_ptr
    assert cache.ple_states[0].conv_state.data_ptr() == ple_ptr
    assert cache.layers[0].conv_state.item() == 1
    assert cache.layers[0].recurrent_state.item() == 2
    assert cache.ple_states[0].conv_state.item() == 3
    main.release()
    assert cache.ple_history_at is tail_callback and table._rows_buffer is tail_buffer
    tail.release()
    assert not hasattr(cache, "ple_history_at") and table._rows_buffer is original


def test_one_row_commit_updates_history_and_lengths_without_state_rollback(monkeypatch):
    model, ctx, cache, _table, _original = _real_holders(monkeypatch)
    tail = graphs.VerifyGraph(model=model, ctx=ctx, cache=cache, start=8, rows=1)
    tail._history_np[:] = [[4, 5, 17]]
    cache.layers[0].conv_state.fill_(31)
    cache.layers[0].recurrent_state.fill_(32)
    cache.ple_states[0].conv_state.fill_(33)
    tail.rewind(8, 1, [17])
    assert cache.length == cache.layers[0].length == 9
    assert cache.ple_token_history.tolist() == [[5, 17]]
    assert cache.layers[0].conv_state.item() == 31
    assert cache.layers[0].recurrent_state.item() == 32
    assert cache.ple_states[0].conv_state.item() == 33
    tail.release()


def test_one_row_capture_does_not_build_empty_rewind_graph(monkeypatch):
    holder = graphs.VerifyGraph.__new__(graphs.VerifyGraph)
    holder.rows = 1
    holder.device = torch.device("cpu")
    holder.cache = SimpleNamespace(layers=[], ple_states={})
    holder._gdn_logs = []
    holder._ple_logs = []
    holder._logs_open = False
    calls = []

    class Stream:
        def wait_stream(self, _other):
            pass

    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **_kwargs: Stream())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_args: Stream())
    monkeypatch.setattr(torch.cuda, "stream", lambda _stream: nullcontext())
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: calls.append("graph") or object())
    monkeypatch.setattr(torch.cuda, "graph", lambda _graph: nullcontext())
    holder._run = lambda: calls.append("run")
    graphs.VerifyGraph._capture(holder)
    assert calls == ["run", "graph", "run"]
    assert holder._rewinds == {} and not holder._logs_open


def test_generic_confidence_fallback_removes_underlying_tail_before_eager(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True, native_limit=26)
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_MTP_DRAFT_GRAPH", "0")
    released = []

    class Recording(_Verify):
        def release(self):
            released.append(self.rows)
            super().release()
            # The shared fixture prohibits head calls after a *native* handoff. This case is a
            # confidence-based eager MTP fallback instead, which is allowed to keep using the head.
            self.model.verify_graph = None

    monkeypatch.setattr(graphs, "VerifyGraph", Recording)
    output, stats = runtime.generate_mtp_draft_lookahead(
        model, ctx, torch.tensor([[7, 8]]), max_new_tokens=6,
        draft_tokens=3, verify_graph=True, draft_p_min=0.99999,
    )
    assert output == list(range(10, 16))
    assert released == [4, 1]
    assert stats["tail_graph_built"] and stats["tail_graph_steps"] == 0
    assert stats["mtp_eager_steps"] > 0


def test_callback_failure_during_tail_releases_remaining_single_graph(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True, native_limit=9)
    monkeypatch.setenv(FLAG, "1")
    built = []

    class Recording(_Verify):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            built.append(self)

    monkeypatch.setattr(graphs, "VerifyGraph", Recording)
    seen = []

    def fail(token):
        seen.append(token)
        if len(seen) == 3:
            raise RuntimeError("injected tail callback failure")

    with pytest.raises(RuntimeError, match="tail callback"):
        _run(model, ctx, max_new=7, captured=True, callback=fail)
    assert seen == [10, 11, 12]
    assert len(built) == 1 and built[0].released


def test_tail_can_replay_last_arena_position_but_rejects_one_past_end(monkeypatch):
    model, ctx, cache, _table, _original = _real_holders(monkeypatch)
    tail = graphs.VerifyGraph(model=model, ctx=ctx, cache=cache, start=8, rows=1)
    cache.length = 31
    tail.graph = SimpleNamespace(replay=lambda: None)
    tail.answer = torch.tensor([19])
    assert tail.replay(31, [17]) == [19]
    tail.rewind(31, 1, [17])
    assert cache.length == 32
    with pytest.raises(ValueError, match="outside the recorded arena"):
        tail.replay(32, [19])
    tail.release()
