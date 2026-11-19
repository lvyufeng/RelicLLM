"""CPU contracts for lookahead emission, decode metrics and recording teardown.

The fake model predicts a position-dependent greedy sequence and the fake recordings use CPU
buffers. These tests prove host control flow, not CUDA arithmetic or checkpoint token parity.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from relicllm.models.qwen4_exp import runtime
from relicllm.models.qwen4_exp import verify_graph as graphs


class _Clock:
    def __init__(self):
        self.now = 0.0

    def perf_counter(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Head:
    def __init__(self, model):
        self.model = model
        self.config = SimpleNamespace(hc_count=1)
        self.calls = []

    def make_cache(self, batch, width, device, dtype):
        assert not self.model.native_limit or width <= self.model.native_limit
        return SimpleNamespace(layers=[SimpleNamespace(length=0)], width=width)

    def attach_rope_table(self, cache, width, dtype):
        assert not self.model.native_limit or width <= self.model.native_limit

    def forward(self, streams, ids, cache, *, past_len=0, return_streams=False):
        assert int(past_len) + ids.shape[1] <= cache.width
        assert not getattr(getattr(self.model, "verify_graph", None), "released", False), \
            "a plain native-boundary tail must never call the head again"
        self.calls.append((int(past_len), ids.shape[1]))
        self.model.clock.advance(1.0)
        answer = self.model.logits(int(past_len) + 2, ids.shape[1])
        return (answer, answer) if return_streams else answer


class _Model:
    def __init__(self, clock, *, eos_position=None, fail_verify=False, native_limit=0):
        self.clock = clock
        self.profiler = None
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.config = SimpleNamespace(ngram_size=3, max_position_embeddings=native_limit)
        self.native_limit = native_limit
        self.lm_head = torch.eye(64)
        self.mtp = _Head(self)
        self.eos_position = eos_position
        self.fail_verify = fail_verify
        self.cache = None
        self.calls = []

    def token(self, position):
        return 3 if position == self.eos_position else position + 8

    def logits(self, start, length):
        result = torch.zeros(1, length, 64)
        for row in range(length):
            result[0, row, self.token(start + row)] = 10.0
        return result

    def make_cache(self, *, batch_size, max_seq_len):
        assert not self.native_limit or max_seq_len <= self.native_limit
        self.cache = SimpleNamespace(
            length=0, width=max_seq_len, state=[], ple_token_history=None,
            layers=[SimpleNamespace(state_log=None, length=0)],
            ple_states={0: SimpleNamespace(state_log=None)},
        )
        return self.cache

    def forward(self, ids, *, cache, past_len=0, return_streams=False):
        assert int(past_len) == cache.length == len(cache.state)
        assert int(past_len) + ids.shape[1] <= cache.width
        self.calls.append((int(past_len), ids.shape[1]))
        self.clock.advance(2.0)
        if self.fail_verify and past_len >= 2:
            raise RuntimeError("injected verify failure")
        cache.length = int(past_len) + ids.shape[1]
        cache.state.extend(ids[0].tolist())
        answer = self.logits(int(past_len) + 1, ids.shape[1])
        return answer, answer


class _Verify:
    def __init__(self, *, model, ctx, cache, start, rows):
        self.model, self.cache = model, cache
        self.rows = rows
        self.start = start
        assert start + rows <= cache.width
        self.stage_s = self.device_s = 0.0
        self.released = False
        model.verify_graph = self
        model.clock.advance(4.0)
        self.streams = torch.empty(rows, 64)

    def replay(self, position, feed):
        assert position == self.cache.length
        assert position + self.rows <= self.cache.width
        self.model.clock.advance(2.0)
        self.cache.length = position + self.rows
        self.cache.state.extend(feed)
        self.cache.layers[0].state_log = ["open"]
        self.cache.ple_states[0].state_log = ["open"]
        return [self.model.token(position + i + 1) for i in range(self.rows)]

    def rewind(self, position, kept, feed):
        self.cache.length = position + kept
        self.cache.state = self.cache.state[:position] + feed[:kept]

    def release(self):
        self.released = True
        self.model.clock.advance(0.5)


class _Draft:
    def __init__(self, *, model, ctx, head_cache, verify, drafts):
        self.model, self.drafts = model, drafts
        self.width = head_cache.width
        assert verify.start + 2 * verify.rows <= self.width
        self.released = False
        self.calls = []
        model.draft_graph = self
        model.clock.advance(3.0)

    def replay(self, position, kept):
        assert position + kept + self.drafts <= self.width
        self.calls.append((position, kept))
        self.model.clock.advance(1.0)
        return [self.model.token(position + kept + i + 1) for i in range(self.drafts)]

    def release(self):
        self.released = True
        self.model.clock.advance(0.5)


def _install(monkeypatch, *, captured=False, **model_options):
    clock = _Clock()
    model = _Model(clock, **model_options)
    monkeypatch.setattr(runtime, "time", clock)
    monkeypatch.setattr(runtime, "_ids", lambda values, ctx: torch.tensor([values], dtype=torch.long))
    monkeypatch.setattr(runtime, "_rewind_to", lambda *_args: None)
    monkeypatch.setenv("POCKETLLM_QWEN4_MTP_HEAD_TABLE", "1")
    monkeypatch.delenv("POCKETLLM_QWEN4_ROUND_TRACE", raising=False)
    monkeypatch.setenv("POCKETLLM_QWEN4_MTP_DRAFT_GRAPH", "1")
    # A CUDA descriptor selects the captured control-flow branch, but every fake tensor remains
    # on CPU and no native recording or kernel is allowed to run in this fixture.
    ctx = runtime.TPContext(0, 1, torch.device("cuda:0" if captured else "cpu"), False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *_args: None)
    if captured:
        original_tensor = torch.tensor

        def cpu_tensor(*args, **kwargs):
            if getattr(kwargs.get("device"), "type", None) == "cuda":
                kwargs["device"] = torch.device("cpu")
            return original_tensor(*args, **kwargs)

        monkeypatch.setattr(torch, "tensor", cpu_tensor)
        original_to = torch.Tensor.to

        def cpu_to(tensor, *args, **kwargs):
            if args and getattr(args[0], "type", None) == "cuda":
                args = (torch.device("cpu"),) + args[1:]
            if getattr(kwargs.get("device"), "type", None) == "cuda":
                kwargs["device"] = torch.device("cpu")
            return original_to(tensor, *args, **kwargs)

        monkeypatch.setattr(torch.Tensor, "to", cpu_to)
        monkeypatch.setattr(graphs, "VerifyGraph", _Verify)
        monkeypatch.setattr(graphs, "DraftGraph", _Draft)
    return model, ctx, clock


def _run(model, ctx, *, max_new=6, drafts=3, captured=False, eos=(), callback=None):
    return runtime.generate_mtp_draft_lookahead(
        model, ctx, torch.tensor([[7, 8]]), max_new_tokens=max_new,
        draft_tokens=drafts, verify_graph=captured, eos_token_ids=eos, on_token=callback,
    )


@pytest.mark.parametrize("budget", [0, -1])
def test_nonpositive_budget_does_not_allocate_or_emit(monkeypatch, budget):
    model, ctx, _clock = _install(monkeypatch)
    seen = []
    output, stats = _run(model, ctx, max_new=budget, callback=seen.append)
    assert output == seen == []
    assert model.cache is None and not model.calls and not model.mtp.calls
    assert stats["decode_tokens"] == stats["decode_tps"] == stats["ttft_s"] == 0.0


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("budget", [1, 2, 3, 4, 5, 6, 9])
def test_callbacks_equal_exact_budget_and_no_terminal_head(monkeypatch, captured, budget):
    model, ctx, _clock = _install(monkeypatch, captured=captured)
    seen = []
    output, stats = _run(model, ctx, max_new=budget, captured=captured, callback=seen.append)
    assert output == seen == list(range(10, 10 + budget))
    assert stats["decode_tokens"] == max(0, budget - 1)
    expected_tps = max(0, budget - 1) / stats["decode_s"] if stats["decode_s"] else 0.0
    assert stats["decode_tps"] == pytest.approx(expected_tps)
    assert stats["tokens_per_trunk_step"] == pytest.approx(
        max(0, budget - 1) / max(1, stats["trunk_steps"])
    )
    if captured and budget > 1:
        assert model.verify_graph.released and model.draft_graph.released
        assert len(model.draft_graph.calls) == max(0, stats["draft_rounds"] - 1)
    elif budget == 1:
        assert len(model.mtp.calls) == 1, "only the prefill head call is needed"


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("eos_position", [2, 3, 4, 5, 6])
def test_eos_and_bonus_stop_before_callbacks_and_catchup(monkeypatch, captured, eos_position):
    model, ctx, _clock = _install(monkeypatch, captured=captured, eos_position=eos_position)
    seen = []
    output, stats = _run(model, ctx, max_new=10, captured=captured, eos=(3,), callback=seen.append)
    expected = list(range(10, eos_position + 8)) + [3]
    assert output == seen == expected
    assert stats["stopped"] == "eos"
    assert stats["decode_tokens"] == len(output) - 1
    if captured:
        assert model.verify_graph.released and model.draft_graph.released
        assert len(model.draft_graph.calls) == max(0, stats["draft_rounds"] - 1)


def test_setup_exclusive_and_inclusive_metrics_and_ttft(monkeypatch):
    model, ctx, clock = _install(monkeypatch, captured=True)
    delivered_at = []
    output, stats = _run(
        model, ctx, max_new=4, captured=True, callback=lambda _id: delivered_at.append(clock.now)
    )
    assert output == [10, 11, 12, 13]
    assert stats["prefill_s"] == 3.0
    assert stats["graph_build_s"] == 7.0
    assert stats["ttft_s"] == delivered_at[0] == 10.0
    assert stats["decode_s"] == stats["decode_s_excluding_setup"] == 4.0
    assert stats["decode_wall_s"] == 11.0
    assert stats["decode_tps_excluding_setup"] == 3 / 4
    assert stats["decode_wall_tps"] == 3 / 11
    assert clock.now == 15.0, "both releases remain outside decode and TTFT timing"


def test_draft_constructor_failure_releases_existing_verify(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True)

    def fail(**_kwargs):
        raise RuntimeError("injected draft construction failure")

    monkeypatch.setattr(graphs, "DraftGraph", fail)
    with pytest.raises(RuntimeError, match="draft construction"):
        _run(model, ctx, captured=True)
    assert model.verify_graph.released
    assert model.cache.layers[0].state_log is None
    assert model.cache.ple_states[0].state_log is None


@pytest.mark.parametrize("where", ["first", "accepted"])
def test_callback_failure_releases_both_recordings_and_logs(monkeypatch, where):
    model, ctx, _clock = _install(monkeypatch, captured=True)
    seen = []

    def fail(token):
        seen.append(token)
        if where == "first" or len(seen) > 1:
            raise RuntimeError("injected callback failure")

    with pytest.raises(RuntimeError, match="callback failure"):
        _run(model, ctx, captured=True, callback=fail)
    assert model.verify_graph.released and model.draft_graph.released
    assert model.cache.layers[0].state_log is None
    assert model.cache.ple_states[0].state_log is None


def test_eager_verify_exception_closes_state_logs(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, fail_verify=True)
    with pytest.raises(RuntimeError, match="verify failure"):
        _run(model, ctx)
    assert model.cache.layers[0].state_log is None
    assert model.cache.ple_states[0].state_log is None


def test_short_trace_is_closed_and_exported(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True)
    events = []
    tracer = SimpleNamespace(
        __enter__=lambda: events.append("enter"),
        __exit__=lambda *_args: events.append("exit"),
        export_chrome_trace=lambda path: events.append(path),
    )
    monkeypatch.setenv("POCKETLLM_QWEN4_ROUND_TRACE", "0:99:short-trace.json")
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: tracer)
    _run(model, ctx, max_new=3, captured=True)
    assert events == ["enter", "exit", "short-trace.json"]
    assert model.verify_graph.released and model.draft_graph.released


def test_verify_constructor_failure_closes_partial_logs(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True)

    def fail(*, cache, **_kwargs):
        cache.layers[0].state_log = ["partial construction"]
        cache.ple_states[0].state_log = ["partial construction"]
        raise RuntimeError("injected verify construction failure")

    monkeypatch.setattr(graphs, "VerifyGraph", fail)
    with pytest.raises(RuntimeError, match="verify construction"):
        _run(model, ctx, captured=True)
    assert model.cache.layers[0].state_log is None
    assert model.cache.ple_states[0].state_log is None


def test_failed_draft_release_still_releases_verify_and_logs(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True)

    class BadRelease(_Draft):
        def release(self):
            self.released = True
            raise RuntimeError("injected release failure")

    monkeypatch.setattr(graphs, "DraftGraph", BadRelease)
    with pytest.raises(RuntimeError, match="release failure"):
        _run(model, ctx, captured=True)
    assert model.verify_graph.released and model.draft_graph.released
    assert model.cache.layers[0].state_log is None
    assert model.cache.ple_states[0].state_log is None


def test_trace_callback_failure_closes_trace_and_both_recordings(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True)
    events = []
    tracer = SimpleNamespace(
        __enter__=lambda: events.append("enter"),
        __exit__=lambda *_args: events.append("exit"),
        export_chrome_trace=lambda path: events.append(path),
    )
    monkeypatch.setenv("POCKETLLM_QWEN4_ROUND_TRACE", "0:99:failed-trace.json")
    monkeypatch.setattr(torch.profiler, "profile", lambda **_kwargs: tracer)

    def fail(_token):
        raise RuntimeError("injected callback failure")

    with pytest.raises(RuntimeError, match="callback failure"):
        _run(model, ctx, captured=True, callback=fail)
    assert events == ["enter", "exit", "failed-trace.json"]
    assert model.verify_graph.released and model.draft_graph.released


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("budget", [1, 2, 3, 4, 7, 8, 9, 12, 17, 24])
def test_exact_native_total_uses_plain_tail_without_shortening(monkeypatch, captured, budget):
    native = 2 + budget
    model, ctx, _clock = _install(monkeypatch, captured=captured, native_limit=native)
    seen = []
    output, stats = _run(model, ctx, max_new=budget, captured=captured, callback=seen.append)
    assert output == seen == list(range(10, 10 + budget))
    assert stats["logical_total_tokens"] == stats["arena_tokens"] == native
    assert stats["native_max_context"] == native
    assert all(position + length <= native for position, length in model.calls)
    assert len(model.cache.state) == model.cache.length
    assert stats["decode_tokens"] == budget - 1
    if budget > 1:
        assert stats["plain_tail_steps"] > 0
        assert stats["plain_tail_reason"] == "native_boundary"
        assert stats["plain_tail_start"] + 2 * (3 + 1) > native
    if captured:
        assert stats["graph_rounds"] + stats["plain_tail_steps"] == stats["trunk_steps"]
        assert stats["mtp_eager_steps"] == 0
    if budget < 8:
        assert stats["graph_rounds"] == 0
        assert not model.mtp.calls, "a request already in the plain tail does not seed the head"
    elif captured:
        assert stats["graph_rounds"] > 0
        assert model.verify_graph.released and model.draft_graph.released
    # Every plain tail step consumes one committed position; the last emitted token is not
    # forwarded just to fill an arena. At the exact boundary it is predicted at native - 2.
    if budget > 1:
        assert model.calls[-1] == (native - 2, 1)


@pytest.mark.parametrize("native,budget", [(10, 9), (2, 1)])
def test_request_beyond_native_maximum_fails_before_allocation(monkeypatch, native, budget):
    model, ctx, _clock = _install(monkeypatch, captured=True, native_limit=native)
    with pytest.raises(ValueError, match="exceeds native.*will not be shortened"):
        _run(model, ctx, max_new=budget, captured=True)
    assert model.cache is None and not model.calls and not model.mtp.calls


def test_zero_drafts_uses_plain_trunk_without_head(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, native_limit=10)
    output, stats = _run(model, ctx, max_new=8, drafts=0)
    assert output == list(range(10, 18))
    assert stats["plain_tail_steps"] == stats["trunk_steps"] == 7
    assert not model.mtp.calls


def test_common_start_barrier_is_setup_not_decode(monkeypatch):
    model, ctx, clock = _install(monkeypatch, captured=True)
    ctx.world_size = 2
    monkeypatch.setattr(runtime, "gather_logits", lambda logits, _ctx: logits)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    barriers = []

    def barrier(*, device_ids):
        barriers.append((list(device_ids), clock.now))
        clock.advance(5.0)

    monkeypatch.setattr(torch.distributed, "barrier", barrier)
    delivered_at = []
    output, stats = _run(
        model, ctx, max_new=4, captured=True, callback=lambda _id: delivered_at.append(clock.now)
    )
    assert output == [10, 11, 12, 13]
    assert barriers == [([0], 10.0)]
    assert stats["graph_build_s"] == 7.0
    assert stats["graph_alignment_s"] == 5.0
    assert stats["graph_setup_s"] == 12.0
    assert stats["ttft_s"] == delivered_at[0] == 15.0
    assert stats["decode_s"] == stats["decode_s_excluding_setup"] == 4.0
    assert stats["decode_wall_s"] == 16.0
    assert stats["decode_tps_excluding_setup"] == 3 / 4
    assert stats["decode_wall_tps"] == 3 / 16


def test_eager_mode_never_runs_graph_setup_barrier(monkeypatch):
    model, ctx, _clock = _install(monkeypatch)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "barrier", lambda **_kwargs: pytest.fail("eager barrier"))
    _output, stats = _run(model, ctx, max_new=4)
    assert stats["graph_setup_s"] == stats["graph_alignment_s"] == 0.0
