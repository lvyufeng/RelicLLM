"""CPU contracts for the captured startup draft chain.

The fakes here prove the host control flow: positions, the seeded first draft, chain tokens, cache
hygiene and teardown. They deliberately substitute the CUDA warm-up/record seam (`_record`), because
a CPU box cannot hold a graph; actual replay arithmetic and real-checkpoint draft parity remain a
separate required GPU measurement.
"""
from types import SimpleNamespace

import pytest
import torch

from relicllm.models.qwen4_exp import runtime
from relicllm.models.qwen4_exp import startup_graph
from relicllm.models.qwen4_exp import verify_graph as graphs
from relicllm.models.qwen4_exp.startup_graph import StartupDraftGraph


HIDDEN = 64
VOCAB = 64
HC = 2
_AUTO = object()


class _MoE:
    """The head's expert backend as `DraftGraph._prepare` leaves it."""

    def __init__(self, *, rowwise=True, capture_safe=True, inner=True):
        self.rowwise = rowwise
        self.inner = SimpleNamespace(_capture_safe=capture_safe) if inner else None


class _Head:
    """A deterministic stand-in for the MTP head.

    The answer is a function of the position and the conditioning id, so a test can recompute the
    expected chain without reading the implementation under test, and a salt changed between capture
    and replay proves the replay recomputed rather than reusing the warm-up's result.
    """

    def __init__(self, *, hc_count=HC, moe=_AUTO):
        self.config = SimpleNamespace(hc_count=hc_count)
        self.moe = _MoE() if moe is _AUTO else moe
        self.calls = []
        self.salt = 0
        self.mutate_length = False
        self.length_writes = 0

    def forward(self, streams, token_ids, cache, *, past_len=0, return_streams=False):
        if isinstance(past_len, torch.Tensor):
            position = int(past_len.reshape(()))
        else:
            position = int(past_len)
        ids = [int(value) for value in token_ids.reshape(-1)]
        seq = int(token_ids.shape[1])
        self.calls.append(SimpleNamespace(
            position=position, ids=ids, device_position=isinstance(past_len, torch.Tensor),
            stack=streams.clone(), seq=seq,
        ))
        if self.mutate_length:
            cache.length = position + seq
            self.length_writes += 1
        answer = self.answer(position, ids[0])
        mixed = torch.zeros(1, seq, HIDDEN, dtype=streams.dtype)
        mixed[0, -1, answer] = 1.0
        stream = torch.full((1, seq, HIDDEN * self.config.hc_count),
                            float(position + ids[0]), dtype=streams.dtype)
        self.calls[-1].mixed = mixed.clone()
        self.calls[-1].stream = stream.clone()
        return (mixed, stream) if return_streams else mixed

    def answer(self, position, token):
        return (position * 7 + token * 3 + 1 + self.salt) % VOCAB


class _Recorder:
    """Replaces the CUDA warm-up/record seam with re-executable closures.

    `record` mirrors production: the chain runs once eagerly and once "into the recording", and the
    returned handle re-runs it on `replay`. That is what lets a test tell a real replay apart from
    code that just returns the capture-time buffer.
    """

    def __init__(self):
        self.ready_calls = 0
        self.empty_calls = 0
        self.recorded = []
        self.runs = 0
        self.fail_at = None

    def ready(self, device):
        self.ready_calls += 1
        return True

    def empty_cache(self, device):
        self.empty_calls += 1

    def record(self, device, run):
        if self.fail_at == "before":
            raise RuntimeError("injected capture failure")
        run()
        if self.fail_at == "after-warmup":
            raise RuntimeError("injected capture failure")
        run()
        self.runs += 2
        graph = SimpleNamespace(replays=0)

        def replay():
            graph.replays += 1
            self.runs += 1
            run()

        graph.replay = replay
        self.recorded.append(graph)
        return graph


def _install(monkeypatch, recorder):
    monkeypatch.setattr(startup_graph, "_capture_ready", recorder.ready)
    monkeypatch.setattr(startup_graph, "_record", recorder.record)
    monkeypatch.setattr(startup_graph, "_empty_cache", recorder.empty_cache)
    return recorder


def _ctx():
    return SimpleNamespace(device=torch.device("cpu"), tp_world_size=1)


def _cache(*, width=64, length=11, cos=True, kv_width=True):
    layer = SimpleNamespace(
        k=torch.zeros(1, 1, width, 8),
        v=torch.zeros(1, 1, width, 8),
        index_k=torch.zeros(1, width, 8),
        length=length,
    )
    if cos:
        layer.cos = torch.zeros(3, 1, width, 8)
        layer.sin = torch.zeros(3, 1, width, 8)
    if kv_width is True:
        layer.kv_width = width
    elif kv_width is not False:
        layer.kv_width = int(kv_width)
    return SimpleNamespace(layers=[layer], length=length), layer


def _inputs(*, width=64, length=11, cos=True, kv_width=True, moe=_AUTO, hc=HC):
    cache, layer = _cache(width=width, length=length, cos=cos, kv_width=kv_width)
    head = _Head(hc_count=hc, moe=moe)
    model = SimpleNamespace(mtp=head, lm_head=torch.eye(VOCAB), dtype=torch.float32)
    stack = torch.full((1, 1, HIDDEN * hc), 0.75)
    return SimpleNamespace(model=model, head=head, cache=cache, layer=layer, stack=stack)


def _build(inputs, *, start=12, first_draft=5, chain_stack=None, drafts=4):
    return StartupDraftGraph(
        model=inputs.model, ctx=_ctx(), head_cache=inputs.cache, start=start,
        first_draft=first_draft,
        chain_stack=inputs.stack if chain_stack is None else chain_stack,
        drafts=drafts,
    )


def _expected(start, first_draft, drafts, *, salt=0):
    """The eager chain's tokens, recomputed from the fake head's rule."""
    out = [int(first_draft)]
    token = int(first_draft)
    for link in range(1, drafts):
        token = (start + link - 1) * 7 + token * 3 + 1 + salt
        token %= VOCAB
        out.append(token)
    return out


def _runs(head, per_run):
    return [head.calls[index:index + per_run]
            for index in range(0, len(head.calls), per_run)]


def test_records_the_eager_chain_positions_tokens_and_seed(monkeypatch):
    recorder = _install(monkeypatch, _Recorder())
    inputs = _inputs()
    graph = _build(inputs, start=12, first_draft=5, drafts=4)
    expected = _expected(12, 5, 4)

    assert len(recorder.recorded) == 1 and graph.graph is recorder.recorded[0]
    runs = _runs(inputs.head, 3)
    assert len(runs) == 2, "the chain runs eagerly once and into the recording once"
    for calls in runs:
        assert [call.position for call in calls] == [12, 13, 14]
        assert [call.ids for call in calls] == [[5], [expected[1]], [expected[2]]]
        assert all(call.device_position for call in calls), "positions must stay device tensors"
        assert all(call.seq == 1 for call in calls)
    assert graph.out.tolist() == expected
    assert graph.token.tolist() == [[5]]
    assert graph.replays == 0


def test_replay_returns_the_chain_and_recomputes_it(monkeypatch):
    recorder = _install(monkeypatch, _Recorder())
    inputs = _inputs()
    graph = _build(inputs)
    expected = _expected(12, 5, 4)
    assert graph.replay() == expected
    assert graph.replays == 1 and recorder.recorded[0].replays == 1
    assert len(_runs(inputs.head, 3)) == 3
    assert [call.position for call in _runs(inputs.head, 3)[-1]] == [12, 13, 14]

    # A salt changed after the recording proves the replay re-executes instead of handing back the
    # warm-up buffer: the returned tokens follow the new arithmetic.
    inputs.head.salt = 9
    salted = _expected(12, 5, 4, salt=9)
    assert salted != expected
    assert graph.replay() == salted
    assert len(recorder.recorded) == 1, "a replay must not build a second recording"


def test_replay_copies_new_inputs_into_stable_buffers(monkeypatch):
    _install(monkeypatch, _Recorder())
    inputs = _inputs()
    graph = _build(inputs)
    new_stack = torch.full_like(inputs.stack, -1.5)
    actual = graph.replay(start=20, first_draft=7, chain_stack=new_stack)
    assert actual == _expected(20, 7, 4)
    calls = _runs(inputs.head, 3)[-1]
    assert [call.position for call in calls] == [20, 21, 22]
    assert calls[0].ids == [7]
    assert torch.equal(calls[0].stack, new_stack)
    assert graph.position.tolist() == 20
    assert graph.stack.data_ptr() != new_stack.data_ptr(), "the recording must own its stack"
    assert torch.equal(graph.stack, new_stack)
    assert actual[0] == 7


def test_replay_validates_the_arena_before_replaying(monkeypatch):
    recorder = _install(monkeypatch, _Recorder())
    inputs = _inputs(width=32)
    graph = _build(inputs, start=12, drafts=4)
    with pytest.raises(ValueError, match="past the head arena"):
        graph.replay(start=30)
    with pytest.raises(ValueError, match="before position 0"):
        graph.replay(start=-1)
    assert recorder.recorded[0].replays == 0


@pytest.mark.parametrize(("drafts", "start", "fits"),
                         [(4, 13, True), (4, 14, False), (2, 15, True)])
def test_constructor_rejects_a_chain_that_does_not_fit(monkeypatch, drafts, start, fits):
    _install(monkeypatch, _Recorder())
    inputs = _inputs(width=16)
    if fits:
        graph = _build(inputs, start=start, drafts=drafts)
        assert graph.width == 16
    else:
        with pytest.raises(ValueError, match="head arena"):
            _build(inputs, start=start, drafts=drafts)


@pytest.mark.parametrize("drafts", [0, 1])
def test_constructor_rejects_a_chain_shorter_than_one_link(monkeypatch, drafts):
    _install(monkeypatch, _Recorder())
    with pytest.raises(ValueError, match="at least one chained draft"):
        _build(_inputs(), drafts=drafts)


def test_requires_a_cuda_device():
    # The real seam, not the fake: on a CPU device the recording is refused instead of faked.
    inputs = _inputs()
    with pytest.raises(RuntimeError, match="CUDA device"):
        _build(inputs)


@pytest.mark.parametrize("case", ["no_cos", "no_width", "wrong_width"])
def test_requires_the_draft_graph_prepared_head_cache(monkeypatch, case):
    recorder = _install(monkeypatch, _Recorder())
    inputs = _inputs(cos=case != "no_cos",
                     kv_width=(False if case == "no_width"
                               else 32 if case == "wrong_width" else True))
    with pytest.raises(ValueError, match="DraftGraph|RoPE|KV width"):
        _build(inputs)
    assert recorder.recorded == []


@pytest.mark.parametrize("case", ["none", "no_rowwise_attr", "rowwise_off", "inner_none",
                                  "capture_unsafe"])
def test_requires_the_draft_graph_prepared_expert_path(monkeypatch, case):
    recorder = _install(monkeypatch, _Recorder())
    moe = _AUTO
    inputs = _inputs()
    if case == "none":
        inputs.head.moe = None
    elif case == "no_rowwise_attr":
        inputs.head.moe = SimpleNamespace(inner=_MoE().inner)
    elif case == "rowwise_off":
        inputs.head.moe = _MoE(rowwise=False)
    elif case == "inner_none":
        inputs.head.moe = _MoE(inner=False)
    else:
        inputs.head.moe = _MoE(capture_safe=False)
    with pytest.raises(RuntimeError, match="expert backend"):
        _build(inputs)
    assert recorder.recorded == []


def test_capture_restores_layer_length_and_installs_no_bindings(monkeypatch):
    _install(monkeypatch, _Recorder())
    inputs = _inputs(length=11)
    inputs.head.mutate_length = True
    layer_keys = set(inputs.layer.__dict__)
    cache_keys = set(inputs.cache.__dict__)
    graph = _build(inputs)
    # One write per link, per eager execution: the recording ran the chain twice.
    assert inputs.head.length_writes == 2 * (graph.drafts - 1)
    assert inputs.layer.length == 11
    assert set(inputs.layer.__dict__) == layer_keys
    assert set(inputs.cache.__dict__) == cache_keys
    assert graph.replays == 0


def test_missing_length_is_restored_as_missing(monkeypatch):
    _install(monkeypatch, _Recorder())
    inputs = _inputs()
    inputs.head.mutate_length = True
    del inputs.layer.length
    del inputs.cache.length
    _build(inputs)
    assert "length" not in inputs.layer.__dict__


@pytest.mark.parametrize("when", ["before", "after-warmup"])
def test_failed_capture_releases_and_restores_length(monkeypatch, when):
    recorder = _install(monkeypatch, _Recorder())
    recorder.fail_at = when
    inputs = _inputs(length=11)
    inputs.head.mutate_length = True
    with pytest.raises(RuntimeError, match="injected capture failure"):
        _build(inputs)
    assert recorder.recorded == []
    assert inputs.layer.length == 11
    assert recorder.empty_calls == 1, "the release path must drop the partial recording"


def test_release_is_idempotent_and_rejects_replay(monkeypatch):
    recorder = _install(monkeypatch, _Recorder())
    graph = _build(_inputs())
    assert recorder.empty_calls == 0, "no cache teardown before the request is finished"
    graph.release()
    assert graph.released and graph.graph is None
    with pytest.raises(RuntimeError, match="released"):
        graph.replay()
    graph.release()
    assert recorder.empty_calls == 1


@pytest.mark.parametrize("mode", ["stream", "collapsed"])
def test_next_link_stack_matches_the_eager_recycle(monkeypatch, mode):
    _install(monkeypatch, _Recorder())
    monkeypatch.setattr(runtime, "_mtp_recycle", lambda: mode)
    inputs = _inputs()
    _build(inputs, drafts=3)
    calls = _runs(inputs.head, 2)[0]
    if mode == "stream":
        assert torch.equal(calls[1].stack, calls[0].stream)
    else:
        collapsed = calls[0].mixed.unsqueeze(-2).expand(-1, -1, HC, -1).reshape(1, 1, HIDDEN * HC)
        assert torch.equal(calls[1].stack, collapsed)


def test_two_recordings_keep_independent_buffers(monkeypatch):
    _install(monkeypatch, _Recorder())
    first, second = _inputs(), _inputs()
    first.head.salt = 0
    second.head.salt = 5
    one = _build(first, first_draft=5)
    two = _build(second, first_draft=9)
    for name in ("token", "stack", "position", "out"):
        assert getattr(one, name).data_ptr() != getattr(two, name).data_ptr()
    before = two.out.tolist()
    one.replay(start=30, first_draft=1)
    assert two.out.tolist() == before
    assert two.replay() == _expected(12, 9, 4, salt=5)


def test_recording_does_not_quantize_or_convert_the_head(monkeypatch):
    _install(monkeypatch, _Recorder())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the startup recording must not quantize the head experts again")

    monkeypatch.setattr(graphs, "_quantize_head_experts", forbidden)
    inputs = _inputs()
    inner = inputs.head.moe.inner
    graph = _build(inputs)
    assert inputs.head.moe.inner is inner
    assert inputs.head.moe.rowwise is True
    assert inner._capture_safe is True
    assert graph.stack.dtype is torch.float32
    assert all(call.stack.dtype is torch.float32 for call in inputs.head.calls)
    assert inputs.model.lm_head.dtype is torch.float32


@pytest.mark.parametrize("case", ["not_tensor", "ndim", "leading", "dtype", "device"])
def test_rejects_a_malformed_chain_stack(monkeypatch, case):
    _install(monkeypatch, _Recorder())
    inputs = _inputs()
    bad = inputs.stack
    if case == "not_tensor":
        bad = [[[0.0]]]
    elif case == "ndim":
        bad = torch.zeros(1, HIDDEN * HC)
    elif case == "leading":
        bad = torch.zeros(2, 1, HIDDEN * HC)
    elif case == "dtype":
        bad = torch.zeros(1, 1, HIDDEN * HC, dtype=torch.float64)
    else:
        bad = torch.zeros(1, 1, HIDDEN * HC, device="meta")
    with pytest.raises((TypeError, ValueError)):
        _build(inputs, chain_stack=bad)


def test_rejects_a_negative_first_draft(monkeypatch):
    _install(monkeypatch, _Recorder())
    with pytest.raises(ValueError, match="token id"):
        _build(_inputs(), first_draft=-1)
    graph = _build(_inputs())
    with pytest.raises(ValueError, match="token id"):
        graph.replay(first_draft=-3)


def test_replay_returns_a_plain_int_list_once(monkeypatch):
    _install(monkeypatch, _Recorder())
    graph = _build(_inputs(), drafts=4)
    result = graph.replay()
    assert isinstance(result, list) and len(result) == 4
    assert all(type(value) is int for value in result)


def test_recording_buffers_stay_stable_across_replays(monkeypatch):
    _install(monkeypatch, _Recorder())
    graph = _build(_inputs())
    names = ("token", "stack", "position", "out")
    pointers = {name: getattr(graph, name).data_ptr() for name in names}
    handle = graph.graph
    graph.replay()
    graph.replay(start=13, first_draft=2)
    assert {name: getattr(graph, name).data_ptr() for name in names} == pointers
    assert graph.graph is handle, "a replay must reuse the one recording"