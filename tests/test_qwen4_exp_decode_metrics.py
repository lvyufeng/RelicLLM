"""Deterministic CPU checks for Qwen4-Exp decode timing and logit selection.

The synthetic clock models host work and queued work explicitly. These tests exercise
wall-clock boundaries, not CUDA kernel latency or a graph's numerical equivalence.
Run from the repository root with the normal runtime dependencies available.
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.distributed as dist

from relicllm.models.qwen4_exp import runtime


class _Clock:
    """A serial synthetic work queue whose unfinished work is paid at a drain."""

    def __init__(self) -> None:
        self.now = 0.0
        self.pending = 0.0
        self.drains = []

    def perf_counter(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds

    def enqueue(self, seconds: float) -> None:
        self.pending += seconds

    def drain(self) -> None:
        self.drains.append((self.now, self.pending))
        self.now += self.pending
        self.pending = 0.0


def _logits(batch: int, length: int) -> torch.Tensor:
    result = torch.zeros(batch, 16)
    for row in range(batch):
        result[row, (length + row + 1) % 16] = 10.0
    return result


class _TimedModel:
    """Predict by position and assert that each forward starts at committed state."""

    def __init__(self, clock: _Clock, *, flat_logits: bool = False) -> None:
        self.clock = clock
        self.flat_logits = flat_logits
        self.profiler = None
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        self.local_layers = []
        self.has_mtp = False
        self.calls = []
        self.graph_calls = []
        self.releases = []
        self.cache = None

    def make_cache(self, *, batch_size: int, max_seq_len: int):
        self.cache = SimpleNamespace(length=0, ple_token_history=None, layers=[], state=[])
        return self.cache

    def forward(
        self, ids, *, cache, past_len=0, last_token_only=False, return_streams=False
    ):
        assert ids.ndim == 2
        assert cache.length == past_len, "decode must start from committed, not speculative state"
        assert 999 not in cache.state, "a declined replay must not survive in the eager cache"
        prefill = last_token_only or past_len == 0
        self.calls.append((int(past_len), tuple(ids.shape), prefill))
        self.clock.advance(2.0 if prefill else 1.0)
        self.clock.enqueue(3.0 if prefill else 4.0)
        cache.length = int(past_len) + ids.shape[1]
        cache.state.extend(int(value) for value in ids[0].tolist())
        result = _logits(ids.shape[0], cache.length)
        if not self.flat_logits:
            seq = 1 if last_token_only else ids.shape[1]
            result = result[:, None].expand(-1, seq, -1)
        if return_streams:
            return result, torch.zeros(ids.shape[0], ids.shape[1], 4)
        return result


class _TimedWindow:
    """A fake graph with idempotent release and a deliberately dirty handback."""

    def __init__(self, factory, *, model, cache, start, steps, select) -> None:
        self.factory = factory
        self.model = model
        self.cache = cache
        self.start = start
        self.steps = steps
        self.select = select
        self.histories = []
        self.released = False
        self.committed = (cache.length, list(cache.state))

    def __len__(self) -> int:
        return self.steps

    def fill_history(self, seen) -> None:
        self.histories.append(list(seen))

    def replay(self, position, token):
        assert not self.released
        assert self.start <= position < self.start + self.steps
        assert token.shape == (token.shape[0], 1)
        assert token.dtype == torch.long
        assert self.cache.length == position
        assert self.histories[-1][-1] == int(token[0, 0])
        self.model.graph_calls.append(position)
        self.factory.clock.advance(1.0)
        self.factory.clock.enqueue(4.0)
        self.cache.length = position + 1
        if position in self.factory.handback_positions:
            # Release must restore the last committed state before eager consumes this
            # same position. Overwriting KV alone cannot repair a recurrent advance.
            self.cache.state.append(999)
            return None
        self.cache.state.append(int(token[0, 0]))
        self.committed = (self.cache.length, list(self.cache.state))
        return self.select(_logits(token.shape[0], position + 1))

    def release(self) -> None:
        if self.released:
            return
        self.cache.length, state = self.committed
        self.cache.state = list(state)
        self.model.releases.append(self.cache.length)
        self.factory.clock.advance(0.5)
        self.factory.clock.drain()
        self.released = True


class _GraphFactory:
    def __init__(self, clock, *, declined_builds=(), handback_positions=()) -> None:
        self.clock = clock
        self.declined_builds = set(declined_builds)
        self.handback_positions = set(handback_positions)
        self.calls = []
        self.windows = []

    def __call__(self, *, ctx, model, cache, start, steps, token, select):
        assert ctx.device.type == "cpu", "this fixture must never ask for CUDA tensors"
        assert token.shape == (token.shape[0], 1)
        self.calls.append((start, steps))
        self.clock.advance(4.0)
        self.clock.enqueue(6.0)
        if len(self.calls) in self.declined_builds:
            return None
        window = _TimedWindow(
            self, model=model, cache=cache, start=start, steps=steps, select=select
        )
        self.windows.append(window)
        return window


def _install_clock(monkeypatch, clock) -> None:
    original_select = runtime.next_token_from_logits
    monkeypatch.setattr(runtime, "time", clock)
    monkeypatch.setattr(runtime, "_synchronize_device", lambda _ctx: clock.drain())

    def select(logits, ctx, *, batch_size=1):
        # Model the decision and its host-visible token read, so queued preceding work
        # completes before a token can be emitted even though the tensors are on CPU.
        clock.advance(0.25)
        clock.drain()
        return original_select(logits, ctx, batch_size=batch_size)

    monkeypatch.setattr(runtime, "next_token_from_logits", select)


def _ctx():
    return runtime.TPContext(0, 1, torch.device("cpu"), False)


def _run(monkeypatch, *, graph=False, max_new=7, factory_options=None, batch=1, flat=False,
         callback=True, chunk_size=16, prompt=None, **generate_options):
    clock = _Clock()
    model = _TimedModel(clock, flat_logits=flat)
    _install_clock(monkeypatch, clock)
    factory = _GraphFactory(clock, **(factory_options or {}))
    if graph:
        monkeypatch.setattr(runtime, "_build_decode_window", factory)
    ids = torch.tensor([prompt or [1, 2]], dtype=torch.long).expand(batch, -1)
    tokens, stats = runtime.generate(
        model, _ctx(), ids, max_new_tokens=max_new, chunk_size=chunk_size,
        graph_window=2 if graph else 0,
        on_token=(lambda _token: clock.advance(0.5)) if callback else None,
        **generate_options,
    )
    return tokens, stats, model, clock, factory


@pytest.mark.parametrize("flat", [False, True])
def test_eager_wall_latency_includes_decisions_callbacks_and_final_drain(monkeypatch, flat):
    tokens, stats, model, clock, _ = _run(monkeypatch, max_new=3, flat=flat)

    assert tokens == [3, 4, 5]
    assert model.calls == [(0, (1, 2), True), (2, (1, 1), False), (3, (1, 1), False)]
    assert model.cache.length == 4, "the final emitted token must not be forwarded again"
    assert clock.pending == 0.0
    assert stats["prefill_s"] == pytest.approx(5.0)
    assert stats["ttft_s"] == pytest.approx(5.25)
    # Three decisions, two real decode forwards and three callbacks, not an async
    # per-call graph sum. The first emitted token was computed by prefill.
    assert stats["decode_s"] == pytest.approx(3 * 0.25 + 2 * 5.0 + 3 * 0.5)
    assert stats["decode_s"] == stats["decode_wall_s"]
    assert stats["decode_s_excluding_setup"] == stats["decode_s"]
    assert stats["capture_s"] == stats["rebuild_s"] == 0.0
    assert stats["emitted_steps"] == stats["generated_tokens"] == 3.0
    assert stats["prefill_first_token_steps"] == stats["prefill_first_tokens"] == 1.0
    assert stats["decode_steps"] == stats["decode_tokens"] == stats["eager_steps"] == 2.0
    assert stats["replay_count"] == stats["replay_attempts"] == 0.0
    assert stats["decode_tps"] == pytest.approx(2 / 12.25)
    assert stats["decode_tps_excluding_setup"] == stats["decode_tps"]


def test_chunked_prefill_seam_precedes_only_the_initial_decision(monkeypatch):
    tokens, stats, model, _, _ = _run(
        monkeypatch, max_new=3, chunk_size=2, prompt=[1, 2, 3, 4, 5], callback=False
    )
    assert tokens == [6, 7, 8]
    assert [past for past, _, _ in model.calls] == [0, 2, 4, 5, 6]
    assert stats["prefill_s"] == pytest.approx(15.0)
    assert stats["ttft_s"] == pytest.approx(15.25)
    assert stats["decode_s"] == pytest.approx(10.75)


def test_graph_wall_and_setup_separated_latency_use_the_same_eager_definition(monkeypatch):
    tokens, stats, model, clock, factory = _run(monkeypatch, graph=True)

    assert tokens == [3, 4, 5, 6, 7, 8, 9]
    assert len(model.calls) == 1, "only the actual prefill should call the eager model"
    assert model.graph_calls == [2, 3, 4, 5, 6, 7]
    assert factory.calls == [(2, 2), (4, 2), (6, 2)]
    assert all(window.released for window in factory.windows)
    assert factory.windows[0].histories == [[1, 2, 3], [1, 2, 3, 4]]
    assert stats["prefill_s"] == pytest.approx(5.0)
    assert stats["ttft_s"] == pytest.approx(5.0 + 0.25 + 10.0)
    assert stats["capture_s"] == pytest.approx(10.0)
    assert stats["rebuild_s"] == pytest.approx(2 * (0.5 + 10.0))
    assert stats["graph_setup_s"] == pytest.approx(31.0)
    work_wall = 6 * 5.0 + 7 * 0.25 + 7 * 0.5
    assert stats["decode_s"] == pytest.approx(31.0 + work_wall)
    assert stats["decode_s_excluding_setup"] == pytest.approx(work_wall)
    assert stats["decode_tps"] == pytest.approx(6 / (31.0 + work_wall))
    assert stats["decode_tps_excluding_setup"] == pytest.approx(6 / work_wall)
    assert stats["decode_steps"] == stats["replay_count"] == stats["replay_attempts"] == 6.0
    assert stats["eager_steps"] == stats["handback_count"] == 0.0
    # Final release is lifecycle cleanup, not decoding a token; it stays outside the
    # measured loop. Queued work is drained both before and after each timed build.
    assert clock.now == pytest.approx(5.0 + stats["decode_s"] + 0.5)
    assert clock.pending == 0.0


def test_graph_handback_counts_attempt_without_success_and_restores_before_eager(monkeypatch):
    tokens, stats, model, _, factory = _run(
        monkeypatch, graph=True, max_new=4, factory_options={"handback_positions": {3}}
    )
    assert tokens == [3, 4, 5, 6]
    assert model.graph_calls == [2, 3]
    assert [past for past, _, _ in model.calls] == [0, 3, 4]
    assert 999 not in model.cache.state
    assert model.cache.state == [1, 2, 3, 4, 5]
    assert len(factory.calls) == 1
    assert factory.windows[0].released
    assert stats["decode_steps"] == 3.0
    assert stats["replay_attempts"] == 2.0
    assert stats["replay_count"] == stats["handback_count"] == 1.0
    assert stats["eager_steps"] == 2.0
    assert stats["capture_s"] == pytest.approx(10.0)
    assert stats["rebuild_s"] == 0.0
    assert stats["decode_s"] == pytest.approx(33.5)
    assert stats["decode_s_excluding_setup"] == pytest.approx(23.5)


def test_declined_initial_build_is_timed_once_and_stays_eager(monkeypatch):
    tokens, stats, model, _, factory = _run(
        monkeypatch, graph=True, max_new=4, factory_options={"declined_builds": {1}}
    )
    assert tokens == [3, 4, 5, 6]
    assert factory.calls == [(2, 2)]
    assert len(model.calls) == 4
    assert stats["capture_s"] == pytest.approx(10.0)
    assert stats["rebuild_s"] == stats["replay_attempts"] == stats["replay_count"] == 0.0
    assert stats["eager_steps"] == 3.0
    assert stats["ttft_s"] == pytest.approx(15.25)
    assert stats["decode_s"] == pytest.approx(28.0)
    assert stats["decode_s_excluding_setup"] == pytest.approx(18.0)


def test_declined_rebuild_keeps_its_setup_cost_and_does_not_retry_each_step(monkeypatch):
    tokens, stats, model, _, factory = _run(
        monkeypatch, graph=True, factory_options={"declined_builds": {2}}
    )
    assert tokens == [3, 4, 5, 6, 7, 8, 9]
    assert factory.calls == [(2, 2), (4, 2)]
    assert model.graph_calls == [2, 3]
    assert [past for past, _, _ in model.calls] == [0, 4, 5, 6, 7]
    assert stats["replay_attempts"] == stats["replay_count"] == 2.0
    assert stats["eager_steps"] == 4.0
    assert stats["capture_s"] == pytest.approx(10.0)
    assert stats["rebuild_s"] == pytest.approx(10.5)
    assert stats["decode_s"] == pytest.approx(55.75)
    assert stats["decode_s_excluding_setup"] == pytest.approx(35.25)


@pytest.mark.parametrize("flat", [False, True])
def test_generate_gathers_local_logits_exactly_once_per_emitted_token(monkeypatch, flat):
    clock = _Clock()
    model = _TimedModel(clock, flat_logits=flat)
    gathers = []

    def all_gather(parts, local):
        assert local.shape == (1, 16), "a second gather would see the full 32-token vocabulary"
        gathers.append(local.clone())
        for part in parts:
            part.copy_(local)

    monkeypatch.setattr(dist, "all_gather", all_gather)
    tokens, stats = runtime.generate(
        model, runtime.TPContext(0, 2, torch.device("cpu"), True),
        torch.tensor([[1, 2]]), max_new_tokens=3,
    )
    assert tokens == [3, 4, 5]
    assert len(gathers) == len(model.calls) == 3
    assert stats["decode_steps"] == 2.0


@pytest.mark.parametrize("flat", [False, True])
def test_batch_counts_decoded_tokens_separately_from_prefill_first_tokens(monkeypatch, flat):
    tokens, stats, model, _, _ = _run(monkeypatch, max_new=3, batch=2, flat=flat)
    assert tokens == [[3, 4, 5], [4, 5, 6]]
    assert model.calls == [(0, (2, 2), True), (2, (2, 1), False), (3, (2, 1), False)]
    assert stats["generated_tokens"] == stats["emitted_steps"] == 3.0
    assert stats["generated_tokens_total"] == 6.0
    assert stats["prefill_first_token_steps"] == 1.0
    assert stats["prefill_first_tokens"] == stats["decode_steps"] == 2.0
    assert stats["decode_tokens"] == 4.0
    assert stats["decode_tps"] == pytest.approx(4 / 12.25)


@pytest.mark.parametrize("head", [False, True])
def test_pp_generate_handles_none_or_batched_sequence_logits_without_shape_broadcast(
    monkeypatch, head
):
    clock = _Clock()
    model = _TimedModel(clock)
    if not head:
        forward = model.forward

        def no_head(*args, **kwargs):
            forward(*args, **kwargs)
            return None

        monkeypatch.setattr(model, "forward", no_head)
    broadcasts = []

    def broadcast(token, src):
        assert src == 1
        assert token.shape == (2, 1)
        broadcasts.append(token.clone())
        if not head:
            step = len(broadcasts)
            token.copy_(torch.tensor([[2 + step], [3 + step]]))

    monkeypatch.setattr(dist, "broadcast", broadcast)
    ctx = runtime.TPContext(int(head), 2, torch.device("cpu"), True, pp_size=2, pp_rank=int(head))
    tokens, stats = runtime.generate(model, ctx, torch.tensor([[1, 2], [1, 2]]), max_new_tokens=3)
    assert tokens == [[3, 4, 5], [4, 5, 6]]
    assert len(broadcasts) == len(model.calls) == 3
    assert stats["decode_steps"] == 2.0


@pytest.mark.parametrize("max_new", [0, 1])
def test_zero_or_one_token_needs_no_decode_forward(monkeypatch, max_new):
    tokens, stats, model, _, _ = _run(monkeypatch, max_new=max_new, callback=False)
    assert tokens == ([] if max_new == 0 else [3])
    assert len(model.calls) == 1
    assert stats["emitted_steps"] == float(max_new)
    assert stats["prefill_first_token_steps"] == float(max_new)
    assert stats["decode_steps"] == stats["decode_tps"] == stats["eager_steps"] == 0.0
    assert stats["ttft_s"] == (0.0 if max_new == 0 else pytest.approx(5.25))


def test_collective_eos_stops_before_work_for_an_unused_next_token(monkeypatch):
    local_eos = [False]
    tokens, stats, model, _, _ = _run(
        monkeypatch, max_new=4, callback=False, eos_token_ids=(3,),
        local_eos=local_eos, on_step=lambda: local_eos[0],
    )
    assert tokens == [3]
    assert stats["stopped"] == "eos"
    assert len(model.calls) == 1
    assert stats["decode_steps"] == stats["eager_steps"] == 0.0


@pytest.mark.parametrize("cancel_before_first", [False, True])
def test_cancellation_counts_only_emitted_work(monkeypatch, cancel_before_first):
    polls = []

    def on_step():
        polls.append(1)
        return cancel_before_first or len(polls) > 1

    tokens, stats, model, _, _ = _run(monkeypatch, on_step=on_step, callback=False)
    assert tokens == ([] if cancel_before_first else [3])
    assert stats["stopped"] == "cancel"
    assert len(model.calls) == 1
    assert stats["decode_steps"] == stats["eager_steps"] == 0.0
    assert stats["prefill_first_token_steps"] == float(not cancel_before_first)
    assert stats["ttft_s"] == (0.0 if cancel_before_first else pytest.approx(5.25))


def test_final_drain_includes_work_queued_by_the_last_callback(monkeypatch):
    clock = _Clock()
    model = _TimedModel(clock)
    _install_clock(monkeypatch, clock)
    _, stats = runtime.generate(
        model, _ctx(), torch.tensor([[1, 2]]), max_new_tokens=1,
        on_token=lambda _token: clock.enqueue(9.0),
    )
    assert stats["ttft_s"] == pytest.approx(5.25)
    assert stats["decode_s"] == pytest.approx(9.25)
    assert clock.pending == 0.0
    assert len(model.calls) == 1


def test_optional_warmup_is_a_noop_on_cpu(monkeypatch):
    def unexpected_cuda_work(*_args, **_kwargs):
        raise AssertionError("optional CPU warmup must not allocate or synchronize CUDA work")

    monkeypatch.setattr(torch, "randn", unexpected_cuda_work)
    monkeypatch.setattr(torch.cuda, "synchronize", unexpected_cuda_work)
    monkeypatch.setattr(torch.cuda, "empty_cache", unexpected_cuda_work)
    assert runtime._warmup_device(_ctx(), torch.float32, 8.0) == 0.0


@pytest.mark.parametrize("mtp_draft", [False, True])
def test_run_tp4_wires_warmup_once_outside_measured_generation(monkeypatch, mtp_draft):
    clock = _Clock()
    model = _TimedModel(clock)
    model.mtp = SimpleNamespace(make_cache=lambda *_args: SimpleNamespace(layers=[]))
    _install_clock(monkeypatch, clock)
    if mtp_draft:
        forward = model.forward

        def synchronous_forward(*args, **kwargs):
            # This smoke leg only tests the former NameError and warmup wiring;
            # MTP timing/algorithm changes are not part of this standard-decode patch.
            result = forward(*args, **kwargs)
            clock.drain()
            return result

        monkeypatch.setattr(model, "forward", synchronous_forward)
    warmups = []

    def warmup(ctx, dtype, seconds):
        assert not model.calls
        assert ctx.device.type == "cpu"
        assert dtype == torch.float32
        warmups.append(seconds)
        clock.advance(8.0)
        return 8.0

    tokenizer = SimpleNamespace(
        encode=lambda _text: SimpleNamespace(ids=[1]), decode=lambda _ids: "synthetic output"
    )
    config = SimpleNamespace(text_config=SimpleNamespace(eos_token_id=(15,)))
    monkeypatch.setattr(runtime, "init_distributed", _ctx)
    monkeypatch.setattr(runtime, "load_model", lambda *_args, **_kwargs: (model, config))
    monkeypatch.setattr(runtime, "_load_tokenizer", lambda _path: tokenizer)
    monkeypatch.setattr(runtime, "_warmup_device", warmup)
    monkeypatch.delenv("POCKETLLM_QWEN4_KERNEL_PROFILE", raising=False)
    result = runtime.run_tp4(
        "/unused", "unused", warmup_s=8.0, raw_prompt=True,
        max_new_tokens=1, dtype=torch.float32, mtp_draft=mtp_draft,
    )
    assert warmups == [8.0]
    assert result["tokens"] == [2]
    assert result["stats"]["warmup_s"] == 8.0
    assert result["stats"]["prefill_s"] == pytest.approx(5.0)
    assert len(model.calls) == 1
    # The MTP leg reaches first-token emission, exercising the old unconditional
    # warmup_s NameError, but intentionally does not test the future drafting algorithm.


def _dirty_request_cache():
    layer = SimpleNamespace(
        conv_state=torch.ones(1, 2, 3), recurrent_state=torch.full((1, 1, 2, 2), 2.0),
        initialized=True, length=9, host_position=8, state_log=[torch.ones(1)],
        k=torch.full((1, 1, 12, 2), 7.0), v=torch.full((1, 1, 12, 2), 8.0),
        index_k=torch.full((1, 12, 2), 9.0),
    )
    ple = SimpleNamespace(conv_state=torch.full((1, 4, 9), 3.0))
    return SimpleNamespace(
        layers=[layer, None], ple_states={4: ple}, length=9,
        ple_token_history=torch.tensor([[4, 5]]),
        cos=torch.ones(1, 12, 2), sin=torch.zeros(1, 12, 2),
    )


@pytest.mark.parametrize("inference_buffers", [False, True])
def test_reset_request_cache_zeroes_small_state_in_place_not_kv_or_rope(inference_buffers):
    with torch.inference_mode(inference_buffers):
        cache = _dirty_request_cache()
    layer, ple = cache.layers[0], cache.ple_states[4]
    state_refs = [layer.conv_state, layer.recurrent_state, ple.conv_state]
    arena_refs = [layer.k, layer.v, layer.index_k, cache.cos, cache.sin]
    arena_values = [value.clone() for value in arena_refs]
    runtime._reset_request_cache(cache)
    for current, original in zip([layer.conv_state, layer.recurrent_state, ple.conv_state], state_refs):
        assert current is original
        assert current.is_inference() is inference_buffers
        assert torch.count_nonzero(current) == 0
    for current, original, value in zip(
        [layer.k, layer.v, layer.index_k, cache.cos, cache.sin], arena_refs, arena_values
    ):
        assert current is original
        torch.testing.assert_close(current, value)
    assert cache.length == layer.length == layer.host_position == 0
    assert layer.initialized is False and layer.state_log is None
    assert cache.ple_token_history is None


@pytest.mark.parametrize("kind", ["opaque", "empty", "no_layers"])
def test_request_reset_accepts_fake_caches_without_layer_metadata(kind):
    if kind == "opaque":
        cache = object()
    elif kind == "empty":
        cache = SimpleNamespace()
    else:
        cache = SimpleNamespace(length=9, ple_token_history=torch.tensor([[4, 5]]))
    runtime._reset_request_cache(cache)
    if kind == "no_layers":
        assert cache.length == 0 and cache.ple_token_history is None


@pytest.mark.parametrize("active", ["owner", "hook", "layer", "ple"])
def test_request_reset_rejects_active_layout_before_any_zero_or_device_work(monkeypatch, active):
    cache = _dirty_request_cache()
    if active == "owner":
        cache._decode_graph_owner = object()
    elif active == "hook":
        cache.ple_history_at = lambda _ids: cache.ple_token_history
    elif active == "layer":
        cache.layers[0].state_slots = {}
    else:
        cache.ple_states[4].state_slots = {}
    layer = cache.layers[0]
    original = [value.clone() for value in (layer.conv_state, layer.recurrent_state, cache.ple_states[4].conv_state)]
    history = cache.ple_token_history
    monkeypatch.setattr(runtime, "_synchronize_device", lambda _ctx: pytest.fail("preflight before device work"))
    with pytest.raises(RuntimeError, match="release"):
        runtime._reset_request_cache(cache, ctx=_ctx())
    for value, expected in zip((layer.conv_state, layer.recurrent_state, cache.ple_states[4].conv_state), original):
        torch.testing.assert_close(value, expected)
    assert cache.length == layer.length == 9 and layer.initialized is True
    assert layer.host_position == 8 and layer.state_log is not None
    assert cache.ple_token_history is history


def test_generate_cache_cleanup_is_drained_and_outside_prefill_ttft_decode(monkeypatch):
    clock = _Clock()
    model = _TimedModel(clock)
    cache = SimpleNamespace(length=9, ple_token_history=torch.tensor([[4, 5]]), state=[])
    _install_clock(monkeypatch, clock)
    reset = runtime._reset_request_cache

    def timed_reset(cache, *, ctx=None):
        clock.advance(7.0)
        clock.enqueue(3.0)
        reset(cache, ctx=ctx)

    monkeypatch.setattr(runtime, "_reset_request_cache", timed_reset)
    tokens, stats = runtime.generate(model, _ctx(), torch.tensor([[1, 2]]), max_new_tokens=1, cache=cache)
    assert tokens == [3]
    assert stats["cache_reset_s"] == pytest.approx(10.0)
    assert stats["prefill_s"] == pytest.approx(5.0)
    assert stats["ttft_s"] == pytest.approx(5.25)
    assert stats["decode_s"] == pytest.approx(0.25)
    assert clock.pending == 0.0 and clock.now == pytest.approx(15.25)


def _tiny_request_model():
    """Real GDN/QSA/PLE/MoE CPU arithmetic, with small deterministic weights."""
    import torch.nn.functional as F

    from relicllm.models.qwen4_exp.config import Qwen4ExpTextConfig
    from relicllm.models.qwen4_exp.layers import GatedResidual
    from relicllm.models.qwen4_exp.model import PLELayer, Qwen4ExpDecoderLayer, Qwen4ExpModel
    from relicllm.models.qwen4_exp.moe import InMemoryMoE
    from relicllm.models.qwen4_exp.weights import DeviceEmbedding

    config = Qwen4ExpTextConfig(
        hidden_size=8, num_hidden_layers=2, vocab_size=32,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        layer_types=["linear_attention", "qwen_sparse_attention"],
        linear_num_key_heads=1, linear_num_value_heads=2,
        linear_key_head_dim=4, linear_value_head_dim=4,
        hc_count=2, hc_lowrank=4, num_experts=4, num_experts_per_tok=2,
        moe_intermediate_size=8, shared_expert_intermediate_size=8,
        indexer_n_heads=2, indexer_kv_heads=1, indexer_head_dim=4,
        indexer_budget=8, indexer_compress_ratio=2,
        ple_layer_ids=[1, 2], ple_embed_dim=3, ple_conv_kernel_size=4,
        heads_per_ngram=1, ngram_vocab_size_base=17, make_ngram_vocab_size_divisible_by=4,
        partial_rotary_factor=0.5, mrope_section=[1, 0, 0],
        eos_token_id=[0], pad_token_id=0, max_position_embeddings=64,
    )
    generator = torch.Generator(device="cpu").manual_seed(761)

    def rand(*shape):
        return torch.randn(*shape, generator=generator, device="cpu") * 0.1

    hidden, hc_hidden, inter = config.hidden_size, config.hc_hidden_size, config.moe_intermediate_size
    gate_up = {index: rand(config.num_experts, 2 * inter, hidden) for index in range(2)}
    down = {index: rand(config.num_experts, hidden, inter) for index in range(2)}
    moe = InMemoryMoE(config.num_experts, gate_up, down)
    layers = []
    for index in range(2):
        weights = {}
        for name in ("attn_hyper_connection", "mlp_hyper_connection"):
            weights[f"{name}.hc_norm"] = torch.ones(hc_hidden)
            weights[f"{name}.input_mix_weight_down"] = rand(config.hc_lowrank, hc_hidden)
            weights[f"{name}.input_mix_weight_up"] = rand(hc_hidden, config.hc_lowrank)
            weights[f"{name}.block_inject_weight"] = rand(config.hc_count, hc_hidden)
        weights["mlp.gate"] = rand(config.num_experts, hidden)
        weights["mlp.shared_expert_gate"] = rand(1, hidden)
        weights["mlp.shared_expert.gate_proj"] = rand(inter, hidden)
        weights["mlp.shared_expert.up_proj"] = rand(inter, hidden)
        weights["mlp.shared_expert.down_proj"] = rand(hidden, inter)
        if index == 0:
            weights.update({
                "linear_attn.in_proj_qkv": rand(16, hidden),
                "linear_attn.in_proj_z": rand(8, hidden),
                "linear_attn.in_proj_b": rand(2, hidden), "linear_attn.in_proj_a": rand(2, hidden),
                "linear_attn.conv1d": rand(16, 1, config.linear_conv_kernel_dim),
                "linear_attn.dt_bias": torch.ones(2), "linear_attn.A_log": torch.full((2,), -1.0),
                "linear_attn.out_proj": rand(hidden, 8), "linear_attn.norm": torch.ones(4),
            })
        else:
            weights.update({
                "self_attn.q_proj": rand(16, hidden), "self_attn.k_proj": rand(4, hidden),
                "self_attn.v_proj": rand(4, hidden), "self_attn.o_proj": rand(hidden, 8),
                "self_attn.q_norm": torch.ones(4), "self_attn.k_norm": torch.ones(4),
                "self_attn.index_qk_proj": rand(12, hidden),
                "self_attn.q_layernorm": torch.ones(4), "self_attn.k_layernorm": torch.ones(4),
            })
        ple_index = config.ple_layer_index(index)
        rows = config.ngram_head_vocab_sizes(ple_index)[2]
        table = rand(rows, config.ple_embed_dim)
        ple_width = config.ngram_heads * config.ple_embed_dim
        ple_weights = {
            "key_proj": rand(hc_hidden, ple_width), "value_proj": rand(hidden, ple_width),
            "norm_key": torch.ones(hc_hidden), "norm_query": torch.ones(hc_hidden),
            "norm_conv": torch.ones(hc_hidden),
            "conv1d": rand(hc_hidden, 1, config.ple_conv_kernel_size),
        }
        ple = PLELayer(
            config, ple_weights, ple_index, torch.device("cpu"),
            ngram_table=lambda ids, table=table: F.embedding(ids, table),
        )
        layers.append(Qwen4ExpDecoderLayer(config, index, weights, moe, device=torch.device("cpu"), ple=ple))
    final_mixer = GatedResidual(
        config, torch.ones(hc_hidden), rand(config.hc_lowrank, hc_hidden),
        rand(hc_hidden, config.hc_lowrank), None,
    )
    model = Qwen4ExpModel(
        config, layers, embed_tokens=DeviceEmbedding(rand(config.vocab_size, hidden)),
        lm_head=rand(config.vocab_size, hidden), final_mixer=final_mixer,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    # The CPU test does not exercise H2D/pin allocation. Preallocate the real token
    # staging buffer so an accelerator-only pinned allocator is not required.
    model.__dict__["_ids_pinned"] = torch.empty(1, 64, dtype=torch.long, device="cpu")
    return model


def test_real_cpu_model_reuses_one_arena_with_fresh_request_logits_and_state(monkeypatch):
    from relicllm.models.qwen4_exp import model as model_module

    monkeypatch.setattr(model_module, "_SCHEDULE", "by_layer")
    monkeypatch.delenv("POCKETLLM_QWEN4_QSA_FIXED_WIDTH", raising=False)
    monkeypatch.delenv("POCKETLLM_QWEN4_INDEXER_BUDGET", raising=False)
    monkeypatch.delenv("POCKETLLM_QWEN4_HOST_KV", raising=False)
    model = _tiny_request_model()
    reused = model.make_cache(1, 24)
    refs = [reused.layers[0].conv_state, reused.layers[0].recurrent_state,
            *(state.conv_state for state in reused.ple_states.values()),
            reused.layers[1].k, reused.layers[1].v, reused.layers[1].index_k]
    forward = model.forward
    observations = []
    starts = []

    def record(ids, **kwargs):
        cache = kwargs["cache"]
        if kwargs["past_len"] == 0:
            starts.append(cache)
            assert cache.length == cache.layers[1].length == 0
            assert cache.layers[0].initialized is False
            assert torch.count_nonzero(cache.layers[0].conv_state) == 0
            assert torch.count_nonzero(cache.layers[0].recurrent_state) == 0
            assert all(torch.count_nonzero(state.conv_state) == 0 for state in cache.ple_states.values())
        result = forward(ids, **kwargs)
        observations.append(result.detach().clone())
        return result

    monkeypatch.setattr(model, "forward", record)
    for prompt in ([[1, 2, 3, 4, 5, 6, 7]], [[9, 0, 10]]):
        ids = torch.tensor(prompt, dtype=torch.long)
        observations.clear()
        tokens, _stats = runtime.generate(model, _ctx(), ids, max_new_tokens=3, chunk_size=2, cache=reused)
        reused_logits = list(observations)
        assert reused.layers[0].initialized is True
        assert torch.count_nonzero(reused.layers[0].conv_state) > 0
        reused_state = [value.clone() for value in [reused.layers[0].conv_state, reused.layers[0].recurrent_state,
                        *(state.conv_state for state in reused.ple_states.values())]]
        prefix = reused.ple_token_history.clone()
        fresh = model.make_cache(1, 24)
        observations.clear()
        reference_tokens, _ = runtime.generate(model, _ctx(), ids, max_new_tokens=3, chunk_size=2, cache=fresh)
        assert tokens == reference_tokens
        assert len(reused_logits) == len(observations)
        for actual, expected in zip(reused_logits, observations):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        for actual, expected in zip(reused_state, [fresh.layers[0].conv_state, fresh.layers[0].recurrent_state,
                                   *(state.conv_state for state in fresh.ple_states.values())]):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(prefix, fresh.ple_token_history)
    assert len(starts) == 4
    assert starts[0] is reused and starts[2] is reused
    assert starts[1] is not reused and starts[3] is not reused
    for current, original in zip(
        [reused.layers[0].conv_state, reused.layers[0].recurrent_state,
         *(state.conv_state for state in reused.ple_states.values()),
         reused.layers[1].k, reused.layers[1].v, reused.layers[1].index_k], refs
    ):
        assert current is original
