"""CPU-only graph ownership/rollback tests, including real CPU PLE hashing.

The fake replay executes recorded state operations and copies from captured host
row addresses. It deliberately does not rerun CPU hash or table gathering. These
checks establish lifecycle/input-staging invariants, not CUDA numerical parity.
"""

from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch

from relicllm.models.qwen4_exp import graph_window as graphs
from relicllm.models.qwen4_exp.attention import GatedDeltaNetCache, _state_at, _update_state
from relicllm.models.qwen4_exp.layers import NGramHasher
from relicllm.models.qwen4_exp.model import PLEState, Qwen4ExpCache


class _Hasher:
    def __init__(self, config):
        self.impl = NGramHasher(config, 1, torch.device("cpu"))
        self.calls = []

    def row_ids(self, history, out_len):
        self.calls.append(history.clone())
        return self.impl.row_ids(history, out_len)


class _HostTable:
    """HostNGramTable's raw-row/H2D contract without needing a pinned allocator."""

    def __init__(self, raw_dtype=torch.float32):
        self.head_dim = 1
        self._gather_dtype = raw_dtype
        self.weight = torch.arange(64).reshape(64, 1).to(raw_dtype)
        self._rows_buffer = torch.full((4, 1), 17, dtype=raw_dtype)
        self.scale = 0.5 if raw_dtype == torch.uint8 else 1.0
        self.gathers = []
        self.last_source = None

    def _host_rows(self, rows):
        if self._rows_buffer is None or self._rows_buffer.shape[0] < rows:
            self._rows_buffer = torch.empty(rows, self.head_dim, dtype=self._gather_dtype)
        return self._rows_buffer[:rows]

    def _gather_into(self, flat, out):
        self.gathers.append(flat.clone())
        out.copy_(self.weight.index_select(0, flat))

    def _to_device(self, out):
        # This CPU clone stands in for the recorded H2D read and unchanged dequant.
        self.last_source = out
        return out.clone().float() * self.scale

    def __call__(self, row_ids):
        out = self._host_rows(row_ids.numel())
        self._gather_into(row_ids.reshape(-1), out)
        return self._to_device(out).reshape(*row_ids.shape, self.head_dim)


class _Model:
    def __init__(self, raw_dtype=torch.float32, *, flat=False, with_ple=True):
        self.config = SimpleNamespace(
            ngram_size=3, heads_per_ngram=1, primary_eos_token_id=0,
            vocab_size=16, seed=7, hidden_size=2, hc_count=1,
            ple_conv_kernel_size=4, indexer_budget=256,
            ngram_head_vocab_sizes=lambda _index: ([11, 13], [0, 11], 24),
        )
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.lm_head = torch.zeros(5, 2)
        self.pp_size = 1
        self.pp_rank = 0
        self.owns_head = True
        self.flat = flat
        self.table = _HostTable(raw_dtype)
        self.hasher = _Hasher(self.config)
        ple = SimpleNamespace(hasher=self.hasher, ngram_table=self.table) if with_ple else None
        self.layers = [SimpleNamespace(ple=None), SimpleNamespace(ple=ple)]
        self.setup_reads = []
        self.forward_calls = []
        self.replaying_sources = None
        self.last_sources = {}
        self.make_cache_calls = 0

    def make_cache(self, *_args, **_kwargs):
        self.make_cache_calls += 1
        raise AssertionError("width discovery must never allocate a scratch cache or reset PLE")

    def rope(self, position_ids, _dtype):
        batch, width = position_ids.shape[1:]
        base = torch.arange(width).float().view(1, width, 1).expand(batch, -1, 2)
        return base.clone(), -base.clone()

    def forward(self, ids, *, cache, past_len=0, last_token_only=True, return_streams=False):
        position = int(past_len)
        batch = ids.shape[0]
        gdn = cache.layers[0]
        previous_conv = _state_at(gdn, "conv_state")
        previous_rec = _state_at(gdn, "recurrent_state")
        if previous_rec is None:
            previous_rec = torch.zeros_like(gdn.recurrent_state)
        self.forward_calls.append((position, self.replaying_sources is not None))
        if self.replaying_sources is None:
            hook = getattr(cache, "ple_history_at", None)
            history = hook(ids) if hook is not None else torch.cat([cache.ple_token_history, ids], dim=1)
            cache.ple_token_history = history[:, -2:].clone()
        else:
            history = None
        delta = ids.float().reshape(batch, 1, 1)
        conv = previous_conv + delta.expand_as(previous_conv)
        rec_delta = delta.reshape(batch, 1, 1, 1) * 2
        rec = previous_rec + rec_delta.expand_as(previous_rec)
        _update_state(gdn, "conv_state", conv)
        _update_state(gdn, "recurrent_state", rec)
        gdn.initialized = True

        ple = self.layers[1].ple
        if ple is not None:
            state = cache.ple_states[1]
            previous_ple = _state_at(state, "conv_state")
            if self.replaying_sources is None:
                row_ids = ple.hasher.row_ids(history, 1)
                embeddings = ple.ngram_table(row_ids)
                self.last_sources[1] = self.table.last_source
                self.setup_reads.append((previous_conv.clone(), previous_ple.clone()))
            else:
                # CPU hash/gather are absent from a CUDA recording. Only this read
                # from the captured pinned source and the state operations replay.
                embeddings = self.table._to_device(self.replaying_sources[1]).reshape(batch, 1, 2, 1)
            ple_delta = embeddings.flatten(1).mean(1).reshape(batch, 1, 1) * 3
            ple_conv = previous_ple + ple_delta.expand_as(previous_ple)
            _update_state(state, "conv_state", ple_conv)
        else:
            ple_conv = torch.zeros(batch, 2, 9)
        sparse = cache.layers[1]
        sparse.k[:, :, position:position + 1].fill_(float(ids[0, 0]))
        sparse.v[:, :, position:position + 1].fill_(2 * float(ids[0, 0]))
        sparse.index_k[:, position:position + 1].fill_(3 * float(ids[0, 0]))
        sparse.length = position + 1
        cache.length = position + 1
        logits = torch.stack([
            conv.flatten(1).sum(1), rec.flatten(1).sum(1), ple_conv.flatten(1).sum(1),
            conv.flatten(1)[:, 0] + ple_conv.flatten(1)[:, 0], torch.zeros(batch),
        ], dim=1)
        streams = torch.stack([conv.flatten(1)[:, 0], ple_conv.flatten(1)[:, 0]], dim=1)
        if not self.flat:
            logits, streams = logits[:, None], streams[:, None]
        return (logits, streams) if return_streams else logits


class _Replay:
    def __init__(self, holder):
        self.holder = holder
        # Preserve the source addresses, not current shared table scratch pointers.
        self.sources = dict(holder.model.last_sources)
        self.calls = 0

    def replay(self):
        self.calls += 1
        self.holder.model.replaying_sources = self.sources
        try:
            with torch.inference_mode():
                self.holder._run_step()
        finally:
            self.holder.model.replaying_sources = None


def _fake_capture(holder):
    with torch.inference_mode():
        holder._run_step()
        holder._run_step()
    holder.graph = _Replay(holder)


@pytest.fixture(autouse=True)
def _cpu_capture(monkeypatch):
    def unexpected_cuda(*_args, **_kwargs):
        raise AssertionError("these lifecycle tests must never start CUDA work")

    for name in ("synchronize", "empty_cache", "Stream", "CUDAGraph"):
        monkeypatch.setattr(torch.cuda, name, unexpected_cuda)
    monkeypatch.setattr(graphs.DecodeGraph, "_capture", _fake_capture)
    monkeypatch.setenv("POCKETLLM_GRAPH_RETURN_MARGIN", "0")
    monkeypatch.delenv("POCKETLLM_QWEN4_GRAPH_PAST_BUDGET", raising=False)


def _cache(batch=1, *, metadata=False, initialized=True, with_ple=True):
    gdn = GatedDeltaNetCache.__new__(GatedDeltaNetCache)
    gdn.conv_state = torch.full((batch, 3, 2), 2.0)
    gdn.recurrent_state = torch.full((batch, 1, 2, 2), 4.0 if initialized else 0.0)
    gdn.initialized = initialized
    gdn.state_log = None
    sparse = SimpleNamespace(
        k=torch.zeros(batch, 1, 12, 2), v=torch.zeros(batch, 1, 12, 2),
        index_k=torch.zeros(batch, 12, 2), length=3, host_kv=False,
    )
    cache = Qwen4ExpCache(layers=[gdn, sparse], length=3)
    cache.ple_token_history = torch.tensor([[11, 12]]).expand(batch, -1).clone()
    cache.cos = torch.zeros(batch, 6, 2)
    cache.sin = torch.ones(batch, 6, 2)
    if with_ple:
        cache.ple_states[1] = PLEState(torch.full((batch, 2, 9), 6.0))
    if metadata:
        cache.kv_width, cache.rope_width = 4, 6
        cache._decode_graph_owner = None
        cache.ple_history_at = lambda ids: torch.cat([cache.ple_token_history, ids], dim=1)
        sparse.cos = torch.full((batch, 6, 2), 7.0)
        sparse.rope_width = 6
        sparse.host_position = 2
        sparse.kv_width = 4
        for state in [gdn, *cache.ple_states.values()]:
            state.state_slots = None
            state.state_slot = None
    return cache


_META = (
    "kv_width", "rope_width", "cos", "sin", "length", "ple_token_history",
    "ple_history_at", "_decode_graph_owner", "host_position", "state_slots",
    "state_slot", "initialized", "state_log",
)
_ABSENT = object()


def _metadata(cache):
    return [(obj, {key: vars(obj).get(key, _ABSENT) for key in _META})
            for obj in [cache, *cache.layers, *getattr(cache, "ple_states", {}).values()]]


def _assert_metadata(saved, *, length=None, history=None, initialized=None):
    for target, attributes in saved:
        for name, expected in attributes.items():
            actual = vars(target).get(name, _ABSENT)
            if expected is _ABSENT:
                assert actual is _ABSENT, f"{name} should be removed from {type(target).__name__}"
            elif name == "length" and length is not None:
                assert actual == length
            elif name == "ple_token_history" and history is not None:
                assert actual is expected, "restore the original host-history buffer identity"
                torch.testing.assert_close(actual, history)
            elif name == "initialized" and initialized is not None:
                assert actual is initialized
            else:
                assert actual is expected, f"restore {name}'s original attribute reference"


def _states(cache):
    return [cache.layers[0].conv_state, cache.layers[0].recurrent_state,
            *(state.conv_state for state in getattr(cache, "ple_states", {}).values())]


def _accepted(holder, cache):
    values = []
    for target, fields in holder._bindings:
        for name in fields:
            values.append(target.state_slots[name].stack[holder._committed_slot])
    return values


def _holder(model=None, cache=None, *, batch=1, steps=2, **kwargs):
    model = model or _Model()
    cache = cache or _cache(batch)
    return graphs.DecodeGraph(
        model=model, cache=cache, start=cache.length, steps=steps,
        token=torch.full((batch, 1), 3, dtype=torch.long), **kwargs,
    )


@pytest.mark.parametrize("pp_size,owns_head,head", [(2, True, True), (2, False, False), (1, False, True)])
def test_unsupported_stage_declines_before_any_mutation_or_cuda_probe(monkeypatch, pp_size, owns_head, head):
    model, cache = _Model(), _cache()
    model.pp_size, model.owns_head = pp_size, owns_head
    if not head:
        model.lm_head = None
    saved = _metadata(cache)
    states = [state.clone() for state in _states(cache)]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("stage guard must precede CUDA probe"))
    assert graphs.build(model=model, cache=cache, start=3, steps=2, token=torch.tensor([[3]])) is None
    _assert_metadata(saved)
    for actual, expected in zip(_states(cache), states):
        torch.testing.assert_close(actual, expected)
    assert model.forward_calls == []
    assert model.make_cache_calls == 0
    assert model.hasher.calls == []


@pytest.mark.parametrize("batch", [1, 2])
def test_width_reads_local_lm_head_without_scratch_forward(monkeypatch, batch):
    model, cache = _Model(), _cache(batch)
    monkeypatch.setattr(model, "forward", lambda *_a, **_k: pytest.fail("no width probe forward"))
    monkeypatch.setattr(graphs.DecodeGraph, "_capture", lambda holder: setattr(holder, "graph", object()))
    holder = _holder(model, cache, batch=batch)
    assert holder.logits_width() == model.lm_head.shape[0] == 5
    assert model.make_cache_calls == 0
    assert holder.width == 12
    holder.release()


@pytest.mark.parametrize("metadata", [False, True])
@pytest.mark.parametrize("initialized", [False, True])
def test_setup_only_writes_slot_one_and_restores_accepted_prefix(metadata, initialized):
    model, cache = _Model(), _cache(metadata=metadata, initialized=initialized)
    saved = _metadata(cache)
    original = [state.clone() for state in _states(cache)]
    refs = list(_states(cache))
    history_ref = cache.ple_token_history
    history_value = history_ref.clone()
    shared_rows = model.table._rows_buffer
    row_value = shared_rows.clone()
    holder = _holder(model, cache)
    assert holder._committed_slot == 0
    assert cache.length == cache.layers[1].length == 3
    assert cache.ple_token_history is history_ref
    torch.testing.assert_close(cache.ple_token_history, history_value)
    assert model.table._rows_buffer is shared_rows
    torch.testing.assert_close(shared_rows, row_value)
    assert len(model.setup_reads) == 2
    for first, second in zip(model.setup_reads[0], model.setup_reads[1]):
        torch.testing.assert_close(first, second)
    for live, ref, expected, accepted in zip(_states(cache), refs, original, _accepted(holder, cache)):
        assert live is ref
        torch.testing.assert_close(live, expected)
        torch.testing.assert_close(accepted, expected)
    assert cache.layers[0].initialized is initialized
    assert cache.layers[0].state_slot is cache.ple_states[1].state_slot is holder.slot
    assert len(cache.ple_states[1].state_slots["conv_state"]) == len(holder) + 2
    holder.release()
    _assert_metadata(saved)


def test_capture_failure_restores_transaction_including_live_buffers(monkeypatch):
    model, cache = _Model(), _cache(metadata=True)
    saved = _metadata(cache)
    original = [state.clone() for state in _states(cache)]
    refs = list(_states(cache))
    prefix = cache.ple_token_history.clone()
    table_rows = model.table._rows_buffer
    kv_prefix = cache.layers[1].k[:, :, :3].clone()

    def fail(holder):
        holder._run_step()
        # A failed implementation bypassing slots must still be repairable from the
        # small frozen buffers. No KV arena clone is needed for this tail write.
        for state in _states(cache):
            state.add_(999)
        saved[0][1]["ple_token_history"].fill_(99)
        cache.layers[1].rope_width = 99
        raise ValueError("injected capture refusal")

    monkeypatch.setattr(graphs.DecodeGraph, "_capture", fail)
    with pytest.raises(ValueError, match="capture refusal"):
        _holder(model, cache)
    _assert_metadata(saved)
    torch.testing.assert_close(cache.ple_token_history, prefix)
    torch.testing.assert_close(cache.layers[1].k[:, :, :3], kv_prefix)
    assert model.table._rows_buffer is table_rows
    for actual, ref, expected in zip(_states(cache), refs, original):
        assert actual is ref
        torch.testing.assert_close(actual, expected)


def test_partial_prepare_failure_removes_installed_layouts(monkeypatch):
    model, cache = _Model(), _cache()
    saved = _metadata(cache)
    layout = graphs.slot_layout
    calls = []

    def fail_second(fields):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("injected PLE layout allocation failure")
        return layout(fields)

    monkeypatch.setattr(graphs, "slot_layout", fail_second)
    with pytest.raises(ValueError, match="allocation failure"):
        _holder(model, cache)
    _assert_metadata(saved)
    assert model.forward_calls == []


def test_build_returns_none_only_after_successful_rollback(monkeypatch):
    model, cache = _Model(), _cache()
    saved = _metadata(cache)
    errors = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(graphs, "_report_capture_failure", errors.append)
    monkeypatch.setattr(graphs.DecodeGraph, "_capture", lambda _h: (_ for _ in ()).throw(ValueError("refused")))
    assert graphs.build(model=model, cache=cache, start=3, steps=2, token=torch.tensor([[3]])) is None
    assert len(errors) == 1
    _assert_metadata(saved)


def test_failed_device_restoration_never_reports_safe_eager_fallback(monkeypatch):
    model, cache = _Model(), _cache()
    saved = _metadata(cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(graphs.DecodeGraph, "_capture", lambda _h: (_ for _ in ()).throw(ValueError("refused")))
    monkeypatch.setattr(graphs.DecodeGraph, "_synchronize", lambda _h: (_ for _ in ()).throw(RuntimeError("poisoned device")))
    monkeypatch.setattr(graphs, "_report_capture_failure", lambda _e: pytest.fail("unsafe rollback cannot be masked"))
    with pytest.raises(graphs.GraphRollbackError, match="unsafe"):
        graphs.build(model=model, cache=cache, start=3, steps=2, token=torch.tensor([[3]]))
    _assert_metadata(saved)


def test_nested_layout_is_refused_until_original_graph_releases(monkeypatch):
    model, cache = _Model(), _cache()
    holder = _holder(model, cache)
    owned_layout = cache.layers[0].state_slots
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(graphs.GraphLifecycleError, match="release"):
        graphs.build(model=model, cache=cache, start=3, steps=2, token=torch.tensor([[3]]))
    assert cache._decode_graph_owner is holder
    assert cache.layers[0].state_slots is owned_layout
    holder.release()
    rebuilt = _holder(model, cache)
    rebuilt.release()


@pytest.mark.parametrize("steps", [1, 2])
def test_success_release_materializes_latest_committed_state_for_eager_handoff(steps):
    model, cache = _Model(), _cache(metadata=True)
    reference_model, reference = _Model(), _cache(metadata=True)
    saved = _metadata(cache)
    refs = list(_states(cache))
    holder = _holder(model, cache)
    seen = [10, 11, 12]
    for token in [3, 7][:steps]:
        seen.append(token)
        holder.fill_history(seen)
        expected = reference_model.forward(torch.tensor([[token]]), cache=reference, past_len=reference.length)
        answer = holder.replay(cache.length, torch.tensor([[token]]))
        assert answer.shape == (1, 1)
        torch.testing.assert_close(holder.outputs, expected[:, 0])
        for accepted, eager in zip(_accepted(holder, cache), _states(reference)):
            torch.testing.assert_close(accepted, eager)
    holder.release()
    _assert_metadata(saved, length=reference.length, history=reference.ple_token_history, initialized=True)
    for live, original_ref, eager in zip(_states(cache), refs, _states(reference)):
        assert live is original_ref
        torch.testing.assert_close(live, eager)
    holder.release()
    actual = model.forward(torch.tensor([[9]]), cache=cache, past_len=cache.length)
    expected = reference_model.forward(torch.tensor([[9]]), cache=reference, past_len=reference.length)
    torch.testing.assert_close(actual, expected)


def test_rotation_seeds_new_slot_zero_from_released_committed_state():
    model, cache = _Model(), _cache()
    reference_model, reference = _Model(), _cache()
    seen = [10, 11, 12]
    for tokens in ([3, 7], [9, 5]):
        holder = _holder(model, cache)
        before = [state.clone() for state in _states(cache)]
        for accepted, value in zip(_accepted(holder, cache), before):
            torch.testing.assert_close(accepted, value)
        for token in tokens:
            seen.append(token)
            holder.fill_history(seen)
            expected = reference_model.forward(torch.tensor([[token]]), cache=reference, past_len=reference.length)
            holder.replay(cache.length, torch.tensor([[token]]))
            torch.testing.assert_close(holder.outputs, expected[:, 0])
        holder.release()
        for live, eager in zip(_states(cache), _states(reference)):
            torch.testing.assert_close(live, eager)


@pytest.mark.parametrize("prior_success", [False, True])
def test_margin_handback_releases_previous_commit_before_same_position_eager(monkeypatch, prior_success):
    model, cache = _Model(), _cache()
    reference_model, reference = _Model(), _cache()
    holder = _holder(model, cache)
    seen = [10, 11, 12]
    if prior_success:
        seen.append(3)
        holder.fill_history(seen)
        holder.replay(3, torch.tensor([[3]]))
        reference_model.forward(torch.tensor([[3]]), cache=reference, past_len=3)
    token = 7 if prior_success else 3
    seen.append(token)
    holder.fill_history(seen)
    position = cache.length
    monkeypatch.setattr(holder, "_should_handback", lambda: True)
    assert holder.replay(position, torch.tensor([[token]])) is None
    assert holder.graph is None
    assert cache.length == reference.length == position
    torch.testing.assert_close(cache.ple_token_history, reference.ple_token_history)
    for live, eager in zip(_states(cache), _states(reference)):
        torch.testing.assert_close(live, eager)
    holder.release()  # Runtime explicitly calls release again after a None replay.
    actual = model.forward(torch.tensor([[token]]), cache=cache, past_len=position)
    expected = reference_model.forward(torch.tensor([[token]]), cache=reference, past_len=position)
    torch.testing.assert_close(actual, expected)


def test_exhausted_span_releases_latest_commit_before_eager():
    model, cache = _Model(), _cache()
    holder = _holder(model, cache)
    seen = [10, 11, 12]
    for token in (3, 7):
        seen.append(token)
        holder.fill_history(seen)
        holder.replay(cache.length, torch.tensor([[token]]))
    committed = [value.clone() for value in _accepted(holder, cache)]
    assert holder.replay(5, torch.tensor([[9]])) is None
    assert cache.length == 5
    assert holder.graph is None
    for actual, expected in zip(_states(cache), committed):
        torch.testing.assert_close(actual, expected)
    holder.release()
    assert holder.replay(5, torch.tensor([[9]])) is None


def test_replaying_same_position_does_not_advance_recurrence_twice():
    model, cache = _Model(), _cache()
    holder = _holder(model, cache)
    outputs = []
    for _ in range(2):
        holder.fill_history([10, 11, 12, 3])
        holder.replay(3, torch.tensor([[3]]))
        outputs.append(holder.outputs.clone())
    torch.testing.assert_close(*outputs)
    reference_model, reference = _Model(), _cache()
    reference_model.forward(torch.tensor([[3]]), cache=reference, past_len=3)
    holder.release()
    for actual, expected in zip(_states(cache), _states(reference)):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("raw_dtype", [torch.float32, torch.uint8])
def test_private_ple_rows_refresh_without_replaying_cpu_hash_or_shared_buffer_pollution(raw_dtype):
    model, cache = _Model(raw_dtype), _cache()
    reference_model, reference = _Model(raw_dtype), _cache()
    original_table_buffer = model.table._rows_buffer
    holder = _holder(model, cache)
    private = holder._ple_rows[0]["buffer"]
    assert model.table._rows_buffer is original_table_buffer
    assert holder.graph.sources[1].data_ptr() == private.data_ptr()
    seen = [10, 11, 12]
    expected_global = original_table_buffer
    for token in (3, 7):
        seen.append(token)
        holder.fill_history(seen)
        row_ids = model.hasher.impl.row_ids(holder.history, 1)
        torch.testing.assert_close(private, model.table.weight.index_select(0, row_ids.reshape(-1)))
        expected = reference_model.forward(torch.tensor([[token]]), cache=reference, past_len=reference.length)
        # An unrelated cache can reuse/reallocate the shared table buffer. The graph
        # must keep reading its own captured address, not this last eager lookup.
        other = _cache(batch=4)
        model.forward(torch.full((4, 1), 9, dtype=torch.long), cache=other, past_len=3)
        expected_global = model.table._rows_buffer
        hash_calls, gather_calls = len(model.hasher.calls), len(model.table.gathers)
        holder.replay(cache.length, torch.tensor([[token]]))
        assert len(model.hasher.calls) == hash_calls
        assert len(model.table.gathers) == gather_calls
        torch.testing.assert_close(holder.outputs, expected[:, 0])
        assert model.table._rows_buffer is expected_global
    holder.release()
    assert model.table._rows_buffer is expected_global, "release must not restore a stale shared scratch pointer"


def test_ple_row_staging_respects_eos_hash_boundaries_and_position():
    model, cache = _Model(), _cache()
    holder = _holder(model, cache)
    holder.fill_history([7, 0, 3])
    ids = model.hasher.impl.row_ids(holder.history, 1)
    reset_ids = model.hasher.impl.row_ids(torch.tensor([[0, 0, 3]]), 1)
    torch.testing.assert_close(ids, reset_ids)
    private = holder._ple_rows[0]["buffer"]
    torch.testing.assert_close(private, model.table.weight.index_select(0, ids.reshape(-1)))
    holder.replay(3, torch.tensor([[3]]))
    assert int(holder.position) == 3
    assert holder._committed_slot == 1
    torch.testing.assert_close(cache.ple_token_history, torch.tensor([[0, 3]]))
    holder.release()


@pytest.mark.parametrize("prior_success", [False, True])
def test_ple_replay_requires_fresh_history_instead_of_silent_placeholder_rows(prior_success):
    model, cache = _Model(), _cache()
    holder = _holder(model, cache)
    if prior_success:
        holder.fill_history([10, 11, 12, 3])
        holder.replay(3, torch.tensor([[3]]))
    position = cache.length
    with pytest.raises(ValueError, match="fill_history"):
        holder.replay(position, torch.tensor([[7 if prior_success else 3]]))
    assert cache.length == position
    assert holder.graph is None
    assert not hasattr(cache.layers[0], "state_slots")


@pytest.mark.parametrize("flat", [False, True])
def test_streams_and_selection_keep_canonical_batch_shapes(flat):
    model, cache = _Model(flat=flat), _cache(batch=2)
    selections = []

    def select(logits):
        assert logits.shape == (2, 5)
        selections.append(logits.clone())
        return logits.argmax(-1, keepdim=True)

    holder = _holder(model, cache, batch=2, wants_streams=True, select=select)
    holder.fill_history([[10, 11, 12, 3], [5, 6, 7, 4]])
    token = holder.replay(3, torch.tensor([[3], [4]]))
    assert token.shape == (2, 1)
    assert holder.streams().shape == (2, 2)
    assert len(selections) == 1
    holder.release()


def test_invalid_token_shape_releases_without_implicit_broadcast():
    model, cache = _Model(), _cache()
    saved = _metadata(cache)
    holder = _holder(model, cache)
    holder.fill_history([10, 11, 12, 3])
    with pytest.raises(ValueError, match="int64"):
        holder.replay(3, torch.tensor([3]))
    _assert_metadata(saved)
    assert holder.graph is None


def test_small_state_snapshot_never_clones_kv_or_rotary_arenas(monkeypatch):
    model, cache = _Model(), _cache()
    arenas = {id(cache.cos), id(cache.sin), id(cache.layers[1].k), id(cache.layers[1].v),
              id(cache.layers[1].index_k)}
    clone = torch.Tensor.clone

    def checked_clone(tensor, *args, **kwargs):
        assert id(tensor) not in arenas, "graph transactions may not clone an arena"
        return clone(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "clone", checked_clone)
    holder = _holder(model, cache)
    holder.release()


def test_unsupported_ple_table_declines_after_clean_rollback(monkeypatch):
    model, cache = _Model(), _cache()
    saved = _metadata(cache)
    model.layers[1].ple.ngram_table = lambda _ids: torch.zeros(1, 1, 2, 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    errors = []
    monkeypatch.setattr(graphs, "_report_capture_failure", errors.append)
    assert graphs.build(model=model, cache=cache, start=3, steps=2, token=torch.tensor([[3]])) is None
    _assert_metadata(saved)
    assert len(errors) == 1
    assert "staging" in str(errors[0])


def test_fake_cache_without_ple_states_remains_supported():
    model, cache = _Model(with_ple=False), _cache(with_ple=False)
    del cache.ple_states
    holder = _holder(model, cache)
    holder.fill_history([10, 11, 12, 3])
    assert holder.replay(3, torch.tensor([[3]])).shape == (1, 1)
    holder.release()


def test_failed_cpu_row_staging_releases_committed_ownership(monkeypatch):
    model, cache = _Model(), _cache()
    saved = _metadata(cache)
    holder = _holder(model, cache)
    holder.fill_history([10, 11, 12, 3])
    monkeypatch.setattr(
        model.hasher, "row_ids", lambda *_a: (_ for _ in ()).throw(ValueError("lookup failed"))
    )
    with pytest.raises(ValueError, match="lookup failed"):
        holder.fill_history([10, 11, 12, 3, 7])
    assert holder.graph is None
    assert not holder._history_ready
    _assert_metadata(saved)


@pytest.mark.parametrize("capture_failure", [False, True])
def test_restore_handles_inference_created_cache_buffers_outside_inference_mode(
    monkeypatch, capture_failure
):
    model = _Model()
    with torch.inference_mode():
        cache = _cache()
    refs = list(_states(cache))
    assert all(value.is_inference() for value in refs)
    assert cache.ple_token_history.is_inference()
    original = [value.clone() for value in refs]
    saved = _metadata(cache)
    if capture_failure:
        def fail(holder):
            with torch.inference_mode():
                holder._run_step()
            raise ValueError("inference capture refused")

        monkeypatch.setattr(graphs.DecodeGraph, "_capture", fail)
        with pytest.raises(ValueError, match="capture refused"):
            _holder(model, cache)
        expected = original
        _assert_metadata(saved)
    else:
        holder = _holder(model, cache)
        holder.fill_history([10, 11, 12, 3])
        holder.replay(3, torch.tensor([[3]]))
        expected = [value.clone() for value in _accepted(holder, cache)]
        # This call intentionally occurs outside torch.inference_mode(). The
        # implementation must guard writes into the original inference buffers.
        assert not torch.is_inference_mode_enabled()
        holder.release()
        _assert_metadata(saved, length=4, history=torch.tensor([[12, 3]]), initialized=True)
    for actual, original_ref, wanted in zip(_states(cache), refs, expected):
        assert actual is original_ref and actual.is_inference()
        torch.testing.assert_close(actual, wanted)


def test_past_budget_guard_remains_an_explicit_validation_boundary(monkeypatch):
    model, cache = _Model(), _cache()
    model.config.indexer_budget = 4
    saved = _metadata(cache)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="validation boundary"):
        graphs.build(model=model, cache=cache, start=3, steps=2, token=torch.tensor([[3]]))
    _assert_metadata(saved)
    monkeypatch.setenv("POCKETLLM_QWEN4_GRAPH_PAST_BUDGET", "1")
    holder = graphs.build(model=model, cache=cache, start=3, steps=2, token=torch.tensor([[3]]))
    assert holder is not None
    holder.release()
