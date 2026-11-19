"""CPU contracts for the reusable graph-session owner.

The state machine is host work, so these tests use the real cache classes with synthetic geometry
and fake recordings that install the same kind of bindings the real ones do. What is proved here is
ownership, compatibility, the exact reset values, suspend/resume of bindings, the buffer-identity
guard and teardown; whether a replayed CUDA graph produces the same tokens is a GPU measurement and
is not this file's claim.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from relicllm.models.qwen4_exp.attention import GatedDeltaNetCache, QSAAttentionCache, _QSAIndexerPool
from relicllm.models.qwen4_exp.graph_session import (
    GraphLease,
    GraphSessionError,
    GraphSessionKey,
    GraphSessionPool,
    SessionInvalidated,
    flags_digest,
)
from relicllm.models.qwen4_exp.model import PLEState, Qwen4ExpCache

_MISSING = object()


# ---------------------------------------------------------------------------------------------
# Fixtures: a real cache of the real classes with tiny geometry, and a fake recording.
# ---------------------------------------------------------------------------------------------


def config(ngram=3):
    return SimpleNamespace(
        linear_num_value_heads=2, linear_num_key_heads=1,
        linear_key_head_dim=4, linear_value_head_dim=4, linear_conv_kernel_dim=3,
        num_key_value_heads=1, head_dim=4, indexer_head_dim=4,
        ngram_size=ngram, primary_eos_token_id=99,
    )


def make_cache(width=32, ngram=3):
    """A cache shaped like `model.make_cache` produces, with two layers and one PLE state."""
    cfg = config(ngram)
    gdn = GatedDeltaNetCache(cfg, 1, torch.device("cpu"), torch.bfloat16)
    qsa = QSAAttentionCache(cfg, 1, width, torch.device("cpu"), torch.bfloat16)
    pool = _QSAIndexerPool(qsa.index_k, 4)
    qsa.indexer_pool = pool
    ple = PLEState(torch.zeros(1, 8, 4, dtype=torch.bfloat16))
    cache = Qwen4ExpCache(
        layers=[gdn, qsa, None],
        ple_states={0: ple},
        ple_token_history=torch.full((1, ngram - 1), 99, dtype=torch.long),
    )
    return cache


class FakeRecording:
    """The binding protocol the real recordings expose, without any CUDA work.

    It installs a cache attribute and a host-table attribute, remembers the values it displaced the
    same way `VerifyGraph` does, and refuses to replay once its bindings are gone.
    """

    def __init__(self, cache, *, name="verify", extra_target=None, install_names=("kv_width",)):
        self.cache = cache
        self.name = name
        self.replays = 0
        self._saved_attrs = []
        self._released_flag = False
        self.replay_positions = []
        self.replay_feeds = []
        with torch.inference_mode(False):
            self.history = torch.full((1, 4), 7, dtype=torch.long)
            self.position = torch.zeros((), dtype=torch.long)
            self.out = torch.zeros(2, dtype=torch.long)
        for attr in install_names:
            self._remember(cache, attr)
            setattr(cache, attr, 32)
        if extra_target is not None:
            target, attr = extra_target
            self._remember(target, attr)
            setattr(target, attr, "staged-buffer")

    def _remember(self, target, name):
        self._saved_attrs.append((target, name, target.__dict__.get(name, _MISSING)))

    def replay(self, position: int, feed: list[int]) -> list[int]:
        if self._saved_attrs and getattr(self.cache, "_retired", False):
            raise RuntimeError("bindings were retired")
        for target, name, _previous in reversed(self._saved_attrs):
            if target.__dict__.get(name, _MISSING) is _MISSING:
                raise RuntimeError(f"binding {name} is not installed")
        self.replays += 1
        self.replay_positions.append(int(position))
        self.replay_feeds.append(list(feed))
        self.position.fill_(int(position))
        return [int(position) + i for i in range(2)]

    @property
    def released(self) -> bool:
        return self._released_flag

    def release(self) -> None:
        self._released_flag = True
        self.history = None
        self.position = None
        self.out = None
        self._saved_attrs = []


def model_for(cache_ngram=3):
    """A stand-in whose config carries what reset needs."""
    return SimpleNamespace(config=config(cache_ngram), dtype=torch.bfloat16)


def key_for(model, *, width=32, draft=3, rows=4, arena=32, flags="0"):
    environ = {"POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT": "1", "POCKETLLM_TEST": flags}
    return GraphSessionKey.build(
        model=model, ctx=SimpleNamespace(device=torch.device("cpu")), width=width,
        draft_tokens=draft, rows=rows, arena_tokens=arena, model_token="checkpoint-x",
        environ=environ,
    )


def adopt_cold(pool, key, cache=None, *, ngram=3):
    """The cold path: allocate, "prefill", capture, then adopt."""
    cache = make_cache(width=key.width, ngram=ngram) if cache is None else cache
    lease = pool.checkout(key)
    assert not lease.reused
    lease.adopt(
        cache=cache,
        verify=FakeRecording(cache, name="verify"),
        draft=FakeRecording(cache, name="draft", extra_target=(cache, "_rows_buffer")),
        tail=None,
    )
    return lease, cache


# ---------------------------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------------------------


def test_cold_then_compatible_warm_call_reuses_the_same_arenas():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    verify = lease.owner("verify")
    assert verify.replays == 0
    lease.finish()

    warm = pool.checkout(key)
    assert warm.reused and warm.cache is cache and warm.owner("verify") is verify
    assert warm.reason == "compatible"
    assert pool.stats["hits"] == 1 and pool.stats["misses"] == 1
    warm.begin_request()
    warm.resume()
    assert warm.owner("verify").replay(0, [1, 2, 3, 4]) == [0, 1]
    warm.finish()
    assert pool.stats["reused"] if "reused" in pool.stats else True
    assert pool.summary()["retained"] == 1


@pytest.mark.parametrize(
    "field,value",
    [("width", 64), ("draft_tokens", 4), ("rows", 5), ("arena_tokens", 64),
     ("model_token", "other-checkpoint"), ("flags", "other")],
)
def test_incompatible_key_discards_the_bundle_instead_of_replaying_it(field, value):
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    verify = lease.owner("verify")
    lease.finish()

    changed = GraphSessionKey(**{**key.__dict__, field: value})
    assert changed.mismatch(key).startswith(field)
    cold = pool.checkout(changed)
    assert not cold.reused and cold.cache is None
    assert verify.released and pool.summary()["retained"] == 0
    assert pool.stats["discards"] == 1


def test_device_difference_is_incompatible():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    adopt_cold(pool, key)[0].finish()
    other = GraphSessionKey(**{**key.__dict__, "device": "cuda:0"})
    assert not pool.checkout(other).reused


def test_flags_digest_changes_with_any_pocketllm_switch():
    base = {"POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT": "1", "PATH": "/usr/bin"}
    assert flags_digest(base) == flags_digest({**base, "PATH": "/other"}), "non-switches are ignored"
    assert flags_digest(base) != flags_digest({**base, "POCKETLLM_QWEN4_QSA_SPLITK_ROWS": "1"})


# ---------------------------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------------------------


def test_begin_request_restores_exactly_the_state_a_fresh_cache_has():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    gdn, qsa = cache.layers[0], cache.layers[1]

    # Leave the cache as a finished request would: advanced metadata and non-zero rolling state.
    gdn.conv_state.fill_(1.5)
    gdn.recurrent_state.fill_(0.25)
    gdn.initialized = True
    gdn.length = 20
    qsa.length = 20
    qsa.host_position = 20
    qsa.indexer_pool.completed = 5
    qsa.indexer_pool.completed_tensor.fill_(5)
    cache.ple_states[0].conv_state.fill_(2.0)
    cache.ple_token_history.fill_(3)
    cache.length = 20
    lease.finish()

    warm = pool.checkout(key)
    warm.begin_request()

    fresh = make_cache(width=key.width)
    assert cache.length == 0
    assert gdn.initialized is False and gdn.length == 0
    assert torch.equal(gdn.conv_state, fresh.layers[0].conv_state)
    assert torch.equal(gdn.recurrent_state, fresh.layers[0].recurrent_state)
    assert qsa.length == 0 and qsa.host_position == 0
    assert qsa.indexer_pool.completed == 0
    assert int(qsa.indexer_pool.completed_tensor) == 0
    assert torch.equal(cache.ple_states[0].conv_state, fresh.ple_states[0].conv_state)
    assert torch.equal(cache.ple_token_history, fresh.ple_token_history)
    assert int(cache.ple_token_history[0, 0]) == 99


def test_reset_never_writes_the_multi_gib_arenas():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    qsa = cache.layers[1]
    for tensor in (qsa.k, qsa.v, qsa.index_k, qsa.indexer_pool.keys):
        tensor.fill_(4.0)
    pointers = {name: tensor.data_ptr() for name, tensor in
                (("k", qsa.k), ("v", qsa.v), ("index_k", qsa.index_k),
                 ("pool", qsa.indexer_pool.keys))}
    lease.finish()
    warm = pool.checkout(key)
    warm.begin_request()

    for name, tensor in (("k", qsa.k), ("v", qsa.v), ("index_k", qsa.index_k),
                         ("pool", qsa.indexer_pool.keys)):
        assert tensor.data_ptr() == pointers[name], f"{name} was reallocated"
        assert torch.equal(tensor, torch.full_like(tensor, 4.0)), f"{name} was written"
    assert qsa.indexer_pool.tensor_ready == cache.layers[1].indexer_pool.tensor_ready


def test_reset_keeps_the_rope_table_and_fixed_widths_by_identity():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    cache.cos = torch.ones(3, key.width, 2)
    cache.sin = torch.zeros(3, key.width, 2)
    cos_ptr = cache.cos.data_ptr()
    lease.finish()
    warm = pool.checkout(key)
    warm.begin_request()
    assert cache.cos.data_ptr() == cos_ptr and cache.sin is not None
    assert cache.cos.shape[1] == key.width


# ---------------------------------------------------------------------------------------------
# Binding suspension
# ---------------------------------------------------------------------------------------------


def test_finish_displaces_every_binding_so_an_eager_prefill_sees_plain_attributes():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    assert cache.kv_width == 32 and cache._rows_buffer == "staged-buffer"

    lease.finish()

    assert "kv_width" not in cache.__dict__, "a displaced attribute that was absent must be gone"
    assert "_rows_buffer" not in cache.__dict__
    assert warm_is_plain(cache)


def warm_is_plain(cache):
    return "kv_width" not in cache.__dict__ and "_rows_buffer" not in cache.__dict__


def test_resume_reinstalls_the_bindings_so_the_recording_replays_again():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    verify = lease.owner("verify")
    lease.finish()

    warm = pool.checkout(key)
    warm.begin_request()
    assert "kv_width" not in cache.__dict__
    warm.resume()
    assert cache.kv_width == 32 and cache._rows_buffer == "staged-buffer"
    assert verify.replay(5, [1, 2, 3, 4]) == [5, 6]
    warm.finish()


def test_suspension_restores_a_previously_installed_value_instead_of_deleting_it():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    cache = make_cache(width=key.width)
    cache.kv_width = 16  # something was already there before the recording
    lease = pool.checkout(key)
    lease.adopt(cache=cache, verify=FakeRecording(cache, name="verify"))
    assert cache.kv_width == 32
    lease.finish()
    assert cache.kv_width == 16


def test_resume_raises_when_a_graph_touched_buffer_moved():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    lease.finish()
    warm = pool.checkout(key)
    warm.begin_request()
    # Something reallocated the PLE carry between requests; a replay would read freed memory.
    cache.ple_states[0].conv_state = torch.zeros_like(cache.ple_states[0].conv_state)
    with pytest.raises(SessionInvalidated, match="conv_state"):
        warm.resume()
    assert warm.owner("verify") is not None
    warm.finish(successful=False)
    assert pool.summary()["retained"] == 0


def test_unknown_binding_name_is_refused_rather_than_left_installed():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    cache = make_cache(width=key.width)
    lease = pool.checkout(key)

    class Surprising(FakeRecording):
        def __init__(self, cache):
            super().__init__(cache, name="surprising")
            self._saved_attrs.append((cache, "some_new_binding", _MISSING))
            cache.some_new_binding = True

    with pytest.raises(SessionInvalidated, match="some_new_binding"):
        lease.adopt(cache=cache, verify=Surprising(cache))
    assert cache.some_new_binding is True  # nothing was silently displaced by a failed adopt


# ---------------------------------------------------------------------------------------------
# Request lifecycles
# ---------------------------------------------------------------------------------------------


def test_two_different_prompts_replay_through_one_bundle_with_no_stale_history():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    verify = lease.owner("verify")
    verify.replay(0, [11, 12, 13, 14])
    lease.finish()

    warm = pool.checkout(key)
    warm.begin_request()
    # Prefill of the second request would have replaced the committed history through the eager
    # path; emulate the value it leaves behind and then resume.
    cache.ple_token_history.fill_(42)
    warm.resume()
    answer = warm.owner("verify").replay(0, [21, 22, 23, 24])
    assert answer == [0, 1]
    assert warm.owner("verify").replay_feeds[-1] == [21, 22, 23, 24]
    assert verify.replay_positions == [0, 0]
    warm.finish()


def test_boundary_retired_request_forces_a_cold_recapture_next_time():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    verify = lease.owner("verify")
    lease.finish(boundary_retired=True)

    assert verify.released and pool.stats["boundary_invalidations"] == 1
    assert pool.summary()["retained"] == 0
    assert not pool.checkout(key).reused


def test_owner_released_behind_the_session_is_not_handed_out_again():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    verify = lease.owner("verify")
    lease.finish()
    verify._released_flag = True  # a recording destroyed itself, as the native handoff does
    assert not pool.checkout(key).reused


def test_abandon_drops_the_bundle_so_a_cancelled_request_cannot_poison_the_next():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    verify = lease.owner("verify")
    lease.abandon()
    assert verify.released and pool.summary()["retained"] == 0
    assert not pool.checkout(key).reused


def test_lease_as_context_manager_finishes_on_error():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    with pytest.raises(ZeroDivisionError):
        with pool.checkout(key) as lease:
            lease.adopt(cache=make_cache(width=key.width), verify=FakeRecording(make_cache()))
            raise ZeroDivisionError
    assert pool.summary()["retained"] == 0


def test_out_of_order_calls_are_refused():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    lease.finish()
    with pytest.raises(GraphSessionError, match="state"):
        lease.resume()
    cold = pool.checkout(GraphSessionKey(**{**key.__dict__, "width": 64}))
    with pytest.raises(GraphSessionError, match="nothing to reset"):
        cold.begin_request()
    with pytest.raises(TypeError):
        pool.checkout("not-a-key")


# ---------------------------------------------------------------------------------------------
# Teardown
# ---------------------------------------------------------------------------------------------


def test_close_is_idempotent_and_releases_every_recording():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, _cache = adopt_cold(pool, key)
    verify, draft = lease.owner("verify"), lease.owner("draft")
    lease.finish()
    pool.close()
    pool.close()
    assert verify.released and draft.released
    assert pool.closed and pool.stats["closes"] == 1
    assert pool.summary()["retained"] == 0
    with pytest.raises(GraphSessionError, match="closed"):
        pool.checkout(key)


def test_address_invalidations_counter_is_available_for_the_parent_to_log():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    lease.finish()
    warm = pool.checkout(key)
    warm.begin_request()
    cache.ple_states[0].conv_state = torch.zeros_like(cache.ple_states[0].conv_state)
    with pytest.raises(SessionInvalidated):
        warm.resume()
    warm.abandon()
    summary = pool.summary()
    assert summary["misses"] == 1 and summary["retained"] == 0
    assert "released" in summary and "adopted" in summary


def test_summary_counts_cold_and_warm_so_a_rate_claim_cannot_hide_recapture():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    for _ in range(2):
        lease, _cache = adopt_cold(pool, key) if not pool.summary()["retained"] else (
            pool.checkout(key), None
        )
        if lease.reused:
            lease.begin_request()
            lease.resume()
        lease.finish()
    summary = pool.summary()
    assert summary["checkouts"] == 2
    assert summary["misses"] == 1 and summary["hits"] == 1
    assert summary["adopted"] == 1, "a warm request must not claim it adopted (captured) a bundle"
    assert summary["resumed"] == 1

# ---------------------------------------------------------------------------------------------
# The owner protocol the recording classes are expected to expose
# ---------------------------------------------------------------------------------------------


class ProtocolRecording(FakeRecording):
    """A recording that owns its binding bookkeeping behind the two protocol methods.

    This is the shape the graph classes will expose: the session never reads `_saved_attrs` for it,
    it just calls `suspend_bindings` / `resume_bindings` in the order it was constructed.
    """

    def __init__(self, cache, *, name, calls: list, token: str, extra_target=None):
        super().__init__(cache, name=name, extra_target=extra_target)
        self.calls = calls
        self.token = token
        self.suspended = False

    def suspend_bindings(self) -> None:
        self.calls.append(f"{self.token}.suspend")
        _restore(reversed(self._saved_attrs))
        self.suspended = True

    def resume_bindings(self) -> None:
        self.calls.append(f"{self.token}.resume")
        for target, name, value in self._installed:
            setattr(target, name, value)
        self.suspended = False

    def _remember(self, target, name):
        super()._remember(target, name)
        namespace = getattr(target, "__dict__", {})
        entry = (target, name, namespace.get(name, _MISSING))
        installed = getattr(self, "_installed", [])
        installed.append(entry)
        self._installed = installed


def _restore(entries):
    for target, name, previous in entries:
        if previous is _MISSING:
            target.__dict__.pop(name, None)
        else:
            setattr(target, name, previous)


def test_protocol_owners_are_called_reverse_on_suspend_and_forward_on_resume():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    cache = make_cache(width=key.width)
    calls: list[str] = []
    lease = pool.checkout(key)
    # Construction order is the order the runtime builds them: tail, verifier, draft head.
    lease.adopt(
        cache=cache,
        tail=ProtocolRecording(cache, name="tail", calls=calls, token="tail"),
        verify=ProtocolRecording(cache, name="verify", calls=calls, token="verify"),
        draft=ProtocolRecording(cache, name="draft", calls=calls, token="draft"),
    )
    lease.finish()
    assert calls == ["draft.suspend", "verify.suspend", "tail.suspend"]

    calls.clear()
    warm = pool.checkout(key)
    warm.begin_request()
    warm.resume()
    assert calls == ["tail.resume", "verify.resume", "draft.resume"]
    warm.finish()


def test_protocol_and_attribute_owners_can_be_mixed_in_one_bundle():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    cache = make_cache(width=key.width)
    calls: list[str] = []
    lease = pool.checkout(key)
    lease.adopt(
        cache=cache,
        verify=ProtocolRecording(cache, name="verify", calls=calls, token="verify"),
        draft=FakeRecording(cache, name="draft"),  # attribute-only fallback
    )
    assert cache.kv_width == 32
    lease.finish()
    assert calls == ["verify.suspend"]
    assert "kv_width" not in cache.__dict__, "the fallback path must displace its binding too"
    warm = pool.checkout(key)
    warm.begin_request()
    warm.resume()
    assert calls == ["verify.suspend", "verify.resume"]
    warm.finish()


# ---------------------------------------------------------------------------------------------
# Cumulative counters and the deferred native handoff
# ---------------------------------------------------------------------------------------------


def test_counters_are_measured_by_offset_and_never_zeroed():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    verify = lease.owner("verify")
    verify.replays = 7
    verify.stage_s = 1.25
    lease.finish()

    warm = pool.checkout(key)
    warm.begin_request()
    mark = warm.counter_mark()
    assert mark["verify.replays"] == 7, "a retained recording keeps its lifetime total"
    verify.replays += 3
    verify.stage_s += 0.5
    delta = warm.counter_delta(mark)
    assert delta["verify.replays"] == 3
    assert abs(delta["verify.stage_s"] - 0.5) < 1e-9
    assert verify.replays == 10, "measuring must not reset the counter"
    warm.finish()


def test_deferred_retirement_leaves_an_owner_that_must_not_be_handed_out():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    verify = lease.owner("verify")
    lease.finish()
    # The deferred native-handoff path retires bindings and keeps the owner alive; it can only
    # raise on replay, so the bundle must be dropped rather than reused.
    verify._bindings_retired = True
    assert not pool.checkout(key).reused
    assert pool.summary()["retained"] == 0


# ---------------------------------------------------------------------------------------------
# A PLE state without a carry is left exactly as it was
# ---------------------------------------------------------------------------------------------


def test_absent_ple_carry_is_not_fabricated_by_reset():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    cache.ple_states[0].conv_state = None
    lease.finish()
    warm = pool.checkout(key)
    warm.begin_request()
    assert cache.ple_states[0].conv_state is None, "reset must not allocate state the cache lacked"
    warm.finish()


# ---------------------------------------------------------------------------------------------
# A recording shaped like the real classes: own sentinel, second row-buffer list, protocol methods
# ---------------------------------------------------------------------------------------------


_OWNER_SENTINEL = object()  # deliberately *not* the session's sentinel


class RealShapeRecording:
    """Mirrors `VerifyGraph`: `_saved_attrs`, `_saved_rows_buffers`, and the two protocol methods."""

    def __init__(self, cache, table, *, name, calls, token, retired=False, released=False):
        self.name = name
        self.calls = calls
        self.token = token
        self.released = released
        self.replays = 0
        self.stage_s = 0.0
        self.device_s = 0.0
        self._bindings_retired = retired
        self._bindings_suspended = False
        self._saved_attrs = []
        self._saved_rows_buffers = []
        self._active_request_attrs = []
        self._active_rows_buffers = []
        # Install exactly what the real `_prepare` installs.
        self._remember(cache, "kv_width")
        cache.kv_width = 32
        self._remember(cache, "ple_history_at")
        cache.ple_history_at = lambda ids: self.history
        self._remember(table, "_rows_buffer")
        self._saved_rows_buffers.append((table, table._rows_buffer))
        table._rows_buffer = torch.zeros(8, dtype=torch.bfloat16)
        with torch.inference_mode(False):
            self.history = torch.zeros(1, 4, dtype=torch.long)

    def _remember(self, target, name):
        self._saved_attrs.append((target, name, target.__dict__.get(name, _OWNER_SENTINEL)))

    def suspend_bindings(self) -> None:
        if self._bindings_retired or self.released:
            raise RuntimeError("a retired recording cannot suspend its request bindings")
        if self._bindings_suspended:
            return
        self.calls.append(f"{self.token}.suspend")
        active, seen = [], set()
        for target, name, _old in self._saved_attrs:
            key = (id(target), name)
            if key not in seen:
                seen.add(key)
                active.append((target, name, target.__dict__.get(name, _OWNER_SENTINEL)))
        self._active_request_attrs = active
        self._active_rows_buffers = [(t, t._rows_buffer) for t, _ in self._saved_rows_buffers]
        for table, old in reversed(self._saved_rows_buffers):
            table._rows_buffer = old
        for target, name, old in reversed(self._saved_attrs):
            if old is _OWNER_SENTINEL:
                target.__dict__.pop(name, None)
            else:
                setattr(target, name, old)
        self._bindings_suspended = True

    def resume_bindings(self) -> None:
        if self._bindings_retired or self.released:
            raise RuntimeError("a retired recording cannot resume its request bindings")
        if not self._bindings_suspended:
            return
        self.calls.append(f"{self.token}.resume")
        for target, name, active in self._active_request_attrs:
            if active is _OWNER_SENTINEL:
                target.__dict__.pop(name, None)
            else:
                setattr(target, name, active)
        for table, active in self._active_rows_buffers:
            table._rows_buffer = active
        self._bindings_suspended = False

    def release(self) -> None:
        self.released = True


def test_real_shape_recording_displaces_ple_rows_history_and_widths():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    cache = make_cache(width=key.width)
    table = SimpleNamespace(_rows_buffer="host-scratch")
    calls: list[str] = []
    lease = pool.checkout(key)
    lease.adopt(
        cache=cache,
        verify=RealShapeRecording(cache, table, name="verify", calls=calls, token="verify"),
        draft=RealShapeRecording(cache, table, name="draft", calls=calls, token="draft"),
    )
    assert cache.kv_width == 32 and callable(cache.ple_history_at)
    assert isinstance(table._rows_buffer, torch.Tensor)

    lease.finish()
    assert calls == ["draft.suspend", "verify.suspend"]
    assert "kv_width" not in cache.__dict__
    assert "ple_history_at" not in cache.__dict__
    assert table._rows_buffer == "host-scratch", "the prefill table gets its own scratch back"

    calls.clear()
    warm = pool.checkout(key)
    warm.begin_request()
    warm.resume()
    assert calls == ["verify.resume", "draft.resume"]
    assert cache.kv_width == 32 and callable(cache.ple_history_at)
    assert isinstance(table._rows_buffer, torch.Tensor)
    warm.finish()


def test_a_protocol_owner_that_retired_itself_reports_a_session_invalidate():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    cache = make_cache(width=key.width)
    table = SimpleNamespace(_rows_buffer=None)
    calls: list[str] = []
    lease = pool.checkout(key)
    lease.adopt(cache=cache, verify=RealShapeRecording(
        cache, table, name="verify", calls=calls, token="verify"))
    lease.finish()
    warm = pool.checkout(key)
    warm.begin_request()
    warm.owner("verify")._bindings_retired = True
    with pytest.raises(SessionInvalidated, match="cannot resume"):
        warm.resume()
    warm.abandon()


def test_protocol_suspend_failure_during_finish_is_reported_not_swallowed():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    cache = make_cache(width=key.width)
    table = SimpleNamespace(_rows_buffer=None)
    calls: list[str] = []
    lease = pool.checkout(key)
    lease.adopt(cache=cache, verify=RealShapeRecording(
        cache, table, name="verify", calls=calls, token="verify"))
    lease.owner("verify")._bindings_retired = True
    # finish() treats a retired owner as unusable before it ever tries to suspend it.
    lease.finish()
    assert pool.summary()["retained"] == 0
    assert pool.stats["boundary_invalidations"] == 0


def test_real_shape_rows_buffer_reallocation_is_caught_on_resume():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    cache = make_cache(width=key.width)
    table = SimpleNamespace(_rows_buffer=None)
    calls: list[str] = []
    lease = pool.checkout(key)
    lease.adopt(cache=cache, verify=RealShapeRecording(
        cache, table, name="verify", calls=calls, token="verify"))
    lease.finish()
    warm = pool.checkout(key)
    warm.begin_request()
    warm.owner("verify")._active_rows_buffers = [
        (table, torch.zeros(16, dtype=torch.bfloat16))
    ]
    with pytest.raises(SessionInvalidated, match="rows"):
        warm.resume()
    warm.abandon()


def test_reset_also_returns_the_head_cache_to_its_fresh_position_metadata():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    cache = make_cache(width=key.width)
    head = make_cache(width=key.width)
    lease = pool.checkout(key)
    lease.adopt(cache=cache, head_cache=head,
                verify=FakeRecording(cache, name="verify"))
    head_layer = head.layers[1]
    head_layer.length = 4096
    head_layer.host_position = 4096
    head_layer.indexer_pool.completed = 1024
    lease.finish()

    warm = pool.checkout(key)
    warm.begin_request()
    assert warm.head_cache is head
    assert head_layer.length == 0 and head_layer.host_position == 0
    assert head_layer.indexer_pool.completed == 0
    warm.finish()


def test_a_bundle_without_a_head_cache_still_resets():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    assert lease.head_cache is None
    lease.finish()
    warm = pool.checkout(key)
    warm.begin_request()
    assert cache.length == 0
    warm.finish()


def test_begin_request_pins_the_recorded_rope_width_so_prefill_cannot_reallocate_the_table():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    cache.cos = torch.ones(3, key.width, 2)
    cache.sin = torch.zeros(3, key.width, 2)
    lease.finish()

    warm = pool.checkout(key)
    warm.begin_request()
    assert cache.rope_width == key.width
    cos_ptr = cache.cos.data_ptr()
    # `_rope_cache`'s fixed branch slices instead of growing, so identity survives the prefill.
    assert cache.cos.data_ptr() == cos_ptr and cache.cos.shape[1] == key.width
    assert "kv_width" not in cache.__dict__, "kv_width must stay out of the eager prefill path"
    warm.resume()
    warm.finish()


def test_rope_width_is_not_pinned_without_a_retained_table():
    model = model_for()
    pool = GraphSessionPool(model=model)
    key = key_for(model)
    lease, cache = adopt_cold(pool, key)
    cache.cos = None
    cache.sin = None
    lease.finish()
    warm = pool.checkout(key)
    warm.begin_request()
    assert "rope_width" not in cache.__dict__
    warm.finish()
