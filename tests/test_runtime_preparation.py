from __future__ import annotations

import pytest

from relicllm.api import EngineArgs
from relicllm.backends.base import RuntimeAdapter
from relicllm.backends.mimo_backend import MimoBackend
from relicllm.backends.qwen4_exp_backend import Qwen4ExpBackend
from relicllm.backends.v41_backend import V41Backend
from relicllm.backends.xing4_backend import Xing4Backend


RUNTIMES = (MimoBackend, Xing4Backend, V41Backend, Qwen4ExpBackend)


@pytest.fixture(params=RUNTIMES, ids=lambda runtime: runtime.name)
def adapter(request):
    runtime = request.param
    return runtime(EngineArgs(model="unused", backend=runtime.name, max_model_len=64))


def test_preparation_has_one_body(adapter):
    assert type(adapter).prepare is RuntimeAdapter.prepare


def test_preparation_keeps_the_load_before_the_cache(adapter, monkeypatch):
    events = []

    def cache():
        assert events[-1] == "load"
        events.append("cache")

    def load():
        events.append("load")
        if isinstance(adapter, V41Backend):
            # V4.1 already builds its store inside the load lock, not after loading.
            cache()

    monkeypatch.setattr(adapter, "_ensure_open", lambda: events.append("open"))
    monkeypatch.setattr(adapter, "_ensure_loaded", load)
    # Qwen4-Exp has no store: the presence of a method must not opt it in.
    monkeypatch.setattr(adapter, "_ensure_prefix_cache", cache, raising=False)

    adapter.prepare()

    expected = ["open", "load"]
    if not isinstance(adapter, Qwen4ExpBackend):
        expected.append("cache")
    assert events == expected


def test_a_closed_adapter_neither_loads_nor_builds_a_cache(adapter, monkeypatch):
    events = []
    adapter._closed = True
    monkeypatch.setattr(adapter, "_ensure_loaded", lambda: events.append("load"))
    monkeypatch.setattr(adapter, "_ensure_prefix_cache", lambda: events.append("cache"), raising=False)

    with pytest.raises(RuntimeError, match="backend is closed"):
        adapter.prepare()

    assert events == []


def test_a_failed_load_never_reaches_cache_preparation(adapter, monkeypatch):
    events = []

    def load():
        events.append("load")
        raise RuntimeError("load failed")

    monkeypatch.setattr(adapter, "_ensure_loaded", load)
    monkeypatch.setattr(adapter, "_ensure_prefix_cache", lambda: events.append("cache"), raising=False)

    with pytest.raises(RuntimeError, match="load failed"):
        adapter.prepare()

    assert events == ["load"]


def test_v41_keeps_its_cache_build_under_the_load_lock(monkeypatch):
    adapter = V41Backend(EngineArgs(model="unused", backend="v41", max_model_len=64))
    events = []

    def load():
        events.append("load")
        adapter._front = object()

    def cache():
        assert adapter._load_lock._is_owned()
        events.append("cache")

    monkeypatch.setattr(adapter, "_load", load)
    monkeypatch.setattr(adapter, "_ensure_prefix_cache", cache)

    adapter.prepare()
    assert events == ["load", "cache"]

    adapter.prepare()
    assert events == ["load", "cache", "cache"]
