"""CPU contract tests for eager token readiness and owned graph staging."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from relicllm.models.qwen4_exp.model import Qwen4ExpModel


class _View:
    def __init__(self, shape, copies):
        self.shape = shape
        self.copies = copies

    def copy_(self, source, *, non_blocking):
        self.copies.append((source, non_blocking))
        return self


class _Buffer:
    def __init__(self, shape, copies):
        self.shape = shape
        self.copies = copies

    def __getitem__(self, slices):
        return _View(tuple(part.stop for part in slices), self.copies)


def test_cpu_token_ids_need_no_allocator_or_copy(monkeypatch):
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    ids = torch.tensor([[3], [4]], dtype=torch.long, device="cpu")
    monkeypatch.setattr(torch, "empty", lambda *a, **k: (_ for _ in ()).throw(AssertionError("allocation")))
    assert model._pinned_ids(ids) is ids


def test_eager_gpu_ids_are_ready_before_cpu_consumers():
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    copies = []
    model._ids_pinned = _Buffer((2, 3), copies)
    source = SimpleNamespace(device=SimpleNamespace(type="cuda"), shape=(2, 1))
    view = model._pinned_ids(source)
    assert view.shape == (2, 1)
    assert copies == [(source, False)]


def test_eager_buffer_growth_still_uses_blocking_copy(monkeypatch):
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    copies, allocations = [], []

    def allocate(rows, width, **kwargs):
        allocations.append((rows, width, kwargs))
        return _Buffer((rows, width), copies)

    monkeypatch.setattr(torch, "empty", allocate)
    source = SimpleNamespace(device=SimpleNamespace(type="cuda"), shape=(2, 4))
    view = model._pinned_ids(source)
    assert view.shape == (2, 4)
    assert copies == [(source, False)]
    assert allocations[0][2] == {"device": "cpu", "dtype": torch.long, "pin_memory": True}


def test_owned_graph_history_bypasses_eager_transfer(monkeypatch):
    model = Qwen4ExpModel.__new__(Qwen4ExpModel)
    history = torch.tensor([[1, 2, 3]], dtype=torch.long, device="cpu")
    observed = []

    class Layer:
        ple = object()

        def __call__(self, hidden, **kwargs):
            observed.append(kwargs["token_history"])
            return hidden

    model.layers = [Layer()]
    model.profiler = None
    model.config = SimpleNamespace(ngram_size=3, primary_eos_token_id=0)
    model._rope_cache = lambda *a: (None, None)
    model._pinned_ids = lambda *a: (_ for _ in ()).throw(AssertionError("captured D2H staging"))
    cache = SimpleNamespace(
        layers=[None], ple_states={}, ple_history_at=lambda _ids: history,
        ple_token_history=None, length=0,
    )
    source = SimpleNamespace(shape=(1, 1))
    marker = object()
    result = model._forward_layers(marker, source, cache=cache, past_len=0,
                                   prof_scope=lambda _name: nullcontext())
    assert result is marker and observed == [history]
    assert cache.length == 1
