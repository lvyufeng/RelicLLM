"""The indexer sort binding: what it replaces, what it refuses, and what it restores.

The anchor test is the one that earns its keep. The binding works by swapping one expression inside
`Indexer.forward`, so if that method is edited the binding must fail loudly rather than quietly keep
reporting a gain from code that no longer runs. Pinning the anchor here means a future edit to the
model shows up as a failing test in this repository instead of as a benchmark that silently measures
nothing.

The refusal tests run against synthetic modules, so they need neither a device nor a checkpoint.
"""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relicllm.models.deepseek_v4_1 import ascend_indexer_sort_binding as binding  # noqa: E402


class _WithAnchor:
    """A stand-in for `Indexer`: the binding keys off a method literally named `forward`."""

    def forward(self, values, keys, offset, tail, width):
        idxs = torch.where(values > -torch.inf, keys + offset, tail).sort(dim=-1).values
        idx = torch.where(idxs < tail, idxs, -1).int()
        return idx


class _WithoutAnchor:
    def forward(self, values, keys, offset, tail, width):
        return torch.zeros_like(keys)


class _WithTwoAnchors:
    def forward(self, values, keys, offset, tail, width):
        idxs = torch.where(values > -torch.inf, keys + offset, tail).sort(dim=-1).values
        idx = torch.where(idxs < tail, idxs, -1).int()
        idxs = torch.where(values > -torch.inf, keys + offset, tail).sort(dim=-1).values
        idx = torch.where(idxs < tail, idxs, -1).int()
        return idx


def _fake_attention(monkeypatch, indexer_base):
    """A stand-in `relicllm.models.deepseek_v4_1.attention` carrying only what install() touches.

    The indexer is a fresh subclass per test: install() writes attributes onto the class it is
    given, so sharing one class object across tests would leak a half-installed binding into the
    next test. The inherited `forward` is still a real function in this file, so `getsource` works.
    """
    module = types.ModuleType("relicllm.models.deepseek_v4_1.attention")
    module.torch = torch
    module.Indexer = type("Indexer", (indexer_base,), {})
    monkeypatch.setitem(sys.modules, module.__name__, module)
    parent = sys.modules.get("relicllm.models.deepseek_v4_1")
    if parent is not None:
        monkeypatch.setattr(parent, "attention", module, raising=False)
    return module


def test_the_anchor_still_exists_exactly_once_in_the_model_method():
    """Fail here, not in a benchmark, if `Indexer.forward` stops containing the expression."""
    from relicllm.models.deepseek_v4_1 import attention

    source = binding._dedented_source(attention.Indexer.forward)
    assert source.count(binding.ANCHOR) == 1, (
        "the indexer selection anchor moved or changed; the binding cannot bind, and any gain it "
        "reported would have come from code that never ran"
    )


def test_install_swaps_the_method_and_describes_what_it_did(monkeypatch):
    module = _fake_attention(monkeypatch, _WithAnchor)
    original = module.Indexer.forward
    owner = binding.install()
    try:
        assert module.Indexer.forward is not original
        assert module.Indexer._relic_indexer_sort_original is original
        described = owner.describe()
        assert described["model_source_touched"] is False
        assert described["anchor_replacements"] == 1
        assert described["numerical_or_performance_acceptance_claimed"] is False
        # The swapped body resolves the fused selection from the namespace install() built.
        # (Its source is not readable: the recompiled method's filename is the exec marker.)
        fused = module.Indexer.forward.__globals__.get("fused_masked_sort_trailing")
        from relic_core.kernels.ascend_indexer_masked_sort import masked_sort_trailing_with_tail
        assert fused is masked_sort_trailing_with_tail
    finally:
        owner.close()


def test_close_restores_the_exact_original_object(monkeypatch):
    module = _fake_attention(monkeypatch, _WithAnchor)
    original = module.Indexer.forward
    owner = binding.install()
    assert owner.close() is True
    assert module.Indexer.forward is original
    for name in (binding._ORIGINAL, binding._OWNER, binding._INSTALLED_FLAG):
        assert not hasattr(module.Indexer, name)


def test_a_second_install_is_refused(monkeypatch):
    module = _fake_attention(monkeypatch, _WithAnchor)
    owner = binding.install()
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            binding.install()
    finally:
        owner.close()


def test_install_refuses_when_the_anchor_is_absent(monkeypatch):
    _fake_attention(monkeypatch, _WithoutAnchor)
    with pytest.raises(RuntimeError, match="anchor found 0 times"):
        binding.install()


def test_install_refuses_when_the_anchor_appears_twice(monkeypatch):
    _fake_attention(monkeypatch, _WithTwoAnchors)
    with pytest.raises(RuntimeError, match="anchor found 2 times"):
        binding.install()


def test_close_refuses_from_another_thread(monkeypatch):
    module = _fake_attention(monkeypatch, _WithAnchor)
    owner = binding.install()
    try:
        failure = []

        def _close():
            try:
                owner.close()
            except RuntimeError as exc:
                failure.append(str(exc))

        worker = threading.Thread(target=_close)
        worker.start()
        worker.join()
        assert failure and "thread that installed it" in failure[0]
        # The refusal left the binding intact rather than half-unwound.
        assert module.Indexer.forward is not module.Indexer._relic_indexer_sort_original
    finally:
        owner.close()


def test_close_refuses_when_the_method_was_replaced_behind_it(monkeypatch):
    module = _fake_attention(monkeypatch, _WithAnchor)
    owner = binding.install()
    original = module.Indexer._relic_indexer_sort_original
    try:
        # A third party replacing the method must be detected rather than overwritten.
        # Note `_WithAnchor.forward` is the patched attribute now, so the interloper has
        # to be a different function.
        module.Indexer.forward = _WithoutAnchor.forward
        with pytest.raises(RuntimeError, match="no longer the one this binding installed"):
            owner.close()
    finally:
        # The guarded close refused, as it should, so unwind the class by hand: this test's
        # whole point is the state close() declines to touch.
        module.Indexer.forward = original
        for name in (binding._ORIGINAL, binding._OWNER, binding._INSTALLED_FLAG):
            if hasattr(module.Indexer, name):
                delattr(module.Indexer, name)


def test_install_is_reversible_and_repeatable(monkeypatch):
    module = _fake_attention(monkeypatch, _WithAnchor)
    original = module.Indexer.forward
    for _ in range(3):
        owner = binding.install()
        owner.close()
        assert module.Indexer.forward is original