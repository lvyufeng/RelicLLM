"""The KV arena width belongs to a cache, not to the process.

A capture needs a fixed-width arena and an eager step does not, and one process can be asked for
both. These check that the width travels with the cache, so a process that captured one request
does not charge the next, eager, one the arena's ~24 ms a step.
"""

from __future__ import annotations

import pytest

from relicllm.models.qwen4_exp.attention import _fixed_kv_width, _kv_width_of


class _Cache:
    """Just the attribute the reader looks for."""

    def __init__(self, kv_width=None):
        if kv_width is not None:
            self.kv_width = kv_width


@pytest.fixture
def no_environment_width(monkeypatch):
    monkeypatch.delenv("POCKETLLM_QWEN4_QSA_FIXED_WIDTH", raising=False)


def test_a_cache_with_a_width_answers_for_itself(no_environment_width):
    assert _kv_width_of(_Cache(kv_width=2048)) == 2048


def test_a_cache_without_one_falls_back_to_the_environment(no_environment_width, monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_FIXED_WIDTH", "512")
    assert _kv_width_of(_Cache()) == 512


def test_the_cache_wins_over_the_environment(no_environment_width, monkeypatch):
    """An eager request must be able to opt out of an arena another request asked for."""
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_FIXED_WIDTH", "2048")
    assert _kv_width_of(_Cache(kv_width=0)) == 2048  # 0 means "no opinion", not "no arena"
    assert _kv_width_of(_Cache(kv_width=384)) == 384


def test_no_cache_and_no_environment_is_the_old_behaviour(no_environment_width):
    assert _kv_width_of(None) == 0
    assert _kv_width_of(_Cache()) == 0
    assert _fixed_kv_width() == 0


def test_a_capture_does_not_leave_the_width_in_the_environment(no_environment_width):
    """What issue #217 is: the width used to be written into `os.environ`, which outlived the
    capture and made every later eager request read an arena it never asked for."""
    import inspect

    from relicllm.models.qwen4_exp import graph_window

    source = inspect.getsource(graph_window)
    assert "os.environ[" not in source, "graph_window must not write to the environment"