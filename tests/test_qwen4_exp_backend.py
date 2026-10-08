"""Qwen3.8-Flash-Next's routing and option surface, hermetic.

Two questions, neither of which needs the checkpoint: whether ``auto`` reaches this runtime for a
Qwen3.8-Flash-Next ``config.json`` and leaves every other checkpoint alone, and whether the options
this adapter declares are the ones it reads. The second is also asked generically in
``tests/test_declared_options.py``; what is here is the part that is this checkpoint's -- the nested
``text_config`` the release actually ships, and the ``0`` this runtime answers where V4.1 leaves the
number to its loader.

The request lifecycle is not exercised here. It needs a loaded model, and the hundreds of gigabytes
a run like that would want are the reason this file stops at the surface -- see ``tests/README.md``
on why a skip is not a pass and why nothing that needs a checkpoint belongs in the baseline.
"""

from __future__ import annotations

import json

import pytest

from relicllm.api import ConfigurationError, EngineArgs, UnsupportedFeatureError
from relicllm.backends import factory, qwen4_exp_backend
from relicllm.scheduler import Scheduler


def _write_config(directory, payload) -> None:
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------------------------
# routing
# ---------------------------------------------------------------------------------------------


def test_auto_selects_qwen4_exp_for_the_released_root_type(tmp_path):
    _write_config(
        tmp_path,
        {
            "model_type": "qwen4_exp",
            "architectures": ["Qwen4ExpForConditionalGeneration"],
        },
    )
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "qwen4_exp"


def test_auto_selects_qwen4_exp_from_the_nested_text_config(tmp_path):
    # The release is a conditional-generation wrapper: the root model_type is the wrapper's and the
    # language stack's repeats inside text_config. Either names this runtime, and the nested walk in
    # `_names_model` is what reaches the inner one.
    _write_config(
        tmp_path,
        {
            "model_type": "qwen4_exp",
            "text_config": {"model_type": "qwen4_exp_text"},
            "vision_config": {"model_type": "qwen4_exp_vision"},
        },
    )
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "qwen4_exp"


def test_auto_refuses_a_neighbouring_qwen(tmp_path):
    # Qwen3.5 is a different runtime's checkpoint and a generic Qwen is nobody's; neither may be
    # captured by a predicate that only looked for the substring "qwen", and with `torch` narrowed
    # to DeepSeek-V4 there is nothing left to catch them.
    for model_type in ("qwen3_5", "qwen2", "qwen3"):
        _write_config(tmp_path, {"model_type": model_type})
        with pytest.raises(UnsupportedFeatureError, match="no backend serves"):
            factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto"))


def test_auto_refuses_a_gguf_qwen4_exp(tmp_path):
    # qwen4_exp is safetensors-only, and the V4 runtime is not a home for a Qwen GGUF.
    _write_config(tmp_path, {"model_type": "qwen4_exp"})
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    with pytest.raises(UnsupportedFeatureError, match="no backend serves"):
        factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto"))


def test_explicit_qwen4_exp_refuses_a_foreign_checkpoint(tmp_path):
    _write_config(tmp_path, {"model_type": "qwen3_5"})
    with pytest.raises(UnsupportedFeatureError, match="Qwen3.8-Flash-Next checkpoints only"):
        factory.select_backend(EngineArgs(model=str(tmp_path), backend="qwen4_exp"))


# ---------------------------------------------------------------------------------------------
# options
# ---------------------------------------------------------------------------------------------


def test_the_option_names_are_the_fields():
    import dataclasses

    assert [option.name for option in qwen4_exp_backend.OPTIONS] == [
        field.name for field in dataclasses.fields(qwen4_exp_backend._Options)
    ]


def test_a_launch_that_names_nothing_is_the_bare_dataclass():
    args = EngineArgs(model="a-qwen4-exp-checkpoint", backend="qwen4_exp")

    assert qwen4_exp_backend._Options.from_args(args) == qwen4_exp_backend._Options()


def test_the_default_chunk_is_the_launchers_own_width():
    # Not the fastest one: the sweep in docs/performance/qwen4_exp_performance.md reaches 753.25
    # tok/s at 8192 and 154.70 at 512, and a chunk that does not fit the card is a failed run rather
    # than a slow one.
    assert qwen4_exp_backend._Options().prefill_chunk == 512


def test_an_unset_expert_cache_is_zero_not_the_loader_constant():
    """"Unset" and "restage every step" are one answer here, and it is the loader's own ``0``.

    V4.1 leaves this to its loader by passing ``None`` through; this runtime's loader reads ``0``
    the same way, so an unset option must arrive as ``0`` rather than as a ``None`` that would
    either crash the arithmetic or be silently a second default.
    """
    args = EngineArgs(model="a-qwen4-exp-checkpoint", backend="qwen4_exp")

    assert qwen4_exp_backend._Options.from_args(args).expert_cache == 0


def test_a_named_expert_cache_reaches_the_option():
    args = EngineArgs(
        model="a-qwen4-exp-checkpoint",
        backend="qwen4_exp",
        backend_options={"expert_cache": 64},
    )

    assert qwen4_exp_backend._Options.from_args(args).expert_cache == 64


def test_an_unknown_option_is_refused_rather_than_ignored():
    args = EngineArgs(
        model="a-qwen4-exp-checkpoint",
        backend="qwen4_exp",
        backend_options={"no_such_lever": 1},
    )
    with pytest.raises(ConfigurationError):
        qwen4_exp_backend._Options.from_args(args)


# ---------------------------------------------------------------------------------------------
# the capacity admission reads
# ---------------------------------------------------------------------------------------------


def _backend(max_model_len: int):
    """An adapter with nothing loaded, which is all these tests need: they read `_max_seq_len`."""
    return qwen4_exp_backend.Qwen4ExpBackend(
        EngineArgs(model="/nonexistent-for-a-capacity-test", max_model_len=max_model_len)
    )


def test_the_capacity_says_the_cache_was_sized_from_the_request_not_the_flag():
    """The flag that stops admission comparing a request to a number that never built its buffer.

    This runtime's QSA key/value and index buffers are allocated per request -- ``prompt_len + budget
    + 1`` at ``models/qwen4_exp/runtime.py:300`` -- and ``--max-model-len`` never reaches an
    allocation. So the capacity this adapter reports has to carry ``sizes_from_request``, or
    admission would refuse requests whose cache was, at that moment, being built to fit them.
    """
    capacity = _backend(max_model_len=8192)._kv_capacity()

    assert capacity.positions == 8192
    assert capacity.sizes_from_request is True


def test_admission_lets_a_request_past_the_configured_context_through():
    """The behaviour that has to survive the lift: this runtime admits what it can build for.

    A ~10k-token request against a `--max-model-len 8192` is the case the check would wrongly refuse
    -- the buffer for it is `prompt + budget + 1`, sized far past the flag. Admission reading a
    request-sized capacity admits it, and `pending_count` says it really queued.
    """
    backend = _backend(max_model_len=8192)
    scheduler = Scheduler(capacity=backend._kv_capacity)

    admitted = scheduler.submit(list(range(10_000)), 1000)

    assert admitted.max_new_tokens == 1000
    assert scheduler.pending_count() == 1


def test_the_details_do_not_claim_the_context_is_the_caches_size():
    """`context` used to read `8192 positions`, the number no allocation here holds.

    A deployment reading the details string decides how wide a context it can afford, so the string
    a run publishes has to name the flag as a ceiling and say where the real size comes from. The
    builder needs only the config and the options, so it is called directly rather than behind a
    load this host cannot perform.
    """
    backend = _backend(max_model_len=8192)
    backend._build_details()

    text = backend._details["context"]

    assert "8192 positions at most" in text
    assert "each request" in text