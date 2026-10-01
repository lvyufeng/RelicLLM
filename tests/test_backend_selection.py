from __future__ import annotations

import json

import pytest

from relicllm.api import EngineArgs, UnsupportedFeatureError
from relicllm.backends import factory


def _write_config(directory, payload) -> None:
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_explicit_backend_is_never_rewritten():
    assert factory.select_backend(EngineArgs(model="missing", backend="torch")) == "torch"
    assert factory.select_backend(EngineArgs(model="missing", backend="v41")) == "v41"


def test_auto_keeps_torch_for_other_qwen_generations(tmp_path):
    # No runtime claims a generic Qwen checkpoint, so the auto path must reach the
    # generic runtime rather than route it to a name-shaped neighbour.
    _write_config(tmp_path, {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "torch"


def test_auto_keeps_torch_for_gguf_checkpoints(tmp_path):
    _write_config(tmp_path, {"model_type": "qwen3_5"})
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "torch"


def test_explicit_v41_is_never_rewritten():
    assert factory.select_backend(EngineArgs(model="missing", backend="v41")) == "v41"


def test_auto_selects_v41_for_a_v41_checkpoint(tmp_path):
    _write_config(
        tmp_path, {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"]}
    )
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "v41"


def test_auto_selects_v41_from_the_nested_text_config(tmp_path):
    # The released checkpoint nests model_type at the root and repeats it inside
    # text_config; the root value alone is the wrapper's, and both name V4.1.
    _write_config(
        tmp_path,
        {"model_type": "deepseek_v41", "text_config": {"model_type": "deepseek_v41_text"}},
    )
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "v41"


def test_auto_keeps_torch_for_a_non_v41_deepseek(tmp_path):
    _write_config(tmp_path, {"model_type": "deepseek_v4"})
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "torch"


def test_auto_keeps_torch_for_a_gguf_v41_checkpoint(tmp_path):
    _write_config(tmp_path, {"model_type": "deepseek_v41"})
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "torch"


def test_explicit_v41_rejects_gguf_checkpoints(tmp_path):
    _write_config(tmp_path, {"model_type": "deepseek_v41"})
    args = EngineArgs(model=str(tmp_path), backend="v41", model_format="gguf")
    with pytest.raises(UnsupportedFeatureError, match="GGUF"):
        factory.select_backend(args)


def test_explicit_v41_rejects_a_foreign_checkpoint(tmp_path):
    _write_config(tmp_path, {"model_type": "qwen3_5"})
    args = EngineArgs(model=str(tmp_path), backend="v41")
    with pytest.raises(UnsupportedFeatureError, match="DeepSeek-V4.1-Flash"):
        factory.select_backend(args)