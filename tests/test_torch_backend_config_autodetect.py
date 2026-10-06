"""Test automatic config selection based on checkpoint metadata."""

import json
import tempfile
from pathlib import Path

import pytest

from relicllm import EngineArgs
from relicllm.api.errors import ConfigurationError
from relicllm.backends.torch_backend import TorchBackend


def test_detect_expert_dtype_from_config_json(tmp_path: Path) -> None:
    """Auto-detect FP4 expert dtype from config.json."""
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "expert_dtype": "fp4",
                "torch_dtype": "bfloat16",
            }
        )
    )

    args = EngineArgs(
        model=str(tmp_path),
        backend="torch",
    )
    backend = TorchBackend(args)
    detected = backend._detect_expert_dtype()

    assert detected == "fp4"


def test_detect_expert_dtype_int8(tmp_path: Path) -> None:
    """Auto-detect int8 expert dtype from config.json."""
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "expert_dtype": "int8",
                "torch_dtype": "bfloat16",
            }
        )
    )

    args = EngineArgs(
        model=str(tmp_path),
        backend="torch",
    )
    backend = TorchBackend(args)
    detected = backend._detect_expert_dtype()

    assert detected == "int8"


def test_detect_expert_dtype_missing_config(tmp_path: Path) -> None:
    """Return None when config.json is missing."""
    args = EngineArgs(
        model=str(tmp_path),
        backend="torch",
    )
    backend = TorchBackend(args)
    detected = backend._detect_expert_dtype()

    assert detected is None


def test_detect_expert_dtype_missing_field(tmp_path: Path) -> None:
    """Return None when config.json lacks expert_dtype."""
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4",
                "torch_dtype": "bfloat16",
            }
        )
    )

    args = EngineArgs(
        model=str(tmp_path),
        backend="torch",
    )
    backend = TorchBackend(args)
    detected = backend._detect_expert_dtype()

    assert detected is None


def _write_profile(directory: Path, name: str) -> Path:
    profile = directory / name
    profile.write_text(json.dumps({"dim": 4096, "n_layers": 43}), encoding="utf-8")
    return profile


def test_runtime_namespace_names_the_default_profile_it_could_not_find(tmp_path: Path) -> None:
    """A missing default profile is refused by name, not passed on as the empty string.

    This used to hand ``""`` to the loader, which is what produced ``open(""): [Errno 2] ... ''``.
    The message named a directory that was never built and not the artifact that was missing, and
    the profile is a *runtime* file -- no profile ships in this checkout -- so there is nothing to
    fall back to and the launch has to be told which one to use.
    """
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps({"model_type": "deepseek_v4", "expert_dtype": "fp4"}), encoding="utf-8"
    )
    args = EngineArgs(model=str(tmp_path), backend="torch")

    with pytest.raises(ConfigurationError) as raised:
        TorchBackend(args)._runtime_namespace()

    message = str(raised.value)
    assert "config_fp4_active.json" in message
    assert "--config-path" in message
    # The dtype it detected is the reason it looked for *this* profile, so the refusal says so.
    assert "expert_dtype=fp4" in message


def test_runtime_namespace_falls_back_to_the_w8a8_profile_when_detection_says_nothing(
    tmp_path: Path,
) -> None:
    """No ``expert_dtype`` in the checkpoint reads as int8-or-unknown, so W8A8 is the profile named."""
    args = EngineArgs(model=str(tmp_path), backend="torch")

    with pytest.raises(ConfigurationError) as raised:
        TorchBackend(args)._runtime_namespace()

    assert "config_w8a8.json" in str(raised.value)


def test_runtime_namespace_takes_a_named_profile_over_auto_detection(tmp_path: Path, monkeypatch) -> None:
    """An explicit ``--config-path`` is the profile, and auto-detection is not consulted at all."""
    profile = _write_profile(tmp_path, "profile.json")
    config_json = tmp_path / "config.json"
    config_json.write_text(
        json.dumps({"model_type": "deepseek_v4", "expert_dtype": "fp4"}), encoding="utf-8"
    )
    backend = TorchBackend(EngineArgs(model=str(tmp_path), backend="torch", config_path=str(profile)))

    def _must_not_be_called() -> str | None:
        raise AssertionError("auto-detection ran despite --config-path")

    monkeypatch.setattr(backend, "_detect_expert_dtype", _must_not_be_called)

    assert backend._runtime_namespace().config == str(profile)
