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


def test_auto_refuses_a_checkpoint_no_runtime_claims(tmp_path):
    # A generic Qwen is nobody's: `torch` is DeepSeek-V4's runtime, not a catch-all, so auto must
    # refuse rather than route it into the DeepSeek-V4 engine.
    _write_config(tmp_path, {"model_type": "qwen2", "architectures": ["Qwen2ForCausalLM"]})
    with pytest.raises(UnsupportedFeatureError, match="no backend serves"):
        factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto"))


def test_auto_refuses_a_gguf_no_runtime_claims(tmp_path):
    # The unreadable-body `.gguf` here reads as no *header* evidence, but the config names qwen3_5,
    # which is evidence against `torch` -- so this is a refusal and not a fall-through.
    _write_config(tmp_path, {"model_type": "qwen3_5"})
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    with pytest.raises(UnsupportedFeatureError, match="no backend serves"):
        factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto"))


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
    # The original V4 is the one architecture `torch` serves, and the only thing auto routes to it.
    _write_config(tmp_path, {"model_type": "deepseek_v4"})
    assert factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto")) == "torch"


def test_auto_refuses_a_gguf_v41_checkpoint(tmp_path):
    # A V4.1 GGUF has no adapter -- v41 is safetensors-only -- and `torch` serves V4, not V4.1, so
    # loading it there would read a V4.1 checkpoint through V4's ModelArgs. Refuse instead.
    _write_config(tmp_path, {"model_type": "deepseek_v41"})
    (tmp_path / "model.gguf").write_bytes(b"GGUF")
    with pytest.raises(UnsupportedFeatureError, match="no backend serves"):
        factory.select_backend(EngineArgs(model=str(tmp_path), backend="auto"))


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


def test_explicit_torch_rejects_a_foreign_checkpoint(tmp_path):
    # The declaration used to say `torch` reads everything; it reads DeepSeek-V4. An explicit
    # `--backend torch` on anything else must refuse for the same reason auto does.
    _write_config(tmp_path, {"model_type": "qwen3_5"})
    args = EngineArgs(model=str(tmp_path), backend="torch")
    with pytest.raises(UnsupportedFeatureError, match="DeepSeek-V4 checkpoints only"):
        factory.select_backend(args)


def test_a_v41_checkpoint_is_not_claimed_by_torch(tmp_path):
    # The substring trap this predicate is written to avoid: `deepseek_v41` contains `deepseek`, and
    # a recogniser that matched loosely would route V4.1 to the V4 runtime.
    _write_config(tmp_path, {"model_type": "deepseek_v41", "architectures": ["DeepseekV41ForCausalLM"]})
    args = EngineArgs(model=str(tmp_path), backend="auto")
    assert factory.select_backend(args) == "v41"


def test_the_architecture_string_alone_does_not_let_torch_claim_v41(tmp_path):
    """The `model_type` exact set is not what saves this one: the released V4.1 config also carries
    `architectures=["DeepseekV41ForCausalLM"]`, whose lowercase form *contains* the needle V4 matches
    on. A plain substring test would therefore route a config that declares only the architecture
    string to the V4 runtime, and `auto` would hide it by reaching `v41` first -- `--backend torch`
    would not. The generation digit has to stop the match."""
    _write_config(tmp_path, {"architectures": ["DeepseekV41ForCausalLM"]})

    from relicllm.backends.capabilities import Verdict, identify

    args = EngineArgs(model=str(tmp_path), backend="torch")
    assert identify("torch", args).verdict is Verdict.REFUSED
    assert identify("v41", args).verdict is Verdict.READ
    with pytest.raises(UnsupportedFeatureError, match="DeepSeek-V4 checkpoints only"):
        factory.select_backend(args)


def test_torch_claims_v4_from_the_architecture_string_alone(tmp_path):
    """The other side of the boundary: V4's own architecture string still reaches `torch`, and does
    not spill into `v41`. Without this the digit rule could pass by refusing both generations."""
    _write_config(tmp_path, {"architectures": ["DeepseekV4ForCausalLM"]})

    from relicllm.backends.capabilities import Verdict, identify

    args = EngineArgs(model=str(tmp_path), backend="auto")
    assert factory.select_backend(args) == "torch"
    assert identify("v41", args).verdict is Verdict.REFUSED


def test_a_gguf_of_a_v4_variant_is_torchs(tmp_path):
    """`general.architecture` is a family, not one spelling: the base export is `deepseek4` and a
    variant suffixes it (`deepseek4_mtp_support` on this host's MTP build). Refusing a V4 variant as
    "not V4" loses a servable model, so the prefix is matched with the separator -- which still
    cannot reach a differently-numbered neighbour."""
    from tests.gguf_test_utils import write_gguf

    path = tmp_path / "variant.gguf"
    write_gguf(path, metadata={"general.architecture": "deepseek4_mtp_support"}, tensors=[])

    assert factory.select_backend(EngineArgs(model=str(path), backend="auto")) == "torch"
    assert factory.select_backend(EngineArgs(model=str(path), backend="torch")) == "torch"