"""The captured MTP loop as an *opt-in* serving mode, hermetic.

What is proved here is dispatch and refusal, not decoding: which loop each rank enters, what the
launch validation refuses before a checkpoint is read, that a request past the checkpoint's native
ceiling is refused before any rank is dispatched, and that a captured request's own setup and
cleanup are charged to its decode seconds. Token parity of the loop itself is a GPU measurement
(``tests/probe_qwen4_mtp_nvfp4.py`` on the leased eight-4090 host), not this file's claim.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from relicllm.api import (
    ConfigurationError,
    EngineArgs,
    GenerationRequest,
    SamplingParams,
    UnsupportedFeatureError,
)
from relicllm.backends import capabilities, qwen4_exp_backend
from relicllm.models.qwen4_exp import runtime as qwen4_runtime


NATIVE = 262144


@pytest.fixture(autouse=True)
def qualified_indexer_geometry(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT", "1")


def test_missing_qualified_indexer_geometry_is_refused(monkeypatch):
    monkeypatch.setenv(qwen4_exp_backend._STRICT_VERIFY_ENV, "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT", "0")
    backend, calls = _install(monkeypatch, args=_args(speculative_method="mtp", speculative_tokens=3))
    with pytest.raises(ConfigurationError, match="INDEXER_ROW_INVARIANT=1"):
        _run(backend)
    assert not calls["plain"] and not calls["mtp"]


def _config(*, eos=2):
    return SimpleNamespace(
        max_position_embeddings=NATIVE,
        text_config=SimpleNamespace(
            max_position_embeddings=NATIVE,
            eos_token_id=eos,
            num_hidden_layers=48,
            num_experts=512,
        ),
    )


def _model(*, has_mtp=True):
    return SimpleNamespace(has_mtp=has_mtp)


def _args(**overrides):
    values = dict(model="/nonexistent", backend="qwen4_exp")
    values.update(overrides)
    return EngineArgs(**values)


def _install(monkeypatch, *, args, model=None, world=1, device="cuda", pp=1, tp=None):
    """A backend whose group, loader and both loops are fakes, so nothing reads a checkpoint."""
    ctx = SimpleNamespace(
        device=torch.device(device, 0) if device == "cuda" else torch.device("cpu"),
        world_size=world,
        rank=0,
        pp_size=pp,
        tp_world_size=world if tp is None else tp,
    )
    monkeypatch.setattr(qwen4_runtime, "init_distributed", lambda: ctx)
    calls = {"plain": [], "mtp": []}

    def plain(*_args, **_kwargs):
        calls["plain"].append((_args, _kwargs))
        return [10, 11], {"stopped": "length", "prefill_s": 0.5, "decode_s": 2.0, "ttft_s": 0.6}

    def lookahead(*_args, **_kwargs):
        calls["mtp"].append((_args, _kwargs))
        return [20, 21, 22], {
            "stopped": "length", "prefill_s": 0.5, "ttft_s": 0.6,
            "decode_s": 1.0, "graph_setup_s": 3.0, "request_cleanup_s": 0.25,
            "decode_s_including_cleanup": 1.25, "decode_wall_s": 4.0,
            "trunk_steps": 1.0,
        }

    monkeypatch.setattr(qwen4_runtime, "generate", plain)
    monkeypatch.setattr(qwen4_runtime, "generate_mtp_draft_lookahead", lookahead)
    backend = qwen4_exp_backend.Qwen4ExpBackend(
        args, loader=lambda _args, _options: (model or _model(), _config())
    )
    backend._init_distributed_for_tests = lambda: None
    return backend, calls


def _patch_group(monkeypatch, *, device="cuda"):
    """Join a fake single-rank group, so the real loader path can run without a card."""
    ctx = SimpleNamespace(
        device=torch.device(device, 0) if device == "cuda" else torch.device("cpu"),
        world_size=1, rank=0, pp_size=1, tp_world_size=1,
    )
    monkeypatch.setattr(qwen4_runtime, "init_distributed", lambda: ctx)
    return ctx


def _run(backend, *, prompt=8, budget=4, stop=None, on_token=None):
    request = GenerationRequest(prompt_tokens=list(range(prompt)), request_id="req-mtp")
    return backend._loop(
        list(range(prompt)), budget, request, on_token=on_token, stop=stop
    )


@pytest.mark.parametrize("method,tokens", [(None, 1), ("", 1)])
def test_default_launch_stays_on_the_plain_loop(monkeypatch, method, tokens):
    backend, calls = _install(monkeypatch, args=_args(speculative_method=method,
                                                      speculative_tokens=tokens))
    generation = _run(backend)
    assert generation.tokens == [10, 11]
    assert len(calls["plain"]) == 1 and not calls["mtp"]
    # The plain loop's own decode interval is used as-is; it already includes its graph work.
    assert generation.decode_seconds == 2.0
    assert backend._model.has_mtp  # the injected model is untouched by the flag being off


@pytest.mark.parametrize("tokens", [3, 4])
def test_mtp_launch_enters_the_captured_loop_with_its_depth_and_cancellation(monkeypatch, tokens):
    monkeypatch.setenv(qwen4_exp_backend._STRICT_VERIFY_ENV, "1")
    backend, calls = _install(monkeypatch, args=_args(speculative_method="mtp",
                                                      speculative_tokens=tokens))
    seen = []
    generation = _run(backend, on_token=seen.append, stop=lambda: False)
    assert generation.tokens == [20, 21, 22]
    assert len(calls["mtp"]) == 1 and not calls["plain"]
    _args_seen, kwargs = calls["mtp"][0]
    assert kwargs["draft_tokens"] == tokens and kwargs["verify_graph"] is True
    assert kwargs["max_new_tokens"] == 4 and kwargs["chunk_size"] == 512
    # A predicate is always handed over, even with no stream stop: the loop's own EOS flag rides
    # the same broadcast, and a rank that honoured it locally would strand its peers.
    predicate = kwargs["on_step"]
    assert callable(predicate) and predicate() is False
    assert kwargs["local_eos"] == [False]
    assert callable(kwargs["on_token"])
    # Setup and deferred cleanup are charged to the request, not left out of its rate.
    assert generation.decode_seconds == pytest.approx(1.25 + 3.0)


def test_mtp_enabled_loads_the_head(monkeypatch, tmp_path):
    monkeypatch.setenv(qwen4_exp_backend._STRICT_VERIFY_ENV, "1")
    _patch_group(monkeypatch)
    seen = {}
    monkeypatch.setattr(
        qwen4_runtime, "load_model",
        lambda directory, ctx, **kwargs: seen.update(directory=directory, **kwargs) or (
            _model(), _config()
        ),
    )
    backend = qwen4_exp_backend.Qwen4ExpBackend(
        _args(model=str(tmp_path), speculative_method="mtp", speculative_tokens=3),
        tokenizer=object(),
    )
    backend._checkpoint_dir = str(tmp_path)
    backend._ensure_loaded()
    assert seen["mtp"] is True and seen["directory"] == str(tmp_path)


def test_plain_launch_does_not_load_the_head(monkeypatch, tmp_path):
    _patch_group(monkeypatch, device="cpu")
    seen = {}
    monkeypatch.setattr(
        qwen4_runtime, "load_model",
        lambda directory, ctx, **kwargs: seen.update(directory=directory, **kwargs) or (
            _model(has_mtp=False), _config()
        ),
    )
    backend = qwen4_exp_backend.Qwen4ExpBackend(_args(model=str(tmp_path)), tokenizer=object())
    backend._checkpoint_dir = str(tmp_path)
    backend._ensure_loaded()
    assert seen["mtp"] is False


@pytest.mark.parametrize("method", ["eagle", "medusa", "MTP "])
def test_another_speculative_method_is_refused_by_name(monkeypatch, method):
    with pytest.raises(UnsupportedFeatureError, match="captured MTP speculative decoder"):
        qwen4_exp_backend.Qwen4ExpBackend(_args(speculative_method=method, speculative_tokens=4))


@pytest.mark.parametrize("tokens", [1, 2, 5, 6, 16])
def test_a_depth_the_adapter_cannot_prove_is_refused(monkeypatch, tokens):
    with pytest.raises(UnsupportedFeatureError, match="3, 4"):
        qwen4_exp_backend.Qwen4ExpBackend(_args(speculative_method="mtp",
                                                speculative_tokens=tokens))


@pytest.mark.parametrize("tokens", [2, 4])
def test_a_depth_without_a_method_is_a_contradiction_not_an_ignored_flag(monkeypatch, tokens):
    with pytest.raises(ConfigurationError, match="needs speculative_method='mtp'"):
        qwen4_exp_backend.Qwen4ExpBackend(_args(speculative_tokens=tokens))


def test_missing_strict_verify_switch_refuses_before_loading(monkeypatch):
    monkeypatch.delenv(qwen4_exp_backend._STRICT_VERIFY_ENV, raising=False)
    backend, _calls = _install(monkeypatch, args=_args(model="/checkpoint",
                                                       speculative_method="mtp",
                                                       speculative_tokens=4))
    with pytest.raises(ConfigurationError, match="VERIFY_ROW_INVARIANT=1"):
        backend._ensure_loaded()
    assert backend._model is None


@pytest.mark.parametrize("world,device,pp,pattern", [
    (1, "cpu", 1, "needs device='cuda'"),
    (8, "cuda", 2, "pure tensor-parallel"),
    (8, "cuda", 1, "pure tensor-parallel"),
])
def test_placement_requirements_are_refused(monkeypatch, world, device, pp, pattern):
    monkeypatch.setenv(qwen4_exp_backend._STRICT_VERIFY_ENV, "1")
    # A pure-TP group must have its tensor-parallel width equal to the world; the (8, cuda, 1)
    # case is narrowed behind it so "one stage" and "not pure TP" are told apart as causes.
    narrower = 4 if pp == 1 and world == 8 else None
    backend, _calls = _install(
        monkeypatch, args=_args(model="/checkpoint", speculative_method="mtp",
                                 speculative_tokens=3),
        world=world, device=device, pp=pp, tp=narrower,
    )
    with pytest.raises(UnsupportedFeatureError, match=pattern):
        backend._ensure_loaded()


def test_batch_width_above_one_is_refused(monkeypatch):
    monkeypatch.setenv(qwen4_exp_backend._STRICT_VERIFY_ENV, "1")
    backend, _calls = _install(monkeypatch, args=_args(model="/checkpoint",
                                                       speculative_method="mtp",
                                                       speculative_tokens=3,
                                                       max_batch_size=2))
    with pytest.raises(UnsupportedFeatureError, match="one mutable cache"):
        backend._ensure_loaded()


def test_a_model_without_its_head_is_refused(monkeypatch):
    monkeypatch.setenv(qwen4_exp_backend._STRICT_VERIFY_ENV, "1")
    backend, _calls = _install(monkeypatch, args=_args(model="/checkpoint",
                                                       speculative_method="mtp",
                                                       speculative_tokens=3),
                               model=_model(has_mtp=False))
    with pytest.raises(ConfigurationError, match="MTP head"):
        backend._ensure_loaded()


def test_worker_rank_runs_the_same_captured_driver(monkeypatch):
    monkeypatch.setenv(qwen4_exp_backend._STRICT_VERIFY_ENV, "1")
    backend, calls = _install(monkeypatch, args=_args(speculative_method="mtp",
                                                      speculative_tokens=4))
    backend._ensure_loaded()
    backend._run_payload({"prompt_ids": [1, 2, 3], "max_new_tokens": 5,
                          "request_id": "req-worker"})
    assert len(calls["mtp"]) == 1 and not calls["plain"]
    _args_seen, kwargs = calls["mtp"][0]
    assert kwargs["draft_tokens"] == 4 and kwargs["max_new_tokens"] == 5
    assert kwargs["on_token"] is None  # a worker streams nothing; rank 0 owns the callback
    assert kwargs["local_eos"] == [False]


def test_plain_worker_payload_is_unchanged(monkeypatch):
    backend, calls = _install(monkeypatch, args=_args())
    backend._ensure_loaded()
    backend._run_payload({"prompt_ids": [1, 2], "max_new_tokens": 3, "request_id": "req-plain"})
    assert len(calls["plain"]) == 1 and not calls["mtp"]


# ---------------------------------------------------------------------------------------------
# request bounds
# ---------------------------------------------------------------------------------------------


def _request(prompt, budget=None):
    return GenerationRequest(
        prompt_tokens=list(range(prompt)), request_id="req-bounds",
        sampling_params=SamplingParams(max_tokens=budget),
    )


def test_a_request_past_the_native_ceiling_is_refused_before_dispatch(monkeypatch):
    monkeypatch.setenv(qwen4_exp_backend._STRICT_VERIFY_ENV, "1")
    backend, calls = _install(monkeypatch, args=_args(speculative_method="mtp",
                                                      speculative_tokens=3))
    backend._ensure_loaded()
    prompt = NATIVE - 8
    with pytest.raises(ConfigurationError, match="native positions"):
        backend._budget(list(range(prompt)), _request(prompt, 128))
    # Nothing was dispatched into the collective, so no loop ran on either side.
    assert not calls["mtp"] and not calls["plain"]


def test_a_request_that_exactly_fills_the_native_ceiling_is_admitted(monkeypatch):
    monkeypatch.setenv(qwen4_exp_backend._STRICT_VERIFY_ENV, "1")
    backend, _calls = _install(monkeypatch, args=_args(speculative_method="mtp",
                                                       speculative_tokens=4))
    backend._ensure_loaded()
    prompt = NATIVE - 128
    assert backend._budget(list(range(prompt)), _request(prompt, 128)) == 128
    # No explicit cap: the loop takes everything the prompt leaves.
    assert backend._budget(list(range(prompt)), _request(prompt)) == 128


def test_plain_budget_keeps_the_configured_model_length(monkeypatch):
    backend, _calls = _install(monkeypatch, args=_args())
    backend._ensure_loaded()
    backend._max_seq_len = 1024
    assert backend._budget(list(range(24)), _request(24)) == 1000


# ---------------------------------------------------------------------------------------------
# declaration and accounting
# ---------------------------------------------------------------------------------------------


def test_the_declaration_advertises_only_the_two_measured_depths():
    declaration = capabilities.runtime_capabilities("qwen4_exp")
    assert declaration.supports_speculative_decoding == ("mtp",)
    prose = declaration.details["speculative"]
    assert "--speculative-tokens 3 or 4" in prose and "experimental" in prose


def test_capabilities_wire_the_declaration_through(monkeypatch):
    backend, _calls = _install(monkeypatch, args=_args())
    backend._build_details()
    capability = backend.capabilities
    assert capability.supports_speculative_decoding == ("mtp",)
    assert "off; plain greedy decode" in capability.details["speculative"]


def test_mtp_capabilities_detail_names_the_depth(monkeypatch):
    monkeypatch.setenv(qwen4_exp_backend._STRICT_VERIFY_ENV, "1")
    backend, _calls = _install(monkeypatch, args=_args(speculative_method="mtp",
                                                       speculative_tokens=4))
    backend._ensure_loaded()
    prose = backend.capabilities.details["speculative"]
    assert "4 drafts" in prose and "experimental" in prose


def test_decode_seconds_charges_setup_and_cleanup_only_for_the_captured_loop():
    captured = qwen4_exp_backend._decode_seconds(
        {"decode_s": 1.0, "graph_setup_s": 3.0, "decode_s_including_cleanup": 1.25,
         "trunk_steps": 1.0}
    )
    assert captured == pytest.approx(4.25)
    # A tree that predates the explicit cleanup scope still charges setup plus the reported cleanup.
    older = qwen4_exp_backend._decode_seconds(
        {"decode_s": 1.0, "graph_setup_s": 2.0, "request_cleanup_s": .5, "trunk_steps": 1.0}
    )
    assert older == pytest.approx(3.5)
    # Plain stats have no trunk_steps marker and their decode_s already covers everything.
    plain = qwen4_exp_backend._decode_seconds(
        {"decode_s": 2.0, "graph_setup_s": 3.0, "request_cleanup_s": .5}
    )
    assert plain == 2.0