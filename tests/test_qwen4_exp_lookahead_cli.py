"""CLI routing exposes qualified lookahead without silently replacing the legacy mode."""
from types import SimpleNamespace
import sys

import pytest
import torch

from relicllm.models.qwen4_exp import runtime


@pytest.fixture(autouse=True)
def qualified_indexer_geometry(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT", "1")


def test_captured_cli_requires_qualified_indexer_geometry(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT", "0")
    monkeypatch.setattr(runtime, "init_distributed", lambda: pytest.fail("must refuse before group init"))
    with pytest.raises(ValueError, match="INDEXER_ROW_INVARIANT=1"):
        runtime.run_tp4("fake", "test", mtp_lookahead=True, mtp_verify_graph=True, mtp_draft_tokens=3)


@pytest.mark.parametrize("mode", ["plain", "legacy", "eager_lookahead", "captured_lookahead"])
def test_run_routes_only_explicit_lookahead_and_loads_head(monkeypatch, mode):
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    ctx = SimpleNamespace(device=torch.device("cuda", 0), pp_size=1, rank=1, world_size=8,
                          initialized=False)
    model = SimpleNamespace(local_layers=[], dtype=torch.bfloat16, has_mtp=True, reader=None)
    config = SimpleNamespace(text_config=SimpleNamespace(eos_token_id=[3]))
    loaded, calls = [], []
    monkeypatch.setattr(runtime, "init_distributed", lambda: ctx)

    def load(path, actual_ctx, **kwargs):
        loaded.append(kwargs)
        return model, config

    monkeypatch.setattr(runtime, "load_model", load)
    monkeypatch.setattr(runtime, "_load_tokenizer", lambda _p: SimpleNamespace(
        encode=lambda _t: SimpleNamespace(ids=[7,8]), decode=lambda _t: "answer"))
    monkeypatch.setattr(runtime, "_warmup_device", lambda *_a: 0)
    monkeypatch.setattr(runtime, "_maybe_kernel_profile", lambda _ctx, fn: fn())

    def operation(name):
        def invoke(*args, **kwargs):
            calls.append((name, kwargs))
            return [10,11], {"decode_s":1.0,"decode_tps":1.0,"prefill_s":0.0}
        return invoke

    monkeypatch.setattr(runtime, "generate", operation("plain"))
    monkeypatch.setattr(runtime, "generate_mtp_draft", operation("legacy"))
    monkeypatch.setattr(runtime, "generate_mtp_draft_lookahead", operation("lookahead"))
    result = runtime.run_tp4("fake", "test", raw_prompt=True,
        mtp_draft=mode=="legacy", mtp_lookahead="lookahead" in mode,
        mtp_verify_graph=mode=="captured_lookahead", mtp_draft_tokens=3)
    assert result["tokens"] == [10,11]
    assert calls[0][0] == ("lookahead" if "lookahead" in mode else mode)
    assert loaded[0]["mtp"] is (mode!="plain")
    if "lookahead" in mode:
        assert calls[0][1]["verify_graph"] is (mode=="captured_lookahead")
        assert calls[0][1]["graph_window"] == 0
        assert result["stats"]["execution_mode"] == mode


@pytest.mark.parametrize("kwargs,match", [
    ({"mtp_verify_graph":True}, "requires --mtp-lookahead"),
    ({"mtp_lookahead":True,"mtp_draft":True}, "legacy"),
    ({"mtp_lookahead":True,"batch":2}, "batch 1"),
    ({"mtp_lookahead":True,"graph_window":4}, "graph-window 0"),
    ({"mtp_lookahead":True,"mtp_verify_graph":True,"mtp_draft_tokens":5}, "changed tokens"),
])
def test_unqualified_cli_combinations_fail_before_loading(monkeypatch, kwargs, match):
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    monkeypatch.setattr(runtime,"init_distributed",lambda:pytest.fail("invalid options reached CUDA init"))
    with pytest.raises(ValueError,match=match):runtime.run_tp4("fake","test",**kwargs)


def test_captured_cli_requires_the_explicit_strict_arithmetic_flag(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT","0")
    with pytest.raises(ValueError,match="ROW_INVARIANT"):
        runtime.run_tp4("fake","test",mtp_lookahead=True,mtp_verify_graph=True,mtp_draft_tokens=3)


def test_main_passes_new_flags_without_implying_legacy_driver(monkeypatch):
    calls=[]
    monkeypatch.setattr(runtime,"run_tp4",lambda *a,**k:calls.append((a,k)))
    monkeypatch.setattr(sys,"argv",["runtime", "--model","fake","--prompt","test",
                                   "--mtp-lookahead","--mtp-verify-graph","--mtp-draft-tokens","3"])
    runtime.main()
    assert calls[0][1]["mtp_lookahead"] and calls[0][1]["mtp_verify_graph"]
    assert calls[0][1]["mtp"] and not calls[0][1]["mtp_draft"]
