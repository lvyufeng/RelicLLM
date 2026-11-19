"""The serving adapter opts into one reusable graph arena, symmetrically across ranks."""
import pytest
from types import SimpleNamespace
from relicllm.models.qwen4_exp import graph_session
from test_qwen4_exp_backend_mtp import _args, _install, _run


def test_explicit_session_is_reused_by_the_adapter_and_closed(monkeypatch):
    monkeypatch.setenv('POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_MTP_SESSION','1')
    model=SimpleNamespace(has_mtp=True,config=SimpleNamespace(ngram_size=3,primary_eos_token_id=2))
    backend,calls=_install(monkeypatch,args=_args(speculative_method='mtp',speculative_tokens=3),model=model)
    _run(backend);_run(backend)
    first=calls['mtp'][0][1]['graph_session'];second=calls['mtp'][1][1]['graph_session']
    assert isinstance(first,graph_session.GraphSessionPool) and first is second
    assert first.model is backend._model
    backend.close();assert first.closed and backend._graph_session is None


def test_close_from_a_token_callback_waits_for_the_active_owner(monkeypatch):
    import torch
    from relicllm.models.qwen4_exp import runtime
    monkeypatch.setenv('POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_MTP_SESSION','1')
    model=SimpleNamespace(has_mtp=True,config=SimpleNamespace(ngram_size=3,primary_eos_token_id=2))
    backend,_calls=_install(monkeypatch,args=_args(speculative_method='mtp',speculative_tokens=3),model=model)
    owners=[]

    def generate(_model,ctx,inputs,**kwargs):
        pool=kwargs['graph_session']
        key=graph_session.GraphSessionKey.build(model=model,ctx=ctx,width=32,draft_tokens=3,rows=4)
        lease=pool.checkout(key)
        owner=SimpleNamespace(released=False,_saved_attrs=[],_saved_rows_buffers=[])
        owner.release=lambda: setattr(owner,'released',True)
        owners.append(owner)
        lease.adopt(cache=SimpleNamespace(layers=[],ple_states={},length=0),verify=owner)
        kwargs['on_token'](100)
        assert pool.closed and not owner.released and backend._graph_session is pool
        assert not backend._closed and not backend.health().ready
        assert kwargs['on_step']()
        lease.finish(successful=False)
        return [100],{'decode_s':0.1,'prefill_s':0.2,'stopped':'cancel'}

    monkeypatch.setattr(runtime,'generate_mtp_draft_lookahead',generate)
    _run(backend,on_token=lambda _token:backend.close())
    assert owners[0].released and backend._closed and backend._graph_session is None


def test_default_serving_keeps_the_request_local_lifetime(monkeypatch):
    monkeypatch.setenv('POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_MTP_SESSION','0')
    backend,calls=_install(monkeypatch,args=_args(speculative_method='mtp',speculative_tokens=3))
    _run(backend)
    assert calls['mtp'][0][1]['graph_session'] is None and backend._graph_session is None


def test_near_native_short_tail_does_not_request_an_incompatible_session(monkeypatch):
    monkeypatch.setenv('POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_MTP_SESSION','1')
    backend,calls=_install(monkeypatch,args=_args(speculative_method='mtp',speculative_tokens=3))
    backend._ensure_loaded()
    backend._config.text_config.max_position_embeddings=12
    backend._run_mtp(list(range(10)),2,'edge',on_token=None,stop=None)
    assert calls['mtp'][0][1]['graph_session'] is None and backend._graph_session is None


def test_one_token_request_does_not_claim_an_unusable_recording(monkeypatch):
    monkeypatch.setenv('POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_MTP_SESSION','1')
    backend,calls=_install(monkeypatch,args=_args(speculative_method='mtp',speculative_tokens=3))
    _run(backend,budget=1)
    assert calls['mtp'][0][1]['graph_session'] is None and backend._graph_session is None
