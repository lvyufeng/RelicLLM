"""Runtime graph reuse is explicit, correctly scoped, and abandoned on prefill failure."""
import pytest
import torch
from relicllm.models.qwen4_exp import runtime, graph_session
from test_qwen4_exp_lookahead_contract import _install, _Verify, _Draft


def _fixture(monkeypatch, **model_options):
    model,ctx,_clock=_install(monkeypatch,captured=True,**model_options)
    monkeypatch.setenv('POCKETLLM_QWEN4_MTP_TAIL_GRAPH','0')
    monkeypatch.setenv('POCKETLLM_QWEN4_MTP_STARTUP_GRAPH','0')
    original=graph_session.GraphLease.begin_request

    def fake_reset(lease):
        original(lease)
        # This CPU fixture represents consumed ids as a Python list; real caches keep recurrent
        # tensors instead. Reset only this fixture-specific ledger after the real metadata reset.
        lease.cache.state=[]

    monkeypatch.setattr(graph_session.GraphLease,'begin_request',fake_reset)
    pool=graph_session.GraphSessionPool(model=model)
    return model,ctx,pool


def test_second_request_reuses_recordings_and_keeps_per_request_rates(monkeypatch):
    model,ctx,pool=_fixture(monkeypatch)
    args=dict(max_new_tokens=6,draft_tokens=3,verify_graph=True,graph_session=pool)
    first,stats1=runtime.generate_mtp_draft_lookahead(model,ctx,torch.tensor([[7,8]]),**args)
    cache=model.cache;verify=model.verify_graph;draft=model.draft_graph
    second,stats2=runtime.generate_mtp_draft_lookahead(model,ctx,torch.tensor([[9,10]]),**args)
    assert first==second==[10,11,12,13,14,15]
    assert model.cache is cache and model.verify_graph is verify and model.draft_graph is draft
    assert stats1['graph_build_s']>0 and not stats1['graph_session_reused']
    assert stats2['graph_build_s']==0 and stats2['graph_session_reused']
    assert stats1['graph_session_retained'] and stats2['graph_session_retained']
    assert pool.summary()['hits']==1 and pool.summary()['adopted']==1
    assert not verify.released and not draft.released
    pool.close()
    assert verify.released and draft.released


def test_prefill_failure_discards_the_lease_and_allows_the_next_request(monkeypatch):
    model,ctx,pool=_fixture(monkeypatch)
    args=dict(max_new_tokens=6,draft_tokens=3,verify_graph=True,graph_session=pool)
    original=model.forward
    model.forward=lambda *_args,**_kwargs: (_ for _ in ()).throw(RuntimeError('prefill failed'))
    with pytest.raises(RuntimeError,match='prefill failed'):
        runtime.generate_mtp_draft_lookahead(model,ctx,torch.tensor([[7,8]]),**args)
    assert pool._active_lease is None
    model.forward=original
    output,stats=runtime.generate_mtp_draft_lookahead(model,ctx,torch.tensor([[7,8]]),**args)
    assert output==[10,11,12,13,14,15] and not stats['graph_session_reused']
    pool.close()


def test_native_tail_suspends_inner_owners_without_invalidating_the_session(monkeypatch):
    model,ctx,pool=_fixture(monkeypatch,native_limit=12)
    monkeypatch.setenv('POCKETLLM_QWEN4_MTP_TAIL_GRAPH','1')
    for cls in (_Verify,_Draft):
        monkeypatch.setattr(cls,'suspend_bindings',lambda owner: setattr(owner,'_bindings_suspended',True),raising=False)
        monkeypatch.setattr(cls,'resume_bindings',lambda owner: setattr(owner,'_bindings_suspended',False),raising=False)
    args=dict(max_new_tokens=10,draft_tokens=3,verify_graph=True,graph_session=pool)
    first,s1=runtime.generate_mtp_draft_lookahead(model,ctx,torch.tensor([[7,8]]),**args)
    second,s2=runtime.generate_mtp_draft_lookahead(model,ctx,torch.tensor([[9,10]]),**args)
    assert first==second==list(range(10,20))
    assert s1['tail_graph_steps']>0 and s2['tail_graph_steps']>0
    assert s1['graph_session_boundary_suspensions']==s2['graph_session_boundary_suspensions']==2
    assert s1['graph_session_retained'] and s2['graph_session_reused']
    assert s2['graph_build_s']==0 and pool.summary()['boundary_invalidations']==0
    pool.close()


def test_group_disagreement_makes_every_rank_cold_before_prefill(monkeypatch):
    model,ctx,pool=_fixture(monkeypatch)
    ctx.world_size=2
    allocate=model.make_cache
    def fresh_cache(**kwargs):
        model.verify_graph=None  # Discard the CPU fixture's old-capture ledger, not model state.
        return allocate(**kwargs)
    monkeypatch.setattr(model,'make_cache',fresh_cache)
    monkeypatch.setattr(runtime,'gather_logits',lambda value,_ctx:value)
    monkeypatch.setattr(runtime,'_mtp_session_group_count',lambda _ctx,ready: int(ready))
    args=dict(max_new_tokens=6,draft_tokens=3,verify_graph=True,graph_session=pool)
    _first,_s1=runtime.generate_mtp_draft_lookahead(model,ctx,torch.tensor([[7,8]]),**args)
    old=model.verify_graph
    output,stats=runtime.generate_mtp_draft_lookahead(model,ctx,torch.tensor([[7,8]]),**args)
    assert output==[10,11,12,13,14,15]
    assert stats['graph_session_group_fallbacks']==1 and not stats['graph_session_reused']
    assert stats['graph_build_s']>0 and old.released and model.verify_graph is not old
    pool.close()


def test_a_peer_validation_failure_unwinds_before_graph_alignment(monkeypatch):
    model,ctx,pool=_fixture(monkeypatch)
    args=dict(max_new_tokens=6,draft_tokens=3,verify_graph=True,graph_session=pool)
    runtime.generate_mtp_draft_lookahead(model,ctx,torch.tensor([[7,8]]),**args)
    votes=iter([1,0])
    monkeypatch.setattr(runtime,'_mtp_session_group_count',lambda _ctx,_ready:next(votes))
    with pytest.raises(graph_session.SessionInvalidated,match="a TP rank"):
        runtime.generate_mtp_draft_lookahead(model,ctx,torch.tensor([[7,8]]),**args)
    assert pool._active_lease is None and model.verify_graph.released
    pool.close()


def test_session_vote_uses_an_explicit_int32_sum(monkeypatch):
    from types import SimpleNamespace
    import torch.distributed as dist
    monkeypatch.setattr(dist,'is_initialized',lambda:True)
    seen=[]
    def reduce(vote,op):
        assert vote.dtype is torch.int32 and vote.shape==(1,) and op is dist.ReduceOp.SUM
        seen.append(vote.item());vote.fill_(3)
    monkeypatch.setattr(dist,'all_reduce',reduce)
    assert runtime._mtp_session_group_count(SimpleNamespace(tp_world_size=8,device='cpu'),True)==3
    assert seen==[1]


def test_failed_adoption_leaves_graphs_to_the_constructor_finalizers(monkeypatch):
    model,ctx,pool=_fixture(monkeypatch)
    def reject(_lease,**_kwargs):raise graph_session.GraphSessionError('adoption failed')
    monkeypatch.setattr(graph_session.GraphLease,'adopt',reject)
    with pytest.raises(graph_session.GraphSessionError,match='adoption failed'):
        runtime.generate_mtp_draft_lookahead(model,ctx,torch.tensor([[7,8]]),max_new_tokens=6,
                                           draft_tokens=3,verify_graph=True,graph_session=pool)
    assert pool._active_lease is None and model.verify_graph.released and model.draft_graph.released
    pool.close()


def test_reentrant_request_is_refused_without_abandoning_the_original(monkeypatch):
    model,ctx,pool=_fixture(monkeypatch)
    key=graph_session.GraphSessionKey.build(model=model,ctx=ctx,width=18,draft_tokens=3,rows=4)
    lease=pool.checkout(key)
    with pytest.raises(graph_session.GraphSessionError,match='active request'):
        runtime.generate_mtp_draft_lookahead(model,ctx,torch.tensor([[7,8]]),max_new_tokens=6,
                                           draft_tokens=3,verify_graph=True,graph_session=pool)
    assert pool._active_lease is lease and not lease._finished
    lease.abandon();pool.close()
