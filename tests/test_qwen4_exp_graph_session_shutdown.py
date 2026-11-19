"""An active request, not the pool's close caller, owns final CUDA graph release."""
from types import SimpleNamespace
import pytest
import torch
from relicllm.models.qwen4_exp.graph_session import GraphSessionPool, GraphSessionKey, GraphSessionError


class Owner:
    def __init__(self,fail_suspend=False):
        self.released=False;self.fail_suspend=fail_suspend;self._saved_attrs=[];self._saved_rows_buffers=[]
        self.ids=torch.zeros(1,dtype=torch.long)
    def suspend_bindings(self):
        if self.fail_suspend:raise RuntimeError('suspend failed')
        self._bindings_suspended=True
    def resume_bindings(self):self._bindings_suspended=False
    def release(self):self.released=True


def adopted(fail_suspend=False):
    model=SimpleNamespace(config=SimpleNamespace(ngram_size=3,primary_eos_token_id=2))
    pool=GraphSessionPool(model=model)
    key=GraphSessionKey.build(model=model,ctx=SimpleNamespace(device='cpu'),width=32,draft_tokens=3,rows=4)
    lease=pool.checkout(key);owner=Owner(fail_suspend)
    cache=SimpleNamespace(layers=[],ple_states={},length=0,ple_token_history=torch.tensor([[1,2]]))
    lease.adopt(cache=cache,verify=owner)
    return pool,key,lease,owner


def test_close_quiesces_but_does_not_destroy_an_active_recording():
    pool,key,lease,owner=adopted()
    pool.close()
    assert pool.closed and not owner.released and not lease._finished
    with pytest.raises(GraphSessionError,match='closed'):pool.checkout(key)
    lease.finish()
    assert owner.released and lease.state=='discarded'
    assert pool.summary()['released']==1 and pool.summary()['finishes']==1
    pool.close();assert pool.summary()['closes']==1


def test_suspend_failure_discards_without_leaking_or_masking_an_error():
    pool,key,lease,owner=adopted(fail_suspend=True)
    lease.finish()
    assert owner.released and lease.state=='discarded'
    assert pool.summary()['released']==1 and pool.summary()['address_invalidations']==1
    assert not pool.checkout(key).reused
    pool.abandon_active();pool.close()


def test_same_pointer_and_shape_with_changed_strides_is_not_replayable():
    from relicllm.models.qwen4_exp.graph_session import SessionInvalidated
    pool,key,lease,owner=adopted()
    owner.ids=torch.arange(4).reshape(2,2)
    # Adopt a fresh fingerprint now that the owner has a two-dimensional buffer.
    lease.abandon();lease=pool.checkout(key);owner.released=False
    lease.adopt(cache=SimpleNamespace(layers=[],ple_states={},length=0),verify=owner)
    lease.finish();warm=pool.checkout(key);warm.begin_request()
    pointer=owner.ids.data_ptr();owner.ids=owner.ids.t()
    assert owner.ids.data_ptr()==pointer and owner.ids.shape==(2,2)
    with pytest.raises(SessionInvalidated,match='changed identity'):warm.resume()
    warm.abandon();pool.close()


def test_abandon_counts_released_owners_and_unadopted_finishes():
    pool,key,lease,owner=adopted()
    lease.abandon()
    assert owner.released and pool.summary()['released']==1
    empty=pool.checkout(key);empty.abandon()
    assert pool.summary()['finishes']==2
    pool.close()
