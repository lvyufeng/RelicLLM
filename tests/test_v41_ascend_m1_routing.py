"""M1 binding against the literal original row-index/copy/sum data flow."""
from __future__ import annotations

from collections import defaultdict

import pytest
import torch

from relicllm.models.deepseek_v4_1 import ascend_m1_routing as binding
from relicllm.models.deepseek_v4_1 import ascend_packed_experts as packed


class Store(torch.nn.Module):
    def __init__(self, owned=(0,1,2,3)):
        super().__init__()
        self.dim=64
        self.device=torch.device("cpu")
        self.dtype=torch.bfloat16
        self.backend="torch"
        self.swiglu_limit=8.0
        self.packed_gemv=lambda *_, **__: None
        self._arenas={}
        self.owned=set(owned)
        self.lookups=self.old_calls=self.routing_calls=0

    def _check_forward_inputs(self,x,weights,indices):
        if x.ndim!=2 or x.shape[1]!=self.dim or weights.shape!=indices.shape or weights.shape[0]!=x.shape[0]:
            raise ValueError("invalid original routed input shape")

    def _routing(self,indices):
        self.routing_calls+=1
        result=defaultdict(lambda:([],[]))
        for row, items in enumerate(indices.tolist()):
            for slot, expert in enumerate(items):
                result[expert][0].append(row)
                result[expert][1].append(slot)
        return dict(result)

    def _owns(self,expert):
        return expert in self.owned

    def _expert_at(self,expert):
        return expert

    def forward(self,x,weights,indices):
        self.old_calls+=1
        self._check_forward_inputs(x,weights,indices)
        buckets=self._routing(indices)
        result=torch.zeros((x.shape[0],self.dim),dtype=torch.float32)
        for expert in sorted(buckets):
            if not self._owns(expert):
                continue
            rows,slots=buckets[expert]
            row_index=torch.tensor(rows,dtype=torch.int64)
            slot_index=torch.tensor(slots,dtype=torch.int64)
            gate_weights=weights.index_select(0,row_index).gather(1,slot_index.unsqueeze(1)).float()
            self.lookups+=1
            contribution=packed.packed_expert_forward(x.index_select(0,row_index),expert,
                swiglu_limit=self.swiglu_limit,weights=gate_weights,dtype=self.dtype,
                backend=self.backend,packed_gemv=self.packed_gemv,prefill=x.shape[0]>1)
            result.index_put_((row_index,),result.index_select(0,row_index)+contribution.float())
        return result


@pytest.fixture
def original_body(monkeypatch):
    def body(x,resident,*,weights,dtype,swiglu_limit,backend,packed_gemv,prefill):
        # One unchanged body serves the literal original dispatcher and the new
        # binding. Its fake resident represents expert-specific static operands.
        value=torch.clamp(x.float()*(resident+1)/8,min=-swiglu_limit,max=swiglu_limit)
        return (weights*value).to(dtype)
    monkeypatch.setattr(packed,"packed_expert_forward",body)
    return body


def operands(rows=1):
    x=torch.randn(rows,64,generator=torch.Generator().manual_seed(1661)).bfloat16()
    weights=torch.tensor([[0.1,0.5,0.4]],dtype=torch.float32).expand(rows,3).contiguous()
    indices=torch.tensor([[3,0,2]],dtype=torch.int64).expand(rows,3).contiguous()
    return x,weights,indices


@pytest.mark.parametrize("owned",[(0,1,2,3),(0,2),()])
def test_M1_matches_every_original_FP32_sum_bit_and_keeps_sorted_routing(owned,original_body,monkeypatch):
    expected_store,store=Store(owned),Store(owned)
    x,weights,indices=operands()
    expected=expected_store(x,weights,indices)
    monkeypatch.setattr(binding,"_on_npu",lambda _:True)
    with binding.M1RoutedDispatch([store]) as manager:
        actual=store(x,weights,indices)
        assert torch.equal(actual.view(torch.int32),expected.view(torch.int32))
        assert store.old_calls==0 and store.routing_calls==1
        assert store.lookups==expected_store.lookups
        assert manager.optimized_calls==1
        assert manager.describe()["host_routing_unchanged"] is True


def test_changed_current_input_and_routing_weights_are_not_cached(original_body,monkeypatch):
    store=Store()
    monkeypatch.setattr(binding,"_on_npu",lambda _:True)
    with binding.M1RoutedDispatch([store]):
        for seed in range(3):
            x,weights,indices=operands()
            x=(x.float()+seed).bfloat16()
            weights=weights*(seed+1)
            expected=Store()(x,weights,indices)
            assert torch.equal(store(x,weights,indices).view(torch.int32),expected.view(torch.int32))


def test_prefill_and_duplicate_M1_routes_stay_on_the_original_dispatcher(original_body,monkeypatch):
    monkeypatch.setattr(binding,"_on_npu",lambda _:True)
    store=Store()
    with binding.M1RoutedDispatch([store]) as manager:
        x,w,ids=operands(2)
        assert torch.equal(store(x,w,ids),Store()(x,w,ids))
        x,w,ids=operands()
        ids=torch.tensor([[0,0,2]])
        assert torch.equal(store(x,w,ids),Store()(x,w,ids))
        assert store.old_calls==2 and manager.original_calls==2
        assert manager.optimized_calls==0


def test_CPU_calls_are_untouched(original_body):
    store=Store()
    x,w,ids=operands()
    expected=Store()(x,w,ids)
    with binding.M1RoutedDispatch([store]) as manager:
        assert torch.equal(store(x,w,ids),expected)
        assert store.old_calls==1 and manager.optimized_calls==0


def test_context_restores_original_bound_forward_and_refuses_nested_owner(original_body):
    store=Store()
    original=store.forward
    manager=binding.M1RoutedDispatch([store])
    with manager:
        with pytest.raises(RuntimeError,match="owns"):
            binding.M1RoutedDispatch([store]).install()
    assert store.forward == original
    assert store.forward.__self__ is original.__self__
    assert store.forward.__func__ is original.__func__
    assert not hasattr(store,binding._OWNER)
    manager.close()


def test_invalid_shapes_raise_the_original_named_error(original_body,monkeypatch):
    store=Store()
    monkeypatch.setattr(binding,"_on_npu",lambda _:True)
    with binding.M1RoutedDispatch([store]):
        x,w,ids=operands()
        with pytest.raises(ValueError,match="original routed input"):
            store(x[:,:32],w,ids)


def test_the_expert_body_seam_is_resolved_at_replay_time(original_body,monkeypatch):
    store=Store()
    monkeypatch.setattr(binding,"_on_npu",lambda _:True)
    calls=[]
    with binding.M1RoutedDispatch([store]):
        def instrumented(*args,**kwargs):
            calls.append(args[1])
            return original_body(*args,**kwargs)
        monkeypatch.setattr(packed,"packed_expert_forward",instrumented)
        store(*operands())
    assert calls==[0,2,3]
