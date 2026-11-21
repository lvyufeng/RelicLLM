"""Exact CPU scheduling proofs for current-token Engram prefetch."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from relicllm.models.deepseek_v4_1 import modules
from relicllm.models.deepseek_v4_1.host_engram_prefetch import HostEngramPrefetch


class Table:
    def __init__(self, dtype):
        self.values = torch.randn(128, 32, generator=torch.Generator().manual_seed(603)).to(dtype)
        self.calls = 0
        self.fail = False

    def lookup(self, ids, device):
        self.calls += 1
        if self.fail:
            raise RuntimeError("deliberate host lookup failure")
        return self.values[ids].to(device)


class Layer(nn.Module):
    def __init__(self, index, table):
        super().__init__()
        config = SimpleNamespace(dim=8, hc_mult=2, norm_eps=1e-6)
        layout = SimpleNamespace(layer_ids=[1,14], max_ngram_size=2, n_heads=1, head_dim=32)
        self.engram = modules.Engram(config, index, layout, table, device=torch.device("cpu"))
        with torch.no_grad():
            self.engram.wkv.weight.copy_(torch.randn(24,32,generator=torch.Generator().manual_seed(604+index)).to(self.engram.wkv.weight.dtype)*0.1)


class Model(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.table = Table(dtype)
        self.layers = nn.ModuleList([Layer(1,self.table),Layer(14,self.table)])
        self.register_buffer("inputs",torch.randn(128,2,8,generator=torch.Generator().manual_seed(605)).to(dtype))

    def forward(self, input_ids, start_pos=0, hash_ids=None, image_mask=None, chunk=None):
        value = self.inputs[input_ids]
        mask = None if image_mask is None else ~image_mask
        for layer in self.layers:
            value = layer.engram(value,hash_ids[:,:,layer.engram.layer_hash_index,:],mask)
        return value


def data(span=1):
    return torch.arange(5,5+span).unsqueeze(0),torch.arange(17,17+span*2).reshape(1,span,2,1)


def equal_bits(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.view(torch.uint8),b.view(torch.uint8))


@pytest.mark.parametrize("dtype",[torch.bfloat16,torch.float32])
@pytest.mark.parametrize("mask",[None,True,False])
def test_original_gate_body_and_cpu_result_bits_are_unchanged(monkeypatch,dtype,mask):
    monkeypatch.setattr(modules,"LINEAR_DTYPE",dtype)
    model=Model(dtype)
    baseline=copy.deepcopy(model)
    ids,hashes=data()
    image_mask=None if mask is None else torch.tensor([[mask]])
    original_body=modules.Engram._forward_pass
    with torch.inference_mode():
        expected=baseline(ids,0,hashes,image_mask)
        with HostEngramPrefetch(model) as manager:
            actual=model(ids,0,hashes,image_mask)
            assert equal_bits(actual,expected)
            assert manager.stats["projection_hits"]==2
            assert manager.stats["lookup_hits"]==2
            assert manager.describe()["pending_results"]==0
            assert model.table.calls==2  # one original lookup per layer, not a duplicate
    assert modules.Engram._forward_pass is original_body


def test_prefill_keeps_the_original_body_without_scheduling(monkeypatch):
    monkeypatch.setattr(modules,"LINEAR_DTYPE",torch.bfloat16)
    model=Model(torch.bfloat16)
    baseline=copy.deepcopy(model)
    ids,hashes=data(3)
    with torch.inference_mode():
        expected=baseline(ids,0,hashes)
        with HostEngramPrefetch(model) as manager:
            assert equal_bits(model(ids,0,hashes),expected)
            assert manager.stats["scheduled"]==0


def test_hash_cache_mutation_does_not_modify_the_submitted_cpu_ids(monkeypatch):
    monkeypatch.setattr(modules,"LINEAR_DTYPE",torch.bfloat16)
    model=Model(torch.bfloat16)
    baseline=copy.deepcopy(model)
    ids,hashes=data()
    saved=hashes.clone()
    with torch.inference_mode():
        expected=baseline(ids,0,saved)
        with HostEngramPrefetch(model) as manager:
            manager.prepare(hashes)
            hashes.fill_(91)
            value=model.inputs[ids]
            for layer in model.layers:
                value=layer.engram(value,saved[:,:,layer.engram.layer_hash_index,:])
            assert equal_bits(value,expected)
            assert manager.stats["projection_hits"]==2
            manager.clear()


def test_different_hashes_never_reuse_an_old_projection(monkeypatch):
    monkeypatch.setattr(modules,"LINEAR_DTYPE",torch.bfloat16)
    model=Model(torch.bfloat16)
    baseline=copy.deepcopy(model)
    ids,hashes=data()
    changed=hashes+11
    with torch.inference_mode():
        expected=baseline(ids,0,changed)
        with HostEngramPrefetch(model) as manager:
            manager.prepare(hashes)
            value=model.inputs[ids]
            for layer in model.layers:
                value=layer.engram(value,changed[:,:,layer.engram.layer_hash_index,:])
            assert equal_bits(value,expected)
            assert manager.stats["projection_hits"]==0
            assert manager.stats["stale_misses"]==2
            manager.clear()


def test_nonzero_rank_does_not_prefetch_or_replace_cpu_calls(monkeypatch):
    monkeypatch.setattr(modules,"LINEAR_DTYPE",torch.bfloat16)
    model=Model(torch.bfloat16)
    original=model.forward
    quantized=modules.quantized_linear
    ids,hashes=data()
    with HostEngramPrefetch(model,rank=1) as manager:
        assert model.forward==original
        assert modules.quantized_linear is quantized
        with torch.inference_mode():model(ids,0,hashes)
        assert manager.stats["scheduled"]==0


def test_worker_errors_propagate_and_all_hooks_are_restored(monkeypatch):
    monkeypatch.setattr(modules,"LINEAR_DTYPE",torch.bfloat16)
    model=Model(torch.bfloat16)
    original=model.forward
    quantized=modules.quantized_linear
    lookup=model.table.lookup
    model.table.fail=True
    ids,hashes=data()
    with pytest.raises(RuntimeError,match="deliberate host lookup"):
        with HostEngramPrefetch(model) as manager:
            model(ids,0,hashes)
    assert model.forward==original
    assert modules.quantized_linear is quantized
    assert model.table.lookup==lookup
    assert manager.pending=={} and manager.executor is None


def test_repeated_positions_remain_bounded_and_restore_the_hash_span(monkeypatch):
    monkeypatch.setattr(modules,"LINEAR_DTYPE",torch.bfloat16)
    model=Model(torch.bfloat16)
    baseline=copy.deepcopy(model)
    ids,hashes=data()
    with torch.inference_mode():
        with HostEngramPrefetch(model) as manager:
            for step in (0,1,2,0,2,1,0):
                current=hashes+step
                assert equal_bits(model(ids,step,current),baseline(ids,step,current))
                assert manager.describe()["pending_results"]==0
                assert manager.describe()["maximum_pending_results"]==2


def test_nested_global_dispatch_install_is_refused(monkeypatch):
    monkeypatch.setattr(modules,"LINEAR_DTYPE",torch.bfloat16)
    model=Model(torch.bfloat16)
    with HostEngramPrefetch(model):
        with pytest.raises(RuntimeError,match="only one explicit"):
            HostEngramPrefetch(model).install()
