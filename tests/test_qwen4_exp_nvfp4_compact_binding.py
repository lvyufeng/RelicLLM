"""The exact compact launch is explicit and never silently measures the old native entry."""
from types import SimpleNamespace
import pytest
import torch
from relicllm.models.qwen4_exp.moe import NvFp4SlotExpertMoE


@pytest.mark.parametrize('rows',[1,3,4,5,8,512])
@pytest.mark.parametrize('compact',[False,True])
def test_compact_dispatch_changes_only_the_native_entry(monkeypatch,rows,compact):
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_EXACT_FAST','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_ROWS','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_COMPACT','1' if compact else '0')
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_COMPACT_MIN_ROWS','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_STORE','0')
    table=tuple(torch.zeros(1) for _ in range(8));calls=[]
    def entry(name):
        def invoke(x,ids,weights,*args):
            assert x.shape==(rows,16) and x.dtype is torch.float32
            assert ids.shape==weights.shape==(rows,3) and ids.dtype is torch.int64
            assert torch.equal(ids,torch.tensor([0,-1,1]).repeat(rows,1))
            assert all(a is b for a,b in zip(args[:6],table[:6]))
            assert args[6:10]==(0,0.0,1.0,1.0) and args[10] is table[6] and args[11] is table[7]
            assert args[12] is False
            calls.append(name);return x+1
        return invoke
    backend=NvFp4SlotExpertMoE.__new__(NvFp4SlotExpertMoE)
    backend._ops=SimpleNamespace(nvfp4_moe_rows_forward_exact_fast=entry('old'),
                                 nvfp4_moe_rows_forward_exact_fast_compact=entry('compact'))
    backend._table=lambda _layer:table;backend.stage_calls=0
    backend.world_size,backend.rank,backend.num_experts=8,1,512
    x=torch.zeros(1,rows,16,dtype=torch.bfloat16)
    actual=backend(0,x,torch.tensor([1,2,9]).repeat(rows,1),torch.ones(rows,3))
    assert actual.shape==x.shape and actual.dtype is x.dtype and torch.equal(actual,x+1)
    assert calls==['compact' if compact else 'old']


@pytest.mark.parametrize('rows,expected',[(1,'old'),(4,'old'),(512,'compact')])
def test_measured_prefill_threshold_leaves_decode_launch_unchanged(monkeypatch,rows,expected):
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_COMPACT','1')
    monkeypatch.delenv('POCKETLLM_QWEN4_NVFP4_COMPACT_MIN_ROWS',raising=False)
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_STORE','0')
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_EXACT_FAST','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_ROWS','1')
    calls=[]
    def entry(name):
        def call(x,*_args):calls.append(name);return x
        return call
    backend=NvFp4SlotExpertMoE.__new__(NvFp4SlotExpertMoE)
    backend._ops=SimpleNamespace(nvfp4_moe_rows_forward_exact_fast=entry('old'),
        nvfp4_moe_rows_forward_exact_fast_compact=entry('compact'))
    backend._table=lambda _layer:tuple(torch.zeros(1) for _ in range(8));backend.stage_calls=0
    backend.world_size,backend.rank,backend.num_experts=8,0,512
    x=torch.zeros(rows,16,dtype=torch.bfloat16)
    assert torch.equal(backend(0,x,torch.zeros(rows,2,dtype=torch.long),torch.ones(rows,2)),x)
    assert calls==[expected]


@pytest.mark.parametrize('reason',['missing_binary','fast_disabled','rows_disabled'])
def test_requested_compact_path_cannot_fall_back_silently(monkeypatch,reason):
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_COMPACT','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_COMPACT_MIN_ROWS','1')
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_STORE','0')
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_EXACT_FAST','0' if reason=='fast_disabled' else '1')
    monkeypatch.setenv('POCKETLLM_QWEN4_NVFP4_ROWS','0' if reason=='rows_disabled' else '1')
    backend=NvFp4SlotExpertMoE.__new__(NvFp4SlotExpertMoE)
    backend._ops=SimpleNamespace(nvfp4_moe_rows_forward_exact_fast=lambda *_args:pytest.fail('wrong fallback'))
    if reason!='missing_binary':backend._ops.nvfp4_moe_rows_forward_exact_fast_compact=lambda *_args:pytest.fail('wrong dispatch')
    backend._table=lambda _layer:();backend.stage_calls=0
    with pytest.raises(RuntimeError,match='compact NVFP4'):
        backend(0,torch.zeros(4,16,dtype=torch.bfloat16),torch.zeros(4,2,dtype=torch.long),torch.ones(4,2))
