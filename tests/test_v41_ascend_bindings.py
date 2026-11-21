"""Runtime seams for the original-format Ascend operator provider.

Device cases require RELIC_ASCEND_EXPERTS_DEVICE=npu:0 and a fresh lease.
CPU/CUDA behavior is preserved; no serving capability guard is relaxed here.
"""
from __future__ import annotations

import importlib.util
import math
import os
from types import SimpleNamespace

import pytest

torch=pytest.importorskip("torch")
from relicllm.models.deepseek_v4_1 import loader, mma
from relicllm.models.deepseek_v4_1.kernels import quantized_linear
from relic_core.kernels.unpack import unpack_fp8_to_bf16


def test_cpu_linear_does_not_enter_the_ascend_binding(monkeypatch):
    from relicllm.models.deepseek_v4_1 import ascend_linear
    def forbid(*args,**kwargs):raise AssertionError("CPU linear entered the NPU provider")
    monkeypatch.setattr(ascend_linear,"quantized_linear_ascend",forbid)
    x=torch.arange(64,dtype=torch.float32).reshape(2,32)
    w=torch.ones(32,32,dtype=torch.float32)
    assert torch.equal(quantized_linear(x,w,quantized=False),torch.nn.functional.linear(x,w))
    quantized_linear(x,w,quantized=True,emulate_reduction=False)


def _selection_fixture(monkeypatch):
    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("NPU device registration required for a constructor-only seam check")
    import torch_npu
    captured={}
    from relicllm.models.deepseek_v4_1 import ascend_experts
    class FakeStore:
        def __init__(self,*args,**kwargs):captured.update(kwargs);self.partial=kwargs["world"]>1
    class EmptyTree(torch.nn.Module):
        def __init__(self,*args,**kwargs):super().__init__();self.routed=kwargs["routed"]
    monkeypatch.setattr(ascend_experts,"AscendRoutedExperts",FakeStore)
    monkeypatch.setattr(loader,"Backbone",EmptyTree)
    monkeypatch.setattr(loader.ShardPlan,"build",lambda *args,**kwargs:object())
    monkeypatch.setattr(loader,"attach_tp",lambda *args:None)
    monkeypatch.setattr(loader,"checkpoint_weights",lambda *args,**kwargs:loader.LoadReport())
    monkeypatch.setattr(loader,"_moe_shape",lambda *args:(384,6))
    config=SimpleNamespace(dim=5120,moe_inter_dim=2304,max_position_embeddings=128,
                           n_layers=1,compress_ratios=(0,),expert_dtype="fp4",swiglu_limit=10.)
    checkpoint=SimpleNamespace(root="unused",keys=lambda:[])
    return config,checkpoint,captured


def test_loader_selects_one_npu_expert_share_without_cuda(monkeypatch):
    config,checkpoint,captured=_selection_fixture(monkeypatch)
    def forbid(*args,**kwargs):raise AssertionError("NPU store entered the CUDA arena")
    monkeypatch.setattr(loader,"DeviceRoutedExperts",forbid)
    result=loader.load_backbone(config,checkpoint,device="cpu",engram=False,
             resident_experts=False,expert_device="npu:0",expert_world=8,world=8,rank=3)
    assert captured["rank"]==3 and captured["world"]==8
    assert str(captured["device"])=="npu:3" and captured["backend"]=="triton"
    assert result.model.routed[0].partial is True


def test_npu_loader_refuses_unsupported_arena_and_deal_by_name(monkeypatch):
    config,checkpoint,_=_selection_fixture(monkeypatch)
    common=dict(device="cpu",engram=False,resident_experts=False,expert_device="npu:0")
    with pytest.raises(ValueError,match="hot/pool/batched"):
        loader.load_backbone(config,checkpoint,expert_pool_rows=1,**common)
    with pytest.raises(ValueError,match="cyclic"):
        loader.load_backbone(config,checkpoint,expert_deal="sorted",**common)
    with pytest.raises(ValueError,match="matching tensor-parallel"):
        loader.load_backbone(config,checkpoint,expert_world=8,**common)


def _npu():
    name=os.environ.get("RELIC_ASCEND_EXPERTS_DEVICE")
    if not name:pytest.skip("requires a leased RELIC_ASCEND_EXPERTS_DEVICE=npu:0")
    import torch_npu
    return torch.device(name)


@pytest.mark.parametrize("leading", [(3,),(2,3)])
def test_npu_dense_linear_preserves_original_code_grid_bits(leading,monkeypatch):
    device=_npu()
    g=torch.Generator().manual_seed(511)
    k,n=64,64
    raw=torch.randint(0,256,(n,k),dtype=torch.uint8,generator=g)
    raw=torch.where((raw&127)==127,torch.full_like(raw,126),raw)
    scale_bytes=torch.randint(115,125,(n//32,k//32),dtype=torch.uint8,generator=g)
    scale=torch.tensor([math.ldexp(1.,int(v)-127) for v in scale_bytes.flatten()],dtype=torch.float32).reshape_as(scale_bytes)
    w=unpack_fp8_to_bf16(raw,scale_bytes,block_shape=(32,32),backend="torch")
    codes=raw.view(torch.float8_e4m3fn)
    mma.attach_source_codes(w,codes,scale)
    x=(torch.randn(*leading,k,generator=g)*0.125).to(torch.bfloat16)
    expected=quantized_linear(x,w,quantized=True,emulate_reduction=True)
    nw=w.to(device)
    mma.attach_source_codes(nw,codes.to(device),scale.to(device))
    def forbid(*args,**kwargs):raise AssertionError("NPU hot path tried float64 exponent recovery")
    monkeypatch.setattr(mma,"e8m0_exponent",forbid)
    monkeypatch.setattr(mma,"codes_from_scale",forbid)
    actual=quantized_linear(x.to(device),nw,quantized=True,emulate_reduction=True)
    torch.npu.synchronize(device)
    assert torch.equal(actual.cpu().view(torch.int16),expected.view(torch.int16))


def test_npu_dense_linear_requires_source_codes_instead_of_silent_fallback():
    device=_npu()
    x=torch.ones(1,32,dtype=torch.bfloat16,device=device)
    w=torch.ones(32,32,dtype=torch.bfloat16,device=device)
    with pytest.raises(RuntimeError,match="original E4M3 codes"):
        quantized_linear(x,w,quantized=True,emulate_reduction=True)
