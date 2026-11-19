"""Packed Q norms must retain one-query dispatch and strides under strict verification."""
import pytest
import torch

from relicllm.models.qwen4_exp import layers


@pytest.mark.parametrize("rows", [2,4,5,6,8])
def test_indexer_row_gap_uses_the_same_fused_dispatch_as_each_decode_query(monkeypatch, rows):
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT","1")
    packed=torch.randn(1,rows,640,dtype=torch.bfloat16)
    query=packed[...,:512].reshape(1,rows,4,128)
    assert not query.is_contiguous() and query[:,:1].is_contiguous()
    seen=[]

    def fused(x,*args):
        seen.append((tuple(x.shape),x.is_contiguous(),x.stride()))
        return x*2 if x.is_contiguous() else None

    monkeypatch.setattr(layers,"_cuda_grouped_rms_norm",fused)
    norm=layers.RMSNorm(torch.zeros(128,dtype=torch.bfloat16),1e-6)
    actual=norm(query)
    assert torch.equal(actual,query*2)
    assert len(seen)==rows and all(shape==(1,1,4,128) and contiguous
                                   for shape,contiguous,_stride in seen)


@pytest.mark.parametrize("rows", [2,4,6,8])
def test_qsa_head_gap_is_not_copied_into_an_incorrect_fused_norm(monkeypatch, rows):
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT","1")
    packed=torch.randn(1,rows,3,512,dtype=torch.bfloat16)
    query=packed[...,:256]
    assert not query.is_contiguous() and not query[:,:1].is_contiguous()
    seen=[]

    def observe(x,*args):
        seen.append((tuple(x.shape),x.is_contiguous(),x.stride()))
        return None

    monkeypatch.setattr(layers,"_cuda_grouped_rms_norm",observe)
    norm=layers.RMSNorm(torch.zeros(256,dtype=torch.bfloat16),1e-6)
    actual=norm(query)
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT","0")
    expected=torch.cat([norm(query[:,row:row+1]) for row in range(rows)],1)
    assert torch.equal(actual,expected)
    assert len(seen)==2*rows
    assert all(shape==(1,1,3,256) and not contiguous and stride[-2]==512
               for shape,contiguous,stride in seen)


def test_default_wider_norm_dispatch_is_untouched(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT","0")
    query=torch.randn(1,4,640,dtype=torch.bfloat16)[...,:512].reshape(1,4,4,128)
    seen=[]
    monkeypatch.setattr(layers,"_cuda_grouped_rms_norm",lambda x,*args: seen.append(x.shape))
    layers.RMSNorm(torch.zeros(128,dtype=torch.bfloat16),1e-6)(query)
    assert seen==[torch.Size([1,4,4,128])]
