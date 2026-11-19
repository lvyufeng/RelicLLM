"""PLE reductions split only query rows, preserving decode's four hc groups together."""
import pytest
import torch
from relicllm.models.qwen4_exp.layers import row_invariant_sequence_sum


@pytest.mark.parametrize("rows", [1,2,4,5,6,8,9])
@pytest.mark.parametrize("enabled", [False,True])
@pytest.mark.parametrize("dtype", [torch.bfloat16,torch.float32])
def test_query_sum_dispatch_shape_dtype_and_values(monkeypatch, rows, enabled, dtype):
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT","1" if enabled else "0")
    x=torch.randn(1,rows,4,2560,dtype=dtype)
    calls=[]
    original=torch.Tensor.sum

    def observed(value,*args,**kwargs):
        calls.append((tuple(value.shape),value.is_contiguous(),kwargs.get("dim")))
        return original(value,*args,**kwargs)

    monkeypatch.setattr(torch.Tensor,"sum",observed)
    output=row_invariant_sequence_sum(x)
    expected=original(x,dim=-1,keepdim=True)
    assert output.shape==(1,rows,4,1) and output.dtype is dtype
    assert torch.equal(output,expected)
    active=enabled and dtype is torch.bfloat16 and 2<=rows<=8
    assert len(calls)==(rows if active else 1)
    if active:assert calls==[((1,1,4,2560),True,-1)]*rows


def test_noncontiguous_input_is_copied_per_query_without_changing_hc_grouping(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT","1")
    x=torch.randn(1,4,4,5120,dtype=torch.bfloat16)[...,::2]
    assert not x.is_contiguous()
    expected=torch.cat([x[:,i:i+1].contiguous().sum(-1,keepdim=True) for i in range(4)],1)
    actual=row_invariant_sequence_sum(x)
    assert torch.equal(actual,expected)
