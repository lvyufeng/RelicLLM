"""Separate FP32 expert slots prevent premature expert/TP narrowing and double counting."""
import pytest
import torch

from relic_core.kernels.global_ffn_slots import prepare_global_ffn_slots, finish_global_ffn_slots


def test_compact_ep_rows_are_aligned_to_global_ascending_expert_slots():
    down = torch.stack([torch.full((32,),1.0),torch.full((32,),2.0),torch.full((32,),99.0)])
    selected=torch.tensor([1,7,-1],dtype=torch.int32)
    ids=torch.tensor([[7,4,1]])
    shared=torch.full((1,32),.125,dtype=torch.bfloat16)
    slots=torch.empty(4,32)
    prepare_global_ffn_slots(down,selected,ids,shared,slots,backend='torch')
    assert torch.equal(slots[:,0],torch.tensor([1.0,0.0,2.0,.125]))
    assert torch.equal(finish_global_ffn_slots(slots,torch.zeros(1,dtype=torch.int32),backend='torch'),torch.full((1,32),3.125))


def test_tp_partials_sum_before_each_expert_bf16_boundary_not_after():
    left=torch.tensor([1.00390625,-1.00390625,0.0]).reshape(3,1).expand(3,32).clone()
    right=torch.tensor([-1.0,1.0,0.0]).reshape(3,1).expand(3,32).clone()
    result=finish_global_ffn_slots(left+right,torch.zeros(1,dtype=torch.int32),backend='torch')
    assert not result.count_nonzero()
    left[1].zero_();right[1].zero_()
    result=finish_global_ffn_slots(left+right,torch.zeros(1,dtype=torch.int32),backend='torch')
    assert (result==.00390625).all()
    wrong=left.bfloat16().float()+right.bfloat16().float()
    assert not wrong.count_nonzero()


def test_different_experts_cannot_be_summed_before_independent_narrowing():
    slots=torch.zeros(3,32)
    slots[0].fill_(1.00390625);slots[1].fill_(-1.0)
    actual=finish_global_ffn_slots(slots,torch.zeros(1,dtype=torch.int32),backend='torch')
    assert not actual.count_nonzero()
    wrong=(slots[0]+slots[1]).bfloat16().float()
    assert (wrong==.00390625).all()


def test_routing_errors_poison_complete_ffn_not_outside_reads():
    assert torch.isnan(finish_global_ffn_slots(torch.ones(7,32),torch.ones(1,dtype=torch.int32),backend='torch')).all()


def test_global_ffn_slots_do_not_accept_bf16_down_partials():
    with pytest.raises(ValueError,match='FP32'):
        prepare_global_ffn_slots(torch.ones(2,32,dtype=torch.bfloat16),torch.tensor([0,1],dtype=torch.int32),
                                torch.tensor([[0,1]]),torch.ones(1,32,dtype=torch.bfloat16),torch.empty(3,32),backend='torch')
