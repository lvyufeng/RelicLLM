"""Expert TP tests: original K32 activation and FP32 reduce-before-narrow."""
from __future__ import annotations

import pytest
import torch

from relic_core.kernels.ascendc_tile_moe import make_tile_moe_workspace
from relic_core.kernels.expert_tp_moe import tp_moe_down_partials, finish_tp_moe
from relic_core.kernels.online_tile_cube import pack_expert_tile_bank


def test_down_reduction_must_happen_before_bf16_narrowing():
    workspace = make_tile_moe_workspace(32, 32, 2, device="cpu")
    workspace.selected.copy_(torch.tensor([0, -1], dtype=torch.int32))
    left = torch.zeros(2, 32)
    right = torch.zeros_like(left)
    left[0].fill_(1.00390625)
    right[0].fill_(-1.0)
    reduced = left + right
    actual = finish_tp_moe(reduced, workspace, backend="torch")
    assert torch.equal(actual, reduced[:1].bfloat16().float())
    wrong = left.bfloat16().float() + right.bfloat16().float()
    assert wrong[0, 0] == 0 and actual[0, 0] != 0


def test_only_tp_leader_emits_to_global_moe_reduce():
    workspace = make_tile_moe_workspace(32, 32, 2, device="cpu")
    workspace.selected.copy_(torch.tensor([0, 1], dtype=torch.int32))
    partial = torch.ones(2, 32)
    assert not finish_tp_moe(partial, workspace, backend="torch", emit=False).count_nonzero()
    assert torch.equal(finish_tp_moe(partial, workspace, backend="torch"), torch.full((1, 32), 2.0))


def test_finalization_checks_poison_and_fp32_boundary():
    workspace = make_tile_moe_workspace(32, 32, 2, device="cpu")
    workspace.route_error.fill_(1)
    assert torch.isnan(finish_tp_moe(torch.ones(2, 32), workspace, backend="torch")).all()
    with pytest.raises(ValueError, match="FP32"):
        finish_tp_moe(torch.ones(2, 32, dtype=torch.bfloat16), workspace, backend="torch")


def test_local_down_partials_are_fp32_and_route_owned_experts_only():
    generator = torch.Generator().manual_seed(7102)
    dim, local_mid = 128, 128
    banks = {}
    for name in ("w1", "w3", "w2"):
        code = torch.randint(0, 256, (4, 128, 64), generator=generator, dtype=torch.uint8)
        scales = torch.full((4, 128, 4), 120, dtype=torch.uint8)
        banks[name] = pack_expert_tile_bank(code, scales)
    workspace = make_tile_moe_workspace(dim, local_mid, 4, device="cpu", group_k=4)
    x = torch.randn(1, dim, generator=generator).bfloat16()
    weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    indices = torch.tensor([[7, 0, 6, 3]])
    result = tp_moe_down_partials(x, weights, indices, banks, ep_rank=0, ep_world=2,
                                n_experts=8, workspace=workspace, backend="torch")
    assert result is workspace.down_accumulation
    assert result.dtype == torch.float32
    assert workspace.selected.tolist() == [0, 6, -1, -1]
    assert result[0:2].abs().max() > 0
    assert not result[2:].count_nonzero()
