"""Nonzero original-operand and scratch-admission tests for the tile MoE oracle."""
import pytest
import torch

from relic_core.kernels.ascendc_tile_moe import device_tile_fp4_moe, make_tile_moe_workspace
from relic_core.kernels.device_selected_moe import device_selected_fp4_moe, make_device_moe_workspace
from relic_core.kernels.online_tile_cube import pack_expert_tile_bank


def fixture():
    generator = torch.Generator().manual_seed(612)
    banks = {}
    for name in ("w1", "w2", "w3"):
        banks[name] = (torch.randint(0, 256, (8, 128, 64), dtype=torch.uint8, generator=generator),
                       torch.full((8, 128, 4), 121, dtype=torch.uint8))
    tiles = {name: pack_expert_tile_bank(*pair) for name, pair in banks.items()}
    x = (torch.randn((1, 128), generator=generator) * 0.125).bfloat16()
    weights = torch.tensor([[0.1, 0.2, 0.15, 0.25, 0.175, 0.125]], dtype=torch.float32)
    return x, weights, banks, tiles


@pytest.mark.parametrize("ids", [[1, 3, 5, 7, 9, 11], [1, 2, 5, 7, 9, 11],
                                  [10, 3, 2, 7, 6, 11], [10, 0, 2, 14, 6, 4]])
@pytest.mark.parametrize("limit", [0.0, 1.25])
def test_tile_composition_matches_natural_original_i8_stage(ids, limit):
    x, weights, banks, tiles = fixture()
    before = {name: tuple(t.clone() for t in pair) for name, pair in tiles.items()}
    indices = torch.tensor([ids], dtype=torch.int64)
    stage = make_device_moe_workspace(128, 128, 6, device="cpu", projection_mode="weight_quant")
    expected = device_selected_fp4_moe(x, weights, indices, banks, rank=0, world=2, n_experts=16,
                                       swiglu_limit=limit, workspace=stage, backend="torch",
                                       projection_mode="weight_quant", fused_epilogue=True)
    scratch = make_tile_moe_workspace(128, 128, 6, device="cpu", group_k=4)
    actual = device_tile_fp4_moe(x, weights, indices, tiles, rank=0, world=2, n_experts=16,
                                 swiglu_limit=limit, workspace=scratch, backend="torch",
                                 certified_bf16_operands=True)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    for name, pair in tiles.items():
        assert all(torch.equal(t, original) for t, original in zip(pair, before[name]))


def test_full_model_scratch_is_bounded_before_bank_allocation():
    scratch = make_tile_moe_workspace(5120, 2304, 6, device="cpu", group_k=4)
    assert scratch.nbytes == 430188
    assert scratch.weight_storage.numel() * scratch.weight_storage.element_size() == 196608
    with pytest.raises(ValueError, match="budget"):
        make_tile_moe_workspace(5120, 2304, 6, device="cpu", group_k=4, max_bytes=430187)


def test_k256_workspace_preserves_small_ring_and_budget_admission():
    scratch = make_tile_moe_workspace(5120, 2304, 6, device="cpu", group_k=8)
    assert scratch.nbytes == 626796
    assert scratch.weight_storage.shape == (24, 2, 4096)
    assert scratch.weight_storage.numel() * scratch.weight_storage.element_size() == 393216
    with pytest.raises(ValueError, match="budget"):
        make_tile_moe_workspace(5120, 2304, 6, device="cpu", group_k=8, max_bytes=626795)
    with pytest.raises(ValueError, match="complete original K32"):
        make_tile_moe_workspace(5120, 1152, 6, device="cpu", group_k=8)


@pytest.mark.parametrize("ids", [[0, 0, 2, 4, 6, 8], [-1, 1, 2, 3, 4, 5], [16, 1, 2, 3, 4, 5]])
def test_tile_cpu_oracle_refuses_duplicate_and_invalid_routes(ids):
    x, weights, _, tiles = fixture()
    scratch = make_tile_moe_workspace(128, 128, 6, device="cpu", group_k=4)
    with pytest.raises(ValueError, match="without replacement"):
        device_tile_fp4_moe(x, weights, torch.tensor([ids]), tiles, rank=0, world=2, n_experts=16,
                            workspace=scratch, backend="torch", certified_bf16_operands=True)


def test_tile_scratch_aliases_are_rejected_before_projection():
    x, weights, _, tiles = fixture()
    scratch = make_tile_moe_workspace(128, 128, 6, device="cpu", group_k=4)
    scratch.up_accumulation = scratch.gate_accumulation
    with pytest.raises(ValueError, match="alias"):
        device_tile_fp4_moe(x, weights, torch.tensor([[0, 2, 4, 6, 8, 10]]), tiles,
                            rank=0, world=2, n_experts=16, workspace=scratch,
                            backend="torch", certified_bf16_operands=True)


def test_original_row_major_data_is_not_a_tile_bank():
    x, weights, banks, _ = fixture()
    scratch = make_tile_moe_workspace(128, 128, 6, device="cpu", group_k=4)
    with pytest.raises(ValueError, match="cyclic tile bank"):
        device_tile_fp4_moe(x, weights, torch.tensor([[0, 2, 4, 6, 8, 10]]), banks,
                            rank=0, world=2, n_experts=16, workspace=scratch,
                            backend="torch", certified_bf16_operands=True)
