"""Explicit tile-bank runtime binding keeps original passthroughs and ownership."""
from types import SimpleNamespace

import pytest
import torch

from relic_core.kernels.online_tile_cube import pack_expert_tile_bank
from relicllm.models.deepseek_v4_1.ascend_tile_routing import TileRoutedDispatch
from relicllm.models.deepseek_v4_1.ascend_device_routing import DeviceRoutedDispatch


def store(*, tiled=True):
    calls = []
    arenas = {}
    for name in ("w1", "w2", "w3"):
        b = torch.zeros((8, 64, 32), dtype=torch.uint8)
        s = torch.full((8, 64, 2), 127, dtype=torch.uint8)
        if tiled:
            b, s = pack_expert_tile_bank(b, s)
        arenas[name] = SimpleNamespace(codes=b, scales=s)
    def original(x, weights, indices):
        calls.append(x.shape[0])
        return torch.zeros((x.shape[0], 64), dtype=torch.float32)
    def readback(*args):
        pytest.fail("original host routing was called in replaced decode")
    return SimpleNamespace(dim=64, inter_dim=64, topk=6, device=torch.device("cpu"),
                           dtype=torch.bfloat16, rank=0, world=2, n_experts=16,
                           owned=tuple(range(0, 16, 2)), swiglu_limit=1.25,
                           _arenas=arenas, _routing=readback, forward=original,
                           _check_forward_inputs=lambda *args: None), calls


def test_tile_decode_and_prefill_restore_without_readback_or_bank_copy():
    routed, calls = store()
    original = routed.forward
    binding = TileRoutedDispatch([routed], backend="torch", certified_bf16_operands=True)
    with binding:
        value = routed.forward(torch.ones(1, 64, dtype=torch.bfloat16),
                               torch.ones(1, 6) / 6, torch.tensor([[0, 2, 4, 6, 8, 10]]))
        assert torch.equal(value, torch.zeros(1, 64))
        assert not calls
        assert binding.banks[id(routed)]["w1"][0] is routed._arenas["w1"].codes
        info = binding.describe()
        assert info["workspace_bytes"] < 1024**2
        assert info["whole_matrix_expanded_weight_scratch_bytes"] == 0
        assert info["routing_D2H_calls"] == 0
        assert info["complete_Engram_CPU"] and info["original_FP32_SiLU_unchanged"]
        routed.forward(torch.ones(2, 64, dtype=torch.bfloat16), torch.ones(2, 6), torch.zeros(2, 6, dtype=torch.int64))
        assert calls == [2]
    assert routed.forward is original


def test_row_major_arena_is_not_silently_reinterpreted_as_tile_or_nz():
    routed, _ = store(tiled=False)
    original = routed.forward
    with pytest.raises(ValueError, match="tile-major"):
        TileRoutedDispatch([routed], backend="torch", certified_bf16_operands=True).install()
    assert routed.forward is original


def test_both_dispatchers_share_one_owner_guard():
    routed, _ = store()
    with TileRoutedDispatch([routed], backend="torch", certified_bf16_operands=True):
        with pytest.raises(RuntimeError, match="another dispatcher"):
            DeviceRoutedDispatch([routed], backend="torch", certified_bf16_operands=True).install()


def test_tile_source_certificate_is_required():
    routed, _ = store()
    with pytest.raises(ValueError, match="certify"):
        TileRoutedDispatch([routed], backend="torch")


def test_multi_layer_tile_workspace_is_shared_and_admitted_before_patching():
    first, _ = store()
    second, _ = store()
    original = first.forward
    with pytest.raises(ValueError, match="budget"):
        TileRoutedDispatch([first, second], backend="torch", certified_bf16_operands=True, max_workspace_bytes=1).install()
    assert first.forward is original
    with TileRoutedDispatch([first, second], backend="torch", certified_bf16_operands=True) as binding:
        assert binding.describe()["stores"] == 2
        assert not binding.describe()["resident_weight_bank_duplicated"]


class ScratchConsumer:
    """Only declared geometry is used; these contracts never execute a consumer."""

    def __init__(self, output_rows="legacy", *, group_k=1):
        self.block_dim = 6
        self.group_k = group_k
        if output_rows != "legacy":
            self.output_rows = output_rows

    def __call__(self, *args, **kwargs):
        pytest.fail("scratch admission must not execute a native consumer")


@pytest.mark.parametrize("rows, expected_rows", [(None, 16), ("legacy", 16), (16, 16), (32, 32)],
                         ids=["no-consumer", "legacy-consumer", "n16", "n32"])
def test_tile_scratch_uses_the_declared_consumer_rows(rows, expected_rows):
    routed, _ = store()
    original = routed.forward
    consumer = None if rows is None else ScratchConsumer(rows)
    with TileRoutedDispatch([routed], consumer=consumer, backend="torch",
                            certified_bf16_operands=True) as binding:
        workspace = binding.workspace
        block_dim = 24 if consumer is None else consumer.block_dim
        group_k = 1 if consumer is None else consumer.group_k
        assert workspace.weight_rows == expected_rows
        assert workspace.weight_storage.shape == (block_dim, 2, expected_rows * 32 * group_k)
        assert getattr(routed, "_relic_device_selected_moe_owner") is binding
    assert routed.forward is original
    assert not hasattr(routed, "_relic_device_selected_moe_owner")


@pytest.mark.parametrize("rows, group_k", [(0, 1), (True, 1), (8, 1), (64, 1), (32, 8)],
                         ids=["zero", "boolean", "n8", "n64", "n32-ring-too-wide"])
def test_invalid_tile_row_geometry_is_refused_before_store_ownership_changes(rows, group_k):
    first, _ = store()
    second, _ = store()
    originals = (first.forward, second.forward)
    binding = TileRoutedDispatch([first, second], consumer=ScratchConsumer(rows, group_k=group_k),
                                backend="torch", certified_bf16_operands=True)
    with pytest.raises(ValueError, match="geometry"):
        binding.install()
    assert not binding.installed and not binding.replacements and binding.workspace is None
    for routed, original in zip((first, second), originals):
        assert routed.forward is original
        assert not hasattr(routed, "_relic_device_selected_moe_owner")


def test_n32_workspace_budget_is_checked_before_patching_any_store():
    control, _ = store()
    with TileRoutedDispatch([control], consumer=ScratchConsumer(16), backend="torch",
                            certified_bf16_operands=True) as baseline:
        n16_budget = baseline.workspace.nbytes
    first, _ = store()
    second, _ = store()
    originals = (first.forward, second.forward)
    binding = TileRoutedDispatch([first, second], consumer=ScratchConsumer(32), backend="torch",
                                certified_bf16_operands=True, max_workspace_bytes=n16_budget)
    with pytest.raises(ValueError, match="budget"):
        binding.install()
    assert not binding.installed and not binding.replacements and binding.workspace is None
    for routed, original in zip((first, second), originals):
        assert routed.forward is original
        assert not hasattr(routed, "_relic_device_selected_moe_owner")
