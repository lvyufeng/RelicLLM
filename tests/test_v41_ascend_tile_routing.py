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
