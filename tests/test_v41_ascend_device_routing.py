"""The no-readback MoE binding is explicit, bounded and does not replace defaults."""
from types import SimpleNamespace

import pytest
import torch

from relicllm.models.deepseek_v4_1.ascend_device_routing import DeviceRoutedDispatch


def store():
    calls = []
    arenas = {name: SimpleNamespace(codes=torch.zeros((8, 64, 32), dtype=torch.uint8),
                                    scales=torch.full((8, 64, 2), 127, dtype=torch.uint8))
              for name in ("w1", "w2", "w3")}
    def original(x, weights, indices):
        calls.append(x.shape[0])
        return torch.zeros((x.shape[0], 64), dtype=torch.float32)
    def readback(*args):
        pytest.fail("legacy host routing was called")
    result = SimpleNamespace(dim=64, inter_dim=64, topk=6, device=torch.device("cpu"),
                             dtype=torch.bfloat16, rank=0, world=2, n_experts=16,
                             owned=tuple(range(0, 16, 2)), swiglu_limit=1.25,
                             _arenas=arenas, _routing=readback, forward=original,
                             _check_forward_inputs=lambda *args: None)
    return result, calls


def test_decode_never_calls_the_legacy_host_router_and_close_restores():
    routed, calls = store()
    original = routed.forward
    binding = DeviceRoutedDispatch([routed], backend="torch", certified_bf16_operands=True)
    with binding:
        result = routed.forward(torch.ones(1, 64, dtype=torch.bfloat16),
                                torch.ones(1, 6) / 6, torch.tensor([[0, 2, 4, 6, 8, 10]]))
        assert result.dtype == torch.float32 and torch.equal(result, torch.zeros_like(result))
        assert not calls
        info = binding.describe()
        assert info["device_routed_calls"] == 1 and info["routing_D2H_calls"] == 0
        assert info["complete_Engram_CPU"] and info["ascending_expert_FP32_sum_unchanged"]
        assert info["native_K_accumulation_changed"]
    assert routed.forward is original


def test_prefill_is_unchanged():
    routed, calls = store()
    with DeviceRoutedDispatch([routed], backend="torch", certified_bf16_operands=True):
        routed.forward(torch.ones(2, 64, dtype=torch.bfloat16),
                       torch.ones(2, 6), torch.zeros(2, 6, dtype=torch.int64))
        assert calls == [2]


def test_source_certificate_is_mandatory():
    routed, _ = store()
    with pytest.raises(ValueError, match="certify"):
        DeviceRoutedDispatch([routed], backend="torch")


def test_shared_workspace_does_not_duplicate_the_resident_bank():
    first, _ = store()
    second, _ = store()
    binding = DeviceRoutedDispatch([first, second], backend="torch", certified_bf16_operands=True).install()
    try:
        info = binding.describe()
        assert info["stores"] == 2 and not info["resident_weight_bank_duplicated"]
        assert info["workspace_bytes"] < 512 * 2**20
        assert binding.banks[id(first)]["w1"][0] is first._arenas["w1"].codes
        assert binding.banks[id(second)]["w1"][0] is second._arenas["w1"].codes
    finally:
        binding.close()


def test_duplicate_dispatcher_is_refused():
    routed, _ = store()
    routed._relic_m1_routed_dispatch_owner = object()
    with pytest.raises(RuntimeError, match="another dispatcher"):
        DeviceRoutedDispatch([routed], backend="torch", certified_bf16_operands=True).install()


def test_no_default_provider_is_patched_by_import():
    from relicllm.models.deepseek_v4_1 import ascend_packed_experts
    before = ascend_packed_experts.packed_expert_forward
    routed, _ = store()
    with DeviceRoutedDispatch([routed], backend="torch", certified_bf16_operands=True):
        assert ascend_packed_experts.packed_expert_forward is before
    assert ascend_packed_experts.packed_expert_forward is before
