"""NPU original activation call-site tests; not a whole-model acceptance claim."""
from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch

from relic_core.kernels import ops
from relic_core.kernels.block_fp8_quant import quantize_block_fp8
from relic_core.kernels.unpack import unpack_fp8_to_bf16
from relicllm.models.deepseek_v4_1 import kernels
from relicllm.models.deepseek_v4_1.ascend_activation import act_quant, fp4_act_quant, fp8_quant_inplace


def _npu_name():
    name = os.environ.get("RELIC_INFERENCE_TEST_DEVICE")
    if not name:
        pytest.skip("requires a leased RELIC_INFERENCE_TEST_DEVICE=npu:0")
    import torch_npu  # noqa: F401 - registers the device

    return name


@pytest.mark.parametrize("shape", [(1, 64), (2, 128), (1, 3, 128)])
def test_npu_inplace_fp8_uses_original_codes_and_scale(shape):
    name = _npu_name()
    generator = torch.Generator().manual_seed(616)
    x = torch.randn(shape, generator=generator).to(torch.bfloat16)
    x.flatten()[0] = -0.0
    raw, scale = quantize_block_fp8(x.reshape(-1, shape[-1]), backend="torch")
    expected = unpack_fp8_to_bf16(raw, scale, block_shape=(1, 32), backend="torch").reshape(shape)
    device = torch.device(name)
    actual = x.to(device)
    result = fp8_quant_inplace(actual, 32, "ue8m0", torch.float8_e8m0fnu)
    torch.npu.synchronize(device)
    assert result is actual
    assert torch.equal(actual.cpu().view(torch.int16), expected.view(torch.int16))


def test_npu_fp8_binding_refuses_non_original_width():
    x = torch.ones(1, 128, dtype=torch.bfloat16, device=_npu_name())
    with pytest.raises(ValueError, match="block32"):
        fp8_quant_inplace(x, 128, "ue8m0", torch.float8_e8m0fnu)


def test_npu_non_inplace_fp8_cannot_masquerade_bytes_as_native_fp8():
    stub = SimpleNamespace(device=SimpleNamespace(type="npu"))
    with pytest.raises(ValueError, match="inplace=True"):
        act_quant(stub, 32, "ue8m0", torch.float8_e8m0fnu, False)


def test_cpu_fp8_call_forwards_the_unchanged_legacy_arguments(monkeypatch):
    x = torch.ones(1, 32, dtype=torch.bfloat16)
    sentinel = object()
    calls = []

    def legacy(*arguments):
        calls.append(arguments)
        return sentinel

    monkeypatch.setattr(ops, "act_quant", legacy)
    assert act_quant(x, 32, "ue8m0", torch.float32, True) is sentinel
    assert calls == [(x, 32, "ue8m0", torch.float32, True)]


@pytest.mark.parametrize("scale_dtype", [None, torch.float8_e8m0fnu])
def test_cpu_indexer_fp4_forwards_only_the_three_legacy_arguments(monkeypatch, scale_dtype):
    x = torch.ones(1, 32, dtype=torch.bfloat16)
    sentinel = object()
    calls = []

    def legacy(*arguments):
        calls.append(arguments)
        return sentinel

    monkeypatch.setattr(ops, "fp4_act_quant", legacy)
    assert fp4_act_quant(x, 32, True, scale_dtype) is sentinel
    assert calls == [(x, 32, True)]


def test_cpu_e4m3_fp4_selects_the_existing_compressed_kv_binding(monkeypatch):
    x = torch.ones(1, 16, dtype=torch.bfloat16)
    sentinel = object()
    calls = []

    def legacy(*arguments):
        calls.append(arguments)
        return sentinel

    monkeypatch.setattr(kernels, "fp4_act_quant_e4m3", legacy)
    assert fp4_act_quant(x, 16, True, torch.float8_e4m3fn) is sentinel
    assert calls == [(x, 16, True)]


def test_fp4_cannot_quietly_select_e8m0_for_an_unknown_scale_dtype():
    with pytest.raises(ValueError, match="E8M0 or E4M3"):
        fp4_act_quant(torch.ones(1, 32), scale_dtype=torch.float32)
