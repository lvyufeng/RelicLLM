"""Exact activation memoization, CPU Engram bypass and reversible provider ownership."""
from __future__ import annotations

import pytest
import torch

from relic_core.kernels import block_fp8_quant, unpack, fused_fp8_activation_decode
from relicllm.models.deepseek_v4_1 import fused_activation_binding as binding


@pytest.fixture
def forced_device_binding(monkeypatch):
    original_quantize = block_fp8_quant.quantize_block_fp8
    original_decode = unpack.unpack_fp8_to_bf16
    calls = []

    def fake_fused(x, backend="auto", out=None, return_codes=False):
        # The CPU fixture uses the original composed oracle. It is not a claim
        # about a fused device kernel; that has separate root-owned NPU tests.
        if backend not in {"auto", "triton"}:
            raise ValueError("fixture only models the fused selection")
        codes, scales = original_quantize(x, backend="torch")
        decoded = original_decode(codes, scales, block_shape=(1, 32), backend="torch")
        calls.append(x.shape)
        return decoded, codes, scales

    monkeypatch.setattr(fused_fp8_activation_decode, "quantize_decode_fp8", fake_fused)
    monkeypatch.setattr(binding, "_on_npu", lambda tensor: isinstance(tensor, torch.Tensor))
    manager = binding.FusedActivationBinding().install()
    try:
        yield manager, original_quantize, original_decode, calls
    finally:
        manager.close()


def activation():
    return torch.randn(1,128,generator=torch.Generator().manual_seed(1651)).bfloat16()


def test_fused_code_buffers_and_reused_BF16_value_match_the_original_bits(forced_device_binding):
    manager, old_quantize, old_decode, calls = forced_device_binding
    x = activation()
    expected_codes, expected_scales = old_quantize(x, backend="torch")
    expected = old_decode(expected_codes, expected_scales, block_shape=(1,32), backend="torch")
    codes, scales = block_fp8_quant.quantize_block_fp8(x, backend="auto")
    gate = unpack.unpack_fp8_to_bf16(codes, scales, block_shape=(1,32), backend="auto")
    up = unpack.unpack_fp8_to_bf16(codes, scales, block_shape=(1,32), backend="auto")
    assert torch.equal(codes,expected_codes) and torch.equal(scales,expected_scales)
    assert torch.equal(gate.view(torch.int16),expected.view(torch.int16))
    assert gate is up
    assert len(calls)==1 and manager.decode_hits==2
    assert manager.describe()["global_registry_of_activations"] is False


@pytest.mark.parametrize("which",["codes", "scales"])
def test_mutated_byte_operands_do_not_return_the_old_decoded_value(which,forced_device_binding):
    manager, _, old_decode, _ = forced_device_binding
    codes, scales = block_fp8_quant.quantize_block_fp8(activation())
    if which == "codes":
        codes[0,0] = 0
    else:
        scales[0,0] += 1
    expected = old_decode(codes,scales,block_shape=(1,32),backend="torch")
    actual = unpack.unpack_fp8_to_bf16(codes,scales,block_shape=(1,32),backend="auto")
    assert torch.equal(actual.view(torch.int16),expected.view(torch.int16))
    assert manager.stale_misses==1 and manager.decode_hits==0


def test_a_cloned_grid_takes_the_original_decoder_not_an_unowned_memo(forced_device_binding):
    manager, _, old_decode, _ = forced_device_binding
    codes, scales = block_fp8_quant.quantize_block_fp8(activation())
    other = scales.clone()
    expected = old_decode(codes,other,block_shape=(1,32),backend="torch")
    actual = unpack.unpack_fp8_to_bf16(codes,other,block_shape=(1,32),backend="auto")
    assert torch.equal(actual,expected)
    assert manager.stale_misses==1


def test_requested_output_buffers_keep_the_original_quantize_contract(forced_device_binding):
    manager, old_quantize, _, calls = forced_device_binding
    x = activation()
    out_codes = torch.empty(x.shape,dtype=torch.uint8)
    out_scales = torch.empty((1,4),dtype=torch.uint8)
    expected_codes,expected_scales = old_quantize(x,backend="torch")
    got_codes,got_scales = block_fp8_quant.quantize_block_fp8(x,backend="torch",out=(out_codes,out_scales))
    assert got_codes is out_codes and got_scales is out_scales
    assert torch.equal(got_codes,expected_codes) and torch.equal(got_scales,expected_scales)
    assert not calls and manager.fused_calls==0


def test_reused_decode_honors_and_validates_out(forced_device_binding):
    _, _, _, _ = forced_device_binding
    codes,scales = block_fp8_quant.quantize_block_fp8(activation())
    expected = unpack.unpack_fp8_to_bf16(codes,scales,block_shape=(1,32))
    out = torch.empty_like(expected)
    actual = unpack.unpack_fp8_to_bf16(codes,scales,block_shape=(1,32),out=out)
    assert actual is out and torch.equal(actual,expected)
    with pytest.raises(TypeError):
        unpack.unpack_fp8_to_bf16(codes,scales,block_shape=(1,32),out=torch.empty(expected.shape,dtype=torch.float32))


def test_cpu_Engram_providers_are_not_sent_to_the_fused_device_operator(monkeypatch):
    original_quantize = block_fp8_quant.quantize_block_fp8
    original_decode = unpack.unpack_fp8_to_bf16
    def forbidden(*_, **__):
        raise AssertionError("CPU Engram entered the NPU fused kernel")
    monkeypatch.setattr(fused_fp8_activation_decode,"quantize_decode_fp8",forbidden)
    x = activation()
    expected_codes,expected_scales = original_quantize(x,backend="torch")
    with binding.FusedActivationBinding() as manager:
        codes,scales = block_fp8_quant.quantize_block_fp8(x,backend="torch")
        actual = unpack.unpack_fp8_to_bf16(codes,scales,block_shape=(1,32),backend="torch")
        expected = original_decode(expected_codes,expected_scales,block_shape=(1,32),backend="torch")
        assert torch.equal(actual.view(torch.int16),expected.view(torch.int16))
        assert manager.CPU_bypasses==1 and manager.fused_calls==0
    assert block_fp8_quant.quantize_block_fp8 is original_quantize
    assert unpack.unpack_fp8_to_bf16 is original_decode


def test_installation_has_one_owner_and_close_restores_originals():
    old_quantize,old_decode = block_fp8_quant.quantize_block_fp8,unpack.unpack_fp8_to_bf16
    first,second = binding.FusedActivationBinding(),binding.FusedActivationBinding()
    with first:
        with pytest.raises(RuntimeError,match="already installed"):
            second.install()
    first.close()
    assert block_fp8_quant.quantize_block_fp8 is old_quantize
    assert unpack.unpack_fp8_to_bf16 is old_decode


def test_fused_errors_propagate_and_context_cleanup_is_not_a_fallback(monkeypatch):
    old_quantize,old_decode = block_fp8_quant.quantize_block_fp8,unpack.unpack_fp8_to_bf16
    monkeypatch.setattr(binding,"_on_npu",lambda _:True)
    def fail(*_, **__):
        raise RuntimeError("named fused compiler failure")
    monkeypatch.setattr(fused_fp8_activation_decode,"quantize_decode_fp8",fail)
    with pytest.raises(RuntimeError,match="named fused compiler failure"):
        with binding.FusedActivationBinding():
            block_fp8_quant.quantize_block_fp8(activation())
    assert block_fp8_quant.quantize_block_fp8 is old_quantize
    assert unpack.unpack_fp8_to_bf16 is old_decode


def test_explicit_same_contract_operator_is_selected_and_reported(monkeypatch):
    old_quantize,old_decode = block_fp8_quant.quantize_block_fp8,unpack.unpack_fp8_to_bf16
    monkeypatch.setattr(binding,"_on_npu",lambda _:True)
    calls=[]
    def static_fused(x,*,backend,return_codes):
        codes,scales=old_quantize(x,backend="torch")
        decoded=old_decode(codes,scales,block_shape=(1,32),backend="torch")
        calls.append((backend,return_codes))
        return decoded,codes,scales
    with binding.FusedActivationBinding(operator=static_fused) as manager:
        codes,scales=block_fp8_quant.quantize_block_fp8(activation())
        decoded=unpack.unpack_fp8_to_bf16(codes,scales,block_shape=(1,32))
        assert torch.equal(decoded,old_decode(codes,scales,block_shape=(1,32),backend="torch"))
        assert manager.describe()["operator"]=="static_fused"
    assert calls==[("auto",True)]


def test_invalid_explicit_operator_is_refused_before_install():
    with pytest.raises(TypeError,match="must be callable"):
        binding.FusedActivationBinding(operator="invalid")


def test_remember_original_decode_reuses_external_fused_outputs(forced_device_binding):
    manager, quantize, decode, _ = forced_device_binding
    codes, scales = quantize(activation(), backend="torch")
    expected = decode(codes, scales, block_shape=(1, 32), backend="torch")
    binding.remember_original_decode(codes, scales, expected)
    actual = unpack.unpack_fp8_to_bf16(codes, scales, block_shape=(1, 32))
    assert actual is expected and manager.decode_hits == 1
    assert manager.fused_calls == 0
    scales[0, 0] += 1
    result = unpack.unpack_fp8_to_bf16(codes, scales, block_shape=(1, 32))
    assert result is not expected and manager.stale_misses == 1


def test_remember_decode_rejects_mismatched_metadata_without_installation():
    codes = torch.zeros((1, 64), dtype=torch.uint8)
    scales = torch.full((1, 2), 127, dtype=torch.uint8)
    value = torch.zeros((1, 64), dtype=torch.bfloat16)
    binding.remember_original_decode(codes, scales, value)
    assert binding._ACTIVE is None
    with pytest.raises(ValueError, match="matching contiguous"):
        binding.remember_original_decode(codes, scales[:, :1], value)
    with pytest.raises(ValueError, match="matching contiguous"):
        binding.remember_original_decode(codes, scales, value.float())


def test_inference_metadata_does_not_throw_for_absent_version():
    with torch.inference_mode():
        value=torch.zeros((1,32),dtype=torch.uint8)
        assert binding._token(value)[-1] is None
