"""CPU contracts for the opt-in original-operand Engram core helpers."""
from __future__ import annotations

import pytest
import torch

from relic_core.kernels.fast_fused_fp8_activation_decode import quantize_decode_fp8_fast
from relic_core.kernels.original_fp8_engram import (
    copy_fp32_signbit, original_engram_gate, original_fp8_engram_projection,
)


def tensors(seed=17, hc=4, dim=128):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, 1, hc, dim, generator=generator).bfloat16()
    kv = torch.randn(1, (hc + 1) * dim, generator=generator).bfloat16()
    q = torch.randn(hc, dim, generator=generator).bfloat16()
    k = torch.randn(hc, dim, generator=generator).bfloat16()
    return x, kv, q.float() * k.float()


def gate_oracle(x, kv, weight, mask, eps=1e-6):
    hc, dim = x.shape[-2:]
    key, value = kv.reshape(1, 1, -1).split([hc * dim, dim], -1)
    key = key.float().reshape(1, 1, hc, dim)
    hidden = x.float()
    rstd = torch.rsqrt(hidden.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (hidden * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
    return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).bfloat16()


@pytest.mark.parametrize("seed", [0, 1, 5, 17])
def test_gate_retains_original_sequence(seed):
    x, kv, weight = tensors(seed)
    mask = torch.ones(1, 1, dtype=torch.bool)
    actual = original_engram_gate(x, kv, weight, mask, eps=1e-6)
    assert torch.equal(actual.view(torch.int16), gate_oracle(x, kv, weight, mask).view(torch.int16))


def test_masked_finite_token_is_unchanged():
    x, kv, weight = tensors()
    actual = original_engram_gate(x, kv, weight, torch.zeros(1, 1, dtype=torch.bool), eps=1e-6)
    assert torch.equal(actual.view(torch.int16), x.view(torch.int16))


def test_zero_inputs_and_nonfinite_poison_are_not_hidden():
    x, kv, weight = tensors()
    x.zero_(); kv.zero_()
    mask = torch.ones(1, 1, dtype=torch.bool)
    assert not original_engram_gate(x, kv, weight, mask, eps=1e-6).count_nonzero()
    kv[:, -1] = float("nan")
    actual = original_engram_gate(x, kv, weight, torch.zeros_like(mask), eps=1e-6)
    expected = gate_oracle(x, kv, weight, torch.zeros_like(mask))
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))


def test_projection_uses_original_fp8_activation_not_int8():
    generator = torch.Generator().manual_seed(29)
    lookup = torch.randn(1, 64, generator=generator).bfloat16()
    codes = torch.randn(64, 96, generator=generator).to(torch.float8_e4m3fn)
    weight = codes.float().bfloat16()
    actual = original_fp8_engram_projection(lookup, weight, certified_original_bf16_weight=True, backend="torch")
    original = quantize_decode_fp8_fast(lookup, backend="torch")
    expected = original @ weight
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_nonquantized_projection_does_not_quantize():
    x = torch.linspace(-0.91, 1.07, 32).reshape(1, -1).bfloat16()
    weight = torch.eye(32).bfloat16()
    actual = original_fp8_engram_projection(x, weight, quantized=False, backend="torch")
    assert torch.equal(actual, x)


@pytest.mark.parametrize("changes", [{"certified_original_bf16_weight": False}, {"backend": "triton"}])
def test_projection_refuses_uncertified_or_wrong_device(changes):
    x = torch.ones(1, 32, dtype=torch.bfloat16)
    weight = torch.ones(32, 64, dtype=torch.bfloat16)
    options = {"backend": "torch", "certified_original_bf16_weight": True, **changes}
    with pytest.raises(ValueError):
        original_fp8_engram_projection(x, weight, **options)


@pytest.mark.parametrize("eps", [0.0, -1.0, float("nan"), float("inf")])
def test_gate_refuses_bad_epsilon(eps):
    x, kv, weight = tensors()
    with pytest.raises(ValueError):
        original_engram_gate(x, kv, weight, torch.ones(1, 1, dtype=torch.bool), eps=eps)


def test_copysign_bits_preserve_signed_zero_inf_and_nan_payload():
    raw = torch.tensor([0, -2147483648, 0x3F800000, -1082130432,
                        0x7F800000, -8388608, 0x7FC12345, -4127931], dtype=torch.int32)
    magnitude = raw.view(torch.float32)
    for signs in (raw.flip(0).contiguous(), raw.roll(3)):
        actual = copy_fp32_signbit(magnitude, signs.view(torch.float32))
        expected = torch.copysign(magnitude, signs.view(torch.float32))
        assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_copysign_bits_refuse_implicit_dtype_conversion():
    with pytest.raises(ValueError):
        copy_fp32_signbit(torch.ones(4, dtype=torch.bfloat16), torch.ones(4))


def test_gate_refuses_incorrect_complete_kv_extent():
    x, kv, weight = tensors()
    with pytest.raises(ValueError):
        original_engram_gate(x, kv[:, :-1].contiguous(), weight, torch.ones(1, 1, dtype=torch.bool), eps=1e-6)
