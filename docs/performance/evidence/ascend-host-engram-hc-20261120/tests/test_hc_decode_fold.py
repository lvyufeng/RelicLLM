"""CPU contracts for the explicit decode HC fold primitives."""
from __future__ import annotations

import pytest
import torch

from relic_core.kernels.hc_decode_fold import fused_hc_decode_pre, fused_hc_decode_post


def operands(dim=5120, seed=17):
    generator = torch.Generator().manual_seed(seed)
    residual = torch.randn(1, 1, 4, dim, generator=generator).bfloat16()
    x = torch.randn(1, 1, dim, generator=generator).bfloat16()
    pre = torch.randn(1, 1, 4, generator=generator)
    post = torch.randn(1, 1, 4, generator=generator)
    comb = torch.randn(1, 1, 4, 4, generator=generator)
    return x, residual, pre, post, comb


@pytest.mark.parametrize("dim", [1, 31, 256, 5120])
def test_pre_matches_original_fp32_then_bf16(dim):
    _, residual, pre, _, _ = operands(dim)
    expected = (pre.unsqueeze(-1) * residual.float()).sum(dim=2).bfloat16()
    actual = fused_hc_decode_pre(residual, pre, backend="torch")
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("dim", [1, 31, 256, 5120])
def test_post_matches_original_column_orientation(dim):
    x, residual, _, post, comb = operands(dim)
    expected = (post.unsqueeze(-1) * x.unsqueeze(-2)
                + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)).bfloat16()
    actual = fused_hc_decode_post(x, residual, post, comb, backend="torch")
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_persistent_out_and_source_immutability():
    x, residual, pre, post, comb = operands(64)
    original = [tensor.clone() for tensor in (x, residual, pre, post, comb)]
    pre_out = torch.empty_like(x)
    post_out = torch.empty_like(residual)
    assert fused_hc_decode_pre(residual, pre, out=pre_out, backend="torch") is pre_out
    assert fused_hc_decode_post(x, residual, post, comb, out=post_out, backend="torch") is post_out
    assert all(torch.equal(value, saved) for value, saved in zip((x, residual, pre, post, comb), original))


def test_nonfinite_inputs_are_not_silently_masked():
    x, residual, pre, post, comb = operands(64)
    residual[:, :, 2, 17] = float("nan")
    expected = (pre.unsqueeze(-1) * residual.float()).sum(dim=2).bfloat16()
    actual = fused_hc_decode_pre(residual, pre, backend="torch")
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    assert torch.isnan(fused_hc_decode_post(x, residual, post, comb, backend="torch")[:, :, :, 17]).all()


def test_pre_and_post_refuse_alias_output():
    x, residual, pre, post, comb = operands(64)
    with pytest.raises(ValueError, match="alias"):
        fused_hc_decode_pre(residual, pre, out=residual[:, :, 0, :], backend="torch")
    with pytest.raises(ValueError, match="alias"):
        fused_hc_decode_post(x, residual, post, comb, out=residual, backend="torch")


def test_refuse_prefill_and_coefficient_dtype_conversion():
    x, residual, pre, post, comb = operands(64)
    with pytest.raises(ValueError):
        fused_hc_decode_pre(residual.expand(1, 2, 4, 64).contiguous(), pre.expand(1, 2, 4).contiguous(), backend="torch")
    with pytest.raises(ValueError):
        fused_hc_decode_post(x, residual, post.bfloat16(), comb, backend="torch")


def test_cpu_cannot_claim_triton_hardware_execution():
    _, residual, pre, _, _ = operands(64)
    with pytest.raises(ValueError):
        fused_hc_decode_pre(residual, pre, backend="triton")
