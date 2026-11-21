"""Direct bit decoder admission and an independent exhaustive code/scale oracle.

CPU bits check the proposed modular exponent construction only. Actual native
vector arithmetic, ring ownership and layout require the separate NPU corpus.
"""
from __future__ import annotations

import pytest
import torch

from relic_core.kernels.unpack import unpack_fp4_to_int8, fp4_int8_scale


def proposed_bits(code, scale):
    low, high = code & 15, code >> 4
    result = []
    for nibble in (low, high):
        magnitude = nibble & 7
        delta = torch.where(magnitude >= 6, 2, torch.where(magnitude >= 4, 1, torch.where(magnitude >= 2, 0, -1)))
        mantissa = torch.where((magnitude == 3) | (magnitude == 5) | (magnitude == 7), 64, 0)
        unit = ((nibble & 8) << 12) | ((127 + delta) << 7) | mantissa
        unit = torch.where(magnitude == 0, 0, unit)
        offset = ((scale - 127) << 7) & 65535
        bits = (unit + offset) & 65535
        bits = torch.where(magnitude == 0, 0, bits)
        bits = torch.where((scale < 2) | (scale > 252), bits | 0x7fc0, bits)
        result.append(bits)
    return torch.stack(result, -1).flatten(-2).to(torch.int16)


def test_every_byte_and_certified_scale_retains_exact_bf16_bits():
    codes = torch.arange(256).to(torch.uint8).reshape(16, 16).repeat(251, 1)
    scales = torch.arange(2, 253).to(torch.uint8).repeat_interleave(16).reshape(-1, 1)
    expected = (unpack_fp4_to_int8(codes, backend="torch").float()
                * fp4_int8_scale(scales, backend="torch").repeat_interleave(32, -1)).bfloat16()
    actual = proposed_bits(codes.to(torch.int32), scales.to(torch.int32))
    assert torch.equal(actual, expected.view(torch.int16))
    assert actual.numel() == 128512


@pytest.mark.parametrize("scale", [0, 1, 253, 254, 255])
def test_uncertified_scales_poison_even_original_zero_codes(scale):
    codes = torch.tensor([[0, 0x88, 0x17, 0xef]], dtype=torch.int32)
    result = proposed_bits(codes, torch.full((1, 1), scale, dtype=torch.int32)).view(torch.bfloat16)
    assert torch.isnan(result).all()


def test_original_positive_and_negative_zero_are_both_folded_to_positive_zero():
    codes = torch.tensor([[0, 0x88, 8, 0x80]], dtype=torch.int32)
    for scale in (2, 126, 127, 128, 252):
        assert not proposed_bits(codes, torch.tensor([[scale]])).count_nonzero()
