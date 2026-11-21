"""Explicit no-ring AIV metadata, original-byte mapping and tail oracle contracts."""
from __future__ import annotations

import pytest
import torch

from relic_core.kernels.direct_aiv_gemv import DirectAIVConsumer
from relic_core.kernels.online_tile_cube import pack_expert_tile_bank
from relic_core.kernels.unpack import unpack_fp4_to_int8, fp4_int8_scale


@pytest.mark.parametrize("k", [64, 288, 576, 1152])
@pytest.mark.parametrize("shared", [True, False])
def test_original_k32_tail_and_device_selected_slot_mapping(k, shared):
    generator = torch.Generator().manual_seed(502 + k)
    codes = torch.randint(0, 256, (3, 32, k // 2), generator=generator, dtype=torch.uint8)
    scales = torch.full((3, 32, k // 32), 121, dtype=torch.uint8)
    banks = pack_expert_tile_bank(codes, scales)
    x = torch.randint(-4, 5, (1 if shared else 3, k), generator=generator).bfloat16()
    ids = torch.tensor([4, 0, -1], dtype=torch.int32)
    op = DirectAIVConsumer(backend="torch")
    actual = op(x, *banks, ids, world=2, certified_bf16_operands=True, output_dtype=torch.float32)
    reference = (unpack_fp4_to_int8(codes, backend="torch").float()
                 * fp4_int8_scale(scales, backend="torch").repeat_interleave(32, -1)).bfloat16()
    expected = torch.zeros((3, 32))
    for group, expert in enumerate((2, 0)):
        expected[group] = (x[0 if shared else group].float()[None] @ reference[expert].float().t())[0]
    assert torch.equal(actual, expected)
    assert op.describe()["expanded_weight_GM_write_read_bytes"] == 0
    assert op.describe()["original_K32_tail_supported"]


def operands():
    b, s = pack_expert_tile_bank(torch.zeros((2, 32, 32), dtype=torch.uint8), torch.full((2, 32, 2), 127, dtype=torch.uint8))
    return torch.ones(1, 64, dtype=torch.bfloat16), b, s, torch.tensor([0, -1], dtype=torch.int32)


def test_native_backend_never_silently_runs_the_cpu_oracle():
    with pytest.raises(ValueError, match="Triton Ascend"):
        DirectAIVConsumer()(*operands(), world=1, certified_bf16_operands=True)


def test_explicit_fp32_tp_partial_is_the_provided_accumulation_storage():
    result = torch.empty(2, 32)
    actual = DirectAIVConsumer(backend="torch")(*operands(), world=1, accumulation=result,
                                               output_dtype=torch.float32, certified_bf16_operands=True)
    assert actual is result and not actual.count_nonzero()


@pytest.mark.parametrize("bad", [-2, 2, 2**31 - 1])
def test_invalid_selected_source_is_poison_not_an_outside_read(bad):
    x, b, s, ids = operands(); ids.fill_(bad)
    assert torch.isnan(DirectAIVConsumer(backend="torch")(x, b, s, ids, world=1, certified_bf16_operands=True)).all()


def test_accumulation_cannot_alias_activation_or_change_projection_dtype():
    x, b, s, ids = operands()
    with pytest.raises(ValueError, match="metadata"):
        DirectAIVConsumer(backend="torch")(x, b, s, ids, world=1, accumulation=x, certified_bf16_operands=True)
    with pytest.raises(ValueError, match="FP32"):
        DirectAIVConsumer(backend="torch")(x, b, s, ids, world=1, output_dtype=torch.float16, certified_bf16_operands=True)


@pytest.mark.parametrize("scale", [0, 1, 253, 254, 255])
def test_invalid_scales_poison_original_zero_weights(scale):
    x, b, s, ids = operands(); s.fill_(scale)
    result = DirectAIVConsumer(backend="torch")(x, b, s, ids, world=1, certified_bf16_operands=True)
    assert torch.isnan(result[0]).all() and not result[1].count_nonzero()
