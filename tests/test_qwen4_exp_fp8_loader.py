"""FP8 checkpoint naming, packing, and host-reader tests."""

from __future__ import annotations

import os

import pytest
import torch

from relicllm.models.qwen4_exp.config import FP8QuantSpec, Qwen4ExpConfig
from relicllm.models.qwen4_exp.quant import FP8Tensor, pack_gate_up_fp8
from relicllm.models.qwen4_exp.weights import (
    HostExpertShard,
    HostNGramTable,
    MmapSafetensors,
    Qwen4ExpCheckpoint,
)

FP8_MODEL = os.environ.get(
    "QWEN4EXP_FP8_MODEL", "/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next-FP8"
)
REAL_FP8 = pytest.mark.skipif(
    not os.path.exists(os.path.join(FP8_MODEL, "model.safetensors.index.json")),
    reason=f"real FP8 checkpoint not found at {FP8_MODEL}",
)


def test_fp8_quant_spec_reads_root_metadata() -> None:
    spec = FP8QuantSpec.from_dict(
        {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head"],
        }
    )
    assert spec is not None
    assert spec.block_size == (128, 128)
    assert spec.activation_scheme == "dynamic"
    assert spec.skips("lm_head.weight")
    assert not spec.skips("model.language_model.layers.0.mlp.experts.0.gate_proj.weight")
    assert FP8QuantSpec.from_dict({"quant_method": "gptq"}) is None


def test_qwen_config_keeps_root_quantization_config() -> None:
    config = Qwen4ExpConfig.from_dict(
        {
            "architectures": ["Qwen4ExpForCausalLM"],
            "text_config": {"num_hidden_layers": 1},
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
        }
    )
    assert config.is_fp8
    assert config.weight_block_size == (128, 128)
    assert config.quantization["quant_method"] == "fp8"


def test_pack_gate_up_keeps_rows_and_scale_tiles_aligned() -> None:
    """A layer's two weights pack into the one fused `[2*inter, hidden]` matrix."""
    gate = FP8Tensor(
        torch.arange(4 * 4, dtype=torch.uint8).reshape(4, 4).view(torch.float8_e4m3fn),
        torch.tensor([[2.0], [3.0]], dtype=torch.bfloat16),
        (2, 4),
    )
    up = FP8Tensor(
        (100 + torch.arange(4 * 4, dtype=torch.uint8)).reshape(4, 4).view(torch.float8_e4m3fn),
        torch.tensor([[5.0], [7.0]], dtype=torch.bfloat16),
        (2, 4),
    )
    packed = pack_gate_up_fp8(gate, up)
    assert packed.code.shape == (8, 4)
    assert packed.scale.shape == (4, 1)
    torch.testing.assert_close(packed.code[:4].view(torch.uint8), gate.code.view(torch.uint8))
    torch.testing.assert_close(packed.code[4:].view(torch.uint8), up.code.view(torch.uint8))
    torch.testing.assert_close(packed.scale[:2], gate.scale)
    torch.testing.assert_close(packed.scale[2:], up.scale)


def test_pack_gate_up_stacks_stacked_expert_grids() -> None:
    """One expert per leading axis: `[E, 640, in]` twice, packed along the matrix's row axis.

    The per-matrix grids keep one tile row per 128 rows, so the packed ten tile rows are gate's
    five then `up`'s five.  The checkpoint instead writes one grid over the fused 1280 rows; that
    one is split at the same seam, and both layouts must dequantize to the same matrix.
    """
    from relicllm.models.qwen4_exp.quant import dequantize_block_fp8

    experts, inter, hidden, block = 2, 8, 4, (2, 4)
    gate = FP8Tensor(
        torch.arange(experts * inter * hidden, dtype=torch.uint8)
        .reshape(experts, inter, hidden)
        .view(torch.float8_e4m3fn),
        torch.full((experts, inter // block[0], 1), 2.0, dtype=torch.bfloat16),
        block,
    )
    up = FP8Tensor(
        (100 + torch.arange(experts * inter * hidden, dtype=torch.uint8))
        .view(torch.float8_e4m3fn)
        .reshape(experts, inter, hidden),
        torch.full((experts, inter // block[0], 1), 3.0, dtype=torch.bfloat16),
        block,
    )
    packed = pack_gate_up_fp8(gate, up)
    tiles = 2 * inter // block[0]
    assert packed.code.shape == (experts, 2 * inter, hidden)
    assert packed.scale.shape == (experts, tiles, 1)
    torch.testing.assert_close(
        packed.scale, torch.cat([gate.scale, up.scale], dim=1)
    )

    fused = torch.cat([gate.scale, up.scale], dim=1)
    from_fused = pack_gate_up_fp8(
        FP8Tensor(gate.code, fused, block),
        FP8Tensor(up.code, fused, block),
    )
    # Compared as codes: NaN is a legal e4m3 code (126 and 127), which `assert_close` calls unequal
    # to itself.
    torch.testing.assert_close(
        from_fused.code.view(torch.uint8), packed.code.view(torch.uint8), rtol=0, atol=0
    )
    torch.testing.assert_close(from_fused.scale, packed.scale)
    want = dequantize_block_fp8(packed.code, fused, block, torch.float32)
    got = dequantize_block_fp8(packed.code, packed.scale, block, torch.float32)
    assert torch.equal(got.isnan(), want.isnan())
    torch.testing.assert_close(got.nan_to_num(0.0), want.nan_to_num(0.0))


@REAL_FP8
def test_real_fp8_expert_names_and_shapes() -> None:
    store = MmapSafetensors(FP8_MODEL)
    checkpoint = Qwen4ExpCheckpoint(FP8_MODEL, store=store)
    assert checkpoint.is_fp8
    assert checkpoint.fp8_expert_keys(0, 0) == [
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
        "model.language_model.layers.0.mlp.experts.0.up_proj.weight",
        "model.language_model.layers.0.mlp.experts.0.up_proj.weight_scale_inv",
        "model.language_model.layers.0.mlp.experts.0.down_proj.weight",
        "model.language_model.layers.0.mlp.experts.0.down_proj.weight_scale_inv",
    ]
    gate_up, down = checkpoint.expert_rows(0, 0)
    assert isinstance(gate_up, FP8Tensor)
    assert isinstance(down, FP8Tensor)
    assert gate_up.code.shape == (1280, 2560)
    assert gate_up.scale.shape == (10, 20)
    assert down.code.shape == (2560, 640)
    assert down.scale.shape == (20, 5)
    assert gate_up.code.dtype is torch.float8_e4m3fn
    assert down.code.dtype is torch.float8_e4m3fn


@REAL_FP8
def test_real_fp8_ple_scale_and_gather() -> None:
    store = MmapSafetensors(FP8_MODEL)
    checkpoint = Qwen4ExpCheckpoint(FP8_MODEL, store=store)
    keys = checkpoint.ngram_shard_keys(1)
    assert len(keys) == 128
    assert store.view(keys[0]).dtype is torch.float8_e4m3fn
    scale = checkpoint.ngram_scale(1)
    assert scale is not None
    table = HostNGramTable(
        [store.view(keys[0])], device=torch.device("cpu"), dtype=torch.float16, scale=scale
    )
    got = table(torch.tensor([[0, 2]], dtype=torch.long))
    assert got.shape == (1, 2, 160)
    assert got.dtype is torch.float16
    assert torch.isfinite(got).all()


def test_fp8_shard_allocates_codes_and_scales_as_one_group(monkeypatch) -> None:
    shard = HostExpertShard(num_layers=1, rank=0, world_size=1, pin_memory=False)
    codes_gu = torch.zeros(2, 4, 4, dtype=torch.float8_e4m3fn)
    codes_dn = torch.zeros(2, 4, 2, dtype=torch.float8_e4m3fn)
    scales_gu = torch.ones(2, 1, 1, dtype=torch.bfloat16)
    scales_dn = torch.ones(2, 1, 1, dtype=torch.bfloat16)

    def read(expert_id: int):
        return (
            FP8Tensor(codes_gu[expert_id], scales_gu[expert_id], (2, 2)),
            FP8Tensor(codes_dn[expert_id], scales_dn[expert_id], (2, 2)),
        )

    shard.load_layer_fp8(0, 2, read, block_size=(2, 2))
    gate_up, down = shard.rows(0, 1)
    assert isinstance(gate_up, FP8Tensor)
    assert isinstance(down, FP8Tensor)
    assert shard.is_fp8
    expected = sum(t.numel() * t.element_size() for t in (codes_gu, codes_dn, scales_gu, scales_dn))
    assert shard.resident_bytes == expected


def test_resident_shard_pin_fallback_converts_all_fp8_buffers(monkeypatch) -> None:
    shard = HostExpertShard(num_layers=2, rank=0, world_size=1, pin_memory=True)
    codes_gu = torch.zeros(1, 4, 4, dtype=torch.float8_e4m3fn)
    codes_dn = torch.zeros(1, 4, 2, dtype=torch.float8_e4m3fn)
    scales_gu = torch.ones(1, 1, 1, dtype=torch.bfloat16)
    scales_dn = torch.ones(1, 1, 1, dtype=torch.bfloat16)

    def read(_expert_id: int):
        return (
            FP8Tensor(codes_gu[0], scales_gu[0], (2, 2)),
            FP8Tensor(codes_dn[0], scales_dn[0], (2, 2)),
        )

    real_empty = torch.empty
    calls = [0]

    def failing_empty(*args, **kwargs):
        if kwargs.get("pin_memory"):
            calls[0] += 1
            if calls[0] == 5:
                raise RuntimeError("simulated memlock limit")
        return real_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", failing_empty)
    shard.load_layer_fp8(0, 1, read, block_size=(2, 2))
    shard.load_layer_fp8(1, 1, read, block_size=(2, 2))
    assert not shard.pinned
    for layer in (0, 1):
        gate_up, down = shard.rows(layer, 0)
        assert not gate_up.code.is_pinned()
        assert not gate_up.scale.is_pinned()
        assert not down.code.is_pinned()
        assert not down.scale.is_pinned()
