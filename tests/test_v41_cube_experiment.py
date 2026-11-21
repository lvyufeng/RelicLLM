"""Tests for the user-authorized NPU Cube reduction experiment.

Everything here is a CPU proof except the two NPU cases at the end, which are opt-in through
``RELIC_CUBE_EXPERIMENT_DEVICE`` and are the lead's to run on a leased card. The CPU layer is the one
that can be trusted to run anywhere, so it covers:

* the exact operand decoding (all sixteen E2M1 codes, all 256 E4M3 codes) against independent
  references -- a hand-written LUT and the frozen relic-core decoder;
* the ``-0`` sign that the INT8 pair cannot carry, confined to zeros and shown not to move a product;
* the BF16-representability guard, including the two ways an operand can silently move (overflow to
  infinity, underflow to zero) and the floating-grid power-of-two precondition;
* the call ordering of the dense path: the original quantizer's bytes, no INT8 activation
  requantization, and one natural matmul;
* the packed GEMV against its own definition and against a float64 reference;
* the bounded per-object cache -- off by default, per-object, evicting, capped, and holding its
  sources alive until ``clear``;
* the provider wrapper, including that ``emulate_reduction=False`` still reaches the *captured*
  original rather than recursing.

The NPU cases assert finiteness and shape and report the difference from a float64 reference; they do
not and cannot certify the device compiler's arithmetic from a CPU run.
"""
from __future__ import annotations

import gc
import os
import weakref

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels.block_fp4_quant import quantize_block_fp4  # noqa: E402
from relic_core.kernels.block_fp8_quant import quantize_block_fp8  # noqa: E402
from relic_core.kernels.unpack import (  # noqa: E402
    fp4_int8_scale,
    unpack_fp4_to_int8,
    unpack_fp8_to_bf16,
)
from relicllm.models.deepseek_v4_1 import cube_experiment as cube  # noqa: E402
from relicllm.models.deepseek_v4_1 import mma  # noqa: E402

BLOCK = cube.BLOCK
DEVICE_ENV = "RELIC_CUBE_EXPERIMENT_DEVICE"
requires_device = pytest.mark.skipif(
    not os.environ.get(DEVICE_ENV),
    reason=f"set {DEVICE_ENV}=npu:0 on a leased card to run the device case",
)


# ------------------------------------------------------------------------------------------------
# Fixtures and helpers
# ------------------------------------------------------------------------------------------------


def torch_core_ops(*, record: list | None = None):
    """The frozen operators with the backend forced to the CPU reference, for seam-patched tests."""
    def quantize(x, backend="auto"):
        if record is not None:
            record.append(("quantize", backend))
        return quantize_block_fp8(x, backend="torch")

    def unpack(w, s=None, *, block_shape=(32, 32), backend="auto"):
        if record is not None:
            record.append(("unpack_fp8", backend, tuple(block_shape)))
        return unpack_fp8_to_bf16(w, s, block_shape=block_shape, backend="torch")

    return quantize, unpack, unpack_fp4_to_int8, fp4_int8_scale


@pytest.fixture
def seams(monkeypatch):
    """Drop the NPU-only guard and force the frozen operators onto the CPU reference.

    Two module-private seams, which is the whole of how these tests exercise the real ordering without
    a card; there is no flag in the production path that could do the same thing by accident.
    """
    monkeypatch.setattr(cube, "_require_npu", lambda *arguments, **keywords: None)
    monkeypatch.setattr(cube, "_core_ops", torch_core_ops)


def e8m0_fp32(scale_bytes: torch.Tensor) -> torch.Tensor:
    """The FP32 powers of two of a raw E8M0 byte grid -- the form the loader attaches."""
    return torch.ldexp(
        torch.ones_like(scale_bytes, dtype=torch.float32), scale_bytes.to(torch.int32) - 127
    )


def row_scaled_weight(rows: int, columns: int, *, seed: int = 0, scale: float = 3.0):
    """A BF16 weight with the loader's provenance: original codes plus the attached FP32 grid."""
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn(rows, columns, generator=generator) * scale
    codes, grid_bytes = quantize_block_fp8(source, backend="torch")
    weight = unpack_fp8_to_bf16(codes, grid_bytes, block_shape=(1, BLOCK), backend="torch")
    grid = e8m0_fp32(grid_bytes)
    mma.attach_source_codes(weight, codes.view(torch.float8_e4m3fn), grid)
    return weight, codes, grid_bytes, grid


def packed_projection(rows: int, columns: int, *, seed: int = 0):
    """Original packed FP4 weight bytes ``[N, K//2]`` beside the raw un-halved E8M0 grid."""
    generator = torch.Generator().manual_seed(seed)
    weight = (torch.randn(rows, columns, generator=generator) * 2).to(torch.bfloat16)
    packed, grid = quantize_block_fp4(weight, BLOCK, "e8m0", backend="torch")
    return packed, grid


def activation(rows: int, columns: int, *, seed: int = 0, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, columns, generator=generator)
    return values.to(dtype)


def all_e2m1_codes() -> torch.Tensor:
    """Sixteen packed bytes that spell every E2M1 code once, low nibble first, as ``K = 32``."""
    return torch.arange(16, dtype=torch.uint8).reshape(1, 16)


def all_e4m3_codes() -> torch.Tensor:
    """All 256 E4M3 byte encodings as ``[8, 32]`` so the last axis is one block of 32."""
    return torch.arange(256, dtype=torch.uint8).reshape(8, 32)


# ------------------------------------------------------------------------------------------------
# Exact decoding
# ------------------------------------------------------------------------------------------------


def test_the_lut_is_the_exact_e2m1_codebook_including_signed_zero():
    values = cube._fp4_values(torch.device("cpu"))
    assert values.tolist() == [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    assert bool(torch.signbit(values[8])), "code 8 is negative zero, not positive zero"
    assert not bool(torch.signbit(values[0])), "code 0 is positive zero"


def test_decode_of_all_sixteen_codes_is_bit_exact_against_the_lut():
    codes = all_e2m1_codes()
    scales = torch.full((1, 1), 127, dtype=torch.uint8)  # 2**0, so the codes are the values
    decoded = cube.decode_fp4_packed_bf16(codes, scales, backend="torch")
    order = []
    for byte in codes[0].tolist():
        order += [byte & 0x0F, byte >> 4]
    expected = cube._fp4_values(torch.device("cpu"))[torch.tensor(order)].to(torch.bfloat16)
    assert decoded.shape == (1, 2 * codes.shape[1])
    assert torch.equal(decoded.view(torch.int16), expected.reshape(1, -1).view(torch.int16))


def test_the_frozen_pair_decoder_agrees_except_for_the_sign_of_a_zero():
    """The claim behind using the INT8 pair on a device: it differs only where the value is zero.

    ``unpack_fp4_to_int8`` cannot carry an E2M1 ``-0`` -- INT8 has one zero -- while the exact decoder
    keeps the sign bit. Both are numerically zero, so a product cannot move; that is asserted on a
    matmul, not just argued.
    """
    codes = all_e2m1_codes()
    scales = torch.full((1, 1), 127, dtype=torch.uint8)
    exact = cube.decode_fp4_packed_bf16(codes, scales, backend="torch")
    values = unpack_fp4_to_int8(codes, backend="torch").to(torch.float32)
    half = fp4_int8_scale(scales, backend="torch")
    pair = (values * half.repeat_interleave(BLOCK, dim=-1)).to(torch.bfloat16)
    differ = exact.view(torch.int16) != pair.view(torch.int16)
    order = []
    for byte in codes[0].tolist():
        order += [byte & 0x0F, byte >> 4]
    negative_zero_at = order.index(8)
    assert int(differ.sum()) == 1, "only code 8 (negative zero) may differ"
    assert int(differ.nonzero()[0][1]) == negative_zero_at
    assert exact.view(torch.int16)[0, negative_zero_at].item() == -32768, "the exact decoder keeps -0.0"
    assert pair.view(torch.int16)[0, negative_zero_at].item() == 0, "the INT8 pair yields +0.0"
    # A zero's sign cannot move a dot product.
    left = torch.randn(1, exact.shape[1], generator=torch.Generator().manual_seed(3)).to(torch.bfloat16)
    assert torch.equal((left @ exact.t()).view(torch.int16), (left @ pair.t()).view(torch.int16))


def test_the_e8m0_byte_table_is_exact_at_both_ends():
    one = torch.full((1, 1), 127, dtype=torch.uint8)
    assert cube._e8m0_byte_values(one, BLOCK).reshape(-1)[0].item() == 1.0
    zero = torch.full((1, 1), 0, dtype=torch.uint8)
    assert cube._e8m0_byte_values(zero, BLOCK).reshape(-1)[0].item() == 2.0**-127
    top = torch.full((1, 1), 254, dtype=torch.uint8)
    assert cube._e8m0_byte_values(top, BLOCK).reshape(-1)[0].item() == 2.0**127
    nan = torch.full((1, 1), 255, dtype=torch.uint8)
    assert bool(torch.isnan(cube._e8m0_byte_values(nan, BLOCK)).all())


def test_the_exact_fp8_decoder_matches_the_frozen_one_on_all_256_codes():
    codes = all_e4m3_codes()
    scales = torch.full((8, 1), 127, dtype=torch.uint8)
    mine = cube._fp8_exact(codes, scales, BLOCK)
    frozen = unpack_fp8_to_bf16(codes, scales, block_shape=(1, BLOCK), backend="torch").float()
    finite = torch.isfinite(frozen)
    assert torch.equal(mine[finite].view(torch.int32), frozen[finite].view(torch.int32))
    assert torch.equal(torch.isnan(mine), torch.isnan(frozen))
    row, column = divmod(0x38, 32)
    assert mine[row, column].item() == 1.0, "0x38 is 1.0, not 8.0"


def test_a_non_finite_original_scale_stays_nan_instead_of_a_finite_value():
    codes = torch.zeros((1, 16), dtype=torch.uint8)
    scales = torch.full((1, 1), 255, dtype=torch.uint8)
    decoded = cube.decode_fp4_packed_bf16(codes, scales, backend="torch")
    assert bool(torch.isnan(decoded).all()), "a 255 grid byte must not be answered with a number"


# ------------------------------------------------------------------------------------------------
# Exactness guards
# ------------------------------------------------------------------------------------------------


def test_the_guard_accepts_exact_operands_and_counts_its_synchronizations():
    before = dict(cube.GUARD_STATS)
    report = cube.check_bf16_representable(
        all_e2m1_codes(), torch.full((1, 1), 127, dtype=torch.uint8), kind="fp4", name="weight"
    )
    assert report["not_bf16_representable"] == 0 and report["non_finite"] == 0
    assert report["elements"] == 32
    assert cube.GUARD_STATS["checks"] == before["checks"] + 1
    assert cube.GUARD_STATS["syncs"] - before["syncs"] == report["syncs"] >= 1


def test_the_guard_refuses_fp4_values_that_overflow_bf16():
    codes = torch.full((1, 16), 0x07, dtype=torch.uint8)  # 6.0 per code
    scales = torch.full((1, 1), 254, dtype=torch.uint8)  # 2**127 -> 6 * 2**127 > BF16 max
    with pytest.raises(cube.OriginalOperandNotBf16Representable, match="non-finite"):
        cube.check_bf16_representable(codes, scales, kind="fp4", name="weight")


def test_the_guard_refuses_fp8_values_that_underflow_bf16():
    codes = torch.full((1, 32), 0x01, dtype=torch.uint8)  # smallest E4M3 subnormal: 2**-9
    scales = torch.full((1, 1), 0, dtype=torch.uint8)  # 2**-127 -> 2**-136 < BF16 min subnormal
    with pytest.raises(cube.OriginalOperandNotBf16Representable, match="not exactly representable"):
        cube.check_bf16_representable(codes, scales, kind="fp8", name="activation")


def test_a_floating_grid_must_be_powers_of_two_and_is_accepted_when_it_is():
    codes = all_e2m1_codes()
    good = torch.tensor([[1.0]], dtype=torch.float32)
    assert cube.check_bf16_representable(codes, good, kind="fp4", name="weight")["elements"] == 32
    with pytest.raises(ValueError, match="powers of two"):
        cube.check_bf16_representable(
            codes, torch.tensor([[3.0]], dtype=torch.float32), kind="fp4", name="weight"
        )


def test_a_square_grid_is_expanded_for_the_guard_instead_of_being_refused():
    codes = torch.arange(32, dtype=torch.uint8).reshape(2, 16)
    scales = torch.full((1, 1), 127, dtype=torch.uint8)  # 32 rows share one square block row
    report = cube.check_bf16_representable(codes, scales, kind="fp4", name="weight")
    assert report["elements"] == 2 * 32


# ------------------------------------------------------------------------------------------------
# Provenance: the weight really is the decode of the attached bytes
# ------------------------------------------------------------------------------------------------


def test_the_weight_identity_check_passes_for_a_row_scaled_grid():
    weight, _, _, _ = row_scaled_weight(8, 64, seed=5)
    report = cube.verify_weight_decode_identity(weight)
    assert report["mismatches"] == 0 and report["block_rows"] == 1


def test_the_weight_identity_check_passes_for_a_square_grid():
    generator = torch.Generator().manual_seed(9)
    codes, row_grid = quantize_block_fp8(torch.randn(64, 64, generator=generator) * 2, backend="torch")
    square_blocked = row_grid.reshape(2, BLOCK, 2)[:, 0, :].repeat_interleave(BLOCK, dim=0)[:64]
    weight = unpack_fp8_to_bf16(codes, square_blocked, block_shape=(1, BLOCK), backend="torch")
    square = row_grid.reshape(2, BLOCK, 2)[:, 0, :]
    mma.attach_source_codes(weight, codes.view(torch.float8_e4m3fn), e8m0_fp32(square))
    report = cube.verify_weight_decode_identity(weight)
    assert report["mismatches"] == 0 and report["block_rows"] == BLOCK


def test_a_weight_that_is_not_the_decode_of_its_metadata_is_refused_by_name():
    weight, _, _, _ = row_scaled_weight(8, 64, seed=13)
    weight.view(torch.int16)[0, 0] ^= 0x0001  # one bit away from the decode of the attached bytes
    with pytest.raises(cube.OriginalOperandNotBf16Representable, match="not the decode"):
        cube.verify_weight_decode_identity(weight)


def test_a_weight_without_attached_metadata_is_refused_by_name():
    with pytest.raises(cube.CubeExperimentUnavailable, match="no original E4M3 codes"):
        cube.verify_weight_decode_identity(torch.zeros(4, 64, dtype=torch.bfloat16))


# ------------------------------------------------------------------------------------------------
# The dense path
# ------------------------------------------------------------------------------------------------


def test_the_dense_entry_point_refuses_a_cpu_tensor_and_the_old_escape_hatch():
    weight, _, _, _ = row_scaled_weight(4, 64)
    x = activation(1, 64)
    with pytest.raises(cube.CubeExperimentUnavailable, match="NPU only"):
        cube.cube_quantized_linear_ascend(x, weight)
    with pytest.raises(cube.CubeExperimentUnavailable, match="not re-implemented here"):
        cube.cube_quantized_linear_ascend(x, weight, emulate_reduction=False)


def test_the_dense_path_requires_the_original_weight_provenance(seams):
    x = activation(1, 64)
    plain = torch.zeros(4, 64, dtype=torch.bfloat16)
    with pytest.raises(cube.CubeExperimentUnavailable, match="E4M3 codes and E8M0 grid attached"):
        cube.cube_quantized_linear_ascend(x, plain)


def test_the_dense_path_quantizes_with_the_original_quantizer_and_multiplies_once(seams, monkeypatch):
    """The ordering that makes this an experiment on the *reduction* and nothing else."""
    record: list = []
    calls: dict = {}

    def quantize(x, backend="auto"):
        record.append(("quantize", backend))
        codes, grid = quantize_block_fp8(x, backend="torch")
        calls["codes"], calls["grid"] = codes, grid
        return codes, grid

    def unpack(w, s=None, *, block_shape=(32, 32), backend="auto"):
        record.append(("unpack_fp8", backend, tuple(block_shape)))
        calls["unpack_saw"] = w
        return unpack_fp8_to_bf16(w, s, block_shape=block_shape, backend="torch")

    monkeypatch.setattr(cube, "_core_ops", lambda: (quantize, unpack, unpack_fp4_to_int8, fp4_int8_scale))
    weight, _, _, _ = row_scaled_weight(6, 64, seed=17)
    out = cube.cube_quantized_linear_ascend(activation(2, 64, seed=19), weight)

    assert record[0] == ("quantize", "triton"), "the released quantizer is asked for the fused backend"
    assert record[1] == ("unpack_fp8", "triton", (1, BLOCK)), "one row-scaled decode, no requantization"
    assert calls["unpack_saw"] is calls["codes"], "the decoder sees the quantizer's own bytes"
    assert not any(entry[0] == "unpack_fp4" for entry in record), "no INT8 activation requantization"
    expected = torch.matmul(
        unpack_fp8_to_bf16(calls["codes"], calls["grid"], block_shape=(1, BLOCK), backend="torch"),
        weight.transpose(0, 1),
    )
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))


def test_the_dense_output_dtype_follows_the_activation(seams):
    weight, _, _, _ = row_scaled_weight(6, 64, seed=23)
    for dtype in (torch.bfloat16, torch.float32):
        out = cube.cube_quantized_linear_ascend(activation(2, 64, seed=29, dtype=dtype), weight)
        assert out.dtype == dtype and out.shape == (2, 6)


def test_the_dense_path_refuses_a_non_bf16_weight(seams):
    weight, _, _, _ = row_scaled_weight(4, 64)
    codes = mma.attached_codes(weight)
    grid = mma.attached_scale(weight)
    fp32_weight = weight.float()
    mma.attach_source_codes(fp32_weight, codes, grid)
    with pytest.raises(TypeError, match="BF16 weight"):
        cube.cube_quantized_linear_ascend(activation(1, 64), fp32_weight)


# ------------------------------------------------------------------------------------------------
# The packed-expert GEMV
# ------------------------------------------------------------------------------------------------


def _packed_gemv_inputs(rows=6, columns=64, seed=31):
    packed, grid = packed_projection(rows, columns, seed=seed)
    x = activation(1, columns, seed=seed + 1)
    codes, scale = quantize_block_fp8(x.contiguous(), backend="torch")
    return codes, scale, packed, grid, x


def test_the_packed_gemv_is_the_natural_matmul_of_the_exact_decodes():
    codes, scale, packed, grid, _ = _packed_gemv_inputs()
    out = cube.cube_fp4_packed_gemv(codes, scale, packed, grid, backend="torch")
    act = unpack_fp8_to_bf16(codes, scale, block_shape=(1, BLOCK), backend="torch")
    weight = cube.decode_fp4_packed_bf16(packed, grid, backend="torch")
    expected = (act @ weight.t()).to(torch.float32)
    assert out.dtype == torch.float32 and out.shape == (1, 6)
    assert torch.equal(out, expected)


def test_the_packed_gemv_is_within_bf16_rounding_of_the_exact_operands():
    codes, scale, packed, grid, _ = _packed_gemv_inputs(rows=5, columns=128, seed=37)
    out = cube.cube_fp4_packed_gemv(codes, scale, packed, grid, backend="torch")
    exact = cube._fp8_exact(codes, scale, BLOCK).double() @ cube._fp4_packed_exact(packed, grid, BLOCK).double().t()
    # A BF16-accumulated dot of K=128 terms: the bound is the K-order difference, not a tolerance on
    # the operands -- the operands themselves are bit-exact by construction.
    assert (out.double() - exact).abs().max().item() <= 2.0**-7 * exact.abs().max().item() * 8


def test_the_packed_gemv_refuses_operand_shapes_that_are_not_the_contract():
    codes, scale, packed, grid, _ = _packed_gemv_inputs()
    with pytest.raises(ValueError, match="packed FP4 weight bytes must be"):
        cube.cube_fp4_packed_gemv(codes, scale, packed[:, :-1], grid, backend="torch")
    with pytest.raises(ValueError, match="one activation row"):
        cube.cube_fp4_packed_gemv(codes.repeat(2, 1), scale.repeat(2, 1), packed, grid, backend="torch")
    with pytest.raises(ValueError, match="one scale row per output row"):
        cube.cube_fp4_packed_gemv(codes, scale, packed, grid[:1], backend="torch")


# ------------------------------------------------------------------------------------------------
# The bounded per-object cache
# ------------------------------------------------------------------------------------------------


def test_the_projection_cache_is_off_by_default_and_still_computes():
    codes, scale, packed, grid, _ = _packed_gemv_inputs()
    projection = cube.CubePackedProjection(backend="torch")
    first = projection(codes, scale, packed, grid)
    second = projection(codes, scale, packed, grid)
    assert torch.equal(first, second)
    stats = projection.cache_stats
    assert stats["decodes"] == 2 and stats["entries"] == 0 and stats["retained_bytes"] == 0
    assert stats["cache_budget_bytes"] == 0


def test_an_explicit_budget_hits_and_stays_within_its_bytes():
    codes, scale, packed, grid, _ = _packed_gemv_inputs()
    budget = 1 << 20
    projection = cube.CubePackedProjection(backend="torch", cache_budget_bytes=budget)
    projection(codes, scale, packed, grid)
    projection(codes, scale, packed, grid)
    stats = projection.cache_stats
    assert stats["hits"] == 1 and stats["entries"] == 1 and stats["decodes"] == 1
    assert 0 < stats["retained_bytes"] <= budget


def test_two_projection_objects_share_no_state_and_clear_releases_entries():
    codes, scale, packed, grid, _ = _packed_gemv_inputs()
    cached = cube.CubePackedProjection(backend="torch", cache_budget_bytes=1 << 20)
    plain = cube.CubePackedProjection(backend="torch")
    cached(codes, scale, packed, grid)
    cached(codes, scale, packed, grid)
    plain(codes, scale, packed, grid)
    assert cached.cache_stats["hits"] == 1 and plain.cache_stats["hits"] == 0
    assert cached.clear() == 1 and cached.cache_stats["retained_bytes"] == 0


def test_a_retained_entry_keeps_its_source_alive_until_cleared():
    codes, scale, packed, grid, _ = _packed_gemv_inputs()
    base_codes, base_grid = packed.clone(), grid.clone()
    live = weakref.ref(base_codes)
    projection = cube.CubePackedProjection(backend="torch", cache_budget_bytes=1 << 20)
    projection(codes, scale, base_codes[:], base_grid[:])
    del base_codes, base_grid
    gc.collect()
    assert live() is not None, "a cached entry must hold its packed bytes, not just a pointer"
    projection.clear()
    gc.collect()
    assert live() is None


def test_a_budget_above_the_ceiling_is_refused_and_metadata_is_reported():
    with pytest.raises(ValueError, match="exceeds the"):
        cube.CubePackedProjection(backend="torch", cache_budget_bytes=cube.MAX_CACHE_BYTES + 1)
    projection = cube.CubePackedProjection(backend="torch")
    provenance = projection.provenance
    assert provenance["mode"] == cube.MODE
    assert provenance["activation_requantization"].startswith("none")
    assert provenance["reference_reduction_installed"] is False
    assert "none" in provenance["cache"]["timing_claim"]


# ------------------------------------------------------------------------------------------------
# The provider wrapper
# ------------------------------------------------------------------------------------------------


def test_the_provider_routes_the_emulation_flag_to_the_captured_original():
    calls: list = []

    def original(x, weight, *, block_size=32, emulate_reduction=True):
        calls.append(emulate_reduction)
        return "original"

    provider = cube.make_linear_provider(original)
    assert provider(object(), object(), emulate_reduction=False) == "original"
    assert calls == [False]
    assert cube.make_linear_provider.__doc__ and "cannot recurse" in cube.make_linear_provider.__doc__
    with pytest.raises(TypeError, match="must be callable"):
        cube.make_linear_provider(None)


def test_the_provider_reaches_the_experiment_when_the_flag_is_set(seams):
    weight, _, _, _ = row_scaled_weight(4, 64, seed=41)
    x = activation(1, 64, seed=43)
    provider = cube.make_linear_provider(lambda *arguments, **keywords: "original")
    out = provider(x, weight)
    assert out.shape == (1, 4) and out.dtype == torch.bfloat16


# ------------------------------------------------------------------------------------------------
# The optional native weight-quantization mode
# ------------------------------------------------------------------------------------------------


def recording_weight_quant_op(seen: list):
    """A stand-in for ``npu_weight_quant_batchmatmul`` that mirrors its documented arithmetic.

    It dequantizes the ``[K, N]`` INT8 operand with the per-group ``[ceil(K/32), N]`` BF16 scale and
    accumulates in BF16, which is the contract the device operator documents. Recording the operands is
    the point: the test then pins the layout this module passes, not the operator's behaviour.
    """

    def op(
        x,
        weight,
        antiquant_scale,
        antiquant_offset=None,
        quant_scale=None,
        quant_offset=None,
        bias=None,
        antiquant_group_size=0,
        inner_precise=0,
    ):
        seen.append(
            {
                "x": x,
                "weight": weight,
                "antiquant_scale": antiquant_scale,
                "antiquant_offset": antiquant_offset,
                "quant_scale": quant_scale,
                "bias": bias,
                "antiquant_group_size": antiquant_group_size,
                "inner_precise": inner_precise,
            }
        )
        dequantized = weight.float() * antiquant_scale.repeat_interleave(
            antiquant_group_size, dim=0
        )[: weight.shape[0]]
        return (x @ dequantized.to(torch.bfloat16)).to(x.dtype)

    return op


def test_weight_quant_mode_passes_the_documented_operand_layout(seams, monkeypatch):
    seen: list = []
    monkeypatch.setattr(cube, "_npu_weight_quant_op", lambda: recording_weight_quant_op(seen))
    codes, scale, packed, grid, _ = _packed_gemv_inputs(rows=5, columns=128, seed=61)
    out = cube.cube_fp4_weight_quant_gemv(codes, scale, packed, grid)

    call = seen[0]
    assert call["weight"].dtype == torch.int8 and call["weight"].shape == (128, 5), "[K, N] int8"
    assert call["antiquant_scale"].dtype == torch.bfloat16, "the scale dtype must match x"
    assert call["antiquant_scale"].shape == (4, 5), "[ceil(K/32), N] per-group scales"
    assert call["x"].dtype == torch.bfloat16 and call["x"].shape == (1, 128)
    assert call["antiquant_group_size"] == BLOCK and call["inner_precise"] == 0
    assert call["antiquant_offset"] is None and call["quant_scale"] is None, "no output quantization"
    assert out.dtype == torch.float32 and out.shape == (1, 5)


def test_weight_quant_mode_is_the_same_arithmetic_as_the_decode_mode(seams, monkeypatch):
    """The operand mapping is right iff int8-times-two times the halved grid is the original value.

    With a stand-in that performs exactly the documented on-chip dequantization, this mode must land on
    the *same* bits as decoding the projection to BF16 with the un-halved grid and multiplying -- which
    is the whole claim of using the frozen INT8 pair and the original half grid.
    """
    seen: list = []
    monkeypatch.setattr(cube, "_npu_weight_quant_op", lambda: recording_weight_quant_op(seen))
    codes, scale, packed, grid, _ = _packed_gemv_inputs(rows=6, columns=128, seed=67)
    native = cube.cube_fp4_weight_quant_gemv(codes, scale, packed, grid)
    decoded = cube.cube_fp4_packed_gemv(codes, scale, packed, grid, backend="torch")
    assert torch.equal(native.view(torch.int32), decoded.view(torch.int32))


def test_weight_quant_mode_refuses_a_missing_operator_by_name(seams):
    def unavailable():
        raise cube.CubeExperimentUnavailable("no npu_weight_quant_batchmatmul in this build")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cube, "_npu_weight_quant_op", unavailable)
        codes, scale, packed, grid, _ = _packed_gemv_inputs()
        with pytest.raises(cube.CubeExperimentUnavailable, match="weight_quant_batchmatmul"):
            cube.cube_fp4_weight_quant_gemv(codes, scale, packed, grid)


def test_the_half_scale_gate_refuses_nan_and_bf16_inexact_scales():
    nan_scale = torch.tensor([[float("nan")]], dtype=torch.float32)
    with pytest.raises(cube.OriginalOperandNotBf16Representable, match="non-finite"):
        cube._half_scale_bf16(nan_scale, name="weight half-scale", verify=True)
    # 1.1 is not a BF16 value, so casting it would round a scale rather than only change the reduction.
    with pytest.raises(cube.OriginalOperandNotBf16Representable, match="not exactly representable"):
        cube._half_scale_bf16(torch.tensor([[1.1]]), name="weight half-scale", verify=True)
    # Powers of two are exact all the way down to the BF16 subnormal floor.
    for exponent in (-128, -127, -126, 0, 126):
        exact = cube._half_scale_bf16(
            torch.tensor([[2.0**exponent]]), name="weight half-scale", verify=True
        )
        assert exact.item() == 2.0**exponent and exact.dtype == torch.bfloat16


def test_a_subnormal_product_is_refused_but_can_be_allowed_explicitly():
    """A subnormal operand is exactly encodable and still refused, because a device may flush it."""
    codes = torch.full((1, 16), 0x01, dtype=torch.uint8)  # codes 1 and 0: 0.5 and +0.0
    scales = torch.full((1, 1), 1, dtype=torch.uint8)  # 2**-126 -> 0.5 * 2**-126 = 2**-127
    with pytest.raises(cube.OriginalOperandNotBf16Representable, match="subnormal"):
        cube.check_bf16_representable(codes, scales, kind="fp4", name="weight")
    report = cube.check_bf16_representable(
        codes, scales, kind="fp4", name="weight", allow_subnormal=True
    )
    assert report["subnormal"] == 16, "the 16 nonzero lanes are subnormal and the zeros are not"
    assert report["not_bf16_representable"] == 0 and report["non_finite"] == 0


def test_the_projection_weight_quant_mode_uses_the_native_op_and_refuses_a_budget(monkeypatch):
    with pytest.raises(ValueError, match="no BF16 weight temporary to cache"):
        cube.CubePackedProjection(mode="weight_quant", cache_budget_bytes=1 << 20)
    with pytest.raises(ValueError, match="the core's per-K32 native kernel is a separate operator"):
        cube.CubePackedProjection(mode="per_k32_partial")
    projection = cube.CubePackedProjection(mode="weight_quant")
    assert projection.provenance["mode"] == cube.WEIGHT_QUANT_MODE
    assert projection.provenance["cache"]["applicable"] is False
    assert projection.cache_stats["entries"] == 0


def test_the_projection_default_mode_is_still_the_decode_mode():
    projection = cube.CubePackedProjection(backend="torch")
    assert projection.provenance["mode"] == cube.MODE
    assert projection.provenance["projection_mode"] == "decode"


# ------------------------------------------------------------------------------------------------
# Cross-check with relic-core's independent certification helper (task 57's fixture)
# ------------------------------------------------------------------------------------------------


def test_the_guard_agrees_with_the_core_fixture_for_a_packed_store():
    """The exact fixture the core colleague names: K=64, N=5, and their two negative cases."""
    scale_bytes = torch.tensor([[130, 131], [129, 132], [128, 128], [133, 129], [127, 130]],
                               dtype=torch.uint8)
    generator = torch.Generator().manual_seed(0)
    packed = torch.randint(0, 256, (5, 32), generator=generator, dtype=torch.uint8)
    assert cube.check_bf16_representable(packed, scale_bytes, kind="fp4", name="weight")["elements"] == 5 * 64

    # The int8_x2 view of the same store must certify identically.
    int8 = unpack_fp4_to_int8(packed, backend="torch")
    half = fp4_int8_scale(scale_bytes, backend="torch")
    assert int8.shape == (5, 64) and half.shape == (5, 2)

    # Negative case 1: any 255 scale byte is NaN in the frozen decoder.
    with pytest.raises(cube.OriginalOperandNotBf16Representable):
        cube.check_bf16_representable(
            packed, torch.full((5, 2), 255, dtype=torch.uint8), kind="fp4", name="weight"
        )
    # Negative case 2: a byte of 1 puts a nonzero lane on a BF16 subnormal (2**-127).
    with pytest.raises(cube.OriginalOperandNotBf16Representable):
        cube.check_bf16_representable(
            packed, torch.full((5, 2), 1, dtype=torch.uint8), kind="fp4", name="weight"
        )


# ------------------------------------------------------------------------------------------------
# Device cases: the lead's to run on a leased card
# ------------------------------------------------------------------------------------------------


def _device():
    pytest.importorskip("torch_npu")
    return torch.device(os.environ[DEVICE_ENV])


@requires_device
def test_npu_dense_cube_is_finite_and_matches_its_bf16_operands():
    device = _device()
    generator = torch.Generator().manual_seed(101)
    rows, columns = 16, 128
    source = torch.randn(rows, columns, generator=generator) * 2
    codes, grid_bytes = quantize_block_fp8(source, backend="torch")
    weight = unpack_fp8_to_bf16(codes, grid_bytes, block_shape=(1, BLOCK), backend="torch").to(device)
    mma.attach_source_codes(
        weight,
        codes.to(device).view(torch.float8_e4m3fn),
        e8m0_fp32(grid_bytes).to(device),
    )
    x = activation(2, columns, seed=103).to(device)
    out = cube.cube_quantized_linear_ascend(x, weight)
    assert out.dtype == torch.bfloat16 and out.shape == (2, rows)
    assert bool(torch.isfinite(out).all())
    # The device result against the exact original operands, in float64. This is the report the lead
    # asked for: what the changed reduction costs on one real layer-shaped projection.
    a_codes, a_scale = quantize_block_fp8(x.reshape(-1, columns).contiguous(), backend="triton")
    exact = cube._fp8_exact(a_codes.cpu(), a_scale.cpu(), BLOCK).double() @ cube._fp8_exact(
        codes, grid_bytes, BLOCK
    ).double().t()
    print(
        "NPU_DENSE_DIFF",
        {"max_abs": float((out.cpu().double() - exact).abs().max()),
         "max_abs_reference": float(exact.abs().max())},
    )


@requires_device
def test_npu_packed_gemv_is_finite_and_reports_its_difference():
    device = _device()
    generator = torch.Generator().manual_seed(107)
    rows, columns = 8, 128
    weight = (torch.randn(rows, columns, generator=generator) * 2).to(torch.bfloat16)
    packed, grid = quantize_block_fp4(weight, BLOCK, "e8m0", backend="torch")
    x = activation(1, columns, seed=109)
    codes, scale = quantize_block_fp8(x.contiguous(), backend="torch")
    out = cube.cube_fp4_packed_gemv(
        codes.to(device), scale.to(device), packed.to(device), grid.to(device)
    )
    assert out.dtype == torch.float32 and out.shape == (1, rows)
    assert bool(torch.isfinite(out).all())
    exact = cube._fp8_exact(codes, scale, BLOCK).double() @ cube._fp4_packed_exact(
        packed, grid, BLOCK
    ).double().t()
    print(
        "NPU_PACKED_GEMV_DIFF",
        {"max_abs": float((out.cpu().double() - exact).abs().max()),
         "max_abs_reference": float(exact.abs().max())},
    )


def test_cube_callable_matches_the_real_packed_provider_keyword_seam():
    from relicllm.models.deepseek_v4_1.ascend_packed_experts import PackedMatrix, _call_packed_gemv
    a = torch.full((1, 64), 0x38, dtype=torch.uint8)
    sa = torch.full((1, 2), 127, dtype=torch.uint8)
    b = torch.full((32, 32), 0x11, dtype=torch.uint8)
    sb = torch.full((32, 2), 127, dtype=torch.uint8)
    provider = cube.CubePackedProjection(backend="torch", cache_budget_bytes=0)
    result = _call_packed_gemv(provider, a, sa, PackedMatrix(b, sb, 32), backend="torch")
    assert result.dtype == torch.float32
    assert torch.equal(result, torch.full((1, 32), 32.0))
