"""Opt-in exact performance bindings preserve code grids and CPU Engram dispatch."""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import cpu_block_gemv
from relic_core.kernels.unpack import unpack_fp8_to_bf16
from relicllm.models.deepseek_v4_1 import kernels, mma


def operands(rows=1, *, grid_rows=32):
    generator = torch.Generator().manual_seed(917)
    n, k = 32, 64
    raw = torch.randint(0, 256, (n, k), dtype=torch.uint8, generator=generator)
    raw = torch.where((raw & 127) == 127, torch.full_like(raw, 126), raw)
    grid = torch.randint(118, 124, (math.ceil(n / grid_rows), k // 32),
                         dtype=torch.uint8, generator=generator)
    scales = torch.tensor([math.ldexp(1.0, int(v)-127) for v in grid.flatten()],
                          dtype=torch.float32).reshape_as(grid)
    weight = unpack_fp8_to_bf16(raw, grid, block_shape=(grid_rows, 32), backend="torch")
    mma.attach_source_codes(weight, raw.view(torch.float8_e4m3fn), scales)
    x = torch.randn(rows, k, generator=generator).to(torch.bfloat16)
    return x, weight


@pytest.mark.parametrize("grid_rows", [1, 32])
def test_compiled_host_binding_passes_original_operands_once(monkeypatch, grid_rows):
    x, weight = operands(grid_rows=grid_rows)
    monkeypatch.delenv("DEEPSEEK_V41_CPU_EXACT_GEMV", raising=False)
    expected = kernels.quantized_linear(x, weight, quantized=True, emulate_reduction=True)
    captured = []
    old_reference = mma.fp8_block_gemm

    def compiled(a, sa, b, sb, *, block_rows, threads):
        captured.append((a, sa, b, sb, block_rows, threads))
        return old_reference(a, sa, b, sb)

    def refuse_old(*args, **kwargs):
        raise AssertionError("the enabled single-row binding entered the old CPU GEMM")

    monkeypatch.setattr(cpu_block_gemv, "fp8_block_gemv", compiled)
    monkeypatch.setattr(mma, "fp8_block_gemm", refuse_old)
    monkeypatch.setenv("DEEPSEEK_V41_CPU_EXACT_GEMV", "1")
    monkeypatch.setenv("DEEPSEEK_V41_CPU_GEMV_THREADS", "64")
    actual = kernels.quantized_linear(x, weight, quantized=True, emulate_reduction=True)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert len(captured) == 1
    a, sa, b, sb, seen_rows, threads = captured[0]
    assert b is mma.attached_codes(weight) and sb is mma.attached_scale(weight)
    assert a.shape == x.shape and sa.shape == (1, 2)
    assert (seen_rows, threads) == (grid_rows, 64)
    assert all(value.device.type == "cpu" for value in (a, sa, b, sb, actual))


@pytest.mark.parametrize("enabled,rows", [(False, 1), (True, 2)])
def test_cpu_default_and_prefill_never_enter_the_new_operator(monkeypatch, enabled, rows):
    x, weight = operands(rows)
    monkeypatch.setenv("DEEPSEEK_V41_CPU_EXACT_GEMV", "1" if enabled else "0")

    def forbidden(*args, **kwargs):
        raise AssertionError("default or multi-row prefill entered a decode-only operator")

    monkeypatch.setattr(cpu_block_gemv, "fp8_block_gemv", forbidden)
    output = kernels.quantized_linear(x, weight, quantized=True, emulate_reduction=True)
    assert output.shape == (rows, weight.shape[0])
    assert output.device.type == "cpu"


def test_explicit_compiled_cpu_request_refuses_missing_original_metadata(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_V41_CPU_EXACT_GEMV", "1")
    x = torch.ones(1, 32, dtype=torch.bfloat16)
    weight = torch.ones(32, 32, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="original E4M3 codes and the scale grid"):
        kernels.quantized_linear(x, weight, quantized=True, emulate_reduction=True)


def test_compiled_flag_does_not_change_nonquantized_cpu_arithmetic(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_V41_CPU_EXACT_GEMV", "1")
    x, weight = operands()
    actual = kernels.quantized_linear(x, weight, quantized=False)
    expected = torch.nn.functional.linear(x, weight)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def test_compile_error_is_not_a_silent_reference_fallback(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_V41_CPU_EXACT_GEMV", "1")
    x, weight = operands()

    def compiler_failure(*args, **kwargs):
        raise RuntimeError("test compiler unavailable")

    monkeypatch.setattr(cpu_block_gemv, "fp8_block_gemv", compiler_failure)
    with pytest.raises(RuntimeError, match="test compiler unavailable"):
        kernels.quantized_linear(x, weight, quantized=True, emulate_reduction=True)


def test_raw_simd_binding_keeps_the_original_cpu_code_grid(monkeypatch):
    from relic_core.kernels import cpu_raw_simd_gemv
    x, weight = operands()
    monkeypatch.setenv("DEEPSEEK_V41_CPU_EXACT_GEMV", "0")
    expected = kernels.quantized_linear(x, weight, quantized=True, emulate_reduction=True)
    old_reference = mma.fp8_block_gemm
    calls = []

    def raw(a, sa, b, sb, *, block_rows, threads):
        calls.append((b, sb, block_rows, threads))
        return old_reference(a, sa, b, sb)

    def forbidden(*args, **kwargs):
        raise AssertionError("the selected raw SIMD path entered the scalar operator")

    monkeypatch.setattr(cpu_raw_simd_gemv, "fp8_raw_simd_gemv", raw)
    monkeypatch.setattr(cpu_block_gemv, "fp8_block_gemv", forbidden)
    monkeypatch.setenv("DEEPSEEK_V41_CPU_EXACT_GEMV", "1")
    monkeypatch.setenv("DEEPSEEK_V41_CPU_GEMV_IMPL", "raw_simd")
    monkeypatch.setenv("DEEPSEEK_V41_CPU_GEMV_THREADS", "96")
    actual = kernels.quantized_linear(x, weight, quantized=True, emulate_reduction=True)
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    assert len(calls) == 1
    b, sb, rows, threads = calls[0]
    assert b is mma.attached_codes(weight) and sb is mma.attached_scale(weight)
    assert (rows, threads) == (32, 96)
    assert actual.device.type == "cpu"


def test_unknown_cpu_implementation_is_not_a_fallback(monkeypatch):
    x, weight = operands()
    monkeypatch.setenv("DEEPSEEK_V41_CPU_EXACT_GEMV", "1")
    monkeypatch.setenv("DEEPSEEK_V41_CPU_GEMV_IMPL", "unknown")
    with pytest.raises(ValueError, match="DEEPSEEK_V41_CPU_GEMV_IMPL"):
        kernels.quantized_linear(x, weight, quantized=True, emulate_reduction=True)
