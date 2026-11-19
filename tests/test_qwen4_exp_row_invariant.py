"""Opt-in BF16 row invariance uses decode-shaped operations, not altered precision.

CPU tests pin helper dispatch, layout, order and call counts. They do not substitute for the
real-checkpoint GPU oracle comparing every layer, recurrent state and full-vocabulary logits.
"""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relicllm.models.qwen4_exp import attention, layers


FLAG = "POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT"


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 9])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_linear_dispatch_preserves_dtype_shape_and_values(monkeypatch, enabled, rows, dtype):
    monkeypatch.setenv(FLAG, "1" if enabled else "0")
    generator = torch.Generator().manual_seed(rows)
    x = torch.randn(1, rows, 16, generator=generator).to(dtype)
    weight = torch.randn(11, 16, generator=generator).to(dtype)
    bias = torch.randn(11, generator=generator).to(dtype)
    original = F.linear
    calls = []

    def record(value, w, b=None):
        calls.append((tuple(value.shape), value.dtype, w is weight, value.is_contiguous()))
        return original(value, w, b)

    monkeypatch.setattr(F, "linear", record)
    output = layers.row_invariant_linear(x, weight, bias)
    active = enabled and dtype is torch.bfloat16 and 2 <= rows <= 8
    expected = (torch.cat([original(x[:, row:row + 1], weight, bias) for row in range(rows)], dim=1)
                if active else original(x, weight, bias))
    assert output.shape == (1, rows, 11) and output.dtype is dtype
    assert torch.equal(output, expected)
    assert len(calls) == (rows if active else 1)
    if active:
        assert all(shape == (1, 1, 16) and same_weight and contiguous
                   for shape, _dtype, same_weight, contiguous in calls)


def test_linear_noncontiguous_leading_axes_restore_original_layout_shape(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    x = torch.arange(2 * 3 * 16, dtype=torch.float32).reshape(2, 3, 16).transpose(0, 1).to(torch.bfloat16)
    weight = torch.eye(16, dtype=torch.bfloat16)
    output = layers.row_invariant_linear(x, weight)
    assert output.shape == x.shape and torch.equal(output, x)


def test_prefill_linear_and_hc_router_use_explicit_helper(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_DENSE_FP16_MIN_ROWS", "1")
    original = layers.row_invariant_linear
    calls = []

    def record(x, weight, bias=None):
        calls.append((tuple(x.shape), x.dtype, weight.dtype))
        return original(x, weight, bias)

    monkeypatch.setattr(layers, "row_invariant_linear", record)
    x = torch.randn(1, 4, 8).to(torch.bfloat16)
    layers.prefill_linear(x, torch.randn(5, 8).to(torch.bfloat16))
    config = SimpleNamespace(hc_count=2, hidden_size=4, rms_norm_eps=1e-6)
    hc = layers.GatedResidual(
        config, torch.zeros(8, dtype=torch.bfloat16),
        torch.randn(3, 8).to(torch.bfloat16), torch.randn(8, 3).to(torch.bfloat16),
        torch.randn(2, 8).to(torch.bfloat16),
    )
    hc(x)
    layers.TopKRouter(torch.randn(5, 8).to(torch.bfloat16), 2)(x.reshape(4, 8))
    assert len(calls) == 5  # prefill + HC down/up/inject + router
    assert all(input_dtype is weight_dtype is torch.bfloat16 for _shape, input_dtype, weight_dtype in calls)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("outputs", [1, 2, 4, 8, 9])
@pytest.mark.parametrize("dilation", [1, 3])
def test_depthwise_convolution_uses_contiguous_one_output_windows(monkeypatch, enabled, outputs, dilation):
    monkeypatch.setenv(FLAG, "1" if enabled else "0")
    channels, kernel = 3, 3
    span = dilation * (kernel - 1) + 1
    x = torch.arange(channels * (span + outputs - 1), dtype=torch.float32).reshape(1, channels, -1)
    x = (x / 100).to(torch.bfloat16)
    weight = torch.randn(channels, 1, kernel).to(torch.bfloat16)
    original = F.conv1d
    calls = []

    def record(value, w, bias=None, stride=1, padding=0, dilation=1, groups=1):
        calls.append(value.clone())
        assert value.is_contiguous()
        return original(value, w, bias, stride, padding, dilation, groups)

    monkeypatch.setattr(F, "conv1d", record)
    result = layers.row_invariant_conv1d(x, weight, dilation=dilation, groups=channels)
    active = enabled and 2 <= outputs <= 8
    expected = (torch.cat([
        original(x[..., row:row + span].contiguous(), weight, dilation=dilation, groups=channels)
        for row in range(outputs)
    ], dim=-1) if active else original(x, weight, dilation=dilation, groups=channels))
    assert result.shape == (1, channels, outputs) and result.dtype is torch.bfloat16
    assert torch.equal(result, expected)
    assert len(calls) == (outputs if active else 1)
    if active:
        assert all(torch.equal(value, x[..., row:row + span]) for row, value in enumerate(calls))


@pytest.mark.parametrize("kind", ["float32", "padding", "stride", "not_depthwise"])
def test_convolution_other_dispatches_remain_single_calls(monkeypatch, kind):
    monkeypatch.setenv(FLAG, "1")
    dtype = torch.float32 if kind == "float32" else torch.bfloat16
    x = torch.randn(1, 2, 6).to(dtype)
    weight = torch.randn(2, 2 if kind == "not_depthwise" else 1, 3).to(dtype)
    kwargs = dict(groups=1 if kind == "not_depthwise" else 2,
                  padding=1 if kind == "padding" else 0, stride=2 if kind == "stride" else 1)
    original = F.conv1d
    calls = []

    def record(*args, **kw):
        calls.append(1)
        return original(*args, **kw)

    monkeypatch.setattr(F, "conv1d", record)
    actual = layers.row_invariant_conv1d(x, weight, **kwargs)
    assert torch.equal(actual, original(x, weight, **kwargs)) and len(calls) == 1


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 9])
@pytest.mark.parametrize("splitk", [False, True])
def test_qsa_small_queries_follow_one_query_dispatch_and_order(monkeypatch, enabled, rows, splitk):
    monkeypatch.setenv(FLAG, "1" if enabled else "0")
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_SPLITK", "1")
    query = torch.arange(rows * 4, dtype=torch.float32).reshape(1, rows, 1, 4).to(torch.bfloat16)
    key = torch.zeros(1, 1, 16, 4, dtype=torch.bfloat16)
    value = torch.ones_like(key)
    selection = torch.arange(rows, dtype=torch.int32).reshape(1, rows, 1).expand(-1, -1, 3).contiguous()
    calls = []

    def dispatch(name):
        def invoke(q, k, v, picks, scale):
            assert k is key and v is value and scale == 0.5
            assert q.is_contiguous() and picks.is_contiguous()
            calls.append((name, q.clone(), picks.clone()))
            return q + 1
        return invoke

    ops = SimpleNamespace(qwen4_exp_qsa_bf16_forward=dispatch("general"))
    if splitk:
        ops.qwen4_exp_qsa_splitk_bf16_forward = dispatch("splitk")
    actual = attention._qsa_forward_rows(ops, query, key, value, selection, 0.5)
    active = enabled and 2 <= rows <= 8
    assert torch.equal(actual, query + 1)
    assert len(calls) == (rows if active else 1)
    expected_name = "splitk" if splitk and (active or rows == 1) else "general"
    assert all(name == expected_name for name, _q, _s in calls)
    if active:
        assert all(torch.equal(q, query[:, row:row + 1]) and torch.equal(picks, selection[:, row:row + 1])
                   for row, (_name, q, picks) in enumerate(calls))
