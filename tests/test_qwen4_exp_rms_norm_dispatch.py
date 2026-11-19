"""The fused norm must be reachable at decode, and must mean the same thing when it is.

Two things are easy to get wrong here and both were: the gate that decides whether the kernel may
run (borrowed from the hyper-connection kernels, which default off with a 128-row floor, so a
one-row decode step never qualified), and the ungrouped case (which only reached the kernel when a
group size had been given, so the model's ordinary norms never did).
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch

from relicllm.models.qwen4_exp import layers


def test_the_norm_kernel_can_be_enabled_at_one_row(monkeypatch):
    # A decode step is one row; the old gate could never accept it because it borrowed the
    # hyper-connection predicate, whose floor is 128 rows.
    monkeypatch.setenv("POCKETLLM_QWEN4_RMS_NORM_CUDA", "1")
    assert layers._rms_norm_cuda_enabled(1) is True


def test_the_norm_kernel_defaults_on(monkeypatch):
    # Three interleaved A/B rounds on eight RTX 4090s at 8192 positions: 38.77 against 37.65 tok/s
    # excluding setup (medians), 25.13 against 25.88 ms p50, identical tokens in all six runs.
    monkeypatch.delenv("POCKETLLM_QWEN4_RMS_NORM_CUDA", raising=False)
    assert layers._rms_norm_cuda_enabled(1) is True


def test_the_norm_kernel_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_RMS_NORM_CUDA", "0")
    assert layers._rms_norm_cuda_enabled(1) is False


def test_the_gate_is_not_the_hyper_connection_one(monkeypatch):
    # The two decisions are separate: turning the norm kernel on must not depend on HC, and
    # turning HC on must not turn the norm on by itself.
    monkeypatch.setenv("POCKETLLM_QWEN4_RMS_NORM_CUDA", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_HC_CUDA", "0")
    assert layers._hc_cuda_enabled(1) is False
    assert layers._rms_norm_cuda_enabled(1) is True
    # And the norm switch alone decides the norm: turning it off is enough.
    monkeypatch.setenv("POCKETLLM_QWEN4_RMS_NORM_CUDA", "0")
    monkeypatch.setenv("POCKETLLM_QWEN4_HC_CUDA", "1")
    assert layers._rms_norm_cuda_enabled(1) is False


def _record_group_size(monkeypatch):
    seen = {}

    def fake(x, weight, group_size, eps):
        seen["group_size"] = group_size
        seen["shape"] = tuple(x.shape)
        return torch.zeros_like(x)

    monkeypatch.setattr(layers, "ext_qwen4_placeholder", None, raising=False)
    monkeypatch.setattr(layers, "_cuda_grouped_rms_norm", fake)
    return seen


@pytest.mark.parametrize("group_size, expected", [(None, 2560), (640, 640)])
def test_an_ungrouped_norm_asks_for_one_group(monkeypatch, group_size, expected):
    seen = _record_group_size(monkeypatch)
    norm = layers.RMSNorm(torch.ones(2560), 1e-6, group_size=group_size)
    x = torch.zeros(1, 2560)
    norm(x)
    assert seen["group_size"] == expected


def test_a_declined_kernel_falls_back_to_the_torch_path(monkeypatch):
    monkeypatch.setattr(layers, "_cuda_grouped_rms_norm", lambda *a, **k: None)
    norm = layers.RMSNorm(torch.zeros(8), 1e-6)
    x = torch.tensor([[3.0] * 8])
    # gain is 1 + weight = 1, so an all-equal row normalizes to 1.
    out = norm(x)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-5)


def test_ungrouped_and_single_group_agree_in_the_fallback(monkeypatch):
    # The two spellings must be the same arithmetic; this checks the reference path both ways.
    monkeypatch.setattr(layers, "_cuda_grouped_rms_norm", lambda *a, **k: None)
    torch.manual_seed(0)
    weight = torch.randn(64)
    x = torch.randn(3, 64)
    plain = layers.RMSNorm(weight, 1e-6)(x)
    grouped = layers.RMSNorm(weight, 1e-6, group_size=64)(x)
    assert torch.equal(plain, grouped)
