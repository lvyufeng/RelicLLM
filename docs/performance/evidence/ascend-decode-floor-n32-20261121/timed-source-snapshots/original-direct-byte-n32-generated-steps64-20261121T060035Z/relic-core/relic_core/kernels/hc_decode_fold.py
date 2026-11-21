"""Opt-in batch-one HC collapse/residual-fold kernels.

Only FP32 multiply, four-term reduction, addition and the original BF16 output
boundary are fused. No coefficient, epsilon, Sinkhorn, normalization or weight
precision is changed. Device reduction ordering must be validated against the
current runtime before a caller labels the transformation bit-neutral.
"""
from __future__ import annotations

import math
import torch

from .ascend_unpack import _npu_launch, supports_triton_ascend


def _input(x, *, shape_tail):
    if (x.shape[:-len(shape_tail)] != (1, 1) or tuple(x.shape[-len(shape_tail):]) != shape_tail
            or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.requires_grad):
        raise ValueError("HC decode operands require contiguous inference BF16 with batch/token one")


def _coeff(value, shape, device):
    if (value.shape != shape or value.dtype != torch.float32 or value.device != device
            or not value.is_contiguous() or value.requires_grad):
        raise ValueError("HC coefficients must be the matching contiguous original FP32 tensors")


def _output(out, shape, device, operands):
    if out is None:
        return torch.empty(shape, dtype=torch.bfloat16, device=device)
    if (out.shape != shape or out.dtype != torch.bfloat16 or out.device != device
            or not out.is_contiguous() or out.requires_grad):
        raise ValueError("HC output must match its contiguous BF16 extent and device")
    if any(out.untyped_storage()._cdata == tensor.untyped_storage()._cdata for tensor in operands):
        raise ValueError("HC output must not alias an input or coefficient")
    return out


def _backend(x, backend):
    if backend not in ("torch", "triton"):
        raise ValueError("choose an explicit CPU oracle or NPU backend")
    if backend == "torch":
        if x.device.type != "cpu":
            raise ValueError("the CPU oracle is not an NPU fallback")
    elif x.device.type != "npu" or not supports_triton_ascend():
        raise ValueError("the fused HC kernel requires the Ascend Triton backend")


@torch.no_grad()
def fused_hc_decode_pre(x, pre_mix, *, backend="triton", out=None):
    """Collapse original BF16 [1,1,4,D] with original FP32 [1,1,4]."""
    if x.ndim != 4 or x.shape[-1] <= 0:
        raise ValueError("HC input must have four dimensions and a positive hidden width")
    dim = x.shape[-1]
    _input(x, shape_tail=(4, dim))
    _coeff(pre_mix, (1, 1, 4), x.device)
    _backend(x, backend)
    result = _output(out, (1, 1, dim), x.device, (x, pre_mix))
    if backend == "torch":
        result.copy_((pre_mix.unsqueeze(-1) * x.float()).sum(dim=2).to(torch.bfloat16))
        return result
    from ._ascend_hc_decode_fold_kernels import hc_decode_pre_kernel
    with _npu_launch(x):
        hc_decode_pre_kernel[(math.ceil(dim / 256),)](x, pre_mix, result, dim, 256, enable_fp_fusion=False)
    return result


@torch.no_grad()
def fused_hc_decode_post(x, residual, post, comb, *, backend="triton", out=None):
    """Fold a sublayer BF16 [1,1,D] into original BF16 [1,1,4,D]."""
    if x.ndim != 3 or x.shape[-1] <= 0:
        raise ValueError("HC sublayer output requires a positive hidden width")
    dim = x.shape[-1]
    _input(x, shape_tail=(dim,))
    _input(residual, shape_tail=(4, dim))
    if residual.device != x.device:
        raise ValueError("HC residual and sublayer output must share the device")
    _coeff(post, (1, 1, 4), x.device)
    _coeff(comb, (1, 1, 4, 4), x.device)
    _backend(x, backend)
    result = _output(out, (1, 1, 4, dim), x.device, (x, residual, post, comb))
    if backend == "torch":
        value = post.unsqueeze(-1) * x.unsqueeze(-2)
        value = value + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=2)
        result.copy_(value.to(torch.bfloat16))
        return result
    from ._ascend_hc_decode_fold_kernels import hc_decode_post_kernel
    with _npu_launch(x):
        hc_decode_post_kernel[(4, math.ceil(dim / 256))](
            x, residual, post, comb, result, dim, 256, enable_fp_fusion=False)
    return result
