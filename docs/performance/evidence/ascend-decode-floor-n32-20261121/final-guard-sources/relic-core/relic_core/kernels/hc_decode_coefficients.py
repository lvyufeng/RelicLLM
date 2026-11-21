"""One FP32 HC coefficient projection and RMS statistic from original operands.

No HC weight narrowing or activation quantization. The native reduction order is
an explicit experiment; affine/sigmoid/softmax/Sinkhorn belong to the caller's
unchanged original provider after this helper.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

from .ascend_unpack import _npu_launch, supports_triton_ascend


@torch.no_grad()
def hc_project_rms(x, coefficients, *, norm_eps, backend="triton", k_tile=1024):
    if (x.ndim != 4 or x.shape[:3] != (1, 1, 4) or x.dtype != torch.bfloat16
            or not x.is_contiguous() or x.requires_grad):
        raise ValueError("HC projection requires one contiguous original four-copy BF16 input")
    k = x.numel()
    if (coefficients.shape != (24, k) or coefficients.dtype != torch.float32
            or coefficients.device != x.device or not coefficients.is_contiguous()):
        raise ValueError("HC coefficients must remain original contiguous FP32 [24,4*dim]")
    if isinstance(norm_eps, bool) or not isinstance(norm_eps, (int, float)) or not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("original HC RMS epsilon must be finite and positive")
    if backend not in ("torch", "triton") or isinstance(k_tile, bool) or not isinstance(k_tile, int) or k_tile not in (256, 512, 1024, 2048):
        raise ValueError("choose an explicit HC backend and bounded local K tile")
    if backend == "torch":
        if x.device.type != "cpu":
            raise ValueError("the HC CPU oracle is not an NPU fallback")
        source = x.flatten(2).float()
        return F.linear(source, coefficients) * torch.rsqrt(source.square().mean(-1, keepdim=True) + norm_eps)
    if x.device.type != "npu" or not supports_triton_ascend():
        raise ValueError("HC native projection requires the explicit Triton Ascend backend")
    from ._ascend_hc_decode_coefficients_kernels import hc_project_rms_kernel
    output = torch.empty((1, 1, 24), dtype=torch.float32, device=x.device)
    with _npu_launch(x):
        hc_project_rms_kernel[(24,)](x, coefficients, output, k, norm_eps, k_tile, enable_fp_fusion=False)
    return output
