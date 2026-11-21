"""Original-operand Engram projection/gate for the opt-in device-compute path.

The caller owns host lookup and distributed output-channel gathering. Quantized
projection uses the released E4M3/E8M0 block32 activation, never integer INT8.
Native full-K FP32 Cube accumulation is a changed-reduction experiment. Gate
operations retain the original FP32 sequence and BF16 output boundary.
"""
from __future__ import annotations

import math
import torch

from .fast_fused_fp8_activation_decode import quantize_decode_fp8_fast

NUMERICAL_CONTRACT = "original_fp8_operands_native_full_K_BF16_output_original_FP32_Engram_gate"


def original_fp8_engram_projection(lookup, weight, *, quantized=True,
                                   certified_original_bf16_weight=False, backend="triton"):
    """Project one gathered BF16 row against a certified transposed weight slice."""
    if backend not in ("torch", "triton"):
        raise ValueError("choose the explicit CPU oracle or NPU backend")
    device_type = "cpu" if backend == "torch" else "npu"
    if lookup.device.type != device_type or weight.device != lookup.device:
        raise ValueError("projection operands must share the explicit backend device")
    if (lookup.ndim != 2 or lookup.shape[0] != 1 or lookup.shape[1] == 0
            or weight.ndim != 2 or weight.shape[1] == 0 or weight.shape[0] != lookup.shape[1]
            or lookup.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not lookup.is_contiguous() or not weight.is_contiguous()
            or lookup.requires_grad or weight.requires_grad):
        raise ValueError("projection requires contiguous inference BF16 [1,K] and [K,N]")
    if quantized:
        if lookup.shape[1] % 32 or certified_original_bf16_weight is not True:
            raise ValueError("certify the original FP8 weight and retain block32 activation quantization")
        lookup = quantize_decode_fp8_fast(lookup, backend=backend)
    return torch.matmul(lookup, weight)


def copy_fp32_signbit(magnitude, sign):
    """Exact IEEE sign injection without torch_npu's copysign CPU fallback."""
    if (magnitude.dtype != torch.float32 or sign.dtype != torch.float32
            or magnitude.shape != sign.shape or magnitude.device != sign.device
            or not magnitude.is_contiguous() or not sign.is_contiguous()):
        raise ValueError("copysign inputs must be matching contiguous FP32 tensors")
    value_bits = torch.bitwise_and(magnitude.view(torch.int32), 0x7FFFFFFF)
    sign_bits = torch.bitwise_and(sign.view(torch.int32), -2147483648)
    return torch.bitwise_or(value_bits, sign_bits).view(torch.float32)


def original_engram_gate(x, kv, channel_weight, token_mask, *, eps, clamp_value=1e-6):
    """Apply the released FP32 gate sequence to complete, already-gathered KV."""
    if (x.ndim != 4 or x.shape[:2] != (1, 1) or x.dtype != torch.bfloat16
            or not x.is_contiguous() or x.requires_grad):
        raise ValueError("Engram gate requires contiguous inference BF16 [1,1,hc,dim]")
    hc, dim = x.shape[2:]
    if (kv.shape != (1, (hc + 1) * dim) or kv.dtype != torch.bfloat16
            or channel_weight.shape != (hc, dim) or channel_weight.dtype != torch.float32
            or token_mask.shape != (1, 1) or token_mask.dtype != torch.bool
            or any(t.device != x.device or not t.is_contiguous() or t.requires_grad
                   for t in (kv, channel_weight, token_mask))):
        raise ValueError("KV, channel weights and mask must match the Engram geometry/device")
    if (isinstance(eps, bool) or isinstance(clamp_value, bool)
            or not math.isfinite(eps) or eps <= 0
            or not math.isfinite(clamp_value) or clamp_value <= 0):
        raise ValueError("Engram normalization epsilon and clamp must be finite and positive")
    key, value = kv.view(1, 1, -1).split([hc * dim, dim], dim=-1)
    key = key.float().unflatten(-1, (hc, dim))
    hidden = x.float()
    rstd = torch.rsqrt(hidden.square().mean(-1) + eps) * torch.rsqrt(key.square().mean(-1) + eps)
    dot = (hidden * channel_weight * key).sum(-1) * rstd * dim**-0.5
    # torch_npu 2.10 copysign falls back to CPU and cannot enter a graph.
    # Preserve its IEEE sign operation exactly using integer device operators.
    gate = torch.sigmoid(copy_fp32_signbit(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    gate = gate.masked_fill(~token_mask.unsqueeze(-1), 0)
    return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)
