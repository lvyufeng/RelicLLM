"""NPU call-site bindings for the original activation quantizers.

Byte-level quantization and unpacking belong to relic-core. The model's NPU
paths retain block-32 E8M0 or block-16/32 E4M3 scales and the in-place activation
dtype; they never construct an unsupported native NPU FP8/FP4 tensor. CPU and
CUDA calls continue through their existing implementation.
"""
from __future__ import annotations

import torch


def fp8_quant_inplace(
    x: torch.Tensor,
    block_size: int,
    scale_fmt: str | None,
    scale_dtype: torch.dtype,
) -> torch.Tensor:
    """Quantize/dequantize the window KV in BF16 using original raw E4/E8 bytes."""
    if x.device.type != "npu" or x.dtype != torch.bfloat16:
        raise ValueError("the V4.1 NPU in-place FP8 activation binding requires a BF16 NPU tensor")
    if block_size != 32 or scale_fmt not in {"ue8m0", "e8m0"}:
        raise ValueError("the original V4.1 NPU FP8 activation uses block32 with E8M0 scales")
    if scale_dtype not in {getattr(torch, "float8_e8m0fnu", None), torch.float32}:
        raise ValueError("the original FP8 activation scale must be E8M0 or its FP32 value")
    from relic_core.kernels.block_fp8_quant import quantize_block_fp8
    from relic_core.kernels.unpack import unpack_fp8_to_bf16

    shape = tuple(x.shape)
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    codes, scales = quantize_block_fp8(flat, backend="triton")
    decoded = unpack_fp8_to_bf16(codes, scales, block_shape=(1, 32), backend="triton")
    x.copy_(decoded.reshape(shape))
    return x


def act_quant(
    x: torch.Tensor,
    block_size: int = 128,
    scale_fmt: str | None = None,
    scale_dtype: torch.dtype = torch.float32,
    inplace: bool = False,
):
    """Keep the old-device interface and bind the NPU window-KV in-place call.

    A non-in-place NPU caller must use relic-core's explicit raw-byte API rather
    than receive uint8 values masquerading as the legacy floating-type results.
    """
    if x.device.type != "npu":
        from relic_core.kernels.ops import act_quant as legacy

        return legacy(x, block_size, scale_fmt, scale_dtype, inplace)
    if not inplace:
        raise ValueError("the V4.1 NPU act_quant binding requires inplace=True; use quantize_block_fp8 for raw bytes")
    return fp8_quant_inplace(x, block_size, scale_fmt, scale_dtype)


def fp4_act_quant(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    scale_dtype: torch.dtype | None = None,
):
    """Bind indexer FP4; optional E4M3 selects the compressed-KV variant.

    Non-in-place NPU results are the core API's packed uint8 codes and raw scale
    bytes, not native floating dtypes. The active indexer calls in-place.
    """
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if scale_dtype is not None and scale_dtype not in {e8m0, torch.float8_e4m3fn}:
        raise ValueError("FP4 activation scales must be E8M0 or E4M3")
    format_name = "e4m3" if scale_dtype == torch.float8_e4m3fn else "e8m0"
    if x.device.type != "npu":
        if format_name == "e4m3":
            from .kernels import fp4_act_quant_e4m3

            return fp4_act_quant_e4m3(x, block_size, inplace)
        from relic_core.kernels.ops import fp4_act_quant as legacy

        return legacy(x, block_size, inplace)
    return fp4_quant_npu(x, block_size, inplace, scale_format=format_name)


def fp4_quant_npu(
    x: torch.Tensor,
    block_size: int = 32,
    inplace: bool = False,
    *,
    scale_format: str = "e8m0",
):
    """Call the original-grid core kernel; execution/compilation errors propagate."""
    if x.device.type != "npu":
        raise ValueError("the V4.1 NPU FP4 activation binding needs an NPU tensor")
    from relic_core.kernels.block_fp4_quant import quantize_block_fp4

    return quantize_block_fp4(
        x, block_size=block_size, scale_format=scale_format, inplace=inplace, backend="triton"
    )
