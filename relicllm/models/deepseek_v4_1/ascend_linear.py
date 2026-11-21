"""NPU bindings for original FP8 linear operands, without float8 arithmetic casts.

The operators live in relic-core. This module only chooses the stored code/grid
layout and the model's output dtype; it does not recover source weight codes.
"""
from __future__ import annotations

import torch

from . import mma


def quantized_linear_ascend(x, weight, *, block_size=32, emulate_reduction=True):
    if x.device.type != "npu" or weight.device != x.device:
        raise ValueError("the Ascend quantized linear needs activation and weight on one NPU")
    if block_size != 32:
        raise ValueError("original V4.1 NPU linear operands use block_size=32")
    if x.dtype not in {torch.bfloat16,torch.float32}:
        raise TypeError("original NPU FP8 activation quantization accepts BF16 or FP32")
    if weight.ndim != 2 or x.shape[-1] != weight.shape[-1]:
        raise ValueError("quantized linear needs a [N,K] weight matching the activation K")
    from relic_core.kernels.block_fp8_quant import quantize_block_fp8
    from relic_core.kernels.block_gemm import fp8_block_gemm
    flat=x.reshape(-1,x.shape[-1]).contiguous()
    a_codes,a_scale=quantize_block_fp8(flat,backend="triton")
    if emulate_reduction:
        codes,scale=mma.attached_codes(weight),mma.attached_scale(weight)
        if codes is None or scale is None:
            raise RuntimeError(
                "NPU quantized linear needs original E4M3 codes and the checkpoint scale grid "
                "attached to the actual weight; source recovery or an unreported matmul fallback "
                "would change the reference reduction"
            )
        if tuple(scale.shape)==(weight.shape[0],weight.shape[1]//32):
            rows=1
        elif tuple(scale.shape)==(-(-weight.shape[0]//32),weight.shape[1]//32):
            rows=32
        else:
            raise ValueError("the attached weight scale grid is not the original block-32 layout")
        import os
        use_decode_gemv = flat.shape[0] == 1 and os.environ.get(
            "DEEPSEEK_V41_ASCEND_DECODE_GEMV", "0"
        ).lower() in {"1", "true", "yes", "on"}
        if use_decode_gemv:
            # Independent K-block fragments followed by the same sequential
            # FP32 accumulation; this is not a native-Cube precision escape.
            implementation = os.environ.get("DEEPSEEK_V41_ASCEND_FP8_GEMV_IMPL", "split")
            if implementation == "float_significand":
                from relic_core.kernels.floating_significand_gemv import fp8_float_significand_gemv
                result = fp8_float_significand_gemv(
                    a_codes, a_scale, codes, scale, block_rows=rows, backend="triton", variant="grouped"
                )
            elif implementation == "split":
                from relic_core.kernels.decode_block_gemv import fp8_decode_gemv
                result = fp8_decode_gemv(
                    a_codes, a_scale, codes, scale, block_rows=rows, backend="triton"
                )
            else:
                raise ValueError("DEEPSEEK_V41_ASCEND_FP8_GEMV_IMPL must be 'split' or 'float_significand'")
        else:
            result=fp8_block_gemm(a_codes,a_scale,codes,scale,block_rows=rows,backend="triton")
    else:
        # This is the user's existing explicit throughput escape hatch. Its
        # activation still follows the original FP8 quantizer, not W8A8.
        from relic_core.kernels.unpack import unpack_fp8_to_bf16
        act=unpack_fp8_to_bf16(a_codes,a_scale,block_shape=(1,32),backend="triton").float()
        result=torch.matmul(act,weight.float().transpose(0,1))
    return result.to(x.dtype).reshape(*x.shape[:-1],weight.shape[0])
