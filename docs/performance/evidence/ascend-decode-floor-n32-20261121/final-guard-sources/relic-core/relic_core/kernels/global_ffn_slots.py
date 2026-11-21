"""One global FFN collective retains each expert's FP32 down partial separately.

Prepare [topk expert slots + one shared slot, dim]. SUM across physical ranks,
then narrow EACH expert to BF16 before ascending-expert FP32 combination. Never
sum all expert FP32 partials into one row before the per-expert boundary.
"""
from __future__ import annotations

import math
import torch

from .ascend_unpack import _npu_launch


def _check(down, selected, ids, shared, slots):
    if down.ndim != 2:
        raise ValueError("global FFN down partials must be a complete FP32 expert-row matrix")
    topk, dim = down.shape
    if (not 1 <= topk <= 8 or dim <= 0 or (topk + 1) * dim > 2**31 - 1
            or down.dtype != torch.float32 or not down.is_contiguous()
            or selected.shape != (topk,) or selected.dtype != torch.int32
            or ids.shape != (1, topk) or ids.dtype not in (torch.int32, torch.int64)
            or shared.shape != (1, dim) or shared.dtype != torch.bfloat16
            or slots.shape != (topk + 1, dim) or slots.dtype != torch.float32):
        raise ValueError("global FFN slots must retain FP32 down, original IDs and BF16 shared partial")
    values = (down, selected, ids, shared, slots)
    if any(value.device != down.device or not value.is_contiguous() or value.requires_grad for value in values):
        raise ValueError("global FFN slot metadata must share one contiguous device")
    if slots.untyped_storage()._cdata in {value.untyped_storage()._cdata for value in values[:-1]}:
        raise ValueError("global FFN slots cannot alias source partials")
    return topk, dim


@torch.no_grad()
def prepare_global_ffn_slots(down, selected, ids, shared, slots, *, backend="triton"):
    topk, dim = _check(down, selected, ids, shared, slots)
    expected = "cpu" if backend == "torch" else "npu"
    if backend not in ("torch", "triton") or down.device.type != expected:
        raise ValueError("global FFN slots need the explicit CPU or NPU backend")
    if backend == "torch":
        slots.zero_()
        source = {expert:slot for slot, expert in enumerate(selected.tolist()) if expert >= 0}
        for slot, expert in enumerate(sorted(ids[0].tolist())):
            if expert in source:
                slots[slot].copy_(down[source[expert]])
        slots[topk].copy_(shared[0].float())
    else:
        from ._ascend_global_expert_slots_kernels import prepare_global_ffn_slots_kernel
        with _npu_launch(down):
            prepare_global_ffn_slots_kernel[(math.ceil(dim / 256), topk + 1)](
                down, selected, ids, shared, slots, dim, topk, 1 << (topk - 1).bit_length(), 256)
    return slots


@torch.no_grad()
def finish_global_ffn_slots(reduced, route_error, *, backend="triton"):
    if (reduced.ndim != 2 or reduced.dtype != torch.float32 or not reduced.is_contiguous()
            or not 2 <= reduced.shape[0] <= 9 or reduced.shape[1] <= 0 or reduced.numel() > 2**31 - 1
            or route_error.shape != (1,) or route_error.dtype != torch.int32 or not route_error.is_contiguous()
            or reduced.requires_grad or route_error.requires_grad or route_error.device != reduced.device):
        raise ValueError("global FFN finalization requires reduced FP32 per-expert slots")
    topk, dim = reduced.shape[0] - 1, reduced.shape[1]
    expected = "cpu" if backend == "torch" else "npu"
    if backend not in ("torch", "triton") or reduced.device.type != expected:
        raise ValueError("global FFN finalization needs its explicit CPU or NPU backend")
    if backend == "torch":
        if int(route_error.item()):
            return torch.full((1, dim), float('nan'))
        result = torch.zeros((1, dim), dtype=torch.float32)
        for slot in range(topk):
            result = result + reduced[slot:slot + 1].bfloat16().float()
        return result + reduced[topk:topk + 1]
    from ._ascend_global_expert_slots_kernels import finish_global_ffn_slots_kernel
    output = torch.empty((1, dim), dtype=torch.float32, device=reduced.device)
    with _npu_launch(reduced):
        finish_global_ffn_slots_kernel[(math.ceil(dim / 256),)](reduced, route_error, output, dim, topk, 256, enable_fp_fusion=False)
    return output
