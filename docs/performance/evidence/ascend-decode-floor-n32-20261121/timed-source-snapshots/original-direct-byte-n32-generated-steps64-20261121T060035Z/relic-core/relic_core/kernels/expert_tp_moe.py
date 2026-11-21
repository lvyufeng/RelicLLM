"""Original-byte expert TP: full-K up/gate, local-K FP32 down partials.

Output-channel up/gate slices retain their original BF16 narrowing. The original
FP32 SiLU/clamp/routing-weight sequence and K32 activation quantizer run locally.
Down partials are FP32 and MUST be reduced across TP before a single original
BF16 per-expert narrowing and ascending-global-expert FP32 combine.
"""
from __future__ import annotations

import math
import torch

from .ascend_unpack import _npu_launch
from .ascendc_tile_moe import _validate, _torch_projection
from .device_selected_moe import _torch_routes, _triton_routes, fused_moe_epilogue, _torch_weight
from .fast_fused_fp8_activation_decode import quantize_decode_fp8_fast
from .tile_major_packed_gemv import unpack_tile_major

CONTRACT = "original_fp4_expert_TP_FP32_down_reduce_before_BF16_ascending_expert_combine"


@torch.no_grad()
def tp_moe_down_partials(x, weights, indices, banks, *, ep_rank, ep_world, n_experts,
                         workspace, consumer=None, swiglu_limit=0.0, backend="triton"):
    """Compute one EP group's local intermediate slice, without narrowing down."""
    _validate(x, weights, indices, banks, workspace, ep_rank, ep_world, n_experts)
    if backend not in ("torch", "triton") or not math.isfinite(swiglu_limit) or swiglu_limit < 0:
        raise ValueError("expert TP requires an explicit backend and finite nonnegative clamp")
    activation = quantize_decode_fp8_fast(x, backend=backend)
    if backend == "torch":
        if x.device.type != "cpu":
            raise ValueError("the TP CPU oracle is not an NPU fallback")
        slots = _torch_routes(indices, weights, rank=ep_rank, world=ep_world,
                              n_experts=n_experts, workspace=workspace)
        gate = _torch_projection(activation, banks["w1"], slots, world=ep_world,
                                 topk=workspace.topk, n=workspace.inter_dim)
        up = _torch_projection(activation, banks["w3"], slots, world=ep_world,
                               topk=workspace.topk, n=workspace.inter_dim)
    else:
        if consumer is None or (consumer.group_k, consumer.block_dim, getattr(consumer, "output_rows", 16)) != (workspace.group_k, workspace.block_dim, workspace.weight_rows):
            raise ValueError("TP consumer and local-K workspace geometry must match")
        _triton_routes(indices, weights, rank=ep_rank, world=ep_world,
                       n_experts=n_experts, workspace=workspace)
        options = dict(world=ep_world, certified_bf16_operands=True, scratch=workspace.weight_storage)
        gate = consumer(activation, *banks["w1"], workspace.selected,
                        accumulation=workspace.gate_accumulation, **options)
        up = consumer(activation, *banks["w3"], workspace.selected,
                      accumulation=workspace.up_accumulation, **options)
    local_mid = fused_moe_epilogue(gate, workspace.gate_weights, up=up,
                                   swiglu_limit=swiglu_limit, backend=backend)
    if backend == "torch":
        partials = workspace.down_accumulation
        partials.zero_()
        for slot, (expert, _) in enumerate(slots):
            code, scale = unpack_tile_major(banks["w2"][0][expert // ep_world],
                                            banks["w2"][1][expert // ep_world], n_output=workspace.dim)
            weight = _torch_weight(code, scale)
            partials[slot] = (local_mid[slot].float()[None] @ weight.float().t())[0]
        return partials
    return consumer(local_mid, *banks["w2"], workspace.selected,
                    accumulation=workspace.down_accumulation, output_dtype=torch.float32, **options)


@torch.no_grad()
def finish_tp_moe(reduced_down_fp32, workspace, *, backend="triton", emit=True):
    """Narrow ONLY after TP reduction; emit one EP contribution on its TP leader."""
    if (reduced_down_fp32.shape != (workspace.topk, workspace.dim)
            or reduced_down_fp32.dtype != torch.float32 or not reduced_down_fp32.is_contiguous()
            or reduced_down_fp32.device != workspace.down_accumulation.device):
        raise ValueError("TP finalization requires complete reduced FP32 per-expert down values")
    if not isinstance(emit, bool) or backend not in ("torch", "triton"):
        raise ValueError("TP emission and backend must be explicit")
    if (backend == "torch" and reduced_down_fp32.device.type != "cpu") or (backend == "triton" and reduced_down_fp32.device.type != "npu"):
        raise ValueError("TP finalization must use its explicitly matched CPU or NPU backend")
    if not emit:
        # Other TP ranks still compute and participate in the down reduction.
        # Returning zero avoids duplicating routed experts in the unchanged
        # TP8 global MoE reduction, while every rank's shared partial remains.
        return torch.zeros((1, workspace.dim), dtype=torch.float32, device=reduced_down_fp32.device)
    contributions = reduced_down_fp32.to(torch.bfloat16)
    if backend == "torch":
        if reduced_down_fp32.device.type != "cpu":
            raise ValueError("the TP CPU oracle is not an NPU fallback")
        if int(workspace.route_error.item()):
            return torch.full((1, workspace.dim), float("nan"))
        result = torch.zeros((1, workspace.dim), dtype=torch.float32)
        for slot, expert in enumerate(workspace.selected.tolist()):
            if expert >= 0:
                result = result + contributions[slot:slot + 1].float()
        return result
    from ._ascend_device_selected_moe_kernels import ordered_combine_kernel
    result = torch.empty((1, workspace.dim), dtype=torch.float32, device=reduced_down_fp32.device)
    with _npu_launch(contributions):
        ordered_combine_kernel[(math.ceil(workspace.dim / 256),)](
            contributions, workspace.selected, workspace.route_error, result,
            workspace.dim, workspace.topk, 256)
    return result
