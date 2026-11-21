"""Opt-in original-FP4 MoE with device selection and bounded native tile consumption.

This composes the independently validated AscendC consumer with the existing
original FP8 encode/decode, original FP32 SiLU, and ascending expert combine.
The CPU branch is an explicit numerical oracle, never a hot NPU fallback.
"""
from dataclasses import dataclass
import math

import torch

from .ascend_unpack import _npu_launch, supports_triton_ascend
from .device_selected_moe import _torch_routes, _triton_routes, _torch_weight, fused_moe_epilogue
from .fast_fused_fp8_activation_decode import quantize_decode_fp8_fast
from .tile_major_packed_gemv import unpack_tile_major

CONTRACT = "original_fp4_tile_ring_native_cube_original_fp8_original_silu_ascending_fp32_combine"


@dataclass
class TileMoEWorkspace:
    selected: torch.Tensor
    gate_weights: torch.Tensor
    group_list: torch.Tensor
    route_error: torch.Tensor
    weight_storage: torch.Tensor
    gate_accumulation: torch.Tensor
    up_accumulation: torch.Tensor
    down_accumulation: torch.Tensor
    dim: int
    inter_dim: int
    topk: int
    group_k: int
    block_dim: int
    weight_rows: int = 16

    @property
    def tensors(self):
        return (self.selected, self.gate_weights, self.group_list, self.route_error,
                self.weight_storage, self.gate_accumulation, self.up_accumulation, self.down_accumulation)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors)


def make_tile_moe_workspace(dim, inter_dim, topk, *, device, block_dim=24, group_k=1,
                            weight_rows=16, max_bytes=512 * 2**20):
    for value in (dim, inter_dim, topk, block_dim, group_k, weight_rows, max_bytes):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("tile workspace geometry and budget must be positive integers")
    if (dim % 32 or inter_dim % 32 or topk > 8 or block_dim > 24 or group_k not in (1, 2, 4, 8)
            or weight_rows not in (16, 32) or weight_rows * group_k > 128):
        raise ValueError("unsupported original tile workspace geometry")
    if dim % (32 * group_k) or inter_dim % (32 * group_k):
        raise ValueError("native tiles must retain complete original K32 blocks")
    required = (topk * 8 + (topk + 1) * 8 + 4 + block_dim * 128 * weight_rows * group_k
                + topk * (2 * inter_dim + dim) * 4)
    if required > max_bytes:
        raise ValueError(f"tile MoE workspace needs {required} bytes, budget is {max_bytes}")
    return TileMoEWorkspace(
        torch.empty(topk, dtype=torch.int32, device=device),
        torch.empty(topk, dtype=torch.float32, device=device),
        torch.empty(topk + 1, dtype=torch.int64, device=device),
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.empty((block_dim, 2, weight_rows * 32 * group_k), dtype=torch.bfloat16, device=device),
        torch.empty((topk, inter_dim), dtype=torch.float32, device=device),
        torch.empty((topk, inter_dim), dtype=torch.float32, device=device),
        torch.empty((topk, dim), dtype=torch.float32, device=device),
        dim, inter_dim, topk, group_k, block_dim, weight_rows)


def _validate(x, weights, indices, banks, workspace, rank, world, n_experts):
    if (isinstance(world, bool) or not isinstance(world, int) or world <= 0
            or isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world):
        raise ValueError("tile MoE requires valid cyclic ownership")
    if isinstance(n_experts, bool) or not isinstance(n_experts, int) or not workspace.topk <= n_experts < 2**31:
        raise ValueError("tile MoE requires an int32 expert count at least topk")
    if x.shape != (1, workspace.dim) or x.dtype != torch.bfloat16 or not x.is_contiguous() or x.requires_grad:
        raise ValueError("tile MoE requires one contiguous original BF16 input row")
    if (indices.shape != (1, workspace.topk) or indices.dtype not in (torch.int32, torch.int64)
            or weights.shape != indices.shape or weights.dtype not in (torch.float32, torch.bfloat16)
            or not weights.is_contiguous() or not indices.is_contiguous() or weights.requires_grad):
        raise ValueError("tile MoE requires contiguous original routing tensors")
    expected_scratch = ((workspace.selected, (workspace.topk,), torch.int32),
                        (workspace.gate_weights, (workspace.topk,), torch.float32),
                        (workspace.group_list, (workspace.topk + 1,), torch.int64),
                        (workspace.route_error, (1,), torch.int32),
                        (workspace.weight_storage, (workspace.block_dim, 2, workspace.weight_rows * 32 * workspace.group_k), torch.bfloat16),
                        (workspace.gate_accumulation, (workspace.topk, workspace.inter_dim), torch.float32),
                        (workspace.up_accumulation, (workspace.topk, workspace.inter_dim), torch.float32),
                        (workspace.down_accumulation, (workspace.topk, workspace.dim), torch.float32))
    if any(t.shape != shape or t.dtype != dtype or not t.is_contiguous() or t.requires_grad
           for t, shape, dtype in expected_scratch):
        raise ValueError("tile MoE scratch metadata is inconsistent")
    owned = len(range(rank, n_experts, world))
    all_tensors = [x, weights, indices, *workspace.tensors]
    for name, n, k in (("w1", workspace.inter_dim, workspace.dim),
                       ("w3", workspace.inter_dim, workspace.dim),
                       ("w2", workspace.dim, workspace.inter_dim)):
        b, s = banks[name]
        if (b.shape != (owned, n // 16, k // 32, 16, 16)
                or s.shape != (owned, n // 16, k // 32, 16)
                or b.dtype != torch.uint8 or s.dtype != torch.uint8
                or not b.is_contiguous() or not s.is_contiguous()):
            raise ValueError(f"{name} must be the lossless original cyclic tile bank")
        all_tensors.extend((b, s))
    if any(t.device != x.device for t in all_tensors):
        raise ValueError("all tile MoE operands and scratch must share a device")
    storage = [t.untyped_storage()._cdata for t in workspace.tensors]
    if len(set(storage)) != len(storage) or set(storage).intersection(t.untyped_storage()._cdata for t in all_tensors[:3] + all_tensors[11:]):
        raise ValueError("tile MoE scratch must not alias operands or other scratch")


def _torch_projection(x, bank, slots, *, world, topk, n):
    result = torch.zeros((topk, n), dtype=torch.bfloat16)
    for group, (expert, _) in enumerate(slots):
        b, s = unpack_tile_major(bank[0][expert // world], bank[1][expert // world], n_output=n)
        weight = _torch_weight(b, s)
        row = x[0 if x.shape[0] == 1 else group]
        result[group] = (row.float()[None] @ weight.float().t())[0].bfloat16()
    return result


@torch.no_grad()
def device_tile_fp4_moe(x, weights, indices, banks, *, rank, world, n_experts,
                         workspace, consumer=None, swiglu_limit=0.0,
                         certified_bf16_operands=False, backend="triton"):
    _validate(x, weights, indices, banks, workspace, rank, world, n_experts)
    if backend not in ("torch", "triton") or not math.isfinite(swiglu_limit) or swiglu_limit < 0:
        raise ValueError("select an explicit tile backend and finite nonnegative clamp")
    if certified_bf16_operands is not True:
        raise ValueError("certify the original E8M0 BF16 operand range at load time")
    if backend == "triton":
        if x.device.type != "npu" or not supports_triton_ascend() or consumer is None:
            raise ValueError("the NPU tile experiment needs its explicit validated native consumer")
        if (consumer.group_k, consumer.block_dim, getattr(consumer, "output_rows", 16)) != (workspace.group_k, workspace.block_dim, workspace.weight_rows):
            raise ValueError("native tile geometry must match persistent scratch")
    elif x.device.type != "cpu":
        raise ValueError("the CPU tile oracle is not a fallback for NPU")
    decoded_x = quantize_decode_fp8_fast(x, backend=backend)
    if backend == "torch":
        slots = _torch_routes(indices, weights, rank=rank, world=world, n_experts=n_experts, workspace=workspace)
        gate = _torch_projection(decoded_x, banks["w1"], slots, world=world, topk=workspace.topk, n=workspace.inter_dim)
        up = _torch_projection(decoded_x, banks["w3"], slots, world=world, topk=workspace.topk, n=workspace.inter_dim)
    else:
        _triton_routes(indices, weights, rank=rank, world=world, n_experts=n_experts, workspace=workspace)
        options = dict(world=world, certified_bf16_operands=True, scratch=workspace.weight_storage)
        gate = consumer(decoded_x, *banks["w1"], workspace.selected, accumulation=workspace.gate_accumulation, **options)
        up = consumer(decoded_x, *banks["w3"], workspace.selected, accumulation=workspace.up_accumulation, **options)
    intermediate = fused_moe_epilogue(gate, workspace.gate_weights, up=up,
                                       swiglu_limit=swiglu_limit, backend=backend)
    if backend == "torch":
        contributions = _torch_projection(intermediate, banks["w2"], slots, world=world, topk=workspace.topk, n=workspace.dim)
        result = torch.zeros((1, workspace.dim), dtype=torch.float32)
        for group in range(len(slots)):
            result = result + contributions[group:group+1].float()
        return result
    contributions = consumer(intermediate, *banks["w2"], workspace.selected,
                             accumulation=workspace.down_accumulation, **options)
    from ._ascend_device_selected_moe_kernels import ordered_combine_kernel
    result = torch.empty((1, workspace.dim), dtype=torch.float32, device=x.device)
    with _npu_launch(contributions):
        ordered_combine_kernel[(math.ceil(workspace.dim / 256),)](
            contributions, workspace.selected, workspace.route_error, result,
            workspace.dim, workspace.topk, 256)
    return result
