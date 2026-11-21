"""Single-row original-byte AIV GEMV without an expanded-weight GM handoff.

This explicit reduction experiment applies original E2M1/E8M0 scales as exact
BF16 bit fields and reduces FP32 products. It preserves input quantization and
projection output boundaries, but does not emulate the exact window reduction.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch

from .ascend_unpack import _npu_launch, supports_triton_ascend
from .online_tile_cube import _validate
from .tile_major_packed_gemv import unpack_tile_major
from .unpack import unpack_fp4_to_int8, fp4_int8_scale


class DirectAIVConsumer:
    """An explicit no-ring M=1 provider, not a default or CPU fallback."""

    def __init__(self, *, backend="triton", block_dim=24, k_tile=256, n_tile=8):
        if backend not in ("torch", "triton"):
            raise ValueError("choose an explicit CPU oracle or Triton Ascend backend")
        if (isinstance(block_dim, bool) or not isinstance(block_dim, int) or not 1 <= block_dim <= 24
                or isinstance(k_tile, bool) or not isinstance(k_tile, int) or k_tile not in (32, 64, 128, 256)
                or isinstance(n_tile, bool) or not isinstance(n_tile, int) or n_tile not in (4, 8, 16)):
            raise ValueError("AIV programs and K32-aligned local tile extent must be explicit")
        self.backend, self.block_dim, self.k_tile, self.n_tile = backend, block_dim, k_tile, n_tile
        self.group_k = 1  # Local tile tails retain complete original K32 blocks.
        self.ring_elements = 512
        self.preload_activation = False
        self.direct_bf16_bits = True

    @torch.no_grad()
    def __call__(self, x, b_tile, s_tile, selected, *, world, n_output=None,
                 certified_bf16_operands=False, scratch=None, accumulation=None,
                 output_dtype=torch.bfloat16):
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("AIV output must preserve BF16 or explicit FP32 TP partials")
        owned, tiles, blocks, groups, n = _validate(x, b_tile, s_tile, selected,
            world=world, n_output=n_output, group_k=1, out=None)
        if certified_bf16_operands is not True:
            raise ValueError("certify original scales at load time before native bit decoding")
        output = accumulation if accumulation is not None else torch.empty((groups, n), dtype=torch.float32, device=x.device)
        if (output.shape != (groups, n) or output.dtype != torch.float32 or output.device != x.device
                or not output.is_contiguous() or output.requires_grad):
            raise ValueError("AIV output accumulation metadata is inconsistent")
        operands = (x, b_tile, s_tile, selected)
        if output.untyped_storage()._cdata in {value.untyped_storage()._cdata for value in operands}:
            raise ValueError("AIV output cannot alias original operands")
        if self.backend == "torch":
            if x.device.type != "cpu":
                raise ValueError("the AIV CPU logic oracle is never an NPU fallback")
            for group, expert in enumerate(selected.tolist()):
                if expert == -1:
                    output[group].zero_()
                elif expert < 0 or expert // world >= owned:
                    output[group].fill_(float("nan"))
                else:
                    code, scale = unpack_tile_major(b_tile[expert // world], s_tile[expert // world], n_output=n)
                    weight = (unpack_fp4_to_int8(code, backend="torch").float()
                              * fp4_int8_scale(scale, backend="torch").repeat_interleave(32, -1)).bfloat16()
                    valid_scale = ((scale >= 2) & (scale <= 252)).repeat_interleave(32, -1)
                    weight = torch.where(valid_scale, weight, float("nan"))
                    output[group].copy_((x[0 if x.shape[0] == 1 else group].float()[None] @ weight.float().t())[0])
        else:
            if x.device.type != "npu" or not supports_triton_ascend():
                raise ValueError("the AIV native path requires the explicit Triton Ascend backend")
            from ._ascend_direct_aiv_gemv_kernels import original_bit_aiv_gemv_kernel
            with _npu_launch(x):
                original_bit_aiv_gemv_kernel[(min(groups * math.ceil(n / self.n_tile), self.block_dim),)](
                    x, b_tile, s_tile, selected, output, n, blocks * 32, tiles, blocks,
                    groups, owned, world, x.shape[0] == 1, self.k_tile, self.n_tile,
                    max(b_tile.numel(), s_tile.numel(), output.numel()) >= 2**31,
                    enable_fp_fusion=False)
        return output if output_dtype == torch.float32 else output.to(torch.bfloat16)

    def describe(self):
        return {"provider": "direct_original_byte_AIV_GEMV", "backend": self.backend,
                "direct_original_BF16_bit_decoder": True, "local_K_tile": self.k_tile,
                "output_rows_per_program": self.n_tile,
                "original_K32_tail_supported": True, "expanded_weight_GM_write_read_bytes": 0,
                "caller_ring_scratch_unused_and_not_counted_as_saved_model_capacity": True,
                "projection_FP32_accumulate_and_explicit_BF16_boundary": True,
                "native_reduction_differs_from_exact_default": True,
                "packed_weight_and_original_scale_bytes_unchanged": True,
                "source": str(Path(__file__).resolve()), "not_a_measured_speedup_claim": True}
