"""Explicit binding for the independently built AscendC tile-consumer prototype.

The wrapper validates metadata without reading device routing back to CPU and
passes only existing NPU pointers plus static geometry to the mixed AIV/AIC
kernel. No build, installation or global provider replacement occurs on import.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import torch

from .online_tile_cube import _validate
from .ascend_unpack import _npu_launch


class AscendCTileCubeConsumer:
    """One explicit native library owner; original packed/tile operands only."""

    def __init__(self, library_path, *, sha256: str, block_dim=24):
        path = Path(library_path).resolve(strict=True)
        if not path.is_file() or len(sha256) != 64 or hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError("native tile consumer needs the exact verified library checksum")
        if isinstance(block_dim, bool) or not isinstance(block_dim, int) or not 1 <= block_dim <= 24:
            raise ValueError("Ascend910B prototype block_dim must be within 1..24")
        self.path = path
        self.sha256 = sha256
        self.block_dim = block_dim
        self._library = ctypes.CDLL(str(path))
        query = self._library.relic_online_tile_cube_k_blocks_per_tile
        query.restype, query.argtypes = ctypes.c_uint32, []
        self.group_k = query()
        if self.group_k not in (1, 2, 4):
            raise ValueError("native library reports an unsupported original K32 grouping")
        self.ring_elements = 512 * self.group_k
        self._launch = self._library.relic_online_tile_cube_launch
        self._launch.restype = None
        self._launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p,
                                 *([ctypes.c_void_p] * 6), *([ctypes.c_uint32] * 7)]

    @torch.no_grad()
    def __call__(self, x, b_tile, s_tile, selected, *, world, n_output=None,
                 certified_bf16_operands=False, scratch=None, accumulation=None):
        owned, tiles, blocks, groups, n = _validate(x, b_tile, s_tile, selected,
            world=world, n_output=n_output, group_k=1, out=None)
        if n != tiles * 16:
            raise ValueError("the native tile prototype currently requires the complete physical output extent")
        if x.device.type != "npu" or certified_bf16_operands is not True:
            raise ValueError("the native consumer requires NPU and load-time exact BF16 certification")
        if blocks % self.group_k:
            raise ValueError("the original K32 grid must be divisible by the native library's tile grouping")
        output = accumulation if accumulation is not None else torch.empty((groups, n), dtype=torch.float32, device=x.device)
        stage = scratch if scratch is not None else torch.empty((self.block_dim, 2, self.ring_elements), dtype=torch.bfloat16, device=x.device)
        for tensor, shape, dtype in ((output, (groups, n), torch.float32),
                                     (stage, (self.block_dim, 2, self.ring_elements), torch.bfloat16)):
            if (tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device != x.device
                    or not tensor.is_contiguous() or tensor.requires_grad):
                raise ValueError("native ring/output scratch metadata is inconsistent")
        if len({t.untyped_storage()._cdata for t in (x, b_tile, s_tile, selected, output, stage)}) != 6:
            raise ValueError("native ring/output scratch must not alias input banks or each other")
        with _npu_launch(x):
            stream = torch.npu.current_stream(x.device).npu_stream
            self._launch(self.block_dim, stream, x.data_ptr(), b_tile.data_ptr(),
                         s_tile.data_ptr(), selected.data_ptr(), output.data_ptr(), stage.data_ptr(),
                         tiles, blocks, groups, owned, world, n, int(x.shape[0] == 1))
        # Preserve the released per-expert BF16 store before the ascending FP32
        # combine. This small output boundary is not an expanded-weight buffer.
        return output.to(torch.bfloat16)

    def describe(self):
        return {"library": str(self.path), "sha256": self.sha256,
                "AIV_to_small_GM_ring_to_L1_to_AIC": True,
                "original_K32_blocks_per_Cube_tile": self.group_k,
                "ring_weight_double_buffer_bytes_per_owner": self.ring_elements * 4,
                "global_expanded_weight_scratch_bytes": self.block_dim * self.ring_elements * 4,
                "whole_matrix_expanded_weight_scratch_bytes": 0,
                "native_full_K_accumulation_differs_from_exact_default": True}
