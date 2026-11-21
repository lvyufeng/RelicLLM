"""Explicit binding for the independently built AscendC tile-consumer prototype.

The wrapper validates metadata without reading device routing back to CPU and
passes only existing NPU pointers plus static geometry to the mixed AIV/AIC
kernel. No build, installation or global provider replacement occurs on import.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path
from threading import RLock

import torch

from .online_tile_cube import _validate
from .ascend_unpack import _npu_launch


def activation_l1_layout(k, slots, *, group_k=4, shared=True, output_rows=16):
    """Static NZ-padded activation admission, matching the native 512 KiB L1."""
    if (isinstance(k, bool) or not isinstance(k, int) or k <= 0
            or isinstance(slots, bool) or not isinstance(slots, int) or not 1 <= slots <= 8
            or isinstance(group_k, bool) or not isinstance(group_k, int) or group_k not in (1, 2, 4, 8)
            or not isinstance(shared, bool) or k % (32 * group_k)
            or isinstance(output_rows, bool) or not isinstance(output_rows, int) or output_rows not in (16, 32)):
        raise ValueError("L1 admission requires valid original-K32 geometry and an explicit shared flag")
    weights = 2 * output_rows * (32 * group_k) * 2
    activation = (1 if shared else slots) * 16 * k * 2
    return {"L1_capacity_bytes": 512 * 1024, "weight_double_buffer_bytes": weights,
            "NZ_activation_bytes": activation, "total_bytes": weights + activation,
            "fits": weights + activation <= 512 * 1024,
            "admission_is_geometry_not_measured_occupancy": True}


def _decode_table_words(group_k, direct_bits, output_rows=16):
    if (isinstance(group_k, bool) or not isinstance(group_k, int) or group_k not in (1, 2, 4, 8)
            or not isinstance(direct_bits, bool) or isinstance(output_rows, bool)
            or not isinstance(output_rows, int) or output_rows not in (16, 32) or output_rows * group_k > 128):
        raise ValueError("decode tables require bounded original-K32 geometry and an explicit bit policy")
    pairs, elements = output_rows * 16 * group_k, output_rows * 32 * group_k
    return 512 + (pairs if direct_bits else elements) + pairs + (512 if direct_bits else 0)


def _decode_table_cpu(group_k, direct_bits, output_rows=16):
    """Exact small metadata only; no checkpoint weight or routing values are read."""
    words = _decode_table_words(group_k, direct_bits, output_rows)
    pairs, elements = output_rows * 16 * group_k, output_rows * 32 * group_k
    units = (0, 0x3f00, 0x3f80, 0x3fc0, 0x4000, 0x4040, 0x4080, 0x40c0,
             0, 0xbf00, 0xbf80, 0xbfc0, 0xc000, 0xc040, 0xc080, 0xc0c0)
    byte_order = [(index + 128) & 255 for index in range(256)]
    lookup = [units[byte & 15] | (units[byte >> 4] << 16) for byte in byte_order]
    scale_lookup = []
    for byte in byte_order:
        if direct_bits:
            offset = ((byte - 127) * 128) & 0xffff
            scale_lookup.append((offset | (offset << 16)) if 2 <= byte <= 252 else 0)
        else:
            scale_lookup.append((byte << 23) if 2 <= byte <= 252 else 0x7fc00000)
    repeat = [index // (16 if direct_bits else 32) * 4 for index in range(pairs if direct_bits else elements)]
    nz = []
    for index in range(pairs):
        half_tile, within = divmod(index, output_rows * 8)
        source_row, pair = divmod(within, 8)
        source_tile, row = divmod(source_row, 16)
        nz.append((source_tile * group_k * 256 + half_tile // 2 * 256
                   + row * 16 + half_tile % 2 * 8 + pair) * 4)
    payload = lookup + scale_lookup + repeat + nz
    if direct_bits:
        payload += [(0xffff if byte & 7 else 0) | (0xffff0000 if byte & 0x70 else 0) for byte in byte_order]
        payload += [0 if 2 <= byte <= 252 else 0x7fc07fc0 for byte in byte_order]
    if len(payload) != words:
        raise RuntimeError("immutable decode table layout differs from the native ABI")
    return torch.tensor([((word + 2**31) % 2**32) - 2**31 for word in payload], dtype=torch.int32)


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
        if self.group_k not in (1, 2, 4, 8):
            raise ValueError("native library reports an unsupported original K32 grouping")
        preload_query = getattr(self._library, "relic_online_tile_cube_preload_activation", None)
        self.preload_activation = False
        if preload_query is not None:
            preload_query.restype, preload_query.argtypes = ctypes.c_uint32, []
            preload = preload_query()
            if preload not in (0, 1):
                raise ValueError("native activation-L1 policy must be an explicit boolean")
            self.preload_activation = bool(preload)
        bits_query = getattr(self._library, "relic_online_tile_cube_direct_bf16_bits", None)
        self.direct_bf16_bits = False
        if bits_query is not None:
            bits_query.restype, bits_query.argtypes = ctypes.c_uint32, []
            bits = bits_query()
            if bits not in (0, 1):
                raise ValueError("native direct-bit decode policy must be an explicit boolean")
            self.direct_bf16_bits = bool(bits)
        self.output_rows = 16
        rows_query = getattr(self._library, "relic_online_tile_cube_output_rows", None)
        if rows_query is not None:
            rows_query.restype, rows_query.argtypes = ctypes.c_uint32, []
            self.output_rows = rows_query()
        if self.output_rows not in (16, 32) or self.output_rows * self.group_k > 128:
            raise ValueError("native output-row geometry must fit the bounded original N16/N32 consumer")
        self.ring_elements = self.output_rows * 32 * self.group_k
        self._launch = self._library.relic_online_tile_cube_launch
        self._launch.restype = None
        self._launch.argtypes = [ctypes.c_uint32, ctypes.c_void_p,
                                 *([ctypes.c_void_p] * 6), *([ctypes.c_uint32] * 7)]
        self.prepared_decode_tables = False
        self.table_words = 0
        self._launch_tables = None
        self._decode_tables = {}
        self._decode_table_versions = {}
        self._cpu_decode_table = None
        self._table_lock = RLock()
        table_query = getattr(self._library, "relic_online_tile_cube_prepared_tables", None)
        if table_query is not None:
            table_query.restype, table_query.argtypes = ctypes.c_uint32, []
            prepared = table_query()
            if prepared not in (0, 1):
                raise ValueError("native prepared-table policy must be an explicit boolean")
            self.prepared_decode_tables = bool(prepared)
        if self.prepared_decode_tables:
            words_query = getattr(self._library, "relic_online_tile_cube_table_words", None)
            self._launch_tables = getattr(self._library, "relic_online_tile_cube_launch_tables", None)
            if words_query is None or self._launch_tables is None:
                raise ValueError("native prepared tables require the complete checked table-launch ABI")
            words_query.restype, words_query.argtypes = ctypes.c_uint32, []
            self.table_words = words_query()
            if self.table_words != _decode_table_words(self.group_k, self.direct_bf16_bits, self.output_rows):
                raise ValueError("native immutable decode table extent differs from the expected ABI")
            self._launch_tables.restype = None
            self._launch_tables.argtypes = [ctypes.c_uint32, ctypes.c_void_p,
                                           *([ctypes.c_void_p] * 7), *([ctypes.c_uint32] * 7)]

    @torch.no_grad()
    def prepare_device_metadata(self, device):
        """Prepare immutable small tables before capture; never expose expanded weights."""
        if not self.prepared_decode_tables:
            return
        device = torch.device(device)
        if device.type != "npu":
            raise ValueError("native decode metadata must be prepared on the explicit NPU device")
        if device.index is None:
            device = torch.device("npu", torch.npu.current_device())
        with self._table_lock:
            table = self._decode_tables.get(device)
            if table is not None:
                try:
                    version = table._version
                except RuntimeError:
                    version = None
                if (table.shape != (self.table_words,) or table.dtype != torch.int32 or table.device != device
                        or not table.is_contiguous() or version != self._decode_table_versions[device]):
                    raise RuntimeError("an immutable native decode table changed after preparation")
                return
            capturing = getattr(torch.npu, "is_current_stream_capturing", None)
            if capturing is not None and capturing():
                raise RuntimeError("prepare native decode metadata before entering NPU graph capture")
            if self._cpu_decode_table is None:
                self._cpu_decode_table = _decode_table_cpu(self.group_k, self.direct_bf16_bits, self.output_rows)
            table = self._cpu_decode_table.to(device=device, non_blocking=False)
            try:
                version = table._version
            except RuntimeError:
                version = None
            self._decode_table_versions[device] = version
            self._decode_tables[device] = table

    @torch.no_grad()
    def __call__(self, x, b_tile, s_tile, selected, *, world, n_output=None,
                 certified_bf16_operands=False, scratch=None, accumulation=None,
                 output_dtype=torch.bfloat16):
        if output_dtype not in (torch.bfloat16, torch.float32):
            raise ValueError("native output must be BF16 or an explicitly requested FP32 TP partial")
        owned, tiles, blocks, groups, n = _validate(x, b_tile, s_tile, selected,
            world=world, n_output=n_output, group_k=1, out=None)
        if n != tiles * 16:
            raise ValueError("the native tile prototype currently requires the complete physical output extent")
        if n % self.output_rows:
            raise ValueError("native output rows must preserve a complete bounded N16/N32 tile")
        if any(value >= 2**32 for value in (owned, tiles, blocks, blocks * 32, groups, world, n)):
            raise ValueError("native tile geometry must fit the uint32 C ABI without truncation")
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
            if self.prepared_decode_tables:
                self.prepare_device_metadata(x.device)
                table = self._decode_tables[x.device]
                self._launch_tables(self.block_dim, stream, x.data_ptr(), b_tile.data_ptr(),
                    s_tile.data_ptr(), selected.data_ptr(), output.data_ptr(), stage.data_ptr(), table.data_ptr(),
                    tiles, blocks, groups, owned, world, n, int(x.shape[0] == 1))
            else:
                self._launch(self.block_dim, stream, x.data_ptr(), b_tile.data_ptr(),
                             s_tile.data_ptr(), selected.data_ptr(), output.data_ptr(), stage.data_ptr(),
                             tiles, blocks, groups, owned, world, n, int(x.shape[0] == 1))
        # TP down projections expose the original FP32 accumulator explicitly;
        # the caller must reduce those partials before the released BF16 store.
        # Default standalone/expert projections retain their existing narrowing.
        return output if output_dtype == torch.float32 else output.to(torch.bfloat16)

    def describe(self):
        return {"library": str(self.path), "sha256": self.sha256,
                "AIV_to_small_GM_ring_to_L1_to_AIC": True,
                "original_K32_blocks_per_Cube_tile": self.group_k,
                "native_output_rows_per_tile": self.output_rows,
                "source_checkpoint_tile_rows_unchanged": 16,
                "activation_L1_preload_requested": self.preload_activation,
                "direct_original_BF16_bit_decoder": self.direct_bf16_bits,
                "immutable_decode_tables_prepared_once_per_device": self.prepared_decode_tables,
                "decode_table_bytes_per_device": self.table_words * 4,
                "decode_table_devices_prepared": [str(device) for device in self._decode_tables],
                "decode_tables_contain_no_checkpoint_weight_or_routing_values": True,
                "activation_L1_capacity_bytes": 512 * 1024,
                "GM_ring_B1_and_MTE1_reuse_fences_preserved": True,
                "ring_weight_double_buffer_bytes_per_owner": self.ring_elements * 4,
                "global_expanded_weight_scratch_bytes": self.block_dim * self.ring_elements * 4,
                "whole_matrix_expanded_weight_scratch_bytes": 0,
                "native_full_K_accumulation_differs_from_exact_default": True}
