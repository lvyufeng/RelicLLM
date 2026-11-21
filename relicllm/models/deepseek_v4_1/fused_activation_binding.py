"""Opt-in exact fused-activation reuse at the original NPU provider seams.

The original E4M3 codes and E8M0 scales are still returned. Their owner also
retains the BF16 value produced in the same exact quantize/decode pass, so native
Cube consumers can reuse it instead of launching a second decoder. CPU Engram
always calls the original CPU quantizer/decoder. Importing this module changes
nothing; installation is explicit and reversible for one private experiment.
"""
from __future__ import annotations

from dataclasses import dataclass
import threading

import torch

_LOCK = threading.RLock()
_ACTIVE = None
_ATTRIBUTE = "_relic_fused_original_fp8_activation"


def _on_npu(tensor):
    return isinstance(tensor, torch.Tensor) and tensor.device.type == "npu"


def _token(tensor):
    if torch.is_inference(tensor):
        version = None
    else:
        try:
            version = tensor._version
        except RuntimeError:
            # Graph replays write both code buffers and the decoded result in the
            # same recorded kernel; unrelated mutation is outside this private owner.
            version = None
    return (tensor.data_ptr(), tensor.untyped_storage()._cdata,
            tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.dtype, tensor.device, version)


@dataclass(frozen=True)
class _Decoded:
    codes_token: tuple
    scales_token: tuple
    value: torch.Tensor


def remember_original_decode(codes, scales, value):
    """Attach an explicitly fused result to its original byte outputs, with identity guards.

    This does not install a provider or add a global cache. The caller owns all
    three outputs of the same kernel; the installed decoder can reuse the BF16
    output after checking the original code/scale tensor identities.
    """
    if (codes.dtype != torch.uint8 or scales.dtype != torch.uint8
            or value.dtype != torch.bfloat16 or tuple(value.shape) != tuple(codes.shape)
            or codes.ndim < 1 or codes.shape[-1] % 32
            or tuple(scales.shape) != (*codes.shape[:-1], codes.shape[-1] // 32)
            or any(t.device != codes.device or not t.is_contiguous() for t in (codes, scales, value))):
        raise ValueError("fused outputs must be matching contiguous original FP8/E8M0 bytes and BF16")
    setattr(codes, _ATTRIBUTE, _Decoded(_token(codes), _token(scales), value))


class FusedActivationBinding:
    """Install an NPU-only exact quantize/decode provider and restore it on close."""

    def __init__(self, *, operator=None):
        if operator is not None and not callable(operator):
            raise TypeError("the explicit fused activation operator must be callable")
        self.operator = operator
        self.installed = False
        self.fused_calls = self.decode_hits = self.stale_misses = self.CPU_bypasses = 0
        self._quant = self._decode = None
        self._original_quantize = self._original_decode = None
        self._quantized = self._decoded = None

    def install(self):
        global _ACTIVE
        from relic_core.kernels import block_fp8_quant, unpack
        from relic_core.kernels.fused_fp8_activation_decode import quantize_decode_fp8
        fused_operator = quantize_decode_fp8 if self.operator is None else self.operator
        with _LOCK:
            if self.installed or _ACTIVE is not None:
                raise RuntimeError("another fused activation binding is already installed")
            self._quant, self._decode = block_fp8_quant, unpack
            self._original_quantize = block_fp8_quant.quantize_block_fp8
            self._original_decode = unpack.unpack_fp8_to_bf16

            def quantized(x, backend="auto", out=None):
                if not _on_npu(x) or out is not None or backend == "torch":
                    if isinstance(x, torch.Tensor) and x.device.type == "cpu":
                        self.CPU_bypasses += 1
                    return self._original_quantize(x, backend=backend, out=out)
                value, codes, scales = fused_operator(x, backend=backend, return_codes=True)
                setattr(codes, _ATTRIBUTE, _Decoded(_token(codes), _token(scales), value))
                self.fused_calls += 1
                return codes, scales

            def decoded(weight, scale=None, *, block_shape=(32, 32), backend="auto", out=None):
                cached = getattr(weight, _ATTRIBUTE, None)
                if cached is not None and _on_npu(weight) and scale is not None and block_shape == (1, 32) and backend in {"auto", "triton"}:
                    if cached.codes_token == _token(weight) and cached.scales_token == _token(scale):
                        value = cached.value
                        if value.dtype != torch.bfloat16 or value.device != weight.device or tuple(value.shape) != tuple(weight.shape) or not value.is_contiguous():
                            raise RuntimeError("the fused activation owner retained invalid BF16 metadata")
                        self.decode_hits += 1
                        if out is None:
                            return value
                        result = unpack._output(out, value.shape, torch.bfloat16, value.device, (weight, scale))
                        result.copy_(value)
                        return result
                    self.stale_misses += 1
                return self._original_decode(weight, scale, block_shape=block_shape, backend=backend, out=out)

            self._quantized, self._decoded = quantized, decoded
            block_fp8_quant.quantize_block_fp8 = quantized
            unpack.unpack_fp8_to_bf16 = decoded
            self.installed = True
            _ACTIVE = self
        return self

    def close(self):
        global _ACTIVE
        with _LOCK:
            if not self.installed:
                return
            if self._quant.quantize_block_fp8 is not self._quantized or self._decode.unpack_fp8_to_bf16 is not self._decoded:
                raise RuntimeError("another owner changed the fused activation provider while installed")
            self._quant.quantize_block_fp8 = self._original_quantize
            self._decode.unpack_fp8_to_bf16 = self._original_decode
            self.installed = False
            _ACTIVE = None

    def describe(self):
        return {"installed": self.installed, "fused_calls": self.fused_calls,
                "decoded_activation_reuses": self.decode_hits, "stale_misses": self.stale_misses,
                "CPU_bypasses": self.CPU_bypasses, "original_E4M3_E8M0_contract_unchanged": True,
                "complete_Engram_CPU": True, "global_registry_of_activations": False,
                "operator": "quantize_decode_fp8" if self.operator is None else getattr(self.operator, "__name__", type(self.operator).__name__)}

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
