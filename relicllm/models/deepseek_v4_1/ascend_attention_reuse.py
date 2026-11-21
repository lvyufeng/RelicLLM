"""Reversible original FP8 activation preparation reuse inside one attention call.

Only immutable attention input/normalized-QR values may be reused. The private
scope retains input storage, clears after every call/capture, and requires its
caller to certify immutability for versionless inference tensors. Nothing is
cached between forward calls or graph recording rounds. Code and E8M0 outputs
are the original quantizer's outputs; native matmul/reduction is unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import functools
from threading import RLock, get_ident

import torch

_ACTIVE = None
_LOCK = RLock()
_OWNER = "_relic_attention_fp8_reuse_owner"


def _stamp(value):
    try:
        version = value._version
    except RuntimeError:
        version = None
    return (value.data_ptr(), value.untyped_storage()._cdata, value.storage_offset(),
            tuple(value.shape), tuple(value.stride()), value.dtype, value.device, version)


@dataclass
class _Prepared:
    source: torch.Tensor
    stamp: tuple
    codes: torch.Tensor
    scales: torch.Tensor
    codes_stamp: tuple
    scales_stamp: tuple


class AttentionFP8Reuse:
    """One thread-scoped quantizer owner; CPU/prefill/default providers stay original."""

    def __init__(self, model=None, *, backend="triton", certify_immutable_inputs=False):
        if backend not in ("torch", "triton") or certify_immutable_inputs is not True:
            raise ValueError("choose an explicit backend and certify immutable attention operands")
        self.model, self.backend = model, backend
        self.device_type = "cpu" if backend == "torch" else "npu"
        self.installed = self.closed = False
        self.quant_module = self.original_quant = self.replacement_quant = None
        self.owner_thread = None
        self.prepared = None
        self.replacements = []
        self.attentions = [] if model is None else [block.attn for block in model.layers]
        if len({id(value) for value in self.attentions}) != len(self.attentions):
            raise ValueError("attention reuse requires distinct attention module instances")
        self.stats = {"scopes": 0, "quantizer_calls": 0, "reuses": 0, "outside_scope_bypasses": 0,
                      "unsupported_bypasses": 0, "peak_live_preparations": 0}

    @contextmanager
    def scope(self):
        with _LOCK:
            if not self.installed or self.closed:
                raise RuntimeError("attention activation reuse must be installed before opening a scope")
            if self.prepared is not None:
                raise RuntimeError("nested attention activation scopes are not supported")
            self.owner_thread, self.prepared = get_ident(), {}
            self.stats["scopes"] += 1
        try:
            yield
        finally:
            with _LOCK:
                self.prepared.clear()
                self.prepared = self.owner_thread = None

    def install(self):
        global _ACTIVE
        from relic_core.kernels import block_fp8_quant
        with _LOCK:
            if self.closed or self.installed or _ACTIVE is not None:
                raise RuntimeError("attention activation reuse already has an owner or this controller is closed")
            for attention in self.attentions:
                if getattr(attention, _OWNER, None) is not None:
                    raise RuntimeError("attention forward already has an activation reuse owner")
            self.quant_module = block_fp8_quant
            self.original_quant = block_fp8_quant.quantize_block_fp8
            def quantize(x, backend="auto", out=None):
                with _LOCK:
                    eligible = (x.device.type == self.device_type and x.dtype in (torch.bfloat16, torch.float32)
                                and x.ndim == 2 and x.shape[0] == 1 and x.shape[-1] > 0 and x.shape[-1] % 32 == 0
                                and x.is_contiguous() and not x.requires_grad and out is None
                                and backend in (self.backend, "auto"))
                    if self.prepared is None or self.owner_thread != get_ident():
                        self.stats["outside_scope_bypasses"] += 1
                        return self.original_quant(x, backend=backend, out=out)
                    if not eligible:
                        self.stats["unsupported_bypasses"] += 1
                        return self.original_quant(x, backend=backend, out=out)
                    stamp = _stamp(x)
                    if stamp in self.prepared:
                        self.stats["reuses"] += 1
                        cached = self.prepared[stamp]
                        if (_stamp(cached.source) != cached.stamp
                                or _stamp(cached.codes) != cached.codes_stamp
                                or _stamp(cached.scales) != cached.scales_stamp):
                            raise RuntimeError("retained attention preparation changed during its immutable scope")
                        return cached.codes, cached.scales
                    if len(self.prepared) >= 8:
                        self.stats["unsupported_bypasses"] += 1
                        return self.original_quant(x, backend=backend, out=out)
                    codes, scales = self.original_quant(x, backend=backend, out=out)
                    self.prepared[stamp] = _Prepared(x, stamp, codes, scales, _stamp(codes), _stamp(scales))
                    self.stats["quantizer_calls"] += 1
                    self.stats["peak_live_preparations"] = max(self.stats["peak_live_preparations"], len(self.prepared))
                    return codes, scales
            self.replacement_quant = quantize
            for attention in self.attentions:
                original = attention.forward
                had_override = "forward" in attention.__dict__
                override = attention.__dict__.get("forward")
                @functools.wraps(original)
                def forward(x, *args, _original=original, **kwargs):
                    if (x.device.type != self.device_type or x.ndim != 3 or x.shape[:2] != (1, 1)
                            or x.dtype != torch.bfloat16):
                        return _original(x, *args, **kwargs)
                    with self.scope():
                        return _original(x, *args, **kwargs)
                attention.forward = forward
                setattr(attention, _OWNER, self)
                self.replacements.append((attention, forward, had_override, override))
            block_fp8_quant.quantize_block_fp8 = quantize
            self.installed, _ACTIVE = True, self
            return self

    def close(self):
        global _ACTIVE
        with _LOCK:
            if not self.installed:
                return
            if self.prepared is not None:
                raise RuntimeError("close the current attention activation scope first")
            if self.quant_module.quantize_block_fp8 is not self.replacement_quant or _ACTIVE is not self:
                raise RuntimeError("another owner changed the quantizer while attention reuse was installed")
            for attention, replacement, *_ in self.replacements:
                if attention.forward is not replacement or getattr(attention, _OWNER, None) is not self:
                    raise RuntimeError("another owner changed the attention reuse forward")
            self.quant_module.quantize_block_fp8 = self.original_quant
            for attention, _, had_override, override in reversed(self.replacements):
                if had_override:
                    attention.forward = override
                else:
                    delattr(attention, "forward")
                delattr(attention, _OWNER)
            self.replacements.clear()
            self.installed, self.closed, _ACTIVE = False, True, None

    def describe(self):
        return {"installed": self.installed, "backend": self.backend,
                "scope": "one immutable batch-one attention invocation",
                "original_E4M3_E8M0_quantization_unchanged": True,
                "no_cache_between_forwards_or_recording_rounds": True,
                "versionless_inference_inputs_require_explicit_caller_immutability": True,
                "provider_counts_include_capture_not_device_replay": True,
                "max_live_preparations_per_scope": 8,
                "supported_attention_modules": len(self.attentions), **self.stats}

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
