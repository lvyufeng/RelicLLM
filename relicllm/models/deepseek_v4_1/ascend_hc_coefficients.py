"""Explicit decode-only FP32 HC projection/RMS fusion, retaining nonlinear providers."""
from __future__ import annotations

import functools
from threading import RLock

import torch

from relic_core.kernels.hc_decode_coefficients import hc_project_rms

_OWNER = "_relic_HC_FP32_projection_owner"
_SOURCE_NAMES = ("hc_attn_fn", "hc_attn_scale", "hc_attn_base", "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base")


def _sources_match(block, sources):
    return all(getattr(block, name, None) is value for name, value in zip(_SOURCE_NAMES, sources))


def _stamp(tensor):
    try:
        version = tensor._version
    except RuntimeError:
        version = None
    return (tensor.untyped_storage()._cdata, tensor.storage_offset(), tuple(tensor.shape),
            tuple(tensor.stride()), tensor.dtype, tensor.device, version)


class FusedHCDecodeCoefficients:
    def __init__(self, model, *, backend="triton", certify_immutable_sources=False, k_tile=1024):
        if backend not in ("torch", "triton") or certify_immutable_sources is not True:
            raise ValueError("certify immutable original FP32 HC sources and choose an explicit backend")
        if isinstance(k_tile, bool) or not isinstance(k_tile, int) or k_tile not in (256, 512, 1024, 2048):
            raise ValueError("HC local K tile must fit the validated bounded geometry")
        from . import modules
        self.model, self.backend, self.k_tile = model, backend, k_tile
        self.split_provider = modules.hc_split_sinkhorn
        self.lock = RLock()
        self.entries, self.replacements = [], []
        self.installed = self.closed = False
        self.selected = self.passthrough = 0
        expected_device = "cpu" if backend == "torch" else "npu"
        if not model.layers or len({id(block) for block in model.layers}) != len(model.layers):
            raise ValueError("HC fusion requires distinct original blocks")
        for block in model.layers:
            if block.hc_mult != 4 or not callable(block.hc_mixes) or getattr(block, _OWNER, None) is not None:
                raise ValueError("HC projection requires unowned original four-copy blocks")
            sources = tuple(getattr(block, name) for name in _SOURCE_NAMES)
            if sources[0].ndim != 2 or sources[0].shape[1] % 4:
                raise ValueError("HC projection requires the original full four-copy FP32 coefficient extent")
            dim = sources[0].shape[1] // 4
            for fn, scale, base in (sources[:3], sources[3:]):
                if (fn.shape != (24, dim * 4) or scale.shape != (3,) or base.shape != (24,)
                        or any(value.dtype != torch.float32 or value.device.type != expected_device
                               or not value.is_contiguous() for value in (fn, scale, base))):
                    raise ValueError("HC projection must keep original FP32 parameter storage and shape")
            if len({value.device for value in sources}) != 1:
                raise ValueError("HC projection sources must share one device")
            self.entries.append((block, block.hc_mixes, "hc_mixes" in block.__dict__, block.__dict__.get("hc_mixes"),
                                 sources, tuple(_stamp(value) for value in sources)))

    def install(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("use a fresh HC coefficient controller after close")
            if self.installed:
                return self
            for block, original, _, _, sources, stamps in self.entries:
                if (block.hc_mixes != original or getattr(block, _OWNER, None) is not None
                        or not _sources_match(block, sources) or tuple(_stamp(value) for value in sources) != stamps):
                    raise RuntimeError("original HC method/source ownership changed before installation")
            for block, original, _, _, sources, stamps in self.entries:
                @functools.wraps(original)
                def mixes(x, fn, scale, base, _block=block, _original=original, _sources=sources, _stamps=stamps):
                    expected = "cpu" if self.backend == "torch" else "npu"
                    if x.ndim != 4 or x.shape[:3] != (1, 1, 4) or x.dtype != torch.bfloat16 or x.device.type != expected:
                        self.passthrough += 1
                        return _original(x, fn, scale, base)
                    if not _sources_match(_block, _sources) or tuple(_stamp(value) for value in _sources) != _stamps:
                        raise RuntimeError("an immutable HC FP32 source changed after installation")
                    if not any(fn is a and scale is b and base is c for a, b, c in (_sources[:3], _sources[3:])):
                        raise ValueError("HC fusion only accepts this block's original parameter triplet")
                    self.selected += 1
                    projected = hc_project_rms(x, fn, norm_eps=_block.norm_eps, backend=self.backend, k_tile=self.k_tile)
                    return self.split_provider(projected, scale, base, _block.hc_mult, _block.hc_sinkhorn_iters, _block.hc_eps)
                block.hc_mixes = mixes
                setattr(block, _OWNER, self)
                self.replacements.append((block, mixes))
            self.installed = True
            return self

    def close(self):
        with self.lock:
            if not self.installed:
                self.closed = True
                self.entries.clear()
                return
            for block, replacement in self.replacements:
                if block.hc_mixes is not replacement or getattr(block, _OWNER, None) is not self:
                    raise RuntimeError("another owner changed installed HC projection fusion")
            error = None
            try:
                if self.backend == "triton":
                    torch.npu.synchronize()
            except BaseException as caught:
                error = caught
            finally:
                for block, _, had_override, previous, _, _ in reversed(self.entries):
                    if had_override:
                        block.hc_mixes = previous
                    else:
                        delattr(block, "hc_mixes")
                    delattr(block, _OWNER)
                self.replacements.clear(); self.entries.clear()
                self.installed, self.closed = False, True
            if error is not None:
                raise error

    def describe(self):
        return {"installed": self.installed, "backend": self.backend, "K_tile": self.k_tile,
                "original_HC_FP32_weights_not_narrowed": True, "original_nonlinear_provider_retained": True,
                "FP32_projection_and_RMS_reduction_order_changed": True,
                "immutable_parameters_required_for_capture_and_replay": True,
                "selected_calls_eager_and_capture_not_device_replay": self.selected,
                "prefill_or_device_passthrough_calls": self.passthrough,
                "not_an_exact_default_or_measured_speedup_claim": True}

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
