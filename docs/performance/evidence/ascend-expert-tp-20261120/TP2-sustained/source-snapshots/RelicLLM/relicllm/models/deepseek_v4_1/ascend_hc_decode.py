"""Reversible decode HC collapse/residual-fold experiment.

Each fold becomes one device kernel, retaining original FP32 coefficients and
BF16 boundaries. Coefficient projection, normalization and Sinkhorn stay on the
existing path. No default or global function is patched. Prefill remains eager
and original. Numerical identity must be proved on current real consumers.
"""
from __future__ import annotations

import functools
from threading import RLock

import torch

from relic_core.kernels.hc_decode_fold import fused_hc_decode_pre, fused_hc_decode_post

_OWNER = "_ascend_decode_hc_fold_owner"


class FusedHCDecodeFold:
    def __init__(self, model, *, backend="triton"):
        if backend not in ("torch", "triton"):
            raise ValueError("choose an explicit HC CPU oracle or NPU backend")
        self.model, self.backend = model, backend
        self.device_type = "cpu" if backend == "torch" else "npu"
        self.lock = RLock()
        self.entries, self.replacements = [], []
        self.installed = self.closed = False
        self.stats = {"pre_selected": 0, "post_selected": 0, "pre_passthrough": 0, "post_passthrough": 0}
        self.coverage = {id(block): {"block": index, "pre": 0, "post": 0}
                         for index, block in enumerate(model.layers)}
        if len(self.coverage) != len(model.layers):
            raise ValueError("HC controller does not support repeated block objects")
        for block in model.layers:
            if getattr(block, "hc_mult", None) != 4 or not callable(block.hc_pre) or not callable(block.hc_post):
                raise ValueError("HC fold experiment requires the original four-copy block interface")
            if getattr(block, _OWNER, None) is not None:
                raise RuntimeError("block already has an HC decode-fold owner")
            self.entries.append((block, block.hc_pre, block.hc_post,
                                 "hc_pre" in block.__dict__, block.__dict__.get("hc_pre"),
                                 "hc_post" in block.__dict__, block.__dict__.get("hc_post")))
        if not self.entries:
            raise ValueError("HC decode experiment requires at least one block")

    def install(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("use a fresh HC fold controller after close")
            if self.installed:
                return self
            for block, original_pre, original_post, *_ in self.entries:
                if (getattr(block, _OWNER, None) is not None
                        or block.hc_pre != original_pre or block.hc_post != original_post):
                    raise RuntimeError("another owner changed HC methods before installation")
            for block, original_pre, original_post, *_ in self.entries:
                @functools.wraps(original_pre)
                def pre(x, pre_mix, _original=original_pre, _coverage=self.coverage[id(block)]):
                    if (x.ndim == 4 and x.shape[:3] == (1, 1, 4)
                            and x.dtype == torch.bfloat16 and x.device.type == self.device_type
                            and x.is_contiguous() and pre_mix.is_contiguous()):
                        self.stats["pre_selected"] += 1
                        _coverage["pre"] += 1
                        return fused_hc_decode_pre(x, pre_mix, backend=self.backend)
                    self.stats["pre_passthrough"] += 1
                    return _original(x, pre_mix)
                @functools.wraps(original_post)
                def post_fold(x, residual, post, comb, _original=original_post,
                              _coverage=self.coverage[id(block)]):
                    if (x.ndim == 3 and x.shape[:2] == (1, 1)
                            and residual.shape == (1, 1, 4, x.shape[-1])
                            and x.dtype == residual.dtype == torch.bfloat16
                            and x.device.type == self.device_type
                            and all(value.is_contiguous() for value in (x, residual, post, comb))):
                        self.stats["post_selected"] += 1
                        _coverage["post"] += 1
                        return fused_hc_decode_post(x, residual, post, comb, backend=self.backend)
                    self.stats["post_passthrough"] += 1
                    return _original(x, residual, post, comb)
                block.hc_pre, block.hc_post = pre, post_fold
                setattr(block, _OWNER, self)
                self.replacements.append((block, pre, post_fold))
            self.installed = True
            return self

    def close(self):
        with self.lock:
            if not self.installed:
                return
            for block, pre, post in self.replacements:
                if (getattr(block, _OWNER, None) is not self or block.hc_pre is not pre
                        or block.hc_post is not post):
                    raise RuntimeError("another owner changed installed HC folds")
            error = None
            try:
                if self.backend == "triton":
                    torch.npu.synchronize()
            except BaseException as caught:
                error = caught
            finally:
                for block, _, _, had_pre, override_pre, had_post, override_post in reversed(self.entries):
                    if had_pre:
                        block.hc_pre = override_pre
                    else:
                        delattr(block, "hc_pre")
                    if had_post:
                        block.hc_post = override_post
                    else:
                        delattr(block, "hc_post")
                    delattr(block, _OWNER)
                self.replacements.clear()
                self.installed, self.closed = False, True
            if error is not None:
                raise error

    def describe(self):
        return {"installed": self.installed, "backend": self.backend, "blocks": len(self.entries),
                "fused_scope": "HC collapse and residual fold only",
                "original_FP32_coefficients_and_BF16_boundaries": True,
                "projection_normalization_and_Sinkhorn_unchanged": True,
                "per_block_coverage": list(self.coverage.values()),
                "selected_counts_include_eager_and_capture_not_device_replay": True, **self.stats}

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
