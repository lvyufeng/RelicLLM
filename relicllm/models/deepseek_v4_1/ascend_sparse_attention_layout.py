"""Explicit, reversible B1 sparse-attention layout experiment for one model.

Only the attention module's existing sparse_attn reference is intercepted, and
only inside an owned attention forward on the installation thread. Other models,
prefill, unsupported metadata and other threads keep their original provider.
Install before recording attention graphs: Python counters count eager/capture
calls, not subsequent device replays. No serving default or kernel registry is
changed by importing this module.
"""
from __future__ import annotations

import functools
import math
from threading import RLock, get_ident

import torch

_ACTIVE = None
_LOCK = RLock()
_OWNER = "_relic_sparse_attention_layout_owner"


class SparseAttentionLayoutBinding:
    """One explicitly certified model owner; no cross-call operand or output cache."""

    def __init__(self, model, *, backend="triton", certified_valid_indices=False,
                 max_workspace_bytes=64 * 2**20):
        if backend not in ("torch", "triton"):
            raise ValueError("choose an explicit CPU oracle or NPU sparse-layout backend")
        if not isinstance(certified_valid_indices, bool) or (backend == "triton" and not certified_valid_indices):
            raise ValueError("NPU sparse layout requires explicit certification of all non-negative index bounds")
        if type(max_workspace_bytes) is not int or max_workspace_bytes <= 0:
            raise ValueError("sparse layout requires an explicit positive workspace budget")
        self.attentions = [block.attn for block in model.layers]
        if not self.attentions or len({id(value) for value in self.attentions}) != len(self.attentions):
            raise ValueError("sparse layout requires a nonempty model with distinct attention instances")
        if any(not callable(getattr(value, "forward", None)) for value in self.attentions):
            raise ValueError("every owned attention must expose its original forward")
        self.model, self.backend = model, backend
        self.device_type = "cpu" if backend == "torch" else "npu"
        self.certified_valid_indices, self.max_workspace_bytes = certified_valid_indices, max_workspace_bytes
        self.installed = self.closed = False
        self.owner_thread = self.current_attention = None
        self.attention_module = self.original_sparse = self.replacement_sparse = None
        self.replacements, self.devices = [], {}
        self._operator = self._workspace_bytes = None
        self.stats = {"owned_B1_scopes": 0, "selected_B1_calls": 0, "outside_scope_bypasses": 0,
                      "unsupported_bypasses": 0, "budget_bypasses": 0, "peak_declared_workspace_bytes": 0}

    def _resolve_operator(self):
        if self._operator is None:
            from relic_core.kernels.ascend_sparse_attention_layout import sparse_attn_layout, layout_workspace_bytes

            self._operator, self._workspace_bytes = sparse_attn_layout, layout_workspace_bytes
        return self._operator, self._workspace_bytes

    def _remember_device(self, device):
        if getattr(device, "type", None) == "npu":
            self.devices[str(device)] = device

    def _eligible_input(self, x):
        return (isinstance(x, torch.Tensor) and x.device.type == self.device_type
                and x.dtype == torch.bfloat16 and x.ndim == 3 and tuple(x.shape[:2]) == (1, 1)
                and x.is_contiguous() and not x.requires_grad)

    def _eligible_sparse(self, q, kv, sink, indices, scale):
        values = (q, kv, sink, indices)
        if not all(isinstance(value, torch.Tensor) for value in values):
            return False
        if (q.device.type != self.device_type or any(value.device != q.device for value in values)
                or any(value.requires_grad for value in (q, kv, indices))
                or (sink.requires_grad and torch.is_grad_enabled())
                or q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16
                or q.ndim != 4 or tuple(q.shape[:2]) != (1, 1) or q.shape[-1] != 512 or q.shape[2] < 1
                or kv.ndim != 3 or kv.shape[0] != 1 or kv.shape[-1] != 512 or not 1 <= kv.shape[1] <= 4096
                or indices.dtype != torch.int32 or indices.ndim != 3 or tuple(indices.shape[:2]) != (1, 1)
                or not 0 <= indices.shape[-1] <= 512
                or sink.dtype not in (torch.bfloat16, torch.float32)
                or tuple(sink.shape) not in ((q.shape[2],), (1, q.shape[2]))
                or any(not value.is_contiguous() for value in values)
                or isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale)):
            return False
        return True

    def _sparse(self, q, kv, sink, indices, scale):
        with _LOCK:
            if (not self.installed or self.closed or self.current_attention is None
                    or get_ident() != self.owner_thread):
                self.stats["outside_scope_bypasses"] += 1
                original = self.original_sparse
                selected = False
            elif not self._eligible_sparse(q, kv, sink, indices, scale):
                self.stats["unsupported_bypasses"] += 1
                original = self.original_sparse
                selected = False
            else:
                selected = True
        if not selected:
            return original(q, kv, sink, indices, scale)
        operator, workspace_bytes = self._resolve_operator()
        required = workspace_bytes(indices.shape[-1], heads=q.shape[2])
        if required > self.max_workspace_bytes:
            with _LOCK:
                self.stats["budget_bypasses"] += 1
            return self.original_sparse(q, kv, sink, indices, scale)
        with _LOCK:
            self.stats["selected_B1_calls"] += 1
            self.stats["peak_declared_workspace_bytes"] = max(self.stats["peak_declared_workspace_bytes"], required)
            self._remember_device(q.device)
        # A bounds certificate comes from the original index-construction caller,
        # not shape metadata. No device reduction or D2H check is inserted here.
        return operator(q, kv, indices, sink, scale, backend=self.backend,
                        certified_valid_indices=self.certified_valid_indices,
                        max_workspace_bytes=self.max_workspace_bytes)

    def install(self):
        global _ACTIVE
        from . import attention as attention_module

        with _LOCK:
            if self.installed or self.closed or _ACTIVE is not None:
                raise RuntimeError("use a fresh sparse-layout owner; another model may already own the hook")
            for value in self.attentions:
                if getattr(value, _OWNER, None) is not None:
                    raise RuntimeError("an attention forward already has a sparse-layout owner")
            original_sparse = attention_module.sparse_attn
            if getattr(original_sparse, _OWNER, None) is not None:
                raise RuntimeError("the attention sparse provider already has an owner")
            self.attention_module, self.original_sparse = attention_module, original_sparse
            self.owner_thread = get_ident()

            @functools.wraps(original_sparse)
            def sparse(q, kv, attn_sink, topk_idxs, softmax_scale):
                return self._sparse(q, kv, attn_sink, topk_idxs, softmax_scale)

            setattr(sparse, _OWNER, self)
            self.replacement_sparse = sparse
            try:
                for value in self.attentions:
                    original = value.forward
                    had_override, override = "forward" in value.__dict__, value.__dict__.get("forward")

                    @functools.wraps(original)
                    def forward(x, *args, _original=original, _attention=value, **kwargs):
                        with _LOCK:
                            owned = (self.installed and not self.closed and get_ident() == self.owner_thread
                                     and self._eligible_input(x))
                            if owned:
                                if self.current_attention is not None:
                                    raise RuntimeError("nested sparse-layout attention scopes are not supported")
                                self.current_attention = _attention
                                self.stats["owned_B1_scopes"] += 1
                                self._remember_device(x.device)
                        if not owned:
                            return _original(x, *args, **kwargs)
                        try:
                            return _original(x, *args, **kwargs)
                        finally:
                            with _LOCK:
                                self.current_attention = None

                    value.forward = forward
                    setattr(value, _OWNER, self)
                    self.replacements.append((value, forward, had_override, override))
                    self._remember_device(getattr(getattr(value, "attn_sink", None), "device", None))
                attention_module.sparse_attn = sparse
                self.installed, _ACTIVE = True, self
            except BaseException:
                for value, _, had_override, override in reversed(self.replacements):
                    if had_override:
                        value.forward = override
                    else:
                        delattr(value, "forward")
                    delattr(value, _OWNER)
                self.replacements.clear()
                self.owner_thread = None
                raise
            return self

    def close(self):
        global _ACTIVE
        with _LOCK:
            if not self.installed:
                if not self.closed:
                    self.closed = True
                return
            if get_ident() != self.owner_thread or self.current_attention is not None:
                raise RuntimeError("close sparse layout on its installation thread after the current scope ends")
            if (_ACTIVE is not self or self.attention_module.sparse_attn is not self.replacement_sparse):
                raise RuntimeError("another owner changed the attention sparse hook")
            for value, replacement, *_ in self.replacements:
                if value.forward is not replacement or getattr(value, _OWNER, None) is not self:
                    raise RuntimeError("another owner changed an attention forward")
            if self.backend == "triton":
                # Do not put unpatch/free in a finally block: a failed fence leaves
                # every registered function and core/operator reference owned for
                # retry, including graph-referenced per-call allocations.
                for device in self.devices.values():
                    torch.npu.synchronize(device)
            self.attention_module.sparse_attn = self.original_sparse
            for value, _, had_override, override in reversed(self.replacements):
                if had_override:
                    value.forward = override
                else:
                    delattr(value, "forward")
                delattr(value, _OWNER)
            self.replacements.clear()
            self.devices.clear()
            self._operator = self._workspace_bytes = None
            self.installed, self.closed, self.owner_thread, _ACTIVE = False, True, None, None

    def describe(self):
        return {"installed": self.installed, "backend": self.backend,
                "scope": "owned B1 attention forward on the installation thread only",
                "certified_valid_indices": self.certified_valid_indices,
                "index_certificate_is_caller_attestation_not_metadata_or_value_validation": True,
                "max_workspace_bytes": self.max_workspace_bytes,
                "workspace_budget_is_declared_tensor_estimate_not_allocator_or_graph_pool_peak": True,
                "core_owns_per_call_layout_scratch": True,
                "source_inputs_quantization_and_index_values_unchanged": True,
                "no_operand_or_output_cache_between_calls": True,
                "defaults_and_other_models_keep_original_provider": True,
                "install_before_attention_graph_recording_required": True,
                "caller_must_retire_recordings_before_restoring_default_execution": True,
                "python_counters_include_capture_not_device_replay": True,
                "numerical_or_performance_acceptance_claimed": False,
                "owned_attention_modules": len(self.attentions), **self.stats}

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
