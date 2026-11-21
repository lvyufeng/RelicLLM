"""Explicit host-table/device-compute Engram experiment, never a default.

Original table objects and original CPU parameter/source-code metadata stay
untouched. Only gathered BF16 rows cross H2D; the hidden stream never goes D2H.
WKV is output-channel TP with native full-K accumulation, gathered as raw BF16
bytes before the original FP32 gate sequence. Prefill and wider inputs retain
the original host implementation. Each decode graph owns persistent inputs;
pinned host reuse is fenced and exported graph results are cloned on device.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import functools
from threading import RLock
import time

import torch
import torch.distributed as dist

from relic_core.kernels.original_fp8_engram import (
    NUMERICAL_CONTRACT, original_engram_gate, original_fp8_engram_projection,
)
from . import mma, modules
from .host_engram_prefetch import _key

_OWNER = "_host_table_device_engram_owner"


def _tensor_stamp(value):
    if value is None:
        return None
    try:
        version = value._version
    except RuntimeError:
        version = None
    return (id(value), value.data_ptr(), value.untyped_storage()._cdata, version,
            tuple(value.shape), tuple(value.stride()), value.storage_offset(), value.dtype, value.device)


def _source_stamp(engram):
    table = engram.embed
    tensors = (engram.wkv.weight, mma.attached_codes(engram.wkv.weight),
               mma.attached_scale(engram.wkv.weight), engram.q_weight, engram.k_weight)
    table_tensors = tuple(getattr(table, name, None) for name in ("codes", "scales", "weight", "scale"))
    table_metadata = tuple(getattr(table, name, None) for name in
                           ("layer_id", "weight_key", "scale_key", "block_size", "head_dim",
                            "rows_total", "code_dtype", "scale_dtype"))
    return (tuple(_tensor_stamp(value) for value in tensors), id(table),
            tuple(_tensor_stamp(value) for value in table_tensors if isinstance(value, torch.Tensor)),
            table_metadata, id(getattr(table, "checkpoint", None)),
            engram.layer_id, engram.layer_hash_index,
            engram._quantized, engram.dim, engram.hc_mult, engram.eps, engram.clamp_value)


def _check_table_host(table):
    from .loader import CheckpointEngramTable
    if not isinstance(table, (CheckpointEngramTable, modules.ResidentEngramTable)):
        raise ValueError("certify a checkpoint-backed or explicitly host-resident Engram table")
    for name in ("codes", "scales", "weight", "scale"):
        value = getattr(table, name, None)
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            raise ValueError("original Engram tables must remain on host")


def _certify_weight(engram, start, stop):
    """Verify only this output shard against original codes/grid, on CPU."""
    from . import cube_experiment
    weight = engram.wkv.weight
    if not engram._quantized:
        return {"quantized": False, "source": "original BF16 parameter"}
    codes, scales = mma.attached_codes(weight), mma.attached_scale(weight)
    if codes is None or scales is None or codes.device.type != "cpu" or scales.device.type != "cpu":
        raise ValueError("original CPU FP8 source codes and E8M0 grid are required")
    if start % 32 or stop % 32:
        raise ValueError("output TP must preserve original FP8 32-row scale blocks")
    reports = []
    for first in range(start, stop, 256):
        last = min(first + 256, stop)
        block_rows = 1 if scales.shape[0] == weight.shape[0] else 32
        source_codes = codes[first:last].view(torch.uint8).contiguous()
        source_scales = scales[first // block_rows:(last + block_rows - 1) // block_rows].contiguous()
        report = cube_experiment.check_bf16_representable(
            source_codes, source_scales, kind="fp8", name=f"Engram{engram.layer_id}/WKV/{first}:{last}")
        decoded = weight[first:last].detach().contiguous()
        mma.attach_source_codes(decoded, source_codes.view(torch.float8_e4m3fn), source_scales)
        cube_experiment.verify_weight_decode_identity(decoded)
        reports.append(report)
    return {"quantized": True, "original_decode_identity": True,
            "elements": sum(row["elements"] for row in reports), "chunks": len(reports)}


@dataclass
class _Entry:
    engram: object
    original_forward: object
    stamp: object
    weight_kn: torch.Tensor
    channel_weight: torch.Tensor
    x: torch.Tensor
    lookup: torch.Tensor
    mask: torch.Tensor
    gathered: torch.Tensor
    pinned_lookup: torch.Tensor
    pinned_mask: torch.Tensor
    source_report: dict
    hash_cols: int
    had_forward_override: bool = False
    forward_override: object = None
    graph: object = None
    output: object = None
    graph_output: object = None
    transfer_done: object = None
    transfer_in_flight: bool = False


class HostTableDeviceEngram:
    """One reversible batch-one controller with original host table residency.

    ``backend='torch'`` is a CPU logic/numerical oracle, not NPU fallback or a
    performance claim. Normal distributed use borrows the existing TP group.
    A custom gather callback is only for explicit tests or an existing group.
    """

    def __init__(self, model, *, device=None, backend="triton", capture_graphs=True,
                 rank=None, world=None, gather=None, lookup_workers=2,
                 max_device_bytes=512 * 2**20):
        if backend not in ("torch", "triton"):
            raise ValueError("choose explicit CPU oracle or NPU backend")
        if isinstance(lookup_workers, bool) or not isinstance(lookup_workers, int) or lookup_workers < 1:
            raise ValueError("lookup_workers must be a positive integer")
        if isinstance(max_device_bytes, bool) or not isinstance(max_device_bytes, int) or max_device_bytes <= 0:
            raise ValueError("the explicit device budget must be a positive integer")
        self.device = torch.device(device if device is not None else next(model.parameters()).device)
        if self.device.type != ("cpu" if backend == "torch" else "npu"):
            raise ValueError("controller device must match its explicit backend")
        if backend == "torch" and capture_graphs:
            raise ValueError("CPU oracle cannot claim NPU graph capture")
        initialized = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if rank is None and initialized else (0 if rank is None else rank)
        self.world = dist.get_world_size() if world is None and initialized else (1 if world is None else world)
        if (isinstance(self.rank, bool) or isinstance(self.world, bool)
                or not isinstance(self.rank, int) or not isinstance(self.world, int)
                or self.world <= 0 or not 0 <= self.rank < self.world):
            raise ValueError("controller requires a valid TP rank/world")
        if self.world > 1 and gather is None and not initialized:
            raise ValueError("multi-rank WKV requires an existing process group or explicit test gather")
        if gather is not None and not callable(gather):
            raise ValueError("the explicit TP gather must be callable")
        if (initialized and gather is None
                and (self.rank != dist.get_rank() or self.world != dist.get_world_size())):
            raise ValueError("default-group TP rank/world must match the initialized process group")
        self.model, self.backend, self.capture_graphs = model, backend, capture_graphs
        self.lookup_workers, self.max_device_bytes = lookup_workers, max_device_bytes
        self.gather = gather if gather is not None else self._gather
        self.lock, self.owner_stream = RLock(), None
        self.installed = self.closed = False
        self.executor, self.pending = None, {}
        self.replacements = []
        self.original_model_forward = model.forward
        self.model_had_forward_override = "forward" in model.__dict__
        self.model_forward_override = model.__dict__.get("forward")
        self.graph_api = self.graph_pool = None
        self.stats = {"decode_calls": 0, "passthrough_calls": 0, "prefetched_lookups": 0,
                      "direct_lookups": 0, "lookup_wait_s": 0.0, "pin_reuse_wait_s": 0.0,
                      "lookup_H2D_bytes": 0, "mask_H2D_bytes": 0,
                      "hidden_state_D2H_bytes": 0, "graph_replays": 0}
        self.entries = {}
        engr = [layer.engram for layer in model.layers if layer.engram is not None]
        if not engr:
            raise ValueError("a device-Engram controller requires actual Engram layers")
        if len({id(engram) for engram in engr}) != len(engr):
            raise ValueError("device-Engram ownership requires distinct layer objects")
        projected = 0
        layouts = []
        for engram in engr:
            _check_table_host(engram.embed)
            if getattr(engram, _OWNER, None) is not None:
                raise RuntimeError("Engram already has a device-compute owner")
            weight = engram.wkv.weight
            if (weight.device.type != "cpu" or weight.dtype != torch.bfloat16 or weight.ndim != 2
                    or weight.shape[0] != (engram.hc_mult + 1) * engram.dim
                    or weight.shape[1] <= 0 or weight.shape[1] % 32
                    or weight.shape[0] % self.world):
                raise ValueError("original CPU WKV geometry is incompatible with output-channel TP")
            local = weight.shape[0] // self.world
            if engram._quantized and local % 32:
                raise ValueError("output-channel TP must not split original FP8 scale blocks")
            if (engram.q_weight.device.type != "cpu" or engram.k_weight.device.type != "cpu"
                    or engram.q_weight.dtype != torch.bfloat16 or engram.k_weight.dtype != torch.bfloat16
                    or engram.q_weight.shape != (engram.hc_mult, engram.dim)
                    or engram.k_weight.shape != engram.q_weight.shape):
                raise ValueError("original CPU gate weights must match the released Engram geometry")
            n, k = weight.shape
            head_dim = getattr(engram.embed, "head_dim", None)
            if head_dim is None and isinstance(engram.embed, modules.ResidentEngramTable):
                head_dim = engram.embed.weight.shape[-1]
            if not isinstance(head_dim, int) or head_dim <= 0 or k % head_dim:
                raise ValueError("the original table head width must divide WKV input width")
            # Final resident weight and explicit persistent inputs; graph-pool
            # intermediates and allocator caches are separately excluded.
            projected += 2 * local * k + 4 * engram.hc_mult * engram.dim
            projected += 2 * engram.hc_mult * engram.dim + 2 * k + 1 + 2 * n
            layouts.append((engram, local, n, k, k // head_dim))
        if projected > max_device_bytes:
            raise ValueError(f"device-Engram explicit storage needs {projected} bytes, above budget")
        self.explicit_device_bytes = projected
        for engram, local, n, k, hash_cols in layouts:
            start, stop = self.rank * local, (self.rank + 1) * local
            source_report = _certify_weight(engram, start, stop)
            weight_kn = engram.wkv.weight[start:stop].detach().t().contiguous().to(self.device)
            channel = (engram.q_weight.detach().float() * engram.k_weight.detach().float()).to(self.device)
            pin = self.backend == "triton"
            entry = _Entry(engram, engram.forward, _source_stamp(engram), weight_kn, channel,
                           torch.empty((1, 1, engram.hc_mult, engram.dim), dtype=torch.bfloat16, device=self.device),
                           torch.empty((1, k), dtype=torch.bfloat16, device=self.device),
                           torch.ones((1, 1), dtype=torch.bool, device=self.device),
                           torch.empty((1, n), dtype=torch.bfloat16, device=self.device),
                           torch.empty((1, k), dtype=torch.bfloat16, device="cpu", pin_memory=pin),
                           torch.ones((1, 1), dtype=torch.bool, device="cpu", pin_memory=pin), source_report, hash_cols)
            entry.had_forward_override = "forward" in engram.__dict__
            entry.forward_override = engram.__dict__.get("forward")
            if pin:
                entry.transfer_done = torch.npu.Event()
            self.entries[id(engram)] = entry

    def _gather(self, output, local):
        dist.all_gather_into_tensor(output, local)

    @torch.inference_mode()
    def _lookup(self, entry, hashes):
        result = entry.engram.embed.lookup(hashes, torch.device("cpu"))
        if result.device.type != "cpu" or result.dtype != torch.bfloat16 or result.numel() != entry.lookup.numel():
            raise ValueError("host lookup must return precisely the original gathered BF16 row")
        return result.reshape(1, -1).contiguous()

    def _clear(self, *, propagate_errors=True):
        old, self.pending = self.pending, {}
        errors = []
        for _, future in old.values():
            if future.cancel():
                continue
            try:
                future.result()
            except BaseException as error:
                errors.append(error)
        if errors and propagate_errors:
            raise errors[0]

    def _prepare(self, hashes):
        self._clear()
        if hashes is None or hashes.ndim != 4 or hashes.shape[:2] != (1, 1):
            return
        if hashes.device.type != "cpu":
            raise ValueError("this explicit host-lookup path requires CPU hash IDs")
        for key, entry in self.entries.items():
            if _source_stamp(entry.engram) != entry.stamp:
                raise RuntimeError("original Engram parameters/table ownership changed")
            ids = hashes[:, :, entry.engram.layer_hash_index, :].detach().clone().contiguous()
            self.pending[key] = (_key(ids), self.executor.submit(self._lookup, entry, ids))

    def _body(self, entry):
        local = original_fp8_engram_projection(
            entry.lookup, entry.weight_kn, quantized=entry.engram._quantized,
            certified_original_bf16_weight=True, backend=self.backend)
        if self.world == 1:
            complete = local
        else:
            self.gather(entry.gathered.view(torch.int8).reshape(-1), local.view(torch.int8).reshape(-1))
            complete = entry.gathered
        entry.output = original_engram_gate(entry.x, complete, entry.channel_weight, entry.mask,
                                             eps=entry.engram.eps, clamp_value=entry.engram.clamp_value)
        return entry.output

    def _stream_guard(self):
        if self.backend != "triton":
            return
        stream = int(torch.npu.current_stream(self.device).npu_stream)
        if self.owner_stream is None:
            self.owner_stream = stream
        elif stream != self.owner_stream:
            raise RuntimeError("device Engram owns one execution stream")

    @torch.inference_mode()
    def _forward(self, entry, x, hashes, mask):
        if (x.device != self.device or x.dtype != torch.bfloat16 or x.shape[:2] != (1, 1)):
            self.stats["passthrough_calls"] += 1
            return entry.original_forward(x, hashes, mask)
        if _source_stamp(entry.engram) != entry.stamp:
            raise RuntimeError("original Engram sources changed after device snapshot")
        if (x.shape != entry.x.shape or not x.is_contiguous() or hashes.device.type != "cpu"
                or hashes.ndim != 3 or hashes.shape[:2] != (1, 1)
                or hashes.shape[-1] != entry.hash_cols):
            raise ValueError("decode Engram inputs do not match the host-table geometry")
        if mask is not None and (mask.shape != (1, 1) or mask.dtype != torch.bool or mask.device.type != "cpu"):
            raise ValueError("decode mask must be original CPU bool [1,1]")
        self._stream_guard()
        pending = self.pending.get(id(entry.engram))
        if pending is not None:
            if pending[0] != _key(hashes):
                raise RuntimeError("prefetched Engram IDs do not match current token IDs")
            started = time.perf_counter()
            lookup = pending[1].result()
            self.stats["lookup_wait_s"] += time.perf_counter() - started
            self.stats["prefetched_lookups"] += 1
        else:
            lookup = self._lookup(entry, hashes)
            self.stats["direct_lookups"] += 1
        if entry.transfer_in_flight:
            started = time.perf_counter()
            entry.transfer_done.synchronize()
            self.stats["pin_reuse_wait_s"] += time.perf_counter() - started
        entry.pinned_lookup.copy_(lookup)
        if mask is None:
            entry.pinned_mask.fill_(True)
        else:
            entry.pinned_mask.copy_(mask)
        entry.lookup.copy_(entry.pinned_lookup, non_blocking=self.backend == "triton")
        entry.mask.copy_(entry.pinned_mask, non_blocking=self.backend == "triton")
        if entry.transfer_done is not None:
            entry.transfer_done.record(torch.npu.current_stream(self.device))
            entry.transfer_in_flight = True
        entry.x.copy_(x)
        self.stats["decode_calls"] += 1
        if self.backend == "triton":
            self.stats["lookup_H2D_bytes"] += entry.lookup.numel() * 2
            self.stats["mask_H2D_bytes"] += 1
        if self.capture_graphs:
            if entry.graph is None:
                from .ascend_graphs import _record
                entry.graph = _record(lambda: self._body(entry), self.graph_pool, 1, self.graph_api)
                entry.graph_output = entry.output
            entry.graph.replay()
            self.stats["graph_replays"] += 1
            return entry.graph_output.clone()  # never silently overwrite a caller's previous output
        return self._body(entry)

    def install(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("use a fresh controller after close")
            if self.installed:
                return self
            if getattr(self.model, _OWNER, None) is not None:
                raise RuntimeError("model already has a host-table device-Engram owner")
            if self.model.forward != self.original_model_forward:
                raise RuntimeError("another model forward owner changed after source preparation")
            if getattr(modules.quantized_linear, "_engram_prefetch", None) is not None:
                raise RuntimeError("close the CPU WKV prefetch before installing device Engram")
            for entry in self.entries.values():
                if (getattr(entry.engram, _OWNER, None) is not None
                        or entry.engram.forward != entry.original_forward):
                    raise RuntimeError("another forward owner changed Engram after source preparation")
            if self.capture_graphs:
                from .ascend_graphs import npu_graph_api, _require_current_device
                self.graph_api = npu_graph_api()
                _require_current_device(self.graph_api, self.device)
                self.graph_pool = self.graph_api.graph_pool_handle()
            self.executor = ThreadPoolExecutor(max_workers=self.lookup_workers, thread_name_prefix="host-table-lookup")
            for entry in self.entries.values():
                @functools.wraps(entry.original_forward)
                def forward(x, hash_ids, token_mask=None, _entry=entry):
                    with self.lock:
                        return self._forward(_entry, x, hash_ids, token_mask)
                entry.engram.forward = forward
                setattr(entry.engram, _OWNER, self)
                self.replacements.append((entry.engram, entry.original_forward, forward))
            @torch.inference_mode()
            @functools.wraps(self.original_model_forward)
            def model_forward(*args, **kwargs):
                with self.lock:
                    hashes = args[2] if len(args) > 2 else kwargs.get("hash_ids")
                    self._prepare(hashes)
                    try:
                        result = self.original_model_forward(*args, **kwargs)
                    except BaseException:
                        self._clear(propagate_errors=False)
                        raise
                    self._clear()
                    return result
            self.model.forward = model_forward
            self.model_forward_replacement = model_forward
            setattr(self.model, _OWNER, self)
            self.installed = True
            return self

    def close(self):
        with self.lock:
            if not self.installed:
                if self.closed:
                    return
                try:
                    self._clear()
                    if self.backend == "triton" and self.entries:
                        torch.npu.synchronize(self.device)
                finally:
                    if self.executor is not None:
                        self.executor.shutdown(wait=True, cancel_futures=True)
                    self.executor = None
                    self.entries.clear()
                    self.graph_api = self.graph_pool = None
                    self.closed = True
                return
            if (getattr(self.model, _OWNER, None) is not self
                    or self.model.forward is not self.model_forward_replacement):
                raise RuntimeError("another owner changed the model's Engram binding")
            for engram, _, replacement in self.replacements:
                if engram.forward is not replacement or getattr(engram, _OWNER, None) is not self:
                    raise RuntimeError("another owner changed installed Engram forward")
            errors = []
            try:
                try:
                    self._clear()
                except BaseException as error:
                    errors.append(error)
                if self.backend == "triton":
                    try:
                        torch.npu.synchronize(self.device)
                    except BaseException as error:
                        errors.append(error)
            finally:
                if self.model_had_forward_override:
                    self.model.forward = self.model_forward_override
                else:
                    delattr(self.model, "forward")
                delattr(self.model, _OWNER)
                for engram, original, _ in reversed(self.replacements):
                    entry = self.entries[id(engram)]
                    if entry.had_forward_override:
                        engram.forward = entry.forward_override
                    else:
                        delattr(engram, "forward")
                    delattr(engram, _OWNER)
                self.executor.shutdown(wait=True, cancel_futures=True)
                self.executor = None
                self.replacements.clear()
                self.entries.clear()
                self.graph_api = self.graph_pool = None
                self.installed, self.closed = False, True
            if errors:
                raise errors[0]

    def describe(self):
        return {"installed": self.installed, "backend": self.backend, "tables_on_host": True,
                "complete_Engram_CPU": False, "WKV_and_gate_on_NPU": self.backend == "triton",
                "original_table_objects_and_parameter_sources_unchanged": True,
                "WKV_output_channel_TP": self.world, "TP_rank": self.rank,
                "numerical_contract": NUMERICAL_CONTRACT,
                "explicit_device_bytes": self.explicit_device_bytes,
                "device_budget_excludes_graph_pools_and_allocator_caches": True,
                "graphs": sum(entry.graph is not None for entry in self.entries.values()),
                "source_certificates": [entry.source_report for entry in self.entries.values()],
                "source_guard_scope": "metadata and available tensor versions; versionless inference sources require caller immutability",
                **self.stats}

    def __enter__(self):
        return self.install()

    def __exit__(self, *_):
        self.close()
