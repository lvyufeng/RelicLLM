"""Explicit current-token CPU Engram prefetch; original gate body is never replaced.

Hash IDs are ready before the backbone starts. Lookup and WKV depend on those IDs,
not on the current hidden state, so they can run early on CPU. While installed,
small memoizing call seams feed those exact CPU results back to the unchanged
Engram body. No projection, normalization or gating is moved to the accelerator.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
import time

import torch
import torch.distributed as dist

from . import mma, modules


def _key(ids):
    if ids.device.type != "cpu":
        raise ValueError("Engram prefetch requires the already-host-resident hash IDs")
    return tuple(ids.shape), str(ids.dtype), ids.contiguous().view(torch.uint8).numpy().tobytes()


def _stamp(engram):
    result = []
    for tensor in (engram.wkv.weight, mma.attached_codes(engram.wkv.weight),
                   mma.attached_scale(engram.wkv.weight)):
        if tensor is None:
            result.append(None)
            continue
        try:
            version = tensor._version
        except RuntimeError:
            version = None
        result.append((id(tensor), tensor.data_ptr(), version, tuple(tensor.shape)))
    names = ("POCKETLLM_V41_MMA_EMULATE", "DEEPSEEK_V41_CPU_EXACT_GEMV",
             "DEEPSEEK_V41_CPU_GEMV_IMPL", "DEEPSEEK_V41_CPU_GEMV_THREADS")
    return tuple(result), tuple(os.environ.get(name) for name in names), engram._quantized


@dataclass
class _Pending:
    engram: object
    hashes: torch.Tensor
    stamp: tuple
    future: object


class HostEngramPrefetch:
    """One bounded CPU result per Engram/current token, installed only explicitly.

    The model's hash front end and Engram body stay untouched. Rank zero alone
    prepares results when the existing exact host-broadcast adapter is in use.
    Only immutable inference weights are supported; their versions and original
    source metadata are checked again before reusing a result.
    """

    def __init__(self, model, *, rank=None, workers=2):
        if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
            raise ValueError("workers must be a positive integer")
        self.model = model
        self.engrams = [layer.engram for layer in model.layers if layer.engram is not None]
        if any(engram.wkv.weight.device.type != "cpu" for engram in self.engrams):
            raise ValueError("all prefetched Engram parameters must remain on CPU")
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        self.enabled = rank == 0 and bool(self.engrams)
        self.workers = workers
        self.pending = {}
        self.executor = None
        self.installed = False
        self.stats = {"scheduled": 0, "projection_hits": 0, "lookup_hits": 0,
                      "stale_misses": 0, "CPU_wait_seconds": 0.0}
        self.original_forward = model.forward
        self.original_quantized = modules.quantized_linear
        self.lookups = {id(e.embed): (e.embed, e.embed.lookup) for e in self.engrams}

    @torch.inference_mode()
    def _compute(self, engram, hashes):
        lookup = self.lookups[id(engram.embed)][1]
        embedding = lookup(hashes, engram.wkv.weight.device)
        kv = self.original_quantized(embedding.flatten(-2), engram.wkv.weight,
                                     quantized=engram._quantized)
        return embedding, kv

    def _result(self, entry):
        start = time.perf_counter()
        value = entry.future.result()  # computation errors are never hidden by a fallback
        self.stats["CPU_wait_seconds"] += time.perf_counter()-start
        return value

    def clear(self, *, propagate_errors=True):
        errors = []
        old, self.pending = self.pending, {}
        for entry in old.values():
            if entry.future.cancel():
                continue
            try:
                entry.future.result()
            except BaseException as error:
                errors.append(error)
        if errors and propagate_errors:
            raise errors[0]

    def prepare(self, hashes):
        self.clear()
        if not self.enabled or hashes is None or hashes.shape[1] != 1:
            return
        if hashes.device.type != "cpu" or hashes.ndim != 4:
            raise ValueError("current-token hashes must be CPU [batch,1,layers,heads]")
        for engram in self.engrams:
            ids = hashes[:, :, engram.layer_hash_index, :].detach().clone().contiguous()
            stamp = _stamp(engram)
            future = self.executor.submit(self._compute, engram, ids)
            self.pending[id(engram.wkv.weight)] = _Pending(engram, ids, stamp, future)
            self.stats["scheduled"] += 1

    def _quantized(self, x, weight, *args, **kwargs):
        entry = self.pending.get(id(weight))
        if entry is not None and x.device.type == "cpu" and not args and (
            kwargs.get("quantized", False) == entry.engram._quantized
            and kwargs.get("block_size", 32) == 32
            and kwargs.get("emulate_reduction") is None
        ):
            if entry.stamp == _stamp(entry.engram):
                embedding, kv = self._result(entry)
                expected = embedding.flatten(-2)
                if x.shape == expected.shape and x.dtype == expected.dtype and torch.equal(
                    x.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
                ):
                    self.stats["projection_hits"] += 1
                    return kv
            self.stats["stale_misses"] += 1
        return self.original_quantized(x, weight, *args, **kwargs)

    def _lookup(self, table_id, hashes, device):
        if torch.device(device).type == "cpu" and self.pending:
            key = _key(hashes)
            for entry in self.pending.values():
                if id(entry.engram.embed) == table_id and _key(entry.hashes) == key:
                    if entry.stamp == _stamp(entry.engram):
                        embedding, _ = self._result(entry)
                        self.stats["lookup_hits"] += 1
                        return embedding
                    self.stats["stale_misses"] += 1
        return self.lookups[table_id][1](hashes, device)

    def install(self):
        if self.installed or not self.enabled:
            return self
        if getattr(modules.quantized_linear, "_engram_prefetch", None) is not None:
            raise RuntimeError("only one explicit Engram prefetch manager may be installed")
        self.executor = ThreadPoolExecutor(max_workers=self.workers, thread_name_prefix="host-engram")
        @torch.inference_mode()
        def forward(*args, **kwargs):
            hashes = args[2] if len(args) > 2 else kwargs.get("hash_ids")
            self.prepare(hashes)
            try:
                result = self.original_forward(*args, **kwargs)
            except BaseException:
                self.clear(propagate_errors=False)
                raise
            self.clear()
            return result
        def quantized(*args, **kwargs):
            return self._quantized(*args, **kwargs)
        quantized._engram_prefetch = self
        modules.quantized_linear = quantized
        self.model.forward = forward
        for table_id, (table, _) in self.lookups.items():
            def lookup(hashes, device, table_id=table_id):
                return self._lookup(table_id, hashes, device)
            table.lookup = lookup
        self.installed = True
        return self

    def close(self, *, propagate_errors=True):
        if not self.installed:
            return
        modules.quantized_linear = self.original_quantized
        self.model.forward = self.original_forward
        for table, lookup in self.lookups.values():
            table.lookup = lookup
        try:
            self.clear(propagate_errors=propagate_errors)
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            self.installed = False

    def describe(self):
        return {"enabled": self.enabled, "installed": self.installed,
                "pending_results": len(self.pending), "maximum_pending_results": len(self.engrams),
                "complete_Engram_CPU": True, "original_gate_body_unchanged": True, **self.stats}

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, traceback):
        self.close(propagate_errors=exc_type is None)
