"""Exact prefill/decode state hand-off for Qwen4-Exp, as transportable bytes.

This module is the transport half of a two-host pipeline split: a prefill process finishes a
prompt and hands its *complete* sequence state to a decode process that never ran the prefill.
The decode process must then produce bit-identical tokens to a monolithic run, so every piece of
state the next step reads is exported -- not an approximation of it, and not a re-derivation of
anything whose bits are not guaranteed to come back the same.

What travels, and why each piece is in the list
-----------------------------------------------
* ``GatedDeltaNetCache.conv_state`` / ``recurrent_state`` -- the recurrent half cannot be rebuilt
  from positions; it is the result of every consumed token. The conv window is the short kernel's
  tail and the recurrent state is the full ``[B, v_heads, k_dim, v_dim]`` scan result.
* ``QSAAttentionCache.k`` / ``v`` -- the consumed prefix of the KV arena. Rows past the prefix are
  unwritten in both processes (the decode process overwrites them before they become visible), so
  only ``[..., :length, :]`` is sent.
* ``QSAAttentionCache.index_k`` -- the indexer's raw per-token keys, again prefix only. These are
  what a later step pools into blocks; recomputing them is exact but pointless work.
* ``_QSAIndexerPool.keys`` -- the *completed* block prefix (mean -> layernorm -> partial RoPE in
  float32), prefix only. Without it the decode process would recompute the whole context's pooling
  on its first step, which is correct but is exactly the cost the pool exists to avoid.
* ``PLEState.conv_state`` and ``ple_token_history`` -- the n-gram history and the PLE short-conv
  window. The history is on the host by construction; both are tiny.
* The small host scalars/tensors a resumed decode needs before its first round: ``past``,
  ``next_token``, and the MTP draft seeds ``first_draft`` / ``chain_stack`` / ``first_conf``.
* The MTP head's own cache, when the drafting path will be used: the head keeps keys its later
  catch-up reads.

What deliberately does **not** travel
-------------------------------------
* ``cos``/``sin`` RoPE tables: derived from position and the model's own theta, recomputed locally.
  Sending them would invite a caller to use another model's phases.
* ``_QSAIndexerPool.signature``: it contains ``id(module)`` and ``data_ptr()`` of the *source*
  process, which mean nothing here. It is rebuilt from this process's own pointers (see
  ``_rebind_indexer_pools``); a copied signature would compare unequal, invalidate the pool, and
  silently redo the whole imported prefix.
* ``state_log`` / CUDA graphs / saved attribute bindings: process-local control state, not
  sequence state. The decode process builds its own recordings around its own caches.
* The KV arenas' unused tail: sending the arena would multiply the transfer by whatever capacity
  the arena was allocated with.

Format
------
::

    zip
      manifest.json        UTF-8 JSON: geometry, per-tensor dtype/shape/sha256, scalars
      t/<tensor name>      raw little-endian element bytes, contiguous, no header

That is deliberately not ``torch.save``: the payload is a dict of tensors plus JSON, and loading a
pickle from another host is an execution primitive this transfer does not need. Bytes are read back
with the dtype recorded in the manifest and are length- and hash-checked before anything is written
into a live cache.

Import is validate-then-apply: every tensor is matched against the destination cache's shape, dtype
and capacity before the first ``copy_`` runs, so a wrong-version or wrong-geometry bundle cannot
leave a half-updated cache behind.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import torch

from relicllm.models.qwen4_exp.attention import (
    GatedDeltaNetCache,
    QSAAttentionCache,
    _QSAIndexerPool,
    _tensor_layout,
    _tensor_version,
)
from relicllm.models.qwen4_exp.model import PLEState, Qwen4ExpCache


FORMAT = "qwen4-exp-pd-state"
VERSION = 1

#: Element dtype -> (torch dtype, storage dtype for the byte blob, itemsize). ``bfloat16`` and
#: ``float16`` are carried as int16 images because NumPy has no native bf16; the bits are identical.
_DTYPES = {
    "bfloat16": (torch.bfloat16, torch.int16, 2),
    "float16": (torch.float16, torch.int16, 2),
    "float32": (torch.float32, torch.float32, 4),
    "int64": (torch.int64, torch.int64, 8),
    "int32": (torch.int32, torch.int32, 4),
    "uint8": (torch.uint8, torch.uint8, 1),
    "float8_e4m3fn": (torch.float8_e4m3fn, torch.uint8, 1),
}
_DTYPE_NAMES = {torch_dtype: name for name, (torch_dtype, _storage, _size) in _DTYPES.items()}


class PDStateError(RuntimeError):
    """A bundle is malformed, or does not fit the cache it is being applied to."""


@dataclass
class PDStateBundle:
    """A frozen source state: JSON manifest plus CPU tensors, still independent of any process.

    The tensors are owned CPU copies (a ``clone()`` of a ``to("cpu")`` view), so the source process
    may keep decoding and reusing its arenas without changing what this bundle holds.
    """

    manifest: dict
    tensors: dict[str, torch.Tensor]

    @property
    def past(self) -> int:
        return int(self.manifest["past"])

    @property
    def next_token(self) -> int:
        return int(self.manifest["next_token"])

    def tensor_names(self) -> list[str]:
        return sorted(self.tensors)


# ---------------------------------------------------------------------------------------------
# Tensor packing
# ---------------------------------------------------------------------------------------------


def _dtype_name(dtype: torch.dtype) -> str:
    try:
        return _DTYPE_NAMES[dtype]
    except KeyError:
        raise PDStateError(f"unsupported state dtype {dtype}") from None


def _storage_view(tensor: torch.Tensor) -> torch.Tensor:
    """A 1-D uint8 view of a contiguous tensor's bytes, without copying."""
    _torch_dtype, storage, _size = _DTYPES[_dtype_name(tensor.dtype)]
    return tensor.contiguous().view(storage).view(torch.uint8).reshape(-1)


def _to_cpu_owned(tensor: torch.Tensor, *, name: str) -> torch.Tensor:
    """An owned, contiguous CPU copy of `tensor`, exact dtype and bits, never a view of a cache."""
    if not isinstance(tensor, torch.Tensor):
        raise PDStateError(f"{name} is not a tensor")
    _dtype_name(tensor.dtype)
    with torch.inference_mode():
        return tensor.detach().to("cpu").contiguous().clone()


def _from_bytes(blob: bytes, dtype: torch.dtype, shape: list[int]) -> torch.Tensor:
    _dtype, _storage, itemsize = _DTYPES[_dtype_name(dtype)]
    expected = itemsize
    for dim in shape:
        expected *= int(dim)
    if len(blob) != expected:
        raise PDStateError(f"tensor blob is {len(blob)} bytes, expected {expected} for {shape}")
    if expected == 0:
        return torch.empty(shape, dtype=dtype)
    # Interpret the bytes as a dtype-agnostic uint8 buffer and reinterpret to the recorded dtype, so
    # a bf16 payload comes back as bf16 rather than as its int16 storage image.
    flat = torch.frombuffer(bytearray(blob), dtype=torch.uint8).view(dtype)
    with torch.inference_mode():
        return flat.view(shape).clone()


def _sha256(view: torch.Tensor) -> str:
    digest = hashlib.sha256()
    with memoryview(view.numpy()) as buffer:
        for start in range(0, len(buffer), 1 << 22):
            digest.update(buffer[start:start + (1 << 22)])
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------
# Freeze (export)
# ---------------------------------------------------------------------------------------------


def _prefix_length(owner, fallback: int) -> int:
    length = getattr(owner, "length", None)
    return int(fallback) if length is None else int(length)


def _indexer_ratio(layer_cache) -> int:
    pool = getattr(layer_cache, "indexer_pool", None)
    if pool is not None and pool.keys.shape[1] > 0:
        return int(layer_cache.index_k.shape[1] // pool.keys.shape[1])
    return int(getattr(layer_cache, "compress_ratio", 0) or 0)


def freeze_state(
    *,
    cache: Qwen4ExpCache,
    past: int,
    next_token: int,
    head_cache: Qwen4ExpCache | None = None,
    first_draft: int | None = None,
    chain_stack: torch.Tensor | None = None,
    first_conf: float | None = None,
    prefill_s: float | None = None,
    tp_rank: int = 0,
    tp_world_size: int = 1,
    native_max_context: int | None = None,
    metadata: dict | None = None,
    batch_size: int = 1,
) -> PDStateBundle:
    """Freeze every piece of sequence state the next decode step reads.

    `past` is the number of consumed positions. Only prefixes up to it are exported; arena rows at
    or past it are not part of the state (both sides overwrite them before they are visible).
    """
    if cache is None:
        raise PDStateError("freeze_state needs a backbone cache")
    past, batch_size = int(past), int(batch_size)
    if past < 0:
        raise PDStateError(f"past must be nonnegative, got {past}")
    if batch_size < 1:
        raise PDStateError(f"batch_size must be positive, got {batch_size}")
    if native_max_context is not None and past > int(native_max_context):
        raise PDStateError(f"past {past} exceeds native context {native_max_context}")
    cache_length = int(getattr(cache, "length", 0) or 0)
    if cache_length not in (0, past):
        # A cache whose own length disagrees with the caller's `past` would export a prefix that is
        # not the one the next step expects; refuse rather than guess which number is right.
        raise PDStateError(f"cache.length {cache_length} disagrees with past {past}")
    if metadata is not None:
        _json_safe(metadata, "metadata")

    tensors: dict[str, torch.Tensor] = {}
    layers: list[dict] = []
    for index, layer_cache in enumerate(cache.layers):
        entry: dict = {"index": int(index), "kind": "none"}
        if layer_cache is None:
            layers.append(entry)
            continue
        if isinstance(layer_cache, GatedDeltaNetCache):
            entry.update(kind="gdn", initialized=bool(getattr(layer_cache, "initialized", False)),
                         length=int(getattr(layer_cache, "length", past)))
            tensors[f"backbone.layer.{index}.conv_state"] = _to_cpu_owned(
                layer_cache.conv_state, name=f"layer {index} conv_state")
            tensors[f"backbone.layer.{index}.recurrent_state"] = _to_cpu_owned(
                layer_cache.recurrent_state, name=f"layer {index} recurrent_state")
        elif isinstance(layer_cache, QSAAttentionCache):
            length = _prefix_length(layer_cache, past)
            if length > past:
                raise PDStateError(
                    f"layer {index} records length {length} past the exported prefix {past}"
                )
            entry.update(kind="qsa", length=length,
                         host_kv=bool(getattr(layer_cache, "host_kv", False)),
                         host_position=int(getattr(layer_cache, "host_position", length)))
            _export_qsa_prefix(tensors, f"backbone.layer.{index}", layer_cache, length,
                               ratio=_indexer_ratio(layer_cache), entry=entry,
                               batch_size=batch_size)
        else:
            raise PDStateError(f"layer {index} has unsupported cache type {type(layer_cache).__name__}")
        layers.append(entry)

    ple_entries: list[int] = []
    for index, state in sorted(cache.ple_states.items()):
        if state is None or getattr(state, "conv_state", None) is None:
            continue
        if not isinstance(state, PLEState):
            raise PDStateError(f"PLE state {index} has unsupported type {type(state).__name__}")
        ple_entries.append(int(index))
        tensors[f"backbone.ple.{index}.conv_state"] = _to_cpu_owned(
            state.conv_state, name=f"ple {index} conv_state")
    if cache.ple_token_history is not None:
        tensors["backbone.ple_token_history"] = _to_cpu_owned(
            cache.ple_token_history, name="ple_token_history")

    head_manifest = None
    if head_cache is not None:
        head_layers: list[dict] = []
        for index, layer_cache in enumerate(head_cache.layers):
            entry = {"index": int(index), "kind": "none"}
            if layer_cache is None:
                head_layers.append(entry)
                continue
            if isinstance(layer_cache, GatedDeltaNetCache):
                entry.update(kind="gdn",
                             initialized=bool(getattr(layer_cache, "initialized", False)),
                             length=int(getattr(layer_cache, "length", 0)))
                tensors[f"head.layer.{index}.conv_state"] = _to_cpu_owned(
                    layer_cache.conv_state, name=f"head layer {index} conv_state")
                tensors[f"head.layer.{index}.recurrent_state"] = _to_cpu_owned(
                    layer_cache.recurrent_state, name=f"head layer {index} recurrent_state")
            elif isinstance(layer_cache, QSAAttentionCache):
                length = _prefix_length(layer_cache, int(getattr(head_cache, "length", 0) or 0))
                entry.update(kind="qsa", length=length,
                             host_kv=bool(getattr(layer_cache, "host_kv", False)),
                             host_position=int(getattr(layer_cache, "host_position", length)))
                _export_qsa_prefix(tensors, f"head.layer.{index}", layer_cache, length,
                                   ratio=_indexer_ratio(layer_cache), entry=entry,
                                   batch_size=batch_size)
            else:
                raise PDStateError(
                    f"head layer {index} has unsupported cache type {type(layer_cache).__name__}"
                )
            head_layers.append(entry)
        head_manifest = {"layers": head_layers,
                         "length": int(getattr(head_cache, "length", 0) or 0),
                         "has_rope_table": getattr(head_cache, "cos", None) is not None}

    if chain_stack is not None:
        tensors["mtp.chain_stack"] = _to_cpu_owned(chain_stack, name="chain_stack")

    manifest: dict = {
        "format": FORMAT,
        "version": VERSION,
        "past": past,
        "next_token": int(next_token),
        "batch_size": batch_size,
        "cache_length": cache_length,
        "native_max_context": None if native_max_context is None else int(native_max_context),
        "tp_rank": int(tp_rank),
        "tp_world_size": int(tp_world_size),
        "backbone": {
            "layers": layers,
            "has_rope_table": getattr(cache, "cos", None) is not None,
            "rope_positions": (int(cache.cos.shape[1])
                               if getattr(cache, "cos", None) is not None else None),
            "ple_indices": ple_entries,
        },
        "head": head_manifest,
        "mtp": {
            "first_draft": None if first_draft is None else int(first_draft),
            "first_conf": None if first_conf is None else float(first_conf),
            "prefill_s": None if prefill_s is None else float(prefill_s),
            "has_chain_stack": chain_stack is not None,
        },
        "metadata": dict(metadata or {}),
        "tensors": [],
    }
    for name in sorted(tensors):
        tensor = tensors[name]
        manifest["tensors"].append({
            "name": name,
            "dtype": _dtype_name(tensor.dtype),
            "shape": [int(dim) for dim in tensor.shape],
            "nbytes": int(tensor.numel()) * _DTYPES[_dtype_name(tensor.dtype)][2],
            "sha256": _sha256(_storage_view(tensor)),
        })
    return PDStateBundle(manifest=manifest, tensors=tensors)


def _export_qsa_prefix(tensors, prefix, layer_cache, length, *, ratio, entry, batch_size) -> None:
    """Export `[..., :length, :]` of k/v/index_k, plus the completed indexer-pool prefix."""
    if length < 0:
        raise PDStateError(f"{prefix} length must be nonnegative, got {length}")
    for attr in ("k", "v", "index_k"):
        tensor = getattr(layer_cache, attr, None)
        if tensor is None:
            raise PDStateError(f"{prefix} has no {attr}")
        if int(tensor.shape[0]) != batch_size:
            raise PDStateError(
                f"{prefix} {attr} batch {tensor.shape[0]} != exported batch {batch_size}"
            )
        positions = int(tensor.shape[1] if attr == "index_k" else tensor.shape[2])
        if length > positions:
            raise PDStateError(f"{prefix} {attr} cannot hold a prefix of {length} positions")
        view = tensor[:, :length] if attr == "index_k" else tensor[:, :, :length]
        tensors[f"{prefix}.{attr}"] = _to_cpu_owned(view, name=f"{prefix} {attr}")
        entry[f"{attr}_positions"] = positions
    pool = getattr(layer_cache, "indexer_pool", None)
    # `tensor_ready` says the entire arena was initialized for replay, not whether eager
    # prefill's completed prefix is valid. Eager writes maintain `completed` while leaving
    # tensor_ready false; dropping that prefix silently forces pooling again on the receiver.
    if pool is None:
        entry.update(indexer_pool=False, indexer_completed=0)
        return
    if ratio <= 0:
        raise PDStateError(f"{prefix} has an indexer pool but no compression ratio")
    # A speculative rewind may leave the pool's device count ahead of the committed layer.
    # Only complete groups visible in this layer's exported prefix are transferable.
    completed = min(int(pool.completed if pool.completed is not None else int(pool.completed_tensor)),
                    length // ratio)
    if completed <= 0:
        entry.update(indexer_pool=False, indexer_completed=0)
        return
    if int(pool.keys.shape[0]) != batch_size:
        raise PDStateError(
            f"{prefix} indexer pool batch {pool.keys.shape[0]} != exported batch {batch_size}"
        )
    if int(pool.keys.shape[1]) < completed:
        raise PDStateError(f"{prefix} pool holds {pool.keys.shape[1]} groups, not {completed}")
    entry.update(indexer_pool=True, indexer_completed=completed, indexer_ratio=int(ratio))
    tensors[f"{prefix}.indexer_keys"] = _to_cpu_owned(
        pool.keys[:, :completed], name=f"{prefix} indexer pool prefix")


# ---------------------------------------------------------------------------------------------
# Thaw (import)
# ---------------------------------------------------------------------------------------------


def _rope_table_ready(cache) -> bool:
    """The head publishes absolute tables on its QSA layer, not on the wrapper cache."""
    if cache is None:
        return False
    if getattr(cache, "cos", None) is not None and getattr(cache, "sin", None) is not None:
        return True
    indexed = [layer for layer in getattr(cache, "layers", [])
               if layer is not None and getattr(layer, "index_k", None) is not None]
    return bool(indexed) and all(getattr(layer, "cos", None) is not None
                                 and getattr(layer, "sin", None) is not None for layer in indexed)


def thaw_state(
    bundle: PDStateBundle,
    *,
    cache: Qwen4ExpCache,
    head_cache: Qwen4ExpCache | None = None,
    model=None,
    tp_rank: int | None = None,
    tp_world_size: int | None = None,
    native_max_context: int | None = None,
    expected_metadata: dict | None = None,
) -> dict:
    """Validate `bundle` against this process's caches, then copy it into them in place.

    `model` is optional and is used only to rebind each restored indexer pool's process-local
    signature (it holds ``id()`` and ``data_ptr()`` values of the *source* process). Without it the
    pools are left invalidated: still correct, because the first step recomputes the prefix from the
    raw index keys that were also imported, but slower by that one-time pooling pass.

    The destination's RoPE tables are *not* part of the payload. Build them before calling this
    (and set the capture-mode ``rope_width`` on the destination cache) if the pool prefix is meant
    to be reused immediately: the pool signature covers the table's layout and version, so a table
    built later leaves the pool invalidated until the next session. The returned report says which
    happened: ``backbone_rope_table_ready``, ``head_rope_table_ready`` and each pool's ``rebound``.

    Returns a report with the counters a resumed decode needs plus what was applied.
    """
    manifest = _validate_manifest(bundle, cache=cache, head_cache=head_cache, tp_rank=tp_rank,
                                  tp_world_size=tp_world_size,
                                  native_max_context=native_max_context,
                                  expected_metadata=expected_metadata)
    plan = _build_plan(bundle, manifest, cache=cache, head_cache=head_cache)

    # No live tensor has been touched above this line. Everything below copies into allocations the
    # destination already owns, so buffer addresses (and recordings reading them) stay valid.
    with torch.inference_mode():
        for copy in plan["copies"]:
            if copy["kind"] == "tensor":
                copy["live"].copy_(copy["frozen"])
            else:
                pool = _ensure_pool(copy["layer_cache"], copy["ratio"], where=copy["where"])
                pool.keys[:, :copy["completed"]].copy_(copy["frozen"])
    _apply_lengths_and_flags(manifest, cache=cache, head_cache=head_cache)
    pools = _finish_pools(plan, model=model, cache=cache, head_cache=head_cache)
    mtp = _require_section(manifest, "mtp")
    head_section = manifest.get("head")
    return {
        "past": int(_require(manifest, "past")),
        "next_token": int(_require(manifest, "next_token")),
        "cache_length": int(_require(manifest, "past")),
        "batch_size": int(_require(manifest, "batch_size")),
        "head_cache_length": (None if head_section is None
                              else int(head_section.get("length", 0))),
        "first_draft": mtp.get("first_draft"),
        "first_conf": mtp.get("first_conf"),
        "prefill_s": mtp.get("prefill_s"),
        "chain_stack": (bundle.tensors["mtp.chain_stack"].clone()
                        if "mtp.chain_stack" in bundle.tensors else None),
        "metadata": dict(manifest.get("metadata") or {}),
        "tensors_applied": [copy["name"] for copy in plan["copies"]],
        "initialized_layers": plan["initialized"],
        "indexer_pools": pools,
        "indexer_pool_rebound": sum(1 for pool in pools if pool["rebound"]),
        "indexer_pool_needs_recompute": sum(1 for pool in pools if not pool["rebound"]),
        "backbone_rope_table_ready": _rope_table_ready(cache),
        "head_rope_table_ready": _rope_table_ready(head_cache),
    }


def _validate_manifest(bundle, *, cache, head_cache, tp_rank, tp_world_size,
                       native_max_context, expected_metadata) -> dict:
    if not isinstance(bundle, PDStateBundle):
        raise PDStateError("thaw_state needs a PDStateBundle")
    manifest = bundle.manifest
    if manifest.get("format") != FORMAT:
        raise PDStateError(f"bundle format is {manifest.get('format')!r}, not {FORMAT!r}")
    if int(manifest.get("version", -1)) != VERSION:
        raise PDStateError(
            f"bundle version {manifest.get('version')} is not supported by this build ({VERSION})"
        )
    if cache is None:
        raise PDStateError("thaw_state needs a destination backbone cache")
    # Required top-level structure, checked here so a truncated manifest cannot fail half way
    # through a copy loop.
    for key in ("past", "next_token", "batch_size", "tensors"):
        _require(manifest, key)
    _require_section(manifest, "backbone")
    _require_section(manifest, "mtp")
    past = int(_require(manifest, "past"))
    native = manifest.get("native_max_context")
    if native_max_context is not None and native is not None and int(native_max_context) != int(native):
        raise PDStateError(
            f"bundle was frozen for native context {native}, this process has {native_max_context}"
        )
    limit = native_max_context if native_max_context is not None else native
    if limit is not None and past > int(limit):
        raise PDStateError(f"bundle past {past} exceeds native context {limit}")
    if tp_rank is not None and int(tp_rank) != int(_require(manifest, "tp_rank")):
        raise PDStateError(f"bundle tensor-parallel rank {manifest['tp_rank']} != {tp_rank}")
    if tp_world_size is not None and int(tp_world_size) != int(_require(manifest, "tp_world_size")):
        raise PDStateError(
            f"bundle tensor-parallel width {manifest['tp_world_size']} != {tp_world_size}"
        )
    if expected_metadata:
        _json_safe(expected_metadata, "expected_metadata")
        actual = manifest.get("metadata") or {}
        for key, value in expected_metadata.items():
            if actual.get(key) != value:
                raise PDStateError(
                    f"bundle metadata {key!r} is {actual.get(key)!r}, expected {value!r}"
                )
    recorded = {entry["name"] for entry in manifest.get("tensors", [])}
    if recorded != set(bundle.tensors):
        raise PDStateError("manifest tensor list and bundle tensors disagree")
    if head_cache is None and manifest.get("head") is not None:
        raise PDStateError("bundle carries an MTP head cache but no head_cache was given")
    if head_cache is not None and manifest.get("head") is None:
        raise PDStateError("head_cache was given but the bundle carries no head cache state")
    for scope, container in (("backbone", cache), ("head", head_cache)):
        if container is None:
            continue
        section = _require_section(manifest, scope)
        entries = section.get("layers", [])
        if len(entries) != len(container.layers):
            raise PDStateError(
                f"{scope} bundle has {len(entries)} layers, destination has "
                f"{len(container.layers)}"
            )
        for expected_index, entry in enumerate(entries):
            index = int(entry["index"])
            if index != expected_index:
                raise PDStateError(f"{scope} layer indices are not the complete ordered cache layout")
            layer_cache = container.layers[index]
            kind = entry["kind"]
            if kind not in {"none", "gdn", "qsa"}:
                raise PDStateError(f"{scope} layer {index}: unknown cache kind {kind!r}")
            if kind == "none":
                if layer_cache is not None:
                    raise PDStateError(f"{scope} layer {index}: bundle empty, destination populated")
                continue
            if layer_cache is None:
                raise PDStateError(f"{scope} layer {index}: bundle populated, destination empty")
            expected = GatedDeltaNetCache if kind == "gdn" else QSAAttentionCache
            if not isinstance(layer_cache, expected):
                raise PDStateError(
                    f"{scope} layer {index} is {type(layer_cache).__name__}, bundle says "
                    f"{expected.__name__}"
                )
    return manifest


def _require(container: dict, key: str):
    """A manifest field that must exist; a missing one is a malformed bundle, not a KeyError."""
    try:
        return container[key]
    except (KeyError, TypeError):
        raise PDStateError(f"bundle manifest is missing {key!r}") from None


def _require_section(manifest: dict, name: str) -> dict:
    section = _require(manifest, name)
    if not isinstance(section, dict):
        raise PDStateError(f"bundle manifest section {name!r} is not an object")
    return section


def _build_plan(bundle, manifest, *, cache, head_cache) -> dict:
    declared = {entry["name"]: entry for entry in _require(manifest, "tensors")}
    past = int(_require(manifest, "past"))
    plan: dict = {"copies": [], "initialized": [], "pools": []}
    for scope, container in (("backbone", cache), ("head", head_cache)):
        if container is None:
            continue
        section = _require_section(manifest, scope)
        for entry in section.get("layers", []):
            index, kind = int(entry["index"]), entry["kind"]
            if kind == "none":
                continue
            layer_cache = container.layers[index]
            if kind == "gdn":
                for attr in ("conv_state", "recurrent_state"):
                    name = f"{scope}.layer.{index}.{attr}"
                    plan["copies"].append(_tensor_copy(
                        name, getattr(layer_cache, attr), bundle, declared))
                plan["initialized"].append(
                    {"scope": scope, "index": index,
                     "initialized": bool(entry.get("initialized", False))})
                continue
            layer_past = past if scope == "backbone" else int(entry.get("length", 0))
            length = int(entry.get("length", layer_past))
            if length < 0 or length > layer_past:
                raise PDStateError(
                    f"{scope} layer {index}: recorded length {length} is outside "
                    f"[0, {layer_past}]"
                )
            for attr in ("k", "v", "index_k"):
                name = f"{scope}.layer.{index}.{attr}"
                live_tensor = getattr(layer_cache, attr)
                positions = int(entry.get(f"{attr}_positions", 0))
                capacity = int(live_tensor.shape[1] if attr == "index_k" else live_tensor.shape[2])
                if positions != capacity:
                    raise PDStateError(
                        f"{name}: source arena {positions} != destination arena {capacity}"
                    )
                if length > capacity:
                    raise PDStateError(
                        f"{name} cannot hold {length} positions (capacity {capacity})")
                view = live_tensor[:, :length] if attr == "index_k" else live_tensor[:, :, :length]
                plan["copies"].append(_tensor_copy(name, view, bundle, declared))
            completed = int(entry.get("indexer_completed", 0) or 0)
            if completed > 0:
                ratio = int(entry.get("indexer_ratio", 0) or 0)
                if ratio <= 0 or completed > length // ratio:
                    raise PDStateError(f"{scope} layer {index}: invalid completed pooled prefix")
                name = f"{scope}.layer.{index}.indexer_keys"
                frozen = _frozen(bundle, declared, name)
                expected_shape = (int(layer_cache.index_k.shape[0]), completed,
                                  int(layer_cache.index_k.shape[-1]))
                if tuple(frozen.shape) != expected_shape or frozen.dtype is not torch.float32:
                    raise PDStateError(f"{name}: pooled prefix must have shape {expected_shape} and dtype float32")
                pool = getattr(layer_cache, "indexer_pool", None)
                expected_capacity = int(layer_cache.index_k.shape[1]) // ratio
                if pool is not None and int(pool.keys.shape[1]) != expected_capacity:
                    # Checked here rather than at copy time so a wrong pool geometry is a validation
                    # failure, before the first `copy_` has touched the cache.
                    raise PDStateError(
                        f"{name}: existing pool has {pool.keys.shape[1]} groups, geometry needs "
                        f"{expected_capacity}"
                    )
                capacity = expected_capacity if pool is None else int(pool.keys.shape[1])
                if completed > capacity:
                    raise PDStateError(
                        f"{name}: destination holds {capacity} groups, bundle has {completed}")
                plan["copies"].append({
                    "kind": "pool", "name": name, "where": f"{scope} layer {index}",
                    "layer_cache": layer_cache, "ratio": ratio, "completed": completed,
                    "frozen": frozen,
                })
                plan["pools"].append({"scope": scope, "index": index,
                                      "layer_cache": layer_cache, "completed": completed,
                                      "ratio": ratio, "rebound": False, "reason": ""})
    for index in _require_section(manifest, "backbone").get("ple_indices", []):
        state = cache.ple_states.get(index)
        if state is None or getattr(state, "conv_state", None) is None:
            raise PDStateError(f"destination cache has no PLE state {index}")
        plan["copies"].append(_tensor_copy(
            f"backbone.ple.{index}.conv_state", state.conv_state, bundle, declared))
    if "backbone.ple_token_history" in declared:
        plan["copies"].append(_tensor_copy(
            "backbone.ple_token_history", cache.ple_token_history, bundle, declared))
    if "mtp.chain_stack" in declared:
        # Returned to the caller rather than written into a cache: the resumed decode builds its own
        # device copy of the draft seed.
        _frozen(bundle, declared, "mtp.chain_stack")
    return plan


def _tensor_copy(name: str, live, bundle, declared) -> dict:
    if live is None:
        raise PDStateError(f"destination cache has no {name}")
    frozen = _frozen(bundle, declared, name)
    if tuple(live.shape) != tuple(frozen.shape):
        raise PDStateError(
            f"{name}: destination {list(live.shape)} != bundle {list(frozen.shape)}"
        )
    if live.dtype is not frozen.dtype:
        raise PDStateError(f"{name}: destination dtype {live.dtype} != bundle {frozen.dtype}")
    return {"kind": "tensor", "name": name, "live": live, "frozen": frozen}


def _frozen(bundle, declared, name: str) -> torch.Tensor:
    try:
        entry = declared[name]
    except KeyError:
        raise PDStateError(f"manifest is missing the tensor for {name}") from None
    try:
        tensor = bundle.tensors[name]
    except KeyError:
        raise PDStateError(f"bundle is missing tensor {name}") from None
    if [int(dim) for dim in tensor.shape] != [int(dim) for dim in entry["shape"]]:
        raise PDStateError(
            f"{name}: bundle shape {list(tensor.shape)} != manifest {entry['shape']}"
        )
    if _dtype_name(tensor.dtype) != entry["dtype"]:
        raise PDStateError(f"{name}: bundle dtype {tensor.dtype} != manifest {entry['dtype']}")
    expected_bytes = int(tensor.numel()) * _DTYPES[entry["dtype"]][2]
    if int(entry["nbytes"]) != expected_bytes:
        raise PDStateError(f"{name}: manifest nbytes {entry['nbytes']} != {expected_bytes}")
    if _sha256(_storage_view(tensor)) != entry["sha256"]:
        raise PDStateError(f"{name}: checksum does not match the manifest")
    return tensor


def _ensure_pool(layer_cache, ratio: int, *, where: str):
    pool = getattr(layer_cache, "indexer_pool", None)
    if pool is None:
        with torch.inference_mode(False):
            pool = _QSAIndexerPool(layer_cache.index_k, ratio)
        layer_cache.indexer_pool = pool
    elif int(pool.keys.shape[1]) != int(layer_cache.index_k.shape[1]) // ratio:
        raise PDStateError(f"{where}: existing indexer pool does not match the bundle's geometry")
    return pool


def _apply_lengths_and_flags(manifest, *, cache, head_cache) -> None:
    past = int(manifest["past"])
    cache.length = past
    for entry in manifest["backbone"].get("layers", []):
        layer_cache = cache.layers[int(entry["index"])]
        if layer_cache is None or entry["kind"] == "none":
            continue
        if hasattr(layer_cache, "length"):
            layer_cache.length = int(entry.get("length", past))
        if entry["kind"] == "gdn":
            layer_cache.initialized = bool(entry.get("initialized", False))
        elif entry["kind"] == "qsa":
            layer_cache.host_position = int(entry.get("host_position", past))
        if hasattr(layer_cache, "state_log"):
            layer_cache.state_log = None
    for state in cache.ple_states.values():
        if state is not None:
            state.state_log = None
    if head_cache is not None and manifest.get("head") is not None:
        head_cache.length = int(manifest["head"].get("length", 0))
        for entry in manifest["head"].get("layers", []):
            layer_cache = head_cache.layers[int(entry["index"])]
            if layer_cache is None or entry["kind"] == "none":
                continue
            if hasattr(layer_cache, "length"):
                layer_cache.length = int(entry.get("length", head_cache.length))
            if entry["kind"] == "gdn":
                layer_cache.initialized = bool(entry.get("initialized", False))
            elif entry["kind"] == "qsa":
                layer_cache.host_position = int(entry.get("length", head_cache.length))
            if hasattr(layer_cache, "state_log"):
                layer_cache.state_log = None


def _finish_pools(plan, *, model, cache, head_cache) -> list[dict]:
    """Publish the copied prefixes and rebind each pool's process-local signature."""
    results: list[dict] = []
    for entry in plan["pools"]:
        pool = getattr(entry["layer_cache"], "indexer_pool", None)
        record = {"scope": entry["scope"], "index": entry["index"],
                  "completed": entry["completed"], "rebound": False, "reason": ""}
        if pool is None:
            record["reason"] = "no pool after copy"
            results.append(record)
            continue
        if model is None:
            pool.invalidate()
            record["reason"] = "no model supplied; prefix will be recomputed once"
            results.append(record)
            continue
        module, cos, sin = _indexer_context(model, entry, cache=cache, head_cache=head_cache)
        if module is None or cos is None or sin is None:
            pool.invalidate()
            record["reason"] = "rope table or indexer module unavailable"
            results.append(record)
            continue
        arena = entry["layer_cache"].index_k
        # The same tuple `QSAIndexer._pool_for_write` will compute on this host: `id()` of this
        # process's module, this process's arena/table layouts and versions.
        pool.signature = (
            id(module), _tensor_layout(arena),
            int(getattr(module, "compress_ratio", entry["ratio"])),
            _tensor_layout(cos), _tensor_version(cos),
            _tensor_layout(sin), _tensor_version(sin),
        )
        arena._qsa_indexer_owner = pool.owner_token
        pool.raw_version = _tensor_version(arena)
        pool.completed = int(entry["completed"])
        pool.completed_tensor.fill_(int(entry["completed"]))
        pool.tensor_ready = True
        record["rebound"] = True
        results.append(record)
    return results


def _indexer_context(model, entry, *, cache, head_cache):
    """The indexer module and RoPE tables its pool signature is keyed on, in this process."""
    if entry["scope"] == "backbone":
        layers = getattr(model, "layers", None) or []
        if entry["index"] >= len(layers) or layers[entry["index"]] is None:
            return None, None, None
        attn = getattr(layers[entry["index"]], "attn", None)
        module = getattr(attn, "indexer", None)
        return module, getattr(cache, "cos", None), getattr(cache, "sin", None)
    head_layers = getattr(head_cache, "layers", None) or []
    if entry["index"] >= len(head_layers) or head_layers[entry["index"]] is None:
        return None, None, None
    head = getattr(model, "mtp", None)
    module = getattr(getattr(head, "layer", None), "attn", None)
    module = getattr(module, "indexer", None)
    layer = head_layers[entry["index"]]
    return module, getattr(layer, "cos", None), getattr(layer, "sin", None)


# ---------------------------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------------------------


def _write_bundle(bundle: PDStateBundle, stream) -> None:
    """Stream a bundle into a binary file object: `manifest.json` last, one entry per tensor."""
    if not isinstance(bundle, PDStateBundle):
        raise PDStateError("_write_bundle needs a PDStateBundle")
    manifest = json.loads(json.dumps(bundle.manifest))
    entries = []
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(bundle.tensors):
            tensor = bundle.tensors[name]
            # One tensor's bytes at a time: a 256K-context bundle is hundreds of MiB a rank, so the
            # whole payload is never materialised in a second buffer.
            blob = _storage_view(tensor).numpy().tobytes()
            entries.append({
                "name": name, "dtype": _dtype_name(tensor.dtype),
                "shape": [int(dim) for dim in tensor.shape],
                "nbytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest(),
                "entry": f"t/{name}",
            })
            archive.writestr(f"t/{name}", blob)
            del blob
        manifest["tensors"] = entries
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True).encode("utf-8"))


def bundle_to_bytes(bundle: PDStateBundle) -> bytes:
    """Serialize a bundle: a zip of `manifest.json` and one raw byte entry per tensor."""
    buffer = io.BytesIO()
    _write_bundle(bundle, buffer)
    return buffer.getvalue()


def bundle_from_bytes(data: bytes) -> PDStateBundle:
    """Load a bundle written by :func:`bundle_to_bytes`, verifying every checksum."""
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise PDStateError("bundle_from_bytes needs bytes")
    try:
        archive = zipfile.ZipFile(io.BytesIO(bytes(data)), "r")
    except zipfile.BadZipFile as error:
        raise PDStateError(f"bundle is not a readable archive: {error}") from None
    with archive:
        try:
            raw = archive.read("manifest.json")
        except KeyError:
            raise PDStateError("bundle has no manifest.json") from None
        except zipfile.BadZipFile as error:
            raise PDStateError(f"bundle archive is corrupt: {error}") from None
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise PDStateError(f"bundle manifest is not valid JSON: {error}") from None
        if not isinstance(manifest, dict) or manifest.get("format") != FORMAT:
            raise PDStateError("bundle manifest does not declare the expected format")
        if int(manifest.get("version", -1)) != VERSION:
            raise PDStateError(f"bundle version {manifest.get('version')} is not supported")
        tensors: dict[str, torch.Tensor] = {}
        for entry in manifest.get("tensors", []):
            name = entry.get("name")
            if not isinstance(name, str) or not name:
                raise PDStateError("bundle manifest has a tensor without a name")
            try:
                blob = archive.read(entry.get("entry", f"t/{name}"))
            except KeyError:
                raise PDStateError(f"bundle is missing the payload for {name}") from None
            except zipfile.BadZipFile as error:
                raise PDStateError(f"{name}: bundle archive is corrupt: {error}") from None
            if hashlib.sha256(blob).hexdigest() != entry.get("sha256"):
                raise PDStateError(f"{name}: payload checksum does not match the manifest")
            specs = _DTYPES.get(entry.get("dtype"))
            if specs is None:
                raise PDStateError(f"{name}: unsupported dtype {entry.get('dtype')!r}")
            tensors[name] = _from_bytes(blob, specs[0], [int(dim) for dim in entry["shape"]])
    return PDStateBundle(manifest=manifest, tensors=tensors)


def save_bundle(bundle: PDStateBundle, path: str | os.PathLike) -> int:
    """Stream a bundle atomically to `path`; returns the number of bytes written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
    try:
        with os.fdopen(handle, "wb") as stream:
            _write_bundle(bundle, stream)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return target.stat().st_size


def load_bundle(path: str | os.PathLike) -> PDStateBundle:
    return bundle_from_bytes(Path(path).read_bytes())


# ---------------------------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------------------------


def tensor_hashes(bundle: PDStateBundle) -> dict[str, str]:
    """Per-tensor content hashes, for deciding which rank replicas are byte-identical."""
    return {name: _sha256(_storage_view(tensor))
            for name, tensor in sorted(bundle.tensors.items())}


def bundle_digest(bundle: PDStateBundle) -> str:
    """One digest over the whole manifest and every tensor hash."""
    payload = json.dumps({"manifest": bundle.manifest, "tensors": tensor_hashes(bundle)},
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def identical_tensors(first: PDStateBundle, second: PDStateBundle) -> dict[str, bool]:
    """Which named tensors hold identical bytes in two bundles (names present in both)."""
    left, right = tensor_hashes(first), tensor_hashes(second)
    return {name: left[name] == right[name] for name in sorted(set(left) & set(right))}


# ---------------------------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------------------------


def _json_safe(value, where: str) -> None:
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise PDStateError(f"{where} is not JSON-serializable: {error}") from None


__all__ = [
    "FORMAT",
    "VERSION",
    "PDStateBundle",
    "PDStateError",
    "bundle_digest",
    "bundle_from_bytes",
    "bundle_to_bytes",
    "freeze_state",
    "identical_tensors",
    "load_bundle",
    "save_bundle",
    "tensor_hashes",
    "thaw_state",
]