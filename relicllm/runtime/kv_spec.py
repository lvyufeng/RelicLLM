"""The shape of the memory a layer's KV cache wants, in a form a host allocator can read.

Nothing in this tree stated this before. Each runtime sized and allocated its own contiguous KV
buffer inside its own adapter, from its own options (`self._model.cache(n)`, `make_cache(...)`, a
bare `torch.zeros(max_seq_len // ratio, head_dim)` inside an attention module), so "what does one
token cost, and in what layout" was answerable only by running the runtime it belonged to. That is
the single missing artefact behind two rows of the performance roadmap: no continuous batching (every
runtime serializes behind one mutable cache) and no paged KV (every cache is one contiguous buffer
prepaid at `max_seq_len`).

This module is that artefact, and it is deliberately the same shape both upstream runtimes settled
on. vLLM (`v1/kv_cache_interface.py`) publishes a family of frozen dataclasses -- ``KVCacheSpec`` ->
``AttentionSpec`` -> ``FullAttentionSpec`` / ``SlidingWindowSpec`` / ``MLAAttentionSpec``, plus
``MambaSpec`` for non-attention state -- and each attention layer implements ``get_kv_cache_spec()``
so the runner collects a ``dict[str, KVCacheSpec]`` it reads generically. SGLang declares no such
thing: its geometry is scattered across ``KVCacheConfigurator`` and hand-written formulas in
``pool_configurator.py``, with a per-family state machine (``DSV4PoolConfigurator``) for DeepSeek-V4.
The two agree on *what* to declare and disagree on *form*; this takes vLLM's form. The reading behind
that choice is recorded in ``docs/architecture/kv_declaration.md``.

## Why this is not "a new allocator", and what "read" means

``MimoV2KVCache``, ``KVLatentCache`` and ``Qwen4ExpCache`` carry ``append`` / ``view`` / ``reset``
semantics the runtimes depend on, so a declaration is not a licence to allocate a different object:
it is passed *into* those constructors, which stop deriving shapes themselves. vLLM and SGLang each
re-allocate from their spec with a generic kernel-side writer; this tree has no such writer and
inventing one is out of scope. The declaration is the shape, not the writer.

## Named caches, not one spec per layer

A v41 layer registers **four** KV-shaped buffers -- ``window_kv_cache``, ``compress_kv_cache``, the
indexer's ``k_cache``, and the compressor's ``kv_state``/``score_state`` -- and a QSA layer registers
two. One spec per layer cannot express that. A spec names one buffer and the layers that share it,
which is vLLM's ``dict[str, KVCacheSpec]`` keyed by cache name and matches this tree's buffer names
one-for-one. Layers that differ in geometry (V4.1's compressor is ratio 2 on three layers and ratio 1
on a fourth) get one spec each -- the same rule vLLM's ``merge`` states, that every layer in a group
must be identical.

## Two costs, and why both are here

A cache has an **allocated** size (what a paged block table lays out) and a **marginal** cost (what
one more token adds). They differ for a sliding-window layer, whose ring holds ``sliding_window``
slots however long the context runs: its ``page_size_bytes`` is real and its ``values_per_token`` is
zero. The triage fit test reads the marginal number; a paged allocator reads the allocated one. A
class that reported only one of them would be read wrong by one of the two callers.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch is imported lazily: this module is on the `--help` and triage paths
    import torch

__all__ = [
    "AttentionSpec",
    "FullAttentionSpec",
    "KVCacheSpec",
    "KVKind",
    "MLASpec",
    "SlidingWindowSpec",
    "StateSpec",
    "SpecConfig",
    "VALID_KV_CACHE_DTYPES",
    "dtype_size",
    "to_torch_dtype",
]

#: Per-architecture aliases: the key a publisher reads -> the attribute the runtime's own config
#: class spells it with. A publisher reads one spelling; the aliases are how the same declaration is
#: fed from a checkpoint's `config.json`, from a runtime dataclass, and from a triage test payload
#: without any of the three having to know the others' names. Flat rather than per-architecture
#: because these are genuine synonyms for the same quantity -- `num_hidden_layers` and `n_layers` are
#: one number -- and the table is deliberately small: a name that is *not* a synonym (MiMo's per-family
#: `swa_*` widths, which no single key can carry) is not aliased and is resolved by the caller.
_CONFIG_ALIASES: dict[str, tuple[str, ...]] = {
    "num_hidden_layers": ("num_hidden_layers", "n_layers", "num_layers"),
    "head_dim": ("head_dim", "attention.key_length"),
    "num_key_value_heads": ("num_key_value_heads", "attention.head_count_kv"),
    "kv_source_layer_ids": ("kv_source_layer_ids", "kv_source_layers"),
    "qk_rope_head_dim": ("qk_rope_head_dim", "rope.dimension_count"),
}


class SpecConfig(Mapping[str, Any]):
    """A view of one architecture's config that answers the names a publisher reads.

    A publisher such as ``models/deepseek_v4_1/kv_spec.py`` reads ``config["num_hidden_layers"]``;
    that runtime's own config class spells the same field ``n_layers``, and its ``compress_ratios``
    comes back as a tuple of ints rather than a list. Rather than teach every publisher every
    spelling -- or, worse, have each runtime feed the publisher a hand-built dict that can silently
    drift from its own config -- the publisher reads a mapping and the adapter bridges the two.

    A mapping rather than an attribute wrapper because the publishers also read *optional* keys
    (``config.get("layer_types")``); ``__getitem__`` raises and ``__iter__``/``__len__`` describe the
    canonical keys, so the view is a real ``Mapping`` and nothing else has to change.
    """

    __slots__ = ("_source", "_items")

    def __init__(self, source: Any) -> None:
        if isinstance(source, Mapping):
            self._source: Any = None
            self._items: Mapping[str, Any] = source
        else:
            self._source = source
            self._items = getattr(source, "__dict__", {})

    def _lookup(self, key: str) -> Any:
        for name in _CONFIG_ALIASES.get(key, (key,)):
            if self._source is not None and hasattr(self._source, name):
                return getattr(self._source, name)
            if name in self._items:
                return self._items[name]
        raise KeyError(key)

    def __getitem__(self, key: str) -> Any:
        return self._lookup(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self._lookup(key)
        except KeyError:
            return default

    def __iter__(self) -> Iterator[str]:
        present = set(self._items)
        if self._source is not None:
            present |= set(self._source.__dict__)
        for key, names in _CONFIG_ALIASES.items():
            if any(name in present for name in names):
                yield key
        for key in present:
            if key not in _CONFIG_ALIASES:
                yield key

    def __len__(self) -> int:
        return sum(1 for _ in self)

    @property
    def source(self) -> Any:
        """The object this view wraps -- a live config, or the mapping itself.

        A publisher for an architecture whose widths are **per family** (MiMo resolves a global and a
        sliding-window key/value width, and each has its own head counts) cannot build an exact
        declaration from flat keys alone: the live config's own resolution -- ``attention(layer)`` --
        is the authority, and this is how a publisher reaches it. A fixture payload or a
        ``config.json`` body is a mapping, has no such accessor, and the publisher falls back to the
        raw keys, which is exactly the path triage already exercised.
        """
        return self._items if self._source is None else self._source


def spec_config(config: Any) -> SpecConfig:
    """The view a publisher reads, from a config mapping, a dataclass instance, or another view.

    A plain mapping keeps its own keys intact -- including a GGUF's namespaced ones, which a publisher
    simply does not read -- because :meth:`SpecConfig.__getitem__` falls back to the underlying
    mapping after the alias table misses.
    """
    return config if isinstance(config, SpecConfig) else SpecConfig(config)

#: What ``--kv-cache-dtype`` may be asked for. ``auto`` is the runtime's own cache dtype, which is
#: bf16 in every runtime today, and the rest are the quantized storages a declaration could name.
#: The set is stated here -- beside the dtype table that knows their widths -- rather than in the
#: argument parser, so a host flag, a runtime's declaration and the byte arithmetic cannot disagree
#: about what a value means. Matches vLLM's ``get_kv_quant_mode`` spellings.
VALID_KV_CACHE_DTYPES: tuple[str, ...] = (
    "auto",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "int8",
    "nvfp4",
)


class KVKind(StrEnum):
    """Which of the four shapes a cache is, so a reader can dispatch without a model name."""

    FULL = "full"
    SLIDING_WINDOW = "sliding_window"
    MLA = "mla"
    STATE = "state"


#: Bytes per scalar, by the dtype string a spec carries. Deliberately a table rather than
#: ``torch.finfo``: this module is imported by triage on the ``--help`` path, and a module-scope
#: ``import torch`` would make a fit check pay a multi-second import for what is arithmetic. The
#: fp8 and fp4 rows exist because a quantized KV cache is the reason ``--kv-cache-dtype`` exists at
#: all; the widths are the storage widths, which is what a byte count needs.
_DTYPE_SIZE: dict[str, int] = {
    "bool": 1,
    "int8": 1,
    "uint8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float64": 8,
}


def dtype_size(name: str) -> int:
    """Bytes one scalar of ``name`` occupies, from the table above.

    Unknown names are a refusal rather than a default: a spec that silently read an unrecognised
    dtype as two bytes would under- or over-state a cache by the factor it got wrong, which is the
    class of mistake this module exists to make impossible.
    """
    try:
        return _DTYPE_SIZE[str(name)]
    except KeyError:
        raise ValueError(
            f"unknown KV dtype {name!r}; known: {', '.join(sorted(_DTYPE_SIZE))}"
        ) from None


def to_torch_dtype(name: str) -> "torch.dtype":
    """The ``torch.dtype`` a spec's dtype string names. Imported here so callers need not.

    Kept beside :func:`dtype_size` rather than in the specs because the two must agree: a spec that
    reported two bytes and then asked torch for a four-byte dtype would allocate twice what it
    declared, and nothing else in this module would notice.
    """
    import torch  # noqa: PLC0415 -- deliberately lazy; see the module docstring

    mapping: dict[str, torch.dtype] = {
        "bool": torch.bool,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    if name in ("float8_e4m3fn", "float8_e5m2"):
        return getattr(torch, name)
    try:
        return mapping[name]
    except KeyError:
        raise ValueError(f"dtype {name!r} has no torch spelling") from None


@dataclass(frozen=True)
class KVCacheSpec:
    """One named KV buffer and the layers that share it.

    ``layer_ids`` is the group, not an index: a spec is emitted once for every layer that has exactly
    this geometry, and two layers with the same shape share one spec because they are the same shape.
    That is vLLM's rule -- every layer in a KV group must carry an identical spec.
    """

    name: str
    """The buffer's name, as the model registers it: ``window_kv_cache``, ``k``, ``latent``, …."""

    layer_ids: tuple[int, ...]
    """The layers that share this buffer. One entry per layer, even when they share a name."""

    dtype: str = "bfloat16"
    """The dtype the cache is stored in, as a string (:func:`to_torch_dtype` resolves it)."""

    @property
    def kind(self) -> KVKind:
        raise NotImplementedError

    @property
    def layer_count(self) -> int:
        return len(self.layer_ids)

    @property
    def values_per_token(self) -> int:
        """Marginal values one more token adds to this cache, summed over its layers.

        Zero for a bounded cache -- a sliding-window ring or a fixed recurrent state -- which is the
        number a fit test wants and the one a naive ``heads * dim`` overstates.
        """
        raise NotImplementedError

    def page_size_bytes(self, block_size: int = 1) -> int:
        """Bytes one block of ``block_size`` tokens occupies, for every layer in the group.

        This is the allocated size, and it is what a paged allocator lays out. A block of one token
        is the default because the marginal number above is the block table's usual unit.
        """
        raise NotImplementedError

    def bytes_per_token_whole_model(self) -> int:
        """Marginal bytes one more context token costs this group, every layer summed.

        The word "whole model" is the fit test's, and it means *unsharded and undivided*: what one
        token would cost if every rank held every layer. It is the marginal number, so a sliding
        window and a fixed state contribute zero here however many bytes they have allocated --
        :meth:`allocated_bytes` is the one that does not.
        """
        return self.values_per_token * self.layer_count * dtype_size(self.dtype)

    def allocated_bytes(self, block_size: int = 1) -> int:
        """Bytes this group's block table actually holds for one block, every layer summed.

        The other cost. A sliding window's contribution is real here and zero above, which is the
        distinction this module keeps both numbers for.
        """
        return self.page_size_bytes(block_size) * self.layer_count


@dataclass(frozen=True)
class AttentionSpec(KVCacheSpec):
    """A key/value cache under a head axis: ``num_kv_heads`` heads of ``head_dim`` keys and a value.

    ``v_head_dim`` is ``None`` when the value width follows the key width -- which is the common case
    and not the reason this field exists. It is set when the two genuinely differ (MiMo-V2.6-Flash
    caches 192-wide keys beside 128-wide values) and set to ``0`` for an *absorbed* cache, which
    stores no value tensor at all because the value is recovered from the key latent.
    """

    num_kv_heads: int = 0
    head_dim: int = 0
    v_head_dim: int | None = None

    @property
    def kind(self) -> KVKind:
        return KVKind.FULL

    @property
    def value_dim(self) -> int:
        """The width of one value vector, or zero when this cache stores no value tensor."""
        return self.head_dim if self.v_head_dim is None else int(self.v_head_dim)

    @property
    def attention_values_per_token(self) -> int:
        """One token's key and value values, before dtype: the GQA formula.

        Kept separate from :attr:`values_per_token` so a subclass can override the *marginal* reading
        without changing the *allocated* one -- a sliding window's page is real and its marginal cost
        is not.
        """
        return int(self.num_kv_heads) * (int(self.head_dim) + self.value_dim)

    @property
    def values_per_token(self) -> int:
        return self.attention_values_per_token

    def page_size_bytes(self, block_size: int = 1) -> int:
        return int(block_size) * self.attention_values_per_token * dtype_size(self.dtype)


@dataclass(frozen=True)
class FullAttentionSpec(AttentionSpec):
    """A cache that grows one row a token, for the whole context."""

    @property
    def kind(self) -> KVKind:
        return KVKind.FULL


@dataclass(frozen=True)
class SlidingWindowSpec(AttentionSpec):
    """A bounded ring: ``sliding_window`` slots, overwritten modulo their length.

    Its ``values_per_token`` is zero and its page size is not. A ring of 128 holds the last 128
    positions however long the context runs, so one more token costs the model nothing *in total* --
    which is exactly the saving a fit test has to see, and exactly the allocation a block table still
    has to provide for.
    """

    sliding_window: int = 0

    @property
    def kind(self) -> KVKind:
        return KVKind.SLIDING_WINDOW

    @property
    def values_per_token(self) -> int:
        return 0


@dataclass(frozen=True)
class MLASpec(AttentionSpec):
    """An MLA cache: one latent a token, either absorbed whole or compressed by a ratio.

    Two readers, one formula. An absorbed cache (Xing4.0-29B-A4B) stores
    ``kv_lora_rank + qk_rope_head_dim`` values per token with no head axis and no value tensor, so it
    is a spec with ``head_dim`` set to that width and ``compress_ratio`` left at 1. A compressed cache
    (DeepSeek-V4 and V4.1) stores ``head_dim // compress_ratio`` values, because ``compress_ratio``
    consecutive tokens pool into one latent. ``head_dim // compress_ratio`` answers both.

    The value is **not** doubled the way a GQA pair is: there is one latent, and the value is
    ``latent @ v_b`` -- recovered after the attention weights, never stored. That is the whole
    saving of MLA and the reason :attr:`values_per_token` is overridden rather than inherited.
    """

    kv_lora_rank: int = 0
    qk_rope_head_dim: int = 0
    compress_ratio: int = 1

    @property
    def kind(self) -> KVKind:
        return KVKind.MLA

    @property
    def values_per_token(self) -> int:
        ratio = int(self.compress_ratio) or 1
        return int(self.head_dim) // ratio

    def page_size_bytes(self, block_size: int = 1) -> int:
        """A compressed block is ``block_size // ratio`` latents, floored at one.

        Flooring at one is not a rounding convenience: a block smaller than the compression ratio
        still has to hold the partial group that is filling up, and the compressor keeps exactly that
        in ``kv_state``. Reporting zero would let a block table under-allocate the layer.
        """
        ratio = int(self.compress_ratio) or 1
        latents = max(1, int(block_size) // ratio)
        return latents * self.values_per_token * dtype_size(self.dtype)


@dataclass(frozen=True)
class StateSpec(KVCacheSpec):
    """A cache with no token axis: a recurrent or convolutional state, sized once.

    Upstream's ``MambaSpec`` row, and the row ``qwen4_exp``'s 36 GatedDeltaNet layers and V4.1's
    compressor need: their buffers are shaped by the head counts and the kernel width, never by the
    context, so they hold the same number of values at one token and at a million.
    """

    shapes: tuple[tuple[int, ...], ...] = ()
    dtypes: tuple[str, ...] = ()

    @property
    def kind(self) -> KVKind:
        return KVKind.STATE

    @property
    def values_per_token(self) -> int:
        return 0

    def page_size_bytes(self, block_size: int = 1) -> int:
        """A state does not scale with tokens, so its "page" is the whole thing, once."""
        return sum(
            _product(shape) * dtype_size(dtype or self.dtype)
            for shape, dtype in _zip_shapes(self.shapes, self.dtypes)
        )


def _product(shape: tuple[int, ...]) -> int:
    total = 1
    for size in shape:
        total *= int(size)
    return total


def _zip_shapes(
    shapes: tuple[tuple[int, ...], ...], dtypes: tuple[str, ...]
) -> list[tuple[tuple[int, ...], str]]:
    """Pair each shape with its dtype, defaulting to the spec's own when fewer dtypes are given."""
    if not dtypes:
        return [(shape, "") for shape in shapes]
    return [(shape, dtypes[index] if index < len(dtypes) else "") for index, shape in enumerate(shapes)]