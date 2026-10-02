"""KV-cache geometry: what one token costs, per layer class, and how the cache is shared.

**The one thing this module exists to stop.** The obvious calculation -- ``n_layers * n_kv_heads *
head_dim``, times two for K and V -- is wrong for most of this repository's checkpoints, and wrong
in a direction nobody checks. Four of the nine models here are *hybrid*: most of their layers hold no
growing cache at all, because they are sliding-window rings (a fixed 128 slots, zero marginal cost
per token) or linear attention (no cache at all). Computing them as full attention overstates the
cache by 3.5x to infinity, and the judge of which is which is ``layer_types`` or ``compress_ratios``
in the checkpoint's own config -- not the layer count, and not the model card.

Three further things vary per model and are recorded here rather than assumed:

* **Sharding.** Whether a rank holds a slice of the cache or a whole copy is a property of the
  allocation site in code, and the answer differs across the runtimes: MiMo shards by KV head,
  while V4.1, MiniMax and GLM each hold a full copy on every rank. Assuming sharding where the code
  replicates is a 4x error at TP4. The default for an unrecognised architecture is therefore
  :data:`REPLICATED`, which is the conservative direction for a fit test.
* **Preallocation.** Whether the cache is sized once at load from ``max_seq_len`` (so
  ``--max-model-len`` is a *prepayment*, paid whether or not the context is used) or sized per
  request (so it is paid only in use). These support different questions and a report that does not
  distinguish them misleads in opposite directions for the two groups.
* **Extra caches.** DeepSeek-V4 and V4.1 attach a second per-token cache (an ``Indexer``) to their
  compressed layers, separate from the attention cache and easy to miss.

Every number a caller gets back carries the ``file:line`` it came from, so a disagreement with the
code is arguable rather than a matter of memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Iterable, Mapping  # noqa: F401  (Mapping is in the public signatures)

from relicllm.triage.keys import (
    architecture_key as _canonical,
    architecture_of as _architecture_of,
    flatten_gguf as _flatten_gguf,
    int_at as _int,
    list_at as _list,
)

__all__ = [
    "Confidence",
    "KvGeometry",
    "LayerGeometry",
    "SHARDED",
    "REPLICATED",
    "SINGLE_CARD",
    "kv_geometry",
    "bytes_per_token_per_rank",
]


class Confidence(StrEnum):
    """How a reported number was arrived at. Carried by every derived quantity in this package."""

    MEASURED = "measured"
    """Printed by the checkpoint itself, or by a recorded run."""

    DERIVED = "derived"
    """Computed from measured inputs, by arithmetic this module shows."""

    ASSUMED = "assumed"
    """A default was used because the checkpoint did not say. :attr:`KvGeometry.notes` names it."""


SHARDED = "sharded"
REPLICATED = "replicated"
SINGLE_CARD = "single_card"

#: Divisor for :data:`SHARDED` when the caller has not said at what width. TP4 is what every recorded
#: run in this repository uses, and the one the thresholds were measured at.
DEFAULT_TP_WIDTH = 4


@dataclass(frozen=True)
class LayerGeometry:
    """One *class* of layer -- not one layer.

    A 43-layer model with three compress ratios has three entries here, not forty-three, which is
    what makes the difference between a cache that grows with context and one that does not visible
    at a glance.
    """

    count: int
    """How many layers are of this class."""

    kind: str
    """``full`` | ``sliding_window`` | ``linear`` | ``compressed``."""

    values_per_token: int
    """Scalar values one token adds to one such layer. Zero for a ring buffer or a linear layer."""

    source: str
    """The allocation site, as ``file.py:NNN``, so the claim is checkable."""

    compress_ratio: int | None = None
    """Slots per token are ``head_dim // ratio`` for a compressed layer; ``None`` when not applicable."""

    kv_heads: int = 0
    """How many KV heads this class's cache is split across. ``0`` means *the whole class shards*.

    This is what makes the per-rank number a derivation rather than a division. A cache is not always
    a multiple of the world size: Qwen3.8-Flash-Next has two KV heads, and at TP4 ``builder.py:267``
    binds each rank to ``max(1, kv_heads // world)`` of them -- one, not half of one. Dividing the
    model's total by four answers 6912 B/token per rank where the rank actually holds 15360.
    """

    shared_values_per_token: int = 0
    """The part of :attr:`values_per_token` that is *not* split — a second cache with no head axis."""

    note: str = ""

    @property
    def grows(self) -> bool:
        """Whether this class costs anything as the context grows."""
        return self.values_per_token > 0

    @property
    def shardable_values_per_token(self) -> int:
        """The part of this class's cache that a rank holds a share of."""
        return max(0, int(self.values_per_token) - int(self.shared_values_per_token))

    def values_per_token_per_rank(self, tp_width: int) -> int:
        """This class's per-token cost on one rank, before dtype.

        Three terms, and a class that reports no ``kv_heads`` falls to the first: the whole class is
        divided by the world size. A class that does gets the head arithmetic instead -- each rank
        takes ``max(1, kv_heads // world)`` heads and the rest is replicated whole.
        """
        world = max(1, int(tp_width))
        shared = int(self.shared_values_per_token)
        if self.kv_heads <= 1 or world == 1:
            return -(-int(self.values_per_token) // world)
        per_head = self.shardable_values_per_token // int(self.kv_heads)
        if per_head * int(self.kv_heads) != self.shardable_values_per_token:
            # Not a whole number of values per head, so this class cannot be split head-wise and the
            # even division is the conservative reading -- more per rank than a share would be.
            return -(-int(self.values_per_token) // world)
        held = max(1, int(self.kv_heads) // world)
        return per_head * held + shared


@dataclass(frozen=True)
class KvGeometry:
    """What one token costs in KV, and the three rules that decide how it is paid for."""

    attention_kind: str
    """``mla_latent`` | ``gqa`` | ``hybrid`` | ``unknown``."""

    layers: tuple[LayerGeometry, ...]

    values_per_token_per_layer: int
    """Summed over every layer class: the whole model's per-token cost, before dtype and sharding."""

    dtype_bytes: int
    """Bytes per scalar in the allocation -- read from the allocation, not from the model card."""

    sharding: str
    """``sharded`` | ``replicated`` | ``single_card``."""

    sharding_source: str
    """``file.py:NNN`` for the rule, or the reason it had to be assumed."""

    preallocated: bool
    """True when the cache is sized once from ``max_seq_len`` -- i.e. the flag is a prepayment."""

    confidence: Confidence

    extra_caches: tuple[str, ...] = ()
    """Second per-token caches beside the attention one, with their cost, e.g. DeepSeek's Indexer."""

    notes: tuple[str, ...] = ()
    """Every default that was used, in words. Empty means the checkpoint and the code both answered."""

    sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def bytes_per_token_whole_model(self) -> int:
        """Unsharded, undivided: what one token costs the whole model at this dtype."""
        return self.values_per_token_per_layer * self.dtype_bytes

    @property
    def exact(self) -> bool:
        """Whether nothing had to be assumed."""
        return self.confidence is not Confidence.ASSUMED

    def values_per_token_per_rank(self, tp_width: int = DEFAULT_TP_WIDTH) -> int:
        """Values one token costs **on one rank**, summed over the layer classes.

        :data:`REPLICATED` and :data:`SINGLE_CARD` are unaffected by ``tp_width`` by definition --
        every rank holds the whole thing -- which is exactly why the distinction is made here rather
        than by a caller dividing by a world size it guessed.
        """
        if self.sharding != SHARDED:
            return int(self.values_per_token_per_layer)
        return sum(
            int(layer.count) * layer.values_per_token_per_rank(tp_width) for layer in self.layers
        )


def bytes_per_token_per_rank(
    geometry: KvGeometry, *, tp_width: int = DEFAULT_TP_WIDTH
) -> int:
    """Bytes one token costs **on one rank**, at the width the runtime actually runs.

    This is the number a card budget is compared against, and the one a sharding mistake moves by
    the TP width. It is *not* the whole model's cost divided by the world size -- that is the
    approximation :meth:`KvGeometry.values_per_token_per_rank` exists to replace, and it is wrong by
    2.2x on Qwen3.8-Flash-Next, whose two KV heads do not divide by four and whose indexer cache is
    replicated whole.
    """
    return geometry.values_per_token_per_rank(tp_width) * int(geometry.dtype_bytes)


# ---------------------------------------------------------------------------------------------
# Per-architecture derivations
#
# Each builder reads whatever the checkpoint itself states and fills the rest from the allocation
# site in code. A builder that has to fall back says so in `notes`; a builder that reads everything
# from the config leaves `notes` empty, and `Confidence.DERIVED` then means "arithmetic on the
# checkpoint's own numbers" rather than "arithmetic on a guess".
# ---------------------------------------------------------------------------------------------


#: Tensor names that mark a layer as *not* holding a growing KV cache. GGUF and HuggingFace spell
#: the same three things differently, so all the spellings are here rather than in a per-format path.
#: Deliberately excludes ``attn_q``/``q_proj``/``attn_output``: a linear-attention layer has those
#: too (as a fused ``attn_qkv`` and an output projection), so they say nothing about the cache.
_LINEAR_LAYER_MARKERS = ("ssm_", "linear_attn", "gated_delta", "conv1d")
_ATTENTION_LAYER_MARKERS = ("attn_k.", "attn_k_", "k_proj.", "attn_v.", "v_proj.", "self_attn.")

_LAYER_INDEX = re.compile(r"(?:^|\.)(?:blk|layers|h)\.(\d+)\.")


def _cache_owning_layers(names: Iterable[str]) -> tuple[set[int], set[int]] | None:
    """Which layer indices hold a growing KV cache, read from the checkpoint's own tensor names.

    This is the fallback that keeps a GGUF honest. A GGUF carries no ``layer_types`` -- that list
    exists only in the HF config -- so a hybrid checkpoint read from GGUF alone would look uniform,
    and costing every layer as full attention overstates a 48-of-64 linear model by 4x. The tensors
    say otherwise and say it checkably: ``blk.N.ssm_a`` exists exactly on the linear layers and
    ``blk.N.attn_k.weight`` exactly on the attention ones, so counting them is a measurement rather
    than a default.

    Returns ``None`` when the naming is not recognised, so an unfamiliar checkpoint falls through to
    the generic path and is marked assumed rather than silently miscounted.
    """
    linear: set[int] = set()
    full: set[int] = set()
    for name in names:
        found = _LAYER_INDEX.search(name)
        if not found or not any(marker in name for marker in _ATTENTION_LAYER_MARKERS + _LINEAR_LAYER_MARKERS):
            continue
        index = int(found.group(1))
        if any(marker in name for marker in _LINEAR_LAYER_MARKERS):
            linear.add(index)
        else:
            full.add(index)
    if not full and not linear:
        return None
    return full - linear, linear


def _layer_types_from_interval(layers: int, interval: int) -> list[str]:
    """Expand ``full_attention_interval`` into the ``layer_types`` list it stands for.

    The convention is the one Qwen3-Next uses and both checkpoints here agree with: every
    ``interval``-th layer, counting from the first, is full attention and the rest are linear. At 64
    layers and an interval of 4 that is indices 3, 7, ... 63 -- sixteen of them, which is both what
    the HF ``layer_types`` says and what the GGUF's tensors say. Two independent sources, so the
    expansion is a reading rather than a guess.
    """
    return [
        "full_attention" if interval > 0 and (index + 1) % interval == 0 else "linear_attention"
        for index in range(int(layers))
    ]


def _synthesize_layer_types(*, layers: int, owning: tuple[set[int], set[int]]) -> list[str]:
    """Turn tensor evidence into the ``layer_types`` list the hybrid builders already read."""
    full, linear = owning
    types = ["linear_attention"] * int(layers)
    for index in full:
        if 0 <= index < layers:
            types[index] = "full_attention"
    return types


def _sliding_window(config: Mapping[str, Any]) -> int:
    return _int(config, "sliding_window", "window_size", default=128) or 128


def _trim_to_backbone(
    config: Mapping[str, Any], ratios: list[Any], layers: int
) -> tuple[list[Any], str]:
    """Drop the trailing entries ``compress_ratios`` carries for the MTP draft layers.

    DeepSeek lists one ratio per layer *including* the multi-token-prediction heads, so the list is
    longer than ``num_hidden_layers`` -- 46 against 43 for V4-Flash, 43 against 40 for V4.1. Those
    trailing entries are all ``0``, i.e. plain sliding windows, so leaving them in does not change
    the bytes; it does change the *layer counts* a report prints, which is how a reader ends up
    believing a 43-layer model has 46. Dropping them makes the class counts sum to the layer count,
    and a non-zero entry among them is returned as a note rather than trimmed away silently: that
    would mean a draft layer grows a cache the backbone arithmetic is not paying for.
    """
    if layers <= 0 or len(ratios) <= layers:
        return ratios, ""
    dropped = ratios[layers:]
    growing = [r for r in dropped if int(r) != 0]
    note = ""
    if growing:
        note = (
            f"{len(dropped)} entries past num_hidden_layers were trimmed, but {len(growing)} of them "
            f"are non-zero ({growing}); the draft layers may hold a cache this fit does not pay for"
        )
    return ratios[:layers], note


def _from_compress_ratios(
    *,
    ratios: list[Any],
    head_dim: int,
    source: str,
    index_dim: int | None,
    index_on_ratio: int | None,
    min_ratio: int,
) -> tuple[list[LayerGeometry], tuple[str, ...]]:
    """DeepSeek's shape: a per-layer ``compress_ratios`` list, with three kinds of entry in it.

    ``0`` is a pure sliding window -- a ring, no marginal cost. ``min_ratio`` is the smallest ratio
    that still stores one slot per few tokens and is the class that actually grows. Everything else
    is a compressed layer at ``head_dim // ratio``.

    ``index_dim`` adds DeepSeek's second cache: the ``Indexer`` is built only on the layers whose
    ratio is ``index_on_ratio`` (``runtime.py:1146``), and it is sized ``index_dim // ratio`` rather
    than ``head_dim // ratio``.
    """
    classes: dict[tuple[int, str], list[int]] = {}
    for ratio in ratios:
        try:
            ratio = int(ratio)
        except (TypeError, ValueError):
            continue
        if ratio == 0:
            key = (0, "sliding_window")
        elif ratio == min_ratio:
            key = (ratio, "compressed_full")
        else:
            key = (ratio, "compressed")
        classes.setdefault(key, []).append(ratio)

    layers: list[LayerGeometry] = []
    for (ratio, kind), members in sorted(classes.items()):
        if ratio == 0:
            layers.append(
                LayerGeometry(
                    count=len(members),
                    kind="sliding_window",
                    values_per_token=0,
                    source=source,
                    note="ring of window_size slots; no marginal cost per token",
                )
            )
            continue
        layers.append(
            LayerGeometry(
                count=len(members),
                kind=kind,
                values_per_token=head_dim // ratio,
                source=source,
                compress_ratio=ratio,
                note=f"{head_dim} latent values per {ratio} tokens",
            )
        )

    extra: list[str] = []
    if index_dim and index_on_ratio and any(r == index_on_ratio for r in ratios if isinstance(r, int)):
        count = sum(1 for r in ratios if r == index_on_ratio)
        per_token = index_dim // index_on_ratio
        layers.append(
            LayerGeometry(
                count=count,
                kind="indexer",
                values_per_token=per_token,
                source="relicllm/models/deepseek_v4/runtime.py:1146",
                compress_ratio=index_on_ratio,
                note="indexer cache, a second per-token buffer beside the attention cache",
            )
        )
        extra.append(f"indexer: {count} layers x {per_token} values/token")
    return layers, tuple(extra)


def _from_layer_types(
    *,
    types: list[Any],
    full: tuple[str, ...],
    linear: tuple[str, ...],
    full_values: int,
    full_source: str,
    windowed: tuple[str, ...] = (),
    full_kv_heads: int = 0,
    full_shared_values: int = 0,
) -> tuple[list[LayerGeometry], tuple[str, ...]]:
    """The hybrid shape: ``config.layer_types`` naming which layers hold a cache at all."""
    notes: list[str] = []
    buckets: dict[str, int] = {}
    for raw in types:
        name = str(raw)
        if name in full:
            buckets["full"] = buckets.get("full", 0) + 1
        elif name in windowed:
            buckets["sliding_window"] = buckets.get("sliding_window", 0) + 1
        elif name in linear:
            buckets["linear"] = buckets.get("linear", 0) + 1
        else:
            buckets["unknown"] = buckets.get("unknown", 0) + 1
    if buckets.get("unknown"):
        notes.append(
            f"{buckets['unknown']} layers carry a layer_type this build does not classify; "
            "they are counted as full attention, which overstates the cache"
        )
        buckets["full"] = buckets.get("full", 0) + buckets.pop("unknown")

    layers: list[LayerGeometry] = []
    if buckets.get("full"):
        layers.append(
            LayerGeometry(
                count=buckets["full"],
                kind="full",
                values_per_token=full_values,
                source=full_source,
                kv_heads=full_kv_heads,
                shared_values_per_token=full_shared_values,
            )
        )
    if buckets.get("sliding_window"):
        layers.append(
            LayerGeometry(
                count=buckets["sliding_window"],
                kind="sliding_window",
                values_per_token=0,
                source=full_source,
                note="ring buffer; no marginal cost per token",
            )
        )
    if buckets.get("linear"):
        layers.append(
            LayerGeometry(
                count=buckets["linear"],
                kind="linear",
                values_per_token=0,
                source=full_source,
                note="linear attention: fixed recurrent state, no per-token KV",
            )
        )
    return layers, tuple(notes)


def _gqa_values(kv_heads: int, head_dim: int) -> int:
    """K and V are stored separately: ``2 * heads * dim`` per token per layer."""
    return 2 * int(kv_heads) * int(head_dim)


def kv_geometry(
    config: Mapping[str, Any] | None,
    *,
    architecture: str | None = None,
    layers: int | None = None,
    tensor_names: Iterable[str] | None = None,
) -> KvGeometry:
    """Derive what one token costs in KV for this checkpoint.

    ``config`` is the checkpoint's own ``config.json`` (HF) or its GGUF metadata; a GGUF table is
    flattened to bare keys first (:func:`_flatten_gguf`). ``architecture`` is ``general.architecture``
    from a GGUF or the HF ``model_type``, and is only used to pick a builder -- the numbers come from
    ``config`` wherever ``config`` carries them.

    ``tensor_names`` is optional and is the checkpoint's own tensor names, from either format. It
    matters for exactly one case and settles it: a GGUF has no ``layer_types``, so a hybrid model
    read from GGUF alone is indistinguishable from a uniform one, and costing every layer as full
    attention overstates Qwen3.8-27B or Bonsai by 4x. The tensors name the linear layers directly
    (:func:`_cache_owning_layers`), which turns that 4x from an assumption into a measurement.

    An unrecognised architecture is not an error: the generic path reads the head counts it can find
    and marks the result :attr:`Confidence.ASSUMED`, which is what stops a caller treating a guess
    as a measurement.
    """
    config = _flatten_gguf(config or {})
    arch = (architecture or _architecture_of(config) or "").lower()
    layers = layers or _int(config, "num_hidden_layers", "n_layers", "block_count", default=0) or 0
    head_dim = _int(config, "head_dim", "attention.key_length", default=128) or 128

    canonical = _canonical(arch)

    trunk_note = ""
    if canonical in _TRUNK_ONLY_ARCHITECTURES and layers:
        draft = _int(config, "nextn_predict_layers", "num_nextn_predict_layers", default=0) or 0
        if draft:
            trunk, layers = layers, max(1, int(layers) - draft)
            trunk_note = (
                f"the layer count is the trunk: {trunk} block(s) are declared, {draft} of them are "
                f"trailing NextN/MTP blocks that this runtime does not build, and no cache is "
                f"allocated for a block that is not built ({_TRUNK_ONLY_ARCHITECTURES[canonical]})"
            )

    evidence_note = ""
    if (
        tensor_names is not None
        and canonical in _HYBRID_ARCHITECTURES
        and _list(config, "layer_types") is None
        and _list(config, "hybrid_layer_pattern") is None
        and _int(config, "full_attention_interval") is None
    ):
        owning = _cache_owning_layers(tensor_names)
        if owning is not None and layers:
            config = {**config, "layer_types": _synthesize_layer_types(layers=layers, owning=owning)}
            evidence_note = (
                f"the config states no layer_types; the {len(owning[0])} full-attention and "
                f"{len(owning[1])} linear layers were counted from the checkpoint's own tensor names "
                "(ssm_a/linear_attn vs attn_k/k_proj), not assumed"
            )

    builder = _BUILDERS.get(canonical)
    if builder is None:
        geometry = _generic(config, architecture=arch, layers=layers, head_dim=head_dim)
    else:
        geometry = builder(config, layers=layers, head_dim=head_dim)
        if geometry is None:
            geometry = _generic(config, architecture=arch, layers=layers, head_dim=head_dim)
    for note in (trunk_note, evidence_note):
        if note:
            geometry = replace(geometry, notes=geometry.notes + (note,))
    return geometry


#: Architectures whose runtime builds the *trunk* only, dropping the trailing NextN/MTP blocks a
#: GGUF's ``block_count`` -- and, on this roster, the HF config's ``num_nextn_predict_layers`` --
#: still counts. The value is the ``file.py:NNN`` that subtracts them.
#:
#: This is per-architecture because the runtimes genuinely disagree, and the disagreement is worth
#: 2.5% of one model's cache and 6.25% of another's. GLM-DSA says so in its own comment
#: ("``params.n_layers`` is the physical block_count (e.g. 79) ... The trunk depth is
#: block_count - nextn_predict_layers") and Xing4's GGUF model takes its block count from the
#: checkpoint's ``config.json``, where ``num_hidden_layers`` is 40 against the GGUF's 41. MiMo is the
#: counter-example that keeps this a table rather than a rule: its checkpoint also declares
#: ``num_nextn_predict_layers: 3``, and every cache it builds ranges over ``range(num_hidden_layers)``
#: -- all 48 of them.
_TRUNK_ONLY_ARCHITECTURES = {
    "glm_dsa": "relicllm/models/glm_dsa/architecture.py:159",
    "xing4": "relicllm/models/xing4_0/gguf_model.py:78",
}


#: The builders that branch on ``layer_types``. Only these can be helped by counting the cache-owning
#: layers from tensor names, and only these should be told about it -- synthesising the list for a
#: uniform model like GLM-DSA would add a note about a split that does not exist.
_HYBRID_ARCHITECTURES = frozenset({"qwen3_5", "qwen4_exp", "mimo"})


def _build_deepseek_v4(config: Mapping[str, Any], *, layers: int, head_dim: int) -> KvGeometry | None:
    """DeepSeek-V4-Flash: MLA latent, per-layer compress ratios, plus an indexer cache.

    The allocation is ``window_size + max_seq_len // compress_ratio`` slots of ``head_dim``
    (``relicllm/models/deepseek_v4/runtime.py:1408-1409``), sized once at load from ``max_seq_len``
    -- so ``--max-model-len`` is a prepayment here. Every rank allocates the whole thing: the
    allocation carries no rank term, and ``self.n_local_heads = args.n_heads // tp_world_size``
    (:1309) divides the attention heads while ``self.head_dim`` (:1301) does not.
    """
    ratios = _list(config, "compress_ratios")
    head_dim = _int(config, "head_dim", default=head_dim) or head_dim
    if not ratios:
        return None
    ratios, trim_note = _trim_to_backbone(config, ratios, layers)
    classes, extra = _from_compress_ratios(
        ratios=ratios,
        head_dim=head_dim,
        source="relicllm/models/deepseek_v4/runtime.py:1408",
        index_dim=_int(config, "index_head_dim", default=0),
        index_on_ratio=4,
        min_ratio=1,
    )
    return KvGeometry(
        attention_kind="mla_latent",
        layers=tuple(classes),
        values_per_token_per_layer=sum(c.count * c.values_per_token for c in classes),
        dtype_bytes=2,
        sharding=REPLICATED,
        sharding_source="relicllm/models/deepseek_v4/runtime.py:1408 — the allocation carries no rank "
        "term; head_dim is not divided by tp_world_size (:1301) while the heads are (:1309)",
        preallocated=True,
        confidence=Confidence.DERIVED,
        extra_caches=extra,
        notes=(
            "max_batch_size defaults to 4 (runtime.py:504), so the cache is allocated for four rows "
            "even though the serving path runs one",
        )
        + ((trim_note,) if trim_note else ()),
        sources=("relicllm/models/deepseek_v4/runtime.py:1408", "relicllm/models/deepseek_v4/runtime.py:504"),
    )


def _build_deepseek_v4_1(config: Mapping[str, Any], *, layers: int, head_dim: int) -> KvGeometry | None:
    """DeepSeek-V4.1-Flash: same compressor shape, but **only a few layers own a growing buffer**.

    This is the sharpest case in the roster, and the one a ratio-only reading gets wrong. V4.1 has
    the same ``compress_ratios`` idea as V4, but a non-zero ratio here means the layer *reads* a
    compressed cache rather than writing one: of its 40 layers, 36 are consumer-only sliding windows
    and just four — the ``kv_source_layer_ids`` — own a growing buffer. Reading the ratio list alone
    counts 38 growing layers where there are 4.

    The indexer is guarded the same way and for the same reason: its ``k_cache`` is registered inside
    ``if self.owns_k`` (``attention.py:809``), and ``owns_k`` is ``layer_id in kv_sources`` (:758), so
    the eight ``index_source_layer_ids`` do *not* each hold a cache — only the four that also source
    the compressor do, and the other four read the keys from those.

    ``CACHE_DTYPE = torch.bfloat16`` (:104). Every rank holds a whole copy: the constructor divides
    ``n_heads`` and ``o_groups`` by ``world`` (:1294) and neither is a cache.
    """
    ratios = _list(config, "compress_ratios")
    sources = _list(config, "kv_source_layer_ids")
    head_dim = _int(config, "head_dim", default=head_dim) or head_dim
    index_dim = _int(config, "index_head_dim", default=0) or 0
    if not ratios or not sources:
        return None
    ratios, trim_note = _trim_to_backbone(config, ratios, layers)
    ratio_at = {int(i): int(r) for i, r in enumerate(ratios)}
    owning = sorted(index for index in (int(i) for i in sources) if index < len(ratios))

    per_token = sum(head_dim // ratio_at[i] for i in owning if ratio_at.get(i))
    index_values = sum(index_dim // ratio_at[i] for i in owning if ratio_at.get(i) and index_dim)
    layers = [
        LayerGeometry(
            count=len(owning),
            kind="compressed_full",
            values_per_token=per_token // len(owning),
            source="relicllm/models/deepseek_v4_1/attention.py:1341",
            note=f"kv_source layers {owning}; the other {len(ratios) - len(owning)} read them",
        ),
        LayerGeometry(
            count=max(0, len(ratios) - len(owning)),
            kind="sliding_window",
            values_per_token=0,
            source="relicllm/models/deepseek_v4_1/attention.py:1341",
            note="consumer-only: a 128-slot ring, no growing buffer",
        ),
    ]
    if index_values:
        layers.append(
            LayerGeometry(
                count=len(owning),
                kind="indexer",
                values_per_token=index_values // len(owning),
                source="relicllm/models/deepseek_v4_1/attention.py:809",
                note="indexer key cache, registered inside `if self.owns_k`",
            )
        )
    extra = (f"indexer: {len(owning)} layers x {index_values // len(owning)} values/token",) if index_values else ()
    return KvGeometry(
        attention_kind="mla_latent",
        layers=tuple(layers),
        values_per_token_per_layer=per_token + index_values,
        dtype_bytes=2,
        sharding=REPLICATED,
        sharding_source="relicllm/models/deepseek_v4_1/attention.py:1294 — the constructor divides "
        "n_heads and o_groups by world, and neither is a cache",
        preallocated=True,
        confidence=Confidence.DERIVED,
        extra_caches=extra,
        notes=(
            f"only the {len(owning)} kv_source layers own a growing buffer; a ratio list alone would "
            f"count {sum(1 for r in ratios if r)} of them",
        )
        + ((trim_note,) if trim_note else ()),
        sources=("relicllm/models/deepseek_v4_1/attention.py:1341", "relicllm/models/deepseek_v4_1/attention.py:809"),
    )


def _build_mimo(config: Mapping[str, Any], *, layers: int, head_dim: int) -> KvGeometry | None:
    """MiMo-V2.6-Flash: GQA over a hybrid stack, and the one runtime that shards its cache.

    K and V have *different* head widths here (192 and 128), so they are read separately rather than
    assumed equal. The sharding is real and is decided in ``device_model.py``: ``shards``/``shard``
    come from the expert-parallel group's ``attention_shards``/``attention_shard`` (:516-523), and
    the cache is built with them (:685-698), so at TP4 each rank keeps one of the four KV heads.
    """
    kv_heads = _int(config, "num_key_value_heads", "attention.head_count_kv", default=0) or 0
    key_dim = _int(config, "head_dim", "attention.key_length", default=head_dim) or head_dim
    value_dim = _int(config, "v_head_dim", "attention.value_length", default=key_dim) or key_dim
    types = _list(config, "layer_types")
    full_values = int(kv_heads) * (int(key_dim) + int(value_dim))
    source = "relicllm/models/mimo_v2/device_attention.py:694"
    if types:
        classes, notes = _from_layer_types(
            types=types,
            full=("full_attention", "global_attention", "full"),
            linear=(),
            windowed=("sliding_attention", "sliding_window"),
            full_values=full_values,
            full_source=source,
        )
    else:
        pattern = _list(config, "hybrid_layer_pattern")
        global_count = sum(1 for flag in pattern if not int(flag)) if pattern else 0
        notes = [
            "layer_types absent; the split came from hybrid_layer_pattern, where **1 means "
            "sliding-window and 0 means global** (mimo_v2/config.py:342) -- reading it the other "
            "way inverts the model"
        ]
        if not pattern:
            # config.py:352 -- an absent pattern is all-zero, i.e. every layer is global. That is
            # the *pessimistic* reading, which is the right direction for a fit test.
            global_count = int(layers)
            notes = ["neither layer_types nor hybrid_layer_pattern is stated; config.py:352 resolves "
                     "that to all-global attention, which overstates the cache"]
        classes = [
            LayerGeometry(global_count, "full", full_values, source),
            LayerGeometry(max(0, int(layers) - global_count), "sliding_window", 0, source,
                          note="ring buffer of `sliding_window` slots; no marginal cost per token"),
        ]
    return KvGeometry(
        attention_kind="hybrid",
        layers=tuple(classes),
        values_per_token_per_layer=sum(c.count * c.values_per_token for c in classes),
        dtype_bytes=2,
        sharding=SHARDED,
        sharding_source="relicllm/models/mimo_v2/device_model.py:516 — shards/shard come from the "
        "expert-parallel group's attention_shards and are passed into the cache constructor (:685)",
        preallocated=True,
        confidence=Confidence.DERIVED,
        notes=notes,
        sources=(source, "relicllm/models/mimo_v2/device_model.py:516"),
    )


def _build_minimax(config: Mapping[str, Any], *, layers: int, head_dim: int) -> KvGeometry:
    """MiniMax-M2.7: uniform GQA, and a cache that is replicated even though the model is tensor-parallel.

    ``reset_cache`` allocates the *full* ``n_kv_heads`` with no rank division
    (``relicllm/models/minimax_m2/architecture.py:123-128``), and the only things TP shards are the
    routed experts and ``lm_head``. So at TP4 the KV cost is four times what dividing by the world
    size would give — and the model page agrees with the code, saying a full-length request does not
    fit the 4x22 GiB baseline.

    The cache is sized per request, from ``prompt + max_new_tokens`` (``runtime/generation.py:93``),
    not from a context field.
    """
    kv_heads = _int(config, "n_kv_heads", "num_key_value_heads", "attention.head_count_kv", default=8) or 8
    head_dim = _int(config, "head_dim", "attention.key_length", default=head_dim) or head_dim
    values = _gqa_values(kv_heads, head_dim)
    return KvGeometry(
        attention_kind="gqa",
        layers=(LayerGeometry(int(layers), "full", values, "relicllm/models/minimax_m2/architecture.py:123"),),
        values_per_token_per_layer=int(layers) * values,
        dtype_bytes=2,
        sharding=REPLICATED,
        sharding_source="relicllm/models/minimax_m2/architecture.py:123 — the allocation uses the full "
        "n_kv_heads and there is no rank division anywhere in the attention",
        preallocated=False,
        confidence=Confidence.DERIVED,
        notes=("the cache is sized per request from prompt + max_new_tokens, not from a context field",),
        sources=("relicllm/models/minimax_m2/architecture.py:123",),
    )


def _build_glm_dsa(config: Mapping[str, Any], *, layers: int, head_dim: int) -> KvGeometry:
    """GLM-5.2 (glm-dsa): an MLA model whose runtime allocates a per-head cache.

    **This builder reports what the code does, not what the architecture says, and the difference is
    recorded rather than smoothed over.** The checkpoint declares ``attention.head_count_kv = 1`` and
    carries a 512-wide ``kv_lora_rank`` plus a 64-wide rope — i.e. one shared latent per token, which
    is what MLA means and what the model-side helper computes (179,712 B/token). The runtime instead
    allocates ``n_heads`` (=64) key and value heads of ``key_length_mla``/``value_length_mla``
    (``relicllm/models/glm_dsa/architecture.py:300-308``), which is 5,177,344 B/token — a **56.9x**
    over-allocation.

    The call is to report the code, because the code is what has to fit on a card. The discrepancy is
    carried in :attr:`KvGeometry.notes` so that fixing the allocation turns this into a visible,
    deliberate change rather than a silent one.
    """
    n_heads = _int(config, "n_heads", "attention.head_count", default=64) or 64
    # Every MLA width is namespaced in a GGUF and at the root in an HF config, and all of them have
    # to be asked for both ways. `key_length_mla` was read by its bare name only, which succeeds on
    # an HF config and silently takes the default on a GGUF -- and because GLM-5.2's declared widths
    # *are* 256 and 256, the default answered correctly and the miss was invisible. A GLM whose MLA
    # widths differ would have been priced at 256 regardless of what it said.
    key_mla = _int(config, "key_length_mla", "attention.key_length_mla", default=256) or 0
    value_mla = _int(config, "value_length_mla", "attention.value_length_mla", default=256) or 0
    # `attention.kv_lora_rank`, not `kv_lora_rank`: a GGUF carries the MLA width under the
    # attention namespace, where the HF config has it at the root. Reading only the HF spelling
    # leaves `kv_lora` at its default and silently drops the discrepancy note -- which is exactly
    # the failure this builder exists to make visible.
    kv_lora = _int(config, "kv_lora_rank", "attention.kv_lora_rank", default=0) or 0
    rope = _int(config, "rope_dim", "rope.dimension_count", default=0) or 0

    code_values = n_heads * (key_mla + value_mla)
    if kv_lora and rope:
        declared_values = kv_lora + rope
        ratio = code_values / declared_values if declared_values else 0.0
        discrepancy = (
            f"the runtime allocates a per-head cache ({n_heads} x ({key_mla}+{value_mla}) = "
            f"{code_values} values/token/layer) while the checkpoint declares one shared latent "
            f"({kv_lora}+{rope} = {declared_values}), a {ratio:.1f}x over-allocation. This report "
            f"uses the runtime's number because the runtime's number is what must fit on a card."
        )
    else:
        discrepancy = ""
    return KvGeometry(
        attention_kind="mla_latent",
        layers=(
            LayerGeometry(
                int(layers),
                "full",
                code_values,
                "relicllm/models/glm_dsa/architecture.py:300",
            ),
        ),
        values_per_token_per_layer=int(layers) * code_values,
        dtype_bytes=2,
        sharding=REPLICATED,
        sharding_source="relicllm/models/glm_dsa/architecture.py:264 — the attention takes no "
        "rank or world argument, and gguf_model.py:337 builds it without one",
        preallocated=False,
        confidence=Confidence.DERIVED,
        notes=tuple(n for n in (discrepancy, "the cache is sized per request, not from a context field") if n),
        sources=("relicllm/models/glm_dsa/architecture.py:300",),
    )


def _build_qwen4_exp(config: Mapping[str, Any], *, layers: int, head_dim: int) -> KvGeometry | None:
    """Qwen3.8-Flash-Next: 12 attention layers and 36 linear ones — a quarter of the layers, at most.

    The KV is sharded, which is the opposite of the three models above: ``builder.py:267-289``
    computes ``q_per_rank`` per rank and binds each rank to one of the two KV heads. The cache is
    sized per request (``runtime.py:286``).
    """
    types = _list(config, "layer_types")
    kv_heads = _int(config, "num_key_value_heads", "attention.head_count_kv", default=0) or 0
    head_dim = _int(config, "head_dim", "attention.key_length", default=head_dim) or head_dim
    # The key is `indexer_head_dim`, not `index_head_dim` -- and the difference is the whole
    # indexer. `QSAAttentionCache.__init__` registers `index_k` as
    # `(batch, max_seq_len, config.indexer_head_dim)` at attention.py:455-457, un-divided by any
    # compress ratio, so it is 128 values per token per QSA layer -- 11% of that layer's cost, and
    # silently absent if the key is misread.
    index_dim = _int(config, "indexer_head_dim", default=0) or 0
    full_values = _gqa_values(kv_heads, head_dim) + index_dim
    source = "relicllm/models/qwen4_exp/attention.py:453"
    if not types:
        return None
    classes, notes = _from_layer_types(
        types=types,
        full=("full_attention", "qwen_sparse_attention"),
        linear=("linear_attention",),
        full_values=full_values,
        full_source=source,
        full_kv_heads=kv_heads,
        # `QSAAttentionCache.index_k` is `(batch, max_seq_len, indexer_head_dim)` -- one tensor per
        # rank, with no head axis to divide and no rank term in its shape
        # (`attention.py:455-457`), and `builder.py:282-283` says so in words. So the indexer is
        # replicated even though the cache around it is sharded, and it is 20% of this model's
        # per-rank cost.
        full_shared_values=index_dim,
    )
    return KvGeometry(
        attention_kind="hybrid",
        layers=tuple(classes),
        values_per_token_per_layer=sum(c.count * c.values_per_token for c in classes),
        dtype_bytes=2,
        sharding=SHARDED,
        sharding_source="relicllm/models/qwen4_exp/builder.py:267 — q_per_rank is computed per rank "
        "and each rank holds the KV heads its query heads bind to",
        preallocated=False,
        confidence=Confidence.DERIVED,
        notes=notes + ("the cache is sized per request from prompt_len + max_new_tokens + 1",),
        sources=(source, "relicllm/models/qwen4_exp/builder.py:267"),
    )


def _build_xing4(config: Mapping[str, Any], *, layers: int, head_dim: int) -> KvGeometry:
    """Xing4.0-29B-A4B: absorbed MLA — one 576-wide latent per token per layer, on a single card.

    Not a K/V pair: the value is ``latent @ v_b`` and is only ever needed after the attention
    weights, so there is one tensor, ``kv_lora_rank + qk_rope_head_dim`` wide
    (``relicllm/models/xing4_0/attention.py:205``). The runtime has no tensor-parallel path at all
    (``xing4_backend.py`` refuses ``--tensor-parallel-size > 1``), so the width is not a variable.
    """
    kv_lora = _int(config, "kv_lora_rank", default=512) or 512
    rope = _int(config, "qk_rope_head_dim", "rope.dimension_count", default=64) or 64
    values = kv_lora + rope
    return KvGeometry(
        attention_kind="mla_latent",
        layers=(LayerGeometry(int(layers), "full", values, "relicllm/models/xing4_0/attention.py:205"),),
        values_per_token_per_layer=int(layers) * values,
        dtype_bytes=2,
        sharding=SINGLE_CARD,
        sharding_source="relicllm/backends/xing4_backend.py:12 — there is no tensor-parallel group "
        "and a width above 1 is refused, so one process holds the cache whole",
        preallocated=True,
        confidence=Confidence.DERIVED,
        notes=(
            "the cache is allocated at the context the backend resolves at load, so --max-model-len "
            "is a prepayment",
        ),
        sources=("relicllm/models/xing4_0/attention.py:205",),
    )


def _build_qwen3_5(config: Mapping[str, Any], *, layers: int, head_dim: int) -> KvGeometry | None:
    """Qwen3.8-27B: 16 full-attention layers out of 64; the other 48 hold a fixed recurrent state.

    Same hybrid shape as the Flash-Next build, and the same GQA geometry (24 query heads over 4 KV
    heads, so TP4 gives each rank exactly one KV head).

    **Ternary-Bonsai-2-27B is this same architecture**, which is not obvious from its name: its GGUF
    declares ``general.architecture = qwen35`` and its tensors are the same 16 ``attn_k`` layers
    beside 48 ``ssm_a`` ones. So one builder answers both, and the tensor-evidence path below is what
    makes the GGUF case work at all -- a GGUF carries no ``layer_types``.
    """
    types = _list(config, "layer_types")
    kv_heads = _int(config, "num_key_value_heads", "attention.head_count_kv", default=0) or 0
    head_dim = _int(config, "head_dim", "attention.key_length", default=head_dim) or head_dim
    full_values = _gqa_values(kv_heads, head_dim)
    source = "relicllm/models/qwen4_exp/builder.py:265"
    interval_note = ""
    if not types:
        # A GGUF states the same split a different way, and states it in the metadata rather than in
        # the tensor names: `qwen35.full_attention_interval = 4`. Reading it here keeps the answer
        # DERIVED from the checkpoint's own declaration instead of falling back to counting tensors.
        interval = _int(config, "full_attention_interval")
        if interval:
            types = _layer_types_from_interval(int(layers), int(interval))
            interval_note = (
                f"layer_types absent; every {interval}th layer is full attention, per the "
                "checkpoint's own full_attention_interval"
            )
    if not types:
        return None
    classes, notes = _from_layer_types(
        types=types,
        full=("full_attention",),
        linear=("linear_attention",),
        full_values=full_values,
        full_source=source,
    )
    if interval_note:
        notes = notes + (interval_note,)
    return KvGeometry(
        attention_kind="hybrid",
        layers=tuple(classes),
        values_per_token_per_layer=sum(c.count * c.values_per_token for c in classes),
        dtype_bytes=2,
        sharding=SHARDED,
        sharding_source="relicllm/models/qwen4_exp/builder.py:265 — the same 24/4 GQA geometry binds "
        "each rank to one KV head",
        preallocated=True,
        confidence=Confidence.DERIVED,
        notes=notes,
        sources=(source,),
    )


def _generic(
    config: Mapping[str, Any], *, architecture: str, layers: int, head_dim: int
) -> KvGeometry:
    """The fallback: read what is there, assume the rest, and say so.

    A model whose layers are all full attention is the *conservative* reading -- it overstates the
    cache, so a fit test that passes here passes under the truth too. That is why this is a
    defensible default and why it is still :attr:`Confidence.ASSUMED`: it is safe to fail on and not
    safe to quote.
    """
    kv_heads = _int(config, "num_key_value_heads", "n_kv_heads", "attention.head_count_kv", default=0)
    head_dim = _int(config, "head_dim", "attention.key_length", default=head_dim) or head_dim
    notes = ["architecture not recognised: every layer is counted as full attention"]
    if kv_heads:
        values = _gqa_values(int(kv_heads), head_dim)
        attention_kind = "gqa"
    else:
        values = head_dim
        attention_kind = "unknown"
        notes.append("no KV-head count found: the cache is costed as one head of head_dim per token")
    notes.append("assumed replicated across ranks, which is the conservative direction")
    return KvGeometry(
        attention_kind=attention_kind,
        layers=(LayerGeometry(max(0, int(layers)), "full", values, "assumed"),),
        values_per_token_per_layer=max(0, int(layers)) * values,
        dtype_bytes=2,
        sharding=REPLICATED,
        sharding_source="assumed: unknown architecture, and replication is the conservative direction",
        preallocated=True,
        confidence=Confidence.ASSUMED,
        notes=tuple(notes),
        sources=(),
    )


#: Architecture key -> builder. A builder returns ``None`` when the checkpoint does not carry what
#: it needs, and :func:`kv_geometry` then falls through to :func:`_generic` rather than half-answering.
_BUILDERS = {
    "deepseek_v4": _build_deepseek_v4,
    "deepseek_v4_1": _build_deepseek_v4_1,
    "mimo": _build_mimo,
    "minimax": _build_minimax,
    "glm_dsa": _build_glm_dsa,
    "qwen4_exp": _build_qwen4_exp,
    "xing4": _build_xing4,
    "qwen3_5": _build_qwen3_5,
}
