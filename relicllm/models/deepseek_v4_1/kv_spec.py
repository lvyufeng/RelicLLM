"""DeepSeek-V4.1-Flash's KV declaration: only a few layers own a *growing* buffer.

The same compressor shape as V4 and the sharpest case in the roster, because a ratio list read alone
gets it backwards. V4.1 has ``compress_ratios`` like V4, but a non-zero ratio here means the layer
*reads* a compressed cache rather than writing one: of 40 layers, just four -- the
``kv_source_layer_ids`` -- own a growing buffer, and costing every non-zero ratio as a growing layer
counts 38 where there are 4.

**Every** layer registers a ``window_kv_cache`` (``attention.py:1341``, unconditional), yet the window
spec below names only the 36 *non-owning* layers. That is a deliberate gap, not an oversight: naming
all forty would put the four source layers in two cache classes at once, and `KvGeometry.layers` sums
per-class counts back to the trunk depth, so a layer counted twice reads as a 44-layer model. The
allocated ring on those four layers is therefore unstated here; stating it needs the declaration to
express "each layer in exactly one class by default, plus this layer also in that one", which is a
triage-side change rather than a publisher-side one.

The indexer is guarded and follows the same list: its ``k_cache`` is registered inside
``if self.owns_k`` (``attention.py:809``), and ``owns_k`` is membership of the source list, so only
the four source layers hold one.

Every rank holds a whole copy. The constructor divides ``n_heads`` and ``o_groups`` by the world size
(``attention.py:1294``) and neither is a cache, so the declaration's sharding is replication and it is
not a variable here.
"""

from __future__ import annotations

from typing import Any, Mapping

from relicllm.runtime.kv_spec import (
    KVCacheSpec,
    MLASpec,
    SlidingWindowSpec,
    spec_config,
)

_DEFAULT_HEAD_DIM = 512
_DEFAULT_WINDOW = 128
_INDEX_NAME = "k_cache"


def _ratios(config: Mapping[str, Any]) -> list[int]:
    """The compress ratios, cut to the trunk layer count the checkpoint declares.

    ``compress_ratios`` covers the backbone plus the MTP/draft layers, and the runtime allocates a
    cache for the trunk only -- 43 ratios for a 40-layer model, 46 for a 43-layer one. ``declared``
    is the same ``num_hidden_layers`` the module is built from, so it is where the list stops; a
    config that states no layer count leaves the list whole rather than guessing a cut.
    """
    ratios = [int(value) for value in (config.get("compress_ratios") or ())]
    declared = int(config.get("num_hidden_layers") or 0)
    if declared and len(ratios) > declared:
        ratios = ratios[:declared]
    return ratios


def _owning_layers(config: Mapping[str, Any], ratios: list[int]) -> list[int]:
    """The layers that source a growing cache, in order and within the ratio list's reach."""
    sources = [int(value) for value in (config.get("kv_source_layer_ids") or ())]
    return sorted(index for index in sources if 0 <= index < len(ratios))


def kv_spec(config: Mapping[str, Any]) -> tuple[KVCacheSpec, ...]:
    """A compressed spec per ratio among the source layers, plus their indexer key cache."""
    config = spec_config(config)
    ratios = _ratios(config)
    if not ratios:
        return ()
    head_dim = int(config.get("head_dim") or _DEFAULT_HEAD_DIM)
    index_dim = int(config.get("index_head_dim") or 0)
    window = int(config.get("window_size") or _DEFAULT_WINDOW)
    owning = set(_owning_layers(config, ratios))

    specs: list[KVCacheSpec] = []
    groups: dict[int, list[int]] = {}
    for layer, ratio in enumerate(ratios):
        if layer in owning and ratio:
            groups.setdefault(ratio, []).append(layer)
    for ratio, layers in sorted(groups.items()):
        specs.append(
            MLASpec(
                name="compress_kv_cache",
                layer_ids=tuple(layers),
                dtype="bfloat16",
                num_kv_heads=1,
                head_dim=head_dim,
                v_head_dim=0,
                compress_ratio=ratio,
            )
        )
        if index_dim:
            specs.append(
                MLASpec(
                    name=_INDEX_NAME,
                    layer_ids=tuple(layers),
                    dtype="bfloat16",
                    num_kv_heads=1,
                    head_dim=index_dim,
                    v_head_dim=0,
                    compress_ratio=ratio,
                )
            )

    # Every layer registers a `window_kv_cache` (`attention.py:1341`), source or not. Declaring it on
    # only the *non-owning* layers keeps this cache class disjoint from the compressed one, which is
    # what `KvGeometry.layers` assumes when it sums per-class counts back to the trunk depth. The
    # owning layers' rings are therefore not named here; see the module docstring for why that is a
    # known, deliberate gap rather than an oversight.
    consuming = tuple(layer for layer in range(len(ratios)) if layer not in owning)
    if consuming:
        specs.append(
            SlidingWindowSpec(
                name="window_kv_cache",
                layer_ids=consuming,
                dtype="bfloat16",
                num_kv_heads=1,
                head_dim=head_dim,
                v_head_dim=0,
                sliding_window=window,
            )
        )
    return tuple(specs)