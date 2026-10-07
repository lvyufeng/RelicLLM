"""DeepSeek-V4-Flash's KV declaration: a per-layer compress ratio, plus an indexer key cache.

Three kinds of entry in one ``compress_ratios`` list, and each is a different cache. ``0`` is a pure
sliding window -- a ring, no marginal cost. Any other ratio is a compressed MLA latent: ``ratio``
consecutive tokens pool into one ``head_dim``-wide latent, so a token costs ``head_dim // ratio``
values. The **indexer** is a second, narrower cache, built only on the layers whose ratio is 4
(``runtime.py:1146``).

The compression is the whole saving of MLA and the reason the declaration is not a K/V pair:
:class:`~relicllm.runtime.kv_spec.MLASpec` carries no value tensor, and ``head_dim // compress_ratio``
is the one formula that answers an absorbed cache (ratio 1) and a compressed one alike.

Two notes the numbers alone do not show. ``max_batch_size`` defaults to 4 (``runtime.py:504``), so the
runtime allocates four rows where serving uses one -- the declaration describes one row and says so.
And the allocation carries no rank term: ``head_dim`` is not divided by the tensor-parallel width
while the heads are (``runtime.py:1301`` vs ``:1309``), so every rank holds the whole cache.
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
#: The indexer is built on the layers whose compress ratio is this one (`runtime.py:1146`).
_INDEX_ON_RATIO = 4
_INDEX_NAME = "k_cache"


def _ratios(config: Mapping[str, Any]) -> list[int]:
    """The compress ratios, cut to the trunk layer count the checkpoint declares.

    ``compress_ratios`` covers the backbone *plus* the MTP/draft layers, and the runtime allocates a
    cache for the trunk only -- 46 ratios for a 43-layer model. ``declared`` is the same
    ``num_hidden_layers`` the module is built from, so it is where the list stops; a config that
    states no layer count leaves the list whole rather than guessing a cut.
    """
    ratios = [int(value) for value in (config.get("compress_ratios") or ())]
    declared = int(config.get("num_hidden_layers") or 0)
    if declared and len(ratios) > declared:
        ratios = ratios[:declared]
    return ratios


def _group(ratios: list[int]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for layer, ratio in enumerate(ratios):
        groups.setdefault(ratio, []).append(layer)
    return groups


def kv_spec(config: Mapping[str, Any]) -> tuple[KVCacheSpec, ...]:
    """A spec per compress ratio, plus the indexer cache on the layers that own one."""
    config = spec_config(config)
    ratios = _ratios(config)
    if not ratios:
        return ()
    head_dim = int(config.get("head_dim") or _DEFAULT_HEAD_DIM)
    index_dim = int(config.get("index_head_dim") or 0)
    window = int(config.get("window_size") or _DEFAULT_WINDOW)

    specs: list[KVCacheSpec] = []
    for ratio, layers in sorted(_group(ratios).items()):
        if ratio == 0:
            specs.append(
                SlidingWindowSpec(
                    name="window_kv_cache",
                    layer_ids=tuple(layers),
                    dtype="bfloat16",
                    num_kv_heads=1,
                    head_dim=head_dim,
                    v_head_dim=0,
                    sliding_window=window,
                )
            )
            continue
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
        if index_dim and ratio == _INDEX_ON_RATIO:
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
    return tuple(specs)