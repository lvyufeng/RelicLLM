"""MiMo-V2.6-Flash's KV declaration: a hybrid stack, and the one runtime that shards a cache.

Two cache classes, because the model is genuinely two. A **global** layer keeps keys 192 wide and
values 128 wide -- the widths differ, which is why the spec carries ``v_head_dim`` separately -- and
grows one row a token. A **sliding-window** layer keeps a 128-slot ring, so its marginal cost a token
is zero and its allocated page is not (`kv_spec.SlidingWindowSpec`).

The windowed/global split is the model's own ``hybrid_layer_pattern``: a resolved list where **1
means sliding-window and 0 means global** (`config.py:342`). Reading it the other way inverts the
model, so this reads it the same way the runtime does. ``layer_types`` is the HF spelling and is used
when present; a GGUF has neither and its ``hybrid_layer_pattern`` is the fallback. When both are
absent every layer is global (`config.py:352`), which is the pessimistic reading and the right
direction for a fit test.
"""

from __future__ import annotations

from typing import Any, Mapping

from relicllm.runtime.kv_spec import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
    spec_config,
)

_DEFAULT_SLIDING_WINDOW = 128

_FULL_TYPES = ("full_attention", "global_attention", "full")
_WINDOW_TYPES = ("sliding_attention", "sliding_window")


def _pattern(config: Mapping[str, Any], layers: int) -> list[bool]:
    """Which layers are sliding-window, one bool a layer, resolved the way ``config.py:342`` does."""
    resolved = getattr(config.source, "resolved_hybrid_layer_pattern", None)
    if resolved:
        return [bool(flag) for flag in resolved]
    types = config.get("layer_types")
    if types:
        return [str(name) in _WINDOW_TYPES for name in types]
    pattern = config.get("hybrid_layer_pattern")
    if pattern:
        return [bool(int(flag)) for flag in pattern]
    return [False] * layers


def kv_spec(config: Mapping[str, Any]) -> tuple[KVCacheSpec, ...]:
    """A global spec and a sliding-window spec, each naming the layers that share it.

    The two families have their own head counts and widths on this checkpoint, and both are resolved
    rather than stated: a live ``MimoV2TextConfig`` answers ``attention(layer)`` for the layer
    itself, which is the only place the per-family fallback (an unstated ``swa_head_dim`` follows the
    *global* head dim, not the other way round) is applied. When that accessor is not available -- a
    triage fixture payload or a ``config.json`` body -- the flat keys are read, which is the path the
    fixture exercises.
    """
    config = spec_config(config)
    layers = int(config.get("num_hidden_layers") or len(config.get("hybrid_layer_pattern") or ()))
    resolve = getattr(config.source, "attention", None)
    live = resolve if callable(resolve) else None

    pattern = _pattern(config, layers)
    window = int(
        config.get("sliding_window") or config.get("sliding_window_size") or _DEFAULT_SLIDING_WINDOW
    )
    specs: list[KVCacheSpec] = []
    for is_swa in (False, True):
        members = tuple(index for index, flag in enumerate(pattern) if flag is is_swa)
        if not members:
            continue
        # A live config answers per layer, which is where an unstated `swa_head_dim` resolves against
        # the *global* head dim; a mapping states one pair of widths for both families.
        shape = live(members[0]) if live is not None else None
        kv_heads = int(shape.num_kv_heads) if shape else int(config.get("num_key_value_heads") or 0)
        key_dim = int(shape.head_dim) if shape else int(config.get("head_dim") or 0)
        value_dim = (
            int(shape.v_head_dim) if shape else int(config.get("v_head_dim") or key_dim)
        )
        cls = SlidingWindowSpec if is_swa else FullAttentionSpec
        window_kwargs = {"sliding_window": window} if is_swa else {}
        specs.append(
            cls(
                name="key",
                layer_ids=members,
                dtype="bfloat16",
                num_kv_heads=kv_heads,
                head_dim=key_dim,
                v_head_dim=value_dim,
                **window_kwargs,
            )
        )
    return tuple(specs)