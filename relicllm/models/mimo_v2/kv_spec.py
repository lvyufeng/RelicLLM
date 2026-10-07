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
)

_DEFAULT_SLIDING_WINDOW = 128

_FULL_TYPES = ("full_attention", "global_attention", "full")
_WINDOW_TYPES = ("sliding_attention", "sliding_window")


def _pattern(config: Mapping[str, Any], layers: int) -> list[bool]:
    """Which layers are sliding-window, one bool a layer, resolved the way ``config.py:342`` does."""
    types = config.get("layer_types")
    if types:
        return [str(name) in _WINDOW_TYPES for name in types]
    pattern = config.get("hybrid_layer_pattern")
    if pattern:
        return [bool(int(flag)) for flag in pattern]
    return [False] * layers


def kv_spec(config: Mapping[str, Any]) -> tuple[KVCacheSpec, ...]:
    """A global spec and a sliding-window spec, each naming the layers that share it."""
    layers = int(config.get("num_hidden_layers") or len(config.get("hybrid_layer_pattern") or ()))
    kv_heads = int(
        config.get("num_key_value_heads") or config.get("attention.head_count_kv") or 0
    )
    key_dim = int(config.get("head_dim") or config.get("attention.key_length") or 0)
    value_dim = int(config.get("v_head_dim") or config.get("attention.value_length") or key_dim)
    window = int(
        config.get("sliding_window")
        or config.get("sliding_window_size")
        or _DEFAULT_SLIDING_WINDOW
    )

    pattern = _pattern(config, layers)
    windowed = tuple(index for index, is_swa in enumerate(pattern) if is_swa)
    global_layers = tuple(index for index, is_swa in enumerate(pattern) if not is_swa)

    specs: list[KVCacheSpec] = []
    if global_layers:
        specs.append(
            FullAttentionSpec(
                name="key",
                layer_ids=global_layers,
                dtype="bfloat16",
                num_kv_heads=kv_heads,
                head_dim=key_dim,
                v_head_dim=value_dim,
            )
        )
    if windowed:
        specs.append(
            SlidingWindowSpec(
                name="key",
                layer_ids=windowed,
                dtype="bfloat16",
                num_kv_heads=kv_heads,
                head_dim=key_dim,
                v_head_dim=value_dim,
                sliding_window=window,
            )
        )
    return tuple(specs)