"""Xing4.0-29B-A4B's KV declaration: one absorbed MLA latent a layer, no value tensor.

Every layer is the same shape, so this publishes one spec over all of them. The cache is *absorbed*:
``KVLatentCache`` (`attention.py:193`) stores ``kv_lora_rank + qk_rope_head_dim`` values a token in a
single tensor, and the value is ``latent @ v_b`` -- recovered after the attention weights, never
stored. That is why this is an :class:`~relicllm.runtime.kv_spec.MLASpec` with ``v_head_dim=0`` and
``num_kv_heads=1``: there is no head axis and no second tensor, and a declaration that carried either
would overstate the cache by the factor it invented.

A leaf module: it reads a config mapping and imports nothing that is not a spec, so triage can build
it without importing torch or the model (the ``models/`` package ``__init__`` is lazy for this
reason).
"""

from __future__ import annotations

from typing import Any, Mapping

from relicllm.runtime.kv_spec import KVCacheSpec, MLASpec

#: The widths the checkpoint states, with the released card's values as the fallback the config class
#: itself uses (`xing4_0/config.py:82`). Spelled here rather than read from the dataclass so this
#: module stays importable without ``config.py``.
_DEFAULT_KV_LORA_RANK = 512
_DEFAULT_QK_ROPE_HEAD_DIM = 64


def kv_spec(config: Mapping[str, Any]) -> tuple[KVCacheSpec, ...]:
    """One :class:`MLASpec` for every layer of this checkpoint."""
    kv_lora = int(config.get("kv_lora_rank") or _DEFAULT_KV_LORA_RANK)
    rope = int(
        config.get("qk_rope_head_dim")
        or config.get("rope.dimension_count")
        or _DEFAULT_QK_ROPE_HEAD_DIM
    )
    layers = int(config.get("num_hidden_layers") or 0)
    return (
        MLASpec(
            name="latent",
            layer_ids=tuple(range(layers)),
            dtype="bfloat16",
            num_kv_heads=1,
            head_dim=kv_lora + rope,
            v_head_dim=0,
            kv_lora_rank=kv_lora,
            qk_rope_head_dim=rope,
        ),
    )