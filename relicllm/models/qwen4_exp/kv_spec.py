"""Qwen3.8-Flash-Next's KV declaration: a hybrid of sparse attention and a linear state.

Two things a K/V cache is not, both present here. Thirty-six of the 48 layers run GatedDeltaNet, a
linear attention whose state is a **fixed** convolution window and recurrent matrix -- no token axis,
so its marginal cost is zero and its allocation is :class:`~relicllm.runtime.kv_spec.StateSpec`, the
``MambaSpec`` row upstream keeps for exactly this. The other twelve run QSA sparse attention and hold
the only growing caches.

A QSA layer holds **two** of them: a key/value pair under a head axis, and the indexer's raw key
beside it -- no head axis, its own width. That is why the declaration is a set of named caches rather
than one a layer: a single spec per layer could carry the pair or the indexer but not both.

The layer split is the checkpoint's own ``layer_types``, read through the same accessor the runtime
uses (`qwen4_exp/config.py:153`). An absent list is the runtime's default expansion, which this
reproduces rather than guesses at.
"""

from __future__ import annotations

from typing import Any, Mapping

from relicllm.runtime.kv_spec import (
    FullAttentionSpec,
    KVCacheSpec,
    StateSpec,
    spec_config,
)

_LINEAR_TYPE = "linear_attention"
_FULL_TYPE = "full_attention"


def _layer_types(config: Mapping[str, Any], layers: int) -> list[str]:
    """The checkpoint's ``layer_types``, or the runtime's default expansion when it states none.

    The default (``qwen4_exp/config.py:109``) is a repeating block of four: three linear layers then
    one full one, over the whole depth. Reproduced here so a config without the list yields the same
    split the model would build, not an all-full overstatement.
    """
    types = config.get("layer_types")
    if types:
        return [str(name) for name in types]
    return [_FULL_TYPE if index % 4 == 3 else _LINEAR_TYPE for index in range(layers)]


def kv_spec(config: Mapping[str, Any]) -> tuple[KVCacheSpec, ...]:
    """The linear state, the QSA key/value pair, and the indexer key -- each naming its layers."""
    config = spec_config(config)
    layers = int(config.get("num_hidden_layers") or len(config.get("layer_types") or ()))
    types = _layer_types(config, layers)
    linear = tuple(index for index, name in enumerate(types) if name == _LINEAR_TYPE)
    full = tuple(index for index, name in enumerate(types) if name != _LINEAR_TYPE)

    kv_heads = int(config.get("num_key_value_heads") or config.get("attention.head_count_kv") or 0)
    head_dim = int(config.get("head_dim") or config.get("attention.key_length") or 0)
    index_dim = int(config.get("indexer_head_dim") or config.get("index_head_dim") or 0)
    k_heads = int(config.get("linear_num_key_heads") or 0)
    v_heads = int(config.get("linear_num_value_heads") or 0)
    k_head_dim = int(config.get("linear_key_head_dim") or 0)
    v_head_dim = int(config.get("linear_value_head_dim") or 0)
    conv_kernel = int(config.get("linear_conv_kernel_dim") or 0)

    specs: list[KVCacheSpec] = []
    if linear:
        # The two tensors `GatedDeltaNetCache` allocates (`qwen4_exp/attention.py:298`): a convolution
        # state `(2*k_heads*k_dim + v_heads*v_dim, kernel-1)` and a recurrent state
        # `(v_heads, k_dim, v_dim)`. They are one cache -- one spec, two shapes -- because they exist
        # on the same layers and share their lifetime: two specs would report 36 layers twice. The
        # batch axis is the host allocator's, so the shapes here are one row's.
        conv_dim = 2 * k_heads * k_head_dim + v_heads * v_head_dim
        specs.append(
            StateSpec(
                name="state",
                layer_ids=linear,
                dtype="bfloat16",
                shapes=((conv_dim, max(0, conv_kernel - 1)), (v_heads, k_head_dim, v_head_dim)),
                dtypes=("bfloat16", "float32"),
            )
        )
    if full:
        specs.append(
            FullAttentionSpec(
                name="k",
                layer_ids=full,
                dtype="bfloat16",
                num_kv_heads=kv_heads,
                head_dim=head_dim,
                v_head_dim=head_dim,
            )
        )
        if index_dim:
            specs.append(
                FullAttentionSpec(
                    name="index_k",
                    layer_ids=full,
                    dtype="bfloat16",
                    num_kv_heads=1,
                    head_dim=index_dim,
                    v_head_dim=0,
                )
            )
    return tuple(specs)