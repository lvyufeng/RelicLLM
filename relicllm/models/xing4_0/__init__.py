"""Xing4.0-29B-A4B, staged.

Stage 3 of [#388](https://github.com/lvyufeng/PocketLLM/issues/388) is the
attention -- `config`, `rope` and the two forms of MLA.  Stage 4 is the
hyper-connection, `hyper_connection` and the `block` that wraps the sublayers in
it.  The MoE, serving and the write-up are [#393](https://github.com/lvyufeng/PocketLLM/issues/393),
so nothing in this package runs the model end to end yet.

The names below are re-exported lazily rather than imported here, the pattern
`models/deepseek_v4_1/__init__.py` documents: every submodule but `config` imports torch, and this
package's KV declaration (`kv_spec.py`) must be readable without it so a fit check on a host that
never loads the model does not have to build one. A lazy re-export keeps
`from relicllm.models.xing4_0 import Xing4_0Params` working unchanged.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DecoderLayer",
    "DecoderLayerWeights",
    "HyperConnection",
    "HyperConnectionWeights",
    "KVLatentCache",
    "MLAAttention",
    "MLAAttentionWeights",
    "Xing4_0Params",
    "YarnParams",
    "rms_norm",
    "sinkhorn",
    "yarn_get_mscale",
]

_LAZY = {
    "Xing4_0Params": "config",
    "YarnParams": "config",
    "yarn_get_mscale": "config",
    "DecoderLayer": "block",
    "DecoderLayerWeights": "block",
    "rms_norm": "block",
    "KVLatentCache": "attention",
    "MLAAttention": "attention",
    "MLAAttentionWeights": "attention",
    "HyperConnection": "hyper_connection",
    "HyperConnectionWeights": "hyper_connection",
    "sinkhorn": "hyper_connection",
}


def __getattr__(name: str) -> Any:
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(f".{module}", __name__), name)


def __dir__() -> list[str]:
    return sorted(__all__)
