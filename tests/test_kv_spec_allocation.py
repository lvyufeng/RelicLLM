"""The allocation sites read the declaration, rather than deriving the same shape a second time.

`tests/test_kv_spec.py` pins the declaration's arithmetic and that triage consumes it. This pins the
other half of #129: that the caches the runtimes actually allocate are sized *from* it. The failure
mode a shape-only test cannot see is drift -- the declaration says one width and the cache allocates
another -- so each test below builds a real cache and compares it against the publisher's own output,
not against a number restated here.

v4, v4.1 and qwen4_exp are not covered yet: their caches register `nn.Module` buffers inside
`Attention`/`Compressor`/`Indexer`, and threading a spec through `Backbone -> Block -> Attention` is a
separate change (see the module docstring of `models/deepseek_v4_1/kv_spec.py` for the one gap that
blocks the v4.1 window). What is testable today is testable here.
"""

from __future__ import annotations

import pytest


def test_the_mimo_cache_takes_its_kv_dimensions_from_the_declaration() -> None:
    """The cache's key/value heads and widths are the declaration's, layer for layer."""
    torch = pytest.importorskip("torch")
    from relicllm.models.mimo_v2.config import MimoV2TextConfig
    from relicllm.models.mimo_v2.device_attention import MimoV2KVCache
    from relicllm.models.mimo_v2.kv_spec import kv_spec

    config = MimoV2TextConfig(
        num_hidden_layers=4,
        hybrid_layer_pattern=(0, 1, 1, 0),
        head_dim=192,
        v_head_dim=128,
        num_key_value_heads=4,
        sliding_window=128,
    )
    cache = MimoV2KVCache(config, capacity=256, layers=[0, 1, 2, 3], device="cpu", dtype=torch.float32)

    declared = {}
    for spec in kv_spec(config):
        for layer in spec.layer_ids:
            declared[layer] = spec

    assert declared, "the publisher declared nothing for a config it can read"
    for layer, spec in declared.items():
        shape = cache.shape(layer)
        assert shape.num_kv_heads == spec.num_kv_heads
        assert shape.head_dim == spec.head_dim
        assert shape.v_head_dim == (spec.v_head_dim if spec.v_head_dim is not None else spec.head_dim)
        # A sliding-window layer's ring is capped at the declared window; a global one is not.
        if getattr(spec, "sliding_window", None):
            assert cache.slots(layer) == int(spec.sliding_window)


def test_the_xing4_cache_width_is_the_declared_latent_width() -> None:
    """`KVLatentCache` built from a publisher's specs holds exactly the declared width.

    The params spelling that predates the declaration is covered by `test_xing4_0_decode_pos.py`,
    which builds one from the real checkpoint; here the declaration is the only subject.
    """
    torch = pytest.importorskip("torch")
    from relicllm.models.xing4_0.attention import KVLatentCache
    from relicllm.models.xing4_0.kv_spec import kv_spec

    specs = kv_spec({"kv_lora_rank": 512, "qk_rope_head_dim": 64, "num_hidden_layers": 8})

    cache = KVLatentCache(batch=1, capacity=16, source=specs, device="cpu", dtype=torch.float32)

    assert cache.width == specs[0].head_dim == 512 + 64
    assert cache.latent.shape == (1, 16, 512 + 64)


def test_the_declared_dtypes_are_the_ones_the_cli_advertises() -> None:
    """`--kv-cache-dtype`'s value set and the byte table cannot disagree about a name."""
    from relicllm.runtime.kv_spec import VALID_KV_CACHE_DTYPES, dtype_size

    assert "auto" in VALID_KV_CACHE_DTYPES  # the runtime's own cache dtype
    for name in VALID_KV_CACHE_DTYPES:
        if name == "auto":
            continue
        # Every non-auto value names a storage the byte table can size: a value the parser accepted
        # but the arithmetic could not would fail deep in an allocation instead of at the flag.
        resolved = {"fp8": "float8_e4m3fn", "fp8_e4m3": "float8_e4m3fn", "fp8_e5m2": "float8_e5m2"}.get(
            name, name
        )
        assert dtype_size(resolved) > 0