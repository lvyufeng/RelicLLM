"""The V4.1 attention buffers read the declaration, rather than deriving the same shape again.

`relicllm/models/deepseek_v4_1/attention.py` is the one served architecture whose KV shapes are
`register_buffer` calls inside the layer that reads them, so the "does the allocator read the
declaration" question is asked here of four buffers at once: `window_kv_cache` (every layer),
`compress_kv_cache` and the compressor's `kv_state`/`score_state` (the `kv_source_layers`), and the
indexer's `k_cache` (the layers that `own_k`). Before #129 each was derived from `_compress_ratio_at`,
`window_size` and `head_dim` independently of the publisher; a drift between the two would be a
declaration that names a width the cache does not allocate.

Hermetic: the toy geometry is the same one `test_models_deepseek_v4_1_attention.py` drives, and no
checkpoint is read. The shapes are compared against the publisher's own output, not against numbers
restated here -- a test that repeated the module's arithmetic would agree with it whatever it said.

V4 (`models/deepseek_v4/runtime.py`) is not covered. Its `Attention` keeps a single `kv_cache` sized
`window_size + max_seq_len // compress_ratio` -- the window and the compressed latent concatenated
into one buffer -- and its `Compressor` scales its state by an overlap factor (`coff`) the publisher
does not carry, so its buffers cannot be read back from the current declaration without widening it
to describe an allocated, non-per-token shape. That is a publisher change, left for its own piece.
"""

from __future__ import annotations

import pytest

from relicllm.models.deepseek_v4_1.config import V41TextConfig
from relicllm.models.deepseek_v4_1.kv_spec import kv_spec

# The toy geometry from `test_models_deepseek_v4_1_attention.py`: ratios 2 and 1 so both compressor
# branches exist, two KV sources, three index sources, and layer 4 owning both published caches.
TOY = dict(
    dim=64,
    n_layers=6,
    n_mtp_layers=0,
    n_heads=4,
    head_dim=32,
    rope_head_dim=8,
    q_lora_rank=32,
    o_groups=2,
    o_lora_rank=16,
    window_size=4,
    compress_ratios=(0, 0, 2, 2, 1, 1),
    kv_source_layers=(2, 4),
    index_source_layers=(2, 4, 5),
    index_n_heads=2,
    index_head_dim=32,
    index_topk=16,
    candidate_source_layer=4,
    candidate_topk_blocks=16,
    candidate_block_size=2,
    norm_eps=1e-6,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=40.0,
    beta_fast=32,
    beta_slow=1,
    original_seq_len=512,
    max_position_embeddings=1024,
)

MAX_SEQ_LEN = 64
MAX_BATCH = 1


def _specs_by_name(config: V41TextConfig) -> dict:
    by_name: dict[str, dict] = {}
    for spec in kv_spec(config):
        for layer in spec.layer_ids:
            by_name.setdefault(spec.name, {})[layer] = spec
    return by_name


def test_the_window_ring_is_the_declared_sliding_window() -> None:
    """Every layer's ring holds the window the declaration names, at the declared width."""
    torch = pytest.importorskip("torch")
    from relicllm.models.deepseek_v4_1.attention import Attention

    config = V41TextConfig(**TOY)
    declared = _specs_by_name(config)["window_kv_cache"]

    for layer_id in range(config.n_layers):
        attention = Attention(layer_id, config, max_batch_size=MAX_BATCH, max_seq_len=MAX_SEQ_LEN)
        spec = declared[layer_id]
        assert attention.window_kv_cache.shape == (
            MAX_BATCH,
            int(spec.sliding_window),
            int(spec.head_dim),
        ), f"layer {layer_id}'s ring is not the declared window"


def test_a_source_layer_sizes_its_compressed_cache_and_compressor_state_from_the_spec() -> None:
    """The compressed latent and the partial group that fills it agree with the declaration.

    One layer, four shapes: the cache the layer grows, the index keys derived from it, and the
    compressor's `kv_state`/`score_state`, which hold exactly one partial group of the same width.
    """
    torch = pytest.importorskip("torch")
    from relicllm.models.deepseek_v4_1.attention import Attention

    config = V41TextConfig(**TOY)
    declared = _specs_by_name(config)
    layer_id = 2  # a KV source at ratio 2, also an index source that owns its keys
    attention = Attention(layer_id, config, max_batch_size=MAX_BATCH, max_seq_len=MAX_SEQ_LEN)

    compress = declared["compress_kv_cache"][layer_id]
    assert attention.compress_kv_cache.shape == (
        MAX_BATCH,
        MAX_SEQ_LEN // int(compress.compress_ratio),
        int(compress.head_dim),
    )

    # The compressor's state is one partial group: `ratio` rows of the same width the cache stores.
    assert attention.compressor is not None
    state_shape = (MAX_BATCH, int(compress.compress_ratio), int(compress.head_dim))
    assert tuple(attention.compressor.kv_state.shape) == state_shape
    assert tuple(attention.compressor.score_state.shape) == state_shape

    index = declared["k_cache"][layer_id]
    assert attention.indexer is not None
    assert attention.indexer.k_cache.shape == (
        MAX_BATCH,
        MAX_SEQ_LEN // int(index.compress_ratio),
        int(index.head_dim),
    )


def test_a_non_source_layer_declares_no_compressed_cache_and_registers_none() -> None:
    """The buffer set is the declared buffer set, not a superset: a reader has nothing unstated.

    A layer that only reads compressed positions is in the index-source list but not the KV-source
    list, so it carries no `compress_kv_cache` and its indexer owns no `k_cache` -- which is exactly
    the condition the publisher uses to decide which layers appear in those two specs.
    """
    torch = pytest.importorskip("torch")
    from relicllm.models.deepseek_v4_1.attention import Attention

    config = V41TextConfig(**TOY)
    declared = _specs_by_name(config)
    read_only_indexer = 5  # index source, not a KV source -> reads the keys, owns none
    assert read_only_indexer not in declared["k_cache"]
    assert read_only_indexer not in declared["compress_kv_cache"]

    attention = Attention(read_only_indexer, config, max_batch_size=MAX_BATCH, max_seq_len=MAX_SEQ_LEN)

    assert not hasattr(attention, "compress_kv_cache")
    assert attention.indexer is not None and not attention.indexer.owns_k
    assert not hasattr(attention.indexer, "k_cache")


def test_the_buffer_follows_the_declaration_even_when_the_config_disagrees(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The line #129 has to hold: a change to the declaration moves the buffer.

    Every test above passes with the constructors reading the config and ignoring the declaration --
    the two agree for any config the publisher itself reads. This forces them apart, so a fall-through
    to the config arithmetic fails here instead of agreeing by coincidence. It is the same check
    `test_kv_spec.py::test_triage_costs_the_declared_geometry_rather_than_recomputing_it` makes on the
    consumer, applied to the allocator.
    """
    torch = pytest.importorskip("torch")
    from dataclasses import replace

    from relicllm.models.deepseek_v4_1 import attention as attention_module

    config = V41TextConfig(**TOY)
    real = attention_module.kv_spec(config)

    def widened(cfg: V41TextConfig):
        """The same declaration, one ring slot deeper and one width wider than the config states."""
        out = []
        for spec in real:
            if spec.name == "window_kv_cache":
                out.append(replace(spec, sliding_window=int(spec.sliding_window) + 1, head_dim=spec.head_dim + 8))
            else:
                out.append(spec)
        return tuple(out)

    monkeypatch.setattr(attention_module, "kv_spec", widened)
    attention = attention_module.Attention(0, config, max_batch_size=MAX_BATCH, max_seq_len=MAX_SEQ_LEN)

    declared = next(s for s in widened(config) if s.name == "window_kv_cache")
    assert attention.window_kv_cache.shape == (
        MAX_BATCH,
        int(declared.sliding_window),
        int(declared.head_dim),
    ), "the ring was sized from the config, not the declaration"