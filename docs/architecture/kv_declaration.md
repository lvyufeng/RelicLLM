# The KV declaration

Every runtime in this tree sized and allocated its own contiguous KV buffer inside its own adapter,
from its own options: `self._model.cache(n)`, `make_cache(...)`, and — for the two DeepSeek runtimes —
a bare `torch.zeros(max_seq_len // ratio, head_dim)` registered as an `nn.Module` buffer inside the
attention layer that reads it. So the two questions a *host* allocator has to ask — how many bytes
does a token cost, and in what layout — were answerable only by running the runtime they belonged to.

That is the one missing artefact behind two rows of the [performance roadmap](../performance/old_hardware_roadmap.md):
no continuous batching (every runtime serializes behind one mutable cache) and no paged KV (every
cache is one contiguous buffer prepaid at `max_seq_len`). [One scheduler, many models](one_scheduler_many_models.md)
named it **② the small declaration** and left its fields open. This page opens them and records the
two artefacts that carry the answer: `relicllm/runtime/kv_spec.py` (the taxonomy) and one
`relicllm/models/<arch>/kv_spec.py` per served architecture (the publishers).

## What the two upstream runtimes do

Both were read on 2026-10-07. They agree about *what* to declare and disagree about *form*, and the
disagreement is worth stating because it is the fork this decision had to pick.

**vLLM** (`v1/kv_cache_interface.py`, v0.1.15) declares it. The per-model surface is a family of
frozen dataclasses:

```
KVCacheSpec            block_size
└ AttentionSpec        + num_kv_heads, head_size, dtype, kv_quant_mode, page_size_padded
  ├ FullAttentionSpec  + head_size_v, sliding_window, attention_chunk_size
  │ └ MLAAttentionSpec + cache_dtype_str, alignment, compress_ratio, model_version
  ├ SlidingWindowSpec  + sliding_window, head_size_v
  └ ChunkedLocalAttentionSpec + attention_chunk_size
└ MambaSpec            shapes, dtypes, mamba_type, mamba_cache_mode
```

Each attention layer implements `get_kv_cache_spec(vllm_config)` and the runner collects a
`dict[str, KVCacheSpec]` keyed by layer name; the scheduler reads it generically
(`isinstance(spec, MambaSpec)`) and never asks what the model is. `page_size_bytes` is a property
each subclass overrides, because a sliding window, an MLA latent and a Mamba state cost a block
different amounts.

**SGLang** (`python/sglang/srt/`, v0.5.20, read at `f4de6ab`) declares nothing of the kind. Geometry
is scattered: `KVCacheConfigurator.__init__` holds model-level facts (`use_mla_backend`,
`sliding_window_size`, `page_size`, `kv_cache_dtype`), the byte cost is a hand-written formula in
`pool_configurator.py::_compute_cell_size` and its per-family siblings (`_compute_dsa_indexer_cell_size`,
`_compute_qsa_cell_size`), and the only per-layer structure is a pair of id lists
(`swa_attention_layer_ids` / `full_attention_layer_ids`) with no shape on them. Its one structured
container is `DSV4PoolConfigurator` — 400 lines of state machine written for DeepSeek-V4's c4/c128
state pools.

Two ways to hold the same knowledge. vLLM concentrates it in a declaration the layers produce and the
layer above consumes; SGLang concentrates it in a consumer that computes it from the config.

## The decision

**Take vLLM's form.** A declaration is a value the model side publishes and a host reads; a
calculator is behaviour that lives on the host and has to be extended for every model. The
[package layers](package_layers.md) rule already names the direction — the models layer produces a
declaration the layer above consumes — and a per-model branch inside a shared allocator is the
duplication that rule exists to prevent, not a smaller version of it.

The taxonomy is `relicllm/runtime/kv_spec.py`, beside the device plane because it is torch-free and
model-agnostic:

- `KVCacheSpec` — a `name`, the `layer_ids` that share it, and a `dtype` string.
- `AttentionSpec` — `num_kv_heads`, `head_dim`, and `v_head_dim` (`None` follows the key width; `0`
  means the cache stores no value tensor at all, which is what "absorbed MLA" is).
  - `FullAttentionSpec`, `SlidingWindowSpec` (a bounded ring), and `MLASpec`
    (`kv_lora_rank`, `qk_rope_head_dim`, `compress_ratio`).
- `StateSpec` — upstream's `MambaSpec` row, for a cache with no token axis: a convolution window or a
  recurrent matrix, shaped by head counts and never by context.

**A named cache, not one spec per layer.** A V4.1 layer registers four KV-shaped buffers
(`window_kv_cache`, `compress_kv_cache`, the indexer's `k_cache`, the compressor's
`kv_state`/`score_state`) and a QSA layer registers two. One spec per layer cannot carry that, so a
spec names one buffer and the layers that share it — vLLM's `dict[str, KVCacheSpec]` by another name,
and one-for-one with this tree's buffer names. Layers that differ in geometry get a spec each: V4.1's
compressor is ratio 2 on its first three source layers and ratio 1 on the fourth, and merging them
would report a shape none of them has.

**A cache may name a layer another cache already names.** That is not the same as double-counting the
layer, and the two must be told apart. V4.1 registers `window_kv_cache` on **every** layer, including
the four that also grow a compressed latent, so those four appear in two specs. The rule triage
applies is *a second buffer on a layer is not a second layer*: the layer is counted once, in the class
that grows, and the ring is counted only for the layers no growing class claims. The declaration names
all forty rings because all forty exist; the count still lands on forty.

**Two costs, because two callers ask.** `page_size_bytes(block_size)` is the allocated size a paged
allocator lays out, and `values_per_token` is the marginal cost one more token adds, which the fit
test budgets against. They differ exactly where it matters: a sliding window reserves a 128-slot ring
and margins zero — which is also why declaring the four source layers' rings is free here: their
marginal contribution is zero, so the verified per-token number is unmoved, and the *allocated* fact
the paged allocator needs is no longer missing. A declaration that reported one number would be read
wrong by one of the two callers — and the fit test's verification story from 2026-10-02 is a *marginal*
number, so the round-trip through this declaration has to reproduce it.

## What a publisher looks like

One module per served architecture, `relicllm/models/<arch>/kv_spec.py`, exporting
`kv_spec(config) -> tuple[KVCacheSpec, ...]` and reading only that architecture's own config keys.
This is the second small thing the model side publishes beside `MoEModelSpec`
(`models/<arch>/spec.py`), and the discovery precedent is `components/moe/registry.py`.

| package | declares |
| --- | --- |
| `models/mimo_v2` | a global `FullAttentionSpec` (keys 192, values 128, 4 heads) and a `SlidingWindowSpec` per `hybrid_layer_pattern` |
| `models/xing4_0` | one absorbed `MLASpec` (576-wide, no value tensor) over every layer |
| `models/qwen4_exp` | a `StateSpec` (conv + recurrent shapes) for the 36 linear layers, and a `FullAttentionSpec` plus an indexer cache for the 12 QSA ones |
| `models/deepseek_v4` | a spec per compress ratio, plus the indexer cache on the layers the runtime builds one for |
| `models/deepseek_v4_1` | the same, restricted to the `kv_source_layer_ids` that own a growing buffer — four layers of forty |

Each publisher is torch-free, and the check is a subprocess
(`tests/test_kv_spec.py::test_the_declarations_are_readable_without_importing_torch`): the numbers
being right is not enough if reading them costs a torch import on a host that only wants to know
whether the model fits. Two package `__init__`s had to become lazy re-exports for this —
`models/mimo_v2/__init__.py` and `models/xing4_0/__init__.py` imported torch submodules eagerly — with
the pattern `models/deepseek_v4_1/__init__.py` already documents.

## The triage consolidation

`relicllm/triage/kv.py` already derived a per-layer KV geometry for nine models, pinned by
`tests/fixtures/triage/kv_geometry.json` and hand-verified against checkpoints. It is the same subject
in a different form, so rather than keep two it now **consumes** the declaration for the five
architectures that publish one. The mapping is a table (`_DECLARED`) and the numbers are read back,
not recomputed: `KvGeometry` keeps its `Confidence`, its `file:line` provenance and every note, so the
report and the fit test are untouched.

Three things the consolidation had to get right, each of which the fixtures caught:

- **The layer count is the trunk.** `compress_ratios` covers the backbone *plus* the MTP/draft layers,
  so a declaration that kept all of them reports 46 layers for a 43-layer model. The publisher cuts
  the list at `num_hidden_layers`, the same place the loader stops.
- **The indexer is a second cache, not a second layer.** It contributes bytes and is excluded from the
  layer count — the `"indexer"` kind string the consumer already knew, kept because the count sum
  reads it.
- **The marginal number is the one that round-trips.** The declaration's `page_size_bytes` is the
  allocated size; triage's `bytes_per_token_whole_model` is the marginal one. The mapping reads
  `values_per_token`, so a sliding window contributes zero here and its ring is a fact the *allocator*
  reads instead.

The two architectures the fixture covers that have no `models/` package (`qwen3_5`, `qwen35`) keep
their own config-reading builder: there is no declaration to consume, and inventing one for a model
this tree does not serve would put a claim in the models layer nothing builds.

**The acceptance gate is that all nine fixtures pass with no edit.** They do, unchanged.

## What this enables, and what it does not

**Enables.** Continuous batching (#33) and paged KV (#34) now have a shape to allocate against: a
host can read `page_size_bytes` and `layer_ids` without importing a model, and the declaration's
`name`/`layer_ids` are the block table's rows. It is also the prerequisite for #14's prefix-cache and
chunked-prefill restore being one implementation rather than four.

**Does not.** This is the declaration, not the consumer — no paged allocation and no block table
land here. It is also not a new allocator: `MimoV2KVCache`, `KVLatentCache` and `Qwen4ExpCache` carry
`append`/`view`/`reset` semantics the runtimes depend on, so the declaration is passed *into* those
constructors, which stop deriving shapes themselves. vLLM and SGLang each re-allocate from their spec
with a generic kernel-side writer; this tree has no such writer, and inventing one is a separate
piece of work. Making the caches independent `nn.Module` tensors — rather than buffers that share the
model's tree — is likewise out of scope here.

## Not decided here

The block size a paged allocator will use, and whether the block table lands before or after the
prefix-cache ports it would make cheaper. Those are #33/#34's, and this declaration is the value they
read.