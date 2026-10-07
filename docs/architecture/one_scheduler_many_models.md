# One scheduler, many models

**Date:** 2026-10-06
**Question:** this repository wants "one scheduler, swappable model implementations". Is that the
shape the two stacks it measures itself against actually have, or does each model get its own spec
that drives scheduling?

**Answer: a model registry keyed by architecture string, and *one* model-agnostic scheduler.**
Neither vLLM nor SGLang has a per-model spec object that the scheduler consults. What is per-model is
a **small declaration** — the shape of the memory a layer wants — not a behaviour interface. The
scheduler never sees a model class at all.

That answer corrects a draft of this work. An earlier proposal here was a per-architecture
`ServingSpec` carrying seven methods (weight loading, single-step forward, tokenizer, prompt encoder,
EOS set, prefix-cache shape, served fields) that the serving skeleton would call through. Both
upstreams are smaller than that, and the size is the point: a seven-method interface per architecture
is the adapter duplication this repository already suffers
([measured](per_method_duplication_2026_09.md)) moved one level up and given a better name. That
draft is rejected below.

Read at each project's tip on 2026-10-06; the file paths and quotes are from the sources listed at
the end.

## What vLLM does

**A registry keyed by the architecture string.** `vllm/model_executor/models/registry.py` maps an HF
architecture name to a `(module, class_name)` pair and resolves it lazily:

```python
_TEXT_GENERATION_MODELS = {
    "LlamaForCausalLM": ("llama", "LlamaForCausalLM"),
}
_VLLM_MODELS = {**_TEXT_GENERATION_MODELS, **_EMBEDDING_MODELS, ...}
```

The entry is a `_LazyRegisteredModel` (module + class name), not a live class, so importing the
registry does not import every model. The lookup is `ModelRegistry.resolve_model_cls(architectures,
model_config)`, with `_normalize_arch` handling the architecture-default suffixes and a Transformers
fallback when nothing matches.

**A single model-agnostic scheduler.** `vllm/v1/core/sched/scheduler.py` holds one concrete
`Scheduler` implementing `SchedulerInterface`. There are no per-architecture subclasses. Its own
comment states the design rule:

> it doesn't distinguish phases, and each request tracks `num_computed_tokens` and
> `num_tokens_with_spec`, which generalizes to chunked prefills, prefix caching, and spec decoding.

That is the load-bearing sentence for this repository. **Chunked prefill, prefix reuse and
speculative decoding — three of the four items on our own performance roadmap — are, upstream,
counters on a request rather than branches in a scheduler.** A scheduler written that way does not
grow a per-model case when a model gains one of them.

**How a model's difference reaches the scheduler.** Not by the scheduler reading the model, but
through three channels, each of them data or a factory:

| Channel | What flows | Example |
| --- | --- | --- |
| config flags | booleans on `vllm_config` | `model_config.uses_mrope`, `is_encoder_decoder`, `supports_multimodal_inputs` |
| the KV declaration | one small object a layer group publishes | `kv_cache_config.kv_cache_groups[i].kv_cache_spec` |
| factories | a connector / manager built for a role | `KVConnectorBase_V1`, `KVCacheManager` |

where the scheduler reads the second generically — `isinstance(group.kv_cache_spec, MambaSpec)`,
`get_kv_cache_spec_kind(...)` — rather than by asking what the model is.

**And the declaration is small.** `vllm/v1/kv_cache_interface.py` is a family of frozen dataclasses,
each a handful of fields:

```text
KVCacheSpec            block_size, dcp_sharded, block_stride_alignment
└ AttentionSpec        + num_kv_heads, head_size, dtype, kv_quant_mode
  └ FullAttentionSpec  + sliding_window, attention_chunk_size, non_causal
    └ MLAAttentionSpec + cache_dtype_str, alignment, model_version, head_size_v=0
  └ SlidingWindowSpec · ChunkedLocalAttentionSpec
└ MambaSpec
```

The variants are the attention *types* (full, MLA, sliding window, chunked local, Mamba, cross,
encoder-only), and what they carry is geometry and dtype — the numbers a block allocator needs to lay
out a block table. A model publishes one of these; it does not publish a way to be scheduled.

## What SGLang does

`sglang/srt/models/registry.py` is the same pattern and less machinery. The registry is a dict of
architecture name to implementation class:

```python
models: Dict[str, Union[Type[nn.Module], str]] = field(default_factory=dict)
```

Registration is module-level discovery rather than a decorator: `import_model_classes()` walks the
submodules and reads an `EntryClass` attribute off each, so a module can expose one class or a list.
Lookup is `resolve_model_cls(architectures)`, with the same normalization-then-fallback shape as
vLLM's. **The registry file contains no scheduling logic at all** — its scope is discovering model
classes and resolving a name to one.

## The shape both share, stated once

| | vLLM | SGLang | RelicLLM |
| --- | --- | --- | --- |
| registry keyed by architecture string | ✅ `_VLLM_MODELS` | ✅ `models` dict | ✅ `components/moe/registry.py` |
| a small per-layer memory/shape declaration | ✅ `KVCacheSpec` family | (inside the model) | ❌ absent |
| one scheduler for every architecture | ✅ | ✅ | ❌ absent |
| the scheduler holds a model class or a forward | ❌ never | ❌ never | — |

So the user-facing claim that motivated this record — *they are all LLMs, so the scheduling should be
the same* — is not a simplification we would be making. **It is what both upstreams do**, and the
discipline that makes it possible is that the per-model surface is kept to a declaration rather than
opened into a behaviour interface.

This also agrees with the invariant this repository already states for the hardware axis: *backend
selection happens at configure time rather than in shared code* (`CLAUDE.md`). Both axes — which
architecture, which accelerator — resolve to a binding chosen up front, so the shared code stays free
of branches on either.

## Where RelicLLM stands

The first row is done, and it is the same pattern. `components/moe/registry.py` keys on
`general.architecture` and hands back a spec per architecture
(`DeepSeekV4Spec`, `GLMDSASpec`, `MiniMaxM2Spec`), each `architecture` a class attribute — the vLLM
shape, in a repository that also contains the anti-pattern.

The decisive local fact is the third row. There **is** a scheduler in this tree:
`relicllm/models/deepseek_v4/pd_scheduler.py`. It is imported by exactly two modules, and both are inside one
model:

```
relicllm/models/deepseek_v4/generation.py
relicllm/models/deepseek_v4/serving.py
```

So the scheduler we have lives *inside* `deepseek_v4`, serves one model, and the other four runtimes
have no scheduler at all — they serialize behind the adapter's own request lock. That is the exact
inverse of the pattern above: not *one scheduler the models plug into*, but *a scheduler inside one
model with the others going without*.

The middle two rows are the missing artefacts:

- **② the small declaration.** Every runtime today preallocates its own contiguous KV buffer sized
  from its own options; nothing states "a block is N tokens, K heads, this dtype" in a form a host
  allocator could read. Without it there is no paged KV and no shared admission — the two structural
  gaps the [performance roadmap](../performance/old_hardware_roadmap.md) names under every model.
- **③ one scheduler.** Admission, KV accounting, sampling and cancellation as one Python module that
  does not know which architecture it is serving, with per-model differences reaching it as the
  config flags, the declaration, and a runtime hook — never as a branch on the architecture name.

## What this rules out

- **A per-architecture `ServingSpec` with the model's behaviour on it.** Seven methods is larger than
  either upstream, and it would let the same lifecycle be written once per model under a shared
  interface — the duplication relocated rather than removed.
- **Moving model knowledge into the adapter.** That is the retired native shape in reverse, and it is
  what put the constraint mask inside `cpp_backend.py`; see
  [the native surface decision](native_surface_decision.md).
- **A blanket rename of `backend`.** The word carries four unrelated meanings in this package, and
  two of them are hardware: `components/moe/cpu_backend.py` and `gpu_prefill_backend.py` are device
  planes, not model selections. A global substitution would rename the device axis wrongly.

## The naming this settles

Measured across the package, `backend` appears 755 times and carries four meanings:

| Meaning | Where | What it should be called |
| --- | --- | --- |
| which weights are served — the user-facing `--backend torch/v41/...` | `api/types.py::_BACKENDS` | **`runtime`** (the package already calls it that everywhere else: `RuntimeCapabilities`, `RuntimeAdapter`, `runtime_engine.py`; `runtime` already appears 1011 times) |
| which accelerator | `components/moe/{cpu,gpu_prefill}_backend.py` | **device** / plane (`runtime/device.py::PLATFORMS`) |
| the host language of the engine — retired | "the retired cpp backend" | history |
| the adapter layer itself | `BackendBase`, `relicllm/backends/` | adapter; it dissolves when ② and ③ land |

Ruling: the **user-facing surface says `runtime`**, matching the package's own vocabulary and both
upstreams' (`--backend` does not exist in either). The hardware axis stays `device`. `backends/` is
the adapter layer and is not renamed by this decision — it is subsumed. Two spellings do not move
whatever the rename does, because they are contracts with existing launch scripts and configs:
**`POCKETLLM_*` environment variables and the `--backend-option` flag name** (see `CLAUDE.md`).

## Relation to the open issue tree

This is a ruling on existing items rather than a new workstream:

- **#18 (port the batch scheduler to Python)** — the same ruling, reached from the other side. It and
  this record converge; #18's "the request lifecycle and scheduler are Python" is the "one scheduler"
  row and is not re-opened.
- **#52 / #53 / #54 (R1/R2/R3: one request lifecycle and one scheduler across every runtime)** — this
  record supplies the shape those items were missing: the scheduler is model-agnostic, and the
  per-runtime surface is the small declaration plus config, not a spec object.
- **#55 (collapse the five adapters into one adapter plus a per-runtime spec)** — its acceptance
  ("one adapter class; adding a runtime is a spec plus a registry entry") is right, with one
  refinement this record fixes: **"spec" means the small declaration**, and the behaviour stays in
  `models/<arch>/` where it already is.

## Not decided here

- The exact fields of ② — the KV declaration — and whether a paged block table lands before or after
  the ports it would make cheaper.
- Where ③ lives (`relicllm/scheduler/`, or `runtime/` beside the device plane), and how it takes over
  the adapter request lock without a behaviour change at width 1.
- The rename's blast radius: `--backend` on the user surface is low-risk; `relicllm/backends/` is a
  large diff and is deliberately *not* in this decision.

Each is a follow-up with its own measurement, per the repository's rule that a lever is only
authorized after it is measured.

## Sources

Read at tip on 2026-10-06:

- [vLLM `vllm/model_executor/models/registry.py`](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/registry.py)
- [vLLM `vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
- [vLLM `vllm/v1/kv_cache_interface.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py)
- [SGLang `python/sglang/srt/models/registry.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/models/registry.py)