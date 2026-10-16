# The capability gap against vLLM and SGLang

**Date:** 2026-10-16
**Status:** living page. It is meant to be edited as each gap closes, and its upstream column is
re-derived rather than remembered.
**Upstream read at:** vLLM `main` and SGLang `main`, on 2026-10-16, with
`scripts/upstream_capability_inventory.py --fetch`.

RelicLLM and the two stacks it is usually measured against are **two kinds of engine, not two
versions of one**. This page is the standing answer to "where is the gap", kept as one document so
the answer is one edit instead of a rewrite each time the question comes back. Every claim about
this tree cites the file or page it was read from; every claim about upstream is read from the
projects' own text, and re-read by the script named above.

## What this page is not

- **Not a benchmark.** No number here is measured against an upstream run, and none may be quoted as
  if it were. The performance numbers are pointers into this tree's own records, taken under the
  conditions those records state — see [benchmarking and reporting rules](../guides/benchmarking.md).
- **Not a feature-parity goal.** Flag counts and model counts are listed because a reader asks, not
  because matching them is the plan; [the positioning](positioning.md) says what this runtime is for.
  A gap below is classified as a *debt* only when closing it is what the runtime's own users need.

## How to refresh the upstream column

The script reads both projects as text — no import, no install — so it re-runs against a future
commit without a build, which is the same contract
[the command-line inventory](cli_surface_design_2026_09.md) made for its numbers.

```bash
git clone --depth 1 https://github.com/vllm-project/vllm /tmp/vllm_ref
git clone --depth 1 https://github.com/sgl-project/sglang /tmp/sglang_ref
python scripts/upstream_capability_inventory.py --vllm /tmp/vllm_ref --sglang /tmp/sglang_ref

# or, with no checkout:
python scripts/upstream_capability_inventory.py --fetch
```

It prints vLLM's two capability bullet lists (`vLLM is fast with:` / `vLLM is flexible and easy to
use with:`), SGLang's `docs/advanced_features/` page names — which are its feature surface now that
its README is an ecosystem page — and SGLang's hardware table. A checkout's HEAD and commit date are
recorded beside the reading; the table below is only true of the revision it was taken at.

## The one answer

**On throughput and concurrency, the two are not comparable objects.** RelicLLM has no scheduling
width above one and no paged-KV writer, so any SLO or goodput comparison would be measuring the wrong
thing. **On "a checkpoint the cards cannot hold"**, RelicLLM runs models that vLLM/SGLang would need
different hardware for. The gap is therefore three different things wearing one name:

| Class | What it is | Is closing it the plan? |
| --- | --- | --- |
| **A — architectural** | the throughput track: continuous batching, paged KV, a synchronous per-token host path, no speculative path | yes, and it is [Phase 4](../performance/old_hardware_roadmap.md) — the largest and last change |
| **B — ecosystem surface** | model count, API breadth, adapter families, hardware plugins | no; a different product, argued in [the positioning](positioning.md) |
| **C — engineering coverage** | inside the same engine: graph-capture coverage, structured-output quality, sampling surface, prefix-reuse depth | partly; it is [Phase 2](../performance/old_hardware_roadmap.md) and it is where the next work is |

The classes are a judgement, and the table states the evidence for each rather than asserting it.

## The comparison

Read the two upstream columns as "claims to have", not "does on this box"; read the RelicLLM column
as "declared and wired", which is what the tree's own tests pin. Sources are named under
[Provenance](#provenance).

| Dimension | RelicLLM (this tree) | vLLM `main` | SGLang `main` | Class |
| --- | --- | --- | --- | --- |
| **Request scheduling** | one request at a time; `Scheduler` is the old `_request_lock` made a visible queue, width 1, and all five runtimes declare `supports_batch=False` (`relicllm/scheduler/core.py`, `relicllm/backends/capabilities.py`) | continuous batching + chunked prefill | continuous batching + chunked prefill | A |
| **KV memory** | each runtime allocates its own contiguous buffer at `max_model_len`; the page table and allocator exist but no runtime builds a cache from them (`relicllm/runtime/kv_spec.py`, `relicllm/runtime/kv_blocks.py`) | PagedAttention | paged KV, and a quantized KV cache | A |
| **Prefix reuse** | a whole-prefix anchor on the host, LRU under a byte budget, keyed by a token hash chain; live on `v41` and `mimo`, absent on `qwen4_exp` (`relicllm/models/prefix_cache.py`) | automatic prefix caching, block-hash granular | RadixAttention, plus HiCache tiers, object storage and a session cache | A / C |
| **Speculative decoding** | nothing on a serving path: `supports_speculative_decoding` is empty for every runtime, `v41` refuses the flag by name, MiMo's MTP/DFlash drafters are read and checked but "not wired to a serving path" | n-gram, suffix, EAGLE, DFlash | speculative decoding, plus an adaptive algorithm | A |
| **Graph capture / compilation** | `xing4` decode graph and `v41` decode graph, default on; **zero `torch.compile`**; `mimo` / `qwen4_exp` not yet | piecewise and full CUDA/HIP graphs, `torch.compile` for kernel generation and graph transforms | CUDA graph including a breakable variant | C |
| **Quantization** | strong at *consuming* low bit: FP4 E2M1, FP8 E4M3, GGUF Q4/Q5/Q8, IQ1/IQ2/IQ3, Q2 in the hot path, without expanding to a full fp32 copy | FP8, MXFP8/MXFP4, NVFP4, INT8, INT4, GPTQ/AWQ, GGUF, compressed-tensors, ModelOpt, TorchAO | quantization, plus a quantized KV cache | B / C |
| **Parallelism** | tensor parallel on one node through the local supervisor, plus a host expert bank; no pipeline, data, expert or context parallel, and **no multi-node** | tensor, pipeline, data, expert and context parallel | tensor, pipeline, DP-attention, expert and context parallel, plus prefill CP | B |
| **Disaggregation** | `--pd-mode` takes `off` or `scheduler`, and both are V4's **intra-box** phase resource policy (`relicllm/models/deepseek_v4/pd_scheduler.py`), not a network split | disaggregated prefill, decode and encode | PD and EPD disaggregation | B |
| **Serving API** | `/health`, `/alive`, `/ready`, `/metrics`, `/v1/models`, `/v1/chat/completions`, `/v1/completions`, `DELETE /v1/requests/{id}`; embeddings are declared off (`relicllm/server/openai.py`) | OpenAI-compatible, plus Anthropic Messages, gRPC and the batch file format | OpenAI-compatible plus a model gateway | B |
| **Structured output** | `response_format` rendered into the prompt by the checkpoint's own template; **no logit mask**, and `mimo`/`xing4`/`qwen4_exp` refuse the field | xgrammar or guidance | constrained decoding, including for reasoning models | C |
| **Sampling surface** | `logprobs` non-streaming and only on `torch`; streaming logprobs off everywhere; penalties / repetition / `min_p` on `torch` only (`relicllm/protocol/contract.py`) | logit processors, parallel sampling, beam search | thinking budget, separate reasoning, reasoning-aware compression | C |
| **Adapters** | none — no LoRA path anywhere in the tree | multi-LoRA over dense and MoE layers | LoRA | B |
| **Model coverage** | seven architectures, hand-written per checkpoint | the project advertises 200+ Hugging Face architectures | a cookbook per model, roughly a hundred | B |
| **Hardware** | CUDA `sm_75`/`sm_89`, Ascend 910A/B through the aclnn op dispatch, and CPU | NVIDIA, AMD, Intel, CPU, plus plugins for TPU, Gaudi, Spyre, Ascend, Rebellions, Apple, MetaX | NVIDIA, AMD, TPU, Ascend, Moore Threads, Apple | B, inverted |

## Class A — the architectural gap

This is the gap that makes the two stacks different objects. None of it is unknown to this
repository; [the roadmap](../performance/old_hardware_roadmap.md) states two of the four rows as its
own diagnosis, and [the scheduler lift](scheduler_2026_10.md) records the width-1 ruling.

- **No continuous batching.** Every runtime declares `supports_batch=False`, and the scheduler is
  explicit that it is one slot: "``acquire`` blocks exactly as the mutex did." The counter shape a
  batcher needs is already there (`num_computed_tokens`, `num_tokens_with_spec`), which is the part
  that was deliberately borrowed from vLLM's scheduler; the batch is not.
- **No paged-KV writer.** `relicllm/runtime/kv_blocks.py` is a real block table with a byte-denominated
  pool and the whole-page-never-rewritten rule, and its own docstring is equally clear that it is
  only half: "no runtime builds a cache out of these pages yet". Without a writer, a block-granular
  prefix cache has nothing to allocate from.
- **A per-token host round trip.** Every step ends `argmax` on the host, and the tree counts 541
  `.item()`/`.cpu()`/`.numpy()`/`.tolist()`/`.synchronize()` sites outside the tests. The roadmap's
  measurement of the result is the sharpest single number here: on DeepSeek-V4.1-Flash the device is
  busy **11%** of the wall.
- **No speculative path.** The drafters exist in research form (DSpark, DFlash2, MiMo's MTP/DFlash
  config) and no runtime's declaration offers one, so a client cannot ask for it and be served.

The corollary the roadmap states is the reason this class is last rather than first: **a faster
kernel cannot fix it.** The device is idle; the levers are ones that remove host work, and that is
why Phase 2 is de-synchronized sampling and graph coverage, and Phase 4 — last, and gated on the
baseline being trustworthy — is the batch.

## Class B — the ecosystem gap

The widest rows in the table are the ones a general serving framework has by construction, and this
runtime has chosen not to have: 200+ architectures against seven, a dozen adapter families against
none, five parallel dimensions against one, a gateway against a single server. That is the
positioning, not an oversight — this engine is written per checkpoint and per hardware family, and
[the positioning](positioning.md) is explicit that a model is worth adapting here only when a modern
part would have to split it across cards.

Two rows deserve their own note:

- **Model count is not a quality measure in either direction.** Upstream's breadth and this tree's
  hand-written depth answer different questions; a hand-written runtime is why V4.1's 475 GiB fits
  four 22 GiB cards and upstream's would not, and it is also why adding an eighth architecture here is
  a project.
- **Hardware support inverts.** vLLM and SGLang list Ascend among a dozen plugin platforms; this tree
  targets old CUDA cards and Ascend as the *primary* problem, including op dispatch through aclnn. The
  support matrices are not comparable because the axis is different.

## Class C — the engineering-coverage gap

These live inside the same engine, so they are the ones a reader can reasonably ask to see closed
first, and they are exactly Phase 2's items:

- **Graph capture is two of five runtimes.** `xing4` measured 4.3× (eager 178 ms → graphed 38 ms) and
  `v41`'s graph is now the default at 2.49× the decode TPOT and 3.3× the decode throughput; `mimo` and
  `qwen4_exp` have no capture, and `torch.compile` appears nowhere in the tree.
- **Structured output is a prompt instruction, not a constraint.** Where upstream holds the model to a
  grammar or a schema, this runtime renders the checkpoint's own `response_format` block and trusts
  the model. That is a weaker guarantee and it is declared as such rather than hidden.
- **Prefix reuse is whole-prefix, not block-granular.** The anchor store resumes a prompt that shares a
  stored prefix, but it cannot share a *block* between two live sequences, so two conversations with
  the same system prompt pay for it twice. Upstream's block-hash cache shares it.
- **The sampling surface is narrow by declaration.** Streaming logprobs are off everywhere, and
  several penalties apply on one runtime only. The tree's own contract refuses a field rather than
  accepting and dropping it, which is the right failure, but the field is still missing.

## Where the tree already holds the upstream shape

Worth recording, because it changes what "closing a gap" costs: the *shapes* of four upstream
mechanisms are already in the tree, deliberately borrowed, with the missing part named. A batch is
then an extension of state that exists rather than a new subsystem.

| Upstream shape | Where this tree holds it | What is still missing |
| --- | --- | --- |
| vLLM's scheduler carries counters, not a phase enum | `relicllm/scheduler/core.py` — `num_computed_tokens` / `num_tokens_with_spec`, with `phase` derived | a second request in the slot |
| vLLM's `dict[str, KVCacheSpec]` | `relicllm/runtime/kv_spec.py` — `FullAttentionSpec` / `SlidingWindowSpec` / `MLASpec` / `StateSpec`, two costs per cache ([the declaration](kv_declaration.md)) | a writer that allocates from it |
| vLLM's "a block hash is published for a full block only" | `relicllm/runtime/kv_blocks.py` — `share_prefix_from` takes whole pages only, default block 16 ([the table](kv_block_table.md)) | a paged cache in a runtime |
| vLLM's `hash_request_tokens` chain | `relicllm/models/prefix_cache.py` — `h_i = blake2b(h_{i-1} ‖ len ‖ block_i)` over 64-token blocks | block-granular sharing between live sequences |
| vLLM's and SGLang's flag conventions | `relicllm/backends/shared_options.py` and the generated `--help` ([the CLI surface](cli_surface_design_2026_09.md)) | nothing; this one is done |

## Which roadmap phase closes which gap

| Gap | Phase | State |
| --- | --- | --- |
| A trustworthy baseline, so any of the below is measurable | Phase 0 | landed as `relicllm bench` + the perf gate; the baseline itself is P0.2's deliverable |
| The fastest model must be reachable at all | Phase 1 | landed for `qwen4_exp` |
| Graph capture for `mimo` and `qwen4_exp`; de-synchronized sampling | Phase 2 | `v41`'s graph default landed; sampling still synchronous |
| The offload path: no fp4→bf16 CPU expansion, pinned expert views | Phase 3 | open |
| Paged KV and continuous batching | Phase 4 | open, and it is the whole of Class A |
| No speculative serving path | — | **no phase names it**; the drafters are research code today |

That last row is stated rather than smoothed over: the roadmap's ordering is host-work-removed per
unit of effort, and speculation did not make the list. If a client asks for it, the honest answer is
a refusal by name, which is what `v41` already does.

## Change log

| Date | What moved |
| --- | --- |
| 2026-10-16 | Page created. First reading: vLLM `main` and SGLang `main`. Class A recorded as the Phase 4 gap, with `scheduler_2026_10.md` and `kv_blocks.py` as the in-tree evidence; Class C recorded as Phase 2's. |

## Provenance

- **This tree.** Claims are read from the files cited in the table, and the two pages that state the
  gap in this repository's own words: [Performance roadmap: becoming the best on old hardware](../performance/old_hardware_roadmap.md)
  and [One scheduler, lifted](scheduler_2026_10.md). The support matrix is
  [the model guides](../models/README.md), and the CLI comparison is
  [the command-line surface](cli_surface_design_2026_09.md), read at vLLM `004e37e` / SGLang `f4de6ab`
  on 2026-09-28.
- **vLLM.** [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) — the `main` README,
  whose two capability lists are the upstream column, and
  [docs.vllm.ai](https://docs.vllm.ai/en/latest/features/) for the feature index.
- **SGLang.** [github.com/sgl-project/sglang](https://github.com/sgl-project/sglang) — the `main`
  README and the `docs/advanced_features/` page list at
  [docs.sglang.io](https://docs.sglang.io/), which is where its feature surface has moved.
- **Machine-readable.** `scripts/upstream_capability_inventory.py` reprints the two upstream readings;
  `--json` is the form to diff between two dates.