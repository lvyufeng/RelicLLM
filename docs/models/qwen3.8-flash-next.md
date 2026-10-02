# Qwen3.8-Flash-Next

A 48-layer hybrid text model — **GatedDeltaNet** linear-attention layers interleaved with a
query-sparse full attention every fourth layer — carrying 512 routed experts activated top-10 and a
95 GiB embedding table. RelicLLM runs the text stack as four processes on four cards, with each
rank's disjoint share of the experts copied into host RAM and only the draw staged to the card,
behind the same OpenAI-compatible server as its other models.

- **Backend**: `--backend qwen4_exp` (also selected automatically from the checkpoint's `config.json`)
- **Parallelism**: 4 GPUs, TP4 over PCIe. The dense projections and the `lm_head` are sharded and the
  routed experts are owned round-robin, `expert_id % world_size == rank`.
- **Context**: 32,768 by default; the checkpoint's own `max_position_embeddings` is 262,144
- **Validated on**: 4×RTX 2080 Ti 22 GiB, PCIe Gen3

## Overview

The checkpoint is a conditional-generation wrapper: `model_type` is `qwen4_exp` at the root and
again as `qwen4_exp_text` inside `text_config`, beside a `vision_config` this runtime does not
execute. `--backend auto` recognizes either spelling and routes here.

| Field | Value |
| --- | ---: |
| Text layers | 48, of which 36 are `linear_attention` and 12 `full_attention` (`full_attention_interval` 4) |
| Hidden size | 2,560 |
| Full attention | 24 query heads, **2** KV heads, head dim 256 |
| Linear attention | GatedDeltaNet: 16 key heads, 48 value heads, a fixed recurrent state |
| Routed experts | 512, activated top-10, `moe_intermediate_size` 640 |
| Vocabulary | 248,320 |
| End of turn | `eos_token_id` 248,044 |
| Dense weights a rank | ~2.6 GiB (attn/linear-attn projections, hyper-connections) |
| Expert shard a rank | 128 of 512 experts × 48 layers = 56.25 GiB, host-resident and pinned |
| PLE table | ~95 GiB, host-side, never staged whole |

Two properties decide how this runtime is built and what its numbers mean.

**Decode is bound by bytes over PCIe, not by arithmetic.** A decode token draws top-10 of 512
experts across 48 layers; round-robin ownership makes that 120 expert copies a rank, 1,099 MiB, or
**103.6 ms of H2D** on PCIe Gen3. At an 8,192-token context the compute underneath is 35.0 ms —
dense 11.2, full attention 8.8, routed MoE 4.4, NCCL 8.8, `lm_head` 1.8 — so **28.6 tok/s is the
ceiling if the H2D were free**. Everything below that ceiling is the expert pipeline, and the only
lever that reaches it is a resident or narrower expert: INT4 experts and fully-resident experts land
on exactly the same 35.0 ms. `--expert-cache` is the flag that buys residency, and the default is
`0` — restage every step. [The performance record](../performance/qwen4_exp_performance.md) carries
the sweep and the projection.

**Prefill is the opposite: fast, and the chunk is its whole tuning surface.** The same 8K prompt in
one 8,192-token chunk measured **786.27 tok/s** at BF16 (14.108 GiB a rank), against 154.70 tok/s at
a 512-token chunk — the chunk amortizes the per-layer dispatch and staging. The chunk also bounds
the QSA score matrix, so it is not free to widen: a single-shot 64K prefill would need a 64K × 64K
mask a layer. `--prefill-chunk-tokens` is the lever.

## Run it

```bash
relicllm serve \
  --model /path/to/Qwen3.8-Flash-Next \
  --backend qwen4_exp \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --prefill-chunk-tokens 8192 \
  --host 0.0.0.0 --port 8000
```

Then an ordinary OpenAI request:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen4_exp","messages":[{"role":"user","content":"Hello!"}],"max_tokens":64}'
```

`--backend auto` selects this runtime from the checkpoint, so the flag is optional.

### Serving options

| Flag | Effect |
| --- | --- |
| `--tensor-parallel-size N` | Ranks. 4 is the validated width; each rank takes one card. |
| `--prefill-chunk-tokens N` | Tokens one prefill forward takes. The prefill rate's main lever (512 → 8,192 is ~5×). |
| `--expert-cache N` | Staged experts the card keeps instead of re-copying each step. `0` restages every step. |
| `--pin-experts` / `--no-pin-experts` | Page-lock the host shard so the H2D copies are asynchronous. On by default. |
| `--host-expert-memory` / `--no-host-expert-memory` | Hold the rank's shard in host RAM rather than reading the mapping. On by default. |
| `--tensor-parallel-startup-timeout S` | Rank startup budget. **Raise this** — see Known limitations. |
| `--max-model-len T` | Positions the caches are sized at. Default 32,768. |

### Non-server entrypoints

`python -m relicllm.models.qwen4_exp.runtime --model DIR --prompt "…"` runs one prompt under
`torchrun`, and is what the measurements above used. It is a single-shot path: it loads, answers, and
exits, so it does not serve.

## What is supported

| Capability | Status |
| --- | --- |
| OpenAI chat / completions | Yes |
| SSE streaming | Yes |
| Cancellation and stop strings | Yes — agreed by a per-step broadcast, so a rank never leaves its peers inside a layer's all-reduce |
| Prefix caching | **No.** A repeated prefix is forwarded again; `supports_prefix_caching` reports false rather than promising a switch that does nothing. |
| Continuous batching | **No.** One mutable cache serves one sequence; `--max-batch-size` above 1 is refused at startup. |
| Sampling | **Greedy only.** The runtime's loop takes `argmax`; `temperature`/`top_p` are not applied. |
| Vision | **Not executed.** The checkpoint ships a `vision_config`; this runtime serves the text stack only. |

## Performance

**Served, measured 2026-10-02 on 4×RTX 2080 Ti, TP4 over PCIe Gen3, one request at a time:**

| Quantity | Measured |
| --- | ---: |
| Time to first token, 512-token prompt, chunk 8,192 | 4.46 s |
| Decode, default `--expert-cache 0` | 275 ms/step (3.6 tok/s) |
| Device memory a rank, resident | 3.3 GiB |
| Cold expert preload, 56.25 GiB a rank | 3,446 s (57 min) |

TTFT here is dominated by the prefill forward; the 4.46 s includes the chunked prefill of the
512-token prompt plus one distribution.

**Decode at 3.6 tok/s is the pipeline, not the model.** A device-utilization sample taken across a
128-token generation read between **7% and 94%**, swinging with each step: the card is idle while
1.1 GiB of experts crosses PCIe, which is precisely the 103.6 ms H2D floor the record predicts. The
28.6 tok/s ceiling is reachable only by taking the experts off that path — `--expert-cache` for
residency, or quantized experts — and is not what the default configuration does.

The measurement conditions above are one run each, not a sweep. Read
[the performance record](../performance/qwen4_exp_performance.md) for the prefill sweep, the phase
attribution and the decode projection, and [the reporting rules](../guides/benchmarking.md) before
comparing any of these to another model's numbers.

## Hardware and memory

Per rank, at a world of four: ~2.6 GiB of dense weights and a 0.30 GiB `lm_head` shard on the card,
3.3 GiB resident in the served configuration, and 56.25 GiB of pinned host RAM for the expert shard.
The 95 GiB embedding table stays host-side. Four ranks therefore want roughly 225 GiB of host RAM
and four 22 GiB cards.

## Known limitations

- **The startup timeout must be raised.** The pinned expert preload took **3,446 s a rank** on a
  cold page cache in the validated run — against the 300 s default of
  `--tensor-parallel-startup-timeout`, and against the ~53 s a warm preload takes once the release
  is in page cache. A cold host that keeps the default will be torn down mid-preload with
  `missing ranks: 0, 1, 2, 3`. Pass an hour or more.
- **Decode is PCIe-bound at the default `--expert-cache 0`.** See **Performance**; 3.6 tok/s is the
  served number and 28.6 tok/s is what the same hardware reaches once the experts stop crossing the
  bus.
- **Greedy only.** A request that needs `temperature` or `top_p` cannot be honoured; the loop takes
  `argmax`.
- **No prefix reuse.** A chat turn that resends its history forwards the history again, and on a
  runtime whose prefill is chunked this is a real per-turn cost.
- **No continuous batching.** Requests serialize behind one cache, so a served deployment answers
  one at a time.
- **No vision.** The checkpoint's `vision_config` is not executed.

## Where the detail is

- [The performance record](../performance/qwen4_exp_performance.md) — the prefill sweep, the phase
  attribution, the decode ceiling and the expert-placement analysis.
- [Hardware adaptation triage](../guides/hardware_triage.md) — where this checkpoint's rates sit
  against the box's other models.
- [The support matrix](README.md) — runtime status for every model.
- [Benchmarking and reporting](../guides/benchmarking.md) — the measurement conventions.