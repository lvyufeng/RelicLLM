# Benchmarking and reporting rules

PocketLLM reports prefill and decode separately because they stress different parts of the runtime. Prefill is a multi-token operation dominated by projection/GEMM, attention over the prompt, and expert batching. Decode is a single-token latency path dominated by recurrent state updates, KV attention, expert staging, kernel launch overhead, and TP communication.

## Required metadata

A benchmark result should record:

- model name and exact checkpoint/quantization variant;
- PocketLLM commit or release;
- runtime (`torch`, `v41`, `mimo`, `xing4`, `qwen4_exp`; record the backend even under `auto`);
- GPU model, per-card memory, GPU count, TP/EP world size, PCIe/NVLink topology;
- CPU model, NUMA layout, system RAM, CUDA/driver/runtime versions when relevant;
- prompt token count, generated token count, context length, and tokenizer source;
- cold/warm state, page-cache prewarm, expert-cache state, and relevant environment switches;
- prefill wall time and tokens/s;
- decode wall time and tokens/s for generated tokens after the first result;
- peak GPU memory per rank and host/pinned memory when it is material;
- token parity, numerical error, or an explicit statement that no reference comparison was run.

## The record `relicllm bench` writes

Most of that list is supplied automatically by `relicllm bench serve`, which launches the server,
measures it on [vLLM's terms](latency_metrics.md), and writes one JSON envelope per run:

```bash
relicllm bench serve --scenario decode --scenario prefill-8k \
    --json-out /tmp/bench.json \
    -- --model /path/to/checkpoint --backend auto --tensor-parallel-size 4
```

The envelope carries `schema`, `tool`, `git_commit`, `host` (hostname, platform, Python) and `cuda`
(device names and count) around a `scenarios` map of the per-workload records. `launch` is the
**literal argv** that started the server when the command launched one, and `null` when `--base-url`
pointed at a server somebody else started — the two are different claims about a number, so the record
keeps them apart rather than filling the gap with a guess.

**Every metadata reader is best-effort and never invents a value.** A field that cannot be read — no
`nvidia-smi`, no torch, no checkout — is **absent** from the record, not `0` and not `"unknown"`. That
is the same rule a missing measurement follows everywhere else here, and it is what makes an absent
`cuda` key mean "unknown host" rather than "no GPU".

What the record does **not** carry is the switches that live in the environment. `POCKETLLM_*`
(`POCKETLLM_XING4_DIR`, the bank names, the residency flags) change where experts live and therefore
what the number means; a run whose result is to be quoted should name the ones it set alongside the
record. The envelope's `env_snapshot` helper exists for a caller that wants them folded in.

## Timing convention

PocketLLM's standard timed generation path measures the first model result as prefill and measures subsequent single-token forwards as decode:

```text
prefill_tps = prompt_tokens / prefill_seconds
decode_tps  = decode_tokens / decode_seconds
```

The first generated token is produced by the prompt forward and therefore belongs to the prefill phase. `decode_tokens` counts only later generated tokens. A benchmark that uses another convention must say so explicitly.

Do not report a combined tokens/s number as a replacement for these two fields. Combined throughput is useful only as an additional end-to-end figure.

**The seam between the two is a device fact, not a host clock.** A forward is asynchronous: the host
returns from the prompt's forward with its kernels still in flight, so a clock read at that moment
does not measure the prefill. Without a synchronisation there, the prompt's remaining work drains
inside the *first* decode step instead — `prefill_seconds` under-reports by the last chunk's device
time and `decode_seconds` over-reports by exactly as much. The size of that error is not a constant:
it follows the width of the last prefill chunk and the depth of the context, so it distorts the two
rates against each other and distorts a decode-versus-context trend most of all.

So a timed path has to drain at the seam. `relicllm/models/xing4_0/generate.py`'s `_drain` is that, and
what it removes is measured in
[Xing4.0-29B-A4B: the prefill/decode seam](../performance/xing4_0_rate_clock_split.md). MiMo has the
same drain at the same seam, added after that measurement: the **position** is pinned by
`tests/test_mimo_serving.py`, but what it removes on MiMo is **not yet measured** — a decode rate read
off that model before the fix is short by its last prefill chunk.

This is one of two conventions in this repository. For client-observed serving
numbers — TTFT, TPOT, ITL, E2EL, throughput and goodput, defined the way vLLM
defines them — see [Serving latency metrics](latency_metrics.md). The two answer
different questions and neither replaces the other; a number that claims to be
`prefill_tps` or `decode_tps` must be measured the way this section describes.

## Comparison rules

Results are directly comparable only when the checkpoint, quantization, runtime, prompt, generation length, hardware, TP/EP layout, and warm/cold policy match. In particular:

- PyTorch heterogeneous expert staging, GGUF raw-block TP, GPU-resident FP4, and Qwen GPU-resident FP8 are different execution paths.
- A 256-token prefill microbenchmark is not a long-context prefill result.
- A synthetic kernel benchmark is evidence about a kernel, not about model-level TPS.
- A first-token match does not prove that a quantized KV-cache path remains sequence-identical for later tokens.
- Experimental opt-in switches must be reported with their values and must not silently replace the default baseline.
- If rank outputs differ only by floating-point reduction order, report the max error and token-level result rather than claiming exact tensor identity.

## Recommended table

Use this shape for model pages:

| Model/checkpoint | Runtime | Hardware / parallelism | Prompt | Decode tokens | Prefill tok/s | Decode tok/s | Peak GPU/rank | Correctness |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |

Then add a short paragraph describing warmup, cache state, and the source script or command.

## PocketLLM baseline machine

Many repository measurements use:

- 4× NVIDIA GeForce RTX 2080 Ti, 22 GiB each;
- Turing architecture, no native BF16/FP8/FP4 tensor cores;
- PCIe Gen3 and no NVLink;
- dual-socket Intel Xeon E5-2696 v4, 1 TiB system memory;
- one process/rank per GPU, usually TP4.

These constraints are part of the result. Moving to a newer GPU, NVLink system, or different NUMA placement can change the bottleneck and invalidate an apparent A/B comparison.

## Reproducibility and honesty rules

1. Prefer real checkpoints and real prompts for end-to-end claims.
2. Keep short and long prompt cases separate.
3. Run repeated measurements serially when comparing single-request latency.
4. Preserve the fastest known command before changing an optimization switch.
5. Report regressions and disabled experiments alongside wins.
6. Keep model architecture specifications separate from runtime support status.
7. Link the test, script, commit, or analysis note that produced each non-trivial number.
8. Establish run-to-run stability before comparing generated tokens across configurations. Repeated
   runs of one identical binary can disagree with each other, and then the comparison measures the
   spread rather than the change. On the Ascend backend with TP all-reduce active, four runs of two
   binaries over a 4966-token prompt produced three distinct greedy step-0 tokens and top logits
   spanning 10.42-10.94 — well above fp16 rounding ([measurement](https://github.com/lvyufeng/relic-engine/blob/master/docs/performance/ascend_gated_delta_slice.md#the-generated-tokens-are-not-a-usable-ab-signal-here)).
   That instance has since been attributed to a pooled workspace race rather than to the platform, and
   the runs **were repeated on the fix**: eight runs of one binary over a 4966-token prompt with the
   TP all-reduce still active produced one identical step-0 token and one identical 9-token sequence,
   against three distinct tokens in four runs before. The rule is kept anyway, because establishing
   the stability is cheap and discovering its absence late is not — the eight runs do not establish
   that the race was the *only* mechanism, only that this stack currently reproduces.
   Numerical A/B comparisons against a host reference inside one process are not affected by this,
   and neither are repeated timings of a kernel; it is specifically generated-token comparison that
   needs the stability established first.
