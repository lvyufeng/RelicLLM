<div class="rll-hero">
<div class="rll-hero__eyebrow">Multi-GPU inference runtime</div>
<h1 class="rll-hero__title">RelicLLM</h1>
<p class="rll-hero__tagline">
Run large language models on multi-GPU systems that were never meant to hold
them — including the cards everyone else stopped optimizing for.
</p>
<p class="rll-hero__badges">
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
<a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
<a href="https://github.com/lvyufeng/RelicLLM/blob/master/README.md#install"><img src="https://img.shields.io/badge/install-from%20source-lightgrey.svg" alt="Install from source"></a>
</p>
<p class="rll-hero__actions">
<a class="rll-btn rll-btn--primary" href="getting-started/">Get started</a>
<a class="rll-btn" href="https://github.com/lvyufeng/RelicLLM">View on GitHub</a>
</p>
</div>

**RelicLLM** is a PyTorch inference runtime for running large language models on multi-GPU systems
built from older accelerators. It owns the model implementations, the serving adapters and the
schedulers; the native kernels come from [relic-core](https://github.com/lvyufeng/relic-core), and
the retired C++ engine lives on in
[relic-engine](https://github.com/lvyufeng/relic-engine) as an archive.

The project started with DeepSeek-V4 on 4×RTX 2080 Ti and now covers DeepSeek-V4, MiniMax-M2.7,
GLM-5.2, Qwen3.8-27B, DeepSeek-V4.1-Flash, MiMo-V2.6-Flash and Ternary-Bonsai-2-27B. It is **not**
a single universal backend: each model has a runtime matched to its architecture and checkpoint
format, and it does not trade away per-hardware kernel optimization for portability.

Four of those models are served end to end over the OpenAI-compatible API:
**Qwen3.8-27B-FP8**, **Ternary-Bonsai-2-27B** on **one** card,
**DeepSeek-V4.1-Flash** through `serve --backend v41`, and
**MiMo-V2.6-Flash** through `serve --backend mimo`.

!!! warning "Status"

    Research and engineering software. Every number on this site is a measurement
    from a specific checkpoint and hardware configuration, not a performance
    guarantee. Read [Benchmarking and reporting rules](guides/benchmarking.md)
    before comparing any two results.

## News

- **[2026/09] Xing4.0-29B-A4B is served end to end on one card.** A 29B mixture-of-experts model —
  MLA attention, 64 routed experts activated top-4 plus one shared, and **four residual streams per
  block** mixed by a matrix hyper-connection — released as an official `IQ4_NL` GGUF that fits a 22 GiB
  card whole, so **all 64 experts of all 38 MoE blocks stay resident** and there is no host bank and no
  tensor parallel group. `serve --backend xing4` answers chat with SSE streaming, cancel and
  `/metrics`: **75.22 tok/s of prefill** at a 4,096-token prompt and **6.72 tok/s of decode**, flat in
  context to 32,768 tokens. The run found two defects the unit tests could not — the four residuals
  were carried at fp16 while the checkpoint's activations reach 1e5, and a shared cache was never reset
  between requests — and both are fixed and guarded. Decode is now launch-bound rather than
  bandwidth-bound: 11,536 launches a step against 46.5 ms of device work.
  [Model page](models/xing4.0-29b-a4b.md) · [Design record](architecture/xing4_0_29b_a4b_design.md)
- **[2026/09] Ternary-Bonsai-2-27B is served end to end on one card.** A 27B hybrid-attention model
  — 48 Gated DeltaNet layers and 16 GQA layers over a dense MLP — released as a GGUF whose weights
  are **1.75 bits each** (GGML type 143, 5.53 GiB), with a Hadamard rotation declared in the file
  and consumed as ternary end to end: prefill is **636.0 tok/s at a 4,096-token prompt** and decode
  **25.9 tok/s** against the same card's upstream reference of 642.5 and 30.7, and 5.53 GiB of
  weights leaves room for a **245,760-token context** (262,144 with an fp8 KV cache).
  [Model page](models/ternary-bonsai-2-27b.md) ·
  [Design record](architecture/bonsai_2_27b_design.md)
- **[2026/09] MiMo-V2.6-Flash is served end to end.** `serve --backend mimo` runs the release as four
  processes on four cards, with the 149.81 GiB of routed experts in a host bank and the 48-layer
  backbone on the GPUs. The attention is divided along the checkpoint's own four-way `qkv_proj`
  partition and joined by an all-gather, so a **262,144-token prompt reaches 104.04 tok/s of prefill**
  and a decode step at that depth is **180.0 ms — 5.56 tok/s**, with the four ranks byte-identical.
  The decode step's softmax, rotation and norms are each one kernel now instead of eighteen, ten and
  two dispatches, which is a short-context step of **156.3 ms and 6.40 tok/s**, and keeping each routed
  layer's hottest experts on the card takes it to **117.3 and 8.53**.
  [Model page](models/mimo-v2.6-flash.md)
- **[2026/09] DeepSeek-V4.1-Flash is served end to end.** `serve --backend v41` runs the released
  475 GiB checkpoint as four processes on four 22 GiB cards, with the 457.8 GiB of routed experts
  pinned in host memory rather than resident on the device. The runtime accepts up to 262,144 tokens
  of context; a 260,244-token prompt measures 150.3–152.0 tok/s of prefill and 3.48–3.54 tok/s of
  decode. Cross-request prefix caching landed in the same batch, so a prompt whose prefix has already
  been served forwards only its tail.
  [Model page](models/deepseek-v4.1-flash.md) ·
  [Run record](performance/deepseek_v4_1_flash_served_gate.md)
- **[2026/09] Qwen3.8-27B-FP8 gained a native OpenAI-compatible server** — health and model
  discovery, streaming and non-streaming chat and completions, per-token log probabilities,
  stop-sequence truncation, request-field refusals and concurrent scheduler admission, all verified
  against a real checkpoint. [Model page](models/qwen3.8-27b-fp8.md)
- **[2026/08] Two external speculative drafters for Qwen3.8-27B.** DSpark came first and
  [DFlash2](architecture/qwen3_8_27b_fp8_design.md#external-dflash2-speculative-decoding) after it, measuring
  2.78× full-request and 3.02× decode on a 512-token fixture with exact token parity in every case.
  Both are opt-in, because their gains are acceptance-dependent.

<details markdown="1">
<summary>More</summary>

- **[2026/08] Qwen3.8-27B-FP8 on the C++/CUDA runtime** — FP8 E4M3 Safetensors text generation at
  TP4, 864.54 tok/s of prefill and 43.22 tok/s of decode on a 512-token prompt, with a 256K context
  path and a persistent TP4 worker that keeps prefix state alive across requests.
- **[2026/07] GLM-5.2 text generation** through the shared GGUF raw-block path.
- **[2026/06] MiniMax-M2.7 on GGUF `UD-IQ1_M`** — ~104.9–107 tok/s full-model 256-token prefill, and
  a 43-layer decode benchmark at 10.32 tok/s after fused RMSNorm.
- **[2026/05] DeepSeek-V4-Flash**, the checkpoint this project started on — FP4/FP8 Safetensors and
  GGUF Q2/IQ2/IQ1 generation, ~401 tok/s of C++ FP4 prefill at 32K–64K.
  [Model page](models/deepseek-v4.md)

</details>

The five visible entries are the same five the [repository README](https://github.com/lvyufeng/RelicLLM#news)
points at, as one-liners; this list is the archive, and is where the numbers behind each entry live.

**These entries were written before the rename, and their commands are the old spelling.** The
installed entry point is `relicllm`; a page that says `pocketllm serve --backend v41` describes a
run performed with the same code under its former name, and the command to type now is
`relicllm serve --backend v41`. The rename reached the package, the CLI and the repository's own
`CLAUDE.md`/README but not the bodies of the measurement records, which are left as written so the
numbers still sit next to the conditions they were taken under.

## What RelicLLM provides

<div class="grid cards" markdown>

- **Model-specific inference paths**

    ---

    Hybrid attention, MLA, GQA, Gated DeltaNet, dense MLPs and routed MoE layers,
    each with the kernels its architecture actually needs.

- **Low-bit execution without expansion**

    ---

    FP4, FP8 E4M3, GGUF Q4/Q5/Q8, IQ1/IQ2/IQ3 and Q2 paths consume quantized
    blocks directly in the hot path. Raw weights are not expanded to a full FP32
    copy where it matters.

- **Consumer-GPU parallelism**

    ---

    TP4/NCCL execution on PCIe-connected GPUs with no NVLink, plus CPU/NUMA
    expert placement for checkpoints that do not fit in device memory.

- **Separate prefill and decode dispatch**

    ---

    Large-row kernels are optimized independently from the single-token latency
    path, so improving one does not cost the other.

- **A host-PyTorch path for a checkpoint the cards cannot hold**

    ---

    `serve --backend v41` runs DeepSeek-V4.1-Flash as four processes, one a card,
    over the 475 GiB checkpoint: the dense tree and the packed FP4 experts execute
    on the GPUs while the routed experts read from a pinned host bank.

- **Inspection and validation tools**

    ---

    GGUF architecture and spec reports, Safetensors audits, tensor-shape checks,
    numerical parity tests and real-checkpoint benchmarks.

</div>

## Measured on RTX 2080 Ti

This table is an index, not the record. Each row's headline is one number from one configuration, and
the thing to read is the model page it links to — which carries the conditions the number was taken
under, the configuration that produced it, and a `## Known limitations` section saying what it does
not cover. Nothing here may be averaged into a single score, and no two rows are directly
comparable unless their checkpoint, prompt, runtime, warm state and measurement convention match; see
[benchmarking and reporting rules](guides/benchmarking.md).

Real checkpoints, PCIe Gen3, no NVLink, single requests, TP4 where applicable and **one card where
the checkpoint fits on one**.

| Model | Checkpoint / format | Validated path | Headline result |
| --- | --- | --- | --- |
| [DeepSeek-V4.1-Flash](models/deepseek-v4.1-flash.md) | Safetensors FP8 E4M3 dense + FP4 E2M1 experts | `serve --backend v41`, host PyTorch, TP4 | 150.3–152.0 tok/s prefill at a 260,244-token prompt, 3.48–3.54 tok/s decode, one request at a time |
| [MiMo-V2.6-Flash](models/mimo-v2.6-flash.md) | Safetensors FP8 E4M3 dense + MXFP4 experts | `serve --backend mimo`, 48 layers on the cards, experts out of a 149.81 GiB host bank, TP4 | 104.04 tok/s prefill at a 262,144-token prompt, 5.56 tok/s decode at that depth, 6.40 tok/s at a short context and 8.53 with experts resident |
| [Qwen3.8-27B-FP8](models/qwen3.8-27b-fp8.md) | Safetensors FP8 E4M3 | C++/CUDA TP4, GPU-resident FP8 | 864.54 tok/s prefill, 43.22 tok/s decode on a 512-token prompt |
| [Ternary-Bonsai-2-27B](models/ternary-bonsai-2-27b.md) | GGUF `PTQ1_0` (GGML type 143), 1.75 bits a weight, 5.53 GiB | native C++/CUDA, **one card**, no flag needed | **636.0 tok/s prefill** at a 4,096-token prompt and 25.9 tok/s decode, against the same card's upstream reference of 642.5 and 30.7 |
| [Xing4.0-29B-A4B](models/xing4.0-29b-a4b.md) | GGUF `IQ4_NL` (GGML type 20), 4.5 bits a weight, 17.94 GiB resident | `serve --backend xing4`, **one card**, all 64 experts of every layer resident | **75.22 tok/s prefill** at a 4,096-token prompt and 6.72 tok/s decode, 79.15 tok/s prefill at 512 tokens; decode is host-launch-bound, not bandwidth-bound |
| [DeepSeek-V4-Flash](models/deepseek-v4.md) | Safetensors FP4/FP8; GGUF Q2/IQ2/IQ1 | PyTorch heterogeneous, C++/CUDA, GGUF TP4 | C++ FP4: ~401 tok/s prefill at 32K–64K; ~3.7 tok/s decode |
| [MiniMax-M2.7](models/minimax-m2.7.md) | GGUF `UD-IQ1_M` | Raw-block CUDA, GGUF TP4 | 256-token prefill ~104.9–107 tok/s; 43-layer decode benchmark 10.32 tok/s |
| [GLM-5.2](models/glm-5.2.md) | GGUF `UD-Q2_K_XL` | Raw-block CUDA, GGUF TP4 | ~0.79 tok/s prefill; ~0.66 tok/s decode |

The [support matrix](models/README.md) is the same runtime status with the format and validation
detail; the model pages separate architecture specifications from what this runtime actually
implements. `inspect`, `smoke` and a benchmark are not automatically equivalent to a production
serving guarantee.

## Documentation

| Directory | What it holds |
| --- | --- |
| [Guides](guides/index.md) | Benchmark reporting rules and the serving API |
| [Model guides](models/README.md) | The support matrix and one page per checkpoint |
| [Performance](performance/index.md) | Run records and bottleneck analyses for capabilities that are live today |
| [Architecture](architecture/index.md) | Per-model design records, the prefix-cache designs, and the CLI surface |
| [Migration](migration/index.md) | Breaking-change notes for this runtime's configuration and defaults |
| [Reports](reports/dsv4_2080ti_report.pdf) | Rendered long-form reports |

Every document in this tree is listed in one of those sections, so nothing is
reachable only by guessing a filename. Pages whose subject was the operator layer or the retired
engine are not here; they sit next to the code they describe, in
[relic-core](https://lvyufeng.github.io/relic-core/) and
[relic-engine](https://github.com/lvyufeng/relic-engine/tree/master/docs) — the cross-repository
links throughout this site point at them.

## Elsewhere in the repository

- [Repository home](https://github.com/lvyufeng/RelicLLM) — the install steps and the news index
- [relic-core](https://github.com/lvyufeng/relic-core) — the kernels, built and installed first

## License

RelicLLM is released under the [MIT License](https://github.com/lvyufeng/RelicLLM/blob/master/LICENSE).
Model weights, tokenizer files, CUDA, PyTorch, GGUF assets and other third-party
components are governed by their own licenses; RelicLLM's code license grants no
additional rights to third-party model assets.