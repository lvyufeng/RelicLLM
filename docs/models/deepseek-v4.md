# DeepSeek-V4-Flash

PocketLLM's most mature model family: a 43-layer MoE with MLA and sparse/indexed attention, 256 routed
experts activated top-6 per token, shipped as Safetensors FP4 experts with FP8 dense tensors and also
available in GGUF Q2/IQ2/IQ1 variants. It runs on a PyTorch plane — the `torch` backend — and used to
run on a native C++/CUDA engine that has since been retired, with separate loading, kernel and
placement policies per format.

- **Backend**: `torch` (host PyTorch, OpenAI-compatible server); the GGUF path is a separate
  entrypoint, and the retired native C++/CUDA engine is **not** available from this repository
- **Parallelism**: TP4
- **Context**: 65,536 tokens in the validated configuration
- **Validated on**: 4×RTX 2080 Ti 22 GiB, PCIe Gen3, no NVLink

## Overview

| Field | Value |
| --- | ---: |
| Transformer layers | 43 |
| Hidden size | 4,096 |
| Attention heads / head dim / RoPE dim | 64 / 512 / 64 |
| Routed experts / active | 256 / top-6 |
| Shared experts | 1 |
| Expert intermediate size | 2,048 |
| Attention | MLA with a sparse/indexed C4 indexer and compressor |
| Original sequence length | 65,536 |

Four execution paths existed, and which one you wanted depended on the format and the memory
situation. Two of them were the C++ engine's and went with it; the two PyTorch ones are what this
repository still has:

- **C++ FP4 Safetensors** — the retired engine's served path. The dense and attention work ran on the
  GPUs, the routed experts were resident or kept in host memory and staged as active quantized blocks,
  with TP4 NCCL reductions, compressed/indexed attention, grouped prefill MoE, deterministic expert
  reduction and the embedded OpenAI-compatible server.
- **PyTorch FP4 Safetensors** — heterogeneous routed-expert serving and performance experiments. This
  is the `torch` backend, selected by `--backend auto` for a `model_type=deepseek_v4` checkpoint.
- **C++ GGUF Q2/IQ2/IQ1** — the retired engine's low-bit path: TP4 generation, grouped prefill and
  active-expert decode out of raw-block kernels. The `torch` backend reads a DeepSeek-V4 GGUF
  (`general.architecture=deepseek4`, including the `deepseek4_mtp_support` variant), but this low-bit
  C++ path is not part of it.
- **DSpark** — an experimental speculative path. The C++ sequential verify implementation was a
  correctness path, not an end-to-end speedup claim.

Decode is limited by PCIe expert staging rather than by Python overhead, which is why where the
experts live matters more here than which kernel runs.

## Run it

```bash
# C++ Safetensors backend, TP4 — historical: the `cpp` backend is retired and this no longer runs
python -m pocketllm serve \
  --model /path/to/DeepSeek-V4-Flash \
  --backend cpp \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --port 8000
```

```bash
# PyTorch OpenAI-compatible server
CKPT_PATH=/path/to/DeepSeek-V4-Flash-w8a8 bash scripts/run_openai_server.sh

# GGUF Q2/IQ2 serving
CKPT_PATH=/path/to/deepseek-v4.gguf \
TOKENIZER_PATH=/path/to/DeepSeek-V4-Flash-tokenizer \
bash scripts/run_gguf_q2_layer_pp.sh
```

Inspect a GGUF checkpoint without running it:

```bash
PYTHONPATH=$PWD python -m relicllm.cli.inspect_gguf \
  --gguf-path /path/to/deepseek-v4.gguf \
  --summary --validate-ds4-q2
```

## What is supported

| Capability | State |
| --- | --- |
| C++ FP4 Safetensors generation and OpenAI-compatible serving | Retired with the `cpp` backend |
| PyTorch FP4 heterogeneous routed-expert serving | Supported, via the `torch` backend |
| C++ GGUF Q2/IQ2/IQ1 TP4 generation | Retired with the `cpp` backend |
| Grouped prefill MoE and deterministic expert reduction | Supported, on by default |
| Host-resident routed experts with bounded staging | Supported |
| 65,536-token context (batched attention) | Supported |
| FlashMemory 1M context | Separate path with its own enablement and validation constraints |
| DSpark speculative decoding | Experimental; the sequential verify path is a correctness path |
| TP4 GGUF Q2 all-reduce precision | FP32, required for parity — the BF16 round trip compounded quantization drift |
| FP4 MoE expert reduction | Deterministic and on by default, to avoid `atomicAdd` run-to-run drift |

## Performance

4×RTX 2080 Ti 22 GiB, TP4, single request, real checkpoints.

| Path | Prompt | Prefill | Decode | Peak GPU/rank |
| --- | ---: | ---: | ---: | ---: |
| C++ FP4 | 2,101 tokens | ~275 tok/s | ~3.7 tok/s | ~7 GiB |
| C++ FP4 | 32,768 tokens | **~402 tok/s** | not reported | ~11.2 GiB |
| C++ FP4 | 65,536 tokens | **~401 tok/s** | ~3.7 tok/s | ~14.5 GiB |
| PyTorch FP4 (host experts) | 2,148 tokens | ~321 tok/s after warmup | 3.49 tok/s | — |
| PyTorch FP4 (host experts) | 65,536 tokens | ~255 tok/s | n/a | — |
| GGUF Q2/IQ2 (warm) | 2,148 tokens | 213–216 tok/s | 3.75–4.44 tok/s | — |

The long-context batched-attention path is what makes the 32K/64K rows what they are: it improved them
by roughly 8.6–8.8× over the earlier per-position implementation, and the earlier alternative
continuation kernels regressed substantially. Read the table as path milestones rather than as one
controlled comparison — each row comes from its own session, and only rows with the same format,
prompt and warm state are directly comparable.

## Known limitations

- **Decode is host-bound on this path.** Host-resident routed experts make it sensitive to PCIe, NUMA,
  page-cache and CPU behaviour; nothing in the current design hides that.
- **One-GPU GGUF Q2 is a smoke path.** It works for very short prompts and is not practical beyond
  them — see [the single-GPU note](deepseek-v4-gguf-q2-single-gpu.md).
- **DSpark's multi-token verify can change floating-point reduction order and token selection.** Read
  [the DSpark note](../performance/dspark.md) before making parity claims.
- **FlashMemory's 1M context is a separate path** with its own enablement and validation constraints,
  not an extension of the validated 65,536-token configuration.
- **The native C++/CUDA paths are retired from this repository.** They were served by
  `pocketllm serve --backend cpp`, which is no longer accepted; the `pocketllm_engine` binary and the
  code behind them belong to the archived
  [relic-engine](https://github.com/lvyufeng/relic-engine). `--backend auto` now reaches this model
  through the `torch` backend, which reads a `model_type=deepseek_v4` Safetensors directory or a
  `deepseek4`-family GGUF.

## Where the detail is

- [Design and measurements](../architecture/deepseek_v4_design.md) — the validated geometry, the
  three paths, the precision rules, and the run records behind each row.
- [DSpark speculative decoding](../performance/dspark.md) and
  [FlashMemory 1M context](https://github.com/lvyufeng/relic-engine/blob/master/docs/performance/flashmemory_1m_context.md).
- [GGUF Q2 on one GPU](deepseek-v4-gguf-q2-single-gpu.md) — the historical single-card measurements.
- [Benchmarking rules](../guides/benchmarking.md) — required before any DeepSeek-V4 number is quoted.
- The support matrix in [models/README.md](README.md).
