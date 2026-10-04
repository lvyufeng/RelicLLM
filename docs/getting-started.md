# Getting started

This page is the short route from a checkout to a running model. The repository
[`README.md`](https://github.com/lvyufeng/PocketLLM#installation) is the
authoritative install text — it carries the caveats about `--no-build-isolation`
and about which prerequisites a fresh virtualenv does not have. What follows is
the same route with the detours removed.

## Requirements

- Python >= 3.10
- PyTorch >= 2.0, < 2.7 — installed *before* PocketLLM, and matching your CUDA toolkit
- CUDA toolkit 11.8+ for GPU execution
- `setuptools >= 68`, `wheel`
- NCCL for tensor parallelism with `TP > 1`

## Install

```bash
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel
pip install pocketllm --no-build-isolation
```

`--no-build-isolation` is what makes the build use the Torch you just installed,
and it also means pip fetches none of the prerequisites above — they must already
be present.

For a working copy:

```bash
git clone https://github.com/lvyufeng/PocketLLM.git
cd PocketLLM
pip install "torch>=2.0,<2.7" "setuptools>=68" wheel
pip install -e . --no-build-isolation
```

!!! note "The native C++/CUDA engine is no longer built here"
    Earlier versions built a `pocketllm_cpp` module and a standalone
    `pocketllm_engine` binary from a `cpp_engine/` tree. Both are retired: this
    repository builds no C/C++ extension and every runtime it ships is PyTorch.
    The engine lives in the archived
    [relic-engine](https://github.com/lvyufeng/relic-engine).

## Verify the install

```bash
python -m pytest tests/ -q
```

Modules that need a GPU or a real checkpoint skip themselves — a skip is not a
pass. See [`tests/README.md`](https://github.com/lvyufeng/RelicLLM/blob/master/tests/README.md)
for the suite's own rules, including why `tests/baseline_failures.txt` is a set of
node ids rather than a count.

## Run something

Python API:

```python
from pocketllm import LLM

llm = LLM(model="/path/to/checkpoint", backend="auto", tensor_parallel_size=4)
print(llm.generate("What is artificial intelligence?").text)
```

OpenAI-compatible server:

```bash
pocketllm serve --model /path/to/checkpoint --backend auto --tensor-parallel-size 4
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "pocketllm", "messages": [{"role": "user", "content": "Hello!"}]}'
```

DeepSeek-V4.1-Flash runs on the host-PyTorch `v41` backend instead, over a
checkpoint far larger than the aggregate VRAM. Startup pins a 457.8 GiB expert
bank in host memory and takes roughly four minutes before the server answers:

```bash
DEEPSEEK_V41_RESIDENT_EXPERTS=1 python -m pocketllm serve \
  --model /path/to/DeepSeek-V4.1-Flash \
  --backend v41 \
  --tensor-parallel-size 4 \
  --max-model-len 32768 \
  --port 8000 \
  --expert-pool-rows 148 \
  --prefill-chunk-tokens 4096 \
  --decode-graphs \
  --threads 22
```

Raise `--max-model-len` to `262144` for the longest context the runtime accepts.
The adapter takes one request lock, so requests are served one at a time.

MiMo-V2.6-Flash is the other four-rank heterogeneous path. Its routed experts live in a host bank
rather than in device memory, and the first start fills that bank — 149.81 GiB, about 12 minutes;
later starts attach to the existing segment:

```bash
python -m pocketllm serve \
  --backend mimo \
  --model /path/to/MiMo-V2.6-Flash \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --port 8000 \
  --prefill-chunk-tokens 2048 \
  --chunk-rows 16
```

It is one request at a time too, and for a firmer reason: every routed layer closes with an
all-reduce that every rank has to reach, so the ranks run the request as a symmetric group and rank
0 broadcasts the whole request before it starts generating. See the
[MiMo-V2.6-Flash model page](models/mimo-v2.6-flash.md) for the numbers and the memory that bank
takes.

**Ternary-Bonsai-2-27B** is not servable from this repository. It is a GGUF whose weights are 1.75
bits each, so a 27B model is 5.53 GiB and one card still has room for a 245,760-token KV cache — but
its header declares `general.architecture=qwen35`, and the `cpp` adapter that was its default backend
has been retired. No runtime here declares the canonical `qwen3_5`, so `--backend auto` refuses the
file by name. [Model page](models/ternary-bonsai-2-27b.md) keeps the retired engine's record.

The same is true of the **Qwen3.8-27B** checkpoints (FP8 / NVFP4 / BF16): they are `qwen3_5`
Safetensors and went with the same retired engine.

## Pick your path

| If you are running | Start here |
| --- | --- |
| DeepSeek-V4 | [DeepSeek-V4](models/deepseek-v4.md), or [GGUF Q2 on one GPU](models/deepseek-v4-gguf-q2-single-gpu.md) |
| DeepSeek-V4.1-Flash | [DeepSeek-V4.1-Flash](models/deepseek-v4.1-flash.md), then [serving it behind the OpenAI server](performance/deepseek_v4_1_flash_served_gate.md) |
| MiMo-V2.6-Flash | [MiMo-V2.6-Flash](models/mimo-v2.6-flash.md) |
| Xing4.0-29B-A4B (**one card**) | [Xing4.0-29B-A4B](models/xing4.0-29b-a4b.md) |
| Qwen3.8-Flash-Next | [Qwen3.8-Flash-Next](models/qwen3.8-flash-next.md) |
| MiniMax-M2.7 | [MiniMax-M2.7](models/minimax-m2.7.md) |
| GLM-5.2 | [GLM-5.2](models/glm-5.2.md) |
| Ternary-Bonsai-2-27B, Qwen3.8-27B | **not servable here** — see [Model support matrix](models/README.md) |
| A model not listed above | [Model support matrix](models/README.md) first |

Before quoting any number you measure or read here, read
[Benchmarking and reporting rules](guides/benchmarking.md).
