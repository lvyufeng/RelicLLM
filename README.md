# RelicLLM

A **multi-GPU inference runtime for older accelerators**, on PyTorch. It owns the model
implementations, the serving adapters and the schedulers; the kernels come from
[relic-core](https://github.com/lvyufeng/relic-core).

## What lives here

| Path | What it is |
|---|---|
| `src/models/` | the model implementations (`deepseek_v4`, `deepseek_v4_1`, `glm_dsa`, `mimo_v2`, `minimax_m2`, `qwen4_exp`, `xing4_0`) |
| `src/{loader,encoding,runtime,components,cli,server}/` | the loader, prefix/KV runtime, MoE components, CLI and server |
| `relicllm/` | the serving package: CLI, HTTP server, supervisor, and the runtime adapters |

## What does NOT live here

- **No C++ engine.** The runtime is PyTorch; native compute arrives as ops from `relic-core`.
  `relicllm.backends.cpp_backend` still exists only because the native engine has not been
  unhooked yet — see the "unhook the native engine" work item.
- **No kernels.** `relic_core.kernels` is imported, not vendored. Build relic-core first:

  ```bash
  pip install -e ../relic-core --no-build-isolation
  pip install -e . --no-build-isolation
  ```

- **Not the single-card / edge runtime** — that is PocketLLM.

## Provenance

Extracted with `git-filter-repo` from the PocketLLM monorepo (`src/` minus `src/csrc` and
`src/kernels`, plus `pocketllm/`), history preserved. The package was renamed `pocketllm` →
`relicllm`, and `src.kernels` imports now resolve to `relic_core.kernels`.
