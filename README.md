# RelicLLM

A **multi-GPU inference runtime for older accelerators**, on PyTorch. It owns the model
implementations, the serving adapters and the schedulers; the kernels come from
[relic-core](https://github.com/lvyufeng/relic-core).

## What lives here

One package, `relicllm/`. It is the serving shell and the model side together:

| Path | What it is |
|---|---|
| `relicllm/models/` | the model implementations (`deepseek_v4`, `deepseek_v4_1`, `glm_dsa`, `mimo_v2`, `minimax_m2`, `qwen4_exp`, `xing4_0`) |
| `relicllm/{loader,encoding,runtime,components}/` | the weight loaders, the prompt encoders, the device plane and the MoE components |
| `relicllm/{cli,server,api,backends,protocol}/` | the CLI, HTTP server, public API, runtime adapters and the request protocol |

`src/` used to be a second top-level package holding the model side. It was merged in: the two called
each other, and the name `src` in `site-packages` belonged to nobody.

## What does NOT live here

- **No C++ engine.** The runtime is PyTorch; the four backends (`v41`, `mimo`, `xing4`, `torch`) are
  all Python. The retired `cpp_engine` and the `cpp` backend that fronted it are gone.
- **No kernels.** `relic_core.kernels` is imported, not vendored. Build relic-core first:

  ```bash
  pip install -e ../relic-core --no-build-isolation --no-deps
  pip install -e . --no-build-isolation --no-deps
  ```

  `--no-deps` on both: `torch` is resolved from the environment, and pip without the flag may
  reinstall a different one and rebuild the kernels against the wrong ABI. The `relic-core`
  dependency line is a local checkout, not a PyPI release.

- **Not the single-card / edge runtime** — that is PocketLLM.

## Provenance

Extracted with `git-filter-repo` from the PocketLLM monorepo (`src/` minus `src/csrc` and
`src/kernels`, plus `pocketllm/`), history preserved. The package was renamed `pocketllm` →
`relicllm`, and `src.kernels` imports now resolve to `relic_core.kernels`. The `src/` tree came across
unchanged and was later merged into `relicllm/` with `git mv`, so history survives at the new paths.
