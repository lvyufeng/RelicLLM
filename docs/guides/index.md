# Guides

Task-oriented documentation: how to measure PocketLLM, how to call it, and how to
release it. Start with [Benchmarking and reporting rules](benchmarking.md) before
quoting any performance number from this site — it defines the measurement
conventions the rest of the documentation assumes.

| Guide | What it covers |
| --- | --- |
| [Benchmarking and reporting rules](benchmarking.md) | The prefill/decode split, what every result record must contain, and the hardware and invocation details a comparable number needs. |
| [Serving latency metrics (vLLM convention)](latency_metrics.md) | TTFT, TPOT, ITL and E2EL as vLLM defines them, the `pocket_*` series that back each one, and how they relate to the prefill/decode convention. |
| [Hardware adaptation triage](hardware_triage.md) | The five outcomes a new checkpoint can earn (production, chat, demo, impossible, candidate), the four gates split by cost, the static fit calculation and its context/batch curve, why KV geometry is not layer count times head count, and the measured gaps the thresholds sit in. |
| [PocketLLM API and backend guide](pocketllm_api.md) | The single user-facing API over the PyTorch execution planes, backend selection, tensor parallelism, and what happened to the batch/scheduler surface. |
| [PyPI release guide](https://github.com/lvyufeng/PocketLLM/blob/master/docs/guides/pypi_release.md) | The single source of truth for releasing `pocketllm` to PyPI, including the Test PyPI dry run. |
| [Building and loading the CUDA extensions](https://github.com/lvyufeng/relic-core/blob/master/docs/guides/cuda_extension_builds.md) | Which `.so` the loader resolves and in what order, the two silent wrong-build failures and the three checks that separate them, and the sources in `src/csrc/` that `setup.py` does not compile. |
| [Ascend SoC generations](https://github.com/lvyufeng/relic-core/blob/master/docs/guides/ascend_soc_generations.md) | Why `910B` and `910B1`–`910B4` are different chips, how to read `Short_SoC_version`, and why they cannot share kernels. |

For install and build instructions, see [Getting started](../getting-started.md).
For per-model runtimes see [Model guides](../models/README.md).
