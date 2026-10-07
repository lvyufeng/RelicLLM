# The package layers

How `relicllm`'s subpackages stack, which direction an import is allowed to run, and the moves that
made the stack that way. The page exists because the answer was previously *unstated*: nothing
stopped a low layer from naming a high one, and six modules did.

The rule is one sentence: **an import may reach its own layer or any layer below it, never one
above.** `tests/test_package_boundaries.py::test_imports_only_reach_downward_through_the_package_stack`
enforces it by parsing every file in the package, and the exception list on that test is the only
place an upward edge may live.

## The stack

Lowest first. Each layer may name the ones under it and no others.

| Layer | Packages | What it is |
|---|---|---|
| **device** | `runtime/` (`device.py`, `ops.py`) | the accelerator plane and the kernel bindings. Names no vendor, no model, no container format |
| **primitives** | `loader/`, `encoding/` | reading checkpoints, and the tokenizer front ends. Container formats live here |
| **protocol** | `api/`, `protocol/` | the request/response contract, the sampling vocabulary, prompt rendering |
| **support** | `components/` | shared building blocks: GGUF quantized ops, MoE placement and backends, the spec registry |
| **models** | `models/<arch>/` | one architecture per directory: weights, kernels' call sites, the per-model generation loop |
| **adapters** | `backends/` | the five runtime adapters and the capability declaration |
| **hosts** | `cli/`, `server/`, `bench/`, `triage/`, and the top-level modules (`engine.py`, `supervisor.py`, …) | entry points: command line, HTTP, benchmarks, triage |

The top-level modules (`relicllm/engine.py`, `relicllm/choices.py`, `relicllm/work_bell.py`) are
*hosts*: they are entry points that wire the layers together, which is why they are allowed to
name any of them.

## Why the boundaries sit there

The two boundaries worth defending are the ones the table above puts a gap in.

**device below everything.** `runtime/device.py` answers "which accelerator is this process on" and
`runtime/ops.py` resolves a kernel binding. Nothing above them names a vendor, and they name nothing
above themselves — measured, both had zero outbound cross-package edges before this work and still
do. The Ascend port ([the device plane](device_plane.md)) depends on exactly that: a second
hardware family changes these two files and nothing else.

**models above support, below adapters.** A model directory may use the shared building blocks
(`components/`) and the loader; it may not name a backend, a server, or the CLI. And nothing below
the models layer may import a model — that is the rule that keeps `backends/` model-agnostic and
lets the request lifecycle be one lifecycle rather than five
([one scheduler, many models](one_scheduler_many_models.md)).

This is the same shape both upstream runtimes settled on, from opposite directions:

- **vLLM** puts `platforms/` under `model_executor/layers/` under `model_executor/models/`, then the
  scheduler in `v1/core/` and the entry points on top. Its models import **zero** modules from
  `v1/core`, `v1/engine` or `entrypoints`; the scheduler never names a model class. What crosses the
  gap both ways is one small declaration, `KVCacheSpec`.
- **SGLang** stacks `platforms/` + `hardware_backend/` under `layers/` under `mem_cache/` under
  `models/`, then `managers/` (scheduler + worker) and `entrypoints/`. `managers/` imports zero files
  from `models/`; the scheduler reaches a model only through `TpModelWorker.forward_batch_generation`.

Two ways to draw the same contract. In both, the layer under the models never imports one, and the
only upward traffic is a declaration the models *produce* and the layer above *consumes*.

## The six imports that violated it, and where they went

Before this work the test above failed with exactly six edges. Each is now a move, not an exception:

| Import | Why it pointed up | Fix |
|---|---|---|
| `runtime/generation.py → components.moe` | a GGUF generation driver sitting in the device plane | moved to `components/gguf/generation.py` |
| `runtime/pd_scheduler.py → components.moe.cpu_backend` | the DeepSeek-V4 PD scheduler sitting in the device plane | moved to `models/deepseek_v4/pd_scheduler.py` |
| `runtime/prefix_snapshot.py` (imported by `models/deepseek_v4`) | V4-only cache in the shared plane | moved to `models/deepseek_v4/prefix_snapshot.py` |
| `encoding/engram.py → models.deepseek_v4_1.config` | the V4.1 Engram front end: tokenizer-side, but the config layout that names it is V4.1's | moved to `models/deepseek_v4_1/engram.py` |
| `loader/mappings/{glm_dsa,minimax_m2}.py → components.moe.spec` | per-model tensor-name tables in the primitive layer | moved to `models/{glm_dsa,minimax_m2}/mappings.py` |
| `backends/torch_backend.py → server.engine` | the adapter imported the serving engine that is its own runtime | moved to `backends/serving_engine.py` |

The `loader ↔ components` cycle noted earlier is gone with the fourth and fifth rows: `loader/` no
longer imports `components/` at all. `loader/mappings/` as a directory no longer exists — the three
files it held were all per-model, and all three are now beside the model they name.

### What stayed an exception

`components/moe/registry.py` imports three model *specs* (`deepseek_v4`, `glm_dsa`, `minimax_m2`).
That is the one sanctioned upward edge, and it is the vLLM `KVCacheSpec` shape: the registry builds
an architecture string → spec table, the specs are the declaration, and the registry discovers them.
The `ALLOWED_UPWARD` entry on the test names the pair and the reason; the finer rule — only
`registry.py`, only those three modules — is the separate
`test_components_moe_only_imports_model_specs_for_registry_discovery`.

## What is not here

- **No `scheduler/` package yet.** `models/deepseek_v4/pd_scheduler.py` is where the scheduler lives
  today, inside the model that uses it. Lifting a model-agnostic core out of it is
  [one scheduler, many models](one_scheduler_many_models.md)'s follow-up, not this page's.
- **No layer for `bench/` and `triage/`.** Both are hosts (they drive a model from outside), so they
  sit at the top. If `triage/` ever grows a library surface another package calls, that is the point
  to give it a layer of its own — not before.