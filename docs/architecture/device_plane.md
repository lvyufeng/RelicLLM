# The device plane

How RelicLLM decides which accelerator a process runs on, and what a second hardware family would
have to satisfy to run on it. The work this page records has one property worth stating before
anything else: **the Ascend half of it has never run on an Ascend device.** This host has four RTX
2080 Ti and no CANN, and every claim below about 910 hardware is a claim about a decision the code
makes, verified on CUDA, rather than about a tensor it produced. The last section says exactly which
of the two each statement is.

## Where Ascend support actually is

RelicLLM is one of three repositories, and the split is what makes this question sharp:

| Repository | Ascend code |
|---|---|
| [RelicLLM](https://github.com/lvyufeng/RelicLLM) | none, before this work. One declaration — `cpp` runtime `devices=("cuda","ascend")` — with nothing behind it |
| [relic-core](https://github.com/lvyufeng/relic-core) | none. `relic_core/kernels/` is CUDA and triton |
| [relic-engine](https://github.com/lvyufeng/relic-engine) | **all of it.** `cpp_engine/backends/ascend/` is 24 C/C++/AscendC source files, 10,012 lines: the kernels, an ACL device runtime and a hand-written IPC all-reduce that falls back to HCCL past a size ceiling |

`relic-engine/CLAUDE.md` states its own status in its first paragraph: *"A frozen archive of the
retired C++ inference engine … No repository depends on this one, no CI runs here, and nothing is
built, released or maintained."* So the Ascend path is not something to reconnect. It is something
to rebuild, and the decision taken was to rebuild it on **`torch_npu`** — the device-agnostic torch
plane — rather than on AscendC kernels or on the retired C++. The archive's kernels are worth
reading; they are not worth resurrecting, because a `torch_npu` plane is what makes these models
*load and run at all* on the card, and that capability does not currently exist in any live
repository.

The target is the first generation: `910A` and the `910B` without a trailing digit —
`Short_SoC_version=Ascend910`, 32 MiB of L2, no cube/vector parallelism. Later 910 parts and the
310/310P family are out of scope for this step, not because they are uninteresting but because a
port that has never run anywhere should not start by spanning four chips.

## The plane, and what it is made of

`relicllm/runtime/device.py`. Everything above it asks *this* module which platform it is on; nothing
above it names a vendor.

Three things are pure, and each is pure for a reason a test uses:

| Piece | What it is | Why it is shaped that way |
|---|---|---|
| `PLATFORMS`, `ACCELERATORS` | `("auto","cuda","ascend","cpu")` and `("cuda","ascend")` | `auto` is a question, never stored; the accelerators are the complement of `cpu` |
| `_DEVICE_TYPE`, `_COLLECTIVE` | `{cuda: cuda, ascend: npu, cpu: cpu}`, `{cuda: nccl, ascend: hccl, cpu: gloo}` | two string tables, because `ascend → npu` and `ascend → hccl` are the two facts a port gets wrong first, and neither needs a device object to state |
| `Accelerator` | `platform`, `torch_device_type`, `distributed_backend` | one value rather than three calls, so a run cannot choose the Ascend device type and the NCCL collective |

`probe_accelerator()` answers what this host can serve, and it takes both of its answers as
arguments. That is what makes the Ascend branch reachable here:

```python
probe_accelerator(cuda_available=False, npu_available=True)   # the Ascend answer, on a CUDA box
```

Both halves have to be named. Supplying only `npu_available=True` leaves `cuda` to the real probe,
which on the four-card box this is written on answers `True` and wins — the call would return CUDA
while reading as though it had asked for Ascend.

The real Ascend test is `import torch_npu` followed by `torch._C._get_privateuse1_backend_name() ==
"npu"`, which is the whole of what `torch_npu` does to this question: it renames torch's PrivateUse1
backend. Nothing imports it at module scope, because a CUDA host must not have it in its import
graph.

`canonical_device()` resolves a named device, and takes the current card index as a *callable* —
so an unindexed `"cuda"` is testable by passing `lambda: 2` rather than by owning a card. Two
policies sit on top of it, because three callers wanted three different answers and merging them
would have silently changed two:

| Function | Policy | Replaces |
|---|---|---|
| `canonical_device` | `None` in, `None` out; never refuses | `deepseek_v4_1/attention.py::canonical_device` and the copies around it |
| `require_device` | a card is required, and *this host* must have one | `minimax_m2/moe_runtime.py::_canonical_cuda_device`, both GGUF loaders' inline versions |
| `accelerator_device` | no card here is an answer, not a failure | `deepseek_v4/loader.py::_cuda_quant_device` |

And three verbs, which go through one shared guard rather than three copies of it — `bind_device`,
`synchronize`, `device_count`. The guard is where an unregistered device type is refused by name, so
the `ModuleNotFoundError: No module named 'torch_npu'` a bare `import` would raise is replaced by a
sentence naming the platform and the package.

Before this, `--device` was accepted, validated, forwarded to every rank, and read by nothing:
`auto` never resolved (the flag's own help text claimed otherwise), and `RuntimeCapabilities.devices`
was compared against nothing, so `--backend mimo --device cpu` was accepted by a runtime declaring
`("cuda",)`. Both are enforced in `factory.select_backend` now, and the behaviour change is carried
by [`migration/platform-is-checked.md`](../migration/platform-is-checked.md).

### The six entry points

The plane answers a question at 325 `torch.cuda` sites in the package. Six of them
*decide*: the place a process becomes bound to a card, and the place a collective backend is named.
Those six ask the plane now, plus the two worker payloads that make the same decision again in a
child process:

| Entry point | What it decides |
|---|---|
| `relicllm/runtime/generation.py::setup_dist` | the GGUF raw-block runtime's device and collective |
| `relicllm/cli/generate_v41.py::setup_distributed` | the same, for the V4.1 launcher |
| `relicllm/models/mimo_v2/ep.py::EpGroup.from_env` | the MiMo expert-parallel group |
| `relicllm/models/qwen4_exp/runtime.py::init_distributed` | the Qwen4-Exp TP context |
| `relicllm/models/deepseek_v4/generation.py::main` | the DeepSeek-V4 runtime's rank setup |
| `relicllm/models/deepseek_v4/serving.py::_init_runtime` | the served path's rank setup |
| `serving.py::_run_payload`, `_run_payload_stream` | the platform, carried to the worker process |
| `relicllm/backends/runtime_engine.py::_bind_device` | the serving bridge's one hard `torch.cuda` |

The platform travels *with* the worker payload rather than being re-probed there, because a worker
that probed independently would be answering a question about a differently configured host.

## Why the plane lives in `relicllm/runtime/`

The device question is asked by the model side — seventeen files under `relicllm/models/` still name
`torch.cuda` — and by `relicllm/backends/`. Those two halves were separate top-level packages when
this module was written, and `CLAUDE.md` named `relicllm/protocol/` as the home for anything shared
across that boundary, warning that a second `src/ → relicllm/` edge would be a cycle forming.
Sitting the module in `relicllm/protocol/` would have added seventeen reverse edges, which is
precisely what that rule existed to prevent.

The rule is gone: `src/` was merged into `relicllm/`, so there is one package and no boundary to
respect. The module stays where it was put. `relicllm/runtime/` is still the right shelf — it is the
layer that holds Torch-and-collectives facts rather than model facts, next to `generation.py` and
`ops.py` — and the model side reaching up into it is an ordinary intra-package import.

## The operator seam

Kernels are not in this repository. Every op arrives as a named binding on one extension module from
`relic-core`, and before this work each call site asked for it directly: `load_cuda_kernel()`, 56
times across 16 files. That is the wrong shape the moment a second answer exists,
because "give me the ops" has a different reply per platform while the *asking* should not.

`relicllm/runtime/ops.py` is that question, asked once. It is a registry, a lookup and a list — not a
wrapper, and it translates nothing. On CUDA, `load_ops()` returns **the same object**
`load_cuda_kernel()` returns, by identity, so the `hasattr(ops, ...)` probes the call sites already
use keep working and the change is a no-op there. After it, `load_cuda_kernel` appears in exactly
one file, `relicllm/runtime/ops.py` — which is where its name comes from now.

`BINDINGS` is measured rather than recalled: the intersection of the built extension's public names
with the names this tree references anywhere in the package, which was **46 of the extension's 57**.
The eleven it leaves out are a mix of second spellings of a kernel the runtime already has
(`int8_gemm_forward` beside `int8_gemm_pair_forward`, `moe_single_token_int8_forward_v2` beside
`moe_single_token_int8_forward`, `gguf_q2k_gemm_dp4a_forward`), six sparse-attention variants no
call site names, and the two `PTQ1_0` ternary kernels — whose GGUF type this tree *can*
read (`relicllm/loader/gguf/ptq1_0.py`) while the runtime that consumes it is the retired C++ engine, so
no Python call site here reaches them.

Two postures already existed in the tree, and the seam has to keep both:

- **probe and fall back.** `models/xing4_0/hyper_connection.py` says it best: *"`None` rather than
  raising: this is one kernel of many in a tree that also builds for Ascend, and a forward pass that
  falls back to the eager arithmetic is correct, just slower."*
- **require.** `components/gguf/quantized_ops.py`, `models/mimo_v2/device_experts.py`,
  `models/deepseek_v4_1/device_experts.py`, `models/glm_dsa/architecture.py` and
  `models/xing4_0/gguf_model.py` raise when the binding is absent, because there the op *is* the
  implementation and there is no slower correct answer.

That is why an unimplemented platform answers `None` rather than raising: a provider that does not
exist yet is exactly the case the first posture was written for.

### The provider gap, stated plainly

`register_provider("ascend", …)` is the hook, and nothing implements it. Making the seam exist makes
the 46 bindings *addressable*; it does not make them portable. Most of them have no torch fallback in
either repository — the five require-posture files above are the reason a pure-`torch_npu` Ascend
plane will need real operator work before those models run, and the honest reading of `BINDINGS` is
as a checklist with 46 entries and zero of them answered for Ascend.

## What is not in this step, and why

| Not here | Why not |
|---|---|
| The `torch_npu` provider itself | A later series. The hook exists; the operators do not, and inventing a torch fallback for `gguf_moe_single_token_iq2_q2k_forward` is a model-level decision rather than a port |
| CUDA graphs (`xing4_0/graphs.py`, `deepseek_v4_1/graphs.py`) | `torch.npu` has its own graph API. A rename would be silently wrong, so this needs a replacement |
| Pinned memory (66 occurrences) | `pin_memory` is a CUDA concept. An NPU has its own host-pinning story and the same call is not it |
| The custom all-reduce (`deepseek_v4/runtime.py`, `custom_allreduce_*` in `BINDINGS`) | A hand-written IPC all-reduce has to be *replaced*, not ported. It exists because 2080 Ti machines lack NVLink-scale collectives; the NPU equivalent is HCCL's own |
| NUMA/PCI affinity (`qwen4_exp/runtime.py`) | It reads `/sys/bus/pci/devices/*/numa_node` and pins CPU cores per card. There is no NPU equivalent of that file |
| The remaining `torch.cuda` sites | 325 occurrences across 26 files; 34 are `is_available()` gates. A mechanical series, one PR per subpackage — `models/` alone is the bulk of it. Measured with `git grep -o torch.cuda -- 'relicllm/**/*.py' \| wc -l`, which is the command to re-run |
| Real validation on a 910A | Not possible here: no CANN, no NPU, and `torch.device("npu")` does not even construct |

Three hazards upstream in `relic-core` will bite before any provider does, and they are **another
repository's PR**:

- `_auto_impl("fp8")` (`relic_core/kernels/ops.py:66`) only consults `get_device_capability()` when
  `torch.cuda.is_available()`. On an NPU host that question is skipped, so the capability rule that
  exists to keep sm_75 off the triton path cannot fire.
- The triton branches at `ops.py:632` (`soft_fp8_blockfp8_weight_dequant`) and `ops.py:968`
  (`soft_fp8_blockfp8_gemm`) are guarded by `_USE_TRITON` and **not** by `.is_cuda`. On a host where
  `import triton` succeeds and the tensors are on an NPU, `_auto_impl("fp8")` skips its capability
  rule and returns `"triton"`, so the auto path selects a triton kernel for a non-CUDA tensor. The
  only branch nearby that does test the device is `_soft_gemm` at `ops.py:509`
  (`impl == "triton" and a_ref.is_cuda and b_ref.is_cuda`). Two other branches have the same
  unguarded shape without the same exposure, and the *reason* differs in each case:
  `blockfp8_act_quant` at `ops.py:573` is kind `fp8_quant`, which `_auto_impl` maps to `"torch"`
  unconditionally, and `soft_bf16_weight_gemm_int8` at `ops.py:921` reads `_INT8_IMPL` (`ops.py:15`),
  whose default is `"torch"`. So the two names to fix are 632 and 968, and the check to add is the
  one their neighbours already make.
- `_USE_TRITON` is latched at import (`ops.py:31`) and cannot be flipped per device — it is set from
  whether `import triton` succeeded, and it is *also* what decides whether the triton branches above
  are taken at all. `DEEPSEEK_FORCE_TORCH_GEMM=1` (`ops.py:12`) suppresses the import and so disables
  all of them, which is the switch to reach for on an NPU host in the meantime.
  `DEEPSEEK_FORCE_FALLBACK` (`ops.py:11`) is defined and read nowhere — an abandoned start at
  exactly this feature, and one that keeps its `DEEPSEEK_` spelling for the provenance reason
  `CLAUDE.md` gives.

## What this can and cannot claim

A device *type* is registered with torch by whoever implements it, and an unregistered one cannot be
named at all:

```python
torch.device("npu")
# RuntimeError: Expected one of cpu, cuda, ipu, xpu, mkldnn, opengl, opencl, ideep, hip, ve,
# fpga, maia, xla, lazy, vulkan, mps, meta, hpu, mtia, privateuseone device type at start of
# device string: npu
```

`torch._C._get_privateuse1_backend_name()` answers `privateuseone` until `torch_npu` renames it. So
on this box an Ascend **decision** is testable and an Ascend **object** is not. Everything below the
line is the decision:

| Verified here | How |
|---|---|
| `ascend → npu`, `ascend → hccl` | the two tables, asserted exactly (`tests/test_device_platform.py`) |
| the host probe prefers CUDA when both are present | `probe_accelerator(cuda_available=True, npu_available=True)` |
| `auto` resolves to the host's platform, and anything else is left alone | injected probe |
| a runtime that does not declare the resolved platform is refused, by name | `select_backend` with an injected accelerator |
| an unregistered device type is refused with the platform named | `canonical_device("npu:1")` on this host |
| a strict caller on an Ascend host is told `torch_npu` is what is missing | `require_device("npu:0", platform="ascend", accelerator=probe_accelerator(cuda_available=False, npu_available=True))` |
| the operator seam is a no-op on CUDA | `load_ops("cuda") is load_cuda_kernel()`, by identity |
| the entry points still produce the same answer on CUDA | `POCKETLLM_GOLDEN=1 pytest tests/test_served_path_golden.py -q -k xing4` |

**Not verified, and not verifiable here:** that a tensor, a collective or a kernel produces the
right answer on a 910. That starts when a 910 host is available and `torch_npu` is installed, and
the first thing it will meet is the provider gap above.

## The numbers Ascend work starts from

The roadmap this project inherits is honest about the difficulty, and the only Ascend measurements
that exist are in `relic-engine`'s archive: Qwen3.8-27B FP16 on 4×910B first generation, TP4, CANN
9.0.0 — **prefill 1261 tok/s against a 2000 target, decode 9.2 tok/s against 100**. That document's
own conclusion is that decode "is not 1.6x away and is not a tuning problem": quantization is
required rather than optional, and the M=1 bandwidth floor for TP4 is 42 ms per token.

A `torch_npu` plane starts further behind than that, because it gives up hand-written AscendC
kernels for native operators. It is still the right first move. The alternative — reconnecting the
archive — is a rewrite of a tree its own README says nothing builds, and the capability that does
not exist today is not speed on a 910; it is *loading a model on one at all*.
