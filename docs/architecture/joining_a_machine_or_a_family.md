# Joining a machine, and joining a family

Two pieces of work arrive looking like one and share almost nothing. **Joining a machine** is
bringing a fourth box up to run the stack that exists: same vendor, same kernels, a different set of
per-host facts. **Joining a family** is teaching the tree a second kind of accelerator — Ascend 910A
today, an MLU or a DCU after it — which is a code change before it is a machine.

They are separated here because they fail differently. A machine that is not joined produces numbers
that are wrong for quiet reasons: a threshold copied from another box, a benchmark run under an
inherited `CUDA_VISIBLE_DEVICES`, a test that skipped and was read as a pass. A family that is not
joined produces a refusal, which is the good failure.

[The device plane](device_plane.md) already owns the second half's *design* — the platform tables,
the provider seam, the four things that must be replaced rather than renamed, and the line between an
Ascend decision and an Ascend tensor. This page does not restate it. What it adds is the **contract**:
the checklist each half has to satisfy, the places the existing abstractions do not generalise, and
the work that has no design at all yet.

### Which half you are on

| You have | You are joining | Start at |
|---|---|---|
| Another CUDA box to run the existing stack | a **machine** | [Part A](#part-a-joining-a-machine) |
| An Ascend 910A, an MLU, a DCU — any non-NVIDIA accelerator | a **family** | [Part B](#part-b-joining-an-accelerator-family) |
| A machine whose accelerator is a family the tree has not seen | **both**, family first | [Part B](#part-b-joining-an-accelerator-family), then [Part A](#part-a-joining-a-machine) |

A machine is operational and mostly unverified. A family is a code change with one honest
prerequisite: the provider seam exists, and nothing implements it.

---

## Part A: joining a machine

Nothing here is a code change. It is a set of facts that are true of one box and are read as though
they were true of all of them, plus the tooling that moves a job onto a box you are not sitting at.

### The per-host facts, and the rule that they are not universal

`CLAUDE.md` states the rule and lists the facts: checkpoint paths, the CUDA toolkit layout,
`/dev/shm` size, and the `POCKETLLM_*` variables. The rule is not decoration — the failure it
prevents has a shape. Read a number taken on a 2080 Ti box as a property of the software and it
becomes a threshold, then a comparison, then a decision about a machine that was never measured.

The reconciliation is the same every time: **a per-host number is a parameter, and the tool that uses
it says so.** [Hardware adaptation triage](../guides/hardware_triage.md) is the model — its card
count, card memory, host memory and every SLO threshold is a CLI flag, and the report prints the
basis string it was given. When you join a box, the work is not to edit the software; it is to
establish the box's own values and pass them.

### The environment contract

Two variable families, and they are not the same kind of thing.

**`POCKETLLM_*` keeps its spelling.** `CLAUDE.md` is explicit that these are a contract with launch
scripts and configs, not module paths, after the rename from `pocketllm`. The same holds for
`/dev/shm/pocketllm_*_experts*` bank names, which survive between runs and are therefore state, not
naming. Do not "fix" either when you touch a launch script.

**The visible-devices variable is platform-dependent, and this is already known.** The runtime names
both spellings in one place:

```python
# relicllm/backends/runtime_engine.py:82
_VISIBLE_DEVICES = ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")
```

So a launcher that hardcodes `CUDA_VISIBLE_DEVICES` works on a CUDA box and silently addresses the
wrong cards — or none — on an NPU box. That is the machine half of a family problem: you can join a
machine correctly and still have a script that only works on half the fleet.

### Moving a job onto a box you are not on

`scripts/remote_build.sh` runs one long job on another machine: detached, locked per job, with the log
and the final status written to files.

```bash
scripts/remote_build.sh run    <host> --dir /srv/src --tag build -- 'cmake --build build -j'
scripts/remote_build.sh status <host> --tag build
scripts/remote_build.sh logs   <host> --tag build --lines 100
scripts/remote_build.sh kill   <host> --tag build
scripts/remote_build.sh list   <host>
```

`<host>` is any ssh destination your own configuration already resolves. The script holds no
credential, asks for no password, and reads no ssh config — it shells out to `ssh <host> …` and
whatever that does is what happens. Run `ssh <host> true` once first; if that works, this works.

Three things go wrong with the obvious approaches and the shape exists to prevent all three: a
foreground job inside an `ssh` call dies when the connection drops; `nohup … &` inside a one-shot
`ssh` command is signalled when the channel closes about as often as not; and two jobs on one machine
contend for the same cores, which presents as "this machine is broken" rather than "you ran two". So
it detaches into a new session, takes a per-job lock (a second run refuses with exit `75`), and writes
both the log and the exit status to files.

One lesson that cost real time, and belongs here rather than in a shell history: killing a remote
build with `pkill -f "nvcc"` can match **the ssh session's own command line**, because that command
line contains the word. The session then kills itself and you see empty output. Use
`pkill -9 -x nvcc` (exact name). The same trap catches any `pkill -f` whose pattern you typed into the
command that is running it.

### Running the suite on a new box, and reading the skips

`tests/README.md` owns the policy and `CLAUDE.md` states the consequence: **a skip is not a pass**,
and an entry in `tests/baseline_failures.txt` is a claim that a test *runs here and fails*. A skip is
absent from that file by rule.

The sharp edge when you join a box is that the baseline check cannot see the difference. It counts
`failed` and `error` outcomes, and a suite that skipped itself for want of a card reports exactly the
same way as a suite that ran. On this tree, 41 modules under `tests/` gate on `torch.cuda.is_available`,
so a box without a usable CUDA device comes up green-with-skips and nothing in the tree
distinguishes *skipped for want of a CUDA card* from *skipped for want of any accelerator at all*.

That is not a blocker for joining a CUDA box — the predicate is right there. It is a blocker for
reading an NPU box's first run as evidence: the suite would skip on the same predicate a CUDA-less box
skips on, and the baseline would be silent about it. A shared "is there an accelerator of any kind"
predicate is what would make the two readable apart, and it does not exist yet.

### The triage probe command does not travel as printed

Triage prints the command that would produce the measurement it is asking for. That command is built
for the box it was written on:

```python
# relicllm/triage/report.py:122-123
f"CUDA_VISIBLE_DEVICES=$(seq -s, 0 {self.hardware.gpu_count - 1}) "
f"/home/lvyufeng/miniconda3/envs/deepseek/bin/python -m relicllm.cli serve "
```

Two things are frozen into it that are true of one machine: the interpreter path, and the
visible-devices variable name. On another box both are wrong, and the wrongness is quiet — the shell
would run, or not, and the failure would read as a bad command rather than as a non-portable one. When
you copy that command to a new box, change both by hand; the flag that would make the tool do it for
you does not exist (see [the open list](#what-is-open)).

### The checklist for a new box

1. `ssh <host> true` succeeds from wherever you will run jobs.
2. `relic-core` is installed from its checkout **before** this package:
   `pip install -e ../relic-core --no-build-isolation --no-deps`, then `pip install -e . --no-build-isolation --no-deps`.
   Without the first, pip tries to fetch `relic-core` from PyPI, where it is not published.
3. Establish the box's own values for: card count, card memory, host memory, `/dev/shm` size,
   checkpoint paths, toolkit layout. These are inputs to triage and to any benchmark, not constants.
4. Run the suite and read the skips, not just the failures: `python -m pytest tests/ -q -rs`, then
   `python scripts/check_test_baseline.py`.
5. Run a long job through `remote_build.sh`, so a dropped connection does not cost you the run.

---

## Part B: joining an accelerator family

This half is a code change, and the code it changes is small because most of the vocabulary already
exists. The work is in what the vocabulary does not cover.

### What is already platform-neutral

Three pieces shipped, and a port reuses them rather than replacing them:

- **The tables.** `PLATFORMS` and `ACCELERATORS` are string tuples; `_DEVICE_TYPE` and `_COLLECTIVE`
  are the two mappings the module docstring calls "the two facts a port gets wrong first"
  (`ascend → npu`, `ascend → hccl`). See [the device plane](device_plane.md) for the wording; the
  point here is that a new family adds *entries*, not machinery.
- **The probe.** `probe_accelerator(*, cuda_available=None, npu_available=None)` takes its answers as
  arguments, so the Ascend branch is reachable on a CUDA box — which is how it is tested without a
  910. A new family needs a third injectable answer (`<family>_available`), not a new probe.
- **The provider seam.** `relicllm/runtime/ops.py` is a registry, a lookup and a list. `load_ops()`
  returns the same object as `load_cuda_kernel()` on CUDA, by identity, so the change is a no-op
  there; on any other platform it consults `_PROVIDERS` and returns `None` when there is no provider,
  which the call sites already read as "fall back to eager arithmetic".

The runtime already speaks platform too: `_VISIBLE_DEVICES` above is one example, and
`relicllm/backends/factory.py` resolves `auto` through `resolve_platform` and refuses a platform the
runtime did not declare — that half is done and needs no change.

### What a runtime must declare

A runtime's entire hardware statement is one field: `RuntimeCapabilities.devices`, a tuple of
platform names, empty by default. Every other capability describes behaviour. The rule that matters is
exercised in one function:

```python
# relicllm/backends/factory.py:123-124
declared = runtime_capabilities(name).devices
if not declared or platform in declared:
    return
```

**An empty tuple is unconstrained, not nothing.** A runtime that has not stated a set has not excluded
one, so the refusal does not fire. This is checked for both selection paths — an explicit `--backend`
and the `auto` walk — and it is why the current census is four runtimes declaring `cuda` and `cpu`
(`torch`, `v41`) or `cuda` alone (`mimo`, `xing4`, `qwen4_exp`), and no runtime declaring `ascend`.

Adding a family therefore means **appending a platform name to the `devices` tuples of the runtimes
that can actually serve it** — and only after that name exists in the plane's tables. Declaring
`ascend` on a runtime whose kernels have no provider is a claim the tree will then enforce, not a
permission it will grant.

### What is CUDA-shaped and leaks: compute capability

This is the one place where an existing shape does not generalise, and it is worth stating plainly
because a wrong abstraction here reads as correct.

The tree has a card descriptor — `CardCapability` in `relicllm/runtime/device.py`, with
`major`/`minor`, a `cc` of `major*100 + minor*10`, and predicates `supports_fp8_tensor_core = major >= 8`
and `supports_fp4_tensor_core = major >= 10`. It answers *which card* where the plane answers *which
platform*, and it is the abstraction this section is about: it is NVIDIA-shaped, and this is where that
matters.

"Is compute capability general?" — **the question is, the encoding is not.** "Compute capability", the
`major.minor` convention and `major*100 + minor*10` are NVIDIA's. An Ascend 910A has no `major`; an
MLU or a DCU has no `sm_xx`. A card descriptor keyed on those numbers cannot describe a non-NVIDIA
accelerator, and the `KNOWN_CAPABILITIES` map is keyed by NVIDIA part numbers.

What saves it is the direction it fails in. A probe on a non-CUDA host returns the unknown descriptor,
which reads as the Turing floor — FP8 tensor core false, FP4 tensor core false. For an NPU that is
**correct about capability and wrong about identity**: "no FP8 tensor core in the NVIDIA sense" is
true, and "this is a 2080 Ti" is false. So the trap is not a crash. It is a report or a golden fixture
that says *Turing* for an NPU, and a reader downstream who believes it. A capability gate fails safe
here; a provenance record does not.

**One of those two is now closed.** A golden fixture records the card it was taken on
(`tests/golden_fixtures.py`), and the reader that produces the label asks the plane for the platform
first: a CUDA host names the descriptor, an Ascend host is written as hardware (`ascend 910B`), and
anything else records nothing. So a fixture cannot say *Turing* for an NPU — it says `""`, which is
the honest answer to a question nothing could read. The field is recorded and never consumed, which is
what keeps it a provenance record rather than a gate that would have to be right about capability too.
The descriptor itself is unchanged: splitting its portable *question* from its NVIDIA *encoding* is
the work below, and it waits on a second family that has a probe to read.

The generalisation to make before a port is to split the two: the **question** ("does this card have
an FP8 tensor core, how many of them") is portable and worth asking of every family; the **encoding**
is one family's, and a second family needs either a discriminator that switches what the numeric
fields mean or a small per-family record behind a common predicate. Appending
`if platform == "ascend": return something` inside `supports_fp8_tensor_core` would be treating an
NVIDIA integer as universal — which is the exact failure this section exists to name.

### What must be replaced, not renamed

Four things, and [the device plane](device_plane.md#what-is-not-in-this-step-and-why) already states
why each one is a replacement rather than a rename. What follows is where they are, so the size of the
job is legible:

| Item | Where | Why a rename is wrong |
|---|---|---|
| Graph capture | `relicllm/models/xing4_0/graphs.py`, `relicllm/models/deepseek_v4_1/graphs.py` | `torch.cuda.CUDAGraph`, `torch.cuda.Stream`, `torch.cuda.graph_pool_handle` — a vendor graph API with different capture semantics, not a namespace |
| Pinned host memory | 66 `pin_memory` occurrences under `relicllm/`, concentrated in `components/moe/cpu_backend.py` | Host pinning is a CUDA concept; an NPU has its own, and the staged expert-transfer paths need a branch |
| The custom all-reduce | `relicllm/models/deepseek_v4/runtime.py` (`_DecodeCustomAllreduce`, gated on `torch.cuda.can_device_access_peer`) | A hand-written IPC all-reduce exists because these machines lack NVLink-scale collectives; the NPU equivalent is HCCL's own |
| NUMA / PCI affinity | `relicllm/models/qwen4_exp/runtime.py` (`_cuda_numa_node`, reading `torch.cuda.get_device_properties`) | It builds a `/sys/bus/pci/devices/*/numa_node` path; an NPU exposes no such id |

The event and stream surface is the same class of problem at more sites — `torch.cuda.Event` and
`torch.cuda.Stream` appear across the MoE prefill backend and the device-expert runtimes — and the
device-plane page's "not in this step" table is the authority on scope.

### The declaration surface, in one place

A runtime's device declaration and the check that reads it are both small enough to state here:

```text
runtime declares:  RuntimeCapabilities.devices = ("cuda",)          # or () for unconstrained
auto resolves:     probe_accelerator() -> Accelerator -> resolve_platform("auto", accelerator)
refusal:           if declared and platform not in declared: UnsupportedFeatureError
```

So a new family is three additions and one registration: entries in the plane's two tables, a third
injectable answer on the probe, the platform name appended to the `devices` of the runtimes that can
serve it, and a provider registered through `register_provider(platform, loader)`. The provider's
module exports whichever members of `BINDINGS` it can supply; incomplete is allowed, and
`missing_bindings` reports the gap rather than refusing. What those bindings are, and why there are 46
of them with no provider, is [the device plane's](device_plane.md#the-provider-gap-stated-plainly)
subject.

### The build and toolchain half

This repository pins nothing. `setup.py` states it is pure Python — the native kernels moved to
relic-core — and there is no `ext_modules`, no `TORCH_CUDA_ARCH_LIST`, no `nvcc`. The arch pin is
entirely relic-core's: its `setup.py` sets `TORCH_CUDA_ARCH_LIST` before building the extension, and
that default is what a second *NVIDIA* arch changes. A port does not edit that default; it replaces
the extension build wholesale.

The import graph is safe as shipped: `relic_core.kernels.cuda_loader` imports only `pathlib`, `ctypes`
and `importlib` and returns `None` when it finds no `.so`, so importing this package off-CUDA does not
require CUDA. `triton` is an optional extra, not a base dependency, so a plain install pulls no vendor
compiler onto an NPU host — which is the shape you want and the one to keep.

One defect to record, because it is the kind of thing that misleads a port: `pyproject.toml` carries
`ninja`, `pybind11` and `cmake` as runtime dependencies of a package that builds no extension, and
`requirements.txt` does not list them. `CLAUDE.md` says those two files must agree. They are leftovers
from before the kernels moved — harmless at import, misleading on any host.

### The memory and accounting half

Triage's static fit does **not** need a port, and the reason is worth stating because the documentation
says the opposite.

The core fit touches no device. `HardwareProfile.gpu_count` and `gpu_memory_gib` (defaults 4 and 22.0
— 2080 Ti facts) are the entire "card", and `total_gpu_bytes`, `per_gpu_bytes` and `usable_per_card`
are arithmetic. `minimum_cards` walks integer counts. `kv.py` derives KV bytes from the checkpoint
config and a per-architecture sharding rule. No `torch`, no capability lookup, no device probe. So a
910A column is not "arithmetic on a platform this tree cannot execute" — the fit will compute.

What *is* CUDA-calibrated is the input, and a second family must restate it rather than inherit it:

- The defaults themselves (4 cards, 22 GiB) are one box's facts, surfaced as `--gpu-count` and
  `--gpu-memory-gib`. The field names say `gpu` and **there is no `--platform` or `--device` flag** to
  associate a fit's numbers with a family — the triage CLI's "box" group has no such option.
- The 15% reserve is a CUDA-driver-model figure; CANN reserves differently, so it is a per-family
  default, not a constant.
- `workspace_bytes` defaults to 0, the "no scratch" exemption that makes the gate free. Whether that
  is as safe on another family is not known.
- The precision ladder keys on `(architecture, role, dtype)` with **no platform field**, so the
  conversion factor for a format is "a property of the runtime" with nowhere to say *which* runtime.
  A family whose kernels consume a format in place where the CUDA path dequantizes it needs a new
  branch, not a new value.

Two concrete CUDA spellings live in triage itself: the `CUDA_VISIBLE_DEVICES` in the printed probe
command (above), and the SLO thresholds, which are absolute numbers measured on the CUDA roster.
`assess()` already takes `thresholds=`, so restating them is a value, not a branch.

**A stale statement to fix.** `docs/guides/hardware_triage.md`'s "deliberately not here" list says a
910A column "would be arithmetic on a platform this tree cannot execute". Its first clause — no
runtime declares `ascend` — is true. Its second is not: the plane declares ascend/npu/hccl in full,
`probe_accelerator` returns the Ascend answer, and the factory has the Ascend arm. The fit is
platform-neutral arithmetic; what is missing is a declaration and a provider, not an executor.

---

### What is a machine and what is a family

| Concern | Machine (a CUDA box) | Family (a non-NVIDIA accelerator) |
|---|---|---|
| Where the work is | operational | code |
| What changes | per-host values, passed as flags | the plane's tables, the probe, `devices` declarations, a provider |
| The honest failure | a number that is right elsewhere read as universal | a refusal, by name |
| Verified by | `pytest` + baseline on the box | nothing yet — the first thing it meets is the provider gap |
| Not a substitute for | the other | the other |

The last row is the one worth keeping. A machine can join the fleet and be green, and tell you nothing
about whether the family works; a family can be declared, reviewable and tested off-hardware, and still
have no box to run on. Neither replaces the other.

### What is open

No answer exists yet for any of these, and a page that implied otherwise would be worse than this
list:

1. **The rendezvous mechanism is NCCL-specific.** The supervisor allocates a file literally named
   `nccl_id` for the rank-0 broadcast and exports it as `POCKETLLM_NCCL_ID_PATH`. The variable name is
   frozen by the `POCKETLLM_*` rule, but the artifact is NCCL's. An HCCL path needs an equivalent
   id/root-info exchange beside it, and that is open.
2. **The NPU replacements are unnamed.** `torch.npu` has its own graph API and its own host-pinning
   story; neither is named or wired. This is the largest body of work in a port.
3. **The capability record is undecided, and now deliberately so.** A golden fixture bypasses it —
   the recorded card is free text, and only the CUDA branch of the label reads the descriptor — so the
   two halves have come apart: the *portable string schema* is settled, and whether `CardCapability`
   grows a family discriminator or is superseded by a per-family record is left until a second family
   has a probe to read. Deciding its shape now would be designing for a caller that does not exist;
   the label only ever needed a string.
4. **The triage platform key does not exist.** A `--platform` flag on the fit, and per-family restatements
   of the reserve and the workspace default, are unspecified.
5. **No accelerator-agnostic skip predicate.** The suite's 41 `torch.cuda.is_available` gates cannot
   express "a card of some kind", so an NPU run's skips are indistinguishable from a CUDA-less run's.
6. **The Ascend thresholds are unset.** The triage SLOs are measured on the CUDA roster; what replaces
   them for a 910A is a measurement nobody has taken here.

The one Ascend performance data that exists is in `relic-engine`'s archive, and
[the device plane](device_plane.md#the-numbers-ascend-work-starts-from) states it with its caveats:
prefill and decode figures that the archive's own roadmap calls "not a tuning problem", against a
`torch_npu` plane that starts further behind because it gives up hand-written AscendC kernels for
native operators. The first thing a port buys is not speed on a 910 — it is *loading a model on one at
all*.