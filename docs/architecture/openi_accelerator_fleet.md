# The OpenI accelerator fleet

RelicLLM exists to run models on accelerators nobody else optimizes for, and most of those are
rented by the card-hour from OpenI (启智社区, `openi.pcl.ac.cn`). This page is the fleet as it stands
on **2026-10-14**, the two members the Ascend port starts with — **910A** and **V100** — and the
constraints that decide what a job may ask for.

Every `spec_id`, card count and price below was read live through
`GET /api/v1/ai_task/creation/required`. That is the same call behind the `openi-bridge` MCP
server's `openi_list_creation_options` tool and behind its CLI fallback `openi-run-on-card`, and it
is read-only until a task is created. The platform's own documentation names card types; it does not
publish the spec shapes an account can actually open, which is why the numbers here come from the
live call rather than from a docs page.

## What an account can open

This page is the **rentable** half of the fleet. The machines the project runs on directly are
eight-card boxes with their own budgets, and
[the eight-card fleet](eight_card_fleet_and_size_lines.md) covers those and the two size lines a
checkpoint has to clear before it is worth downloading.

Two facts about the response are easy to miss and both bound the work.

**`card_count` belongs to the spec, not to the card.** The same `ASCEND910` is offered as a one-,
two- and four-card spec, and there is no flag that asks for a fifth — a four-card spec is the
ceiling for this account, not a hint that more exist. So "how many cards can this model use" is two
questions: what the model needs, and what the largest spec for its card offers. This page answers
the second; [Hardware adaptation triage](../guides/hardware_triage.md) answers the first.

**`limit_count` is 1.** The account runs one task at a time, so a second job queues rather than
starting, and a port that wants to measure two cards in parallel cannot. Debug tasks also bill by
the card-hour against a points balance, in six-minute blocks.

## The two we start with

Both are 32 GiB cards and both are old, which is [the positioning](positioning.md): a model that has
to be split across cards on a modern part should fit one of these whole.

| Target | Compute source | `spec_id` | Cards | Card memory | Host | Price (点/卡时) |
|---|---|---:|---:|---:|---:|---:|
| 910A | NPU | 328 | 1 | 32 GiB | 24 cores / 96 GiB | 2 |
| 910A | NPU | 339 | 2 | 32 GiB | 48 cores / 128 GiB | 4 |
| 910A | NPU | 340 | 4 | 32 GiB | 96 cores / 256 GiB | 8 |
| V100 | GPU | 210 | 1 | 32 GiB | 8 cores / 50 GiB | 6 |
| V100 | GPU | 211 | 2 | 32 GiB | 16 cores / 100 GiB | 12 |

The card under `ASCEND910` is a 昇腾 910ProA, reported by `npu-smi` as HBM 32768 MB on an aarch64
host; `ASCEND-D910B` (spec 389, 64 GiB) and `ASCEND-910C` (spec 392) are the newer parts and are not
what this step targets. The GPU spec 210 is a Tesla V100S-PCIE-32GB at 32768 MiB on driver
570.124.06 and CUDA 12.8.

V100 also lists a spec 224 at 1 点/卡时 with **zero cards** — a placeholder the platform advertises
and no account can order. Read the card count, not the price, when a spec looks unusually cheap.

### Why these two, and what each really costs

**910A is a new family.** It is the first non-NVIDIA accelerator the tree is being taught, and
[The device plane](device_plane.md) already owns that design: the platform tables, the
`load_ops` provider seam, and the four things that must be replaced rather than renamed. The target
was fixed to the first generation — `910A`, and the `910B` without a trailing digit — because a port
that has never run anywhere should not start by spanning four chips. A 910A cube unit also rounds
both fp32 operands to fp16 before it multiplies, which the raw-block provider's tests pin to 1.02e-6
against a card's own answer; that it is fp16 and not bf16 is the whole reason the provider computes
in the activation's width rather than declaring fp32.

**V100 is a CUDA machine, not a family.** It runs the same torch plane the 2080 Ti boxes run, so it
joins as a machine: a different set of per-host facts, no new backend. What it is *not* is free.
`sm_70` is older than the `sm_75` every relic-core kernel is built for, and that build's fatbin
carries 7.5 and 8.9 SASS with no PTX, so a Volta card has no kernel image to execute at all — the
launch fails with "no kernel image is available" rather than falling back.

Adding `7.0` to `TORCH_CUDA_ARCH_LIST` is what it needed, and the cost is narrower than the arch
gap suggests. The build failed with 36 nvcc errors, every one in `cuda_kernel_impl.cu` and all but
two of them in the four FP4 grouped-MoE kernels, which multiply with the int8 tensor-core wmma
`m16n16k16` fragment — a shape Turing and newer define and Volta does not have at all. Those four
bodies now compile empty under sm_70 (`RELIC_INT8_WMMA`) while the host launch site keeps the
symbol and refuses the path by name with the device's capability in the message; the remaining two
errors were an nvcc 11.1 front-end complaint about taking the address of a `const` lambda capture.
The vendored llama.cpp MMA paths in `relic_core/csrc/llama_mmq/` already carry a non-Turing
fallback and compiled unchanged, and the GGUF raw-block path is arch-neutral by construction —
scalar fp32 dequantization and `__shfl_down_sync` reductions, with no tensor-core instruction
anywhere in it. The change is [relic-core#39](https://github.com/lvyufeng/relic-core/pull/39).

### CANN 9.0, and the machine behind the 910A target

The Ascend work is pinned to **CANN 9.0 or newer**. `torch_npu`'s operator coverage for the blocks
these models read moved across the 8.x → 9.x line, and a build that lands on 8.x fails in ways that
look like a model bug. The concrete 910A host for this port is `217`, an 8-card box running CANN
9.0.0, and the runtime is developed against it directly rather than through a rental queue.

Note that 217's eight cards and OpenI's four-card spec are the same silicon at two ceilings: the
spec is what a rental can ask for, and the box is what a port can be measured on once it is there.

### The dispatch those cards actually run

217's eight cards are 910A silicon — `ascend_ops_install.info` names `Ascend-cann-910-ops`, the PCI
id is `0xD801`, and `npu-smi` counts 30 cube units each — but `acl.get_soc_name()` and
`torch.npu.get_device_name(0)` both answer `Ascend910B`. `torch_npu` reads that string to choose
the default operator dispatch: `CheckJitDisableInner()` in
`torch_npu/csrc/framework/interface/EnvVariables.cpp` compares the soc version against `Ascend910B1`
and only disables the graph-compiling backend at or above it, so an enumeration that lands below
`910B1` runs the TBE path on a card that is not a 910B.

Both dispatches compute the same answers on the ops these models use — a 39-op probe on 217 passes
39/39 either way — so this is a cost question, and the cost is compilation. With an empty operator
cache the default writes 31 `te_*` entries for those 39 ops;
`torch_npu.npu.set_compile_mode(jit_compile=False)` (which is
`_npu_setOption({"jitCompile": "disable"})`) writes none. That first pass dominates the wall clock:
the Xing4.0 provider's test module goes from 473.60 s to 15.56 s on the same card and the same 71
tests. RelicLLM turns the aclnn dispatch on in `bind_device()` for any soc version starting with
`Ascend910`, and `RELICLLM_ASCEND_DISPATCH=jit` asks for the other one back. The switch that looks
like the same thing from the CANN side, `ACL_OP_INIT_MODE=2`, is refused on this build with
`Acl op is disabled!` and is not a substitute.

## The rest of the fleet

For completeness, the same live call verifies the other accelerator families OpenI offers, each
with its own SMI. They are not ports anyone has started; they are the map of what "support more
chips" would eventually mean.

| Compute source | Card | `spec_id` | Card memory | Verified with |
|---|---|---:|---:|---|
| NPU | 昇腾 910B2 | 389 | 64 GiB | `npu-smi` 25.2.1 |
| NPU | 昇腾 910C | 392 | 64 GiB/芯 (双芯) | `npu-smi` 26.0.rc3 |
| GCU | 燧原 Enflame T20 | 229 | 32 GiB | `efsmi` |
| MLU | 寒武纪 MLU290-M5 | 184 | 32 GiB | `cnmon` |
| DCU | 海光 K100_AI | 342 | — | `hy-smi` / `rocm-smi` |
| ILUVATAR-GPGPU | 天数智芯 MR-V100 | 388 | 32 GiB | `ixsmi` |
| BIREN-GPU | 壁仞 Biren106M | 347 | ~32 GiB | `brsmi` |

Each row is a card that boots and answers its vendor's own SMI inside a C2Net debug task; the
per-family layout, images and evidence live in the `openi-bridge` resource catalog that produced
this table. CPU is the one compute source with no orderable spec at all.

## Where a new family enters the tree

Adding one of the rows above is a code change before it is a machine, and the two halves fail
differently — a machine that is not joined produces quietly wrong numbers, a family that is not
joined produces a refusal. [Joining a machine, and joining a family](joining_a_machine_or_a_family.md)
is the contract for both, and states the checklist: the platform and accelerator tables in
`relicllm/runtime/device.py`, the visible-devices variable, the collective backend, and a provider
behind `load_ops`. The 910A work in this repository is the first walk through that checklist, and
what it establishes — a provider that answers the GGUF raw blocks in plain torch, registered only on
an Ascend host — is the template the next family copies.
