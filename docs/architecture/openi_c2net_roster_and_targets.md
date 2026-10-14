# OpenI C2Net: the roster, and the first two targets

RelicLLM exists for the older end of the accelerator market. OpenI (启智社区,
`openi.pcl.ac.cn`) is a public platform whose `C2Net` cluster is rentable by the hour, and its
roster is not hypothetical: it carries the cards this repository is written for, and it carries
older ones beside them. This page records that roster as measured, then states what the first two
targets — **Ascend 910A** and **Tesla V100** — each need before RelicLLM runs there.

Both targets have since been run, and neither took the path this page first expected. The V100
took the `sm_70` build below and runs the Xing4.0 raw blocks on the OpenI shape itself; the 910A
work left the rental queue for the account's own eight-card box, which is where the CANN floor and
the operator dispatch could be settled. [The OpenI accelerator fleet](openi_accelerator_fleet.md)
is the current reading of the roster and records what landed; this page keeps the measurements
behind it, including the ones that made the later plan change.

It is mostly a *machine* record in the sense [Joining a machine, and joining a family]
(joining_a_machine_or_a_family.md) uses, and one of the two targets is also a *family*. The device
plane that answers the family question already exists; [The device plane](device_plane.md) owns its
design and this page does not restate it.

## How these numbers were taken

Everything below is a reading, not a recollection, and the two ways of reading are both listed with
the task id that produced them so a later reader can re-run either.

| What | How |
|---|---|
| The roster (which cards, which `spec_id`, which price) | `GET /api/v1/ai_task/creation/required` per `compute_source`, through [openi-bridge](https://github.com/lvyufeng/openi-bridge) — the account-specific, live list |
| What is on a card | a `DEBUG` task on that `spec_id`, with a Python probe run through the task's own Jupyter kernel (`openi_run_on_card`) |

A probe is a few seconds inside a `RUNNING` task, so the tasks behind the measurements —
`903634`–`903686` on the account — cost nothing. Queue time is not billed, and a `RUNNING` window
under six minutes is not either.

The billing unit is worth measuring rather than quoting, because the roster's price column is easy
to misread. The account's own decrease records (`points_records`) give one formula across 70 charged
runs, 40 on a single card and 30 on four:
`charge = unit_price × floor(RUNNING_seconds / 360) / 10`. Two things follow from it. A run is
charged in whole six-minute blocks, so `00:09:25` costs what `00:06:45` does and anything under six
minutes is free. And `unit_price` is the price of the **whole shape**, not of a card: the `spec_id`
340 records charge `8 × units / 10` for four cards, which is 8 points an hour for the node rather
than 32.

## The roster, as measured on 2026-10-14

Prices are points per hour for the shape as listed — a ×2 row is twice the ×1 row and buys the same
per-card rate. `spec_id` is the value `--spec-id` takes to open that exact shape.

| `compute_source` | cards (`spec_id`) | price | note |
|---|---|---|---|
| NVIDIA `GPU` | A100 40G (`237`), V100S 32G (`210`), V100×2 (`211`) | 9 / 6 / 12 | CUDA |
| Ascend `NPU` | 910C (`392`), 910B 64G (`389`), 910A 32G (`328`), 910A×2 (`339`), 910A×4 (`340`) | 8 / 4 / 2 / 4 / 8 | CANN |
| Enflame `GCU` | T20 (`229`), T20×2 (`230`), S60 (`335`) | 1 / 2 / 2 | |
| Cambricon `MLU` | MLU290 (`184`), ×2 (`189`), ×4 (`190`) | 1 / 2 / 4 | |
| Hygon `DCU` | BW1000 64G (`354`), K100_AI 64G (`322`, `342`), Z100L 32G (`269`) | 5 / 3 / 1 | ROCm/DTK lineage |
| Iluvatar `ILUVATAR-GPGPU` | BI-V100 (`266`), ×2 (`265`), BI-V150 64G (`387`), MR-V100 (`388`), ×2 (`264`), ×4 (`315`) | 1 / 2 / 2 / 1 / 2 / 4 | |
| MetaX `METAX-GPGPU` | C500 64G (`333`) | 2 | a spec exists; no task has ever scheduled (see the bridge's README) |
| Biren `BIREN-GPU` | BIREN106M 12G (`347`) | 1 | |
| `CPU` | — | — | no spec is offered |

Two things follow from the table and matter for planning. Every non-NVIDIA family here is a
*family* port in the device plane's sense, not a machine: `GCU`, `MLU`, `DCU`, `ILUVATAR`,
`METAX` and `BIREN` each bring their own torch fork and their own device type, exactly as Ascend
does. And the DCU column is the one the plane currently gets *wrong*: Hygon's stack is ROCm-lineage,
so `torch.cuda.is_available()` answers `True` while the device is a DCU, and `--device auto` would
resolve to `cuda` and send the process at relic-core's CUDA extension. That is a trap rather than a
gap, and it is not on the first-two-targets path.

## Target 1: Ascend 910A

The 910A is the chip the device plane was written for; this section is what a real one reports.

### What a 910A reports (spec `328`)

Task `903640` on `spec_id` 328 landed on this board, and task `903643` ran the Python probe in it:

| Reading | Value |
|---|---|
| Board | `910ProA`, HBM 32768 MiB, `npu-smi` 24.1.0, `NB_ASCEND_VERSION=Ascend910` |
| Image | `cann8_3_rc1_mindspore2_7` (the platform's verified image for this spec) |
| Interpreter | `/home/ma-user/anaconda3/envs/PyTorch-2.1.0/bin/python`, CPython 3.10.6 |
| torch | 2.1.0, built with no CUDA (`torch.version.cuda is None`) |
| torch_npu | 2.1.0.post10 |
| `torch.npu.is_available()` | `True`, `torch.npu.device_count() == 1`, name `Ascend910ProA` |
| `torch.cuda.is_available()` | `False` |
| `torch._C._get_privateuse1_backend_name()` | `"npu"` |
| an `npu` matmul | returns |

The last three lines are the point. `_npu_available()` in `relicllm/runtime/device.py` is `import
torch_npu` followed by that exact privateuse-one check, so `probe_accelerator()` on this box returns
the Ascend `Accelerator` — the decision the plane was written to make, now observed on Ascend
hardware rather than injected through the probe's arguments. The 910A distinguishes itself from CUDA
by the plane's own rule, and `torch>=2.0` (this repository's floor) is satisfied by the image.

### Four cards is the ceiling per task

The NPU spec list holds three 910A shapes and no wider one, and `allowed_worker_num` is `[1]` for
every `job_type` (`DEBUG`, `TRAIN`, `INFERENCE`, `ONLINEINFERENCE`, `HPC`), so a task takes one node
with no multi-node shape to ask for:

| `spec_id` | shape | price |
|---|---|---|
| `328` | 910A ×1 | 2 |
| `339` | 910A ×2 | 4 |
| `340` | 910A ×4 | 8 |

**Four cards is the whole ceiling per task**, and all four are on that one node. Task `903694` on
`spec_id` 340 measured it, pinned to the same `cann8_3_rc1_mindspore2_7` image the ×1 probe used:
`torch.npu.device_count() == 4`, every one named `Ascend910ProA`, and `npu-smi info` showing NPU
4/5/6/7 each with 32768 MB HBM. There is no eight-card 910A shape to grow into; the ×4 row is the
widest thing the platform sells.

The four cards are scoped by `ASCEND_VISIBLE_DEVICES=4,5,6,7`, and that spelling matters: the
container maps the bought cards to NPU ids 4 through 7, so there is no `/dev/davinci0` in the
container's view and `ASCEND_RT_VISIBLE_DEVICES` is unset. `_VISIBLE_DEVICES` in
`relicllm/backends/runtime_engine.py` reads the `CUDA_` and `ASCEND_RT_` names, so on this box it
matches nothing and `visible_card_count` answers "the whole host" — which is the right four here for
the same reason it is right on the V100: a C2Net task's card set is fixed by the container runtime,
not by a variable a launcher sets.

One shape needs an image pinned onto it. `spec_id` 340's platform default,
`llama-torchfold-0914-910c`, comes back `CREATED_FAILED`; the 8.3.RC1 image runs. That is a gap in
the caller rather than the platform — `VERIFIED_IMAGES` in
[openi-bridge](https://github.com/lvyufeng/openi-bridge) carries a `328` entry and no `340` one — and
it is the kind of table row a later run adds, not a change in this tree.

### The gap is the operator seam, and it is already sized

What is missing is unchanged from [the device plane](device_plane.md#the-provider-gap-stated-plainly):
`register_provider("ascend", …)` is the hook, `_PROVIDERS` is empty, and the 46 entries of
`BINDINGS` have no Ascend answer. The five require-posture files are the reason a pure-`torch_npu`
plane needs real operator work before those models run; the four items that must be *replaced*
rather than renamed (graph capture, pinned memory, the custom all-reduce, NUMA/PCI affinity) are
listed there too. Nothing in this measurement changes that reading.

The seam has since been walked: [RelicLLM#194](https://github.com/lvyufeng/RelicLLM/pull/194) adds
`register_provider("ascend", …)` and an answer for the six bindings Xing4.0-29B-A4B calls — the four
`gguf_quant_*` ops and the grouped MoE, each of them "unpack the block, then a matmul" in plain
torch, plus the hyper-connection op that already answers `None`. It registers only on an Ascend host,
so the CUDA box keeps its own bindings.

### CANN 9.0 is a requirement the rental catalog does not meet

The image above is CANN **8.3.RC1**, and the requirement is CANN **9.0 or later**. Two live catalog
reads bound how far this account gets, and neither carries a 9.x entry:

The image above is CANN **8.3.RC1**, and the requirement is CANN **9.0 or later**. Two live catalog
reads bound how far this account gets, and neither carries a 9.x entry:

| Read | Endpoint | Result |
|---|---|---|
| The account's NPU images | `ai_task/creation/required` at `compute_source=NPU` | 29 images, the newest CANN **8.5.0** (`mindspore_2.7.2-cann_8.5.0-py_3.11`) |
| The 910A spec's own images | `ai_task/creation/image_by_spec` at `spec_id=328` | 6 images, the newest CANN **8.3.RC1** (`cann8_3_rc1_mindspore2_7`, the one the probe ran on) |

`spec_id` 339 and 340 — the 910A×2 and ×4 shapes — return those same six. The catalog is closed
rather than merely short: `can_use_all_images` is `False` at both the account and the per-spec level,
and `is_subscriber` is `False` for this account, so the create form offers only what those two reads
list. `image_url` is a separate field the create endpoint does honour — the NVIDIA GPU path in
[openi-bridge](https://github.com/lvyufeng/openi-bridge) depends on it — but there is no 9.x image in
this catalog for it to name, and whether a 9.x address supplied by hand would pull on a C2Net 910A is
untested. Meeting the CANN floor therefore needs a 9.x image registered as a platform image, which
is a request to the platform rather than a flag in this tree — item 1 of *What is open* below.

That is a bound on the *rental* path, and the 910A work did not stay on it. The port moved to the
account's own box — `217`, eight 910A cards, CANN **9.0.0**, innerversion
`V100R001C10SPC001B250` — where the floor is met by the host rather than by an image anyone has to
approve. Everything measured about the 910A from here on is from `217`; the spec `328` readings above
stay because they are what a rental reports. What that box then reports about itself — eight cards
that are 910A silicon but name themselves `Ascend910B`, and the operator dispatch that string
decides — is [The OpenI accelerator fleet](openi_accelerator_fleet.md)'s subject, and it is the one
`217` finding this page does not restate.

## Target 2: Tesla V100

The V100 is older than the 2080 Ti this tree was written on, so it is squarely the target rather
than a stretch.

### What a V100 reports (spec `210`)

Task `903656` on `spec_id` 210, probed with the platform's fallback image
`swr.cn-south-1.myhuaweicloud.com/openi/cuda111_python37_pytorch191:v1`:

| Reading | Value |
|---|---|
| Card | `Tesla V100S-PCIE-32GB`, 32768 MiB, driver 570.124.06 (CUDA 12.8 ceiling) |
| `torch.cuda.get_device_capability(0)` | `(7, 0)` — **`sm_70`, Volta** |
| torch in that image | 1.10.0, CPython 3.7.11, built against CUDA 11.1 (`nvcc` 11.1 present) |

The image is only tall enough to read the card: RelicLLM needs `torch>=2.0`, so a V100 run takes a
newer CUDA 12.x image. The capability reading is the fact to carry forward.

### Two cards is the ceiling per task

The GPU spec list holds two V100 shapes and no larger one, and `allowed_worker_num` is `[1]` for every
`job_type` (`DEBUG`, `TRAIN`, `INFERENCE`, `ONLINEINFERENCE`, `HPC`), so a task takes one node with no
multi-node shape to ask for:

| `spec_id` | shape | price |
|---|---|---|
| `210` | V100 ×1 | 6 |
| `211` | V100 ×2 | 12 |
| `224` | `acc_card_type` V100 with `acc_cards_num` 0 | 1 |

**Two cards is the whole ceiling per task**, and both are on that one node. Task `903686` on
`spec_id` 211 measured it: `torch.cuda.device_count() == 2`, both `Tesla V100S-PCIE-32GB`, `sm_70`,
80 SMs, 32494 MiB each.

The `224` row is a trap rather than a shape: it advertises `acc_card_type` V100 at one point an hour,
but `acc_cards_num` and `gpu_mem_gi_b` are both `0`. It is a CPU-only shape carrying the card-type
string, so a table that reads the type and not the count sells a V100 for one point an hour and hands
over none.

The pair arrives as two UUIDs in `NVIDIA_VISIBLE_DEVICES`, with `CUDA_VISIBLE_DEVICES` unset. That is
worth knowing before a launch: `_VISIBLE_DEVICES` in `relicllm/backends/runtime_engine.py` reads the
`CUDA_` and `ASCEND_RT_` spellings, so on this box it finds nothing and `visible_card_count` answers
"the whole host" — which happens to be the right pair, because the container is handed exactly the
cards the spec bought, so `--device-ids` names them `0,1`. A C2Net task scopes its cards through the
container runtime, not through the variable a launcher would set.

Two consequences for a port. `spec_id` 211 is the only multi-card V100 on the platform, so a
tensor-parallel reading there is a TP2 over PCIe with no larger shape to grow into. And that shape is
still `sm_70`, so the missing gencode below is not a single-card problem.

### The arch list has no `sm_70`, and that is the same failure one generation down

`relic-core/setup.py` pins `TORCH_CUDA_ARCH_LIST=7.5;8.9`, which expands to `sm_75` and `sm_89`
gencodes only. The dual-arch work measured what a missing gencode does: a wrong-arch `.so` *loads*
(`cuda_loader` only `exec_module`s; it never launches a kernel) and dies at the first launch with
`no kernel image is available for execution on the device`. A V100 is that case with the numbers
moved down — `7.5` covers Turing and up, and Volta is below it.

### The toolchain is a hard partition, not a preference

CUDA **13.0** removed offline compilation for Maxwell, Pascal and Volta: the toolkit's own release
notes say support for those architectures "have been removed in CUDA Toolkit 13.0 major version
release", and the `nvcc` target list starts at `sm_75` (Turing). The build box the dual-arch work
uses runs CUDA 13.0, so **`sm_70` cannot be added to that fatbin at all** — a V100 gencode has to
come from a CUDA 12.x `nvcc`. One build cannot cover `sm_70` and `sm_75`/`sm_89` together, which
means either a second toolchain for the V100 artifact or a runtime choice between two builds.

### Below `sm_75`, some kernels are compiled out

Volta is the generation before the `mma` instructions these kernels branch on. In relic-core:

- `common_shim.cuh:87` defines `TURING_MMA_AVAILABLE` only when `__CUDA_ARCH__ >= 750`, so on
  `sm_70` the MMQ path takes its non-tensor-core `#else` arm.
- `qwen4_exp_qsa.cu:27` guards its MMA path at `__CUDA_ARCH__ >= 750` the same way.
- The GGUF MMQ headers do carry Volta arms (`#if __CUDA_ARCH__ >= GGML_CUDA_CC_VOLTA`), and `dp4a`
  is available from `sm_61`, so the dp4a int8 paths are the ones most likely to survive on Volta —
  but that is a reading of the guards, not a measurement, and the V100 is where it gets measured.

What the plane already answers about a V100 is correct and needs no change: `probe_card_capability`
reads `(7, 0)`, and `supports_fp8_tensor_core` (`major >= 8`) is `False`, which is the right answer
for Volta. `KNOWN_CAPABILITIES` has no `700`, so the card reports `known=False` with the driver's
own name — a naming gap, not a behaviour one.

## What goes onto them first

Two cards is the V100 ceiling above, and it is the reading that picks the model. Four of the five
runtimes in this tree are written at four ranks and none of them is a sharding choice: MiMo's
attention is divided along the checkpoint's own four-way `qkv_proj` partition, V4.1's decode and host
expert bank are four processes, Qwen3.8-Flash-Next's embedding table and expert shard are a four-way
split, and DeepSeek-V4's GGUF path stages four ways. A 2×32 GiB task therefore cannot place any of
them, and the single-card shapes that fit are one: **Xing4.0-29B-A4B**.

It is also the model whose `sm_70` cost is zero kernels. Its quantized operators —
`gguf_quant_gemm_forward` at decode, `gguf_quant_gemm_prefill_forward` at prefill,
`gguf_quant_gemm_pair_forward` for the gate and up projections,
`gguf_quant_embedding_forward`, and `gguf_moe_prefill_grouped_forward` for the 64 resident experts —
are all IQ4_NL (GGML type 20), and every one of them lands in relic-core's generic float kernels:
the block is dequantized and dotted in fp32 by `gguf_quant_block_dot_256`, with no `__dp4a` and no
tensor-core instruction anywhere on the path. `relic_core/csrc/cuda_kernel_impl.cu` carries no
`__CUDA_ARCH__` guard at all, and `xing4_hyper_connection.cu` — the one op that is its own file, and
the one with a documented `None`-and-fall-back posture in `models/xing4_0/hyper_connection.py` —
carries none either. A V100 therefore needs the `sm_70` gencode and nothing else for this model:
the same CUDA 12.x build item 2 below already asks for. That held — the `sm_70` artifact runs the raw
blocks on `spec_id` 210 with no further kernel work, at the relative error item 2 below records.

The paths that are *not* in that position are worth naming, because they explain the order. The
vendored MMA code compiles to `NO_DEVICE_CODE` whenever `TURING_MMA_AVAILABLE` is undefined, and
`llama_mmq/common_shim.cuh:86` defines it only at `__CUDA_ARCH__ >= 750`. Two paths take it
unconditionally rather than behind a gate: PTQ1_0 (GGML type 143, Ternary-Bonsai-2-27B) for prefill,
and the same family's decode is a `__dp4a` trit GEMV, which Volta does have. So the 1.75-bit model
would be a new prefill kernel for a runtime that is retired here in any case, and Q4_K/Q5_K prefill
— MiniMax's — is behind its own `GGUF_Q4K_Q5K_MMA` gate and falls through to the float kernel when
the gate is off.

On the 910A the same choice is the smallest one, for a different reason. The operator seam needs five
answers for this model rather than forty-six — the four `gguf_quant_*` calls and the grouped MoE —
and each of them is "unpack the block, then a matmul", which is a torch-level answer rather than an
AscendC kernel. The hyper-connection kernel is the sixth and is already written to answer `None` and
run the eager arithmetic, so the first Ascend run does not need it. And because Xing4.0 is TP1,
there is no HCCL tensor-parallel path to get working before the model runs at all, which every other
runtime here needs first.

The natural second step is MiniMax-M2.7, and it is a 910A×4 target only. Its decode is already a
grouped IQ2_XXS `__dp4a` path and its prefill is the gated MMA one, so with the gate off it also
rides the float kernels and needs no new arithmetic; what it needs instead is four cards (about
16 GiB a card) and an HCCL reduction, which is exactly the shape `spec_id` 340 sells and `spec_id`
211 does not.

## What is open

1. **A CANN ≥ 9.0 image on OpenI.** Still open, and still a bound on renting a 910A here: the
   account's NPU image list stops at 8.5.0 and `can_use_all_images` is `False` for a non-subscriber
   account. It no longer blocks the port, because the port runs on `217`, which carries CANN 9.0.0
   itself. A 9.x image registered by the platform — or supplied through `image_url` and verified to
   pull — is what puts a 910A *rental* on the same footing.
2. **A CUDA 12.x build for `sm_70`.** Done, as
   [relic-core#39](https://github.com/lvyufeng/relic-core/pull/39). `TORCH_CUDA_ARCH_LIST=7.0`
   produced 36 nvcc errors, every one in `cuda_kernel_impl.cu`; all but two were the four FP4
   grouped-MoE kernels, which multiply with the int8 tensor-core `m16n16k16` wmma fragment that
   Volta does not have. Those bodies now compile empty under `RELIC_INT8_WMMA` and the host launch
   site refuses the path by name with the device's capability in the message. The extension built
   this way carries nine `sm_70` cubins and a `sm_70` PTX image and no `sm_75` one, so it is a
   *different artifact* from the `sm_75`/`sm_89` fatbin exactly as this page predicted — the build
   targets the card it was made on. Measured on `spec_id` 210: Xing4.0's IQ4_NL raw-block GEMM
   against its fp16 reference, max relative error 2.4e-3 at one row and 2.7e-3 at four.
3. **The sm_70 kernel audit.** Half answered. The vendored MMA paths still compile to
   `NO_DEVICE_CODE` below `sm_75`, so PTQ1_0's prefill arm and any other unconditional `mma` user is
   still absent on Volta; nothing here has exercised them, and the models this page proposes do not
   reach them. What is settled is the part that mattered: the GGUF raw-block path is arch-neutral by
   construction, and it is the whole path Xing4.0 takes.
4. **The Ascend operator provider.** Done, as
   [RelicLLM#194](https://github.com/lvyufeng/RelicLLM/pull/194) — the six bindings above, verified
   on `217` against a 910A. The remaining 40 entries of `BINDINGS` are per-model work rather than a
   gate, which is what `missing_bindings` was written to report.
5. **The DCU trap.** Not on this path, recorded here because the roster makes it reachable: a Hygon
   card answers `torch.cuda.is_available() == True`, so the host probe has to read `torch.version.hip`
   before it may call the platform `cuda`.
