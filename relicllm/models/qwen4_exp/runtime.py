"""TP4 heterogeneous runtime for Qwen4-Exp.

Launch layout: one process per GPU, NCCL for the per-layer all-reduce.  Dense
weights are sharded (see `builder.build_heterogeneous`); routed experts are
loaded into each rank's disjoint host-RAM shard at startup, and the 95 GiB PLE
table stays host-resident through the existing mapping.  Only active experts are
staged to GPU during a step.

Device memory per rank (BF16, 4 ranks):
  dense sharded weights   ~2.6 GiB   (attn/linear-attn projections, HC mixers)
  lm_head shard            0.30 GiB
  KV + conv/recurrent      grows with context
  staged experts        top_k * 6.6 MiB per layer, freed each step

Host memory per rank:
  resident expert shard    56.25 GiB (128 of 512 experts x 48 layers)

Entry point: `python -m relicllm.models.qwen4_exp.runtime --model DIR --prompt ...`
under torchrun, or `run_tp4()` from another module.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from relicllm.models.qwen4_exp.builder import build_heterogeneous, layer_range_for
from relicllm.models.qwen4_exp.config import Qwen4ExpConfig
from relicllm.models.qwen4_exp.model import Qwen4ExpModel
from relicllm.models.qwen4_exp.quant import FP8Tensor, pack_gate_up_fp8
from relicllm.models.qwen4_exp.weights import MmapSafetensors, Qwen4ExpCheckpoint
from relicllm.runtime.device import bind_device, probe_accelerator


@dataclass
class TPContext:
    rank: int
    world_size: int
    device: torch.device
    initialized: bool
    #: Pipeline-parallel layout. ``pp_size == 1`` is every configuration this runtime had before
    #: pipeline support, so the default reproduces the old behaviour exactly.
    #:
    #: The two dimensions are deliberately kept apart, because ``world_size`` alone cannot express
    #: both: it is the *process group* size (TP x PP), which is what a collective must use, while
    #: every sharding rule in this model -- expert round-robin, head slicing, the vocab-sharded head,
    #: the per-layer all-reduce -- is a *TP* rule and must read ``tp_world_size``. Under a pure-PP
    #: launch ``world_size`` is 16 and ``tp_world_size`` is 1: the difference between a correct
    #: forward and 16 ranks each summing a sixteenth of the wrong thing.
    pp_size: int = 1
    pp_rank: int = 0
    #: Set by ``load_model`` once the config is read, so ``layer_range`` resolves for callers that
    #: hold the context before the model exists (the preload, the log line).
    num_layers: int = 0

    @property
    def tp_world_size(self) -> int:
        """Ranks the tensor-parallel sharding rules divide by.

        Under pipeline parallelism the stages own disjoint layers and nothing within a layer is
        split, so this is 1 even though the process group spans both machines.
        """
        return 1 if self.pp_size > 1 else self.world_size

    @property
    def tp_rank(self) -> int:
        return 0 if self.pp_size > 1 else self.rank

    @property
    def layer_range(self) -> tuple[int, int]:
        """The ``[start, end)`` slice of layers this pipeline stage owns.

        Contiguous and balanced, the same split ``deepseek_v4``'s layer-PP policy uses: the first
        ``n_layers % pp_size`` stages take one extra layer. Contiguity is what keeps each stage's
        expert preload a sequential read of the checkpoint rather than a scatter.

        Raises rather than guessing when ``num_layers`` has not been set yet. It is set by
        ``load_model`` from the checkpoint's config, so any caller reading this before then has no
        layer count to divide and would get a silently wrong answer -- ``(0, 0)`` under PP, which
        reads as "this stage owns no layers" and is exactly the kind of empty work that looks like a
        slow run rather than a bug. Failing loudly here cost one misread log line; failing silently
        would have cost a wasted bring-up.
        """
        if not self.num_layers:
            raise RuntimeError(
                "TPContext.layer_range was read before load_model set num_layers; the layer count "
                "comes from the checkpoint's config and is not known at init_distributed() time"
            )
        if self.pp_size <= 1:
            return (0, self.num_layers)
        return layer_range_for(self.num_layers, self.pp_size, self.pp_rank)


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


def _parse_cpu_list(spec: str) -> list[int]:
    cpus: list[int] = []
    for part in spec.strip().split(","):
        if not part:
            continue
        if "-" in part:
            begin, end = (int(value) for value in part.split("-", 1))
            cpus.extend(range(begin, end + 1))
        else:
            cpus.append(int(part))
    return cpus


def _pci_ids_from_smi() -> list[str]:
    """CUDA-index-ordered PCI ids, or [] when nvidia-smi cannot be asked.

    The id is normalised to sysfs' ``domain:bus:device.function`` form: nvidia-smi
    prints the domain as 8 hex digits and upper-case, sysfs uses 4 and lower.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if out.returncode != 0:
        return []
    ids = []
    for line in out.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        parts = value.lower().split(":")
        if len(parts) == 3:
            domain, bus, slot = parts
            ids.append(f"{int(domain, 16):04x}:{bus}:{slot}")
    return ids


def _pci_ids_from_sysfs() -> list[str]:
    """CUDA-index-ordered PCI ids read from sysfs, or [] if it cannot be read.

    Only the NVIDIA display/3D functions are considered, in the sorted order sysfs
    lists them -- which is the order the CUDA runtime enumerates them on a box with
    no CUDA_VISIBLE_DEVICES reordering.
    """
    try:
        entries = sorted(os.listdir("/sys/bus/pci/devices"))
    except OSError:
        return []
    ids = []
    for entry in entries:
        try:
            with open(f"/sys/bus/pci/devices/{entry}/vendor") as f:
                if f.read().strip() != "0x10de":
                    continue
            with open(f"/sys/bus/pci/devices/{entry}/class") as f:
                if not f.read().strip().startswith("0x03"):
                    continue
        except OSError:
            continue
        ids.append(entry)
    return ids


def _pci_id(device: torch.device) -> str | None:
    """This device's PCI id as ``domain:bus:device.function``.

    Read from the torch device properties where the build exposes them, which
    landed after torch 2.6; otherwise resolved to CUDA index order through
    nvidia-smi and then sysfs. Returns None when none of the three is available,
    which the caller treats as "no NUMA information" rather than an error.
    """
    properties = torch.cuda.get_device_properties(device)
    domain = getattr(properties, "pci_domain_id", None)
    bus = getattr(properties, "pci_bus_id", None)
    slot = getattr(properties, "pci_device_id", None)
    if domain is not None and bus is not None and slot is not None:
        return f"{domain:04x}:{bus:02x}:{slot:02x}.0"

    index = device.index if device.index is not None else torch.cuda.current_device()
    candidates = _pci_ids_from_smi() or _pci_ids_from_sysfs()
    if index >= len(candidates):
        return None
    return candidates[index]


def _cuda_numa_node(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    pci_id = _pci_id(device)
    if pci_id is None:
        return None
    try:
        with open(f"/sys/bus/pci/devices/{pci_id}/numa_node") as f:
            node = int(f.read())
    except (OSError, ValueError):
        return None
    return node if node >= 0 else None


def _physical_core_groups(cpus: set[int]) -> list[list[int]]:
    core_map: dict[tuple[int, int], list[int]] = {}
    for cpu in sorted(cpus):
        topology = f"/sys/devices/system/cpu/cpu{cpu}/topology"
        try:
            with open(os.path.join(topology, "physical_package_id")) as f:
                package_id = int(f.read())
            with open(os.path.join(topology, "core_id")) as f:
                core_id = int(f.read())
        except (OSError, ValueError):
            continue
        core_map.setdefault((package_id, core_id), []).append(cpu)
    return [sorted(siblings) for _, siblings in sorted(core_map.items())]


def _cpu_affinity_for_device(
    device: torch.device,
    local_rank: int,
    local_world_size: int,
) -> list[int] | None:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return None
    node = _cuda_numa_node(device)
    if node is None:
        return None
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist") as f:
            node_cpus = set(_parse_cpu_list(f.read()))
    except (OSError, ValueError):
        return None

    allowed_cpus = set(os.sched_getaffinity(0))
    core_groups = _physical_core_groups(node_cpus & allowed_cpus)
    if not core_groups:
        return None

    visible_ranks = min(local_world_size, torch.cuda.device_count())
    peer_ranks = [
        rank
        for rank in range(visible_ranks)
        if _cuda_numa_node(torch.device(f"cuda:{rank}")) == node
    ]
    if local_rank not in peer_ranks:
        return None

    peer_index = peer_ranks.index(local_rank)
    base = len(core_groups) // len(peer_ranks)
    extra = len(core_groups) % len(peer_ranks)
    begin = peer_index * base + min(peer_index, extra)
    count = base + (1 if peer_index < extra else 0)
    selected = core_groups[begin : begin + count]
    return sorted(cpu for siblings in selected for cpu in siblings)


def _bound_thread_pools(cpus: list[int], *, rank: int) -> None:
    """Cap the process's thread pools at the CPUs it was just bound to.

    Binding affinity and sizing a thread pool are two different facts, and this runtime used to
    establish only the first. `sched_setaffinity` says which CPUs the process *may* run on; it does
    not say how many threads to start, and `torch` sizes its intra-op pool from the cores the *host*
    reports -- 192 on the box this was found on. So each rank spawned ~168 threads and then had all
    of them contend for the 12 CPUs the affinity call had left it, eight ranks at once. The measured
    symptom is a load average above 600 during the expert preload, a machine that stops answering
    ssh, and a startup that takes an order of magnitude longer than the same work on a slower host.

    `v41` and `deepseek_v4` both cap their pools (`torch.set_num_threads`, `OMP_NUM_THREADS`); this
    runtime was the one that did not. The count is the affinity set's size, which is the allocation
    the rest of this function already computed.

    `OMP_NUM_THREADS` is set as well as the torch API because the preload's work is `torch.cat` and
    pinning, which reach OpenMP through the same pool: the env var is what a library that reads it
    at first use will honour, and the API call is what torch's own dispatcher honours now.
    `torch.set_num_interop_threads` is called defensively -- it may only be set before any parallel
    work has started, so a later call raises and that is not worth failing a load over.
    """
    threads = max(1, len(cpus))
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, str(threads))
    try:
        import torch

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(threads)
    except (ImportError, RuntimeError) as exc:  # pragma: no cover - defensive
        print(f"[qwen4exp] rank {rank}: thread pools left at their default ({exc})", flush=True)
        return
    print(f"[qwen4exp] rank {rank}: thread pools capped at {threads}", flush=True)


def _configure_cpu_affinity(ctx: TPContext) -> list[int] | None:
    if not _env_enabled("POCKETLLM_QWEN4_CPU_AFFINITY", "1") or ctx.device.type != "cuda":
        return None
    local_rank = ctx.device.index or 0
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(ctx.world_size)))
    cpus = _cpu_affinity_for_device(ctx.device, local_rank, local_world_size)
    if not cpus:
        if ctx.rank == 0:
            print("[qwen4exp] NUMA-local CPU affinity unavailable", flush=True)
        return None
    os.sched_setaffinity(0, cpus)
    cpu_spec = ",".join(str(cpu) for cpu in cpus)
    print(
        f"[qwen4exp] rank {ctx.rank}: GPU {local_rank} bound to CPUs "
        f"{cpu_spec} ({len(cpus)} logical CPUs)",
        flush=True,
    )
    _bound_thread_pools(cpus, rank=ctx.rank)
    return cpus


def init_distributed() -> TPContext:
    """Join the torchrun-provided process group, or fall back to single-rank.

    The pipeline layout is read from ``PIPELINE_PARALLEL_SIZE`` when it is set. It is not derived
    from ``WORLD_SIZE``, because the two mean different things and the group has to be joined before
    either can be resolved: ``WORLD_SIZE`` is the group, and a run may put all of it into pipeline
    stages *or* into tensor sharding. Leaving the variable unset is exactly the old behaviour --
    ``pp_size = 1`` -- so a torchrun without it is a pure-TP launch.
    """
    import torch.distributed as dist

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    accelerator = probe_accelerator()

    pp_size = int(os.environ.get("PIPELINE_PARALLEL_SIZE", "1") or "1")
    pp_rank = int(os.environ.get("PIPELINE_PARALLEL_RANK", str(rank)) or str(rank))
    if pp_size > world_size:
        raise ValueError(
            f"PIPELINE_PARALLEL_SIZE={pp_size} exceeds WORLD_SIZE={world_size}: the stages are a "
            "partition of the process group, so there cannot be more of them than there are ranks"
        )

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend=accelerator.distributed_backend, rank=rank, world_size=world_size
            )
        bind_device(local_rank, platform=accelerator.platform)
        return TPContext(
            rank,
            world_size,
            torch.device(accelerator.torch_device_type, local_rank),
            True,
            pp_size=pp_size,
            pp_rank=pp_rank,
        )

    # Single rank, and the only branch here that keeps a host fallback: with no group to join a
    # run on a host with no accelerator is a legitimate configuration rather than a misconfiguration,
    # which is why this one asks the accelerator rather than insisting on it.
    if not accelerator.is_accelerator:
        return TPContext(rank, world_size, torch.device("cpu"), False)
    bind_device(local_rank, platform=accelerator.platform)
    return TPContext(
        rank, world_size, torch.device(accelerator.torch_device_type, local_rank), False
    )


def make_all_reduce(ctx: TPContext):
    """Sum a row-parallel block output across ranks. None when nothing is sharded.

    Keyed on ``tp_world_size`` and not ``world_size``: a row-parallel block exists because a layer's
    columns were split across ranks, and under pipeline parallelism no layer is split -- the stages
    own *different* layers -- so there is nothing to sum. Reducing over the process group there would
    sum across stages that hold unrelated activations.
    """
    if ctx.tp_world_size == 1:
        return None
    import torch.distributed as dist

    def all_reduce(tensor: torch.Tensor) -> torch.Tensor:
        # The tensor is a fresh block output, so an in-place reduce is safe and
        # avoids an extra allocation per layer.
        tensor = tensor.contiguous()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    return all_reduce


def next_token_from_logits(last_logits: torch.Tensor | None, ctx: TPContext, *, batch_size: int = 1) -> torch.Tensor:
    """The greedy next token, agreed across every rank.

    ``last_logits`` are the last position's logits, shape ``[batch, vocab]``, or ``None`` on a
    pipeline stage that did not produce any (every stage but the last). Under ``pp_size == 1`` this
    is the plain gather-then-argmax the loop always did. Under pipeline parallelism the last stage
    takes the argmax and broadcasts it, because every stage needs the token to feed its own layers on
    the next step.

    The broadcast source is the last stage, not rank 0, and deliberately so: the token is a fact
    about the head and the head lives there. Rank 0's broadcast stays the cancel/EOS flag, so the two
    facts keep their two owners.
    """
    if ctx.pp_size <= 1:
        return gather_logits(last_logits, ctx).argmax(-1, keepdim=True)

    import torch.distributed as dist

    if ctx.pp_rank == ctx.pp_size - 1:
        token = last_logits.argmax(-1, keepdim=True).to(torch.long)
    else:
        token = torch.zeros((batch_size, 1), dtype=torch.long, device=ctx.device)
    dist.broadcast(token, src=ctx.pp_size - 1)
    return token


def gather_logits(logits: torch.Tensor, ctx: TPContext) -> torch.Tensor:
    """Concatenate the vocab-sharded lm_head outputs into full logits.

    Keyed on ``tp_world_size`` for the same reason as ``make_all_reduce``: the vocab is column-sharded
    only when the head itself is split. Under pipeline parallelism exactly one stage owns the
    (unsplit) head, so its logits are already the full vocabulary and gathering would concatenate 16
    copies of it.
    """
    if ctx.tp_world_size == 1:
        return logits
    import torch.distributed as dist

    parts = [torch.empty_like(logits) for _ in range(ctx.tp_world_size)]
    dist.all_gather(parts, logits.contiguous())
    return torch.cat(parts, dim=-1)


@dataclass
class ResidentBudget:
    """What a device-resident expert shard would cost, measured against what is free.

    A dataclass rather than a bool because the *report* is the point: the caller
    and the operator both need to know the requirement and the headroom, and a
    refusal that only says "no" leaves the next box to be sized by guesswork.
    """

    #: Bytes this rank's share of the routed experts occupies, all layers.
    experts_bytes: int
    #: Free device memory before the experts were uploaded -- the budget the run spends out of.
    free_bytes: int
    #: Free device memory after the upload, i.e. what the dense weights, KV and activations share.
    free_after_bytes: int
    #: Total bytes on the card.
    total_bytes: int


def resident_expert_bytes(
    config: Qwen4ExpConfig,
    *,
    pp_size: int = 1,
    pp_rank: int = 0,
    world_size: int = 1,
) -> int:
    """Bytes a rank holds if this rank's experts are resident for the whole run.

    Derived from the config and the checkpoint's own layout rather than from a
    table, so it is a *formula in cards and TP* and not a TP8 constant: one
    routed expert is `gate_up [2*inter, hidden] + down [hidden, inter]` plus its
    block scales, this rank owns every `world_size`-th expert of each layer it
    runs, and under pipeline parallelism it runs only its own layer slice.  At
    TP8 over 48 layers that is 14.06 GiB; at TP16 with 96 layers it is 7.03 GiB
    -- the V100 PP16 case, which fits on a 16 GiB card, so the mode is not a
    24 GiB feature.

    The dtype is read from the config's `quantization_config`: an FP8 checkpoint
    stores a one-byte code an element (the layout the reader's `expert_bytes`
    reports), a BF16 one stores two.  Both scale placements are counted -- the
    128x128 tiles of a block-scaled FP8 checkpoint and a single per-expert
    scalar -- because the difference is a rounding error next to the codes and a
    formula that quietly drops one is worse than one that over-counts by 0.1%.
    """
    text = config.text_config
    start, end = layer_range_for(text.num_hidden_layers, pp_size, pp_rank)
    layers = max(0, end - start)
    per_rank_experts = (int(text.num_experts) + int(world_size) - 1) // int(world_size) if world_size > 0 else 0
    if layers == 0 or per_rank_experts == 0:
        return 0

    inter = int(text.moe_intermediate_size)
    hidden = int(text.hidden_size)
    elements = 2 * inter * hidden + hidden * inter
    spec = config.fp8
    if spec is None:
        per_expert_raw = elements * 2
    else:
        block = spec.block_size
        tiles = ((2 * inter) // block[0]) * (hidden // block[1]) + (hidden // block[0]) * (inter // block[1])
        per_expert_raw = elements + tiles * 2
    return per_expert_raw * per_rank_experts * layers


def require_resident_budget(
    config: Qwen4ExpConfig,
    ctx: TPContext,
    *,
    reserve_bytes: int,
    free_at_start: int | None = None,
) -> ResidentBudget:
    """Check that the whole run fits, in one place, and refuse if it does not.

    The requirement is the *complete* per-rank footprint: this rank's experts
    (`experts_bytes`, the formula in :func:`resident_expert_bytes`, which is the
    only part that moves with TP and the card count) plus `reserve_bytes`, the
    dense weights, the KV cache and the activations of a run of the intended
    length.  `reserve_bytes` is a caller's fact and not one this function can
    derive -- how long the sequences are is decided by the caller, and the dense
    weight is a constant of the checkpoint that the caller has already measured.

    The measured side is `free_at_start` when the caller has it: free device
    memory *before* the experts were uploaded.  That is the budget the run is
    spending out of.  Read after the fill it would already have had
    `experts_bytes` subtracted, and comparing the requirement against that number
    would double-count it -- every residency would come out impossible.  Taken
    here as an argument rather than read from the device for exactly that reason.

    Refusing is the point of the mode: a residency that does not fit must fail
    with a named error saying what was needed and what was found, rather than OOM
    several layers into a forward -- on this box that is eight ranks deep and
    hours into a checkpoint load.
    """
    if ctx.device.type != "cuda":
        return ResidentBudget(0, 0, 0, 0)
    free_now, total_bytes = (int(v) for v in torch.cuda.mem_get_info(ctx.device))
    experts_bytes = resident_expert_bytes(
        config,
        pp_size=ctx.pp_size,
        pp_rank=ctx.pp_rank,
        world_size=ctx.tp_world_size,
    )
    measured = int(free_at_start) if free_at_start is not None else free_now
    if experts_bytes == 0:
        return ResidentBudget(0, measured, free_now, int(total_bytes))
    if measured < experts_bytes + reserve_bytes:
        raise RuntimeError(
            f"rank {ctx.rank}: {expert_resident_requirement(config, ctx)} -- the experts plus "
            f"{reserve_bytes / (1 << 30):.2f} GiB held back for the dense weights, KV and "
            f"activations -- but only {measured / (1 << 30):.2f} GiB of "
            f"{total_bytes / (1 << 30):.2f} GiB was free on {ctx.device} before the upload. "
            f"Drop --expert-resident to stage experts from host RAM instead (that is the "
            f"2080 Ti path and has no device-side expert cost), or use a box with more room "
            f"a card."
        )
    return ResidentBudget(int(experts_bytes), measured, free_now, int(total_bytes))


def expert_resident_requirement(config: Qwen4ExpConfig, ctx: TPContext) -> str:
    """The resident requirement as a sentence, in the units a plan is made in.

    Reported as a formula in cards and TP rather than as a TP8 constant, because
    the question the caller has is "does this fit *here*", and the answer moves
    with both: one expert is 4.6881 MiB, a rank holds `num_experts / TP` of them a
    layer over its share of the layers, so TP8 over 48 layers is 14.06 GiB a rank
    and TP16 over 96 layers is 7.03 GiB -- which is why the mode is not a
    24 GiB-only feature and belongs on a V100 PP16 box as much as on a 3090.
    """
    text = config.text_config
    start, end = ctx.layer_range
    layers = max(0, end - start)
    per_rank = max(1, int(text.num_experts) // max(1, ctx.tp_world_size))
    gib = resident_expert_bytes(
        config, pp_size=ctx.pp_size, pp_rank=ctx.pp_rank, world_size=ctx.tp_world_size
    ) / (1 << 30)
    return (
        f"device-resident experts need {gib:.2f} GiB "
        f"({int(text.num_experts)} experts / {ctx.tp_world_size} way tensor parallel "
        f"= {per_rank} a rank x {layers} layers)"
    )


def load_model(
    model_dir: str,
    ctx: TPContext,
    *,
    dtype: torch.dtype = torch.bfloat16,
    expert_cache_capacity: int = 0,
    host_expert_memory: bool = True,
    pin_experts: bool = True,
    expert_resident: bool = False,
    expert_reserve_bytes: int = 3 << 30,
    expert_decode_device_gib: float | None = None,
    profiler=None,
    expert_deep_cache: float | None = None,
    expert_fused_fp8: bool = False,
) -> tuple[Qwen4ExpModel, Qwen4ExpConfig]:
    _configure_cpu_affinity(ctx)
    config = Qwen4ExpConfig.from_pretrained(model_dir)
    ctx.num_layers = config.text_config.num_hidden_layers
    checkpoint = Qwen4ExpCheckpoint(model_dir, store=MmapSafetensors(model_dir))
    # The resident mode *is* a placement, so it implies the host preload machinery
    # and replaces its destination rather than running beside it: there is no
    # separate "resident shard" and "host shard", only the one shard this rank
    # owns, and where it was put.
    if expert_resident:
        host_expert_memory = True
    deep_cache = expert_deep_cache is not None
    if deep_cache:
        # The deep cache replaces every placement decision above, so it must not also pay for one:
        # no host preload (the file mapping is what it reads), no resident shard, no LRU.
        host_expert_memory = False
    if host_expert_memory:
        # Pull this rank's routed experts off the disk-backed mapping and into
        # host RAM before any step runs, so steady-state staging never seeks.
        import torch.distributed as dist

        distributed = ctx.world_size > 1 and dist.is_initialized()
        # Where this rank's experts go.  The host bank is the 2080 Ti path and is
        # what the staging machinery exists for; the device bank is the same shard
        # with a different destination, and `build_heterogeneous` picks the backend
        # from this placement rather than from a flag of its own.
        shard_device = ctx.device if expert_resident else None
        # Free memory *before* any of this rank's experts land.  The check needs the budget
        # the run is spending out of, not what is left after it has spent it -- measured
        # afterwards, the requirement would be compared against a number that had already
        # had the requirement subtracted from it, and every residency would look impossible.
        free_at_start = (
            int(torch.cuda.mem_get_info(ctx.device)[0])
            if shard_device is not None
            else None
        )
        started = time.perf_counter()
        # Under pipeline parallelism a rank owns its stage's layers and nothing within them is
        # split, so it preloads every expert of those layers: `world_size=1` here is the expert
        # placement, which is a TP rule and therefore 1 when the stages own disjoint layers.
        # `layer_range` is what keeps that from being the whole 112 GiB per rank.
        shard = checkpoint.preload_experts(
            config.text_config.num_hidden_layers,
            num_experts=config.text_config.num_experts,
            rank=ctx.tp_rank,
            world_size=ctx.tp_world_size,
            layer_range=ctx.layer_range if ctx.pp_size > 1 else None,
            pin=pin_experts,
            device=shard_device,
            packed_gate_up=expert_fused_fp8,
            progress=ctx.rank == 0 and os.environ.get("QWEN4EXP_PRELOAD_PROGRESS") == "1",
            barrier=dist.barrier if distributed else None,
        )
        elapsed = time.perf_counter() - started
        start, end = ctx.layer_range
        where = "device" if shard_device is not None else ("pinned" if shard.pinned else "pageable")
        print(
            f"[qwen4exp] rank {ctx.rank}: {shard.num_local_experts} experts/layer "
            f"x {shard.stats()['layers']} layers resident, "
            f"{shard.resident_bytes / (1 << 30):.2f} GiB "
            f"({where}) in {elapsed:.1f}s"
            + (f" (pipeline stage {ctx.pp_rank}/{ctx.pp_size}, layers {start}:{end})"
               if ctx.pp_size > 1 else ""),
            flush=True,
        )
        # The fill has happened by now, so this is the report and the *second* half of the
        # check: `free_at_start` is what the run had to spend, and what is left here is what
        # the dense weights and the KV will have to share.  A residency that does not fit
        # says so at this point rather than OOM eight ranks deep in a forward.
        if expert_resident:
            budget = require_resident_budget(
                config,
                ctx,
                reserve_bytes=int(expert_reserve_bytes),
                free_at_start=free_at_start,
            )
            print(
                f"[qwen4exp] rank {ctx.rank}: resident experts {budget.experts_bytes / (1 << 30):.2f} GiB, "
                f"{budget.free_after_bytes / (1 << 30):.2f} GiB left of "
                f"{budget.total_bytes / (1 << 30):.2f} GiB after the fill, "
                f"{expert_reserve_bytes / (1 << 30):.1f} GiB held back for activations",
                flush=True,
            )
        if distributed:
            dist.barrier()
    model = build_heterogeneous(
        config.text_config,
        checkpoint,
        device=ctx.device,
        dtype=dtype,
        rank=ctx.tp_rank,
        world_size=ctx.tp_world_size,
        all_reduce=make_all_reduce(ctx),
        expert_cache_capacity=expert_cache_capacity,
        pp_size=ctx.pp_size,
        pp_rank=ctx.pp_rank,
        profiler=profiler,
        expert_deep_cache=expert_deep_cache,
        expert_fused_fp8=expert_fused_fp8,
    )
    if deep_cache:
        _decode_experts(
            model, checkpoint, ctx, expert_deep_cache, expert_decode_device_gib, expert_reserve_bytes
        )
    return model, config


def _expert_layer_source(checkpoint, ctx: TPContext, num_experts: int, layer_idx: int):
    """This rank's stored rows of one layer, laid out as `HostExpertShard.layer_tensors` lays them.

    The shard copies those rows into one buffer per projection, scale grids and all; here they are
    gathered off the mapping, so a layer costs a read of this rank's rows and no more, and the
    layout (local row first, gate|up packed) is the one the cache decodes or stages.

    The gather has an output buffer from the start and copies each row into it as it goes, which is
    also the safe way to read these rows: a view the mapping hands back shares memory with a
    million other tensors, and a reader that collects views and copies them afterwards can have
    read them into bytes another read replaced.
    """
    block = checkpoint.fp8.block_size
    ids = list(range(ctx.tp_rank, num_experts, ctx.tp_world_size))
    if not ids:
        raise ValueError(f"rank {ctx.tp_rank} owns no experts out of {num_experts}")

    def owned(proj: str) -> FP8Tensor:
        code0, grid0 = checkpoint.fp8_expert_projection(layer_idx, ids[0], proj)
        codes = torch.empty((len(ids), *code0.shape), dtype=code0.dtype)
        grids = torch.empty((len(ids), *grid0.shape), dtype=grid0.dtype)
        codes[0].copy_(code0)
        grids[0].copy_(grid0)
        for local, expert in enumerate(ids[1:], start=1):
            code, grid = checkpoint.fp8_expert_projection(layer_idx, expert, proj)
            codes[local].copy_(code)
            grids[local].copy_(grid)
        return FP8Tensor(codes, grids, block)

    gate, up = owned("gate"), owned("up")
    return pack_gate_up_fp8(gate, up), owned("down")


def _decode_experts(
    model, checkpoint, ctx: TPContext, scale: float, budget_gib: float | None, reserve_bytes: int
) -> None:
    """Decode this rank's expert layers into its deep cache, as far as the card holds.

    Per layer: read this rank's stored rows off the mapping, decode them, drop the mapped pages.
    Whole-layer granularity is what keeps the peak at one layer's codes and scales on the host and
    one decoded layer on the card beyond the layers already kept.

    The card is a real constraint here and not an assertion: a rank owns 48 x 64 experts, which
    decoded is about 39 GiB, and an 8-way width does not put that on a 24 GiB card.  So the decode
    stops when the next layer no longer fits and the remaining layers stay with the host stove --
    which is also the honest deployment: a card with no room runs the cache empty.
    """
    from relicllm.models.qwen4_exp.moe import DeepFallbackMoE

    # Layers share one backend; reach it through the first layer this rank holds, which is not
    # necessarily layer 0 under pipeline parallelism.
    layer_moe = model.local_layers[0].moe
    inner = getattr(layer_moe, "inner", layer_moe)
    if not isinstance(inner, DeepFallbackMoE):
        raise TypeError(f"deep cache asked for but the backend is {type(inner).__name__}")
    cache = inner.cache
    start, end = ctx.layer_range
    free_before = int(torch.cuda.mem_get_info(ctx.device)[0])
    # A budget is a share of what the card held after the dense weights, and the allocator's cached
    # blocks are part of that -- the decode reuses them.  The default leaves the run's own reserve.
    budget = (
        free_before - int(reserve_bytes) if budget_gib is None else int(budget_gib * (1 << 30))
    )
    started = time.perf_counter()
    decoded: list[int] = []
    for layer_idx in range(start, end):
        stored = _expert_layer_source(checkpoint, ctx, cache.num_experts, layer_idx)
        # What the decoded layer will take: two bf16 values per stored code.
        layer_bytes = 4 * int(stored[0].code.numel() + stored[1].code.numel())
        if cache.decoded_bytes + layer_bytes > max(0, budget):
            break
        cache.decode_layer(layer_idx, stored)
        del stored
        decoded.append(layer_idx)
        if ctx.rank == 0 and len(decoded) % 8 == 0:
            print(
                f"[qwen4exp] rank 0: deep cache through layer {layer_idx} "
                f"({cache.decoded_bytes / (1 << 30):.2f} GiB)",
                flush=True,
            )
    free_after = int(torch.cuda.mem_get_info(ctx.device)[0])
    print(
        f"[qwen4exp] rank {ctx.rank}: deep cache {len(decoded)}/{end - start} of this rank's layers "
        f"{'[%d..%d]' % (decoded[0], decoded[-1]) if decoded else ''}, "
        f"{cache.decoded_bytes / (1 << 30):.2f} GiB bf16 at scale {scale:.3g} in "
        f"{time.perf_counter() - started:.1f}s; {free_after / (1 << 30):.2f} GiB free of "
        f"{free_before / (1 << 30):.2f}{'' if len(decoded) == end - start else ' (the rest staged)'}",
        flush=True,
    )


def generate(
    model: Qwen4ExpModel,
    ctx: TPContext,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    chunk_size: int = 512,
    eos_token_ids: tuple[int, ...] = (),
    on_token=None,
    on_step=None,
    local_eos: list[bool] | None = None,
    graph_window: int = 0,
    cache=None,
) -> tuple[list[int], dict[str, float]]:
    """Greedy decode with chunked prefill.

    Chunking keeps the QSA attention matrix bounded: a 64K single-shot prefill
    would need a 64K x 64K mask per layer, which does not fit.  The GatedDeltaNet
    and QSA caches carry state across chunks (validated by
    `test_chunked_prefill_matches_single_shot`).

    `on_step` is asked before every decode step and stops the loop when it returns
    true, the same contract `mimo_v2.generate` and `xing4_0.generate` take.  It runs
    on *every* rank, not rank 0 alone: this loop is entered by all of them, so a rank
    that stopped on its own would leave its peers inside the per-layer all-reduce
    that nobody else enters.  The caller is the one that can make the predicate a
    collective -- see `Qwen4ExpBackend._loop`, which broadcasts rank 0's decision --
    and that is why it is a callback here rather than a local `self._is_cancelled`
    read.  `stopped` in the returned stats names what ended the loop.

    **`local_eos` is what keeps the *end-of-sequence* exit inside that same
    collective.** Without it this loop broke out on EOS on its own, and that is not a
    local decision: the token is each rank's own `argmax`, so any step where the
    ranks disagree -- a near-tie in the gathered logits is enough -- let one rank
    leave while its peers walked into the next `on_step` and sat in its broadcast.
    The measured result was `WorkNCCL(OpType=BROADCAST, NumelIn=1, NumelOut=1,
    Timeout(ms)=600000) ran for 600018 ms` and `taking the entire process down`,
    on 8xV100 and on 8xRTX 4090 alike; it is not a slow request, it is a hung one.

    With `local_eos` given, EOS sets that flag instead of breaking.  The caller's
    predicate -- the one it already hands `on_step` -- reads it, so the answer is
    agreed on the same broadcast the cancel check rides, and every rank leaves
    together.  A caller that passes nothing keeps the old local behaviour, which is
    correct at world size 1 and is what the CLI entry point uses.

    Timing is drained, not wall-clocked at the seam: both `torch.cuda.synchronize()`
    calls are load-bearing (`docs/guides/benchmarking.md`), because a forward is
    asynchronous and a host-clock seam reads two passes as one.
    """
    prompt_len = input_ids.shape[1]
    if cache is None:
        cache = model.make_cache(
            batch_size=input_ids.shape[0], max_seq_len=prompt_len + max_new_tokens + 1
        )
    else:
        # A caller-owned arena is reused across requests, and a captured window is captured
        # against *its* addresses -- which is what makes the capture a one-off instead of a cost
        # every decode pays.  The length has to be reset or the prefill would write past the
        # previous request's history and the layers would read it as context.
        cache.length = 0
        cache.ple_token_history = None

    profiler = model.profiler
    prof_scope = profiler.scope if profiler else lambda _name: nullcontext()

    t0 = time.perf_counter()
    logits = None
    past = 0
    with torch.inference_mode(), prof_scope("prefill"):
        while past < prompt_len:
            piece = input_ids[:, past : past + chunk_size]
            logits = model.forward(
                piece,
                cache=cache,
                past_len=past,
                last_token_only=True,
            )
            past += piece.shape[1]
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    # Under TP the head is vocab-sharded, so the last position's logits are gathered into the full
    # vocabulary here; under PP exactly one stage has them and no gather applies.
    logits = gather_logits(logits[:, -1], ctx) if ctx.pp_size <= 1 and logits is not None else logits
    produced: list[int] = []
    _steps = 0
    rows = int(input_ids.shape[0])
    produced_rows: list[list[int]] = [[] for _ in range(rows)]
    stopped = "length"
    t1 = time.perf_counter()
    t_first = None
    next_id = next_token_from_logits(logits, ctx, batch_size=input_ids.shape[0])

    # A run of captured steps, when the caller asked for one.  Everything below treats it as
    # optional: an unusable window leaves the loop eager, which is the path it is checked against.
    window = None
    _replay_count = 0
    _step_s = 0.0
    _capture_s = 0.0
    _window_size = 0
    _window_start = 0
    _window_size_used = 0
    if graph_window > 0 and ctx.device.type == "cuda" and max_new_tokens > graph_window:
        from relicllm.models.qwen4_exp.graph_window import build

        _capture_t0 = time.perf_counter()
        window = build(
            model=model,
            cache=cache,
            start=past,
            steps=min(graph_window, max_new_tokens),
            token=next_id,
            select=lambda logits: next_token_from_logits(
                logits, ctx, batch_size=input_ids.shape[0]
            ),
        )
        _window_size = min(graph_window, max_new_tokens)
        _window_start = past
        _window_size_used = (
            min(_window_size, max_new_tokens) if window is not None else 0
        )
        # The capture is a one-off setup cost, not part of the step rate: charging it to the
        # decode makes a window look slower the longer it is, which is the opposite of what it
        # buys.  It is reported on its own.
        _capture_s = time.perf_counter() - _capture_t0
        capture_s = _capture_s
        if window is None and ctx.rank == 0:
            print("[qwen4exp] decode graph window unavailable; stepping eagerly", flush=True)

    with torch.inference_mode(), prof_scope("decode"):
        for _ in range(max_new_tokens):
            if on_step is not None and on_step():
                # The predicate is the collective, so it is the thing that decided. `local_eos`
                # separates the two reasons it can say stop, and it is read only after the flag
                # came back agreed.
                stopped = "eos" if (local_eos is not None and local_eos[0]) else "cancel"
                break
            _t_c = time.perf_counter()
            # A batch records a token a row, a single sequence records one a step -- and
            # `produced[i]` stays row 0 for a batch, so a reader that only ever saw the
            # single-sequence shape is unchanged.
            step_tokens = [int(v) for v in next_id.reshape(-1).tolist()]
            token = step_tokens[0]
            _step_s += time.perf_counter() - _t_c
            _steps += 1
            produced.append(token)
            for row, value in enumerate(step_tokens):
                produced_rows[row].append(value)
            if t_first is None:
                t_first = time.perf_counter() - t1
            if on_token is not None and ctx.rank == 0:
                # Rank 0 alone, and that is a collective hazard rather than an optimisation: every
                # other rank has already left this line for the next step's collective, so a
                # callback that raises here unwinds rank 0 out of a broadcast its peers are waiting
                # in. The caller owns making it safe -- `Qwen4ExpBackend._rank0_callback` catches
                # and turns it into a `cancel()` that `on_step` broadcasts -- because only the
                # caller knows what the failure means for the request.
                on_token(token)
            if token in eos_token_ids:
                if local_eos is None:
                    stopped = "eos"
                    break
                # Recorded, not acted on: the caller's predicate reads this on the next step and
                # the broadcast is what ends the loop, so no rank leaves on its own. See the
                # `local_eos` paragraph in the docstring for what acting on it locally cost.
                local_eos[0] = True
            if _window_size > 0 and past >= _window_start + _window_size_used:
                # This round's positions are spent.  Its graphs are worth nothing now, so they are
                # given back and the next stretch is captured in their place -- a generation longer
                # than one window stays on the captured path instead of dropping to eager for the
                # rest of its steps.
                if window is not None:
                    window.release()
                    window = None
                remaining = max_new_tokens - len(produced)
                if remaining > 1:
                    _round_t0 = time.perf_counter()
                    # The window is a fixed size, not `remaining - 1`: the size is part of the
                    # key a captured window is filed under, so letting it track the tokens left
                    # would make every round a new window and no request could reuse the ones an
                    # earlier request already captured.
                    _next_window = build(
                        model=model,
                        cache=cache,
                        start=past,
                        steps=_window_size,
                        token=next_id,
                        select=lambda logits: next_token_from_logits(
                            logits, ctx, batch_size=input_ids.shape[0]
                        ),
                    )
                    # Setup again, not a step: the same reason the first window's capture is
                    # taken out of the decode rate.
                    _capture_s += time.perf_counter() - _round_t0
                    if _next_window is not None:
                        window = _next_window
                        _window_start = past
                        # The window was captured whole; only the positions this generation will
                        # actually reach are replayed from it.
                        _window_size_used = min(len(window), remaining)
            if window is not None and window.start <= past < window.start + window.steps:
                _replay_count += 1
                # A replay returns nothing, so the token it produced is read from the buffer the
                # graph writes.  `past` advances because this position is spent either way.
                _t_a = time.perf_counter()
                next_id = window.replay(past, next_id.to(ctx.device))
                # The position's cache clone is what the next one is cloned from, so it has to be
                # handed back before the generation moves on.
                window.commit(past)
                _step_s += time.perf_counter() - _t_a
                past += 1
                continue
            step = model.forward(next_id.to(ctx.device), cache=cache, past_len=past)
            past += 1
            # Under PP only the last stage produced logits; every other stage got `None` back and
            # only needs the token the last stage is about to broadcast.
            if ctx.pp_size <= 1:
                last = gather_logits(step[:, -1], ctx)
            else:
                last = step[:, -1] if step is not None else None
            next_id = next_token_from_logits(last, ctx, batch_size=input_ids.shape[0])
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    if _replay_count and _step_s:
        # A captured decode is timed over its steps, not over the wall clock around them: the
        # captures are setup and the wall clock also holds them.
        decode_s = _step_s
    else:
        decode_s = time.perf_counter() - t1
    if ctx.rank == 0:
        print(f"[qwen4exp] replayed {_replay_count}/{len(produced)} steps", flush=True)

    stats = {
        "prompt_tokens": float(prompt_len),
        "generated_tokens": float(len(produced)),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "ttft_s": prefill_s + (t_first if t_first is not None else 0.0),
        "stopped": stopped,
        "prefill_tps": prompt_len / prefill_s if prefill_s > 0 else 0.0,
        # Tokens, not steps: a batch of eight produces eight times the tokens in the same
        # steps, and a rate that counted steps would report the batch as no faster.
        "decode_tps": (rows * _steps) / decode_s if decode_s > 0 else 0.0,
    }
    # A batch answers a list a row; one row answers the flat list every caller already reads.
    if rows > 1:
        return produced_rows, stats
    return produced, stats


def _load_tokenizer(model_dir: str):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))


def _apply_chat_template(model_dir: str, prompt: str) -> str:
    """Wrap a prompt in the checkpoint's chat template if jinja2 is available."""
    template_path = os.path.join(model_dir, "chat_template.jinja")
    if not os.path.exists(template_path):
        return prompt
    try:
        from jinja2 import Template
    except ImportError:
        return prompt
    with open(template_path) as f:
        template = Template(f.read())
    return template.render(
        messages=[{"role": "user", "content": prompt}], add_generation_prompt=True
    )


def run_tp4(
    model_dir: str,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    chunk_size: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    expert_cache_capacity: int = 0,
    host_expert_memory: bool = True,
    pin_experts: bool = True,
    expert_resident: bool = False,
    expert_reserve_bytes: int = 3 << 30,
    raw_prompt: bool = False,
    enable_profile: bool = False,
    expert_deep_cache: float | None = None,
    expert_decode_device_gib: float | None = None,
    expert_fused_fp8: bool = False,
    graph_window: int = 0,
) -> dict:
    from relicllm.models.qwen4_exp.profiler import Profiler

    ctx = init_distributed()
    profiler = Profiler(enabled=enable_profile, device=ctx.device) if enable_profile else None

    if ctx.rank == 0:
        print(f"[rank{ctx.rank}] loading {model_dir} world_size={ctx.world_size}", flush=True)

    t0 = time.perf_counter()
    model, config = load_model(
        model_dir,
        ctx,
        dtype=dtype,
        expert_cache_capacity=expert_cache_capacity,
        host_expert_memory=host_expert_memory,
        pin_experts=pin_experts,
        expert_resident=expert_resident,
        expert_reserve_bytes=expert_reserve_bytes,
        profiler=profiler,
        expert_deep_cache=expert_deep_cache,
        expert_decode_device_gib=expert_decode_device_gib,
        expert_fused_fp8=expert_fused_fp8,
    )
    load_s = time.perf_counter() - t0
    if ctx.rank == 0:
        mem = torch.cuda.memory_allocated(ctx.device) / 2**30 if ctx.device.type == "cuda" else 0.0
        print(f"[rank{ctx.rank}] loaded in {load_s:.1f}s, {mem:.2f} GiB on device", flush=True)

    # All layers share one MoE backend instance; reach it through the first layer this rank holds.
    # Not `layers[0]`: under pipeline parallelism only the first stage owns layer 0, so every other
    # rank would read `None` here.
    moe_backend = getattr(model.local_layers[0], "moe", None) if model.local_layers else None
    moe_backend = getattr(moe_backend, "inner", moe_backend)  # unwrap ShardedMoE
    moe_phase_profile = enable_profile and os.environ.get("POCKETLLM_QWEN4_MOE_PHASE_PROFILE") == "1"
    if moe_phase_profile and hasattr(moe_backend, "phase_stats"):
        moe_backend.phase_stats = {}

    tokenizer = _load_tokenizer(model_dir)
    text = prompt if raw_prompt else _apply_chat_template(model_dir, prompt)
    input_ids = torch.tensor([tokenizer.encode(text).ids], dtype=torch.long)

    eos = tuple(config.text_config.eos_token_id)
    produced, stats = generate(
        model,
        ctx,
        input_ids,
        max_new_tokens=max_new_tokens,
        chunk_size=chunk_size,
        eos_token_ids=eos,
        graph_window=graph_window,
    )
    stats["load_s"] = load_s

    result = {"tokens": produced, "text": tokenizer.decode(produced), "stats": stats}
    if ctx.rank == 0:
        print(f"[rank0] output: {result['text']!r}", flush=True)
        print(f"[rank0] stats: {json.dumps(stats, indent=2)}", flush=True)

        if moe_phase_profile and getattr(moe_backend, "phase_stats", None):
            ps = moe_backend.phase_stats
            print(
                "\n[rank0] moe phases: "
                f"stage {ps.get('stage_s', 0.0):.3f}s  "
                f"compute {ps.get('compute_s', 0.0):.3f}s  "
                f"expert-calls {ps.get('experts', 0)} over {ps.get('calls', 0)} moe calls",
                flush=True,
            )

        # Always, profiled or not.  `--profile` suppresses `_prefetch` and switches
        # `_dispatch_experts` to a synchronizing path, so a profiled run cannot say what the
        # counters would have been -- the run that reports them is exactly the run that
        # destroyed what they measure.  The values are not gated; only this print was, and an
        # acceptance criterion that needs `stage_bytes == 0` has to be readable off the run it
        # is a criterion for.
        if moe_backend is not None and hasattr(moe_backend, "stage_calls"):
            staged = moe_backend.stage_calls
            hits = moe_backend.cache_hits
            gib = moe_backend.stage_bytes / 2**30
            print(
                f"[rank0] staging: {staged} misses / {hits} hits "
                f"({100.0 * hits / max(1, staged + hits):.1f}% hit), {gib:.2f} GiB H2D",
                flush=True,
            )

        if profiler and profiler.enabled:
            print("\n" + profiler.aggregate_report())
            if _env_enabled("POCKETLLM_QWEN4_PROFILE_TREE"):
                print("\n" + profiler.report())

    if ctx.initialized:
        import torch.distributed as dist

        dist.barrier()
        dist.destroy_process_group()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen4-Exp TP heterogeneous inference")
    parser.add_argument("--model", required=True, help="checkpoint directory")
    parser.add_argument("--prompt", default="Hello, who are you?")
    parser.add_argument("--prompt-file", help="read the prompt from a file instead of --prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument(
        "--expert-cache",
        type=int,
        default=0,
        help="number of staged experts to keep on device (0 = restage every step)",
    )
    parser.add_argument(
        "--mmap-experts",
        action="store_true",
        help="diagnostic fallback: read routed experts from the mmap on demand",
    )
    parser.add_argument(
        "--pageable-experts",
        action="store_true",
        help="keep the resident host shard pageable instead of requesting pinned memory",
    )
    parser.add_argument("--raw-prompt", action="store_true", help="skip the chat template")
    parser.add_argument("--profile", action="store_true", help="enable hierarchical profiler")
    parser.add_argument(
        "--graph-window",
        type=int,
        default=0,
        help=(
            "EXPERIMENTAL and not working: a window of one position replays correctly, two or "
            "more do not -- each graph is recorded against the cache as it was at its own "
            "position, and the graphs share one mutable cache, so the second capture is read by "
            "the first replay. Captures also cost about 0.3s a position. Left behind because the "
            "supporting work (the warmup, the cache save/restore, the logits selector) is right; "
            "0 leaves the decode eager, which is the only working path today"
        ),
    )
    parser.add_argument(
        "--expert-resident",
        action="store_true",
        help=(
            "hold this rank's routed experts on the device instead of in host RAM, so a decode "
            "step stages nothing (experts must fit on the card; see --expert-reserve-gib). "
            "Off by default: the 2080 Ti path stages from host and has no device-side expert cost"
        ),
    )
    parser.add_argument(
        "--expert-reserve-gib",
        type=float,
        default=3.0,
        help="device memory to leave free for activations and KV under --expert-resident",
    )
    parser.add_argument(
        "--expert-deep-cache",
        type=float,
        metavar="SCALE",
        default=None,
        help=(
            "hold each layer's experts on the device, decoded to bf16 at a fixed weight scale, "
            "so a decode step reads a row instead of staging or dequantizing one. Reads the "
            "checkpoint's own FP8 codes; the scale is the value t where every code * t <= 448 "
            "(pick it from the checkpoint's weight distribution, not from guesswork)"
        ),
    )
    parser.add_argument(
        "--expert-decode-gib",
        type=float,
        default=None,
        help=(
            "device memory the deep cache may use; layers beyond it stay on the host. Default is "
            "every layer the card can still hold after the dense weights"
        ),
    )
    parser.add_argument(
        "--expert-fused-fp8",
        action="store_true",
        help=(
            "with --expert-resident, run each step's top-k FP8 experts in one fused kernel pair "
            "that reads the codes and block scales in the load, instead of a per-slot Python loop. "
            "WARNING: this path currently produces wrong logits (a decode of a real prompt "
            "degenerates to repeated tokens) while the per-expert loop beside it does not; it is "
            "kept for measurement and is not a working decode path yet"
        ),
    )
    args = parser.parse_args()

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read()

    run_tp4(
        args.model,
        prompt,
        max_new_tokens=args.max_new_tokens,
        chunk_size=args.chunk_size,
        dtype=getattr(torch, args.dtype),
        expert_cache_capacity=args.expert_cache,
        host_expert_memory=not args.mmap_experts,
        pin_experts=not args.pageable_experts,
        expert_resident=args.expert_resident,
        expert_deep_cache=args.expert_deep_cache,
        expert_decode_device_gib=args.expert_decode_gib,
        expert_reserve_bytes=int(args.expert_reserve_gib * (1 << 30)),
        expert_fused_fp8=args.expert_fused_fp8,
        raw_prompt=args.raw_prompt,
        enable_profile=args.profile,
        graph_window=args.graph_window,
    )


if __name__ == "__main__":
    main()
