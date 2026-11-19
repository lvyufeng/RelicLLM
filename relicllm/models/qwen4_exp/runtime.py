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
import torch.nn.functional as F

from relicllm.models.qwen4_exp.builder import build_heterogeneous, layer_range_for
from relicllm.models.qwen4_exp.config import Qwen4ExpConfig
from relicllm.models.qwen4_exp.model import Qwen4ExpModel, restore_cache, snapshot_cache
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
    #: Reusable all-gather output for the vocab-sharded head.  Set by a decode loop that replays a
    #: capture, because the gather runs between replays and an allocation there can be handed the
    #: memory the recording owns.
    gather_parts: list | None = None
    #: Reusable ``(values, indices)`` pair for the vocab-sharded greedy pick.  Allocated
    #: beside ``gather_parts`` for the same reason: the exchange runs between replays, and an
    #: allocation there can be handed memory the recording owns.
    argmax_parts: tuple | None = None
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
        if os.environ.get("POCKETLLM_QWEN4_SKIP_REDUCE") == "1":
            # A ceiling measurement, not a mode: the block outputs are never summed, so the text is
            # wrong and every rank answers from its own shard.  What it does answer honestly is how
            # much of a decode step is the row-parallel sum -- the one number that decides whether
            # a smaller TP degree, or a replica layout, is the right shape for this box.
            return tensor
        # The tensor is a fresh block output, so an in-place reduce is safe and
        # avoids an extra allocation per layer.
        tensor = tensor.contiguous()
        rows = tensor.numel() // tensor.shape[-1]
        if (os.environ.get("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "0") == "1"
                and tensor.dtype is torch.bfloat16 and 1 < rows <= 8):
            # BF16 ring reduction order changes with payload size. Even bit-identical inputs to
            # the first layer differed after a four-row sum versus one-row decode. Keep each
            # collective's element count equal to decode's; grouping/coalescing the rows also
            # changed the arithmetic in the controlled test, so these are deliberately separate.
            if (os.environ.get("POCKETLLM_QWEN4_VERIFY_REDUCE_LAYOUT", "serial") == "rank_chunk"
                    and tensor.device.type == "cuda" and ctx.tp_world_size == 8
                    and rows in (4, 5, 6, 8) and tensor.shape[-1] == 2560):
                # Hardware-scoped experimental layout, not a generic associative reduction:
                # retaining the one-row ring owner's 320-element chunks for each row preserved
                # BF16 rounding in cancellation-heavy eager/captured probes on this host. The
                # ordinary payload did not. Only these row spans passed cancellation-heavy probes;
                # full generated-sequence parity still gates the opt-in flag for each draft depth.
                packed = tensor.reshape(rows, 8, 320).permute(1, 0, 2).contiguous()
                dist.all_reduce(packed, op=dist.ReduceOp.SUM)
                tensor.copy_(packed.permute(1, 0, 2).reshape_as(tensor))
            else:
                for row in tensor.reshape(rows, tensor.shape[-1]):
                    dist.all_reduce(row, op=dist.ReduceOp.SUM)
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    return all_reduce


def _last_token_logits(logits: torch.Tensor | None, *, batch_size: int) -> torch.Tensor:
    """Normalize a head's ``[B, V]`` or ``[B, S, V]`` output to ``[B, V]``.

    Only the sequence axis is removed. Flattening other shapes would mix rows or silently
    broadcast a pipeline head's extra singleton into a token buffer.
    """
    if logits is None:
        raise ValueError("the head must provide logits for next-token selection")
    if logits.ndim == 3 and logits.shape[1] > 0:
        logits = logits[:, -1]
    if logits.ndim != 2 or logits.shape[0] != batch_size or logits.shape[1] == 0:
        raise ValueError(
            f"expected logits [B, V] or [B, S, V] with B={batch_size}, got {tuple(logits.shape)}"
        )
    return logits


def _decode_token(token: torch.Tensor, *, batch_size: int) -> torch.Tensor:
    """Require one int64 token per row, without implicit reshaping or broadcasting."""
    if token.shape != (batch_size, 1):
        raise ValueError(f"expected token shape {(batch_size, 1)}, got {tuple(token.shape)}")
    if token.dtype != torch.long:
        raise TypeError(f"expected int64 tokens, got {token.dtype}")
    return token


def next_token_from_logits(
    last_logits: torch.Tensor | None, ctx: TPContext, *, batch_size: int = 1
) -> torch.Tensor:
    """The greedy next token, shape ``[B, 1]``, agreed across every rank.

    The input is the head's *local*, ungathered ``[B, V]`` or ``[B, S, V]`` logits, or
    ``None`` on a pipeline stage without a head. This helper alone owns the TP gather;
    callers must not gather the same logits before handing them here. Under pipeline
    parallelism the last stage takes the argmax and broadcasts it to every other stage.

    The broadcast source is the last stage, not rank 0: the token belongs to the head
    there, while rank 0 still owns the cancel/EOS flag.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if ctx.pp_size <= 1:
        local = _last_token_logits(last_logits, batch_size=batch_size)
        if _argmax_exchange_enabled() and ctx.tp_world_size > 1:
            return argmax_sharded(local, ctx)
        return gather_logits(local, ctx).argmax(-1, keepdim=True)

    import torch.distributed as dist

    if ctx.pp_rank == ctx.pp_size - 1:
        local = _last_token_logits(last_logits, batch_size=batch_size)
        token = local.argmax(-1, keepdim=True)
    else:
        token = torch.zeros((batch_size, 1), dtype=torch.long, device=ctx.device)
    dist.broadcast(token, src=ctx.pp_size - 1)
    return token


def _argmax_exchange_enabled() -> bool:
    """Whether the greedy pick exchanges ranks' maxima instead of the whole vocabulary.

    Off by default, and measured rather than assumed: on eight RTX 4090s at 8192 positions it
    produced identical tokens and no speedup (43.90 ms against 43.56 ms p50), because the
    vocabulary gather it replaces is not on the critical path -- the profile that said it was
    had charged the collective for waiting behind the recorded step.  Kept because the
    arithmetic is exact and the byte count is genuinely lower (two numbers a rank instead of a
    vocabulary slice), which may matter on a fabric where that transfer is the cost; it is not
    a win here and it is not the default.
    """
    return os.environ.get("POCKETLLM_QWEN4_ARGMAX_EXCHANGE") == "1"


def _argmax_all_gather(local: torch.Tensor, ctx: TPContext) -> list[torch.Tensor]:
    """Gather one scalar row per rank. Split out so the arithmetic is testable off a device."""
    import torch.distributed as dist

    world = ctx.tp_world_size
    parts = ctx.argmax_parts
    if parts is None or parts[0].dtype != local.dtype or parts[0].shape != local.shape:
        parts = [torch.empty_like(local) for _ in range(world)]
        ctx.argmax_parts = parts
    dist.all_gather(parts, local.contiguous())
    return parts


def argmax_sharded(local_logits: torch.Tensor, ctx: TPContext) -> torch.Tensor:
    """Greedy pick over a column-sharded vocabulary without gathering the vocabulary.

    ``torch.cat(parts).argmax()`` returns the first maximum in concatenated order, which is the
    lowest *global* index among equal maxima.  Three facts make the cheap form exact: every shard
    owns one contiguous slice, so a local index becomes global by adding ``rank * width``; ranking
    the shard maxima and then taking the lowest rank among ties is that same rule, because the
    shards are contiguous and ordered by rank; and a float32 cast of the values is lossless for
    both dtypes this head produces, so the comparison is bit-for-bit the one the gather made.

    What crosses the wire is two numbers a rank instead of a vocabulary slice a rank.

    Returns ``[batch, 1]`` int64, equal to ``gather_logits(...).argmax(-1, keepdim=True)``.
    """
    world = ctx.tp_world_size
    batch = int(local_logits.shape[0])
    if world == 1:
        return local_logits.argmax(-1, keepdim=True)
    width = int(local_logits.shape[-1])
    values, local_index = local_logits.max(dim=-1)             # [batch]
    global_index = local_index + int(ctx.tp_rank) * width
    if int(ctx.tp_rank) * width + width > (1 << 24):
        raise ValueError("a float32 index exchange cannot represent this vocabulary")
    # One row a rank: column 0 is the shard's best value, column 1 its global index.
    packed = torch.stack([values.to(torch.float32), global_index.to(torch.float32)], dim=-1)
    shipped = _argmax_all_gather(packed, ctx)                   # world x [batch, 2]
    table = torch.stack(shipped, dim=0)                         # [world, batch, 2]
    best = table[..., 0].max(dim=0).values                      # [batch]
    ranks = torch.arange(world, device=table.device).reshape(-1, 1).expand(world, batch)
    unreachable = torch.full_like(ranks, world, dtype=torch.long)
    # Lowest rank among equal maxima, which is the lowest global index.
    winner = torch.where(table[..., 0] == best.reshape(1, -1), ranks, unreachable).min(dim=0).values
    token = table[..., 1].gather(0, winner.reshape(1, -1)).reshape(-1, 1)
    return token.to(torch.long)


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

    reusable = (
        ctx.gather_parts is not None
        and ctx.gather_parts[0].shape == logits.shape
        # The captured decode hands this float32 (the window stores its output widened) while the
        # eager step hands it bfloat16, so shape alone would reuse a buffer of the wrong dtype and
        # `all_gather` refuses the pair rather than converting.
        and ctx.gather_parts[0].dtype == logits.dtype
    )
    if reusable:
        # Reused across steps when the caller has preallocated them.  A decode loop that replays a
        # capture must not allocate between replays -- the allocator can hand back memory the
        # graph recorded and the next replay writes through a stale address -- so the loop keeps
        # one buffer and the gather writes into it.
        dist.all_gather(ctx.gather_parts, logits.contiguous())
    else:
        # Allocated with inference mode *off*, because these outlive the call. A buffer created
        # inside `torch.inference_mode` is an inference tensor, and `all_gather` writing into one
        # from a later call raises "Inplace update to inference tensor outside InferenceMode" --
        # which is what a second `generate_mtp_draft` in one process hit. A decode loop that
        # replays a capture keeps one buffer for the same reason it must not reallocate: the
        # allocator could hand back memory the recording owns.
        with torch.inference_mode(False):
            parts = [torch.empty_like(logits) for _ in range(ctx.tp_world_size)]
        dist.all_gather(parts, logits.contiguous())
        ctx.gather_parts = parts
    return torch.cat(ctx.gather_parts, dim=-1)


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


def _head_pin_budget(ctx: TPContext, mtp: bool, reserve_bytes: int) -> int:
    """Bytes the MTP head may hold resident: what the card has free, less the caller's reserve.

    Measured at the moment the head is built, which is after the trunk's bank has been filled, so
    this is the leftover rather than a guess from the total.  Pinning is all-or-nothing a layer, so
    a budget that is too small pins nothing and the head stages its experts instead -- the failure
    is a slow draft, not a wrong one.
    """
    if not mtp or ctx.device.type != "cuda":
        return 0
    free, _ = torch.cuda.mem_get_info(ctx.device)
    return max(0, int(free) - int(reserve_bytes))


def mtp_expert_bytes(config: Qwen4ExpConfig, *, world_size: int = 1) -> int:
    """Bytes a rank holds for the MTP head's routed experts, one layer's worth.

    The same formula as :func:`resident_expert_bytes` with a single layer, which is all the head
    has.  Its own config is used because the MTP block can be quantised differently from the trunk
    -- the ModelOpt NVFP4 build leaves the MTP layer at FP8 -- so its experts are not the same
    size as the trunk's, and estimating one from the other would be wrong by the ratio of the two
    formats.
    """
    return resident_expert_bytes(config, world_size=world_size) // max(
        1, int(config.text_config.num_hidden_layers)
    )


def require_resident_budget(
    config: Qwen4ExpConfig,
    ctx: TPContext,
    *,
    reserve_bytes: int,
    free_at_start: int | None = None,
    extra_experts_bytes: int = 0,
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
    # `extra_experts_bytes` is a second bank that has to live on the card too -- the MTP head's
    # experts.  Counted in the requirement rather than left to the activation reserve, because a
    # residency that passes the check and then OOMs halfway through building the head is precisely
    # the failure this function exists to turn into a named error.
    experts_bytes += int(extra_experts_bytes)
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


def _default_tensor_core_qsa() -> None:
    """Ask for the tiled sparse-attention kernel unless the launch already decided.

    The selector in `relic-core` reads the variable itself and treats *unset* as "use the scalar
    kernel", so this is not a preference being imposed -- it is the difference between a run that
    has heard of the tiled path and one that has not.  Kept as a function so the decision is
    testable without a card or a model.
    """
    if os.environ.get("POCKETLLM_QWEN4_QSA_MMA") is None:
        os.environ["POCKETLLM_QWEN4_QSA_MMA"] = "1"


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
    expert_resident_layers: bool = False,
    graph_window: int = 0,
    expert_nvfp4: bool = False,
    mtp: bool = False,
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
    if expert_nvfp4:
        # Every expert is already whole-device, so the host copy has nothing to stage for and
        # reading 56 GiB a rank of FP8 codes this checkpoint does not contain would fail.
        host_expert_memory = False
    if graph_window > 0 and expert_resident_layers and not expert_fused_fp8:
        # A capture cannot record a step that reads the host, and the pinned-layer path gathers its
        # experts by naming rows in a Python dict.  The bank can serve the same step from a
        # device-side stack of the bytes it already holds, which costs nothing extra and is what
        # makes `--graph-window` usable together with `--expert-resident-layers`.  Set here rather
        # than asked of the caller: it is a consequence of the two flags, not a third choice.
        os.environ.setdefault("POCKETLLM_QWEN4_CAPTURE_BANK", "1")
    # The extension carries a tensor-core sparse-attention kernel beside the scalar one, and its
    # selector takes the scalar path unless this variable is *set at all* -- unset and set-to-empty
    # both mean scalar, so the fast kernel is off in every launch that has not heard of it.  On
    # eight RTX 4090s at the native 2048 budget it is 1.62x the decode rate at 8192 positions
    # (19.19 against 31.09 tok/s excluding setup, 51.7 against 30.5 ms inter-token) and 1.23x at
    # 262144, with identical tokens.  Set here as a default a launch may still override:
    # `POCKETLLM_QWEN4_QSA_MMA=0` is a working opt-out, and the scalar kernel remains the fallback
    # for any shape the tiled one does not cover.  It is set at this scope, not inside a branch, so
    # the choice does not depend on which other flags a run happens to pass.
    _default_tensor_core_qsa()
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
                # The head's own experts are a second bank on the same card. Zero when there is no
                # head, which keeps the trunk-only paths exactly as they were.
                extra_experts_bytes=(mtp_expert_bytes(config, world_size=ctx.tp_world_size)
                                     if mtp else 0),
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
        expert_resident_layers=expert_resident_layers,
        expert_nvfp4=expert_nvfp4,
        expert_reserve_bytes=expert_reserve_bytes,
        mtp=mtp,
        # What is left on the card after the trunk's bank and the reserve, for the head's own
        # experts: it runs once a round under drafting, so staging it every time defeats the
        # point.  The pin is all-or-nothing a layer, so an estimate that is too generous simply
        # pins nothing rather than half a layer.
        mtp_pin_bytes=_head_pin_budget(ctx, mtp, expert_reserve_bytes),
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


def _synchronize_device(ctx: TPContext) -> None:
    """Drain this rank's CUDA work at a wall-clock boundary."""
    if ctx.device.type == "cuda":
        torch.cuda.synchronize(ctx.device)


def _build_decode_window(*, ctx: TPContext, model, cache, start, steps, token, select):
    """Refuse unsupported devices/PP before a graph builder can mutate request state."""
    if ctx.device.type != "cuda" or ctx.pp_size > 1:
        return None
    from relicllm.models.qwen4_exp.graph_window import build

    return build(model=model, cache=cache, start=start, steps=steps, token=token, select=select)


def _warmup_device(ctx: TPContext, dtype: torch.dtype, seconds: float) -> float:
    """Run optional throwaway CUDA work before any measured prefill or decode.

    This touches neither model weights nor caches. It is a clock-ramp hint, not a promise
    that a device's clocks are locked, and on a CPU it deliberately does nothing.
    """
    if seconds <= 0 or ctx.device.type != "cuda":
        return 0.0
    _synchronize_device(ctx)
    started = time.perf_counter()
    with torch.inference_mode():
        scratch = torch.randn(4096, 4096, device=ctx.device, dtype=dtype) * 0.01
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            for _ in range(8):
                scratch = scratch @ scratch
            scratch = scratch / (scratch.abs().max() + 1.0)
    _synchronize_device(ctx)
    del scratch
    torch.cuda.empty_cache()
    return time.perf_counter() - started


def _reset_request_cache(cache, *, ctx: TPContext | None = None) -> None:
    """Clear sequence state for a new request, not an MTP/prefix continuation.

    The raw KV/index arenas and rotary tables keep their storage: resetting logical
    lengths masks the old tail, and prefill overwrites every position it consumes.
    Active graph layouts must be released by their owner before this helper runs.
    """
    owners = [cache]
    owners.extend(getattr(cache, "layers", ()) or ())
    owners.extend((getattr(cache, "ple_states", {}) or {}).values())
    owners = list({id(owner): owner for owner in owners if owner is not None}.values())
    if (getattr(cache, "_decode_graph_owner", None) is not None
            or getattr(cache, "ple_history_at", None) is not None) or any(
        getattr(owner, "state_slots", None) is not None for owner in owners
    ):
        raise RuntimeError(
            "release the active decode graph/state-slot layout or history staging before resetting a request"
        )
    # The preflight above must precede both mutation and device work. A caller can
    # supply an arena after an asynchronous eager forward, so drain before clearing it.
    if ctx is not None:
        _synchronize_device(ctx)
    with torch.inference_mode():
        for owner in owners:
            for name in ("conv_state", "recurrent_state"):
                value = getattr(owner, name, None)
                if isinstance(value, torch.Tensor):
                    value.zero_()
            if hasattr(owner, "initialized"):
                owner.initialized = False
            if hasattr(owner, "length"):
                owner.length = 0
            if hasattr(owner, "host_position"):
                owner.host_position = 0
            if hasattr(owner, "state_log"):
                owner.state_log = None
        if hasattr(cache, "ple_token_history"):
            # None takes the same EOS-prefix initialization path as a fresh cache.
            cache.ple_token_history = None
    if ctx is not None:
        _synchronize_device(ctx)


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

    A supplied `cache` is reusable storage for a *new* request, not continuation
    state. GDN/PLE buffers are zeroed in place and logical lengths/history reset;
    active graph layouts must be released first. `cache_reset_s` reports this cleanup
    including its device drains, outside `prefill_s`, `ttft_s`, and `decode_s`. Newly
    allocated caches already start empty and report zero reset time. Arena allocation
    and request cleanup are not part of the measured prefill/decode intervals.

    `prefill_s` ends at the drained, actual prefill seam. `decode_s` (also named
    `decode_wall_s`) is drained wall time from that seam through the decode loop, for
    both graph and eager paths. It includes initial gather/selection, graph setup,
    rebuilds, host token reads and callbacks; it is not a sum of asynchronous GPU calls.
    `capture_s` is the initial build, `rebuild_s` is later release/build setup, and both
    are drained separately. Subtracting them gives `decode_s_excluding_setup`, an
    algebraic wall-time residual, not isolated GPU compute or per-replay latency.
    Final window teardown after the drained decode loop is lifecycle cleanup and is
    outside this interval; releases needed for a rebuild or handback remain inside it.

    `ttft_s` starts with measured prefill and ends when the first token reaches the
    host, before `on_token`. The initial graph build currently precedes that emission,
    so TTFT explicitly includes its cost as well as the initial gather and decision.
    It is zero when no token is emitted. `emitted_steps` counts one token per batch
    row per iteration; the first comes from prefill (`prefill_first_token_steps`),
    and only the remaining `decode_steps` contribute to the aggregate `decode_tps`.
    """
    prompt_len = input_ids.shape[1]
    cache_reset_s = 0.0
    if cache is None:
        cache = model.make_cache(
            batch_size=input_ids.shape[0], max_seq_len=prompt_len + max_new_tokens + 1
        )
    else:
        reset_started = time.perf_counter()
        _reset_request_cache(cache, ctx=ctx)
        cache_reset_s = time.perf_counter() - reset_started

    profiler = model.profiler
    prof_scope = profiler.scope if profiler else lambda _name: nullcontext()

    _synchronize_device(ctx)
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
    _synchronize_device(ctx)
    prefill_done = time.perf_counter()
    prefill_s = prefill_done - t0

    produced: list[int] = []
    rows = int(input_ids.shape[0])
    produced_rows: list[list[int]] = [[] for _ in range(rows)]
    stopped = "length"
    t_first = None
    # Selection owns exactly one gather, and accepts the PP head's [B, 1, V] as well
    # as the eager/graph [B, V] shape. A zero-token request has no decision to make.
    next_id = next_token_from_logits(logits, ctx, batch_size=rows) if max_new_tokens > 0 else None
    prompt_ids = [int(v) for v in input_ids[0].tolist()]

    window = None
    _replay_count = 0
    _replay_attempts = 0
    _rebase_count = 0
    _rebase_s = 0.0
    _eager_steps = 0
    _capture_s = 0.0
    _rebuild_s = 0.0
    _window_size = 0
    _window_start = 0
    _window_size_used = 0
    _handback = 0
    if graph_window > 0 and max_new_tokens > graph_window:
        _synchronize_device(ctx)
        _capture_t0 = time.perf_counter()
        window = _build_decode_window(
            ctx=ctx,
            model=model,
            cache=cache,
            start=past,
            steps=min(graph_window, max_new_tokens),
            token=next_id,
            select=lambda logits: next_token_from_logits(logits, ctx, batch_size=rows),
        )
        _synchronize_device(ctx)
        _capture_s = time.perf_counter() - _capture_t0
        if window is not None:
            _window_size = min(graph_window, max_new_tokens)
            _window_start = past
            _window_size_used = min(_window_size, max_new_tokens - 1)
        elif ctx.rank == 0:
            print("[qwen4exp] decode graph window unavailable; stepping eagerly", flush=True)

    with torch.inference_mode(), prof_scope("decode"):
        for emitted_index in range(max_new_tokens):
            if on_step is not None and on_step():
                # The predicate is the collective; local EOS only names its stop reason.
                stopped = "eos" if (local_eos is not None and local_eos[0]) else "cancel"
                break

            if emitted_index > 0:
                # Consult the stop predicate before work for the next emitted token. The
                # prefill already produced the first token, and the final token needs no
                # unused forward after it (N emitted tokens require N-1 decode forwards).
                if (_window_size > 0 and past >= _window_start + _window_size_used
                        and not (window is not None and getattr(window, "reuse_windows", False))):
                    _synchronize_device(ctx)
                    _round_t0 = time.perf_counter()
                    if window is not None:
                        window.release()
                        window = None
                    remaining = max_new_tokens - len(produced)
                    if remaining > 1:
                        # Preserve the fixed window size used for recording/reuse. Only
                        # the positions this request will actually reach are replayed.
                        window = _build_decode_window(
                            ctx=ctx,
                            model=model,
                            cache=cache,
                            start=past,
                            steps=_window_size,
                            token=next_id,
                            select=lambda logits: next_token_from_logits(
                                logits, ctx, batch_size=rows
                            ),
                        )
                        _window_start = past
                        _window_size_used = min(len(window), remaining) if window is not None else 0
                    if window is None:
                        # A declined build leaves this request eager rather than retrying
                        # the same unsupported capture on every subsequent token.
                        _window_size = 0
                    _synchronize_device(ctx)
                    _rebuild_s += time.perf_counter() - _round_t0

                if window is not None and (window.start <= past < window.start + window.steps
                                           or getattr(window, "reuse_windows", False)):
                    # History ends in the token this forward embeds, already in produced.
                    # Adding next_id again would repeat that token in the PLE n-gram.
                    window.fill_history(prompt_ids + produced)
                    _replay_attempts += 1
                    before_rebases = getattr(window, "rebase_count", 0)
                    before_rebase_s = getattr(window, "rebase_s", 0.0)
                    replayed = window.replay(past, next_id.to(ctx.device))
                    _rebase_count += getattr(window, "rebase_count", 0) - before_rebases
                    _rebase_s += getattr(window, "rebase_s", 0.0) - before_rebase_s
                    if replayed is None:
                        # The recording's release restores committed state and detaches
                        # its slot layout before eager overwrites this same position.
                        _handback += 1
                        window.release()
                        window = None
                        step = model.forward(next_id.to(ctx.device), cache=cache, past_len=past)
                        _eager_steps += 1
                        next_id = next_token_from_logits(step, ctx, batch_size=rows)
                    else:
                        next_id = _decode_token(replayed, batch_size=rows)
                        _replay_count += 1
                else:
                    # PP stages without a head return None; the selector broadcasts the
                    # head stage's [B, 1] decision without ever gathering full PP logits.
                    step = model.forward(next_id.to(ctx.device), cache=cache, past_len=past)
                    _eager_steps += 1
                    next_id = next_token_from_logits(step, ctx, batch_size=rows)
                past += 1

            next_id = _decode_token(next_id, batch_size=rows)
            step_tokens = [int(v) for v in next_id[:, 0].tolist()]
            token = step_tokens[0]
            produced.append(token)
            for row, value in enumerate(step_tokens):
                produced_rows[row].append(value)
            if t_first is None:
                t_first = time.perf_counter() - t0
            if on_token is not None and ctx.rank == 0:
                # The caller must turn callback failures into a collective cancellation;
                # an exception escaping rank 0 would strand its peers in the next on_step.
                on_token(token)
            if token in eos_token_ids:
                if local_eos is None:
                    stopped = "eos"
                    break
                # Do not act locally: every rank reaches the caller's next stop predicate.
                local_eos[0] = True
    _synchronize_device(ctx)
    decode_s = time.perf_counter() - prefill_done
    emitted_steps = len(produced)
    prefill_first_token_steps = int(emitted_steps > 0)
    decode_steps = emitted_steps - prefill_first_token_steps
    decode_tokens = rows * decode_steps
    graph_setup_s = _capture_s + _rebuild_s
    decode_s_excluding_setup = max(0.0, decode_s - graph_setup_s)
    if ctx.rank == 0:
        print(f"[qwen4exp] replayed {_replay_count}/{decode_steps} decode steps "
              f"({_replay_attempts} attempts), {_handback} handed back to eager", flush=True)
        if graph_window > 0 and _replay_attempts == 0:
            why = ("no CUDA device" if ctx.device.type != "cuda"
                   else "window unavailable or no replay positions reached")
            print(f"[qwen4exp] no graph replay ({why}); these steps ran eagerly", flush=True)

    # Give the recording back before returning. A live graph holds captured NCCL work, and
    # destroying the process group under one hangs: the teardown barrier is reached by every rank
    # and then never completes, and the watchdog aborts the job after the answer has already been
    # printed. The rotation above releases a window when it is spent; this is the one still
    # serving when the generation ends, and it used to be dropped on the floor.
    if window is not None:
        window.release()
        window = None
    _synchronize_device(ctx)

    stats = {
        "prompt_tokens": float(prompt_len),
        # Keep the legacy per-row token count; aggregate counts are explicit for batches.
        "generated_tokens": float(emitted_steps),
        "generated_tokens_total": float(rows * emitted_steps),
        "emitted_steps": float(emitted_steps),
        "prefill_first_token_steps": float(prefill_first_token_steps),
        "prefill_first_tokens": float(rows * prefill_first_token_steps),
        "decode_steps": float(decode_steps),
        "decode_tokens": float(decode_tokens),
        "cache_reset_s": cache_reset_s,
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_wall_s": decode_s,
        "capture_s": _capture_s,
        "rebuild_s": _rebuild_s,
        "graph_setup_s": graph_setup_s,
        "decode_s_excluding_setup": decode_s_excluding_setup,
        "ttft_s": t_first if t_first is not None else 0.0,
        "stopped": stopped,
        "prefill_tps": prompt_len / prefill_s if prefill_s > 0 else 0.0,
        "decode_tps": decode_tokens / decode_s if decode_s > 0 else 0.0,
        "decode_tps_excluding_setup": (
            decode_tokens / decode_s_excluding_setup if decode_s_excluding_setup > 0 else 0.0
        ),
        "replay_count": float(_replay_count),
        "replay_attempts": float(_replay_attempts),
        "graph_rebase_count": float(_rebase_count),
        "graph_rebase_s": _rebase_s,
        "handback_count": float(_handback),
        "eager_steps": float(_eager_steps),
    }
    # A batch answers a list a row; one row answers the flat list every caller already reads.
    if rows > 1:
        return produced_rows, stats
    return produced, stats


def mtp_accuracy(
    model: Qwen4ExpModel,
    ctx: TPContext,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    chunk_size: int = 512,
    eos_token_ids: tuple[int, ...] = (),
    max_seq_len: int | None = None,
) -> dict:
    """Generate greedily and score the MTP head's draft against what the backbone actually chose.

    This is the Phase-1 check that the head is wired correctly, and it is a *number* rather than an
    impression: run the backbone one token at a time, ask the MTP head -- once per generated token --
    what it thinks follows the token the backbone just chose, and count how often the backbone
    agreed.  A head that was loaded but is fed the wrong state, the wrong position or the wrong
    stream layout still returns confident logits; the only thing that gives it away is that they
    are not the backbone's next tokens.

    The head's cache is *not* the backbone's: it is advanced in lockstep, one position a step, from
    the same token stream, so a draft at position `p` attends over the draft keys of `[0, p)`.

    Baseline: uniform chance is `1 / vocab_size`.  The backbone and the head at the same position
    draft the *same* quantity (the token after next) only when they are asked at the same place,
    so the comparison here is against the following token, not the current one.
    """
    if model.mtp is None:
        raise RuntimeError("mtp_accuracy needs a model built with mtp=True")

    prompt_len = input_ids.shape[1]
    width = max_seq_len or (prompt_len + max_new_tokens + 2)
    cache = model.make_cache(batch_size=input_ids.shape[0], max_seq_len=width)
    mtp_cache = model.mtp.make_cache(input_ids.shape[0], width, model.device, model.dtype)

    with torch.inference_mode():
        logits, streams = None, None
        past = 0
        while past < prompt_len:
            piece = input_ids[:, past : past + chunk_size]
            logits, piece_streams = model.forward(
                piece, cache=cache, past_len=past, return_streams=True
            )
            # Give the head the prompt as history, the same way the draft loop does.  Without it
            # the head answers from a single token's context and its "prediction" is really an
            # echo of what it was handed -- which scores well against the token at the same
            # position and is worth nothing as a draft.
            #
            # The ids are the ones the trunk *consumed* to reach each state, not the ones it went
            # on to produce: position `i`'s state is fused with position `i`'s token to draft
            # `i + 1`.  Handing it `piece[:, 1:]` paired every state with the token it is supposed
            # to be predicting, and a head holding the answer still returns confident logits --
            # measured at 1/23 next-token with that pairing against 0/23 accepted by the draft
            # loop, which is how it was found.
            if piece.shape[1] > 1:
                model.mtp.forward(
                    piece_streams[:, :-1], piece[:, :-1].to(ctx.device), mtp_cache, past_len=past
                )
            streams = piece_streams[:, -1:]
            # The head's key for the chunk's last position is written by the first draft call, so
            # that position is not re-seeded here.  `head_pos` is where that call must place it.
            past += piece.shape[1]

    chosen: list[int] = []
    drafts: list[int] = []
    hits = 0
    #: The next-token quantity a speculative decoder needs -- the same comparison `hits` makes,
    #: spelled out so a reader cannot mistake it for an echo rate.
    next_hits = 0
    #: How often the trunk's chosen token is anywhere in the head's top 5.  On a 256k vocabulary
    #: top-1 alone is noisy; agreement@5 shows whether the head is tracking the trunk's
    #: distribution at all or merely producing well-formed logits.
    top5_hits = 0
    per_step: list[dict[str, int]] = []
    #: How often the *second* draft of a chained round matched the trunk's token two positions
    #: ahead.  Reported apart from `next_hits` because it answers a different question: `next_hits`
    #: is a head that re-derives the trunk's own answer, while this is a head standing in for a
    #: state the trunk never computed.
    depth2_hits = 0
    depth2_total = 0
    depth2_pending = None
    next_id = next_token_from_logits(logits, ctx, batch_size=input_ids.shape[0])

    # The head drafts the successor of the token the trunk consumed at `past - 1`, so it is fused
    # with *that* token's id.  For the first generated token that is the prompt's last id; after
    # that it is the token emitted on the previous step.
    consumed = input_ids[:, -1:].to(ctx.device)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            # `streams` is the trunk's state at position `past - 1` and `consumed` is the token
            # the trunk consumed there, so the head is fed both at the same position and predicts
            # that position's successor -- which is exactly the token this round is about.  Fusing
            # with `next_id` instead hands the head the answer it is meant to guess, and fusing a
            # state and a token from *different* positions costs about an eighth of the accuracy
            # (measured: 47% against 41% on held-out text).
            # Score the *chained* draft first: it is the one that pays, and whether this
            # checkpoint can produce it at all decides if multi-token speculation is worth
            # building.  Step one re-derives the trunk's own next token; step two recycles step
            # one's hidden in place of the state the trunk never computed, which is the
            # extrapolation an EAGLE head is trained for and a single-module MTP may not be.
            if depth2_pending is not None:
                depth2_total += 1
                depth2_hits += int(depth2_pending == int(next_id.reshape(-1)[0]))
                depth2_pending = None
            probe_before = snapshot_cache(mtp_cache)
            step1_mixed, step1_stream = model.mtp.forward(
                streams, consumed, mtp_cache, past_len=past - 1, return_streams=True
            )
            step1_token = torch.tensor(
                [[int(gather_logits(
                    torch.nn.functional.linear(step1_mixed[:, -1], model.lm_head), ctx
                ).argmax())]], dtype=torch.long, device=ctx.device,
            )
            if _mtp_recycle() == "collapsed":
                recycled = step1_mixed.unsqueeze(-2).expand(
                    -1, -1, model.mtp.config.hc_count, -1
                ).reshape_as(step1_stream)
            else:
                recycled = step1_stream
            step2_mixed, _ = model.mtp.forward(
                recycled, step1_token, mtp_cache, past_len=past, return_streams=True
            )
            depth2_pending = int(gather_logits(
                torch.nn.functional.linear(step2_mixed[:, -1], model.lm_head), ctx
            ).argmax())
            restore_cache(probe_before, mtp_cache)
            depth2_total_note = depth2_total

            draft = model.draft_logits(
                streams, consumed, mtp_cache=mtp_cache, past_len=past - 1
            )
            draft = gather_logits(draft[:, -1], ctx) if ctx.pp_size <= 1 else draft
            draft_row = draft.reshape(-1)
            draft_token = int(draft_row.argmax())
            draft_top5 = torch.topk(draft_row, 5).indices.tolist()
            token = int(next_id.reshape(-1)[0])
            chosen.append(token)
            drafts.append(draft_token)
            agree = int(draft_token == token)
            hits += agree
            per_step.append(
                {
                    "position": past,
                    "backbone": token,
                    "mtp": draft_token,
                    "hit": agree,
                    "top5": draft_top5,
                }
            )
            if token in eos_token_ids:
                break
            consumed = next_id.to(ctx.device)
            logits, streams = model.forward(
                consumed, cache=cache, past_len=past, return_streams=True
            )
            past += 1
            next_id = next_token_from_logits(logits, ctx, batch_size=input_ids.shape[0])

    # A draft emitted at step `i` is the head's prediction for the token the trunk chooses *at*
    # that step, because the head is fed that position's state and that position's token.  The two
    # statistics below are therefore the same comparison; `hits` is kept as the name the printout
    # has always used, and `next_hits` spells out that it is the next-token quantity a speculative
    # decoder needs rather than an echo.
    next_hits = hits
    top5_hits = sum(
        1 for step in per_step[: len(chosen)] if step["backbone"] in step["top5"]
    )
    total = len(chosen)
    vocab = int(model.config.vocab_size)
    return {
        "predictions": total,
        "hits": hits,
        "next_hits": next_hits,
        "depth2_hits": depth2_hits,
        "depth2_total": depth2_total,
        "depth2_accuracy": depth2_hits / max(1, depth2_total),
        "next_accuracy": next_hits / max(1, total),
        "next_top5_hits": top5_hits,
        "next_top5_accuracy": top5_hits / max(1, total),
        "accuracy": hits / total if total else 0.0,
        "chance": 1.0 / vocab,
        "vocab_size": vocab,
        "input_mode": getattr(model.mtp, "input_mode", "?"),
        "per_step": per_step if ctx.rank == 0 else [],
    }


def _begin_state_log(cache) -> None:
    """Ask every linear layer to record its state after each position it consumes.

    Off unless a verify round is about to run several positions and may keep only some of them:
    the log costs one state a position (13.8 MiB a rank for all 36 layers), and turning it on also
    asks the recurrence to step a position at a time.
    """
    for layer in cache.layers:
        if layer is not None and hasattr(layer, "state_log"):
            layer.state_log = []


def _end_state_log(cache, past: int) -> None:
    """Close the log and pull this step's speculative tail out of the caches.

    Two things are left behind by a verify forward that ran past the accepted prefix: the per-layer
    KV length, which each QSA layer advanced to `past + K` on its own, and the indexer's raw-key
    rows.  The rows are addressed by position and get written over by whatever consumes them next,
    but a length that is too long would read them, so it comes back to `past`.
    """
    for layer in cache.layers:
        if layer is None:
            continue
        if hasattr(layer, "state_log"):
            layer.state_log = None
        if hasattr(layer, "length"):
            layer.length = past


def _rewind_to(cache, accepted: int) -> None:
    """Put the linear layers back to the state after `accepted` of the logged positions.

    The verify forward consumed every draft in one go, so on anything short of a full accept the
    rolling state is further along than the sequence is.  The log is what makes the shorter
    distance recoverable: entry `k` is the state after position `k`, already computed, so this is
    a copy rather than a replay.
    """
    if accepted <= 0:
        return
    for layer in cache.layers:
        if layer is None:
            continue
        log = getattr(layer, "state_log", None)
        if not log:
            continue
        row = min(accepted, len(log)) - 1
        window, recurrent = log[row]
        # In place: a capture, or a staged copy, holds these buffers' addresses.
        layer.conv_state.copy_(window.to(layer.conv_state.dtype))
        layer.recurrent_state.copy_(recurrent.to(layer.recurrent_state.dtype))
        layer.initialized = True


def _begin_ple_log(cache) -> None:
    """Ask every PLE short convolution to record the window a verify forward convolves.

    The 36 linear layers are not the only per-sequence state a verify advances: the PLE layer's
    short-convolution carry also moves by every position fed, and `_rewind_to` does not reach it.
    Left alone, a partial accept keeps the carry of rejected drafts and every later PLE injection
    mixes them in.
    """
    for state in (getattr(cache, "ple_states", None) or {}).values():
        state.state_log = []


def _rewind_ple_conv(cache, kept: int) -> None:
    """Put each PLE carry back to the window after `kept` positions of the logged verify, then
    close the log."""
    for state in (getattr(cache, "ple_states", None) or {}).values():
        log = getattr(state, "state_log", None)
        state.state_log = None
        if not log or state.conv_state is None:
            continue
        window = log[-1]
        width = int(state.conv_state.shape[-1])
        if kept < 0 or kept + width > window.shape[-1]:
            raise RuntimeError(
                f"PLE rewind to {kept} positions does not fit a {window.shape[-1]}-wide window"
            )
        with torch.inference_mode():
            state.conv_state.copy_(window[..., kept : kept + width].to(state.conv_state.dtype))


def _rewind_ple(cache, snapshot, accepted_ids, ngram: int) -> None:
    """Put the PLE token history back to the accepted prefix.

    `Qwen4ExpModel.forward` advances it once per call, to the tail of whatever it was handed -- so
    a verify forward that consumed K drafts leaves the history K tokens further along than the
    sequence is, and every later step then builds its n-gram window from tokens the trunk never
    chose.  This model hands a 51B n-gram term to all 48 layers, so that does not degrade the
    output, it scrambles it; it also costs draft accuracy, because the head reads the same window.

    The snapshot holds the history from before the verify, so the repair is to append only what
    was accepted.
    """
    if snapshot is None or cache.ple_token_history is None:
        return
    # `snapshot_cache` answers one entry per cache it was handed, and each entry is
    # `(layers, length, ple_history)`; the caller passed one cache, so the history is `[0][2]`.
    frozen = snapshot[0][2] if snapshot[0] is not None else None
    if frozen is None:
        return
    if accepted_ids:
        tail = torch.tensor([list(accepted_ids)], dtype=torch.long)
        merged = torch.cat([frozen.cpu(), tail], dim=1)
    else:
        merged = frozen.cpu()
    cache.ple_token_history = merged[:, -(ngram - 1) :].clone()


def _mtp_head_table() -> bool:
    """Whether the lookahead head's cache carries its absolute RoPE table (`attach_rope_table`).

    On by default: without it the head's indexer rotates pooled blocks with the wrong phase and
    rebuilds its block pool every call. `POCKETLLM_QWEN4_MTP_HEAD_TABLE=0` restores the old
    behaviour, to measure the draft acceptance against.
    """
    return os.environ.get("POCKETLLM_QWEN4_MTP_HEAD_TABLE", "1") != "0"


def _mtp_recycle() -> str:
    """Which hidden the next draft link is conditioned on.

    `stream` recycles the four-stream stack the head's decoder layer ends on; `collapsed`
    re-broadcasts the post-final-norm vector, which is what the reference implementation does.
    Only the second draft of a round distinguishes them, and it turned out to decide whether the
    chain predicts at all.

    Traced on this checkpoint at a 2048 position prompt, three drafts a round.  Under `stream`
    every link after the first repeats the first -- `drafts=[3841, 3841, 3841]` against a trunk that
    chose `[3841, 13477, 3841]` -- because feeding the head its own four-stream output leaves it
    echoing the token it was just handed.  Under `collapsed` the chain predicts:
    `drafts=[3841, 4087, 13]`, and over 32 tokens it needs 26 trunk steps where `stream` needs 32.
    The acceptance is 1.23 tokens a step rather than 1.00.

    `stream` also produced the wrong text: its sixth token was 13477 where the plain decode (and
    `collapsed`) give 37550.  So the default is `collapsed`, and the switch stays for pricing.
    """
    value = os.environ.get("POCKETLLM_QWEN4_MTP_RECYCLE", "collapsed").lower()
    return value if value in {"stream", "collapsed"} else "collapsed"


def _set_host_positions(cache, position: int) -> None:
    """Tell a host-resident arena where the next write lands.

    The integer is what makes the write a slice instead of a device-indexed `index_copy_`; taking
    it from the position tensor at the point of use would be a host read of a device value, which a
    capture cannot contain.  Both drivers know it already.
    """
    if cache is None:
        return
    for layer in cache.layers:
        if layer is not None and getattr(layer, "host_kv", False):
            layer.host_position = int(position)


def _maybe_kernel_profile(ctx, fn):
    """Run `fn` under a CUDA kernel profile when asked, and print what the step is made of.

    The hierarchical profiler answers "which phase" by timing scopes on the host; it cannot say
    which *kernel* a phase spends on, and after a capture the host timeline is not where the time
    is.  This is the other half: Torch's own profiler, keyed by kernel name, so a fusion target is
    chosen from counts and device time instead of from an assumption.
    """
    if os.environ.get("POCKETLLM_QWEN4_KERNEL_PROFILE") != "1":
        return fn()
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as prof:
        out = fn()
    if ctx.rank == 0:
        keyed = prof.key_averages()
        total = sum(entry.count for entry in keyed)
        print("\n=== kernels by device time ===", flush=True)
        print(keyed.table(sort_by="cuda_time_total", row_limit=25), flush=True)
        print(f"\n=== {total} launches, by call count ===", flush=True)
        for entry in sorted(keyed, key=lambda e: -e.count)[:25]:
            print(f"  {entry.count:7d}  {entry.key[:70]}", flush=True)
    return out


def _trunk_step(window, model, ctx, cache, token: torch.Tensor, past: int):
    """One trunk step at `past`: replayed from the capture when it covers this position.

    `model.forward` leaves the state in `cache` and hands back the streams; a replay writes the
    same cache through `state_slots` and returns only the chosen token, so the streams come from
    the cache the recording just wrote.  Both paths are `(logits, streams)` to the caller, which
    is what lets the draft loop run on either without knowing which it got -- and a replay that
    declines (the near-tie margin) simply falls through to the eager step for that position.
    """
    if window is not None:
        replayed = window.replay(past, token.to(ctx.device))
        if replayed is not None:
            streams = window.streams()
            # `to()` already makes a new tensor when the dtype differs, but a capture's buffer is
            # overwritten by the next replay -- so the copy is taken here rather than left to rest
            # on the dtypes differing.  The head keeps this across a round that may replay the same
            # position again.
            return replayed, (streams.to(model.dtype).clone()
                              if streams is not None else None)
    # A capture's player takes a flat token; `model.forward` wants `[b, seq]`.  Callers hand both
    # shapes (a decode step has one row, a speculative verify has several), so the widening is
    # done here once rather than at each call site.
    if token.dim() == 1:
        token = token.unsqueeze(0)
    _set_host_positions(cache, past)
    logits, streams = model.forward(token, cache=cache, past_len=past, return_streams=True)
    return logits, streams


def generate_mtp_draft(
    model: Qwen4ExpModel,
    ctx: TPContext,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    chunk_size: int = 512,
    eos_token_ids: tuple[int, ...] = (),
    on_token=None,
    graph_window: int = 0,
    draft_tokens: int = 1,
    draft_p_min: float = 0.0,
) -> tuple[list[int], dict[str, float]]:
    """Greedy decode that drafts several tokens per round with the MTP head and verifies them.

    One round is: the backbone answers position `p` and proposes `t`; the MTP head, fed the trunk's
    state at `p` plus the embedding of `t`, drafts `d` for `p + 1`; the backbone then verifies `d`
    by running it.  If its own answer for `p + 1` is also `d`, the round accepted two tokens for one
    backbone step instead of one.  If it disagrees, the draft is discarded and the backbone's token
    is kept -- which is the token a plain decode would have produced, so the *output is identical
    to `generate`*, only sometimes produced faster.

    Rolling back is the delicate part, and it is why `snapshot_cache`/`restore_cache` exist.  A
    rejected draft has already run the whole stack on `d`: it wrote a QSA row at `p + 1` and
    advanced the recurrence on the 36 GatedDeltaNet layers.  The QSA rows are addressed by position,
    so the next write covers them; the linear layers' state is the *result* of the step, so it is
    restored from a snapshot taken just before the draft ran.  Without the restore the model would
    have "seen" a token its history does not contain and would drift from the unspeculative chain
    within a few steps -- which is exactly what `--mtp-draft` is checked against below.

    The head's own cache is threaded separately and advances in step with the backbone's: a draft
    for `p + 1` is conditioned on the token at `p + 1` only *after* that token is settled, so the
    head's key for a position is written when the position's token becomes final.  That is why the
    accept and reject paths both end by feeding the head the same pair -- `(state at p, token p)` --
    and why the rejected path has to rewind the head too.

    Scope: batch 1, no pipeline.  A pipeline stage does not own the head, so that one is a real
    constraint rather than a simplification to restore later.

    `graph_window` replays the trunk's step from a recording instead of running it, which matters
    here more than anywhere else: every round costs a trunk step, and on a host whose per-launch
    cost is high that step is the whole budget.  The recording takes `(position, token)`, which is
    exactly what a speculative round needs -- the position is the one being written and the token
    is either the trunk's own or the draft under test -- so a rejected round simply replays over
    the same positions again.  The head is not captured: it runs once a round with shapes the
    capture did not see, and it is one layer against forty-eight.
    """
    if model.mtp is None:
        raise RuntimeError("generate_mtp_draft needs a model built with mtp=True")
    if graph_window > 0 and os.environ.get("POCKETLLM_QWEN4_MTP_GRAPH", "0") != "1":
        # Not a limitation of the idea, a state of the code: the capture is built and the first
        # replay faults with an illegal memory access, because the step a draft round replays is
        # not the step `graph_window` knows how to record -- it has to write K rows at once and
        # hand back the per-position recurrent state a partial accept rewinds to.
        #
        # The second of those now exists: the fused gated-delta scan has a variant that returns the
        # state after every position, verified equal to a shorter scan from the same start, so a
        # rewind has the values it needs without a replay. Whether that is enough to make the
        # recording legal is what POCKETLLM_QWEN4_MTP_GRAPH=1 asks, by letting the capture be
        # attempted instead of refused.
        raise RuntimeError(
            "--mtp-draft cannot be combined with --graph-window yet: the verify step's capture "
            "faults on its first replay.  Run the draft eagerly (correct, break-even), or the "
            "plain captured decode without --mtp-draft."
        )
    if ctx.pp_size > 1:
        raise RuntimeError("--mtp-draft is not supported under pipeline parallelism")
    if int(input_ids.shape[0]) != 1:
        raise RuntimeError("--mtp-draft decodes one sequence at a time")

    prompt_len = input_ids.shape[1]
    width = prompt_len + max_new_tokens + 2
    cache = model.make_cache(batch_size=1, max_seq_len=width)
    # A multi-token round drafts a chain of tokens, and every link has to attend over the
    # backbone's keys for the positions before it.  A private arena cannot show it those, so the
    # head borrows the backbone's; one token a round does not need it and keeps its own.
    mtp_cache = model.mtp.make_cache(1, width, model.device, model.dtype)
    prof_scope = model.profiler.scope if model.profiler else lambda _name: nullcontext()

    t0 = time.perf_counter()
    with torch.inference_mode(), prof_scope("prefill"):
        past = 0
        logits = streams = None
        while past < prompt_len:
            piece = input_ids[:, past : past + chunk_size]
            # The prefill keeps this chunk's whole stream, not just the last position: seeding the
            # head needs every prompt position's trunk state, and the last one is what the first
            # round drafts from.
            logits, piece_streams = model.forward(
                piece, cache=cache, past_len=past, return_streams=True
            )
            # Seed the head as the prompt arrives, so the first draft attends over real history
            # instead of an empty cache.  Head position `i` is conditioned on the token the trunk
            # consumed there, so the states and the ids are paired index for index -- not shifted.
            if piece.shape[1] > 1:
                model.mtp.forward(
                    piece_streams[:, :-1],
                    piece[:, :-1].to(ctx.device),
                    mtp_cache,
                    past_len=past,
                )
            # Only the last position is carried into the decode: it is the state the next token is
            # drafted from, and keeping the whole prompt would hold a `[1, prompt, hc * hidden]`
            # copy of every chunk for the rest of the generation.
            streams = piece_streams[:, -1:]
            past += piece.shape[1]
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    # The first token the trunk chose, from the last prompt position's logits.  The loop below
    # starts by emitting whatever `next_id` holds, so it has to be set here rather than being
    # carried in from a chunk that has already gone out of scope.
    next_id = next_token_from_logits(logits, ctx, batch_size=1)

    # A capture, when asked for.  Built after the prefill so the window starts at the first
    # generated position.  `trunk_step` then runs or replays each step by position, which is all a
    # speculative round needs: a rejected draft is simply replayed over the same positions again.
    window = None
    if graph_window > 0 and ctx.device.type == "cuda":
        from relicllm.models.qwen4_exp.graph_window import build

        window = build(
            model=model,
            cache=cache,
            start=past,
            steps=min(int(graph_window), max(1, max_new_tokens - 1)),
            token=next_id,
            wants_streams=True,
            # Without this the replay answers with the argmax of *this rank's shard*, which is not
            # the token the model chose: the vocabulary is column-sharded, so every rank would
            # return a different token and the loop would follow one of them.  The gather is a
            # collective, which is why it lives outside the capture.
            select=lambda logits: next_token_from_logits(logits, ctx, batch_size=1),
        )
        if window is not None and ctx.tp_world_size > 1:
            # The gather runs between replays, so its output cannot be allocated lazily: an
            # allocation there can be handed memory the recording owns, and the next replay then
            # writes through an address the graph baked.  Sized here, once, before the capture.
            width = window.logits_width()
            ctx.gather_parts = [
                torch.empty(1, width, dtype=torch.float32, device=ctx.device)
                for _ in range(ctx.tp_world_size)
            ]
            ctx.argmax_parts = [
                torch.empty(1, 2, dtype=torch.float32, device=ctx.device)
                for _ in range(ctx.tp_world_size)
            ]
        if ctx.rank == 0:
            print(
                f"[qwen4exp] mtp draft: capture window from {past}: "
                f"{'built' if window is not None else 'declined, staying eager'}",
                flush=True,
            )

    produced: list[int] = []
    rounds = accepted = rejected = 0
    trunk_steps = 0
    t1 = time.perf_counter()
    #: The token the trunk consumed to reach `streams` -- what the head is conditioned on.  The
    #: prompt's last id, because the prefill left the trunk at the first generated position.
    consumed = input_ids[:, -1:].to(ctx.device)

    with torch.inference_mode(), prof_scope("decode"):
        while len(produced) < max_new_tokens:
            token = int(next_id.reshape(-1)[0])
            produced.append(token)
            if on_token is not None and ctx.rank == 0:
                on_token(token)
            if token in eos_token_ids or len(produced) >= max_new_tokens:
                break

            # The trunk consumes the token it chose.  This is the one forward every round must
            # make whatever happens next: it produces both the next token and the state the head
            # drafts from.
            consumed = next_id.to(ctx.device)
            logits, streams = model.forward(
                consumed, cache=cache, past_len=past, return_streams=True
            )
            trunk_steps += 1
            past += 1
            next_id = next_token_from_logits(logits, ctx)

            # -- draft -- a chain of head calls, no trunk forward between them.
            #
            # This is the shape MTP speculation is built for and the reason a one-token round
            # cannot win: the head's own output *is* the state for the next draft step, so K
            # candidates cost K-1 head passes -- each a single layer -- against a trunk step that
            # would otherwise cost a full forward.
            head_before = snapshot_cache(mtp_cache)
            trunk_before = None if window is not None else snapshot_cache(cache)
            drafts: list[int] = []
            head_stack = streams
            head_tok = consumed
            head_at = past - 1
            for _ in range(draft_tokens):
                if len(produced) + len(drafts) >= max_new_tokens:
                    break
                mixed, step_stream = model.mtp.forward(
                    head_stack, head_tok, mtp_cache, past_len=head_at, return_streams=True
                )
                # What the next link is conditioned on.  The reference implementation recycles
                # the *post-final-norm* hidden -- one collapsed vector -- rather than the four
                # streams this layer ends on, so the two conventions are a one-line difference and
                # only the second draft onward can tell them apart.  `stream` (the default) keeps
                # the streams; `collapsed` broadcasts the mixed vector back to `hc_count` rows.
                if _mtp_recycle() == "collapsed":
                    head_stack = mixed.unsqueeze(-2).expand(
                        -1, -1, model.mtp.config.hc_count, -1
                    ).reshape_as(step_stream)
                else:
                    head_stack = step_stream
                step_logits = gather_logits(F.linear(mixed[:, -1], model.lm_head), ctx)
                drafted = int(step_logits.argmax(-1).reshape(-1)[0])
                # Stop drafting when the head is not sure of this one.  The chain is
                # autoregressive, so an unsure link makes every later link worse -- and a link
                # that is going to be rejected anyway still has to be verified by a trunk row.
                # Confidence is the top-1 probability, which for a 256k vocabulary separates
                # "the head knows" from "the head is guessing" far better than a logit margin.
                confidence = float(step_logits.softmax(-1).max()) if draft_p_min > 0 else 1.0
                if os.environ.get("POCKETLLM_QWEN4_MTP_TRACE") == "1" and ctx.rank == 0:
                    print(f"[chain] at={head_at} tok={int(head_tok.reshape(-1)[0])} "
                          f"-> {drafted} p={confidence:.3f}", flush=True)
                if draft_p_min > 0 and confidence < draft_p_min:
                    break
                drafts.append(drafted)
                head_tok = _ids([drafted], ctx)
                head_at += 1
            rounds += 1
            if not drafts:
                continue

            # -- verify -- one trunk forward over every draft position at once.
            #
            # Each position's answer is compared against the draft for *that* position, and the
            # accepted run is the prefix that matches; the first mismatch ends it, because the
            # position after an accepted token is only known once the token itself is accepted.
            # -- verify -- every draft in one trunk forward, then keep the prefix that matched.
            #
            # The feed is the draft chain itself at the positions it was drafted for: row `i`
            # consumes `drafts[i]` at `past + i` and answers `past + i + 1`.  So a draft is checked
            # against the row *before* it, except the first, which is checked against `next_id` --
            # the answer the trunk already gave for `past`.
            verify_tokens = _ids(drafts, ctx)
            _begin_state_log(cache)
            verify_logits, _ = model.forward(
                verify_tokens, cache=cache, past_len=past, return_streams=True
            )
            trunk_steps += 1
            rows = gather_logits(verify_logits[0], ctx).argmax(-1).tolist()
            chosen = [int(next_id.reshape(-1)[0])] + rows
            accepted_run = 0
            for drafted, got in zip(drafts, chosen):
                if drafted != got:
                    break
                accepted_run += 1
            if os.environ.get("POCKETLLM_QWEN4_MTP_TRACE") == "1" and ctx.rank == 0:
                print(f"[trace] past={past} drafts={drafts} chosen={chosen[:len(drafts)]} "
                      f"accepted={accepted_run}", flush=True)

            # The verify forward consumed all of them, so the recurrence is as far as `past + K - 1`
            # whatever was accepted.  The log is what makes the shorter distance recoverable.
            _rewind_to(cache, accepted_run)
            _end_state_log(cache, past + accepted_run)
            _rewind_ple(
                cache, trunk_before, drafts[:accepted_run], model.config.ngram_size
            )
            if accepted_run == 0:
                rejected += 1
                restore_cache(head_before, mtp_cache)
                # The rows this forward wrote are the drafts, and the sequence keeps none of them;
                # the next round writes the position the trunk chooses over the top of them.
                continue

            accepted += 1
            for drafted in drafts[:accepted_run]:
                produced.append(drafted)
                if on_token is not None and ctx.rank == 0:
                    on_token(drafted)
            if produced and produced[-1] in eos_token_ids:
                break
            if len(produced) >= max_new_tokens:
                break
            past += accepted_run
            # The token after the accepted prefix is the answer for position `past + accepted_run`,
            # which is row `accepted_run` -- present because the feed ran one row past the last
            # draft it had to judge.
            next_id = torch.tensor([[chosen[accepted_run]]], dtype=torch.long, device=ctx.device)
            cache.length = past

    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    decode_s = time.perf_counter() - t1

    stats = {
        "prompt_tokens": float(prompt_len),
        "generated_tokens": float(len(produced)),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "ttft_s": prefill_s,
        "stopped": "length" if len(produced) >= max_new_tokens else "eos",
        "decode_tps": len(produced) / decode_s if decode_s > 0 else 0.0,
        "draft_rounds": float(rounds),
        "draft_accepted": float(accepted),
        "draft_rejected": float(rejected),
        "draft_accept_rate": accepted / rounds if rounds else 0.0,
        "trunk_steps": float(trunk_steps),
        # Tokens a trunk forward, against 1.0 for an unspeculated decode: the ratio the accept
        # rate buys, and the number that says whether the round was worth taking.
        "tokens_per_trunk_step": len(produced) / max(1, trunk_steps),
    }
    return produced, stats


def generate_mtp_draft_onestep(
    model: Qwen4ExpModel,
    ctx: TPContext,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    chunk_size: int = 512,
    eos_token_ids: tuple[int, ...] = (),
    on_token=None,
    graph_window: int = 0,
    draft_tokens: int = 1,
    draft_p_min: float = 0.0,
) -> tuple[list[int], dict[str, float]]:
    """Greedy decode that drafts several tokens per round with the MTP head and verifies them.

    One round is: the backbone answers position `p` and proposes `t`; the MTP head, fed the trunk's
    state at `p` plus the embedding of `t`, drafts `d` for `p + 1`; the backbone then verifies `d`
    by running it.  If its own answer for `p + 1` is also `d`, the round accepted two tokens for one
    backbone step instead of one.  If it disagrees, the draft is discarded and the backbone's token
    is kept -- which is the token a plain decode would have produced, so the *output is identical
    to `generate`*, only sometimes produced faster.

    Rolling back is the delicate part, and it is why `snapshot_cache`/`restore_cache` exist.  A
    rejected draft has already run the whole stack on `d`: it wrote a QSA row at `p + 1` and
    advanced the recurrence on the 36 GatedDeltaNet layers.  The QSA rows are addressed by position,
    so the next write covers them; the linear layers' state is the *result* of the step, so it is
    restored from a snapshot taken just before the draft ran.  Without the restore the model would
    have "seen" a token its history does not contain and would drift from the unspeculative chain
    within a few steps -- which is exactly what `--mtp-draft` is checked against below.

    The head's own cache is threaded separately and advances in step with the backbone's: a draft
    for `p + 1` is conditioned on the token at `p + 1` only *after* that token is settled, so the
    head's key for a position is written when the position's token becomes final.  That is why the
    accept and reject paths both end by feeding the head the same pair -- `(state at p, token p)` --
    and why the rejected path has to rewind the head too.

    Scope: batch 1, no pipeline.  A pipeline stage does not own the head, so that one is a real
    constraint rather than a simplification to restore later.

    `graph_window` replays the trunk's step from a recording instead of running it, which matters
    here more than anywhere else: every round costs a trunk step, and on a host whose per-launch
    cost is high that step is the whole budget.  The recording takes `(position, token)`, which is
    exactly what a speculative round needs -- the position is the one being written and the token
    is either the trunk's own or the draft under test -- so a rejected round simply replays over
    the same positions again.  The head is not captured: it runs once a round with shapes the
    capture did not see, and it is one layer against forty-eight.
    """
    if model.mtp is None:
        raise RuntimeError("generate_mtp_draft needs a model built with mtp=True")
    if graph_window > 0 and os.environ.get("POCKETLLM_QWEN4_MTP_GRAPH", "0") != "1":
        # Not a limitation of the idea, a state of the code: the capture is built and the first
        # replay faults with an illegal memory access, because the step a draft round replays is
        # not the step `graph_window` knows how to record -- it has to write K rows at once and
        # hand back the per-position recurrent state a partial accept rewinds to.
        #
        # The second of those now exists: the fused gated-delta scan has a variant that returns the
        # state after every position, verified equal to a shorter scan from the same start, so a
        # rewind has the values it needs without a replay. Whether that is enough to make the
        # recording legal is what POCKETLLM_QWEN4_MTP_GRAPH=1 asks, by letting the capture be
        # attempted instead of refused.
        raise RuntimeError(
            "--mtp-draft cannot be combined with --graph-window yet: the verify step's capture "
            "faults on its first replay.  Run the draft eagerly (correct, break-even), or the "
            "plain captured decode without --mtp-draft."
        )
    if ctx.pp_size > 1:
        raise RuntimeError("--mtp-draft is not supported under pipeline parallelism")
    if int(input_ids.shape[0]) != 1:
        raise RuntimeError("--mtp-draft decodes one sequence at a time")

    prompt_len = input_ids.shape[1]
    width = prompt_len + max_new_tokens + 2
    cache = model.make_cache(batch_size=1, max_seq_len=width)
    # A multi-token round drafts a chain of tokens, and every link has to attend over the
    # backbone's keys for the positions before it.  A private arena cannot show it those, so the
    # head borrows the backbone's; one token a round does not need it and keeps its own.
    mtp_cache = model.mtp.make_cache(1, width, model.device, model.dtype)
    prof_scope = model.profiler.scope if model.profiler else lambda _name: nullcontext()

    t0 = time.perf_counter()
    with torch.inference_mode(), prof_scope("prefill"):
        past = 0
        logits = streams = None
        while past < prompt_len:
            piece = input_ids[:, past : past + chunk_size]
            # The prefill keeps this chunk's whole stream, not just the last position: seeding the
            # head needs every prompt position's trunk state, and the last one is what the first
            # round drafts from.
            logits, piece_streams = model.forward(
                piece, cache=cache, past_len=past, return_streams=True
            )
            # Seed the head as the prompt arrives, so the first draft attends over real history
            # instead of an empty cache.  Head position `i` is conditioned on the token the trunk
            # consumed there, so the states and the ids are paired index for index -- not shifted.
            if piece.shape[1] > 1:
                model.mtp.forward(
                    piece_streams[:, :-1],
                    piece[:, :-1].to(ctx.device),
                    mtp_cache,
                    past_len=past,
                )
            # Only the last position is carried into the decode: it is the state the next token is
            # drafted from, and keeping the whole prompt would hold a `[1, prompt, hc * hidden]`
            # copy of every chunk for the rest of the generation.
            streams = piece_streams[:, -1:]
            past += piece.shape[1]
    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    prefill_s = time.perf_counter() - t0

    # The first token the trunk chose, from the last prompt position's logits.  The loop below
    # starts by emitting whatever `next_id` holds, so it has to be set here rather than being
    # carried in from a chunk that has already gone out of scope.
    next_id = next_token_from_logits(logits, ctx, batch_size=1)

    # A capture, when asked for.  Built after the prefill so the window starts at the first
    # generated position.  `trunk_step` then runs or replays each step by position, which is all a
    # speculative round needs: a rejected draft is simply replayed over the same positions again.
    window = None
    if graph_window > 0 and ctx.device.type == "cuda":
        from relicllm.models.qwen4_exp.graph_window import build

        window = build(
            model=model,
            cache=cache,
            start=past,
            steps=min(int(graph_window), max(1, max_new_tokens - 1)),
            token=next_id,
            wants_streams=True,
            # Without this the replay answers with the argmax of *this rank's shard*, which is not
            # the token the model chose: the vocabulary is column-sharded, so every rank would
            # return a different token and the loop would follow one of them.  The gather is a
            # collective, which is why it lives outside the capture.
            select=lambda logits: next_token_from_logits(logits, ctx, batch_size=1),
        )
        if window is not None and ctx.tp_world_size > 1:
            # The gather runs between replays, so its output cannot be allocated lazily: an
            # allocation there can be handed memory the recording owns, and the next replay then
            # writes through an address the graph baked.  Sized here, once, before the capture.
            width = window.logits_width()
            ctx.gather_parts = [
                torch.empty(1, width, dtype=torch.float32, device=ctx.device)
                for _ in range(ctx.tp_world_size)
            ]
            ctx.argmax_parts = [
                torch.empty(1, 2, dtype=torch.float32, device=ctx.device)
                for _ in range(ctx.tp_world_size)
            ]
        if ctx.rank == 0:
            print(
                f"[qwen4exp] mtp draft: capture window from {past}: "
                f"{'built' if window is not None else 'declined, staying eager'}",
                flush=True,
            )

    produced: list[int] = []
    rounds = accepted = rejected = 0
    trunk_steps = 0
    #: Rounds where the head, asked for the token that was just emitted, did not answer it. A
    #: non-zero count means the carried drafting state is not where this loop thinks it is.
    head_miss = 0
    t1 = time.perf_counter()
    #: The token the trunk consumed to reach `streams` -- what the head is conditioned on.  The
    #: prompt's last id, because the prefill left the trunk at the first generated position.
    consumed = input_ids[:, -1:].to(ctx.device)

    with torch.inference_mode(), prof_scope("decode"):
        # One trunk step a round.
        #
        # The token this loop emits is the trunk's own answer, carried over from the round before, so
        # the verify pass consumes it *together with* the drafts instead of a separate forward
        # consuming it alone.  That is the difference that matters: the old loop spent two trunk
        # steps to emit `1 + accepted` tokens, which caps tokens-per-step at `(1 + drafts) / 2` and
        # reaches 2.0 only at perfect acceptance; this one spends one step for `1 + accepted`, so the
        # same accept rates give roughly twice the tokens a step and three drafts at a 0.75 accept
        # rate lands near the objective's 3.
        head_stack = streams
        head_tok = consumed
        head_at = past - 1
        while len(produced) < max_new_tokens:
            token = int(next_id.reshape(-1)[0])
            produced.append(token)
            if on_token is not None and ctx.rank == 0:
                on_token(token)
            if token in eos_token_ids or len(produced) >= max_new_tokens:
                break

            # -- draft -- from the state the previous round's verify left behind, a chain of head
            # calls with no trunk forward between them.
            head_before = snapshot_cache(mtp_cache)
            trunk_before = None if window is not None else snapshot_cache(cache)
            drafts: list[int] = []
            stack = head_stack
            tok = head_tok
            at = head_at
            # The first head output predicts the position of the token just emitted. That token is
            # already known, so the output is a check on the carried state rather than a draft -- and
            # being one position earlier than the trivial draft is what makes a single trunk step per
            # round possible: the feed below carries the emitted token and the drafts together, so
            # the trunk consumes the token and judges the drafts in the same pass.
            check = None
            for step in range(draft_tokens + 1):
                if len(produced) + len(drafts) >= max_new_tokens:
                    break
                mixed, step_stream = model.mtp.forward(
                    stack, tok, mtp_cache, past_len=at, return_streams=True
                )
                if _mtp_recycle() == "collapsed":
                    stack = mixed.unsqueeze(-2).expand(
                        -1, -1, model.mtp.config.hc_count, -1
                    ).reshape_as(step_stream)
                else:
                    stack = step_stream
                step_logits = gather_logits(F.linear(mixed[:, -1], model.lm_head), ctx)
                drafted = int(step_logits.argmax(-1).reshape(-1)[0])
                confidence = float(step_logits.softmax(-1).max()) if draft_p_min > 0 else 1.0
                if step == 0:
                    check = drafted
                    tok = _ids([drafted], ctx)
                    at += 1
                    continue
                if draft_p_min > 0 and confidence < draft_p_min:
                    break
                drafts.append(drafted)
                tok = _ids([drafted], ctx)
                at += 1
            if check is None or check != token:
                head_miss += 1
            rounds += 1

            # -- verify -- the emitted token and every draft in a single forward.
            #
            # Row `i` consumes `feed[i]` and answers for the position after it, so draft `j` is
            # judged by row `j - 1` and the answer to the last accepted draft is the bonus token.
            # That bonus is what the next round emits, without a forward of its own.
            feed = [token] + drafts
            _begin_state_log(cache)
            verify_logits, verify_streams = model.forward(
                _ids(feed, ctx), cache=cache, past_len=past, return_streams=True
            )
            trunk_steps += 1
            rows = gather_logits(verify_logits[0], ctx).argmax(-1).tolist()
            accepted_run = 0
            for drafted, got in zip(drafts, rows):
                if drafted != got:
                    break
                accepted_run += 1
            if os.environ.get("POCKETLLM_QWEN4_MTP_TRACE") == "1" and ctx.rank == 0:
                print(f"[trace-1step] past={past} feed={feed} rows={rows} "
                      f"accepted={accepted_run}", flush=True)

            # The forward consumed `1 + len(drafts)` positions; the sequence keeps `1 + accepted`.
            kept = 1 + accepted_run
            _rewind_to(cache, kept)
            _end_state_log(cache, past + kept)
            _rewind_ple(cache, trunk_before, feed[1:kept], model.config.ngram_size)
            if accepted_run == 0:
                rejected += 1
                restore_cache(head_before, mtp_cache)
            else:
                accepted += 1
            for drafted in drafts[:accepted_run]:
                produced.append(drafted)
                if on_token is not None and ctx.rank == 0:
                    on_token(drafted)
            past += kept
            cache.length = past
            # The next token is the row past the accepted prefix, and the state the next round
            # drafts from is that row's -- one trunk step, not two.
            next_id = torch.tensor([[rows[accepted_run]]], dtype=torch.long, device=ctx.device)
            head_stack = verify_streams[:, accepted_run : accepted_run + 1]
            head_tok = _ids([feed[accepted_run]], ctx)
            head_at = past - 1
            if produced and produced[-1] in eos_token_ids:
                break
            if len(produced) >= max_new_tokens:
                break

    if ctx.device.type == "cuda":
        torch.cuda.synchronize()
    decode_s = time.perf_counter() - t1

    stats = {
        "prompt_tokens": float(prompt_len),
        "generated_tokens": float(len(produced)),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "ttft_s": prefill_s,
        "stopped": "length" if len(produced) >= max_new_tokens else "eos",
        "decode_tps": len(produced) / decode_s if decode_s > 0 else 0.0,
        "draft_rounds": float(rounds),
        "draft_accepted": float(accepted),
        "draft_rejected": float(rejected),
        "draft_accept_rate": accepted / rounds if rounds else 0.0,
        "trunk_steps": float(trunk_steps),
        # Tokens a trunk forward, against 1.0 for an unspeculated decode: the ratio the accept
        # rate buys, and the number that says whether the round was worth taking.
        "tokens_per_trunk_step": len(produced) / max(1, trunk_steps),
        "head_state_misses": float(head_miss),
    }
    return produced, stats


def _mtp_session_group_count(ctx, ready: bool) -> int:
    """All TP ranks must agree before one can recapture, release, or replay a retained arena."""
    import torch.distributed as dist
    if ctx.tp_world_size <= 1 or not dist.is_initialized():
        return int(ready)
    vote = torch.tensor([int(ready)], device=ctx.device, dtype=torch.int32)
    dist.all_reduce(vote, op=dist.ReduceOp.SUM)
    return int(vote.item())


def _guard_graph_session_request(operation):
    """A failed prefill must return its reusable request lease before setup's finalizer exists."""
    from functools import wraps

    @wraps(operation)
    def guarded(*args, **kwargs):
        session = kwargs.get("graph_session")
        previous = getattr(session, "_active_lease", None) if session is not None else None
        try:
            return operation(*args, **kwargs)
        except BaseException:
            if session is not None and getattr(session, "_active_lease", None) is not previous:
                session.abandon_active()
            raise
    return guarded


@_guard_graph_session_request
def generate_mtp_draft_lookahead(
    model: Qwen4ExpModel,
    ctx: TPContext,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int = 32,
    chunk_size: int = 512,
    eos_token_ids: tuple[int, ...] = (),
    on_token=None,
    graph_window: int = 0,
    draft_tokens: int = 1,
    draft_p_min: float = 0.0,
    verify_graph: bool = False,
    on_step=None,
    local_eos: list[bool] | None = None,
    prefilled_state=None,
    on_prefill=None,
    graph_session=None,
) -> tuple[list[int], dict[str, float]]:
    """Greedy decode with the standard MTP convention: head(h[i], t[i+1]) predicts t[i+2].

    The head's pairing is shifted by one from ``generate_mtp_draft``: the trunk state at
    position ``i`` is fused with the *next* token's embedding (``t[i+1]``), not the token
    consumed at ``i``.  A probe measured this convention at **85.4%** first-draft acceptance
    against 70.2% for the un-shifted one, matching DeepSeek-V3 MTP acceptance ranges.

    Each round: the trunk verifies the emitted token plus K drafts in one forward; the
    head's catch-up call (over the accepted verify rows plus the bonus token) both seeds the
    head cache correctly for the accepted positions and produces the first draft of the next
    round, so no head pass is wasted on re-deriving a known token.

    `decode_s` / `decode_s_excluding_setup` retain the steady-state interval after initial graph
    construction; `decode_wall_s` adds setup and the common-start alignment back. Both rates count only delivered
    tokens after the prefill-produced first token (`decode_tokens`). TTFT ends at the first
    host-visible token, before its callback, and includes graph construction when one is built.
    `drafts_accepted` counts delivered drafts; `drafts_accepted_verified` also counts any accepted
    fixed-width padding beyond the output's EOS/budget boundary. Prefix histograms are grouped by
    the actual number of drafts proposed in a round, including confidence-shortened rounds. The
    position counters use zero-based draft positions and distinguish verified from delivered
    survival without any additional device reads. Head-prefix histograms count only catch-ups
    actually executed, with index zero unused; the replay histogram is their captured subset.

    `on_step` and `local_eos` are the same cooperative-cancellation pair `generate` takes, and the
    semantics are the same: the predicate is asked on *every* rank, so a rank cannot leave the loop
    on a local decision while its peers enter a collective.  Two differences follow from a round
    emitting up to `draft_tokens + 1` tokens at once:

    * The predicate is asked at round boundaries -- before the token a round has already produced is
      emitted, again right after that emission, and once more after the accepted prefix's callbacks
      but before the head catch-up -- so a cancellation can trail by at most one round's emitted
      tokens (`draft_tokens + 1`) rather than one token.  Those three sites are reached by every
      rank unconditionally; none is gated on data a rank computes for itself, which is what keeps
      the poll counts equal across ranks.
    * A token in a multi-token round may be end-of-sequence before the next poll.  With
      `local_eos`, EOS sets the flag instead of breaking, and the next poll -- which the caller's
      predicate makes a collective -- names the exit, so every rank leaves together.  Without the
      flag the old local exit stands, which is correct at world size 1 and is what the CLI uses.

    Cancelling at a round boundary leaves the trunk cache at the committed accepted prefix: the
    verified round has already been rewound to it and the logical lengths are advanced before the
    last poll, so a caller that only tears the request down sees a coherent cache.  Speculative
    head state for a round that will not run is discarded rather than kept stale.
    """
    if max_new_tokens <= 0:
        return [], {
            "prompt_tokens": float(input_ids.shape[1]), "generated_tokens": 0.0,
            "prefill_first_tokens": 0.0, "decode_tokens": 0.0,
            "prefill_s": 0.0, "decode_s": 0.0, "decode_s_excluding_setup": 0.0,
            "decode_wall_s": 0.0, "ttft_s": 0.0, "stopped": "length",
            "decode_tps": 0.0, "decode_tps_excluding_setup": 0.0, "decode_wall_tps": 0.0,
            "draft_rounds": 0.0, "draft_accepted": 0.0, "draft_rejected": 0.0,
            "draft_accept_rate": 0.0, "drafts_proposed": 0.0, "drafts_accepted": 0.0,
            "drafts_accepted_verified": 0.0, "per_draft_accept": 0.0,
            "per_draft_accept_verified": 0.0,
            "accepted_prefix_histogram_verified_by_width": {},
            "accepted_prefix_histogram_delivered_by_width": {},
            "draft_position_proposed": [], "draft_position_survived_verified": [],
            "draft_position_survived_delivered": [],
            "head_kept_prefix_histogram": [], "head_replay_kept_prefix_histogram": [],
            "trunk_steps": 0.0, "head_calls": 0.0, "tokens_per_trunk_step": 0.0,
            "generated_tokens_per_trunk_step": 0.0, "graph_rounds": 0.0,
            "graph_build_s": 0.0, "graph_setup_s": 0.0, "graph_alignment_s": 0.0,
            "plain_tail_steps": 0.0, "plain_tail_s": 0.0,
            "tail_graph_steps": 0.0, "tail_graph_s": 0.0, "tail_graph_built": False,
            "plain_tail_start": None, "plain_tail_reason": "", "mtp_eager_steps": 0.0,
            "arena_tokens": 0.0,
            "native_max_context": float(getattr(model.config, "max_position_embeddings", 0) or 0),
            "logical_total_tokens": float(input_ids.shape[1]), "verify_s": 0.0, "head_s": 0.0,
            "verify_ms_per_round": 0.0, "head_ms_per_round": 0.0,
            "rewind_ms_per_round": 0.0, "stage_ms_per_round": 0.0,
            "verify_device_ms_per_round": 0.0,
        }
    if chunk_size <= 0 or draft_tokens < 0 or input_ids.shape[1] == 0:
        raise ValueError("lookahead needs a nonempty prompt, positive chunk_size and nonnegative drafts")
    if model.mtp is None:
        raise RuntimeError("generate_mtp_draft_lookahead needs a model built with mtp=True")
    if ctx.pp_size > 1:
        raise RuntimeError("--mtp-draft is not supported under pipeline parallelism")
    if int(input_ids.shape[0]) != 1:
        raise RuntimeError("--mtp-draft decodes one sequence at a time")

    prompt_len = input_ids.shape[1]
    # Two draft spans of slack: a recorded round always drafts the full `draft_tokens`, and its
    # chain writes head keys up to `draft_tokens` positions past the verify span.
    native_limit = int(getattr(model.config, "max_position_embeddings", 0) or 0)
    if native_limit and prompt_len + max_new_tokens > native_limit:
        raise ValueError(
            f"prompt ({prompt_len}) plus requested output ({max_new_tokens}) exceeds native "
            f"maximum context ({native_limit}); the request will not be shortened"
        )
    width = prompt_len + max_new_tokens + 2 * draft_tokens + 4
    if native_limit:
        width = min(width, native_limit)
    # The widest head recording needs two row spans at capture time. Rather than reserve
    # speculative positions beyond the native limit, finish with ordinary one-token trunk calls.
    full_round_span = 2 * (draft_tokens + 1)
    plain_tail = draft_tokens == 0 or prompt_len + full_round_span > width
    plain_tail_start = prompt_len if plain_tail else None
    plain_tail_reason = ("drafts_disabled" if draft_tokens == 0 else
                         "native_boundary" if plain_tail and native_limit else "")
    session_lease = None
    session_reused = False
    session_mark = {}
    session_recording_counters = {}
    session_reset_s = 0.0
    session_retained = False
    session_group_fallbacks = 0
    stage_counter_base = device_counter_base = 0.0
    session_boundary_suspensions = 0
    if graph_session is not None:
        if (not verify_graph or ctx.device.type != "cuda" or max_new_tokens <= 1
                or plain_tail or graph_session.model is not model):
            raise ValueError("a graph session needs compatible captured MTP with at least two output tokens")
        from relicllm.models.qwen4_exp.graph_session import GraphSessionKey
        key = GraphSessionKey.build(model=model, ctx=ctx, width=width,
                                    draft_tokens=draft_tokens, rows=draft_tokens + 1)
        session_lease = graph_session.checkout(key)
        session_reused = session_lease.reused
        reusable_ranks = _mtp_session_group_count(ctx, session_reused)
        if reusable_ranks not in (0, ctx.tp_world_size):
            # A rank-local eviction cannot send just that rank into CUDA graph construction while
            # peers replay old collectives. Return every lease and make the whole group cold.
            session_lease.abandon()
            session_lease = graph_session.checkout(key)
            session_reused = False
            session_group_fallbacks = 1
    if session_reused:
        reset_started = time.perf_counter()
        with torch.inference_mode():
            session_lease.begin_request()
        cache, mtp_cache = session_lease.cache, session_lease.head_cache
        session_reset_s = time.perf_counter() - reset_started
    else:
        cache = model.make_cache(batch_size=1, max_seq_len=width)
        mtp_cache = None if plain_tail else model.mtp.make_cache(1, width, model.device, model.dtype)
        if mtp_cache is not None and _mtp_head_table():
            model.mtp.attach_rope_table(mtp_cache, width, model.dtype)
    if prefilled_state is not None and not session_reused:
        # Pool fingerprints include local RoPE storage. Construct those tables before importing
        # state so the receiver can reuse the exact transmitted FP32 pooled keys without recompute.
        with torch.inference_mode(False):
            positions = torch.arange(width, device=ctx.device).view(1, 1, -1)
            cache.cos, cache.sin = model.rope(positions.expand(3, 1, -1), model.dtype)
            cache.rope_width = width
            cache.share_rope_table()
    prof_scope = model.profiler.scope if model.profiler else lambda _name: nullcontext()

    t0 = time.perf_counter()
    pd_restore_s = 0.0
    pd_source_prefill_s = 0.0
    pd_restore_report = None

    def head_token(mixed: torch.Tensor) -> tuple[int, float]:
        """The head's greedy token for its last row, agreed across ranks, and its probability."""
        step_logits = gather_logits(F.linear(mixed[:, -1], model.lm_head), ctx)
        token = int(step_logits.argmax(-1).reshape(-1)[0])
        confidence = float(step_logits.float().softmax(-1).max()) if draft_p_min > 0 else 1.0
        return token, confidence

    def recycle(mixed: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
        """The stack the next chain link is conditioned on (see `_mtp_recycle`)."""
        if _mtp_recycle() == "collapsed":
            return mixed.unsqueeze(-2).expand(
                -1, -1, model.mtp.config.hc_count, -1
            ).reshape_as(stream)
        return stream

    if prefilled_state is not None:
        from relicllm.models.qwen4_exp.pd_state import thaw_state

        import hashlib
        restore_started = time.perf_counter()
        if prefilled_state.past != prompt_len:
            raise ValueError("PD state position differs from the supplied prompt length")
        report = thaw_state(
            prefilled_state, cache=cache, head_cache=mtp_cache, model=model,
            tp_rank=ctx.rank, tp_world_size=ctx.tp_world_size, native_max_context=native_limit,
            expected_metadata={"input_sha256": hashlib.sha256(
                input_ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest()},
        )
        if report["past"] != prompt_len:
            raise ValueError("PD state position differs from the supplied prompt length")
        past = report["past"]
        next_id = _ids([report["next_token"]], ctx)
        if not plain_tail:
            if report["first_draft"] is None or report["chain_stack"] is None:
                raise ValueError("PD MTP state lacks its first draft or recycled head stack")
            first_draft = int(report["first_draft"])
            first_conf = float(report["first_conf"] if report["first_conf"] is not None else 1.0)
            chain_stack = report["chain_stack"].to(device=ctx.device, dtype=model.dtype)
        if ctx.device.type == "cuda":
            torch.cuda.synchronize(ctx.device)
        pd_restore_report = report
        pd_restore_s = time.perf_counter() - restore_started
        pd_source_prefill_s = float(report["prefill_s"] or 0.0)
        prefill_s = time.perf_counter() - t0
    else:
        with torch.inference_mode(), prof_scope("prefill"):
            past = 0
            logits = None
            last_streams = None
            last_start = 0
            while past < prompt_len:
                piece = input_ids[:, past : past + chunk_size]
                logits, piece_streams = model.forward(
                    piece, cache=cache, past_len=past, return_streams=True
                )
                end = past + piece.shape[1]
                if not plain_tail and end < prompt_len:
                    # Head position `i` fuses the trunk state at `i` with the token at `i + 1`, so a
                    # chunk's ids are shifted by one and borrow the next chunk's first token.
                    shifted = input_ids[:, past + 1 : end + 1].to(ctx.device)
                    model.mtp.forward(piece_streams, shifted, mtp_cache, past_len=past)
                elif not plain_tail:
                    # The last position's successor is the first generated token, known only once
                    # the trunk has answered, so this chunk is seeded after the loop.
                    last_streams = piece_streams
                    last_start = past
                past = end
            next_id = next_token_from_logits(logits, ctx, batch_size=1)
            if not plain_tail:
                shifted = torch.cat(
                    [input_ids[:, last_start + 1 :].to(ctx.device), next_id.to(ctx.device)], dim=1
                )
                mixed, stream = model.mtp.forward(
                    last_streams, shifted, mtp_cache, past_len=last_start, return_streams=True
                )
                # Carried into the first round: the head's answer at position `past - 1`, which is its
                # draft for `past + 1`, and the stack the second draft of a round chains from.
                first_draft, first_conf = head_token(mixed)
                chain_stack = recycle(mixed[:, -1:], stream[:, -1:])
            del last_streams, logits

        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        prefill_s = time.perf_counter() - t0


    if on_prefill is not None:
        # The hook sees live request state before capture or any generated-token callback. A
        # transport freezes owned CPU bytes; no CUDA graph or process-local pointer crosses hosts.
        on_prefill({
            "cache": cache, "head_cache": mtp_cache, "past": past,
            "next_token": int(next_id.reshape(-1)[0]),
            "first_draft": first_draft if not plain_tail else None,
            "chain_stack": chain_stack if not plain_tail else None,
            "first_conf": first_conf if not plain_tail else None,
            "prefill_s": prefill_s, "width": width, "native_max_context": native_limit,
        })

    graph = draft_graph = tail_graph = startup_graph = tracer = None
    tail_graph_built = False
    startup_graph_built = False
    startup_graph_replays = 0
    startup_head_s = 0.0
    graph_build_s = graph_setup_s = graph_alignment_s = 0.0
    stage_s_final = device_s_final = 0.0
    trace_path = ""
    ttft_s = 0.0
    next_drafts: list[int] | None = None
    defer_graph_release = os.environ.get("POCKETLLM_QWEN4_MTP_DEFER_GRAPH_RELEASE", "0") == "1"
    retired_graphs: list = []
    deferred_boundary_graphs = 0
    boundary_retire_s = request_cleanup_s = deferred_graph_cleanup_s = 0.0

    def retire_boundary_owner(owner) -> None:
        nonlocal deferred_boundary_graphs, session_boundary_suspensions
        if session_lease is not None:
            # Keep reversible bindings and all recording buffers in the session. The one-row tail
            # is restored by suspending the inner owners in reverse construction order; finish will
            # suspend that outer tail too, ready for a compatible request's eager prefill.
            owner.suspend_bindings()
            session_boundary_suspensions += 1
        elif defer_graph_release:
            # Restore only Python bindings. The recording, pinned buffers and outstanding rewind
            # sources remain owned through final token delivery; final cleanup reports their cost.
            retired_graphs.append(owner)
            deferred_boundary_graphs += 1
            owner.retire_bindings()
        else:
            owner.release()

    def release_retired_owners() -> None:
        if not retired_graphs:
            return
        owner = retired_graphs.pop(0)
        try:
            owner.release()
        finally:
            release_retired_owners()

    def enter_plain_tail() -> None:
        """Restore the committed-prefix bindings before a one-row boundary tail."""
        nonlocal graph, draft_graph, next_drafts, plain_tail, stage_s_final, device_s_final
        nonlocal plain_tail_start, plain_tail_reason, boundary_retire_s
        started = time.perf_counter()
        try:
            try:
                if draft_graph is not None:
                    retire_boundary_owner(draft_graph)
            finally:
                draft_graph = None
                if graph is not None:
                    stage_s_final += graph.stage_s
                    device_s_final += graph.device_s
                    try:
                        retire_boundary_owner(graph)
                    finally:
                        graph = None
        finally:
            boundary_retire_s += time.perf_counter() - started
        next_drafts = None
        plain_tail = True
        plain_tail_start = past
        plain_tail_reason = "native_boundary" if native_limit and width == native_limit else "arena_boundary"

    setup_started = time.perf_counter()
    try:
        if session_reused:
            from relicllm.models.qwen4_exp.graph_session import SessionInvalidated
            resume_error = None
            try:
                session_lease.resume()
            except SessionInvalidated as exc:
                resume_error = exc
                graph_session.stats["address_invalidations"] += 1
            if _mtp_session_group_count(ctx, resume_error is None) != ctx.tp_world_size:
                # Every rank fails at this deterministic seam, before graph-alignment/replay. A
                # rank-local moved-buffer error must never strand other ranks in a collective.
                raise SessionInvalidated("a TP rank's retained graph buffers moved during prefill") from resume_error
            tail_graph = session_lease.owner("tail")
            graph = session_lease.owner("verify")
            draft_graph = session_lease.owner("draft")
            startup_graph = session_lease.owner("startup")
            tail_graph_built = tail_graph is not None
            startup_graph_built = startup_graph is not None
            session_mark = session_lease.counter_mark()
            stage_counter_base = getattr(graph, "stage_s", 0.0)
            device_counter_base = getattr(graph, "device_s", 0.0)
        if (not session_reused and verify_graph and max_new_tokens > 1 and ctx.device.type == "cuda"
                and os.environ.get("POCKETLLM_QWEN4_MTP_TAIL_GRAPH", "0") == "1"):
            from relicllm.models.qwen4_exp.verify_graph import VerifyGraph

            # Install the one-row recording first. The wider verifier saves its staging metadata
            # and restores it at native handoff, so both captures own independent pinned PLE rows
            # while reading/writing the same stable live recurrent-state buffers.
            started = time.perf_counter()
            tail_graph = VerifyGraph(model=model, ctx=ctx, cache=cache, start=past, rows=1)
            tail_graph_built = True
            graph_build_s += time.perf_counter() - started
        if not session_reused and verify_graph and not plain_tail and max_new_tokens > 1 and ctx.device.type == "cuda":
            from relicllm.models.qwen4_exp.verify_graph import VerifyGraph

            started = time.perf_counter()
            graph = VerifyGraph(model=model, ctx=ctx, cache=cache, start=past, rows=draft_tokens + 1)
            graph_build_s += time.perf_counter() - started
            if ctx.rank == 0:
                print(f"[qwen4exp] verify graph: {draft_tokens + 1} rows recorded at {past} "
                      f"in {graph_build_s:.1f}s", flush=True)
        if not session_reused and graph is not None and _mtp_head_table() and draft_p_min <= 0 \
                and os.environ.get("POCKETLLM_QWEN4_MTP_DRAFT_GRAPH", "1") != "0":
            from relicllm.models.qwen4_exp.verify_graph import DraftGraph

            started = time.perf_counter()
            draft_graph = DraftGraph(
                model=model, ctx=ctx, head_cache=mtp_cache, verify=graph, drafts=draft_tokens
            )
            if ctx.rank == 0:
                print(f"[qwen4exp] draft graph: catch-up over {draft_tokens + 1} rows and "
                      f"{draft_tokens - 1} chain links recorded in "
                      f"{time.perf_counter() - started:.1f}s", flush=True)
            graph_build_s += time.perf_counter() - started
        if (not session_reused and draft_graph is not None and draft_tokens > 1 and draft_p_min <= 0
                and ctx.device.type == "cuda"
                and os.environ.get("POCKETLLM_QWEN4_MTP_STARTUP_GRAPH", "0") == "1"):
            from relicllm.models.qwen4_exp.startup_graph import StartupDraftGraph

            started = time.perf_counter()
            startup_graph = StartupDraftGraph(
                model=model, ctx=ctx, head_cache=mtp_cache, start=past,
                first_draft=first_draft, chain_stack=chain_stack, drafts=draft_tokens,
            )
            startup_graph_built = True
            graph_build_s += time.perf_counter() - started
        if session_lease is not None and not session_reused:
            session_lease.adopt(cache=cache, head_cache=mtp_cache, tail=tail_graph,
                                verify=graph, draft=draft_graph, startup=startup_graph)
            session_mark = session_lease.counter_mark()
        if graph is not None or tail_graph is not None:
            # Compile/capture completion varies across ranks. Align once before the measured
            # decode interval, so another rank's remaining setup cannot inflate the first replay.
            if ctx.device.type == "cuda":
                import torch.distributed as dist

                if ctx.tp_world_size > 1 and dist.is_initialized():
                    alignment_started = time.perf_counter()
                    torch.cuda.synchronize(ctx.device)
                    device_index = ctx.device.index
                    if device_index is None:
                        device_index = torch.cuda.current_device()
                    dist.barrier(device_ids=[device_index])
                    torch.cuda.synchronize(ctx.device)
                    graph_alignment_s = time.perf_counter() - alignment_started
            graph_setup_s = time.perf_counter() - setup_started
        graph_rounds = 0
        verify_s = head_s = rewind_s = 0.0

        produced: list[int] = []
        rounds = accepted = rejected = 0
        stopped = "length"
        trunk_steps = plain_tail_steps = tail_graph_steps = 0
        plain_tail_s = tail_graph_s = 0.0
        head_calls = 0
        drafts_proposed = drafts_accepted = verified_drafts_accepted = 0
        # Allocate host counters before the decode interval. Updating them only uses the already
        # host-visible prefix length; fixed-size arrays avoid per-round dictionaries or GPU work.
        prefix_hist_verified = [[0] * (count + 1) for count in range(draft_tokens + 1)]
        prefix_hist_delivered = [[0] * (count + 1) for count in range(draft_tokens + 1)]
        position_proposed = [0] * draft_tokens
        position_verified = [0] * draft_tokens
        position_delivered = [0] * draft_tokens
        head_kept_hist = [0] * (draft_tokens + 2)
        head_replay_kept_hist = [0] * (draft_tokens + 2)
        t1 = time.perf_counter()

        # `POCKETLLM_QWEN4_ROUND_TRACE=<skip>:<rounds>:<path>` records a kernel trace of `rounds`
        # recorded rounds after `skip` warm ones, on rank 0, to price the round's pieces offline.
        trace_spec = os.environ.get("POCKETLLM_QWEN4_ROUND_TRACE", "")
        trace_skip = trace_rounds = 0
        trace_path = ""
        if trace_spec and ctx.rank == 0 and graph is not None:
            trace_skip, trace_rounds, trace_path = trace_spec.split(":", 2)
            trace_skip, trace_rounds = int(trace_skip), int(trace_rounds)

        with torch.inference_mode(), prof_scope("decode"):
            while True:
                if trace_path and rounds == trace_skip and tracer is None:
                    from torch.profiler import ProfilerActivity, profile

                    tracer = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                    tracer.__enter__()
                if tracer is not None and rounds == trace_skip + trace_rounds:
                    torch.cuda.synchronize()
                    completed_trace, tracer = tracer, None
                    completed_trace.__exit__(None, None, None)
                    completed_trace.export_chrome_trace(trace_path)
                    print(f"[qwen4exp] round trace written to {trace_path}", flush=True)
                    trace_path = ""
                if on_step is not None and on_step():
                    # The predicate is the collective; a local EOS only names the stop reason.
                    stopped = "eos" if (local_eos is not None and local_eos[0]) else "cancel"
                    break
                token = int(next_id.reshape(-1)[0])
                produced.append(token)
                if len(produced) == 1:
                    ttft_s = time.perf_counter() - t0
                if on_token is not None and ctx.rank == 0:
                    on_token(token)
                if token in eos_token_ids:
                    if on_step is None or local_eos is None:
                        stopped = "eos"
                        break
                    # Do not leave on our own: the poll below is reached by every rank and the
                    # caller's predicate decides there. Breaking here would strand the peers in
                    # the cooperative check, which is the hung request `generate` documents.
                    local_eos[0] = True
                if len(produced) >= max_new_tokens:
                    # A request can end on its budget and on an EOS token at the same emission;
                    # the delivered prefix is what names which (the old derived rule).
                    stopped = "eos" if produced[-1] in eos_token_ids else "length"
                    break
                if on_step is not None and on_step():
                    # Asked before any trunk or head work for the token this round would produce:
                    # an EOS above, or a stop/error rank 0 recorded in the callback, ends here.
                    stopped = "eos" if (local_eos is not None and local_eos[0]) else "cancel"
                    break

                if not plain_tail and past + full_round_span > width:
                    enter_plain_tail()
                if plain_tail:
                    if past >= width:
                        raise RuntimeError("one-token tail would exceed the native KV arena")
                    phase = time.perf_counter()
                    if tail_graph is not None:
                        answer = tail_graph.replay(past, [token])
                        tail_graph.rewind(past, 1, [token])
                        tail_graph_steps += 1
                        next_id = torch.tensor([[answer[0]]], dtype=torch.long, device=ctx.device)
                        tail_graph_s += time.perf_counter() - phase
                    else:
                        step_logits, _unused_streams = model.forward(
                            _ids([token], ctx), cache=cache, past_len=past, return_streams=True
                        )
                        plain_tail_steps += 1
                        next_id = next_token_from_logits(step_logits, ctx, batch_size=1)
                        plain_tail_s += time.perf_counter() - phase
                    trunk_steps += 1
                    past += 1
                    cache.length = past
                    continue

                # -- draft -- `token` sits at position `past` and has not been consumed yet. The
                # first draft (for `past + 1`) was produced by the previous round's catch-up; each
                # further link drafts one position later from the previous link's own output.
                room = max_new_tokens - len(produced)
                drafts: list[int] = []
                phase = time.perf_counter()
                if next_drafts is not None:
                    drafts = list(next_drafts)
                elif startup_graph is not None and startup_graph_replays == 0:
                    # Real generation work remains after t1 and after the first token callback;
                    # only recording/warmup belongs to initial setup. Keep its owners until cleanup.
                    drafts = (startup_graph.replay(start=past, first_draft=first_draft,
                                                   chain_stack=chain_stack)
                              if session_reused else startup_graph.replay())
                    startup_graph_replays += 1
                    head_calls += draft_tokens - 1
                elif draft_tokens > 0 and room > 0 and (draft_p_min <= 0 or first_conf >= draft_p_min):
                    drafts.append(first_draft)
                    stack = chain_stack
                    tok = _ids([first_draft], ctx)
                    at = past
                    # A recorded round has one width, so while it is live every round drafts the full
                    # `draft_tokens`; drafts past the budget are verified and then cut from the output,
                    # which keeps the last round on the recording instead of an eager forward.
                    limit = draft_tokens if graph is not None else min(draft_tokens, room)
                    while len(drafts) < limit:
                        mixed, stream = model.mtp.forward(
                            stack, tok, mtp_cache, past_len=at, return_streams=True
                        )
                        head_calls += 1
                        drafted, confidence = head_token(mixed)
                        if draft_p_min > 0 and confidence < draft_p_min:
                            break
                        drafts.append(drafted)
                        stack = recycle(mixed, stream)
                        tok = _ids([drafted], ctx)
                        at += 1
                initial_or_round_head_s = time.perf_counter() - phase
                head_s += initial_or_round_head_s
                if rounds == 0:
                    startup_head_s += initial_or_round_head_s
                rounds += 1
                drafts_proposed += len(drafts)

                # -- verify -- `token` and every draft in one trunk forward. Row `i` answers for
                # position `past + i + 1`, so draft `j` (0-based) is judged by row `j`.
                feed = [token] + drafts
                phase = time.perf_counter()
                mark = torch.profiler.record_function("round.verify") if tracer is not None else nullcontext()
                with mark:
                    replayed = graph is not None and len(feed) == graph.rows
                    if graph is not None and not replayed:
                        # An eager trunk forward cannot run beside the recording: the cache's PLE history
                        # is staged for the replay, so the forward would hash the wrong n-grams. A round of
                        # another width therefore ends the recording for the rest of the request.
                        if draft_graph is not None:
                            raise RuntimeError("a recorded draft round produced the wrong draft count")
                        stage_s_final += graph.stage_s
                        device_s_final += graph.device_s
                        graph.release()
                        graph = None
                        # Generic eager fallback must remove the underlying one-row staging too:
                        # only a tail graph replay refreshes its pinned history, not eager forward.
                        if tail_graph is not None:
                            tail_graph.release()
                            tail_graph = None
                    if replayed:
                        rows = graph.replay(past, feed)
                        verify_streams = graph.streams.unsqueeze(0)
                        graph_rounds += 1
                    else:
                        history_before = (
                            None if cache.ple_token_history is None
                            else ((None, None, cache.ple_token_history.clone()),)
                        )
                        _begin_state_log(cache)
                        _begin_ple_log(cache)
                        verify_logits, verify_streams = model.forward(
                            _ids(feed, ctx), cache=cache, past_len=past, return_streams=True
                        )
                        rows = gather_logits(verify_logits[0], ctx).argmax(-1).reshape(-1).tolist()
                    trunk_steps += 1
                verify_s += time.perf_counter() - phase
                accepted_run = 0
                for drafted, got in zip(drafts, rows):
                    if drafted != got:
                        break
                    accepted_run += 1
                if os.environ.get("POCKETLLM_QWEN4_MTP_TRACE") == "1" and ctx.rank == 0:
                    print(f"[trace-la] past={past} feed={feed} rows={rows} "
                          f"accepted={accepted_run}", flush=True)

                verified_drafts_accepted += accepted_run
                # Both are decisions already made from the host-visible gates; the increments
                # themselves add no read, synchronization, or tensor operation.
                prefix_hist_verified[len(drafts)][accepted_run] += 1
                for position in range(len(drafts)):
                    position_proposed[position] += 1
                    if position < accepted_run:
                        position_verified[position] += 1
                # Fixed-width recording may verify past the output budget or an accepted EOS. Only
                # the deliverable prefix is committed/emitted, and callbacks must never see the rest.
                accepted_run = min(accepted_run, max_new_tokens - len(produced))
                for index, drafted in enumerate(drafts[:accepted_run]):
                    if drafted in eos_token_ids:
                        accepted_run = index + 1
                        break
                prefix_hist_delivered[len(drafts)][accepted_run] += 1
                for position in range(accepted_run):
                    position_delivered[position] += 1
                # The forward consumed `len(feed)` positions; the sequence keeps `kept` of them.
                kept = 1 + accepted_run
                if replayed:
                    phase = time.perf_counter()
                    graph.rewind(past, kept, feed)
                    rewind_s += time.perf_counter() - phase
                else:
                    _rewind_to(cache, kept)
                    _end_state_log(cache, past + kept)
                    _rewind_ple_conv(cache, kept)
                    _rewind_ple(cache, history_before, feed[:kept], model.config.ngram_size)
                drafts_accepted += accepted_run
                if drafts:
                    if accepted_run == 0:
                        rejected += 1
                    else:
                        accepted += 1
                for drafted in drafts[:accepted_run]:
                    produced.append(drafted)
                    if on_token is not None and ctx.rank == 0:
                        on_token(drafted)
                if any(t in eos_token_ids for t in drafts[:accepted_run]):
                    if on_step is None or local_eos is None:
                        stopped = "eos"
                        break
                    # Same deferral as the single token above: name it locally, let the poll below
                    # decide collectively.
                    local_eos[0] = True
                if len(produced) >= max_new_tokens:
                    stopped = "eos" if produced[-1] in eos_token_ids else "length"
                    break
                if on_step is not None and on_step():
                    # The accepted prefix's callbacks have run and rank 0 may have recorded a stop
                    # or an error. Leave before the head catch-up spends a round on drafts that
                    # will never be emitted. The verified round is already rewound to `kept`, so
                    # commit that prefix before leaving rather than leaving a half-advanced cache.
                    stopped = "eos" if (local_eos is not None and local_eos[0]) else "cancel"
                    past += kept
                    cache.length = past
                    next_drafts = None
                    break
                bonus = int(rows[accepted_run])
                # This bonus is already verified. If it ends the request, emit it now rather than
                # building a head catch-up/chain whose results will never be used.
                if bonus in eos_token_ids or len(produced) + 1 >= max_new_tokens:
                    produced.append(bonus)
                    if on_token is not None and ctx.rank == 0:
                        on_token(bonus)
                    # The bonus is a row of this round's verify, identical on every rank, so this
                    # exit is already collective: no poll is needed to make the ranks agree.
                    if bonus in eos_token_ids:
                        stopped = "eos"
                        if local_eos is not None:
                            local_eos[0] = True
                    else:
                        stopped = "length"
                    break
                next_id = torch.tensor([[bonus]], dtype=torch.long, device=ctx.device)
                if past + kept + full_round_span > width:
                    # The next round cannot safely speculate at the native boundary. Its bonus
                    # is already known, so discard speculative head state and keep the trunk's
                    # rewound GDN/PLE prefix. No head catch-up is needed for a plain trunk tail.
                    past += kept
                    cache.length = past
                    enter_plain_tail()
                    continue

                # -- head catch-up -- rewrite head positions `past .. past + kept - 1` with the true
                # pairing `(h_j, t_{j+1})`: the trunk's verify streams and the kept feed shifted by
                # one, ending in the bonus token. The chain's speculative keys at those positions are
                # overwritten; later ones are never read, because a QSA call reads only up to its own
                # end and the pooled indexer keys are invalidated from the first group written. The
                # last row is the head's answer at the new `past - 1`: the next round's first draft.
                phase = time.perf_counter()
                head_kept_hist[kept] += 1
                if draft_graph is not None:
                    # The catch-up and the whole next chain in one replay (see `DraftGraph`).
                    with (torch.profiler.record_function("round.draft") if tracer is not None
                          else nullcontext()):
                        next_drafts = draft_graph.replay(past, kept)
                    head_replay_kept_hist[kept] += 1
                    head_calls += draft_tokens
                else:
                    catch_ids = _ids(feed[1:kept] + [bonus], ctx)
                    mixed, stream = model.mtp.forward(
                        verify_streams[:, :kept], catch_ids, mtp_cache,
                        past_len=past, return_streams=True,
                    )
                    head_calls += 1
                    first_draft, first_conf = head_token(mixed)
                    chain_stack = recycle(mixed[:, -1:], stream[:, -1:])
                head_s += time.perf_counter() - phase
                past += kept
                cache.length = past

        if ctx.device.type == "cuda":
            torch.cuda.synchronize()
        decode_s = time.perf_counter() - t1
        if graph is not None:
            stage_s_final += graph.stage_s
            device_s_final += graph.device_s
        if session_reused:
            stage_s_final -= stage_counter_base
            device_s_final -= device_counter_base
    finally:
        cleanup_started = time.perf_counter()
        if session_lease is not None:
            owns_recordings = session_lease.cache is cache
            session_recording_counters = session_lease.counter_delta(session_mark)
            completed_request = "decode_s" in locals() and stopped != "cancel"
            session_lease.finish(boundary_retired=bool(retired_graphs), successful=completed_request)
            session_retained = session_lease.state == "suspended"
            if owns_recordings:
                # The session retained or released the entire adopted bundle. A failed adoption
                # leaves constructor-owned graphs to the original nested finalizers below.
                graph = draft_graph = tail_graph = startup_graph = None
                retired_graphs = []
        # Close short/failed traces too. Nested finalizers ensure one failed teardown cannot skip
        # the other recording's restoration or leave per-position logging enabled on the cache.
        try:
            try:
                if tracer is not None:
                    completed_trace, tracer = tracer, None
                    completed_trace.__exit__(None, None, None)
                    if trace_path:
                        completed_trace.export_chrome_trace(trace_path)
            finally:
                try:
                    if draft_graph is not None:
                        draft_graph.release()
                finally:
                    try:
                        if graph is not None:
                            graph.release()
                    finally:
                        try:
                            # Reverse installation order: active main bindings restore the tail;
                            # metadata-retired owners already did so and cannot rebind it later.
                            if tail_graph is not None:
                                tail_graph.release()
                        finally:
                            for layer in cache.layers:
                                if layer is not None and hasattr(layer, "state_log"):
                                    layer.state_log = None
                            for state in cache.ple_states.values():
                                state.state_log = None
        finally:
            try:
                if startup_graph is not None:
                    startup_graph.release()
            finally:
                deferred_started = time.perf_counter()
                try:
                    release_retired_owners()
                finally:
                    deferred_graph_cleanup_s += time.perf_counter() - deferred_started
                    request_cleanup_s = time.perf_counter() - cleanup_started

    decoded_tokens = max(0, len(produced) - 1)
    decode_wall_s = graph_setup_s + decode_s
    decode_s_including_cleanup = decode_s + request_cleanup_s
    stats = {
        "prompt_tokens": float(prompt_len),
        "generated_tokens": float(len(produced)),
        "prefill_s": prefill_s,
        "decode_s": decode_s,
        "decode_s_excluding_setup": decode_s,
        "decode_wall_s": decode_wall_s,
        "decode_tokens": float(decoded_tokens),
        "prefill_first_tokens": float(bool(produced)),
        "ttft_s": ttft_s,
        "stopped": stopped,
        # The first generated token was produced by prefill, outside this decode interval.
        "decode_tps": decoded_tokens / decode_s if decode_s > 0 else 0.0,
        "decode_tps_excluding_setup": decoded_tokens / decode_s if decode_s > 0 else 0.0,
        "decode_wall_tps": decoded_tokens / decode_wall_s if decode_wall_s > 0 else 0.0,
        "draft_rounds": float(rounds),
        "draft_accepted": float(accepted),
        "draft_rejected": float(rejected),
        "draft_accept_rate": accepted / rounds if rounds else 0.0,
        "drafts_proposed": float(drafts_proposed),
        "drafts_accepted": float(drafts_accepted),
        "drafts_accepted_verified": float(verified_drafts_accepted),
        "per_draft_accept": drafts_accepted / drafts_proposed if drafts_proposed else 0.0,
        "per_draft_accept_verified": (verified_drafts_accepted / drafts_proposed
                                      if drafts_proposed else 0.0),
        # Width-keyed dicts, observed widths only, so the JSON stays small and unambiguous.
        "accepted_prefix_histogram_verified_by_width": {
            str(width_key): row for width_key, row in enumerate(prefix_hist_verified)
            if sum(row)
        },
        "accepted_prefix_histogram_delivered_by_width": {
            str(width_key): row for width_key, row in enumerate(prefix_hist_delivered)
            if sum(row)
        },
        "draft_position_proposed": position_proposed,
        "draft_position_survived_verified": position_verified,
        "draft_position_survived_delivered": position_delivered,
        "head_kept_prefix_histogram": head_kept_hist,
        "head_replay_kept_prefix_histogram": head_replay_kept_hist,
        "trunk_steps": float(trunk_steps),
        "head_calls": float(head_calls),
        "tokens_per_trunk_step": decoded_tokens / max(1, trunk_steps),
        "generated_tokens_per_trunk_step": len(produced) / max(1, trunk_steps),
        "graph_rounds": float(graph_rounds),
        "graph_build_s": graph_build_s,
        "graph_setup_s": graph_setup_s,
        "graph_alignment_s": graph_alignment_s,
        "plain_tail_steps": float(plain_tail_steps),
        "plain_tail_s": plain_tail_s,
        "tail_graph_steps": float(tail_graph_steps),
        "tail_graph_s": tail_graph_s,
        "tail_graph_built": tail_graph_built,
        "startup_graph_built": startup_graph_built,
        "startup_graph_replays": float(startup_graph_replays),
        "startup_head_s": startup_head_s,
        "graph_session_reused": session_reused,
        "graph_session_retained": session_retained,
        "graph_session_reset_s": session_reset_s,
        "graph_session_boundary_suspensions": session_boundary_suspensions,
        "graph_session_recording_counters": session_recording_counters,
        "graph_session_group_fallbacks": session_group_fallbacks,
        "pd_prefilled": prefilled_state is not None,
        "pd_restore_s": pd_restore_s,
        "pd_source_prefill_s": pd_source_prefill_s,
        "pd_indexer_pool_rebound": (pd_restore_report or {}).get("indexer_pool_rebound", 0),
        "pd_indexer_pool_needs_recompute": (pd_restore_report or {}).get("indexer_pool_needs_recompute", 0),
        "pd_backbone_rope_table_ready": (pd_restore_report or {}).get("backbone_rope_table_ready", False),
        "pd_head_rope_table_ready": (pd_restore_report or {}).get("head_rope_table_ready", False),
        "boundary_retire_s": boundary_retire_s,
        "deferred_boundary_graphs": float(deferred_boundary_graphs),
        "deferred_graph_cleanup_s": deferred_graph_cleanup_s,
        "request_cleanup_s": request_cleanup_s,
        "decode_s_including_cleanup": decode_s_including_cleanup,
        "decode_tps_including_cleanup": (decoded_tokens / decode_s_including_cleanup
                                        if decode_s_including_cleanup > 0 else 0.0),
        "plain_tail_start": float(plain_tail_start) if plain_tail_start is not None else None,
        "plain_tail_reason": plain_tail_reason,
        "mtp_eager_steps": float(trunk_steps - graph_rounds - plain_tail_steps - tail_graph_steps),
        "arena_tokens": float(width),
        "native_max_context": float(native_limit),
        "logical_total_tokens": float(prompt_len + len(produced)),
        "verify_s": verify_s,
        "head_s": head_s,
        "verify_ms_per_round": 1000.0 * verify_s / max(1, rounds),
        "head_ms_per_round": 1000.0 * head_s / max(1, rounds),
        "rewind_ms_per_round": 1000.0 * rewind_s / max(1, rounds),
        "stage_ms_per_round": 1000.0 * stage_s_final
        / max(1, rounds),
        "verify_device_ms_per_round": 1000.0
        * device_s_final / max(1, rounds),
    }
    return produced, stats


def _ids(values: list[int], ctx: TPContext) -> torch.Tensor:
    return torch.tensor([values], dtype=torch.long, device=ctx.device)


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
    batch: int = 1,
    warmup_s: float = 0.0,
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
    expert_resident_layers: bool = False,
    expert_nvfp4: bool = False,
    graph_window: int = 0,
    mtp: bool = False,
    mtp_draft_tokens: int = 1,
    mtp_accuracy_report: bool = False,
    mtp_draft: bool = False,
    mtp_draft_p_min: float = 0.0,
    mtp_lookahead: bool = False,
    mtp_verify_graph: bool = False,
) -> dict:
    from relicllm.models.qwen4_exp.profiler import Profiler

    if mtp_verify_graph and not mtp_lookahead:
        raise ValueError("--mtp-verify-graph requires --mtp-lookahead")
    if mtp_lookahead and mtp_draft:
        raise ValueError("choose --mtp-lookahead or the legacy --mtp-draft path, not both")
    if mtp_lookahead and (batch != 1 or graph_window != 0):
        raise ValueError("--mtp-lookahead needs batch 1 and --graph-window 0")
    if mtp_verify_graph and mtp_draft_tokens not in (3, 4):
        raise ValueError("captured lookahead is qualified for draft depths 3/4 only; depth 5 changed tokens")
    if mtp_verify_graph and os.environ.get("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "0") != "1":
        raise ValueError("captured lookahead requires POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT=1")
    if mtp_verify_graph and os.environ.get("POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT", "0") != "1":
        raise ValueError("captured lookahead requires POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT=1")

    ctx = init_distributed()
    if mtp_verify_graph and (ctx.device.type != "cuda" or ctx.pp_size != 1):
        raise ValueError("captured lookahead needs CUDA with pure tensor parallelism")
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
        expert_resident_layers=expert_resident_layers,
        graph_window=graph_window,
        expert_nvfp4=expert_nvfp4,
        mtp=mtp or mtp_draft or mtp_lookahead,
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
    if batch > 1:
        # Replicate the prompt rather than pad: every row then has the same prompt length, which
        # keeps the step shapes rectangular and makes the measurement about concurrency rather
        # than about how a shorter row idles.
        input_ids = input_ids.expand(batch, -1).contiguous()

    # Optional clock-ramp work belongs outside measured prefill/decode, and does not
    # touch the model or request cache. Both standard and MTP paths use the same helper.
    warmup_elapsed_s = _warmup_device(ctx, model.dtype, warmup_s)
    eos = tuple(config.text_config.eos_token_id)
    if mtp_lookahead:
        produced, stats = generate_mtp_draft_lookahead(
            model, ctx, input_ids, max_new_tokens=max_new_tokens, chunk_size=chunk_size,
            eos_token_ids=eos, graph_window=0, draft_tokens=mtp_draft_tokens,
            draft_p_min=mtp_draft_p_min, verify_graph=mtp_verify_graph,
        )
        stats["execution_mode"] = "captured_lookahead" if mtp_verify_graph else "eager_lookahead"
    elif mtp_draft:
        produced, stats = generate_mtp_draft(
            model,
            ctx,
            input_ids,
            max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
            eos_token_ids=eos,
            graph_window=graph_window,
            draft_tokens=mtp_draft_tokens,
            draft_p_min=mtp_draft_p_min,
        )
    else:
        produced, stats = _maybe_kernel_profile(ctx, lambda: generate(
            model,
            ctx,
            input_ids,
            max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
            eos_token_ids=eos,
            graph_window=graph_window,
        ))
    stats["load_s"] = load_s
    stats["warmup_s"] = warmup_elapsed_s

    # A batch decodes one list a row; the single-sequence form has always been a flat list, so the
    # first row is what gets shown and the rest stay in `tokens` for whoever wants them.
    flat = produced[0] if produced and isinstance(produced[0], list) else produced
    result = {
        "tokens": produced,
        "text": tokenizer.decode(flat),
        "stats": stats,
        "rows": len(produced) if produced and isinstance(produced[0], list) else 1,
    }
    if mtp_accuracy_report and model.has_mtp:
        # A second generation, run one token at a time so the backbone's pre-mixer state exists at
        # every step: the head is scored against the tokens the *first* pass produced, which is the
        # same greedy chain this run would follow.
        measured = mtp_accuracy(
            model,
            ctx,
            input_ids,
            max_new_tokens=max_new_tokens,
            chunk_size=chunk_size,
            eos_token_ids=eos,
        )
        result["mtp"] = measured
        if ctx.rank == 0:
            print(
                f"[rank0] mtp draft: {measured['hits']}/{measured['predictions']} next tokens "
                f"correct ({100.0 * measured['accuracy']:.1f}%, chance "
                f"{100.0 * measured['chance']:.5f}%), next-token "
                f"{measured['next_hits']}/{measured['predictions']} "
                f"(chained {measured['depth2_hits']}/{measured['depth2_total']}) "
                f"({100.0 * measured['next_accuracy']:.1f}%), top-5 "
                f"{measured['next_top5_hits']}/{measured['predictions']} "
                f"({100.0 * measured['next_top5_accuracy']:.1f}%), "
                f"input mode {measured['input_mode']}",
                flush=True,
            )
            for step in measured["per_step"]:
                mark = "hit " if step["hit"] else "miss"
                print(
                    f"[rank0]   pos {step['position']:4d}  backbone {step['backbone']:6d}  "
                    f"mtp {step['mtp']:6d}  {mark}",
                    flush=True,
                )
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
    # `allow_abbrev=False`: `--expert-resident` is a prefix of `--expert-resident-layers`, and the
    # abbreviation rule would resolve the longer name onto the shorter flag.
    parser = argparse.ArgumentParser(
        description="Qwen4-Exp TP heterogeneous inference", allow_abbrev=False
    )
    parser.add_argument("--model", required=True, help="checkpoint directory")
    parser.add_argument("--prompt", default="Hello, who are you?")
    parser.add_argument(
        "--warmup-s",
        type=float,
        default=0.0,
        help="seconds of throwaway CUDA compute before measured prefill/decode; this is a clock-ramp "
        "hint on hosts where clocks cannot be locked, not a locked-clock guarantee",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="decode this many concurrent copies of the prompt; the aggregate rate is what a "
        "server reports, so this is how a single-stream number is turned into a capacity one",
    )
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
            "capture the decode step and replay it, which is 3-4x the eager rate on 8xRTX 3090 "
            "(33 tok/s against 8). This is the number of positions one capture serves before it "
            "is remade; 0 leaves the decode eager. The step and the eager loop differ in the last "
            "bits of the logits, so a near-tied token can resolve differently -- the first few "
            "tokens of a reply match and a later one may not. Ask for the eager loop when "
            "token-for-token agreement matters"
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
        "--expert-resident-layers",
        action="store_true",
        help=(
            "hold as many whole layers' experts on the card as the free memory allows, staging "
            "only the layers that do not fit. On 4x2080Ti that pins 23 of 48 layers and the "
            "pinned ones take the batched expert path, which was worth 4.64 against 3.99 tok/s. "
            "Pair it with --expert-cache: the two cover different layers."
        ),
    )
    parser.add_argument(
        "--expert-nvfp4",
        action="store_true",
        help=(
            "read the routed experts from an NVIDIA ModelOpt NVFP4 checkpoint and run them with "
            "relic-core's fused FP4 MoE kernel, holding every layer on the card. A rank's 48 "
            "layers are 14.9 GiB against 21.5 GiB, so no layer stages a step. Needs a checkpoint "
            "quantised with modelopt and a kernel built for the 16-element grouping it stores."
        ),
    )
    parser.add_argument(
        "--mtp",
        action="store_true",
        help=(
            "also load the checkpoint's MTP draft head (one extra decoder layer, `mtp.*`, and its "
            "own 512 routed experts). Nothing in the default forward pass reads it; it exists for "
            "--mtp-accuracy and for speculative decoding. Costs the head's dense half (~40 MiB a "
            "rank replicated) plus whatever its experts' backend holds"
        ),
    )
    parser.add_argument(
        "--mtp-accuracy",
        action="store_true",
        help=(
            "with --mtp: generate once, then re-generate one token at a time and report how often "
            "the MTP head's draft is the token the backbone actually chose. This is the check that "
            "the head is wired correctly -- a misloaded head still returns confident logits, it "
            "just predicts the wrong tokens"
        ),
    )
    parser.add_argument(
        "--mtp-draft-p-min",
        type=float,
        default=0.0,
        help=(
            "stop drafting when the head's top-1 probability falls below this.  A chain of drafts "
            "is autoregressive, so an unsure link poisons the ones after it; this is the knob the "
            "llama.cpp qwen4exp recipes use (--spec-draft-p-min 0.75).  0 drafts the full count "
            "every round."
        ),
    )
    parser.add_argument(
        "--mtp-draft-tokens",
        type=int,
        default=1,
        help=(
            "draft count for the selected MTP mode. Captured --mtp-lookahead is currently qualified "
            "at depths 3 and 4; depth 5 changed a greedy token in the full-output check and is refused. "
            "Default 1 retains the legacy mode's existing behaviour."
        ),
    )
    parser.add_argument(
        "--mtp-lookahead",
        action="store_true",
        help=(
            "opt into shifted convention-B MTP lookahead. Implies --mtp; batch 1, pure tensor "
            "parallelism and --graph-window 0. Pair with --mtp-verify-graph for the qualified "
            "captured path; this is not a 100 tok/s guarantee."
        ),
    )
    parser.add_argument(
        "--mtp-verify-graph",
        action="store_true",
        help=(
            "capture --mtp-lookahead verification; requires --mtp-draft-tokens 3 or 4 and "
            "POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT=1. Initial recording cost remains in TTFT/setup."
        ),
    )
    parser.add_argument(
        "--mtp-draft",
        action="store_true",
        help=(
            "decode with one MTP draft a step and verify it against the backbone, accepting the "
            "draft when greedy decode would have chosen it anyway. Implies --mtp. The output is "
            "identical to a plain decode; the speed is a property of the checkpoint's draft "
            "accuracy. Batch 1, no pipeline, no --graph-window"
        ),
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
        batch=args.batch,
        warmup_s=args.warmup_s,
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
        expert_resident_layers=args.expert_resident_layers,
        expert_nvfp4=args.expert_nvfp4,
        raw_prompt=args.raw_prompt,
        enable_profile=args.profile,
        graph_window=args.graph_window,
        mtp=args.mtp or args.mtp_accuracy or args.mtp_draft or args.mtp_lookahead,
        mtp_accuracy_report=args.mtp_accuracy,
        mtp_draft=args.mtp_draft,
        mtp_lookahead=args.mtp_lookahead,
        mtp_verify_graph=args.mtp_verify_graph,
        mtp_draft_tokens=args.mtp_draft_tokens,
        mtp_draft_p_min=args.mtp_draft_p_min,
    )


if __name__ == "__main__":
    main()
