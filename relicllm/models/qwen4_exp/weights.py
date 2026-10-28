"""Safetensors access for Qwen4-Exp: mmap reads plus host-resident experts.

The checkpoint is 335 GiB, of which 225 GiB is routed experts and 95 GiB is the
PLE n-gram table.  Neither fits in 4x22 GiB of VRAM, so both live on the host and
the GPU only ever sees the rows a step actually touches.

The reading itself is `relicllm.loader.safetensors.MmapSafetensors`, which maps each
shard once and hands out torch views over the raw bytes; it is re-exported here
because this module is where readers of it look.  Reading experts through the
mapping alone is not enough, though: the mapping is backed by a mechanical disk
here, so a page that is not resident costs a seek.  `HostExpertShard` therefore
copies this rank's routed experts into host RAM once at startup (pinned when the
memlock limit allows), and `Qwen4ExpCheckpoint` serves `expert_rows` from that
copy afterwards.  Under TP the shards are disjoint
(`expert_id % world_size == rank`), so the four ranks together hold exactly one
copy of the expert set.

The FP8 build of the same model halves both of those: routed experts drop from
229.7 to 114.9 GiB and the PLE table from 95.4 to 47.7 GiB.  It also changes the
expert *layout* -- `experts.{id}.{gate,up,down}_proj.weight` with a
`weight_scale_inv` each, instead of two packed `[512, ...]` tensors -- so the
reader assembles the runtime's packed `[2*inter, hidden]` convention itself and
everything downstream sees one shape regardless of which checkpoint is loaded.
"""

from __future__ import annotations

import json
import os

import torch

from relicllm.loader.safetensors import MmapSafetensors, TensorEntry
from relicllm.models.qwen4_exp.quant import FP8Tensor, fp8_scalar_dequantize, pack_gate_up_fp8

__all__ = [
    "HostEmbedding",
    "HostExpertShard",
    "HostNGramTable",
    "MmapSafetensors",
    "Qwen4ExpCheckpoint",
    "TensorEntry",
    "DeviceEmbedding",
]


def _gate_up_plane_shape(row_shape: torch.Size) -> tuple[int, ...]:
    """`[2*inter, hidden]` codes as `[ceil(hidden/16), 2*inter, 16]`, the fused kernel's plane."""
    rows, cols = row_shape
    if cols % 16:
        raise ValueError(f"a packed gate/up plane needs a hidden size of a multiple of 16, got {cols}")
    return (cols // 16, rows, 16)


def pack_gate_up_plane(codes: torch.Tensor, out: torch.Tensor, *, device=None) -> None:
    """Write `[..., 2*inter, hidden]` e4m3 codes into `out`'s plane layout, one 16-code chunk at a time.

    The transposition `relic_core`'s `qwen4_exp_fp8_pack_t` does on the card, done here without a
    second copy of the rows: each chunk is a view the reader assembles into a buffer the shard
    already owns, which is the point -- a resident rank has no memory to spare for the row copy
    *and* the plane.

    Both sides are handled as bytes.  A `float8_e4m3fn` permute is a *value* gather -- this torch
    stores a viewed e4m3 with its low bit in the sign position, and promotes it back through a
    scalar loop -- which is both lossy and slow; the planes are copied, so the codes have to be
    read and written as the bytes they are.
    """
    rows, cols = codes.shape[-2:]
    chunks = cols // 16
    lead = codes.shape[:-2]
    flat = out.reshape(*lead, chunks, rows, 16)
    source = codes.view(torch.uint8).reshape(*lead, rows, chunks, 16)
    target = flat.view(torch.uint8)
    target.copy_(source.permute(*range(len(lead)), -2, -3, -1), non_blocking=device is not None)


def unpack_gate_up_plane(planes: torch.Tensor) -> torch.Tensor:
    """`[..., ceil(hidden/16), 2*inter, 16]` planes back to the row layout `[..., 2*inter, hidden]`."""
    lead = planes.shape[:-3]
    chunks, rows, width = planes.shape[-3:]
    out = torch.empty(*lead, rows, chunks * width, dtype=torch.uint8, device=planes.device)
    out.view(*lead, rows, chunks, width).copy_(planes.view(torch.uint8).permute(*range(len(lead)), -2, -3, -1))
    return out.view(torch.float8_e4m3fn)


class HostExpertShard:
    """This rank's routed experts, copied out of the mapping into host RAM.

    One pair of contiguous host tensors per layer, indexed by *local* expert id
    (`expert_id // world_size`), so a 48-layer TP4 shard is 96 allocations rather
    than 12,288 expert-row ones.  Pinned memory is requested because the H2D
    staging in `HostExpertMoE` is the consumer, but a 56 GiB pin can exceed
    `ulimit -l`; on failure the shard falls back to pageable memory once and says
    so, rather than dying at load time.

    An FP8 checkpoint adds a second buffer per projection for the block scales,
    so a layer holds four tensors instead of two.  They are still allocated as
    one group: a shard where the codes are pinned and the scales are not would
    stall every staging copy on the unpinned half.

    ``device`` turns the shard from a *host* bank into a device-resident one.
    It is the whole of what the resident mode needs on this side: the layout is
    already per-layer and per-local-expert, so the same four buffers and the
    same ``rows()`` serve either placement, and the only difference is which
    device they are allocated on.  ``pin_memory`` is a host concept and is
    ignored when a device is named -- pinned memory is what makes an
    asynchronous H2D copy cheap, and a resident shard does not copy at all.
    Codes and scales stay quantized on the card and are dequantized at the
    GEMM, exactly as the host path does; expanding them to bf16 would be
    28 GiB a rank and would not fit on a 24 GiB card.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        rank: int = 0,
        world_size: int = 1,
        pin_memory: bool = True,
        device: torch.device | str | None = None,
        packed_gate_up: bool = False,
    ) -> None:
        if world_size < 1 or not (0 <= rank < world_size):
            raise ValueError(f"bad shard placement: rank={rank} world_size={world_size}")
        #: Store gate/up in the fused kernel's plane layout instead of `[2*inter, hidden]`.  The
        #: fused decode reads that layout, and holding both would double a resident layer; so it is
        #: a property of how the layer is *written*, chosen before the first copy, and
        #: `layer_tensors` still answers with the row layout by reading the planes back.
        self.packed_gate_up = bool(packed_gate_up)
        self.num_layers = int(num_layers)
        self.rank = int(rank)
        self.world_size = int(world_size)
        #: ``None`` is the host bank every configuration had before the resident mode, which is why
        #: it stays the default: the Turing boxes that need staging must not change behaviour
        #: because a big-memory box gained a mode.
        self.device = None if device is None else torch.device(device)
        # A resident shard is not pinned: it is not staged from, so there is no copy for pinning to
        # accelerate, and a pinned 14 GiB buffer would just hold locked host pages for nothing.
        self.pin_requested = bool(pin_memory) and self.device is None
        self.pinned = False
        self.resident_bytes = 0
        self.num_local_experts = 0
        self.block_size: tuple[int, int] | None = None
        self._gate_up: dict[int, torch.Tensor] = {}
        self._down: dict[int, torch.Tensor] = {}
        self._gate_up_scale: dict[int, torch.Tensor] = {}
        self._down_scale: dict[int, torch.Tensor] = {}
        self._loaded: set[int] = set()
        self._pin_fallback_reported = False

    @property
    def is_fp8(self) -> bool:
        return self.block_size is not None

    # -- placement --------------------------------------------------------

    def owns(self, expert_id: int) -> bool:
        return (int(expert_id) % self.world_size) == self.rank

    def local_index(self, expert_id: int) -> int:
        return int(expert_id) // self.world_size

    def local_expert_ids(self, num_experts: int) -> list[int]:
        return [e for e in range(int(num_experts)) if self.owns(e)]

    # -- allocation -------------------------------------------------------

    def _fall_back_to_pageable(self, exc: RuntimeError) -> None:
        """Convert already loaded layers before retrying the failed allocation."""
        self.pin_requested = False
        if not self._pin_fallback_reported:
            print(
                f"[qwen4exp] pinning the expert shard failed ({exc}); "
                "falling back to pageable host memory",
                flush=True,
            )
            self._pin_fallback_reported = True
        if self.pinned:
            # Do not leave a mixed shard behind: pageable clones replace all
            # earlier pinned layers before the current layer is allocated.
            for store in (self._gate_up, self._down, self._gate_up_scale, self._down_scale):
                for layer, tensor in list(store.items()):
                    store[layer] = tensor.clone()
            self.pinned = False
            import gc

            gc.collect()

    def _alloc_group(
        self,
        specs: list[tuple[tuple[int, ...], torch.dtype]],
    ) -> list[torch.Tensor]:
        """All of a layer's buffers, pinned together or pageable together.

        Allocating them one at a time could leave a layer half-pinned if the
        memlock limit is hit partway through, so a failure on any of them
        discards the whole group and the layer is retried as pageable.

        On a resident shard the group is allocated on ``self.device`` instead and
        ``pinned`` stays False: the flag means "these are page-locked host
        buffers", which is a claim about a host bank this shard is not.
        """
        if self.device is not None:
            return [torch.empty(shape, dtype=dtype, device=self.device) for shape, dtype in specs]
        if self.pin_requested:
            allocated: list[torch.Tensor] = []
            try:
                for shape, dtype in specs:
                    allocated.append(torch.empty(shape, dtype=dtype, pin_memory=True))
                self.pinned = True
                return allocated
            except RuntimeError as exc:  # memlock limit, most likely
                # Release the partial group first: holding pinned buffers here
                # would keep locked pages during the clones below.
                allocated.clear()
                self._fall_back_to_pageable(exc)
        return [torch.empty(shape, dtype=dtype) for shape, dtype in specs]

    def load_layer(
        self,
        layer_idx: int,
        gate_up_all: torch.Tensor,
        down_all: torch.Tensor,
    ) -> int:
        """Copy this rank's rows of one layer in; returns the bytes copied."""
        if layer_idx in self._loaded:
            return 0
        num_experts = gate_up_all.shape[0]
        if down_all.shape[0] != num_experts:
            raise ValueError(
                f"expert count mismatch in layer {layer_idx}: "
                f"gate_up {gate_up_all.shape[0]} vs down {down_all.shape[0]}"
            )
        ids = self.local_expert_ids(num_experts)
        gate_up, down = self._alloc_group(
            [
                ((len(ids), *gate_up_all.shape[1:]), gate_up_all.dtype),
                ((len(ids), *down_all.shape[1:]), down_all.dtype),
            ]
        )
        # Sequential in expert id, which is also sequential in file offset: the
        # backing store is a mechanical disk, so ordering matters more than
        # parallelism here.
        for local, expert_id in enumerate(ids):
            self._copy_row(gate_up[local], gate_up_all[expert_id])
            self._copy_row(down[local], down_all[expert_id])
        self._gate_up[layer_idx] = gate_up
        self._down[layer_idx] = down
        self._loaded.add(layer_idx)
        self.num_local_experts = len(ids)
        copied = gate_up.numel() * gate_up.element_size() + down.numel() * down.element_size()
        self.resident_bytes += copied
        return copied

    def load_layer_fp8(
        self,
        layer_idx: int,
        num_experts: int,
        expert_reader,
        *,
        block_size: tuple[int, int],
    ) -> int:
        """Copy this rank's FP8 rows of one layer in; returns the bytes copied.

        `expert_reader(expert_id)` yields `(gate_up, down)` as `FP8Tensor`s
        already packed into the runtime's `[2*inter, hidden]` convention.  The
        codes and the scale grids each get one contiguous buffer per projection,
        so a layer is four allocations no matter how many experts it holds.
        """
        if layer_idx in self._loaded:
            return 0
        ids = self.local_expert_ids(num_experts)
        if not ids:
            raise ValueError(
                f"rank {self.rank} of {self.world_size} owns no experts out of {num_experts}"
            )
        first_gate_up, first_down = expert_reader(ids[0])
        gate_up_spec = (
            _gate_up_plane_shape(first_gate_up.code.shape) if self.packed_gate_up else first_gate_up.code.shape
        )
        buffers = self._alloc_group(
            [
                ((len(ids), *gate_up_spec), first_gate_up.code.dtype),
                ((len(ids), *first_down.code.shape), first_down.code.dtype),
                ((len(ids), *first_gate_up.scale.shape), first_gate_up.scale.dtype),
                ((len(ids), *first_down.scale.shape), first_down.scale.dtype),
            ]
        )
        gate_up, down, gate_up_scale, down_scale = buffers
        for local, expert_id in enumerate(ids):
            pair = (first_gate_up, first_down) if local == 0 else expert_reader(expert_id)
            if self.packed_gate_up:
                pack_gate_up_plane(pair[0].code, gate_up[local], device=self.device)
            else:
                self._copy_row(gate_up[local], pair[0].code)
            self._copy_row(gate_up_scale[local], pair[0].scale)
            self._copy_row(down[local], pair[1].code)
            self._copy_row(down_scale[local], pair[1].scale)
        self._gate_up[layer_idx] = gate_up
        self._down[layer_idx] = down
        self._gate_up_scale[layer_idx] = gate_up_scale
        self._down_scale[layer_idx] = down_scale
        self._loaded.add(layer_idx)
        self.num_local_experts = len(ids)
        self.block_size = (int(block_size[0]), int(block_size[1]))
        copied = sum(t.numel() * t.element_size() for t in buffers)
        self.resident_bytes += copied
        return copied

    def _copy_row(self, destination: torch.Tensor, source: torch.Tensor) -> None:
        """One expert's row of one buffer into this shard's placement of it.

        The host bank's ``copy_`` is unchanged.  A resident shard crosses PCIe once per row,
        and does it with ``non_blocking=True``: this runs outside any ``torch.cuda.stream``
        context, so a blocking copy would be one synchronise per expert row -- 1536 of them a
        layer at TP16, which is the difference between a preload bounded by the link and one
        bounded by round-trip latency.  Pageable host memory still makes the driver stage it,
        so this is about not waiting on each row, not about overlapping it.

        Deliberately one hop: ``source.to(device, non_blocking=True)`` first and then a
        ``copy_`` into the destination would be two, and would allocate a temporary the size of
        an expert row per call for the caching allocator to churn through.
        """
        if self.device is None:
            destination.copy_(source)
            return
        destination.copy_(source, non_blocking=True)

    # -- access -----------------------------------------------------------

    def has_layer(self, layer_idx: int) -> bool:
        return layer_idx in self._loaded

    def rows(self, layer_idx: int, expert_id: int):
        """One expert's `(gate_up, down)`, in the row layout a GEMM takes.

        A `packed_gate_up` shard stores its gate/up codes as the fused kernel's planes -- one
        expert's codes spread over `hidden/16` chunks rather than laid out by row -- so indexing
        that table by local id answers with a plane slice, not with the expert's matrix.  The row
        form has to be read back out of it, which is what the per-expert backends consume.

        The whole layer is unpacked and then indexed, not one expert at a time: unpacking is a
        permute and a copy over the layer's contiguous code block, and a caller that wants one
        expert still pays for the layer.  Backends that read every expert per step (the grouped
        prefill, the resident loop) want `layer_tensors` anyway; this door is for the callers that
        want a single row.
        """
        local = self.local_index(expert_id)
        gate_up = self._gate_up[layer_idx]
        gate_up = unpack_gate_up_plane(gate_up) if self.packed_gate_up else gate_up
        gate_up = gate_up[local]
        down = self._down[layer_idx][local]
        if self.block_size is None:
            return gate_up, down
        return (
            FP8Tensor(gate_up, self._gate_up_scale[layer_idx][local], self.block_size),
            FP8Tensor(down, self._down_scale[layer_idx][local], self.block_size),
        )

    def layer_tensors(self, layer_idx: int):
        """This rank's contiguous local expert tensors, for grouped prefill.

        Returns `FP8Tensor`s for an FP8 shard, whose leading dimension is the
        local expert index rather than a matrix row. A shard that stored the planes
        (`packed_gate_up`) answers with the row layout read back out of them, which
        is what the prefill's grouped GEMM takes.
        """
        gate_up = self._gate_up[layer_idx]
        if self.packed_gate_up:
            gate_up = unpack_gate_up_plane(gate_up)
        down = self._down[layer_idx]
        if self.block_size is None:
            return gate_up, down
        return (
            FP8Tensor(gate_up, self._gate_up_scale[layer_idx], self.block_size),
            FP8Tensor(down, self._down_scale[layer_idx], self.block_size),
        )

    def gate_up_planes(self, layer_idx: int) -> torch.Tensor:
        """This rank's gate/up codes as the fused kernel's planes, `[local, ceil(hidden/16), 2*inter, 16]`.

        Only a `packed_gate_up` shard holds them; any other shard would have to build them, which is
        a second copy of its codes and is what the flag exists to avoid.
        """
        if not self.packed_gate_up:
            raise ValueError("this shard stores gate/up in the row layout, not as planes")
        return self._gate_up[layer_idx]

    def gate_up_plane_scale(self, layer_idx: int) -> torch.Tensor:
        """The scale grid that goes with `gate_up_planes`, without unpacking the codes.

        `layer_tensors` would answer the same scale, but it answers by unpacking the whole layer's
        gate/up codes back into rows first, and a caller that only wants the scale should not pay
        for that -- a decode step asks every layer, every step.
        """
        return self._gate_up_scale[layer_idx]

    def down_tensors(self, layer_idx: int) -> FP8Tensor | torch.Tensor:
        """This rank's down projection for `layer_idx`, without touching the gate/up codes.

        `layer_tensors` answers both, and on a `packed_gate_up` shard that means unpacking the
        gate/up planes back into rows -- a per-layer permute and copy of the whole layer.  A step
        that has already read the planes has no use for that copy, so this is the door for it.
        """
        down = self._down[layer_idx]
        if self.block_size is None:
            return down
        return FP8Tensor(down, self._down_scale[layer_idx], self.block_size)

    def stats(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "layers": len(self._loaded),
            "local_experts": self.num_local_experts,
            "resident_bytes": self.resident_bytes,
            "pinned": self.pinned,
            "fp8": self.is_fp8,
            "device": None if self.device is None else str(self.device),
        }


class Qwen4ExpCheckpoint:
    """Names and lazily reads Qwen4-Exp tensors from a checkpoint directory."""

    LM_PREFIX = "model.language_model"

    def __init__(
        self,
        root: str,
        *,
        store: MmapSafetensors | None = None,
        fp8: "FP8QuantSpec | None" = None,
    ) -> None:
        self.root = root
        self.store = store if store is not None else MmapSafetensors(root)
        self.expert_shard: HostExpertShard | None = None
        self.fp8 = fp8 if fp8 is not None else self._detect_fp8()

    def _detect_fp8(self) -> "FP8QuantSpec | None":
        """Read `quantization_config` from the checkpoint's own `config.json`.

        Detection is by config, not by tensor naming: the config is what declares
        the block size and the skip list, and a checkpoint that names tensors one
        way but scales them another would be a packaging bug we want to surface
        rather than guess around.
        """
        from relicllm.models.qwen4_exp.config import FP8QuantSpec

        path = os.path.join(self.root, "config.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            raw = json.load(f)
        return FP8QuantSpec.from_dict(raw.get("quantization_config"))

    @property
    def is_fp8(self) -> bool:
        return self.fp8 is not None

    # -- naming -----------------------------------------------------------

    def layer_prefix(self, layer_idx: int) -> str:
        return f"{self.LM_PREFIX}.layers.{layer_idx}"

    def expert_key(self, layer_idx: int, which: str) -> str:
        return f"{self.layer_prefix(layer_idx)}.mlp.experts.{which}"

    def fp8_expert_key(self, layer_idx: int, expert_id: int, proj: str) -> str:
        """Name of one FP8 expert projection (`gate`, `up` or `down`)."""
        return f"{self.expert_key(layer_idx, str(int(expert_id)))}.{proj}_proj.weight"

    # -- routed experts ---------------------------------------------------

    def fp8_expert_rows(
        self, layer_idx: int, expert_id: int
    ) -> tuple[FP8Tensor, FP8Tensor]:
        """One FP8 expert straight from the mapping, packed like the BF16 layout.

        The checkpoint stores `gate_proj` and `up_proj` separately; the runtime
        expects them row-concatenated as `[2*inter, hidden]` with gate first,
        which is the order the BF16 build ships and which `swiglu_expert`'s
        `chunk(2)` assumes.
        """
        if self.fp8 is None:
            raise ValueError("checkpoint is not FP8; use expert_rows instead")
        block = self.fp8.block_size

        def read(proj: str) -> FP8Tensor:
            return FP8Tensor(*self.fp8_expert_projection(layer_idx, expert_id, proj), block)

        return pack_gate_up_fp8(read("gate"), read("up")), read("down")

    def fp8_expert_projection(
        self, layer_idx: int, expert_id: int, proj: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One expert projection's `(codes, scale_inv)` views of the mapping.

        Unpacked, and served uncached: a reader that wants a rank's rows of one layer holds all of
        them at once, and `view` would keep a handle on each of the 150k expert tensors it looked at.
        """
        if self.fp8 is None:
            raise ValueError("checkpoint is not FP8; use expert_rows instead")
        key = self.fp8_expert_key(layer_idx, expert_id, proj)
        return self.store.entry_view(self.store.entry(key)), self.store.entry_view(
            self.store.entry(f"{key}_scale_inv")
        )

    def fp8_expert_keys(self, layer_idx: int, expert_id: int) -> list[str]:
        """Every mapped tensor name backing one FP8 expert (codes and scales)."""
        keys = []
        for proj in ("gate", "up", "down"):
            key = self.fp8_expert_key(layer_idx, expert_id, proj)
            keys.extend((key, f"{key}_scale_inv"))
        return keys

    def preload_experts(
        self,
        num_layers: int,
        *,
        num_experts: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        layer_range: tuple[int, int] | None = None,
        pin: bool = True,
        progress: bool = False,
        barrier=None,
        release_mapping: bool = True,
        device: torch.device | str | None = None,
        packed_gate_up: bool = False,
    ) -> HostExpertShard:
        """Copy this rank's routed experts into host RAM and serve them from there.

        Called once at load time.  Afterwards `expert_rows` no longer touches the
        mapping for owned experts, so steady-state staging reads RAM instead of
        risking a disk seek.  Foreign experts are never loaded; under TP they are
        another rank's responsibility and `ShardedMoE` masks them out.

        `barrier` (when given) is called after each layer.  The ranks interleave
        rows within one layer, so keeping them on the same layer keeps the
        mechanical disk's head in one region instead of letting four processes
        seek across the whole 225 GiB expert set independently.

        `release_mapping` drops each layer's mapped pages from this process once
        it has been copied.  Without it a rank ends up holding its 56 GiB copy
        *and* 56 GiB of mapped expert pages (measured RSS 113 GiB), which at four
        ranks is most of the machine's RAM for no benefit.

        `num_experts` is only consulted for an FP8 checkpoint, where the expert
        count cannot be read off a packed tensor's leading dimension.

        `device` names where the shard *lands*.  Left None it is the host bank
        this method has always filled; given a device, the same rows are filled
        into device buffers instead and `pin` is ignored, which is what makes
        this rank's experts resident for the whole run rather than a source to
        stage from.  The layer loop, the barrier and the `advise_dontneed` are
        unchanged either way -- releasing the mapped pages matters more, not
        less, when the copy that just read them went to the card.
        """
        shard = HostExpertShard(
            num_layers=num_layers,
            rank=rank,
            world_size=world_size,
            pin_memory=pin,
            device=device,
            packed_gate_up=packed_gate_up,
        )
        # Only this stage's layers under pipeline parallelism. Without it a rank would preload an
        # expert set for layers it never runs -- at PP16 that is the whole 112 GiB of host RAM per
        # rank instead of its 7 GiB share, which is the difference between fitting and not.
        start, end = layer_range if layer_range is not None else (0, num_layers)
        for layer_idx in range(start, min(end, num_layers)):
            if self.fp8 is not None:
                self._preload_layer_fp8(
                    shard,
                    layer_idx,
                    num_experts=num_experts,
                    barrier=barrier,
                    release_mapping=release_mapping,
                )
            else:
                self._preload_layer_packed(
                    shard,
                    layer_idx,
                    barrier=barrier,
                    release_mapping=release_mapping,
                )
            # Drop local aliases before the next layer.  The mmap views remain
            # cached by the store, but no temporary packed/FP8 row assembly is
            # retained by the preload loop.
            import gc

            gc.collect()
            if progress:
                gib = shard.resident_bytes / (1 << 30)
                print(
                    f"[qwen4exp] rank {rank}: experts resident through layer "
                    f"{layer_idx} ({gib:.2f} GiB)",
                    flush=True,
                )
        self.expert_shard = shard
        return shard

    def _preload_layer_packed(
        self,
        shard: HostExpertShard,
        layer_idx: int,
        *,
        barrier,
        release_mapping: bool,
    ) -> None:
        gate_up_key = self.expert_key(layer_idx, "gate_up_proj")
        down_key = self.expert_key(layer_idx, "down_proj")
        shard.load_layer(layer_idx, self.store.view(gate_up_key), self.store.view(down_key))
        if barrier is not None:
            # Siblings finish the same layer before anyone drops its pages, so
            # MADV_DONTNEED never pulls a region a sibling is still reading.
            barrier()
        if release_mapping:
            self.store.advise_dontneed(gate_up_key)
            self.store.advise_dontneed(down_key)

    def _preload_layer_fp8(
        self,
        shard: HostExpertShard,
        layer_idx: int,
        *,
        num_experts: int | None,
        barrier,
        release_mapping: bool,
    ) -> None:
        assert self.fp8 is not None
        if num_experts is None:
            raise ValueError(
                "preload_experts needs num_experts for an FP8 checkpoint: the count "
                "is not recoverable from a packed tensor's shape"
            )
        shard.load_layer_fp8(
            layer_idx,
            int(num_experts),
            lambda expert_id: self.fp8_expert_rows(layer_idx, expert_id),
            block_size=self.fp8.block_size,
        )
        if barrier is not None:
            barrier()
        if release_mapping:
            # 6 keys per expert rather than 2 per layer, so this is the one place
            # where the FP8 layout costs materially more bookkeeping.
            for expert_id in shard.local_expert_ids(int(num_experts)):
                for key in self.fp8_expert_keys(layer_idx, expert_id):
                    self.store.advise_dontneed(key)

    def expert_rows(self, layer_idx: int, expert_id: int):
        """One expert's `(gate_up, down)` as host tensors (no copy, no device transfer).

        Served from the resident shard when this rank owns the expert and the
        shard is loaded; otherwise from the mapping, which is both the
        `--mmap-experts` diagnostic path and what the tiny test fixtures use.

        Returns `FP8Tensor`s for an FP8 checkpoint; `HostExpertMoE` dequantizes.
        """
        shard = self.expert_shard
        if shard is not None and shard.has_layer(layer_idx) and shard.owns(expert_id):
            return shard.rows(layer_idx, expert_id)
        if self.fp8 is not None:
            return self.fp8_expert_rows(layer_idx, expert_id)
        gate_up = self.store.view(self.expert_key(layer_idx, "gate_up_proj"))[expert_id]
        down = self.store.view(self.expert_key(layer_idx, "down_proj"))[expert_id]
        return gate_up, down

    # -- PLE n-gram table -------------------------------------------------

    def ngram_shard_keys(self, layer_idx: int) -> list[str]:
        prefix = f"{self.layer_prefix(layer_idx)}.ple.ple_embedding.ngram_embedding.shard_"
        keys = [k for k in self.store.keys() if k.startswith(prefix)]
        return sorted(keys, key=lambda k: int(k.rsplit("_", 1)[1].split(".")[0]))

    def ngram_scale(self, layer_idx: int) -> torch.Tensor | None:
        """The FP8 n-gram table's single shared scale, or None when unquantized.

        Unlike the experts, the table is quantized per tensor: all 128 shards
        share one scalar, stored once next to them.
        """
        key = f"{self.layer_prefix(layer_idx)}.ple.ple_embedding.ngram_embedding.weight_scale"
        if key not in self.store:
            return None
        return self.store.view(key).reshape(())


class HostNGramTable:
    """Row lookup into the sharded PLE embedding, kept in host RAM.

    Shards are equal-height slices of one logical `(total_rows, head_dim)` table,
    so a row id maps to `(shard, row_in_shard)` by integer division.  Gathering
    on the host keeps 95 GiB off the GPU; only the gathered rows cross PCIe.

    An FP8 table halves that to 47.7 GiB.  Its 128 shards share one scalar scale,
    so the gather is done on the raw codes and the scale is applied once on the
    device, after the (much smaller) gathered rows have crossed PCIe.
    """

    def __init__(
        self,
        shards: list[torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
        scale: torch.Tensor | None = None,
    ) -> None:
        assert shards, "PLE table needs at least one shard"
        self.shards = shards
        self.rows_per_shard = shards[0].shape[0]
        self.head_dim = shards[0].shape[1]
        self.device = device
        self.dtype = dtype
        self.total_rows = sum(s.shape[0] for s in shards)
        self.is_fp8 = shards[0].dtype is torch.float8_e4m3fn
        if self.is_fp8 and scale is None:
            raise ValueError("an FP8 n-gram table needs its weight_scale")
        # The scale is one scalar and every step multiplies by it, so it is moved once here rather
        # than per call: a `.to(device)` inside a step is a host-to-device copy, and under a
        # capture an unpinned one is refused outright.
        self.scale = scale.to(device) if scale is not None else None
        # `index_select` has no float8 CPU kernel, so gather over a byte alias of
        # the same storage and reinterpret afterwards.  This is a view, not a copy.
        self._gather_shards = (
            [s.view(torch.uint8) for s in shards] if self.is_fp8 else shards
        )
        self._gather_dtype = torch.uint8 if self.is_fp8 else shards[0].dtype
        #: The pinned host buffer the gathered rows are written into; see `_host_rows`.
        self._rows_buffer: torch.Tensor | None = None

    def __call__(self, row_ids: torch.Tensor) -> torch.Tensor:
        """row_ids: (..., ngram_heads) -> (..., ngram_heads, head_dim) on device."""
        rows = int(row_ids.numel() // row_ids.shape[-1]) * int(row_ids.shape[-1])
        flat = row_ids.reshape(-1).to("cpu", dtype=torch.long)
        out = self._host_rows(rows)
        self._gather_into(flat, out)
        return self._to_device(out).reshape(*row_ids.shape, self.head_dim)

    def _host_rows(self, rows: int) -> torch.Tensor:
        """The host buffer the gathered rows land in, pinned and kept across calls.

        A decode step reads `ngram_heads` rows, several kilobytes, and the crossing back to the
        device is the only part of this table a step pays for.  Pinned memory makes that crossing
        one copy instead of a staged one, and keeping the buffer makes its address the same every
        step -- which is what lets a captured graph replay the copy at all.
        """
        buffer = self._rows_buffer
        if buffer is None or buffer.shape[0] < rows:
            with torch.inference_mode(False):
                buffer = torch.empty(rows, self.head_dim, dtype=self._gather_dtype,
                                     pin_memory=True)
            self._rows_buffer = buffer
        return buffer[:rows]

    def _gather_into(self, flat: torch.Tensor, out: torch.Tensor) -> None:
        """Fill `out` with the rows `flat` names, shard by shard."""
        shard_idx = torch.div(flat, self.rows_per_shard, rounding_mode="floor")
        local_idx = flat - shard_idx * self.rows_per_shard
        for s in shard_idx.unique().tolist():
            picks = (shard_idx == s).nonzero(as_tuple=True)[0]
            if s >= len(self._gather_shards):
                # Row lands in the divisor padding past the real table; upstream
                # leaves those rows at their (zero) init value.
                out[picks] = 0
                continue
            out[picks] = self._gather_shards[s].index_select(0, local_idx[picks])

    def _to_device(self, out: torch.Tensor) -> torch.Tensor:
        if self.is_fp8:
            codes = out.to(self.device, non_blocking=True).view(torch.float8_e4m3fn)
            return fp8_scalar_dequantize(codes, self.scale, self.dtype)
        return out.to(self.device, dtype=self.dtype, non_blocking=True)


class HostEmbedding:
    """Token embedding gathered on the host, moved to device per step.

    ``weight`` is ``None`` on a pipeline stage that does not own the embedding; calling it there is a
    programming error rather than a shape problem, so it says so instead of raising a ``TypeError``
    several frames later.
    """

    def __init__(
        self, weight: torch.Tensor | None, *, device: torch.device, dtype: torch.dtype
    ) -> None:
        self.weight = weight
        self.device = device
        self.dtype = dtype

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError(
                "this pipeline stage does not own the token embedding; only the first stage embeds"
            )
        flat = token_ids.reshape(-1).to("cpu", dtype=torch.long)
        rows = self.weight.index_select(0, flat)
        return rows.to(self.device, dtype=self.dtype).reshape(*token_ids.shape, self.weight.shape[1])


class DeviceEmbedding:
    """Token embedding resident on the compute device."""

    def __init__(self, weight: torch.Tensor) -> None:
        self.weight = weight

    def __call__(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(token_ids.to(self.weight.device), self.weight)
