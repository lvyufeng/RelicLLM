"""A bounded Ascend store for the routed experts, executing their original packed FP4 weights.

The released checkpoint stores a routed expert as three packed E2M1 matrices beside a per-row E8M0
grid, and the port has two ways to run them: expand each matrix to a dense BF16 weight on the host
(what `CheckpointRoutedExperts` does, one 67.5 MiB dense expert at a time), or execute the checkpoint
bytes where they are. This module is the second one for Ascend, and it is deliberately built out of
parts that are already accepted rather than new kernel work:

* the weight side is the frozen `online_unpack.fp4_expert_int8`: every nibble widened to **twice**
  its E2M1 value in INT8, with the file's own E8M0 grid decoded to FP32 and halved, so
  ``int8 * (scale / 2)`` is the checkpoint's floating weight and the compensation is applied exactly
  once. Nothing is requantized, no dense weight is built, and no byte of the checkpoint is rewritten;
* the activation side is the released kernel's own quantizer, ``block_fp8_quant.quantize_block_fp8``
  (block 32, E8M0, amax floor ``1e-4``), producing the e4m3 bytes and E8M0 bytes the reference's
  ``act_quant`` produces -- never an INT8 activation, which is the CUDA device path's documented
  divergence and not this rule;
* the reduction is ``block_gemm.fp4_int8_block_gemm``: the reference's own two-half, truncate-toward-
  zero, window-13 block rule (the operator's own docstring says "a reference reduction, not native
  FP4/FP8 hardware arithmetic"), with the same ``(fragment * s_a) * s_b`` scale order applied once per
  block and the same sequential FP32 accumulation in K order across blocks.

What this buys, and what it does not
------------------------------------
Per expert the store retains **37.97 MiB** -- 33.75 MiB of INT8 codes plus 4.22 MiB of FP32
half-scales -- instead of the 67.5 MiB dense BF16 widening, and it never materialises the dense form
at all. (Arithmetic on the released shape: `w1`/`w3` are `[2304, 5120]`, `w2` is `[5120, 2304]`, and
their grids are `[2304, 160]`, `[2304, 160]` and `[5120, 72]` of FP32.) That is the whole reason it
is worth having: one layer's 384 experts are **14.24 GiB in this form and 25.31 GiB dense**, so a
layer does not fit either way, and a 40-layer model fits neither. The store is therefore a
**byte-budgeted LRU** whose default budget (512 MiB) is shared by every store in the process -- about
13 resident experts -- so 40 layers degrade into one cache rather than into 40 private caches that
each believe they are alone.

What the arithmetic is, and what that does not establish
--------------------------------------------------------
The operands are the reference's and the reduction is the reference's two-half rule, implemented as
a vectorised FP32 transcription: two sequential 16-wide halves, each cut toward zero onto a
power-of-two grid anchored by its own largest nominal exponent, the first half's sum joining the
second as a peer, and one renormalisation at the end. `block_gemm`'s tensor path and
`mma.fp8_block_gemm` are two independent transcriptions of that rule, and on the CPU fixtures in
`tests/test_v41_ascend_experts.py` this store's projection path agrees with the port's `mma` path
**bit for bit** at FP32 -- which is evidence that both spell the rule the same way, not that either
matches a card.

So what is **not** claimed: that this reproduces the reference on hardware (the port's own parity
measurement belongs to `mma`'s record, not to this store), that it is native FP4/FP8 arithmetic (the
operands are integers and bytes and the reduction is FP32), or that a bounded model run works. The
one thing this file does claim is the operand mapping and the order around it: exact INT8-times-two
values with the file's own grid halved, the released activation quantizer, and the reference's
SwiGLU/routing placement.

Order and placement, which is where an expert store is easiest to get wrong
--------------------------------------------------------------------------
`modules.expert_forward` is the contract, and every asymmetry in it is reproduced:

1. ``w1`` and ``w3`` are two *separate* linears over the same activation, so each quantizes it on its
   own (the quantizer is deterministic, so the two byte tensors agree, but the call structure is the
   reference's and is not folded into one);
2. each FP32 accumulator is narrowed to the store's dtype and then widened for the clamp -- the
   reference's GEMM outputs BF16 and the clamp runs in FP32 on that narrowed value;
3. the clamp is applied in FP32 **before** ``silu``, and ``swiglu_limit`` is 10.0 on this release
   with expert activations that exceed it, so clamping in BF16 would round a value about to be
   clamped;
4. the routing weight multiplies the **intermediate** ``[rows, inter]`` activation, not the expert's
   output;
5. the ``w2`` operand is the BF16 cast of that intermediate, quantized *after* the cast;
6. the ``w2`` result is narrowed to the store's dtype and enters the layer's FP32 routed sum, which
   walks experts in **ascending id order** exactly like the reference and the other two stores.

Expert parallelism: `partial` and the deal
------------------------------------------
``world``/``rank`` deal whole experts cyclically (``expert % world == rank``), which is the same rule
`device_experts.deal_rule` uses for the ``id`` layout, and ``partial`` is an **instance** attribute
computed once in ``__init__``: with ``world > 1`` a store returns one rank's share of the routed sum
and `MoE.forward` joins it with the shared expert on a single all-reduce, while ``world == 1`` returns
the whole routed sum. A store that reports the wrong one multiplies the routed term by ``world`` and
still looks plausible, so the deal and the flag are one decision made in one place. Nothing here cuts
a weight tensor: the tensor-parallel plan cuts *parameters*, and the routed experts are not
parameters, so their only split is this deal.

What is not claimed
-------------------
No graph capture (the routing indices are read on the host once per call -- one synchronization, and
graphs are not claimed), no batched multi-token expert kernel, no NPU GEMM inside a quantized
``int8 x int8`` unit (the reduction consumes an e4m3 activation and INT8-times-two weights as
integers in FP32, which is a *calculation*, not a hardware path), no whole-model capability: the
``v41`` Ascend capability guard is untouched, and an executor for the routed experts is one of
several things a bounded run still needs. Sharing the two ``w1``/``w3`` quantizations is a legal
optimization (the quantizer is deterministic) and is *not* taken here, because a store is the wrong
place to change a semantic the reference states.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import os
import weakref

import torch
import torch.nn.functional as F

from .online_unpack import Fp4ExpertInt8, fp4_expert_int8

__all__ = [
    "AscendExpertCache",
    "AscendRoutedExperts",
    "DEFAULT_CACHE_BYTES",
    "ResidentExpert",
    "expert_forward_ascend",
]

#: The reference's block width along K (`fp8_block_size == block_K == 32`). Both the activation
#: quantizer and the weight grid are stated on it, so every matrix dimension here is a multiple.
BLOCK = 32

#: Default byte budget of the process-wide expert cache: **512 MiB shared by every store**, not a
#: per-layer count. One released expert is 37.97 MiB in this representation, so this is about 13
#: resident experts -- enough for a decode step's handful of experts with room for the next step, and
#: small enough that 40 layers cannot turn it into 40 private caches. A caller that wants a different
#: trade-off (or a private cache) injects one; nothing else in this file reads a second constant.
DEFAULT_CACHE_BYTES = 512 * 1024 * 1024

_PROJECTIONS = ("w1", "w2", "w3")
_BACKENDS = frozenset({"auto", "torch", "triton"})
_DEVICE_TYPES = frozenset({"cpu", "npu"})
_FLOAT_DTYPES = frozenset({torch.bfloat16, torch.float32})
_DEFAULT_DTYPE = torch.bfloat16

#: One cache per process, created on first use. Two stores share it unless one was handed another,
#: which is what makes the budget a *global* bound rather than a per-layer one.
_GLOBAL_CACHE: "AscendExpertCache | None" = None


def _core():
    """The byte-level operators, imported where they are used.

    Lazy for the same reason `online_unpack._core` is: on a host whose installed relic-core predates
    these operators, importing this module has to stay possible so the legacy path can still load.
    """
    from relic_core.kernels import block_fp8_quant, block_gemm

    return block_fp8_quant, block_gemm


def global_cache() -> "AscendExpertCache":
    """The process-wide cache every store uses unless it was given one.

    `load_backbone` may hand all of a model's stores a single instance explicitly -- the budget is
    the same either way, but one object is easier to report on. This accessor exists so a caller that
    does not care does not have to build one.
    """
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = AscendExpertCache()
    return _GLOBAL_CACHE


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


@dataclass(frozen=True)
class ResidentExpert:
    """One expert's three projections, widened exactly, resident on one device.

    Each projection is a `Fp4ExpertInt8`: INT8 values (``2 * e2m1``) beside the file's E8M0 grid
    halved to FP32. The two are only meaningful together -- halving the scale *is* the compensation
    for doubling the value -- so they travel as one object and are never passed to the GEMM apart.

    ``block`` is the K block width the grid was written on (32 on this release). It is carried rather
    than assumed because a grid read on the wrong width is a silent mis-scale, and because the GEMM
    checks the shapes it is handed rather than trusting the caller.
    """

    w1: Fp4ExpertInt8
    w2: Fp4ExpertInt8
    w3: Fp4ExpertInt8
    block: int

    @property
    def nbytes(self) -> int:
        """Resident bytes: the INT8 codes plus the FP32 half-scale grids, nothing else."""
        total = 0
        for projection in (self.w1, self.w2, self.w3):
            total += _tensor_nbytes(projection.values)
            total += _tensor_nbytes(projection.half_scale)
        return total


class AscendExpertCache:
    """A byte-budgeted LRU of widened experts, shared by every layer of a process.

    The budget is in **bytes rather than experts** because the three projections are not the same
    size (``w1``/``w3`` are ``[inter, dim]``, ``w2`` is ``[dim, inter]``, and the FP32 grids are not
    proportional to the INT8 payload), and because a count would let a larger model configuration
    silently exceed the memory it was sized for.

    Keys are opaque tuples that the stores build, and they **must** carry the checkpoint's identity,
    the layer id and the device as well as the expert id. The measured bug `loader.py:1050-1057`
    records is exactly what keying on the expert id alone does: layer 6 answered with layer 5's
    bytes. Two devices are two entries for the same reason -- a tensor resident on ``npu:0`` is not
    usable by a store on ``npu:1``.

    An expert whose own bytes exceed the budget is **streamed and not retained**: it is loaded, used
    for the call that asked for it, and dropped, without evicting anything else to make room. That is
    the behaviour a tiny budget needs to stay correct rather than to fit a size it cannot.
    """

    def __init__(self, budget_bytes: int = DEFAULT_CACHE_BYTES, *, name: str = "ascend-experts") -> None:
        if isinstance(budget_bytes, bool) or not isinstance(budget_bytes, int):
            raise TypeError("the expert cache budget is a byte count, an int")
        if budget_bytes < 0:
            raise ValueError("the expert cache budget cannot be negative")
        self.budget_bytes = budget_bytes
        self.name = name
        self._entries: "OrderedDict[tuple, tuple[ResidentExpert, int]]" = OrderedDict()
        self._bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.streamed = 0

    # -- what a caller reads --------------------------------------------------

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def retained_bytes(self) -> int:
        return self._bytes

    @property
    def entries(self) -> int:
        return len(self._entries)

    def snapshot(self) -> dict:
        """Counters and occupancy, as one plain dict a caller can log or assert on."""
        accesses = self.hits + self.misses
        return {
            "name": self.name,
            "budget_bytes": self.budget_bytes,
            "retained_bytes": self._bytes,
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "streamed": self.streamed,
            "hit_rate": (self.hits / accesses) if accesses else None,
        }

    # -- the cache itself -----------------------------------------------------

    def get(self, key: tuple) -> ResidentExpert | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self.hits += 1
        self._entries.move_to_end(key)
        return entry[0]

    def put(self, key: tuple, resident: ResidentExpert, nbytes: int) -> bool:
        """Retain `resident`, or report that it was too large to retain.

        Returns ``True`` when the entry is now resident and ``False`` when it was streamed. The
        caller keeps using the object either way -- this decides only whether the *next* call can
        skip the decode.
        """
        size = int(nbytes)
        if size <= 0:
            raise ValueError("a resident expert's byte size must be positive")
        previous = self._entries.pop(key, None)
        if previous is not None:
            self._bytes -= previous[1]
        if self.budget_bytes <= 0 or size > self.budget_bytes:
            # Bigger than the whole budget: keeping it would evict everything and then still not fit.
            self.streamed += 1
            return False
        self._entries[key] = (resident, size)
        self._bytes += size
        while self._bytes > self.budget_bytes:
            _, (_, freed) = self._entries.popitem(last=False)
            self._bytes -= freed
            self.evictions += 1
        return True

    def clear(self) -> int:
        """Drop every entry in this cache object and return how many were held.

        This is local cleanup of one object's tensors, not a process-wide resource reset: another
        cache, another store's entries in a shared cache, and the device allocator are all untouched.
        """
        dropped = len(self._entries)
        self._entries.clear()
        self._bytes = 0
        return dropped

    def clear_prefix(self, prefix: tuple) -> int:
        """Drop the entries whose key starts with `prefix` -- one store's share of a shared cache."""
        width = len(prefix)
        doomed = [key for key in self._entries if key[:width] == prefix]
        for key in doomed:
            _, freed = self._entries.pop(key)
            self._bytes -= freed
        return len(doomed)

    def entries_for(self, prefix: tuple) -> int:
        width = len(prefix)
        return sum(1 for key in self._entries if key[:width] == prefix)


def _checkpoint_identity(checkpoint) -> tuple:
    """A cache-key token for a checkpoint, and the weak reference that retires it.

    A real `V41Checkpoint` has a ``root``: a path identifies the file set, so two loads of the same
    directory share entries and two different directories never do. A checkpoint object without a
    root (a fixture, or a future wrapper) is identified by its object identity, which is only safe
    while that object is alive -- so a finalizer purges the entries when it dies rather than letting
    a recycled ``id`` serve another checkpoint's bytes.
    """
    root = getattr(checkpoint, "root", None)
    if isinstance(root, (bytes, os.PathLike)):
        root = os.fspath(root)
    if isinstance(root, str) and root:
        return ("path", os.path.realpath(root)), None
    try:
        reference = weakref.ref(checkpoint)
    except TypeError:  # a checkpoint that cannot be weak-referenced keeps its entries for the process
        return ("object", id(checkpoint)), None
    return ("object", id(checkpoint)), reference


def _device_key(device: torch.device) -> tuple:
    return (device.type, device.index)


class AscendRoutedExperts:
    """One layer's routed experts executed from the checkpoint's original packed FP4 bytes.

    An interface implementation rather than an `nn.Module`: like the other two stores, it holds no
    `Parameter`, and the layer's experts must not appear in ``state_dict()``. `MoE` reads ``partial``
    off it and calls ``forward``; nothing else about it is part of the tree.

    Construction is cheap and does no I/O: the first ``expert()``/``forward()`` reads bytes and
    decodes them, so a run that never routes an expert pays nothing. NPU construction or execution
    errors propagate by name; unlike the CUDA arena, this provider does not silently fall back to
    the host store and change the execution device.

    ``device`` is the device the experts live on, and it is also the device the activation must be on.
    ``dtype`` is the width the two GEMMs narrow to and the activation is read at; it must be the
    tree's activation width (`LINEAR_DTYPE`, bf16 on this release), not a per-store choice.
    ``backend`` selects the operator implementation: ``triton`` (the fused Ascend path, NPU only),
    ``torch`` (the same-device reference, CPU only -- this is the oracle fixture mode), or ``auto``,
    which resolves to the device's one legal choice. There is no silent substitution: a compiler
    error or a missing backend raises instead of running the other one, because a store that quietly
    changes its arithmetic is a store whose numbers cannot be compared to anything.
    """

    def __init__(
        self,
        checkpoint,
        layer_id: int,
        *,
        n_experts: int,
        dim: int,
        inter_dim: int,
        topk: int,
        swiglu_limit: float = 0.0,
        device: torch.device | str,
        dtype: torch.dtype = _DEFAULT_DTYPE,
        rank: int = 0,
        world: int = 1,
        cache: AscendExpertCache | None = None,
        backend: str = "triton",
    ) -> None:
        self.checkpoint = checkpoint
        self.layer_id = int(layer_id)
        self.n_experts = _positive_int(n_experts, "n_experts")
        self.dim = _block_multiple(dim, "dim")
        self.inter_dim = _block_multiple(inter_dim, "inter_dim")
        self.topk = _positive_int(topk, "topk")
        if self.topk > self.n_experts:
            raise ValueError(
                f"topk {self.topk} exceeds the {self.n_experts} experts it has to choose from"
            )
        self.swiglu_limit = _non_negative_float(swiglu_limit, "swiglu_limit")
        self.device = _supported_device(device)
        self.dtype = _supported_dtype(dtype)
        if isinstance(world, bool) or not isinstance(world, int) or world < 1:
            raise ValueError(f"world must be a positive int, got {world!r}")
        if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world:
            raise ValueError(f"rank must be an int in [0, {world}), got {rank!r}")
        self.rank = rank
        self.world = world
        self.backend = _resolve_backend(backend, self.device)
        if cache is not None and not isinstance(cache, AscendExpertCache):
            raise TypeError("cache must be an AscendExpertCache or None")
        self.cache = global_cache() if cache is None else cache
        # `partial` is decided once, here: the deal and the flag are the same decision, and
        # `MoE.forward` reads it to decide whether the shared expert joins the routed sum on one
        # all-reduce or is reduced on its own.
        self.partial = world > 1
        self.misses = 0
        identity, retire = _checkpoint_identity(checkpoint)
        self._identity = identity
        self._prefix = (identity, self.layer_id, _device_key(self.device))
        if retire is not None:
            cache_ref, prefix = self.cache, self._prefix
            weakref.finalize(checkpoint, cache_ref.clear_prefix, prefix)

    # -- identity helpers -----------------------------------------------------

    def _key(self, expert: int, which: str) -> str:
        return f"layers.{self.layer_id}.ffn.experts.{expert}.{which}.weight"

    def _scale_key(self, expert: int, which: str) -> str:
        # The pairing rule lives in the loader (`scale_key`); importing it lazily keeps this module
        # importable without the loader, which is what lets `load_backbone` import it in either order
        # without a cycle.
        from .loader import scale_key

        return scale_key(self._key(expert, which))

    def _cache_key(self, expert: int) -> tuple:
        return (*self._prefix, expert)

    def _owns(self, expert: int) -> bool:
        return expert % self.world == self.rank

    def __repr__(self) -> str:
        return (
            f"AscendRoutedExperts(layer={self.layer_id}, experts={self.n_experts}, "
            f"topk={self.topk}, device={self.device}, dtype={self.dtype}, backend={self.backend}, "
            f"rank={self.rank}/{self.world}, budget_bytes={self.cache.budget_bytes})"
        )

    # -- reporting ------------------------------------------------------------

    @property
    def cache_stats(self) -> dict:
        """The shared cache's counters. A process-wide view: read it, do not infer this layer's."""
        return self.cache.snapshot()

    @property
    def resident_experts(self) -> int:
        """How many of *this* store's experts are in the shared cache right now."""
        return self.cache.entries_for(self._prefix)

    def clear_cache(self) -> int:
        """Drop this store's entries from the shared cache; other layers' entries stay."""
        return self.cache.clear_prefix(self._prefix)

    # -- experts --------------------------------------------------------------

    def expert(self, expert: int) -> ResidentExpert:
        """One expert's three projections, widened exactly and resident on the store's device.

        A hit returns the cached object; a miss reads the file's packed bytes and the file's E8M0
        grid through the frozen `fp4_expert_int8`, which stages bytes to the device and decodes
        there. Nothing is expanded to a dense weight on either path.
        """
        expert = self._expert_id(expert)
        if not self._owns(expert):
            raise ValueError(
                f"expert {expert} is not owned by rank {self.rank} of {self.world}; the deal is "
                f"expert % world, so this rank owns {sorted(self._owned_experts())[:8]}..."
            )
        key = self._cache_key(expert)
        resident = self.cache.get(key)
        if resident is not None:
            return resident
        self.misses += 1
        resident = self._load(expert)
        self.cache.put(key, resident, resident.nbytes)
        return resident

    def _load(self, expert: int) -> ResidentExpert:
        projections = {}
        for which in _PROJECTIONS:
            key = self._key(expert, which)
            projections[which] = fp4_expert_int8(
                self.checkpoint,
                key,
                self._scale_key(expert, which),
                device=self.device,
                backend=self.backend,
            )
        block = projections["w1"].block
        if any(projection.block != block for projection in projections.values()):
            raise ValueError(
                f"expert {expert} of layer {self.layer_id} mixes scale block widths "
                f"{[projection.block for projection in projections.values()]}; the three "
                "projections of an expert are written on one grid"
            )
        expected = {
            "w1": (self.inter_dim, self.dim),
            "w2": (self.dim, self.inter_dim),
            "w3": (self.inter_dim, self.dim),
        }
        for which, shape in expected.items():
            actual = tuple(projections[which].values.shape)
            if actual != shape:
                raise ValueError(
                    f"{self._key(expert, which)} widens to {actual}; this store was built for a "
                    f"{shape[0]}x{shape[1]} matrix ({which} is [out, in]), so it does not describe "
                    "the layer it was attached to"
                )
        return ResidentExpert(
            w1=projections["w1"], w2=projections["w2"], w3=projections["w3"], block=block
        )

    def _owned_experts(self):
        for expert in range(self.n_experts):
            if self._owns(expert):
                yield expert

    # -- the forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Run one layer's routed experts. Returns the routed sum in FP32, `[n, dim]`.

        ``x`` is the activation this store's dtype and device, ``weights``/``indices`` are the gate's
        ``[n, topk]``. The index tensor is read on the host **once** per call -- one synchronization,
        which is why this path is not graph-capturable and does not claim to be -- and the experts
        are then walked in ascending id order, each contribution added into one FP32 accumulator.
        """
        self._check_forward_inputs(x, weights, indices)
        n = int(x.shape[0])
        if n == 0:
            return torch.zeros((0, self.dim), dtype=torch.float32, device=self.device)
        buckets = self._routing(indices)
        y = torch.zeros((n, self.dim), dtype=torch.float32, device=self.device)
        for expert in sorted(buckets):
            if not self._owns(expert):
                continue
            rows, slots = buckets[expert]
            row_index = torch.tensor(rows, dtype=torch.int64, device=self.device)
            slot_index = torch.tensor(slots, dtype=torch.int64, device=self.device)
            # The reference's `weights[rows, slot, None]`: the routing weight of *this* expert for
            # each row, broadcast over the intermediate width later. Widened to FP32 exactly, which
            # is the width the reference multiplies in.
            gate_weights = weights.index_select(0, row_index).gather(1, slot_index.unsqueeze(1))
            gate_weights = gate_weights.to(torch.float32)
            resident = self.expert(expert)
            contribution = expert_forward_ascend(
                x.index_select(0, row_index),
                resident,
                swiglu_limit=self.swiglu_limit,
                weights=gate_weights,
                dtype=self.dtype,
                backend=self.backend,
            )
            # Ascending expert id into one FP32 accumulator, each contribution already narrowed to
            # the store's dtype by its own GEMM -- the reference's own accumulation order.
            contribution = contribution.to(torch.float32)
            # When this expert took every row, the scatter writes each row exactly once from the
            # same value an elementwise add would use, in the same order, so the FP32 bits are the
            # same and the whole-index machinery is not needed. That is the batch-one decode case for
            # every contribution the token makes, and the index path costs about ninety microseconds
            # there for a scatter of one row. `rows` is built by walking the routing table in row
            # order, so it is ascending and duplicate-free: full cover is exactly this test.
            if len(rows) == n and rows[0] == 0 and rows[-1] == n - 1:
                y += contribution
            else:
                y.index_put_((row_index,), y.index_select(0, row_index) + contribution)
        return y

    # -- validation -----------------------------------------------------------

    def _check_forward_inputs(self, x, weights, indices) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("x must be a torch.Tensor")
        if x.ndim != 2:
            raise ValueError(f"x is {x.ndim}-dimensional; the experts consume [n, dim]")
        if tuple(x.shape)[1] != self.dim:
            raise ValueError(f"x is {tuple(x.shape)}; this store's experts are {self.dim} wide")
        if x.device != self.device:
            raise RuntimeError(
                f"AscendRoutedExperts holds its experts on {self.device} but was handed an "
                f"activation on {x.device}; the two are multiplied, so they have to agree"
            )
        if x.dtype != self.dtype:
            raise RuntimeError(
                f"AscendRoutedExperts narrows its expert GEMMs to {self.dtype} but was handed an "
                f"activation at {x.dtype}. The reference's expert output is BF16 and the clamp runs "
                "on that narrowed value, so a different width is a different computation"
            )
        if not isinstance(indices, torch.Tensor):
            raise TypeError("indices must be a torch.Tensor")
        if not isinstance(weights, torch.Tensor):
            raise TypeError("weights must be a torch.Tensor")
        if indices.ndim != 2 or weights.ndim != 2:
            raise ValueError(
                f"indices and weights are [{indices.ndim}D] and [{weights.ndim}D]; the gate emits "
                "[n, topk] for both"
            )
        if tuple(indices.shape) != tuple(weights.shape):
            raise ValueError(
                f"indices {tuple(indices.shape)} and weights {tuple(weights.shape)} disagree; they "
                "are the gate's two outputs for the same rows"
            )
        if tuple(indices.shape)[0] != tuple(x.shape)[0]:
            raise ValueError(
                f"{tuple(x.shape)[0]} rows of activation but {tuple(indices.shape)[0]} rows of "
                "routing; every row's experts need its row of activation"
            )
        if tuple(indices.shape)[1] != self.topk:
            raise ValueError(
                f"the gate emitted {tuple(indices.shape)[1]} experts per row; this store was built "
                f"for topk={self.topk}"
            )
        if indices.dtype not in _INTEGER_DTYPES:
            raise TypeError(f"indices has dtype {indices.dtype}, which is not an integer dtype")
        if not weights.is_floating_point():
            raise TypeError(f"weights has dtype {weights.dtype}, which is not a floating dtype")
        if indices.device != self.device or weights.device != self.device:
            raise RuntimeError(
                f"the routing is on {indices.device}/{weights.device} but the experts are on "
                f"{self.device}; the rows it names are rows of an activation on that device"
            )

    def _routing(self, indices: torch.Tensor) -> dict[int, tuple[list[int], list[int]]]:
        """Group the gate's output by expert, one host read of the whole index tensor.

        Also the range check and the uniqueness check: the gate's ``topk`` is without replacement, so
        a repeated expert inside one row means the routing is not what this store was built to
        consume, and the reference's own ``y[idx] +=`` idiom collides on it. Both are refused by name
        here rather than turned into a plausible wrong number deep in the loop.
        """
        host = indices.detach().to(device="cpu", dtype=torch.int64).tolist()
        buckets: dict[int, tuple[list[int], list[int]]] = {}
        for row, routing in enumerate(host):
            if self.topk > 1 and len(set(routing)) != self.topk:
                raise ValueError(
                    f"row {row} routes to experts {routing} with a repeat; the gate's topk is "
                    "without replacement, so a repeated id is not a routing this store can add up"
                )
            for slot, expert in enumerate(routing):
                if expert < 0 or expert >= self.n_experts:
                    raise ValueError(
                        f"row {row} routes to expert {expert}; this layer has {self.n_experts} "
                        f"experts, so the ids are in [0, {self.n_experts})"
                    )
                rows, slots = buckets.setdefault(expert, ([], []))
                rows.append(row)
                slots.append(slot)
        return buckets

    def _expert_id(self, expert: int) -> int:
        if isinstance(expert, bool) or not isinstance(expert, int):
            raise TypeError(f"an expert id is an int, got {type(expert).__name__}")
        if not 0 <= expert < self.n_experts:
            raise ValueError(f"expert {expert} is outside [0, {self.n_experts})")
        return expert


_INTEGER_DTYPES = frozenset(
    {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _block_multiple(value, name: str) -> int:
    value = _positive_int(value, name)
    if value % BLOCK:
        raise ValueError(
            f"{name} is {value}; the activation quantizer and the expert scale grid are stated on "
            f"{BLOCK}-wide blocks along the contracted axis, so every width here is a multiple"
        )
    return value


def _non_negative_float(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return float(value)


def _supported_device(device) -> torch.device:
    resolved = torch.device(device) if not isinstance(device, torch.device) else device
    if resolved.type not in _DEVICE_TYPES:
        raise ValueError(
            f"an Ascend expert store runs on 'npu' (production) or 'cpu' (the oracle fixture); got "
            f"{resolved.type}. The CUDA store is `device_experts.DeviceRoutedExperts`"
        )
    return resolved


def _supported_dtype(dtype: torch.dtype) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype must be a torch.dtype, got {type(dtype).__name__}")
    if dtype not in _FLOAT_DTYPES:
        raise ValueError(
            f"dtype is {dtype}; the activation quantizer takes BF16 or FP32 (and the released tree "
            "is BF16), so a store that narrowed to anything else would be quantizing a cast value "
            "the reference never produces"
        )
    return dtype


def _resolve_backend(backend: str, device: torch.device) -> str:
    if not isinstance(backend, str) or backend not in _BACKENDS:
        raise ValueError(f"backend must be one of {sorted(_BACKENDS)}, got {backend!r}")
    try:
        block_fp8_quant, _ = _core()
    except ImportError as missing:
        raise RuntimeError(
            "this store is built out of relic-core's block-32 quantizer and block GEMM, and the "
            f"installed relic-core does not provide them ({missing}); install it from its checkout "
            "(`pip install -e ../relic-core --no-build-isolation --no-deps`)"
        ) from missing
    if device.type == "npu":
        if backend == "torch":
            raise ValueError(
                "backend='torch' is the CPU oracle fixture path; on a card use backend='triton' or "
                "'auto'. Running the CPU reference against NPU tensors would fall back to a "
                "different arithmetic than the one being measured"
            )
        if not block_fp8_quant.supports_triton_ascend():
            raise RuntimeError(
                "backend resolves to the fused Ascend operators but no compatible Triton Ascend "
                "backend is installed; the store refuses to build rather than silently running the "
                "CPU reference on the card"
            )
        return "triton"
    if backend == "triton":
        raise ValueError(
            "backend='triton' is the fused Ascend path and needs NPU tensors; the CPU oracle "
            "fixture is backend='torch'"
        )
    if getattr(torch, "float8_e4m3fn", None) is None:  # pragma: no cover - a torch without fp8
        raise RuntimeError(
            "the CPU oracle path ends in a native float8_e4m3fn cast and this torch build has no "
            "such dtype; use a build that has it, or run the store on the card"
        )
    return "torch"


def expert_forward_ascend(
    x: torch.Tensor,
    resident: ResidentExpert,
    *,
    swiglu_limit: float = 0.0,
    weights: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """`modules.expert_forward` executed from original bytes: `x` `[rows, dim]` -> `[rows, dim]`.

    The three linears are the block-32 activation quantizer and the block-32 FP4/INT8 GEMM, whose
    reduction is the reference's two-half rule; everything around them is the reference's, in its
    order:

    * ``w1`` and ``w3`` each quantize ``x`` on their own, then narrow their FP32 accumulators to
      ``dtype`` and widen them for the clamp -- the reference's GEMM outputs BF16 and the clamp is
      FP32 on that value;
    * ``up`` is clamped to ``[-limit, limit]``, ``gate`` only from above, ``silu(gate) * up``;
    * the routing weight multiplies that **intermediate**, not the result;
    * ``w2`` is handed the ``dtype`` cast of it, which is what is quantized;
    * the ``w2`` accumulator is narrowed to ``dtype`` and returned at that width.

    ``dtype`` defaults to ``x.dtype``. ``weights`` is the routing weight per row, ``[rows, 1]``.
    """
    dtype = x.dtype if dtype is None else dtype
    block_fp8_quant, block_gemm = _core()
    quantize = block_fp8_quant.quantize_block_fp8
    gemm = block_gemm.fp4_int8_block_gemm

    # Two independent quantizations, because the reference is two separate `Linear` calls. The
    # quantizer is deterministic, so the bytes agree; the call structure is kept rather than folded
    # so this seam does not quietly become "the reference with one step shared".
    gate_codes, gate_scale = quantize(x, backend=backend)
    gate = gemm(
        gate_codes, gate_scale, resident.w1.values, resident.w1.half_scale, backend=backend
    )
    gate = gate.to(dtype).float()
    up_codes, up_scale = quantize(x, backend=backend)
    up = gemm(up_codes, up_scale, resident.w3.values, resident.w3.half_scale, backend=backend)
    up = up.to(dtype).float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    out = F.silu(gate) * up
    if weights is not None:
        out = weights * out
    out_codes, out_scale = quantize(out.to(dtype), backend=backend)
    w2 = gemm(out_codes, out_scale, resident.w2.values, resident.w2.half_scale, backend=backend)
    return w2.to(dtype)