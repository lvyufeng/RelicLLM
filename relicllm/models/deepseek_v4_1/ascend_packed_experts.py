"""Original packed expert bytes resident in HBM, without ever widening them in memory.

`ascend_experts.AscendRoutedExperts` executes a layer's routed experts from the checkpoint's original
packed FP4 bytes, but it keeps a **bounded LRU of widened experts**: INT8-times-two codes plus FP32
half-scales, 37.97 MiB per expert, with one process-wide 512 MiB budget shared by every layer. That
budget exists because 384 experts of one layer fit in neither representation, so a run of many layers
degrades into one cache and a decode step re-reads weights it evicted.

This module is the other trade: spend HBM once, up front, on the **bytes the file already holds**.
One released expert is three packed E2M1 matrices (5,898,240 code bytes each) beside three raw E8M0
grids (368,640 bytes each) -- **18,800,640 bytes**, 17.93 MiB, against 37.97 MiB widened and 67.5 MiB
dense BF16. For the experts a rank owns under the expert-parallel deal, all 40 layers of this release
are **1920 experts = 33.62 GiB per card**, which fits a 64 GiB part; the widened form would be
71.19 GiB and does not.

Why residency is worth having
-----------------------------
The 512 MiB LRU holds about 13 widened experts, so a decode step that draws 6 experts per layer misses
on most layers and pays the file read plus the widening again on each miss. Residency replaces that
with one resident copy, and the per-token cost becomes the arithmetic alone. Byte accounting is all
this file claims: it makes **no** tokens-per-second claim, and none of the numbers above is a
measurement of anything but the released shapes' sizes.

The bytes are the file's, and they travel unmodified
----------------------------------------------------
* Every E2M1 low-nibble-first code byte and every raw E8M0 scale byte is copied from
  `checkpoint.packed(key)` -- the shard mapping, or the resident bank when one is attached -- through
  a ``uint8`` **dtype view**, never a numeric cast. The encodings are the payload, so ``.to(uint8)``
  would convert values and destroy exactly what is about to be executed.
* No widening is stored: the resident form is the packed form. The INT8-times-two representation
  appears only as a **bounded temporary inside one call**, and only for more than one row.
* No requantization, no dense weight, no checkpoint rewrite, and no native FP4/FP8 arithmetic claim:
  the reduction stays relic-core's reference two-half block rule, which the bounded store also runs.

How the arithmetic runs
-----------------------
``relic_core.kernels.decode_block_gemv.fp4_packed_decode_gemv`` is the M=1 path: it consumes the
original packed weight bytes beside the raw scale bytes directly, so a decode step widens no weight at
all. It is resolved **by name** at the call, and when it is not installed the M=1 path raises
:class:`PackedDecodeUnavailable` rather than running another kernel: a store that quietly substitutes
arithmetic is a store whose numbers cannot be compared to anything.

For more than one row -- a prefill -- the provider converts **one projection at a time** to the
accepted INT8-times-two operands (``unpack_fp4_to_int8`` + ``fp4_int8_scale``, the frozen operators the
bounded store uses) and calls the accepted ``fp4_int8_block_gemm``. The temporary is one projection
(37.2 MiB for ``w1``), not one expert and never a layer, and it is dropped when the projection is
done. The operand mapping is the bounded store's own, so the two providers agree where their paths
meet -- which is what the CPU fixtures assert.

Order and placement, which is where an expert store is easiest to get wrong
--------------------------------------------------------------------------
`modules.expert_forward` is the contract, and every asymmetry is reproduced exactly as the bounded
store states it: the gate and up accumulators narrow to the store dtype and widen again for an FP32
clamp; ``swiglu_limit`` is applied **before** ``silu``; the routing weight multiplies the
**intermediate** activation; the ``w2`` operand is the BF16 cast of that intermediate, quantized after
the cast; the ``w2`` result is narrowed and enters one FP32 routed sum walked in ascending expert id
order.

The one optimization taken is the one the bounded store documents as legal and does not take: ``w1``
and ``w3`` share **one** activation quantization, because the released quantizer is deterministic and
the two byte tensors are identical by construction. That is measured here rather than assumed -- the
shared bytes are compared against two independent quantizations and the call count is asserted.

Expert parallelism
------------------
``world``/``rank`` deal whole experts cyclically (``expert % world == rank``), the same ``id`` rule the
bounded store and `device_experts.deal_rule` use, and ``partial`` is computed once in ``__init__``:
with ``world > 1`` the store returns one rank's share of the routed sum and `MoE.forward` joins it
with the shared expert on one all-reduce. Only owned experts are resident, so the HBM cost is
``1 / world`` of the model's packed experts.

What is not claimed
-------------------
No token parity, no TPS, no graph capture (the routing indices are read on the host once per call), no
batched multi-token kernel, no native FP4/FP8 arithmetic, and no bounded-run capability: the ``v41``
Ascend capability guard is untouched. This provider is **opt-in** and allocates nothing until
``preload=True`` or :meth:`AscendPackedRoutedExperts.preload` is called.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .ascend_experts import BLOCK, AscendExpertCache, AscendRoutedExperts
from .online_unpack import Fp4ExpertInt8

__all__ = [
    "AscendPackedRoutedExperts",
    "DEFAULT_RESERVE_BYTES",
    "PackedDecodeUnavailable",
    "PackedExpert",
    "PackedMatrix",
    "packed_expert_bytes",
    "packed_expert_forward",
    "packed_model_bytes",
    "packed_store_bytes",
]

_PROJECTIONS = ("w1", "w2", "w3")

#: Default headroom a store requires beside its own residency. The 33.62 GiB of packed experts is
#: only part of what a run needs on the same card: the dense half is about 6 GiB in BF16, the driver
#: and allocator hold their own reservations, and the activation/cache working set grows with the
#: context. 12 GiB is the explicit preflight floor for that, so a store is refused rather than
#: admitted when the card cannot hold both its experts and the run around them. It is a floor, not a
#: measurement: the launcher passes its own number when it has measured one.
DEFAULT_RESERVE_BYTES = 12 * 2**30

#: Released checkpoint geometry. The three estimators below take their own numbers; these exist so the
#: budget arithmetic can be stated once, in the tests, against the shapes this provider was written
#: for: 384 experts a layer, 40 layers, 40 of them carrying routed experts.
RELEASE_DIM = 5120
RELEASE_INTER_DIM = 2304
RELEASE_EXPERTS = 384
RELEASE_LAYERS = 40


class PackedDecodeUnavailable(RuntimeError):
    """The M=1 packed decode operator is not installed in the relic-core that is on the path.

    Raised, never swallowed: falling back to the prefill path's temporary widening would make a
    measurement of "the packed decode path" describe a different path, and it would allocate the
    INT8 operand the residency design exists to avoid.
    """


def _core():
    """The byte-level operators, imported where they are used.

    Lazy for the same reason `ascend_experts._core` is: importing this module has to stay possible on
    a host whose installed relic-core predates these operators, so the legacy path can still load.
    """
    from relic_core.kernels import block_fp8_quant, block_gemm, unpack

    return block_fp8_quant, block_gemm, unpack


def _stored_bytes(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """`tensor`'s original 8-bit payload as a `uint8` view, without changing a byte of it.

    A view and not a cast: the checkpoint stores these in containers whose *encoding* is the payload
    (packed E2M1 nibbles, E8M0 exponent bytes), so ``.to(torch.uint8)`` would convert the values and
    destroy exactly what is about to be executed.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must have a dense strided layout")
    if tensor.element_size() != 1:
        raise TypeError(
            f"{name} is {tensor.dtype}, which is not an 8-bit original-code container; this provider "
            "keeps the file's own encodings and does not widen them at load"
        )
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


def _positive_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _non_negative_int(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a byte count, an int, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{name} cannot be negative, got {value}")
    return value


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def packed_expert_bytes(dim: int, inter_dim: int, *, block: int = BLOCK) -> int:
    """Resident bytes of one expert in its **original packed** form.

    ``w1``/``w3`` are ``[inter_dim, dim]`` and ``w2`` is ``[dim, inter_dim]``; each packed matrix is
    half as many code bytes as values, and each grid is one byte per output row and ``block``-wide K
    block. For the released geometry this is exactly **18,800,640 bytes** (17.93 MiB), 2x smaller than
    the widened INT8 form (37.97 MiB) and 3.75x smaller than the dense BF16 widening (67.5 MiB).
    """
    dim = _positive_int(dim, "dim")
    inter_dim = _positive_int(inter_dim, "inter_dim")
    block = _positive_int(block, "block")
    if dim % block or inter_dim % block:
        raise ValueError(
            f"dim {dim} / inter_dim {inter_dim} are not whole numbers of {block}-wide K blocks, so "
            "the raw scale grids this counts cannot exist"
        )
    total = 0
    for out_features, in_features in (
        (inter_dim, dim),
        (dim, inter_dim),
        (inter_dim, dim),
    ):
        total += out_features * (in_features // 2)
        total += out_features * (in_features // block)
    return total


def packed_store_bytes(
    n_experts: int, dim: int, inter_dim: int, *, rank: int = 0, world: int = 1, block: int = BLOCK
) -> int:
    """Resident bytes of **one store**: the experts one rank owns under the cyclic deal."""
    n_experts = _positive_int(n_experts, "n_experts")
    if isinstance(world, bool) or not isinstance(world, int) or world < 1:
        raise ValueError(f"world must be a positive int, got {world!r}")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world:
        raise ValueError(f"rank must be an int in [0, {world}), got {rank!r}")
    owned = sum(1 for expert in range(n_experts) if expert % world == rank)
    return owned * packed_expert_bytes(dim, inter_dim, block=block)


def packed_model_bytes(
    n_layers: int,
    n_experts: int,
    dim: int,
    inter_dim: int,
    *,
    rank: int = 0,
    world: int = 1,
    block: int = BLOCK,
) -> int:
    """Resident bytes of one rank's stores across a backbone: the number a budget is set against.

    For the released geometry at ``world=8``: 40 layers x 48 owned experts x 18,800,640 bytes =
    **36,097,228,800 bytes (33.62 GiB)**. That sum is what a launcher compares with the card's free
    HBM before it constructs any store, so the arithmetic is a function here rather than a note.
    """
    n_layers = _positive_int(n_layers, "n_layers")
    return n_layers * packed_store_bytes(
        n_experts, dim, inter_dim, rank=rank, world=world, block=block
    )


def _allocate(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Allocate one arena. A separate function so a test can prove validation precedes allocation."""
    return torch.empty(shape, dtype=torch.uint8, device=device)


def _device_free_bytes(device: torch.device) -> int | None:
    """Free bytes on `device`, or `None` when this stack cannot report them.

    Only ``npu`` is asked, and only through the allocator's own report -- the same information a
    launcher would read. The two returned numbers are read as ``(free, total)`` like CUDA's
    ``mem_get_info``; taking the **smaller** of them is the conservative reading, so a build that
    returned them the other way round cannot make this check believe there is more room than there
    is. An unavailable probe returns ``None`` and the caller records that instead of inventing a
    number.
    """
    if device.type != "npu":
        return None
    mem_get_info = getattr(getattr(torch, "npu", None), "mem_get_info", None)
    if mem_get_info is None:
        return None
    try:
        reported = mem_get_info(device)
    except TypeError:  # a build whose probe takes no device argument
        try:
            reported = mem_get_info()
        except Exception:
            return None
    except Exception:
        return None
    try:
        first, second = int(reported[0]), int(reported[1])
    except (TypeError, IndexError, ValueError):
        return None
    return min(first, second)


@dataclass(frozen=True)
class PackedMatrix:
    """One projection in the checkpoint's own encoding, resident on one device.

    ``codes`` is ``[out, in // 2]`` of ``uint8``: two E2M1 codes per byte, **low nibble first** along
    K. ``scales`` is ``[out, in // 32]`` of ``uint8``: the raw original E8M0 exponent bytes, one per
    output row and 32-wide K block. ``block`` is the K block width the grid was written on.

    Both tensors are views into a store's arena, so constructing one of these allocates nothing. They
    are treated as immutable: nothing in this module writes to them after the arena is filled, and
    the values are the file's bytes rather than a representation derived from them.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    block: int

    @property
    def nbytes(self) -> int:
        return _tensor_nbytes(self.codes) + _tensor_nbytes(self.scales)

    @property
    def shape(self) -> tuple[int, int]:
        """The logical ``(out_features, in_features)`` matrix this encoding describes."""
        return int(self.codes.shape[0]), int(self.codes.shape[1]) * 2

    def to_int8_x2(self, *, backend: str = "auto") -> Fp4ExpertInt8:
        """The accepted INT8-times-two operands of this projection, as a **bounded temporary**.

        This is the prefill path's conversion, not a resident form: the returned tensors are one
        projection's worth of INT8 codes plus FP32 half-scales, and the caller drops them when the
        projection is done. The widening is relic-core's frozen ``unpack_fp4_to_int8`` and the grid is
        its frozen ``fp4_int8_scale`` -- the same two operators the bounded store uses -- so the
        operand mapping is that store's, not a second spelling of it.
        """
        _, _, unpack = _core()
        if not self.codes.is_contiguous() or not self.scales.is_contiguous():
            raise ValueError(
                "a packed projection handed to the widening is not contiguous; the arena's own views "
                "are, so this means a caller built one by hand"
            )
        values = unpack.unpack_fp4_to_int8(self.codes, backend=backend)
        half_scale = unpack.fp4_int8_scale(self.scales, backend=backend)
        return Fp4ExpertInt8(values=values, half_scale=half_scale, block=self.block)


@dataclass(frozen=True)
class PackedExpert:
    """One expert's three projections in the file's encoding, resident on one device."""

    w1: PackedMatrix
    w2: PackedMatrix
    w3: PackedMatrix
    block: int

    @property
    def nbytes(self) -> int:
        """Resident bytes: the packed codes plus the raw scale grids, nothing else."""
        return self.w1.nbytes + self.w2.nbytes + self.w3.nbytes

    def matrices(self) -> dict[str, PackedMatrix]:
        return {"w1": self.w1, "w2": self.w2, "w3": self.w3}


@dataclass(frozen=True)
class _ProjectionArena:
    """One projection's residency for every expert a store owns: two arrays, not two per expert.

    ``codes`` is ``[owned, out, in // 2]`` and ``scales`` is ``[owned, out, in // 32]``, both
    ``uint8``. One allocation each per projection per layer means 6 tensors a layer -- not 5,760
    per-expert tensors for a full 40-layer, 48-expert card -- and it is also what lets an expert be a
    view rather than a copy.
    """

    which: str
    codes: torch.Tensor
    scales: torch.Tensor

    @property
    def nbytes(self) -> int:
        return _tensor_nbytes(self.codes) + _tensor_nbytes(self.scales)

    def matrix(self, slot: int) -> PackedMatrix:
        return PackedMatrix(
            codes=self.codes[slot], scales=self.scales[slot], block=BLOCK
        )


def _import_packed_decode():
    """The relic-core M=1 operator, or a named error saying why it is not usable.

    The module is checked in, so an import failure means the installed relic-core is older than this
    provider; a module without the operator means the same. Neither is substituted.
    """
    try:
        from relic_core.kernels import decode_block_gemv
    except ImportError as missing:
        raise PackedDecodeUnavailable(
            "the M=1 packed decode operator is not installed in the relic-core on this path "
            f"(relic_core.kernels.decode_block_gemv: {missing}). This provider does not fall back to "
            "the prefill path's temporary widening, because that would allocate exactly the INT8 "
            "operand residency exists to avoid and would make the measurement describe another path. "
            "Install relic-core from its checkout, or pass packed_gemv= explicitly for an oracle"
        ) from missing
    function = getattr(decode_block_gemv, "fp4_packed_decode_gemv", None)
    if function is None:
        raise PackedDecodeUnavailable(
            "the installed relic-core has relic_core.kernels.decode_block_gemv but no "
            "`fp4_packed_decode_gemv` in it; the M=1 packed decode operator is what this provider "
            "calls, and it refuses to substitute the prefill path"
        )
    if not callable(function):
        raise PackedDecodeUnavailable(
            f"relic_core.kernels.decode_block_gemv.fp4_packed_decode_gemv is {type(function).__name__}, "
            "not callable; refusing to treat it as the M=1 operator"
        )
    return function


def _resolve_packed_gemv(packed_gemv) -> "callable":
    """The M=1 operator to call: an explicit injection, or relic-core's by name.

    An explicit callable is the oracle seam: the tests inject a same-contract reference so the store's
    ordering can be checked without a card. It is never a quiet fallback -- passing one says so.
    """
    if packed_gemv is None:
        return _import_packed_decode()
    if not callable(packed_gemv):
        raise TypeError(
            f"packed_gemv must be callable or None, got {type(packed_gemv).__name__}; it stands in for "
            "relic-core's M=1 packed decode operator"
        )
    return packed_gemv


def _call_packed_gemv(function, a_codes, a_scale, matrix: PackedMatrix, *, backend: str):
    """Call the M=1 operator with this provider's operand order.

    The contract, in one place because it is the only coupling to relic-core's decode operator:
    ``fp4_packed_decode_gemv(a_codes, a_scale, b_packed, b_e8m0, *, backend=...)``. All four operands
    are 2-D and contiguous: the activation's raw E4M3 codes ``[1, K]`` and raw E8M0 bytes
    ``[1, K//32]``, and the weight's packed E2M1 bytes ``[N, K//2]`` (low nibble first along K) beside
    its **original, un-halved** E8M0 grid ``[N, K//32]``. The operator widens the nibbles itself and
    derives its own half-scale, so nothing here reinterprets an encoding or pre-slices a grid -- the
    arenas hold the file's bytes and they are passed as they are.

    One error is mapped and only one: the shared launcher's "no compatible Triton Ascend backend"
    ``RuntimeError`` becomes :class:`PackedDecodeUnavailable`, because that is a named refusal to run
    this path rather than a kernel result. Every other exception -- in particular a compile error or a
    shape/dtype refusal from the operator -- propagates verbatim: a compile failure must be seen, not
    relabelled and not answered by another kernel.
    """
    try:
        return function(a_codes, a_scale, matrix.codes, matrix.scales, backend=backend)
    except RuntimeError as error:
        if "Triton Ascend backend is required" in str(error):
            raise PackedDecodeUnavailable(
                f"the M=1 packed decode operator cannot run on {matrix.codes.device}: {error}. This "
                "provider does not fall back to the prefill path's temporary widening"
            ) from error
        raise


class AscendPackedRoutedExperts(AscendRoutedExperts):
    """One layer's routed experts, resident in HBM in the checkpoint's original packed encoding.

    An interface implementation rather than an `nn.Module`, like the bounded store it specializes:
    it holds no `Parameter`, so the experts never appear in ``state_dict()``; `MoE` reads ``partial``
    and calls ``forward``; `tp.py` does not cut it, because the routed experts are not parameters and
    their only split is the expert deal.

    It **inherits** the bounded store's validation and routing -- the constructor's shape checks, the
    ``[n, topk]`` routing contract, the range/uniqueness checks, and the ascending-id forward walk --
    and replaces exactly two things: what an expert is made of, and which operator projects it.

    Construction validates and allocates **nothing**. ``preload=True`` (or a later :meth:`preload`)
    validates every owned expert's stored bytes and then allocates six arena tensors per layer and
    copies the file's bytes into them. That split exists because the validation failures are the ones
    worth failing before 33.62 GiB of HBM is touched.

    ``budget_bytes`` is the most HBM this store may hold and defaults to ``required_bytes +
    reserve_bytes``; a budget below ``required_bytes`` is refused rather than built. ``reserve_bytes``
    defaults to :data:`DEFAULT_RESERVE_BYTES` (12 GiB) because the experts are not the only thing on
    the card: the dense half, the driver and the run's working set need room beside them, so the
    preflight compares the residency **plus the reserve** with the card's free memory.
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
        dtype: torch.dtype = torch.bfloat16,
        rank: int = 0,
        world: int = 1,
        backend: str = "triton",
        budget_bytes: int | None = None,
        reserve_bytes: int = DEFAULT_RESERVE_BYTES,
        preload: bool = False,
        packed_gemv=None,
    ) -> None:
        # A private zero-budget cache keeps the parent's attribute contract satisfied without letting
        # a packed store retain anything in the process-wide widened LRU. Nothing below writes to it,
        # and `cache_stats` is replaced, so the budget it reports is this store's own.
        super().__init__(
            checkpoint,
            layer_id,
            n_experts=n_experts,
            dim=dim,
            inter_dim=inter_dim,
            topk=topk,
            swiglu_limit=swiglu_limit,
            device=device,
            dtype=dtype,
            rank=rank,
            world=world,
            cache=AscendExpertCache(0),
            backend=backend,
        )
        self.block = BLOCK
        self.owned: tuple[int, ...] = tuple(self._owned_experts())
        self._slots: dict[int, int] = {expert: slot for slot, expert in enumerate(self.owned)}
        self._expected_shapes: dict[str, tuple[int, int]] = {
            "w1": (self.inter_dim, self.dim),
            "w2": (self.dim, self.inter_dim),
            "w3": (self.inter_dim, self.dim),
        }
        self.required_bytes = len(self.owned) * packed_expert_bytes(
            self.dim, self.inter_dim, block=self.block
        )
        self.reserve_bytes = _non_negative_int(reserve_bytes, "reserve_bytes")
        if budget_bytes is None:
            budget_bytes = self.required_bytes + self.reserve_bytes
        budget_bytes = _non_negative_int(budget_bytes, "budget_bytes")
        if budget_bytes < self.required_bytes:
            raise ValueError(
                f"layer {self.layer_id}: {len(self.owned)} owned experts need "
                f"{self.required_bytes} packed bytes but the budget is {budget_bytes}; residency is "
                "this provider's whole design, so a store that cannot hold its experts is refused "
                "instead of built as a slower copy of the bounded store"
            )
        self.budget_bytes = budget_bytes
        self.packed_gemv = packed_gemv
        self.lookups = 0
        self._arenas: dict[str, _ProjectionArena] | None = None
        self._headroom: dict | None = None
        if preload:
            self.preload()

    # -- reporting ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AscendPackedRoutedExperts(layer={self.layer_id}, experts={self.n_experts}, "
            f"owned={len(self.owned)}, topk={self.topk}, device={self.device}, dtype={self.dtype}, "
            f"backend={self.backend}, rank={self.rank}/{self.world}, "
            f"required_bytes={self.required_bytes}, preloaded={self.allocated})"
        )

    @property
    def allocated(self) -> bool:
        return self._arenas is not None

    @property
    def resident_bytes(self) -> int:
        if self._arenas is None:
            return 0
        return sum(arena.nbytes for arena in self._arenas.values())

    @property
    def resident_experts(self) -> int:
        """How many of *this* store's experts are resident right now -- all of them, once preloaded."""
        return len(self.owned) if self._arenas is not None else 0

    @property
    def arenas(self) -> int:
        """Arena objects held: three projections a layer, each owning a code and a scale array."""
        return 0 if self._arenas is None else len(self._arenas)

    @property
    def cache_stats(self) -> dict:
        """Residency, in the shape the widened store's snapshot uses, plus this store's own numbers.

        The keys the CLI reads (``budget_bytes``/``retained_bytes``/``entries`` and the hit counters)
        mean the residency here rather than an LRU's: ``misses`` and ``evictions`` are always zero
        because a preloaded store cannot miss, and ``hit_rate`` is ``None`` because a lookup that can
        never miss does not have one.
        """
        return {
            "name": f"ascend-packed-experts/layer{self.layer_id}",
            "budget_bytes": self.budget_bytes,
            "retained_bytes": self.resident_bytes,
            "entries": self.resident_experts,
            "hits": self.lookups,
            "misses": 0,
            "evictions": 0,
            "streamed": 0,
            "hit_rate": None,
            "required_bytes": self.required_bytes,
            "owned_experts": len(self.owned),
            "preloaded": self.allocated,
            "headroom": self._headroom,
        }

    @property
    def expert_stats(self) -> dict:
        """Per-store facts a launcher logs next to the routing decision."""
        return {
            "layer_id": self.layer_id,
            "n_experts": self.n_experts,
            "owned_experts": len(self.owned),
            "resident_experts": self.resident_experts,
            "topk": self.topk,
            "block": self.block,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "backend": self.backend,
            "rank": self.rank,
            "world": self.world,
            "partial": self.partial,
            "packed_expert_bytes": packed_expert_bytes(self.dim, self.inter_dim, block=self.block),
            "required_bytes": self.required_bytes,
            "budget_bytes": self.budget_bytes,
            "resident_bytes": self.resident_bytes,
            "preloaded": self.allocated,
        }

    # -- residency ------------------------------------------------------------

    def preload(self) -> int:
        """Validate every owned expert's bytes, then make them resident. Returns the bytes held.

        Validation runs first and **before any allocation**: every projection of every owned expert
        has to be packed fp4 with a 32-wide grid, and its stored shapes have to be the shapes this
        store was built for. Only then are the six arena arrays a layer allocates, and only then does
        the file's bytes get copied into them.

        Idempotent: a second call on a preloaded store does nothing and reports the bytes held.
        """
        if self._arenas is not None:
            return self.resident_bytes
        self._validate_checkpoint_bytes()
        headroom = self._check_memory_headroom()
        codes_shape = lambda out_features, in_features: (  # noqa: E731 - two small shape closures
            len(self.owned), out_features, in_features // 2
        )
        scales_shape = lambda out_features, in_features: (  # noqa: E731
            len(self.owned), out_features, in_features // self.block
        )
        arenas: dict[str, _ProjectionArena] = {}
        for which in _PROJECTIONS:
            out_features, in_features = self._expected_shapes[which]
            arenas[which] = _ProjectionArena(
                which=which,
                codes=_allocate(codes_shape(out_features, in_features), self.device),
                scales=_allocate(scales_shape(out_features, in_features), self.device),
            )
        for slot, expert in enumerate(self.owned):
            for which in _PROJECTIONS:
                key = self._key(expert, which)
                scale_name = self._scale_key(expert, which)
                codes = _stored_bytes(self.checkpoint.packed(key), f"{key} codes").contiguous()
                scales = _stored_bytes(self.checkpoint.packed(scale_name), f"{scale_name}").contiguous()
                # `copy_` between two `uint8` tensors moves bytes; it is not a numeric conversion of
                # the encodings, and the shapes were checked above, so nothing here can reinterpret
                # a nibble or an exponent byte.
                arenas[which].codes[slot].copy_(codes)
                arenas[which].scales[slot].copy_(scales)
        self._arenas = arenas
        self._headroom = headroom
        return self.resident_bytes

    def release(self) -> int:
        """Drop this store's resident arenas and return how many experts were held.

        Local cleanup of one object's tensors: no device reset, no shared allocator flush, and no
        other store's residency is touched.
        """
        dropped = self.resident_experts
        self._arenas = None
        self._headroom = None
        return dropped

    def clear_cache(self) -> int:
        """`release`, spelled the way the widened store's callers spell it. Local, as above."""
        return self.release()

    def _validate_checkpoint_bytes(self) -> None:
        """Refuse a checkpoint that is not the packed original this store was built for."""
        for expert in self.owned:
            for which in _PROJECTIONS:
                key = self._key(expert, which)
                scale_name = self._scale_key(expert, which)
                if not self.checkpoint.is_quantized(key):
                    raise ValueError(
                        f"{key} has no scale stored beside it, so the routed experts of this "
                        "checkpoint are not in the packed-fp4 form this provider requires"
                    )
                if not self.checkpoint.is_packed_fp4(key):
                    raise ValueError(
                        f"{key} is not stored as packed fp4; this provider keeps the file's packed "
                        "bytes and does not expand a dense weight at load"
                    )
                block = self.checkpoint.block_size(key)
                if block != self.block:
                    raise ValueError(
                        f"{key} derives a {block}-wide K block, but the released routed experts use "
                        f"{self.block}; a different width would apply each scale to the wrong number "
                        "of values"
                    )
                out_features, in_features = self._expected_shapes[which]
                codes = _stored_bytes(self.checkpoint.packed(key), f"{key} codes")
                scales = _stored_bytes(self.checkpoint.packed(scale_name), f"{scale_name}")
                expected_codes = (out_features, in_features // 2)
                expected_scales = (out_features, in_features // block)
                if tuple(codes.shape) != expected_codes:
                    raise ValueError(
                        f"{key} is {tuple(codes.shape)}; a {which} of this store's layer is "
                        f"{expected_codes} packed code bytes ({out_features}x{in_features} values, "
                        "two per byte), so this checkpoint does not describe the layer it was "
                        "attached to"
                    )
                if tuple(scales.shape) != expected_scales:
                    raise ValueError(
                        f"{scale_name} is {tuple(scales.shape)}; the original grid keeps one scale "
                        f"per output row and {block}-wide K block, so it must be "
                        f"{expected_scales}"
                    )

    def _check_memory_headroom(self) -> dict | None:
        """Refuse to start filling when the card cannot hold this store, and record what was seen."""
        free = _device_free_bytes(self.device)
        if free is None:
            return None
        needed = self.required_bytes + self.reserve_bytes
        if free < needed:
            raise RuntimeError(
                f"layer {self.layer_id}: the {len(self.owned)} owned experts need "
                f"{self.required_bytes} packed bytes plus a {self.reserve_bytes}-byte reserve, but "
                f"only {free} bytes are free on {self.device}; residency is the point of this "
                "provider, so it refuses rather than building a store that cannot hold its experts"
            )
        return {
            "free_bytes": free,
            "required_bytes": self.required_bytes,
            "reserve_bytes": self.reserve_bytes,
        }

    # -- experts --------------------------------------------------------------

    def expert(self, expert: int) -> PackedExpert:
        """One expert's three projections as views of this store's arenas.

        No I/O and no widening: a preloaded store has every owned expert resident, so this is a
        three-slice view. Asking for an expert this rank does not own, or for any expert before the
        store is preloaded, is refused by name -- the store does not load on demand, because
        on-demand loading is the bounded store's design and mixing the two would make the resident
        bytes a lie.
        """
        expert = self._expert_id(expert)
        if not self._owns(expert):
            raise ValueError(
                f"expert {expert} is not owned by rank {self.rank} of {self.world}; the deal is "
                f"expert % world, and this store holds {list(self.owned[:8])}..."
            )
        if self._arenas is None:
            raise RuntimeError(
                f"layer {self.layer_id}: expert {expert} was asked for before the store preloaded; "
                "this provider makes every owned expert resident at startup and does not load on "
                "demand. Construct with preload=True or call preload() first"
            )
        self.lookups += 1
        return self._expert_at(expert)

    def _expert_at(self, expert: int) -> PackedExpert:
        slot = self._slots[expert]
        arenas = self._arenas
        if arenas is None:  # pragma: no cover - every caller checks `allocated` first
            raise RuntimeError("the packed expert store is not preloaded")
        return PackedExpert(
            w1=arenas["w1"].matrix(slot),
            w2=arenas["w2"].matrix(slot),
            w3=arenas["w3"].matrix(slot),
            block=self.block,
        )

    # -- the forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Run one layer's routed experts from resident packed bytes. Returns FP32 `[n, dim]`.

        The walk is the bounded store's, unchanged: the routing is read on the host once, experts are
        visited in ascending id order, each contribution is narrowed by its own GEMM to the store
        dtype, widened back to FP32, and added into one accumulator. One row takes the M=1 packed
        decode operator; more rows take the bounded temporary widening and the accepted block GEMM.
        """
        self._check_forward_inputs(x, weights, indices)
        n = int(x.shape[0])
        if n == 0:
            return torch.zeros((0, self.dim), dtype=torch.float32, device=self.device)
        if self._arenas is None:
            raise RuntimeError(
                f"layer {self.layer_id}: forward was called before the store preloaded; construct "
                "with preload=True or call preload() first. This provider does not load on demand"
            )
        buckets = self._routing(indices)
        # One decision for the whole call: a decode step is one row and takes the packed operator, a
        # prefill takes the bounded widening. An expert that owns a single row of a multi-row prefill
        # must take the same path as the rest of that call, so this is not re-derived per expert.
        prefill = n > 1
        gemv = None if prefill else _resolve_packed_gemv(self.packed_gemv)
        y = torch.zeros((n, self.dim), dtype=torch.float32, device=self.device)
        for expert in sorted(buckets):
            if not self._owns(expert):
                continue
            rows, slots = buckets[expert]
            row_index = torch.tensor(rows, dtype=torch.int64, device=self.device)
            slot_index = torch.tensor(slots, dtype=torch.int64, device=self.device)
            gate_weights = weights.index_select(0, row_index).gather(1, slot_index.unsqueeze(1))
            gate_weights = gate_weights.to(torch.float32)
            self.lookups += 1
            contribution = packed_expert_forward(
                x.index_select(0, row_index),
                self._expert_at(expert),
                swiglu_limit=self.swiglu_limit,
                weights=gate_weights,
                dtype=self.dtype,
                backend=self.backend,
                packed_gemv=gemv,
                prefill=prefill,
            )
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


def packed_expert_forward(
    x: torch.Tensor,
    resident: PackedExpert,
    *,
    swiglu_limit: float = 0.0,
    weights: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
    backend: str = "auto",
    packed_gemv=None,
    prefill: bool | None = None,
) -> torch.Tensor:
    """`modules.expert_forward` from resident packed bytes: `x` `[rows, dim]` -> `[rows, dim]`.

    `rows == 1` (a decode step) projects through the M=1 packed decode operator, which reads the
    weight's original nibbles and exponent bytes directly; `rows > 1` (a prefill) widens one
    projection at a time to the accepted INT8-times-two operands and calls the accepted block GEMM.
    Both return the same FP32 accumulator the bounded store's GEMM returns, and everything around the
    projections is the reference's order:

    * the FP32 accumulator narrows to ``dtype`` and widens again for the FP32 clamp;
    * ``up`` is clamped to ``[-limit, limit]`` and ``gate`` only from above, before ``silu``;
    * the routing weight multiplies the **intermediate** ``[rows, inter]`` activation;
    * ``w2`` is handed the ``dtype`` cast of that intermediate, which is what is quantized;
    * the ``w2`` accumulator narrows to ``dtype`` and is returned at that width.

    The gate and up projections share **one** activation quantization: they read the same `x`, and the
    released quantizer is deterministic, so the second call would compute the same bytes. The tests
    compare the shared bytes against two independent quantizations and count the calls, so the claim
    is measured rather than assumed.

    ``packed_gemv`` is the oracle seam: ``None`` resolves relic-core's operator by name, and an
    explicit callable (the tests' same-contract reference) replaces it. ``dtype`` defaults to
    ``x.dtype``.

    ``prefill`` decides **per call**, not per contribution, which projection path this is. It matters
    because the store routes a whole call and then hands each expert the rows that drew it: in a
    multi-row prefill an expert can own exactly one of those rows, and deciding by ``rows`` alone would
    silently run the decode operator for that contribution and the widening for the others. The store
    therefore passes ``prefill=n > 1`` for the call; a standalone caller that leaves it ``None`` gets
    the obvious reading of its own row count.
    """
    dtype = x.dtype if dtype is None else dtype
    if resident.block != BLOCK:
        raise ValueError(
            f"the resident expert declares a {resident.block}-wide K block; the activation quantizer "
            f"and the operator contract are stated on {BLOCK}"
        )
    block_fp8_quant, _, _ = _core()
    quantize = block_fp8_quant.quantize_block_fp8
    rows = int(x.shape[0])
    if prefill is None:
        prefill = rows > 1
    gemv = None if prefill else _resolve_packed_gemv(packed_gemv)
    if gemv is not None and rows != 1:
        raise ValueError(
            f"the packed decode operator is defined for one activation row, but this call was told "
            f"it is not a prefill and has {rows} rows; the store decides that per call, so this is a "
            "caller building the arguments by hand"
        )

    # One quantization, read by both projections. The reference calls the quantizer twice; the bytes
    # are identical because the quantizer is deterministic, and this is the only place this provider
    # departs from the bounded store's call structure. `tests/test_v41_ascend_packed_experts.py`
    # asserts both halves of that: the bytes equal an independent second quantization, and the call
    # count is two per expert forward (this one and the `w2` operand).
    codes, scales = quantize(x, backend=backend)

    gate = _project(codes, scales, resident.w1, gemv=gemv, backend=backend)
    up = _project(codes, scales, resident.w3, gemv=gemv, backend=backend)
    gate = gate.to(dtype).float()
    up = up.to(dtype).float()
    if swiglu_limit > 0:
        up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
        gate = torch.clamp(gate, max=swiglu_limit)
    out = F.silu(gate) * up
    if weights is not None:
        out = weights * out
    out_codes, out_scales = quantize(out.to(dtype), backend=backend)
    w2 = _project(out_codes, out_scales, resident.w2, gemv=gemv, backend=backend)
    return w2.to(dtype)


def _project(
    a_codes: torch.Tensor,
    a_scale: torch.Tensor,
    matrix: PackedMatrix,
    *,
    gemv,
    backend: str,
) -> torch.Tensor:
    """One projection: the M=1 packed operator, or the bounded widening plus the block GEMM."""
    if gemv is not None:
        return _call_packed_gemv(gemv, a_codes, a_scale, matrix, backend=backend)
    _, block_gemm, _ = _core()
    weight = matrix.to_int8_x2(backend=backend)  # bounded: one projection, dropped below
    out = block_gemm.fp4_int8_block_gemm(
        a_codes, a_scale, weight.values, weight.half_scale, backend=backend
    )
    del weight
    return out