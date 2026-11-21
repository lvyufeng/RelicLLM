"""The packed expert store, resident in one **tile-major** HBM bank per projection.

`ascend_packed_experts.AscendPackedRoutedExperts` already keeps the released experts in HBM in the
file's own packed encoding -- 18,800,640 bytes an expert, 33.62 GiB a card at the released geometry --
and executes them without storing a widened form. What it does **not** change is the *order* of those
bytes: a projection stays ``[out, in // 2]`` code bytes beside ``[out, in // 32]`` scale bytes, i.e.
row-major over output features. The M=1 decode operator therefore reads a weight matrix one output
row at a time, and on Ascend that is a strided gather across HBM for every decoded token.

This module is the same residency with the *same bytes* in a **tile-major** order: a projection is one
4-D bank of 16x16 tiles, so the values a single tile-shaped kernel slice needs -- sixteen output rows
over one 32-wide K block -- are one contiguous run. Byte counts are unchanged; only the permutation
is. The released checkpoint, the encodings, the quantization and the reduction are all untouched:

* the bytes still come from ``checkpoint.packed(...)`` through a ``uint8`` **dtype view** and are
  copied, never converted, widened or requantized;
* the resident form is still the packed form -- no INT8-times-two bank, no dense BF16 bank, no second
  copy of the model's experts;
* the arithmetic around the projections is inherited, not re-implemented: this class **subclasses**
  the frozen packed provider and overrides only (a) how the residency is laid out and filled and
  (b) which matrix object a resident expert is made of.

The tile
--------
A projection is ``N`` output features by ``K`` in-features, ``[N, K]``. It is cut into
``NT = ceil(N / 16)`` row tiles of sixteen rows and ``KB = K / 32`` K blocks of thirty-two values --
one block is exactly one E8M0 scale, so a tile is the smallest rectangle that owns a whole set of
scales for a row. A tile is stored as ``[16, 16]`` code bytes: sixteen rows by sixteen bytes, and
sixteen bytes is thirty-two values at two codes a byte, low nibble first along K. The bank is

    codes   [NT, KB, 16, 16]  uint8     NT * KB * 256 == N * K / 2   bytes
    scales  [NT, KB, 16]      uint8     NT * KB * 16  == N * K / 32  bytes

per owned expert, in one allocation each per projection per layer -- six tensors a layer, the same
count the row-major packed store allocates, holding the same number of bytes. The scale tile's last
axis is its **sixteen rows**, not sixteen blocks: that is what puts one row group's scales for one K
block in sixteen contiguous bytes.

Why 16 and 32 and not a wider tile: 16 rows is the tile shape the decode operator is written for, and
32 values is the scale's own period, so a tile never straddles a scale group. A wider tile would need
each tile's scales to be a separate rectangle *and* would put more than one row group behind one
pointer; this keeps the invariant "one tile, one contiguous run, one set of scales".

Padding is not needed and is not paid, but is still correct
----------------------------------------------------------
A tile spans sixteen rows, so a bank needs ``ceil(N / 16)`` tiles. For this checkpoint it is exactly
``N / 16``: a projection's output count is ``inter_dim`` or ``dim``, and each of those is also a
projection's **in**-feature count, which the block-32 grid requires to be a multiple of 32 -- and a
multiple of 32 is a multiple of 16. :func:`_padded_rows` still pads for a geometry that is not this
one, :func:`tile_matrix_bytes` charges for it when it does, and the decode path trims the padded
tail through the operator's own ``n_output`` instead of returning rows the model never asked for. The
tests assert the released geometry pays nothing, and that the padded case is charged and trimmed.

What is inherited and must stay inherited
-----------------------------------------
Everything that decides *numbers*: the constructor's shape and budget validation, the ``[n, topk]``
routing contract, the cyclic expert deal and ownership, the ascending-expert-id walk into one FP32
routed sum, the single activation quantization shared by ``w1``/``w3``, the narrow/widen/FP32-clamp
order, ``swiglu_limit`` before ``silu``, the routing weight on the **intermediate** activation, and
the BF16 cast of that intermediate before the ``w2`` quantization. This module overrides
:meth:`~AscendTileMajorRoutedExperts.preload` and
:meth:`~AscendTileMajorRoutedExperts._expert_at` and nothing else about arithmetic -- ``forward`` and
``packed_expert_forward`` are the base's, unchanged.

The two projection paths
------------------------
``rows == 1`` (a decode step) goes through the base's M=1 hook, which calls the function this store
resolved at construction: relic-core's **tile-major** packed decode operator (module
``relic_core.kernels.tile_major_packed_gemv``), looked up by name, or an explicitly injected callable.
Because the base's hook passes ``(a_codes, a_scale, codes, scales, backend=...)``, the resolver
signature is exactly that: the two activation operands, then this store's two tile banks.

``rows > 1`` (a prefill) is unchanged in kind: the base widens **one projection at a time** to the
accepted INT8-times-two operands and calls the accepted block GEMM. This store supplies that
operand's bytes by inverting the permutation for that one projection into a bounded temporary, which
is dropped when the projection is done -- never an expert, never a layer, never resident.

The permutation is written twice on purpose
------------------------------------------
:func:`pack_code_tiles`/:func:`pack_scale_tiles` and their inverses are this store's own statement of
the layout, and relic-core ships `pack_tile_major`/`unpack_tile_major` as the operator's. They are
byte-for-byte the same function of the same inputs, and the tests compare the two rather than trusting
either: an agreement test is what makes "the bank the kernel expects" a checked claim instead of two
transcriptions that happen to agree today.

What is not claimed
-------------------
No token parity, no tokens-per-second claim, no graph capture, no native FP4/FP8 arithmetic, and no
bounded-run capability: the ``v41`` Ascend capability guard is untouched. The provider is **opt-in**
and allocates nothing until ``preload=True`` or :meth:`preload` is called. The old row-major provider
remains the default and is not modified by this file; the whole-model decision to prefer this one is
the launcher's, and needs a whole-model bit-exact proof first -- a layout is a hypothesis until then.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .ascend_experts import BLOCK
from .ascend_packed_experts import (
    _PROJECTIONS,
    _allocate,
    _stored_bytes,
    _tensor_nbytes,
    AscendPackedRoutedExperts,
    DEFAULT_RESERVE_BYTES,
    PackedExpert,
)

__all__ = [
    "AscendTileMajorRoutedExperts",
    "TileDecodeUnavailable",
    "TileMatrix",
    "TILE_DECODER_MODULE",
    "TILE_DECODER_NAME",
    "TILE_ROWS",
    "pack_code_tiles",
    "pack_scale_tiles",
    "tile_matrix_bytes",
    "unpack_code_tiles",
    "unpack_scale_tiles",
]

#: Rows per tile. Sixteen is the row group the decode operator's slice is shaped for, and it keeps a
#: tile's scale rectangle integral because thirty-two values (one E8M0 scale) is sixteen code bytes.
TILE_ROWS = 16

#: The E8M0 byte for `2**0`: the identity scale, used only to fill a padded row tail (which every
#: read masks out). Kept equal to relic-core's `IDENTITY_SCALE` so a padded bank is byte-comparable.
_IDENTITY_SCALE = 127

#: Where relic-core's tile-major packed decode operator lives, and its name. One spelling, one lookup:
#: the store resolves it once and never substitutes the row-major decoder, which would read the tile
#: bank as if it were ``[N, K // 2]`` and produce plausible garbage instead of an error.
TILE_DECODER_MODULE = "relic_core.kernels.tile_major_packed_gemv"
TILE_DECODER_NAME = "fp4_packed_tile_major_decode_gemv"


class TileDecodeUnavailable(RuntimeError):
    """The tile-major packed decode operator is not installed in the relic-core on this path.

    Raised, never swallowed. The row-major operator is **not** a fallback: it reads a different byte
    order, so answering with it would return numbers from the wrong addresses rather than an error,
    and answering with the prefill path's temporary widening would allocate exactly the INT8 operand
    residency exists to avoid.
    """


def _padded_rows(rows: int) -> int:
    """``rows`` rounded up to a whole number of tiles; zero stays zero."""
    if rows <= 0:
        return 0
    return ((rows + TILE_ROWS - 1) // TILE_ROWS) * TILE_ROWS


def _shape_args(rows: int, cols: int) -> tuple[int, int, int]:
    """Validate a logical ``[rows, cols]`` projection and return ``(padded, row_tiles, k_blocks)``."""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows <= 0:
        raise ValueError(f"rows must be a positive int, got {rows!r}")
    if isinstance(cols, bool) or not isinstance(cols, int) or cols <= 0:
        raise ValueError(f"cols must be a positive int, got {cols!r}")
    if cols % BLOCK:
        raise ValueError(
            f"cols {cols} is not a whole number of {BLOCK}-wide K blocks, so the raw E8M0 grid this "
            "tile-major bank reorders cannot exist"
        )
    padded = _padded_rows(rows)
    return padded, padded // TILE_ROWS, cols // BLOCK


def pack_code_tiles(codes: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Reorder row-major code bytes ``[rows, cols // 2]`` into ``[NT, KB, 16, 16]``.

    A pure permutation plus (for a width that needs it) a zero-padded row tail: the bytes are moved,
    never reinterpreted, so a nibble is still a nibble and an exponent byte is still an exponent byte.
    """
    padded, row_tiles, k_blocks = _shape_args(rows, cols)
    expected = (rows, cols // 2)
    if tuple(codes.shape) != expected:
        raise ValueError(f"code bytes must be {expected}, got {tuple(codes.shape)}")
    if codes.dtype != torch.uint8:
        raise TypeError(f"code bytes must be uint8, got {codes.dtype}")
    if padded == rows:
        full = codes
    else:
        full = codes.new_zeros((padded, cols // 2))
        full[:rows].copy_(codes)
    # [NT, 16 rows, KB, 16 code bytes] -> [NT, KB, 16 rows, 16 code bytes]
    return full.reshape(row_tiles, TILE_ROWS, k_blocks, TILE_ROWS).permute(0, 2, 1, 3).contiguous()


def pack_scale_tiles(scales: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Reorder the raw E8M0 grid ``[rows, cols // 32]`` into ``[NT, KB, 16]`` (last axis: rows)."""
    padded, row_tiles, k_blocks = _shape_args(rows, cols)
    expected = (rows, cols // BLOCK)
    if tuple(scales.shape) != expected:
        raise ValueError(f"scale bytes must be {expected}, got {tuple(scales.shape)}")
    if scales.dtype != torch.uint8:
        raise TypeError(f"scale bytes must be uint8, got {scales.dtype}")
    if padded == rows:
        full = scales
    else:
        # The padded tail is masked out of every read, so its value cannot change a result; `127` is
        # the identity scale (`2**0`) and matches relic-core's own pack, which keeps the two banks
        # comparable byte for byte even on a geometry that needs padding.
        full = scales.new_full((padded, cols // BLOCK), _IDENTITY_SCALE)
        full[:rows].copy_(scales)
    return full.reshape(row_tiles, TILE_ROWS, k_blocks).permute(0, 2, 1).contiguous()


def unpack_code_tiles(tiles: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """The inverse of :func:`pack_code_tiles`: back to row-major ``[rows, cols // 2]``."""
    padded, row_tiles, k_blocks = _shape_args(rows, cols)
    expected = (row_tiles, k_blocks, TILE_ROWS, TILE_ROWS)
    if tuple(tiles.shape) != expected:
        raise ValueError(f"code tiles must be {expected}, got {tuple(tiles.shape)}")
    row_major = tiles.permute(0, 2, 1, 3).reshape(padded, cols // 2)
    return row_major[:rows].contiguous()


def unpack_scale_tiles(tiles: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """The inverse of :func:`pack_scale_tiles`: back to ``[rows, cols // 32]``."""
    padded, row_tiles, k_blocks = _shape_args(rows, cols)
    expected = (row_tiles, k_blocks, TILE_ROWS)
    if tuple(tiles.shape) != expected:
        raise ValueError(f"scale tiles must be {expected}, got {tuple(tiles.shape)}")
    row_major = tiles.permute(0, 2, 1).reshape(padded, cols // BLOCK)
    return row_major[:rows].contiguous()


def tile_matrix_bytes(rows: int, cols: int) -> int:
    """Resident bytes of one projection in the tile-major form, padding included.

    For a geometry whose ``rows`` is already a multiple of sixteen this equals the row-major count
    ``rows * cols / 2 + rows * cols / 32`` exactly -- the released geometry always is, because both
    the output and in-feature counts of these projections are multiples of the 32-wide K block.
    """
    padded, row_tiles, k_blocks = _shape_args(rows, cols)
    return row_tiles * k_blocks * TILE_ROWS * TILE_ROWS + row_tiles * k_blocks * TILE_ROWS


@dataclass(frozen=True)
class TileMatrix:
    """One projection in tile-major order, as views of a store's arena.

    ``codes`` is ``[NT, KB, 16, 16]`` and ``scales`` is ``[NT, KB, 16]``, both ``uint8`` and both
    views -- constructing one of these allocates nothing. ``shape`` is the **logical**
    ``(out_features, in_features)`` matrix the encoding describes; it is not recoverable from the
    tiles alone once padding is in play, which is why it travels with them.

    ``inverse_codes``/``inverse_scales`` are the permutation's inverse, carried on the object so the
    prefill path's bounded temporary can be built without reaching for a module global a test would
    then have to patch. The carrier holds **strong references** to every field: a store's arenas, the
    shape and the two helpers outlive any one call.
    """

    codes: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, int]
    block: int = BLOCK
    inverse_codes: object = field(default=unpack_code_tiles)
    inverse_scales: object = field(default=unpack_scale_tiles)

    @property
    def block32(self) -> int:
        """The K-block width the raw grid was written on; 32 for the released experts."""
        return self.block

    @property
    def nbytes(self) -> int:
        return _tensor_nbytes(self.codes) + _tensor_nbytes(self.scales)

    @property
    def codesTile4D(self) -> torch.Tensor:
        """The code bank as stored: ``[NT, KB, 16, 16]``."""
        return self.codes

    @property
    def scalesTile3D(self) -> torch.Tensor:
        """The scale bank as stored: ``[NT, KB, 16]``."""
        return self.scales

    def to_int8_x2(self, *, backend: str = "auto"):
        """The accepted INT8-times-two operands of this projection, as a **bounded temporary**.

        This is the prefill path, not a resident form: the tile bank is inverted back to the file's
        row-major byte order **for this one projection**, widened by relic-core's frozen
        ``unpack_fp4_to_int8``/``fp4_int8_scale``, and dropped when the projection's GEMM is done. The
        inversion is the exact inverse of :func:`pack_code_tiles`/:func:`pack_scale_tiles`, so what
        the frozen operators see is the file's own bytes in the file's own order.
        """
        from .ascend_packed_experts import _core
        from .online_unpack import Fp4ExpertInt8

        _, _, unpack = _core()
        rows, cols = self.shape
        codes = self.inverse_codes(self.codes, rows, cols)
        scales = self.inverse_scales(self.scales, rows, cols)
        values = unpack.unpack_fp4_to_int8(codes, backend=backend)
        half_scale = unpack.fp4_int8_scale(scales, backend=backend)
        return Fp4ExpertInt8(values=values, half_scale=half_scale, block=self.block)


@dataclass(frozen=True)
class _TileArena:
    """One projection's residency for every owned expert, in tile-major order.

    ``codes`` is ``[owned, NT, KB, 16, 16]`` and ``scales`` is ``[owned, NT, KB, 16]``, both ``uint8``
    -- one allocation each per projection per layer, so an expert is a view and not a copy, and the
    layer's whole residency is six tensors rather than one per expert.
    """

    which: str
    codes: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, int]
    pack_codes: object = field(default=pack_code_tiles)
    pack_scales: object = field(default=pack_scale_tiles)
    inverse_codes: object = field(default=unpack_code_tiles)
    inverse_scales: object = field(default=unpack_scale_tiles)

    @property
    def nbytes(self) -> int:
        return _tensor_nbytes(self.codes) + _tensor_nbytes(self.scales)

    def matrix(self, slot: int) -> TileMatrix:
        return TileMatrix(
            codes=self.codes[slot],
            scales=self.scales[slot],
            shape=self.shape,
            block=BLOCK,
            inverse_codes=self.inverse_codes,
            inverse_scales=self.inverse_scales,
        )


def _import_tile_decode():
    """relic-core's tile-major M=1 operator, or a named error saying why it is not usable.

    The row-major decoder is deliberately **not** consulted: it reads a different byte order, so a
    store that answered with it would be reading addresses that hold another tile's bytes.
    """
    import importlib

    try:
        module = importlib.import_module(TILE_DECODER_MODULE)
    except ImportError as missing:
        raise TileDecodeUnavailable(
            "the tile-major packed decode operator is not installed in the relic-core on this path "
            f"({TILE_DECODER_MODULE}: {missing}). This store holds a tile-major bank, so the "
            "row-major operator is not a fallback -- it would read the wrong bytes. Install "
            "relic-core from its checkout, or pass tile_gemv= explicitly for an oracle"
        ) from missing
    function = getattr(module, TILE_DECODER_NAME, None)
    if function is None:
        raise TileDecodeUnavailable(
            f"the installed relic-core has {TILE_DECODER_MODULE} but no `{TILE_DECODER_NAME}` in it; "
            "a tile-major store cannot be decoded by the row-major operator, and this store does not "
            "fall back to the prefill path's widening"
        )
    if not callable(function):
        raise TileDecodeUnavailable(
            f"{TILE_DECODER_MODULE}.{TILE_DECODER_NAME} is {type(function).__name__}, not callable; "
            "refusing to treat it as the M=1 operator"
        )
    return function


class AscendTileMajorRoutedExperts(AscendPackedRoutedExperts):
    """One layer's routed experts resident in HBM in the file's bytes, **tile-major**.

    A specialization of the frozen packed provider: the same experts, the same encodings, the same
    arithmetic, the same deal, the same budget admission -- with the resident bytes permuted into
    sixteen-row tiles so a decode step's slice is contiguous instead of strided.

    Construction validates and allocates **nothing**. ``preload=True`` (or a later :meth:`preload`)
    validates every owned expert's stored bytes, checks the card's headroom, and only then allocates
    six tile arenas a layer and copies the file's bytes into them through the permutation.

    ``tile_gemv`` is the oracle seam: ``None`` resolves relic-core's tile-major operator by name at
    the first decode call (so constructing a store on a host without it is still possible, and a store
    used only for prefills never needs it), and an explicit callable replaces it -- the tests inject a
    same-contract reference. Passing one says so; there is no quiet fallback in either direction.
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
        tile_gemv=None,
        pack_codes=None,
        pack_scales=None,
        inverse_codes=None,
        inverse_scales=None,
    ) -> None:
        if tile_gemv is not None and not callable(tile_gemv):
            raise TypeError(
                f"tile_gemv must be callable or None, got {type(tile_gemv).__name__}; it stands in for "
                "relic-core's tile-major M=1 packed decode operator"
            )
        self._tile_gemv_injected = tile_gemv
        self._tile_gemv_resolved = None
        # The base validates the deal, the shapes and the budget, and sets every attribute the
        # inherited forward reads. `packed_gemv` is this store's own dispatch, so the base's M=1 hook
        # reaches the tile operator and never the row-major one.
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
            backend=backend,
            budget_bytes=budget_bytes,
            reserve_bytes=reserve_bytes,
            preload=False,
            packed_gemv=self._tile_gemv_dispatch,
        )
        self._pack_codes = pack_codes or pack_code_tiles
        self._pack_scales = pack_scales or pack_scale_tiles
        self._inverse_codes = inverse_codes or unpack_code_tiles
        self._inverse_scales = inverse_scales or unpack_scale_tiles
        # Tile geometry is decided once, from the same expected shapes the base validated against.
        self._tile_shapes: dict[str, tuple[int, int]] = {
            which: self._expected_shapes[which] for which in _PROJECTIONS
        }
        # The base charged the row-major count, which equals the tile-major count whenever the output
        # count divides into whole tiles -- always, for a block-32 geometry. Recompute so a geometry
        # that does need a padded tail is charged for it rather than overrunning its own arena.
        row_major = self.required_bytes
        self.required_bytes = len(self.owned) * sum(
            tile_matrix_bytes(*self._tile_shapes[which]) for which in _PROJECTIONS
        )
        if self.required_bytes != row_major:
            padded_budget = budget_bytes if budget_bytes is not None else (
                row_major + self.reserve_bytes
            )
            if padded_budget < self.required_bytes:
                raise ValueError(
                    f"layer {self.layer_id}: the padded tile bank needs {self.required_bytes} bytes "
                    f"but the budget is {padded_budget}; residency is this provider's whole design, "
                    "so a store that cannot hold its experts is refused rather than built"
                )
            self.budget_bytes = padded_budget
        if preload:
            self.preload()

    # -- the M=1 dispatch -----------------------------------------------------

    def _logical_rows(self, b_codes: torch.Tensor) -> int | None:
        """The logical output count a tile bank holds, or ``None`` when it is the physical one.

        Every projection of this layer is a candidate by its physical geometry; when they disagree
        about the logical count the answer is ambiguous and the operator's default (physical) is used,
        which is exact for the released geometry and only wrong for a padded bank the validator cannot
        produce. Returning ``None`` means "omit ``n_output``", not "unknown".
        """
        physical_rows = int(b_codes.shape[0])
        k_blocks = int(b_codes.shape[1])
        candidates = {
            rows
            for which in _PROJECTIONS
            for rows, cols in (self._tile_shapes[which],)
            if _shape_args(rows, cols)[1] == physical_rows
            and _shape_args(rows, cols)[2] == k_blocks
        }
        if len(candidates) != 1:
            return None
        rows = candidates.pop()
        return None if rows == physical_rows * TILE_ROWS else rows

    def _tile_gemv_dispatch(self, a_codes, a_scale, b_codes, b_scales, *, backend="auto"):
        """The base's M=1 hook, answered by the tile operator.

        Called as ``function(a_codes, a_scale, matrix.codes, matrix.scales, backend=...)``, so the
        operands are this store's two tile banks. The operator is resolved on first use and cached: an
        injected callable wins, otherwise relic-core's is imported **by name** once. ``n_output`` is
        passed only when this store's bank carries a padded row tail, which the released geometry
        never does.
        """
        if self._tile_gemv_resolved is None:
            self._tile_gemv_resolved = self._tile_gemv_injected or _import_tile_decode()
        trim = self._logical_rows(b_codes)
        if trim is None:
            return self._tile_gemv_resolved(a_codes, a_scale, b_codes, b_scales, backend=backend)
        return self._tile_gemv_resolved(
            a_codes, a_scale, b_codes, b_scales, backend=backend, n_output=trim
        )

    # -- reporting ------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"AscendTileMajorRoutedExperts(layer={self.layer_id}, experts={self.n_experts}, "
            f"owned={len(self.owned)}, topk={self.topk}, device={self.device}, dtype={self.dtype}, "
            f"backend={self.backend}, rank={self.rank}/{self.world}, "
            f"required_bytes={self.required_bytes}, preloaded={self.allocated})"
        )

    @property
    def strategy(self) -> str:
        """Which residency layout this store holds. Read by launchers that log the choice."""
        return "tile-major"

    @property
    def tile_shapes(self) -> dict:
        """Per projection: logical ``(rows, cols)``, padded rows, row tiles, K blocks and shapes."""
        report = {}
        for which in _PROJECTIONS:
            rows, cols = self._tile_shapes[which]
            padded, row_tiles, k_blocks = _shape_args(rows, cols)
            report[which] = {
                "rows": rows,
                "cols": cols,
                "padded_rows": padded,
                "row_tiles": row_tiles,
                "k_blocks": k_blocks,
                "codes_shape": (row_tiles, k_blocks, TILE_ROWS, TILE_ROWS),
                "scales_shape": (row_tiles, k_blocks, TILE_ROWS),
                "bytes": tile_matrix_bytes(rows, cols),
            }
        return report

    @property
    def cache_stats(self) -> dict:
        """The base's snapshot plus the layout this store is holding.

        Every base key keeps its meaning -- ``retained_bytes`` is the residency, ``entries`` is the
        experts held, and the hit counters are the store's own because a preloaded store cannot miss.
        ``strategy`` is what a launcher logs beside them to say *which* residency was measured.
        """
        stats = super().cache_stats
        # The base names the snapshot after the row-major provider; this store's residency is a
        # different object, so a log that says which one was measured is the point of the field.
        stats["name"] = f"ascend-tile-major-experts/layer{self.layer_id}"
        stats["strategy"] = self.strategy
        stats["tile_rows"] = TILE_ROWS
        stats["tile_banks"] = 0 if self._arenas is None else len(self._arenas)
        return stats

    @property
    def expert_stats(self) -> dict:
        stats = super().expert_stats
        stats["strategy"] = self.strategy
        stats["tile_rows"] = TILE_ROWS
        stats["tile_shapes"] = self.tile_shapes
        return stats

    # -- residency ------------------------------------------------------------

    def preload(self) -> int:
        """Validate, then make every owned expert resident in the tile-major bank.

        Validation and the headroom check are the base's and run **before any allocation**, so a
        checkpoint that is not the packed original, or a card that cannot hold the store, fails
        without touching HBM. Then six tile arenas a layer are allocated and each owned expert's
        stored bytes are permuted into their slot -- one host-side permutation per projection, then
        one contiguous copy into the bank, so no per-row discrete copies are issued.

        Idempotent: a second call on a preloaded store does nothing and reports the bytes held.
        """
        if self._arenas is not None:
            return self.resident_bytes
        self._validate_checkpoint_bytes()
        headroom = self._check_memory_headroom()
        arenas: dict[str, _TileArena] = {}
        for which in _PROJECTIONS:
            rows, cols = self._tile_shapes[which]
            padded, row_tiles, k_blocks = _shape_args(rows, cols)
            arenas[which] = _TileArena(
                which=which,
                codes=_allocate(
                    (len(self.owned), row_tiles, k_blocks, TILE_ROWS, TILE_ROWS), self.device
                ),
                scales=_allocate((len(self.owned), row_tiles, k_blocks, TILE_ROWS), self.device),
                shape=(rows, cols),
                pack_codes=self._pack_codes,
                pack_scales=self._pack_scales,
                inverse_codes=self._inverse_codes,
                inverse_scales=self._inverse_scales,
            )
        for slot, expert in enumerate(self.owned):
            for which in _PROJECTIONS:
                key = self._key(expert, which)
                scale_name = self._scale_key(expert, which)
                rows, cols = self._tile_shapes[which]
                codes = _stored_bytes(self.checkpoint.packed(key), f"{key} codes").contiguous()
                scales = _stored_bytes(self.checkpoint.packed(scale_name), f"{scale_name}").contiguous()
                # A permutation followed by `copy_` between `uint8` tensors: the encodings move, they
                # are not converted, and the shapes were checked by the base's validation just above.
                arenas[which].codes[slot].copy_(self._pack_codes(codes, rows, cols))
                arenas[which].scales[slot].copy_(self._pack_scales(scales, rows, cols))
        self._arenas = arenas
        self._headroom = headroom
        return self.resident_bytes

    def _expert_at(self, expert: int) -> PackedExpert:
        """One expert's three projections as tile views of this store's arenas.

        ``PackedExpert`` is the base's carrier and its field annotations are not enforced at runtime,
        which is what lets a tile matrix travel through the inherited ``forward`` and
        ``packed_expert_forward`` unchanged: those only read ``.w1``/``.w2``/``.w3`` and the methods a
        matrix has to answer (``shape``, ``nbytes``, ``to_int8_x2``, plus the two tile views and the
        ``.codes``/``.scales`` operands the M=1 hook passes).
        """
        slot = self._slots[expert]
        arenas = self._arenas
        if arenas is None:  # pragma: no cover - every caller checks `allocated` first
            raise RuntimeError("the tile-major expert store is not preloaded")
        return PackedExpert(
            w1=arenas["w1"].matrix(slot),
            w2=arenas["w2"].matrix(slot),
            w3=arenas["w3"].matrix(slot),
            block=self.block,
        )