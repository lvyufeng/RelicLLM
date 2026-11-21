"""Reversible same-size CubeB-order physical layout for packed FP4 weights.

This module is a *format*, not a kernel: it rewrites the bytes of a packed-FP4 weight
projection into an order an Ascend Cube unit can consume without a gather, and back again.
It decodes nothing, quantizes nothing, and never changes a value -- the output has exactly
as many bytes as the input, so the transform is a permutation of bytes and is lossless by
construction. Signed zeros, both FP4 zeros, saturating codes and any code a consumer treats
as non-finite all survive because they are never interpreted.

Recorded format: ``relic.fp4.cube-n32-k128.v1``.

Source layout
-------------
``codes``  ``uint8``/``int8`` ``[N, K/2]`` -- two E2M1 codes per byte, low nibble first, so
byte ``j`` of a row holds logical columns ``2j`` and ``2j + 1``. This is the released
packed-FP4 layout (the one ``relic_core.kernels.unpack`` reads), and it is why one output
element below is a *byte* rather than a value.

``scales`` ``uint8`` ``[N, K/32]`` -- one raw E8M0 byte per 32-wide K block.

Geometry: ``N % 32 == 0`` and ``K % 128 == 0``. Those are the Cube tile's own multiples, so
a projection either fits the format exactly or is refused: there is no padding, no tail
handling and no masked last tile.

CubeB order
-----------
The 32 rows of a tile are addressed as ``(n16, row)`` -- ``n16`` picks one of two 16-row
halves, ``row`` the row inside the half -- and the tile's 128 K values as ``(k16, kpair)``
-- ``k16`` picks one of eight 16-value groups, ``kpair`` the packed byte inside that group.
The only physical statement of the order is these two lines:

.. code-block:: python

    codes_cube  = codes.view(N // 32, 2, 16, K // 128, 8, 8).permute(0, 3, 4, 1, 2, 5)
    scales_cube = scales.view(N // 32, 32, K // 128, 4).permute(0, 2, 3, 1)

Their shapes are ``[N/32, K/128, 8, 2, 16, 8]`` and ``[N/32, K/128, 4, 32]``. Spelled as
coordinate arithmetic (``//`` and ``%`` are on the corresponding extent), for source byte
``codes[n, j]``:

* ``nb, n16, row = n // 32, (n % 32) // 16, n % 16``
* ``kb, k16, kpair = j // 64, (j % 64) // 8, j % 8``
* destination ``codes_cube[nb, kb, k16, n16, row, kpair]``

and for ``scales[m, s]`` with ``s = kb * 4 + group``:

* ``nb, nrow = m // 32, m % 32``
* destination ``scales_cube[nb, kb, group, nrow]``

The code permutation is an involution on the dims it touches -- ``(1 3)(2 4)`` -- so its
own inverse is stated with the same tuple. The scale permutation is the three-cycle
``(1 2 3)``, so recovery states its inverse explicitly:

.. code-block:: python

    codes  = codes_cube.permute(0, 3, 4, 1, 2, 5).reshape(N, K // 2)
    scales = scales_cube.permute(0, 3, 1, 2).reshape(N, K // 32)

The nibble pair is never separated: ``kpair`` addresses whole bytes, and the two codes
inside a byte stay ``2 * kpair`` and ``2 * kpair + 1``, so the low-nibble-first convention
survives the move.

Byte budget
-----------
``N*K/2`` code bytes plus ``N*K/32`` scale bytes in, and exactly the same out:
``codes_bytes + scales_bytes == payload_bytes == source_bytes``, with no padding term.
:meth:`CubeGeometry.report` states this explicitly, so a caller can assert the invariant
instead of trusting it.

Scope and device policy
-----------------------
Bounded to *one projection at a time*: the utilities take a single ``[N, K]`` weight and
materialize a single payload of the same size. There is deliberately no bank-level driver,
because that is where the resident footprint would stop being the caller's choice.

This is an offline CPU-side repack. Inputs are accepted on the CPU and on a non-CUDA
accelerator (an NPU-resident weight can be serialized for a host-side repack); the payload
is always materialized on the CPU, and an explicit non-CPU output device is refused, as is a
CUDA input -- a GPU copy here would be an implicit transfer, which this utility does not
perform. Nothing here allocates device memory or launches work.

``torch`` is imported inside the entry points, so importing this module is free and the
geometry helpers work without a tensor library.

Digests are the caller's business: this module does not hash a payload. Hash the source
``codes``/``scales`` bytes yourself if a provenance check is needed; keeping it out keeps
the API one payload in, one payload out.
"""
from __future__ import annotations

from dataclasses import dataclass
import operator

__all__ = [
    "FORMAT",
    "CubeGeometry",
    "CubePackedFP4",
    "cube_geometry",
    "convert_fp4_cube",
    "recover_fp4_cube",
]

#: The recorded format name. A payload that does not declare this is refused, so a v2 byte
#: order can never be read with v1's formulas.
FORMAT = "relic.fp4.cube-n32-k128.v1"

#: Cube tile extents: 32 rows, 128 K values (64 packed code bytes).
CUBE_ROWS = 32
CUBE_K = 128
#: Scale granularity along K, unchanged by the layout.
SCALE_K = 32

#: The four CubeB axes, derived from the tile extents rather than restated.
#: ``row`` is the extent inside a 16-row half; ``n16`` counts the halves.
_HALF_ROWS = 16
_ROWS_N16 = CUBE_ROWS // _HALF_ROWS  # 2
#: A 16-value group is 8 packed bytes, so 128 K values are 8 groups of 8 packed bytes.
_KPAIR = 8
_K16 = CUBE_K // (_KPAIR * 2)  # 8
#: 128 K values are four 32-wide scale blocks.
_SCALE_GROUPS = CUBE_K // SCALE_K  # 4

#: The dim orders, written once so the forward and inverse statements cannot drift.
#: Codes: ``(nb, n16, row, kb, k16, kpair) -> (nb, kb, k16, n16, row, kpair)``.
_CODE_PERMUTE = (0, 3, 4, 1, 2, 5)
#: Scales: ``(nb, nrow, kb, group) -> (nb, kb, group, nrow)``, and its inverse.
_SCALE_PERMUTE = (0, 2, 3, 1)
_SCALE_PERMUTE_INVERSE = (0, 3, 1, 2)

#: Code containers: the packed bytes, and the signed spelling of the same bytes.
_CODE_DTYPES = ("uint8", "int8")


def _checked_count(value, name: str) -> int:
    """A positive tensor extent, refusing ``bool`` and anything not integral.

    ``bool`` is an ``int`` subclass, so ``True`` would otherwise pass as a count of one and
    a caller's mistaken flag would silently become a shape.
    """
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        count = operator.index(value)
    except TypeError:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}") from None
    if count <= 0:
        raise ValueError(f"{name} must be positive, got {count}")
    return count


@dataclass(frozen=True)
class CubeGeometry:
    """The shapes and the exact byte budget of one ``[n, k]`` projection.

    Constructing one validates the tile multiples, so an unusable geometry cannot be
    carried around as a value. The shapes are properties rather than stored fields: they
    are pure functions of ``(n, k)``, and duplicating them invites a mismatch.
    """

    n: int
    k: int

    def __post_init__(self) -> None:
        n = _checked_count(self.n, "n")
        k = _checked_count(self.k, "k")
        if n % CUBE_ROWS:
            raise ValueError(
                f"n must be a multiple of {CUBE_ROWS} for {FORMAT}, got n={n} (the format "
                f"has no padding, so a partial 32-row tile is refused rather than masked)"
            )
        if k % CUBE_K:
            raise ValueError(
                f"k must be a multiple of {CUBE_K} for {FORMAT}, got k={k} (the format "
                f"has no padding, so a partial 128-K tile is refused rather than masked)"
            )

    # -- source layout -------------------------------------------------------------

    @property
    def source_codes_shape(self) -> tuple[int, int]:
        """``[n, k/2]`` packed code bytes, two E2M1 codes per byte."""
        return (self.n, self.k // 2)

    @property
    def source_scales_shape(self) -> tuple[int, int]:
        """``[n, k/32]`` raw E8M0 scale bytes."""
        return (self.n, self.k // SCALE_K)

    # -- cube layout ---------------------------------------------------------------

    @property
    def codes_view_shape(self) -> tuple[int, ...]:
        """The source bytes read as ``(nb, n16, row, kb, k16, kpair)`` before the permute."""
        return (
            self.n // CUBE_ROWS,
            _ROWS_N16,
            _HALF_ROWS,
            self.k // CUBE_K,
            _K16,
            _KPAIR,
        )

    @property
    def scales_view_shape(self) -> tuple[int, ...]:
        """The source scales read as ``(nb, nrow, kb, group)`` before the permute."""
        return (self.n // CUBE_ROWS, CUBE_ROWS, self.k // CUBE_K, _SCALE_GROUPS)

    @property
    def codes_shape(self) -> tuple[int, ...]:
        """``[n/32, k/128, 8, 2, 16, 8]``: ``(nb, kb, k16, n16, row, kpair)``."""
        return (self.n // CUBE_ROWS, self.k // CUBE_K, _K16, _ROWS_N16, _HALF_ROWS, _KPAIR)

    @property
    def scales_shape(self) -> tuple[int, ...]:
        """``[n/32, k/128, 4, 32]``: ``(nb, kb, group, nrow)``."""
        return (self.n // CUBE_ROWS, self.k // CUBE_K, _SCALE_GROUPS, CUBE_ROWS)

    # -- budget --------------------------------------------------------------------

    @property
    def codes_bytes(self) -> int:
        return self.n * (self.k // 2)

    @property
    def scales_bytes(self) -> int:
        return self.n * (self.k // SCALE_K)

    @property
    def payload_bytes(self) -> int:
        """Bytes after the transform: the two tensors, with no padding."""
        return self.codes_bytes + self.scales_bytes

    @property
    def source_bytes(self) -> int:
        """Bytes before the transform. Equal to :attr:`payload_bytes` by construction."""
        return self.codes_bytes + self.scales_bytes

    @property
    def padding_bytes(self) -> int:
        """Always ``0``: the format is size-exact, and this is the field that says so."""
        return 0

    def report(self) -> dict:
        """A JSON-friendly statement of the format, shapes and byte budget.

        Deliberately carries no digest: hashing stays with the caller, which keeps this
        cheap enough to call once per projection.
        """
        return {
            "format": FORMAT,
            "n": self.n,
            "k": self.k,
            "cube_rows": CUBE_ROWS,
            "cube_k": CUBE_K,
            "scale_k": SCALE_K,
            "source_codes_shape": list(self.source_codes_shape),
            "source_scales_shape": list(self.source_scales_shape),
            "codes_shape": list(self.codes_shape),
            "scales_shape": list(self.scales_shape),
            "codes_bytes": self.codes_bytes,
            "scales_bytes": self.scales_bytes,
            "source_bytes": self.source_bytes,
            "payload_bytes": self.payload_bytes,
            "padding_bytes": self.padding_bytes,
            "size_exact": self.payload_bytes == self.source_bytes,
        }

    def __str__(self) -> str:
        return f"{FORMAT}(n={self.n}, k={self.k}, bytes={self.payload_bytes})"


def cube_geometry(n, k) -> CubeGeometry:
    """The geometry of one ``[n, k]`` projection, validated. Needs no tensor library."""
    return CubeGeometry(n, k)


@dataclass(frozen=True)
class CubePackedFP4:
    """One projection in cube order: the two byte tensors plus the format they obey.

    The payload is the unit a native consumer is handed, so it carries its own format name
    and geometry and validates them on construction -- a tensor of the right size under the
    wrong declared geometry is refused instead of being silently reinterpreted.

    ``source_codes_dtype`` records only the *spelling* of the original code bytes
    (``"uint8"`` or ``"int8"``) so recovery can hand back the same dtype. It changes no
    byte: the two spellings are the same storage, and the transform never reads a code.
    """

    codes: "torch.Tensor"  # noqa: F821 - a string annotation keeps the import lazy
    scales: "torch.Tensor"  # noqa: F821
    geometry: CubeGeometry
    source_codes_dtype: str = "uint8"
    format: str = FORMAT

    def __post_init__(self) -> None:
        if self.format != FORMAT:
            raise ValueError(
                f"payload declares format {self.format!r}, but this reader implements "
                f"{FORMAT!r}; refusing to apply v1 formulas to another byte order"
            )
        if not isinstance(self.geometry, CubeGeometry):
            raise TypeError(
                f"geometry must be a CubeGeometry, got {type(self.geometry).__name__}"
            )
        if self.source_codes_dtype not in _CODE_DTYPES:
            raise ValueError(
                f"source_codes_dtype must be one of {_CODE_DTYPES}, got "
                f"{self.source_codes_dtype!r}"
            )
        _checked_payload_tensor(self.codes, "payload codes", self.geometry.codes_shape)
        _checked_payload_tensor(self.scales, "payload scales", self.geometry.scales_shape)
        _check_no_overlap(self.codes, self.scales, "payload codes", "payload scales")

    @property
    def n(self) -> int:
        return self.geometry.n

    @property
    def k(self) -> int:
        return self.geometry.k

    @property
    def payload_bytes(self) -> int:
        return self.geometry.payload_bytes

    def report(self) -> dict:
        """The geometry report plus the recorded code spelling."""
        report = dict(self.geometry.report())
        report["source_codes_dtype"] = self.source_codes_dtype
        return report


# -------------------------------------------------------------------------- validation


def _torch():
    """The tensor library, imported on use so importing this module stays free."""
    import torch

    return torch


def _device_kind(device) -> str:
    """The device type of a tensor, or of an explicit device argument, as a string."""
    kind = getattr(device, "type", None)
    if kind is None:
        kind = str(device).split(":")[0]
    return str(kind).lower()


def _require_device(tensor, name: str, allowed: tuple[str, ...]) -> None:
    """Refuse a device outside ``allowed``, naming the transfer the utility will not do."""
    kind = _device_kind(tensor.device)
    if kind in allowed:
        return
    if kind == "cuda":
        raise ValueError(
            f"{name} is on a CUDA device; this is an offline CPU-side repack and will not "
            f"perform an implicit GPU transfer -- move the projection to the CPU first"
        )
    raise ValueError(
        f"{name} is on an unsupported device {kind!r}; expected one of {allowed}"
    )


def _checked_byte_layout(
    tensor, name: str, shape: tuple[int, ...] | None, allowed: tuple[str, ...]
):
    """A contiguous ``uint8``, grad-free tensor of exactly ``shape`` on an allowed device."""
    torch = _torch()
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != torch.uint8:
        raise TypeError(
            f"{name} must be uint8 (raw bytes), got {tensor.dtype}; this format stores "
            f"bytes, so a numeric dtype would mean values were being reinterpreted"
        )
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous (strides {tuple(tensor.stride())}); the layout "
            f"formulas index raw bytes"
        )
    if tensor.requires_grad:
        # Defensive: no integer dtype can carry a grad today, so this cannot fire for the
        # accepted dtypes. It stays so the contract is stated where a wider dtype would
        # have to re-earn it.
        raise ValueError(f"{name} must not require grad; this is a byte permutation")
    _require_device(tensor, name, allowed)
    return tensor


def _checked_payload_tensor(tensor, name: str, shape: tuple[int, ...]):
    """A payload tensor: always the CPU, since the payload is the offline artifact."""
    return _checked_byte_layout(tensor, name, shape, ("cpu",))


def _checked_codes(codes):
    """``codes`` as a contiguous ``uint8`` CPU/NPU tensor, without a numeric cast.

    An ``int8`` projection is the same bytes as ``uint8``; ``view`` reinterprets the storage
    while ``.to(torch.uint8)`` would be a numeric conversion, so the view is what keeps a
    negative-spelled code byte identical on the way in and out.
    """
    torch = _torch()
    if not isinstance(codes, torch.Tensor):
        raise TypeError(f"codes must be a torch.Tensor, got {type(codes).__name__}")
    if codes.dtype not in (torch.uint8, torch.int8):
        raise TypeError(
            f"codes must be the packed uint8/int8 bytes, got {codes.dtype}; this is the "
            f"released two-codes-per-byte packing, not a per-code element dtype"
        )
    if codes.ndim != 2:
        raise ValueError(
            f"codes must be a 2-D [N, K/2] tensor, got shape {tuple(codes.shape)}"
        )
    if not codes.is_contiguous():
        raise ValueError(
            f"codes must be contiguous (strides {tuple(codes.stride())}); the layout "
            f"formulas index raw bytes"
        )
    if codes.requires_grad:
        raise ValueError("codes must not require grad; this is a byte permutation")
    _require_device(codes, "codes", ("cpu", "npu"))
    return codes.view(torch.uint8)


def _checked_scales(scales):
    """``scales`` as a contiguous ``uint8`` E8M0 byte tensor on the CPU or an NPU."""
    return _checked_byte_layout(scales, "scales", None, ("cpu", "npu"))


def _resolve_output_device(device) -> str:
    """The device the payload lands on: always the CPU, refused if asked otherwise."""
    if device is None:
        return "cpu"
    kind = _device_kind(device)
    if kind != "cpu":
        raise ValueError(
            f"the payload is always materialized on the CPU, so output device {kind!r} is "
            f"refused; an on-device copy would be an implicit transfer, which this offline "
            f"repack does not perform"
        )
    return kind


def _check_no_overlap(left, right, left_name: str, right_name: str) -> None:
    """Refuse two tensors that share storage. Different devices cannot overlap."""
    torch = _torch()
    if _device_kind(left.device) != _device_kind(right.device):
        return
    if torch._C._overlaps(left, right):
        raise ValueError(f"{left_name} and {right_name} must not share storage")


# -------------------------------------------------------------------------- conversion


def convert_fp4_cube(codes, scales, *, device=None) -> CubePackedFP4:
    """Rewrite one ``[N, K]`` packed-FP4 projection into cube order.

    ``codes`` ``[N, K/2]`` ``uint8``/``int8`` and ``scales`` ``[N, K/32]`` ``uint8`` are read
    as bytes and returned in the recorded cube order; nothing is decoded. The result has the
    same byte count as the input (see :meth:`CubeGeometry.report`), so the transform is
    reversible by construction.

    The payload is a fresh allocation: neither returned tensor shares storage with an input,
    so rewriting or freeing the source cannot disturb a payload already handed to a consumer.

    ``device`` may only be ``None`` or ``"cpu"``. Inputs are accepted on the CPU or on an NPU
    (serialized for an offline repack); a CUDA input or an explicit non-CPU output is refused
    rather than transferred.
    """
    torch = _torch()
    if not isinstance(codes, torch.Tensor):
        raise TypeError(f"codes must be a torch.Tensor, got {type(codes).__name__}")
    source_codes_dtype = "int8" if codes.dtype == torch.int8 else "uint8"
    raw_codes = _checked_codes(codes)
    raw_scales = _checked_scales(scales)

    geometry = CubeGeometry(raw_codes.shape[0], raw_codes.shape[1] * 2)
    if tuple(raw_scales.shape) != geometry.source_scales_shape:
        raise ValueError(
            f"scales must have shape {geometry.source_scales_shape} for a "
            f"[{geometry.n}, {geometry.k}] projection, got {tuple(raw_scales.shape)}"
        )
    target = _resolve_output_device(device)

    # The view only splits existing extents, so it cannot move a byte; the permute plus
    # contiguous() is the single materialization.
    codes_cube = raw_codes.view(geometry.codes_view_shape).permute(_CODE_PERMUTE).contiguous()
    scales_cube = raw_scales.view(geometry.scales_view_shape).permute(_SCALE_PERMUTE).contiguous()
    if _device_kind(raw_codes.device) != target:
        codes_cube = codes_cube.to(target)
    if _device_kind(raw_scales.device) != target:
        scales_cube = scales_cube.to(target)

    payload = CubePackedFP4(
        codes=codes_cube,
        scales=scales_cube,
        geometry=geometry,
        source_codes_dtype=source_codes_dtype,
    )
    # The payload must not alias the source: a later write to `codes` would otherwise change
    # bytes a consumer already believes are stable.
    for left, left_name in ((payload.codes, "payload codes"), (payload.scales, "payload scales")):
        for right, right_name in ((codes, "codes"), (scales, "scales")):
            if _device_kind(left.device) == _device_kind(right.device) and torch._C._overlaps(
                left, right
            ):
                raise RuntimeError(
                    f"{left_name} aliases {right_name}; the layout step must allocate"
                )  # pragma: no cover - contiguous() always allocates
    return payload


def recover_fp4_cube(payload: CubePackedFP4, *, codes_dtype=None):
    """Rewrite a cube-order payload back to the source ``[N, K/2]``/``[N, K/32]`` layout.

    Byte-exact inverse of :func:`convert_fp4_cube`: the returned bytes equal the bytes the
    payload was built from, including signed-zero and saturating codes, because no code is
    ever interpreted.

    ``codes_dtype`` selects only the spelling of the recovered code bytes -- ``None``
    restores the dtype the projection arrived with, ``"uint8"``/``"int8"`` (or the
    ``torch.dtype``) overrides it. Either way the storage is the same bytes.
    """
    torch = _torch()
    if not isinstance(payload, CubePackedFP4):
        raise TypeError(
            f"payload must be a CubePackedFP4, got {type(payload).__name__}; recovery "
            f"refuses to guess a layout from a bare tuple"
        )
    # Re-validate rather than trusting the fields: a payload can be constructed by hand (by
    # the native consumer's loader), and its declared geometry is what recovery uses.
    geometry = payload.geometry
    _checked_payload_tensor(payload.codes, "payload codes", geometry.codes_shape)
    _checked_payload_tensor(payload.scales, "payload scales", geometry.scales_shape)

    codes = payload.codes.permute(_CODE_PERMUTE).contiguous().view(geometry.source_codes_shape)
    scales = (
        payload.scales.permute(_SCALE_PERMUTE_INVERSE)
        .contiguous()
        .view(geometry.source_scales_shape)
    )

    if codes_dtype is None:
        requested = payload.source_codes_dtype
    elif isinstance(codes_dtype, str):
        requested = codes_dtype
    else:
        requested = str(codes_dtype).replace("torch.", "")
    if requested not in _CODE_DTYPES:
        raise ValueError(
            f"codes_dtype must be one of {_CODE_DTYPES}, got {requested!r}; recovery "
            f"reinterprets bytes and does not convert them numerically"
        )
    if requested == "int8":
        codes = codes.view(torch.int8)

    for left, left_name in ((codes, "recovered codes"), (scales, "recovered scales")):
        if torch._C._overlaps(left, payload.codes) or torch._C._overlaps(left, payload.scales):
            raise RuntimeError(
                f"{left_name} aliases the payload; recovery must allocate"
            )  # pragma: no cover - contiguous() always allocates
    return codes, scales