"""Online decoding of the released checkpoint's original floating-format weights.

The released V4.1 weights are floating formats in byte containers: a dense projection is one
``float8_e4m3fn`` code per byte beside an E8M0 grid, and a routed expert is two E2M1 nibbles per
byte beside an E8M0 grid with one scale per output row and 32-wide K block. A loader that fills an
ordinary bf16 tree has to turn those back into numbers, and there are two ways to do it:

* **expand on the host and copy the result** -- what the incumbent path does: it expands each weight
  on the host (an FP32 intermediate, four bytes per element of host work), narrows it to the
  parameter's BF16, and copies that two-byte-per-element tensor to the card where the checkpoint
  stored one byte of codes;
* **stage the original bytes and decode on the device** -- what this module is for. The weight
  crosses the link as the bytes the file holds, and the decode runs where the parameter lives.

The second one is not an optimization detail on Ascend: the incumbent path ends in
``mma.codes_from_scale``, whose last step casts a float32 quotient to ``float8_e4m3fn``. torch_npu
2.10 answers that cast with ``aclnnInplaceCopy`` error **561103** on 910B3, so the load itself fails
before a single token is generated. The checkpoint already holds the codes that cast was trying to
recover, so there is nothing to recover: stage them and decode.

What is preserved, and what is not:

* Original E4M3 codes, original E8M0 scales and their original block layout are read as bytes and
  never rewritten, requantized, or replaced by a row-wise integer scheme. The decode is E4M3 value x
  E8M0 power-of-two in FP32, narrowed once to BF16 in the order ``modules.dequantize_rows`` uses.
  The scale the *operator* reads is the raw stored byte container; the scale the *metadata* carries
  is that same grid decoded to exact FP32 on the host, because the device must not be asked to cast
  a float8 scale container -- that cast is the failure this path exists to avoid, and a uint8 cast
  would read 0..255 numerically instead of the exponent it encodes.
* The packed-FP4 adaptor widens each nibble to **twice** its value in INT8 and keeps the original
  grid divided by two, so ``int8 * (scale / 2)`` reconstructs the original floating weight in FP32:
  exact arithmetic, with the caveat ``Fp4ExpertInt8.to_float32`` records -- the product of a small
  integer and a power of two needs at most four significand bits, but a device that flushes FP32
  subnormals would zero the ``2**-128`` edge. That is a lossless widening of values the file already
  contains, not new quantization: there is no amax, no zero-point and no replacement scale. Integer
  zero cannot carry E2M1's negative-zero sign, so that one bit does not survive into INT8 -- the
  packed bytes themselves are untouched.
* Neither operation is native FP4/FP8 hardware arithmetic. ``fp4_expert_int8`` still hands back a
  representation rather than executing an expert; the optional ``ascend_experts`` provider consumes
  it through relic-core's separately validated original-format block GEMM.
* The unpacked values are wider execution of the original encodings. A BF16 result that looks
  sensible is not token parity, and nothing here claims it.

Only the model-level contract lives here -- which layout a key has, whether a parameter may take the
BF16 decode, and what a staged weight hands back. The byte-level operators are in ``relic-core``
(``relic_core.kernels.unpack``), imported lazily: the runtime has to import on hosts whose installed
relic-core predates them, where only the legacy load path is reachable anyway.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

__all__ = [
    "BLOCK",
    "Fp4ExpertInt8",
    "StagedFp8Weight",
    "decode_dense_fp8_weight",
    "fp4_expert_int8",
    "infer_fp8_block_shape",
    "is_online_unpack_parameter",
    "stage_dense_fp8_weight",
]

#: The reference's block width along K: `kernel.py`'s `block_K = group_size = fp8_block_size = 32`.
#: Every quantized tensor in this checkpoint uses it -- a dense grid is square 32x32 and an expert or
#: Engram grid is 32 wide along K -- so a key whose derived width disagrees is refused rather than
#: decoded against a grid it was not written on.
BLOCK = 32

_E4M3 = torch.float8_e4m3fn
_E8M0 = getattr(torch, "float8_e8m0fnu", None)


def _core():
    """The byte-level operators, imported where they are used rather than at module import.

    ``loader`` imports this module unconditionally, and the runtime has to stay importable on a host
    whose installed relic-core predates ``relic_core.kernels.unpack``; only the legacy path is
    reachable there, and it must not become an ImportError at import time.
    """
    from relic_core.kernels import unpack as core_unpack

    return core_unpack


def _payload_bytes(tensor: torch.Tensor, name: str) -> torch.Tensor:
    """`tensor`'s original 8-bit codes as a `uint8` view, without changing a byte of them.

    A view and not a cast: the checkpoint stores these as floating dtypes whose *encoding* is the
    payload, so ``.to(torch.uint8)`` would convert values and destroy exactly what is being decoded.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must have a dense strided layout")
    if tensor.element_size() != 1:
        raise TypeError(f"{name} is {tensor.dtype}, not an 8-bit original-code container")
    if tensor.dtype == torch.uint8:
        return tensor
    return tensor.view(torch.uint8)


def _e8m0_dtype() -> torch.dtype:
    if _E8M0 is None:
        raise RuntimeError(
            "this torch has no E8M0 dtype, so the checkpoint's original scales cannot be read; "
            "the online unpack path needs torch>=2.8"
        )
    return _E8M0


def infer_fp8_block_shape(
    weight_shape: tuple[int, ...], scale_shape: tuple[int, ...]
) -> tuple[int, int]:
    """Which ``(block_rows, block_cols)`` grid the file wrote for this weight and scale pair.

    The checkpoint uses two layouts and they are not interchangeable -- a decoder that read either
    as the other would produce plausible wrong numbers rather than fail:

    * **square**, for every dense projection: ``scale[out // 32, in // 32]``, one exponent per 32x32
      tile, which is ``block_shape=(32, 32)``;
    * **row per row**, for Engram tables and packed-fp4 experts: ``scale[rows, in // 32]``, one
      exponent per output row and 32-wide K block, which is ``block_shape=(1, 32)``.

    The grid decides, not the name: ``block_cols`` is derived from the two widths and a leading
    (batch) axis has to agree exactly. The input axis must divide exactly -- a ragged K is refused,
    because the width is what the block grid is counted along -- while the output axis may end in a
    short block, since the operator's contract is `ceil(rows / block_cols)`.

    **What the runtime actually stages is the released checkpoint's own layout**, which is fully
    tiled: every dense grid is ``[out // 32, in // 32]``, so both axes are multiples of the block
    width, and the experts and Engram rows keep one scale per row and 32-wide K block. The ragged
    row form is the operator's generality, not a shape this checkpoint ships, and it is not claimed
    here as one this loader exercises.

    Empty axes are refused, so this is also not a statement that a degenerate grid is accepted:
    ``rows``, ``columns``, ``grid_rows`` and ``grid_columns`` all have to be positive.
    """
    weight_shape = tuple(int(size) for size in weight_shape)
    scale_shape = tuple(int(size) for size in scale_shape)
    if len(weight_shape) < 2:
        raise ValueError(f"a scaled fp8 weight must have rank >= 2, got {weight_shape}")
    if len(scale_shape) != len(weight_shape):
        raise ValueError(
            f"weight {weight_shape} and scale {scale_shape} must have the same rank"
        )
    if weight_shape[:-2] != scale_shape[:-2]:
        raise ValueError(
            f"weight {weight_shape} and scale {scale_shape} disagree on their leading axes"
        )
    rows, columns = weight_shape[-2:]
    grid_rows, grid_columns = scale_shape[-2:]
    if min(rows, columns, grid_rows, grid_columns) <= 0:
        raise ValueError(f"weight {weight_shape} / scale {scale_shape} has an empty axis")
    if columns % grid_columns:
        raise ValueError(
            f"scale {scale_shape} does not divide the {columns}-wide input axis of {weight_shape}"
        )
    block_columns = columns // grid_columns
    # The square grid is asked for first, which canonicalises the one shape the two layouts cannot
    # tell apart: a single output row. With M = 1 both layouts have one grid row and index scale row
    # 0 for the only element, so `(32, 32)` is returned rather than refusing a grid that is merely
    # ambiguous. More rows are decided by the grid itself -- ceil(M / 32) rows is the square layout,
    # M rows is the row-per-row one -- and neither reading is optional for a real table.
    if grid_rows == -(-rows // block_columns):
        return block_columns, block_columns
    if grid_rows == rows:
        return 1, block_columns
    raise ValueError(
        f"scale {scale_shape} is neither a row-per-row grid nor a square "
        f"{block_columns}x{block_columns} grid for a {weight_shape} weight: {rows} output rows "
        f"want {rows} scale rows or {-(-rows // block_columns)} of them"
    )


def is_online_unpack_parameter(parameter: torch.Tensor) -> bool:
    """Whether an automatic load should decode this parameter from the original bytes.

    Both halves are requirements rather than preferences. The device is the one with no float8 cast
    (see the module docstring); the dtype is what the decode produces, and staging a BF16 result
    into an fp32 parameter would round a contract the incumbent path keeps exactly -- an fp32
    dequantized weight holds the exact product of an E4M3 code and a power of two, where a BF16
    intermediate holds 8 bits of it. So an fp32 parameter never takes this path automatically, and
    asking for it explicitly is refused instead of silently satisfied.
    """
    return (
        isinstance(parameter, torch.Tensor)
        and parameter.device.type == "npu"
        and parameter.dtype == torch.bfloat16
    )


@dataclass(frozen=True)
class StagedFp8Weight:
    """A dense fp8 weight decoded on its own device, with the identities the metadata needs.

    ``decoded`` is what the parameter takes. ``codes`` are the original weight bytes viewed as
    ``float8_e4m3fn`` on the same device and already cut to this rank. ``scale`` is the original E8M0
    grid decoded to exact **FP32** powers of two, also on that device: that is the form
    ``mma.attach_scale`` is handed, because a float8 cast of the stored scale container on the device
    is the operation this path avoids, and an integer cast of it would read the exponent field as the
    numeric byte 0..255. ``mma.attach_source_codes`` hangs both on the parameter so the reduction
    rule reads the file's own octave instead of recovering one.
    """

    decoded: torch.Tensor
    codes: torch.Tensor
    scale: torch.Tensor
    block_shape: tuple[int, int]


def _e8m0_grid_fp32(scale_bytes: torch.Tensor) -> torch.Tensor:
    """The stored E8M0 grid as exact FP32 powers of two, decoded on the host.

    ``mma.e8m0_exponent`` accepts either the byte container or FP32 powers of two, and for metadata
    the FP32 form is the safe one: casting a float8 container on an NPU is precisely the operation
    that raises ``aclnnInplaceCopy`` 561103, and casting it as plain bytes would give the numeric byte
    rather than the exponent. The grid is 1/1024 of the weight it scales -- 4x2 for the 128x64
    fixture in the tests, 160 per 5120-wide row in the release -- so the host loop that keeps it exact is
    negligible, and ``math.ldexp`` makes every value exact in FP32 (including 2**-127, a subnormal
    there but still exact).
    """
    flat = [int(byte) for byte in scale_bytes.reshape(-1).tolist()]
    if any(byte == 255 for byte in flat):
        raise ValueError(
            "an E8M0 scale byte of 255 is NaN, so this grid cannot describe a weight's octave"
        )
    values = torch.tensor([math.ldexp(1.0, byte - 127) for byte in flat], dtype=torch.float64)
    return values.reshape(tuple(scale_bytes.shape)).to(torch.float32)


def decode_dense_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    block_shape: tuple[int, int] | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Decode original E4M3 codes and their E8M0 grid to BF16, on the weight's own device.

    ``block_shape`` defaults to the layout the two shapes imply (see :func:`infer_fp8_block_shape`);
    pass it only to pin a layout a shape cannot distinguish. The payload is never copied to the CPU:
    the operators run where the arguments are, and an unscaled weight is refused -- a dense fp8
    weight without its grid has no exact value.
    """
    if scale is None:
        raise ValueError("a dense fp8 weight needs its original E8M0 grid; there is no exact decode without it")
    if block_shape is None:
        block_shape = infer_fp8_block_shape(tuple(weight.shape), tuple(scale.shape))
    return _core().unpack_fp8_to_bf16(weight, scale, block_shape=tuple(block_shape), backend=backend)


def stage_dense_fp8_weight(
    checkpoint,
    key: str,
    *,
    plan=None,
    device: torch.device | str | None = None,
    backend: str = "auto",
) -> StagedFp8Weight:
    """Stage one dense fp8 weight's **original bytes** to ``device`` and decode them there.

    The order matters and is the point of the function: fetch the stored bytes, cut them for this
    rank, move the bytes, decode on the destination. Nothing expands the weight on the host, so the
    link carries one byte per element of stored codes rather than the two-byte BF16 (or the four-byte
    FP32 host intermediate) the incumbent path produces before that copy.

    The cut is ``ShardPlan``'s, applied to raw bytes. It lands on whole tiles because a rank owns
    whole 32-wide K blocks (``tp.py`` refuses a cut that would split one), so cutting before decoding
    is the same partition as cutting the dense weight, and a tensor the plan does not shard passes
    through untouched. ``block_size`` is the width the checkpoint's own shapes derive; the inferred
    grid has to agree with it, which is what stops a rowwise table's grid from being read as a
    square one here.

    Two representations of the grid leave the function, deliberately: the operator decodes against
    the stored byte container (that is what its signature takes), and the returned metadata grid is
    the same exponents as FP32, decoded on the host. See :func:`_e8m0_grid_fp32` for why the card is
    never asked to cast the container.

    ``checkpoint`` is a :class:`relicllm.models.deepseek_v4_1.loader.V41Checkpoint`; it is typed
    loosely to keep this module importable without the loader (the loader imports this one).
    """
    if not checkpoint.is_quantized(key):
        raise ValueError(f"{key} has no scale stored beside it, so it has no original grid to decode")
    if checkpoint.is_packed_fp4(key):
        raise ValueError(
            f"{key} is packed fp4; a routed expert or Engram row is a store rather than a filled "
            "parameter, and its representation comes from fp4_expert_int8"
        )
    scale = checkpoint.tile_scale(key)
    if scale is None:
        raise ValueError(f"{key} has no original E8M0 grid; there is no exact decode without it")
    block = checkpoint.block_size(key)
    raw_bytes = _payload_bytes(checkpoint.packed(key), f"{key} weight codes")
    scale_bytes = _payload_bytes(scale, f"{key} scale")
    if plan is not None:
        raw_bytes = plan.local_value(key, raw_bytes)
        scale_bytes = plan.local_scale(key, scale_bytes, block)
    # The mapping's views are contiguous already, and a plan's cut returns a contiguous copy; this is
    # a no-op in both cases and exists so a caller's strided tensor is made dense once, here, rather
    # than copied again inside the operator.
    raw_bytes = raw_bytes.contiguous()
    scale_bytes = scale_bytes.contiguous()
    block_shape = infer_fp8_block_shape(tuple(raw_bytes.shape), tuple(scale_bytes.shape))
    if block_shape[1] != block:
        raise ValueError(
            f"{key} derives a {block}-wide K block from its stored shapes, but its scale grid "
            f"implies {block_shape[1]}; refusing to decode against a grid the file does not use"
        )
    target = torch.device(device) if device is not None else torch.device("cpu")
    # A dtype view on the host, then one typed transfer of the original bytes: the floating dtypes
    # are how the operator is told what the bytes mean, and the transfer keeps the dtype rather than
    # converting the container. The operator's scale argument stays the raw stored byte container.
    codes = raw_bytes.view(_E4M3).to(target)
    operator_scale = scale_bytes.view(_e8m0_dtype()).to(target)
    decoded = _core().unpack_fp8_to_bf16(
        codes, operator_scale, block_shape=block_shape, backend=backend
    )
    # The metadata grid is FP32 and decoded here, on the host, so the card never casts the container.
    metadata_scale = _e8m0_grid_fp32(scale_bytes).to(target)
    return StagedFp8Weight(
        decoded=decoded, codes=codes, scale=metadata_scale, block_shape=block_shape
    )


@dataclass(frozen=True)
class Fp4ExpertInt8:
    """Packed E2M1 experts widened exactly to INT8, with the original grid halved to match.

    Reconstruct the original floating weight in FP32 with :meth:`to_float32`: each INT8 entry is an
    E2M1 value times two, and each scale is the file's E8M0 exponent times ``2**-1``, so the product
    is the checkpoint's own value and the compensation is applied exactly once. There is no
    requantization to undo: the codes are the file's codes, widened.

    The two things this does **not** do: it does not preserve the sign of an E2M1 negative zero
    (integer zero has one encoding), and it does not itself run a routed-expert GEMM. The optional
    ``ascend_experts`` provider supplies that execution path; the representation's value/scale
    contract remains independent of the executor.
    """

    values: torch.Tensor
    half_scale: torch.Tensor
    block: int

    def to_float32(self) -> torch.Tensor:
        """The original floating weight this representation encodes, in FP32.

        The arithmetic is exact in FP32 for every finite entry the release can produce: a small
        integer times a power of two needs at most four significand bits. The one condition that is
        not arithmetic is the executing device's FP32 mode. ``half_scale`` byte 0 is ``2**-128``,
        which is itself subnormal in FP32, so a product like ``1 * 2**-128`` is subnormal too and a
        device that runs FP32 with flush-to-zero would return 0 where the CPU returns the subnormal.
        That is a property of the device rather than of this representation, so it is **reported**
        by a test instead of being papered over, and no bit-parity claim is made for the flushed
        edge. The scale bits themselves survive: the operator decodes them through a host table, so
        the stored byte values are preserved on the way to the device.
        """
        scaled = self.half_scale.to(torch.float32).repeat_interleave(self.block, dim=-1)
        return self.values.to(torch.float32) * scaled


def fp4_expert_int8(
    checkpoint,
    key: str,
    scale_name: str,
    *,
    device: torch.device | str | None = None,
    backend: str = "auto",
) -> Fp4ExpertInt8:
    """Widen one packed-fp4 expert matrix to INT8 x2 and decode its original E8M0 grid.

    ``key`` is the expert's matrix (``[out, in // 2]`` of ``I8``, low nibble first) and
    ``scale_name`` the tensor stored beside it (``[out, in // 32]`` of ``E8M0``). The row axis is the
    output axis, so the grid keeps one exponent per output row -- the fp4 layout, and the reason this
    is not the dense square grid. The block width is derived from those two shapes and must be the
    reference's 32: a different width would apply each scale to the wrong number of values.

    Nothing is copied through the CPU when ``device`` is given; the bytes are staged and the widening
    runs there.
    """
    if not checkpoint.is_quantized(key):
        raise ValueError(f"{key} has no scale stored beside it, so it is not packed fp4 with a grid")
    if not checkpoint.is_packed_fp4(key):
        raise ValueError(f"{key} is not stored as packed fp4")
    block = checkpoint.block_size(key)
    if block != BLOCK:
        raise ValueError(
            f"{key} derives a {block}-wide K block; the released routed experts use {BLOCK}, and a "
            "different width would apply each scale to the wrong number of values"
        )
    packed = checkpoint.packed(key)
    scale = checkpoint.packed(scale_name)
    if packed.ndim != 2 or scale.ndim != 2:
        raise ValueError(
            f"{key} is {tuple(packed.shape)} with a {scale_name} of {tuple(scale.shape)}; a packed "
            "expert matrix and its row grid are both two-dimensional"
        )
    rows, packed_columns = tuple(int(size) for size in packed.shape)
    total_columns = packed_columns * 2
    if total_columns % block:
        raise ValueError(
            f"{key} is {packed_columns} packed bytes wide, i.e. {total_columns} values, which is not "
            f"a whole number of {block}-wide K blocks"
        )
    expected = (rows, total_columns // block)
    if tuple(scale.shape) != expected:
        raise ValueError(
            f"{key} keeps one original scale per output row and {block}-wide K block, so "
            f"{scale_name} must be {expected}; it is {tuple(scale.shape)}"
        )
    target = torch.device(device) if device is not None else torch.device("cpu")
    packed_bytes = _payload_bytes(packed, f"{key} packed codes").contiguous().to(target)
    scale_bytes = _payload_bytes(scale, f"{scale_name}").contiguous().to(target)
    core = _core()
    return Fp4ExpertInt8(
        values=core.unpack_fp4_to_int8(packed_bytes, backend=backend),
        half_scale=core.fp4_int8_scale(scale_bytes, backend=backend),
        block=block,
    )