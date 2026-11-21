"""NPU Cube throughput experiment: a **changed** K reduction, opt-in only.

The user has explicitly authorized a separate native-Cube accumulation *test*. This module is that
test's runtime half, and nothing else: no default is changed, no serving capability is relaxed and no
existing provider, loader, quantizer or call site is edited. Whoever wants the experiment installs it
process-locally, for one run::

    from relicllm.models.deepseek_v4_1 import ascend_linear, cube_experiment

    ascend_linear.quantized_linear_ascend = cube_experiment.make_linear_provider(
        ascend_linear.quantized_linear_ascend
    )

`make_linear_provider` captures the **original** callable, so the old path stays reachable and the
patch can never recurse into itself.

Two experimental reduction modes exist and they are not interchangeable
----------------------------------------------------------------------
The root's Cube baseline primitive (a BF16 ``torch.matmul`` on a ``[1, 5120] x [5120, 2304]``
projection) was measured separately; that primitive is not this module's claim, and this module's
composition was **not** timed. The reductions this module implements are:

``MODE = "full-scaled-native-bf16"`` (default)
    Every operand is decoded to BF16 *with its original E8M0 scale already applied*, and the natural
    matmul reduces all ``K`` in one FP32 accumulator. This is :func:`cube_quantized_linear_ascend` and
    :func:`cube_fp4_packed_gemv`, i.e. ``CubePackedProjection(mode="decode")``.

``WEIGHT_QUANT_MODE = "weight-quant-i8x2-antiquant-bf16"`` (opt-in, ``mode="weight_quant"``)
    The frozen INT8-times-two widening of the file's packed E2M1 bytes (the original values, ``-12..
    12``) plus the **original halved** E8M0 grid are handed to the device's native weight-quantized
    matmul as a per-group antiquantization, so the weight is dequantized on chip and no BF16 weight
    temporary is ever materialized. Still not W4A8: the activation is the released FP8 quantizer's own
    bytes decoded exactly, with no INT8 activation. Its scaling placement is the same as the mode
    above (the original scale is folded into the operand before the product), which is exactly what
    makes it the like-for-like alternative to the decode mode; the mode's own accumulation still
    differs from the reference window rule.

Both are **different arithmetic** from the per-K32 partial-scale native kernel handled in relic-core
(task 57), which applies scales *after* each block's dot product and fuses the decode into the kernel.
The modes therefore produce different numbers from each other as well as from the reference, so every
report here names its mode and the results must never be quoted as one number.

What is preserved, exactly
--------------------------
* The **activation** side runs the released block-32 quantizer unchanged
  (`relic_core.kernels.block_fp8_quant.quantize_block_fp8`): the same amax floor, the same "ceiling of
  ``amax * fl32(1/448)``" E8M0 scale, the same E4M3 round-to-nearest. Those raw bytes are the operand
  -- there is no INT8 activation requantization anywhere in this module.
* The **weight** side is the loader's own BF16 parameter, which the online-unpack path produced from
  the file's E4M3 bytes and E8M0 grid, and this module *requires* that provenance: it reads the
  attached raw codes/grid (`mma.attached_codes` / `mma.attached_scale`) and refuses a weight that has
  none rather than reverse-engineering codes from a dense parameter.
* Full Engram stays on the host, computed by the original CPU code on rank 0 and broadcast byte for
  byte. That is why the experiment must **not** be installed by clearing `POCKETLLM_V41_MMA_EMULATE`
  globally: that switch also changes the CPU Engram's `wkv` reduction, which is the correctness
  control, not part of this experiment.
* The reference reduction (`mma`'s two-16-term-half window rule, `_MMA1`) is untouched and remains the
  control every measurement is compared against.

What changes, and only here
---------------------------
The K reduction. Operands are decoded exactly to BF16 -- an E4M3 code has a four-bit significand and
an E2M1 code has a three-bit one, so every finite original value times a power-of-two scale is exactly
representable in BF16 -- and the product is then handed to the device's natural matmul, which
accumulates in FP32 and rounds once to BF16. The reference rule instead splits K into 16-term halves
with exponent anchors, a truncation window and a carry. The two therefore disagree by ULPs on ordinary
inputs, and this module labels that difference everywhere it appears: `REDUCTION_NOTE` is in every
`provenance` report, and the results are **not** the reference's numbers.

The natural matmul also returns BF16 rather than an unrounded FP32 accumulator, so an FP32 caller sees
one rounding the reference path does not apply. The output *dtype* is still the caller's activation
dtype, so call sites do not change.

Cost shape, stated because it decides where this is usable
----------------------------------------------------------
Decoding an original packed projection to a BF16 temporary is the composition's own expense and the
root measured it in the ~15 ms class for a released expert projection, against a Cube primitive in the
tens of microseconds; a decode-then-matmul call therefore does not inherit the primitive's speed. That
is why the fused per-K32 native kernel is the **preferred** path for packed experts and this module's
packed operator is the explicit comparison baseline. A bounded weight cache (`CubePackedProjection`)
exists to separate decode cost from matmul cost in a controlled run, **not** as a speedup claim: the
root's routing draws each expert once per step, so the measured hit rate on the original routing was
zero and no static cache benefit is asserted.

Exactness guards
----------------
`check_bf16_representable` proves the operand contract instead of assuming it: values whose exact FP32
decoding is not exactly a BF16 value (overflow to infinity, underflow to zero, or a subnormal whose
low bits would round) are counted and refused **by name**. It is a load-time guard -- two device
synchronizations per call, counted in `GUARD_STATS` -- and the per-call path runs none, so no decode
step pays for a host read. `decode_fp4_packed_bf16` never invents a finite value for a non-finite
block: an original ``255`` E8M0 byte stays NaN, and the load-time guard is what names it.

Language: English comments and docstrings, per the repository convention.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from . import mma

__all__ = [
    "BLOCK",
    "EXPERIMENT_ID",
    "GUARD_STATS",
    "MAX_CACHE_BYTES",
    "MODE",
    "REDUCTION_NOTE",
    "WEIGHT_QUANT_MODE",
    "CubeExperimentUnavailable",
    "CubePackedProjection",
    "OriginalOperandNotBf16Representable",
    "check_bf16_representable",
    "cube_fp4_packed_gemv",
    "cube_fp4_weight_quant_gemv",
    "cube_quantized_linear_ascend",
    "decode_fp4_packed_bf16",
    "make_linear_provider",
    "verify_weight_decode_identity",
]

#: The released K block. This module is stated on it and refuses anything else.
BLOCK = 32

EXPERIMENT_ID = "ascend-cube-natural-f32-accum/v1"

#: Which arithmetic a measurement of this module describes. The core's per-K32 partial-scale kernel is
#: a different mode, deliberately not this string.
MODE = "full-scaled-native-bf16"

#: The optional native on-chip weight-dequantization mode: the frozen INT8-times-two operand and the
#: original **half** grid handed to ``torch_npu.npu_weight_quant_batchmatmul`` as a per-group
#: antiquantization, so no BF16 weight temporary is materialized. Still not W4A8 and still no
#: activation requantization -- the activation is the original FP8 grid decoded exactly to BF16.
WEIGHT_QUANT_MODE = "weight-quant-i8x2-antiquant-bf16"

#: Every provenance report carries this sentence. The experiment's numbers are not the reference's.
REDUCTION_NOTE = (
    f"mode {MODE}: operands are unchanged original bytes decoded exactly to BF16 with their original "
    "E8M0 scale applied, and K is reduced by the device's natural matmul (FP32 accumulate, one "
    "rounding to BF16) instead of relic-core's reference two-half window rule. The reduction is "
    "changed, so results are not reference-quality."
)

#: Hard ceiling on one `CubePackedProjection`'s decoded-weight cache. The packed expert stores are the
#: reason: a bounded cache is the difference between a temporary and a second copy of the weights.
MAX_CACHE_BYTES = 256 * 2**20

#: Reduction counts for the guards that read device memory back. A per-call guard would put a
#: synchronization inside a decode step, so the counters exist to make any such cost visible.
GUARD_STATS: dict[str, int] = {"checks": 0, "syncs": 0}

#: The eight E2M1 magnitudes in code order. Codes 8..15 are their negatives, and code 8 is ``-0.0``:
#: the sign of a zero is part of the encoding even though the INT8 widening cannot carry it.
_FP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class CubeExperimentUnavailable(RuntimeError):
    """The experiment cannot run here: wrong device, no fused backend, or the wrong entry point.

    Raised instead of silently substituting another backend, because a measurement that quietly ran
    the reference arithmetic would describe a path this module does not implement.
    """


class OriginalOperandNotBf16Representable(ValueError):
    """An original operand is not exactly representable in BF16, so the experiment would move it.

    This is the guard's refusal: the experiment's claim is that it only changes the *reduction*, and a
    value that BF16 cannot hold exactly would make that claim false.
    """


# -- small helpers -------------------------------------------------------------------------------


def _require_npu(tensor: torch.Tensor, what: str) -> None:
    """Refuse anything that is not on an NPU. Module-private so tests can pin the decision."""
    if tensor.device.type != "npu":
        raise CubeExperimentUnavailable(
            f"{what} is the NPU Cube experiment and runs on an NPU only, got {tensor.device}. "
            "Nothing falls back to another device; the default paths are unchanged."
        )


def _core_ops():
    """The frozen relic-core operators this module composes, imported where they are used.

    Lazy for the same reason the other Ascend bindings are: importing this module must stay possible on
    a core checkout that predates the operators. Module-private and returned as a tuple so a test can
    substitute CPU-forced wrappers and exercise the real ordering without a card.
    """
    from relic_core.kernels import block_fp8_quant, unpack

    return (
        block_fp8_quant.quantize_block_fp8,
        unpack.unpack_fp8_to_bf16,
        unpack.unpack_fp4_to_int8,
        unpack.fp4_int8_scale,
    )


def _select_decode_backend(backend: str, device: torch.device) -> str:
    """``"triton"`` on an NPU, ``"torch"`` as an explicit CPU oracle; never a silent switch."""
    if not isinstance(backend, str) or backend not in {"auto", "torch", "triton"}:
        raise ValueError("backend must be one of 'auto', 'torch' or 'triton'")
    if device.type == "npu":
        if backend == "torch":
            raise CubeExperimentUnavailable(
                "the frozen pair decoders are Triton-NPU kernels; the Torch LUT decoder exists as a "
                "CPU oracle and is not selected on a device"
            )
        return "triton"
    if backend == "triton":
        raise CubeExperimentUnavailable(
            "the fused Triton decoders need an NPU tensor; pass backend='torch' for the CPU oracle"
        )
    return "torch"


def _checked_codes(
    codes: torch.Tensor, name: str, *, encoding: str = "fp4", columns: int | None = None
) -> torch.Tensor:
    """Validate a raw 8-bit code tensor and return it as ``uint8`` without changing a byte.

    ``encoding="fp8"`` also accepts the loader's ``float8_e4m3fn`` view of the stored bytes, which is
    a reinterpretation rather than a cast; ``encoding="fp4"`` is packed nibbles and takes only 8-bit
    containers, because no float8 type describes two codes in one byte.
    """
    if encoding not in {"fp4", "fp8"}:
        raise ValueError(f"encoding must be 'fp4' or 'fp8', got {encoding!r}")
    if not isinstance(codes, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(codes).__name__}")
    if codes.layout != torch.strided:
        raise ValueError(f"{name} must have a dense strided layout")
    if codes.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got shape {tuple(codes.shape)}")
    e4m3 = getattr(torch, "float8_e4m3fn", None)
    accepted = {torch.uint8, torch.int8} if encoding == "fp4" else {torch.uint8, torch.int8, e4m3}
    if codes.dtype not in accepted:
        raise TypeError(
            f"{name} is {codes.dtype}; this experiment reads the file's 8-bit encodings in a "
            f"container ({'uint8/int8' if encoding == 'fp4' else 'uint8/int8/float8_e4m3fn view'}) "
            "and does not convert them"
        )
    if not codes.is_contiguous():
        raise ValueError(f"{name} must be contiguous; no implicit copy is made")
    if columns is not None and int(codes.shape[1]) != columns:
        raise ValueError(f"{name} must be [rows, {columns}], got {tuple(codes.shape)}")
    return codes if codes.dtype == torch.uint8 else codes.view(torch.uint8)


def _fp4_values(device: torch.device) -> torch.Tensor:
    """The sixteen E2M1 values as FP32, indexed by code, with ``code 8 == -0.0``."""
    return torch.tensor(
        [*_FP4_MAGNITUDES, *(-magnitude for magnitude in _FP4_MAGNITUDES)],
        dtype=torch.float32,
        device=device,
    )


def _e8m0_byte_values(scale_bytes: torch.Tensor, block: int) -> torch.Tensor:
    """The original (un-halved) ``2**(byte - 127)`` grid as FP32, expanded along K.

    Built from bit patterns rather than a float power: byte ``0`` is ``2**-127``, which is an FP32
    *subnormal* and therefore not an exponent field, and byte ``255`` is NaN in the frozen decoders.
    Indexing a 256-entry table keeps both ends exact and needs no libm power.
    """
    table = (torch.arange(256, dtype=torch.int64, device=scale_bytes.device) << 23).to(torch.int32).clone()
    table[0] = 0x00400000  # 2**-127, the subnormal encoding of the smallest original scale
    table[255] = 0x7FC00000  # NaN, the frozen operators' reading of byte 255
    values = table.view(torch.float32)[scale_bytes.reshape(-1).to(torch.int64)]
    return values.reshape(scale_bytes.shape).repeat_interleave(block, dim=-1)


def _grid_values(
    scale: torch.Tensor, block: int, *, verify_power_of_two: bool = False
) -> tuple[torch.Tensor, int]:
    """An E8M0 grid as FP32 values expanded along K, in either of the two forms the tree carries.

    The checkpoint stores **bytes** (``uint8``/``int8`` containers, or the ``float8_e8m0fnu`` view of
    them), while `mma.attach_scale` hangs the **host-decoded powers of two** on a weight -- both are
    the same exponents and both appear in this module's operands, so one helper reads both rather than
    making each call site know which it was handed.

    ``verify_power_of_two`` checks that a floating grid really is a grid of powers of two, which is the
    precondition for "exactly representable in BF16". That check is a host read (one synchronization),
    so it is counted and it is only ever requested from a load-time guard. A byte container is not
    checked because an E8M0 byte *is* an exponent.

    Returns ``(values, syncs)``.
    """
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"the E8M0 grid must be a torch.Tensor, got {type(scale).__name__}")
    if scale.layout != torch.strided or not scale.is_contiguous():
        raise ValueError("the E8M0 grid must be contiguous")
    e8m0 = getattr(torch, "float8_e8m0fnu", None)
    if scale.dtype in {torch.uint8, torch.int8} or (e8m0 is not None and scale.dtype == e8m0):
        raw = scale if scale.dtype == torch.uint8 else scale.view(torch.uint8)
        return _e8m0_byte_values(raw, block), 0
    if not scale.is_floating_point():
        raise TypeError(
            f"the E8M0 grid is {scale.dtype}; expected raw bytes or the decoded FP32 powers of two"
        )
    syncs = 0
    if verify_power_of_two:
        mantissa, _ = torch.frexp(scale.to(torch.float64))
        syncs += 1
        GUARD_STATS["syncs"] += 1
        if not bool((mantissa.abs() == 0.5).all()):
            raise ValueError(
                "the attached grid is not a grid of powers of two, so it is not an E8M0 scale and "
                "the exactness this experiment claims does not hold"
            )
    return scale.to(torch.float32).repeat_interleave(block, dim=-1), syncs


def _fp4_packed_exact(codes: torch.Tensor, scales: torch.Tensor, block: int) -> torch.Tensor:
    """Exact FP32 values of original packed E2M1 codes under the original grid, ``[N, K]``."""
    rows, packed_columns = int(codes.shape[0]), int(codes.shape[1])
    low = (codes & 0x0F).to(torch.int64)
    high = (codes >> 4).to(torch.int64)
    nibbles = torch.stack((low, high), dim=-1).reshape(rows, packed_columns * 2)
    values = _fp4_values(codes.device)[nibbles]
    # `verify_power_of_two` is a no-op for a raw byte grid and a real check for the decoded FP32 grid
    # a weight carries; the FP4 path needs it as much as the FP8 path, since its values are exact only
    # if the grid really is a grid of powers of two.
    grid, _ = _grid_values(scales, block, verify_power_of_two=True)
    return values * grid


def _fp8_exact(codes: torch.Tensor, scales: torch.Tensor, block: int) -> torch.Tensor:
    """Exact FP32 values of raw E4M3 codes under an E8M0 grid, ``[N, K]``."""
    raw = codes.to(torch.int64)
    sign = (raw >> 7) & 1
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    # E4M3 significands are the eight values ``(8 + mantissa) * 2**-3`` for a normal number and
    # ``mantissa * 2**-9`` for a subnormal one. Folding the ``2**-3`` into the exponent is what makes
    # code ``0x38`` decode to ``1.0`` and not to ``8.0``.
    significand = torch.where(exponent == 0, mantissa, 8 + mantissa)
    power = torch.where(exponent == 0, -9, exponent - 10)
    grid, _ = _grid_values(scales, block, verify_power_of_two=True)
    magnitude = torch.ldexp(significand.to(torch.float32), power) * grid
    # E4M3's two NaN encodings (exponent 15 with significand 15) are NaN here too, and so is the
    # scale's own NaN byte: the same reading the frozen decoders take.
    nan = (raw & 127) == 127
    values = torch.where(sign != 0, -magnitude, magnitude)
    return torch.where(nan, torch.full_like(values, float("nan")), values)


def _natural_bf16_matmul(activation: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """``[M, K] @ [N, K].T`` on the device's natural matmul, BF16 in and BF16 out.

    Module-private and device-agnostic so the *arithmetic* can be checked on a CPU oracle: the
    contract is exactly "the operands the caller built, one native matmul", nothing reinterpreted.
    """
    if activation.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16:
        raise TypeError(
            "the Cube experiment multiplies BF16 operands only; got "
            f"{activation.dtype} and {weight.dtype}"
        )
    if activation.ndim != 2 or weight.ndim != 2:
        raise ValueError("the natural matmul takes two rank-2 operands")
    if int(activation.shape[1]) != int(weight.shape[1]):
        raise ValueError(
            f"K mismatch: activation {tuple(activation.shape)} against weight {tuple(weight.shape)}"
        )
    return torch.matmul(activation, weight.transpose(0, 1))


# -- exact decoding ------------------------------------------------------------------------------


def decode_fp4_packed_bf16(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    block: int = BLOCK,
    backend: str = "auto",
) -> torch.Tensor:
    """Decode one projection of original packed FP4 bytes to BF16 ``[N, K]``.

    ``codes`` is ``[N, K // 2]`` of the checkpoint's own layout -- two E2M1 codes per byte, **low
    nibble first** along K -- and ``scales`` is the original ``[N, K // block]`` raw E8M0 grid,
    **un-halved**. Nothing is requantized and no activation bytes are involved.

    Every finite original value is exactly representable in BF16, so the result is an exact decode
    followed by one narrowing that cannot round: ``E2M1`` codes are magnitudes ``{0, 0.5, 1, 1.5, 2,
    3, 4, 6}`` (a three-bit significand) and the scale is a power of two. Two ends are still handled
    carefully rather than assumed away:

    * an E2M1 ``-0`` code keeps BF16's sign bit on the device path as well (the frozen INT8 widening
      cannot carry it, so ``+0`` and ``-0`` are the one place the two decoders differ -- both are
      numerically zero and cannot move a dot product, which the tests assert);
    * the original ``255`` E8M0 byte is NaN in the frozen decoders, and this decoder keeps it NaN
      rather than inventing a finite value. Callers that require finite weights verify once at load
      (`check_bf16_representable`), not once per call.

    ``backend="auto"`` decodes with the frozen pair operators on an NPU and with the exact bit-pattern
    decoder elsewhere; ``"torch"`` is the explicit CPU oracle and is refused on an NPU, while
    ``"triton"`` needs one.
    """
    if isinstance(block, bool) or not isinstance(block, int) or block != BLOCK:
        raise ValueError(f"the original FP4 grid is {BLOCK}-wide, got block={block!r}")
    raw = _checked_codes(codes, "packed FP4 codes", encoding="fp4")
    packed_columns = int(raw.shape[1])
    if packed_columns <= 0 or packed_columns % (BLOCK // 2):
        raise ValueError(
            f"packed FP4 codes must be [N, K//2] with K a multiple of {BLOCK}, got {packed_columns} "
            "code columns"
        )
    rows, width = int(raw.shape[0]), packed_columns * 2
    raw_scales = _checked_scales(scales, "FP4 E8M0 grid", width // BLOCK)
    if int(raw_scales.shape[0]) != rows:
        raise ValueError(
            f"the FP4 grid has {raw_scales.shape[0]} rows but there are {rows} code rows; the "
            "original grid keeps one scale row per output row and must be the matched pair"
        )
    if raw.device != raw_scales.device:
        raise ValueError("packed FP4 codes and their E8M0 grid must be on one device")
    implementation = _select_decode_backend(backend, raw.device)
    if implementation == "torch":
        return _fp4_packed_exact(raw, raw_scales, BLOCK).to(torch.bfloat16)
    _, _, unpack_fp4_to_int8, fp4_int8_scale = _core_ops()
    # The frozen pair: INT8 holds exactly twice each E2M1 value and the decoded grid is the original
    # scale halved, so their product is the original value. Matched pair and no other arithmetic.
    values = unpack_fp4_to_int8(raw, backend=implementation).to(torch.float32)
    half_scale = fp4_int8_scale(raw_scales, backend=implementation)
    return (values * half_scale.repeat_interleave(BLOCK, dim=-1)).to(torch.bfloat16)


def _checked_scales(scales: torch.Tensor, name: str, blocks: int) -> torch.Tensor:
    """Validate a raw E8M0 grid of ``[rows, blocks]`` bytes and return its ``uint8`` container."""
    raw = _checked_codes(scales, name, encoding="fp4")
    if int(raw.shape[1]) != blocks:
        raise ValueError(f"{name} must be [rows, {blocks}], got {tuple(raw.shape)}")
    return raw


def _blocked_grid_for_rows(
    scale: torch.Tensor, rows: int, columns: int, *, name: str = "grid"
) -> torch.Tensor:
    """Normalise an attached grid to the ``[rows, columns // 32]`` form the decoders read.

    Two layouts appear in the tree and neither may be guessed: the file's dense projections store a
    **square** 32x32 grid (``ceil(rows / 32)`` scale rows), while the row-scaled tables -- and the
    activation quantizer -- store **one** scale row per output row. This expands a square grid along
    rows so every output row carries its block's scale, and refuses a shape that is neither, which is
    what stops a rowwise table's grid from being read as a square one.
    """
    if not isinstance(scale, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(scale).__name__}")
    if scale.ndim != 2 or int(scale.shape[1]) != columns // BLOCK:
        raise ValueError(
            f"{name} must be [rows, {columns // BLOCK}], got {tuple(scale.shape)}"
        )
    grid_rows = int(scale.shape[0])
    if grid_rows == rows:
        return scale
    if grid_rows == -(-rows // BLOCK):
        return scale.repeat_interleave(BLOCK, dim=0)[:rows]
    raise ValueError(
        f"{name} is {tuple(scale.shape)}; neither the row-scaled ({rows}, {columns // BLOCK}) nor the "
        f"square ({-(-rows // BLOCK)}, {columns // BLOCK}) layout"
    )


def verify_weight_decode_identity(weight: torch.Tensor) -> dict:
    """Prove a BF16 weight **is** the decode of the original codes/grid attached to it.

    One comparison and one synchronization, counted in ``GUARD_STATS``. This is a load-time check -- a
    per-call version would put a host read inside a decode step -- and it is the check that licenses
    using the loader's parameter as the original weight.

    ``block_shape`` is derived, not guessed: the file's dense projections store a square 32x32 grid and
    the row-scaled tables store one row of scales per output row, and a grid of neither shape is
    refused rather than assumed to be one of them. Both grid forms the loader can attach are accepted
    (`_grid_values`).
    """
    codes = mma.attached_codes(weight)
    scale = mma.attached_scale(weight)
    if codes is None or scale is None:
        raise CubeExperimentUnavailable(
            "this weight carries no original E4M3 codes / E8M0 grid, so it did not come from the "
            "online-unpack loader and this experiment cannot claim its operands are the file's"
        )
    rows, columns = int(weight.shape[0]), int(weight.shape[1])
    if columns % BLOCK:
        raise ValueError(f"K {columns} is not a whole number of {BLOCK}-wide blocks")
    raw = _checked_codes(codes, "attached E4M3 codes", encoding="fp8")
    if tuple(raw.shape) != (rows, columns):
        raise ValueError(
            f"the attached codes are {tuple(raw.shape)} and the weight is {(rows, columns)}"
        )
    grid_rows = int(scale.shape[0]) if scale.ndim == 2 else -1
    if grid_rows == rows:
        block_rows = 1
    elif grid_rows == -(-rows // BLOCK):
        block_rows = BLOCK
    else:
        raise ValueError(
            f"the attached grid is {tuple(scale.shape)}; neither the row-scaled ({rows}, "
            f"{columns // BLOCK}) nor the square ({-(-rows // BLOCK)}, {columns // BLOCK}) layout"
        )
    # `_fp8_exact` expands the K axis itself, so this hands it the blocked form with the rows already
    # expanded for a square grid -- not a grid that has been expanded twice.
    blocked = _blocked_grid_for_rows(scale, rows, columns, name="attached grid")
    decoded = _fp8_exact(raw, blocked, BLOCK).to(torch.bfloat16)
    GUARD_STATS["checks"] += 1
    GUARD_STATS["syncs"] += 1
    mismatches = int((decoded.view(torch.int16) != weight.view(torch.int16)).sum().item())
    if mismatches:
        raise OriginalOperandNotBf16Representable(
            f"{mismatches} of {weight.numel()} weight values are not the decode of the codes and grid "
            "attached to them; the experiment's operands would not be the file's own"
        )
    return {"elements": int(weight.numel()), "mismatches": 0, "block_rows": block_rows, "syncs": 1}


def check_bf16_representable(
    codes: torch.Tensor,
    scales: torch.Tensor,
    *,
    kind: str = "fp4",
    block: int = BLOCK,
    name: str = "operand",
    allow_subnormal: bool = False,
) -> dict:
    """Refuse an original operand whose exact value BF16 cannot hold, and report what was counted.

    ``kind="fp4"`` reads packed E2M1 bytes with their original grid; ``kind="fp8"`` reads raw E4M3
    codes (the loader's ``float8_e4m3fn`` view is accepted too) with an E8M0 grid in either form. The
    exact FP32 value is constructed from the bytes -- no device float power, no cast -- rounded to
    BF16 and compared **bit for bit**; the guard therefore catches overflow to infinity, underflow to
    zero and a subnormal whose low bits would round, all three of which are cases where the experiment
    would have moved a value rather than only the reduction.

    A value that is *exactly* representable only as a BF16 **subnormal** is refused by default even
    though the encoding is exact: accelerators flush subnormals to zero under the usual FTZ modes, so a
    subnormal operand is one the device may silently zero. ``allow_subnormal=True`` opts into treating
    encoding-exactness as sufficient; the stricter default is the one that can be checked against a
    cross-walk with the relic-core certification helper.

    Cost: the counts are read back, so three synchronizations (plus one if the grid arrived as floating
    powers and needed its power-of-two check), counted in ``GUARD_STATS``. This is a load-time guard;
    the per-call paths in this module run none.

    Returns ``{"kind", "elements", "not_bf16_representable", "non_finite", "subnormal", "syncs"}``.
    """
    if kind not in {"fp4", "fp8"}:
        raise ValueError(f"kind must be 'fp4' or 'fp8', got {kind!r}")
    if isinstance(block, bool) or not isinstance(block, int) or block != BLOCK:
        raise ValueError(f"the original grid is {BLOCK}-wide, got block={block!r}")
    if kind == "fp4":
        raw = _checked_codes(codes, f"{name} packed FP4 codes", encoding="fp4")
        width = int(raw.shape[1]) * 2
    else:
        raw = _checked_codes(codes, f"{name} E4M3 codes", encoding="fp8")
        width = int(raw.shape[1])
    if width % BLOCK:
        raise ValueError(f"{name} width {width} is not a whole number of {BLOCK}-wide blocks")
    if not isinstance(scales, torch.Tensor):
        raise TypeError(f"{name} grid must be a torch.Tensor, got {type(scales).__name__}")
    if scales.ndim != 2 or int(scales.shape[1]) != width // BLOCK:
        raise ValueError(
            f"{name} grid must be [rows, {width // BLOCK}], got {tuple(scales.shape)}"
        )
    rows = int(raw.shape[0])
    if int(scales.shape[0]) not in {rows, -(-rows // BLOCK)}:
        raise ValueError(
            f"{name}: {rows} code rows against a grid of {tuple(scales.shape)}; the original grid is "
            "either one scale row per output row or one per 32-row block"
        )
    # A square grid is expanded along rows here; a row-scaled one is already the matched pair.
    blocked = _blocked_grid_for_rows(scales, rows, width, name=f"{name} grid")
    syncs_before = GUARD_STATS["syncs"]
    exact = (
        _fp4_packed_exact(raw, blocked, BLOCK)
        if kind == "fp4"
        else _fp8_exact(raw, blocked, BLOCK)
    )
    rounded = exact.to(torch.bfloat16).to(torch.float32)
    not_representable = (rounded.view(torch.int32) != exact.view(torch.int32)).sum()
    non_finite = (~torch.isfinite(exact)).sum()
    # Exactness is not enough at the bottom of the range: a BF16 subnormal is flushed to zero by the
    # accelerator's usual FTZ behaviour, which would move the operand rather than only the reduction.
    normal_floor = torch.finfo(torch.bfloat16).tiny
    subnormal = (~non_finite) & (exact != 0) & (exact.abs() < normal_floor)
    GUARD_STATS["checks"] += 1
    GUARD_STATS["syncs"] += 3
    subnormal_count = int(subnormal.sum().item())
    report = {
        "kind": kind,
        "elements": int(exact.numel()),
        "not_bf16_representable": int(not_representable.item()),
        "non_finite": int(non_finite.item()),
        "subnormal": subnormal_count,
        # Counted, not asserted: a floating grid also pays the power-of-two check inside `_grid_values`,
        # so the honest number is whatever the counters moved by rather than a constant.
        "syncs": GUARD_STATS["syncs"] - syncs_before,
    }
    if report["not_bf16_representable"] or report["non_finite"]:
        raise OriginalOperandNotBf16Representable(
            f"{name}: {report['not_bf16_representable']} of {report['elements']} values are not "
            f"exactly representable in BF16 and {report['non_finite']} are non-finite, so this "
            "operand would be moved rather than only the reduction. The experiment refuses it; the "
            "reference path is unaffected and remains the control"
        )
    if subnormal_count and not allow_subnormal:
        raise OriginalOperandNotBf16Representable(
            f"{name}: {subnormal_count} of {report['elements']} values are exactly representable only "
            "as BF16 subnormals, which the accelerator may flush to zero. Pass allow_subnormal=True "
            "to accept encoding-exactness alone"
        )
    return report


# -- the two experimental entry points -----------------------------------------------------------


def cube_quantized_linear_ascend(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    block_size: int = BLOCK,
    emulate_reduction: bool = True,
    verify_operands: bool = False,
) -> torch.Tensor:
    """Dense NPU linear with the reference reduction **replaced** by a natural Cube matmul.

    Drop-in shape and dtype for
    :func:`relicllm.models.deepseek_v4_1.ascend_linear.quantized_linear_ascend`:
    ``[..., K] @ [N, K].T -> [..., N]`` at ``x.dtype``. What it does:

    1. Runs the released block-32 activation quantizer on ``x`` -- unchanged, the same bytes the
       reference path quantizes.
    2. Decodes those bytes to BF16 with the frozen FP8 decoder. No INT8 activation requantization is
       applied, so the activation operand is the original encoding decoded exactly.
    3. Requires the weight's original provenance: the loader's BF16 parameter with the file's own E4M3
       codes and E8M0 grid attached. A weight without them is refused rather than reverse engineered.
    4. Multiplies the two BF16 operands with the device's natural matmul and returns ``x.dtype``.

    This is ``MODE = "full-scaled-native-bf16"``: the original scale is applied to each operand before
    the dot product. The relic-core per-K32 partial-scale kernel is a *different* mode and the two must
    not be quoted as one number.

    ``emulate_reduction=True`` is the mode this entry point implements: it is the flag the model passes
    when it wants the reference emulation, and this module is what replaces that emulation under an
    explicit, process-local installation. ``False`` is the *existing* throughput escape hatch and is
    **not** re-implemented here; pass it to the original function through :func:`make_linear_provider`
    instead of letting two definitions drift.

    ``verify_operands`` runs :func:`check_bf16_representable` on the activation bytes (two
    synchronizations) and is meant for a one-shot acceptance run, not a decode step. The weight's own
    guards -- :func:`verify_weight_decode_identity` and a ``kind="fp8"`` representability check -- are
    load-time calls with the same reasoning, and the root checks them once from CPU metadata.
    """
    if isinstance(emulate_reduction, bool) and not emulate_reduction:
        raise CubeExperimentUnavailable(
            "emulate_reduction=False is the pre-existing natural escape hatch inside "
            "ascend_linear.quantized_linear_ascend and is not re-implemented here; install this module "
            "with make_linear_provider(original) so that branch keeps running the original function"
        )
    _require_npu(x, "the Cube dense linear")
    if block_size != BLOCK:
        raise ValueError(f"original V4.1 linear operands use block_size={BLOCK}, got {block_size}")
    if x.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError(f"the activation must be BF16 or FP32, got {x.dtype}")
    if weight.dtype != torch.bfloat16:
        raise TypeError(
            f"the Cube experiment multiplies a BF16 weight; got {weight.dtype}. The loader's original "
            "online-unpack parameter is BF16, and an FP32 weight would need a different operand claim"
        )
    if weight.ndim != 2 or int(x.shape[-1]) != int(weight.shape[1]):
        raise ValueError(f"the weight must be [N, {x.shape[-1]}], got {tuple(weight.shape)}")
    if weight.device != x.device:
        raise ValueError("activation and weight must be on one NPU")
    if not weight.is_contiguous():
        raise ValueError("the weight must be contiguous; no implicit copy is made")
    codes = mma.attached_codes(weight)
    scale = mma.attached_scale(weight)
    if codes is None or scale is None:
        raise CubeExperimentUnavailable(
            "the Cube dense linear needs the original E4M3 codes and E8M0 grid attached to the actual "
            "weight; recovering them from the dense parameter would be a different operand"
        )

    quantize, unpack_fp8_to_bf16, _, _ = _core_ops()
    flat = x.reshape(-1, x.shape[-1]).contiguous()
    activation_codes, activation_scale = quantize(flat, backend="triton")
    if verify_operands:
        check_bf16_representable(activation_codes, activation_scale, kind="fp8", name="activation")
    activation = unpack_fp8_to_bf16(
        activation_codes, activation_scale, block_shape=(1, BLOCK), backend="triton"
    )
    product = _natural_bf16_matmul(activation, weight)
    return product.to(x.dtype).reshape(*x.shape[:-1], int(weight.shape[0]))


def cube_fp4_packed_gemv(
    a_codes: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_e8m0: torch.Tensor,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    """One-projection M=1 GEMV from original packed expert bytes, on the natural matmul.

    The four positional operands are exactly the contract
    ``ascend_packed_experts.AscendPackedRoutedExperts(packed_gemv=...)`` calls:
    ``(a_codes [1, K], a_scale [1, K//32], b_packed [N, K//2], b_e8m0 [N, K//32])``, all raw bytes,
    the weight grid **un-halved**. Output is FP32 ``[1, N]`` -- the shape and dtype the replaced
    operator returns -- so the surrounding expert arithmetic keeps its own ordering.

    Only this **one projection** is ever widened: the BF16 weight is a bounded temporary dropped when
    the call returns. Nothing here makes a full BF16 expert resident, which would be 67.5 MiB an expert
    against the packed store's 17.93 MiB.

    The activation is the released quantizer's own E4M3/E8M0 bytes, decoded exactly; there is no INT8
    activation requantization. ``backend="torch"`` is the explicit CPU oracle and runs the whole
    composition on the host; on an NPU the operand decoders are the frozen Triton kernels.

    This composition decodes the projection on every call, so it is the **comparison baseline** and not
    the fast path: relic-core's fused per-K32 native kernel avoids the BF16 temporary entirely.
    """
    if not (isinstance(backend, str) and backend in {"auto", "torch", "triton"}):
        raise ValueError("backend must be one of 'auto', 'torch' or 'triton'")
    if backend != "torch":
        _require_npu(a_codes, "the Cube packed-expert GEMV")
    raw_a = _checked_codes(a_codes, "activation E4M3 codes", encoding="fp8")
    if int(raw_a.shape[0]) != 1:
        raise ValueError(
            f"the packed-expert GEMV is defined for one activation row [1, K], got {tuple(raw_a.shape)}"
        )
    width = int(raw_a.shape[1])
    if width <= 0 or width % BLOCK:
        raise ValueError(f"the GEMV needs a positive K divisible by {BLOCK}, got {width}")
    raw_sa = _checked_scales(a_scale, "activation E8M0 grid", width // BLOCK)
    if int(raw_sa.shape[0]) != 1:
        raise ValueError(f"activation scales must be [1, {width // BLOCK}]")
    raw_b = _checked_codes(b_packed, "packed FP4 weight bytes", encoding="fp4")
    columns = int(raw_b.shape[0])
    if columns <= 0 or int(raw_b.shape[1]) != width // 2:
        raise ValueError(f"packed FP4 weight bytes must be [N, {width // 2}], got {tuple(raw_b.shape)}")
    raw_sb = _checked_scales(b_e8m0, "FP4 E8M0 grid", width // BLOCK)
    if int(raw_sb.shape[0]) != columns:
        raise ValueError(
            f"the FP4 grid has {raw_sb.shape[0]} rows against {columns} weight rows; the original grid "
            "keeps one scale row per output row"
        )
    devices = {raw_a.device, raw_sa.device, raw_b.device, raw_sb.device}
    if len(devices) != 1:
        raise ValueError("all four GEMV operands must be on one device")

    _, unpack_fp8_to_bf16, _, _ = _core_ops()
    activation = unpack_fp8_to_bf16(raw_a, raw_sa, block_shape=(1, BLOCK), backend=backend)
    weight = decode_fp4_packed_bf16(raw_b, raw_sb, backend=backend)
    product = _natural_bf16_matmul(activation, weight)
    del weight  # the one-projection temporary ends with the call
    return product.to(torch.float32)


class CubePackedProjection:
    """A callable M=1 packed-expert operator with an **optional, bounded, per-object** weight cache.

    Instances are drop-in for the ``packed_gemv=`` injection point, so an experiment can compare a
    cached and an uncached run of the same arithmetic. The cache is decoded **BF16 weights only** -- the
    packed bytes stay the store's, and nothing here becomes a second resident copy of an expert.

    ``cache_budget_bytes`` defaults to ``0``: every call decodes the projection again, which is the
    arithmetic the uncached run measures. An explicit budget retains recently used projections up to
    that many bytes, evicting least-recently-used first; it is capped at :data:`MAX_CACHE_BYTES`
    (256 MiB, about five experts' worth of three projections at the released geometry) and the cap is a
    new `ValueError`, so a configuration cannot quietly become a weight cache.

    ``mode="weight_quant"`` selects :func:`cube_fp4_weight_quant_gemv` instead: the frozen INT8-times-two
    operand and the original halved grid go to the device's native weight-quantized matmul, so no BF16
    weight exists to cache and a non-zero budget is refused rather than ignored.

    **No cache benefit is claimed.** The root's real routing draws each expert once per step, so the
    measured hit rate was zero; the cache exists to separate decode cost from matmul cost in a
    controlled run, and a timing claim has to come from an actual device measurement.

    There is deliberately **no process-global cache**: two instances share nothing, the accounting is
    per object, and :meth:`clear` releases this object's entries. An entry holds a reference to the
    packed bytes it decoded, so a recycled allocator address cannot be answered with another
    projection's decode; the other side of that is that entries must be cleared when the owning store's
    arenas are released -- the packed store documents its arenas as immutable for their lifetime.
    """

    def __init__(
        self,
        *,
        block: int = BLOCK,
        cache_budget_bytes: int = 0,
        backend: str = "auto",
        max_cache_bytes: int = MAX_CACHE_BYTES,
        mode: str = "decode",
    ) -> None:
        if isinstance(block, bool) or not isinstance(block, int) or block != BLOCK:
            raise ValueError(f"the original FP4 grid is {BLOCK}-wide, got block={block!r}")
        for value, label in (
            (cache_budget_bytes, "cache_budget_bytes"),
            (max_cache_bytes, "max_cache_bytes"),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an int, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"{label} cannot be negative, got {value}")
        if cache_budget_bytes > max_cache_bytes:
            raise ValueError(
                f"cache_budget_bytes {cache_budget_bytes} exceeds the {max_cache_bytes}-byte ceiling; "
                "this is a bounded comparison cache, not a resident weight store"
            )
        if not (isinstance(backend, str) and backend in {"auto", "torch", "triton"}):
            raise ValueError("backend must be one of 'auto', 'torch' or 'triton'")
        if mode not in {"decode", "weight_quant"}:
            raise ValueError(
                f"mode must be 'decode' or 'weight_quant', got {mode!r}; the core's per-K32 native "
                "kernel is a separate operator and not a mode of this object"
            )
        if mode == "weight_quant" and cache_budget_bytes:
            # The on-chip path never builds a BF16 weight, so there is nothing for this cache to hold;
            # accepting a budget would describe a saving this mode cannot make.
            raise ValueError(
                "mode='weight_quant' has no BF16 weight temporary to cache; use the default decode "
                "mode for a bounded weight cache, or drop cache_budget_bytes"
            )
        self.block = block
        self.backend = backend
        self.mode = mode
        self.cache_budget_bytes = cache_budget_bytes
        self.max_cache_bytes = max_cache_bytes
        self._entries: "OrderedDict[tuple, tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]" = (
            OrderedDict()
        )
        self._retained_bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._decodes = 0

    # -- reporting ------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"CubePackedProjection(backend={self.backend!r}, block={self.block}, "
            f"cache_budget_bytes={self.cache_budget_bytes}, entries={len(self._entries)}, "
            f"retained_bytes={self._retained_bytes})"
        )

    @property
    def cache_stats(self) -> dict:
        """This object's own counters. No counter here is shared with another instance."""
        return {
            "name": "cube-packed-projection",
            "cache_budget_bytes": self.cache_budget_bytes,
            "retained_bytes": self._retained_bytes,
            "entries": len(self._entries),
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "decodes": self._decodes,
            "max_cache_bytes": self.max_cache_bytes,
        }

    @property
    def provenance(self) -> dict:
        """What this object's numbers are, stated where a report can quote it."""
        native = self.mode == "weight_quant"
        return {
            "experiment": EXPERIMENT_ID,
            "mode": WEIGHT_QUANT_MODE if native else MODE,
            "projection_mode": self.mode,
            "reduction": REDUCTION_NOTE,
            "weight_source": "original packed E2M1 bytes + raw un-halved E8M0 grid",
            "activation_source": "released block-32 E4M3/E8M0 quantizer bytes, decoded exactly",
            "activation_requantization": "none (no INT8 activation operand)",
            "widening": (
                "on-chip antiquantization from the frozen INT8-times-two operand and the original "
                "halved grid; no BF16 weight temporary"
                if native
                else "one projection per call, BF16, dropped when the call returns"
            ),
            "preferred_fast_path": "relic-core fused per-K32 partial-scale native kernel "
            "(a different mode; not this object's arithmetic)",
            "cache": {
                "scope": "per object, bounded, decoded BF16 weights only",
                "budget_bytes": self.cache_budget_bytes,
                "max_cache_bytes": self.max_cache_bytes,
                "key": "packed/grid identity (pointer, shape, device) with a retained reference",
                "invalidation": "clear() when the owning store's arenas are released or refilled",
                "timing_claim": "none; the measured hit rate on the original routing was zero",
                "applicable": not native,
            },
            "reference_reduction_installed": False,
        }

    # -- use -------------------------------------------------------------------

    def clear(self) -> int:
        """Drop this object's cached decodes and return how many entries went."""
        dropped = len(self._entries)
        self._entries.clear()
        self._retained_bytes = 0
        return dropped

    def __call__(
        self,
        a_codes: torch.Tensor,
        a_scale: torch.Tensor,
        b_packed: torch.Tensor,
        b_e8m0: torch.Tensor,
        *,
        backend: str | None = None,
    ) -> torch.Tensor:
        """The four-operand provider seam, with its explicit backend keyword."""
        if backend not in {None, "auto"}:
            _select_decode_backend(backend, a_codes.device)
            if self.backend not in {"auto", backend}:
                raise ValueError("the provider backend keyword conflicts with its configured backend")
        if self.mode == "weight_quant":
            return cube_fp4_weight_quant_gemv(a_codes, a_scale, b_packed, b_e8m0)
        if self.backend != "torch":
            _require_npu(a_codes, "the Cube packed-expert GEMV")
        raw_b = _checked_codes(b_packed, "packed FP4 weight bytes", encoding="fp4")
        weight = self._weight(raw_b, b_e8m0)
        return self._product(a_codes, a_scale, raw_b, b_e8m0, weight)

    def _product(
        self,
        a_codes: torch.Tensor,
        a_scale: torch.Tensor,
        raw_b: torch.Tensor,
        b_e8m0: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        """Decode the single activation row, then one natural matmul against the cached weight."""
        if int(raw_b.shape[0]) != int(weight.shape[0]):
            raise ValueError("the decoded weight lost its projection's rows")
        _, unpack_fp8_to_bf16, _, _ = _core_ops()
        raw_a = _checked_codes(a_codes, "activation E4M3 codes", encoding="fp8")
        width = int(weight.shape[1])
        if int(raw_a.shape[0]) != 1 or int(raw_a.shape[1]) != width:
            raise ValueError(
                f"the activation must be [1, {width}], got {tuple(raw_a.shape)}; the packed-expert "
                "GEMV is defined for exactly one row"
            )
        raw_sa = _checked_scales(a_scale, "activation E8M0 grid", width // BLOCK)
        if int(raw_sa.shape[0]) != 1:
            raise ValueError(f"activation scales must be [1, {width // BLOCK}]")
        if len({raw_a.device, raw_sa.device, weight.device}) != 1:
            raise ValueError("all four GEMV operands must be on one device")
        activation = unpack_fp8_to_bf16(raw_a, raw_sa, block_shape=(1, BLOCK), backend=self.backend)
        return _natural_bf16_matmul(activation, weight).to(torch.float32)

    def _key(self, raw_b: torch.Tensor, b_e8m0: torch.Tensor) -> tuple:
        return (
            int(raw_b.data_ptr()),
            int(b_e8m0.data_ptr()),
            tuple(raw_b.shape),
            tuple(b_e8m0.shape),
            str(raw_b.device),
        )

    def _weight(self, raw_b: torch.Tensor, b_e8m0: torch.Tensor) -> torch.Tensor:
        """The projection's decoded BF16 weight: from this object's cache, or decoded and cached."""
        key = self._key(raw_b, b_e8m0)
        entry = self._entries.get(key)
        if entry is not None:
            # The retained reference is the identity check: a freed and reallocated address cannot be
            # answered with another projection's decode, because the entry holds the pair alive.
            if entry[0] is raw_b and entry[1] is b_e8m0:
                self._entries.move_to_end(key)
                self._hits += 1
                return entry[2]
            del self._entries[key]
            self._retained_bytes -= entry[3]
        self._misses += 1
        raw_scales = _checked_scales(b_e8m0, "FP4 E8M0 grid", int(raw_b.shape[1]) * 2 // BLOCK)
        if int(raw_scales.shape[0]) != int(raw_b.shape[0]):
            raise ValueError(
                f"the FP4 grid has {raw_scales.shape[0]} rows against {raw_b.shape[0]} weight rows; "
                "the original grid keeps one scale row per output row"
            )
        decoded = decode_fp4_packed_bf16(raw_b, raw_scales, block=self.block, backend=self.backend)
        self._decodes += 1
        if self.cache_budget_bytes == 0:
            return decoded
        nbytes = int(decoded.numel()) * int(decoded.element_size())
        if nbytes > self.cache_budget_bytes:
            return decoded  # a projection larger than the whole budget is not a cache candidate
        while self._entries and self._retained_bytes + nbytes > self.cache_budget_bytes:
            _, (_, _, _, dropped_bytes) = self._entries.popitem(last=False)
            self._retained_bytes -= dropped_bytes
            self._evictions += 1
        self._entries[key] = (raw_b, b_e8m0, decoded, nbytes)
        self._retained_bytes += nbytes
        return decoded


def _fused_backend(tensor: torch.Tensor) -> str:
    """The frozen decoders' backend for this device: Triton on an NPU, the CPU reference elsewhere.

    This is not a fallback. Every entry point that reaches it is NPU-only through :func:`_require_npu`,
    so the CPU answer exists for the seam-patched tests and for the exactness oracles, and a device run
    always asks for the fused kernels.
    """
    return "triton" if tensor.device.type == "npu" else "torch"


def _npu_weight_quant_op():
    """The device's native weight-quantized matmul, or a named refusal.

    ``torch_npu.npu_weight_quant_batchmatmul(x, weight, antiquant_scale, antiquant_offset=None,
    quant_scale=None, quant_offset=None, bias=None, antiquant_group_size=0, inner_precise=0)``:
    ``x`` is ``[M, K]`` FP16/BF16, ``weight`` is ``[K, N]`` int8, and a per-group
    ``antiquant_scale`` is ``[ceil(K, group), N]`` in ``x``'s dtype. Without a ``quant_scale`` the
    output has ``x``'s dtype.

    Resolved by name at the call: this build may not ship the operator, and a module that silently
    replaced a missing on-chip dequantization with the decode path would describe arithmetic it did
    not run. Module-private so a test can substitute a recording fake and pin the operand layout.
    """
    try:
        import torch_npu  # noqa: F401 - the operator lives on the torch_npu module
    except ImportError as error:  # pragma: no cover - depends on the install
        raise CubeExperimentUnavailable(
            "the weight-quant mode needs torch_npu's native weight-quantized matmul, which is not "
            "importable here; the decode mode is the portable path"
        ) from error
    op = getattr(torch_npu, "npu_weight_quant_batchmatmul", None)
    if op is None:  # pragma: no cover - depends on the build
        raise CubeExperimentUnavailable(
            "this torch_npu build exposes no npu_weight_quant_batchmatmul, so the on-chip weight "
            "dequantization cannot run; no other operator is substituted"
        )
    return op


def _half_scale_bf16(half_scale: torch.Tensor, *, name: str, verify: bool) -> torch.Tensor:
    """The original half grid cast to BF16, exactly, with the two failure modes refused by name.

    Every finite E8M0 half-scale is a power of two in ``[2**-128, 2**126]``, and BF16 holds all of
    them exactly -- including the subnormal end, since BF16's min subnormal is ``2**-133``. So the
    cast cannot round, and the two ways it could still be wrong are refused: a non-finite grid (the
    original ``255`` byte, which the frozen decoder reads as NaN) and a value that is not a power of
    two. ``verify`` is the load-time gate -- the checks read the grid back, so they are off the
    per-call path and counted in ``GUARD_STATS`` when asked for.
    """
    cast = half_scale.to(torch.bfloat16)
    if not verify:
        return cast
    GUARD_STATS["checks"] += 1
    GUARD_STATS["syncs"] += 2
    if not bool(torch.isfinite(half_scale).all()):
        raise OriginalOperandNotBf16Representable(
            f"{name}: the original grid contains a non-finite scale (the frozen decoder reads the "
            "255 byte as NaN), so the on-chip antiquantization would multiply by NaN"
        )
    if not torch.equal(cast.float().view(torch.int32), half_scale.view(torch.int32)):
        raise OriginalOperandNotBf16Representable(
            f"{name}: a scale is not exactly representable in BF16, so the antiquantization would "
            "round a scale rather than only change the reduction"
        )
    return cast


def cube_fp4_weight_quant_gemv(
    a_codes: torch.Tensor,
    a_scale: torch.Tensor,
    b_packed: torch.Tensor,
    b_e8m0: torch.Tensor,
    *,
    antiquant_group_size: int = BLOCK,
    inner_precise: int = 0,
    verify_operands: bool = False,
) -> torch.Tensor:
    """One-projection M=1 GEMV through the device's **native on-chip weight dequantization**.

    ``MODE = WEIGHT_QUANT_MODE``. The four positional operands are the same contract
    ``ascend_packed_experts.AscendPackedRoutedExperts(packed_gemv=...)`` calls, and the result is FP32
    ``[1, N]`` like the replaced operator, so this is a drop-in alternative to
    :func:`cube_fp4_packed_gemv` that never materializes a BF16 weight temporary:

    * the weight operand is the frozen INT8-times-two widening of the file's packed E2M1 bytes
      (``-12..12``; the original encoded values, not a new quantization) transposed to ``[K, N]``;
    * the antiquantization scale is the **original half grid** (``fp4_int8_scale``: the E8M0 exponent
      halved, matching the doubled values) transposed to ``[ceil(K / group), N]`` and cast to BF16,
      which is exact for a finite grid (`_half_scale_bf16`);
    * the activation is the released block-32 E4M3/E8M0 quantizer's own bytes decoded exactly to
      BF16. There is no INT8 activation and no new activation quantization, so this is **not** W4A8;
    * the device dequantizes the weight on chip and accumulates naturally, so the K reduction differs
      from the reference rule exactly as the decode mode's does.

    Only one projection's INT8 codes plus their grid exist at a time (11.8 MiB of codes for ``w1`` at
    the released geometry), which is the point of the mode: a resident INT8 bank would be a second
    copy of the weights and is deliberately not created.

    ``verify_operands`` runs the scale gate (finite, exactly BF16) and the packed-operand
    representability guard at load time; leave it off a decode step, because both read device memory
    back. Absent operator support is a named refusal, never a silent switch of arithmetic.
    """
    if isinstance(antiquant_group_size, bool) or not isinstance(antiquant_group_size, int):
        raise TypeError("antiquant_group_size must be an int")
    if antiquant_group_size != BLOCK:
        raise ValueError(
            f"the original FP4 grid is {BLOCK}-wide, so the antiquantization group is {BLOCK}; got "
            f"{antiquant_group_size}"
        )
    if inner_precise not in {0, 1}:
        raise ValueError(f"inner_precise must be 0 (high precision) or 1, got {inner_precise!r}")
    _require_npu(a_codes, "the weight-quant GEMV")
    raw_a = _checked_codes(a_codes, "activation E4M3 codes", encoding="fp8")
    if int(raw_a.shape[0]) != 1:
        raise ValueError(
            f"the weight-quant GEMV is defined for one activation row [1, K], got {tuple(raw_a.shape)}"
        )
    width = int(raw_a.shape[1])
    if width <= 0 or width % BLOCK:
        raise ValueError(f"the GEMV needs a positive K divisible by {BLOCK}, got {width}")
    raw_sa = _checked_scales(a_scale, "activation E8M0 grid", width // BLOCK)
    if int(raw_sa.shape[0]) != 1:
        raise ValueError(f"activation scales must be [1, {width // BLOCK}]")
    raw_b = _checked_codes(b_packed, "packed FP4 weight bytes", encoding="fp4")
    columns = int(raw_b.shape[0])
    if columns <= 0 or int(raw_b.shape[1]) != width // 2:
        raise ValueError(f"packed FP4 weight bytes must be [N, {width // 2}], got {tuple(raw_b.shape)}")
    raw_sb = _checked_scales(b_e8m0, "FP4 E8M0 grid", width // BLOCK)
    if int(raw_sb.shape[0]) != columns:
        raise ValueError(
            f"the FP4 grid has {raw_sb.shape[0]} rows against {columns} weight rows; the original grid "
            "keeps one scale row per output row"
        )
    if len({raw_a.device, raw_sa.device, raw_b.device, raw_sb.device}) != 1:
        raise ValueError("all four GEMV operands must be on one device")
    if verify_operands:
        check_bf16_representable(raw_b, raw_sb, kind="fp4", name="weight")

    op = _npu_weight_quant_op()
    fused = _fused_backend(raw_a)
    _, unpack_fp8_to_bf16, unpack_fp4_to_int8, fp4_int8_scale = _core_ops()
    activation = unpack_fp8_to_bf16(raw_a, raw_sa, block_shape=(1, BLOCK), backend=fused)
    # One projection's INT8-times-two operand and the matching original half grid, transposed to the
    # [K, N] / [ceil(K / group), N] layout the operator documents, then dropped when the call returns.
    weight = unpack_fp4_to_int8(raw_b, backend=fused).transpose(0, 1).contiguous()
    half_scale = _half_scale_bf16(
        fp4_int8_scale(raw_sb, backend=fused).transpose(0, 1).contiguous(),
        name="weight half-scale",
        verify=verify_operands,
    )
    product = op(
        activation,
        weight,
        half_scale,
        antiquant_offset=None,
        quant_scale=None,
        quant_offset=None,
        bias=None,
        antiquant_group_size=antiquant_group_size,
        inner_precise=inner_precise,
    )
    del weight
    return product.to(torch.float32)


def make_linear_provider(original):
    """Wrap the **original** dense provider so the experiment is explicit and cannot recurse.

    ``emulate_reduction=True`` -- the flag the model passes when it wants the reference emulation -- is
    routed to this module; ``False`` -- the pre-existing natural escape hatch -- keeps running
    ``original``. Capturing the callable is what makes an in-process monkeypatch safe::

        from relicllm.models.deepseek_v4_1 import ascend_linear, cube_experiment
        ascend_linear.quantized_linear_ascend = cube_experiment.make_linear_provider(
            ascend_linear.quantized_linear_ascend
        )

    Nothing is installed by importing this module, and nothing outside the patch is touched.
    """
    if not callable(original):
        raise TypeError(f"original must be callable, got {type(original).__name__}")

    def provider(x, weight, *, block_size: int = BLOCK, emulate_reduction: bool = True):
        if not emulate_reduction:
            return original(x, weight, block_size=block_size, emulate_reduction=False)
        return cube_quantized_linear_ascend(x, weight, block_size=block_size)

    provider.__name__ = "cube_experiment_linear_provider"
    provider.__doc__ = (
        "Dense linear provider: reference emulation replaced by the authorized Cube experiment, the "
        "pre-existing natural escape hatch left on the original function."
    )
    return provider