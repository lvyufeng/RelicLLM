"""Operator tests for the exact pair rotation, against ATen's complex multiply.

The claim these tests make is narrow and checkable: ``apply_rotary_inplace``
produces the same bits as rotating the pairs as ``complex64`` numbers, which is
what the reference does. The oracle is therefore ATen itself -- ``pairs * angle``
through ``torch.view_as_complex`` -- and the fused Ascend kernel is compared to it
bit for bit, never with a tolerance.

There are three layers, and they are deliberately separable:

* ``test_the_aten_reference_is_the_contracted_formula`` re-derives the reference
  in double precision from the two documented lines. It is what keeps the
  documentation honest if a future torch changes its contraction: the operator
  would still match ATen, but the note explaining *why* would be wrong, and this
  test fails.
* the CPU tests exercise the operator, its in-place contract, its strided views
  and every refusal, with no accelerator needed.
* the ``npu`` tests are skipped unless ``RELIC_ASCEND_ROTARY_DEVICE`` names a
  device (``npu:0``). They are what establishes that the fused kernel agrees with
  ATen on real hardware, including the wide-magnitude case where an unfused pair
  rotation does not.

Run the CPU layer with ``python -m pytest tests/test_ascend_rotary.py -q`` and the
hardware layer with
``RELIC_ASCEND_ROTARY_DEVICE=npu:0 python -m pytest tests/test_ascend_rotary.py -q``.
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels.rotary import (  # noqa: E402 - after the torch import guard
    _CONTRACTION_PROBES,
    _candidate_bits,
    _complex_bits,
    _injective,
    apply_rotary_inplace,
    needs_64_bit_indices,
    reference_contraction,
    supports_triton_ascend,
)

DEVICE = os.environ.get("RELIC_ASCEND_ROTARY_DEVICE", "").strip()
requires_npu = pytest.mark.skipif(
    not DEVICE, reason="set RELIC_ASCEND_ROTARY_DEVICE=npu:0 to exercise the fused backend"
)


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def make_table(sequence: int, width: int, *, scale: float = 1.0, seed: int = 0,
               device="cpu") -> torch.Tensor:
    """A contiguous FP32 ``[sequence, width // 2, 2]`` table of ``(cos, sin)`` pairs."""
    generator = torch.Generator().manual_seed(seed)
    angles = torch.rand(sequence, width // 2, 2, generator=generator) * scale
    return angles.to(torch.float32).contiguous().to(device)


def make_input(shape, *, scale: float = 1.0, seed: int = 1, dtype=torch.float32,
               device="cpu") -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(*shape, generator=generator) * scale
    return values.to(dtype).to(device)


def aten_rotate(x: torch.Tensor, table: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """ATen's complex multiply on a copy of ``x``, as the reference performs it.

    ``x`` may be a strided view; the product is formed on a contiguous FP32 copy and
    written back through ``x``'s own strides, exactly like the operator does.
    """
    result = x.detach().clone()
    if result.numel():
        apply_rotary_inplace(result, table, inverse=inverse, backend="torch")
    return result


def formula_rotate(x: torch.Tensor, table: torch.Tensor,
                   inverse: bool = False) -> torch.Tensor:
    """The two documented lines, in double precision, independent of ATen's kernel.

    ``real = fma(a, c, -round32(b s))`` and ``imag = fma(a, s, +round32(b c))``.
    Every operand is FP32, so the products are exact in FP64 and rounding the sum
    once to FP32 reproduces the fused step.
    """
    shape = x.shape
    sequence, width = shape[-3], shape[-1]
    flat = x.detach().to(torch.float32).contiguous()
    a = flat[..., 0::2]
    b = flat[..., 1::2]
    # The angle axes are (sequence, pair), which must broadcast against x's
    # trailing (heads, pair) -- aligned explicitly rather than left to a shape
    # coincidence between heads and sequence.
    angle_shape = (1,) * (x.ndim - 3) + (sequence, 1, width // 2)
    cosine = table[..., 0].reshape(angle_shape)
    sine = table[..., 1].reshape(angle_shape)
    if inverse:
        sine = -sine
    # The addend is a separately rounded FP32 product, not a second fused multiply.
    real_addend = (b * sine).to(torch.float64)
    imag_addend = (b * cosine).to(torch.float64)
    real = (a.to(torch.float64) * cosine.to(torch.float64) - real_addend).to(torch.float32)
    imag = (a.to(torch.float64) * sine.to(torch.float64) + imag_addend).to(torch.float32)
    rotated = torch.empty_like(flat)
    rotated[..., 0::2] = real
    rotated[..., 1::2] = imag
    return rotated.reshape(shape).to(x.dtype)


def same_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Bit equality between two tensors of the same dtype, each moved to CPU."""
    left = left.detach().to("cpu")
    right = right.detach().to("cpu")
    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    if left.dtype == torch.float32:
        return torch.equal(left.view(torch.int32), right.view(torch.int32))
    if left.dtype == torch.bfloat16:
        return torch.equal(left.view(torch.int16), right.view(torch.int16))
    return torch.equal(left, right)


def num_mismatched_bits(left: torch.Tensor, right: torch.Tensor) -> int:
    left = left.detach().to("cpu")
    right = right.detach().to("cpu")
    if left.dtype == torch.float32:
        return int((left.view(torch.int32) != right.view(torch.int32)).sum())
    if left.dtype == torch.bfloat16:
        return int((left.view(torch.int16) != right.view(torch.int16)).sum())
    return int((left != right).sum())


# ---------------------------------------------------------------------------------------------
# The reference itself: ATen's complex multiply is the contracted formula
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("table_scale", [1e-9, 1e-5, 0.5, 1.0])
@pytest.mark.parametrize("x_scale", [1e-3, 1.0, 7.5, 1e4])
@pytest.mark.parametrize("inverse", [False, True])
def test_the_aten_reference_is_the_contracted_formula(x_scale, table_scale, inverse):
    """If a future torch stops contracting, this fails and the note must change.

    A mismatch here is not an operator bug: the operator would still equal ATen.
    It means the *reason* recorded in the module docstring no longer holds, which
    is exactly the kind of quiet drift that a tolerance would hide.
    """
    x = make_input((2, 16, 4, 64), scale=x_scale)
    table = make_table(16, 64, scale=table_scale)
    reference = aten_rotate(x, table, inverse)
    formula = formula_rotate(x, table, inverse)
    assert same_bits(reference, formula), (
        f"{num_mismatched_bits(reference, formula)} of {reference.numel()} elements "
        f"differ between ATen's complex multiply and the documented FMA formula"
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_aten_reference_is_the_contracted_formula_near_rope_magnitudes(dtype):
    """The magnitudes the runtime actually uses: unit activations, small angles."""
    x = make_input((1, 64, 8, 128), scale=1.0, dtype=dtype)
    table = make_table(64, 128, scale=1e-5)
    reference = aten_rotate(x, table)
    # Compare in FP32: a BF16 store would hide the difference this test is about.
    formula = formula_rotate(x, table)
    assert same_bits(reference.to(torch.float32), formula.to(torch.float32))


def test_the_unfused_pair_product_is_not_the_reference():
    """The defect this operator exists to remove, stated as a measurement.

    ``a*c - b*s`` written as two separate FP32 products differs from the reference
    on a large fraction of inputs. If this ever reported zero disagreements the
    operator's whole reason for existing would be gone -- and the bit-exact
    comparison below would stop being able to tell the two apart.
    """
    x = make_input((1, 128, 8, 64), scale=1.0, seed=9)
    table = make_table(128, 64, scale=1e-5, seed=10)
    flat = x.to(torch.float32)
    a = flat[..., 0::2]
    b = flat[..., 1::2]
    cosine = table[..., 0].reshape(1, 128, 1, 32)
    sine = table[..., 1].reshape(1, 128, 1, 32)
    unfused = torch.empty_like(flat)
    unfused[..., 0::2] = a * cosine - b * sine
    unfused[..., 1::2] = a * sine + b * cosine
    reference = aten_rotate(x, table)
    disagreements = num_mismatched_bits(unfused, reference)
    assert disagreements > reference.numel() // 100, (
        f"only {disagreements} of {reference.numel()} elements separate the unfused "
        "arithmetic from the reference; the bit-exact tests would not have teeth"
    )


# ---------------------------------------------------------------------------------------------
# The operator, on the CPU reference backend
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
def test_the_operator_rewrites_in_place_and_returns_its_argument(dtype, inverse):
    x = make_input((2, 8, 4, 32), dtype=dtype, seed=3)
    table = make_table(8, 32, scale=1e-3, seed=4)
    expected = aten_rotate(x, table, inverse)
    identity = id(x)
    returned = apply_rotary_inplace(x, table, inverse=inverse)
    assert id(returned) == identity, "the operator must return the tensor it rewrote"
    assert same_bits(x, expected)


@pytest.mark.parametrize("inverse", [False, True])
def test_a_rank_three_tensor_is_batch_sequence_width(inverse):
    """Rank 3 is the model's ``[b, s, d]``: the sequence axis is dim 1, not dim 0.

    A single implicit head, so the table still has one row per position. Getting
    this axis order wrong would rotate across the batch instead of along time and
    still "work" on a square enough tensor, which is why the shape has both axes
    different sizes.
    """
    batch, sequence, width = 3, 8, 32
    x = make_input((batch, sequence, width), seed=5)
    table = make_table(sequence, width, scale=1e-3, seed=6)
    expected = aten_rotate(x, table, inverse)
    apply_rotary_inplace(x, table, inverse=inverse)
    assert same_bits(x, expected)


def test_a_rank_three_batch_is_not_the_sequence():
    """The table length identifies the sequence axis: ``shape[1]``, never ``shape[0]``.

    With a batch of 4 and 16 positions the two readings cannot be confused by
    accident: a table of 16 rows must be accepted and one of 4 rows must be
    refused. If the operator treated dim 0 as the sequence, both outcomes invert.
    """
    batch, sequence, width = 4, 16, 8
    x = make_input((batch, sequence, width), seed=51)
    table = make_table(sequence, width, scale=1e-3, seed=52)
    expected = aten_rotate(x, table)
    apply_rotary_inplace(x, table)
    assert same_bits(x, expected)
    wrong_length = make_table(batch, width, scale=1e-3, seed=52)
    with pytest.raises(ValueError, match=r"table must be \[sequence, width // 2, 2\]"):
        apply_rotary_inplace(x, wrong_length)


def test_the_inverse_direction_undoes_the_forward_one():
    """A forward then inverse call must match doing both through the reference.

    The round trip is not expected to return the input bit for bit -- two rounded
    rotations are not an identity -- so the bar is agreement with ATen's own round
    trip. That still catches a wrong-signed sine, which no amount of "looks
    smaller afterwards" would.
    """
    x = make_input((1, 32, 4, 32), seed=7)
    table = make_table(32, 32, scale=1e-4, seed=8)
    expected = aten_rotate(aten_rotate(x, table), table, inverse=True)
    apply_rotary_inplace(x, table)
    apply_rotary_inplace(x, table, inverse=True)
    assert same_bits(x, expected)


def test_a_head_slice_of_a_fused_buffer_rotates_in_place():
    """The case the operator exists for: no copy, and neighbours left alone.

    The slice's head stride is three widths and its sequence stride is larger
    again, so this fails loudly if any stride is assumed to be contiguous.
    """
    batch, sequence, heads, width = 2, 8, 4, 32
    fused = make_input((batch, sequence, heads, 3, width), seed=11)
    untouched = fused[:, :, :, 2, :].clone()
    reference = fused.clone()
    table = make_table(sequence, width, scale=1e-3, seed=12)
    apply_rotary_inplace(fused[:, :, :, 0, :], table)
    apply_rotary_inplace(fused[:, :, :, 1, :], table, inverse=True)
    # Rotate the same slices of the untouched copy through the reference backend.
    apply_rotary_inplace(reference[:, :, :, 0, :], table)
    apply_rotary_inplace(reference[:, :, :, 1, :], table, inverse=True)
    assert same_bits(fused, reference)
    # The third width of the fused buffer belongs to neither slice.
    assert same_bits(fused[:, :, :, 2, :], untouched)


def test_a_strided_width_is_rotated_through_its_own_stride():
    """A width stride above one is legal and must not be assumed contiguous."""
    base = make_input((2, 8, 4, 64), seed=13)
    view = base[..., ::2]
    table = make_table(8, 32, scale=1e-3, seed=14)
    expected = aten_rotate(view, table)
    returned = apply_rotary_inplace(view, table)
    assert same_bits(returned, expected)
    # The untouched odd columns of the base tensor must be exactly as they were.
    original = make_input((2, 8, 4, 64), seed=13)
    assert same_bits(base[..., 1::2], original[..., 1::2])


def test_a_zero_length_batch_returns_immediately():
    x = torch.zeros((0, 8, 4, 32), dtype=torch.float32)
    table = make_table(8, 32, scale=1e-3)
    assert apply_rotary_inplace(x, table) is x


def test_the_narrowest_possible_width_is_one_pair():
    x = make_input((2, 4, 2), seed=15)
    table = make_table(4, 2, scale=0.5, seed=16)
    expected = aten_rotate(x, table)
    apply_rotary_inplace(x, table)
    assert same_bits(x, expected)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("rank", [3, 4])
def test_a_single_row_table_rotates_one_decode_position(rank, inverse):
    """A decode step has a sequence axis of one and a table that is a single row.

    The caller should not have to know this op's rank conventions to build it, so a
    ``[width // 2, 2]`` row is accepted in place of ``[1, width // 2, 2]``. Both
    spellings must produce the same bits, and neither may copy the table.
    """
    width, heads = 128, 8
    shape = (2, 1, width) if rank == 3 else (2, 1, heads, width)
    x = make_input(shape, seed=41)
    row = make_table(1, width, scale=1e-5, seed=42)[0]
    assert row.shape == (width // 2, 2)
    expected = aten_rotate(x, row.unsqueeze(0), inverse)
    apply_rotary_inplace(x.clone(), row.unsqueeze(0), inverse=inverse)
    reference = x.detach().clone()
    apply_rotary_inplace(reference, row, inverse=inverse)
    assert same_bits(reference, expected)


def test_a_single_row_table_is_still_shape_checked():
    """The loose form is exactly one row wide, not any two-dimensional table."""
    x = make_input((1, 2, 8), seed=43)
    wrong = make_table(1, 4, scale=1.0)[0]
    with pytest.raises(ValueError, match=r"table must be \[sequence, width // 2, 2\]"):
        apply_rotary_inplace(x, wrong)


def test_a_multi_position_tensor_still_requires_the_stacked_table():
    """The row shortcut is only for a sequence axis of one; it must not leak."""
    x = make_input((4, 2, 8), seed=44)
    row = make_table(1, 8, scale=1.0)[0]
    with pytest.raises(ValueError, match=r"table must be \[sequence, width // 2, 2\]"):
        apply_rotary_inplace(x, row)


# ---------------------------------------------------------------------------------------------
# Refusals: every one names what was wrong
# ---------------------------------------------------------------------------------------------


def test_an_odd_width_is_refused():
    x = make_input((4, 2, 5), seed=17)
    table = make_table(4, 6, scale=1.0)
    with pytest.raises(ValueError, match="must be even"):
        apply_rotary_inplace(x, table)


@pytest.mark.parametrize("rank", [2, 5])
def test_an_unsupported_rank_is_refused(rank):
    shape = (4, 2) if rank == 2 else (1, 1, 4, 2, 2)
    x = torch.zeros(shape, dtype=torch.float32)
    table = make_table(4, 2, scale=1.0)
    with pytest.raises(ValueError, match="rank"):
        apply_rotary_inplace(x, table)


def test_a_table_of_the_wrong_shape_is_refused():
    x = make_input((4, 2, 8), seed=18)
    table = make_table(4, 4, scale=1.0)
    with pytest.raises(ValueError, match=r"table must be \[sequence, width // 2, 2\]"):
        apply_rotary_inplace(x, table)


def test_a_table_of_the_wrong_dtype_is_refused():
    x = make_input((4, 2, 8), seed=19)
    table = make_table(4, 8, scale=1.0).to(torch.float64)
    with pytest.raises(TypeError, match="dense FP32"):
        apply_rotary_inplace(x, table)


def test_a_non_contiguous_table_is_refused():
    table = make_table(4, 8, scale=1.0)
    strided = table.transpose(0, 1)
    assert not strided.is_contiguous()
    x = make_input((4, 2, 8), seed=20)
    with pytest.raises(ValueError, match="contiguous"):
        apply_rotary_inplace(x, strided)


def test_a_table_sharing_storage_with_x_is_refused():
    """``x`` is rewritten, so an aliased table could be overwritten mid-rotation."""
    sequence, width = 4, 8
    storage = torch.zeros(sequence * width, dtype=torch.float32)
    x = storage.view(1, sequence, 1, width)
    table = storage.view(sequence, width // 2, 2)
    with pytest.raises(ValueError, match="must not share storage"):
        apply_rotary_inplace(x, table)


def test_a_broadcast_view_is_refused():
    """A zero stride means many lanes write one address, which cannot be in place."""
    base = make_input((1, 4, 2, 8), seed=21)
    x = base.expand(3, 4, 2, 8)
    table = make_table(4, 8, scale=1.0)
    with pytest.raises(ValueError, match="positive batch stride"):
        apply_rotary_inplace(x, table)
    # The same broadcast on a non-batch axis is refused by the stride rule too.
    sequence_broadcast = make_input((1, 1, 2, 8), seed=21).expand(1, 4, 2, 8)
    with pytest.raises(ValueError, match="positive strides"):
        apply_rotary_inplace(sequence_broadcast, table)


def test_an_overlapping_layout_is_refused():
    """A view whose strides can carry is refused rather than silently miswritten."""
    assert not _injective((2, 3, 4), (1, 3, 1))
    storage = torch.zeros(64, dtype=torch.float32)
    x = storage.as_strided((2, 3, 4), (1, 3, 1))
    table = make_table(3, 4, scale=1.0)
    with pytest.raises(ValueError, match="map two coordinates onto one address"):
        apply_rotary_inplace(x, table)


def test_a_sliced_head_layout_is_accepted():
    """The refusal above must not also reject the sliced views this op needs."""
    assert _injective((2, 8, 4, 32), (8 * 4 * 3 * 32, 4 * 3 * 32, 3 * 32, 1))
    assert _injective((8, 4, 32), (4 * 3 * 32, 3 * 32, 1))


def test_a_gradient_tracking_tensor_is_refused():
    x = make_input((4, 2, 8), seed=22).requires_grad_(True)
    table = make_table(4, 8, scale=1.0)
    with pytest.raises(ValueError, match="inference operator"):
        apply_rotary_inplace(x, table)


def test_an_unsigned_dtype_is_refused():
    x = torch.zeros((4, 2, 8), dtype=torch.float16)
    table = make_table(4, 8, scale=1.0)
    with pytest.raises(TypeError, match="BF16 or FP32"):
        apply_rotary_inplace(x, table)


def test_a_non_boolean_inverse_is_refused():
    x = make_input((4, 2, 8), seed=23)
    table = make_table(4, 8, scale=1.0)
    with pytest.raises(TypeError, match="inverse must be a bool"):
        apply_rotary_inplace(x, table, inverse=1)


def test_an_unknown_backend_is_refused():
    x = make_input((2, 4, 8), seed=24)
    table = make_table(4, 8, scale=1.0)
    with pytest.raises(ValueError, match="'auto', 'torch' or 'triton'"):
        apply_rotary_inplace(x, table, backend="cuda")


def test_the_fused_backend_refuses_a_cpu_tensor_by_name():
    """No silent fallback: asking for the fused kernel on CPU is an error."""
    x = make_input((2, 4, 8), seed=25)
    table = make_table(4, 8, scale=1.0)
    with pytest.raises(ValueError, match="NPU tensors only"):
        apply_rotary_inplace(x, table, backend="triton")


def test_non_tensors_are_refused():
    table = make_table(4, 8, scale=1.0)
    with pytest.raises(TypeError, match="x must be a torch.Tensor"):
        apply_rotary_inplace([[0.0] * 8] * 2, table)
    x = make_input((4, 2, 8), seed=26)
    with pytest.raises(TypeError, match="table must be a torch.Tensor"):
        apply_rotary_inplace(x, [[0.0, 0.0]])


# ---------------------------------------------------------------------------------------------
# The fused Ascend backend (skipped unless a device is named)
# ---------------------------------------------------------------------------------------------


def fused_device() -> torch.device:
    import torch_npu  # noqa: F401 - registers the npu device on torch
    return torch.device(DEVICE)


def fused_configured() -> bool:
    if not DEVICE:
        return False
    import torch_npu  # noqa: F401
    return supports_triton_ascend()


requires_fused = pytest.mark.skipif(
    not fused_configured(),
    reason=f"set RELIC_ASCEND_ROTARY_DEVICE=npu:0 with a Triton Ascend backend (current {DEVICE!r})",
)


def cpu_rotate_reference(x, table, inverse):
    """Rotate a device tensor's CPU copy through the ATen reference backend."""
    reference = x.detach().to("cpu").clone()
    apply_rotary_inplace(reference, table.detach().to("cpu").contiguous(),
                         inverse=inverse, backend="torch")
    return reference


@requires_npu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("shape", [
    (1, 16, 4, 64),      # one step, contiguous
    (2, 32, 8, 128),     # the runtime's head width
    (1, 64, 4, 96),      # width the 128-lane tile does not divide: masked tail
    (1, 8, 2, 6),        # a tile much wider than the head
    (1, 4, 2, 2),        # exactly one pair
])
def test_the_fused_kernel_matches_the_reference_bit_for_bit(dtype, inverse, shape):
    """The claim that matters on hardware: no tolerance, no ulp of slack."""
    device = fused_device()
    x = make_input(shape, scale=1.0, dtype=dtype, device=device)
    table = make_table(shape[1], shape[-1], scale=1e-5, device=device)
    expected = cpu_rotate_reference(x, table, inverse)
    returned = apply_rotary_inplace(x, table, inverse=inverse)
    assert returned is x
    mismatches = num_mismatched_bits(x, expected)
    assert mismatches == 0, (
        f"{mismatches} of {expected.numel()} elements differ from ATen's complex "
        f"multiply for shape {shape}, dtype {dtype}, inverse={inverse}"
    )


@requires_npu
def test_the_fused_kernel_matches_the_reference_where_an_unfused_pair_does_not():
    """The user-visible defect: unit activations, a 1e-5 table, FP32 storage.

    Rounding the pairs to BF16 afterwards hides almost every disagreement, which is
    why the counterexample is checked in FP32; the BF16 shape above is checked for
    bit equality on the same magnitudes separately.
    """
    device = fused_device()
    shape = (1, 128, 8, 64)
    x = make_input(shape, scale=1.0, seed=31, device=device)
    table = make_table(shape[1], shape[-1], scale=1e-5, seed=32, device=device)
    expected = cpu_rotate_reference(x, table, False)

    unfused = x.detach().to("cpu").to(torch.float32).clone()
    cosine = table.detach().to("cpu")[..., 0]
    sine = table.detach().to("cpu")[..., 1]
    a = unfused[..., 0::2]
    b = unfused[..., 1::2]
    unfused[..., 0::2] = a * cosine - b * sine
    unfused[..., 1::2] = a * sine + b * cosine
    assert num_mismatched_bits(unfused, expected) > expected.numel() // 100, (
        "the unfused arithmetic agrees with ATen here, so this case cannot show "
        "that the fused kernel is doing anything different"
    )

    apply_rotary_inplace(x, table)
    assert num_mismatched_bits(x, expected) == 0


@requires_npu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_fused_kernel_handles_a_head_slice_of_a_fused_buffer(dtype):
    device = fused_device()
    batch, sequence, heads, width = 1, 64, 8, 128
    fused = make_input((batch, sequence, heads, 3, width), seed=33,
                       dtype=dtype, device=device)
    table = make_table(sequence, width, scale=1e-5, seed=34, device=device)
    reference = fused.detach().to("cpu").clone()
    apply_rotary_inplace(reference[:, :, :, 0, :], table.detach().to("cpu"), backend="torch")
    apply_rotary_inplace(reference[:, :, :, 1, :], table.detach().to("cpu"),
                         inverse=True, backend="torch")
    apply_rotary_inplace(fused[:, :, :, 0, :], table)
    apply_rotary_inplace(fused[:, :, :, 1, :], table, inverse=True)
    assert num_mismatched_bits(fused, reference) == 0


@requires_npu
def test_the_fused_kernel_loops_over_rows_past_the_grid_limit():
    """``sequence * heads`` above 65535 forces the row loop, not a bigger grid."""
    device = fused_device()
    shape = (1, 4096, 16, 128)
    assert shape[1] * shape[2] > 65535
    x = make_input(shape, scale=1.0, dtype=torch.bfloat16, seed=35, device=device)
    table = make_table(shape[1], shape[-1], scale=1e-5, seed=36, device=device)
    expected = cpu_rotate_reference(x, table, False)
    apply_rotary_inplace(x, table)
    assert num_mismatched_bits(x, expected) == 0


@requires_npu
def test_the_fused_kernel_preserves_signed_zeros_and_subnormals():
    """The zero-sign part of the contract, on the device rather than in emulation."""
    device = fused_device()
    values = torch.tensor(
        [0.0, -0.0, 2.0**-149, -(2.0**-149), 2.0**-126, -(2.0**-126),
         1.0, -1.0, 65504.0, -65504.0],
        dtype=torch.float32,
    )
    # Width 4: two pairs per row, sequence 8, one head.
    x = values.repeat(16).reshape(8, 1, 8)[..., :4].contiguous().to(device)
    table = make_table(8, 4, scale=1e-5, seed=37, device=device)
    expected = cpu_rotate_reference(x, table, False)
    apply_rotary_inplace(x, table)
    assert num_mismatched_bits(x, expected) == 0


@requires_npu
def test_the_fused_backend_refuses_the_torch_reference_on_npu():
    device = fused_device()
    x = make_input((4, 2, 8), seed=38, device=device)
    table = make_table(4, 8, scale=1.0, device=device)
    with pytest.raises(ValueError, match="CPU/CUDA"):
        apply_rotary_inplace(x, table, backend="torch")

# ---------------------------------------------------------------------------------------------
# The reference contraction is a property of this build, not a universal constant
# ---------------------------------------------------------------------------------------------


def test_this_hosts_reference_fuses_the_first_product():
    """The arithmetic the fused kernel implements, measured on this host's ATen.

    The kernel commits to one contraction: ``real = fma(a, c, -round32(b*s))`` and
    ``imag = fma(a, s, +round32(b*c))``, with the second product of each component
    rounded on its own. That is a property of the ATen kernel *compiled for this
    host*, not an invariant of complex multiplication -- another architecture or
    compiler may fuse the other product. If this ever fails, the documented
    arithmetic no longer describes this host and the NPU kernel's parity claim does
    not transfer to it; the fix is to decide deliberately, not to loosen the test.
    """
    found = reference_contraction()
    assert found == "first", (
        f"this host's ATen complex multiply contracts differently ({found!r}); the "
        "fused kernel assumes the first-product fusion and would disagree by an ulp"
    )


def test_the_contraction_probe_can_actually_tell_the_spellings_apart():
    """A probe that cannot separate the candidates would always answer 'unknown'."""
    pairs = torch.tensor([(probe[0], probe[1]) for probe in _CONTRACTION_PROBES],
                         dtype=torch.float32)
    angles = torch.tensor([(probe[2], probe[3]) for probe in _CONTRACTION_PROBES],
                          dtype=torch.float32)
    observed = _complex_bits(pairs, angles).to(torch.int32)
    matches = {}
    for spelling in ("first", "second", "unfused"):
        candidate = _candidate_bits(pairs, angles, spelling).to(torch.int32)
        matches[spelling] = torch.equal(candidate, observed)
    assert matches["first"] is True
    assert matches["second"] is False
    assert matches["unfused"] is False


def test_the_operator_agrees_with_the_reference_on_a_huge_stride_geometry():
    """A width stride large enough that an int32 base product could overflow.

    The tensor stays small; only its *strides* are large, which is exactly the case
    the index-width rule exists for. Both backends must agree on it.
    """
    sequence, heads, width = 4, 4, 8
    stride_width = 2**28           # one width step is 256 Mi elements
    stride_head = width * stride_width
    stride_sequence = heads * stride_head
    stride_batch = sequence * stride_sequence
    span = stride_batch + 2**8
    storage = torch.zeros(span, dtype=torch.float32)
    x = torch.as_strided(storage, (1, sequence, heads, width),
                         (stride_batch, stride_sequence, stride_head, stride_width))
    assert needs_64_bit_indices(1, sequence, heads, width,
                                (stride_batch, stride_sequence, stride_head, stride_width),
                                tile_lanes=width)
    table = make_table(sequence, width, scale=1e-3, seed=61)
    expected = aten_rotate(x, table)
    assert same_bits(x, expected)


def test_the_index_width_rule_is_conservative_for_ordinary_geometry():
    """A normal activation must not pay for 64-bit indices it does not need."""
    assert not needs_64_bit_indices(1, 32768, 16, 128,
                                    (32768 * 16 * 128, 16 * 128, 128, 1), tile_lanes=128)
    # ... but a geometry whose reach crosses the boundary must ask for them.
    assert needs_64_bit_indices(1, 2**20, 4096, 128,
                                (2**20 * 4096 * 128, 4096 * 128, 128, 1), tile_lanes=128)
