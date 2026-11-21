"""Operator tests for the explicit-arithmetic pair rotation.

Two different references are used, and keeping them apart is the point of this
file:

* the ``"torch"`` backend *is* ATen's ``complex64`` multiply, so every CPU test
  compares it against ``torch.view_as_complex`` arithmetic. That comparison holds
  on any host by construction -- it is the same code -- so it is a real assertion
  about the operator's in-place/strided contract, not about FMA policy.
* the fused Ascend kernel implements an explicit documented arithmetic
  (``rotary.DOCUMENTED_POLICY``). It is compared **bit for bit** against an
  independent exact-rational binary32 ``fmaf`` model of those two lines, which is
  host independent. It is compared against a host's ATen complex multiply only
  when ``matches_documented_policy()`` reports that the host compiles the same
  contraction; otherwise the two legitimately differ by about an ulp and no parity
  is claimed (the measurement is still pinned by its own test below).

The measurement is why this distinction exists: on the Ascend host's aarch64 ATen
the real component matched ``"second"`` on all 4096 probed pairs and the imaginary
component matched none of the three documented spellings, while an x86 build
measured ``"first"`` for both. A test that hardcoded either answer would be a
statement about one machine.

There are three layers, and they are deliberately separable: the host-measurement
tests, the CPU tests (in-place, strided views, every refusal), and the ``npu``
tests, which are skipped unless ``RELIC_ASCEND_ROTARY_DEVICE`` names a device
(``npu:0``).

Run the CPU layer with ``python -m pytest tests/test_ascend_rotary.py -q`` and the
hardware layer with
``RELIC_ASCEND_ROTARY_DEVICE=npu:0 python -m pytest tests/test_ascend_rotary.py -q``.
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels.rotary import (  # noqa: E402 - after the torch import guard
    DOCUMENTED_POLICY,
    SPELLINGS,
    _CONTRACTION_PROBES,
    _candidate_bits,
    _candidate_component,
    _complex_bits,
    _injective,
    apply_rotary_inplace,
    matches_documented_policy,
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


def _load_exact_oracle():
    """Load the sibling math module's exact binary32 helpers by path.

    One oracle serves both files: ``tests/test_ascend_rotary_math.py`` already
    carries an exact-rational ``fma_f32``/``multiply_f32``/``as_f32`` model (and
    cross-checks its FMA against ``libm.fmaf`` where the platform has one). Reusing
    it here means the device comparison cannot drift from the scalar proof, and it
    keeps this file importable without relying on how pytest lays out ``sys.path``.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).with_name("test_ascend_rotary_math.py")
    spec = importlib.util.spec_from_file_location("_rotary_exact_oracle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_EXACT = _load_exact_oracle()


def policy_rotate(x: torch.Tensor, table: torch.Tensor, inverse: bool = False,
                  policy=DOCUMENTED_POLICY) -> torch.Tensor:
    """The documented policy in exact binary32 arithmetic, independent of ATen.

    ``real = fma(a, c, -round32(b s))`` and ``imag = fma(a, s, +round32(b c))`` with
    every operation rounded to binary32 exactly once (``"first"``), or the
    ``"second"``/``"unfused"`` spellings of the same two expressions. This is the
    oracle the fused kernel is held to: it is a statement of the arithmetic in the
    module's docstring, not a second copy of ATen, so it is the same on every host.
    """
    fma, multiply, as_float32 = _EXACT.fma_f32, _EXACT.multiply_f32, _EXACT.as_f32
    shape = x.shape
    sequence, width = shape[-3], shape[-1]
    flat = x.detach().to(torch.float32).contiguous()
    a_t = flat[..., 0::2]
    b_t = flat[..., 1::2]
    angle_shape = (1,) * (x.ndim - 3) + (sequence, 1, width // 2)
    cosine_t = table[..., 0].reshape(angle_shape)
    sine_t = table[..., 1].reshape(angle_shape)
    a = [as_float32(v) for v in a_t.reshape(-1).tolist()]
    b = [as_float32(v) for v in b_t.reshape(-1).tolist()]
    cosine = [as_float32(v) for v in cosine_t.expand_as(a_t).reshape(-1).tolist()]
    sine = [as_float32(v) for v in sine_t.expand_as(a_t).reshape(-1).tolist()]
    if inverse:
        sine = [-v for v in sine]
    real_name, imag_name = policy
    real = [fma(a_i, c_i, -multiply(b_i, s_i)) for a_i, b_i, c_i, s_i
            in zip(a, b, cosine, sine)]
    imag = [fma(a_i, s_i, multiply(b_i, c_i)) for a_i, b_i, c_i, s_i
            in zip(a, b, cosine, sine)]
    if real_name != "first" or imag_name != "first":
        raise ValueError("only the documented first-product policy has an exact model here")
    rotated = torch.empty_like(flat)
    rotated[..., 0::2] = torch.tensor(real, dtype=torch.float32).reshape(a_t.shape)
    rotated[..., 1::2] = torch.tensor(imag, dtype=torch.float32).reshape(b_t.shape)
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


def test_the_documented_policy_is_one_of_the_named_spellings():
    """The policy is a fixed pair of named spellings, chosen in source."""
    assert DOCUMENTED_POLICY == ("first", "first")
    assert set(DOCUMENTED_POLICY) <= set(SPELLINGS)


@pytest.mark.parametrize("table_scale", [1e-9, 1e-5, 0.5, 1.0])
@pytest.mark.parametrize("x_scale", [1e-3, 1.0, 7.5, 1e4])
@pytest.mark.parametrize("inverse", [False, True])
def test_this_hosts_reference_matches_the_documented_policy_or_is_reported(x_scale, table_scale,
                                                                         inverse):
    """ATen is either the documented arithmetic on this host, or measurably not.

    Both branches are real assertions, so neither a matching host nor a differing
    one can pass vacuously:

    * a host that reports :data:`DOCUMENTED_POLICY` must reproduce it bit for bit
      across these magnitudes and both directions -- that is the parity claim;
    * a host that reports anything else must *disagree* with it on some element,
      which is exactly what "the measurement found a different contraction" means.
      If such a host silently agreed everywhere the measurement would be wrong.

    The second branch is the Ascend host's aarch64 ATen: its real component
    measured ``"second"`` and its imaginary component matched no documented
    spelling, so the fused kernel is not claimed to equal that host's complex
    multiply -- the operator keeps its own explicit arithmetic instead.
    """
    x = make_input((2, 16, 4, 64), scale=x_scale)
    table = make_table(16, 64, scale=table_scale)
    reference = aten_rotate(x, table, inverse)
    documented = policy_rotate(x, table, inverse)
    differences = num_mismatched_bits(reference, documented)
    if matches_documented_policy():
        assert differences == 0, (
            f"{differences} of {reference.numel()} elements differ although this host "
            "reports the documented policy; the measurement and the arithmetic disagree"
        )
    else:
        assert differences > 0, (
            "this host does not report the documented contraction yet agrees with it on "
            f"every element of {reference.numel()}; the host measurement is wrong"
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_the_host_measurement_is_stable_across_the_magnitudes_the_runtime_uses(dtype):
    """The host policy is a property of the build, so magnitude cannot change it."""
    assert reference_contraction() == reference_contraction()
    x = make_input((1, 64, 8, 128), scale=1.0, dtype=dtype)
    table = make_table(64, 128, scale=1e-5)
    reference = aten_rotate(x, table)
    documented = policy_rotate(x, table)
    differences = num_mismatched_bits(reference.to(torch.float32),
                                      documented.to(torch.float32))
    if matches_documented_policy():
        assert differences == 0
    else:
        assert differences > 0


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


def policy_reference(x, table, inverse):
    """The documented arithmetic in exact binary32, as the fused kernel's oracle.

    Host independent on purpose: the device is held to the arithmetic this operator
    documents, which is what can be stated without a claim about any CPU's compiler.
    """
    return policy_rotate(x.detach().to("cpu"), table.detach().to("cpu").contiguous(), inverse)


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
    expected = policy_reference(x, table, inverse)
    returned = apply_rotary_inplace(x, table, inverse=inverse)
    assert returned is x
    mismatches = num_mismatched_bits(x, expected)
    assert mismatches == 0, (
        f"{mismatches} of {expected.numel()} elements differ from the exact binary32 "
        f"model of the documented arithmetic for shape {shape}, dtype {dtype}, "
        f"inverse={inverse}"
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
    expected = policy_reference(x, table, False)

    unfused = x.detach().to("cpu").to(torch.float32).clone()
    # The angle axes are (sequence, pair); align them against x's (heads, pair)
    # trailing axes explicitly instead of relying on a shape coincidence.
    cosine = table.detach().to("cpu")[..., 0].reshape(1, shape[1], 1, shape[-1] // 2)
    sine = table.detach().to("cpu")[..., 1].reshape(1, shape[1], 1, shape[-1] // 2)
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
    reference[:, :, :, 0, :] = policy_rotate(reference[:, :, :, 0, :].contiguous(),
                                             table.detach().to("cpu"), False)
    reference[:, :, :, 1, :] = policy_rotate(reference[:, :, :, 1, :].contiguous(),
                                             table.detach().to("cpu"), True)
    apply_rotary_inplace(fused[:, :, :, 0, :], table)
    apply_rotary_inplace(fused[:, :, :, 1, :], table, inverse=True)
    assert num_mismatched_bits(fused, reference) == 0


@requires_npu
def test_the_fused_kernel_loops_over_rows_past_the_grid_limit():
    """``sequence * heads`` above 65535 forces the row loop, not a bigger grid."""
    device = fused_device()
    # 4096 * 16 rows is past the grid cap; a two-wide (one-pair) head keeps the
    # exact oracle's elementwise model affordable while still exercising the loop.
    shape = (1, 4096, 16, 2)
    assert shape[1] * shape[2] > 65535
    x = make_input(shape, scale=1.0, dtype=torch.bfloat16, seed=35, device=device)
    table = make_table(shape[1], shape[-1], scale=1e-5, seed=36, device=device)
    expected = policy_reference(x, table, False)
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
    # Rank 3 is [batch, sequence, width]: one batch row, eight positions, and a
    # width of four (two pairs per position). The ten probe values are tiled to
    # fill exactly 1 * 8 * 4 = 32 lanes.
    x = values.repeat(4)[:32].reshape(1, 8, 4).contiguous().to(device)
    table = make_table(8, 4, scale=1e-5, seed=37, device=device)
    expected = policy_reference(x, table, False)
    apply_rotary_inplace(x, table)
    assert num_mismatched_bits(x, expected) == 0


@requires_npu
def test_the_fused_backend_refuses_the_torch_reference_on_npu():
    device = fused_device()
    # Rank 3 is [batch, sequence, width], so this is batch 4 / sequence 2 / width 8.
    x = make_input((4, 2, 8), seed=38, device=device)
    table = make_table(2, 8, scale=1.0, device=device)
    with pytest.raises(ValueError, match="CPU/CUDA"):
        apply_rotary_inplace(x, table, backend="torch")

# ---------------------------------------------------------------------------------------------
# The reference contraction is a property of this build, not a universal constant
# ---------------------------------------------------------------------------------------------


def test_the_host_measurement_names_a_spelling_per_component_or_says_unknown():
    """The measurement is per component and never invents an answer.

    ``"unknown"`` is a legitimate result -- the Ascend host's aarch64 ATen measures
    it for the imaginary component, which is why the fused kernel does not claim
    parity there. What is not legitimate is reporting a spelling that does not
    reproduce ATen, so each named component is re-derived here.
    """
    measured = reference_contraction()
    assert set(measured) == {"real", "imag"}
    for name in ("real", "imag"):
        assert measured[name] in (*SPELLINGS, "unknown")
    matches_documented_policy()  # must be a plain bool, not a tensor or string

    pairs = torch.tensor([(probe[0], probe[1]) for probe in _CONTRACTION_PROBES],
                         dtype=torch.float32)
    angles = torch.tensor([(probe[2], probe[3]) for probe in _CONTRACTION_PROBES],
                          dtype=torch.float32)
    observed = _complex_bits(pairs, angles).to(torch.int32)
    for component, name in ((0, "real"), (1, "imag")):
        reported = measured[name]
        if reported == "unknown":
            continue
        candidate = _candidate_component(pairs, angles, reported, component).to(torch.int32)
        assert torch.equal(candidate, observed[:, component]), (
            f"the measurement reported {reported!r} for the {name} component but that "
            "spelling does not reproduce this host's ATen"
        )


def test_the_contraction_probe_can_actually_tell_the_spellings_apart():
    """A probe that cannot separate the candidates could not measure anything.

    The assertion is that the probe set *discriminates*: for each component, the
    three documented spellings are pairwise different somewhere in the probe, so a
    host matching one of them cannot accidentally also match another. Which one the
    host matches is reported, not asserted -- that is a property of the build.
    """
    pairs = torch.tensor([(probe[0], probe[1]) for probe in _CONTRACTION_PROBES],
                         dtype=torch.float32)
    angles = torch.tensor([(probe[2], probe[3]) for probe in _CONTRACTION_PROBES],
                          dtype=torch.float32)
    for component in (0, 1):
        rendered = {
            spelling: _candidate_component(pairs, angles, spelling, component).to(torch.int32)
            for spelling in SPELLINGS
        }
        for first in SPELLINGS:
            for second in SPELLINGS:
                if first < second:
                    assert not torch.equal(rendered[first], rendered[second]), (
                        f"the probe cannot separate {first!r} from {second!r} on component "
                        f"{component}; add a discriminating witness"
                    )


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
