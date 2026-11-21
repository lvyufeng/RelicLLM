"""Contracts for the reversible CubeB-order packed-FP4 layout.

The point of these tests is that the layout is *lossless and exactly sized*: whatever the
transform does, the bytes it hands back must be the bytes it was given, in the same count,
with the same dtype spelling. Two kinds of evidence are used, and they are deliberately
independent of each other:

* an **independent scalar coordinate oracle** below, written as nested loops over the
  documented index arithmetic with its own literals -- it never calls ``view``, ``permute``
  or any module constant for the mapping, so it states the formula a second time;
* a **byte-level roundtrip**, which needs no formula at all: convert then recover and
  compare storage.

Corpus choices are aimed at the ways a permutation can be wrong rather than merely at
coverage: all 256 packed byte values in one projection (which includes both FP4 zeros,
both signed zeros, the largest magnitude and the code a consumer treats as non-finite),
all 256 scale byte values in another, non-square and equal tile counts, and a payload whose
bytes are all distinct so a transposed axis cannot hide behind symmetry.

Nothing here runs a kernel, allocates device memory, or hashes anything.
"""
from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from relic_core.kernels import cube_packed_format as cube  # noqa: E402

#: Tile extents restated as literals on purpose: the oracle must not read the module's
#: constants, or a wrong constant would agree with itself.
ROWS = 32
KTILE = 128
HALF_ROWS = 16
HALVES = 2
KPAIR = 8
K16 = 8
SCALE_GROUPS = 4


# ------------------------------------------------------------------- the scalar oracle


def _oracle_codes_forward(source: list[int], n: int, k: int) -> list[int]:
    """``[N, K/2]`` bytes -> ``[N/32, K/128, 8, 2, 16, 8]`` bytes, one coordinate at a time.

    Destination flat index is
    ``((((nb * ktiles + kb) * 8 + k16) * 2 + n16) * 16 + row) * 8 + kpair``
    with ``nb, n16, row = n // 32, (n % 32) // 16, n % 16`` and
    ``kb, k16, kpair = j // 64, (j % 64) // 8, j % 8``.
    """
    ktiles = k // KTILE
    packed = k // 2
    out = [0] * len(source)
    for row_index in range(n):
        nb, n16, row = row_index // ROWS, (row_index % ROWS) // HALF_ROWS, row_index % HALF_ROWS
        base = row_index * packed
        for j in range(packed):
            kb, k16, kpair = j // (K16 * KPAIR), (j % (K16 * KPAIR)) // KPAIR, j % KPAIR
            destination = (
                (((nb * ktiles + kb) * K16 + k16) * HALVES + n16) * HALF_ROWS + row
            ) * KPAIR + kpair
            out[destination] = source[base + j]
    return out


def _oracle_codes_inverse(cube_bytes: list[int], n: int, k: int) -> list[int]:
    """The inverse of :func:`_oracle_codes_forward`, by reading the same coordinate pairs."""
    ktiles = k // KTILE
    packed = k // 2
    out = [0] * len(cube_bytes)
    for row_index in range(n):
        nb, n16, row = row_index // ROWS, (row_index % ROWS) // HALF_ROWS, row_index % HALF_ROWS
        base = row_index * packed
        for j in range(packed):
            kb, k16, kpair = j // (K16 * KPAIR), (j % (K16 * KPAIR)) // KPAIR, j % KPAIR
            destination = (
                (((nb * ktiles + kb) * K16 + k16) * HALVES + n16) * HALF_ROWS + row
            ) * KPAIR + kpair
            out[base + j] = cube_bytes[destination]
    return out


def _oracle_scales_forward(source: list[int], n: int, k: int) -> list[int]:
    """``[N, K/32]`` scales -> ``[N/32, K/128, 4, 32]`` scales.

    Destination flat index is ``(((nb * ktiles + kb) * 4 + group) * 32 + nrow)`` with
    ``nb, nrow = n // 32, n % 32`` and ``kb, group = s // 4, s % 4``.
    """
    ktiles = k // KTILE
    blocks = k // 32
    out = [0] * len(source)
    for row_index in range(n):
        nb, nrow = row_index // ROWS, row_index % ROWS
        base = row_index * blocks
        for s in range(blocks):
            kb, group = s // SCALE_GROUPS, s % SCALE_GROUPS
            destination = ((nb * ktiles + kb) * SCALE_GROUPS + group) * ROWS + nrow
            out[destination] = source[base + s]
    return out


def _oracle_scales_inverse(cube_bytes: list[int], n: int, k: int) -> list[int]:
    """The inverse of :func:`_oracle_scales_forward`."""
    ktiles = k // KTILE
    blocks = k // 32
    out = [0] * len(cube_bytes)
    for row_index in range(n):
        nb, nrow = row_index // ROWS, row_index % ROWS
        base = row_index * blocks
        for s in range(blocks):
            kb, group = s // SCALE_GROUPS, s % SCALE_GROUPS
            destination = ((nb * ktiles + kb) * SCALE_GROUPS + group) * ROWS + nrow
            out[base + s] = cube_bytes[destination]
    return out


# --------------------------------------------------------------------------- helpers


def _all_bytes_pattern(count: int) -> list[int]:
    """``count`` bytes cycling through all 256 values, so every packed byte is present."""
    return [index % 256 for index in range(count)]


def _flat(tensor: torch.Tensor) -> list[int]:
    return tensor.reshape(-1).tolist()


def _make(n: int, k: int, *, code_seed: int = 0):
    """A source pair with distinct code bytes and a scale grid covering all 256 values."""
    codes = torch.tensor(
        [(code_seed + index) % 256 for index in range(n * (k // 2))], dtype=torch.uint8
    ).reshape(n, k // 2)
    scales = torch.tensor(_all_bytes_pattern(n * (k // 32)), dtype=torch.uint8).reshape(
        n, k // 32
    )
    return codes, scales


# ---------------------------------------------------------------------------- geometry


@pytest.mark.parametrize(
    ("n", "k"),
    [(32, 128), (64, 256), (96, 384), (32, 1280), (256, 128)],
)
def test_geometry_reports_exact_shapes_and_a_padding_free_budget(n, k):
    """The shape statement and the byte budget, for real tile counts."""
    geometry = cube.cube_geometry(n, k)
    assert geometry.codes_shape == (n // 32, k // 128, 8, 2, 16, 8)
    assert geometry.scales_shape == (n // 32, k // 128, 4, 32)
    assert geometry.source_codes_shape == (n, k // 2)
    assert geometry.source_scales_shape == (n, k // 32)

    assert geometry.codes_bytes == n * k // 2
    assert geometry.scales_bytes == n * k // 32
    # Size-exactness is the whole premise: same bytes out, and no padding term.
    assert geometry.payload_bytes == geometry.source_bytes
    assert geometry.payload_bytes == n * k // 2 + n * k // 32
    assert geometry.padding_bytes == 0

    report = geometry.report()
    assert report["format"] == cube.FORMAT == "relic.fp4.cube-n32-k128.v1"
    assert report["size_exact"] is True
    assert report["padding_bytes"] == 0
    assert report["payload_bytes"] == geometry.payload_bytes
    assert report["n"] == n and report["k"] == k


@pytest.mark.parametrize(
    ("n", "k"),
    [
        (48, 128),  # n not a multiple of 32
        (31, 128),  # a partial row tile
        (32, 96),  # k not a multiple of 128
        (32, 64),  # a partial K tile
        (0, 128),  # empty
        (-32, 128),  # negative
    ],
)
def test_partial_tiles_are_refused_rather_than_padded(n, k):
    """The format has no padding, so a geometry that would need some is not a geometry."""
    with pytest.raises(ValueError):
        cube.cube_geometry(n, k)


@pytest.mark.parametrize(("n", "k"), [(True, 128), (32, False), (32.0, 128), (32, 128.0)])
def test_counts_must_be_integers_and_a_bool_is_not_a_count(n, k):
    """``bool`` is an ``int`` subclass, so it has to be refused by name, not by accident."""
    with pytest.raises(TypeError):
        cube.cube_geometry(n, k)


# ------------------------------------------------------------------- independent oracle


@pytest.mark.parametrize(("n", "k"), [(32, 128), (64, 256), (96, 128)])
def test_code_cube_order_matches_the_independent_coordinate_oracle(n, k):
    """Every destination byte, against a loop written from the index arithmetic."""
    codes, scales = _make(n, k)
    payload = cube.convert_fp4_cube(codes, scales)

    assert tuple(payload.codes.shape) == (n // 32, k // 128, 8, 2, 16, 8)
    assert payload.codes.dtype == torch.uint8
    assert payload.codes.is_contiguous()
    assert _flat(payload.codes) == _oracle_codes_forward(codes.reshape(-1).tolist(), n, k)


@pytest.mark.parametrize(("n", "k"), [(32, 128), (64, 256), (96, 128)])
def test_scale_cube_order_matches_the_independent_coordinate_oracle(n, k):
    """The scale order is a different permutation from the code order; state it separately."""
    codes, scales = _make(n, k)
    payload = cube.convert_fp4_cube(codes, scales)

    assert tuple(payload.scales.shape) == (n // 32, k // 128, 4, 32)
    assert payload.scales.dtype == torch.uint8
    assert payload.scales.is_contiguous()
    assert _flat(payload.scales) == _oracle_scales_forward(scales.reshape(-1).tolist(), n, k)


def test_the_oracle_agrees_with_a_full_roundtrip_of_its_own():
    """The oracle's inverse has to invert the oracle before it can judge the module."""
    n, k = 64, 256
    codes, scales = _make(n, k)
    forward = _oracle_codes_forward(codes.reshape(-1).tolist(), n, k)
    assert _oracle_codes_inverse(forward, n, k) == codes.reshape(-1).tolist()
    forward_scales = _oracle_scales_forward(scales.reshape(-1).tolist(), n, k)
    assert _oracle_scales_inverse(forward_scales, n, k) == scales.reshape(-1).tolist()


def test_the_layout_is_not_the_identity():
    """A no-op would satisfy every roundtrip test, so pin that the bytes actually move."""
    n, k = 32, 128
    codes, scales = _make(n, k)
    payload = cube.convert_fp4_cube(codes, scales)
    assert _flat(payload.codes) != codes.reshape(-1).tolist()
    assert _flat(payload.scales) != scales.reshape(-1).tolist()


# --------------------------------------------------------------------- roundtrip, bytes


@pytest.mark.parametrize(("n", "k"), [(32, 128), (64, 256), (96, 384)])
def test_roundtrip_is_byte_exact_over_all_256_packed_byte_values(n, k):
    """Every possible packed byte value, through convert and back, byte for byte.

    ``0x00``/``0x80`` carry a +0/-0 nibble, ``0x08``/``0x88`` the other zero pair, and
    ``0x7F``/``0xFF`` the code a consumer treats as non-finite. None of them is interpreted,
    so all of them must come back identical.
    """
    codes, scales = _make(n, k)
    assert set(codes.reshape(-1).tolist()) == set(range(256))
    payload = cube.convert_fp4_cube(codes, scales)
    recovered_codes, recovered_scales = cube.recover_fp4_cube(payload)

    assert recovered_codes.dtype == torch.uint8
    assert recovered_codes.shape == codes.shape
    assert recovered_scales.shape == scales.shape
    assert _flat(recovered_codes) == _flat(codes)
    assert _flat(recovered_scales) == _flat(scales)


@pytest.mark.parametrize(("n", "k"), [(32, 128), (64, 256)])
def test_scale_bytes_are_a_pure_permutation_including_the_extremes(n, k):
    """Every scale byte value is legal here: this format does not interpret scales.

    That is deliberately unlike the FP8 GEMV entry point, which refuses ``0xFF``; a byte
    permutation has no opinion about what a byte means, and it must not acquire one.
    """
    codes, _ = _make(n, k)
    count = n * (k // 32)
    # Repeat the boundary values so they are present whatever the grid size is, then fill
    # the remainder with a sweep. 0x00/0x01 are the smallest grids, 0xFE/0xFF the largest.
    extremes = (0x00, 0x01, 0x7F, 0x80, 0xFE, 0xFF)
    pattern = (list(extremes) * (count // len(extremes) + 1))[:count]
    scales = torch.tensor(pattern, dtype=torch.uint8).reshape(n, k // 32)
    for extreme in extremes:
        assert extreme in scales.reshape(-1).tolist()
    payload = cube.convert_fp4_cube(codes, scales)
    _, recovered_scales = cube.recover_fp4_cube(payload)
    assert _flat(recovered_scales) == _flat(scales)


def test_int8_codes_are_reinterpreted_as_bytes_not_converted_numerically():
    """An int8 projection keeps its bytes: the codes are a bit view, never a value."""
    n, k = 32, 128
    codes, scales = _make(n, k)
    signed = codes.view(torch.int8)
    assert signed.dtype == torch.int8
    assert signed.reshape(-1).tolist()[128] == codes.reshape(-1).tolist()[128] - 256

    payload = cube.convert_fp4_cube(signed, scales)
    assert payload.codes.dtype == torch.uint8
    assert payload.source_codes_dtype == "int8"
    recovered, _ = cube.recover_fp4_cube(payload)
    # Same spelling back by default, and the same storage underneath.
    assert recovered.dtype == torch.int8
    assert recovered.view(torch.uint8).reshape(-1).tolist() == codes.reshape(-1).tolist()

    as_unsigned, _ = cube.recover_fp4_cube(payload, codes_dtype="uint8")
    assert as_unsigned.dtype == torch.uint8
    assert _flat(as_unsigned) == _flat(codes)

    explicit_signed, _ = cube.recover_fp4_cube(payload, codes_dtype=torch.int8)
    assert explicit_signed.dtype == torch.int8
    assert explicit_signed.view(torch.uint8).reshape(-1).tolist() == codes.reshape(-1).tolist()


def test_every_payload_tensor_is_the_recorded_dtype_and_contiguous():
    n, k = 64, 256
    codes, scales = _make(n, k)
    payload = cube.convert_fp4_cube(codes, scales)
    for tensor in (payload.codes, payload.scales):
        assert tensor.dtype == torch.uint8
        assert tensor.is_contiguous()
        assert tensor.device.type == "cpu"


def test_the_payload_bytes_add_up_to_the_source_bytes():
    """The budget is asserted from the tensors, not from the constants."""
    n, k = 96, 384
    codes, scales = _make(n, k)
    payload = cube.convert_fp4_cube(codes, scales)
    counted = sum(tensor.numel() * tensor.element_size() for tensor in (payload.codes, payload.scales))
    assert counted == payload.payload_bytes == payload.geometry.payload_bytes
    assert counted == codes.numel() * codes.element_size() + scales.numel() * scales.element_size()
    assert payload.report()["size_exact"] is True
    assert payload.report()["source_codes_dtype"] == "uint8"
    json.dumps(payload.report())  # the report is meant to be serialized


# ------------------------------------------------------------------------- ownership


def test_the_payload_does_not_alias_the_source():
    """Rewriting the source after the conversion must not disturb the payload."""
    n, k = 32, 128
    codes, scales = _make(n, k)
    payload = cube.convert_fp4_cube(codes, scales)
    snapshot_codes, snapshot_scales = _flat(payload.codes), _flat(payload.scales)

    assert not torch._C._overlaps(payload.codes, codes)
    assert not torch._C._overlaps(payload.scales, scales)
    codes.fill_(0x5A)
    scales.fill_(0x5A)
    assert _flat(payload.codes) == snapshot_codes
    assert _flat(payload.scales) == snapshot_scales


def test_the_recovered_tensors_do_not_alias_the_payload():
    """Rewriting the payload after recovery must not disturb what was recovered."""
    n, k = 32, 128
    codes, scales = _make(n, k)
    payload = cube.convert_fp4_cube(codes, scales)
    recovered_codes, recovered_scales = cube.recover_fp4_cube(payload)
    snapshot = _flat(recovered_codes)

    assert not torch._C._overlaps(recovered_codes, payload.codes)
    assert not torch._C._overlaps(recovered_scales, payload.scales)
    payload.codes.fill_(0x5A)
    payload.scales.fill_(0x5A)
    assert _flat(recovered_codes) == snapshot
    assert _flat(recovered_scales) == _flat(scales)


# -------------------------------------------------------------------------- refusals


def test_a_payload_with_a_foreign_format_name_is_refused():
    """A v2 byte order must not be read with v1's formulas."""
    n, k = 32, 128
    codes, scales = _make(n, k)
    payload = cube.convert_fp4_cube(codes, scales)
    with pytest.raises(ValueError, match="refusing to apply v1 formulas"):
        cube.CubePackedFP4(
            codes=payload.codes,
            scales=payload.scales,
            geometry=payload.geometry,
            format="relic.fp4.cube-n32-k128.v2",
        )


def test_a_payload_whose_tensors_do_not_match_its_geometry_is_refused():
    n, k = 32, 128
    codes, scales = _make(n, k)
    payload = cube.convert_fp4_cube(codes, scales)
    wrong_geometry = cube.cube_geometry(n * 2, k)
    with pytest.raises(ValueError):
        cube.CubePackedFP4(codes=payload.codes, scales=payload.scales, geometry=wrong_geometry)
    with pytest.raises(ValueError):
        cube.CubePackedFP4(
            codes=payload.codes.reshape(-1), scales=payload.scales, geometry=payload.geometry
        )
    with pytest.raises(TypeError):
        cube.CubePackedFP4(
            codes=payload.codes.to(torch.int8), scales=payload.scales, geometry=payload.geometry
        )


def test_recovery_refuses_anything_that_is_not_a_payload():
    n, k = 32, 128
    codes, scales = _make(n, k)
    payload = cube.convert_fp4_cube(codes, scales)
    with pytest.raises(TypeError, match="refuses to guess a layout"):
        cube.recover_fp4_cube((payload.codes, payload.scales))
    with pytest.raises(ValueError, match="codes_dtype"):
        cube.recover_fp4_cube(payload, codes_dtype="int16")


@pytest.mark.parametrize(
    ("bad_codes", "expected"),
    [
        ((32, 128, 1), ValueError),  # three dimensions
        ((32, 32), ValueError),  # K/2 that is not a whole number of tiles
    ],
)
def test_bad_code_shapes_are_refused(bad_codes, expected):
    scales = torch.zeros((bad_codes[0], bad_codes[1] // 32 or 1), dtype=torch.uint8)
    codes = torch.zeros(bad_codes, dtype=torch.uint8)
    with pytest.raises(expected):
        cube.convert_fp4_cube(codes, scales)


def test_non_byte_dtypes_and_non_contiguous_inputs_are_refused():
    n, k = 32, 128
    codes, scales = _make(n, k)
    with pytest.raises(TypeError, match="packed uint8/int8 bytes"):
        cube.convert_fp4_cube(codes.to(torch.int16), scales)
    with pytest.raises(TypeError, match="uint8"):
        cube.convert_fp4_cube(codes, scales.to(torch.int8))
    with pytest.raises(ValueError, match="contiguous"):
        cube.convert_fp4_cube(codes.t().contiguous().t(), scales)  # shape is right, layout is not


def test_non_tensor_inputs_are_refused():
    """The entry point takes tensors, not nested lists that would lose the uint8 width.

    A ``requires_grad`` input is not exercised here on purpose: no integer dtype can carry a
    grad, so the check in the module is defensive and cannot be reached from this entry
    point. Asserting a refusal that cannot happen would pin nothing.
    """
    n, k = 32, 128
    codes, scales = _make(n, k)
    with pytest.raises(TypeError, match="torch.Tensor"):
        cube.convert_fp4_cube(codes.tolist(), scales)
    with pytest.raises(TypeError, match="torch.Tensor"):
        cube.convert_fp4_cube(codes, scales.tolist())


def test_a_mismatched_scale_grid_is_refused():
    n, k = 32, 128
    codes, _ = _make(n, k)
    with pytest.raises(ValueError, match="scales must have shape"):
        cube.convert_fp4_cube(codes, torch.zeros((n, k // 32 + 1), dtype=torch.uint8))
    with pytest.raises(ValueError, match="scales must have shape"):
        cube.convert_fp4_cube(codes, torch.zeros((n // 2, k // 32), dtype=torch.uint8))


def test_the_payload_is_always_the_cpu_and_an_explicit_other_device_is_refused():
    """The repack is offline: it never moves bytes onto an accelerator by itself."""
    n, k = 32, 128
    codes, scales = _make(n, k)
    payload = cube.convert_fp4_cube(codes, scales, device="cpu")
    assert payload.codes.device.type == "cpu"
    with pytest.raises(ValueError, match="always materialized on the CPU"):
        cube.convert_fp4_cube(codes, scales, device="npu")
    with pytest.raises(ValueError):
        cube.convert_fp4_cube(codes, scales, device="meta")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device to build the input")
def test_a_cuda_input_is_refused_rather_than_transferred():
    n, k = 32, 128
    codes, scales = _make(n, k)
    with pytest.raises(ValueError, match="implicit GPU transfer"):
        cube.convert_fp4_cube(codes.cuda(), scales.cuda())


@pytest.mark.skipif(not hasattr(torch, "npu") or not torch.npu.is_available(),
                    reason="needs an Ascend NPU to build the input")
def test_an_npu_input_is_serialized_to_the_cpu():
    """An NPU-resident weight may be repacked offline, and the payload still lands on the CPU."""
    n, k = 32, 128
    codes, scales = _make(n, k)
    payload = cube.convert_fp4_cube(codes.npu(), scales.npu())
    assert payload.codes.device.type == "cpu"
    assert payload.scales.device.type == "cpu"
    reference = cube.convert_fp4_cube(codes, scales)
    assert _flat(payload.codes) == _flat(reference.codes)
    assert _flat(payload.scales) == _flat(reference.scales)