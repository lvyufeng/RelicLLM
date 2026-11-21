"""Online unpack of the original checkpoint bytes, checked without a card.

`relicllm/models/deepseek_v4_1/online_unpack.py` exists because the incumbent fill path ends in a
`float8_e4m3fn` cast that torch_npu 2.10 rejects on a 910B3 (`aclnnInplaceCopy`, 561103), and because
the checkpoint already contains the bytes that cast was trying to recover. The tests here are about
the *runtime's* half of that: which grid a stored scale describes, that a plan's cut happens on the
bytes rather than on an expansion, that the parameter ends up with the file's own codes attached, and
that every CPU/CUDA path is untouched.

The oracle is written from the format, not from the implementation: `_e4m3_value`, `_e2m1_value` and
`_e8m0_value` below decode the released encodings by hand in float64, and the expected BF16 output is
that exact product narrowed once. A test that compared the operator against itself would pass on a
swapped nibble order or a scale applied along the wrong axis, which is what these conventions get
wrong.

What is deliberate about the fixtures:

* the dense weight carries codes across the whole E4M3 range that are *finite*, because the fill
  comparisons are exact (`torch.equal` on bf16 bit patterns) and a NaN fixture would make two
  correct implementations disagree with themselves;
* one packed-fp4 expert is 8 x 16 bytes, i.e. K = 32, so its grid is one scale per row and block;
  a second one is 8 x 32 bytes (K = 64) on the same 32-wide block, and its grid is deliberately one
  column short so the adapter's length check is the thing that refuses it;
* the row-scaled table is the Engram layout (`[rows, K]` codes, `[rows, K / 32]` scales) because
  reading it as the dense square grid is the mistake that produces plausible wrong numbers.

No test here runs a model, a GEMM or a card: the NPU cases are opt-in and skipped unless the host has
a working torch_npu, and even those compare decodes against each other rather than claiming parity.
"""

from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from relic_core.kernels import unpack as core_unpack

from relicllm.models.deepseek_v4_1 import mma
from relicllm.models.deepseek_v4_1 import online_unpack
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint, checkpoint_weights, scale_key
from relicllm.models.deepseek_v4_1.online_unpack import (
    BLOCK,
    Fp4ExpertInt8,
    StagedFp8Weight,
    decode_dense_fp8_weight,
    fp4_expert_int8,
    infer_fp8_block_shape,
    is_online_unpack_parameter,
    stage_dense_fp8_weight,
)
from relicllm.models.deepseek_v4_1.tp import ShardPlan

E4M3 = torch.float8_e4m3fn
E8M0 = torch.float8_e8m0fnu

# One name from each layout the release has, in the file's own spelling. `attn.wq_b.weight` is in
# `tp._SPLITS` as a rows cut, which is what makes it usable for the plan test below.
DENSE_KEY = "layers.0.attn.wq_b.weight"
DENSE_SCALE = scale_key(DENSE_KEY)
ENGRAM_KEY = "layers.0.engram.embed.weight"
ENGRAM_SCALE = scale_key(ENGRAM_KEY)
EXPERT_KEY = "layers.0.ffn.experts.0.w1.weight"
EXPERT_SCALE = scale_key(EXPERT_KEY)
SHORT_EXPERT_KEY = "layers.0.ffn.experts.1.w1.weight"
SHORT_EXPERT_SCALE = scale_key(SHORT_EXPERT_KEY)
PLAIN_KEY = "layers.0.ffn.norm.weight"


# -- the format, decoded by hand ---------------------------------------------------------------


def _e4m3_value(byte: int) -> float:
    """One E4M3FN code as its exact value: bias 7, three mantissa bits, no infinities."""
    exponent, mantissa = (byte >> 3) & 15, byte & 7
    if exponent == 15 and mantissa == 7:
        return math.copysign(float("nan"), -1.0 if byte & 128 else 1.0)
    if exponent == 0:
        magnitude = math.ldexp(float(mantissa), -9)
    else:
        magnitude = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return math.copysign(magnitude, -1.0 if byte & 128 else 1.0)


def _e8m0_value(byte: int) -> float:
    """One E8M0 byte as its exact power of two; every byte but 255 is finite."""
    return float("nan") if byte == 255 else math.ldexp(1.0, byte - 127)


def _e2m1_value(code: int) -> float:
    """One E2M1 nibble as its exact value, sign bit included (so code 8 is negative zero)."""
    magnitudes = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    magnitude = magnitudes[code & 7]
    return -magnitude if code & 8 else magnitude


def _as_bytes(tensor: torch.Tensor) -> torch.Tensor:
    """The stored encoding as bytes, whatever floating container the checkpoint reads it through."""
    return tensor if tensor.dtype == torch.uint8 else tensor.view(torch.uint8)


def _bf16_reference(codes: torch.Tensor, scale: torch.Tensor, block_shape: tuple[int, int]) -> torch.Tensor:
    """The expected weight: exact fp64 product, narrowed once through fp32 into bf16."""
    codes, scale = _as_bytes(codes), _as_bytes(scale)
    block_rows, block_columns = block_shape
    values = torch.tensor(
        [[_e4m3_value(int(byte)) for byte in row] for row in codes.tolist()], dtype=torch.float64
    )
    scales = torch.tensor(
        [[_e8m0_value(int(byte)) for byte in row] for row in scale.tolist()], dtype=torch.float64
    )
    expanded = scales.repeat_interleave(block_rows, dim=0).repeat_interleave(block_columns, dim=1)
    return (values * expanded[: values.shape[0], : values.shape[1]]).to(torch.float32).to(torch.bfloat16)


def _fp4_reference(packed: torch.Tensor, scale: torch.Tensor, block: int = BLOCK) -> torch.Tensor:
    """The original floating weight a packed E2M1 matrix and its half-grid encode, in fp32.

    Negative zero is reproduced here (it survives float64); the INT8 representation cannot carry its
    sign, and the tests below assert that difference rather than hiding it behind a tolerance.
    """
    packed, scale = _as_bytes(packed), _as_bytes(scale)
    rows, half = packed.shape
    values = torch.zeros(rows, half * 2, dtype=torch.float64)
    for row in range(rows):
        for column in range(half):
            byte = int(packed[row, column]) & 0xFF
            values[row, 2 * column] = _e2m1_value(byte & 0x0F)
            values[row, 2 * column + 1] = _e2m1_value((byte >> 4) & 0x0F)
    scales = torch.tensor(
        [[_e8m0_value(int(byte)) for byte in row] for row in scale.tolist()], dtype=torch.float64
    )
    expanded = scales.repeat_interleave(block, dim=1)[:, : values.shape[1]]
    return (values * expanded).to(torch.float32)


def _assert_bf16_equal(got: torch.Tensor, want: torch.Tensor) -> None:
    """Bit equality over the finite entries, and NaN where the reference is NaN."""
    assert got.dtype == torch.bfloat16 and tuple(got.shape) == tuple(want.shape)
    assert torch.equal(torch.isnan(got), torch.isnan(want))
    finite = ~torch.isnan(want)
    assert torch.equal(got[finite], want[finite]), "decoded BF16 bit patterns differ"


# -- fixtures written to disk, in the layouts the release uses ---------------------------------

# Every finite E4M3 pattern class: subnormal, the smallest normal, the widest significands, both
# zeros, the largest finite value and their negatives. No byte 0x7F/0xFF: those are NaNs, and the
# exact-comparison fixtures must not contain a value that is unequal to itself.
_DENSE_CODE_PATTERN = (0x00, 0x80, 0x01, 0x08, 0x38, 0x3C, 0x3D, 0x41, 0x70, 0x7E, 0xB8, 0xC1)


def _e4m3_bytes(rows: int, columns: int) -> torch.Tensor:
    pattern = _DENSE_CODE_PATTERN
    return torch.tensor(
        [[pattern[(row * columns + column) % len(pattern)] for column in range(columns)]
         for row in range(rows)],
        dtype=torch.uint8,
    )


def _scale_bytes(exponents) -> torch.Tensor:
    """E8M0 bytes from exponents, keeping the shape the caller wrote."""
    return torch.tensor(exponents, dtype=torch.uint8)


def _packed_fp4_bytes(rows: int, packed_columns: int) -> torch.Tensor:
    """Packed E2M1 nibbles cycling through all sixteen codes in *both* positions.

    The two nibbles are not independent samples: walking one 0..15 cycle across the byte positions
    puts the even codes in the low nibble and the odd ones in the high nibble, so code 8 -- E2M1's
    negative zero -- never appears in the high half and a test of "the sign is lost" would silently
    cover only one position. The high nibble is therefore the low one rotated by eight, which gives
    each position the whole domain, negative zero included.
    """
    low = (torch.arange(rows * packed_columns).reshape(rows, packed_columns) % 16).to(torch.uint8)
    high = ((low.to(torch.int64) + 8) % 16).to(torch.uint8)
    return (low | (high << 4)).to(torch.uint8).contiguous()


def _dense_scale_fp32(ckpt) -> torch.Tensor:
    """The dense fixture's stored grid as the exact FP32 powers of two the metadata carries."""
    stored = ckpt.packed(DENSE_SCALE).view(torch.uint8)
    return torch.tensor(
        [[_e8m0_value(int(byte)) for byte in row] for row in stored.tolist()], dtype=torch.float32
    )


def _write_checkpoint(root: str) -> None:
    from safetensors.torch import save_file

    tensors = {
        DENSE_KEY: _e4m3_bytes(128, 64).view(E4M3),
        # Exponents stay low enough that `largest code x scale` is finite in fp32 (448 x 2**117 <
        # 2**128): an overflowing fixture would make the incumbent path inf and its code recovery
        # undefined, so the two paths could not be compared exactly.
        DENSE_SCALE: _scale_bytes([[117, 116], [115, 114], [113, 112], [111, 110]]).view(E8M0),
        ENGRAM_KEY: _e4m3_bytes(40, 64).view(E4M3),
        ENGRAM_SCALE: _scale_bytes(
            [[108 + (index % 10), 113 - (index % 7)] for index in range(40)]
        ).view(E8M0),
        EXPERT_KEY: _packed_fp4_bytes(8, 16).view(torch.int8),
        EXPERT_SCALE: _scale_bytes([[127], [126], [128], [125], [129], [124], [127], [126]]).view(E8M0),
        SHORT_EXPERT_KEY: _packed_fp4_bytes(8, 32).view(torch.int8),
        SHORT_EXPERT_SCALE: _scale_bytes([[127]] * 8).view(E8M0),
        PLAIN_KEY: torch.arange(64, dtype=torch.float32).to(torch.bfloat16),
    }
    save_file(tensors, f"{root}/model.safetensors", metadata={"format": "pt"})
    with open(f"{root}/model.safetensors.index.json", "w") as handle:
        json.dump({"metadata": {}, "weight_map": {name: "model.safetensors" for name in tensors}}, handle)


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    root = str(tmp_path_factory.mktemp("v41-online-unpack"))
    _write_checkpoint(root)
    checkpoint = V41Checkpoint(root)
    yield checkpoint
    checkpoint.close()


class _Tree(nn.Module):
    """The smallest module whose parameter names are checkpoint keys.

    `checkpoint_weights` walks `named_parameters`, so a name only has to round-trip: the path before
    the last dot becomes nested `nn.Module`s and the leaf is the `weight` parameter itself.
    """

    def __init__(self, entries: dict[str, tuple[tuple[int, ...], torch.dtype, str]]) -> None:
        super().__init__()
        for name, (shape, dtype, device) in entries.items():
            parent_path = name.rpartition(".")[0]
            parent = self
            for part in parent_path.split("."):
                child = getattr(parent, part, None)
                if child is None:
                    child = nn.Module()
                    parent.add_module(part, child)
                parent = child
            parent.weight = nn.Parameter(torch.zeros(shape, dtype=dtype, device=device))


def _dense_tree(shape=(128, 64), dtype=torch.bfloat16, device="cpu") -> _Tree:
    """One dense parameter, defaulting to the checkpoint's own shape: without a plan nothing is cut."""
    return _Tree({DENSE_KEY: (shape, dtype, device)})


def _weight(tree: nn.Module) -> torch.Tensor:
    """The single parameter `_dense_tree` registered, by its full name.

    `_Tree` nests its modules to reproduce the checkpoint's dotted key, so the root has no `weight`
    attribute -- the parameter lives at `layers.0.attn.wq_b.weight`, which is exactly the name the
    fill walked to fill it.
    """
    parameters = dict(tree.named_parameters())
    assert tuple(parameters) == (DENSE_KEY,), tuple(parameters)
    return parameters[DENSE_KEY]


def _plan() -> ShardPlan:
    """A real rows cut: `attn.wq_b.weight` is `heads * head_dim` rows, two ranks of 64."""
    config = SimpleNamespace(
        n_heads=4, o_groups=2, index_n_heads=2, moe_inter_dim=64,
        head_dim=32, o_lora_rank=16, index_head_dim=32,
    )
    return ShardPlan.build(
        config, rank=1, world=2, reduce=lambda value: value, gather=lambda value: value
    )


# -- the two fill paths, on the same bytes -----------------------------------------------------


def test_online_fill_matches_the_incumbent_fill_bit_for_bit(ckpt) -> None:
    """BF16 is where the two paths have to agree, and they have to agree exactly.

    The incumbent expands on the host in fp32 and lets `copy_` narrow; the online path decodes on the
    device and narrows inside the operator. The product of a four-bit significand and a power of two
    is exact in both widths, so the only rounding is the final one and the bit patterns are the same.
    """
    incumbent, online = _dense_tree(), _dense_tree()
    checkpoint_weights(incumbent, ckpt)
    checkpoint_weights(online, ckpt, online_unpack=True)

    reference = _bf16_reference(
        ckpt.packed(DENSE_KEY).view(torch.uint8), ckpt.packed(DENSE_SCALE).view(torch.uint8), (BLOCK, BLOCK)
    )
    _assert_bf16_equal(_weight(incumbent), reference)
    _assert_bf16_equal(_weight(online), reference)
    assert torch.equal(_weight(online), _weight(incumbent))


def test_the_online_fill_attaches_the_file_codes_and_the_file_grid(ckpt) -> None:
    """The metadata the GEMM reads is the checkpoint's, not a recovery of it.

    On this host `codes_from_scale` also succeeds, so the two spellings can be compared: the file's
    codes and the recovered ones are the same tensor here. What the online path adds is that it never
    needs the cast that fails on NPU -- and that its octave is the file's by construction.

    The grid arrives as FP32 powers of two rather than as the stored container, which is the safe
    form on a device with no float8 cast; `mma.e8m0_exponent` is the reader the GEMM uses, so running
    it here is what says the metadata is not merely attached but *readable*.
    """
    incumbent, online = _dense_tree(), _dense_tree()
    checkpoint_weights(incumbent, ckpt)
    checkpoint_weights(online, ckpt, online_unpack=True)

    stored_codes = ckpt.packed(DENSE_KEY).view(torch.uint8)
    stored_scale = ckpt.packed(DENSE_SCALE).view(torch.uint8)
    expected_scale = _dense_scale_fp32(ckpt)
    for tree in (incumbent, online):
        parameter = _weight(tree)
        # Read the metadata off the parameter object itself. `attach_scale`/`attach_codes` are
        # `setattr` on that object, so a fresh `.detach()` -- a new wrapper sharing storage but not
        # attributes -- would report `None` for metadata that is present and in use.
        codes = mma.attached_codes(parameter)
        assert codes is not None and codes.dtype == E4M3 and codes.device == parameter.device
        assert torch.equal(codes.view(torch.uint8), stored_codes)
    assert torch.equal(
        mma.attached_codes(_weight(online)), mma.attached_codes(_weight(incumbent))
    )
    # The incumbent attaches the stored container; the online path attaches the same grid as FP32.
    assert mma.attached_scale(_weight(incumbent)).dtype == E8M0
    online_scale = mma.attached_scale(_weight(online))
    assert online_scale.dtype == torch.float32 and online_scale.device == _weight(online).device
    assert torch.equal(online_scale, expected_scale)
    exponents = (stored_scale.reshape(-1).to(torch.int64) - 127).reshape(stored_scale.shape)
    assert torch.equal(mma.e8m0_exponent(online_scale), exponents)


def test_the_online_fill_never_recovers_codes_from_the_scale(ckpt, monkeypatch) -> None:
    """The metadata must be the file's bytes, not a quotient cast back to `float8_e4m3fn`.

    `codes_from_scale` is the recovery, and its last step is the float8 cast that torch_npu answers
    with 561103. Pinning it to fail proves the online fill attaches what it staged and does not reach
    for the recovery at all -- which is the whole reason this path exists.
    """
    def _refuse(*args, **kwargs):
        raise AssertionError("the online fill recovered codes from the weight instead of staging them")

    monkeypatch.setattr(mma, "codes_from_scale", _refuse)
    tree = _dense_tree()
    report = checkpoint_weights(tree, ckpt, online_unpack=True)
    assert DENSE_KEY in report.loaded and DENSE_KEY in report.quantized
    codes = mma.attached_codes(_weight(tree))
    assert codes is not None
    assert torch.equal(codes.view(torch.uint8), ckpt.packed(DENSE_KEY).view(torch.uint8))


def test_auto_does_not_stage_a_cpu_parameter(ckpt, monkeypatch) -> None:
    """`online_unpack=None` is where it is *required*, and a CPU bf16 parameter does not require it.

    Pinned by making the staging entry point fail: the default fill must not reach it. This is the
    guarantee that the CUDA and CPU configurations keep the code path they were measured on.
    """
    def _refuse(*args, **kwargs):
        raise AssertionError("staging ran for a CPU parameter under the automatic selection")

    monkeypatch.setattr(online_unpack, "stage_dense_fp8_weight", _refuse)
    tree = _dense_tree()
    report = checkpoint_weights(tree, ckpt)
    assert DENSE_KEY in report.loaded and DENSE_KEY in report.quantized


def test_the_plan_cuts_the_raw_bytes_not_an_expansion(ckpt) -> None:
    """The TP cut lands on the stored codes and the stored grid, and the decode happens after.

    A rank owns whole 32-row bands of a rows-cut weight, so cutting the bytes is the same partition
    as cutting the dense tensor -- which is the property that lets the link carry one byte per
    element. The assertion is on identities: the cut codes are the file's rows 64..128, and the
    decoded weight is what those rows decode to, not what the whole tensor decoded to and then split.
    """
    plan = _plan()
    staged = stage_dense_fp8_weight(ckpt, DENSE_KEY, plan=plan)
    assert isinstance(staged, StagedFp8Weight)
    assert tuple(staged.decoded.shape) == (64, 64)
    assert staged.block_shape == (BLOCK, BLOCK)

    stored_codes = ckpt.packed(DENSE_KEY).view(torch.uint8)[64:128]
    stored_scale = ckpt.packed(DENSE_SCALE).view(torch.uint8)[2:4]
    assert torch.equal(staged.codes.view(torch.uint8), stored_codes)
    assert torch.equal(
        staged.scale,
        torch.tensor(
            [[_e8m0_value(int(byte)) for byte in row] for row in stored_scale.tolist()],
            dtype=torch.float32,
        ),
    )
    _assert_bf16_equal(staged.decoded, _bf16_reference(stored_codes, stored_scale, (BLOCK, BLOCK)))

    incumbent, online = _dense_tree((64, 64)), _dense_tree((64, 64))
    checkpoint_weights(incumbent, ckpt, plan=plan)
    checkpoint_weights(online, ckpt, plan=plan, online_unpack=True)
    assert torch.equal(_weight(online), _weight(incumbent))


def test_a_forced_online_fill_refuses_an_fp32_parameter(ckpt) -> None:
    """A BF16 intermediate is a different contract from an exact fp32 expansion, so it is refused."""
    tree = _dense_tree((128, 64), dtype=torch.float32)
    with pytest.raises(ValueError, match="bfloat16"):
        checkpoint_weights(tree, ckpt, online_unpack=True)
    parameter = _weight(tree)
    assert torch.equal(parameter, torch.zeros_like(parameter)), "the refusal must not half-fill"


def test_a_packed_fp4_parameter_is_refused_by_the_dense_fill_path(ckpt) -> None:
    """A routed expert is a store, not a filled parameter; naming one here is a tree that is wrong."""
    tree = _Tree({EXPERT_KEY: ((8, 32), torch.bfloat16, "cpu")})
    with pytest.raises(ValueError, match="packed fp4"):
        checkpoint_weights(tree, ckpt, online_unpack=True)


# -- which grid the file wrote -----------------------------------------------------------------


def test_the_block_shape_follows_the_stored_grid() -> None:
    """Square 32x32 for dense projections, one row per output row for tables and experts."""
    assert infer_fp8_block_shape((64, 64), (2, 2)) == (BLOCK, BLOCK)
    assert infer_fp8_block_shape((40, 64), (40, 2)) == (1, BLOCK)
    assert infer_fp8_block_shape((128, 5120), (4, 160)) == (BLOCK, BLOCK)
    # A short last row block is the operator's `ceil(rows / block_cols)` contract. It is *not* a
    # shape the release ships -- every dense grid is `[out // 32, in // 32]` -- it is here so the
    # resolver matches the operator it feeds rather than being narrower than it.
    assert infer_fp8_block_shape((48, 64), (2, 2)) == (BLOCK, BLOCK)
    # Leading batch axes have to agree exactly and pass through.
    assert infer_fp8_block_shape((3, 64, 64), (3, 2, 2)) == (BLOCK, BLOCK)
    # A single output row (M = 1) is the shape the two layouts cannot tell apart: one 32-row block
    # and one row-per-row band both index scale row 0, so it is canonicalised to the square grid
    # rather than refused. The 32-row table below is *not* ambiguous -- its stored grid has one row
    # per row, and a square grid for 32 rows would have a single row of its own.
    assert infer_fp8_block_shape((1, 64), (1, 2)) == (BLOCK, BLOCK)
    assert infer_fp8_block_shape((BLOCK, 64), (1, 2)) == (BLOCK, BLOCK)
    assert infer_fp8_block_shape((BLOCK, 64), (BLOCK, 2)) == (1, BLOCK)


@pytest.mark.parametrize(
    "weight_shape, scale_shape, message",
    [
        ((64, 64), (3, 2), "neither a row-per-row grid nor a square"),
        ((64, 64), (2, 3), "does not divide"),
        ((2, 64, 64), (3, 2, 2), "leading axes"),
        ((64,), (2,), "rank"),
        # A degenerate grid is refused rather than accepted, which is why the doc does not claim one.
        ((0, 64), (0, 2), "empty axis"),
        ((64, 0), (2, 0), "empty axis"),
    ],
)
def test_a_grid_that_does_not_describe_the_weight_is_refused(weight_shape, scale_shape, message) -> None:
    """Guessing here means decoding an octave that belongs to another tensor."""
    with pytest.raises(ValueError, match=message):
        infer_fp8_block_shape(weight_shape, scale_shape)


def test_a_row_scaled_table_is_decoded_with_one_scale_per_row(ckpt) -> None:
    """The Engram layout, decoded the way `dequantize_rows` forms it: fp32 product, then bf16."""
    staged = stage_dense_fp8_weight(ckpt, ENGRAM_KEY)
    codes = ckpt.packed(ENGRAM_KEY).view(torch.uint8)
    scale = ckpt.packed(ENGRAM_SCALE).view(torch.uint8)
    assert staged.block_shape == (1, BLOCK)
    _assert_bf16_equal(staged.decoded, _bf16_reference(codes, scale, (1, BLOCK)))

    # The two layouts are not interchangeable: declaring either one against the other's stored grid
    # is refused by the operator rather than decoded into a plausible wrong matrix.
    dense_codes = ckpt.packed(DENSE_KEY).view(E4M3)
    dense_scale = ckpt.packed(DENSE_SCALE).view(E8M0)
    with pytest.raises(ValueError):
        decode_dense_fp8_weight(dense_codes, dense_scale, block_shape=(1, BLOCK))
    with pytest.raises(ValueError):
        decode_dense_fp8_weight(codes.view(E4M3), scale.view(E8M0), block_shape=(BLOCK, BLOCK))
    # And an all-NaN scale byte cannot describe an octave, so staging refuses it before attaching.
    with pytest.raises(ValueError, match="cannot describe a weight's octave"):
        online_unpack._e8m0_grid_fp32(torch.tensor([[255]], dtype=torch.uint8))


# -- the packed-fp4 representation ---------------------------------------------------------------


def test_the_fp4_adapter_widens_exactly_and_halves_the_original_grid(ckpt) -> None:
    """Every nibble becomes twice its value in INT8, and the original grid keeps its layout."""
    result = fp4_expert_int8(ckpt, EXPERT_KEY, EXPERT_SCALE)
    assert isinstance(result, Fp4ExpertInt8)
    packed = ckpt.packed(EXPERT_KEY)
    scale = ckpt.packed(EXPERT_SCALE).view(torch.uint8)

    expected_values = torch.tensor(
        [
            [
                value
                for byte in (int(packed[row, column]) & 0xFF for column in range(packed.shape[1]))
                for value in (
                    int(round(_e2m1_value(byte & 0x0F) * 2)),
                    int(round(_e2m1_value((byte >> 4) & 0x0F) * 2)),
                )
            ]
            for row in range(packed.shape[0])
        ],
        dtype=torch.int8,
    )
    assert result.values.dtype == torch.int8
    assert tuple(result.values.shape) == (8, 2 * packed.shape[1])
    assert torch.equal(result.values, expected_values)
    assert result.half_scale.dtype == torch.float32
    assert torch.equal(
        result.half_scale, torch.tensor([[2.0 ** (int(byte) - 128)] for byte in scale.reshape(-1)], dtype=torch.float32)
    )
    assert torch.equal(result.to_float32(), _fp4_reference(packed, scale))
    # The documented loss: integer zero has one encoding, so the sign of an E2M1 negative zero does
    # not survive the widening. Both zero codes land on the same INT8 value.
    bytes_view = packed.view(torch.uint8)
    low_zero = (bytes_view & 0x0F) == 8
    high_zero = ((bytes_view >> 4) & 0x0F) == 8
    assert low_zero.any() and high_zero.any(), "the fixture must exercise code 8 in both nibbles"
    # Interleaved by column, not by row: column 2*c is the low nibble and 2*c+1 the high one.
    assert torch.equal(result.values[..., 0::2][low_zero], torch.zeros_like(result.values[..., 0::2][low_zero]))
    assert torch.equal(result.values[..., 1::2][high_zero], torch.zeros_like(result.values[..., 1::2][high_zero]))


def test_the_fp4_adapter_refuses_the_wrong_stored_kind_or_grid(ckpt) -> None:
    """Three separate contracts, and each has to fail loudly rather than decode something."""
    with pytest.raises(ValueError, match="not stored as packed fp4"):
        fp4_expert_int8(ckpt, DENSE_KEY, DENSE_SCALE)
    with pytest.raises(ValueError, match="one original scale per output row"):
        fp4_expert_int8(ckpt, EXPERT_KEY, DENSE_SCALE)
    # K = 64 here but the grid keeps ONE column, so the derived block would be 64 rather than 32.
    with pytest.raises(ValueError, match="released routed experts use 32"):
        fp4_expert_int8(ckpt, SHORT_EXPERT_KEY, SHORT_EXPERT_SCALE)


# -- the operators underneath, over their whole domain -----------------------------------------


def test_the_operators_cover_the_whole_code_domain() -> None:
    """All 256 E4M3 codes and all 256 packed bytes, against the hand decode.

    This is the contract the runtime calls: if the operator's domain is narrower than the format's,
    the failure would appear as one wrong weight rather than as an error.
    """
    codes = torch.arange(256, dtype=torch.uint8)
    decoded = core_unpack.unpack_fp8_to_bf16(codes.view(E4M3))
    expected = torch.tensor([_e4m3_value(byte) for byte in range(256)], dtype=torch.float64)
    _assert_bf16_equal(decoded, expected.to(torch.float32).to(torch.bfloat16))

    # Scaled, on the per-row grid, so a scale error cannot hide behind a single-exponent fixture.
    scale_bytes = torch.tensor([[0], [1], [126], [127], [128], [254], [255]], dtype=torch.uint8)
    scaled = core_unpack.unpack_fp8_to_bf16(
        codes.view(E4M3).reshape(256, 1).expand(256, len(scale_bytes)).contiguous(),
        scale_bytes.view(E8M0).reshape(1, -1).expand(256, len(scale_bytes)).contiguous(),
        block_shape=(1, 1),
    )
    expected_scaled = torch.tensor(
        [[_e4m3_value(byte) * _e8m0_value(int(scale)) for scale in scale_bytes.reshape(-1)]
         for byte in range(256)],
        dtype=torch.float64,
    )
    _assert_bf16_equal(scaled, expected_scaled.to(torch.float32).to(torch.bfloat16))

    packed = torch.arange(256, dtype=torch.uint8)
    widened = core_unpack.unpack_fp4_to_int8(packed)
    expected_widened = torch.tensor(
        [
            value
            for byte in range(256)
            for value in (int(_e2m1_value(byte & 0x0F) * 2), int(_e2m1_value((byte >> 4) & 0x0F) * 2))
        ],
        dtype=torch.int8,
    )
    assert torch.equal(widened, expected_widened)

    scale_codes = torch.arange(256, dtype=torch.uint8).view(E8M0)
    halved = core_unpack.fp4_int8_scale(scale_codes)
    expected_halved = torch.tensor(
        [_e8m0_value(byte) / 2.0 for byte in range(256)], dtype=torch.float32
    )
    assert torch.equal(torch.isnan(halved), torch.isnan(expected_halved))
    finite = ~torch.isnan(expected_halved)
    assert torch.equal(halved[finite], expected_halved[finite])


# -- the metadata helper's own contract --------------------------------------------------------


def test_attaching_source_codes_writes_the_gemm_metadata_and_validates_its_arguments() -> None:
    """The helper is plumbing, but it is the plumbing the reduction rule reads: it must be exact."""
    weight = torch.zeros(64, 64, dtype=torch.bfloat16)
    codes = torch.arange(64 * 64, dtype=torch.uint8).reshape(64, 64).view(E4M3)
    scale = torch.zeros(2, 2, dtype=torch.uint8).view(E8M0)
    mma.attach_source_codes(weight, codes, scale)
    assert torch.equal(mma.attached_codes(weight).view(torch.uint8), codes.view(torch.uint8))
    assert torch.equal(mma.attached_scale(weight).view(torch.uint8), scale.view(torch.uint8))

    with pytest.raises(TypeError, match="float8_e4m3fn"):
        mma.attach_source_codes(weight, codes.view(torch.uint8), scale)
    with pytest.raises(ValueError, match="source codes are"):
        mma.attach_source_codes(weight, codes[:32], scale)
    with pytest.raises(ValueError, match="contiguous"):
        mma.attach_source_codes(weight, codes.t(), scale)


# -- the selection gate ------------------------------------------------------------------------


def test_the_gate_selects_only_a_bf16_parameter(ckpt) -> None:
    """Without an NPU the answer is no, and it is no for the dtype reason too."""
    assert is_online_unpack_parameter(torch.zeros(4, 4, dtype=torch.bfloat16)) is False
    assert is_online_unpack_parameter(torch.zeros(4, 4, dtype=torch.float32)) is False
    # A plain (unquantized) parameter is never decoded from bytes, whatever the gate says.
    report_tree = _Tree({PLAIN_KEY: ((64,), torch.bfloat16, "cpu")})
    report = checkpoint_weights(report_tree, ckpt, online_unpack=True)
    assert PLAIN_KEY in report.loaded and PLAIN_KEY not in report.quantized


# -- the card, when there is one ----------------------------------------------------------------


def _npu_ready() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    try:
        return bool(torch.npu.is_available()) and torch.npu.device_count() > 0
    except Exception:
        return False


requires_npu = pytest.mark.skipif(not _npu_ready(), reason="requires a working torch_npu stack")


@requires_npu
def test_the_npu_decode_is_bit_identical_to_the_cpu_decode(ckpt) -> None:
    """The decode is specified by its bytes, so the device must not move a bit.

    The operator gathers pre-rounded BF16 patterns from a host-built table precisely so that a
    device's FP32 subnormal mode cannot change the answer; this is the test that says that held.
    """
    cpu = stage_dense_fp8_weight(ckpt, DENSE_KEY)
    npu = stage_dense_fp8_weight(ckpt, DENSE_KEY, device="npu:0")
    assert npu.decoded.device.type == "npu" and npu.codes.device.type == "npu"
    assert torch.equal(npu.decoded.cpu(), cpu.decoded)
    assert torch.equal(npu.codes.cpu().view(torch.uint8), cpu.codes.view(torch.uint8))

    table_cpu = stage_dense_fp8_weight(ckpt, ENGRAM_KEY)
    table_npu = stage_dense_fp8_weight(ckpt, ENGRAM_KEY, device="npu:0")
    assert torch.equal(table_npu.decoded.cpu(), table_cpu.decoded)


@requires_npu
def test_the_npu_fill_is_selected_automatically_and_matches_the_cpu_fill(ckpt) -> None:
    """The real integration: a bf16 NPU parameter takes the staging path with no flag at all.

    This is the load that previously failed at `mma.codes_from_scale`; it has to produce the same
    values the CPU path does and leave the file's codes and grid attached on the card.
    """
    cpu_tree = _dense_tree()
    checkpoint_weights(cpu_tree, ckpt)
    npu_tree = _dense_tree(device="npu:0")
    assert is_online_unpack_parameter(_weight(npu_tree)) is True
    report = checkpoint_weights(npu_tree, ckpt)
    assert DENSE_KEY in report.quantized
    assert torch.equal(_weight(npu_tree).cpu(), _weight(cpu_tree))
    codes = mma.attached_codes(_weight(npu_tree))
    scale = mma.attached_scale(_weight(npu_tree))
    assert codes is not None and scale is not None, "the auto fill attached no metadata pair"
    assert codes.device.type == "npu" and scale.device.type == "npu"
    # The pair on the card is the file's, not a recovery: the codes are the stored bytes and the grid
    # is the exact FP32 decode of the stored exponents.
    assert torch.equal(codes.cpu().view(torch.uint8), ckpt.packed(DENSE_KEY).view(torch.uint8))
    assert scale.dtype == torch.float32
    assert torch.equal(scale.cpu(), _dense_scale_fp32(ckpt))


@requires_npu
def test_the_fp4_low_scale_edge_is_reported_rather_than_assumed() -> None:
    """E8M0 byte 0 halves to 2**-128, so a small INT8 value times it is subnormal in FP32.

    The representation is exact and the operator preserves the stored scale bits -- it decodes them
    through a host table rather than by casting on the device. What is *not* guaranteed is the
    multiply on hardware that runs FP32 with flush-to-zero: there the product is 0 where the CPU
    returns the subnormal. That is the device's arithmetic mode, not this representation, so the
    test asserts the bits it must assert and reports the flush instead of pretending either way.
    """
    scale_bytes = torch.zeros(2, 1, dtype=torch.uint8)
    cpu_scale = core_unpack.fp4_int8_scale(scale_bytes.view(E8M0))
    npu_scale = core_unpack.fp4_int8_scale(scale_bytes.view(E8M0).to("npu:0"))
    assert torch.equal(npu_scale.cpu(), cpu_scale), "the stored scale bits must survive the transfer"
    assert torch.equal(cpu_scale, torch.tensor([[2.0 ** -128], [2.0 ** -128]], dtype=torch.float32))

    values = torch.tensor(
        [[1, -1, 12, -12] + [0] * 28, [1, -1, 12, -12] + [0] * 28], dtype=torch.int8
    )
    cpu = Fp4ExpertInt8(values=values, half_scale=cpu_scale, block=BLOCK).to_float32()
    npu = Fp4ExpertInt8(
        values=values.to("npu:0"), half_scale=npu_scale, block=BLOCK
    ).to_float32().cpu()
    flushed = (cpu != 0) & (npu == 0)
    if flushed.any():
        pytest.skip(
            "this device flushes FP32 subnormal products to zero, so to_float32 is exact only "
            "where the product is representable without flush-to-zero"
        )
    assert torch.equal(npu, cpu)


@requires_npu
def test_the_npu_fp4_representation_matches_the_cpu_representation(ckpt) -> None:
    """The widened INT8 values and their half-scales are a function of the bytes, not of the card."""
    cpu = fp4_expert_int8(ckpt, EXPERT_KEY, EXPERT_SCALE)
    npu = fp4_expert_int8(ckpt, EXPERT_KEY, EXPERT_SCALE, device="npu:0")
    assert npu.values.device.type == "npu"
    assert torch.equal(npu.values.cpu(), cpu.values)
    assert torch.equal(npu.half_scale.cpu(), cpu.half_scale)
    assert torch.equal(npu.to_float32().cpu(), cpu.to_float32())
