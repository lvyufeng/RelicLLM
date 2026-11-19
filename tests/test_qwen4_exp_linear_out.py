"""The optional cat-free strict linear must be faithful, guarded, and never silently promoted.

Three properties are tested separately, because they answer different questions:

* **Faithfulness of the accepted geometry.** With the switches on, the helper must issue exactly
  the same one-row `linear` requests the reference loop issues and write each answer into one
  shared allocation. A test builds that reference independently, straight from ATen's `linear.out`,
  and requires bit equality. This holds on CPU and is the property the helper is allowed to claim.
* **The rejected preserved-rank mode is refused, not served.** On 8xRTX 4090 the preserved-rank
  staging (`LINEAR_OUT_2D` unset) failed the bit oracle on 22 of 269 cases -- 20 of the 22
  rank-three real-projection configurations, the adversarial vectors and the captured replay --
  while two-dimensional staging passed every case. The helper therefore returns ``None`` for
  `x.ndim > 2` unless two-dimensional staging is explicitly selected, and these tests assert that
  refusal *and* that no `linear.out` dispatch happens for it. A refused call is a pure fallback;
  it is never compared as if it had been accepted.
* **Why CPU cannot qualify it.** `linear.out` and the composite behind `F.linear` pick different
  accumulation paths at rank three on this build. That is recorded with a direct ATen diagnostic,
  outside the helper gate, and is the reason promotion is a device-oracle decision.

The remaining tests are guards: a default-off switch, the strict-row prerequisite, dtype/device/
shape/row-count/bias/autograd refusals, and an explicit failure when the requested overload is
missing instead of a silent downgrade.

The device tests only instantiate geometries the helper accepts: two-dimensional staging for every
leading shape, preserved rank only where `x.ndim == 2`. Every one of the 22 configurations the
eight-card oracle rejected has a CPU refusal test instead.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn.functional as F

from relicllm.models.qwen4_exp import linear_out
from relicllm.models.qwen4_exp.linear_out import linear_out_enabled, qwen4_exp_linear_out


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------

_ENABLE_VARS = (
    "POCKETLLM_QWEN4_LINEAR_OUT",
    "POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT",
    "POCKETLLM_QWEN4_LINEAR_OUT_2D",
)


def enable(monkeypatch) -> None:
    """Turn on every switch the helper requires for the qualified request path."""
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT_2D", "1")


def enable_without_2d(monkeypatch) -> None:
    """The experiment on, the qualified two-dimensional staging absent (the rejected mode)."""
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    monkeypatch.delenv("POCKETLLM_QWEN4_LINEAR_OUT_2D", raising=False)


def reference_linear(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    """The loop being replaced, verbatim from `row_invariant_linear`.

    Kept as a local copy rather than imported: a change to the model helper must show up here as a
    difference, not silently redefine what "the same one-row call" means.
    """
    rows = x.numel() // x.shape[-1]
    flat = x.reshape(rows, x.shape[-1])
    single_shape = (1,) * (x.ndim - 1) + (x.shape[-1],)
    output = torch.cat([
        F.linear(flat[row:row + 1].reshape(single_shape).contiguous(), weight, bias).reshape(1, -1)
        for row in range(rows)
    ], dim=0)
    return output.reshape(*x.shape[:-1], weight.shape[0])


def manual_out_reference(x: torch.Tensor, weight: torch.Tensor, bias=None, *,
                         stage_2d: bool = True) -> torch.Tensor:
    """The same row loop expressed with `linear.out` and one shared buffer, built independently.

    This is what the helper claims to be. It is written out here so the equality test does not
    simply compare the helper with itself. ``stage_2d=False`` keeps the caller's rank, which is the
    rejected mode; it is used only by the direct diagnostic, never to define an accepted answer.
    """
    rows = x.numel() // x.shape[-1]
    if stage_2d or x.ndim == 2:
        input_single = (1, x.shape[-1])
        output_single = (1, weight.shape[0])
    else:
        input_single = (1,) * (x.ndim - 1) + (x.shape[-1],)
        output_single = (1,) * (x.ndim - 1) + (weight.shape[0],)
    flat = x.reshape(rows, x.shape[-1])
    buffer = torch.empty(rows, weight.shape[0], dtype=x.dtype, device=x.device)
    for row in range(rows):
        staged = flat[row:row + 1].reshape(input_single).contiguous()
        target = buffer[row:row + 1].reshape(output_single)
        torch.ops.aten.linear.out(staged, weight, bias, out=target)
    return buffer.reshape(*x.shape[:-1], weight.shape[0])


def bit_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.dtype is right.dtype and left.shape == right.shape and torch.equal(
        left.view(torch.int16), right.view(torch.int16)
    )


def differing_elements(left: torch.Tensor, right: torch.Tensor) -> int:
    """BF16 bit differences; NaN payloads included, so an equal count is a strong statement."""
    return int((left.view(torch.int16) != right.view(torch.int16)).sum())


def make(shape, in_features, out_features, *, bias=True, seed=0, dtype=torch.bfloat16):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(*shape, in_features, dtype=dtype, generator=generator)
    weight = torch.randn(out_features, in_features, dtype=dtype, generator=generator)
    b = torch.randn(out_features, dtype=dtype, generator=generator) if bias else None
    return x, weight, b


#: The actual leading shapes the strict verifier uses. Rank two, three and four all occur; the
#: rank-three-and-four entries are served only through two-dimensional staging.
LEADING_GEOMETRIES = [
    (2, (2,)),
    (2, (3,)),
    (2, (4,)),
    (2, (5,)),
    (2, (6,)),
    (2, (8,)),
    (3, (1, 4)),
    (3, (2, 4)),
    (4, (1, 2, 2)),
    (4, (1, 1, 6)),
]

#: The leading shapes the helper accepts with preserved rank: only the two-dimensional ones.
RANK_TWO_LEADING = [(2,), (3,), (4,), (5,), (6,), (8,)]

#: The leading shapes that are refused without two-dimensional staging. These are exactly the
#: geometries whose preserved-rank request the eight-card device oracle rejected.
REJECTED_WITHOUT_2D = [(1, 4), (2, 4), (1, 2, 2), (1, 1, 6)]

#: The device shape census for the strict verifier: rows are always four, and these are the actual
#: projection widths. The trailing entries are the hyper-connection mixes and the language-model
#: head, which are the widest and narrowest requests the loop issues.
REAL_SHAPES = [
    (2560, 6),
    (2560, 1280),
    (2560, 768),
    (2560, 256),
    (2560, 640),
    (2560, 1536),
    (2560, 80),
    (2560, 2560),
    (10240, 320),
    (10240, 4),
    (2560, 31040),
]


# ---------------------------------------------------------------------------------------------
# Faithfulness: the helper is exactly the row loop it replaces, expressed with linear.out
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ndim,lead", LEADING_GEOMETRIES)
@pytest.mark.parametrize("in_features,out_features", [(16, 12), (2560, 80), (320, 2560)])
@pytest.mark.parametrize("bias", [True, False])
def test_helper_is_bit_identical_to_the_manual_out_reference(monkeypatch, ndim, lead, in_features,
                                                            out_features, bias):
    enable(monkeypatch)
    x, weight, b = make(lead, in_features, out_features, bias=bias)
    result = qwen4_exp_linear_out(x, weight, b)
    expected = manual_out_reference(x, weight, b, stage_2d=True)
    assert result is not None
    assert bit_equal(result, expected), (
        f"{differing_elements(result, expected)} elements differ from the out-API reference"
    )
    assert result.shape == (*x.shape[:-1], out_features)


def test_the_result_owns_one_contiguous_allocation(monkeypatch):
    enable(monkeypatch)
    x, weight, b = make((1, 4), 2560, 768)
    result = qwen4_exp_linear_out(x, weight, b)
    assert result is not None
    assert result.is_contiguous()
    # One allocation of exactly the output, not a concatenation of per-row results.
    assert result.untyped_storage().size() == result.numel() * result.element_size()


def test_each_row_is_written_at_its_own_offset(monkeypatch):
    """The shared buffer must place row `r` at `r * out_features`, with the row's input."""
    enable(monkeypatch)
    x, weight, b = make((1, 4), 2560, 80)
    result = qwen4_exp_linear_out(x, weight, b)
    assert result is not None
    rows = 4
    flat = result.reshape(rows, 80)
    source = x.reshape(rows, 2560)
    for row in range(rows):
        alone = torch.empty(1, 80, dtype=torch.bfloat16)
        torch.ops.aten.linear.out(source[row:row + 1].contiguous(), weight, b, out=alone)
        assert bit_equal(flat[row:row + 1], alone), (
            f"row {row} is not the out result for its own input"
        )


def test_rank_two_reproduces_the_replaced_reference_on_this_build(monkeypatch):
    """At rank two this build routes both spellings identically, so equality is a real invariant.

    A future torch build that changes this is a signal, not a nuisance: it would mean the
    promotion gate has to be re-established for two-dimensional call sites as well.
    """
    enable_without_2d(monkeypatch)
    for in_features, out_features in REAL_SHAPES:
        x, weight, b = make((4,), in_features, out_features)
        result = qwen4_exp_linear_out(x, weight, b)
        assert result is not None, "rank two must stay accepted without the 2D switch"
        assert bit_equal(result, reference_linear(x, weight, b)), (
            f"{in_features}->{out_features} diverges at rank two on this build"
        )


def test_the_cpu_build_routes_preserved_rank_through_a_different_path(monkeypatch):
    """Direct-ATen diagnostic for the rejected mode, deliberately outside the helper gate.

    This is *not* an accepted helper call: it shows with `linear.out` called directly at the
    caller's rank that the preserved-rank request disagrees with the composite `F.linear` at rank
    three on this build, which is the CPU analogue of the device rejection. The helper's own answer
    for the same shapes is asserted separately, as a refusal.

    If a future build routes the two identically, this assertion fails on purpose: the module
    docstring's CPU claim and the reason CPU cannot qualify the path would both need re-deriving.
    """
    enable_without_2d(monkeypatch)
    diverged = []
    for in_features, out_features in REAL_SHAPES:
        for lead in ((1, 4), (2, 2)):
            x, weight, b = make(lead, in_features, out_features)
            preserved = manual_out_reference(x, weight, b, stage_2d=False)
            two_dimensional = manual_out_reference(x, weight, b, stage_2d=True)
            composite = reference_linear(x, weight, b)
            assert preserved.shape == composite.shape == two_dimensional.shape
            assert torch.isfinite(preserved.float()).all()
            if not bit_equal(preserved, composite):
                diverged.append((lead, in_features, out_features,
                                 differing_elements(preserved, composite), preserved.numel()))
            # The two-dimensional request is the one the device oracle accepted; on this build it is
            # also the one that folds to the same matrix-vector request the composite makes.
            assert bit_equal(two_dimensional, composite), (
                f"two-dimensional staging should reproduce the composite at "
                f"{in_features}->{out_features} lead {lead}"
            )
    assert diverged, (
        "this build routes preserved-rank linear.out through the same accumulation as F.linear; "
        "re-derive the module docstring and the device qualification before trusting the CPU claim"
    )


def test_noncontiguous_input_is_staged_exactly_like_the_reference(monkeypatch):
    """The reference `.reshape().contiguous()` per-row staging is what keeps gapped views sane.

    The helper must apply the same staging, so a gapped input reaches the kernel as the same bytes
    the reference would have produced.
    """
    enable(monkeypatch)
    base, _unused, _b = make((1, 4), 512, 80)
    gapped = base[..., ::2]
    weight = torch.randn(80, 256, dtype=torch.bfloat16)
    b = torch.randn(80, dtype=torch.bfloat16)
    assert not gapped.is_contiguous() and gapped.shape[-1] == 256
    result = qwen4_exp_linear_out(gapped, weight, b)
    assert result is not None
    assert bit_equal(result, manual_out_reference(gapped, weight, b, stage_2d=True))
    assert result.shape == (*gapped.shape[:-1], 80)


def test_the_helper_does_not_mutate_its_inputs(monkeypatch):
    enable(monkeypatch)
    x, weight, b = make((1, 4), 2560, 80)
    before = (x.clone(), weight.clone(), b.clone())
    qwen4_exp_linear_out(x, weight, b)
    assert torch.equal(x, before[0]) and torch.equal(weight, before[1])
    assert torch.equal(b, before[2])


# ---------------------------------------------------------------------------------------------
# The rejected mode: refused for production, with no dispatch
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("lead", REJECTED_WITHOUT_2D)
@pytest.mark.parametrize("in_features,out_features", REAL_SHAPES)
@pytest.mark.parametrize("bias", [True, False])
def test_preserved_rank_above_two_is_refused_without_the_two_dimensional_switch(
        monkeypatch, lead, in_features, out_features, bias):
    """Every configuration the eight-card oracle rejected must be a pure fallback here.

    The helper is not allowed to serve the preserved-rank request and let a caller compare it as a
    candidate: 22 of 269 device cases disagreed. Returning ``None`` is the whole contract, and the
    dispatch counter below proves the refusal happened before any kernel was asked.
    """
    enable_without_2d(monkeypatch)
    dispatched = []
    monkeypatch.setattr(linear_out, "_linear_out_op",
                        lambda: dispatched.append(True) or pytest.fail(
                            "a refused shape must not reach linear.out"))
    x, weight, b = make(lead, in_features, out_features, bias=bias)
    assert x.ndim > 2
    assert qwen4_exp_linear_out(x, weight, b) is None
    assert dispatched == [], "the refused preserved-rank call dispatched an out kernel"


def test_a_refused_rank_does_not_even_lookup_the_out_overload(monkeypatch):
    """The refusal must precede the overload lookup, so a build without it also falls back."""
    enable_without_2d(monkeypatch)
    monkeypatch.setattr(linear_out, "_available_overloads",
                        lambda: pytest.fail("a refused shape must not inspect aten overloads"))
    x, weight, b = make((1, 4), 16, 8)
    assert qwen4_exp_linear_out(x, weight, b) is None


def test_rank_two_is_accepted_without_the_two_dimensional_switch(monkeypatch):
    """Rank two is the one geometry where the two staging spellings cannot disagree."""
    enable_without_2d(monkeypatch)
    x, weight, b = make((4,), 2560, 80)
    result = qwen4_exp_linear_out(x, weight, b)
    assert result is not None
    assert bit_equal(result, manual_out_reference(x, weight, b, stage_2d=False))


# ---------------------------------------------------------------------------------------------
# Row staging rank: a separate explicit choice, because it changes which ATen path answers
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("ndim,lead", LEADING_GEOMETRIES)
@pytest.mark.parametrize("in_features,out_features", REAL_SHAPES)
@pytest.mark.parametrize("bias", [True, False])
def test_the_two_dimensional_mode_reproduces_the_old_reference_bit_for_bit(monkeypatch, ndim, lead,
                                                                         in_features, out_features,
                                                                         bias):
    """The qualified staging form, compared against the reference it would replace.

    The reference loop hands each row to `F.linear` at the caller's rank, and that composite folds
    singleton dimensions down to a matrix before it reaches the kernel. Staging a plain matrix
    reproduces that folded request, on CPU and on the eight-card device oracle.
    """
    enable(monkeypatch)
    x, weight, b = make(lead, in_features, out_features, bias=bias)
    result = qwen4_exp_linear_out(x, weight, b)
    assert result is not None
    assert bit_equal(result, reference_linear(x, weight, b)), (
        f"{in_features}->{out_features} at ndim {ndim} diverges from the reference"
    )
    assert result.shape == (*x.shape[:-1], out_features)


def test_the_two_dimensional_mode_writes_one_matrix_row_per_call(monkeypatch):
    enable(monkeypatch)
    calls = []
    real = torch.ops.aten.linear.out

    def recording_op(staged, weight, bias=None, *, out):
        calls.append((tuple(staged.shape), tuple(out.shape), staged.is_contiguous()))
        return real(staged, weight, bias, out=out)

    monkeypatch.setattr(linear_out, "_linear_out_op", lambda: recording_op)
    x, weight, b = make((1, 4), 2560, 80)
    assert qwen4_exp_linear_out(x, weight, b) is not None
    assert calls == [((1, 2560), (1, 80), True)] * 4


@pytest.mark.parametrize("lead", [(4,), (1, 4), (2, 2)])
def test_the_staged_input_carries_the_old_input_bits_and_row_order(monkeypatch, lead):
    """Oracle for the input side: row `r` must reach the kernel with the reference's own bits.

    The staged request is two-dimensional in every accepted configuration, so the operand has to
    match the old loop's staging element for element and row for row.
    """
    enable(monkeypatch)
    x, weight, b = make(lead, 2560, 768)
    staged_rows = []
    real = torch.ops.aten.linear.out

    def recording_op(staged, weight, bias=None, *, out):
        staged_rows.append(staged.clone())
        return real(staged, weight, bias, out=out)

    monkeypatch.setattr(linear_out, "_linear_out_op", lambda: recording_op)
    assert qwen4_exp_linear_out(x, weight, b) is not None

    rows = x.numel() // x.shape[-1]
    flat = x.reshape(rows, x.shape[-1])
    assert len(staged_rows) == rows
    for row in range(rows):
        # The reference loop's own staging, rebuilt here rather than sampled from the helper.
        expected = flat[row:row + 1].reshape((1,) * (x.ndim - 1) + (x.shape[-1],)).contiguous()
        # Same values in the same order, whatever rank the kernel was asked for.
        assert torch.equal(staged_rows[row].reshape(-1), expected.reshape(-1)), (
            f"row {row} reached the kernel with different input bits"
        )
        assert tuple(staged_rows[row].shape) == (1, 2560)


def test_the_two_dimensional_switch_defaults_off(monkeypatch):
    """Unset means the rejected request: rank two served, rank three refused."""
    enable(monkeypatch)
    monkeypatch.delenv("POCKETLLM_QWEN4_LINEAR_OUT_2D", raising=False)
    calls = []
    real = torch.ops.aten.linear.out

    def recording_op(staged, weight, bias=None, *, out):
        calls.append(tuple(staged.shape))
        return real(staged, weight, bias, out=out)

    monkeypatch.setattr(linear_out, "_linear_out_op", lambda: recording_op)
    accepted, weight, b = make((4,), 2560, 80)
    assert qwen4_exp_linear_out(accepted, weight, b) is not None
    assert calls == [(1, 2560)] * 4
    calls.clear()
    refused, weight, b = make((1, 4), 2560, 80)
    assert qwen4_exp_linear_out(refused, weight, b) is None
    assert calls == []


def test_the_two_dimensional_switch_is_not_read_while_the_helper_is_off(monkeypatch):
    monkeypatch.delenv("POCKETLLM_QWEN4_LINEAR_OUT", raising=False)
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT_2D", "1")
    x, weight, b = make((4,), 16, 8)
    assert qwen4_exp_linear_out(x, weight, b) is None


# ---------------------------------------------------------------------------------------------
# The binding predicate: the model may only be pointed at the qualified request path
# ---------------------------------------------------------------------------------------------


def test_the_binding_predicate_requires_all_three_switches(monkeypatch):
    for name in _ENABLE_VARS:
        monkeypatch.delenv(name, raising=False)
    assert linear_out_enabled() is False
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT", "1")
    assert linear_out_enabled() is False, "strict rows are a prerequisite"
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    assert linear_out_enabled() is False, "the rejected preserved-rank mode must not be bound"
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT_2D", "1")
    assert linear_out_enabled() is True


def test_the_binding_predicate_defaults_off(monkeypatch):
    for name in _ENABLE_VARS:
        monkeypatch.delenv(name, raising=False)
    assert linear_out_enabled() is False


def test_the_helper_stays_opt_in_when_the_binding_predicate_is_false(monkeypatch):
    """A caller that binds on `linear_out_enabled` keeps the reference loop, as shipped."""
    from relicllm.models.qwen4_exp import layers
    for name in _ENABLE_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    monkeypatch.setattr(linear_out, "qwen4_exp_linear_out",
                        lambda *_args: pytest.fail("default-off helper reached"))
    x, weight, b = make((1, 4), 2560, 80)
    actual = layers.row_invariant_linear(x, weight, b)
    assert torch.equal(actual, reference_linear(x, weight, b))


# ---------------------------------------------------------------------------------------------
# Guards: everything the experiment does not cover must fall back rather than guess
# ---------------------------------------------------------------------------------------------


def test_the_switch_defaults_off(monkeypatch):
    monkeypatch.delenv("POCKETLLM_QWEN4_LINEAR_OUT", raising=False)
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    x, weight, b = make((4,), 16, 8)
    assert qwen4_exp_linear_out(x, weight, b) is None


@pytest.mark.parametrize("value", ["0", "false", "no", "", "true", "yes"])
def test_only_exactly_one_enables_the_helper(monkeypatch, value):
    """The strict-row contract spells its switches ``== "1"``; a truthy spelling must not opt in."""
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT", value)
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    x, weight, b = make((4,), 16, 8)
    assert qwen4_exp_linear_out(x, weight, b) is None


def test_the_strict_row_switch_is_a_prerequisite(monkeypatch):
    """The helper is an implementation detail of the strict-row contract, never a widening of it."""
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT", "1")
    monkeypatch.delenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", raising=False)
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT_2D", "1")
    x, weight, b = make((4,), 16, 8)
    assert qwen4_exp_linear_out(x, weight, b) is None
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT", "0")
    assert qwen4_exp_linear_out(x, weight, b) is None


@pytest.mark.parametrize("rows", [1, 9, 16])
def test_row_counts_outside_the_strict_window_fall_back(monkeypatch, rows):
    enable(monkeypatch)
    x, weight, b = make((rows,), 16, 8)
    assert qwen4_exp_linear_out(x, weight, b) is None


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
def test_every_strict_row_count_is_accepted(monkeypatch, rows):
    enable(monkeypatch)
    x, weight, b = make((rows,), 16, 8)
    result = qwen4_exp_linear_out(x, weight, b)
    assert result is not None and result.shape == (rows, 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_only_bfloat16_is_accepted(monkeypatch, dtype):
    enable(monkeypatch)
    x, weight, b = make((4,), 16, 8, dtype=dtype)
    assert qwen4_exp_linear_out(x, weight, b) is None
    x, weight, b = make((4,), 16, 8)
    assert qwen4_exp_linear_out(x, weight.to(dtype), b) is None


def test_a_one_dimensional_input_is_refused(monkeypatch):
    enable(monkeypatch)
    weight, b = torch.randn(8, 16, dtype=torch.bfloat16), torch.randn(8, dtype=torch.bfloat16)
    assert qwen4_exp_linear_out(torch.randn(16, dtype=torch.bfloat16), weight, b) is None


def test_a_mismatched_inner_dimension_is_refused(monkeypatch):
    enable(monkeypatch)
    x, _weight, b = make((4,), 16, 8)
    assert qwen4_exp_linear_out(x, torch.randn(8, 17, dtype=torch.bfloat16), b) is None


def test_a_non_matrix_weight_is_refused(monkeypatch):
    enable(monkeypatch)
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    b = torch.randn(8, dtype=torch.bfloat16)
    assert qwen4_exp_linear_out(x, torch.randn(8, 2, 16, dtype=torch.bfloat16), b) is None


def test_a_device_mismatch_is_refused(monkeypatch):
    enable(monkeypatch)
    x, weight, b = make((4,), 16, 8)
    # A meta weight has a different device and can never be executed with a CPU input.
    assert qwen4_exp_linear_out(x, weight.to("meta"), b) is None


@pytest.mark.parametrize("bad_bias", ["wrong_width", "two_dimensions", "float32"])
def test_an_unsupported_bias_is_refused(monkeypatch, bad_bias):
    enable(monkeypatch)
    x, weight, b = make((4,), 16, 8)
    if bad_bias == "wrong_width":
        bias = torch.randn(9, dtype=torch.bfloat16)
    elif bad_bias == "two_dimensions":
        bias = torch.randn(1, 8, dtype=torch.bfloat16)
    else:
        bias = torch.randn(8, dtype=torch.float32)
    assert qwen4_exp_linear_out(x, weight, bias) is None


def test_an_empty_input_is_refused(monkeypatch):
    enable(monkeypatch)
    x = torch.empty(0, 16, dtype=torch.bfloat16)
    weight = torch.randn(8, 16, dtype=torch.bfloat16)
    assert qwen4_exp_linear_out(x, weight, None) is None


def test_grad_tracking_operands_are_refused(monkeypatch):
    """`out=` kernels cannot build a graph; the helper must decline instead of raising."""
    enable(monkeypatch)
    x = torch.randn(4, 16, requires_grad=True)
    weight = torch.randn(8, 16, requires_grad=True)
    assert qwen4_exp_linear_out(x, weight, None) is None
    assert qwen4_exp_linear_out(x.detach(), weight.detach(), None) is None  # dtype is float32


def test_inference_mode_is_the_expected_caller(monkeypatch):
    enable(monkeypatch)
    x, weight, b = make((4,), 16, 8)
    with torch.inference_mode():
        result = qwen4_exp_linear_out(x, weight, b)
    assert result is not None and result.shape == (4, 8)


# ---------------------------------------------------------------------------------------------
# Plumbing: one out call a row, with the expected shapes and no resize of the shared buffer
# ---------------------------------------------------------------------------------------------


def test_one_out_call_per_row_with_matrix_shapes(monkeypatch):
    enable(monkeypatch)
    calls = []
    real = torch.ops.aten.linear.out

    def recording_op(staged, weight, bias=None, *, out):
        calls.append((tuple(staged.shape), tuple(out.shape), out.dtype, staged.is_contiguous()))
        return real(staged, weight, bias, out=out)

    monkeypatch.setattr(linear_out, "_linear_out_op", lambda: recording_op)
    x, weight, b = make((1, 4), 2560, 80)
    result = qwen4_exp_linear_out(x, weight, b)
    assert result is not None
    assert calls == [((1, 2560), (1, 80), torch.bfloat16, True)] * 4


def test_a_flat_out_target_is_silently_resized_so_the_natural_rank_is_used(monkeypatch):
    """Guards the leading-shape target decision with the failure it prevents.

    ATen resizes an `out` tensor whose shape does not match the kernel's natural rank. A flat
    `(1, out)` target therefore comes back with a different shape than it was passed, so a caller
    iterating row views could not rely on the handle it built. The helper passes the kernel's
    natural rank, and this test records the resize that would otherwise be silent.
    """
    torch.manual_seed(0)
    staged = torch.randn(1, 1, 8, dtype=torch.bfloat16)
    weight = torch.randn(4, 8, dtype=torch.bfloat16)
    flat = torch.empty(1, 4, dtype=torch.bfloat16)
    assert flat.shape == (1, 4)
    torch.ops.aten.linear.out(staged, weight, None, out=flat)
    assert flat.shape == (1, 1, 4), "a flat target is resized to the kernel's rank, which is the hazard"


def test_a_missing_out_overload_fails_loudly_when_requested(monkeypatch):
    """An explicit request must not be quietly served by the old path."""
    enable(monkeypatch)
    monkeypatch.setattr(linear_out, "_available_overloads", lambda: ["default"])
    x, weight, b = make((4,), 16, 8)
    with pytest.raises(RuntimeError, match="linear.out"):
        qwen4_exp_linear_out(x, weight, b)


# ---------------------------------------------------------------------------------------------
# Source contract: the helper may only use the call it claims, and stays out of the model path
# ---------------------------------------------------------------------------------------------


def _source() -> str:
    return (pathlib.Path(__file__).resolve().parents[1]
            / "relicllm/models/qwen4_exp/linear_out.py").read_text()


def test_the_helper_uses_only_the_out_linear_call():
    import re

    source = _source()
    assert "torch.ops.aten.linear.out" in source
    for forbidden in (r"\btorch\.mm\b", r"\btorch\.bmm\b", r"\btorch\.matmul\b", r"\baddmm\b",
                      r"\beinsum\b", r"\bfloat16\b", r"\ballow_tf32\b"):
        assert re.search(forbidden, source) is None, (
            f"the frozen experiment must not introduce {forbidden!r}"
        )


def test_the_switches_are_documented_as_off_by_default():
    source = _source()
    assert '_ENABLE_ENV = "POCKETLLM_QWEN4_LINEAR_OUT"' in source
    assert '_STRICT_ENV = "POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT"' in source
    assert '_STAGE_2D_ENV = "POCKETLLM_QWEN4_LINEAR_OUT_2D"' in source
    assert 'os.environ.get(name, "0") == "1"' in source, "switches are spelled exactly 1"
    assert "def linear_out_enabled()" in source


def test_the_rejected_mode_is_refused_before_dispatch_in_source():
    """The guard order is part of the contract, so a refactor cannot move it behind the lookup."""
    source = _source()
    refusal = "if x.ndim > 2 and not _stage_two_dimensional_rows():"
    assert refusal in source
    assert source.index(refusal) < source.index("op = _linear_out_op()"), (
        "the rejected rank must return None before the out overload is looked up"
    )


# ---------------------------------------------------------------------------------------------
# Device-qualified oracle. Skipped without a card: a skip is not a pass, and these are the only
# tests that authorise promotion of a shape. Every parametrised configuration here is one the
# helper accepts; the 22 configurations the device rejected are CPU refusal tests above.
# ---------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cuda():
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return torch.device("cuda", torch.cuda.current_device())


#: (stage, leading) pairs the helper actually serves: two-dimensional staging for every leading
#: shape, preserved rank only where the request is already two-dimensional.
ACCEPTED_STAGES = [
    ("2", (4,)),
    ("2", (1, 4)),
    ("2", (2, 2)),
    ("same", (4,)),
]


@pytest.mark.parametrize("stage,leading", ACCEPTED_STAGES)
@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize("in_features,out_features", REAL_SHAPES)
def test_gpu_bits_match_the_recorded_reference_for_each_real_shape(monkeypatch, cuda, stage,
                                                                   leading, bias, in_features,
                                                                   out_features):
    """The promotion gate: every served shape must reproduce the one-row reference bit for bit.

    The reference is the same loop the model uses today, executed on the same device with the same
    operands, so a pass here means the replacement gave up nothing on that shape. Bias and no-bias
    are separate cases: the device rejection report distinguishes them, so the oracle must too.
    """
    enable(monkeypatch)
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT_2D", "1" if stage == "2" else "0")
    x, weight, b = make(leading, in_features, out_features, bias=bias)
    x, weight = x.to(cuda), weight.to(cuda)
    b = None if b is None else b.to(cuda)
    result = qwen4_exp_linear_out(x, weight, b)
    assert result is not None and result.device == cuda
    assert bit_equal(result, reference_linear(x, weight, b))


@pytest.mark.parametrize("stage,leading", ACCEPTED_STAGES)
def test_gpu_adversarial_vectors_stay_bit_identical(monkeypatch, cuda, stage, leading):
    """Cancellation-heavy and rounding-tie inputs, which is where a reordered sum shows up."""
    enable(monkeypatch)
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT_2D", "1" if stage == "2" else "0")
    torch.manual_seed(0)
    for in_features, out_features in ((2560, 768), (2560, 2560), (80, 2560)):
        x = torch.randn(*leading, in_features, dtype=torch.bfloat16)
        # Alternating large magnitudes force cancellation inside the reduction.
        signs = torch.sign(torch.randn(in_features))
        x = (x.float() * signs).to(torch.bfloat16)
        weight = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        # A near-tie row: many identical products, so the sum's order decides the last bits.
        weight[0] = x.reshape(-1, in_features)[0]
        x, weight = x.to(cuda), weight.to(cuda)
        for bias in (None, torch.randn(out_features, dtype=torch.bfloat16, device=cuda)):
            result = qwen4_exp_linear_out(x, weight, bias)
            assert result is not None
            assert bit_equal(result, reference_linear(x, weight, bias))


@pytest.mark.parametrize("stage,leading", ACCEPTED_STAGES)
def test_gpu_result_survives_a_captured_replay(monkeypatch, cuda, stage, leading):
    """The path exists to run inside a graph, so it must record and replay with stable addresses."""
    enable(monkeypatch)
    monkeypatch.setenv("POCKETLLM_QWEN4_LINEAR_OUT_2D", "1" if stage == "2" else "0")
    x, weight, b = make(leading, 2560, 768)
    x, weight, b = x.to(cuda), weight.to(cuda), b.to(cuda)
    reference = reference_linear(x, weight, b)
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        qwen4_exp_linear_out(x, weight, b)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        captured = qwen4_exp_linear_out(x, weight, b)
    assert captured is not None
    graph.replay()
    torch.cuda.synchronize()
    assert bit_equal(captured, reference)


def test_gpu_refused_rank_dispatches_no_kernel(monkeypatch, cuda):
    """On a real device the rejected request must still be a refusal, not a served candidate."""
    enable_without_2d(monkeypatch)
    higher = torch.randn(1, 4, 2560, dtype=torch.bfloat16, device=cuda)
    weight = torch.randn(80, 2560, dtype=torch.bfloat16, device=cuda)
    assert qwen4_exp_linear_out(higher, weight, None) is None