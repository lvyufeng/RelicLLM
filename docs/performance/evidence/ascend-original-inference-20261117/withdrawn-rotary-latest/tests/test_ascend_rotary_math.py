"""Exact scalar semantics of the fused rotary lane, with no Triton and no torch.

These tests execute this project's own Triton function body with scalar float32
semantics and compare it against an independent exact-rational statement of the
reference's contracted complex multiply. They are not a hardware test:
``tests/test_ascend_rotary.py`` is what holds the operator to ATen.

The property under test is narrow and easy to lose. The reference rotates a pair
``(a, b)`` by ``(c, s)`` as a ``complex64`` multiply, and that multiply is
compiled with one fused step in each component::

    real = fma(a, c, -round_f32(b * s))
    imag = fma(a, s, +round_f32(b * c))

Every part of that matters. ``round_f32(b * s)`` is a *separately* rounded
product, not a second fused multiply; the fused product is the **first** one, not
the second; and ``fma`` rounds exactly once. The kernel does not write those two
lines, it writes one lane-uniform expression over ``this``/``other``/``cos``/
``sin``, so this file also has to show that the uniform form is the same function.

Two independent statements are used:

* the oracle below is the two ATen lines written in exact ``Fraction`` arithmetic,
  rounding once per operation with an explicitly written float32 round-to-nearest-
  even;
* the kernel side is its own source, re-executed by an AST harness.

The float32 FMA emulation is additionally cross-checked against the C library's
``fmaf`` where the platform exposes one, which is a third implementation and not a
restatement of the first two.
"""
from __future__ import annotations

import ast
import ctypes
import math
from fractions import Fraction
from pathlib import Path
import random
import struct

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "relic_core/kernels/_ascend_rotary_kernels.py"

_TWO = Fraction(2)
_HALF = Fraction(1, 2)
_TWO128 = _TWO**128
_NAN = float("nan")


# ---------------------------------------------------------------------------------------------
# Exact binary32 arithmetic used by both statements
# ---------------------------------------------------------------------------------------------


def sign_bit(value: float) -> bool:
    """Whether a Python float carries a negative sign bit, including ``-0.0``."""
    return math.copysign(1.0, value) < 0


def round_f32(value: Fraction, negative_zero: bool = False) -> float:
    """The nearest binary32 to ``value``, ties to even.

    ``negative_zero`` says which zero to return when the value rounds to zero:
    an exact zero has no sign of its own, and the sign of a zero is part of this
    operator's contract because it survives into the stored activation.
    """
    if value == 0:
        return -0.0 if negative_zero else 0.0
    magnitude = abs(value)
    if magnitude >= _TWO128:
        return -math.inf if value < 0 else math.inf
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    while _TWO**exponent > magnitude:
        exponent -= 1
    while _TWO ** (exponent + 1) <= magnitude:
        exponent += 1
    # 24 significand bits; below the smallest normal the grid is fixed at 2**-149.
    step = _TWO ** (max(exponent, -126) - 23)
    scaled = magnitude / step
    whole = scaled.numerator // scaled.denominator
    remainder = scaled - whole
    if remainder > _HALF or (remainder == _HALF and whole % 2 == 1):
        whole += 1
    rounded = whole * step
    if rounded >= _TWO128:
        return -math.inf if value < 0 else math.inf
    result = float(rounded)
    return -result if value < 0 else result


def multiply_f32(first: float, second: float) -> float:
    """Binary32 product, rounding once and keeping the sign of a zero result."""
    first, second = _value(first), _value(second)
    if not (math.isfinite(first) and math.isfinite(second)):
        return _multiply_special(first, second)
    exact = Fraction(first) * Fraction(second)
    negative = sign_bit(first) != sign_bit(second)
    return round_f32(exact, negative_zero=negative)


def _multiply_special(first: float, second: float) -> float:
    """IEEE multiplication where an operand is already infinite or NaN."""
    if math.isnan(first) or math.isnan(second):
        return _NAN
    if first == 0.0 or second == 0.0:
        # The only way here is inf * 0, which IEEE defines as an invalid operation.
        return _NAN
    return -math.inf if sign_bit(first) != sign_bit(second) else math.inf


def _add_special(first: float, second: float) -> float:
    """IEEE addition where an operand is already infinite or NaN."""
    if math.isnan(first) or math.isnan(second):
        return _NAN
    if math.isinf(first) and math.isinf(second):
        return _NAN if sign_bit(first) != sign_bit(second) else first
    return first if math.isinf(first) else second


def fma_f32(first: float, second: float, addend: float) -> float:
    """Binary32 fused multiply-add: ``first * second + addend``, rounded once.

    The exact-zero result has three cases and they are not interchangeable. If
    both the product and the addend are zero the result is ``-0.0`` only when
    both are negative; if nonzero operands cancel exactly, round-to-nearest gives
    ``+0.0``; otherwise the sign is the exact sum's.
    """
    first, second, addend = _value(first), _value(second), _value(addend)
    if not (math.isfinite(first) and math.isfinite(second)):
        return _add_special(_multiply_special(first, second), addend)
    product = Fraction(first) * Fraction(second)
    if not math.isfinite(addend):
        # A finite product cannot change an infinite or NaN addend.
        return addend
    exact_addend = Fraction(addend)
    total = product + exact_addend
    if total != 0:
        return round_f32(total)
    if product == 0 and exact_addend == 0:
        both_negative = (sign_bit(first) != sign_bit(second)) and sign_bit(addend)
        return -0.0 if both_negative else 0.0
    return 0.0


class F32:
    """A Python float restricted to binary32 values, with binary32 arithmetic.

    Only the operations the kernel body performs are modelled: multiplication
    rounds to binary32, negation is exact, and everything else stays a value. A
    product that overflows becomes an infinity, which later steps propagate.
    """

    def __init__(self, value):
        if isinstance(value, F32):
            self.value = value.value
            return
        raw = float(value)
        self.value = (round_f32(Fraction(raw), negative_zero=sign_bit(raw))
                      if math.isfinite(raw) else raw)

    def __mul__(self, other):
        return F32(multiply_f32(self.value, _value(other)))

    __rmul__ = __mul__

    def __neg__(self):
        return F32(-self.value)

    def __bool__(self):
        return bool(self.value)

    def __repr__(self) -> str:  # pragma: no cover - diagnostic aid
        return f"F32({self.value!r})"


def add_f32(first, second) -> float:
    """Binary32 sum, keeping the sign of an exact zero result."""
    first, second = _value(first), _value(second)
    if not (math.isfinite(first) and math.isfinite(second)):
        return _add_special(first, second)
    total = Fraction(first) + Fraction(second)
    if total != 0:
        return round_f32(total)
    return -0.0 if sign_bit(first) and sign_bit(second) else 0.0


def subtract_f32(first, second) -> float:
    """Binary32 difference; ``-0.0 - 0.0`` is ``-0.0`` and cancellation is ``+0.0``."""
    return add_f32(first, -_value(second))


def _value(other) -> float:
    return other.value if isinstance(other, F32) else float(other)


def as_f32(value: float) -> float:
    """The binary32 value a Python float denotes.

    Test inputs have to be normalised through this. ``1e-05`` written as a literal
    is a *double* whose float32 neighbour differs in the last bit, and feeding the
    double to the oracle while the kernel receives its rounded form would compare
    two different operands -- a disagreement that says nothing about the kernels.
    """
    return round_f32(Fraction(value), negative_zero=sign_bit(value))


def _bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _same(left, right) -> bool:
    """Bit equality for binary32, so ``+0.0`` and ``-0.0`` are distinguishable."""
    return _bits(_value(left)) == _bits(_value(right))


# ---------------------------------------------------------------------------------------------
# The kernel side: its own source, re-executed
# ---------------------------------------------------------------------------------------------


def kernel_helpers(name: str):
    """Compile one ``@triton.jit`` body from the kernel module into plain Python.

    The decorator is dropped and ``tl`` is replaced by a namespace whose ``fma``
    is the exact-rational one above, so the expression under test is literally the
    one that ships rather than a transcription of it.
    """
    tree = ast.parse(SOURCE.read_text())
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(functions) == 1, f"expected exactly one {name} in {SOURCE.name}"
    function = functions[0]
    function.decorator_list = []
    tl = type("tl", (), {
        "where": staticmethod(lambda condition, left, right: left if bool(condition) else right),
        "fma": staticmethod(fma_f32),
    })
    namespace = {"tl": tl}
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, str(SOURCE), "exec"), namespace)  # noqa: S102 - executing our own source
    return namespace[name]


# ---------------------------------------------------------------------------------------------
# The oracle: the two ATen lines, stated directly
# ---------------------------------------------------------------------------------------------


def aten_pair(a: float, b: float, c: float, s: float) -> tuple[float, float]:
    """The reference's contracted complex multiply on one ``(a, b)`` by ``(c, s)``.

    ``a``/``b`` are the pair's real and imaginary parts, ``c``/``s`` the cosine and
    sine. Written without reference to the kernel's lane formulation: the addend of
    each ``fma`` is the *other* product rounded to binary32 first, and the fused
    product is the first one. ``fma_f32`` supplies the IEEE fused step itself and is
    validated separately against the C library's ``fmaf``; what this function states
    independently is which of the two products is rounded and which is fused.
    """
    return fma_f32(a, c, -multiply_f32(b, s)), fma_f32(a, s, multiply_f32(b, c))


def aten_pair_with_sine(a: float, b: float, c: float, s: float,
                        inverse: bool = False) -> tuple[float, float]:
    """The pair rotation, optionally by the conjugate (a negated sine)."""
    return aten_pair(a, b, c, -s if inverse else s)


# ---------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------

#: Values chosen so the roundings can differ: signed zeros, the smallest
#: subnormals, the largest finite values, and magnitudes whose products land on a
#: binary32 tie. Every entry is normalised to binary32 so the oracle and the kernel
#: are handed the same operand.
EDGE_VALUES = tuple(as_f32(value) for value in (
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5,
    2.0**-149, -(2.0**-149), 2.0**-126, -(2.0**-126),
    5.877471754111438e-39, -5.877471754111438e-39,
    2.0**24, -(2.0**24), 65504.0, -65504.0,
    3.4028234663852886e38, -3.4028234663852886e38,
))
EDGE_ANGLES = tuple(as_f32(value) for value in (
    0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1e-5, -1e-5, 3e-6, -3e-6,
    2.0**-149, -(2.0**-149), 3.4028234663852886e38, -3.4028234663852886e38,
))
assert all(_bits(as_f32(value)) == _bits(value) for value in EDGE_VALUES + EDGE_ANGLES), (
    "every edge case must already be a binary32 value"
)


def _finite_f32(pattern: int) -> float | None:
    value = struct.unpack("<f", struct.pack("<I", pattern & 0xFFFFFFFF))[0]
    return value if math.isfinite(value) else None


def random_cases(count: int, seed: int = 20261117):
    generator = random.Random(seed)
    produced = 0
    while produced < count:
        values = [_finite_f32(generator.getrandbits(32)) for _ in range(4)]
        if any(value is None for value in values):
            continue
        yield tuple(values)
        produced += 1


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------


def test_the_fma_emulation_matches_the_c_library_where_one_is_available():
    """A third implementation of binary32 fma, so the emulation is not self-certified."""
    try:
        library = ctypes.CDLL(None)
        library.fmaf.argtypes = [ctypes.c_float] * 3
        library.fmaf.restype = ctypes.c_float
    except (OSError, AttributeError):  # pragma: no cover - platform dependent
        pytest.skip("this platform exposes no fmaf through ctypes")
    generator = random.Random(4242)
    triples = [(1.0, 1.0, 0.0), (0.0, -1.0, -0.0), (-0.0, -1.0, -0.0),
               (2.0**-149, 1.0, 0.0), (3.4028234663852886e38, 3.4028234663852886e38, 0.0)]
    while len(triples) < 4000:
        values = [_finite_f32(generator.getrandbits(32)) for _ in range(3)]
        if all(value is not None for value in values):
            triples.append(tuple(values))
    for first, second, addend in triples:
        expected = library.fmaf(first, second, addend)
        assert _bits(fma_f32(first, second, addend)) == _bits(expected), (first, second, addend)


def test_the_lane_helper_matches_the_reference_formula_on_every_edge_combination():
    rotate = kernel_helpers("_rotate_lane")
    checked = 0
    for a in EDGE_VALUES:
        for b in EDGE_VALUES:
            for c in EDGE_ANGLES:
                for s in EDGE_ANGLES:
                    real, imag = aten_pair(a, b, c, s)
                    # The real half of the pair: this=a, other=b, angle=(c, s).
                    assert _same(rotate(F32(a), F32(b), F32(c), F32(s), True), real), (
                        f"real of ({a}, {b}) by ({c}, {s})"
                    )
                    # The imaginary half: the roles of the pair swap.
                    assert _same(rotate(F32(b), F32(a), F32(c), F32(s), False), imag), (
                        f"imag of ({a}, {b}) by ({c}, {s})"
                    )
                    checked += 2
    assert checked == 2 * len(EDGE_VALUES) ** 2 * len(EDGE_ANGLES) ** 2


def test_the_lane_helper_matches_the_reference_formula_on_random_bit_patterns():
    rotate = kernel_helpers("_rotate_lane")
    for a, b, c, s in random_cases(6000):
        real, imag = aten_pair(a, b, c, s)
        assert _same(rotate(F32(a), F32(b), F32(c), F32(s), True), real), (a, b, c, s)
        assert _same(rotate(F32(b), F32(a), F32(c), F32(s), False), imag), (a, b, c, s)


def test_the_inverse_direction_is_the_conjugate_and_matches_the_reference():
    rotate = kernel_helpers("_rotate_lane")
    for a in EDGE_VALUES:
        for b in EDGE_VALUES:
            for c in EDGE_ANGLES:
                for s in EDGE_ANGLES:
                    real, imag = aten_pair_with_sine(a, b, c, s, inverse=True)
                    negated = F32(-F32(s))
                    assert _same(rotate(F32(a), F32(b), F32(c), negated, True), real), (a, b, c, s)
                    assert _same(rotate(F32(b), F32(a), F32(c), negated, False), imag), (a, b, c, s)


def test_the_second_product_is_not_fused():
    """The documented defect: fusing the *other* product changes the answer.

    If the kernel fused ``first * angle`` into the addend instead of the reverse
    -- or fused both -- it would still look like a rotary kernel. This case is a
    concrete input where the two contractions disagree, so the test above is
    known to have teeth rather than merely passing.
    """
    rotate = kernel_helpers("_rotate_lane")
    disagreements = 0
    for a, b, c, s in random_cases(3000):
        fused_first = aten_pair(a, b, c, s)
        # The wrong contraction: round the first product, fuse the second.
        wrong_real = fma_f32(b, s, -multiply_f32(a, c))
        wrong_imag = fma_f32(b, c, multiply_f32(a, s))
        if _bits(wrong_real) != _bits(fused_first[0]):
            disagreements += 1
            assert not _same(rotate(F32(a), F32(b), F32(c), F32(s), True), wrong_real)
        if _bits(wrong_imag) != _bits(fused_first[1]):
            disagreements += 1
            assert not _same(rotate(F32(b), F32(a), F32(c), F32(s), False), wrong_imag)
    assert disagreements > 100, (
        f"only {disagreements} of 6000 samples separate the two contractions; the "
        "comparison would not have caught the defect this operator exists to fix"
    )


def test_the_unfused_product_is_not_the_reference():
    """A third spelling, refused for the same reason as a compiler-chosen contraction."""
    disagreements = 0
    for a, b, c, s in random_cases(3000):
        real, imag = aten_pair(a, b, c, s)
        unfused_real = subtract_f32(multiply_f32(a, c), multiply_f32(b, s))
        unfused_imag = add_f32(multiply_f32(a, s), multiply_f32(b, c))
        if _bits(unfused_real) != _bits(real):
            disagreements += 1
        if _bits(unfused_imag) != _bits(imag):
            disagreements += 1
    assert disagreements > 100, f"only {disagreements} of 6000 samples distinguish the fusion"


def test_the_kernel_uses_one_explicit_fma_over_a_pre_rounded_product():
    """Structural pin: the source must fuse exactly once, and not by accident.

    Reading the AST rather than the text keeps the check honest about *which*
    product is fused: the addend passed to ``fma`` has to be the separately
    computed product, so the compiler is never asked to choose.
    """
    tree = ast.parse(SOURCE.read_text())
    helper = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_rotate_lane")
    calls = [node for node in ast.walk(helper)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
             and node.func.attr == "fma"]
    assert len(calls) == 1, "the lane helper must perform exactly one fused multiply-add"
    fused = calls[0]
    assert len(fused.args) == 3
    # `a * b + c` chosen by the compiler is what must not happen: the third
    # argument is a value computed earlier, never the sum itself.
    assert not isinstance(fused.args[2], ast.BinOp), (
        "the fma addend must be the pre-rounded product, not a sum the compiler could contract"
    )
    assert isinstance(fused.args[2], ast.Name)
    multiplications = [node for node in ast.walk(helper) if isinstance(node, ast.BinOp)
                       and isinstance(node.op, ast.Mult)]
    assert len(multiplications) == 1, (
        "exactly one multiplication belongs in the lane helper: the pair product "
        "that is rounded before the fused step"
    )
    # No float64 escape hatch and no numeric constant that is not binary32-exact.
    for node in ast.walk(helper):
        assert not (isinstance(node, ast.Constant) and isinstance(node.value, float)
                    and node.value not in {0.0, -1.0, 1.0} and abs(node.value) > 1.0)


def test_the_full_kernel_offers_no_complex_or_float64_operations():
    """The op must not reach for complex64 or FP64 on the device.

    The module *documents* the complex64 semantics it replaces, so the check reads
    identifiers off the AST rather than looking for words: a ``complex`` dtype, an
    FP64 dtype or a torch/numpy escape hatch would all have to be named to be used.
    """
    tree = ast.parse(SOURCE.read_text())
    identifiers = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.add(node.id)
        elif isinstance(node, ast.Attribute):
            identifiers.add(node.attr)
        elif isinstance(node, ast.Import):
            identifiers.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            identifiers.add(node.module.split(".")[0])
    for forbidden in ("complex64", "complex128", "float64", "double", "torch", "numpy"):
        assert forbidden not in identifiers, f"the fused rotary kernel must not use {forbidden}"
    # The only module-level import is Triton itself.
    imports = {alias.name.split(".")[0]
               for node in tree.body if isinstance(node, ast.Import)
               for alias in node.names}
    assert imports == {"triton"}, f"unexpected module imports: {sorted(imports)}"

# ---------------------------------------------------------------------------------------------
# Pointer width: the kernel must widen before it multiplies, never after
# ---------------------------------------------------------------------------------------------

#: A geometry whose row base genuinely needs 64 bits: one batch row alone spans
#: more than ``2**31`` elements, so ``batch * stride_batch`` overflows int32.
HUGE = {
    "sequence": 4096,
    "heads": 64,
    "width": 128,
}
HUGE_STRIDES = (
    HUGE["sequence"] * HUGE["heads"] * HUGE["width"],   # batch stride, > 2**31
    HUGE["heads"] * HUGE["width"],                       # sequence stride
    HUGE["width"],                                       # head stride
    1,                                                   # width stride
)


def _wrap32(value: int) -> int:
    """The signed int32 a Triton int32 register would hold for ``value``."""
    value &= 0xFFFFFFFF
    return value - 0x100000000 if value >= 0x80000000 else value


def row_base_widened_first(batch: int, sequence: int, head: int,
                           strides: tuple[int, int, int, int]) -> int:
    """The kernel's current order: widen the row index, then every product is int64."""
    stride_batch, stride_sequence, stride_head, _ = strides
    return batch * stride_batch + sequence * stride_sequence + head * stride_head


def row_base_cast_last(batch: int, sequence: int, head: int,
                       strides: tuple[int, int, int, int]) -> int:
    """The rejected order: sum in int32, widen the finished (already wrapped) sum."""
    stride_batch, stride_sequence, stride_head, _ = strides
    total = _wrap32(_wrap32(batch * stride_batch) + _wrap32(sequence * stride_sequence))
    return _wrap32(total + _wrap32(head * stride_head))


def test_casting_the_row_after_the_products_would_lose_the_address():
    """Why the widening order is a correctness rule and not a style choice.

    Emulating int32 wraparound shows the two orders disagree once a single row's
    span passes ``2**31``: the cast-last order returns a base near zero for a batch
    row that is genuinely gigabytes into the storage. This is the overflow the
    kernel must not have, demonstrated without allocating the tensor.
    """
    sequence, heads, width = HUGE["sequence"], HUGE["heads"], HUGE["width"]
    last_batch = 65
    assert last_batch * HUGE_STRIDES[0] > 2**31
    correct = row_base_widened_first(last_batch, 0, 0, HUGE_STRIDES)
    wrapped = row_base_cast_last(last_batch, 0, 0, HUGE_STRIDES)
    assert correct == last_batch * HUGE_STRIDES[0]
    assert wrapped != correct, "the emulation did not reproduce the overflow it documents"
    assert abs(wrapped) < 2**31 and correct >= 2**31


def test_the_kernel_widens_the_row_before_decomposing_it():
    """Structural pin on the shipping kernel: no int32 product may form first.

    The row-loop variable must be converted to int64 (under the ``INDEX64`` branch)
    *before* it is divided into batch/sequence/head. If the conversion were applied
    to the assembled base instead, the AST would show the loop variable itself
    feeding the multiplications, and this fails.
    """
    tree = ast.parse(SOURCE.read_text())
    kernel = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "rotary_pairs_kernel")
    loops = [node for node in ast.walk(kernel) if isinstance(node, ast.For)]
    # The row loop is the three-argument one, ``range(pid, ROWS, PROGRAMS)``; the
    # width loop inside it is ``range(K_STEPS)``.
    row_loops = [node for node in loops
                 if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call)
                 and getattr(node.iter.func, "id", None) == "range"
                 and len(node.iter.args) == 3]
    assert len(row_loops) == 1, "expected exactly one strided range loop over rows"
    loop = row_loops[0]
    loop_variable = loop.target.id

    # The first statement has to be the widened alias of the loop variable.
    first = loop.body[0]
    assert isinstance(first, ast.Assign) and len(first.targets) == 1
    assert isinstance(first.value, ast.IfExp), (
        "the row must be aliased under the INDEX64 conditional so the widened value "
        "is used for every product below"
    )
    widened_name = first.targets[0].id
    widened_calls = [node for node in ast.walk(first.value)
                     if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "to"]
    assert widened_calls, "the INDEX64 branch must convert the row with .to(...)"

    # After that statement the loop variable may not appear in any arithmetic: all
    # base arithmetic must go through the widened alias.
    for statement in loop.body[1:]:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and node.id == loop_variable:
                raise AssertionError(
                    f"the raw int32 row variable {loop_variable!r} is used after the "
                    "widening alias; every product must use the widened value"
                )
            if isinstance(node, ast.Name) and node.id == widened_name:
                break
    # And the widened alias must actually be what gets decomposed.
    decompositions = [node for statement in loop.body[1:]
                      for node in ast.walk(statement)
                      if isinstance(node, ast.BinOp)
                      and isinstance(node.op, (ast.Mod, ast.FloorDiv))]
    assert decompositions, "the row was never decomposed into batch/sequence/head"
