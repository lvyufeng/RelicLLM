"""The Ascend routed-expert store, checked on the CPU oracle path without a card.

`relicllm/models/deepseek_v4_1/ascend_experts.py` runs a layer's routed experts from the checkpoint's
original packed FP4 bytes: exact INT8-times-two weights with the file's E8M0 grid halved, the
released block-32 FP8 activation quantizer, and the block-32 reduction. These tests are about the
*store's* half of that contract -- the operand mapping, the order the reference states for SwiGLU and
the routing weight, the byte-budgeted cache, and the expert-parallel deal -- not about the GEMM, whose
own evidence lives with the operators.

Two oracles, and neither is the store:

* `_transcribe` below is the reference's arithmetic written out again from
  `.coord/expert-semantics-v3.md`, with knobs that reproduce the three plausible mistakes (clamp after
  `silu`, routing weight on the output, no BF16 narrowing). Tests assert the store matches the
  reference shape **bit for bit** and that each wrong shape differs, so an assertion here fails for
  the reason it names rather than passing because both sides share a bug;
* `modules.expert_forward(..., quantized=True)` is the port's existing reference-reduction path, run
  on dense weights built here from the same bytes (the same construction
  `CheckpointRoutedExperts._expert_matrix` uses). `block_gemm` and `mma.fp8_block_gemm` are two
  transcriptions of the same two-half rule, so the store is expected to match that path **exactly**
  on BF16 activations -- and to differ from `quantized=False`, which is a plain dense matmul and has
  no activation quantization at all. Both directions are asserted, because agreement alone would
  also be produced by a store that quietly ran the dense path.

The fixture is a fake checkpoint exposing the four `V41Checkpoint` methods the store reads
(`packed`, `is_quantized`, `is_packed_fp4`, `block_size`) over hand-built bytes: original packed
E2M1 nibbles, original E8M0 scale containers. Nothing expands a dense weight for the store, and no
test here runs on hardware -- the one NPU case is opt-in and skipped unless a device is named.

Local runs of this file use a CPU Torch build; the accepted numbers for the released stack come from
the private environment, so a pass here is a unit result, not a hardware acceptance.
"""

from __future__ import annotations

import math
import random
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("relic_core.kernels.block_gemm")
pytest.importorskip("relic_core.kernels.block_fp8_quant")

from relic_core.kernels.block_fp8_quant import quantize_block_fp8  # noqa: E402
from relic_core.kernels.block_gemm import fp4_int8_block_gemm  # noqa: E402
from relic_core.kernels.unpack import fp4_int8_scale, unpack_fp4_to_int8  # noqa: E402

from relicllm.models.deepseek_v4_1 import ascend_experts, mma  # noqa: E402
from relicllm.models.deepseek_v4_1.ascend_experts import (  # noqa: E402
    DEFAULT_CACHE_BYTES,
    AscendExpertCache,
    AscendRoutedExperts,
    ResidentExpert,
    expert_forward_ascend,
)
from relicllm.models.deepseek_v4_1.kernels import block_fp8_act_quant  # noqa: E402
from relicllm.models.deepseek_v4_1.loader import scale_key  # noqa: E402
from relicllm.models.deepseek_v4_1.modules import expert_forward  # noqa: E402

BLOCK = 32
DIM = 64
INTER = 96
TOPK = 2
N_EXPERTS = 4
DTYPE = torch.bfloat16

#: E2M1 magnitudes in code order -- what a packed nibble means, independent of any operator.
E2M1_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)

#: The INT8-times-two widening the store relies on, written out from the format.
E2M1_INT8_X2 = tuple(
    (-1 if code & 8 else 1) * int(E2M1_MAGNITUDES[code & 7] * 2) for code in range(16)
)

#: Scale bytes used by the fixtures: ordinary octaves, so no fixture overflows FP32 unless a test is
#: about exactly that. `2 ** (b - 127)` for b in [120, 136) is [2**-7, 2**9).
_SCALE_BYTES = range(120, 136)


def _e2m1_value(code: int) -> float:
    return (-1.0 if code & 8 else 1.0) * E2M1_MAGNITUDES[code & 7]


def _e8m0_value(byte: int) -> float:
    return math.ldexp(1.0, byte - 127) if byte < 255 else float("nan")


def _halved_e8m0_value(byte: int) -> float:
    """The same scale halved: the compensation for an INT8-times-two weight."""
    return math.ldexp(1.0, byte - 128) if byte < 255 else float("nan")


def _e8m0_dtype() -> torch.dtype:
    dtype = getattr(torch, "float8_e8m0fnu", None)
    return torch.uint8 if dtype is None else dtype


def _as_e8m0(scale_bytes: torch.Tensor) -> torch.Tensor:
    """The stored E8M0 container, as the released checkpoint holds it (one byte per block)."""
    dtype = _e8m0_dtype()
    return scale_bytes if dtype is torch.uint8 else scale_bytes.view(dtype)


def _nibbles(packed: torch.Tensor) -> torch.Tensor:
    """The packed bytes' codes along K: low nibble first, then the high one."""
    low = (packed & 15).to(torch.int64)
    high = ((packed >> 4) & 15).to(torch.int64)
    return torch.stack((low, high), dim=-1).flatten(-2)


class FakeCheckpoint:
    """A read-only checkpoint over hand-built original bytes, exposing only what the store reads.

    The shapes are the released ones' relationships, not their sizes: a projection is `[out, in/2]`
    of packed nibbles beside `[out, in/32]` of E8M0 bytes. Scale bytes are drawn from ordinary
    octaves so the fixture stays inside FP32, and `scales=` overrides them for edge cases.
    """

    def __init__(
        self,
        *,
        root: str = "fixture://layer0",
        dim: int = DIM,
        inter: int = INTER,
        experts: int = N_EXPERTS,
        layers: tuple[int, ...] = (0,),
        seed: int = 0,
        block: int = BLOCK,
        scale_value: int | None = None,
        fp4: bool = True,
        store_scale: bool = True,
    ) -> None:
        self.root = root
        self.dim = dim
        self.inter = inter
        self.experts = experts
        self.layers = tuple(layers)
        self.block = block
        self._fp4 = fp4
        self._store_scale = store_scale
        self._tensors: dict[str, torch.Tensor] = {}
        generator = random.Random(seed)
        for layer in self.layers:
            for expert in range(experts):
                for which in ("w1", "w2", "w3"):
                    out, inn = (dim, inter) if which == "w2" else (inter, dim)
                    key = f"layers.{layer}.ffn.experts.{expert}.{which}.weight"
                    packed = torch.tensor(
                        [generator.randrange(256) for _ in range(out * (inn // 2))],
                        dtype=torch.uint8,
                    ).reshape(out, inn // 2)
                    self._tensors[key] = packed
                    if not store_scale:
                        continue
                    blocks = inn // block
                    if scale_value is None:
                        drawn = [generator.choice(_SCALE_BYTES) for _ in range(out * blocks)]
                    else:
                        drawn = [scale_value] * (out * blocks)
                    raw = torch.tensor(drawn, dtype=torch.uint8).reshape(out, blocks)
                    self._tensors[scale_key(key)] = _as_e8m0(raw)

    # -- the four methods `fp4_expert_int8` reads ------------------------------

    def packed(self, key: str) -> torch.Tensor:
        return self._tensors[key]

    def is_quantized(self, key: str) -> bool:
        try:
            sibling = scale_key(key)
        except ValueError:
            return False
        return sibling in self._tensors

    def is_packed_fp4(self, key: str) -> bool:
        return self._fp4

    def block_size(self, key: str) -> int:
        return self.block

    # -- what the tests read --------------------------------------------------

    def scale_bytes(self, key: str) -> torch.Tensor:
        """The stored E8M0 bytes as a `uint8` view, for the independent oracle."""
        return self._tensors[scale_key(key)].view(torch.uint8)

    def dense(self, key: str) -> torch.Tensor:
        """The original floating weight this fixture encodes, computed here from the format."""
        packed = self._tensors[key]
        scale = self.scale_bytes(key)
        codes = _nibbles(packed)
        values = torch.tensor([_e2m1_value(code) for code in range(16)], dtype=torch.float64)
        decoded = torch.tensor([_e8m0_value(int(byte)) for byte in range(255)] + [float("nan")],
                              dtype=torch.float64)
        grid = decoded[scale.reshape(-1).to(torch.int64)].reshape(scale.shape)
        widened = grid.repeat_interleave(self.block, dim=-1)
        return (values[codes] * widened).to(DTYPE)


def _store(checkpoint=None, *, layer: int = 0, cache=None, bucket: int | None = None,
           world: int = 1, rank: int = 0, topk: int = TOPK, swiglu_limit: float = 10.0,
           device: str = "cpu", dim: int = DIM, inter: int = INTER,
           experts: int = N_EXPERTS, dtype: torch.dtype = DTYPE,
           backend: str = "torch") -> AscendRoutedExperts:
    checkpoint = FakeCheckpoint() if checkpoint is None else checkpoint
    if cache is None:
        cache = AscendExpertCache(0 if bucket is None else bucket)
    return AscendRoutedExperts(
        checkpoint,
        layer,
        n_experts=experts,
        dim=dim,
        inter_dim=inter,
        topk=topk,
        swiglu_limit=swiglu_limit,
        device=device,
        dtype=dtype,
        rank=rank,
        world=world,
        cache=cache,
        backend=backend,
    )


def _transcribe(
    x: torch.Tensor,
    resident: ResidentExpert,
    *,
    swiglu_limit: float = 0.0,
    weights: torch.Tensor | None = None,
    dtype: torch.dtype = DTYPE,
    backend: str = "torch",
    narrow: bool = True,
    clamp_before_silu: bool = True,
    weight_on_intermediate: bool = True,
    cast_before_w2_quant: bool = True,
) -> torch.Tensor:
    """The reference's expert arithmetic, spelled out here with the plausible mistakes switchable.

    This is the specification side of the wiring tests. It calls the same operators, so a mismatch is
    about *order*, *placement* or *width* -- which is exactly what a store can get wrong while every
    individual operator is right.
    """

    def project(rows: torch.Tensor, projection) -> torch.Tensor:
        codes, scales = quantize_block_fp8(rows, backend=backend)
        out = fp4_int8_block_gemm(
            codes, scales, projection.values, projection.half_scale, backend=backend
        )
        return out.to(dtype).float() if narrow else out

    gate = project(x, resident.w1)
    up = project(x, resident.w3)
    if clamp_before_silu:
        if swiglu_limit > 0:
            up = torch.clamp(up, min=-swiglu_limit, max=swiglu_limit)
            gate = torch.clamp(gate, max=swiglu_limit)
        out = torch.nn.functional.silu(gate) * up
    else:
        out = torch.nn.functional.silu(gate) * up
        if swiglu_limit > 0:
            out = torch.clamp(out, min=-swiglu_limit, max=swiglu_limit)
    if weights is not None:
        out = weights * out if weight_on_intermediate else out
    operand = out.to(dtype) if cast_before_w2_quant else out
    codes, scales = quantize_block_fp8(operand, backend=backend)
    result = fp4_int8_block_gemm(
        codes, scales, resident.w2.values, resident.w2.half_scale, backend=backend
    )
    w2 = result.to(dtype)
    return w2 if weight_on_intermediate or weights is None else (weights * w2.float())


def _indices_from(table: list[list[int]]) -> torch.Tensor:
    """`[[e0, e1], ...]` -- one row of expert ids per row -- as the gate's int64 output."""
    return torch.tensor(table, dtype=torch.int64)


def _routing_weights(table: list[list[float]]) -> torch.Tensor:
    return torch.tensor(table, dtype=torch.float32)


# --------------------------------------------------------------------------------------------
# The operand mapping: INT8-times-two values beside the halved grid.
# --------------------------------------------------------------------------------------------


def test_widened_operands_reconstruct_the_original_floating_weight_exactly():
    """`int8 * (scale / 2)` is the checkpoint's value, bit for bit, over every code and octave."""
    packed = torch.arange(256, dtype=torch.int32).to(torch.uint8).reshape(1, 256)
    values = unpack_fp4_to_int8(packed, backend="torch").reshape(-1).tolist()
    # Low nibble is the even K element, high nibble the odd one -- the order the store's widening,
    # the packed format and the released kernel all agree on.
    expected = []
    for byte in range(256):
        expected.extend((E2M1_INT8_X2[byte & 15], E2M1_INT8_X2[(byte >> 4) & 15]))
    assert values == expected, "the INT8 view is not exactly twice the E2M1 value"

    for byte in (0, 1, 2, 64, 127, 200, 254):
        raw = torch.tensor([[byte]], dtype=torch.uint8)
        half = fp4_int8_scale(raw, backend="torch").reshape(-1)[0].item()
        if byte == 255:  # pragma: no cover - 255 is NaN and not in the loop's range
            continue
        assert math.isclose(half, _halved_e8m0_value(byte), rel_tol=0.0, abs_tol=0.0), (
            f"scale byte {byte} halved to {half}, not {_halved_e8m0_value(byte)}"
        )
        assert torch.tensor(half).view(torch.int32).item() == torch.tensor(
            _halved_e8m0_value(byte)
        ).view(torch.int32).item(), "the halved scale's FP32 bits are not the exact power of two"

    # The one edge the E8M0 byte cannot spell: 2**-128 is an FP32 subnormal, and it is preserved.
    zero = fp4_int8_scale(torch.zeros((1, 1), dtype=torch.uint8), backend="torch")
    assert zero.reshape(-1)[0].view(torch.int32).item() == 0x00200000, (
        "scale byte 0 must be 2**-128 (0x00200000), the subnormal the halved grid needs"
    )
    assert math.isnan(fp4_int8_scale(torch.full((1, 1), 255, dtype=torch.uint8),
                                    backend="torch").reshape(-1)[0].item())

    # And the reconstruction itself, in FP32: the doubled integer times the halved scale is the
    # original nibble times the original scale, bitwise, for every code and a spread of octaves.
    # E2M1's negative zero is the one value the INT8 view cannot carry: integer zero has one
    # encoding, so the reconstruction is `+0` there while the file encodes `-0`. That is the
    # documented loss, pinned rather than papered over.
    integer = torch.tensor([[float(E2M1_INT8_X2[code]) for code in range(16)]])
    plain = torch.tensor([[_e2m1_value(code) for code in range(16)]])
    live = [code for code in range(16) if code != 8]  # code 8 is E2M1 -0
    for byte in (0, 1, 2, 64, 127, 200, 254):
        doubled = integer[:, live] * torch.tensor(_halved_e8m0_value(byte), dtype=torch.float32)
        original = plain[:, live] * torch.tensor(_e8m0_value(byte), dtype=torch.float32)
        assert torch.equal(doubled.view(torch.int32), original.view(torch.int32)), (
            f"scale byte {byte}: doubled value x halved scale is not the original product"
        )
    negative_zero = plain[0, 8] * torch.tensor(_e8m0_value(127), dtype=torch.float32)
    widened_zero = integer[0, 8] * torch.tensor(_halved_e8m0_value(127), dtype=torch.float32)
    assert negative_zero.view(torch.int32).item() == torch.tensor(-0.0).view(torch.int32).item()
    assert widened_zero.view(torch.int32).item() == 0, (
        "the documentation says INT8 cannot carry E2M1's negative zero; it did"
    )


def test_the_store_reports_int8_times_two_values_and_the_halved_grid():
    """The cache holds INT8 codes and FP32 half-scales -- never a dense widening."""
    checkpoint = FakeCheckpoint()
    store = _store(checkpoint)
    resident = store.expert(1)
    assert isinstance(resident, ResidentExpert)
    assert resident.block == BLOCK
    for which, (out, inn) in (("w1", (INTER, DIM)), ("w2", (DIM, INTER)), ("w3", (INTER, DIM))):
        projection = getattr(resident, which)
        assert tuple(projection.values.shape) == (out, inn)
        assert projection.values.dtype == torch.int8
        assert tuple(projection.half_scale.shape) == (out, inn // BLOCK)
        assert projection.half_scale.dtype == torch.float32
        # The values are the file's own codes widened, not a requantization.
        widened = projection.values.to(torch.int64).reshape(-1).tolist()
        assert set(widened) <= set(E2M1_INT8_X2), f"{which} holds values outside the codebook"

    expected_bytes = sum(
        getattr(resident, which).values.numel()
        + 4 * getattr(resident, which).half_scale.numel()
        for which in ("w1", "w2", "w3")
    )
    assert resident.nbytes == expected_bytes
    # A dense BF16 widening of the same expert would be twice the INT8 payload plus the grids.
    dense_bytes = sum(
        2 * getattr(resident, which).values.numel() for which in ("w1", "w2", "w3")
    ) + 4 * sum(getattr(resident, which).half_scale.numel() for which in ("w1", "w2", "w3"))
    assert resident.nbytes < dense_bytes, "the resident form is not smaller than a dense BF16 one"


# --------------------------------------------------------------------------------------------
# The forward: order, placement and width.
# --------------------------------------------------------------------------------------------


def test_forward_matches_the_reference_shaped_transcription_bitwise():
    """One row per expert, so only one contribution is in play: the store must match exactly."""
    checkpoint = FakeCheckpoint()
    store = _store(checkpoint, topk=1, bucket=None)
    torch.manual_seed(4)
    x = torch.randn(3, DIM, dtype=DTYPE)
    indices = _indices_from([[0], [1], [2]])
    weights = _routing_weights([[0.5], [1.25], [2.0]])
    store_output = store.forward(x, weights, indices)
    expected = torch.zeros_like(store_output)
    for row, expert in enumerate(indices.reshape(-1).tolist()):
        contribution = _transcribe(
            x[row: row + 1],
            store.expert(expert),
            swiglu_limit=store.swiglu_limit,
            weights=weights[row: row + 1],
        )
        expected[row] = contribution.to(torch.float32)
    assert torch.equal(store_output, expected), "the store's forward is not the reference's order"


def test_the_gemm_result_is_narrowed_to_the_store_dtype_before_the_clamp(monkeypatch):
    """The reference's GEMM outputs BF16 and its clamp runs on that narrowed value in FP32.

    Narrowing first and widening later is not cosmetic, but it is also not always *visible* in the
    final answer: the SwiGLU result is cast to BF16 before the `w2` quantization, which absorbs a
    one-ulp difference most of the time. So the assertion is on the value that reaches `silu` --
    every element of it must be exactly representable in the store's dtype -- rather than on a
    downstream difference that a fixture would have to be tuned to produce.
    """
    checkpoint = FakeCheckpoint()
    store = _store(checkpoint, topk=1, bucket=None)
    torch.manual_seed(5)
    x = torch.randn(2, DIM, dtype=DTYPE)
    indices = _indices_from([[0], [3]])
    weights = _routing_weights([[1.0], [1.0]])
    seen: list[torch.Tensor] = []
    real_silu = torch.nn.functional.silu

    def recording_silu(gate):
        seen.append(gate.detach().clone())
        return real_silu(gate)

    monkeypatch.setattr(ascend_experts, "F", SimpleNamespace(silu=recording_silu))
    store_row = store.forward(x, weights, indices)[:1]
    monkeypatch.undo()
    assert seen, "the forward never reached silu"
    for gate in seen:
        assert gate.dtype == torch.float32, "the clamp/silu operand is not the widened accumulator"
        assert torch.equal(gate, gate.to(DTYPE).float()), (
            "the gate accumulator reaches silu without being narrowed to the store's dtype and "
            "re-widened, which is not what the reference's BF16 GEMM hands its clamp"
        )
    # And the value that comes out is the reference-shaped transcription's, bit for bit.
    narrowed = _transcribe(x[:1], store.expert(0), swiglu_limit=store.swiglu_limit,
                           weights=weights[:1], narrow=True)
    assert torch.equal(store_row, narrowed.to(torch.float32))


def test_the_clamp_runs_before_silu_in_the_accumulator_that_reaches_it():
    """`swiglu_limit` binds on this release; clamping after `silu` is a different expert."""
    checkpoint = FakeCheckpoint(scale_value=127)  # scale 2**0: activations exceed the limit
    store = _store(checkpoint, topk=1, bucket=None, swiglu_limit=0.5)
    torch.manual_seed(6)
    x = torch.randn(1, DIM, dtype=DTYPE) * 4.0
    indices = _indices_from([[0]])
    weights = _routing_weights([[1.0]])
    resident = store.expert(0)
    before = _transcribe(x, resident, swiglu_limit=0.5, weights=weights, clamp_before_silu=True)
    after = _transcribe(x, resident, swiglu_limit=0.5, weights=weights, clamp_before_silu=False)
    assert not torch.equal(before, after), (
        "the clamp does not bind on this fixture, so the order it runs in is unobservable here"
    )
    assert torch.equal(store.forward(x, weights, indices), before.to(torch.float32))


def test_the_routing_weight_multiplies_the_intermediate_not_the_expert_output():
    checkpoint = FakeCheckpoint()
    store = _store(checkpoint, topk=1, bucket=None)
    torch.manual_seed(7)
    x = torch.randn(1, DIM, dtype=DTYPE)
    indices = _indices_from([[2]])
    # Not a power of two: a power-of-two weight commutes with the intermediate quantization
    # exactly, so it could not tell the two placements apart.
    weights = _routing_weights([[0.3]])
    resident = store.expert(2)
    intermediate = _transcribe(x, resident, swiglu_limit=store.swiglu_limit, weights=weights,
                               weight_on_intermediate=True)
    output_side = _transcribe(x, resident, swiglu_limit=store.swiglu_limit, weights=weights,
                              weight_on_intermediate=False)
    assert not torch.equal(intermediate, output_side), (
        "scaling the intermediate and scaling the output agree on this fixture, so the placement "
        "of the routing weight is not pinned"
    )
    assert torch.equal(store.forward(x, weights, indices), intermediate.to(torch.float32))


def test_the_w2_operand_is_the_dtype_cast_and_is_quantized_after_it():
    checkpoint = FakeCheckpoint()
    store = _store(checkpoint, topk=1, bucket=None)
    torch.manual_seed(8)
    x = torch.randn(1, DIM, dtype=DTYPE)
    indices = _indices_from([[1]])
    weights = _routing_weights([[1.0]])
    resident = store.expert(1)
    cast = _transcribe(x, resident, swiglu_limit=store.swiglu_limit, weights=weights,
                       cast_before_w2_quant=True)
    uncast = _transcribe(x, resident, swiglu_limit=store.swiglu_limit, weights=weights,
                         cast_before_w2_quant=False)
    assert not torch.equal(cast, uncast), (
        "quantizing the FP32 intermediate and quantizing its BF16 cast agree on this fixture"
    )
    assert torch.equal(store.forward(x, weights, indices), cast.to(torch.float32))


def test_each_activation_quantization_is_independent_and_identical(monkeypatch):
    """`w1` and `w3` are two linears: two quantizations, deterministic, byte-identical."""
    real_quantize, real_gemm = quantize_block_fp8, fp4_int8_block_gemm
    recorded: list[torch.Tensor] = []

    def spy_quantize(x, backend="auto", out=None):
        codes, scales = real_quantize(x, backend=backend, out=out)
        recorded.append(codes.clone())
        return codes, scales

    monkeypatch.setattr(
        ascend_experts,
        "_core",
        lambda: (
            SimpleNamespace(quantize_block_fp8=spy_quantize, supports_triton_ascend=lambda: False),
            SimpleNamespace(fp4_int8_block_gemm=real_gemm),
        ),
    )
    store = _store(topk=1, bucket=None)
    torch.manual_seed(9)
    x = torch.randn(2, DIM, dtype=DTYPE)
    store.forward(x, _routing_weights([[1.0], [1.0]]), _indices_from([[0], [0]]))
    # Three linears per expert: w1, w3 and w2.
    assert len(recorded) == 3, f"expected three quantizations per expert, saw {len(recorded)}"
    assert torch.equal(recorded[0], recorded[1]), (
        "the w1 and w3 activations were not quantized to the same bytes"
    )
    assert recorded[2].shape == (2, INTER), "the w2 activation was not quantized at the inter width"


def test_the_activation_stays_fp8_bytes_and_the_weight_stays_int8(monkeypatch):
    """No INT8 activation, no dense weight: the operand dtypes are the contract."""
    real_quantize, real_gemm = quantize_block_fp8, fp4_int8_block_gemm
    seen: list[tuple] = []

    def spy_gemm(a, sa, b, sb, *, backend="auto", out=None):
        seen.append((a.dtype, sa.dtype, b.dtype, sb.dtype))
        return real_gemm(a, sa, b, sb, backend=backend, out=out)

    monkeypatch.setattr(
        ascend_experts,
        "_core",
        lambda: (
            SimpleNamespace(quantize_block_fp8=real_quantize, supports_triton_ascend=lambda: False),
            SimpleNamespace(fp4_int8_block_gemm=spy_gemm),
        ),
    )
    store = _store(topk=1, bucket=None)
    torch.manual_seed(10)
    store.forward(torch.randn(1, DIM, dtype=DTYPE), _routing_weights([[1.0]]),
                  _indices_from([[0]]))
    assert seen, "the forward ran no GEMM"
    for activation_codes, activation_scale, weight_values, weight_scale in seen:
        assert activation_codes == torch.uint8, "the activation is not e4m3 bytes"
        assert activation_scale == torch.uint8, "the activation grid is not E8M0 bytes"
        assert weight_values == torch.int8, "the weight is not the exact INT8-times-two view"
        assert weight_scale == torch.float32, "the weight grid is not the halved FP32 grid"


def test_forward_accumulates_experts_in_ascending_id_order(monkeypatch):
    """The reference walks expert ids ascending; the loop order is pinned by recording it."""
    store = _store(FakeCheckpoint(experts=6), topk=3, bucket=None, experts=6)
    order: list[int] = []
    real_expert = AscendRoutedExperts.expert

    def recording(self, expert):
        order.append(expert)
        return real_expert(self, expert)

    monkeypatch.setattr(AscendRoutedExperts, "expert", recording)
    torch.manual_seed(11)
    x = torch.randn(2, DIM, dtype=DTYPE)
    indices = _indices_from([[5, 1, 3], [4, 0, 2]])
    weights = _routing_weights([[0.4, 0.6, 1.0], [1.0, 0.5, 0.25]])
    output = store.forward(x, weights, indices)
    assert order == sorted(order), f"experts were walked as {order}, not in ascending id order"
    assert output.dtype == torch.float32, "the routed sum is not FP32"


def test_forward_agrees_with_the_ascending_sum_of_per_expert_contributions():
    store = _store(FakeCheckpoint(experts=6), topk=3, bucket=None, experts=6)
    torch.manual_seed(12)
    x = torch.randn(3, DIM, dtype=DTYPE)
    indices = _indices_from([[5, 1, 3], [4, 0, 2], [3, 5, 4]])
    weights = _routing_weights([[0.4, 0.6, 1.0], [1.0, 0.5, 0.25], [0.9, 0.1, 0.7]])
    output = store.forward(x, weights, indices)
    expected = torch.zeros_like(output)
    table = indices.tolist()
    for expert in sorted({expert for row in table for expert in row}):
        rows = [row for row, routing in enumerate(table) if expert in routing]
        slots = [routing.index(expert) for routing in table if expert in routing]
        row_index = torch.tensor(rows, dtype=torch.int64)
        slot_index = torch.tensor(slots, dtype=torch.int64)
        routing_weight = weights.index_select(0, row_index).gather(1, slot_index[:, None])
        contribution = expert_forward_ascend(
            x.index_select(0, row_index),
            store.expert(expert),
            swiglu_limit=store.swiglu_limit,
            weights=routing_weight,
            dtype=DTYPE,
            backend="torch",
        )
        expected.index_put_(
            (row_index,), expected.index_select(0, row_index) + contribution.to(torch.float32)
        )
    assert torch.equal(output, expected)


def test_reordering_the_gate_s_slots_does_not_change_the_result():
    """Routing weight and expert id travel together, so the gate's column order is irrelevant."""
    store = _store(topk=2, bucket=None, experts=4, cache=AscendExpertCache(0))
    torch.manual_seed(13)
    x = torch.randn(2, DIM, dtype=DTYPE)
    first = store.forward(
        x, _routing_weights([[0.25, 0.75], [1.0, 0.5]]), _indices_from([[3, 0], [1, 2]])
    )
    swapped = store.forward(
        x, _routing_weights([[0.75, 0.25], [0.5, 1.0]]), _indices_from([[0, 3], [2, 1]])
    )
    assert torch.equal(first, swapped), "the store depends on the gate's slot order"


# --------------------------------------------------------------------------------------------
# The activation quantizer, the reduction, and the port's existing oracle path.
# --------------------------------------------------------------------------------------------


def test_the_activation_quantizer_agrees_with_the_runtime_cpu_oracle_on_the_bf16_grid():
    """The official spelling (reciprocal, bit ceiling) and the port's (divide, log2/ceil) agree.

    They are not the same function in general -- the port's `2 ** ceil(log2(amax / 448))` can
    under-count an octave from 2**4 upward, and the reciprocal rounds -- but the activation is BF16
    and its grid is ~4.5e-3 relative around `448 * 2**k`, far coarser than the ~1e-7 window where the
    two spellings differ. That is the documented reason the store may run the released spelling
    while the port's oracle path keeps its own, and this test is where the agreement is pinned.
    """
    grid = [0.0, 1e-4, 448.0, 448.0 * 2.0 ** -3, 448.0 * 2.0 ** 6, 3.0e38]
    values = list(grid)
    generator = random.Random(21)
    values.extend(generator.uniform(-1e4, 1e4) for _ in range(90))  # 96 values: three blocks
    x = torch.tensor(values, dtype=torch.float32).to(DTYPE).reshape(-1, BLOCK)
    codes, scales = quantize_block_fp8(x, backend="torch")
    reference, reference_scale = block_fp8_act_quant(x)
    assert torch.equal(codes, reference.view(torch.uint8)), (
        "the released quantizer's codes disagree with the port's oracle on the BF16 grid"
    )
    expected = torch.tensor(
        [math.ldexp(1.0, int(byte) - 127) if int(byte) < 255 else float("nan")
         for byte in scales.reshape(-1).tolist()],
        dtype=torch.float32,
    )
    assert torch.equal(reference_scale.reshape(-1).view(torch.int32), expected.view(torch.int32))


def test_the_store_matches_the_ports_mma_reduction_path_exactly(monkeypatch):
    """The store and the port's existing reference path agree, projection for projection.

    `relicllm.models.deepseek_v4_1.mma` transcribes the reference GEMM's two-half truncation rule
    and `relic_core.kernels.block_gemm` is a second transcription of it. Running both end to end on
    the same bytes and the same activation quantization is the strongest check available without a
    card: the store's operands, its SwiGLU order and its widths all have to be right for the two to
    land on the same BF16 values.

    What agreement does **not** establish is hardware parity -- the port's own measurement against
    the released kernel is `mma`'s record, not this store's -- and it is stated here so a reader does
    not read this test as one.
    """
    monkeypatch.setenv("POCKETLLM_V41_MMA_EMULATE", "1")
    checkpoint = FakeCheckpoint()
    dense = {}
    for which in ("w1", "w2", "w3"):
        key = f"layers.0.ffn.experts.0.{which}.weight"
        weight = checkpoint.dense(key)
        # Exactly what `CheckpointRoutedExperts._expert_matrix` hangs on the dense weight: the
        # file's own grid, and the e4m3 codes the reference's fp4 kernel casts the nibbles to.
        mma.attach_scale(weight, checkpoint.packed(scale_key(key)))
        mma.attach_codes(weight, mma.e4m3_codes_from_e2m1(checkpoint.packed(key)))
        dense[which] = weight
    store = _store(checkpoint, topk=1, bucket=None)
    torch.manual_seed(22)
    x = torch.randn(16, DIM, dtype=DTYPE)
    weights = torch.full((16, 1), 1.0)
    store_output = store.forward(x, weights, _indices_from([[0]] * 16))
    reference = expert_forward(
        x, dense["w1"], dense["w2"], dense["w3"], store.swiglu_limit, weights, quantized=True
    ).to(torch.float32)
    assert torch.equal(store_output, reference), (
        "the store's expert arithmetic is not the port's reference-reduction path: "
        f"{(store_output != reference).sum().item()} of {reference.numel()} elements differ"
    )
    # And it is not a dense matmul wearing the same weights: without the activation quantization the
    # same three matrices produce a materially different expert, which is what makes the agreement
    # above informative rather than vacuous.
    plain = expert_forward(
        x, dense["w1"], dense["w2"], dense["w3"], store.swiglu_limit, weights, quantized=False
    ).to(torch.float32)
    differing = int((store_output != plain).sum())
    assert differing > plain.numel() // 2, (
        "the store's output is indistinguishable from a dense BF16 expert, so the activation "
        f"quantization is not reaching it ({differing} of {plain.numel()} elements differ)"
    )
    relative = float((store_output - plain).abs().max()) / float(plain.abs().max())
    assert relative > 1e-2, f"the quantized and dense paths differ by only {relative:.3e} relative"


# --------------------------------------------------------------------------------------------
# Expert parallelism.
# --------------------------------------------------------------------------------------------


def test_the_deal_is_cyclic_and_partial_is_decided_once():
    checkpoint = FakeCheckpoint(experts=6)
    whole = _store(checkpoint, topk=1, bucket=0, experts=6, world=1)
    left = _store(checkpoint, topk=1, bucket=0, experts=6, world=2, rank=0)
    right = _store(checkpoint, topk=1, bucket=0, experts=6, world=2, rank=1)
    assert whole.partial is False, "a single-rank store returns the whole routed sum"
    assert left.partial is True and right.partial is True, "a two-rank deal returns one rank's share"
    for expert in range(6):
        assert left._owns(expert) == (expert % 2 == 0)
        assert right._owns(expert) == (expert % 2 == 1)
    with pytest.raises(ValueError, match="not owned by rank 0"):
        left.expert(1)
    assert left.expert(0).w1.values.shape == right.expert(1).w1.values.shape


def test_two_rank_partial_sums_compose_to_the_whole_routed_sum():
    checkpoint = FakeCheckpoint(experts=6)
    whole = _store(checkpoint, topk=1, bucket=0, experts=6, world=1)
    left = _store(checkpoint, topk=1, bucket=0, experts=6, world=2, rank=0)
    right = _store(checkpoint, topk=1, bucket=0, experts=6, world=2, rank=1)
    torch.manual_seed(23)
    x = torch.randn(3, DIM, dtype=DTYPE)
    indices = _indices_from([[0], [3], [5]])
    weights = _routing_weights([[1.0], [0.5], [2.0]])
    expected = whole.forward(x, weights, indices)
    composed = left.forward(x, weights, indices) + right.forward(x, weights, indices)
    # Each row contributes through exactly one rank here, so the FP32 adds are the same adds.
    assert torch.equal(composed, expected), "the two partial sums do not compose to the whole sum"


def test_a_multi_expert_row_composes_across_ranks_within_rounding():
    checkpoint = FakeCheckpoint(experts=6)
    whole = _store(checkpoint, topk=2, bucket=0, experts=6, world=1)
    left = _store(checkpoint, topk=2, bucket=0, experts=6, world=2, rank=0)
    right = _store(checkpoint, topk=2, bucket=0, experts=6, world=2, rank=1)
    torch.manual_seed(24)
    x = torch.randn(2, DIM, dtype=DTYPE)
    indices = _indices_from([[0, 3], [5, 2]])  # every row straddles the two ranks
    weights = _routing_weights([[0.5, 0.5], [1.0, 1.0]])
    expected = whole.forward(x, weights, indices)
    composed = left.forward(x, weights, indices) + right.forward(x, weights, indices)
    # A row split across ranks is added in a different order than the single-rank walk, and FP32
    # addition is not associative; the difference is that rounding and nothing else.
    scale = float(expected.abs().max())
    assert float((composed - expected).abs().max()) <= 4 * abs(scale) * 2.0 ** -23


# --------------------------------------------------------------------------------------------
# The cache.
# --------------------------------------------------------------------------------------------


def _resident_bytes(checkpoint, store) -> int:
    return store.expert(0).nbytes


def test_the_cache_is_a_byte_budgeted_lru():
    checkpoint = FakeCheckpoint()
    probe = _store(checkpoint, cache=AscendExpertCache(0))
    expert_bytes = probe.expert(0).nbytes
    cache = AscendExpertCache(2 * expert_bytes)
    store = _store(checkpoint, cache=cache)
    assert store.expert(0) is not None
    assert store.expert(1) is not None
    assert cache.entries == 2 and cache.retained_bytes == 2 * expert_bytes
    # Touch 0 so 1 becomes the least recently used, then load 2: 1 is what gives way.
    store.expert(0)
    store.expert(2)
    assert cache.entries == 2, "the budget was not enforced"
    assert cache.retained_bytes <= cache.budget_bytes
    assert cache.evictions == 1
    stats = store.cache_stats
    assert stats["hits"] == 1 and stats["misses"] == 3
    assert store.resident_experts == 2


def test_the_cache_streams_an_expert_larger_than_the_budget_instead_of_evicting():
    checkpoint = FakeCheckpoint()
    probe = _store(checkpoint, cache=AscendExpertCache(0))
    expert_bytes = probe.expert(0).nbytes
    cache = AscendExpertCache(expert_bytes - 1)
    store = _store(checkpoint, cache=cache)
    resident = store.expert(0)
    assert resident is not None, "an oversized expert was not available at all"
    assert cache.entries == 0, "an expert bigger than the budget was retained"
    assert cache.streamed == 1
    assert cache.evictions == 0, "streaming an oversized expert evicted something else"
    assert store.resident_experts == 0
    # The value is still the same one the cache would have held.
    cached = _store(checkpoint, cache=AscendExpertCache(8 * expert_bytes)).expert(0)
    assert torch.equal(resident.w1.values, cached.w1.values)


def test_cache_keys_isolate_checkpoint_layer_and_device():
    first = FakeCheckpoint(root="fixture://a", seed=1)
    second = FakeCheckpoint(root="fixture://b", seed=2)
    layered = FakeCheckpoint(root="fixture://a", seed=1, layers=(0, 1))
    cache = AscendExpertCache(8 * 1024 * 1024)
    one = _store(first, cache=cache)
    two = _store(second, cache=cache)
    other_layer = _store(layered, layer=1, cache=cache)
    # Different files, same (layer, expert): different bytes.
    assert not torch.equal(one.expert(0).w1.values, two.expert(0).w1.values), (
        "two checkpoints shared a cache entry, which is the layer-6-reads-layer-5 bug's cousin"
    )
    # Same file, same expert id, different layer: different bytes.
    assert not torch.equal(one.expert(0).w1.values, other_layer.expert(0).w1.values), (
        "one checkpoint's layer 0 served layer 1's bytes"
    )
    assert cache.entries == 3
    # The device is part of the key even when the weights are not: an entry resident on one device is
    # not usable by a store on another.
    assert one._cache_key(0)[2] == ("cpu", None)
    default_index = _store(first, cache=cache, device="cpu:0")
    assert default_index._cache_key(0) != one._cache_key(0)


def test_clearing_one_store_leaves_the_other_stores_resident():
    checkpoint = FakeCheckpoint(layers=(0, 1))
    cache = AscendExpertCache(8 * 1024 * 1024)
    first = _store(checkpoint, layer=0, cache=cache)
    second = _store(checkpoint, layer=1, cache=cache)
    first.expert(0)
    second.expert(0)
    assert cache.entries == 2
    assert first.clear_cache() == 1
    assert first.resident_experts == 0
    assert second.resident_experts == 1, "clearing one layer's entries dropped another layer's"
    assert cache.clear() == 1


def test_the_default_cache_is_one_shared_byte_budget():
    assert DEFAULT_CACHE_BYTES == 512 * 1024 * 1024, (
        "the default budget is a documented number: 512 MiB shared by every store in the process"
    )
    shared = ascend_experts.global_cache()
    assert shared is ascend_experts.global_cache()
    bare = AscendRoutedExperts(
        FakeCheckpoint(), 0, n_experts=N_EXPERTS, dim=DIM, inter_dim=INTER, topk=TOPK,
        device="cpu", backend="torch",
    )
    assert bare.cache is shared, (
        "a store built without a cache did not join the process-wide one"
    )
    assert shared.budget_bytes == DEFAULT_CACHE_BYTES


# --------------------------------------------------------------------------------------------
# Refusals, by name.
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "factory, message",
    [
        (lambda: _store(dim=48), "multiple"),
        (lambda: _store(inter=100), "multiple"),
        (lambda: _store(topk=0), "positive"),
        (lambda: _store(topk=N_EXPERTS + 1), "exceeds"),
        (lambda: _store(dtype=torch.float16), "quantizer"),
        (lambda: _store(backend="triton"), "needs NPU tensors"),
        (lambda: _store(backend="cuda"), "one of"),
        (lambda: _store(world=0), "positive"),
        (lambda: _store(world=2, rank=2), r"rank must be an int in \[0, 2\)"),
        (lambda: _store(swiglu_limit=-1), "non-negative"),
        (lambda: _store(device="cuda"), "Ascend expert store"),
        (lambda: _store(cache=object()), "AscendExpertCache"),
    ],
)
def test_construction_refuses_arguments_it_cannot_honour(factory, message):
    with pytest.raises((ValueError, TypeError), match=message):
        factory()


def test_forward_refuses_inputs_it_cannot_consume():
    store = _store(topk=2, bucket=0)
    good_x = torch.randn(2, DIM, dtype=DTYPE)
    good_indices = _indices_from([[0, 1], [2, 3]])
    good_weights = _routing_weights([[0.5, 0.5], [0.5, 0.5]])
    cases = [
        (torch.randn(2, DIM, dtype=torch.float32), good_weights, good_indices, "activation at"),
        (torch.randn(3, DIM, dtype=DTYPE), good_weights, good_indices, "rows"),
        (torch.randn(2, DIM // 2, dtype=DTYPE), good_weights, good_indices, "wide"),
        (torch.randn(2, DIM, 1, dtype=DTYPE), good_weights, good_indices, "dimensional"),
        (good_x, _routing_weights([[0.5], [0.5]]), _indices_from([[0], [1]]), "topk=2"),
        (good_x, good_weights, torch.tensor([[0.0, 1.0], [2.0, 3.0]]), "integer"),
        (good_x, good_weights.to(torch.int64), good_indices, "floating"),
        (good_x, _routing_weights([[0.5, 0.5]]), good_indices, "disagree"),
        (good_x, good_weights, _indices_from([[0, 4], [2, 3]]), "this layer has"),
        (good_x, good_weights, _indices_from([[-1, 1], [2, 3]]), "this layer has"),
        (good_x, good_weights, _indices_from([[1, 1], [2, 3]]), "repeat"),
    ]
    for x, weights, indices, message in cases:
        with pytest.raises((ValueError, TypeError, RuntimeError), match=message):
            store.forward(x, weights, indices)


def test_a_device_mismatch_is_refused_by_name():
    store = _store(topk=1, bucket=0)
    x = torch.zeros(1, DIM, dtype=DTYPE, device="meta")
    with pytest.raises(RuntimeError, match="activation on"):
        store.forward(x, _routing_weights([[1.0]]), _indices_from([[0]]))


def test_a_checkpoint_that_is_not_packed_fp4_is_refused():
    store_unpacked = _store(FakeCheckpoint(fp4=False), bucket=0)
    with pytest.raises(ValueError, match="not stored as packed fp4"):
        store_unpacked.expert(0)
    store_unscaled = _store(FakeCheckpoint(store_scale=False), bucket=0)
    with pytest.raises(ValueError, match="no scale stored beside it"):
        store_unscaled.expert(0)
    store_wrong_block = _store(FakeCheckpoint(block=16), bucket=0)
    with pytest.raises(ValueError, match="16-wide K block"):
        store_wrong_block.expert(0)


def test_an_expert_matrix_that_does_not_match_the_layer_is_refused():
    checkpoint = FakeCheckpoint(inter=INTER)
    store = _store(checkpoint, inter=INTER * 2, bucket=0)  # layer built at the wrong inter width
    with pytest.raises(ValueError, match="does not describe the layer"):
        store.expert(0)


def test_a_forward_with_no_rows_is_an_empty_fp32_tensor():
    store = _store(topk=2, bucket=0)
    output = store.forward(
        torch.zeros(0, DIM, dtype=DTYPE),
        torch.zeros(0, 2, dtype=torch.float32),
        torch.zeros(0, 2, dtype=torch.int64),
    )
    assert tuple(output.shape) == (0, DIM)
    assert output.dtype == torch.float32


# --------------------------------------------------------------------------------------------
# Opt-in: the same store on a real device.
# --------------------------------------------------------------------------------------------


def test_the_npu_store_matches_the_cpu_oracle(monkeypatch):
    """Opt-in. Requires a leased Ascend device; not run by the suite and not evidence on its own."""
    import os

    device = os.environ.get("RELIC_ASCEND_EXPERTS_DEVICE")
    if not device:
        pytest.skip("set RELIC_ASCEND_EXPERTS_DEVICE=npu:0 to run the device case")
    import torch_npu  # noqa: F401  (registers the device)
    from relic_core.kernels.block_fp8_quant import supports_triton_ascend

    if not supports_triton_ascend():
        pytest.skip("no compatible Triton Ascend backend installed")
    checkpoint = FakeCheckpoint()
    cpu_store = _store(checkpoint, topk=1, bucket=0, device="cpu", backend="torch")
    device_store = _store(checkpoint, topk=1, bucket=0, device=device, backend="triton")
    torch.manual_seed(31)
    x = torch.randn(2, DIM, dtype=DTYPE)
    indices = _indices_from([[0], [3]])
    weights = _routing_weights([[0.8], [1.3]])
    expected = cpu_store.forward(x, weights, indices)
    actual = device_store.forward(x.to(device), weights.to(device), indices.to(device)).to("cpu")
    scale = float(expected.abs().max())
    assert float((actual - expected).abs().max()) <= 4 * scale * 2.0 ** -23, (
        "the device store and the CPU oracle disagree by more than the reduction's rounding"
    )

def test_the_full_cover_add_is_the_scatter_it_replaces():
    """The batch-one decode accumulation, held against the index_put_ form it short-circuits.

    `forward` adds each expert's contribution into one FP32 accumulator. When an expert took every
    row it now adds elementwise instead of scattering through an index tensor, because with unique
    ascending rows a full cover writes each row exactly once from the same value. This test builds
    the scatter by hand for the same inputs and asserts the two are bit-equal, so the shortcut cannot
    drift into being a different function.
    """
    from relicllm.models.deepseek_v4_1.ascend_experts import expert_forward_ascend

    store = _store(FakeCheckpoint(experts=6), topk=3, bucket=None, experts=6)
    torch.manual_seed(21)
    x = torch.randn(1, DIM, dtype=DTYPE)
    indices = _indices_from([[1, 4, 2]])
    weights = _routing_weights([[0.3, 0.9, 0.5]])

    got = store.forward(x, weights, indices)

    # the same accumulation, written out with the index tensor that the shortcut removes
    table = indices.tolist()
    expected = torch.zeros_like(got)
    for expert in sorted({e for row in table for e in row}):
        rows = [r for r, routing in enumerate(table) if expert in routing]
        slots = [routing.index(expert) for routing in table if expert in routing]
        row_index = torch.tensor(rows, dtype=torch.int64)
        slot_index = torch.tensor(slots, dtype=torch.int64)
        routing_weight = weights.index_select(0, row_index).gather(1, slot_index[:, None])
        contribution = expert_forward_ascend(
            x.index_select(0, row_index), store.expert(expert),
            swiglu_limit=store.swiglu_limit, weights=routing_weight,
            dtype=DTYPE, backend="torch")
        expected.index_put_(
            (row_index,), expected.index_select(0, row_index) + contribution.to(torch.float32))

    assert got.dtype == torch.float32
    assert torch.equal(got.view(torch.int32), expected.view(torch.int32)), (
        "the elementwise full-cover add and the index_put_ scatter disagree at FP32 bits")
