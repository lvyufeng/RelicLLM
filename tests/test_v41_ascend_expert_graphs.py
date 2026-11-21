"""The routed-expert recording round against a fake recording backend, on the host.

What this file can establish, and what it cannot
------------------------------------------------
The fake backend records rather than executes: its "graph" is a `_FakeGraph`, and its `replay` cannot
re-run captured **device** kernels. So the test's recording wrapper attaches the recorded body to the
graph object and the fake replay re-executes that body on the CPU. That is the CPU analogue of a
replay, and it is exactly what makes these tests meaningful: the assertions below are about *this
module* -- that it records one graph per resident expert into one shared pool, that a replay reads
the staging buffers rather than the capture-time inputs, that the result buffer is allocated outside
the pool, that the dispatch falls back to the original body for everything it did not record, and
that nothing it captures is a second copy of the weights. None of it is evidence about Ascend's
recorder, replay cost, or the model's tokens; those are the lead's hardware runs.

The eager oracle is the **original** `ascend_packed_experts.packed_expert_forward` with the same
operator and the same arguments, run on the same host tensors. Bit equality against it is the claim
"a replay is the same numbers as the body it recorded", which is the only correctness claim this
module can make on a CPU fixture.

The operator injected here is a same-contract CPU oracle: it takes the packed decode operator's four
operands and returns `[1, N]` FP32, decoding with exact bit arithmetic (the released E2M1 codebook
and the E8M0 byte table) and multiplying naturally. It is *not* the frozen Ascend reduction; it does
not have to be, because every assertion compares two runs of the *same* operator.
"""

from __future__ import annotations

import contextlib
import types

import pytest
import torch

from relicllm.models.deepseek_v4_1 import ascend_expert_graphs, ascend_graphs, ascend_packed_experts
from relicllm.models.deepseek_v4_1.loader import scale_key

BLOCK = ascend_packed_experts.BLOCK
#: The dispatcher as it ships, captured once so a test that leaks an installed round fails loudly.
PRISTINE_DISPATCHER = ascend_packed_experts.packed_expert_forward
#: The released E2M1 codebook, including the negative zero the INT8 widening cannot carry.
_E2M1 = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


# ------------------------------------------------------------------------------------------------
# A same-contract CPU oracle for the packed decode operator
# ------------------------------------------------------------------------------------------------


def _grid(scales: torch.Tensor) -> torch.Tensor:
    """The exact power of two an E8M0 byte names, `2 ** (byte - 127)`."""
    return torch.ldexp(torch.ones_like(scales, dtype=torch.float32), (scales.to(torch.int32) - 127))


def _fp4_exact(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """`[N, K]` exact FP32 values of packed E2M1 bytes under their original grid."""
    rows, packed_columns = int(codes.shape[0]), int(codes.shape[1])
    low = (codes & 0x0F).to(torch.int64)
    high = (codes >> 4).to(torch.int64)
    nibbles = torch.stack((low, high), dim=-1).reshape(rows, packed_columns * 2)
    values = torch.tensor(_E2M1, dtype=torch.float32)[nibbles]
    return values * _grid(scales).repeat_interleave(BLOCK, dim=-1)


def _fp8_exact(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """`[rows, K]` exact FP32 values of raw E4M3 bytes under an E8M0 grid."""
    raw = codes.to(torch.int64)
    sign = (raw >> 7) & 1
    exponent = (raw >> 3) & 15
    mantissa = raw & 7
    significand = torch.where(exponent == 0, mantissa, 8 + mantissa)
    power = torch.where(exponent == 0, -9, exponent - 10).to(torch.int32)
    magnitude = torch.ldexp(significand.to(torch.float32), power)
    magnitude = magnitude * _grid(scales).repeat_interleave(BLOCK, dim=-1)
    values = torch.where(sign != 0, -magnitude, magnitude)
    nan = (raw & 127) == 127
    return torch.where(nan, torch.full_like(values, float("nan")), values)


def cpu_packed_gemv(a_codes, a_scale, b_packed, b_e8m0, *, backend="auto"):
    """The packed decode operator's contract, exactly: four original byte operands, `[rows, N]` FP32."""
    return _fp8_exact(a_codes, a_scale) @ _fp4_exact(b_packed, b_e8m0).transpose(0, 1)


# ------------------------------------------------------------------------------------------------
# A fake recording backend
# ------------------------------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.waited: list[object] = []

    def wait_stream(self, other: "_FakeStream") -> None:
        self.waited.append(other)


class _FakePool:
    """One round's pool handle. A fresh object per `graph_pool_handle` call, so sharing is testable."""

    def __init__(self, index: int) -> None:
        self.index = index


class _FakeGraph:
    """One recording. `body` is attached by the test's recording wrapper; `replay` re-runs it."""

    def __init__(self, backend: "_FakeBackend") -> None:
        self.backend = backend
        self.body = None
        self.replays = 0

    def replay(self) -> None:
        if self.body is None:  # pragma: no cover - the fixture always attaches one
            raise RuntimeError("this fake replay has no recorded body to re-execute")
        self.replays += 1
        self.backend.replays += 1
        self.backend.events.append("replay")
        self.body()


class _FakeCapture:
    def __init__(self, backend: "_FakeBackend", graph: _FakeGraph, pool) -> None:
        self.backend = backend
        self.graph = graph
        self.pool = pool

    def __enter__(self):
        self.backend.captures += 1
        self.backend.events.append("capture")
        self.backend.captured_with.append(self.pool)
        if self.backend.fail_on_capture == self.backend.captures:
            raise RuntimeError("the fake backend refused this capture")
        return self

    def __exit__(self, *exc_info) -> bool:
        # A capture's allocations would come out of the pool. A few bytes is enough for the budget
        # assertions to be about the round's accounting rather than about torch.
        self.backend.allocated += self.backend.allocated_bytes_per_capture
        self.backend.reserved += self.backend.reserved_bytes_per_capture
        return False


class _FakeBackend:
    """The `torch.npu` recording surface, recorded rather than executed."""

    def __init__(
        self,
        *,
        fail_on_capture: int | None = None,
        pool_bytes_per_capture: int = 7,
        allocated_bytes_per_capture: int | None = None,
        reserved_bytes_per_capture: int = 0,
        expose_memory_reserved: bool = False,
    ) -> None:
        self.events: list[str] = []
        self.captures = 0
        self.replays = 0
        self.allocated = 0
        self.reserved = 0
        # `None` keeps the historical behaviour: every capture grows `allocated` by
        # `pool_bytes_per_capture`. A test can instead split the two counters.
        self.allocated_bytes_per_capture = (
            pool_bytes_per_capture if allocated_bytes_per_capture is None else allocated_bytes_per_capture
        )
        self.reserved_bytes_per_capture = reserved_bytes_per_capture
        self.expose_memory_reserved = expose_memory_reserved
        self.synchronize_calls = 0
        self.pool_calls = 0
        self.flushes = 0
        self.fail_on_capture = fail_on_capture
        self.pool_bytes_per_capture = pool_bytes_per_capture
        self.captured_with: list[object] = []
        self.graphs: list[_FakeGraph] = []
        self._current = _FakeStream("current")

    def Stream(self) -> _FakeStream:
        return _FakeStream("side")

    def current_stream(self) -> _FakeStream:
        return self._current

    def stream(self, side: _FakeStream):
        @contextlib.contextmanager
        def run():
            self.events.append(f"stream:{side.name}")
            self.events.append("warm")
            yield

        return run()

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.events.append("sync")

    def NPUGraph(self) -> _FakeGraph:
        graph = _FakeGraph(self)
        self.graphs.append(graph)
        return graph

    def graph_pool_handle(self) -> _FakePool:
        self.pool_calls += 1
        return _FakePool(self.pool_calls)

    def memory_allocated(self) -> int:
        return self.allocated

    def memory_reserved(self) -> int:
        return self.reserved

    def current_device(self) -> int:
        return 0

    def empty_cache(self) -> None:
        self.flushes += 1


def _api(backend: _FakeBackend, *, device_type: str = "cpu") -> ascend_graphs._NpuGraphApi:
    module = types.SimpleNamespace(
        Stream=backend.Stream,
        current_stream=backend.current_stream,
        stream=backend.stream,
        synchronize=backend.synchronize,
        NPUGraph=backend.NPUGraph,
        graph_pool_handle=backend.graph_pool_handle,
        memory_allocated=backend.memory_allocated,
        current_device=backend.current_device,
        empty_cache=backend.empty_cache,
    )
    if backend.expose_memory_reserved:
        module.memory_reserved = backend.memory_reserved

    def graph(graph_object, pool=None, stream=None):
        return _FakeCapture(backend, graph_object, pool)

    module.graph = graph
    return ascend_graphs._NpuGraphApi(module, device_type=device_type)


# ------------------------------------------------------------------------------------------------
# A resident packed store, built the way the real arena builds its views
# ------------------------------------------------------------------------------------------------


class _FakeStore:
    """One layer's resident packed experts: three arena pairs, sliced into `PackedExpert` views."""

    def __init__(
        self,
        layer_id: int,
        *,
        experts: tuple[int, ...] = (0, 2),
        dim: int = 32,
        inter_dim: int = 64,
        swiglu_limit: float = 0.0,
        seed: int = 0,
        allocated: bool = True,
        operator=cpu_packed_gemv,
    ) -> None:
        self.layer_id = int(layer_id)
        self.dim = dim
        self.dtype = torch.bfloat16
        self.device = torch.device("cpu")
        self.backend = "torch"
        self.swiglu_limit = swiglu_limit
        self.packed_gemv = operator
        self.owned = tuple(experts)
        self.allocated = allocated
        generator = torch.Generator().manual_seed(1000 + int(layer_id) + seed)
        shapes = {"w1": (inter_dim, dim), "w2": (dim, inter_dim), "w3": (inter_dim, dim)}
        self._arenas: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for which, (out_features, in_features) in shapes.items():
            codes = torch.randint(
                0, 256, (len(experts), out_features, in_features // 2),
                generator=generator, dtype=torch.int64,
            ).to(torch.uint8)
            # A tight, mostly-positive exponent range keeps the fixture's values in a range where a
            # bit comparison is about the replay wiring and not about overflow.
            scales = torch.randint(
                120, 135, (len(experts), out_features, in_features // BLOCK),
                generator=generator, dtype=torch.int64,
            ).to(torch.uint8)
            self._arenas[which] = (codes, scales)

    def _expert_at(self, expert: int) -> ascend_packed_experts.PackedExpert:
        slot = self.owned.index(int(expert))
        return ascend_packed_experts.PackedExpert(
            w1=self._matrix("w1", slot),
            w2=self._matrix("w2", slot),
            w3=self._matrix("w3", slot),
            block=BLOCK,
        )

    def _matrix(self, which: str, slot: int) -> ascend_packed_experts.PackedMatrix:
        codes, scales = self._arenas[which]
        return ascend_packed_experts.PackedMatrix(codes=codes[slot], scales=scales[slot], block=BLOCK)


def _model(stores) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        layers=[
            types.SimpleNamespace(layer_id=store.layer_id, ffn=types.SimpleNamespace(routed=store))
            for store in stores
        ]
    )


# ------------------------------------------------------------------------------------------------
# The round under test
# ------------------------------------------------------------------------------------------------


class _Round:
    """A fixture bundle: the stores, the fake backend, and the manager over them."""

    def __init__(
        self,
        *,
        stores=None,
        backend: _FakeBackend | None = None,
        warmup: int = 1,
        max_graphs: int = 64,
        max_pool_bytes: int = 512 * 2**20,
        api_device_type: str = "cpu",
    ) -> None:
        self.backend = _FakeBackend() if backend is None else backend
        self.stores = _FakeStore(0) , _FakeStore(1, experts=(1,))
        if stores is not None:
            self.stores = tuple(stores)
        self.eager = PRISTINE_DISPATCHER
        self.manager = ascend_expert_graphs.RoutedExpertGraphs(
            _model(self.stores),
            warmup=warmup,
            api=_api(self.backend, device_type=api_device_type),
            max_graphs=max_graphs,
            max_pool_bytes=max_pool_bytes,
        )

    # -- convenience ---------------------------------------------------------------------

    def install(self) -> "_Round":
        self.manager.install()
        return self

    def capture(self) -> "_Round":
        self.manager.install().capture_all()
        return self

    @property
    def store(self) -> _FakeStore:
        return self.stores[0]

    def dispatch(self, x, store=None, resident=None, *, weight=1.0, **kwargs):
        store = self.store if store is None else store
        resident = store._expert_at(store.owned[0]) if resident is None else resident
        routing = torch.tensor([[weight]], dtype=torch.float32)
        return ascend_packed_experts.packed_expert_forward(
            x,
            resident,
            swiglu_limit=kwargs.pop("swiglu_limit", store.swiglu_limit),
            weights=routing,
            dtype=kwargs.pop("dtype", store.dtype),
            backend=store.backend,
            packed_gemv=store.packed_gemv,
            prefill=kwargs.pop("prefill", False),
            **kwargs,
        )

    def eager_forward(self, x, store=None, resident=None, *, weight=1.0, **kwargs):
        store = self.store if store is None else store
        resident = store._expert_at(store.owned[0]) if resident is None else resident
        routing = torch.tensor([[weight]], dtype=torch.float32)
        return self.eager(
            x,
            resident,
            swiglu_limit=kwargs.pop("swiglu_limit", store.swiglu_limit),
            weights=routing,
            dtype=kwargs.pop("dtype", store.dtype),
            backend=store.backend,
            packed_gemv=store.packed_gemv,
            prefill=kwargs.pop("prefill", False),
        )


@pytest.fixture
def build_round(monkeypatch):
    """A round factory that attaches recorded bodies and always restores the module dispatcher."""
    rounds: list[_Round] = []

    def record(body, pool, warmup, api):
        graph = ascend_graphs._record(body, pool, warmup, api)
        graph.body = body
        return graph

    monkeypatch.setattr(ascend_expert_graphs, "_record", record)

    def factory(**kwargs) -> _Round:
        round_ = _Round(**kwargs)
        rounds.append(round_)
        return round_

    yield factory
    for round_ in rounds:
        round_.manager.release()
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER, (
        "a test leaked an installed dispatcher"
    )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    """The value's bits, so "equal" means equal and not "close"."""
    return tensor.contiguous().view(torch.int16 if tensor.dtype == torch.bfloat16 else torch.int32)


def _row(dim: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, dim, generator=generator).to(torch.bfloat16)


# ------------------------------------------------------------------------------------------------
# Install and capture
# ------------------------------------------------------------------------------------------------


def test_install_patches_the_module_once_and_reports_itself(build_round):
    round_ = build_round()
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER
    round_.manager.install()
    installed = ascend_packed_experts.packed_expert_forward
    assert installed is not PRISTINE_DISPATCHER
    assert getattr(installed, ascend_expert_graphs.DISPATCHER_MARKER) is round_.manager
    assert round_.manager.describe()["installed"] is True
    assert round_.manager.describe()["captured"] is False


def test_a_second_round_is_refused_while_one_owns_the_dispatcher(build_round):
    first = build_round().install()
    second = build_round()
    with pytest.raises(RuntimeError, match="single-owner|already owns"):
        second.manager.install()
    # The first round is untouched by the refusal.
    assert getattr(ascend_packed_experts.packed_expert_forward, ascend_expert_graphs.DISPATCHER_MARKER) is first.manager


def test_a_second_round_on_the_same_model_is_refused(build_round):
    round_ = build_round()
    same_model = ascend_expert_graphs.RoutedExpertGraphs(
        round_.manager.model, api=_api(round_.backend)
    )
    round_.manager.install()
    with pytest.raises(RuntimeError, match="already installed for this model"):
        same_model.install()


def test_capture_all_records_one_warmed_graph_per_resident_expert(build_round):
    round_ = build_round().capture()
    report = round_.manager.describe()
    # Two layers x (2 and 1 owned experts).
    assert report["graphs"] == 3
    assert report["experts_expected"] == 3
    assert report["covered_layers"] == [0, 1]
    assert round_.backend.captures == 3
    assert round_.backend.replays == 0
    # Every recording warmed on a side stream before it recorded, and every one synchronized.
    assert round_.backend.events == (
        ["stream:side", "warm", "sync", "capture"] * 3
    )
    assert round_.backend.synchronize_calls == 3


def test_every_capture_lands_in_the_round_single_pool(build_round):
    round_ = build_round().capture()
    assert round_.backend.pool_calls == 1, "one pool handle for the whole round"
    assert len(round_.backend.captured_with) == 3
    assert {id(pool) for pool in round_.backend.captured_with} == {id(round_.manager.pool)}


def test_capture_all_is_idempotent(build_round):
    round_ = build_round().capture()
    second = round_.manager.capture_all()
    assert second["already_captured"] is True and second["captured"] == 0
    assert round_.backend.captures == 3


def test_capture_all_needs_install_first(build_round):
    round_ = build_round()
    with pytest.raises(RuntimeError, match="needs install"):
        round_.manager.capture_all()


# ------------------------------------------------------------------------------------------------
# What a replay reads
# ------------------------------------------------------------------------------------------------


def test_a_dispatch_before_capture_runs_the_original_body(build_round):
    round_ = build_round().install()
    x = _row(round_.store.dim, 1)
    produced = round_.dispatch(x)
    control = round_.eager_forward(x)
    assert torch.equal(_bits(produced), _bits(control))
    assert round_.backend.replays == 0
    assert round_.manager.passthroughs == 1


def test_a_replay_reads_the_current_row_and_routing_weight(build_round):
    round_ = build_round().capture()
    for step, (seed, weight) in enumerate(((11, 0.25), (12, -1.5), (13, 2.0))):
        x = _row(round_.store.dim, seed)
        produced = round_.dispatch(x, weight=weight)
        control = round_.eager_forward(x, weight=weight)
        assert torch.equal(_bits(produced), _bits(control)), f"step {step} is not the eager body's bits"
        assert produced.shape == (1, round_.store.dim)
    assert round_.backend.replays == 3
    assert round_.backend.captures == 3, "a replay never records again"
    assert round_.manager.replays == 3
    assert round_.manager.passthroughs == 0


def test_the_result_is_the_shared_sink_allocated_outside_the_pool(build_round):
    round_ = build_round().install()
    sink = round_.manager.sink
    assert sink is not None
    sink_pointer = sink.data_ptr()
    assert round_.backend.captures == 0, "the staging buffers must exist before any capture"
    round_.manager.capture_all()
    first = round_.dispatch(_row(round_.store.dim, 21))
    assert first is sink, "a replay hands back the staging sink, not a pool allocation"
    second = round_.dispatch(_row(round_.store.dim, 22))
    assert second is sink
    assert sink.data_ptr() == sink_pointer, "the sink never moves and is never reallocated"
    assert round_.backend.allocated > 0, "the captures did consume pool memory"
    described = round_.manager.describe()["staging"]
    assert described["sink"]["inside_capture_pool"] is False
    assert described["input"]["inside_capture_pool"] is False
    assert described["route_weight"]["inside_capture_pool"] is False


def test_the_routed_summation_loop_consumes_the_shared_sink_in_order(build_round):
    """The store sums each contribution before dispatching the next expert; one sink is enough."""
    round_ = build_round().capture()
    store = round_.store
    x = _row(store.dim, 31)
    routing = ((store.owned[0], 0.5), (store.owned[1], -0.25))

    def run(forward):
        total = torch.zeros((1, store.dim), dtype=torch.float32)
        for expert, weight in routing:
            resident = store._expert_at(expert)
            contribution = forward(x, resident=resident, weight=weight)
            # The store's own consumer expression, in one row: read the contribution, add it.
            total = total + contribution.to(torch.float32)
        return total

    produced = run(round_.dispatch)
    control = run(lambda x, **kwargs: round_.eager_forward(x, **kwargs))
    assert torch.equal(_bits(produced), _bits(control))
    assert round_.backend.replays == 2, "each drawn expert replayed once"
    assert round_.manager.graphs == 3


def test_two_experts_of_one_round_replay_their_own_constants_into_one_sink(build_round):
    round_ = build_round().capture()
    store = round_.store
    x = _row(store.dim, 41)

    # The answer has to be snapshotted before the next replay: the sink is one buffer by design, so
    # holding the returned tensor across a dispatch and reading it later reads the *next* expert's
    # answer. That is why the store consumes each contribution before dispatching the next.
    first_holder = round_.dispatch(x, resident=store._expert_at(store.owned[0]))
    first_bits = _bits(first_holder).clone()
    first_control = round_.eager_forward(x, resident=store._expert_at(store.owned[0]))
    assert torch.equal(first_bits, _bits(first_control))

    second_holder = round_.dispatch(x, resident=store._expert_at(store.owned[1]))
    second_bits = _bits(second_holder).clone()
    second_control = round_.eager_forward(x, resident=store._expert_at(store.owned[1]))
    assert torch.equal(second_bits, _bits(second_control))

    # One buffer, two experts, and each recording carries its own expert's constants.
    assert first_holder is second_holder is round_.manager.sink
    assert not torch.equal(first_bits, second_bits), (
        "two experts with different resident bytes replayed the same answer"
    )
    assert round_.backend.replays == 2 and round_.backend.captures == 3


# ------------------------------------------------------------------------------------------------
# Everything that is not a recorded single-row decode runs the original body
# ------------------------------------------------------------------------------------------------


def test_a_prefill_call_takes_the_original_path(build_round):
    round_ = build_round().capture()
    store = round_.store
    x = torch.randn(3, store.dim, generator=torch.Generator().manual_seed(51)).to(torch.bfloat16)
    produced = round_.dispatch(x, prefill=True)
    control = round_.eager_forward(x, prefill=True)
    assert torch.equal(_bits(produced), _bits(control))
    assert round_.backend.replays == 0
    assert round_.manager.passthroughs == 1


def test_a_multi_row_call_takes_the_original_path(build_round):
    round_ = build_round().capture()
    store = round_.store
    x = torch.randn(2, store.dim, generator=torch.Generator().manual_seed(52)).to(torch.bfloat16)
    produced = round_.dispatch(x, prefill=None)
    control = round_.eager_forward(x, prefill=None)
    assert torch.equal(_bits(produced), _bits(control))
    assert round_.backend.replays == 0


def test_a_different_swiglu_limit_takes_the_original_path(build_round):
    round_ = build_round().capture()
    x = _row(round_.store.dim, 53)
    produced = round_.dispatch(x, swiglu_limit=0.5)
    control = round_.eager_forward(x, swiglu_limit=0.5)
    assert torch.equal(_bits(produced), _bits(control))
    assert round_.backend.replays == 0
    assert round_.manager.passthroughs == 1


def test_a_different_routing_weight_shape_takes_the_original_path(build_round):
    round_ = build_round().capture()
    store = round_.store
    x = _row(store.dim, 54)
    resident = store._expert_at(store.owned[0])
    produced = ascend_packed_experts.packed_expert_forward(
        x, resident, weights=torch.ones((2, 1), dtype=torch.float32), dtype=store.dtype,
        backend=store.backend, packed_gemv=store.packed_gemv, prefill=False,
    )
    control = round_.eager(
        x, resident, weights=torch.ones((2, 1), dtype=torch.float32), dtype=store.dtype,
        backend=store.backend, packed_gemv=store.packed_gemv, prefill=False,
    )
    assert torch.equal(_bits(produced), _bits(control))
    assert round_.backend.replays == 0


def test_a_different_operator_object_takes_the_original_path(build_round):
    round_ = build_round().capture()
    store = round_.store
    x = _row(store.dim, 55)

    def same_contract_but_another_object(a_codes, a_scale, b_packed, b_e8m0, *, backend="auto"):
        return cpu_packed_gemv(a_codes, a_scale, b_packed, b_e8m0, backend=backend)

    resident = store._expert_at(store.owned[0])
    produced = ascend_packed_experts.packed_expert_forward(
        x, resident, weights=torch.ones((1, 1), dtype=torch.float32), dtype=store.dtype,
        backend=store.backend, packed_gemv=same_contract_but_another_object, prefill=False,
    )
    control = round_.eager(
        x, resident, weights=torch.ones((1, 1), dtype=torch.float32), dtype=store.dtype,
        backend=store.backend, packed_gemv=same_contract_but_another_object, prefill=False,
    )
    assert torch.equal(_bits(produced), _bits(control))
    assert round_.backend.replays == 0, "an unrecorded operator is never answered by a recording"


def test_an_unrecorded_expert_identity_takes_the_original_path(build_round):
    round_ = build_round().capture()
    # Same geometry, same layer id, different arena: a different resident expert.
    stranger = _FakeStore(0, seed=99)
    x = _row(stranger.dim, 56)
    produced = round_.dispatch(x, store=stranger, resident=stranger._expert_at(stranger.owned[0]))
    control = round_.eager_forward(x, store=stranger, resident=stranger._expert_at(stranger.owned[0]))
    assert torch.equal(_bits(produced), _bits(control))
    assert round_.backend.replays == 0


# ------------------------------------------------------------------------------------------------
# Bounds, refusals and cleanup
# ------------------------------------------------------------------------------------------------


def test_a_store_that_is_not_preloaded_is_refused(build_round):
    round_ = build_round(stores=(_FakeStore(0, allocated=False),))
    with pytest.raises(RuntimeError, match="not preloaded"):
        round_.manager.install()
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER


def test_a_non_packed_store_is_not_recorded(build_round):
    class _BoundedStore:
        def __init__(self) -> None:
            self.layer_id = 0
            self.cache = object()

    model = types.SimpleNamespace(
        layers=[types.SimpleNamespace(layer_id=0, ffn=types.SimpleNamespace(routed=_BoundedStore()))]
    )
    manager = ascend_expert_graphs.RoutedExpertGraphs(model, api=_api(_FakeBackend()))
    with pytest.raises(RuntimeError, match="no resident packed routed-expert store"):
        manager.install()
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER


def test_a_caching_operator_is_refused_because_it_would_be_a_second_weight_bank(build_round):
    class _CachingOperator:
        def __init__(self, budget: int) -> None:
            self.cache_budget_bytes = budget

        @property
        def cache_stats(self) -> dict:
            return {"budget_bytes": self.cache_budget_bytes, "entries": 0}

    round_ = build_round(stores=(_FakeStore(0, operator=_CachingOperator(1 << 20)),))
    with pytest.raises(RuntimeError, match="second full weight bank|cache budget"):
        round_.manager.install()
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER


def test_a_zero_budget_caching_operator_is_allowed(build_round):
    class _ZeroBudgetOperator:
        """A caching operator that declares no budget: it retains nothing, so it is not a bank."""

        def __init__(self) -> None:
            self.cache_budget_bytes = 0
            self.calls = 0

        @property
        def cache_stats(self) -> dict:
            return {"budget_bytes": 0, "entries": 0}

        def __call__(self, a_codes, a_scale, b_packed, b_e8m0, *, backend="auto"):
            self.calls += 1
            return cpu_packed_gemv(a_codes, a_scale, b_packed, b_e8m0, backend=backend)

    operator = _ZeroBudgetOperator()
    round_ = build_round(stores=(_FakeStore(0, operator=operator),))
    round_.manager.install().capture_all()
    assert round_.manager.describe()["operators"][0]["declared_cache_budget"] == 0
    x = _row(round_.store.dim, 57)
    produced = round_.dispatch(x)
    control = round_.eager_forward(x)
    assert torch.equal(_bits(produced), _bits(control))
    assert round_.backend.replays == 1


def test_the_real_caching_operators_are_detected_under_their_own_spellings(build_round):
    """The classes this runtime can actually hand over do not report their budget the same way.

    `CubePackedProjection` carries `cache_budget_bytes` and reports it in `cache_stats()`;
    `CubeWeightCache` carries `budget` and reports it in `describe()`. Both also report a capacity
    *ceiling*, which must not be mistaken for a configured budget once the budget is zero.
    """
    from relicllm.models.deepseek_v4_1 import cube_experiment, cube_weight_cache

    projection = cube_experiment.CubePackedProjection(cache_budget_bytes=1 << 20, backend="torch")
    found, budget, source = ascend_expert_graphs._declared_cache_budget(projection)
    assert (found, budget) == (True, 1 << 20) and "cache_budget_bytes" in source

    cache = cube_weight_cache.CubeWeightCache(
        budget_bytes=1 << 20, certified_immutable_sources=True
    )
    found, budget, source = ascend_expert_graphs._declared_cache_budget(cache)
    assert (found, budget) == (True, 1 << 20) and "budget" in source

    # A zero budget is not a bank, even though both objects still advertise a ceiling.
    zero_projection = cube_experiment.CubePackedProjection(cache_budget_bytes=0, backend="torch")
    assert ascend_expert_graphs._declared_cache_budget(zero_projection)[1] == 0
    zero_cache = cube_weight_cache.CubeWeightCache(budget_bytes=0, certified_immutable_sources=True)
    assert ascend_expert_graphs._declared_cache_budget(zero_cache)[1] == 0
    assert zero_projection.cache_stats["max_cache_bytes"] > 0, "the ceiling is not the budget"

    # And the round refuses the positive one by name while the zero-budget one installs.
    refused = build_round(stores=(_FakeStore(0, operator=cache),))
    with pytest.raises(RuntimeError, match="second full weight bank|weight-cache budget"):
        refused.manager.install()
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER

    allowed = build_round(stores=(_FakeStore(0, operator=zero_cache),))
    allowed.manager.install()
    assert allowed.manager.describe()["operators"][0]["declared_cache_budget"] == 0


def test_a_partial_wrapper_is_inspected_through_to_its_wrapped_operator(build_round):
    import functools

    from relicllm.models.deepseek_v4_1 import cube_weight_cache

    cache = cube_weight_cache.CubeWeightCache(budget_bytes=1 << 20, certified_immutable_sources=True)
    wrapped = functools.partial(cache, backend="auto")
    found, budget, _ = ascend_expert_graphs._declared_cache_budget(wrapped)
    assert (found, budget) == (True, 1 << 20)


def test_a_store_reloaded_under_a_live_round_falls_back_instead_of_replaying(build_round):
    """New arenas are new bytes: the recording must not answer for them, and the report must say so."""
    round_ = build_round().capture()
    store = round_.store
    assert round_.manager.describe()["sources_changed_since_capture"] == []

    # Re-preload this store: new arenas, same shapes, same layer id, different bytes.
    replacement = _FakeStore(0, seed=7)
    store._arenas = replacement._arenas
    assert round_.manager.describe()["sources_changed_since_capture"] == [0]

    x = _row(store.dim, 61)
    produced = round_.dispatch(x)
    control = round_.eager_forward(x)
    assert torch.equal(_bits(produced), _bits(control)), "a reloaded arena must not be answered by an old graph"
    assert round_.backend.replays == 0
    assert round_.manager.passthroughs == 1


# ------------------------------------------------------------------------------------------------
# The real provider, not a double: a plain store with `.release()`/`.clear_cache()` and no `__call__`
# ------------------------------------------------------------------------------------------------


class _StubPackedCheckpoint:
    """The two accessors `AscendPackedRoutedExperts` validates against, over in-memory packed bytes.

    The real store is a plain object -- not a module and not callable -- and the root's M1 probe failed
    only because it assumed otherwise. This fixture is the real class over a stub checkpoint, so the
    round is exercised against the contract the runtime actually has: `owned`, `_expert_at`,
    `allocated`, `release()`/`clear_cache()`, and `forward(x, weights, indices)` as the caller.
    """

    identity = "stub-packed-checkpoint"

    def __init__(self, *, layer_id: int, experts: tuple[int, ...], dim: int, inter_dim: int,
                 seed: int = 0) -> None:
        generator = torch.Generator().manual_seed(4000 + seed)
        shapes = {"w1": (inter_dim, dim), "w2": (dim, inter_dim), "w3": (inter_dim, dim)}
        self._bytes: dict[str, torch.Tensor] = {}
        for expert in experts:
            for which, (out_features, in_features) in shapes.items():
                key = f"layers.{layer_id}.ffn.experts.{expert}.{which}.weight"
                self._bytes[key] = torch.randint(
                    0, 256, (out_features, in_features // 2), generator=generator, dtype=torch.int64
                ).to(torch.uint8)
                self._bytes[scale_key(key)] = torch.randint(
                    120, 135, (out_features, in_features // BLOCK), generator=generator, dtype=torch.int64
                ).to(torch.uint8)

    def packed(self, key: str) -> torch.Tensor:
        return self._bytes[key]

    def is_quantized(self, key: str) -> bool:
        return scale_key(key) in self._bytes

    def is_packed_fp4(self, key: str) -> bool:
        return True

    def block_size(self, key: str) -> int:
        return BLOCK


def _real_store(*, layer_id: int = 0, experts: tuple[int, ...] = (0, 1, 2), seed: int = 0):
    """A real `AscendPackedRoutedExperts` whose operator is the CPU same-contract oracle.

    ``world == 1`` with a contiguous ``experts`` tuple, because a `world` deal would own a
    different set than the stub checkpoint holds bytes for.
    """
    store = ascend_packed_experts.AscendPackedRoutedExperts(
        _StubPackedCheckpoint(layer_id=layer_id, experts=experts, dim=32, inter_dim=64, seed=seed),
        layer_id,
        n_experts=len(experts),
        dim=32,
        inter_dim=64,
        topk=2,
        swiglu_limit=0.0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        rank=0,
        world=1,
        backend="torch",
        reserve_bytes=0,
        preload=True,
    )
    store.packed_gemv = cpu_packed_gemv
    return store


def test_the_real_provider_contract_is_what_this_round_assumes():
    store = _real_store()
    # The facts the root's M1 probe got wrong, asserted directly.
    assert callable(store) is False, "the store is a plain object, not a callable/module"
    assert isinstance(store, torch.nn.Module) is False
    for name in ascend_expert_graphs._REQUIRED_STORE_ATTRS:
        assert hasattr(store, name), f"the round needs {name} and the real store lacks it"
    assert hasattr(store, "release") and hasattr(store, "clear_cache")
    assert not hasattr(store, "clear"), "cleanup is release()/clear_cache(), and this round calls neither"
    assert store.packed_gemv is cpu_packed_gemv


def test_the_real_store_forward_runs_through_the_dispatcher_bit_for_bit(build_round):
    """The caller path the runtime actually uses: `store.forward(x, weights, indices)`.

    Host routing, the ascending-id walk and the FP32 `index_put_` summation all run unchanged; only
    the per-expert `packed_expert_forward` call is answered by a replay. The control is the same
    store's forward taken **before** any recording, so this is a whole-caller-path bit comparison.
    """
    store = _real_store(experts=(0, 1, 2))
    x = _row(32, 71)
    weights = torch.tensor([[0.5, -0.25]], dtype=torch.float32)
    indices = torch.tensor([[0, 2]], dtype=torch.int64)
    control = store.forward(x, weights, indices).clone()
    assert bool(torch.isfinite(control).all())

    round_ = build_round(stores=(store,))
    round_.manager.install().capture_all()
    produced = store.forward(x, weights, indices)
    assert torch.equal(_bits(produced), _bits(control)), (
        "the real store's routed sum changed under the recording round"
    )
    assert round_.backend.replays == 2, "both drawn experts replayed"
    assert round_.manager.passthroughs == 0

    # A second step with a different routing weight and row: still the same numbers as eager.
    store.packed_gemv = cpu_packed_gemv  # unchanged; the operator identity is what matches
    other_weights = torch.tensor([[1.25, 0.75]], dtype=torch.float32)
    other_indices = torch.tensor([[2, 0]], dtype=torch.int64)
    round_.manager.release()
    eager_again = store.forward(_row(32, 72), other_weights, other_indices).clone()
    round_.manager.install().capture_all()
    replayed = store.forward(_row(32, 72), other_weights, other_indices)
    assert torch.equal(_bits(replayed), _bits(eager_again))


def test_the_real_store_cleanup_contract_is_not_used_by_the_round(build_round):
    """The round never calls into the store's lifecycle, so the store is still the caller's to close."""
    store = _real_store()
    round_ = build_round(stores=(store,))
    round_.manager.install().capture_all()
    assert store.allocated is True
    round_.manager.release()
    # Cleanup is the store's own: `release()`, not a `clear()` this round might have guessed at.
    store.release()
    assert store.allocated is False


def test_a_store_on_another_device_type_is_refused_before_anything_is_allocated(build_round):
    """A recording captures on the backend's device; a store on another one cannot be its operand."""
    store = _FakeStore(0)
    store.device = torch.device("cuda:0")
    round_ = build_round(stores=(store,))
    with pytest.raises(RuntimeError, match="captures on 'cpu'|same device"):
        round_.manager.install()
    # Refused before the pool handle and before any staging buffer exists.
    assert round_.backend.pool_calls == 0
    assert round_.manager.pool is None
    assert round_.manager.input is None and round_.manager.sink is None
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER


def test_an_npu_round_refuses_a_store_on_another_card(build_round):
    """On a real card the store must be on the *current* device, not merely of type `npu`.

    Reuses `ascend_graphs._require_current_device`, so the message is that helper's: it names
    `set_device` rather than failing inside a capture. No allocation happens first.
    """
    store = _FakeStore(0)
    # This host has no `npu` registered in torch, so the card is a stand-in carrying exactly what the
    # checks read: the device *type* and its *index*. No tensor is ever allocated on it.
    store.device = types.SimpleNamespace(type="npu", index=0, __str__=lambda self: "npu:0")
    backend = _FakeBackend()
    backend.current_device = lambda: 1  # the process is on another card
    round_ = build_round(stores=(store,), backend=backend, api_device_type="npu")
    with pytest.raises(RuntimeError, match="set_device|current device"):
        round_.manager.install()
    assert backend.pool_calls == 0
    assert round_.manager.pool is None and round_.manager.sink is None
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER


def test_an_in_place_write_to_a_versioned_source_stops_matching_the_recording(build_round):
    """A recorded entry is not reused after its arena bytes are written in place.

    The store's arenas are written once at preload and are treated as immutable; this pins what
    happens if that obligation is broken anyway. The version counter is part of the key, so the next
    dispatch misses every entry and runs the **unchanged eager body** on the current bytes -- never a
    stale graph.
    """
    round_ = build_round().capture()
    store = round_.store
    x = _row(store.dim, 81)
    # `_arenas[which]` is the (codes, scales) pair, so `[0]` is the codes arena [experts, out, cols].
    codes_arena = store._arenas["w1"][0]
    assert codes_arena._version == 0, "the fixture's arena is an ordinary versioned tensor"

    before = round_.dispatch(x)
    assert torch.equal(_bits(before), _bits(round_.eager_forward(x)))
    assert round_.backend.replays == 1

    # An in-place write to the arena, wide enough that the answer must move: every code byte of
    # expert 0's w1 projection. One nibble would leave the BF16 answer unchanged and prove nothing.
    with torch.no_grad():
        codes_arena[0] = torch.bitwise_not(codes_arena[0])
    assert codes_arena._version == 1

    after = round_.dispatch(x)
    control = round_.eager_forward(x)
    assert torch.equal(_bits(after), _bits(control)), (
        "a mutated source must be answered by the eager body on the current bytes"
    )
    assert not torch.equal(_bits(before), _bits(after)), "the write must be visible in the answer"
    assert round_.backend.replays == 1, "no replay happened after the mutation"
    assert round_.backend.captures == 3, "and nothing was silently re-recorded"
    assert round_.manager.passthroughs == 1
    # The entry is still registered; it is the key that no longer matches the source.
    assert round_.manager.graphs == 3


def test_a_handbuilt_matrix_with_the_same_pointer_but_another_stride_stops_matching(build_round):
    """Same pointer, same shape, same offset -- different elements. A pointer is not an identity.

    A strided view of the arena can share its first address with the recorded projection while reading
    different bytes. Strides are part of the key, so such a call takes the eager body instead of
    replaying a graph that would read the wrong elements. This is not reachable through the provider
    (it slices contiguous arena views and the round refuses a strided projection at install); it is the
    hand-built case the key exists to reject.
    """
    round_ = build_round().capture()
    store = round_.store
    expert = store.owned[0]
    recorded = store._expert_at(expert)

    def strided_like(matrix, matrix_factory):
        """Same data_ptr/shape/offset as `matrix`, but every column read two bytes apart."""
        out_features, packed_columns = matrix.codes.shape
        codes = torch.as_strided(
            matrix.codes, (out_features, packed_columns), (packed_columns, 2), matrix.codes.storage_offset()
        )
        assert codes.data_ptr() == matrix.codes.data_ptr()
        assert codes.shape == matrix.codes.shape and codes.stride() != matrix.codes.stride()
        return matrix_factory(codes=codes, scales=matrix.scales, block=matrix.block)

    handbuilt = ascend_packed_experts.PackedExpert(
        w1=strided_like(recorded.w1, ascend_packed_experts.PackedMatrix),
        w2=recorded.w2,
        w3=recorded.w3,
        block=recorded.block,
    )
    x = _row(store.dim, 82)
    produced = round_.dispatch(x, resident=handbuilt)
    control = round_.eager_forward(x, resident=handbuilt)
    assert torch.equal(_bits(produced), _bits(control)), (
        "a differently-strided view of the same pointer must not be answered by a recording"
    )
    assert round_.backend.replays == 0, "the stride difference must not match a recorded entry"
    assert round_.backend.captures == 3
    assert round_.manager.passthroughs == 1


def test_an_inference_tensor_source_reports_no_version_and_still_matches(build_round):
    """Inference tensors have no version counter, so the key records `None` and still matches.

    Both sides of a match see the same tensor, so the absent counter is symmetric: a round recorded
    and replayed inside `torch.inference_mode()` still replays (the immutable-arena obligation is what
    keeps that sound). This is the path the runtime actually takes, since generation runs under
    inference mode.
    """
    with torch.inference_mode():
        store = _FakeStore(0)
        assert torch.is_inference(store._arenas["w1"][0])
        assert ascend_expert_graphs._tensor_version(store._arenas["w1"][0]) is None
        round_ = build_round(stores=(store,))
        round_.manager.install().capture_all()
        before = round_.manager.replays
        x = _row(store.dim, 83)
        produced = round_.dispatch(x)
        control = round_.eager_forward(x)
        assert torch.equal(_bits(produced), _bits(control))
        assert round_.manager.replays == before + 1, "an inference-tensor source still replays"
        assert round_.backend.captures == round_.manager.graphs == 2


def test_reserved_only_growth_trips_the_pool_budget(build_round):
    """A capture pool can grow the *reserved* counter while `memory_allocated` does not move.

    A temporary freed inside the recorded region leaves the allocator holding capacity it does not
    count as allocated. Budgeting on the allocated delta alone would accept such a round; the round
    enforces the largest non-negative delta instead.
    """
    round_ = build_round(
        max_pool_bytes=10,
        backend=_FakeBackend(
            allocated_bytes_per_capture=0, reserved_bytes_per_capture=10, expose_memory_reserved=True
        ),
    )
    round_.manager.install()
    with pytest.raises(RuntimeError, match="above the 10-byte ceiling"):
        round_.manager.capture_all()
    assert round_.backend.captures == 2, "the second capture crosses the ceiling"
    # The whole round is rolled back, exactly like any other over-budget round.
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER
    assert round_.manager.graphs == 0 and round_.manager.captured is False


def test_the_report_separates_allocated_and_reserved_deltas(build_round):
    """Both counters are reported, and `pool_bytes` is the enforced maximum with its metric named."""
    round_ = build_round(
        backend=_FakeBackend(
            allocated_bytes_per_capture=3, reserved_bytes_per_capture=7, expose_memory_reserved=True
        )
    )
    report = round_.manager.install().capture_all()
    assert report["allocated_delta"] == 9, "three captures x 3 bytes"
    assert report["reserved_delta"] == 21, "three captures x 7 bytes"
    assert report["pool_bytes"] == 21, "pool_bytes is the delta the budget is enforced on"
    assert report["memory_reserved_available"] is True
    assert report["physical_footprint_measured"] is False
    assert "not a measured physical footprint" in report["pool_bytes_metric"]
    # The same numbers, and the same label, are visible in describe().
    described = round_.manager.describe()
    assert described["pool_bytes"] == 21
    assert described["memory"]["allocated_delta"] == 9
    assert described["memory"]["reserved_delta"] == 21
    assert described["pool_bytes_metric"] == report["pool_bytes_metric"]


def test_the_report_says_so_when_the_reserved_counter_is_unavailable(build_round):
    """A backend without `memory_reserved` reports `None` rather than inventing a number."""
    round_ = build_round()  # the default fake exposes no reserved counter
    report = round_.manager.install().capture_all()
    assert report["memory_reserved_available"] is False
    assert report["reserved_delta"] is None
    assert report["allocated_delta"] == 21, "three captures x 7 bytes of allocated growth"
    assert report["pool_bytes"] == report["allocated_delta"], "the budget falls back to allocated"
    assert round_.manager.describe()["memory"]["reserved_delta"] is None


def test_a_released_round_reports_no_allocator_growth(build_round):
    round_ = build_round().capture()
    report = round_.manager.release()
    assert report["allocated_delta"] == 21
    assert report["reserved_delta"] is None
    assert report["physical_footprint_measured"] is False
    # The counters describe a released round from here on.
    described = round_.manager.describe()
    assert described["pool_bytes"] == 0
    assert described["memory"]["allocated_delta"] == 0


def test_the_graph_ceiling_is_checked_before_any_capture(build_round):
    round_ = build_round(max_graphs=2)
    with pytest.raises(RuntimeError, match="above the 2 ceiling"):
        round_.manager.install()
    assert round_.backend.captures == 0
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER


def test_exceeding_the_pool_budget_releases_the_whole_round(build_round):
    round_ = build_round(max_pool_bytes=10, backend=_FakeBackend(pool_bytes_per_capture=7))
    round_.manager.install()
    with pytest.raises(RuntimeError, match="above the 10-byte ceiling"):
        round_.manager.capture_all()
    # No half-recorded state: the dispatcher is back and no graph is retained.
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER
    assert round_.manager.installed is False
    assert round_.manager.graphs == 0
    assert round_.manager.pool is None
    assert round_.manager.sink is None
    assert round_.backend.captures == 2, "the failing capture is the second one"


def test_a_capture_failure_releases_the_whole_round(build_round):
    round_ = build_round(backend=_FakeBackend(fail_on_capture=2))
    round_.manager.install()
    with pytest.raises(RuntimeError, match="refused this capture"):
        round_.manager.capture_all()
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER
    assert round_.manager.graphs == 0
    assert round_.manager.captured is False


def test_release_restores_the_dispatcher_and_drops_every_recording(build_round):
    round_ = build_round().capture()
    assert round_.backend.replays == 0
    report = round_.manager.release()
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER
    assert report["graphs"] == 3
    assert report["dispatcher_restored"] is True
    assert round_.manager.pool is None and round_.manager.sink is None
    assert round_.manager.describe()["installed"] is False
    assert round_.manager.describe()["graphs"] == 0
    # After release the pristine body runs again, and the released round cannot dispatch.
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER


def test_release_is_idempotent_and_does_not_claim_a_foreign_dispatcher(build_round):
    round_ = build_round().install()
    foreign = lambda *args, **kwargs: None  # noqa: E731 - a stand-in for another owner
    ascend_packed_experts.packed_expert_forward = foreign
    try:
        report = round_.manager.release()
        assert report["dispatcher_restored"] is False
        assert ascend_packed_experts.packed_expert_forward is foreign, "a foreign owner is not overwritten"
    finally:
        ascend_packed_experts.packed_expert_forward = PRISTINE_DISPATCHER
    second = round_.manager.release()
    assert second["graphs"] == 0


def test_a_recording_round_holds_no_second_weight_bank(build_round):
    round_ = build_round().capture()
    entries = round_.manager._entries
    assert len(entries) == 3
    for entry in entries.values():
        # Nothing but the original views: no tensor of the entry's own, so no widened operand copy.
        assert [value for value in vars(entry).values() if isinstance(value, torch.Tensor)] == []
        for matrix in (entry.resident.w1, entry.resident.w2, entry.resident.w3):
            assert matrix.codes.dtype == torch.uint8 and matrix.scales.dtype == torch.uint8
        assert entry.resident.block == BLOCK
        assert entry.graph.body is not None
    # The entries point into the store's own arenas: the same pointers the store hands out.
    store = round_.store
    for expert in store.owned:
        resident = store._expert_at(expert)
        assert ascend_expert_graphs._expert_identity(resident) in entries


def test_describe_names_the_unchanged_contract_and_where_the_engram_lives(build_round):
    round_ = build_round().capture()
    described = round_.manager.describe()
    assert described["numerical_contract"] == "original packed_expert_forward, unchanged; replay only"
    assert described["host_routing_remains_host_side"] is True
    assert described["complete_engram_remains_on_cpu"] is True
    assert described["operators"][0]["declared_cache_budget"] == 0
    assert described["dim"] == round_.store.dim
    assert described["dtype"] == "bfloat16"
    assert described["device"] == "cpu"


def test_the_context_manager_releases(build_round):
    round_ = build_round()
    with round_.manager as installed:
        installed.install()
        assert ascend_packed_experts.packed_expert_forward is not PRISTINE_DISPATCHER
    assert ascend_packed_experts.packed_expert_forward is PRISTINE_DISPATCHER


# ------------------------------------------------------------------------------------------------
# The module keeps its own promises about what it does not touch
# ------------------------------------------------------------------------------------------------

MODULE_PATH = ascend_expert_graphs.__file__


def test_the_module_patches_one_module_level_attribute_and_nothing_else():
    """The dispatcher is the only module-level attribute this file writes.

    An `Assign` whose target is `<name>.<attribute>` and whose value is a plain name (not `self`) is
    a write to another module's global. There is exactly one, and it is the documented seam.
    """
    import ast

    tree = ast.parse(open(MODULE_PATH, encoding="utf-8").read())
    external: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue
            if isinstance(target.value, ast.Name) and target.value.id != "self":
                external.append(ast.unparse(target))
    expected = [
        "ascend_packed_experts.packed_expert_forward",
        "ascend_packed_experts.packed_expert_forward",
    ]
    assert external == expected, (
        f"expected exactly one patch site and one restore site, got {external}"
    )