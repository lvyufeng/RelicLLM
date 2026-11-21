"""The Ascend decode-graph adapter against a fake recording backend, and its NPU fixtures.

Two halves, and they answer different questions.

**The CPU half** drives `ascend_graphs` with a backend that is a *plain object the test passes in*
(`NpuDecodeGraphs(..., api=...)`), never a patched `torch`. That is deliberate: patching a global
backend is how a CPU test in a session that is also running CUDA work picks up another device's
code, and the adapter therefore takes its backend as an explicit argument instead of reaching for
one. What the fake backend establishes is the adapter's own contract -- that the warm passes run
before a capture and on a side stream, that captures go into the round's one pool, that a dual
compressor records two A bodies and the step picks between them from its position, that a failed
capture leaves neither a sticky `compress_variant` nor a half-recorded "ready" layer, that the
`SharedAttentionRuntime` is held strongly for as long as the recordings live, and that a tree which
cannot be recorded is refused **by name** before any capture happens. It executes no device work:
its "graph" records the call and a replay is a no-op, so nothing here is evidence about replay
semantics or speed on hardware.

**The NPU half** is the bounded fixture the lead runs (`RELIC_ASCEND_GRAPHS_DEVICE=npu:0`): T1 is a
standalone primitive recording (integer add and an exactly representable float multiply, replayed
after the inputs change, so a stale or missing replay is a wrong result and not a tolerance), and T2
is a small attention-body recording -- inverse real-pair RoPE on a **strided tail view**
`q[..., -rd:]` of a wider buffer, with the position reaching the body as a 0-dim tensor -- replayed
at two positions and compared to the same body run eagerly on the same card. Neither T1 nor T2 loads
a checkpoint, and neither is a whole-model claim.
"""

from __future__ import annotations

import ast
import contextlib
import gc
import importlib
import os
import types
import weakref
from pathlib import Path

import pytest
import torch
from torch import nn

from relicllm.models.deepseek_v4_1 import ascend_graphs, mma
from relicllm.models.deepseek_v4_1.decode_pos import Pos

MODULE_PATH = Path(ascend_graphs.__file__)
DEVICE_ENV = "RELIC_ASCEND_GRAPHS_DEVICE"


# ------------------------------------------------------------------------------------------------
# A recording backend that is only a Python object
# ------------------------------------------------------------------------------------------------


class _FakeStream:
    def __init__(self, name: str) -> None:
        self.name = name
        self.waited: list[object] = []

    def wait_stream(self, other: _FakeStream) -> None:
        self.waited.append(other)


class _FakeGraph:
    """One recording. `replay` counts only: a fake cannot re-run the captured kernels."""

    def __init__(self, backend: _FakeBackend) -> None:
        self.backend = backend
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1
        self.backend.replays += 1


class _FakeBackend:
    """The `torch.npu` surface, recorded rather than executed.

    `events` is the ordering evidence: a warm pass is `"warm"` and a capture is `"capture"`, so
    `["warm", "capture"] * n` says every recording warmed before it recorded. `pool_handle` is a
    fresh object per call, so "every capture used the round's pool" is an identity check.
    `fail_on_capture` makes the n-th capture raise, which is how the partial-failure paths are
    driven.
    """

    def __init__(self, *, fail_on_capture: int | None = None, pool_argument: str = "keyword") -> None:
        self.events: list[str] = []
        self.names: list[str] = []
        self.captures = 0
        self.replays = 0
        self.allocated = 0
        self.synchronize_calls = 0
        self.pool_calls = 0
        self.empty_cache_calls = 0
        self.fail_on_capture = fail_on_capture
        self.captured_with: list[object] = []
        self.pool_argument = pool_argument

    # -- the API surface ---------------------------------------------------------------------

    def Stream(self) -> _FakeStream:
        return _FakeStream("side")

    def current_stream(self) -> _FakeStream:
        return self._current

    def stream(self, side: _FakeStream):
        @contextlib.contextmanager
        def run():
            self.names.append(f"stream:{side.name}")
            self.events.append("warm")
            yield

        return run()

    def synchronize(self) -> None:
        self.synchronize_calls += 1
        self.events.append("sync")

    def NPUGraph(self):
        self.events.append("new-graph")
        return _FakeGraph(self)

    def graph_pool_handle(self):
        self.pool_calls += 1
        return object()

    def memory_allocated(self) -> int:
        return self.allocated

    def current_device(self) -> int:
        return self.current

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1


def _fake_npu_module(backend: _FakeBackend):
    """A `torch.npu`-shaped namespace around `backend`."""
    module = types.SimpleNamespace()
    backend._current = _FakeStream("current")
    backend.current = 0
    module.Stream = backend.Stream
    module.current_stream = backend.current_stream
    module.stream = backend.stream
    module.synchronize = backend.synchronize
    module.NPUGraph = backend.NPUGraph
    module.graph_pool_handle = backend.graph_pool_handle
    module.memory_allocated = backend.memory_allocated
    module.current_device = backend.current_device
    module.empty_cache = backend.empty_cache

    if backend.pool_argument == "keyword":

        def graph(graph_object, pool=None, stream=None):
            return _FakeCapture(backend, graph_object, pool)

    elif backend.pool_argument == "positional":

        def graph(graph_object, handle, stream=None):
            return _FakeCapture(backend, graph_object, handle)

    elif backend.pool_argument == "none":

        def graph(graph_object):
            return _FakeCapture(backend, graph_object, None)

    else:  # pragma: no cover - a test asking for an unknown spelling is a test bug
        raise AssertionError(backend.pool_argument)
    module.graph = graph
    return module


class _FakeCapture:
    def __init__(self, backend: _FakeBackend, graph_object, pool) -> None:
        self.backend = backend
        self.graph = graph_object
        self.pool = pool

    def __enter__(self):
        self.backend.captures += 1
        self.backend.events.append("capture")
        self.backend.captured_with.append(self.pool)
        if self.backend.fail_on_capture == self.backend.captures:
            raise RuntimeError("the fake backend refused this capture")

    def __exit__(self, *exc_info) -> bool:
        # A capture's allocations would come out of the pool; seven bytes is enough for the
        # accounting assertion to be about the adapter's bookkeeping and not about torch.
        self.backend.allocated += 7
        return False


def _api(backend: _FakeBackend, *, device_type: str = "cpu") -> ascend_graphs._NpuGraphApi:
    return ascend_graphs._NpuGraphApi(_fake_npu_module(backend), device_type=device_type)


# ------------------------------------------------------------------------------------------------
# A block small enough to run on the host, with the shape conventions `_body_a` expects
# ------------------------------------------------------------------------------------------------


class _TinyCompressor:
    def __init__(self, ratio: int) -> None:
        self.compress_ratio = ratio
        self.compress_variant = None


class _TinyAttn(nn.Module):
    def __init__(self, compressor: _TinyCompressor | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.compressor = compressor
        self.variants: list[object] = []

    def forward(self, x, pos, shared):
        self.variants.append(None if self.compressor is None else self.compressor.compress_variant)
        return x + 1.0


class _TinyRouted:
    partial = False

    def __init__(self) -> None:
        self.calls = 0

    def forward(self, x, weights, indices):
        self.calls += 1
        return x * 2.0


class _TinyScale(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = factor

    def forward(self, x):
        return x * self.factor


class _TinyFfn(nn.Module):
    def __init__(self, dim: int = 4) -> None:
        super().__init__()
        self.dim = dim
        self.shared_experts = _TinyScale(3.0)
        self.routed = _TinyRouted()
        self.tp = None

    def gate(self, x, image_mask):
        rows = x.size(0)
        weights = torch.ones(rows, 1, dtype=x.dtype)
        indices = torch.zeros(rows, 1, dtype=torch.int64)
        return weights, indices


class _TinyBlock(nn.Module):
    """`Block.forward`'s structure, at `dim=4`, `hc_mult=2`, with arithmetic that is its own oracle."""

    def __init__(self, layer_id: int = 0, compressor: _TinyCompressor | None = None) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.attn = _TinyAttn(compressor)
        self.attn_norm = nn.Identity()
        self.ffn_norm = nn.Identity()
        self.ffn = _TinyFfn()
        # The hyper-connection coefficient sets the inherited bodies read. Plain tensors: this
        # fixture's `hc_mixes` ignores them, and a Parameter would only exercise the placement walk
        # that the tests below drive on purpose.
        for name in ("hc_attn_fn", "hc_attn_scale", "hc_attn_base",
                     "hc_ffn_fn", "hc_ffn_scale", "hc_ffn_base"):
            setattr(self, name, torch.zeros(1))

    def hc_mixes(self, x, hc_fn, hc_scale, hc_base):
        return x, x, x

    def hc_pre(self, x, pre_mix):
        return x.mean(dim=2)

    def hc_post(self, x, residual, post, comb):
        return residual + x.unsqueeze(2)


class _TinyModel(nn.Module):
    def __init__(self, blocks, width: int = 4) -> None:
        super().__init__()
        self.layers = nn.ModuleList(blocks)
        self.register_buffer("window_kv_cache", torch.zeros(1, width, dtype=torch.float32))

    def forward(self, *args, **kwargs):  # pragma: no cover - `capture_pass` takes a callable
        raise AssertionError("the fixtures drive `capture_pass` with an explicit callable")


def _tensors(block: _TinyBlock, *, scale: float = 0.0):
    x = torch.arange(8, dtype=torch.bfloat16).reshape(1, 1, 2, 4) * 0.5 + scale
    pre_mix = torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)
    return x, pre_mix


def _layer_graphs(block: _TinyBlock, api, *, warmup: int = 1) -> ascend_graphs.NpuLayerGraphs:
    graphs = ascend_graphs.NpuLayerGraphs(block, api.graph_pool_handle(), warmup)
    graphs.api = api
    return graphs


# ------------------------------------------------------------------------------------------------
# The adapter does not reach for CUDA or for an environment-selected backend
# ------------------------------------------------------------------------------------------------


def _executed_source() -> str:
    """The module's source with every string constant and comment removed.

    A docstring may *name* ``torch.cuda`` -- this module's exists to explain the difference -- so a
    search of the raw text would be a test of prose. The AST keeps code and drops prose.
    """
    tree = ast.parse(MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                node.body = body[1:] or [ast.Pass()]
    return ast.dump(ast.fix_missing_locations(tree))


def test_the_adapter_touches_no_cuda_namespace():
    """It is the *CUDA* module's adapter counterpart; a CUDA reference is a copy-paste bug."""
    dumped = _executed_source()
    assert "'cuda'" not in dumped
    assert "attr='cuda'" not in dumped


def test_the_scan_that_forbids_cuda_finds_it_in_the_cuda_module():
    """The teeth of the test above: the same scan over `graphs.py` does find the CUDA calls."""
    source = (MODULE_PATH.parent / "graphs.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    cuda_uses = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "cuda"
        and isinstance(node.value, ast.Name) and node.value.id == "torch"
    ]
    assert cuda_uses, "the CUDA module must use torch.cuda, or the scan proves nothing"


def test_only_the_capture_is_replaced_and_the_bodies_are_the_inherited_ones():
    """The adapter is three backend calls wide; anything else defined here is a silent fork.

    Named explicitly rather than counted, because the failure this guards against is somebody
    fixing a body in the adapter instead of in `graphs.py` and the two drifting apart.
    """
    import inspect

    from relicllm.models.deepseek_v4_1 import graphs

    dunder = {"__module__", "__doc__", "__annotations__", "__firstlineno__", "__static_attributes__"}
    assert set(vars(ascend_graphs.NpuLayerGraphs)) - dunder == {"api", "_capture"}
    for inherited in ("_body_a", "_body_b", "_routed", "_graph_for", "_replay", "step", "ready"):
        assert inspect.getattr_static(ascend_graphs.NpuLayerGraphs, inherited) is inspect.getattr_static(
            graphs.LayerGraphs, inherited
        ), f"{inherited} must be inherited, not reimplemented"
    for inherited in ("capture_pass", "pool_bytes", "marks", "reset_marks"):
        assert inspect.getattr_static(ascend_graphs.NpuDecodeGraphs, inherited) is inspect.getattr_static(
            graphs.DecodeGraphs, inherited
        ), f"{inherited} must be inherited"
    # snapshot/restore are the cache rewind; the inherited `capture_pass` calls `graphs.py`'s, and
    # the adapter must not carry a second copy of the rewind.
    assert not hasattr(ascend_graphs, "snapshot")
    assert not hasattr(ascend_graphs, "restore")
    import inspect as _inspect

    assert "snapshot" in _inspect.getsource(graphs.DecodeGraphs.capture_pass)
    assert "restore" in _inspect.getsource(graphs.DecodeGraphs.capture_pass)


def test_the_pool_is_made_on_the_card_the_caches_are_on():
    """A pool is an allocator view of one card, so it is created with that card current."""
    backend = _FakeBackend()
    api = _api(backend, device_type="npu")
    model = _cached_model(devices=(_Device("npu", 0),))
    backend.current = 0
    graphs = ascend_graphs.NpuDecodeGraphs(model, api=api)
    assert backend.pool_calls == 1
    assert str(graphs.device) == "npu:0"
    graphs.release()

    backend.current = 1
    with pytest.raises(RuntimeError, match="set_device"):
        ascend_graphs.NpuDecodeGraphs(model, api=api)


def test_no_environment_variable_selects_the_npu_graphs():
    """Opt-in is an explicit constructor, so a stray variable cannot switch a run onto graphs."""
    dumped = _executed_source()
    assert "attr='environ'" not in dumped
    assert "os.getenv" not in dumped


# ------------------------------------------------------------------------------------------------
# API discovery, by name, with no device work
# ------------------------------------------------------------------------------------------------


def test_a_missing_recording_name_is_refused_and_named():
    module = types.SimpleNamespace(Stream=object())  # present but nothing else
    with pytest.raises(RuntimeError, match="graph_pool_handle") as refused:
        ascend_graphs.npu_graph_api(module)
    for name in ("NPUGraph", "graph", "graph_pool_handle", "memory_allocated", "current_device"):
        assert name in str(refused.value)


def test_a_host_without_torch_npu_is_refused_by_name():
    if hasattr(torch, "npu"):
        pytest.skip("this host has a torch.npu and the refusal is about the host that does not")
    with pytest.raises(RuntimeError, match="torch.npu"):
        ascend_graphs.npu_graph_api()


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [("keyword", "keyword"), ("positional", "positional"), ("none", None)],
)
def test_the_api_report_reads_how_this_build_passes_a_pool(spelling, expected):
    backend = _FakeBackend(pool_argument=spelling)
    report = ascend_graphs.npu_graph_api_report(_fake_npu_module(backend))
    assert report["module_present"] is True
    assert report["available"]["graph"] is True
    assert report["pool_accepted_as"] == expected
    if expected is None:
        assert any("private pool" in note for note in report["notes"])


def test_a_build_that_cannot_share_one_pool_is_refused_rather_than_recording():
    backend = _FakeBackend(pool_argument="none")
    api = _api(backend)
    with pytest.raises(RuntimeError, match="pool"):
        ascend_graphs._record(lambda: None, api.graph_pool_handle(), 1, api)


# ------------------------------------------------------------------------------------------------
# Recording order, pool identity and the replay path
# ------------------------------------------------------------------------------------------------


def test_every_recording_warms_first_and_captures_into_the_round_pool():
    backend = _FakeBackend()
    api = _api(backend)
    block = _TinyBlock()
    graphs = _layer_graphs(block, api)
    x, pre_mix = _tensors(block)

    graphs.step(x, 0, pre_mix, None, shared=object())

    # One A body and one B body, each warmed once and then captured once, in that order.
    assert backend.events.count("capture") == 2
    assert backend.events.index("warm") < backend.events.index("capture")
    # Both captures went into the pool the layer was handed, and neither made a second one.
    assert backend.captured_with == [graphs.pool, graphs.pool]
    assert backend.pool_calls == 1
    assert graphs.pool_bytes == 14
    assert graphs.ready is True


def test_the_first_step_records_and_the_next_one_only_replays():
    backend = _FakeBackend()
    api = _api(backend)
    graphs = _layer_graphs(_TinyBlock(), api)
    x, pre_mix = _tensors(graphs.block)
    shared = object()

    graphs.step(x, 0, pre_mix, None, shared)
    after_capture = backend.captures
    routed_after_capture = graphs.block.ffn.routed.calls
    graphs.step(x, 1, pre_mix, None, shared)

    assert backend.captures == after_capture, "the second step must not capture again"
    assert after_capture == 2
    # A, the eager routed call, B -- once per step, and the routed call is outside every recording.
    assert graphs.block.ffn.routed.calls == routed_after_capture + 1
    assert backend.replays == 4


def test_the_host_routed_call_runs_between_the_recordings_never_inside_one():
    """`_routed` is the eager half: a `.tolist()` cannot be inside a recording, and it is not.

    The fake backend counts captures, so where the routed call happened relative to them is
    observable: it runs once during the capture pass *before* the first recording is made, and once
    per step after the A replay -- never between a capture's ``__enter__`` and its body.
    """
    backend = _FakeBackend()
    api = _api(backend)
    block = _TinyBlock()
    observed: list[int] = []
    original = block.ffn.routed.forward

    def forward(x, weights, indices):
        observed.append(backend.captures)
        return original(x, weights, indices)

    block.ffn.routed.forward = forward
    graphs = _layer_graphs(block, api)
    x, pre_mix = _tensors(block)

    # The first step records and then replays itself, so the routed call happens twice: once
    # before any recording exists, once between the two replays.
    graphs.step(x, 0, pre_mix, None, object())
    assert observed == [0, 2], "the routed call must precede the recordings and follow A's replay"
    assert backend.captures == 2
    graphs.step(x, 1, pre_mix, None, object())
    assert observed == [0, 2, 2], "a replay step must not capture, so no new capture is seen"
    assert backend.captures == 2


def test_a_captured_step_matches_the_same_bodies_run_eagerly():
    """The adapter must not change what the body computes: same block, same result."""
    backend = _FakeBackend()
    api = _api(backend)
    graphs = _layer_graphs(_TinyBlock(), api)
    x, pre_mix = _tensors(graphs.block)

    # Eager: the inherited bodies, driven by hand on a private sink.
    eager = ascend_graphs.NpuLayerGraphs(graphs.block, None, 1)
    from relicllm.models.deepseek_v4_1.graphs import _put

    _put(eager.sink, "x_in", x)
    _put(eager.sink, "pre_in", pre_mix)
    eager._body_a(Pos.of(0), object())
    eager._routed()
    eager._body_b()

    got_x, got_pre = graphs.step(x, 0, pre_mix, None, object())
    assert torch.equal(got_x, eager.sink.x_out)
    assert torch.equal(got_pre, eager.sink.ffn_pre)


def test_a_dual_compressor_records_both_variants_and_the_step_picks_by_position():
    backend = _FakeBackend()
    api = _api(backend)
    compressor = _TinyCompressor(4)
    block = _TinyBlock(compressor=compressor)
    graphs = _layer_graphs(block, api)
    x, pre_mix = _tensors(block)

    graphs.step(x, 0, pre_mix, None, object())

    assert graphs.dual is True
    assert graphs.graph_a_emit is not None and graphs.graph_a_fill is not None
    assert graphs.graph_a is None
    # The first entry is the real pass `_capture` runs before recording anything -- it allocates the
    # sink slots, and it runs whatever variant the module would have picked (`None`: no override).
    # Then the emitting body saw the forced `True` (warm + capture) and the filling body the forced
    # `False`, and the flag the module carries afterwards is cleared rather than left on one branch.
    assert block.attn.variants == [None, True, True, False, False]
    assert compressor.compress_variant is None
    # (3 + 1) % 4 == 0 emits; (2 + 1) % 4 != 0 fills.
    assert graphs._graph_for(Pos.of(3)) is graphs.graph_a_emit
    assert graphs._graph_for(Pos.of(2)) is graphs.graph_a_fill
    # And the replay actually takes the recording the position names.
    before = (graphs.graph_a_emit.replays, graphs.graph_a_fill.replays)
    graphs.step(x, 3, pre_mix, None, object())
    assert graphs.graph_a_emit.replays == before[0] + 1
    assert graphs.graph_a_fill.replays == before[1]
    graphs.step(x, 2, pre_mix, None, object())
    assert graphs.graph_a_fill.replays == before[1] + 1


def test_a_failed_capture_leaves_no_sticky_variant_and_no_half_recorded_layer():
    """The second capture (the filling body) fails: nothing may look replayable afterwards."""
    backend = _FakeBackend(fail_on_capture=2)
    api = _api(backend)
    compressor = _TinyCompressor(4)
    block = _TinyBlock(compressor=compressor)
    graphs = _layer_graphs(block, api)
    x, pre_mix = _tensors(block)

    with pytest.raises(RuntimeError, match="refused this capture"):
        graphs.step(x, 0, pre_mix, None, object())

    assert compressor.compress_variant is None, "a capture that raised must not leave the flag set"
    assert graphs.graph_a_emit is None, "the emit body was recorded but must not be advertised"
    assert graphs.graph_a_fill is None
    assert graphs.graph_b is None
    assert graphs.ready is False
    # The next step retries the capture instead of replaying an unrecorded graph.
    backend.fail_on_capture = None
    graphs.step(x, 0, pre_mix, None, object())
    assert graphs.ready is True


def test_the_shared_runtime_is_held_by_the_recording_round():
    """A published buffer must outlive the forward that recorded it; the round owns the runtime."""
    backend = _FakeBackend()
    api = _api(backend)
    graphs = _layer_graphs(_TinyBlock(), api)
    x, pre_mix = _tensors(graphs.block)

    class _Runtime:
        pass

    shared = _Runtime()
    reference = weakref.ref(shared)
    graphs.step(x, 0, pre_mix, None, shared)
    del shared
    gc.collect()

    held = graphs.runtime
    assert reference() is not None, "the runtime died while its recordings were still live"
    assert held is reference()


# ------------------------------------------------------------------------------------------------
# The round: capture pass, release, and the refusals that happen before a capture
# ------------------------------------------------------------------------------------------------


def _round(model: _TinyModel, backend: _FakeBackend) -> ascend_graphs.NpuDecodeGraphs:
    return ascend_graphs.NpuDecodeGraphs(model, warmup=1, api=_api(backend))


def test_capture_pass_restores_the_caches_it_ran_the_pass_on():
    backend = _FakeBackend()
    model = _TinyModel([_TinyBlock()])
    graphs = _round(model, backend)
    block = model.layers[0]
    x, pre_mix = _tensors(block)

    def forward():
        # The pass writes the caches, as a real decode step does.
        model.window_kv_cache.add_(1.0)
        block.decode_graph.step(x, 0, pre_mix, None, object())

    graphs.capture_pass(forward)

    assert graphs.captured is True
    assert torch.equal(model.window_kv_cache, torch.zeros_like(model.window_kv_cache)), (
        "the capture pass's own cache writes have to be rewound, or the first real step starts "
        "one step ahead of the eager column"
    )
    assert block.decode_graph is graphs.layers[0]


def test_release_detaches_every_layer_and_forgets_the_graphs():
    backend = _FakeBackend()
    model = _TinyModel([_TinyBlock(), _TinyBlock(layer_id=1)])
    graphs = _round(model, backend)
    x, pre_mix = _tensors(model.layers[0])
    graphs.capture_pass(lambda: [block.decode_graph.step(x, 0, pre_mix, None, object())
                                 for block in model.layers])
    layers = list(graphs.layers)

    graphs.release()

    assert [block.decode_graph for block in model.layers] == [None, None]
    assert graphs.layers == [] and graphs.captured is False and graphs.pool is None
    for layer in layers:
        assert layer.graph_a is None and layer.graph_b is None
        assert layer.graph_a_emit is None and layer.graph_a_fill is None
        assert layer.runtime is None
    assert backend.empty_cache_calls == 1


def test_release_after_a_partial_failure_leaves_every_covered_layer_eager():
    backend = _FakeBackend(fail_on_capture=1)  # the very first body fails
    model = _TinyModel([_TinyBlock()])
    graphs = _round(model, backend)
    x, pre_mix = _tensors(model.layers[0])

    with pytest.raises(RuntimeError, match="refused this capture"):
        graphs.capture_pass(lambda: model.layers[0].decode_graph.step(x, 0, pre_mix, None, object()))
    graphs.release()

    assert model.layers[0].decode_graph is None
    assert graphs.layers == [] and graphs.pool is None


class _Device:
    """A hashable device stand-in, for the fixtures that must report a card without a card."""

    def __init__(self, kind: str, index: int | None = None) -> None:
        self.type = kind
        self.index = index

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _Device) and (self.type, self.index) == (other.type, other.index)

    def __hash__(self) -> int:
        return hash((self.type, self.index))

    def __str__(self) -> str:
        return self.type if self.index is None else f"{self.type}:{self.index}"


def _cached_model(*, devices=None, width: int = 4):
    """A model whose `window_kv_cache` buffer reports each of `devices`, in order."""
    devices = (_Device("cpu"),) if devices is None else devices
    model = types.SimpleNamespace()
    blocks = []
    for layer_id, device in enumerate(devices):
        block = types.SimpleNamespace(
            layer_id=layer_id,
            named_parameters=lambda: iter(()),
            named_buffers=lambda: iter(()),
            named_modules=lambda: iter(()),
            attn=types.SimpleNamespace(compressor=None),
        )
        blocks.append(block)
    model.layers = blocks
    model.named_buffers = lambda: iter([
        (f"layers.{index}.attn.window_kv_cache", types.SimpleNamespace(device=device))
        for index, device in enumerate(devices)
    ])
    return model


def test_a_host_resident_cache_is_refused_by_name():
    """The production backend is `npu`; a host tree must be refused, not recorded empty."""
    model = _cached_model(devices=(torch.device("cpu"),))
    api = types.SimpleNamespace(device_type="npu")
    with pytest.raises(RuntimeError, match="cpu"):
        ascend_graphs.NpuDecodeGraphs(model, api=api)


def test_a_tree_spread_over_two_cards_is_refused():
    model = _cached_model(devices=(_Device("npu", 0), _Device("npu", 1)))
    api = types.SimpleNamespace(device_type="npu")
    with pytest.raises(RuntimeError, match="spread over"):
        ascend_graphs.NpuDecodeGraphs(model, api=api)


def test_a_model_without_an_attention_cache_is_refused():
    model = types.SimpleNamespace(layers=[], named_buffers=lambda: iter(()))
    api = types.SimpleNamespace(device_type="npu")
    with pytest.raises(RuntimeError, match="window_kv_cache"):
        ascend_graphs.NpuDecodeGraphs(model, api=api)


def test_recording_the_emulation_off_body_is_refused(monkeypatch):
    monkeypatch.setenv("POCKETLLM_V41_MMA_EMULATE", "0")
    with pytest.raises(RuntimeError, match="MMA_EMULATE"):
        ascend_graphs.NpuDecodeGraphs(_cached_model(), api=types.SimpleNamespace(device_type="cpu"))


def _quantized_part(block, *, with_metadata: bool, half_attached: bool = False):
    """Attach a quantized submodule to `block.attn`, staged or not (see `mma.attach_source_codes`)."""
    quantized = nn.Module()
    quantized._quantized = True
    quantized.proj = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)
    block.attn.add_module("quantized_part", quantized)
    if with_metadata:
        weight = quantized.proj.weight
        codes = torch.zeros_like(weight, dtype=torch.float8_e4m3fn)
        mma.attach_source_codes(weight, codes, torch.ones(4, 1, dtype=torch.float32))
        if half_attached:
            delattr(weight, mma.CODES_ATTR)
    return quantized


def test_the_routed_expert_store_is_exempt_even_though_it_is_quantized():
    """The FP4 store sets `_quantized` too, and its weights are INT8x2 plus a half-scale grid.

    It is *the* store itself at `ffn.routed`, not only its children, so the exemption has to match
    the subtree root as well -- otherwise a correct NPU tree is refused for not carrying E4M3 codes
    its FP4 weights were never meant to have.
    """
    model = _TinyModel([_TinyBlock()])
    block = model.layers[0]
    store = nn.Module()
    store._quantized = True
    store.experts = nn.ModuleDict({"0": nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)})
    block.ffn.routed = store  # replaces the eager fixture store for this construction-only check
    graphs = ascend_graphs.NpuDecodeGraphs(model, warmup=1, api=_api(_FakeBackend()))
    graphs.release()


def test_a_quantized_module_with_no_staged_metadata_is_refused():
    """A module that says "quantized" with no checkpoint grid under it was not staged the NPU way."""
    model = _TinyModel([_TinyBlock()])
    _quantized_part(model.layers[0], with_metadata=False)
    with pytest.raises(RuntimeError, match="quantized_part") as refused:
        ascend_graphs.NpuDecodeGraphs(model, warmup=1, api=_api(_FakeBackend()))
    assert "loader.load_backbone" in str(refused.value)


def test_a_half_attached_quantized_pair_is_refused():
    """Codes without their grid are a plausible wrong answer, so they are refused, not recorded."""
    model = _TinyModel([_TinyBlock()])
    _quantized_part(model.layers[0], with_metadata=True, half_attached=True)
    with pytest.raises(RuntimeError, match="half of the checkpoint's quantized pair"):
        ascend_graphs.NpuDecodeGraphs(model, warmup=1, api=_api(_FakeBackend()))


def test_a_metadata_free_linear_under_a_quantized_module_is_accepted():
    """`wo_a` is an einsum against the dense weight and legitimately carries no codes.

    The check must not be "every linear under a quantized module is staged", or the real V4.1
    attention would be refused: `wo_a` sits under `Attention._quantized` and never reaches
    `quantized_linear`.
    """
    model = _TinyModel([_TinyBlock()])
    quantized = _quantized_part(model.layers[0], with_metadata=True)
    dense = nn.Linear(4, 4, bias=False, dtype=torch.bfloat16)  # the `wo_a` role: no metadata
    quantized.add_module("dense_einsum_weight", dense)
    graphs = ascend_graphs.NpuDecodeGraphs(model, warmup=1, api=_api(_FakeBackend()))
    graphs.release()


def test_the_checks_accept_a_real_v4_1_block_structure():
    """The real `Block` -- attention, indexer, compressors, MoE -- is what these checks run against.

    Built from the modules suite's own tiny config, so the structure is the model's rather than a
    fixture's. Three things are pinned here, and all three are the false-positive shapes:

    * `attn.wo_a` is under a quantized `Attention` and carries **no** codes (it is applied as an
      einsum against the dense weight), and the consistency check must not object;
    * `ffn.routed` is the FP4 expert store, a quantized module whose weights are INT8x2 plus the
      original half-scale grid, and the host-subtree exemption must cover the store itself;
    * a fresh block (no checkpoint staged) is accepted by the placement check, and the *staged*
      refusal is the one an fp8-configured block produces.
    """
    try:
        modules_tests = importlib.import_module("test_models_deepseek_v4_1_modules")
    except ImportError:
        pytest.skip(
            "the V4.1 modules suite, which owns the tiny block config this check is built from, "
            "is not staged on this host"
        )
    from relicllm.models.deepseek_v4_1.modules import Block

    config = modules_tests._cfg(**modules_tests._BLOCK_FIELDS)
    block = Block(config, 0, 1, 16)
    device = torch.device("cpu")

    modules = dict(block.named_modules())
    assert "attn.wo_a" in modules, "the fixture is not the real block structure"
    assert mma.attached_scale(modules["attn.wo_a"].weight) is None, (
        "wo_a is the metadata-free linear under a quantized module this check must tolerate"
    )
    assert "ffn.routed" in modules and ascend_graphs._is_host_side("ffn.routed"), (
        "the routed expert store is the subtree root and must be exempt as a root"
    )
    # No half-attached pair anywhere, and every parameter/buffer is on the one device.
    ascend_graphs._require_capturable_block(block, device)

    staged = Block(modules_tests._cfg(**modules_tests._BLOCK_FIELDS, dtype="fp8"), 0, 1, 16)
    with pytest.raises(RuntimeError, match="load_backbone"):
        ascend_graphs._require_capturable_block(staged, device)


def test_a_body_that_needs_missing_metadata_refuses_before_any_recording():
    """The weight-level refusal is the body's, and it lands before the first capture.

    A static check cannot tell which linears a body quantizes (see the test above), so the missing
    pair is raised by `quantized_linear`, which the warm pass runs before recording. This pins that
    ordering: no capture is made when the body cannot run.
    """
    backend = _FakeBackend()
    api = _api(backend)
    block = _TinyBlock()
    graphs = _layer_graphs(block, api)
    x, pre_mix = _tensors(block)

    def refuses(*args, **kwargs):
        raise RuntimeError("NPU quantized linear needs original E4M3 codes and the checkpoint grid")

    block.ffn.gate = refuses
    with pytest.raises(RuntimeError, match="E4M3 codes"):
        graphs.step(x, 0, pre_mix, None, object())
    assert backend.captures == 0
    assert graphs.ready is False


def test_the_host_subtrees_are_exempt_from_the_placement_check():
    """Engram and the routed store are on the host by design and neither is inside a recording."""
    device = torch.device("cpu")
    block = types.SimpleNamespace(
        named_parameters=lambda: iter([
            ("attn.wq_a.weight", torch.zeros(1)),
            ("engram.wkv.weight", torch.zeros(1)),
            ("ffn.routed.experts.0.w1", torch.zeros(1)),
        ]),
        named_buffers=lambda: iter([("attn.window_kv_cache", torch.zeros(1))]),
        named_modules=lambda: iter(()),
    )
    ascend_graphs._require_capturable_block(block, device)


def test_a_cpu_parameter_inside_a_block_is_refused():
    device = torch.device("cpu")
    block = types.SimpleNamespace(
        named_parameters=lambda: iter([("attn.moved", torch.zeros(1, device="meta"))]),
        named_buffers=lambda: iter(()),
        named_modules=lambda: iter(()),
    )
    with pytest.raises(RuntimeError, match="attn.moved"):
        ascend_graphs._require_capturable_block(block, device)


def test_a_cpu_buffer_inside_a_block_is_refused():
    device = torch.device("cpu")
    block = types.SimpleNamespace(
        named_parameters=lambda: iter(()),
        named_buffers=lambda: iter([("attn.moved", torch.zeros(1, device="meta"))]),
        named_modules=lambda: iter(()),
    )
    with pytest.raises(RuntimeError, match="attn.moved"):
        ascend_graphs._require_capturable_block(block, device)


def test_capturing_on_another_card_than_the_caches_is_refused():
    npu_device = _Device("npu", 2)
    backend = _FakeBackend()
    api = _api(backend, device_type="npu")
    backend.current = 2
    ascend_graphs._require_current_device(api, npu_device)  # current
    backend.current = 0
    with pytest.raises(RuntimeError, match="set_device"):
        ascend_graphs._require_current_device(api, npu_device)


# ------------------------------------------------------------------------------------------------
# T1 and T2: the bounded fixtures the lead runs on a leased card
# ------------------------------------------------------------------------------------------------


def _assert_same_bits(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    """`torch.equal` with a message -- the message goes in the `assert`, not in `torch.equal`.

    `torch.equal` takes exactly two tensors, so `torch.equal(a, b, "why")` is a `TypeError` that only
    fires on the path that calls it. Both NPU fixtures compare on integer views (`int32` for exact
    float products, `int16` for bf16 rotations) precisely so "same value" means "same bits".
    """
    assert actual.shape == expected.shape, (
        f"{what}: shapes differ, {tuple(actual.shape)} vs {tuple(expected.shape)}"
    )
    differing = int((actual.reshape(-1) != expected.reshape(-1)).sum())
    assert torch.equal(actual, expected), (
        f"{what}: {differing} of {actual.numel()} elements differ"
    )


def test_the_bit_comparison_helper_accepts_equal_bits_and_rejects_different_ones():
    """Host coverage for the comparison the NPU fixtures use.

    Kept as a CPU test on purpose: the fixtures themselves only run on a leased card, and a
    three-argument `torch.equal` reached a hardware run because nothing local executed that line.
    """
    same = torch.tensor([1, 2, 3], dtype=torch.int32)
    _assert_same_bits(same, same.clone(), "identical tensors")
    _assert_same_bits(torch.zeros(0, dtype=torch.int32), torch.zeros(0, dtype=torch.int32), "empty")

    with pytest.raises(AssertionError, match="1 of 3 elements differ"):
        _assert_same_bits(torch.tensor([1, 2, 4], dtype=torch.int32), same, "one different")
    with pytest.raises(AssertionError, match="shapes differ"):
        _assert_same_bits(torch.zeros(2, dtype=torch.int32), same, "shape mismatch")
    # A signed zero and a plain zero are different bit patterns, which is the whole point of
    # comparing through integer views rather than through `torch.equal` on the floats.
    minus_zero = torch.tensor([-0.0], dtype=torch.float32).view(torch.int32)
    plus_zero = torch.tensor([0.0], dtype=torch.float32).view(torch.int32)
    with pytest.raises(AssertionError, match="1 of 1 elements differ"):
        _assert_same_bits(minus_zero, plus_zero, "signed zero")


def _leased_device():
    name = os.environ.get(DEVICE_ENV)
    if not name:
        pytest.skip(f"set {DEVICE_ENV}=npu:0 on a leased card to run the NPU graph fixtures")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.fail("explicitly requested NPU is unavailable; this is not a pass")
    device = torch.device(name)
    torch.npu.set_device(device)
    return device


def test_npu_t1_primitive_recording_replays_exact_bits_after_its_inputs_change():
    """T1: a standalone recording of integer add and an exact float multiply.

    The values are chosen so every result is exactly representable: the replay is compared on bits,
    with no tolerance, and the inputs are changed between replays -- a recording that replayed a
    stale result, or replayed nothing, is a wrong answer here rather than a rounding difference.
    """
    device = _leased_device()
    api = ascend_graphs.npu_graph_api()
    pool = api.graph_pool_handle()

    add_a = torch.tensor([[1, -2, 3, -4], [5, 6, -7, 8]], dtype=torch.int32, device=device)
    add_b = torch.tensor([[10, 20, -30, 40], [-50, 60, 70, -80]], dtype=torch.int32, device=device)
    added = torch.zeros_like(add_a)
    mul_a = torch.tensor([[1.5, -2.25, 3.0, -0.5]], dtype=torch.float32, device=device)
    mul_b = torch.tensor([[2.0, 4.0, -8.0, 16.0]], dtype=torch.float32, device=device)
    multiplied = torch.zeros_like(mul_a)

    def body():
        added.copy_(add_a + add_b)
        multiplied.copy_(mul_a * mul_b)

    graph = ascend_graphs._record(body, pool, warmup=1, api=api)
    graph.replay()
    torch.npu.synchronize()
    _assert_same_bits(added.cpu(), (add_a + add_b).cpu(), "the first replay's integer sum")
    _assert_same_bits(
        multiplied.view(torch.int32).cpu(),
        (mul_a * mul_b).view(torch.int32).cpu(),
        "the first replay's exactly representable product",
    )

    # Change every input and replay: the graph has to read the buffers again.
    add_a.fill_(-9)
    add_b.fill_(4)
    mul_a.fill_(-0.25)
    mul_b.fill_(64.0)
    graph.replay()
    torch.npu.synchronize()
    _assert_same_bits(
        added.cpu(), torch.full_like(added, -5).cpu(), "the changed-input integer sum"
    )
    _assert_same_bits(
        multiplied.view(torch.int32).cpu(),
        (mul_a * mul_b).view(torch.int32).cpu(),
        "the changed-input product, which has to replay bit for bit",
    )


def test_npu_t2_a_body_inverse_rope_on_a_strided_tail_at_two_positions():
    """T2: an A-body-shaped recording, with the position as a tensor and a strided tail view.

    The body is inverse real-pair RoPE on `q[..., -rd:]`, a **head slice of a wider buffer**, which
    is the non-contiguous form the model rotates; the table row is selected by a 0-dim index tensor,
    which is how the position reaches a recording. Two positions are replayed and each is compared
    against the same body run eagerly on the same card, so the comparison is about the recording and
    not about a cross-device rounding.
    """
    device = _leased_device()
    api = ascend_graphs.npu_graph_api()
    pool = api.graph_pool_handle()

    from relicllm.models.deepseek_v4_1.attention import (
        apply_rotary_emb_pairs,
        precompute_freqs_cis,
    )

    rope_dim, heads, seqlen = 32, 2, 4
    table = precompute_freqs_cis(
        rope_dim, seqlen, 0, 10_000.0, 40.0, 32, 1, pairs=True
    ).to(device)
    generator = torch.Generator().manual_seed(20261118)
    base = torch.randn(1, 1, heads, 2 * rope_dim, generator=generator).to(torch.bfloat16).to(device)
    work = base.clone()
    counter = torch.zeros((), dtype=torch.int64, device=device)
    produced = torch.zeros_like(work)

    def body():
        # Inverse rotation of the last `rope_dim` columns only, in place on the strided view, with
        # the table row chosen by the position tensor rather than by a Python index.
        row = table.index_select(0, counter.reshape(1))
        apply_rotary_emb_pairs(work[..., -rope_dim:], row, inverse=True)
        produced.copy_(work)

    graph = ascend_graphs._record(body, pool, warmup=1, api=api)

    def run_eager(step: int) -> torch.Tensor:
        counter.fill_(step)
        work.copy_(base)
        body()
        torch.npu.synchronize()
        return produced.clone()

    results = {}
    for step in (1, 3):
        eager = run_eager(step)
        counter.fill_(step)
        work.copy_(base)
        produced.zero_()
        graph.replay()
        torch.npu.synchronize()
        replayed = produced.clone()
        _assert_same_bits(
            replayed.view(torch.int16).cpu(),
            eager.view(torch.int16).cpu(),
            f"the recording at position {step} against the eager body",
        )
        results[step] = replayed

    assert not torch.equal(
        results[1].view(torch.int16), results[3].view(torch.int16)
    ), "both positions produced the same bits, so the position tensor never reached the body"