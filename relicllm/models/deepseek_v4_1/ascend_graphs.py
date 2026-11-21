"""The decode graphs above, captured with ``torch.npu`` instead of ``torch.cuda``.

`graphs.py` is the design and it is device-neutral except for three things: the stream a body is
warmed on, the pool every recording is captured into, and the graph object a replay calls. This
module is those three, and nothing else: `NpuLayerGraphs` inherits `_body_a`, `_routed`, `_body_b`,
`_graph_for`, `step` and `_replay` verbatim, and `NpuDecodeGraphs` inherits `capture_pass`,
`pool_bytes`, `marks` and `reset_marks`. Only `_capture` is replaced (it is where a recording is
made), plus the constructor and `release` on the round (a CUDA pool handle and a CUDA cache flush
have no NPU spelling).

What stays eager is what was eager: `Backbone.forward`'s embedding, the **host** Engram module and
the routed expert call between the two halves. `_routed` is inherited unchanged, so the host read
inside the expert store still happens outside every recording -- a ``.tolist()`` inside a capture
would be a host synchronisation a recording cannot hold.

**What this is not.** It is a recording adapter, not a correctness or speed claim. Nothing here has
been compared against an eager ascend decode step, so no logit parity and no tokens-per-second
number is asserted by this module. `NpuDecodeGraphs` is opt-in by explicit construction only -- no
CLI flag, no environment variable, no serving path chooses it -- and a caller adopts it after its
own parity run.

**The contract it refuses rather than guesses.** The reference reduction must be enabled
(``POCKETLLM_V41_MMA_EMULATE`` not disabled) and every quantized linear inside a block must carry
the file's own E4M3 codes and E8M0 grid. Both are the conditions `kernels.quantized_linear` needs to
run the reference arithmetic from stored bytes; without them the body would silently take the
throughput escape hatch or recover codes at capture time (a device-to-host ``torch.equal``, which a
recording cannot perform). A tree that cannot meet the contract is refused by name, layer by layer,
**before** any recording is made -- not discovered inside a capture.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch

from relicllm.models.deepseek_v4_1 import mma
from relicllm.models.deepseek_v4_1.graphs import (
    CAPTURE_WARMUP,
    DecodeGraphs,
    LayerGraphs,
)
from relicllm.models.deepseek_v4_1.kernels import _reference_reduction_enabled

__all__ = [
    "NPU_GRAPH_DEVICE_TYPE",
    "NpuDecodeGraphs",
    "NpuLayerGraphs",
    "npu_graph_api",
    "npu_graph_api_report",
]

#: The device type these recordings land on. The CUDA path spells the same decision as
#: ``graphs.GRAPH_DEVICE_TYPE``; the two are separate names because each backend refuses a tree
#: whose caches are not on *its* device, and one shared constant would let one backend accept the
#: other's tree.
NPU_GRAPH_DEVICE_TYPE = "npu"

def _is_host_side(name: str) -> bool:
    """Whether this module path is one of the host-only subtrees, root included.

    The root matters: the routed expert store is a module at `ffn.routed` (not only its children),
    it sets `_quantized` for the FP4 experts, and its weights are widened INT8 plus the original
    half-scale grid rather than E4M3 codes. Matching only `"ffn.routed."` would walk it as if it
    were a dense quantized linear and refuse a tree that is correct.
    """
    return any(name == prefix.rstrip(".") or name.startswith(prefix) for prefix in _HOST_SUBTREES)


#: Subtrees of a block that are deliberately not on the card. The Engram module is built on the host
#: by `modules._engram_device` and the routed expert store reads host memory by design, so both are
#: exempt from the placement check below -- and neither is inside a recording: the Engram runs in
#: `Backbone.forward` before the block, and `_routed` runs between the two halves.
_HOST_SUBTREES = ("engram.", "ffn.routed.")

#: ``torch.npu`` names these recordings use, and what each one is for. Resolved by name so a build
#: that lacks one says which one instead of failing later inside a capture.
_REQUIRED_APIS = {
    "Stream": "the side stream a body is warmed on",
    "current_stream": "whose stream is the caller's before and after",
    "stream": "the `with` that runs the warm bodies on the side stream",
    "synchronize": "the drain between warming and capturing",
    "NPUGraph": "the recording object a replay calls",
    "graph": "the capture context manager",
    "graph_pool_handle": "one pool shared by every recording in the round",
    "memory_allocated": "what the recordings cost, measured around the captures",
    "current_device": "the device the capture lands on, checked against the caches",
}


class _NpuGraphApi:
    """The small ``torch.npu`` surface these recordings use, resolved once.

    ``device_type`` is the type the backend captures on. It is ``"npu"`` for every object
    :func:`npu_graph_api` builds, and the tests build one that says ``"cpu"`` so the same code path --
    the warm bodies, the capture, the replay, the sink -- can run on a CPU fixture without a card and
    without touching a global. Nothing in production can pass a different value: the constructor of
    ``NpuDecodeGraphs`` resolves this class itself unless a test hands one in, and the class refuses
    to be built for any module that is not ``torch.npu``.
    """

    __slots__ = (
        "NPUGraph",
        "Stream",
        "current_device",
        "current_stream",
        "device_type",
        "empty_cache",
        "graph",
        "graph_pool_handle",
        "memory_allocated",
        "module",
        "stream",
        "synchronize",
    )

    def __init__(self, module: Any, *, device_type: str = NPU_GRAPH_DEVICE_TYPE) -> None:
        self.module = module
        self.device_type = device_type
        for name in _REQUIRED_APIS:
            setattr(self, name, getattr(module, name))
        # Optional: a build without a cache flush is not a build that cannot record.
        self.empty_cache = getattr(module, "empty_cache", None)


def npu_graph_api(module: Any = None) -> _NpuGraphApi:
    """``torch.npu``'s recording surface, or a refusal that names what is missing.

    ``module`` exists for the CPU tests, which pass a fake backend in. A caller that omits it gets
    ``torch.npu`` -- and a stack without an Ascend build, or with an older ``torch_npu``, is told
    which names it lacks rather than failing inside a capture with an ``AttributeError`` from a
    nested frame.
    """
    if module is None:
        module = getattr(torch, "npu", None)
        if module is None:
            raise RuntimeError(
                "the NPU decode graphs need an Ascend build of torch (`torch.npu` does not exist "
                "here); the eager decode loop is the path on this host"
            )
    missing = [f"{name} ({why})" for name, why in _REQUIRED_APIS.items() if not hasattr(module, name)]
    if missing:
        raise RuntimeError(
            "this torch_npu build does not expose the recording API the decode graphs need: "
            + ", ".join(missing)
            + ". Nothing was captured. Report this list rather than adapting the adapter blindly."
        )
    return _NpuGraphApi(module)


def npu_graph_api_report(module: Any = None) -> dict[str, Any]:
    """What this stack exposes of the recording API, discovered without capturing anything.

    A handshake for a host whose ``torch_npu`` version is not the one the adapter was written
    against: it reports each required name and ``torch.npu.graph``'s own signature, so whether a
    shared ``pool`` can be passed is read off the build rather than assumed. It runs no device work.
    """
    if module is None:
        module = getattr(torch, "npu", None)
    report: dict[str, Any] = {
        "module_present": module is not None,
        "requires": dict(_REQUIRED_APIS),
        "available": {},
        "graph_signature": None,
        "pool_accepted_as": None,
        "notes": [],
    }
    if module is None:
        report["notes"].append("torch.npu does not exist; no NPU recording is possible on this host")
        return report
    report["available"] = {name: hasattr(module, name) for name in _REQUIRED_APIS}
    if hasattr(module, "graph"):
        kinds = _graph_pool_kind(module.graph)
        report["pool_accepted_as"] = kinds
        try:
            report["graph_signature"] = str(inspect.signature(module.graph))
        except (TypeError, ValueError) as error:  # a C function has no signature
            report["notes"].append(f"torch.npu.graph has no inspectable signature: {error}")
    # `_graph_pool_kind` answering None is the one answer that blocks adoption: a recording without
    # the round's pool is a private pool per graph, which is the memory blow-up the CUDA module's
    # header describes.
    if report["pool_accepted_as"] is None:
        report["notes"].append(
            "torch.npu.graph takes no pool; every recording would get a private pool, so the "
            "adapter refuses to capture until this is resolved"
        )
    return report


def _graph_pool_kind(graph) -> str | None:
    """How ``graph`` accepts the round's pool: ``"keyword"``, ``"positional"`` or ``None``.

    Read off the signature when there is one. ``torch.cuda.graph`` spells it
    ``graph(cuda_graph, pool=None, ...)``, so either spelling reaches the same parameter; a build
    whose ``pool`` parameter is named differently but is still the second positional argument is
    covered by the positional branch. A callable with no inspectable signature is treated as
    keyword-capable, because passing ``pool=`` to it then either works or raises by name at the call
    -- and that attempt happens inside a construction the caller asked for, not silently.
    """
    try:
        parameters = list(inspect.signature(graph).parameters.values())
    except (TypeError, ValueError):
        return "keyword"
    if "pool" in {parameter.name for parameter in parameters}:
        return "keyword"
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return "positional" if len(positional) >= 2 else None


def _capture(graph, pool, api: _NpuGraphApi):
    """The ``with`` that records into ``pool``, in whichever spelling this build accepts."""
    kind = _graph_pool_kind(api.graph)
    if kind == "keyword":
        return api.graph(graph, pool=pool)
    if kind == "positional":
        return api.graph(graph, pool)
    raise RuntimeError(
        "torch.npu.graph accepts no pool argument, so every graph would be captured into a private "
        "pool and a forty-layer tree would pay that pool forty times. Refusing to record; the "
        "eager decode loop is the alternative on this build."
    )


def _record(body, pool, warmup: int, api: _NpuGraphApi):
    """Record ``body`` into ``pool`` after running it for real on a side stream.

    The CUDA module's `graphs._record`, line for line, with ``torch.npu`` in place of ``torch.cuda``.
    The warm bodies are not decoration: they are what the allocator needs, so that a capture finds
    every allocation it wants already in the free list. They also write the caches, which is why
    `DecodeGraphs.capture_pass` rewinds them afterwards -- the rewind is inherited, not repeated.
    """
    side = api.Stream()
    side.wait_stream(api.current_stream())
    with api.stream(side):
        for _ in range(warmup):
            body()
    api.current_stream().wait_stream(side)
    api.synchronize()
    graph = api.NPUGraph()
    with _capture(graph, pool, api):
        body()
    return graph


def _require_current_device(api: _NpuGraphApi, device: torch.device) -> None:
    """Refuse to capture on one card while the tree's caches are on another.

    A recording is made on the *current* device, and its replay runs wherever the caller's stream is.
    Those have to be the same card as the buffers the body reads, or the capture succeeds and the
    replay is a wrong answer rather than an error -- the class of failure this whole adapter has to
    avoid. Checked by name instead of switching the process's device, because a process driving
    several cards would have its device moved out from under the other work by a capture.
    """
    index = device.index
    if index is None:
        return
    current = int(api.current_device())
    if current != int(index):
        raise RuntimeError(
            f"the NPU decode graphs record on the current device ({current}) while this tree's "
            f"attention caches are on npu:{index}; call torch.npu.set_device({index}) in this "
            "process before capturing, and do not capture from a thread with another card current"
        )


def _cache_devices(model: torch.nn.Module) -> set:
    """Every device an attention cache lives on, from the buffers named like one."""
    return {
        buffer.device
        for name, buffer in model.named_buffers()
        if name.endswith("window_kv_cache")
    }


def _capture_device(model: torch.nn.Module, backend_type: str) -> torch.device | object:
    """The one card the caches are on, or a refusal that says why this tree cannot be recorded.

    Three separate answers, and the second is the one worth stating: a cache on the host belongs to a
    tree with no card to record beside (the CUDA module's constructor says what an empty recording
    does there -- the loop repeats one token), and caches on two cards mean one recording cannot
    describe the tree at all.

    ``backend_type`` is the type the backend records on rather than a hard-coded ``"npu"``: that is
    the same ``"npu"`` for every resolution production performs, and it is what lets the CPU fixtures
    drive this class with a declared CPU backend without a card. The refusal a host-resident tree
    gets is therefore stated in terms of the two types it is between, which is what a reader needs.
    """
    devices = _cache_devices(model)
    if not devices:
        raise RuntimeError(
            "the NPU decode graphs replay the attention layers, and this model has no "
            "`window_kv_cache` buffer to size the step against -- it is not a V4.1 backbone"
        )
    if len(devices) > 1:
        raise RuntimeError(
            "the NPU decode graphs record one step on one card, and this tree's attention caches "
            f"are spread over {sorted(str(device) for device in devices)}; a single recording "
            "cannot describe that tree"
        )
    device = devices.pop()
    if device.type != backend_type:
        raise RuntimeError(
            f"the decode graphs are captured onto a {backend_type} device and replayed beside the "
            f"attention caches, and this model's are on {device}. A capture here records an empty "
            "graph that replays without advancing anything, so the generation would repeat one "
            "token; run the eager loop instead of asking a host-resident tree for graphs."
        )
    return device


def _linear_weights(module: torch.nn.Module):
    """``(name, weight)`` for every ``nn.Linear`` outside the host-only subtrees."""
    seen: set[int] = set()
    for name, child in module.named_modules():
        if _is_host_side(name) or not isinstance(child, torch.nn.Linear):
            continue
        if id(child) in seen:
            continue
        seen.add(id(child))
        yield f"{name}.weight", child.weight


def _require_consistent_metadata(block: torch.nn.Module) -> None:
    """Refuse a weight that carries half of the checkpoint's quantized pair.

    `mma.attach_scale` and `mma.attach_codes` are attached together, and the pair is what makes the
    stored bytes self-describing: the E4M3 codes are only the codes *of* the grid beside them. A
    weight left with one of the two is the state `mma.attach_scale` documents at length -- the
    surviving half is a runnable wrong answer, since plausible codes quantize to plausible codes --
    so it is refused here rather than recorded.

    This is deliberately **not** "every linear under a quantized module must carry the pair". It
    cannot be: `Attention._quantized` is set for the layer, and `wo_a` under it is applied as an
    einsum against the dense weight rather than through `quantized_linear` (`attention.py`, the
    block-diagonal `torch.einsum("bsgd,grd->bsgr", ...)` line), so it is legitimately metadata-free.
    Which linears the body quantizes is a property of the body, and the body is the thing that
    raises when it is missing the pair -- see `_require_capturable_block`.
    """
    for name, weight in _linear_weights(block):
        codes = mma.attached_codes(weight)
        scale = mma.attached_scale(weight)
        if (codes is None) != (scale is None):
            raise RuntimeError(
                f"{name} carries only half of the checkpoint's quantized pair "
                f"(codes attached: {codes is not None}, grid attached: {scale is not None}); the "
                "surviving half is a plausible wrong answer rather than an error, so the layer is "
                "refused instead of recorded"
            )


def _require_staged_quantized_modules(block: torch.nn.Module) -> None:
    """Refuse a `_quantized` module whose linears carry no checkpoint metadata at all.

    `_quantized` is the model's own record of "this weight was fp8 in the file" (`cfg.dtype ==
    "fp8"`), and it is what makes `quantized_linear` take the reference path. A module that says it
    is quantized while *none* of its linears carries the file's grid was therefore not staged through
    the original-byte loader: recording it would record the dense fallback, or fail later inside the
    capture. One metadata-carrying linear is the signal that the staging ran -- the individual
    weights that need the pair are enforced by the body that reads them.
    """
    for module_name, module in block.named_modules():
        if not getattr(module, "_quantized", False) or _is_host_side(module_name):
            continue
        if any(mma.attached_scale(weight) is not None for _, weight in _linear_weights(module)):
            continue
        raise RuntimeError(
            f"{module_name} is configured quantized, but none of its linears carries the "
            "checkpoint's E4M3 codes and E8M0 grid. Load the tree with online original-byte staging "
            "(`loader.load_backbone` on an NPU device) before capturing; a recording of the dense "
            "fallback would not be the reference reduction."
        )


def _require_capturable_block(block: torch.nn.Module, device: torch.device) -> None:
    """Refuse a block whose body cannot be recorded against the reference arithmetic.

    The placement half is the same check the CUDA path gets for free -- an eager CUDA body fails on a
    mixed tree with a device error, while a *recorded* body would freeze the addresses of whichever
    device the warm pass happened to run on.

    The metadata half is specific to this stack, and it is split in two on purpose. What can be
    decided from the tree is decided here: a half-attached pair is refused, and a module that claims
    to be quantized with no staged metadata anywhere under it is refused. What *cannot* be decided
    here is which individual linears the body quantizes -- `wo_a` sits under a quantized `Attention`
    and is an einsum against the dense weight -- so a weight missing the pair is refused by the body
    that needs it (`kernels.quantized_linear` -> `ascend_linear.quantized_linear_ascend`, which names
    the requirement). That refusal still lands **before any recording**: `_capture` runs the three
    real bodies first, so the exception comes out of a warm pass with zero captures made.
    """
    for name, parameter in block.named_parameters():
        if _is_host_side(name):
            continue
        if parameter.device != device:
            raise RuntimeError(
                f"{name} is on {parameter.device} and the NPU decode graphs record on {device}; a "
                "recorded body would capture the address of the wrong device"
            )
    for name, buffer in block.named_buffers():
        if _is_host_side(name):
            continue
        if buffer.device != device:
            raise RuntimeError(
                f"{name} is a {buffer.device} buffer in a block recorded on {device}"
            )
    _require_consistent_metadata(block)
    _require_staged_quantized_modules(block)


class NpuLayerGraphs(LayerGraphs):
    """One block's recorded decode forward, on an Ascend card.

    Only `_capture` differs from `graphs.LayerGraphs`; the two bodies, the routed call, the replay
    and the choice of which A recording a position replays are inherited unchanged. `api` is set by
    `NpuDecodeGraphs` on the instances it builds; a `NpuLayerGraphs` constructed on its own resolves
    the real ``torch.npu`` surface on its first capture.
    """

    api: _NpuGraphApi | None = None

    def _capture(self, pos, shared) -> None:  # type: ignore[override]
        """Record both halves, through the NPU backend, all-or-nothing.

        The order is the CUDA body's and it is load-bearing: `self.runtime` first (the published
        buffers outlive the forward that recorded them), then one real pass so every sink slot is
        allocated outside the pool, then the recordings.

        Two things this override adds besides the backend:

        * **the compressor's variant flag is restored in a `finally`.** `compress_variant` is a
          sticky attribute on a live module -- `True` forces the emitting body, `False` the filling
          one -- and a capture that raised between the two would otherwise leave the *next* forward
          running the wrong compressor branch. It is cleared to `None` whatever happens.
        * **the graph attributes are assigned only after every recording succeeded.**
          `LayerGraphs.ready` calls a dual-compressor layer ready on `graph_b` alone, so a layer that
          recorded its emit body and then failed would report itself ready and replay
          ``graph_a_fill`` -- which is ``None``. Keeping the assignments at the end makes a failed
          capture leave the layer not-ready, which is what makes `step` retry it.
        """
        api = self.api if self.api is not None else npu_graph_api()
        device = self.sink.x_in.device
        if device.type != api.device_type:
            raise RuntimeError(
                f"the NPU decode graphs record on {api.device_type} but this step's activation is "
                f"on {device}; the recording would capture an address on another device"
            )
        if device.type == NPU_GRAPH_DEVICE_TYPE:
            _require_current_device(api, device)
        self.runtime = shared
        self._body_a(pos, shared)
        self._routed()
        self._body_b()
        before = api.memory_allocated()
        compressor = self.compressor
        emit = fill = single = None
        try:
            if self.dual:
                compressor.compress_variant = True
                emit = _record(self._body_a_for(pos, shared), self.pool, self.warmup, api)
                compressor.compress_variant = False
                fill = _record(self._body_a_for(pos, shared), self.pool, self.warmup, api)
            else:
                single = _record(self._body_a_for(pos, shared), self.pool, self.warmup, api)
            recorded_b = _record(self._body_b, self.pool, self.warmup, api)
        finally:
            if compressor is not None:
                compressor.compress_variant = None
        self.graph_a_emit, self.graph_a_fill, self.graph_a = emit, fill, single
        self.graph_b = recorded_b
        self.pool_bytes = api.memory_allocated() - before


class NpuDecodeGraphs(DecodeGraphs):
    """A round of recorded decode steps on one Ascend card, and the pass that records them.

    The same arrangement as `graphs.DecodeGraphs` -- one shared pool, one `NpuLayerGraphs` per block,
    `capture_pass` inherited -- with the CUDA-specific pieces replaced: the pool handle, the cache
    flush, and the refusals that a recording of *this* tree would be a wrong answer. Constructing one
    is the opt-in: no CLI flag and no environment variable selects it.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_ids=None,
        warmup: int = CAPTURE_WARMUP,
        *,
        api: _NpuGraphApi | None = None,
    ) -> None:
        api = npu_graph_api() if api is None else api
        if not _reference_reduction_enabled():
            raise RuntimeError(
                "the NPU decode graphs record the reference reduction, and `POCKETLLM_V41_MMA_EMULATE` "
                "is disabled in this environment; recording the emulation-off body would record the "
                "throughput escape hatch instead. Unset the variable, or run the eager loop."
            )
        self.model = model
        self.warmup = warmup
        self.api = api
        # `api.device_type` is "npu" on every backend production resolves, and the only way to see
        # anything else is a caller that handed a backend in -- which is the CPU-test seam and not a
        # configuration of this adapter. The tree still has to match it, so a host-resident model is
        # refused by name for the type the recording would actually land on.
        self.device = _capture_device(model, api.device_type)
        # The pool is an allocator view *of one card*, so it is created with that card current --
        # the same check a layer's capture makes, made once here as well, because a pool made on the
        # wrong device is the failure that would otherwise surface as a wrong replay rather than as
        # an error.
        if self.device.type == NPU_GRAPH_DEVICE_TYPE:
            _require_current_device(api, self.device)
        self.pool = api.graph_pool_handle()
        self.captured = False
        self.layers: list[NpuLayerGraphs] = []
        wanted = None if layer_ids is None else set(layer_ids)
        for block in model.layers:
            if wanted is not None and block.layer_id not in wanted:
                continue
            # Layer by layer, before any recording: a tree that fails this is refused by name rather
            # than captured up to the layer that cannot be.
            _require_capturable_block(block, self.device)
            graphs = NpuLayerGraphs(block, self.pool, warmup)
            graphs.api = api
            block.decode_graph = graphs
            self.layers.append(graphs)

    def release(self) -> None:
        """Uninstall and forget the recordings. A round after this needs a new `NpuDecodeGraphs`.

        The graph objects are dropped **before** the cache flush so the pool's memory is freeable,
        and each layer's `decode_graph` is cleared so a layer that outlives the round runs the eager
        body rather than replaying into memory nothing owns. This is also the cleanup for a round
        whose capture failed part way through: the layers that were never recorded hold no graphs and
        the ones that failed hold none either (see `NpuLayerGraphs._capture`), so releasing always
        leaves every covered layer eager.
        """
        for graphs in self.layers:
            graphs.block.decode_graph = None
            graphs.graph_a = None
            graphs.graph_a_emit = None
            graphs.graph_a_fill = None
            graphs.graph_b = None
            graphs.runtime = None
        self.layers = []
        self.captured = False
        self.pool = None
        empty = self.api.empty_cache
        if empty is not None:
            empty()