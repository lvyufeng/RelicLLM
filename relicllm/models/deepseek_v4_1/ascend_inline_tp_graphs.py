"""Private prototype: one NPU graph per block, with the expert-TP body recorded inline.

This is the ``A -> expert TP -> B`` fusion that `task-105` scouted: instead of a decode step
replaying three graphs a block (``graph_a``, the dispatch's own expert graph, ``graph_b``), each
block records **one** graph whose body is

    _body_a(pos, shared)                      # everything up to and including the gate
    manager._execute(shard, ffn_in, w, idx)   # the expert-TP body, inline -- never .replay()
    _put(sink, "y_in", ...)
    _body_b()                                 # shared expert, TP8 reduce, residual fold

so a step is one ``replay()`` a block and no host copies of the routing tensors. Nothing about the
arithmetic moves: `_execute` is the same call the eager path makes, the same FP32 down-reduction
before the BF16 narrowing, the same ascending-expert combine kernel, the same group-2 then group-8
collectives. The sinks, the compression variants and the position contract are the existing ones.

**Three things are load-bearing and easy to get wrong.**

* **The expert body must be called inline.** A captured ``graph.replay()`` inside another capture is
  not a recording of the work; it is a host no-op, and the fused graph would then replay a block
  whose expert half never ran. This module therefore calls `manager._execute` directly and never
  `shard.forward` (which is the closure that owns the dispatch's own graph, and is what would nest).
* **The recordings are one per block, so the compressor's two bodies become two *full* graphs.** A
  block whose compressor can emit has two different A bodies; with A and B fused, that is two fused
  variants and `_graph_for` picks one per step from `pos.emits(ratio)`, exactly as before.
* **Capture position is a contract, not a detail.** The compressed-index branch is a host answer
  (``(pos.host + 1) // ratio == 0``), so the recorded body has the shape that branch produced. A
  recording made before the first closed group must not be replayed after one: the index tensor's
  width changes and the graph would write the wrong shape. `context_limit` is clamped for such a
  layer at capture time and every replay is refused past it, by name.

The class is a prototype and is deliberately not installed anywhere by default. The intended
wrapper sequence keeps one dispatch for the whole trial:

    dispatch = ExpertTPDispatch(..., capture_graphs=True)   # reference column and native 8-teacher
    ...reference / teacher pass runs, recording the dispatch's own per-layer graphs...
    api.synchronize()                                       # fence: no replay is in flight
    dispatch.graphs.clear()                                 # retire the recordings, keep pool/banks/groups
    dispatch.capture_graphs = False                         # no page faults from inside this round
    fuser = InlineTPDecodeGraphs(model, dispatch)

Nothing else changes hands: the banks, the tile workspace, the TP-2/EP groups and the dispatch's
graph pool are retained, and the dispatch keeps the execution-stream owner it latched during the
teacher. That is why the constructor validates ``owner_stream`` against the stream it will record on
rather than demanding it be unset -- after this retirement it is *supposed* to be set, and to the
same stream. A `graph_pool` left over from the retired round is retained, not reused: this round
records into the parent's own pool.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Callable, Iterable

import torch

from relicllm.models.deepseek_v4_1.graphs import restore, snapshot, _put
from relicllm.models.deepseek_v4_1.ascend_graphs import (
    NPU_GRAPH_DEVICE_TYPE,
    NpuDecodeGraphs,
    NpuLayerGraphs,
    _record,
    _require_current_device,
    npu_graph_api,
)
from relicllm.models.deepseek_v4_1.decode_pos import Pos

__all__ = [
    "InlineTPDecodeGraphs",
    "InlineTPLayerGraphs",
    "InlineTPGraphRefused",
    "describe_inline_tp_graphs",
]

#: A recording that lands on one card, on one stream, from one thread. The dispatch's own graph path
#: (``_forward_for``) does not run here, so ``manager.owner_stream`` stays ``None``; what replaces it
#: is this guard, which is checked on every replay rather than only inside the eager thin path.
_DEFAULT_MAX_POOL_BYTES = 1 << 30


class InlineTPGraphRefused(RuntimeError):
    """A refusal that names the precondition which does not hold.

    Separate from a plain ``RuntimeError`` so a caller (and the tests) can tell a *precondition* --
    wrong manager state, foreign thread or stream, a position the recordings do not cover -- from a
    failure of the recording itself. Both are refusals, not fallbacks: nothing here silently runs the
    eager body instead.
    """


# -- the dispatch it fuses ------------------------------------------------------------------------


def _stream_handle(handle: Any) -> int | None:
    """The raw stream id behind either shape of handle, or ``None`` if it has none.

    `ExpertTPDispatch._forward_for` latches ``torch.npu.current_stream(device).npu_stream`` -- an
    **int** -- while `torch.npu.current_stream()` returns a **wrapper**. Comparing those two objects
    directly is false even for the same stream, and comparing attributes is false too because the
    int has no ``npu_stream``. Both sides are therefore normalised to the raw id first; a ``repr``
    comparison can never bridge ``0`` and ``Stream(0)``, which is exactly the mistake this fixes.
    """
    if isinstance(handle, int) and not isinstance(handle, bool):
        return handle
    raw = getattr(handle, "npu_stream", None)
    if isinstance(raw, int) and not isinstance(raw, bool):
        return raw
    return None


def _same_stream(first: Any, second: Any) -> bool:
    """Whether two handles name the same device stream, in either of the shapes the tree uses.

    The raw id is compared first, because the dispatch stores an int and the API hands back a
    wrapper; wrapper equality, and ``repr`` as a last resort, only cover the cases where neither
    side exposes an id at all. This is a refusal guard, so a false negative is a refusal with a
    message rather than a wrong recording.
    """
    if first is second:
        return True
    left, right = _stream_handle(first), _stream_handle(second)
    if left is not None and right is not None:
        return left == right
    try:
        if bool(first == second):
            return True
    except Exception:  # noqa: BLE001 - an exotic handle must not mask the comparison below
        pass
    return repr(first) == repr(second)


def _require_inline_manager(
    manager: Any, *, device: torch.device | None = None, stream: Any = None
) -> None:
    """Refuse a manager this prototype cannot fuse, before a single graph is recorded.

    The three states that matter are the group's own: ``capture_graphs`` must be off (otherwise
    ``shard.forward`` would nest a replay, and the dispatch would also be recording its own graphs
    into a pool this round does not own), ``manager.graphs`` must be empty (a leftover entry means a
    replay was already captured for a layer, whose body would then be recorded twice into two
    different graphs), and the execution-stream owner must either be unset or be **this** stream.

    The last one is the retirement sequence's requirement rather than a relaxation: a dispatch that
    served the reference column and the native teacher with ``capture_graphs=True`` keeps the stream
    it latched. What has to hold is that the fused round records and replays on that same stream,
    because the expert body reads the TP-2 ring that was built for it. A *different* stream is the
    mistake; a set one is expected.

    ``device=None`` skips the card comparison, for the pass the constructor makes *before* the parent
    has resolved the recording device from the tree; everything else is checked there too, so a
    dispatch that cannot be fused is refused before any block is rebound.
    """
    from relicllm.models.deepseek_v4_1.ascend_expert_tp import COMPACT_BANK_FORMAT64

    if getattr(manager, "installed", False) is not True:
        raise InlineTPGraphRefused("the expert TP dispatch is not installed; install it before recording")
    if getattr(manager, "capture_graphs", True) is not False:
        raise InlineTPGraphRefused(
            "retire the expert TP dispatch's own graphs first: synchronize, clear `manager.graphs` "
            "and set `capture_graphs=False`. This prototype records the expert body inline, and a "
            "dispatch that also captures it would nest a replay"
        )
    if getattr(manager, "graphs", None):
        raise InlineTPGraphRefused(
            "the expert TP dispatch still holds captured graphs; retire them (synchronize, then "
            "`manager.graphs.clear()`) before recording the fused body"
        )
    owner = getattr(manager, "owner_stream", None)
    if owner is not None and (stream is None or not _same_stream(owner, stream)):
        raise InlineTPGraphRefused(
            "the expert TP dispatch's execution-stream owner is a different stream than this round "
            "would record on; the fused body and the TP-2 ring have to run on the stream the "
            "dispatch latched"
        )
    if getattr(manager, "bank_format", None) != COMPACT_BANK_FORMAT64:
        raise InlineTPGraphRefused(
            "inline TP layer graphs require the certified compact N64/K128 scale-LUT expert banks"
        )
    if getattr(manager, "scale_lut_certified", False) is not True:
        raise InlineTPGraphRefused("inline TP layer graphs require a scale-LUT certified expert bank")
    if device is not None and getattr(manager, "device", None) != device:
        raise InlineTPGraphRefused(
            f"the expert TP dispatch runs on {getattr(manager, 'device', None)} but the recordings land "
            f"on {device}; the fused body would capture an address on another device"
        )


def _shards_by_layer(manager: Any) -> dict[int, Any]:
    """``{layer_id: shard}`` from the dispatch's own replacement list, checked for identity.

    The shard is what the routed line of ``MoE.forward`` holds, so the mapping has to be the one the
    tree actually uses: a block whose ``ffn.routed`` is something else is a block this prototype must
    not record, because the fused body would read the shard's banks while the eager body read
    another store.
    """
    shards: dict[int, Any] = {}
    for entry in getattr(manager, "replacements", ()):
        try:
            ffn, _original, shard = entry
        except (TypeError, ValueError):  # pragma: no cover - defensive on a foreign manager
            raise InlineTPGraphRefused("the dispatch's replacement entries are not (ffn, original, shard)")
        layer_id = getattr(shard, "layer_id", None)
        if not isinstance(layer_id, int) or isinstance(layer_id, bool):
            raise InlineTPGraphRefused("an expert TP shard does not name its layer id")
        if layer_id in shards:
            raise InlineTPGraphRefused(f"two expert TP shards claim layer {layer_id}")
        if getattr(ffn, "routed", None) is not shard:
            raise InlineTPGraphRefused(
                f"layer {layer_id}: the block's routed store is not the shard this dispatch installed"
            )
        shards[layer_id] = shard
    if not shards:
        raise InlineTPGraphRefused("the expert TP dispatch installed no shards")
    return shards


# -- the execution-stream / thread guard ----------------------------------------------------------


class _ExecutionGuard:
    """One stream and one thread for the recordings and their replays.

    The dispatch's graph path refuses a second execution stream inside ``_forward_for``; the inline
    body never reaches that check, so the same promise is made here for both halves of the step. The
    stream is latched when the pass that records starts, because that is the stream every body in the
    step -- including the graph's own replays -- has to run on.
    """

    __slots__ = ("api", "stream", "thread")

    def __init__(self, api: Any) -> None:
        self.api = api
        self.thread = threading.get_ident()
        self.stream = None

    def latch(self) -> None:
        self.thread = threading.get_ident()
        self.stream = self.api.current_stream()

    def check(self) -> None:
        if threading.get_ident() != self.thread:
            raise InlineTPGraphRefused(
                "the inline TP layer graphs replay on the thread that recorded them; a graph replay "
                "from another thread would race the stream they were captured on"
            )
        if self.stream is not None and self.api.current_stream() != self.stream:
            raise InlineTPGraphRefused(
                "the inline TP layer graphs replay on the stream they were recorded on; the caller's "
                "current stream is a different one"
            )





# -- one block -----------------------------------------------------------------------------------


class InlineTPLayerGraphs(NpuLayerGraphs):
    """One block: ``A -> expert TP (inline) -> B`` recorded into a single graph per variant.

    ``ready`` is deliberately *not* the base property. The base calls a dual-compressor layer ready
    on ``graph_b`` alone, which is exactly the state a fused recording never produces; here a layer is
    ready when the one graph it needs -- or both variants -- exists, so a capture that failed between
    the variants leaves the layer not ready and the next step retries it.
    """

    #: Read by a trial wrapper (and the tests) to tell this layer's recording apart from a plain
    #: `NpuLayerGraphs` without isinstance gymnastics: the two expose different `ready` rules.
    is_inline_tp_layer_graph = True

    def __init__(
        self,
        block: torch.nn.Module,
        pool,
        warmup: int,
        manager: Any,
        shard: Any,
        *,
        guard: _ExecutionGuard,
        max_supported_context: int,
    ) -> None:
        super().__init__(block, pool, warmup)
        if isinstance(max_supported_context, bool) or not isinstance(max_supported_context, int):
            raise InlineTPGraphRefused("max_supported_context must be an int")
        if max_supported_context < 1:
            raise InlineTPGraphRefused("max_supported_context must cover at least one position")
        self.manager = manager
        self.shard = shard
        self.guard = guard
        self.max_supported_context = max_supported_context
        #: Clamped at capture time: a recording made before the first closed compressed group cannot
        #: be replayed after one, because the index tensor's width changes there.
        self.context_limit = max_supported_context
        self.fused = None
        self.fused_emit = None
        self.fused_fill = None
        #: Held until `release`'s fence: the graphs, the shared runtime whose published buffers they
        #: hold raw addresses into, and the sink. Dropping any of them while a replay is in flight is
        #: the use-after-free the fence exists to prevent.
        self.retained: list[Any] = []
        #: Driver counts for the trial's telemetry. Host-side integers only: incrementing one costs a
        #: Python add and changes no kernel, no buffer and no arithmetic.
        self.replays = 0
        self.replays_emit = 0
        self.replays_fill = 0

    def routing_indices(self) -> torch.Tensor | None:
        """The gate's `[1, topk]` device indices the last replay consumed, or ``None`` before one.

        The routing tensor is one of the sink slots, so a graph replay writes it into the same buffer
        every step and it stays readable afterwards -- which is what lets a profiler callback route
        telemetry from `sink.indices` after the replay instead of needing the gate's output
        separately. It is valid only until the next step's replay overwrites it: read it inside the
        callback, not after the loop.
        """
        return getattr(self.sink, "indices", None)

    # -- readiness -------------------------------------------------------------------------------

    @property
    def ready(self) -> bool:
        """Whether every fused graph this layer needs has been recorded."""
        if self.dual:
            return self.fused_emit is not None and self.fused_fill is not None
        return self.fused is not None

    # -- the entry point -------------------------------------------------------------------------

    def step(self, x, start_pos, pre_mix, image_mask, shared):  # type: ignore[override]
        """`LayerGraphs.step` with every refusal moved ahead of its first sink write.

        The inherited body copies ``x`` and ``pre_mix`` into the sink *before* it validates anything
        and only then captures or replays. A call that is going to be refused -- another thread,
        another stream, another card, a position outside the contract -- would therefore already have
        overwritten the two buffers a legitimate step in flight is reading, which is a race no graph
        capture makes safe. These are the same checks the body would eventually reach; they are just
        made first. `_replay` keeps its own copies because it is also reachable directly.
        """
        if image_mask is not None:
            # Delegated so the refusal and its message stay the base body's own. It raises before its
            # first `_put`, so nothing is written.
            return super().step(x, start_pos, pre_mix, image_mask, shared)
        self.guard.check()
        pos = Pos.of(start_pos)
        self._guard_position(pos)
        api = self.api if self.api is not None else npu_graph_api()
        # The recording's device, once it exists; before the first capture the step's own activation
        # is the only device there is to agree on, so both inputs still have to match each other.
        recorded = self.sink.x_in.device if self.sink.x_in is not None else x.device
        if x.device != recorded or pre_mix.device != recorded:
            raise InlineTPGraphRefused(
                f"this step's inputs are on {x.device} / {pre_mix.device} but the recordings are on "
                f"{recorded}; copying them into the sink would key a replay to another device"
            )
        if recorded.type == NPU_GRAPH_DEVICE_TYPE:
            _require_current_device(api, recorded)
        return super().step(x, start_pos, pre_mix, None, shared)

    # -- the fused body --------------------------------------------------------------------------

    def _tp_body(self) -> None:
        """`MoE.forward`'s routed line, inline: one call into the dispatch, no graph of its own."""
        sink = self.sink
        ffn = self.block.ffn
        view = sink.ffn_in.view(-1, ffn.dim)
        out = self.manager._execute(self.shard, view, sink.weights, sink.indices)
        # The same `_put` the eager path uses: the first pass clones a slot, every pass after copies
        # into it, and the recording holds that buffer's address rather than the returned tensor's.
        _put(sink, "y_in", out)

    def _fused_body(self, pos: "int | Pos", shared) -> None:
        """The whole block, in the order `Block.forward` runs it."""
        self._body_a(pos, shared)
        self._tp_body()
        self._body_b()

    # -- position contract -----------------------------------------------------------------------

    def _guard_position(self, pos: "int | Pos") -> int:
        """Refuse a position these recordings do not reproduce, and return its host value."""
        host = int(Pos.of(pos).host)
        if host < 1:
            raise InlineTPGraphRefused(
                f"position {host} is the prefill body, which is never recorded; the fused graphs "
                "start at the first decode position"
            )
        if host >= self.context_limit:
            raise InlineTPGraphRefused(
                f"position {host} is outside the {self.context_limit} positions these recordings "
                "reproduce; the compressed-index shape changes at the first closed group, so run the "
                "eager loop past that point"
            )
        return host

    def _resolve_context_limit(self, host: int) -> int:
        """How far this layer's recordings can be replayed, from the capture position.

        An index source whose capture position is *before* its first closed group records the empty
        index tensor. That shape is right only while the position stays in the same regime, so the
        limit is clamped to the last such position; a capture after the first group needs no clamp,
        because the branch is monotone from there on.
        """
        limit = self.max_supported_context
        # The index-source branches key on the *attention* layer's compress ratio, which is not
        # `self.ratio`: `self.ratio` is read off a compressor and is 0 on a layer that consumes
        # compressed positions without owning the KV compressor. Keying the clamp on the compressor's
        # ratio would miss the frozen empty-index case on exactly such a layer.
        ratio = getattr(self.block.attn, "compress_ratio", 0)
        if getattr(self.block.attn, "is_index_source", False):
            if isinstance(ratio, int) and not isinstance(ratio, bool) and ratio > 1:
                if (host + 1) // ratio == 0:
                    limit = min(limit, ratio - 1)
        return limit

    # -- capture ---------------------------------------------------------------------------------

    def _capture(self, pos: "int | Pos", shared) -> None:  # type: ignore[override]
        """Record the whole block, once per compressor variant, all-or-nothing.

        The order is the base class's and it is load-bearing: `self.runtime` first (the published
        buffers outlive the forward that recorded them), then one real pass so every sink slot is
        allocated outside the pool, then the recordings. Two differences:

        * one ``_record`` per variant rather than one for A and one for B, because A and B no longer
          have separate graphs to share;
        * ``compress_variant`` is restored in a ``finally`` and the graph attributes are assigned only
          after every recording succeeded, so a failed capture leaves the layer not ready (and the
          next step retries) instead of half-installed.
        """
        api = self.api if self.api is not None else npu_graph_api()
        device = self.sink.x_in.device
        if device.type != api.device_type:
            raise InlineTPGraphRefused(
                f"the NPU decode graphs record on {api.device_type} but this step's activation is on "
                f"{device}; the recording would capture an address on another device"
            )
        if device.type == NPU_GRAPH_DEVICE_TYPE:
            _require_current_device(api, device)
        _require_inline_manager(
            self.manager, device=device,
            stream=self.guard.stream if self.guard.stream is not None else api.current_stream(),
        )
        if getattr(self.block.ffn, "routed", None) is not self.shard:
            raise InlineTPGraphRefused(
                f"layer {getattr(self.block, 'layer_id', '?')}: the block no longer routes to the TP "
                "shard this recording was built for"
            )
        host = self._guard_position(pos)
        self.context_limit = self._resolve_context_limit(host)

        self.runtime = shared
        # Retained before anything is recorded: the runtime owns the published buffers the body holds
        # raw addresses into, and the sink owns every slot, so both must outlive a partial capture.
        self.retained.append((self.runtime, self.sink))
        self._body_a(pos, shared)
        self._tp_body()
        self._body_b()
        before = api.memory_allocated()
        compressor = self.compressor
        emit = fill = single = None
        try:
            if self.dual:
                compressor.compress_variant = True
                emit = _record(lambda: self._fused_body(pos, shared), self.pool, self.warmup, api)
                # Retained the moment it exists. The emitting graph has already captured nodes when
                # the filling variant is recorded, so a failure there must not let it be collected
                # before the release fence has run.
                self.retained.append(emit)
                compressor.compress_variant = False
                fill = _record(lambda: self._fused_body(pos, shared), self.pool, self.warmup, api)
                self.retained.append(fill)
            else:
                single = _record(lambda: self._fused_body(pos, shared), self.pool, self.warmup, api)
                self.retained.append(single)
        finally:
            if compressor is not None:
                compressor.compress_variant = None
        # Only now: a half-recorded layer must stay not-ready so `step` records it again.
        self.fused_emit, self.fused_fill, self.fused = emit, fill, single
        self.pool_bytes = api.memory_allocated() - before

    # -- replay ----------------------------------------------------------------------------------

    def _graph_for(self, pos: Pos):  # type: ignore[override]
        """Which fused recording this step replays -- the same host answer the base class uses."""
        if not self.dual:
            return self.fused
        return self.fused_emit if pos.emits(self.ratio) else self.fused_fill

    def _replay(self, pos: Pos, shared) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """One replay, and nothing else: no eager routed call between two halves.

        `marks` keeps the base class's keys so a caller's existing report still reads; ``routed`` and
        ``b`` stay zero because there is no separate section to time, and ``fused`` carries the
        whole-block figure under the ``a`` key.

        Three guards run before the launch and none of them changes the body: the execution stream and
        thread (one stream recorded, one stream replays), the position contract, and -- on a real card
        -- the current device by name. The device check is not redundant with the stream check: raw
        stream ids restart at 0 on every card, so a foreign card's default stream looks identical.
        """
        self.guard.check()
        self._guard_position(pos)
        graph = self._graph_for(pos)
        if graph is None:
            raise InlineTPGraphRefused(
                "this layer has no fused recording for the step's compression phase; it was captured "
                "for the other variant"
            )
        api = self.api if self.api is not None else npu_graph_api()
        device = self.sink.x_in.device
        if device.type == NPU_GRAPH_DEVICE_TYPE:
            # The stream guard above cannot see this: every card's default stream carries the same
            # raw id (0), so `current_stream()` on a *different* card compares equal to the one the
            # recording was made on. A replay launched there would run the graph against another
            # card's memory. The device is therefore checked by name, immediately before the launch.
            _require_current_device(api, device)
        sink = self.sink
        began = time.perf_counter()
        graph.replay()
        self.marks["a"] += time.perf_counter() - began
        self.replays += 1
        if self.dual:
            if pos.emits(self.ratio):
                self.replays_emit += 1
            else:
                self.replays_fill += 1
        return sink.x_out, sink.ffn_pre


def _saved_decode_graphs(model: Any) -> list[tuple[Any, Any]]:
    """What every block held in `decode_graph` before this constructor touched it.

    Taken before the parent's ``__init__`` runs, because that is what rebinds each covered block. If
    the round is refused afterwards -- a stale dispatch, a layer with no shard -- the model must not
    be left pointing at recordings that will never be captured.
    """
    saved: list[tuple[Any, Any]] = []
    for block in getattr(model, "layers", ()):
        saved.append((block, getattr(block, "decode_graph", None)))
    return saved


def _restore_decode_graphs(saved: list[tuple[Any, Any]]) -> None:
    """Put back the `decode_graph` bindings a failed construction replaced."""
    for block, previous in saved:
        block.decode_graph = previous


# -- the round -----------------------------------------------------------------------------------


class InlineTPDecodeGraphs(NpuDecodeGraphs):
    """One round of fused per-block recordings, and the pass that records them.

    Same shape as `NpuDecodeGraphs` -- the pool, the device, the refusals and `capture_pass` are the
    parent's -- with each covered block holding an `InlineTPLayerGraphs` instead of a
    `NpuLayerGraphs`. The dispatch is a constructor argument and is validated before any recording:
    this prototype fuses *that* dispatch's body, so the dispatch has to have retired its own graphs
    (`capture_graphs=False` with `graphs` empty, after an explicit synchronize) and must not have been
    promised to a different execution stream. Both are refusals, not fallbacks -- a nested replay or a
    foreign stream would produce a wrong replay rather than an error.

    `api` may be the CPU stub `ascend_graphs`' own tests use: every count and every refusal is the
    same through it, and only a card makes the recordings evidence about hardware. `describe()`
    reports `api_device_type` for exactly that reason.
    """

    #: Tests swap this for a probe layer that records the body order without a real block.
    LAYER_CLASS = InlineTPLayerGraphs

    #: Round-level counterpart of the layer's flag, for a wrapper that asks the driver first.
    is_inline_tp_decode_graphs = True

    def __init__(
        self,
        model: torch.nn.Module,
        manager: Any,
        *,
        warmup: int = 1,
        max_supported_context: int = 128,
        layer_ids: Iterable[int] | None = None,
        api: Any = None,
        max_pool_bytes: int | None = _DEFAULT_MAX_POOL_BYTES,
    ) -> None:
        # Resolve the backend and validate the dispatch *before* the parent rebinds `decode_graph` on
        # every covered block: a refusal here must leave the model exactly as it was, and there are
        # no captured nodes to retire yet.
        api = npu_graph_api() if api is None else api
        stream = api.current_stream()
        _require_inline_manager(manager, stream=stream)
        shards = _shards_by_layer(manager)
        saved = _saved_decode_graphs(model)
        try:
            super().__init__(model, layer_ids, warmup, api=api)
            self.manager = manager
            self.max_pool_bytes = max_pool_bytes
            self._guard = _ExecutionGuard(self.api)
            self._released = False
            self._retained: list[Any] = []
            # The stream this round records on, read once: the dispatch's owner has to be this one,
            # and latching it here makes the promise explicit before a single body runs.
            self.execution_stream = stream
            _require_inline_manager(manager, device=self.device, stream=stream)
            covered = [graphs.block for graphs in self.layers]
            if not covered:
                raise InlineTPGraphRefused("no block is covered by this round")
            # Validate the whole coverage mapping before any block is rebound to a fused layer, so a
            # dispatch that covers only part of the round is one refusal and not a partial install.
            for block in covered:
                if shards.get(block.layer_id) is None:
                    raise InlineTPGraphRefused(
                        f"layer {block.layer_id} is covered by this round but the expert TP dispatch "
                        "installed no shard for it; the fused body has no expert half to record"
                    )
            self.layers = []
            for block in covered:
                graphs = self.LAYER_CLASS(
                    block, self.pool, warmup, manager, shards[block.layer_id],
                    guard=self._guard, max_supported_context=max_supported_context,
                )
                graphs.api = self.api
                block.decode_graph = graphs
                self.layers.append(graphs)
        except BaseException:
            # Nothing was captured: this is a pure binding rollback, and it must cover the parent's
            # own partial install as well as anything refused after it.
            _restore_decode_graphs(saved)
            raise

    # -- the capture pass ------------------------------------------------------------------------

    def capture_pass(self, forward: Callable[[], object]) -> None:
        """Record every layer from one real step, then rewind the caches.

        `DecodeGraphs.capture_pass` with two additions: the execution stream and the calling thread
        are latched here, before the step that records, and the allocator delta the round spent is
        checked against an explicit budget afterwards. The budget is the controller's own
        ``memory_allocated`` delta -- node memory the driver keeps is not part of it -- so it is a
        guard against an obviously wrong recording, not a physical-peak measurement.
        """
        if self._released:
            raise InlineTPGraphRefused("this round was released; construct a new one")
        self._guard.latch()
        saved = snapshot(self.model)
        try:
            forward()
        finally:
            restore(self.model, saved)
        self.captured = True
        self._retained.append((self._guard, *self.layers))
        spent = self.pool_bytes
        if self.max_pool_bytes is not None and spent > self.max_pool_bytes:
            raise InlineTPGraphRefused(
                f"the inline TP layer graphs spent {spent} controller bytes, over the "
                f"{self.max_pool_bytes}-byte guard; call release() and record a smaller round"
            )

    # -- what the caller reports with ------------------------------------------------------------

    def marks(self) -> dict[str, float]:
        """The host-side clocks. ``fused`` is the whole block; ``routed``/``b`` are always zero."""
        out = super().marks()
        out["fused"] = out["a"]
        return out

    def describe(self) -> dict[str, Any]:
        """A compact report of what was recorded, for a trial log.

        ``driver_counts`` is what a wrapper labels the profiler scope with: a fused round is one
        replay a block, so ``replays`` is the whole-block figure and there is no separate
        ``graph_a``/``graph_b`` count to report. ``api_device_type`` says whether this was recorded
        on a card or through the CPU stub -- the two produce the same counts and only one of them is
        evidence about hardware.
        """
        counts = {
            "layers": len(self.layers),
            "graphs": sum(2 if graphs.dual else 1 for graphs in self.layers),
            "variants": sum(2 if graphs.dual else 1 for graphs in self.layers),
            "replays": sum(graphs.replays for graphs in self.layers),
            "replays_emit": sum(graphs.replays_emit for graphs in self.layers),
            "replays_fill": sum(graphs.replays_fill for graphs in self.layers),
        }
        return {
            "kind": "inline-tp-layer-graphs",
            "layers": len(self.layers),
            "captured": self.captured,
            "pool_bytes": self.pool_bytes,
            "max_pool_bytes": self.max_pool_bytes,
            "variants": counts["variants"],
            "api_device_type": getattr(self.api, "device_type", None),
            "driver_counts": counts,
            "context_limits": {
                str(getattr(graphs.block, "layer_id", "?")): graphs.context_limit
                for graphs in self.layers
            },
            "marks": self.marks(),
        }

    # -- release ---------------------------------------------------------------------------------

    def release(self) -> None:
        """Fence, then forget. A fence that fails retains everything so the release can be retried.

        The base class drops the graph references and then flushes the cache, which is right for a
        round with no replay in flight and is exactly wrong for one that may still have one: a pool
        freed under an in-flight replay is a device fault, not an exception. So the stream is
        synchronised **first**, and nothing is cleared unless it returned. If it raises, every
        reference stays -- graphs, sinks, the shared runtime that owns the published buffers, the
        pool and the manager -- and a second `release` retries the fence.

        It also refuses, before the fence, when the process is on a different card than the round was
        recorded on. That is a refusal rather than a best effort for the same reason: the fence is
        device-scoped, and every reference is retained so a caller that restores the device can retry.
        """
        if self._released:
            return
        # Validate ownership before the fence: releasing a layer another owner has taken over would
        # clear a binding this round does not own.
        for graphs in self.layers:
            block = graphs.block
            if getattr(block, "decode_graph", None) is not graphs:
                raise InlineTPGraphRefused(
                    f"layer {getattr(block, 'layer_id', '?')}: `decode_graph` is not this round's "
                    "recording; another owner replaced it, so this round no longer owns the layer"
                )
        # The fence. A failure here leaves every reference in place on purpose.
        #
        # `synchronize()` is device-scoped and a *different* card's default stream has the same raw
        # id as this round's, so the stream guard cannot tell the two apart. On a foreign card the
        # fence would wait on the wrong one and then free references this round still owns -- the
        # exact use-after-free it exists to prevent. The device is therefore checked by name first;
        # a refusal retains every reference, so a caller that restores the device can retry.
        if getattr(self, "device", None) is not None and self.device.type == NPU_GRAPH_DEVICE_TYPE:
            _require_current_device(self.api, self.device)
        self.api.synchronize()
        for graphs in self.layers:
            graphs.block.decode_graph = None
            graphs.graph_a = None
            graphs.graph_a_emit = None
            graphs.graph_a_fill = None
            graphs.graph_b = None
            graphs.fused = None
            graphs.fused_emit = None
            graphs.fused_fill = None
            graphs.runtime = None
            graphs.guard = None
            graphs.manager = None
            graphs.shard = None
            graphs.retained.clear()
        self.layers = []
        self.captured = False
        self.pool = None
        # The dispatch is dropped only now, after the fence: a retry after a failed fence still finds
        # this round intact and still holding every reference the retry needs.
        self.manager = None
        self._retained = []
        self._released = True
        empty = self.api.empty_cache
        if empty is not None:
            empty()


def describe_inline_tp_graphs(round_: "InlineTPDecodeGraphs") -> dict[str, Any]:
    """`InlineTPDecodeGraphs.describe` for a caller that holds the round behind a protocol."""
    return round_.describe()


#: The underscore-spelled names the trial wrapper was specified with; both spellings import the same
#: objects, so a caller can use either without a second definition.
_InlineTPLayerGraphs = InlineTPLayerGraphs
_InlineTPDecodeGraphs = InlineTPDecodeGraphs