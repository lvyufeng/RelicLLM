"""Record each resident routed expert's packed body once and replay it per decode step.

The routed experts are the one part of a V4.1 decode step that is neither on the recorded attention
body nor captureable *as a layer*: the routing indices are read on the host, so the layer walks its
experts in Python and calls :func:`ascend_packed_experts.packed_expert_forward` once per expert the
token drew. That call is a fixed sequence of device operations over fixed operands -- the resident
packed bytes of *that* expert -- with only two things varying between steps: the activation row and
the routing weight. So it can be recorded once per expert, into one shared pool, and replayed from
staging buffers that live outside the capture.

What is recorded, exactly
-------------------------
:func:`ascend_packed_experts.packed_expert_forward` itself, **unchanged**, called with

* the activation staging buffer ``input`` (``[1, dim]``, the store's dtype),
* the routing weight ``route_weight`` (``[1, 1]`` FP32),
* the resident expert's own :class:`~.ascend_packed_experts.PackedExpert` views (the file's bytes,
  read through the same code the eager path runs),
* the store's ``swiglu_limit`` / ``dtype`` / ``backend`` and its resolved packed operator,

and ending with ``sink.copy_(...)``, so the recorded region leaves its result in one staging buffer
allocated **outside** the pool. A replay is therefore: copy the current row in, copy the current
routing weight in, replay, hand back ``sink``. No arithmetic is rewritten, no kernel is substituted,
no operand is re-encoded: this module is a recording schedule, and the numbers it produces are the
numbers the eager body produces, bit for bit.

What stays outside a recording, and must
----------------------------------------
* **The host routing.** ``_routing`` reads the gate's indices on the host once per call; this module
  does not touch it and does not pretend a replay removes that read. The walk stays ``topk`` Python
  iterations per token and one dispatch per drawn expert.
* **The complete Engram.** Tables, hashes, lookups, projections and the gate stay on the CPU where
  the run already puts them; nothing here moves any of it.
* **The attention blocks and the dense projections.** They belong to
  :mod:`~.ascend_graphs`'s round, which is a separate recording.

The shared sink and stream order
--------------------------------
All experts write into one ``sink`` (``[1, dim]``). That is sound because the store consumes each
contribution before the next expert is dispatched: ``AscendPackedRoutedExperts.forward`` hands
``y.index_put_(..., y.index_select(...) + contribution.to(torch.float32))`` to the same stream that
the next replay writes on, so the read of the sink is ordered before the next recording's write. A
caller that keeps ``sink`` past the next dispatch would be reading a buffer that has been rewritten;
:attr:`RoutedExpertGraphs.sink` is therefore documented as a value to *consume*, not to retain, and
the store's own loop already does that.

Refusals
--------
Nothing here degrades into a stale recording. A call that does not match a registered expert triple
-- a prefill, more than one row, a different routing-weight shape, a different ``swiglu_limit``, a
different operator object, an expert this round did not record -- simply runs the original eager
body, and the pass-through count is reported. A provider with a **positive per-expert weight cache
budget** is refused at :meth:`RoutedExpertGraphs.install`, because recording per expert on top of a
store that also retains a widened bank per expert is the second full copy this design exists to
avoid. A capture failure, or a capture round that exceeds its graph or pool budget, releases the
whole round: there is no half-recorded state that would replay some experts and not others.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from . import ascend_packed_experts
from .ascend_graphs import _record, _require_current_device, npu_graph_api

__all__ = [
    "DEFAULT_MAX_GRAPHS",
    "DEFAULT_MAX_POOL_BYTES",
    "DISPATCHER_MARKER",
    "RoutedExpertGraphs",
]

#: Recording ceiling: 40 layers x 48 owned experts is 1,920 graphs a rank, so this bound leaves room
#: without being an unbounded promise about what a pool can hold.
DEFAULT_MAX_GRAPHS = 4096
#: What the pool accounting number is, in one place, so a report cannot be read as a physical size.
#: The backend's allocator counters measure *accounting*: a temporary freed inside a capture, and the
#: allocator's reserved-but-unused capacity, are only visible through the reserved counter (when the
#: build exposes one). Neither is a measured physical footprint of the recorded graphs.
POOL_BYTES_METRIC = (
    "max of the non-negative allocator deltas (memory_allocated, and memory_reserved when the "
    "backend exposes it) across the recording round; allocator accounting, not a measured physical "
    "footprint of the graphs"
)
#: Recording pool ceiling: a **bound** this round refuses past, not a measured capacity. What one
#: recording costs depends on the operator, because an operator that widens an operand *inside* the
#: capture puts that temporary in the pool as well -- the native Cube experiment's INT8-times-two
#: projection is several MiB per projection, so a full 1,920-expert round can exceed 512 MiB. The
#: count is therefore a deliberate refusal threshold that a caller raises once it has measured its own
#: operator and tree; it is not a promise that 40 layers of experts fit under this number, and the
#: comparison uses the allocator deltas below rather than a physical measurement.
DEFAULT_MAX_POOL_BYTES = 512 * 2**20
#: Set on the dispatcher this module installs, so a second round is refused by name rather than
#: silently replacing (or being replaced by) another owner of the module attribute.
DISPATCHER_MARKER = "_relic_routed_expert_graph_owner"
#: The attributes a capture takes from a store. Checked by name at install so a different store is
#: refused for the attribute it lacks, rather than failing inside a body.
_REQUIRED_STORE_ATTRS = (
    "owned",
    "_expert_at",
    "allocated",
    "dim",
    "dtype",
    "device",
    "backend",
    "swiglu_limit",
    "packed_gemv",
    "layer_id",
)
#: One installed round per model, keyed by ``id(model)``. A second install on the same tree would
#: capture into the first round's pool and patch the same global twice.
_MANAGERS: dict[int, "RoutedExpertGraphs"] = {}


def _tensor_version(tensor) -> int | None:
    """``tensor._version``, or ``None`` when this tensor does not track one.

    Inference tensors -- what ``torch.inference_mode()`` creates -- have no version counter, and an
    in-place write inside that mode has no counter to bump, so the counter is reported as absent
    instead of guessed. A recording whose sources carry no counter is therefore matched on
    pointer/shape/stride only, and the arena's **immutability** is the obligation that keeps that
    sound: this provider's arenas are written once at ``preload`` and never again. Reading the flag
    through ``torch.is_inference`` first keeps the hot path free of a raised exception for exactly the
    tensors that never have a counter.
    """
    try:
        is_inference = getattr(torch, "is_inference", None)
        if is_inference is not None and is_inference(tensor):
            return None
        return int(tensor._version)
    except RuntimeError:  # a build or a mode without a version counter
        return None


def _tensor_identity(tensor: torch.Tensor) -> tuple:
    """Every address-level fact a replay depends on for one resident tensor.

    A data pointer alone is not enough: a hand-built matrix can share a pointer and a shape with the
    recorded one while reading different elements through another stride or offset, and a recorded
    graph would then replay against the wrong operands. Strides and the storage offset are what pin
    "the same bytes in the same order", and the version counter is what notices an in-place write.
    """
    return (
        str(tensor.device),
        str(tensor.dtype),
        tuple(int(size) for size in tensor.shape),
        tuple(int(step) for step in tensor.stride()),
        int(tensor.storage_offset()),
        int(tensor.data_ptr()),
        _tensor_version(tensor),
    )


def _matrix_identity(matrix) -> tuple:
    """A stable identity for one projection's resident bytes.

    The store hands out a **fresh** ``PackedMatrix`` view per call (``_expert_at`` slices the arena),
    so object identity cannot key a recording. The arena slots' data pointers can: they are fixed for
    as long as the arena lives, and this module's entries hold the original views strongly, which is
    what keeps the arena -- and therefore the pointers -- alive underneath a recording.
    """
    return (_tensor_identity(matrix.codes), _tensor_identity(matrix.scales), int(matrix.block))


def _expert_identity(resident) -> tuple:
    """The key a recorded expert is looked up by: its three projection identities and the block."""
    return (
        _matrix_identity(resident.w1),
        _matrix_identity(resident.w2),
        _matrix_identity(resident.w3),
        int(resident.block),
    )


def _read_report(operator, name: str) -> dict | None:
    """An operator's own reporting helper, as a property or a no-argument method, when it has one."""
    attribute = getattr(type(operator), name, None)
    try:
        if isinstance(attribute, property):
            report = getattr(operator, name)
        else:
            report = getattr(operator, name, None)
            if callable(report):
                report = report()
    except Exception:  # a reporting helper that raises reports nothing
        return None
    return report if isinstance(report, dict) else None


def _declared_cache_budget(operator) -> tuple[bool, int, str]:
    """The per-expert weight-cache budget an operator declares, if it declares one.

    ``False`` means "no per-expert weight cache was declared": a plain function, or a wrapper around
    one, retains nothing between calls. A declared **positive** budget is what this round refuses,
    because a store that keeps a widened operand per expert beside the packed bank is a second copy
    of the weights, and recording per expert on top of it would be priced as if it were free.

    Several spellings are probed, because the operators this runtime can hand over do not agree on
    one -- an attribute or a ``cache_stats``/``describe()`` report, under ``budget``,
    ``budget_bytes`` or ``cache_budget_bytes``. Every probe is collected and the **largest** value is
    answered, so a zero under one name cannot mask a positive budget under another. A capacity
    ceiling such as ``max_cache_bytes`` is deliberately not probed: a provider configured for zero
    caching still reports its ceiling, and reading that as a configured budget would refuse an
    operator that retains nothing.
    """
    budget_names = ("cache_budget_bytes", "budget_bytes", "budget")
    report_names = ("cache_stats", "describe")
    seen: set[int] = set()
    biggest, source = 0, ""
    declared = False
    current = operator
    while current is not None and id(current) not in seen and len(seen) < 8:
        seen.add(id(current))
        for name in budget_names:
            value = getattr(current, name, None)
            if isinstance(value, int) and not isinstance(value, bool):
                declared = True
                if int(value) > biggest:
                    biggest, source = int(value), name
        for report_name in report_names:
            report = _read_report(current, report_name)
            if report is None:
                continue
            for name in budget_names:
                value = report.get(name)
                if isinstance(value, int) and not isinstance(value, bool):
                    declared = True
                    if int(value) > biggest:
                        biggest, source = int(value), f"{report_name}().{name}"
        # `functools.partial` and the like: the budget lives on the wrapped object.
        current = getattr(current, "func", None)
    return (declared, biggest, source)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


@dataclass
class _Coverage:
    """One layered store this round records, and the dispatch constants its graphs were built with."""

    store: Any
    layer_id: int
    operator: Any
    dtype: torch.dtype
    backend: str
    swiglu_limit: float

    @property
    def experts(self) -> tuple[int, ...]:
        return tuple(int(expert) for expert in self.store.owned)


@dataclass
class _Entry:
    """One recorded expert: its graph, the original views it reads, and the dispatch it belongs to.

    It deliberately holds **no tensor of its own**. The only bytes it can reach are the resident
    arenas through ``resident`` -- which is what stops a recording round from being a second weight
    bank -- and the graph, whose pool memory belongs to the round.
    """

    layer_id: int
    expert: int
    key: tuple
    resident: Any
    graph: Any
    dtype: torch.dtype
    backend: str
    swiglu_limit: float
    operator: Any
    capture_seconds: float
    replays: int = 0


class RoutedExpertGraphs:
    """A round of recorded routed-expert bodies for one resident-packed V4.1 tree.

    Constructing one changes nothing, allocates no graph and touches no global: :meth:`install`
    patches the module-level dispatcher, :meth:`capture_all` records every owned expert of every
    resident packed store, and :meth:`release` puts the original dispatcher back. That split exists
    so a caller can record up front, outside any timed window, and so a failed round has one place
    to undo itself.

    ``api`` is the recording adapter, resolved from :func:`~.ascend_graphs.npu_graph_api` unless one
    is handed in. A caller that hands one in is the CPU test seam: every production resolution is
    ``torch.npu``, and this class makes no claim about replay semantics for anything else.

    A round holds each recorded expert's original ``PackedExpert`` views, and those views are what
    keep the arena they slice alive -- which is what makes a data-pointer identity safe to key a
    recording by. The consequence is stated rather than hidden: **release this round before releasing
    the stores it recorded**, or the old arenas stay allocated; :meth:`describe` reports
    ``sources_changed_since_capture`` when a store has been reloaded underneath a live round, and a
    dispatch after that falls back to the eager body instead of replaying against new bytes.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        warmup: int = 1,
        api=None,
        max_graphs: int = DEFAULT_MAX_GRAPHS,
        max_pool_bytes: int = DEFAULT_MAX_POOL_BYTES,
    ) -> None:
        if model is None:
            raise ValueError("a recording round needs the tree whose routed experts it records")
        for name, value in (("warmup", warmup), ("max_graphs", max_graphs), ("max_pool_bytes", max_pool_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        self.model = model
        self.warmup = warmup
        self._api = api
        self.max_graphs = max_graphs
        self.max_pool_bytes = max_pool_bytes
        self.api = None
        self.pool = None
        self.input: torch.Tensor | None = None
        self.route_weight: torch.Tensor | None = None
        self.sink: torch.Tensor | None = None
        self._coverage: list[_Coverage] = []
        self._skipped_layers: list[int] = []
        self._entries: dict[tuple, _Entry] = {}
        self._original = None
        self._dispatcher = None
        self._installed = False
        self._captured = False
        self._dim = 0
        self._dtype: torch.dtype | None = None
        self._device: torch.device | None = None
        self._pool_bytes = 0
        self._allocated_delta = 0
        self._reserved_delta: int | None = None
        self._memory_reserved = None
        self._capture_seconds = 0.0
        self._replays = 0
        self._passthroughs = 0

    # -- properties -----------------------------------------------------------

    @property
    def installed(self) -> bool:
        return self._installed

    @property
    def captured(self) -> bool:
        return self._captured

    @property
    def graphs(self) -> int:
        return len(self._entries)

    @property
    def replays(self) -> int:
        return self._replays

    @property
    def passthroughs(self) -> int:
        """Calls that did not match a recording and ran the original eager body."""
        return self._passthroughs

    # -- install --------------------------------------------------------------

    def install(self) -> "RoutedExpertGraphs":
        """Validate the tree, allocate the staging buffers and patch the dispatcher.

        No graph is recorded here. The pool handle is created once, here, because one pool shared by
        every recording is the difference between a round and 1,920 private pools.
        """
        if self._installed:
            return self
        if id(self.model) in _MANAGERS:
            raise RuntimeError(
                "a RoutedExpertGraphs round is already installed for this model; one round owns the "
                "dispatcher and the pool, so a second would capture into the first round's pool and "
                "patch the same module attribute twice. Release the installed round first"
            )
        installed = getattr(ascend_packed_experts, "packed_expert_forward", None)
        owner = getattr(installed, DISPATCHER_MARKER, None)
        if owner is not None and owner is not self:
            raise RuntimeError(
                "another recording round already owns ascend_packed_experts.packed_expert_forward; "
                "this adapter is single-owner by design, so the owner has to release it first"
            )
        api = npu_graph_api() if self._api is None else self._api
        # Optional second allocator counter. `memory_allocated` alone under-reports a capture pool: a
        # temporary freed inside a recorded region, and the allocator's reserved-but-unused capacity,
        # only show up in `memory_reserved`. Probed once, and treated as absent when the build (or a
        # fake) does not expose it -- the round then reports the allocated delta alone and says so.
        reserved_probe = getattr(getattr(api, "module", None), "memory_reserved", None)
        self._memory_reserved = reserved_probe if callable(reserved_probe) else None
        coverage = self._discover()
        first = coverage[0]
        dim, dtype, device = int(first.store.dim), first.dtype, first.store.device
        # A recording is made on the backend's current device and its replay runs wherever the caller's
        # stream is, so the buffers a body reads have to be on the card the capture lands on. Checked
        # here -- before the pool handle and before any staging allocation -- because the failure mode
        # otherwise is a capture that succeeds and a replay that is a wrong answer.
        self._require_capturable_device(api, device)
        pool = api.graph_pool_handle()
        # Allocated before any capture: these are the buffers a replay writes into and reads from, so
        # they must not be part of any recording's pool memory. Built into locals first, so a failed
        # allocation leaves no half-installed round behind.
        input_row = torch.zeros((1, dim), dtype=dtype, device=device)
        route_weight = torch.zeros((1, 1), dtype=torch.float32, device=device)
        sink = torch.zeros((1, dim), dtype=dtype, device=device)
        original = installed
        dispatcher = self._make_dispatcher(original)
        setattr(dispatcher, DISPATCHER_MARKER, self)
        self.api = api
        self._coverage = coverage
        self._dim = dim
        self._dtype = dtype
        self._device = device
        self.pool = pool
        self.input = input_row
        self.route_weight = route_weight
        self.sink = sink
        self._original = original
        self._dispatcher = dispatcher
        ascend_packed_experts.packed_expert_forward = dispatcher
        self._installed = True
        _MANAGERS[id(self.model)] = self
        return self

    @staticmethod
    def _require_capturable_device(api, device: torch.device) -> None:
        """Refuse a tree whose resident experts are not on the device this backend records on.

        Two separate questions, and both are asked before anything is allocated:

        * the **type** has to match the backend's. A store on one device type recorded through a
          backend for another is not a slow path, it is a recording whose operands a replay cannot
          reach: the capture would succeed and the answer would be wrong.
        * on a real ``npu`` backend the store also has to be on the **current** card, which is what
          :func:`~.ascend_graphs._require_current_device` states -- a pool and a replay belong to the
          card that was current when they were made.

        The CPU seam stays explicit: a caller that hands in a backend declaring ``device_type="cpu"``
        with a host-resident store passes the type check and skips the card check, because a host
        fixture has no card to be current. Nothing in production resolves that seam -- every
        resolution is ``torch.npu`` -- so a mismatch there is refused rather than papered over.
        """
        if device.type != api.device_type:
            raise RuntimeError(
                f"the resident packed experts are on {device} but this recording backend captures on "
                f"{api.device_type!r}; a replay reads the store's own arenas, so the two have to be "
                "the same device. Nothing was allocated and nothing was patched"
            )
        if api.device_type == "npu":
            # Reuse the attention adapter's check so the condition and its single source of truth are
            # shared, but say it in this round's terms: the helper's own message is about attention
            # caches, and these recordings read expert arenas.
            try:
                _require_current_device(api, device)
            except RuntimeError as error:
                raise RuntimeError(
                    f"the resident packed experts are on {device} but this process's current NPU "
                    f"card is {int(api.current_device())}; a recording and its pool belong to the "
                    f"card that was current when they were made. {error}"
                ) from error

    def _discover(self) -> list[_Coverage]:
        """Every resident packed store in the tree, validated, or a refusal that names the problem."""
        coverage: list[_Coverage] = []
        skipped: list[int] = []
        for layer in getattr(self.model, "layers", ()) or ():
            routed = getattr(getattr(layer, "ffn", None), "routed", None)
            if routed is None:
                continue
            if not hasattr(routed, "owned") or not hasattr(routed, "_expert_at"):
                # A bounded or otherwise non-resident store is not this round's to record: its
                # expert bytes are not resident, so there is nothing stable to capture against.
                skipped.append(int(getattr(layer, "layer_id", -1)))
                continue
            coverage.append(self._validate(routed, layer))
        if not coverage:
            raise RuntimeError(
                "no resident packed routed-expert store was found on this tree; a recording round "
                "captures the packed provider's expert bodies, and there is nothing here to record"
            )
        self._skipped_layers = skipped
        devices = {str(one.store.device) for one in coverage}
        dtypes = {one.dtype for one in coverage}
        dims = {int(one.store.dim) for one in coverage}
        if len(devices) > 1 or len(dtypes) > 1 or len(dims) > 1:
            raise RuntimeError(
                "the resident packed stores disagree about their geometry "
                f"(devices {sorted(devices)}, dtypes {sorted(_dtype_name(one) for one in dtypes)}, "
                f"dims {sorted(dims)}); one round stages one activation row on one card"
            )
        expected = sum(len(one.experts) for one in coverage)
        if expected > self.max_graphs:
            raise RuntimeError(
                f"this round would record {expected} expert graphs, above the {self.max_graphs} "
                "ceiling; raise max_graphs deliberately rather than recording an unbounded round"
            )
        return coverage

    def _validate(self, store, layer) -> _Coverage:
        suffix = getattr(layer, "layer_id", "?")
        missing = [name for name in _REQUIRED_STORE_ATTRS if not hasattr(store, name)]
        if missing:
            raise RuntimeError(
                f"layer {suffix}: the routed store is not the resident packed provider (it has no "
                f"{', '.join(missing)}); only packed resident experts can be recorded, because a "
                "recording reads their arena views"
            )
        if not store.allocated:
            raise RuntimeError(
                f"layer {suffix}: the packed store is not preloaded, so it has no arena for a "
                "recording to read; preload it before installing a recording round"
            )
        operator = store.packed_gemv
        found, budget, field = _declared_cache_budget(operator)
        if found and budget > 0:
            raise RuntimeError(
                f"layer {suffix}: the packed operator declares a per-expert weight-cache budget of "
                f"{budget} bytes ({field}); recording per expert on top of a store that also retains "
                "a widened operand per expert would be the second full weight bank this design "
                "refuses. Pass the same operator with a zero cache budget"
            )
        # The operator the store will actually call: the store passes the *resolved* operator per
        # call, and a recording has to call the same object for its replay to belong to this store.
        resolved = ascend_packed_experts._resolve_packed_gemv(operator)
        for expert in store.owned:
            resident = store._expert_at(int(expert))
            for matrix in (resident.w1, resident.w2, resident.w3):
                if not isinstance(matrix.codes, torch.Tensor) or not isinstance(matrix.scales, torch.Tensor):
                    raise RuntimeError(
                        f"layer {suffix}: expert {expert} returned non-tensor projection bytes"
                    )
                if matrix.codes.dtype != torch.uint8 or matrix.scales.dtype != torch.uint8:
                    raise RuntimeError(
                        f"layer {suffix}: expert {expert} returned "
                        f"{matrix.codes.dtype}/{matrix.scales.dtype} bytes; a recording reads the "
                        "original packed uint8 payload, not a widened form"
                    )
                if matrix.codes.ndim != 2 or matrix.scales.ndim != 2:
                    raise RuntimeError(
                        f"layer {suffix}: expert {expert} returned a {matrix.codes.ndim}-D projection"
                    )
                if not matrix.codes.is_contiguous() or not matrix.scales.is_contiguous():
                    raise RuntimeError(
                        f"layer {suffix}: expert {expert} returned a strided projection; the arena's "
                        "own slice views are contiguous, so this means a hand-built expert"
                    )
                if int(matrix.block) != ascend_packed_experts.BLOCK:
                    raise RuntimeError(
                        f"layer {suffix}: expert {expert} declares a {int(matrix.block)}-wide K block; "
                        f"the released packed geometry is {ascend_packed_experts.BLOCK}"
                    )
        return _Coverage(
            store=store,
            layer_id=int(getattr(layer, "layer_id", -1)),
            operator=resolved,
            dtype=store.dtype,
            backend=store.backend,
            swiglu_limit=float(store.swiglu_limit),
        )

    # -- capture --------------------------------------------------------------

    def capture_all(self) -> dict:
        """Record one graph per owned expert of every covered store, then report what was recorded.

        Idempotent: a round that has already captured reports zero new graphs rather than recording a
        second set. On any failure -- a capture that raises, or a pool that grows past
        ``max_pool_bytes`` -- the whole round is released, so a failed round leaves no half-recorded
        state that would replay some experts and run others eagerly.
        """
        if not self._installed:
            raise RuntimeError("capture_all needs install() first; nothing is patched to record into")
        if self._captured:
            return {
                "already_captured": True,
                "captured": 0,
                "graphs": len(self._entries),
                **self._memory_report(),
            }
        baselines = self._memory_baselines()
        started = time.perf_counter()
        captured = 0
        try:
            for coverage in self._coverage:
                for expert in sorted(coverage.experts):
                    self._capture_one(coverage, expert)
                    captured += 1
                    allocated_delta, reserved_delta = self._memory_deltas(*baselines)
                    spent = self._spent(allocated_delta, reserved_delta)
                    if spent > self.max_pool_bytes:
                        raise RuntimeError(
                            f"the recording round's allocator growth is {spent} bytes "
                            f"(memory_allocated delta {allocated_delta}, memory_reserved delta "
                            f"{reserved_delta}) after {captured} graphs, above the "
                            f"{self.max_pool_bytes}-byte ceiling; raise max_pool_bytes deliberately "
                            "or record fewer layers"
                        )
        except BaseException:
            self.release()
            raise
        self._allocated_delta, self._reserved_delta = self._memory_deltas(*baselines)
        self._pool_bytes = self._spent(self._allocated_delta, self._reserved_delta)
        self._capture_seconds = time.perf_counter() - started
        self._captured = True
        return {
            "captured": captured,
            "graphs": len(self._entries),
            **self._memory_report(),
            "capture_seconds": self._capture_seconds,
            "warmup": self.warmup,
            "layers": [
                {
                    "layer_id": coverage.layer_id,
                    "experts": len(coverage.experts),
                    "backend": coverage.backend,
                    "swiglu_limit": coverage.swiglu_limit,
                }
                for coverage in self._coverage
            ],
        }

    # -- allocator accounting -------------------------------------------------

    def _memory_baselines(self) -> tuple[int, int | None]:
        """The allocator counters before the first capture; reserved is ``None`` when not exposed."""
        return (int(self.api.memory_allocated()), self._memory_reserved_value())

    def _memory_reserved_value(self) -> int | None:
        """The backend's reserved counter, or ``None`` when it is absent or unreadable."""
        if self._memory_reserved is None:
            return None
        try:
            return int(self._memory_reserved())
        except Exception:  # a counter that cannot be read is reported as unavailable, never guessed
            return None

    def _memory_deltas(self, allocated_baseline: int, reserved_baseline: int | None) -> tuple[int, int | None]:
        """Non-negative growth since the baselines, separately for the two counters.

        Clamped at zero because a capture may *release* capacity (a warm pass can hand buffers back),
        and a shrinking counter is not negative growth. Both are reported: the allocated delta is what
        the pool holds at the end of the round, and the reserved delta is what the allocator kept for
        it -- the second is where a temporarily freed capture allocation and reserved capacity show up,
        and it is the one `memory_allocated` alone cannot express.
        """
        allocated_delta = max(0, int(self.api.memory_allocated()) - int(allocated_baseline))
        if self._memory_reserved is None or reserved_baseline is None:
            return allocated_delta, None
        reserved_now = self._memory_reserved_value()
        if reserved_now is None:
            return allocated_delta, None
        return allocated_delta, max(0, reserved_now - int(reserved_baseline))

    @staticmethod
    def _spent(allocated_delta: int, reserved_delta: int | None) -> int:
        """The number the budget is enforced on: the largest non-negative delta available."""
        return allocated_delta if reserved_delta is None else max(allocated_delta, reserved_delta)

    def _memory_report(self) -> dict:
        """Allocator accounting for this round, with `pool_bytes` kept and labelled rather than bare."""
        return {
            "pool_bytes": self._pool_bytes,
            "pool_bytes_metric": POOL_BYTES_METRIC,
            "allocated_delta": self._allocated_delta,
            "reserved_delta": self._reserved_delta,
            "memory_reserved_available": self._memory_reserved is not None,
            "physical_footprint_measured": False,
        }

    def _capture_one(self, coverage: _Coverage, expert: int) -> None:
        resident = coverage.store._expert_at(int(expert))
        key = _expert_identity(resident)
        if key in self._entries:
            raise RuntimeError(
                f"layer {coverage.layer_id}: two authorities claim the same resident bytes; a "
                "recording would be ambiguous, so the round refuses instead of guessing"
            )
        body = self._body(resident, coverage)
        started = time.perf_counter()
        # `_record` warms the body on a side stream before recording it, which is what makes the
        # capture find its allocations already in the free list. One shared pool for every expert.
        graph = _record(body, self.pool, self.warmup, self.api)
        self._entries[key] = _Entry(
            layer_id=coverage.layer_id,
            expert=int(expert),
            key=key,
            resident=resident,
            graph=graph,
            dtype=coverage.dtype,
            backend=coverage.backend,
            swiglu_limit=coverage.swiglu_limit,
            operator=coverage.operator,
            capture_seconds=time.perf_counter() - started,
        )

    def _body(self, resident, coverage: _Coverage):
        """The recorded region: the original eager body, ending in a copy into the shared sink."""
        original = self._original

        def body():
            produced = original(
                self.input,
                resident,
                swiglu_limit=coverage.swiglu_limit,
                weights=self.route_weight,
                dtype=coverage.dtype,
                backend=coverage.backend,
                packed_gemv=coverage.operator,
                prefill=False,
            )
            self.sink.copy_(produced)

        return body

    # -- the dispatcher -------------------------------------------------------

    def _make_dispatcher(self, original):
        """The module-level ``packed_expert_forward`` this round installs.

        Its signature is the original's, because every caller already passes these arguments by
        keyword. A call that matches a recorded expert replays it; every other call runs ``original``
        exactly as it would have without this round installed.
        """

        def packed_expert_forward(
            x: torch.Tensor,
            resident,
            *,
            swiglu_limit: float = 0.0,
            weights: torch.Tensor | None = None,
            dtype: torch.dtype | None = None,
            backend: str = "auto",
            packed_gemv=None,
            prefill: bool | None = None,
        ) -> torch.Tensor:
            entry = self._match(
                x,
                resident,
                swiglu_limit=swiglu_limit,
                weights=weights,
                dtype=dtype,
                prefill=prefill,
                packed_gemv=packed_gemv,
                backend=backend,
            )
            if entry is None:
                self._passthroughs += 1
                return original(
                    x,
                    resident,
                    swiglu_limit=swiglu_limit,
                    weights=weights,
                    dtype=dtype,
                    backend=backend,
                    packed_gemv=packed_gemv,
                    prefill=prefill,
                )
            self.input.copy_(x)
            self.route_weight.copy_(weights)
            entry.graph.replay()
            entry.replays += 1
            self._replays += 1
            return self.sink

        return packed_expert_forward

    def _match(self, x, resident, *, swiglu_limit, weights, dtype, prefill, packed_gemv, backend):
        """The recording this call belongs to, or ``None`` for the eager body.

        Every check is a host-side shape/dtype/identity comparison -- no device read, no
        synchronisation -- and any mismatch answers ``None``, which is the eager path. That is the
        only safe direction: a recording whose operands changed is not a recording of this call.
        """
        if not self._captured:
            return None
        if not isinstance(x, torch.Tensor) or x.ndim != 2:
            return None
        if tuple(x.shape) != (1, self._dim):
            return None
        if x.device != self._device or x.dtype != self._dtype:
            return None
        # Prefill takes the widening-and-block-GEMM path, which is a different body. Compared by
        # identity rather than membership, so a non-bool argument is never coerced into a match.
        if prefill is not None and prefill is not False:
            return None
        if not isinstance(weights, torch.Tensor):
            return None
        if tuple(weights.shape) != (1, 1) or weights.dtype != torch.float32:
            return None
        if weights.device != self._device:
            return None
        if not isinstance(swiglu_limit, (int, float)) or isinstance(swiglu_limit, bool):
            return None
        if dtype is not None and dtype != self._dtype:
            return None
        try:
            key = _expert_identity(resident)
        except AttributeError:
            # Not a packed resident expert at all: nothing this round recorded.
            return None
        entry = self._entries.get(key)
        if entry is None:
            return None
        # The remaining checks belong to *this* entry, because they can differ per store.
        if float(swiglu_limit) != entry.swiglu_limit:
            return None
        if backend != entry.backend:
            return None
        if packed_gemv is not entry.operator:
            return None
        if dtype is not None and dtype != entry.dtype:
            return None
        return entry

    # -- release and report ---------------------------------------------------

    def release(self) -> dict:
        """Restore the original dispatcher and drop every recording, the pool and the buffers.

        The graphs are dropped before the allocator cache flush, so the pool's memory is freeable;
        the dispatcher is only restored if it is still **this** round's, so a foreign replacement of
        the module attribute is reported rather than overwritten.
        """
        restored = False
        foreign = None
        current = getattr(ascend_packed_experts, "packed_expert_forward", None)
        if self._installed and self._original is not None:
            if getattr(current, DISPATCHER_MARKER, None) is self:
                ascend_packed_experts.packed_expert_forward = self._original
                restored = True
            elif current is not self._original:
                foreign = f"{type(current).__name__}"
        report = {
            "graphs": len(self._entries),
            "replays": self._replays,
            "passthroughs": self._passthroughs,
            **self._memory_report(),
            "capture_seconds": self._capture_seconds,
            "dispatcher_restored": restored,
            "dispatcher_replaced_by_another_owner": foreign,
        }
        self._entries.clear()
        self._coverage = []
        self._captured = False
        self._installed = False
        self._dispatcher = None
        self._original = None
        self.pool = None
        self.input = None
        self.route_weight = None
        self.sink = None
        self._pool_bytes = 0
        # The counters describe a *released* round from here on; `install` re-probes the capability.
        self._allocated_delta = 0
        self._reserved_delta = None
        self._memory_reserved = None
        _MANAGERS.pop(id(self.model), None)
        if self.api is not None:
            empty = getattr(self.api, "empty_cache", None)
            if empty is not None:
                empty()
        return report

    def describe(self) -> dict:
        """JSON-able state of this round. Runs no device work and captures nothing."""
        return {
            "installed": self._installed,
            "captured": self._captured,
            "warmup": self.warmup,
            "graphs": len(self._entries),
            "experts_expected": sum(len(one.experts) for one in self._coverage),
            "covered_layers": [one.layer_id for one in self._coverage],
            "skipped_layers": list(self._skipped_layers),
            "max_graphs": self.max_graphs,
            # Kept at the top level for a caller that already reads it, with the metric spelled out
            # beside it and both deltas inside `memory`.
            "pool_bytes": self._pool_bytes,
            "pool_bytes_metric": POOL_BYTES_METRIC,
            "memory": self._memory_report(),
            "max_pool_bytes": self.max_pool_bytes,
            "capture_seconds": self._capture_seconds,
            "replays": self._replays,
            "passthroughs": self._passthroughs,
            "device": None if self._device is None else str(self._device),
            "dtype": None if self._dtype is None else _dtype_name(self._dtype),
            "dim": self._dim,
            "staging": {
                "input": None if self.input is None else {"shape": list(self.input.shape),
                                                          "dtype": _dtype_name(self.input.dtype),
                                                          "inside_capture_pool": False},
                "route_weight": None if self.route_weight is None else {
                    "shape": list(self.route_weight.shape),
                    "dtype": _dtype_name(self.route_weight.dtype),
                    "inside_capture_pool": False,
                },
                "sink": None if self.sink is None else {"shape": list(self.sink.shape),
                                                        "dtype": _dtype_name(self.sink.dtype),
                                                        "inside_capture_pool": False},
            },
            "operators": [
                {
                    "layer_id": one.layer_id,
                    "backend": one.backend,
                    "swiglu_limit": one.swiglu_limit,
                    "declared_cache_budget": _declared_cache_budget(one.operator)[1],
                }
                for one in self._coverage
            ],
            "sources_changed_since_capture": self._changed_sources(),
            "held_source_arenas": len({entry.layer_id for entry in self._entries.values()}),
            "numerical_contract": "original packed_expert_forward, unchanged; replay only",
            "host_routing_remains_host_side": True,
            "complete_engram_remains_on_cpu": True,
        }

    def _changed_sources(self) -> list[int]:
        """Layers whose store no longer offers the arena identity the recordings were made against.

        A store that was released and preloaded again allocates new arenas, so every recording for it
        stops matching (a dispatch falls back to the eager body, never to a stale replay). The
        recordings still hold the **old** views, which is what keeps those arenas from being recycled
        underneath a graph -- so a round must be released before its stores are. This reports the
        layers where that has happened, so the condition is visible rather than inferred.
        """
        if not self._captured:
            return []
        changed: list[int] = []
        for coverage in self._coverage:
            store = coverage.store
            try:
                expert = int(store.owned[0])
                current = _expert_identity(store._expert_at(expert))
            except Exception:
                changed.append(coverage.layer_id)
                continue
            recorded = {
                entry.key for entry in self._entries.values() if entry.layer_id == coverage.layer_id
            }
            if current not in recorded:
                changed.append(coverage.layer_id)
        return changed

    def __enter__(self) -> "RoutedExpertGraphs":
        return self

    def __exit__(self, *exc_info) -> bool:
        self.release()
        return False