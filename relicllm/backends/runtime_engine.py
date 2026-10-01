"""What a supervised rank of a Python runtime is made of.

The serving lifecycle is Python (see RelicLLM#18): admission, KV accounting, sampling and
cancellation are ordinary Python in this package. What is left here is the part of that lifecycle
that is *plumbing* rather than scheduling -- the supervised worker loop, the card a rank binds, and
the shape a runtime presents to the bridge.

[`RankedWorker`] is the worker half: a rank joins the process group, runs its adapter as a plain
object, and serves the messages rank 0 sends it. `visible_card_count` and `card_for_rank` are what
a rank works its card out from, ahead of the runtime being selected.

The native `BatchScheduler` this module once dressed the Python runtimes in has been removed along
with the `cpp` backend: it was a C++ library, it lived in the archived `relic-engine`, nothing
built it, and at a declared width of one it was a queue rather than a batcher -- which is what the
adapters' own request lock already is. [`RuntimeRun`], `RuntimeSpec`, `engine_class`,
`make_runtime_engine` and `scheduler_gauges` went with it.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, ClassVar

from relicllm.api import (
    ConfigurationError,
    RequestCancelledError,
    UnsupportedFeatureError,
)


def cancel_key(context: Any, request_id: int) -> str:
    """The id a client's `cancel` arrives under for a request.

    The runtime is handed its own request as `context`, and the request's id is what the HTTP layer
    knows that request by. Falling back to the fallback id keeps a runtime whose context was never
    published cancellable at all, rather than silently uncancellable.
    """
    return str(getattr(context, "request_id", None) or request_id)


def device_index(device: Any) -> int:
    """The card index a device names, or -1 for a host device.

    Accepts what a launcher option carries (``"cpu"``, ``"cuda"``, ``"cuda:2"``), an already-typed
    card index, and what torch reports (``torch.device`` and ``torch.device("cpu")``), because an
    adapter that reads one and not the other is a runtime that binds the wrong card on the one path
    nobody exercised.

    A `str` is read as a `str`. `getattr(x, "index")` used to be asked first, and `"cuda:3".index`
    is `str.index` -- a builtin method -- so `int()` of it raised on every string device, and an
    `int` device fell through to the suffix parse and answered **0**: a rank asked for card 3 bound
    card 0. Both were live on the v41 and torch routes, which pass a string and an int
    respectively; the branch below is keyed on the type rather than on attribute presence so that
    neither can happen again.
    """
    if device is None:
        return -1
    if isinstance(device, bool):  # a bool is an int, and never a card
        raise TypeError(f"{device!r} is not a device")
    if isinstance(device, int):
        return int(device)
    if isinstance(device, str):
        text = device
    else:
        index = getattr(device, "index", None)
        if isinstance(index, int):
            return int(index)
        text = str(device)
    if text.startswith("cpu"):
        return -1
    _, _, suffix = text.partition(":")
    try:
        return int(suffix)
    except ValueError:
        return 0


#: The variables a launcher narrows the visible card set with, per platform. Read together rather
#: than by the selected runtime's vendor, because who chose the vendor is not settled when a rank
#: works out its card, and a launcher that sets one of them means the same thing on either.
_VISIBLE_DEVICES = ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES")


def visible_card_count(env: Mapping[str, str] | None = None) -> int | None:
    """How many cards this process can see, when the launcher narrowed the list; ``None`` otherwise.

    ``None`` and ``0`` are different answers and are kept apart: a variable nobody set means the
    whole host, and a variable set to the empty string means no card at all -- which is how the
    suite runs a CPU-only test on a card-bearing host. Collapsing them would make "narrowed to one"
    and "narrowed to none" the same input to :func:`card_for_rank`.
    """
    source = os.environ if env is None else env
    for variable in _VISIBLE_DEVICES:
        value = source.get(variable)
        if value is not None:
            return len([item for item in value.split(",") if item.strip()])
    return None


def card_for_rank(
    device_ids: Sequence[int] = (),
    *,
    rank: int = 0,
    world: int = 1,
    visible: int | None = None,
) -> int:
    """Which card this rank runs on, as an index into the set the process can see.

    Two answers, and the difference between them is what ``--device-ids`` bought. **Named**: the
    list is the world's cards in rank order, so this rank's is ``device_ids[rank]`` -- one flag
    names every rank's card, and the launcher's ``CUDA_VISIBLE_DEVICES=$rank`` beside ``--device 0``
    has nothing left to do. **Unnamed**: the rule every runtime already had, kept because a launcher
    that still narrows visibility per rank is still correct -- a single process takes card 0, a
    sharded one takes its rank, and a visibility narrowed to one card *is* index 0 of what this
    process can see.

    The indices are into the visible set and not physical, which is what makes the second spelling
    reachable: ``--device-ids 0`` beside a per-rank ``CUDA_VISIBLE_DEVICES`` means what ``--device 0``
    means today, and ``--device-ids 2,3`` on an unnarrowed host names cards 2 and 3. A launcher that
    does both gets the intersection, which is the only reading that keeps the narrowed case working.
    """
    if device_ids:
        if rank >= len(device_ids):
            # Also refused in `EngineArgs.__post_init__` against the world, which is the check that
            # can be made before anything loads; this one is the same statement for a caller that
            # reached here with a rank the args never saw (an EP group's, say).
            raise ConfigurationError(
                f"device_ids names {len(device_ids)} card(s), so rank {rank} has none; "
                "name one per rank, in rank order"
            )
        return int(device_ids[rank])
    if world <= 1:
        return 0
    if visible is not None and visible <= 1:
        return 0
    return rank


class RankedWorker:
    """The worker-rank half of a Python runtime adapter: rebuild, announce, serve, unwind.

    One body for two runtimes, and the reason it is one is that the loop is the same program on
    both: refuse the two shapes that cannot work, load the checkpoint, announce readiness *after*
    the load's closing barrier, then take one message at a time off the broadcast until rank 0
    says to stop. ``v41`` and ``mimo`` each wrote that out, and each wrote it around its own four
    details rather than in terms of them -- so a change to any of the invariants below was a change
    in two places that had to agree.

    The four details, which are hooks rather than branches because they are facts about a runtime
    and not about being a worker rank:

    - **How a message arrives.** ``v41`` receives through its doorbell-plus-collective
      (:meth:`_recv_worker_message` defaults to the plain collective), which is what keeps an idle
      worker parked on the host instead of inside an NCCL poll kernel.
    - **What "this request is over" means.** A cancel and an abort unwind every rank together and
      are swallowed; a runtime with a second such exception declares it in :attr:`_WORKER_ABORTS`.
    - **What serving one message takes.** ``v41`` hands its payload runner the payload *and* the two
      request-scoped arguments it may be called with from the serial path; ``mimo``'s takes the
      payload alone. See :meth:`_run_worker_request`, whose base body is ``v41``'s arity because
      that arity is the union of the two call shapes.
    - **What has to happen before the group is torn down.** ``mimo`` closes with a barrier so rank 0
      cannot reap the ranks under a worker still finishing its last request;
      :meth:`_worker_drained` is that, and a no-op otherwise.

    Two guards in this body are worth stating because they are the reason it exists at all. The
    rank comes from the group and not from the arguments -- a supervised launch sets both, a
    ``torchrun`` one only sets the environment, and rank 0 must never enter this loop. And
    ``_ensure_loaded`` runs *here*, after the guards, because these runtimes join the group and load
    inside this method: "constructed" is not a state worth announcing, and announcing before the
    load's barrier would tell the parent a rank is up while it is still inside a collective.
    """

    #: What rank 0 broadcasts to end the loop. One spelling, because the payload that ends the loop
    #: is the same message on both runtimes and was two literals that only happened to agree.
    _WORKER_SHUTDOWN = "shutdown"

    #: The exceptions a worker rank swallows as "the loop unwound on every rank together".
    #:
    #: Not an error on a worker: the ranks agree per step, so a cancelled or stopped request reaches
    #: every rank at the same token. A runtime whose loop has a second way to unwind declares it
    #: here -- ``v41`` has ``_AbortGeneration`` beside the cancel -- and one that narrows this is a
    #: runtime whose other exceptions really are desynchronizations.
    _WORKER_ABORTS: ClassVar[tuple[type[BaseException], ...]] = (RequestCancelledError,)

    def run_worker(self, on_ready: Callable[[], None] | None = None) -> None:
        """Load, announce, then serve rank 0's requests until it says to stop."""
        self._ensure_open()
        self._init_distributed()
        if self._world <= 1:
            raise UnsupportedFeatureError(
                "run_worker serves rank 0's requests over a process group; a single-process run "
                "serves them through the HTTP server instead"
            )
        if self._rank == 0:
            raise UnsupportedFeatureError("run_worker must not be called on rank 0")
        self._ensure_loaded()
        if on_ready is not None:
            on_ready()
        while not self._closed:
            payload = self._recv_worker_message()
            if not isinstance(payload, Mapping) or payload.get("op") == self._WORKER_SHUTDOWN:
                break
            if payload.get("op") != "generate":
                continue
            try:
                self._run_worker_request(payload)
            except self._WORKER_ABORTS:
                # Every rank unwinds together, so a cancelled request is not a desynchronized group.
                pass
        self._worker_drained()

    def _recv_worker_message(self) -> Any:
        """The next message rank 0 sent, blocking until it arrives.

        The plain collective, which is what a runtime without a doorbell waits in. ``v41``
        overrides it: its bell lets the wait happen on the host, so an idle worker costs a sleeping
        ``recv`` rather than a device-side poll kernel holding a core at 100%.
        """
        import torch.distributed as dist

        box: list[Any] = [None]
        dist.broadcast_object_list(box, src=0)
        return box[0]

    def _run_worker_request(self, payload: Mapping[str, Any]) -> None:
        """Serve one request for the collective's sake and not for its answer.

        The base body is ``v41``'s call shape, and that is deliberate: ``_run_payload`` is the one
        method both the serial path and the worker loop reach, ``v41`` calls it with the surface the
        serial path passes, and ``mimo``'s takes the payload alone. A runtime in the second case
        overrides this rather than widening its payload runner for a caller that never passes them.
        """
        self._run_payload(payload, None, None)

    def _worker_drained(self) -> None:
        """What has to happen before the group is torn down, once the loop has ended.

        A no-op by default. ``mimo`` overrides it with the barrier that keeps rank 0 from reaping
        the group under a worker still finishing its last request.
        """


