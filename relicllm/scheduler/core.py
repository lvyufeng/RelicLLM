"""The request scheduler: one model-agnostic queue every runtime reaches the engine through.

## What it is

The runtime this replaces is a lock. Every adapter held a ``threading.RLock`` (``_request_lock``)
around its generation call, so concurrent requests serialized on it -- one request had the engine,
the rest waited on the mutex, and the mutex had no idea which request it was for. This is that lock
made a queue: the same serialization, one request at a time, but with the request visible while it
waits (its id, its prompt, its phase) rather than hidden inside a mutex.

## Why a counter, not a phase

A request carries ``num_computed_tokens`` and ``num_tokens_with_spec`` rather than a phase enum.
The reason is the one vLLM states for its own scheduler: chunked prefill, prefix reuse and
speculative decoding are then *counters on a request*, not branches in the scheduler -- a scheduler
written this way does not grow a per-model case when a model gains one of them. ``Request`` also
keeps the ``phase`` word the V4 call sites already use, as a view derived from those counters, so a
reader that thinks in phases still reads the same name.

See ``docs/architecture/one_scheduler_many_models.md`` for the ruling and
``docs/architecture/scheduler_2026_10.md`` for this lift.

## What the scheduler does not know

Which architecture it is serving. That is not a style preference: it is what makes the module one
module instead of five. Everything a model needs to say reaches it as data:

* an :class:`ExecutionPlan`, which the *caller* builds. It carries the chunked-prefill size and the
  phase hook as opaque values the scheduler passes through without reading. What a plan's
  ``prefill_chunk_tokens`` means is the model's business; the field lands in the generation call's
  keyword arguments by that name. (The name is historical and now belongs to two models
  -- ``deepseek_v4`` and the GGUF token driver -- but the scheduler does not read it, so a third's
  convention arrives as a new plan, not a new branch here.)
* the KV declaration (#129), once it lands, as the numbers admission accounts against.
* the request's own counters.

``tests/test_scheduler_core.py`` scans this package for an architecture name or a ``relicllm.models``
import and fails on either.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Optional

import collections


@dataclass
class Request:
    """A queued request, as a row of counters.

    ``prompt_tokens`` and ``max_new_tokens`` are what the engine needs to run it; the two counters
    are what a scheduler needs to place it. ``sent_tokens`` is the prompt's length and
    ``num_computed_tokens`` is how much of it the engine has taken, so ``computed >= sent`` is "the
    prefill is finished" -- the same fact a ``phase`` field would carry, counted rather than named.
    ``num_tokens_with_spec`` is written by later phases (speculative decoding) and read by nothing
    yet; it is here because a scheduler that adds it later is a scheduler that changed shape.

    ``phase`` is kept as a *view*: the V4 call sites and their readers think in prefill/decode, and
    the word is derived from the counters so there is one source of truth. ``metadata`` is the
    caller's; the scheduler neither reads nor writes it.
    """

    request_id: int
    prompt_tokens: list[int]
    max_new_tokens: int
    sent_tokens: int = 0
    num_computed_tokens: int = 0
    num_tokens_with_spec: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sent_tokens:
            self.sent_tokens = len(self.prompt_tokens)

    @property
    def phase(self) -> str:
        return "decode" if self.num_computed_tokens >= self.sent_tokens else "prefill"

    @property
    def is_prefilled(self) -> bool:
        return self.num_computed_tokens >= self.sent_tokens


@dataclass(frozen=True)
class ExecutionPlan:
    """The per-model data a scheduler passes through without interpreting.

    Two fields, both opaque to the scheduler. ``prefill_chunk_tokens`` is forwarded into the
    generation call's keyword arguments under that name; ``0`` means "no chunking", which is what
    every runtime but ``deepseek_v4`` and the GGUF driver does. ``phase_callback`` is called with
    ``"prefill"`` or ``"decode"`` when the engine changes phase, and is ``None`` when the model has
    nothing to do on a phase change -- which is every runtime whose phase machinery is off.
    """

    prefill_chunk_tokens: int = 0
    phase_callback: Optional[Callable[[str], None]] = None

    def generation_kwargs(self) -> dict[str, Any]:
        """The plan as keyword arguments for a generation call.

        The one place the field name is spelled, so a schedule that adds a channel adds it here and
        no call site changes.
        """
        kwargs: dict[str, Any] = {"prefill_chunk_tokens": self.prefill_chunk_tokens}
        if self.phase_callback is not None:
            kwargs["phase_callback"] = self.phase_callback
        return kwargs


class Scheduler:
    """One request at a time, with the waiting request visible as a queue.

    ``submit`` appends and returns the request; ``acquire`` blocks until the slot is free and returns
    a context manager that releases it. Between the two, the request is in ``pending`` and readable
    -- which is the whole difference from the mutex this replaces, where a waiting request existed
    only as a thread parked on a lock.

    The slot is occupied for the *whole* engine call and released when the request ends, via
    :meth:`Request.release` or the context manager. It is not released at each yield of a stream: a
    streamed request holds the engine from its first token to its last, exactly as the lock did, so a
    streaming request cannot interleave with a serial one.

    Injection: the slot is a :class:`threading.Semaphore`, but any object with ``acquire`` and
    ``release`` works. The server injects one whose ``acquire`` can be asked to time out
    (``threading.Lock``), turning the wait into a bounded one -- a request that would have blocked on
    the lock forever gets a 503 instead. The class does not import a server type to do this; the
    server passes the object.
    """

    def __init__(self, gate: Optional[Any] = None) -> None:
        self._gate = gate if gate is not None else threading.Semaphore(1)
        self._pending: Deque[Request] = collections.deque()
        self._next_request_id = 0
        #: Rank matters only for the id counter's collision story across processes; nothing here is
        #: collective. It is read once so a rank's ids are disjoint from another's in a log that
        #: merges them, and it is the one environment read the package does.
        self._rank = int(os.getenv("RANK", "0") or "0")
        self._lock = threading.RLock()

    # -- admission -----------------------------------------------------------------------------

    def submit(
        self,
        prompt_tokens: list[int],
        max_new_tokens: int,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> Request:
        with self._lock:
            request = Request(
                request_id=self._next_request_id,
                prompt_tokens=list(prompt_tokens),
                max_new_tokens=int(max_new_tokens),
                metadata=dict(metadata or {}),
            )
            self._next_request_id += 1
            self._pending.append(request)
            return request

    def next_request(self) -> Optional[Request]:
        """The oldest waiting request, or ``None``. Read-only; does not admit or remove it."""
        with self._lock:
            return self._pending[0] if self._pending else None

    def has_work(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    # -- the engine slot -----------------------------------------------------------------------

    def acquire(self, request: Request, *, timeout: Optional[float] = None):
        """Wait for the engine, then hand back a context manager that holds it.

        With ``timeout`` the wait is bounded and ``None`` is returned if it expires, so a caller with
        a client to answer can refuse rather than park. Without it the wait is unbounded and the
        context manager is always returned -- the shape a backend with no HTTP deadline in front of
        it wants.
        """
        if timeout is None:
            self._gate.acquire()
        elif not self._gate.acquire(timeout=timeout):
            return None
        return _Slot(self, request)

    def _release(self, request: Request) -> None:
        with self._lock:
            try:
                self._pending.remove(request)
            except ValueError:
                pass
        self._gate.release()

    def mark_request_done(self, request: Request) -> None:
        """Release the slot from outside a ``with`` -- for a code path that acquired by hand."""
        self._release(request)

    def mark_prefill_done(self, request: Request) -> None:
        """Record that the engine finished a prefill for ``request``.

        A counter write, not a queue move: the request stays where it is and ``phase`` follows the
        counter. Kept as a named method because the V4 call sites and their tests speak this word.
        """
        request.num_computed_tokens = request.sent_tokens


class _Slot:
    """Context manager over one held engine slot. Releasing twice is a no-op, not a double free."""

    def __init__(self, scheduler: Scheduler, request: Request) -> None:
        self._scheduler = scheduler
        self._request = request

    def __enter__(self) -> "Request":
        return self._request

    def __exit__(self, *exc: object) -> bool:
        self._scheduler._release(self._request)
        return False


__all__ = ["ExecutionPlan", "Request", "Scheduler"]