"""Reusable capture arenas: keep a request's caches and recordings across compatible calls.

A cold request pays for the whole of its graph setup: the cache arena is allocated, the trunk
verifier, the draft head and the native tail are recorded, and at teardown all of it is destroyed.
That setup is host-visible latency on every request. This module owns the *lifetime* of that bundle
so a later request with the same geometry can hand the same arenas and recordings back.

What it deliberately does not own is how a recording is made. The parent builds the caches and the
graphs exactly as it does today (``model.make_cache`` and the ``VerifyGraph`` / ``DraftGraph`` /
``StartupDraftGraph`` constructors); a bundle is *adopted* once it exists. This keeps the two
concerns apart: recording geometry stays with the classes that know it, and only ownership,
compatibility, reset and binding suspension live here.

Per-request reset
-----------------
A reused cache must start in the state a fresh one starts in, without writing the multi-GiB KV and
indexer arenas. That is sound because of two contracts the decode path already relies on:

* KV / raw-index rows are addressed by position. A round writes the rows its positions name before
  anything reads them, and the sparse selection masks every block at or past ``blocks_visible``
  (``attention.py``), so rows past this request's committed prefix cannot be selected no matter
  what bytes they hold. A cold cache holds zeros there; a reused cache holds the previous request's
  bytes; neither is read.
* The rolling state is *not* addressed by position -- the 36 GatedDeltaNet layers' conv window and
  recurrence, and each PLE layer's short-conv carry -- so those must be put back to the exact
  values a fresh cache holds: zeros, with ``initialized`` cleared. That is 14 MiB a rank.

``host_position`` and ``length`` are reset with the metadata, and the completed-prefix bookkeeping of
the opt-in indexer pool is cleared to zero rather than invalidating the pool object. The graph
captured writes into that pool's buffers by address, so the object and its allocations must survive;
its *counts* must not claim groups this request has not written.

Binding suspension
------------------
A recording is not inert between requests. ``VerifyGraph`` installs, on the cache and on every PLE
layer's host table, a staged PLE history and its own pinned row buffers (``ple_history_at`` and
``table._rows_buffer``). While those are installed an eager prefill forward would hash the previous
request's n-gram window instead of the ids it was given, and ``DraftGraph`` likewise leaves the
head's expert backend on its recordable ``rowwise`` path, which a multi-row prefill call must not
use. So a reused bundle is *suspended* between requests: every binding its owners installed is
reverted to the value it held before the recording, and `resume` installs them again once prefill
finishes. The owners' own ``release``/``retire_bindings`` state is never touched, and a binding this
module does not know about is refused rather than silently left installed.

Buffer identity
---------------
Suspension is only meaningful if the tensors a recording reads keep their addresses. Nothing here
reallocates them, but the prefill between two requests is ordinary eager code, so `resume` compares
a captured ``data_ptr``/shape/dtype fingerprint for every graph-touched buffer and raises
:class:`SessionInvalidated` on any difference. The caller's correct response is to fall back to a
cold request; returning a bundle whose recording points at freed memory would be a wrong answer, not
a slow one.

CPU testability
---------------
The state machine is pure host work. Tests use the real cache classes with synthetic geometry and
fake recordings that implement the same small binding protocol, so ownership, reset values, suspend
and resume are checked without a card.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field

import torch

__all__ = [
    "GraphLease",
    "GraphSessionError",
    "GraphSessionKey",
    "GraphSessionPool",
    "SessionInvalidated",
    "flags_digest",
]

#: Binding names a recording may install and this module knows how to displace. Anything else means
#: a recording class grew a new cache binding; the session refuses rather than guessing.
_ALLOWED_BINDINGS = frozenset({"kv_width", "rope_width", "ple_history_at", "_rows_buffer", "rowwise"})

#: Owner attributes that are graph-owned buffers, fingerprinted at adoption and re-checked on resume.
_OWNER_BUFFERS = (
    "history", "ids", "position", "outputs", "streams", "answer", "_gathered",
    "select", "out", "token", "stack",
)

#: Cumulative counters a retained recording keeps across every request that reuses it. They are
#: measured by taking a mark, never by zeroing them: the total is the evidence of reuse.
_COUNTER_ATTRS = ("replays", "stage_s", "device_s", "boundary_retire_s")

_MISSING = object()


class GraphSessionError(RuntimeError):
    """A bundle cannot be reused as asked, or the caller used the lease out of order."""


class SessionInvalidated(GraphSessionError):
    """A retained bundle no longer matches the buffers its recordings were made against."""


def flags_digest(environ: dict | None = None) -> str:
    """A stable digest of every ``POCKETLLM_*`` switch, so a flag change cannot reuse a bundle.

    The recordings bake dispatch decisions (fused or torch, packed or serial collectives) into
    their graph. A bundle made under one set of switches and replayed under another would silently
    run the old decisions, so the key carries the switches rather than assuming they are constant.
    """
    source = os.environ if environ is None else environ
    pairs = sorted((key, str(value)) for key, value in source.items() if key.startswith("POCKETLLM_"))
    digest = hashlib.sha256()
    for key, value in pairs:
        digest.update(f"{key}={value}\n".encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class GraphSessionKey:
    """Everything a bundle's recordings depend on that can differ between two requests.

    ``width`` is the arena width the graphs were recorded over; ``rows`` and ``draft_tokens`` are the
    verifier and chain geometries. A request whose width, depth or switches differ must not replay
    these recordings -- positions are device tensors, but the arena, the row span and the native tail
    are not. ``model_token`` is the caller's identity for the loaded model (checkpoint, or a hash);
    it defaults to ``id(model)``, which is only meaningful inside one process.
    """

    model_token: str
    width: int
    draft_tokens: int
    rows: int
    device: str
    flags: str
    arena_tokens: int

    @classmethod
    def build(
        cls,
        *,
        model,
        ctx,
        width: int,
        draft_tokens: int,
        rows: int,
        arena_tokens: int | None = None,
        model_token: str | None = None,
        environ: dict | None = None,
    ) -> "GraphSessionKey":
        """Derive the key for a launch from the live model and context (no GPU work, no read)."""
        if model_token is None:
            named = getattr(model, "checkpoint_path", None) or getattr(model, "model_id", None)
            model_token = f"named:{named}" if named else f"id:{id(model)}"
        device = getattr(ctx, "device", None)
        return cls(
            model_token=str(model_token),
            width=int(width),
            draft_tokens=int(draft_tokens),
            rows=int(rows),
            device=str(device),
            flags=flags_digest(environ),
            arena_tokens=int(width if arena_tokens is None else arena_tokens),
        )

    def mismatch(self, other: "GraphSessionKey") -> str:
        """The first field that differs, or ``""`` when the two keys are interchangeable."""
        for name in ("model_token", "width", "draft_tokens", "rows", "device", "flags",
                     "arena_tokens"):
            mine, theirs = getattr(self, name), getattr(other, name)
            if mine != theirs:
                return f"{name}: {mine!r} != {theirs!r}"
        return ""

    def matches(self, other: "GraphSessionKey") -> bool:
        return not self.mismatch(other)


# ---------------------------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------------------------


def _fingerprint(tensor) -> tuple | None:
    """Identity of one graph-touched tensor, without reading any device value."""
    if not isinstance(tensor, torch.Tensor):
        return None
    return (tensor.data_ptr(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype,
            str(tensor.device))


def _cache_fingerprint(cache, owners, head_cache=None) -> dict:
    """Every buffer a retained recording reads by address, keyed by a readable name."""
    found: dict = {}

    def add(name: str, value) -> None:
        stamp = _fingerprint(value)
        if stamp is not None:
            found[name] = stamp

    add("cache.cos", getattr(cache, "cos", None))
    add("cache.sin", getattr(cache, "sin", None))
    for index, layer in enumerate(getattr(cache, "layers", []) or []):
        if layer is None:
            continue
        for attr in ("k", "v", "index_k"):
            add(f"layer{index}.{attr}", getattr(layer, attr, None))
        pool = getattr(layer, "indexer_pool", None)
        if pool is not None:
            add(f"layer{index}.pool.keys", getattr(pool, "keys", None))
            add(f"layer{index}.pool.completed", getattr(pool, "completed_tensor", None))
        for attr in ("conv_state", "recurrent_state"):
            add(f"layer{index}.{attr}", getattr(layer, attr, None))
    for index, state in (getattr(cache, "ple_states", None) or {}).items():
        add(f"ple{index}.conv_state", getattr(state, "conv_state", None))
    for order, owner in enumerate(owners):
        if owner is None:
            continue
        for attr in _OWNER_BUFFERS:
            add(f"owner{order}.{attr}", getattr(owner, attr, None))
        for index, (table, _old) in enumerate(
                getattr(owner, "_saved_rows_buffers", []) or []):
            add(f"owner{order}.rows{index}._rows_buffer", getattr(table, "_rows_buffer", None))
    if head_cache is not None:
        found.update({"head." + name: stamp for name, stamp in
                      _cache_fingerprint(head_cache, []).items()})
        for index, layer in enumerate(getattr(head_cache, "layers", []) or []):
            if layer is not None:
                add(f"head.layer{index}.cos", getattr(layer, "cos", None))
                add(f"head.layer{index}.sin", getattr(layer, "sin", None))
    return found


def _binding_plan(owners) -> list:
    """Work out how to displace and re-install each recording's cache bindings.

    Two implementations are accepted, because the recording classes own this knowledge:

    * **Owner protocol** (preferred). A recording may expose ``suspend_bindings()`` and
      ``resume_bindings()``. It captured the values it installed when it installed them, so the
      session only has to call those two methods; it never needs to know which attributes are
      involved. This is the intended contract for ``VerifyGraph``, ``DraftGraph`` and the one-row
      tail graph.
    * **Recorded attributes** (fallback). Without the protocol, the recording's ``_saved_attrs``
      list of ``(target, name, previous)`` is read and the currently installed value is taken off
      the target, which is enough to revert and re-install. This keeps the session usable before the
      protocol lands, at the cost of the session knowing the attribute names.

    Either way the list is kept in *construction* order; suspension walks it in reverse and resume
    in forward order (see :class:`_Bundle`). The names are still validated in the fallback case,
    because an attribute this module does not recognise would otherwise be silently left installed
    across a prefill.
    """
    plan = []
    for owner in owners:
        if owner is None:
            continue
        protocol = callable(getattr(owner, "suspend_bindings", None)) and callable(
            getattr(owner, "resume_bindings", None)
        )
        saved = list(getattr(owner, "_saved_attrs", []) or [])
        # A recording swaps each PLE host table's row scratch buffer through a second list, so an
        # eager prefill would otherwise hash into the recording's own pinned rows.
        saved.extend((table, "_rows_buffer", old)
                     for table, old in getattr(owner, "_saved_rows_buffers", []) or [])
        installed = []
        for target, name, _previous in saved:
            if name not in _ALLOWED_BINDINGS:
                raise SessionInvalidated(
                    f"{type(owner).__name__} installs an unknown cache binding {name!r}; this "
                    "session cannot safely suspend it -- release the bundle and rebuild cold"
                )
            namespace = getattr(target, "__dict__", {})
            installed.append((target, name, namespace[name] if name in namespace else _MISSING))
        plan.append({"owner": owner, "protocol": protocol, "installed": installed, "saved": saved})
    return plan


def _is_absent(value) -> bool:
    """Whether a recorded displaced value means "this attribute did not exist".

    Each recording module carries its own ``_MISSING = object()`` sentinel, so identity against this
    module's own sentinel would silently fail and install that sentinel as a cache attribute. The
    shape is unambiguous instead: a bare ``object()`` instance is not a value any binding holds.
    """
    return type(value) is object


def _restore(entries) -> None:
    """Put each ``(target, name, previous)`` back, deleting attributes that were absent before."""
    for target, name, previous in entries:
        if _is_absent(previous):
            if name in getattr(target, "__dict__", {}):
                delattr(target, name)
        else:
            setattr(target, name, previous)


def _call_protocol(owner, method: str) -> None:
    """Run one binding-protocol method, reporting a retired recording as a session invalidate."""
    try:
        getattr(owner, method)()
    except RuntimeError as error:
        raise SessionInvalidated(
            f"{type(owner).__name__}.{method}() refused: {error}; rebuild this bundle cold"
        ) from error


@dataclass
class _Bundle:
    """One retained acquisition: the caches, every recording, and how to displace their bindings.

    ``owners`` is in *construction* order, which is the order the recordings installed their
    bindings. Displacement therefore runs in reverse -- the recording built last is the one holding
    the innermost binding -- and re-installation runs forward, putting them back in the same order
    they went on. The one-row tail graph is built first, then the wide verifier, then the draft head,
    so suspension is ``draft -> verify -> tail`` and resume is ``tail -> verify -> draft``.
    """

    key: GraphSessionKey
    cache: object
    head_cache: object
    owners: list
    names: list
    plan: list
    fingerprint: dict
    suspended: bool = False

    def all_owners(self):
        return [(name, owner) for name, owner in zip(self.names, self.owners) if owner is not None]

    def suspend(self) -> None:
        """Revert every installed binding so ordinary eager requests may run on these caches."""
        if self.suspended:
            return
        for entry in reversed(self.plan):
            if entry["protocol"]:
                _call_protocol(entry["owner"], "suspend_bindings")
            else:
                _restore(reversed(entry["saved"]))
        self.suspended = True

    def resume(self) -> None:
        """Re-install the bindings a replay needs, proving the buffers did not move.

        The identity check runs *after* the bindings are back, because a suspended binding is
        deliberately holding the prefill value rather than the recording's (a PLE table's row
        buffer is the prefill scratch while suspended). What the check can still catch is the set
        of graph-touched buffers no binding owns -- the RoPE table, the KV/indexer arenas, the
        indexer pool, the rolling state and the recording's own device buffers. Those are the ones
        an eager prefill of the next request might legitimately reallocate.
        """
        if not self.suspended:
            return
        for entry in self.plan:
            if entry["protocol"]:
                _call_protocol(entry["owner"], "resume_bindings")
                continue
            for target, name, value in entry["installed"]:
                if _is_absent(value):
                    if name in getattr(target, "__dict__", {}):
                        delattr(target, name)
                else:
                    setattr(target, name, value)
        current = _cache_fingerprint(self.cache, self.owners, self.head_cache)
        for name, stamped in self.fingerprint.items():
            now = current.get(name)
            if now != stamped:
                raise SessionInvalidated(
                    f"{name} changed identity since the recordings were made "
                    f"({stamped} -> {now}); the recording cannot be replayed over it"
                )
        self.suspended = False

    def release(self) -> None:
        """Drop the recordings and their buffers. Idempotent; used when a bundle is discarded."""
        try:
            self.suspend()
        except (SessionInvalidated, RuntimeError):
            # An owner already retired its bindings (the native handoff does), so there is nothing
            # left to displace; dropping the recordings is all that remains.
            pass
        for owner in reversed(self.owners):
            if owner is None:
                continue
            release = getattr(owner, "release", None)
            if callable(release):
                release()


# ---------------------------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------------------------


def _reset_cache(cache, *, ngram: int, eos_token_id: int) -> None:
    """Put a cache back to the state ``model.make_cache`` would have produced, cheaply.

    Rolling state is zeroed in place (its buffers belong to a recording); position-addressed arenas
    are left exactly as they are, because every row this request reads is written first and the
    sparse selection masks the rest. Pool counts are cleared while the pool object, its buffers and
    its validity flag are kept: the recording writes that pool by address.
    """
    if cache is None:
        return None
    cache.length = 0
    for layer in getattr(cache, "layers", []) or []:
        if layer is None:
            continue
        if hasattr(layer, "state_log"):
            layer.state_log = None
        if hasattr(layer, "initialized"):
            layer.initialized = False
        for attr in ("conv_state", "recurrent_state"):
            tensor = getattr(layer, attr, None)
            if isinstance(tensor, torch.Tensor):
                tensor.zero_()
        if hasattr(layer, "length"):
            layer.length = 0
        if hasattr(layer, "host_position"):
            layer.host_position = 0
        pool = getattr(layer, "indexer_pool", None)
        if pool is not None:
            # Counts, not validity: a fresh request has written no groups yet, but the buffers the
            # recording wrote through must stay the ones it holds.
            pool.completed = 0
            completed = getattr(pool, "completed_tensor", None)
            if isinstance(completed, torch.Tensor):
                completed.zero_()
    for state in (getattr(cache, "ple_states", None) or {}).values():
        if hasattr(state, "state_log"):
            state.state_log = None
        carry = getattr(state, "conv_state", None)
        if isinstance(carry, torch.Tensor):
            carry.zero_()
    history = getattr(cache, "ple_token_history", None)
    if isinstance(history, torch.Tensor):
        history.fill_(int(eos_token_id))
    elif history is None:
        cache.ple_token_history = torch.full(
            (1, int(ngram) - 1), int(eos_token_id), dtype=torch.long, device="cpu"
        )
    # The RoPE table and the fixed widths stay: they are position-derived over the whole arena and
    # are what a replay indexes, so rebuilding them would move the addresses the recording reads.
    return None


def _model_ple_spec(model) -> tuple[int, int]:
    config = getattr(model, "config", None)
    ngram = int(getattr(config, "ngram_size", 0) or 0)
    eos = getattr(config, "primary_eos_token_id", None)
    if eos is None:
        eos = getattr(config, "eos_token_id", 0)
    if isinstance(eos, (list, tuple)):
        eos = eos[0] if eos else 0
    if ngram < 1:
        raise GraphSessionError("reset needs the model config's ngram_size")
    return ngram, int(eos)


# ---------------------------------------------------------------------------------------------
# Lease and pool
# ---------------------------------------------------------------------------------------------


class GraphLease:
    """One request's claim on a bundle: either freshly adopted, or adopted and about to be reused.

    Order for a cold request::

        lease = pool.checkout(key)            # lease.reused is False
        cache = model.make_cache(...)         # the caller allocates and prefills as today
        verify, draft, tail, startup = ...    # capture as today
        lease.adopt(cache=..., verify=..., ...)
        ... decode ...
        lease.finish(boundary_retired=...)    # returns the bundle to the pool, or drops it

    Order for a warm request::

        lease = pool.checkout(key)            # lease.reused is True; bindings are suspended
        lease.begin_request()                 # reset rolling state; prefill may run
        ... prefill on lease.cache / lease.head_cache ...
        lease.resume()                        # re-install bindings, prove buffers did not move
        ... decode ...
        lease.finish(...)
    """

    def __init__(self, pool: "GraphSessionPool", bundle: _Bundle | None, reused: bool, reason: str):
        self._pool = pool
        self._bundle = bundle
        self.reused = bool(reused)
        self.reason = reason
        self.state = "ready"
        self._adopted = bundle is not None
        self._finished = False

    # -- what a request needs ---------------------------------------------------------------

    @property
    def cache(self):
        return None if self._bundle is None else self._bundle.cache

    @property
    def head_cache(self):
        return None if self._bundle is None else self._bundle.head_cache

    def owner(self, name: str):
        """A retained recording by the name it was adopted under (``verify``, ``draft``, ...)."""
        if self._bundle is None:
            raise GraphSessionError("this lease holds no bundle")
        try:
            return self._bundle.owners[self._bundle.names.index(name)]
        except ValueError:
            raise GraphSessionError(f"the bundle has no recording named {name!r}") from None

    def adopt(self, *, cache, head_cache=None, owners: dict | None = None, **named) -> None:
        """Take ownership of a freshly built cache and its recordings.

        ``named`` maps a caller-chosen name (``verify``, ``draft``, ``tail``, ``startup``) to the
        recording, in **construction order**; the order is preserved because teardown and binding
        suspension both depend on the order recordings were installed (reverse for suspension,
        forward for resume). A ``None`` recording is still recorded as an owned slot so a partial
        configuration (no tail graph, no startup graph) is expressible.

        The startup recording needs no binding work: it installs no cache attributes, and its seed
        is per-request input (``first_draft`` and the eager chain's stack) that the caller supplies
        through ``StartupDraftGraph.replay(start=..., first_draft=..., chain_stack=...)``. The
        session exposes the object through ``owner("startup")`` and deliberately does not own the
        seed.
        """
        if self._bundle is not None:
            raise GraphSessionError("this lease already holds a bundle")
        if cache is None:
            raise GraphSessionError("adopt needs the cache the recordings were built over")
        merged = {}
        if owners:
            merged.update(owners)
        merged.update(named)
        names = list(merged)
        if not names:
            raise GraphSessionError("adopt needs at least one recording (or an explicit empty map)")
        owned = [merged[name] for name in names]
        plan = _binding_plan(owned)
        self._bundle = _Bundle(
            key=self._pool.key,
            cache=cache,
            head_cache=head_cache,
            owners=owned,
            names=names,
            plan=plan,
            fingerprint=_cache_fingerprint(cache, owned, head_cache),
        )
        self._adopted = True
        self.state = "active"
        self._pool.stats["adopted"] += 1

    def counter_mark(self) -> dict:
        """Snapshot the owners' cumulative counters so a request can measure its own share.

        ``VerifyGraph`` and ``DraftGraph`` accumulate ``replays``/``stage_s``/``device_s`` over their
        whole life, and a retained bundle outlives one request. Those counters must not be zeroed --
        the totals are the evidence that a recording was reused rather than recaptured -- so a
        request takes a mark at the start and subtracts it at the end.
        """
        marks = {}
        if self._bundle is not None:
            for name, owner in self._bundle.all_owners():
                for attr in _COUNTER_ATTRS:
                    value = getattr(owner, attr, None)
                    if isinstance(value, (int, float)):
                        marks[f"{name}.{attr}"] = value
        return marks

    def counter_delta(self, mark: dict) -> dict:
        """This request's share of the owners' cumulative counters, given a mark before it ran."""
        delta = {}
        if self._bundle is None:
            return delta
        for name, owner in self._bundle.all_owners():
            for attr in _COUNTER_ATTRS:
                key = f"{name}.{attr}"
                if key not in mark:
                    continue
                value = getattr(owner, attr, None)
                if isinstance(value, (int, float)):
                    delta[key] = value - mark[key]
        return delta

    def begin_request(self) -> None:
        """Reset a reused bundle to a fresh request's state. Prefill runs after this.

        This resets the rolling state (the 36 GatedDeltaNet conv/recurrent buffers and the PLE
        short-conv carry), the position metadata and the PLE n-gram window, and nothing else. The
        multi-GiB KV/indexer arenas are left untouched -- a round writes the rows its positions name
        before anything reads them, and the sparse selection masks every block at or past the
        committed prefix, so their previous contents cannot be reached. Cumulative owner counters
        are also left alone; use `counter_mark` / `counter_delta`.

        The startup recording's seed is per-request input and is not reset here: the caller supplies
        it by replaying that graph with its own `first_draft` and chain stack.
        """
        if self._bundle is None:
            raise GraphSessionError("nothing to reset: the caller must adopt a cold bundle first")
        if self.state not in {"ready", "active"}:
            raise GraphSessionError(f"begin_request in state {self.state!r}")
        ngram, eos = self._pool.ple_spec
        _reset_cache(self._bundle.cache, ngram=ngram, eos_token_id=eos)
        # The draft head's cache is a second arena with its own QSA lengths and indexer pool. The
        # prefill seeds the head eagerly, so it must start from the same position metadata a fresh
        # request's head cache has -- a leftover length would place that seeding at the wrong offset.
        _reset_cache(self._bundle.head_cache, ngram=ngram, eos_token_id=eos)
        self._pin_rope_width()
        self.state = "prefilling"
        self._pool.stats["reset"] += 1

    def _pin_rope_width(self) -> None:
        """Keep the retained RoPE table at its recorded width across the prefill.

        Without `cache.rope_width`, `_rope_cache` grows the table to ``ceil(total_len / 256) * 256``
        and *reallocates* when that exceeds the retained span. The retained span is the arena width,
        which is wider than the prompt by only the output budget, so a small budget could round past
        it. Setting the recorded width makes `_rope_cache` take its fixed branch and slice the
        existing table instead. Only `rope_width` is pinned: `kv_width` would switch attention onto
        its tensor-position KV path, which eager prefill does not use.
        """
        cache = self._bundle.cache
        cos = getattr(cache, "cos", None)
        if isinstance(cos, torch.Tensor) and cos.shape[1] >= self._bundle.key.width:
            cache.rope_width = self._bundle.key.width

    def resume(self) -> None:
        """Re-install the recordings' bindings after prefill. Raises if any buffer moved."""
        if self._bundle is None:
            raise GraphSessionError("nothing to resume")
        if self.state not in {"prefilling", "active"}:
            raise GraphSessionError(f"resume in state {self.state!r}")
        if self._bundle.suspended:
            self._bundle.resume()
        self.state = "active"
        self._pool.stats["resumed"] += 1

    # -- end of request ---------------------------------------------------------------------

    def finish(self, *, boundary_retired: bool = False, successful: bool = True) -> None:
        """Close the request: keep the bundle for reuse, or drop it when it cannot be replayed.

        A request that took its native-boundary tail destroyed the main recordings (or left them
        metadata-retired, which cannot replay), so the bundle is discarded and the next compatible
        request records again. An unsuccessful request is discarded for the same reason: a cancelled
        decode may have left the caches at a position no reset accounts for.
        """
        if self._finished:
            return
        self._finished = True
        if self._pool._active_lease is self:
            self._pool._active_lease = None
        bundle = self._bundle
        self._pool.stats["finishes"] += 1
        if bundle is None:
            self.state = "finished"
            return
        usable = (bool(successful) and not self._pool._closed and not boundary_retired
                  and self._pool._healthy(bundle))
        if boundary_retired:
            self._pool.stats["boundary_invalidations"] += 1
        if usable:
            try:
                bundle.suspend()
            except (SessionInvalidated, RuntimeError):
                self._pool.stats["address_invalidations"] += 1
                usable = False
        if usable:
            self._pool._return(bundle)
            self.state = "suspended"
        else:
            try:
                bundle.release()
            finally:
                self._pool.stats["released"] += 1
                self.state = "discarded"

    def abandon(self) -> None:
        """Give the bundle back without claiming it may be reused (cancellation, failure, error)."""
        self.finish(boundary_retired=False, successful=False)

    def __enter__(self) -> "GraphLease":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._finished:
            self.finish(successful=exc_type is None)


class GraphSessionPool:
    """The owner. Holds at most ``capacity`` compatible bundles and hands out leases."""

    def __init__(self, *, model, capacity: int = 1):
        if capacity < 1:
            raise ValueError("a session pool holds at least one bundle")
        self.model = model
        self.capacity = int(capacity)
        self.ple_spec = _model_ple_spec(model)
        self._bundles: list[_Bundle] = []
        self._closed = False
        self._active_lease: GraphLease | None = None
        self.key: GraphSessionKey | None = None
        self.stats = {
            "checkouts": 0, "hits": 0, "misses": 0, "discards": 0, "adopted": 0,
            "reset": 0, "resumed": 0, "finishes": 0, "boundary_invalidations": 0,
            "address_invalidations": 0, "released": 0, "closes": 0,
        }

    # -- lookup -----------------------------------------------------------------------------

    def checkout(self, key: GraphSessionKey) -> GraphLease:
        """Claim a bundle for one request.

        A compatible retained bundle is handed out suspended and must be reset with
        ``lease.begin_request()`` before the caller prefills. Anything retained under a different
        key is discarded first: replaying a recording over a different arena or geometry is exactly
        the case this refuses to guess about.
        """
        if self._closed:
            raise GraphSessionError("the session pool is closed")
        if not isinstance(key, GraphSessionKey):
            raise TypeError("checkout needs a GraphSessionKey")
        if self._active_lease is not None and not self._active_lease._finished:
            raise GraphSessionError("the graph session already has an active request")
        self.stats["checkouts"] += 1
        self.key = key
        reason = ""
        for index, bundle in enumerate(self._bundles):
            difference = bundle.key.mismatch(key)
            if not difference:
                if self._healthy(bundle):
                    self._bundles.pop(index)
                    self.stats["hits"] += 1
                    self._active_lease = GraphLease(self, bundle, reused=True, reason="compatible")
                    return self._active_lease
                # A recording destroyed itself behind this session's back (the native handoff does
                # exactly that), so the bundle can no longer replay and is dropped, not handed out.
                self._bundles.pop(index)
                bundle.release()
                self.stats["released"] += 1
                self.stats["discards"] += 1
                break
            reason = difference
        if self._bundles:
            self._discard_all(reason="incompatible: " + (reason or "key differs"))
        self.stats["misses"] += 1
        self._active_lease = GraphLease(self, None, reused=False, reason="cold")
        return self._active_lease

    # -- ownership --------------------------------------------------------------------------

    @staticmethod
    def _healthy(bundle: _Bundle) -> bool:
        """Whether every recording a bundle retains is still replayable."""
        for _name, owner in bundle.all_owners():
            if getattr(owner, "released", False):
                return False
            if getattr(owner, "_bindings_retired", False):
                # Its `_saved_attrs` were cleared when it retired, so nothing can re-install the
                # bindings it needs; replay would raise. Drop the bundle instead of handing it out.
                return False
        return True

    def _return(self, bundle: _Bundle) -> None:
        """Put a suspended bundle back, evicting beyond capacity and dropping stale entries."""
        self._bundles.append(bundle)
        while len(self._bundles) > self.capacity:
            stale = self._bundles.pop(0)
            stale.release()
            self.stats["discards"] += 1
            self.stats["released"] += 1

    def _discard_all(self, *, reason: str) -> None:
        for bundle in self._bundles:
            bundle.release()
            self.stats["released"] += 1
        self._bundles = []
        self.stats["discards"] += 1
        self.stats["last_discard_reason"] = reason

    # -- lifetime ---------------------------------------------------------------------------

    def abandon_active(self) -> None:
        """Discard an interrupted request, including failures before capture/setup starts."""
        if self._active_lease is not None:
            self._active_lease.abandon()

    def close(self) -> None:
        """Release everything retained. Idempotent."""
        if self._closed:
            return
        self._closed = True
        # Closing refuses new requests immediately, but only an active request may release the
        # recordings it can still be replaying. Its finish() sees _closed and discards the bundle.
        count = len(self._bundles)
        for bundle in self._bundles:
            bundle.release()
            self.stats["released"] += 1
        self._bundles = []
        self.stats["closed_bundles"] = count
        self.stats["closes"] += 1

    def __enter__(self) -> "GraphSessionPool":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def closed(self) -> bool:
        return self._closed

    def summary(self) -> dict:
        """Counters plus what is currently retained, for a log line."""
        out = dict(self.stats)
        out["retained"] = len(self._bundles)
        out["closed"] = self._closed
        return out