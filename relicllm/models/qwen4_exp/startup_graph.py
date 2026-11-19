"""A captured startup draft chain: the `K - 1` head calls that follow prefill.

The first speculative round is special. Every later round gets its drafts from `DraftGraph`, which
replays one recording over the trunk's answers for the round just verified. The first round has no
round behind it: its first draft is the prefill's own last head answer, and the `K - 1` drafts after
it are an eager chain -- one head forward and one host argmax read per link, issued from Python
after the request has already delivered its first token. At 256K this contributes to the first
inter-token latency. The generation work must remain charged to decode; only its recording and
warmup are setup.

`StartupDraftGraph` records exactly that chain once, during setup, so the first round can replay it
like any other. It changes no arithmetic: the recording makes the same `K - 1` one-query
`head.forward` calls at the same positions, runs the same `lm_head` projection and greedy pick
(`verify_graph`'s `_greedy_rows` over `_gather_buffer`), and hands the next link the same recycled
stack. What it removes is the host round trip per link -- the argmax no longer has to reach the CPU
before the next forward can be issued.

Four properties hold it together:

* **The seed is the prefill's answer, not a recomputation.** `out[0]` is the caller's `first_draft`,
  copied from a stable device buffer; the recording never re-runs the head for it.
* **Positions are a device tensor.** Link `i` (1-based) runs at `start + i - 1`, which is exactly
  where the eager chain ran it: `start` is the position of the token `first_draft` succeeds.
* **It writes only future, speculative head state.** The recording writes raw KV rows and indexer
  keys at positions at or past `start`, which the trunk has not written and which the first round's
  `DraftGraph` catch-up overwrites (with the true `(h_j, t_{j+1})` pairing) before anything reads
  them. The head has neither recurrence nor a PLE term, so no rolling state needs saving; the only
  cache metadata the recording touches is `layer.length`, which it puts back afterwards.
* **The head cache was already prepared by `DraftGraph`.** Its absolute RoPE table, its fixed KV
  width and its device-indexed expert path are what make the head capturable at all. This module
  neither builds nor converts any of them -- it fails loudly when they are missing rather than
  silently recording a host-read path or quantizing the experts a second time.

The recording is held for the whole request, not just the first round. Its host launch is cheap,
but the actual head work is nonzero and must be measured. Release it with `release()` at request
cleanup; it installs no cache bindings of its own, so there is nothing to retire before then.

GPU parity -- replayed drafts against the eager chain's, on the real checkpoint -- is a separate
measurement. The contracts this module and its tests pin are host control flow, positions and
teardown.
"""

from __future__ import annotations

import torch

from relicllm.models.qwen4_exp.verify_graph import _gather_buffer, _greedy_rows

__all__ = ["StartupDraftGraph"]

_MISSING = object()


def _capture_ready(device) -> bool:
    """Whether a CUDA graph can be warmed up and recorded on `device`.

    A seam on purpose: CPU tests substitute it (with `_record`) so the chain's positions, seed and
    teardown can be checked without a card, while production refuses to pretend a graph exists.
    """
    return device.type == "cuda" and torch.cuda.is_available()


def _empty_cache(device) -> None:
    """Return the eager allocator's cached blocks after a recording is dropped or installed.

    Also a seam, so the tests can prove the cleanup happens exactly once, at `release`, and never
    on the decode path.
    """
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _record(device, run):
    """Warm `run` up outside a recording, then record it as one CUDA graph on a private pool.

    The warm-up allocates the tensors the recording keeps (the gather landing buffer) and lets lazy
    kernels compile, so the recording itself holds only the chain's own work. It runs eagerly,
    before the measured interval, and its results are discarded: `replay` re-executes the recorded
    work, so the recording is never a host shortcut around the decode loop.
    """
    torch.cuda.synchronize(device)
    side = torch.cuda.Stream(device=device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side), torch.inference_mode():
        run()
    torch.cuda.current_stream(device).wait_stream(side)
    torch.cuda.synchronize(device)
    # The recording allocates from a private pool, which cannot borrow the blocks the eager
    # allocator is still holding; at 256K those cached blocks are most of what is free.
    _empty_cache(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.inference_mode():
        run()
    torch.cuda.synchronize(device)
    return graph


class StartupDraftGraph:
    """The prefill tail's `K - 1` draft links as one recording (see the module docstring).

    Construct it after `DraftGraph` has prepared the head cache and the prefill has produced the
    first draft, then `replay()` once the first token has been delivered. `replay` returns the same
    `K` drafts the eager chain would have produced, with `out[0]` being the seed it was built with.
    """

    def __init__(
        self,
        *,
        model,
        ctx,
        head_cache,
        start: int,
        first_draft: int,
        chain_stack: torch.Tensor,
        drafts: int,
    ) -> None:
        self.model = model
        self.head = model.mtp
        self.ctx = ctx
        self.cache = head_cache
        self.drafts = int(drafts)
        self.start = int(start)
        self.device = ctx.device
        self.graph = None
        self.replays = 0
        self._released = False
        self._gathered = None
        self._gathered_ready = False
        self._length_before = _MISSING
        self._length_pending = False

        if self.drafts < 2:
            raise ValueError("a startup graph needs at least one chained draft")
        if self.start < 0:
            raise ValueError("the startup chain cannot start before position 0")
        if not _capture_ready(self.device):
            raise RuntimeError("the startup draft graph needs a CUDA device to record on")

        layer = head_cache.layers[0]
        self.layer = layer
        if getattr(layer, "cos", None) is None:
            raise ValueError("the head cache needs its absolute RoPE table (attach_rope_table)")
        self.width = int(layer.k.shape[2])
        if getattr(layer, "kv_width", None) != self.width:
            raise ValueError(
                "the head cache's fixed KV width is not installed; build the DraftGraph first so "
                "its prepared width, expert path and RoPE are already in place"
            )
        self._require_prepared_experts()
        if self.start + self.drafts - 1 > self.width:
            raise ValueError("the head arena does not hold the whole startup chain")
        first = int(first_draft)
        if first < 0:
            raise ValueError("first_draft must be a token id")
        if not isinstance(chain_stack, torch.Tensor):
            raise TypeError("chain_stack must be a tensor")
        if chain_stack.ndim != 3 or chain_stack.shape[0] != 1 or chain_stack.shape[1] != 1:
            raise ValueError(
                "chain_stack must be the eager chain's [1, 1, hc_count * hidden] stack"
            )
        if chain_stack.device != self.device:
            raise ValueError("chain_stack must be on the head's device")
        if chain_stack.dtype is not getattr(model, "dtype", chain_stack.dtype):
            raise ValueError("chain_stack dtype must match the head's activation dtype")

        with torch.inference_mode(False):
            self.token = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            self.stack = torch.empty(
                chain_stack.shape, dtype=chain_stack.dtype, device=self.device
            )
            self.position = torch.full((), self.start, dtype=torch.long, device=self.device)
            self.out = torch.zeros(self.drafts, dtype=torch.long, device=self.device)
            self.stack.copy_(chain_stack.detach())
            self.token.fill_(first)
            # The recording copies this buffer into `out[0]`; filling it here keeps the object
            # meaningful even before the first replay.
            self.out[0] = first

        self._length_before = layer.__dict__.get("length", _MISSING)
        self._length_pending = True
        try:
            self.graph = _record(self.device, self._run)
        except BaseException:
            self.release()
            raise
        finally:
            self._restore_length()

    def _require_prepared_experts(self) -> None:
        """Fail unless `DraftGraph` already put the head's experts on a recordable path.

        The two recordable backends are the whole-device NVFP4 bank (a K-row kernel with no host
        read) and the FP8 stack whose device-indexed path `_capture_safe` selects. Both are
        installed by `DraftGraph._prepare`; installing them here would quantize or convert the
        experts a second time, so this only checks.
        """
        moe = getattr(self.head, "moe", None)
        if moe is None or not hasattr(moe, "rowwise"):
            raise RuntimeError("the head's expert backend is not a ShardedMoE to run row-wise")
        inner = getattr(moe, "inner", None)
        from relicllm.models.qwen4_exp.moe import NvFp4SlotExpertMoE

        if isinstance(inner, NvFp4SlotExpertMoE):
            return
        if inner is None or not hasattr(inner, "_capture_safe"):
            raise RuntimeError("the head's expert backend has no device-indexed path")
        if not getattr(moe, "rowwise", False) or not getattr(inner, "_capture_safe", False):
            raise RuntimeError(
                "the head's expert backend is not prepared for a recording; build the DraftGraph "
                "first instead of re-preparing it here"
            )

    def _restore_length(self) -> None:
        """Put the head layer's `length` back to what it held before this recording.

        The layer's forward advances it in Python while the recording is being made; on replay that
        Python does not run, so this is setup hygiene, not per-round bookkeeping.
        """
        if not self._length_pending:
            return
        self._length_pending = False
        if self._length_before is _MISSING:
            if "length" in getattr(self.layer, "__dict__", {}):
                delattr(self.layer, "length")
        else:
            self.layer.length = self._length_before

    def _token(self, mixed: torch.Tensor) -> torch.Tensor:
        """The head's greedy token for its last row, on device.

        The same projection, gather and argmax pair `DraftGraph` records, so a replay reproduces
        the eager chain's token rather than approximating it.
        """
        local = torch.nn.functional.linear(mixed[:, -1], self.model.lm_head)
        if not self._gathered_ready:
            with torch.inference_mode(False):
                self._gathered = _gather_buffer(local, self.ctx)
            self._gathered_ready = True
        return _greedy_rows(local, self.ctx, self._gathered).reshape(1, 1)

    def _recycle(self, mixed: torch.Tensor, stream: torch.Tensor) -> torch.Tensor:
        """The stack the next chain link is conditioned on (same rule as the eager chain)."""
        from relicllm.models.qwen4_exp.runtime import _mtp_recycle

        if _mtp_recycle() == "collapsed":
            return mixed.unsqueeze(-2).expand(
                -1, -1, self.head.config.hc_count, -1
            ).reshape_as(stream)
        return stream

    def _run(self) -> None:
        """The chain itself: seed copy, then the `K - 1` one-query head links.

        Everything here is device work on stable buffers -- no host read of a generated token, no
        cache binding, no allocation outside the recording's own pool.
        """
        self.out[0:1].copy_(self.token.reshape(1))
        token = self.token
        stack = self.stack
        for link in range(1, self.drafts):
            mixed, stream = self.head.forward(
                stack, token, self.cache,
                past_len=self.position + (link - 1), return_streams=True,
            )
            token = self._token(mixed)
            self.out[link:link + 1].copy_(token.reshape(1))
            stack = self._recycle(mixed, stream)

    def replay(
        self,
        *,
        start: int | None = None,
        first_draft: int | None = None,
        chain_stack: torch.Tensor | None = None,
    ) -> list[int]:
        """Replay the recorded chain and return its `K` drafts, the first for `start + 1`.

        `start`, `first_draft` and `chain_stack` default to the values the recording was built
        with; passing them copies new inputs into the recording's stable buffers first. `start` must
        leave room for the whole chain in the head arena, and the head cache must hold the prefix it
        implies -- the same precondition the eager chain runs under.
        """
        if self.released:
            raise RuntimeError("the startup draft graph has been released")
        position = self.start if start is None else int(start)
        if position < 0:
            raise ValueError("the startup chain cannot start before position 0")
        if position + self.drafts - 1 > self.width:
            raise ValueError("the startup chain would run past the head arena")
        if first_draft is not None:
            first = int(first_draft)
            if first < 0:
                raise ValueError("first_draft must be a token id")
            self.token.fill_(first)
        if chain_stack is not None:
            if (not isinstance(chain_stack, torch.Tensor)
                    or chain_stack.shape != self.stack.shape
                    or chain_stack.dtype is not self.stack.dtype
                    or chain_stack.device != self.stack.device):
                raise ValueError("chain_stack must repeat the recorded [1, 1, ...] stack layout")
            self.stack.copy_(chain_stack.detach())
        self.position.fill_(position)
        self.graph.replay()
        self.replays += 1
        return self.out.tolist()

    @property
    def released(self) -> bool:
        """Whether `release` has dropped this recording's graph and buffers."""
        return self._released

    def release(self) -> None:
        """Drop the recording and its buffers. Idempotent, and safe after a failed construction."""
        if self._released:
            return
        self._released = True
        self._restore_length()
        self.graph = None
        self.token = None
        self.stack = None
        self.position = None
        self.out = None
        self._gathered = None
        self._gathered_ready = False
        _empty_cache(self.device)