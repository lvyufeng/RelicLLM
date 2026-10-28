"""A window of captured decode steps, one graph per position.

A decode step is launch-bound: the GPU finishes a layer's work long before the host has queued the
next one, and on 8x3090 the host is the bound by a factor of about three (a step is 120 ms eager
against 31 ms replayed).  What a capture cannot do on its own is advance: `past_len` is a Python
integer that selects a rope slice and a cache offset, so a step is only replayable at the position
it was recorded at.

So a window is a run of positions, each with its own graph:

* position `p` has an input buffer and a graph captured at `past_len = start + p`;
* a generation writes a step's token into that position's buffer and replays its graph;
* the token each step produces is read back from a buffer the graph writes, not from the graph's
  return value -- a replay returns nothing.

Capturing costs about 0.2 s a position (each capture runs the step once as a warmup), so the
window is a trade: it pays for itself only past a few hundred generated tokens.  The runtime asks
for it only when `max_new_tokens` is large enough to justify it.

Failure is not fatal.  Anything that refuses the capture -- an unpinned crossing, a host read
inside the step, a collective the graph will not hold -- leaves the caller on the eager loop, which
is the definition these graphs are checked against anyway.
"""
from __future__ import annotations

import os

import torch


class GraphWindow:
    """`steps` captured decode steps starting at `start`, replayed by position."""

    def __init__(self, *, model, cache, start: int, steps: int, token: torch.Tensor,
                 select=None) -> None:
        self.model = model
        #: Turns one position's sharded logits into the token the model chose.  It has to run
        #: outside the capture, because it gathers the shards and that is a collective.
        self.select = select
        self.cache = cache
        self.start = start
        self.steps = steps
        #: The token each position reads.  One buffer a position, because a graph reads its input
        #: at a fixed address and two positions are two captures.
        self.inputs = [token.clone() for _ in range(steps)]
        #: The host end of the token copy, pinned so the transfer is one operation.
        self._host_token = torch.zeros(1, 1, dtype=torch.long, pin_memory=True)
        #: Where each graph leaves its answer.  Written from inside the graph, so the caller reads
        #: this after a replay without the graph having to return anything.
        self.outputs: list[torch.Tensor] = []
        #: One cache clone a position, so no capture writes another graph's history.
        self.caches: list = []
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self._capture()

    @staticmethod
    def _copy_span(dst, src) -> None:
        """Copy what two caches of different sequence lengths have in common."""
        if dst is None or src is None or dst.shape == src.shape:
            if dst is not None and src is not None:
                dst.copy_(src)
            return
        if dst.dim() != src.dim() or dst.shape[:-1] != src.shape[:-1]:
            return
        n = min(dst.shape[-1], src.shape[-1])
        dst[..., :n] = src[..., :n]

    def _arena_len(self) -> int:
        """How long the caller's KV arena is, so a clone can be the same shape."""
        for layer in self.cache.layers:
            if layer is None:
                continue
            for name in ("k", "index_k"):
                tensor = getattr(layer, name, None)
                if tensor is not None:
                    return int(tensor.shape[2] if tensor.dim() == 4 else tensor.shape[1])
        return self.start + self.steps + 2

    def _clone_cache(self, src):
        """A cache that starts out holding what `src` holds, and is otherwise its own."""
        # The same arena size as the caller's, not one sized for the window: a smaller clone had
        # to be copied into the caller's cache through a truncated slice, and the truncation left
        # the tail of every layer disagreeing with the real one.  The length is read off the
        # caller's own KV rather than guessed from the window.
        clone = self.model.make_cache(1, self._arena_len())
        tensors = ("k", "v", "index_k", "conv_state", "recurrent_state")
        with torch.inference_mode():
            for dst, source in zip(clone.layers, src.layers):
                if dst is None or source is None:
                    continue
                for name in tensors:
                    # The two arenas are sized for different sequences, so only the span both
                    # have is copied; past that the clone is blank, which is what the caller's
                    # cache holds there too.
                    self._copy_span(getattr(dst, name, None), getattr(source, name, None))
                for name in ("length", "initialized"):
                    if hasattr(dst, name):
                        setattr(dst, name, getattr(source, name))
            clone.length = src.length
            clone.ple_token_history = (
                None if src.ple_token_history is None else src.ple_token_history.clone()
            )
            clone.cos = src.cos
            clone.sin = src.sin
        return clone

    def _capture(self) -> None:
        """Record one graph a position, chained: each clone starts from what the last one did.

        The cache is what made this hard.  A graph is recorded against the KV it reads, and the
        step it records *writes* that KV -- so two positions sharing one cache record and replay
        into each other, and the first answer comes out wrong the moment there is a second
        position.  Cloning per position fixes the overlap but not the chaining: a clone taken from
        the window's opening state is missing every position the window has already decoded, and
        the answer comes out repeating itself.

        So the clones are made in order, and each is seeded from the previous one's result.  The
        history the *caller* has when it reaches a position is the history the previous position
        left, which is exactly what the previous clone holds.
        """
        model = self.model
        width = self._logits_width()
        holders = [
            torch.zeros(1, width, dtype=torch.float32, device=self.inputs[p].device)
            for p in range(self.steps)
        ]
        stream = torch.cuda.Stream()
        previous = self.cache
        for p in range(self.steps):
            past = self.start + p
            cache = self._clone_cache(previous)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                model.forward(self.inputs[p], cache=cache, past_len=past, last_token_only=True)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                out = model.forward(self.inputs[p], cache=cache, past_len=past, last_token_only=True)
                if out is not None:
                    holders[p].copy_(out[:, -1].float())
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            self.graphs.append(graph)
            self.caches.append(cache)
            previous = cache
        self.outputs = holders

    def _logits_width(self) -> int:
        """How wide one position's logits are on this rank.

        Read from a step on a cache of the window's own: asking the real one would write the
        placeholder token into the very positions the caller is about to decode at.
        """
        with torch.inference_mode():
            scratch = self.model.make_cache(1, self.start + self.steps + 2)
            probe = self.model.forward(
                self.inputs[0], cache=scratch, past_len=self.start, last_token_only=True
            )
            del scratch
            torch.cuda.empty_cache()
        return int(probe.shape[-1])

    def replay(self, position: int, token: torch.Tensor) -> torch.Tensor:
        """Run position `position`, which reads `token` and answers into `outputs[position]`.

        The graph runs against its own cache clone, so nothing another position did to the shared
        one reaches it.  The token still has to be written into the buffer the graph reads before
        the replay -- blocking, because the transfer and the replay are not ordered otherwise.
        """
        idx = position - self.start
        self._host_token.copy_(token)
        self.inputs[idx].copy_(self._host_token)
        self.graphs[idx].replay()
        if self.select is None:
            return self.outputs[idx].argmax(-1, keepdim=True)
        return self.select(self.outputs[idx])

    def __len__(self) -> int:
        return self.steps

    def release(self) -> None:
        """Give the graphs and their cache clones back, so the next window has the room."""
        self.graphs.clear()
        self.outputs.clear()
        self.inputs.clear()
        self.caches.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def commit(self, position: int) -> None:
        """Copy position `position`'s cache clone back into the caller's.

        The next window clones from the caller's cache, so a position's work has to land there
        before the generation moves on -- otherwise the next window would start from a cache that
        never heard of it.
        """
        src = self.caches[position - self.start]
        tensors = ("k", "v", "index_k", "conv_state", "recurrent_state")
        with torch.inference_mode():
            for dst, source in zip(self.cache.layers, src.layers):
                if dst is None or source is None:
                    continue
                for name in tensors:
                    self._copy_span(getattr(dst, name, None), getattr(source, name, None))
                for name in ("length", "initialized"):
                    if hasattr(dst, name):
                        setattr(dst, name, getattr(source, name))
            self.cache.length = src.length
            self.cache.ple_token_history = (
                None if src.ple_token_history is None else src.ple_token_history.clone()
            )


def build(*, model, cache, start: int, steps: int, token: torch.Tensor, select=None) -> GraphWindow | None:
    """Capture a window, or None when the step will not be captured.

    Windows are reused across generations on the same model.  A graph is recorded at a position
    and reads its input from a buffer, not from the request, so a window captured for one decode
    serves any other decode that reaches the same positions: what it holds is the shape of the
    step, and the token that goes in is copied each replay.  Capturing costs about 0.3 s a
    position, which is 30 eager steps' worth, so a second generation would otherwise pay it again
    for graphs it has already made.

    The buffers the step writes are allocated outside inference mode: tensors made inside it are
    inference tensors, and the in-place updates a later step makes to them are rejected.
    """
    if steps <= 0 or not torch.cuda.is_available():
        return None
    with torch.inference_mode(False):
        token_buf = token.clone()
    try:
        window = GraphWindow(
            model=model, cache=cache, start=start, steps=steps, token=token_buf, select=select
        )
        return window
    except Exception as exc:  # noqa: BLE001 - falling back to eager is the contract
        # torchrun routes a child's stdout to the launcher, which drops lines that arrive while it
        # is tearing the group down, so the failure goes to a file as well as to the console.
        import traceback

        detail = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        try:
            with open("/tmp/qwen4exp_graph_window_error.txt", "a") as fh:
                fh.write(detail + "\n---\n")
        except OSError:
            pass
        print("[graph_window] capture failed: " + detail.splitlines()[0], flush=True)
        return None
