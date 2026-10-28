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

    def __init__(self, *, model, cache, start: int, steps: int, token: torch.Tensor) -> None:
        self.model = model
        self.cache = cache
        self.start = start
        self.steps = steps
        #: The token each position reads.  One buffer a position, because a graph reads its input
        #: at a fixed address and two positions are two captures.
        self.inputs = [token.clone() for _ in range(steps)]
        #: Where each graph leaves its answer.  Written from inside the graph, so the caller reads
        #: this after a replay without the graph having to return anything.
        self.outputs: list[torch.Tensor] = []
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self._capture()

    def _capture(self) -> None:
        model, cache = self.model, self.cache
        # Every shape the step touches has to have run before the capture, on a side stream: the
        # first cuBLAS call for a shape picks its algorithm, and picking it inside a capture
        # invalidates the capture with an error that names no call.  The warmup walks a cache of
        # its own to the same position so the shapes match without disturbing the captured one.
        # The shapes to warm are the ones a captured step uses, and a step at position `p` uses
        # them at both ends of the window: the cache slice it writes and the position it reads.
        # Walking the whole window would be `steps` passes for shapes that repeat, so this warms
        # the two ends and lets the per-position warmup below cover the rest.
        # A cache of its own: warming on the one about to be captured would write the positions
        # the capture then records, so the graphs would replay into slots already filled.
        endpoints = sorted({self.start, self.start + self.steps - 1})
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm), torch.inference_mode():
            # The warmup cache only has to reach the far end of the window, and it is as large as
            # the capture's, so it is freed before the captures start: two of them held at once is
            # two of the largest allocations on a card that already holds the experts.
            warm_cache = model.make_cache(1, self.start + self.steps + 2)
            for past in endpoints:
                for _ in range(2):
                    model.forward(self.inputs[0], cache=warm_cache, past_len=past, last_token_only=True)
            del warm_cache
            torch.cuda.empty_cache()
        torch.cuda.current_stream().wait_stream(warm)
        torch.cuda.synchronize()

        # One stream and one pool for the whole window.  A pool a capture allocates from is held
        # for as long as any graph that used it lives, so a pool a position would be sixteen pools
        # the window never gives back; sharing them means the window's memory is bounded by what
        # one step allocates, not by its length.
        stream = torch.cuda.Stream()
        pool = torch.cuda.graph_pool_handle()
        # The per-position warmup is one pass, not two: the shapes repeat from position to
        # position and the endpoint warmup above already ran them, so a second pass here only
        # doubles the setup.  What is left per position is the capture itself, which is one pass.
        for p in range(self.steps):
            past = self.start + p
            holder = torch.zeros(1, 1, dtype=torch.long, device=self.inputs[p].device)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                model.forward(self.inputs[p], cache=cache, past_len=past, last_token_only=True)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                # The argmax is inside the graph for the same reason the step is: read back to the
                # host, it would be the one sync a replay costs.
                out = model.forward(self.inputs[p], cache=cache, past_len=past, last_token_only=True)
                if out is not None:
                    holder.copy_(out[:, -1].argmax(-1, keepdim=True))
            torch.cuda.synchronize()
            self.graphs.append(graph)
            self.outputs.append(holder)

    def replay(self, position: int, token: torch.Tensor) -> torch.Tensor:
        """Run position `position`, which reads `token` and answers into `outputs[position]`."""
        idx = position - self.start
        self.inputs[idx].copy_(token)
        self.graphs[idx].replay()
        return self.outputs[idx]

    def __len__(self) -> int:
        return self.steps

    def release(self) -> None:
        """Give the window's graphs back, so the next stretch can be captured in their place.

        A captured graph holds the pool it allocated from for as long as it lives, and on 8x3090
        a window of about sixty-four positions is what the resident experts leave room for.  A
        generation longer than one window therefore has to capture in rounds, and the round that
        has been replayed is worth nothing once its positions are spent.
        """
        self.graphs.clear()
        self.outputs.clear()
        self.inputs.clear()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def build(*, model, cache, start: int, steps: int, token: torch.Tensor) -> GraphWindow | None:
    """Capture a window, or None when the step will not be captured.

    The buffers the step writes are allocated outside inference mode: tensors made inside it are
    inference tensors, and the in-place updates a later step makes to them are rejected.
    """
    print(f"[graph_window] build start={start} steps={steps} cuda={torch.cuda.is_available()}",
          flush=True)
    if steps <= 0 or not torch.cuda.is_available():
        print(f"[graph_window] refusing: steps={steps}", flush=True)
        return None
    with torch.inference_mode(False):
        token_buf = token.clone()
    try:
        window = GraphWindow(model=model, cache=cache, start=start, steps=steps, token=token_buf)
        print(f"[graph_window] captured {len(window.graphs)} graphs from {start}", flush=True)
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
