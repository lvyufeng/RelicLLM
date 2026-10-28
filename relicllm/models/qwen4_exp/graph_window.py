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

        # A stream shared by the window, but a pool per graph.  Sharing the pool looked like the
        # saving it is not: positions recorded from the same pool hand their buffers back and the
        # next capture reuses them, so a position replayed later reads a buffer another position
        # has since written -- two runs over eight positions disagree from the third one on.
        # Separate pools cost memory and are the only version that answers the same thing twice.
        stream = torch.cuda.Stream()
        # Capturing runs the step, and running the step writes the KV cache -- so the capture
        # leaves the positions it recorded filled with the placeholder tokens the graphs were
        # built around.  The caller's cache has to come back as it was, or the generation would
        # read a history it never produced.  The window records what it is about to overwrite and
        # puts it back below.
        # The rotary table lives on the cache and is rebuilt when the sequence outgrows it -- a
        # new allocation, at a new address.  A captured step holds the old address, so the first
        # replay after a rebuild is an illegal memory access.  Growing it once, to the far end of
        # the window, keeps the address the graphs recorded.
        with torch.inference_mode():
            model._rope_cache(self.start + self.steps + 1, 1, cache)
        saved = self._save_cache()
        # The step's answer is its logits, not a token: the head is vocab-sharded, so an argmax
        # taken here would be over this rank's slice of the vocabulary, and the pick would be its
        # local best rather than the model's.  Gathering the shards is a collective, which a
        # capture will not hold -- so the graph leaves the logits in a buffer and the caller
        # gathers and takes the argmax after the replay.  The width is the step's own output, and
        # the probe that reads it runs before the save, so the save covers what it writes.
        width = self._logits_width()
        # Every output buffer is allocated before any capture runs.  Allocated inside the loop, a
        # position's buffer was handed back by the allocator and reused by a later capture, so the
        # position replayed afterwards wrote into memory another graph was using: one position
        # worked, two did not, and the only difference was when the buffer was created.
        holders = [
            torch.zeros(1, width, dtype=torch.float32, device=self.inputs[p].device)
            for p in range(self.steps)
        ]
        for p in range(self.steps):
            past = self.start + p
            holder = holders[p]
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.inference_mode():
                model.forward(self.inputs[p], cache=cache, past_len=past, last_token_only=True)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            # The warmup wrote this position into the cache, so the capture that follows would
            # read a history with one more token in it than the caller has -- a different step,
            # recorded at the right offset, answering a question nobody asked.  Put the cache
            # back before recording.
            self._restore_cache(saved)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=torch.cuda.graph_pool_handle()):
                out = model.forward(self.inputs[p], cache=cache, past_len=past, last_token_only=True)
                if out is not None:
                    holder.copy_(out[:, -1].float())
            torch.cuda.synchronize()
            # The `copy_` inside the capture is recorded, not performed: a graph holds the work an
            # instantiation later repeats, and until it repeats it the buffer is still whatever it
            # was allocated as -- zeros.  One replay here fills it, so the window's outputs are
            # answers from the moment it exists rather than after some later position happens to
            # replay them.
            graph.replay()
            torch.cuda.synchronize()
            self.graphs.append(graph)
            self.outputs.append(holder)
        self._restore_cache(saved)

    #: The cache tensors a capture writes, and therefore the ones it has to put back. A
    #: full-attention layer's KV is addressed by position, so only the captured span moves; a
    #: linear-attention layer's conv window and recurrent state are not addressed by position at
    #: all -- one capture advances them for good -- so those are saved whole.
    _POSITIONAL = ("k", "v", "index_k")

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

    def _save_cache(self):
        """Everything the captures are about to write, and the length to restore."""
        span = slice(self.start, self.start + self.steps + 1)
        saved = []
        # Cloned outside inference mode: a tensor made inside it is an inference tensor, and
        # copying one back into the cache is an in-place update the cache refuses.
        with torch.inference_mode(False):
            for layer in self.cache.layers:
                if layer is None:
                    continue
                entry = {}
                for name in self._POSITIONAL:
                    tensor = getattr(layer, name, None)
                    if tensor is not None:
                        entry[name] = (
                            tensor[:, :, span].clone() if tensor.dim() == 4 else tensor[:, span].clone()
                        )
                for name in ("conv_state", "recurrent_state"):
                    tensor = getattr(layer, name, None)
                    if tensor is not None:
                        entry[name] = tensor.clone()
                # A linear layer starts uninitialised and the first step flips it; a capture flips
                # it too, and restoring only the tensors would leave it reading a recurrent state
                # the caller never had.
                if hasattr(layer, "initialized"):
                    entry["initialized"] = layer.initialized
                # A full-attention layer keeps its own `length`, and a step sets it to
                # `past_len + 1`; a capture leaves it at the last position it recorded, so the
                # next eager step would read a longer history than the caller has.
                if hasattr(layer, "length"):
                    entry["length"] = layer.length
                saved.append((layer, entry))
        # The rope table is deliberately left grown: shrinking it would free the very buffers the
        # graphics recorded, which is the failure this pre-grow exists to prevent.
        return saved, self.cache.length, self.cache.ple_token_history

    def _restore_cache(self, saved) -> None:
        """Put the cache back the way the captures found it.

        Inside `inference_mode`, which is where the cache's tensors were made: writing into them
        from outside it is the in-place update that mode refuses.
        """
        entries, length, ple_history = saved
        span = slice(self.start, self.start + self.steps + 1)
        with torch.inference_mode():
            for layer, entry in entries:
                for name, tensor in entry.items():
                    if name in ("initialized", "length"):
                        setattr(layer, name, tensor)
                        continue
                    target = getattr(layer, name)
                    if name in self._POSITIONAL:
                        # The KV a capture wrote is left where it is: the replay writes the same
                        # position again, and restoring a slice the replay is about to overwrite
                        # only makes the window disagree with itself.  What has to be put back is
                        # the bookkeeping around it.
                        continue
                    target.copy_(tensor)
            self.cache.length = length
            # The n-gram history is what every PLE layer reads to pick its rows; a capture appends
            # the placeholder token to it, and a generation that kept that would be reading a
            # prompt it never received.
            self.cache.ple_token_history = (
                None if ple_history is None else ple_history.clone()
            )

    def replay(self, position: int, token: torch.Tensor) -> torch.Tensor:
        """Run position `position`, which reads `token` and answers into `outputs[position]`.

        The token goes into a pinned host buffer and crosses on the copy, which is the same path
        a graph records; a pageable `.copy_` from the device is a synchronising read-back the
        capture cannot hold.  The buffer is one int64, so the crossing is the latency of the copy
        rather than its size -- and it is on the same stream as the replay, so it is in order.
        """
        idx = position - self.start
        # The step captured at this position saw `length = position + 1` -- what one decoded token
        # leaves behind.  `_restore_cache` puts back the length the capture found, which is one
        # short, and the layer would then read a shorter history than the step was recorded
        # against: the first full-attention layer that reads it drifts by about four, and eleven
        # layers later the drift is nan.
        # A graph was recorded reading the cache with exactly this much history, so the cache has
        # to be at that length when it replays -- not at whatever the previous position left it.
        # `cache.length` and every layer's own are the whole difference between the two readings.
        self.cache.length = position + 1
        for layer in self.cache.layers:
            if layer is not None and hasattr(layer, "length"):
                layer.length = position + 1
        # And the positions past this one have to be blank: a second graph's capture wrote its own
        # placeholder there, and a KV slot beyond `length` that still holds another graph's answer
        # is read by the linear layers, which carry state across positions rather than addressing
        # it.  Blanking the tail restores the cache the recording was made against.
        if idx + 1 < self.steps:
            span = slice(position + 1, self.start + self.steps)
            with torch.inference_mode():
                for layer in self.cache.layers:
                    if layer is None:
                        continue
                    for name in self._POSITIONAL:
                        tensor = getattr(layer, name, None)
                        if tensor is not None:
                            if tensor.dim() == 4:
                                tensor[:, :, span] = 0
                            else:
                                tensor[:, span] = 0
        # Blocking, deliberately.  `non_blocking=True` hands the transfer to the copy queue and
        # returns, and the replay queued right after it can reach the device first -- the graph
        # then reads the placeholder token the window was built around instead of this step's.
        # The payload is one int64; the sync costs less than the ordering mistake did.
        self._host_token.copy_(token)
        self.inputs[idx].copy_(self._host_token)
        self.graphs[idx].replay()
        if self.select is None:
            return self.outputs[idx].argmax(-1, keepdim=True)
        return self.select(self.outputs[idx])

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
