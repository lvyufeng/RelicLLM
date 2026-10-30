"""A decode step captured once and replayed at any position.

A decode step is launch-bound: the GPU finishes a layer's work long before the host has queued the
next one, and on 8x3090 the host is the bound by a factor of about four (a step is 122 ms eager
against 31 ms replayed).

What a capture cannot do by itself is advance. `past_len` used to be a Python integer that selected
a rope slice and a cache offset, so a step was only replayable at the position it was recorded at,
and the obvious workaround -- one graph a position -- needs a cache clone a position and a way of
seeding each clone from the one before it. That is a great deal of machinery, and the machinery is
where the errors were: a clone seeded from a step that ran a placeholder token answers as if the
sequence before it did not exist; one position a clone can be left holding a key the sequence has
not produced yet; and every one of those reads back as a decode one step ahead of eager.

So the step is made position-independent instead, and there is exactly one graph:

* the KV write takes a device position (`attention._write_kv`);
* the KV read is a fixed-width arena, with the selection mask saying what is visible
  (`POCKETLLM_QWEN4_QSA_FIXED_WIDTH`);
* each linear layer's recurrence lives one entry a position, so a replay writes the next entry
  rather than advancing the one it read (`attention.slot_layout`);
* the rope table is built to the arena's width before the recording, because a table that grows
  on demand reallocates inside a capture;
* the n-gram window the PLE term hashes lives on the host, in a row this object refills between
  replays -- assembling it inside the step would be a device-to-host copy, which a capture refuses.

What stays frozen in the recording is the shape of the step, which is what a graph is for.

**What it costs, in tokens.** A replay and an eager step of the same input agree on the first
token and then diverge where two tokens are near-tied: both execute the same fused kernels in a
different order, so the logits differ in the last bits, and greedy decoding turns that into a
different choice once the top two are closer than the difference. Measured on this model, the first
five tokens of a reply match the eager loop and the sixth does not; the visible effect is a
phrasing change inside the model's reasoning, with the answer itself unchanged. A caller that needs
token-for-token agreement with the eager loop has to ask for the eager loop.
"""
from __future__ import annotations

import os

import torch

from .attention import GatedDeltaNetCache, slot_layout


class DecodeGraph:
    """One recorded decode step, replayed at whatever position the caller sets."""

    def __init__(self, *, model, cache, start: int, steps: int, token: torch.Tensor,
                 select=None) -> None:
        self.model = model
        #: Turns one step's sharded logits into the token the model chose.  It has to run outside
        #: the capture: it gathers the shards, and that is a collective.
        self.select = select
        self.cache = cache
        self.start = int(start)
        self.steps = int(steps)
        device = token.device

        #: The token the step reads.  A graph pins the address, so the caller copies into this
        #: before each replay rather than handing over a new tensor.
        with torch.inference_mode(False):
            self.inputs = token.clone()
        #: The pinned host end of that copy: the transfer and the replay are not otherwise ordered.
        self._host_token = torch.zeros(1, 1, dtype=torch.long, pin_memory=True)
        #: The position the step runs at -- one tensor, refilled before each replay.  This is what
        #: lets a single recording serve every position.
        with torch.inference_mode(False):
            self.position = torch.full((), self.start, dtype=torch.long, device=device)
            #: The same thing counted from the start of the window, which is what indexes the
            #: per-position slots.  Two tensors, not one: the position is absolute (it addresses
            #: the KV and the rope table) and the slot is an offset (it indexes `steps + 2`
            #: entries), so sharing one runs the slot past the end of its layout.
            self.slot = torch.zeros((), dtype=torch.long, device=device)
        #: How wide the KV arena is; the recording bakes it into the key shapes.
        self.width = self._arena_len()
        #: The n-gram window's width, and the host row the PLE term reads.
        self._ngram = int(model.config.ngram_size)
        self.history = torch.full(
            (1, self._ngram), int(model.config.primary_eos_token_id),
            dtype=torch.long, pin_memory=True,
        )
        self._linear = [i for i, layer in enumerate(cache.layers)
                        if isinstance(layer, GatedDeltaNetCache)]
        #: Where the step leaves its answer.  Written from inside the graph, so the caller reads
        #: it after a replay without the graph having to return anything.
        with torch.inference_mode(False):
            self.outputs = torch.zeros(1, self._logits_width(), dtype=torch.float32, device=device)
        self._prepare()
        self._capture()

    def _arena_len(self) -> int:
        """How long the caller's KV arena is; the recording bakes this into its key shapes."""
        for layer in self.cache.layers:
            if layer is None:
                continue
            for name in ("k", "index_k"):
                tensor = getattr(layer, name, None)
                if tensor is not None:
                    return int(tensor.shape[2] if tensor.dim() == 4 else tensor.shape[1])
        return self.start + self.steps + 2

    def _logits_width(self) -> int:
        """How wide one step's logits are on this rank.

        Read off a step on a scratch cache: asking the real one would write the placeholder token
        into the position the caller is about to decode at.
        """
        with torch.inference_mode():
            scratch = self.model.make_cache(1, self.start + self.steps + 2)
            probe = self.model.forward(
                self.inputs, cache=scratch, past_len=self.start, last_token_only=True
            )
            del scratch
            torch.cuda.empty_cache()
        return int(probe.shape[-1])

    def _prepare(self) -> None:
        """Lay the generation out before the recording: arena, rope table, slots, n-gram row."""
        # The arena width has to be set before the first step runs: the layers read it while the
        # step does, and the recording bakes the shape it produces.
        if os.environ.get("POCKETLLM_QWEN4_QSA_FIXED_WIDTH", "0") in ("0", ""):
            os.environ["POCKETLLM_QWEN4_QSA_FIXED_WIDTH"] = str(self.width)
        with torch.inference_mode(False):
            self.cache.rope_width = self.width
            if self.cache.cos is None or self.cache.cos.shape[1] < self.width:
                positions = torch.arange(self.width, device=self.position.device).view(1, 1, -1)
                self.cache.cos, self.cache.sin = self.model.rope(
                    positions.expand(3, 1, -1), self.model.dtype
                )
            # One entry a position, so a replay reads its own and writes the next.  The position
            # tensor drives both, which is what keeps the index out of the recording.
            for i in self._linear:
                layer = self.cache.layers[i]
                self.cache.layers[i].state_slots = slot_layout({
                    "conv_state": [layer.conv_state.clone() for _ in range(self.steps + 2)],
                    "recurrent_state": [layer.recurrent_state.clone()
                                        for _ in range(self.steps + 2)],
                })
                self.cache.layers[i].state_slot = self.slot
            self.cache.ple_history_at = lambda _ids: self.history.reshape(1, -1)

    def _run_step(self) -> None:
        hidden = self.model.forward(self.inputs, cache=self.cache, past_len=self.position,
                                    last_token_only=True)
        self.outputs.copy_(hidden[:, -1].float())

    def _capture(self) -> None:
        # A warmup on a side stream: cuBLAS picks its algorithms on first sight of a shape, and a
        # pick made during a capture invalidates it.
        torch.cuda.synchronize()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.inference_mode():
            self._run_step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._run_step()

    def fill_history(self, seen: list[int]) -> None:
        """Point the PLE window at the last `ngram` ids of `seen`, which ends in this step's token."""
        window = seen[-self._ngram:]
        self.history.fill_(int(self.model.config.primary_eos_token_id))
        self.history[0, self._ngram - len(window):] = torch.tensor(window, dtype=torch.long)

    def replay(self, position: int, token: torch.Tensor) -> torch.Tensor:
        """Run the step at `position`, reading `token`, and return the token it chose."""
        self._host_token.copy_(token)
        self.inputs.copy_(self._host_token)
        with torch.inference_mode(False):
            self.position.fill_(position)
            self.slot.fill_(position - self.start)
            # A tensor position makes `forward` leave `length` alone, and the indexer reads it.
            self.cache.length = position + 1
        self.graph.replay()
        if self.select is None:
            return self.outputs.argmax(-1, keepdim=True)
        return self.select(self.outputs)

    def __len__(self) -> int:
        """How many positions this recording serves: the slot layout is sized for them."""
        return self.steps

    def release(self) -> None:
        self.graph = None
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def build(*, model, cache, start: int, steps: int, token: torch.Tensor, select=None):
    """Capture the step, or None when it will not be captured.

    Failure is not fatal and that is the contract: anything that refuses the capture -- an
    unpinned crossing, a host read inside the step, a collective the graph will not hold --
    leaves the caller on the eager loop, which is the definition these graphs are checked
    against anyway.
    """
    if steps <= 0 or not torch.cuda.is_available():
        return None
    with torch.inference_mode(False):
        token_buf = token.clone()
    try:
        return DecodeGraph(model=model, cache=cache, start=start, steps=steps, token=token_buf,
                           select=select)
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