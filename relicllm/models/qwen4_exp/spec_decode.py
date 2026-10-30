"""A whole generation call as one CUDA graph, built from the model's own layer calls.

The decode step is launch-bound long before it is bandwidth-bound. Measured on the 8xRTX 3090 box:
a kernel's launch-to-launch time is 8.3 us eager, and a 48-layer step issues several thousand of
them, so ~60% of the wall clock is the host queueing work rather than the GPU running it. A graph
removes that host cost, and a graph of the *whole sequence* removes it once instead of once a step.

Three properties of the step make that possible, and each is a constraint on this module:

* **The only host decision a step makes is the token.** Greedy decode is an argmax over the gathered
  logits; routers, routes and block scales all stay on the device. So the token buffer is a static
  input, `torch.argmax` writes into it, and the next step reads the same address.
* **The sequence position has to be an address, not an integer.** Every step advances the cache
  and the rope table by one, which a fixed graph cannot do through a Python counter. The KV cache
  is therefore laid out *step-major* -- slot `p` holds position `p`, size it for the whole
  generation -- and each step narrows the tensors to its own slot, so the graph's per-step node
  simply points at a different offset. Caches whose step is a recurrence rather than a write (a
  linear attention's running state) arrive here pre-advanced: one array per position, step `p`
  reading entry `p - 1` and writing `p`.
* **Nothing per-step may be allocated.** The graphs are captured with the caching allocator in a
  private memory pool, so an allocation inside one is illegal; every scratch tensor a step needs
  has to exist already.

What this module does *not* do is re-implement a layer. It builds one `(cache, hidden, ids,
past_len)` per step against the same `Qwen4ExpDecoderLayer.__call__` the eager path runs, so the
captured arithmetic is the eager arithmetic -- the thing that differs between a graph and the loop
is where the launches go, not what they compute. That is also how a divergence is debugged: run
the same steps eagerly and compare step by step.

Scope: a single sequence (batch 1). Speculation needs per-step hidden states the graph does not
keep, and a batch needs one graph per batch shape; neither is here yet.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


@dataclass
class SlotCache:
    """One cached entry per position, narrowed to a single step before a layer sees it.

    `entries[i]` is position `i`'s state; a layer that reads `cache[kv_len]` and writes
    `cache[kv_len]` (convolutions and q/k/v rows owe one position) wants entry `p` and is handed a
    view of it, while a layer whose write is a function of *past* state as well (a linear attention
    recurrence) is handed `entries[p - 1]`, which is the previous step's output slot and is the
    entry it writes this step. `start` is the first position the generation writes.
    """

    entries: list
    start: int

    def at(self, position: int):
        index = max(position - 1, 0) if position > self.start else position - 1
        return self.entries[max(index, 0)]


class _SlotList(list):
    """`cache.layers[i]` per step: a list is what the layers index, the values are per-step."""

    def __init__(self, slots):
        super().__init__(slots)


def _torch_cat(tensors):
    """One `torch.cat` over a list that may hold None: the chunked collective's own op.

    Every rank contributes the same number of pieces of the same shape, so the concatenation is
    a fixed [stages * rows, width] buffer -- and it is the *only* tensor the collective needs to
    move for a step's tokens, instead of one broadcast per stage.
    """


@dataclass
class TokenExchange:
    """How a step's next token reaches the stage that holds the embeddings.

    On one rank the token is already in the graph's own buffer. Across a pipeline it has to be
    gathered: the last stage writes each step's token into its slot of a `[stages, 1]` buffer, the
    buffer is all-gathered once, and the first stage reads its own copy. The step ids a stage
    embeds are therefore a slice of a gradient-free int64 tensor that the collective fills.
    """

    tokens: torch.Tensor          # [3, 1, steps] int64: the ids every stage embeds, per step
    gather: Callable[[torch.Tensor], None]


def build_grid(model, ctx, *, steps: int, start: int, exchange: TokenExchange,
               all_reduce=None) -> "GraphDecode":
    """Lay out the per-step caches and the static inputs a graph of `steps` steps needs."""
    config = model.config
    width = start + steps
    layers = {idx: layer for idx, layer in enumerate(model.layers) if layer is not None}
    with torch.inference_mode():
        sample = model.layers[next(iter(layers))].make_cache(1, width, model.device, model.dtype)
        cos, sin = model._rope_cache(width, 1)
    from relicllm.models.qwen4_exp.model import Qwen4ExpCache

    spiked = Qwen4ExpCache(layers=[])
    spiked.cos = cos.reshape(3, 1, width, -1)
    spiked.sin = sin.reshape(3, 1, width, -1)
    return GraphDecode(
        model=model,
        ctx=ctx,
        steps=steps,
        start=start,
        width=width,
        sample=sample,
        rope=(spiked.cos, spiked.sin),
        exchange=exchange,
        all_reduce=all_reduce,
    )


class GraphDecode:
    """`steps` captures of one step each, replayed from one call.

    The eager path is still the definition: `capture` runs each step eagerly once (the warmup the
    capture needs anyway), then records it; `replay` issues all the records on one stream. Kept
    separate so a caller can run `capture` then compare a replay against an eager run of the same
    steps -- which is the test, since a graph that captured the wrong thing still replays fast.
    """

    def __init__(self, *, model, ctx, steps: int, start: int, width: int, sample,
                 rope, exchange: TokenExchange, all_reduce=None):
        self.model = model
        self.ctx = ctx
        self.steps = steps
        self.start = start
        self.width = width
        self.exchange = exchange
        self.all_reduce = all_reduce
        self.cos, self.sin = rope
        self.last_stage = ctx.pp_rank == ctx.pp_size - 1
        self.stages = ctx.pp_size
        # One token per step, shared: the last stage's argmax writes step p's slot.
        self.tokens = exchange.tokens
        self._sample = sample
        # What a caller reads: the collected tokens for the whole generation.
        self.out_tokens = self.tokens[0, 0] if all_reduce is None else None
        self.logits_tail = None

    # -- caches ---------------------------------------------------------------

    def _cache_for_step(self, step: int):
        """The caches a step writes: one list per position, each step narrowing its own slot."""
        raise NotImplementedError

    # -- the step -------------------------------------------------------------

    def _step(self, step: int) -> None:
        model, ctx = self.model, self.ctx
        ids = self.tokens[:, :, step]
        past = self.start + step
        hidden = model.embed_tokens(ids).to(model.dtype)
        hidden = hidden.repeat(1, 1, model.config.hc_count)
        cache = model.make_cache(1, self.width) if step < 0 else self._cache_for_step(step)
        cos = self.cos[:, :, past : past + 1]
        sin = self.sin[:, :, past : past + 1]
        for idx, layer in enumerate(model.layers):
            if layer is None:
                continue
            hidden = layer(
                hidden,
                cos=cos,
                sin=sin,
                cache=cache.layers[idx],
                past_len=past,
                token_history=ids,
                use_cache=True,
                profiler=None,
            )
        if self.last_stage:
            hidden = model.final_mixer(hidden[:, -1:])
            logits = F.linear(hidden, model.lm_head)
            if self.stages > 1:
                nxt = logits.argmax(-1)
                self.exchange.next_slot(step).copy_(nxt)
                self.exchange.gather(self.exchange.buffer)
            else:
                torch.argmax(logits[0, -1], -1, keepdim=True,
                             out=self.tokens[0, 0, step].reshape(1))