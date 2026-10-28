"""Qwen3.8-Flash-Next under RelicLLM's backend contract.

A Qwen3.8-Flash-Next checkpoint needs its own adapter rather than a flag on
:mod:`torch_backend`, because it is none of the runtimes already here: this one serves a hybrid
GatedDeltaNet + QSA text stack with routed experts and a host-resident embedding table, through
``relicllm/models/qwen4_exp/``, and nothing in that package reads a GGUF file or the DeepSeek
runtime's plan.

**Why this adapter exists at all.** ``models/qwen4_exp/runtime.py`` could measure the checkpoint
from a ``torchrun`` launch, and ``docs/performance/qwen4_exp_performance.md`` records it as one of
the fastest prefill rates on this box -- 753.25 tokens a second at an 8192-token chunk, above every
model that *was* served. What no harness could do is keep the process alive across requests: the
checkpoint is read once, each rank's host expert shard is filled once, and the model then answers an
OpenAI request. That difference -- one prompt to a process, versus a process to a service -- is the
whole of what is added here.

**What outlives a request is nothing, and that is stated rather than hidden.** Unlike ``v41``,
``mimo`` and ``xing4`` this runtime has no prefix store: the loop forward-passes whatever prompt it
is given, so a chat turn that resends its history pays for the history every time.
``reads_prefix_cache`` is False accordingly -- a client reads "a repeated prefix is *not* resumed"
rather than a switch that quietly does nothing.

**Requests serialize.** One mutable cache serves one sequence here, exactly as for the other
runtimes, so ``supports_batch`` is False and the lock at the backend boundary is the price.

**Tensor parallelism is the supervisor's**, identically to V4.1 and MiMo: one process a rank, the
group off the environment ``_supervise_rank_zero`` sets, and the per-layer ``all_reduce`` that
makes a rank mandatory rather than optional. Every rank runs the same greedy loop and the collective
returns the same sum everywhere, so the tokens agree without a message between them and only rank 0
returns them. Cancellation is the one thing that cannot be local -- a rank that stopped on its own
would leave its peers inside a layer -- so a cancel is a per-step ``broadcast`` of one int32 from
rank 0, the seam ``ShardedWorkerMixin._step_sync`` writes once for this runtime and MiMo.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from relicllm.api import (
    BackendCapabilities,
    ConfigurationError,
    GenerationRequest,
)

from relicllm.scheduler import KVCapacity

from .base import RuntimeAdapter
from .capabilities import IGNORED_OPTIONS, declared_capabilities
from .options import BackendOption, Group, Kind, decode_args
from .runtime_engine import card_for_rank, visible_card_count
from .sharded import ShardedWorkerMixin
from .shared_options import EXPERT_CACHE, PREFILL_CHUNK

DEFAULT_MAX_SEQ_LEN = 32768
"""Positions the cache is sized at when ``--max-model-len`` is not given.

Unlike MiMo's this is not a per-token byte budget a launcher is choosing to spend: the GatedDeltaNet
state is a fixed size and the QSA cache grows with the context, and both are small against the dense
weights this rank already holds. The released checkpoint's own ``max_position_embeddings`` is larger
than a first deployment sends, so the smaller number is the default and ``--max-model-len`` raises it
for a long-context service.
"""

DEFAULT_PREFILL_CHUNK = 512
"""Tokens one prefill forward takes.

The chunk keeps the QSA score matrix bounded -- a single-shot 64K prefill would need a 64K x 64K
mask a layer, which does not fit -- so it is not a free width to widen. It is also the prefill rate's
main lever, measured on this host (``docs/performance/qwen4_exp_performance.md``): a clean strict-BF16
sweep gave 154.70, 257.88, 409.14, 580.68 and 753.25 tokens a second for chunks 512, 1024, 2048, 4096
and 8192, a wider chunk amortising the per-layer dispatch and staging. The default is the launcher's
own 512 rather than the fastest, because the width a deployment should raise it to is a function of
its card room and its context, and a chunk that does not fit is not a slow run but a failed one.
"""


#: Every option this runtime reads. ``prefill_chunk`` is the shared concept, declared once and
#: answered here with this runtime's own constant; the rest are this runtime's.
OPTIONS: tuple[BackendOption, ...] = (
    replace(PREFILL_CHUNK, default=DEFAULT_PREFILL_CHUNK),
    replace(
        EXPERT_CACHE,
        default=0,
        # This runtime's own resolution: unset is not "no cache", it is zero experts cached.
        help=EXPERT_CACHE.help + "; 0 restages every step",
    ),
    BackendOption(
        "pin_experts",
        Kind.FLAG,
        True,
        "page-lock the host expert shard, which is what makes the H2D copies asynchronous",
        group=Group.EXPERT,
    ),
    BackendOption(
        "host_expert_memory",
        Kind.FLAG,
        True,
        "hold this rank's expert shard in host RAM rather than reading it from the mapping",
        group=Group.EXPERT,
    ),
    BackendOption(
        "expert_resident",
        Kind.FLAG,
        False,
        "hold this rank's expert shard on the *device* instead, so a decode step stages nothing; "
        "needs the experts to fit on the card, and the loader refuses by name when they do not",
        group=Group.EXPERT,
    ),
    BackendOption(
        "expert_reserve",
        Kind.BYTES,
        3 * (1 << 30),
        "device memory left free for activations and KV when the experts are device-resident; "
        "takes a suffix (3g)",
        group=Group.EXPERT,
    ),
)


@dataclass(slots=True)
class _Options:
    """The launcher's levers, resolved once at construction.

    Every one of these changes what the run does, which is why an unknown key is refused rather than
    ignored: ``prefill_chunk`` is the width a prompt goes through at, ``expert_cache`` is how many
    staged experts the card keeps instead of copying a token, and the two flags are where the host
    shard lives and whether it is page-locked. The field names are the declarations' canonical names,
    which is what lets ``tests/test_declared_options.py`` compare the two lists.
    """

    prefill_chunk: int = DEFAULT_PREFILL_CHUNK
    expert_cache: int = 0
    pin_experts: bool = True
    host_expert_memory: bool = True
    expert_resident: bool = False
    expert_reserve: int = 3 << 30

    @classmethod
    def from_args(cls, args: Any) -> "_Options":
        values = decode_args(OPTIONS, args, runtime="qwen4_exp", ignored=IGNORED_OPTIONS)
        return cls(
            prefill_chunk=int(values["prefill_chunk"]),
            # ``None`` is *unset* here -- a launch that named neither the flag nor the option -- and
            # the loader's own ``0`` is "restage every step". The two are one answer for this
            # runtime, so an unset capacity becomes the loader's default rather than reaching it.
            expert_cache=0 if values["expert_cache"] is None else int(values["expert_cache"]),
            pin_experts=bool(values["pin_experts"]),
            host_expert_memory=bool(values["host_expert_memory"]),
            expert_resident=bool(values["expert_resident"]),
            expert_reserve=int(values["expert_reserve"]),
        )


class Qwen4ExpBackend(ShardedWorkerMixin, RuntimeAdapter):
    """One Qwen3.8-Flash-Next checkpoint, one process a rank, one request at a time."""

    #: Read by `RuntimeAdapter._tokenize`, which is the only place a runtime's name is needed.
    _RUNTIME_LABEL = "Qwen3.8-Flash-Next"

    name = "qwen4_exp"

    def __init__(
        self,
        args: Any,
        *,
        loader: Callable[[Any, _Options], Any] | None = None,
        tokenizer: Any = None,
    ) -> None:
        super().__init__()
        self.args = args
        self._options = _Options.from_args(args)
        self._checkpoint_dir = str(getattr(args, "model", "") or "")
        self._tokenizer_path = str(getattr(args, "tokenizer_path", "") or "")
        self._max_seq_len = int(getattr(args, "max_model_len", 0) or 0) or DEFAULT_MAX_SEQ_LEN
        if self._max_seq_len < 16:
            raise ConfigurationError(
                f"max_model_len of {self._max_seq_len} leaves no room for a prompt and an answer"
            )
        #: The cards the launch named, in rank order; empty when it named none.
        self._device_ids = tuple(getattr(args, "device_ids", ()) or ())
        self._loader = loader
        self._tokenizer = tokenizer
        self._model: Any = None
        self._config: Any = None
        self._ctx: Any = None
        self._device: Any = None
        self._world = 1
        self._rank = 0
        self._distributed = False
        self._details: dict[str, Any] = {}
        #: What the rank-0 token callback raised, held until the group has unwound so it can be
        #: re-raised from the request that owns it. See :meth:`_rank0_callback` for why it cannot
        #: be raised where it was caught.
        self._callback_failure: list[BaseException] = []

    # ------------------------------------------------------------------ lifecycle

    def _init_distributed(self) -> None:
        """Join the process group the launcher set, if there is one.

        Read from the environment and not from the arguments, because a ``torchrun`` launch sets only
        the environment and a supervised one sets both: two sources that could disagree are one
        source too many, and the group is the thing that has to be right.

        ``qwen4_exp.runtime.init_distributed`` is that reader -- it takes ``RANK``/``WORLD_SIZE``/
        ``LOCAL_RANK``, which ``TensorParallelSupervisor._child_env`` sets beside the ``TP_*`` names
        -- so this method exists only to record what it returned. A world of one with no accelerator
        is a legitimate host run there rather than a misconfiguration, and its device is ``cpu``.
        """
        if self._distributed:
            return
        from relicllm.models.qwen4_exp.runtime import init_distributed

        self._bind_named_card()
        self._ctx = init_distributed()
        self._world, self._rank = self._ctx.world_size, self._ctx.rank
        self._device = self._ctx.device
        self._distributed = True

    def _bind_named_card(self) -> None:
        """Make ``--device-ids`` reach the bind, since this runtime reads the card itself.

        ``init_distributed`` binds ``LOCAL_RANK`` of what the process can see. That is the right card
        for a supervised or ``torchrun`` launch -- both set ``LOCAL_RANK`` to the rank -- but it is
        not a *named* card on an unnarrowed host, and a launch that narrowed visibility to the wrong
        rank would bind a card another rank owns. Rewriting ``LOCAL_RANK`` to the named index, in the
        space the process can see, is what makes the flag mean the same thing here as on MiMo.
        ``card_for_rank`` is that mapping -- ``device_ids[rank]`` when named, ``rank`` when not, and
        ``0`` for a world of one -- and it is written here rather than after the bind because the
        bind is inside ``init_distributed``.
        """
        if not self._device_ids:
            return
        index = card_for_rank(
            self._device_ids,
            rank=int(getattr(self.args, "tensor_parallel_rank", 0) or 0),
            world=int(getattr(self.args, "tensor_parallel_size", 1) or 1),
            visible=visible_card_count(),
        )
        os.environ["LOCAL_RANK"] = str(index)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._state_lock:
            if self._model is not None:
                return
            self._init_distributed()
            if self._loader is not None:
                # An injected loader is a test's or an embedding application's model, and it comes
                # with its own tokenizer: nothing here reaches for the checkpoint, so a run with one
                # never needs a checkpoint directory on disk.
                self._model, self._config = self._loader(self.args, self._options)
                self._build_details()
                self._ready = True
                return
            self._load()

    def _load(self) -> None:
        """Read the checkpoint, fill this rank's host expert shard, and build the tree once.

        Every rank fills its own disjoint shard of the routed experts into host RAM before any step
        runs, so steady-state staging never seeks, and the model is built over the ``all_reduce`` a
        rank has to be present for. Both are startup work this runtime measures in minutes, which is
        why the supervisor's rendezvous timeout is measured in hours.
        """
        from relicllm.models.qwen4_exp.runtime import load_model

        if not os.path.isdir(self._checkpoint_dir):
            raise ConfigurationError(f"no checkpoint at {self._checkpoint_dir}")
        options = self._options
        self._model, self._config = load_model(
            self._checkpoint_dir,
            self._ctx,
            dtype=torch.bfloat16,
            expert_cache_capacity=options.expert_cache,
            host_expert_memory=options.host_expert_memory,
            pin_experts=options.pin_experts,
            expert_resident=options.expert_resident,
            expert_reserve_bytes=options.expert_reserve,
        )
        self._build_details()
        if self._tokenizer is None:
            self._tokenizer = self._open_tokenizer()
        self._publish_ready()

    def _build_details(self) -> None:
        text = getattr(self._config, "text_config", None)
        layers = getattr(text, "num_hidden_layers", None)
        experts = getattr(text, "num_experts", None)
        self._details = {
            "execution": "relicllm/models/qwen4_exp PyTorch runtime, GatedDeltaNet + QSA",
            "experts": (
                f"{experts} routed experts a layer, this rank's shard resident in host RAM, only "
                f"the draw staged to the card"
                if experts
                else "no routed experts"
            ),
            "attention": (
                f"{layers} layers, hybrid GatedDeltaNet (a fixed recurrent state) and QSA (a cache "
                f"that grows with the context)"
                if layers
                else "hybrid GatedDeltaNet and QSA"
            ),
            "context": (
                f"{self._max_seq_len} positions at most; each request's QSA cache is sized to "
                f"its own prompt plus its budget"
            ),
            "prefill": f"one forward a {self._options.prefill_chunk}-token chunk",
            "tensor_parallel": (
                f"world {self._world}, a per-layer all-reduce" if self._world > 1 else "one rank"
            ),
            "cancellation": "per-step broadcast; not inside a prefill chunk",
        }

    # ------------------------------------------------------------------ contract

    @property
    def capabilities(self) -> BackendCapabilities:
        # No prefix store, so the declaration's False stands -- stated through the same call the
        # others use so the answer cannot drift from the class. `max_batch_size` is the instance's
        # reading of the serialized scheduler, as on MiMo.
        return declared_capabilities(
            self.name,
            details={
                **self._details,
                "scheduler": "one mutable cache, serialized at the backend boundary",
                "max_batch_size": 1,
                "device": str(self._device),
            },
            reads_prefix_cache=False,
        )

    def _host_expert_moe(self) -> Any:
        """This run's ``HostExpertMoE``, or None where there is no host bank.

        ``moe`` belongs to the decoder layer, and the model holds those in ``layers``
        (``models/qwen4_exp/model.py``), so the counters are one level below where a
        model-level lookup would find them. ``ShardedMoE`` wraps the bank when the
        world size is above one, so the wrapper is unwrapped here too. Every layer
        shares one bank, so the first layer that has one answers for the run.
        """
        candidates: list[Any] = []
        model = self._model
        if model is not None:
            direct = getattr(model, "moe", None)
            if direct is not None:
                candidates.append(direct)
            for layer in getattr(model, "layers", ()) or ():
                layer_moe = getattr(layer, "moe", None)
                if layer_moe is not None:
                    candidates.append(layer_moe)
                    break
        for candidate in candidates:
            inner = getattr(candidate, "inner", candidate)
            if inner is not None and hasattr(inner, "stage_bytes"):
                return inner
        return None

    def _expert_stage_metrics(self) -> dict[str, float]:
        """The routed-expert staging counters, or nothing when this run has no host bank.

        These are the numbers that say whether the experts are *resident* or merely *cached*: a
        deployment that sizes ``--expert-cache`` to hold every expert it owns stops staging after the
        first pass over the set, and one that does not keeps paying the PCIe cost every step. Without
        them on ``/metrics`` a decode rate cannot be told apart from a bigger LRU -- which is the
        whole question this runtime's decode is asked to answer.

        They live on ``HostExpertMoE``, one level below the ``ShardedMoE`` wrapper the model holds, so
        both are unwrapped here rather than made public on the wrapper for a scrape's sake.
        """
        inner = self._host_expert_moe()
        if inner is None:
            return {}
        return {
            # `_total` is the server's convention for a counter; the rest are gauges.
            "qwen4_exp_expert_stage_bytes_total": float(inner.stage_bytes),
            "qwen4_exp_expert_stage_calls_total": float(inner.stage_calls),
            "qwen4_exp_expert_cache_hits_total": float(inner.cache_hits),
            "qwen4_exp_expert_cache_capacity": float(getattr(inner, "cache_capacity", 0) or 0),
        }

    def metrics(self) -> dict[str, float]:
        """What one engine holds between requests, for ``/metrics``.

        The card's own numbers, which no request owns a share of. ``gpu_memory_allocated`` is what the
        loader left resident -- the dense shard and the caches -- and is the quantity a deployment
        reads against its card size when deciding how wide a context it can afford.
        """
        if self._model is None or self._device is None:
            return {}
        allocated = 0
        if getattr(self._device, "type", "") == "cuda":
            allocated = int(torch.cuda.memory_allocated(self._device))
        return {
            "qwen4_exp_device_bytes": float(allocated),
            #: The configured ceiling, **not** the cache's size: this runtime's QSA buffers are
            #: sized per request (``prompt + budget + 1``) and never from ``--max-model-len``, so
            #: no resident allocation holds this many positions. Kept under the name it was
            #: published as -- a scrape contract -- with the meaning stated here; the details
            #: string is where that is said to a caller.
            "qwen4_exp_context_positions": float(self._max_seq_len),
            "qwen4_exp_world": float(self._world),
        }

    def _kv_capacity(self) -> KVCapacity:
        """This run's context ceiling, marked as a number no cache was sized from.

        The QSA key/value and index buffers are built per request -- ``prompt_len + budget + 1`` at
        ``models/qwen4_exp/runtime.py:300`` -- so ``--max-model-len`` never reaches an allocation
        here. Checking a request against it would compare the request to a bound that had no part in
        building the buffer it runs in, and would refuse a request the runtime would have served.
        :attr:`KVCapacity.sizes_from_request` is what says so, and admission then admits everything,
        which is what this runtime did before the check had a name.
        """
        return KVCapacity(
            positions=int(self._max_seq_len),
            label="this run's context",
            sizes_from_request=True,
        )

    # -------------------------------------------------------------------- requests

    def _checkpoint_eos_token_id(self) -> Any:
        """Qwen4-Exp's end-of-turn id lives on the language tower under its text config."""
        text = getattr(self._config, "text_config", None)
        return getattr(text, "eos_token_id", None) if text is not None else None

    def _loop(
        self,
        prompt_ids: Sequence[int],
        budget: int,
        request: GenerationRequest,
        *,
        on_token: Callable[[int], None] | None,
        stop: Callable[[], bool] | None = None,
    ) -> Any:
        """This runtime's loop, with the cancellation seam the routed layers demand.

        No ``marks`` parameter: the wall clock an answer's total is measured from is written by
        whichever caller put the loop under its own lock, and both the serial and the streamed path
        do. What this method owns is the step boundary -- ``_step_sync`` is a collective here, not a
        local flag -- and that is why a serial request passes a ``stop`` as well.

        The loop is greedy -- ``models/qwen4_exp/runtime.py`` samples with ``argmax`` -- so the
        request's sampler is not forwarded, because there is nothing here for it to change.
        """
        from relicllm.models.qwen4_exp.runtime import generate

        self._ensure_loaded()
        # The prompt and the budget are what the workers reproduce, so rank 0 broadcasts them before
        # it enters the collective the workers must be inside.
        self._dispatch(request, prompt_ids, budget)
        # `local_eos` is the loop's own end-of-sequence finding, handed back to this rank's
        # predicate so it rides the broadcast `_step_sync` already performs instead of ending the
        # loop on one rank. See `generate`'s `local_eos` paragraph: acting on it locally is what
        # produced the 600-second NCCL broadcast timeout on both the V100 and the 4090 boxes.
        local_eos = [False]
        tokens, stats = generate(
            self._model,
            self._ctx,
            torch.tensor([list(prompt_ids)], dtype=torch.long),
            max_new_tokens=budget,
            chunk_size=self._options.prefill_chunk,
            eos_token_ids=tuple(sorted(self._eos_tokens())),
            on_token=self._rank0_callback(on_token, request.request_id),
            on_step=self._step_sync(
                request.request_id,
                None if stop is None else (lambda: stop() or local_eos[0]),
            ),
            local_eos=local_eos,
        )
        # The callback's exception has to leave here rather than from inside the loop: by the time
        # it is recorded the ranks have unwound together, and this is the frame that owns the
        # request. Raising it from the callback is the hang `_rank0_callback` documents.
        self._raise_callback_failure()
        return _Generation.from_stats(tokens, stats)

    def _rank0_callback(
        self, on_token: Callable[[int], None] | None, request_id: str
    ) -> Callable[[int], None] | None:
        """Wrap the token callback so a failure stops the group, not just rank 0.

        ``on_token`` runs on rank 0 alone -- the loop calls it under ``ctx.rank == 0`` and every
        other rank has already moved on to the *next* step's collective -- so an exception escaping
        it unwinds rank 0 out of a broadcast its peers are already waiting in. The request does not
        fail; it hangs, and then the watchdog kills the process:

            WorkNCCL(OpType=BROADCAST, NumelIn=1, NumelOut=1, Timeout(ms)=600000)
            ran for 600000 milliseconds before timing out.

        That shape is shared with the end-of-sequence bug ``local_eos`` fixes, but it is a second,
        independent exit: ``local_eos`` covers a token *disagreement*, and this covers one rank
        raising where the others cannot see it. Measured on 8 x RTX 4090 by injecting a raise at
        the callback: rank 0 went idle at 0% utilization while the other seven spun at 100% until
        the watchdog fired, and a second request to the still-``ready`` server never returned.

        The exception is recorded and translated into a ``cancel()``, which is the one channel that
        reaches the other ranks: :meth:`ShardedWorkerMixin._step_sync` broadcasts rank 0's
        cancellation flag at the top of the next step, so every rank leaves at a boundary they all
        arrive at. The failed step is still forwarded and its ``forward`` still runs -- dropping out
        *here* is the desync this method exists to remove. The recorded exception is re-raised by
        the caller once the group has unwound, because it cannot be raised in place for the same
        reason it cannot be allowed to propagate.
        """
        if on_token is None:
            return None

        def guarded(token: int) -> None:
            # Once a failure is recorded the callback stays silent: the loop is only running out
            # the steps every rank must still take, and a second raise would be noise on top of a
            # request that is already ending.
            if self._callback_failure:
                return
            try:
                on_token(token)
            except Exception as exc:  # noqa: BLE001 -- catching everything is the point
                self._callback_failure.append(exc)
                self.cancel(request_id)

        return guarded

    def _raise_callback_failure(self) -> None:
        """Re-raise what :meth:`_rank0_callback` recorded, once the group has unwound."""
        if not self._callback_failure:
            return
        raise self._callback_failure.pop()

    def _run_payload(self, payload: Mapping[str, Any]) -> None:
        """Run the same loop rank 0 is running, for the collective's sake and not for its answer.

        The prompt ids and the budget are the ones rank 0 tokenized and resolved, and the answer is
        discarded: the tokens agree across ranks because the logits do, so only rank 0's leave.
        """
        from relicllm.models.qwen4_exp.runtime import generate

        prompt_ids = [int(token) for token in payload["prompt_ids"]]
        # The worker carries the flag too, so both sides of the broadcast are asking the same
        # question on every step. It decides nothing here -- this rank's answer is discarded -- it
        # only has to reach the collective in step with rank 0.
        local_eos = [False]
        generate(
            self._model,
            self._ctx,
            torch.tensor([prompt_ids], dtype=torch.long),
            max_new_tokens=int(payload["max_new_tokens"]),
            chunk_size=self._options.prefill_chunk,
            eos_token_ids=tuple(sorted(self._eos_tokens())),
            on_step=self._step_sync(
                str(payload["request_id"]), lambda: local_eos[0]
            ),
            local_eos=local_eos,
        )


@dataclass(slots=True)
class _Generation:
    """The loop's answer in the shape :meth:`RuntimeAdapter._result` reads.

    ``models/qwen4_exp/runtime.py::generate`` returns ``(tokens, stats)`` rather than the
    ``Generation`` dataclass the other runtimes hand back, so the two are reconciled here rather than
    by teaching ``_result`` about a second shape. There is no ``cached_tokens`` to carry -- no prefix
    store -- so it stays at the base's zero, which is what "nothing was resumed" means.
    """

    tokens: list[int]
    stopped: str = "length"
    prefill_seconds: float = 0.0
    decode_seconds: float = 0.0
    ttft_seconds: float = 0.0
    cached_tokens: int = 0

    @classmethod
    def from_stats(cls, tokens: Sequence[int], stats: Mapping[str, float]) -> "_Generation":
        return cls(
            tokens=[int(token) for token in tokens],
            stopped=str(stats.get("stopped", "length")),
            prefill_seconds=float(stats.get("prefill_s", 0.0)),
            decode_seconds=float(stats.get("decode_s", 0.0)),
            ttft_seconds=float(stats.get("ttft_s", 0.0)),
        )

    @property
    def step_seconds(self) -> float:
        """Seconds a decode step, which is the per-token latency ``tpot`` reports.

        The decode clock covers the whole loop including the first token's own forward, so a
        generation of n tokens has n-1 gaps between them; one token has no gap and reports the whole
        decode, which is the only number there is.
        """
        return self.decode_seconds / max(1, len(self.tokens) - 1)


__all__ = [
    "Qwen4ExpBackend",
    "DEFAULT_MAX_SEQ_LEN",
    "DEFAULT_PREFILL_CHUNK",
]