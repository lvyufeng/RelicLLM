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
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from relicllm.api import (
    BackendCapabilities,
    ConfigurationError,
    GenerationRequest,
)

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
        self._request_lock = threading.RLock()
        self._distributed = False
        self._details: dict[str, Any] = {}

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
            "context": f"{self._max_seq_len} positions",
            "prefill": f"one forward a {self._options.prefill_chunk}-token chunk",
            "tensor_parallel": (
                f"world {self._world}, a per-layer all-reduce" if self._world > 1 else "one rank"
            ),
            "cancellation": "per-step broadcast; not inside a prefill chunk",
        }

    def _open_tokenizer(self) -> Any:
        """The checkpoint's own tokenizer, out of the directory the release ships.

        A conditional-generation wrapper ships the language tokenizer beside ``config.json``; the
        template is its own (``chat_template.jinja``) and a prompt rendered any other way is a prompt
        the model was not trained on.
        """
        path = self._tokenizer_path or self._checkpoint_dir
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover - the repo's requirements carry it
            raise ConfigurationError(
                "serving a Qwen3.8-Flash-Next checkpoint needs `transformers` for its tokenizer"
            ) from exc
        try:
            return AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(
                f"no tokenizer could be read from {path}: {exc}; pass --tokenizer-path"
            ) from exc

    def _say(self, message: str) -> None:
        if self._world <= 1 or self._rank == 0:
            print(f"[qwen4_exp] {message}", flush=True)

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
            "qwen4_exp_context_positions": float(self._max_seq_len),
            "qwen4_exp_world": float(self._world),
        }

    # -------------------------------------------------------------------- requests

    def _eos_tokens(self) -> set[int]:
        """The ids that end a turn, from the config and the tokenizer's own end-of-text.

        Both sources are read because a run that stopped on one and not the other would run every
        answer to its budget. There is no default and no fallback: a checkpoint that names neither is
        one whose answers cannot end, and a served request against it should say so rather than emit
        its whole budget.
        """
        found: set[int] = set()
        text = getattr(self._config, "text_config", None)
        ids = getattr(text, "eos_token_id", None) if text is not None else None
        if isinstance(ids, int):
            found.add(int(ids))
        elif ids:
            found.update(int(token) for token in ids)
        for candidate in (
            getattr(self._tokenizer, "eos_token_id", None),
            getattr(self._tokenizer, "eos_token_ids", None),
        ):
            if isinstance(candidate, int):
                found.add(int(candidate))
            elif candidate:
                found.update(int(token) for token in candidate)
        if not found:
            raise ConfigurationError(
                "this checkpoint names no end-of-turn token, so a request would run to its budget; "
                "send a prompt_tokens request with an explicit max_tokens"
            )
        return found

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
        tokens, stats = generate(
            self._model,
            self._ctx,
            torch.tensor([list(prompt_ids)], dtype=torch.long),
            max_new_tokens=budget,
            chunk_size=self._options.prefill_chunk,
            eos_token_ids=tuple(sorted(self._eos_tokens())),
            on_token=on_token,
            on_step=self._step_sync(request.request_id, stop),
        )
        return _Generation.from_stats(tokens, stats)

    def _run_payload(self, payload: Mapping[str, Any]) -> None:
        """Run the same loop rank 0 is running, for the collective's sake and not for its answer.

        The prompt ids and the budget are the ones rank 0 tokenized and resolved, and the answer is
        discarded: the tokens agree across ranks because the logits do, so only rank 0's leave.
        """
        from relicllm.models.qwen4_exp.runtime import generate

        prompt_ids = [int(token) for token in payload["prompt_ids"]]
        generate(
            self._model,
            self._ctx,
            torch.tensor([prompt_ids], dtype=torch.long),
            max_new_tokens=int(payload["max_new_tokens"]),
            chunk_size=self._options.prefill_chunk,
            eos_token_ids=tuple(sorted(self._eos_tokens())),
            on_step=self._step_sync(str(payload["request_id"])),
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