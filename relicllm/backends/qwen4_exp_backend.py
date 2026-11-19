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
import queue
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch

from relicllm.api import (
    BackendCapabilities,
    ConfigurationError,
    GenerationRequest,
    RequestCancelledError,
    TokenEvent,
    UnsupportedFeatureError,
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


#: The speculative methods this adapter can serve, and the only one it implements.
#:
#: ``mtp`` is the checkpoint's own multi-token-prediction head driven by the *captured* lookahead
#: loop in ``models/qwen4_exp/runtime.py``: a K-row trunk verify inside one CUDA graph, a recorded
#: head catch-up, and a recorded one-row tail at the native boundary. It is the only speculative
#: decoder here, so another method is refused by name rather than silently decoding plainly.
_SPECULATIVE_METHODS: tuple[str, ...] = ("mtp",)

#: Draft depths that passed a complete 128-token greedy-parity run on the eight-4090 host.
#:
#: Depth is not a free knob on this checkpoint: a five-draft round changed the emitted token at
#: index 67 (every rank agreed, so it is arithmetic rather than a rank disagreement) while identical
#: tokens at 3 and 4 drafts held at 8K and at the full 262144 positions. A depth outside this set is
#: refused for that reason, and widening it needs a new complete-output comparison first.
_SPECULATIVE_DEPTHS: tuple[int, ...] = (3, 4)

#: The environment switch that makes the K-row verify arithmetic reproduce decode's.
#:
#: ``models/qwen4_exp/runtime.py`` reads it to run dense, convolution, reduction and QSA work per row;
#: with it off a verify round is a batched forward whose last bits differ from plain decode, so the
#: captured path would not be the token-identical answer the plain loop gives. It is required rather
#: than set here, because turning it on changes the arithmetic of every forward this process runs.
_STRICT_VERIFY_ENV = "POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT"


def _native_positions(config: Any) -> int:
    """The checkpoint's own position ceiling: prompt plus answer may not exceed it.

    The released config nests the language tower under ``text_config``, and the same number is in
    either place. Zero means the config did not say, which the speculative path refuses rather than
    guessing a bound from ``--max-model-len`` -- those are different numbers here, and the arena is
    sized from the checkpoint's.
    """
    for holder in (getattr(config, "text_config", None), config):
        value = getattr(holder, "max_position_embeddings", None) if holder is not None else None
        if value:
            return int(value)
    return 0


def _decode_seconds(stats: Mapping[str, float]) -> float:
    """The decode interval a caller waited through, for the loop's accounting.

    A plain loop's ``decode_s`` already includes its own graph work. The captured lookahead loop
    reports the steady interval separately from graph construction and from the resource cleanup it
    defers until after the last token, and both are time the request held the turn -- so they are
    charged here rather than dropped. Folding cleanup into the denominator is also what keeps a
    reported rate from improving merely because work moved past the last token.
    """
    if "trunk_steps" not in stats:
        return float(stats.get("decode_s", 0.0))
    including_cleanup = stats.get("decode_s_including_cleanup")
    if including_cleanup is not None:
        return float(including_cleanup) + float(stats.get("graph_setup_s", 0.0))
    wall = stats.get("decode_wall_s")
    if wall is None:
        wall = float(stats.get("decode_s", 0.0)) + float(stats.get("graph_setup_s", 0.0))
    return float(wall) + float(stats.get("request_cleanup_s", 0.0) or 0.0)


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


class _MTPEventQueue(queue.Queue):
    """A missing SSE reader cannot strand rank 0 inside a blocking token callback."""
    def __init__(self, adapter, request_id: str, maxsize: int):
        super().__init__(maxsize=maxsize)
        self.adapter, self.request_id = adapter, request_id
        self._reader_closed = self._aborted = False

    def close_reader(self) -> None:
        self._reader_closed = self._aborted = True

    def _offer_terminal(self, item) -> None:
        while True:
            try:
                queue.Queue.put(self, item, block=False)
                return
            except queue.Full:
                try:
                    queue.Queue.get(self, block=False)
                except queue.Empty:
                    pass

    def put(self, item, block=True, timeout=None):
        cancelled = (self._reader_closed or self.adapter._mtp_closing
                     or self.adapter._is_cancelled(self.request_id))
        self._aborted |= cancelled
        terminal = item is None or (isinstance(item, TokenEvent) and item.finish_reason == "cancelled")
        if self._aborted and terminal:
            self._offer_terminal(item)
            return
        if cancelled:
            raise RequestCancelledError("the MTP event stream has no active reader")
        if not block:
            return queue.Queue.put(self, item, block=False)
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            wait = 0.05 if deadline is None else max(0.0, min(0.05, deadline - time.monotonic()))
            try:
                return queue.Queue.put(self, item, block=True, timeout=wait)
            except queue.Full:
                if (self._reader_closed or self.adapter._mtp_closing
                        or self.adapter._is_cancelled(self.request_id)):
                    self._aborted = True
                    if terminal:
                        self._offer_terminal(item)
                        return
                    raise RequestCancelledError("the MTP event stream was cancelled while backpressured")
                if deadline is not None and time.monotonic() >= deadline:
                    raise


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
        #: Drafts a round of the captured MTP loop proposes; zero means plain greedy decode.
        self._speculative_tokens = self._resolve_speculative(args)
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
        self._graph_session: Any = None
        self._mtp_active = 0
        self._mtp_closing = False
        self._mtp_teardown_started = False
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

    # ------------------------------------------------------------------ speculation

    @staticmethod
    def _resolve_speculative(args: Any) -> int:
        """The draft depth this launch asks for, or zero for plain greedy decode.

        The launch opts in with ``--speculative-method mtp`` plus ``--speculative-tokens``, which are
        the two global arguments every runtime receives; nothing is inferred from the environment.
        Three failures are answered here, before a checkpoint is read, because each of them is a
        launch this adapter cannot serve rather than a request it can refuse later: another method
        (nothing else is implemented), a depth outside the ones that passed complete-output parity,
        and a depth named without a method (which would otherwise be silently ignored).
        """
        method = getattr(args, "speculative_method", None)
        tokens = int(getattr(args, "speculative_tokens", 1) or 1)
        if not method:
            if tokens != 1:
                raise ConfigurationError(
                    f"speculative_tokens={tokens} needs speculative_method='mtp'; this backend "
                    "decodes plainly when no method is named"
                )
            return 0
        if str(method) not in _SPECULATIVE_METHODS:
            raise UnsupportedFeatureError(
                "backend='qwen4_exp' has only the captured MTP speculative decoder "
                f"({', '.join(_SPECULATIVE_METHODS)}); speculative_method={method!r} is not supported"
            )
        if tokens not in _SPECULATIVE_DEPTHS:
            raise UnsupportedFeatureError(
                f"backend='qwen4_exp' serves MTP at {_SPECULATIVE_DEPTHS} drafts a round only: "
                f"speculative_tokens={tokens} was measured to change the emitted tokens on this "
                "checkpoint, and a depth this adapter cannot prove token-identical is refused "
                "rather than run"
            )
        return tokens

    @property
    def _mtp_enabled(self) -> bool:
        """Whether this launch serves the captured MTP loop instead of plain greedy decode."""
        return self._speculative_tokens > 0

    @property
    def _native_limit(self) -> int:
        """The loaded checkpoint's position ceiling, or zero before a model is loaded."""
        return _native_positions(self._config) if self._config is not None else 0

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
            # Before the checkpoint is read: a launch that cannot run the captured loop should fail
            # in seconds rather than after the minutes a shard load takes.
            self._check_speculative_placement()
            if self._loader is not None:
                # An injected loader is a test's or an embedding application's model, and it comes
                # with its own tokenizer: nothing here reaches for the checkpoint, so a run with one
                # never needs a checkpoint directory on disk.
                self._model, self._config = self._loader(self.args, self._options)
                self._check_speculative_model()
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
            expert_fused_fp8=os.environ.get("POCKETLLM_QWEN4_NVFP4_EXPERTS", "0") == "1",
            expert_reserve_bytes=options.expert_reserve,
            # The speculative MTP loop needs the head as part of this build: its first draft is
            # produced from the prefill's last streams, and a head added later would need a second
            # pass over the checkpoint to acquire the same weights.
            mtp=self._mtp_enabled,
        )
        self._check_speculative_model()
        self._build_details()
        if self._tokenizer is None:
            self._tokenizer = self._open_tokenizer()
        self._publish_ready()

    def _stream_event_queue(self, request_id: str, depth: int):
        return (_MTPEventQueue(self, request_id, depth) if self._mtp_enabled
                else queue.Queue(maxsize=depth))

    def health(self):
        status = super().health()
        return replace(status, status="stopped", ready=False) if self._mtp_closing else status

    def close(self) -> None:
        """Quiesce active MTP first; never free a recording or send shutdown during its replay."""
        with self._state_lock:
            self._mtp_closing = True
            self._cancelled |= self._active_requests
            active = self._mtp_active
        if self._graph_session is not None:
            self._graph_session.close()  # Refuses checkout; an active lease releases itself later.
        if not active:
            self._close_quiescent()

    def _close_quiescent(self) -> None:
        with self._state_lock:
            if self._mtp_teardown_started:
                return
            self._mtp_teardown_started = True
        try:
            if self._graph_session is not None:
                self._graph_session.close()
                self._graph_session = None
        finally:
            super().close()

    def _check_speculative_placement(self) -> None:
        """Refuse a launch whose shape the captured MTP loop cannot run in.

        Four requirements, each of them structural rather than tunable. The recorded verify is a CUDA
        graph, so it needs a card. It is a *tensor-parallel* recording -- one row a rank plus the
        collectives that make the row's answer agree -- so a pipeline stage has nothing to record.
        The loop owns the single mutable cache, so a batch width above one has no second sequence to
        speculate for. And the K-row verify only reproduces decode's arithmetic when the strict
        row-invariant switches are on, which this process cannot decide for a launch that may be
        serving other runtimes: an operator sets them on the command line.
        """
        if not self._mtp_enabled:
            return
        ctx = self._ctx
        if getattr(self._device, "type", "") != "cuda":
            raise UnsupportedFeatureError(
                "the captured MTP loop records CUDA graphs; backend='qwen4_exp' needs "
                f"device='cuda' for speculative_method='mtp' and resolved {self._device}"
            )
        pipeline = int(getattr(ctx, "pp_size", 1) or 1)
        if pipeline != 1 or int(getattr(ctx, "tp_world_size", 1) or 1) != int(self._world):
            raise UnsupportedFeatureError(
                "speculative_method='mtp' is a pure tensor-parallel path: its verify graph holds "
                f"this rank's row of one round, and this launch resolved {pipeline} pipeline "
                f"stage(s) over a world of {self._world}"
            )
        batch = int(getattr(self.args, "max_batch_size", 1) or 1)
        if batch != 1:
            raise UnsupportedFeatureError(
                "the captured MTP loop owns the process's one mutable cache; "
                f"max_batch_size={batch} cannot be served with speculative_method='mtp'"
            )
        required = (_STRICT_VERIFY_ENV, "POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT")
        missing = [flag for flag in required if os.environ.get(flag, "0") != "1"]
        if missing:
            raise ConfigurationError(
                "speculative_method='mtp' requires the qualified decode-shaped verifier and "
                "FP32 indexer-score geometry; set " + ", ".join(f"{flag}=1" for flag in missing)
            )

    def _check_speculative_model(self) -> None:
        """Refuse a loaded model that cannot seed the recorded loop.

        The head is what produces a round's drafts: without it there is nothing to verify, and the
        runtime's own loader only builds one when asked. The position ceiling is what the arena is
        clamped to, so a checkpoint that does not state one leaves the loop unable to bound a request
        -- and a wrong guess there would be a failed request rather than a slow one.
        """
        if not self._mtp_enabled:
            return
        if not getattr(self._model, "has_mtp", False):
            raise ConfigurationError(
                "speculative_method='mtp' needs the checkpoint's MTP head, which this model was "
                "built without; the head is loaded with the tree and cannot be added per request"
            )
        if self._native_limit <= 0:
            raise ConfigurationError(
                "speculative_method='mtp' sizes its arena from the checkpoint's "
                "max_position_embeddings, which this config does not state"
            )

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
            "speculative": (
                f"off; plain greedy decode" if not self._mtp_enabled
                else (
                    f"captured MTP lookahead, {self._speculative_tokens} drafts a round, greedy and "
                    f"token-identical to plain decode; experimental, needs "
                    f"{_STRICT_VERIFY_ENV}=1"
                )
            ),
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

    def _host_bank(self) -> Any:
        """This run's ``HostExpertMoE``, or None where the experts are device-resident.

        ``moe`` belongs to the decoder layer and the model keeps those in ``layers``
        (``models/qwen4_exp/model.py``), so the counters sit one level below where a model-level
        lookup would find them. ``ShardedMoE`` wraps the bank when the world size is above one and
        is unwrapped here too. Every layer shares one bank, so the first layer that has one answers
        for the run.
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
        """The routed-expert staging counters, or nothing where the experts are resident.

        These are the numbers that tell a *resident* bank from a *cached* one, and a decode rate
        cannot be read without them: a deployment that holds every expert it owns stops staging
        after the first pass and one that does not keeps paying the PCIe cost every step. Two runs
        with the same tokens a second and different ``stage_bytes`` are not the same result -- one
        is bounded by the card and the other by the link, and the next request's rate depends on
        which. Nothing else publishes them.
        """
        bank = self._host_bank()
        if bank is None:
            return {}
        return {
            # `_total` is the convention for a counter the server does not own; the rest are gauges.
            "qwen4_exp_expert_stage_bytes_total": float(bank.stage_bytes),
            "qwen4_exp_expert_stage_calls_total": float(bank.stage_calls),
            "qwen4_exp_expert_cache_hits_total": float(bank.cache_hits),
            "qwen4_exp_expert_cache_capacity": float(getattr(bank, "cache_capacity", 0) or 0),
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
            **self._expert_stage_metrics(),
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

    def _budget(self, prompt_ids: Sequence[int], request: GenerationRequest) -> int:
        """How many tokens this request may generate; the checkpoint's ceiling on the MTP path.

        A plain request keeps the base arithmetic, whose ceiling is ``--max-model-len``. The captured
        loop does not size anything from that flag: its arena is the checkpoint's own position
        ceiling, clamped, and its verify round cannot truncate itself -- so a request whose prompt
        plus budget crosses ``max_position_embeddings`` is refused here, before any rank is dispatched
        into the collective, rather than failing inside a graph with the group already in it.
        """
        if not self._mtp_enabled:
            return super()._budget(prompt_ids, request)
        self._ensure_loaded()
        native = self._native_limit
        prompt = len(prompt_ids)
        total = prompt + int(request.sampling_params.token_budget(native - prompt))
        if prompt >= native or total > native:
            raise ConfigurationError(
                f"the captured MTP path decodes inside the checkpoint's {native} native positions "
                f"and this request asks for {prompt} prompt tokens plus "
                f"{total - prompt} generated ones; shorten the prompt or the budget"
            )
        return int(request.sampling_params.token_budget(native - prompt))

    def _checkpoint_eos_token_id(self) -> Any:
        """Qwen4-Exp's end-of-turn id lives on the language tower under its text config."""
        text = getattr(self._config, "text_config", None)
        return getattr(text, "eos_token_id", None) if text is not None else None

    def _loop(
        self, prompt_ids: Sequence[int], budget: int, request: GenerationRequest, *,
        on_token: Callable[[int], None] | None, stop: Callable[[], bool] | None = None,
    ) -> Any:
        if not self._mtp_enabled:
            return self._loop_active(prompt_ids, budget, request, on_token=on_token, stop=stop)
        # Bracket the broadcast as well as generation: close cannot tear down the group in the
        # short gap between dispatching a request and entering its captured MTP owner.
        with self._state_lock:
            if self._mtp_closing:
                raise RuntimeError("backend is closing")
            self._mtp_active += 1
        try:
            return self._loop_active(prompt_ids, budget, request, on_token=on_token, stop=stop)
        finally:
            with self._state_lock:
                self._mtp_active -= 1
                close_after_request = self._mtp_closing and self._mtp_active == 0
            if close_after_request:
                self._close_quiescent()

    def _loop_active(
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
        if self._mtp_enabled:
            # One selector for both sides of the process group: the worker runs this same driver in
            # `_run_payload`, off the same launch arguments, so both ranks enter the same recording.
            tokens, stats = self._run_mtp(
                prompt_ids,
                budget,
                request.request_id,
                on_token=self._rank0_callback(on_token, request.request_id),
                stop=stop,
            )
        else:
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

    def _run_mtp(
        self, prompt_ids: Sequence[int], budget: int, request_id: str, *,
        on_token: Callable[[int], None] | None, stop: Callable[[], bool] | None,
    ) -> tuple[list[int], dict[str, float]]:
        with self._state_lock:
            if self._mtp_closing:
                raise RuntimeError("backend is closing")
            self._mtp_active += 1
        try:
            return self._run_mtp_active(prompt_ids, budget, request_id,
                                        on_token=on_token, stop=stop)
        finally:
            with self._state_lock:
                self._mtp_active -= 1
                close_after_request = self._mtp_closing and self._mtp_active == 0
            if close_after_request:
                self._close_quiescent()

    def _run_mtp_active(
        self,
        prompt_ids: Sequence[int],
        budget: int,
        request_id: str,
        *,
        on_token: Callable[[int], None] | None,
        stop: Callable[[], bool] | None,
    ) -> tuple[list[int], dict[str, float]]:
        """One request through the captured lookahead loop, run identically on every rank.

        Called by rank 0 from :meth:`_loop` and by every worker from :meth:`_run_payload`, which is
        what makes the mode one decision rather than two: the rank that answers and the ranks that
        must be inside its collectives enter the same recording with the same prompt and budget.

        The predicate handed to ``on_step`` is never ``None``. The loop's end-of-sequence exit is a
        *flag* on this rank, and a rank that honoured it alone would walk out of a step its peers are
        still inside -- the 600-second broadcast timeout ``generate``'s ``local_eos`` paragraph
        records. Folding the flag into the broadcast predicate is what turns it into an exit every
        rank takes together, and it is why the MTP path folds it in even when there is no stream
        whose stop string it also has to carry.
        """
        from relicllm.models.qwen4_exp.runtime import generate_mtp_draft_lookahead

        local_eos = [False]
        session = None
        request_width = min(self._native_limit, len(prompt_ids) + budget + 2 * self._speculative_tokens + 4)
        has_verify_room = len(prompt_ids) + 2 * (self._speculative_tokens + 1) <= request_width
        if (budget > 1 and has_verify_room
                and os.environ.get("POCKETLLM_QWEN4_MTP_SESSION", "0") == "1"):
            from relicllm.models.qwen4_exp.graph_session import GraphSessionPool
            if self._graph_session is None:
                self._graph_session = GraphSessionPool(model=self._model, capacity=1)
            session = self._graph_session

        def halt() -> bool:
            # Only rank 0 evaluates `stop`; `_step_sync` broadcasts the answer to the others.
            return self._mtp_closing or local_eos[0] or (stop() if stop is not None else False)

        return generate_mtp_draft_lookahead(
            self._model,
            self._ctx,
            torch.tensor([list(prompt_ids)], dtype=torch.long),
            max_new_tokens=int(budget),
            chunk_size=self._options.prefill_chunk,
            eos_token_ids=tuple(sorted(self._eos_tokens())),
            on_token=on_token,
            draft_tokens=self._speculative_tokens,
            verify_graph=True,
            on_step=self._step_sync(request_id, halt),
            local_eos=local_eos,
            graph_session=session,
        )

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
        if self._mtp_enabled:
            # The same driver rank 0 entered, off the same launch arguments: this rank's answer is
            # discarded, but its rows have to be in the verify graph's grid for the collectives to
            # match. No token callback and no stream stop -- neither exists on a worker.
            self._run_mtp(
                prompt_ids,
                int(payload["max_new_tokens"]),
                str(payload["request_id"]),
                on_token=None,
                stop=None,
            )
            return
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
            # The captured lookahead loop reports its setup and its deferred cleanup separately from
            # the steady interval; they are time this request held the turn and are charged here.
            decode_seconds=_decode_seconds(stats),
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