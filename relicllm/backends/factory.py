"""Backend selection and construction."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from relicllm.api import (
    BackendUnavailableError,
    ConfigurationError,
    EngineArgs,
    UnsupportedFeatureError,
)

from . import capabilities, cli_surface
from .capabilities import runtime_capabilities
from .mimo_backend import MimoBackend
from .torch_backend import TorchBackend
from .v41_backend import V41Backend
from .worker import WORKERS, program
from .xing4_backend import Xing4Backend

# The device vocabulary and the probe. The module is a leaf -- stdlib and torch, nothing from this
# package -- so importing it here is free of the half-initialized-`relicllm` hazard that
# `relicllm/api/types.py` has to watch for.
from relicllm.runtime.device import Accelerator, resolve_platform
from relicllm.runtime.device import probe_accelerator as _probe_accelerator


# ---------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------


def _refuse_a_capability_the_runtime_lacks(name: str, args: EngineArgs) -> None:
    """Refuse a request for something the runtime does not declare, before anything loads.

    The one instance so far, and it is the one that motivated the declaration: the CLI accepts
    ``--max-batch-size`` and ``--enable-batching`` on every backend, and on a runtime with no
    scheduler they used to be read by nothing. A width is a request for concurrency; a runtime that
    declares ``supports_batch=False`` cannot deliver it, and the refusal costs a process start
    rather than a model load.

    The width and the flag are checked separately because they are two different asks: a width
    above 1 is a request for rows, and ``--enable-batching`` on its own is a request for the batch
    *path* even at width 1. Only the second is why the flag exists rather than the width alone.
    """
    if runtime_capabilities(name).supports_batch:
        return
    asked: list[str] = []
    if args.max_batch_size > 1:
        asked.append(f"--max-batch-size {args.max_batch_size}")
    if args.enable_batching:
        asked.append("--enable-batching")
    if not asked:
        return
    raise UnsupportedFeatureError(
        f"backend={name!r} declares supports_batch=False: it runs one request at a time, so "
        f"{' and '.join(asked)} asks for concurrency it cannot deliver. Drop "
        f"{'that' if len(asked) == 1 else 'those'}; no runtime here serves more than one request "
        f"at a time"
    )


def _refuse_options_the_runtime_does_not_read(name: str, args: EngineArgs) -> None:
    """Refuse a flag the selected runtime does not read, in the parent, before anything loads.

    The counterpart of the flag generation in :mod:`relicllm.backends.cli_surface`, and the reason
    one flat namespace is safe: every runtime's flags are on the same command line, so
    ``--expert-pool-rows`` is offered to a MiMo launch too. Silently ignoring it is the failure this
    whole surface exists to prevent -- the run is then measured on a lever nobody pulled -- and the
    refusal can be specific about why, because the declarations say who does read it.

    A ``--backend-option`` key of the same name is *not* this check's business: the adapter's own
    decoder refuses an undeclared key with the runtime's own message, and it has to keep doing so for
    a key that will never have a flag.
    """
    unread = cli_surface.unread_options(name, args)
    if not unread:
        return
    flags = ", ".join(cli_surface.cli_name(option) for option in unread)
    readers = cli_surface.readers_of(unread[0])
    if readers:
        where = (
            f"it is {readers[0]}'s" if len(readers) == 1 else f"it belongs to {', '.join(readers)}"
        )
    else:
        where = "no runtime declares it"
    raise ConfigurationError(
        f"backend={name!r} does not read {flags}: {where}. A tuning option that silently does "
        "nothing is how a run ends up measured on the wrong lever"
    )


def _refuse_a_platform_the_runtime_lacks(
    name: str, args: EngineArgs, *, accelerator: Accelerator | None = None
) -> Accelerator:
    """Resolve ``--device`` and refuse a platform the runtime does not declare.

    Two things were true of ``--device`` before this and both were wrong in the same direction. The
    ``auto`` default was *stored*, never read -- nothing in the tree resolved it, so the flag's own
    help text ("``auto`` asks the build") described behaviour that did not exist. And
    ``RuntimeCapabilities.devices`` was a declaration no one consulted, so ``--backend mimo
    --device cpu`` was accepted on a runtime whose own declaration says ``("cuda",)``.

    Both are answered here because this is the only place that has both halves: the requested
    platform lives on ``args`` and the declared set lives on the runtime's capabilities. The
    resolved :class:`Accelerator` is returned rather than probed twice, so a caller that needs the
    device type or the collective asks for the same answer this check was made against.

    An empty ``devices`` is read as *unconstrained* rather than as *nothing*: the field defaults to
    ``()``, and a runtime that has not stated a set has not excluded one.
    """
    accelerator = _probe_accelerator() if accelerator is None else accelerator
    platform = resolve_platform(args.device, accelerator)
    declared = runtime_capabilities(name).devices
    if not declared or platform in declared:
        return accelerator
    raise UnsupportedFeatureError(
        f"backend={name!r} declares devices={declared!r}, so it cannot run on {platform!r}"
        + (
            f", which is what --device auto resolved to on this host"
            if args.device == "auto"
            else ""
        )
        + f". Ask for one of {', '.join(declared)}, or use a runtime that serves {platform!r}"
    )


def select_backend(args: EngineArgs, *, accelerator: Accelerator | None = None) -> str:
    """Select a backend without silently changing an explicit user choice.

    Both questions -- which checkpoint is this, and can this runtime serve it -- are answered from
    ``capabilities.RUNTIMES``. The two halves used to be four predicate functions plus four
    ``_reject_unsupported_*`` bodies, which is two chances per adapter to disagree about which
    checkpoints are its.
    """
    if args.backend != "auto":
        refusal = capabilities.refusal(args.backend, args)
        if refusal:
            raise UnsupportedFeatureError(refusal)
        _refuse_a_capability_the_runtime_lacks(args.backend, args)
        _refuse_a_platform_the_runtime_lacks(args.backend, args, accelerator=accelerator)
        _refuse_options_the_runtime_does_not_read(args.backend, args)
        return args.backend
    # `auto` asks a different question than the explicit path does: not "is this provably not
    # yours" but "does this checkpoint identify you", so a checkpoint presenting no evidence falls
    # through to the generic runtime rather than being routed on the strength of a backend being
    # importable.
    for name in capabilities.AUTO_ORDER:
        if capabilities.identify(name, args).routes_here:
            _refuse_a_capability_the_runtime_lacks(name, args)
            _refuse_a_platform_the_runtime_lacks(name, args, accelerator=accelerator)
            _refuse_options_the_runtime_does_not_read(name, args)
            return name
    raise AssertionError("capabilities.AUTO_ORDER has no fallback")


def create_backend(args: EngineArgs, **injected: Any):
    """Construct the selected adapter.

    ``injected`` is intentionally useful for tests and embedding applications;
    production callers normally only pass ``EngineArgs``.

    ``accelerator=`` is the one injection that is not an adapter: it answers "what does this host
    have", which a test on a CUDA box has to be able to answer differently to reach the Ascend arm.
    """
    selected = select_backend(args, accelerator=injected.get("accelerator"))
    if selected == "torch":
        return TorchBackend(
            args,
            runtime=injected.get("runtime"),
            serving_engine=injected.get("serving_engine"),
            runtime_loader=injected.get("runtime_loader"),
        )
    if selected == "v41":
        def construct(resolved: EngineArgs) -> V41Backend:
            return V41Backend(
                resolved,
                front=injected.get("front"),
                loader=injected.get("loader"),
                tokenizer=injected.get("tokenizer"),
            )

        if _needs_supervision(args, injected):
            def build(resolved: EngineArgs) -> V41Backend:
                # Rank 0 loads inside the rendezvous window.  The environment that names the
                # group belongs to this process only while this call is on the stack, and this
                # adapter reads the group at load time rather than at construction time, so a
                # rank-0 backend handed back unloaded would find the rendezvous gone.
                backend = construct(resolved)
                backend.prepare()
                return backend

            return _supervise_rank_zero(
                args,
                worker="v41",
                build=build,
                torch_rendezvous=True,
            )
        return construct(args)
    if selected == "xing4":
        return Xing4Backend(
            args,
            loader=injected.get("loader"),
            tokenizer=injected.get("tokenizer"),
        )
    if selected == "mimo":
        def construct(resolved: EngineArgs) -> MimoBackend:
            return MimoBackend(
                resolved,
                loader=injected.get("loader"),
                tokenizer=injected.get("tokenizer"),
            )

        if _needs_supervision(args, injected):
            def build(resolved: EngineArgs) -> MimoBackend:
                # Rank 0 loads inside the rendezvous window, for the reason the V4.1 build does:
                # the group is read at load time and the environment that names it belongs to
                # this process only while this call is on the stack.
                backend = construct(resolved)
                backend.prepare()
                return backend

            return _supervise_rank_zero(
                args,
                worker="mimo",
                build=build,
                torch_rendezvous=True,
            )
        return construct(args)
    raise BackendUnavailableError(f"unsupported backend {selected!r}")


def _needs_supervision(args: EngineArgs, injected: Mapping[str, Any]) -> bool:
    """Whether rank 0 should launch the rest of its tensor-parallel group.

    A non-empty ``POCKETLLM_NCCL_ID_PATH`` or a supplied ``nccl_id_path`` means someone
    else owns the ranks -- the supervisor that started this process, or an embedding
    application -- and spawning here would start a second group at the same rendezvous.
    """
    return (
        args.tensor_parallel_size > 1
        and args.tensor_parallel_rank == 0
        and not args.backend_options.get("nccl_id_path")
        and not os.environ.get("POCKETLLM_NCCL_ID_PATH")
        and not injected.get("_supervised_child")
    )


def _supervise_rank_zero(
    args: EngineArgs,
    *,
    worker: str,
    build: Callable[[EngineArgs], Any],
    torch_rendezvous: bool = False,
) -> Any:
    """Launch ranks 1..N-1 and build rank 0 here, under the same rendezvous.

    Rank 0 stays in this process, so anything the supervisor puts in a child's
    environment has to be assembled here instead.  The rendezvous is published two
    ways for exactly that reason: ``backend_options['nccl_id_path']`` is scoped to the
    backend that needs it, and the process environment -- which is the only channel the
    workers have -- is set for the length of ``build`` and then put back.  Leaking the
    environment into the next ``create_backend`` call is what makes a second engine
    conclude it is already supervised, skip spawning, and then block forever in a
    rendezvous waiting for ranks nobody started.

    ``worker`` is a key of :data:`~relicllm.backends.worker.WORKERS`, and it is the whole of
    what a child is told about which runtime it is: the same program runs for all of them and
    the name arrives in the environment, which is the channel the children have anyway.
    """
    if worker not in WORKERS:
        raise BackendUnavailableError(f"no worker program for backend {worker!r}")
    from ..supervisor import TensorParallelConfig, TensorParallelSupervisor

    supervisor = TensorParallelSupervisor(
        TensorParallelConfig(
            # Rank 0 remains in the parent process, while the supervisor owns the remaining
            # actual TP ranks.  Keep the full world size in child environments so NCCL sees
            # the same group as rank 0, including the TP2 case with one child.
            command=(sys.executable, "-c", program()),
            world_size=args.tensor_parallel_size,
            child_ranks=tuple(range(1, args.tensor_parallel_size)),
            env=_worker_env(args, worker),
        )
    )
    supervisor.start()

    nccl_id_path = str(supervisor.nccl_id_path)
    environment = {"POCKETLLM_NCCL_ID_PATH": nccl_id_path}
    if torch_rendezvous:
        # The same variables the supervisor gave the children, resolved the way it
        # resolves them: a caller-supplied address wins, and everything else is a local
        # group.  A rank 0 that met the others anywhere else would be a different group.
        environment.update({
            "MASTER_ADDR": os.environ.get("MASTER_ADDR") or "127.0.0.1",
            "MASTER_PORT": str(supervisor.master_port),
            "RANK": "0",
            "WORLD_SIZE": str(args.tensor_parallel_size),
            "LOCAL_RANK": "0",
        })

    resolved = replace(
        args, backend_options={**args.backend_options, "nccl_id_path": nccl_id_path}
    )
    with _environment(environment):
        try:
            backend = build(resolved)
        except BaseException:
            # The workers are useless without rank 0, and leaving them alive would hold
            # device memory and block a retry.
            supervisor.cleanup()
            raise

    # Store supervisor reference so it can be cleaned up
    backend._supervisor = supervisor

    return backend


@contextmanager
def _environment(values: Mapping[str, str]) -> Iterator[None]:
    """Set ``values`` in the process environment, then put every name back."""
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            _restore_env(name, value)


# EngineArgs fields a worker rank has to agree with rank 0 on.  Fields that are
# per-rank (tensor_parallel_rank, device) or already forwarded explicitly are
# deliberately absent.
#
# `enable_batching` and `max_batch_size` stay for the same reason as the rest: they
# are refused on a runtime that cannot honour them, so a worker rank has to resolve
# them the way rank 0 did or the two would disagree about whether the launch is even
# legal.
_WORKER_SHARED_ARGS = (
    "prefill_chunk_tokens",
    "enable_prefix_caching",
    "attention_window",
    "attention_sink_tokens",
    "speculative_method",
    "speculative_tokens",
    "max_batch_size",
    "enable_batching",
    "model_format",
    "dtype",
    # The launch's placement, which every rank has to agree on: a worker that defaulted the card
    # list would put its own weights on a card rank 0 did not name, and the collective would find
    # out at the first all-reduce rather than at startup.
    "device",
    "device_ids",
)


def _worker_arg_overrides(args: EngineArgs) -> dict[str, Any]:
    """Collect the EngineArgs values a worker rank must match."""
    return {name: getattr(args, name) for name in _WORKER_SHARED_ARGS}


def _worker_env(args: EngineArgs, worker: str) -> dict[str, str]:
    """The environment a worker rank needs to rebuild rank 0's engine.

    Every rank is handed the same variables, whichever runtime it is serving.  The two trailing
    path variables are read by the runtimes that resolve a tokenizer or a config by path -- a rank
    that resolved a different one would load a different model under the same process group --
    and ignored by the ones that read neither.  Selecting them per runtime here would be a second
    place that has to know which runtime reads what, and the first is
    :data:`~relicllm.backends.worker.WORKERS`.

    Both option tiers travel, and they have to. ``POCKETLLM_BACKEND_OPTIONS`` is what the launch
    named outright; ``POCKETLLM_RESOLVED_OPTIONS`` is what its flags settled -- a prefix store's byte
    budget, a prefill width. A rank that defaulted one of those while rank 0 honoured it would
    evict a different prefix or chunk at a different boundary, and the two would disagree about
    where a resumed request starts without either of them knowing.

    ``POCKETLLM_WORKER_BACKEND`` is what tells the one child program which runtime it is; it comes
    from the registry key rather than from ``args.backend``, which may have been ``auto``.
    """
    return {
        "POCKETLLM_WORKER_BACKEND": worker,
        "POCKETLLM_CHECKPOINT": args.checkpoint_dir,
        "POCKETLLM_TP_SIZE": str(args.tensor_parallel_size),
        "POCKETLLM_MAX_MODEL_LEN": str(args.max_model_len or 8192),
        "POCKETLLM_KV_CACHE_DTYPE": str(args.kv_cache_dtype or "auto"),
        "POCKETLLM_BACKEND_OPTIONS": json.dumps(args.backend_options),
        "POCKETLLM_RESOLVED_OPTIONS": json.dumps(args.resolved_options),
        "POCKETLLM_WORKER_ARGS": json.dumps(_worker_arg_overrides(args)),
        "POCKETLLM_CONFIG_PATH": str(args.config_path or ""),
        "POCKETLLM_TOKENIZER_PATH": str(args.tokenizer_path or ""),
    }


def _restore_env(name: str, previous: str | None) -> None:
    """Put ``name`` back the way it was, distinguishing unset from empty."""
    if previous is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = previous

