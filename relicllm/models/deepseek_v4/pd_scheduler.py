"""DeepSeek-V4's PD-mode phase policy and the execution facade that carries it.

This is the V4 half of what used to be one file. The *queue* left for
:mod:`relicllm.scheduler` -- it was a scheduler with no callers and is now a scheduler with five, the
model-agnostic one. What stays here is what is V4's alone: the phase policy that pins CPU sets, OMP
thread counts, NUMA nodes and the INT8 attention variant *by phase* for the host-resident-expert PD
run, and the facade that hands the model its chunked-prefill size and phase hook.

Why the split falls here, and not at "generic vs specific":

* The queue is not V4's. Four other runtimes reach the engine one request at a time and had nothing;
  the scheduler gives them the one queue too.
* The phase policy *is* V4's. ``apply_phase_resources`` reads ``DEEPSEEK_PD_*`` and pins the process
  to cores -- machinery for a run that puts experts on the host, which no other runtime in this tree
  does. Lifting it into the shared scheduler would have made the shared scheduler a V4 module in
  disguise, which is the failure the lift existed to avoid.

``relicllm/scheduler/core.py`` reads none of the ``DEEPSEEK_PD_*`` names;
``tests/test_scheduler_core.py`` scans the shared package to keep it that way.
"""

from __future__ import annotations

import os
import signal
from dataclasses import dataclass
from typing import Any, Callable, Iterator, List, Optional


_ATTN_INT8_SUFFIXES = (
    "WQ_A_INT8",
    "WQ_B_INT8",
    "WKV_INT8",
    "WO_A_INT8",
    "WO_B_INT8",
    "INDEXER_WQ_B_INT8",
)

_VALID_PHASES = ("prefill", "decode")


@dataclass
class PDExecutionConfig:
    """The phase policy and chunk size a PD run executes with.

    ``phase_policy`` is the V4 resource pinning above, or ``None`` when the run has not asked for it
    (every run that has not set a ``DEEPSEEK_PD_*`` variable). ``prefill_chunk_tokens`` is the
    chunked-prefill size, forwarded to the generation call under that name.
    """

    phase_policy: Optional["PDPhasePolicy"] = None
    prefill_chunk_tokens: int = 0

    @property
    def phase_callback(self) -> Optional[Callable[[str], None]]:
        if self.phase_policy is None or not self.phase_policy.has_phase_overrides():
            return None
        return self.phase_policy.apply_phase_resources

    @classmethod
    def from_env(cls, phase_policy: Optional["PDPhasePolicy"] = None) -> "PDExecutionConfig":
        prefill_chunk_tokens = int(os.getenv("DEEPSEEK_SERVING_PREFILL_CHUNK_TOKENS", "0") or "0")
        return cls(phase_policy=phase_policy, prefill_chunk_tokens=prefill_chunk_tokens)


class PDExecutionFacade:
    """Runs a model's generation with the PD run's chunk size and phase hook attached.

    A thin adapter over one of the model's two generation entry points (serial or streamed): it is
    the single place the ``prefill_chunk_tokens`` keyword and the phase callback are added, so the
    two call paths cannot drift in which they pass.
    """

    def __init__(self, generate_fn: Callable, generate_stream_fn: Callable, config: PDExecutionConfig) -> None:
        self._generate_fn = generate_fn
        self._generate_stream_fn = generate_stream_fn
        self.config = config

    @classmethod
    def from_env(
        cls,
        generate_fn: Callable,
        generate_stream_fn: Callable,
        phase_policy: Optional["PDPhasePolicy"] = None,
    ) -> "PDExecutionFacade":
        return cls(generate_fn, generate_stream_fn, PDExecutionConfig.from_env(phase_policy))

    def run(
        self,
        model: Any,
        prompt_tokens: List[List[int]],
        max_new_tokens: int,
        eos_id: int,
        temperature: float,
        **generate_kwargs,
    ):
        return self._call_with_kwargs(
            self._generate_fn,
            model,
            prompt_tokens,
            max_new_tokens,
            eos_id,
            temperature,
            generate_kwargs,
        )

    def stream(
        self,
        model: Any,
        prompt_tokens: List[List[int]],
        max_new_tokens: int,
        eos_id: int,
        temperature: float,
        **generate_kwargs,
    ) -> Iterator[dict[str, Any]]:
        yield from self._call_with_kwargs(
            self._generate_stream_fn,
            model,
            prompt_tokens,
            max_new_tokens,
            eos_id,
            temperature,
            generate_kwargs,
        )

    def _call_with_kwargs(
        self,
        generate_fn: Callable,
        model: Any,
        prompt_tokens: List[List[int]],
        max_new_tokens: int,
        eos_id: int,
        temperature: float,
        extra_kwargs: dict[str, Any],
    ):
        kwargs: dict[str, Any] = dict(extra_kwargs)
        kwargs["prefill_chunk_tokens"] = self.config.prefill_chunk_tokens
        phase_callback = self.config.phase_callback
        if phase_callback is not None:
            kwargs["phase_callback"] = phase_callback
        return generate_fn(model, prompt_tokens, max_new_tokens, eos_id, temperature, **kwargs)


class PDPhasePolicy:
    """V4's per-phase resource policy, read from ``DEEPSEEK_PD_*``.

    Not a scheduler: it holds no queue and admits nothing. The one scheduler is
    :class:`relicllm.scheduler.Scheduler`; the model's generation loop calls
    :meth:`apply_phase_resources` when the phase changes (through the facade's ``phase_callback``),
    and this decides what a phase change does to the host -- the INT8 attention variant in the
    environment, the server's run/pause, the CPU set, the OMP thread count and the NUMA node. All of
    it is off unless a ``DEEPSEEK_PD_*`` variable is set, which is why a run without any behaves
    exactly as one that never had this class.
    """

    def __init__(self) -> None:
        self._current_phase: Optional[str] = None
        self._current_runtime_threads: Optional[int] = None
        self._pause_server_during_prefill = os.getenv("DEEPSEEK_PD_PAUSE_SERVER_DURING_PREFILL", "0").lower() in {"1", "true", "yes"}
        rank = int(os.getenv("RANK", "0"))
        server_pid_env = os.getenv("DEEPSEEK_PD_SERVER_PID") if rank == 0 else None
        try:
            self._server_pid: Optional[int] = int(server_pid_env) if server_pid_env else None
        except ValueError:
            self._server_pid = None
        self._server_paused: Optional[bool] = None

    def has_phase_overrides(self) -> bool:
        return (
            any(os.getenv(f"DEEPSEEK_PD_{phase.upper()}_{suffix}") for phase in _VALID_PHASES for suffix in ("CPUS", "OMP_THREADS", "NUMA_NODE"))
            or os.getenv("DEEPSEEK_PD_PAUSE_SERVER_DURING_PREFILL", "0").lower() in {"1", "true", "yes"}
            or os.getenv("DEEPSEEK_PD_PHASE_AUTO_SELECT", "0").lower() in {"1", "true", "yes"}
        )

    def apply_phase_resources(self, phase: str) -> None:
        if phase not in _VALID_PHASES:
            raise ValueError(f"unknown phase: {phase!r}")
        os.environ["DEEPSEEK_PD_ACTIVE_PHASE"] = phase
        if self._current_phase == phase:
            return
        self._current_phase = phase
        self._apply_phase_attention_env(phase)
        self._set_server_paused(phase == "prefill")

        cpus_env = os.getenv(f"DEEPSEEK_PD_{phase.upper()}_CPUS")
        if cpus_env and hasattr(os, "sched_setaffinity"):
            cpus = _parse_cpu_list(cpus_env)
            if cpus:
                try:
                    os.sched_setaffinity(0, set(cpus))
                except OSError:
                    pass

        omp_env = os.getenv(f"DEEPSEEK_PD_{phase.upper()}_OMP_THREADS")
        if omp_env:
            try:
                self._current_runtime_threads = max(1, int(omp_env))
                os.environ["OMP_NUM_THREADS"] = str(self._current_runtime_threads)
                try:
                    import relicllm.components.moe.cpu_backend as cpu_routed_backend
                    cpu_routed_backend.configure_cpu_routed_runtime(omp_threads=self._current_runtime_threads)
                except Exception:
                    pass
            except ValueError:
                pass

        numa_env = os.getenv(f"DEEPSEEK_PD_{phase.upper()}_NUMA_NODE")
        if numa_env:
            _try_apply_numa_node(numa_env)

    def _apply_phase_attention_env(self, phase: str) -> None:
        for suffix in _ATTN_INT8_SUFFIXES:
            phase_key = f"DEEPSEEK_PD_{phase.upper()}_{suffix}"
            if phase_key in os.environ:
                os.environ[f"DEEPSEEK_ACTIVE_{suffix}"] = os.environ[phase_key]
            else:
                os.environ.pop(f"DEEPSEEK_ACTIVE_{suffix}", None)

    def _set_server_paused(self, paused: bool) -> None:
        if not self._pause_server_during_prefill or self._server_pid is None:
            return
        if self._server_paused is paused:
            return
        try:
            os.kill(self._server_pid, signal.SIGSTOP if paused else signal.SIGCONT)
            self._server_paused = paused
        except ProcessLookupError:
            self._server_pid = None
        except OSError:
            pass


def _parse_cpu_list(spec: str) -> List[int]:
    cpus: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            try:
                cpus.extend(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        else:
            try:
                cpus.append(int(part))
            except ValueError:
                continue
    return sorted(set(cpus))


def _try_apply_numa_node(spec: str) -> None:
    try:
        nodes = [int(x) for x in spec.split(",") if x.strip()]
    except ValueError:
        return
    if not nodes:
        return
    try:
        import ctypes
        libnuma = ctypes.CDLL("libnuma.so.1", use_errno=True)
    except OSError:
        return
    try:
        libnuma.numa_run_on_node.restype = ctypes.c_int
        libnuma.numa_run_on_node.argtypes = [ctypes.c_int]
        libnuma.numa_run_on_node(nodes[0])
    except Exception:
        return


def run_single_request(
    generate_fn: Callable,
    model,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float,
    phase_policy: Optional[PDPhasePolicy] = None,
    **generate_kwargs,
):
    prefill_chunk_tokens = int(generate_kwargs.pop("prefill_chunk_tokens", 0) or 0)
    facade = PDExecutionFacade(
        generate_fn,
        lambda *args, **kwargs: iter(()),
        PDExecutionConfig(phase_policy=phase_policy or PDPhasePolicy(), prefill_chunk_tokens=prefill_chunk_tokens),
    )
    if generate_kwargs:
        return facade._call_with_kwargs(
            generate_fn,
            model,
            prompt_tokens,
            max_new_tokens,
            eos_id,
            temperature,
            generate_kwargs,
        )
    return facade.run(model, prompt_tokens, max_new_tokens, eos_id, temperature)


def run_single_stream_request(
    generate_stream_fn: Callable,
    model,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    temperature: float,
    phase_policy: Optional[PDPhasePolicy] = None,
    **generate_kwargs,
):
    prefill_chunk_tokens = int(generate_kwargs.pop("prefill_chunk_tokens", 0) or 0)
    facade = PDExecutionFacade(
        lambda *args, **kwargs: None,
        generate_stream_fn,
        PDExecutionConfig(phase_policy=phase_policy or PDPhasePolicy(), prefill_chunk_tokens=prefill_chunk_tokens),
    )
    if generate_kwargs:
        yield from facade._call_with_kwargs(
            generate_stream_fn,
            model,
            prompt_tokens,
            max_new_tokens,
            eos_id,
            temperature,
            generate_kwargs,
        )
        return
    yield from facade.stream(model, prompt_tokens, max_new_tokens, eos_id, temperature)