"""One request scheduler, for every runtime.

The package holds a single model-agnostic queue: a request is a row of counters, admission is
:meth:`~relicllm.scheduler.core.Scheduler.submit`, and the engine is held by one request at a time.
It is the shared replacement for the per-adapter request lock, and the place the counter-based state
that later phases extend lives.

Nothing here may name an architecture: no module under this package imports ``relicllm.models`` or
spells a runtime's name, and ``tests/test_scheduler_core.py`` scans for both. Per-model difference
reaches the scheduler as a :class:`~relicllm.scheduler.core.ExecutionPlan` and a request's own
counters, never as a branch on which model is being served.
"""

from relicllm.scheduler.core import ExecutionPlan, Request, Scheduler

__all__ = ["ExecutionPlan", "Request", "Scheduler"]