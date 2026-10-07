from __future__ import annotations

import ast
import threading
import time
from pathlib import Path

import pytest

from relicllm.scheduler import ExecutionPlan, Request, Scheduler


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "relicllm"
SCHEDULER_ROOT = PACKAGE_ROOT / "scheduler"

#: The runtime names ``--backend`` accepts, minus the ``auto`` sentinel. The scheduler must not know
#: any of them; a hit is the anti-pattern issue #130 exists to prevent.
ARCHITECTURE_NAMES = ("torch", "v41", "mimo", "xing4", "qwen4_exp")


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_the_scheduler_does_not_import_a_model_package() -> None:
    """No module under ``relicllm/scheduler/`` imports ``relicllm.models``."""
    violations: list[str] = []
    for path in _python_files(SCHEDULER_ROOT):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("relicllm.models"):
                violations.append(f"{path.relative_to(REPO_ROOT)} imports {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("relicllm.models"):
                        violations.append(f"{path.relative_to(REPO_ROOT)} imports {alias.name}")

    assert not violations, "\n".join(violations)


def test_the_scheduler_names_no_architecture() -> None:
    """The scheduler's own source never spells a runtime name.

    The shape of ``tests/test_runtime_capabilities.py``'s anti-drift scan. Matched against string
    literals and identifiers rather than a substring of the whole file, so a docstring may *discuss*
    the rule -- ``core.py`` names the runtimes it was lifted from -- without tripping it.
    """
    violations: list[str] = []
    for path in _python_files(SCHEDULER_ROOT):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            value: str | None = None
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value
            elif isinstance(node, ast.Name):
                value = node.id
            elif isinstance(node, ast.Attribute):
                value = node.attr
            if value is not None and value in ARCHITECTURE_NAMES:
                violations.append(
                    f"{path.relative_to(REPO_ROOT)}:{node.lineno} spells {value!r}"
                )

    assert not violations, "\n".join(violations)


def test_the_scheduler_reads_no_scheduling_policy_from_the_environment() -> None:
    """``os.environ`` is not a policy channel. The only read is the rank, for id spacing."""
    allowed = {"RANK"}
    violations: list[str] = []
    for path in _python_files(SCHEDULER_ROOT):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            key: str | None = None
            if (
                isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)
            ):
                key = node.slice.value
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "getenv"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                key = node.args[0].value
            if key is not None and key not in allowed:
                violations.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno} reads {key}")

    assert not violations, "\n".join(violations)


def test_a_request_is_a_row_of_counters_and_phase_follows_them() -> None:
    request = Request(request_id=0, prompt_tokens=[1, 2, 3], max_new_tokens=8)

    assert request.sent_tokens == 3
    assert request.num_computed_tokens == 0
    assert request.phase == "prefill"
    assert not request.is_prefilled

    request.num_computed_tokens = 3

    assert request.phase == "decode"
    assert request.is_prefilled


def test_admission_hands_requests_out_fifo_and_frees_the_slot_on_exit() -> None:
    scheduler = Scheduler()
    first = scheduler.submit([1], 2)
    second = scheduler.submit([3], 4)

    assert scheduler.pending_count() == 2
    assert scheduler.next_request() is first

    with scheduler.acquire(first) as held:
        assert held is first
        assert scheduler.has_work()

    # The slot is free again and the first request is gone from the queue.
    assert scheduler.pending_count() == 1
    assert scheduler.next_request() is second


def test_the_slot_serializes_two_threads() -> None:
    scheduler = Scheduler()
    order: list[str] = []
    holding = threading.Event()
    release = threading.Event()

    def hold(name: str, wait: bool) -> None:
        request = scheduler.submit([1], 1)
        with scheduler.acquire(request):
            order.append(f"{name}-in")
            if wait:
                holding.set()
                release.wait(timeout=5)
            order.append(f"{name}-out")

    first = threading.Thread(target=hold, args=("a", True))
    second = threading.Thread(target=hold, args=("b", False))
    first.start()
    assert holding.wait(timeout=5)
    second.start()
    # Give the second thread time to reach acquire and block on it.
    time.sleep(0.05)
    assert order == ["a-in"]

    release.set()
    first.join(timeout=5)
    second.join(timeout=5)

    assert order == ["a-in", "a-out", "b-in", "b-out"]


def test_a_bounded_wait_returns_none_rather_than_blocking() -> None:
    scheduler = Scheduler(gate=threading.Lock())
    holder = scheduler.submit([1], 1)
    waiter = scheduler.submit([2], 1)

    with scheduler.acquire(holder):
        assert scheduler.acquire(waiter, timeout=0.01) is None


def test_the_plan_is_passed_through_without_the_scheduler_reading_it() -> None:
    phases: list[str] = []
    plan = ExecutionPlan(prefill_chunk_tokens=512, phase_callback=phases.append)

    assert plan.generation_kwargs() == {
        "prefill_chunk_tokens": 512,
        "phase_callback": plan.phase_callback,
    }
    # And the callback is only carried, never invoked by the scheduler.
    assert phases == []
    assert ExecutionPlan().generation_kwargs() == {"prefill_chunk_tokens": 0}