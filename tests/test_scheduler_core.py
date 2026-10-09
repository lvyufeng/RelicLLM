from __future__ import annotations

import ast
import threading
import time
from pathlib import Path

import pytest

from relicllm.scheduler import AdmissionRefused, ExecutionPlan, KVCapacity, Request, Scheduler


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


# ---------------------------------------------------------------------------------------------
# Admission -- the capacity the run reports, checked before anything queues
# ---------------------------------------------------------------------------------------------


def test_a_request_over_the_capacity_is_refused_before_it_enters_the_queue() -> None:
    """The refusal is outside the queue, which is the whole reason admission is here.

    A request that cannot run must not become one the engine is asked to pick up, and the queue is
    where the engine looks. `pending_count` is the assertion that says so: a refused request is
    absent, not merely unwound afterwards.
    """
    scheduler = Scheduler(capacity=lambda: KVCapacity(positions=8))
    prompt = [1, 2, 3]

    with pytest.raises(AdmissionRefused, match=r"needs 12 positions \(3 prompt tokens and 9 new\)"):
        scheduler.submit(prompt, 9)

    assert scheduler.pending_count() == 0
    assert not scheduler.has_work()

    # The boundary is inclusive: exactly filling the context is answered.
    admitted = scheduler.submit(prompt, 5)
    assert scheduler.pending_count() == 1
    assert admitted.max_new_tokens == 5


def test_a_refusal_names_the_buffer_the_run_reported() -> None:
    """The message is built from the capacity, so a run with an unusual buffer says so."""
    scheduler = Scheduler(
        capacity=lambda: KVCapacity(positions=4, label="this run's context")
    )

    with pytest.raises(AdmissionRefused, match="this run's context were sized at 4"):
        scheduler.submit([1, 2, 3], 2)


def test_no_capacity_provider_means_no_admission_rule() -> None:
    """`None` is "this run does not check", which is what the torch route and the fakes get.

    The check is opt-in per adapter rather than a default every scheduler inherits, so a backend
    whose route never had a context refusal does not acquire one by being handed a scheduler.
    """
    scheduler = Scheduler()

    assert scheduler.capacity() is None
    # Far past any plausible context, and admitted, because nothing here is checking.
    assert scheduler.submit(list(range(10_000)), 10_000).max_new_tokens == 10_000


def test_a_capacity_that_sizes_from_the_request_admits_anything() -> None:
    """The runtime whose cache is built per request is not checked against a configured number.

    `qwen4_exp` is the one: its QSA buffers are `prompt + budget + 1`, so a capacity derived from
    `--max-model-len` would refuse requests the runtime would have served. The flag is what makes
    that explicit rather than a check that happens to pass.
    """
    scheduler = Scheduler(
        capacity=lambda: KVCapacity(positions=8, sizes_from_request=True)
    )

    assert scheduler.submit(list(range(1000)), 1000).max_new_tokens == 1000
    assert scheduler.capacity().sizes_from_request is True


def test_the_capacity_provider_is_read_at_submit_time_not_at_construction() -> None:
    """A capacity that arrives after the scheduler was built is the one that is checked.

    This is the reason the argument is a callable. `BackendBase.__init__` builds the scheduler before
    a subclass has read the config its capacity comes from, so a value captured at construction
    would be `None` forever.
    """
    box: dict[str, KVCapacity | None] = {"capacity": None}
    scheduler = Scheduler(capacity=lambda: box["capacity"])

    # Nothing to check against yet: the request is admitted.
    assert scheduler.submit([1, 2, 3], 9).max_new_tokens == 9

    # The load happened, and now the same request is refused.
    box["capacity"] = KVCapacity(positions=8)
    with pytest.raises(AdmissionRefused):
        scheduler.submit([1, 2, 3], 9)


def test_the_refusal_type_is_the_callers_to_choose() -> None:
    """The queue raises what it is told to, so it needs no import of the API to say no.

    ``scheduler`` is the bottom layer with ``runtime`` and may not import ``relicllm.api`` (the
    package stack in ``tests/test_package_boundaries.py``). A serving adapter passes
    ``ConfigurationError`` -- which the server maps to a 400 -- and a caller with no HTTP layer gets
    this module's ``AdmissionRefused``. Same division as the gate: the caller passes the object.
    """
    class Refused(RuntimeError):
        pass

    scheduler = Scheduler(
        capacity=lambda: KVCapacity(positions=1), refusal=Refused
    )

    with pytest.raises(Refused):
        scheduler.submit([1, 2], 1)