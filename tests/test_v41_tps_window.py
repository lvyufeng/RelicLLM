"""The reported TPS must come from a warmup-separated, synchronised, fixed-length window.

The headline number this work reports rests on the timed window in the private probe. Round 36 found
that a *different* harness in the same file divided by the timed steps while its accumulators still
held the warmup, inflating every figure by 1.25x. The TPS window is the one number that must not
carry that class of error, so it gets the same treatment: these tests read the probe's own source and
assert the properties the reported figure depends on. They need no card and no checkpoint.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / ".coord/profile_actual_cube.py"

pytestmark = pytest.mark.skipif(not PROBE.is_file(), reason="the private probe is not in this checkout")


def _source():
    return PROBE.read_text()


def _function(name: str) -> str:
    text = _source()
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(text, node) or ""
    raise AssertionError(f"{name} is gone from the probe")


def test_timed_steps_is_sixteen():
    """The objective requires at least sixteen warmup-separated forwards."""
    text = _source()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("TIMED_STEPS"):
            assert stripped.replace(" ", "").endswith("=16"), stripped
            return
    raise AssertionError("TIMED_STEPS is not defined in the probe")


def test_the_timer_starts_after_the_barrier_and_sync():
    """Ranks must be released together and the device drained before the clock starts."""
    body = _function("_run_rank") if "def _run_rank" in _source() else _source()
    clock = body.index("start_wall = time.perf_counter()")
    before = body[:clock]
    assert "dist.barrier()" in before, "a barrier must precede the timed window"
    assert before.rindex("sync()") > before.rindex("dist.barrier()") - 1
    # and the loop that follows is the fixed-length one
    loop = body[clock:]
    assert "for step in range(TIMED_STEPS)" in loop


def test_each_timed_step_is_synchronised():
    """A step that is not synchronised measures enqueue, not completion."""
    text = _source()
    clock = text.index("start_wall = time.perf_counter()")
    loop = text[clock:text.index("wall = time.perf_counter()-start_wall")]
    assert "sync()" in loop, "each timed step must synchronise before the next"
    assert loop.count("TIMED_STEPS") >= 1


def test_tps_is_the_step_count_over_the_wall():
    """The conversion must divide the fixed step count by the measured span, nothing else.

    Checked structurally rather than as text, because the dict literal spans lines and a string
    match would be defeated by formatting rather than by a wrong formula.
    """
    tree = ast.parse(_source())
    wanted = {"sustained_single_stream_TPS": ("TIMED_STEPS", "wall"),
              "timed_decode_seconds": ("wall",)}
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key, value in zip(node.keys, node.values):
            if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
                continue
            if key.value not in wanted:
                continue
            names = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
            assert set(wanted[key.value]) <= names, (
                f"{key.value} must be built from {wanted[key.value]}, found {sorted(names)}"
            )
            if key.value == "sustained_single_stream_TPS":
                # The shape matters, not just the names: a mutation test showed that
                # `TIMED_STEPS / (wall * 0.8)` still mentions both names and would have slipped
                # through a names-only check, while scaling the reported rate by 1.25.
                assert isinstance(value, ast.BinOp) and isinstance(value.op, ast.Div), (
                    "TPS must be a division"
                )
                assert isinstance(value.left, ast.Name) and value.left.id == "TIMED_STEPS"
                assert isinstance(value.right, ast.Name) and value.right.id == "wall", (
                    "the divisor must be the measured wall itself, not an expression over it"
                )
            seen.add(key.value)
    assert seen == set(wanted), f"missing receipt keys: {set(wanted) - seen}"


def test_the_receipt_states_that_warmup_is_excluded():
    """The objective requires the context and the warmup separation to be stated."""
    text = _source()
    assert "timed_warmup_steps_excluded" in text
    assert '"timed_steps": TIMED_STEPS' in text


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
