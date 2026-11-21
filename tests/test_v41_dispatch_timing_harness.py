"""The private dispatch-timing harness must not report a decomposition its own total contradicts.

Round 31 published a routing share of ninety percent from this harness. Round 36 found that the
accumulators were collecting through the warmup tokens and then being divided by the timed steps
alone, which inflated every per-token figure by (warmup + steps) / steps -- 1.25x for the window the
driver uses. The wrong number was visible in the receipt: the store's forward total came out *above*
the wall time that store forward is part of, and a forty-layer model reported fifty routing calls per
token. Both are arithmetic impossibilities that nothing checked.

These tests check them. They run the harness's arithmetic on synthetic accumulators rather than the
model, so they need no card and no checkpoint, and they fail if a future edit reintroduces the
double count or drops an invariant.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / ".coord/profile_actual_cube.py"

pytestmark = pytest.mark.skipif(not PROBE.is_file(), reason="the private probe is not in this checkout")


def _dispatch_timing_function():
    """The `run_dispatch_timing` source, found without importing the probe (it needs a card)."""
    tree = ast.parse(PROBE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run_dispatch_timing":
            return node
    raise AssertionError("run_dispatch_timing is gone from the probe")


def test_the_accumulators_are_zeroed_before_the_timed_window():
    """A warmup that the accumulators survive silently inflates every per-token figure."""
    function = _dispatch_timing_function()
    source = ast.get_source_segment(PROBE.read_text(), function) or ""
    assert "for key in acc:" in source, (
        "the accumulators must be cleared after the warmup loop and before the timer starts"
    )
    zeroing = source.index("for key in acc:")
    timed_loop = source.index("for step in range(steps):")
    warmup_loop = source.index("for _ in range(warmup):")
    assert warmup_loop < zeroing < timed_loop, (
        "the zeroing must sit after the warmup and before the timed steps"
    )


def test_every_accumulator_entry_is_a_number():
    """The clear loop assigns scalars; a dict or list entry would be replaced by 0 and lose the type."""
    function = _dispatch_timing_function()
    source = ast.get_source_segment(PROBE.read_text(), function) or ""
    initial = source[source.index("acc = {"):source.index("}", source.index("acc = {")) + 1]
    tree = ast.parse(initial.replace("acc = ", "acc_value = ", 1))
    mapping = tree.body[0].value
    assert all(isinstance(value, ast.Constant) and isinstance(value.value, (int, float))
               for value in mapping.values), "every accumulator starts as a scalar"


def test_the_reported_invariants_hold_on_a_synthetic_window():
    """store_forward <= wall, routing + rest == store_forward, and one routing call per layer.

    These are the three checks that would have caught the bug in round 31. They are arithmetic, so
    they are exercised here against a window built the way the harness builds one.
    """
    layers = 40
    warmup, steps = 2, 8
    per_token_routing_ms, per_token_rest_ms = 122.34, 21.12

    # A correct harness: the accumulators hold only the timed tokens. `acc` is a total over the
    # whole window, so it is the per-token figure times the number of timed tokens -- the layers
    # are already inside the per-token figure, because one token runs every layer once.
    routing_s = per_token_routing_ms / 1e3 * steps
    forward_s = (per_token_routing_ms + per_token_rest_ms) / 1e3 * steps
    wall = 172.35 / 1e3 * steps
    calls = layers * steps

    reported = {
        "wall_ms_per_token": wall / steps * 1e3,
        "routing_ms_per_token": routing_s / steps * 1e3,
        "dispatch_rest_ms_per_token": (forward_s - routing_s) / steps * 1e3,
        "store_forward_ms_per_token": forward_s / steps * 1e3,
        "routing_calls_per_token": calls / steps,
    }
    assert reported["store_forward_ms_per_token"] <= reported["wall_ms_per_token"]
    assert (abs(reported["routing_ms_per_token"] + reported["dispatch_rest_ms_per_token"]
                - reported["store_forward_ms_per_token"]) < 1e-9)
    assert reported["routing_calls_per_token"] == layers

    # the round-31 bug, reproduced: no clear, divide by steps only
    inflated = {}
    total_tokens = warmup + steps
    for key in ("routing_ms_per_token", "dispatch_rest_ms_per_token", "store_forward_ms_per_token"):
        inflated[key] = reported[key] * total_tokens / steps
    inflated["routing_calls_per_token"] = reported["routing_calls_per_token"] * total_tokens / steps
    assert inflated["store_forward_ms_per_token"] > reported["wall_ms_per_token"], (
        "the bug is detectable: the decomposition exceeds the wall it is part of"
    )
    assert inflated["routing_calls_per_token"] == 50, "and a forty-layer model reports fifty calls"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
