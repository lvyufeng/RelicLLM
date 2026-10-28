"""`_run_experts_fixed` must be the dispatch loop's arithmetic, held to a fixed shape.

The decode path runs one bound Python call per *slot* instead of one per distinct expert, so the
step is identical whatever the router draws. That is only worth having if the answer is the loop's.
The loop's slots are the top-k in ascending expert id (it derives them from a stable `argsort`), so
a free route's slot order is the raw one; under TP a foreign id is a sentinel the loop skips, and a
slot whose weight is zero must not contribute.

These are CPU tests: bf16 matmul is exact enough on the host for an equality check, and the cost of
running them without a card is the whole point of testing the arithmetic here rather than on one.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from relicllm.models.qwen4_exp import moe  # noqa: E402
from relicllm.models.qwen4_exp.quant import FP8Tensor  # noqa: E402


_BLOCK = (128, 128)


def _geometry():
    return dict(experts=8, top_k=3, hidden=256, intermediate=320)


def _operands(g, *, dtype=torch.bfloat16, device="cpu"):
    gen = torch.Generator().manual_seed(0)
    w = g["experts"]
    gate_up = torch.randn(g["experts"], 2 * g["intermediate"], g["hidden"], generator=gen,
                          dtype=torch.float32).to(dtype).to(device)
    down = torch.randn(g["experts"], g["hidden"], g["intermediate"], generator=gen,
                       dtype=torch.float32).to(dtype).to(device)
    # Per-expert FP8 codes with block scales, flat so that expert `i` is a row slice of both: the
    # shape `HostExpertShard` holds on the card, one pair of tensors per layer.
    code = torch.randint(-126, 127, (w, 2 * g["intermediate"], g["hidden"]), generator=gen,
                         dtype=torch.int8).to(device)
    down_code = torch.randint(-126, 127, (w, g["hidden"], g["intermediate"]), generator=gen,
                              dtype=torch.int8).to(device)
    # The block grid follows the *stacked* rows, so the scales are laid out per expert rather than
    # as per-expert grids (a 2-D weight has exactly one grid, which is why the shard holds codes).
    gu_scale = (-(-(2 * g["intermediate"]) // _BLOCK[0]), -(-g["hidden"] // _BLOCK[1]))
    down_scale = (-(-g["hidden"] // _BLOCK[0]), -(-g["intermediate"] // _BLOCK[1]))
    scales = torch.rand(w, *gu_scale, generator=gen, dtype=torch.float32).to(device) * 0.05
    down_scales = torch.rand(w, *down_scale, generator=gen, dtype=torch.float32).to(device) * 0.05
    return {
        "gate_up": gate_up,
        "down": down,
        "fp8_code": code,
        "fp8_down_code": down_code,
        "fp8_scales": scales,
        "fp8_down_scales": down_scales,
    }


def _fp8_rows(op, g, local_row):
    """Expert `local_row` as the `FP8Tensor`s the shard hands back."""
    return (
        FP8Tensor(op["fp8_code"][local_row], op["fp8_scales"][local_row], _BLOCK),
        FP8Tensor(op["fp8_down_code"][local_row], op["fp8_down_scales"][local_row], _BLOCK),
    )


def _reference(g, op, hidden, indices, weights, fp8=False):
    """The loop, spelled out slot by slot -- one expert per call, no grouping."""
    out = torch.zeros_like(hidden)
    for slot, expert_id in enumerate(indices[0].tolist()):
        if fp8:
            gate_up_raw, down_raw = _fp8_rows(op, g, expert_id)
            gate_up = gate_up_raw.dequantize(hidden.dtype)
            down = down_raw.dequantize(hidden.dtype)
        else:
            gate_up, down = op["gate_up"][expert_id], op["down"][expert_id]
        contrib = moe.swiglu_expert(hidden, gate_up, down)
        out += (contrib * weights[0, slot].to(contrib.dtype)).to(out.dtype)
    return out


def test_the_deep_cache_row_is_the_route_id_not_a_local_line():
    """The cache indexes rows by the id it owns (`id % world_size`), so nothing is renumbered.

    A 0-based local line would have turned expert 3 into row 0 for the rank whose first id is 24,
    which is a different expert's weights under the same route.
    """
    g = _geometry()
    op = _operands(g)
    hidden = torch.randn(1, g["hidden"], dtype=torch.bfloat16)
    indices = torch.tensor([[5, 2, 7]], dtype=torch.long)
    weights = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.bfloat16)
    reads = []

    def rows(row):
        reads.append(row)
        return op["gate_up"][row], op["down"][row]

    got = moe._run_experts_fixed(hidden, weights, indices, g["experts"], rows, g["experts"])
    want = _reference(g, op, hidden, indices, weights)

    assert reads == [5, 2, 7]
    assert torch.equal(got, want)


def test_the_deep_cache_masks_a_foreign_id_the_way_the_shard_does():
    """The cache holds this rank's rows only, so a foreign id must become the sentinel.

    Read raw, expert 5 on the rank that owns ids 2 and 6 would be row 5 of a 2-row table -- past
    its end -- or, unsentinel-ised, someone else's expert under a real weight.
    """
    g = _geometry()
    op = _operands(g)
    cache = moe.DeepCacheMoE(g["experts"], torch.device("cpu"), scale=1.0, world_size=4, rank=2)
    layer = 0
    # Stored compact, one row per owned id -- expert 2 is line 0, expert 6 is line 1.
    mine = torch.arange(2, g["experts"], 4)
    cache.decode_layer(layer, (op["gate_up"][mine], op["down"][mine]))
    assert cache.num_local_experts == 2

    route = [5, 2, 7]
    hidden = torch.randn(1, g["hidden"], dtype=torch.bfloat16)
    indices = torch.tensor([route], dtype=torch.long)
    weights = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.bfloat16)
    masked = cache.local_rows(indices)
    assert masked.tolist() == [[g["experts"], 2, g["experts"]]]

    got = cache(layer, hidden, indices, weights)
    contrib = moe.swiglu_expert(hidden, op["gate_up"][2], op["down"][2])
    want = (contrib * weights[0, 1].to(contrib.dtype)).to(got.dtype)
    assert torch.equal(got, want)


def test_a_layer_decoded_twice_is_kept_once():
    """Resuming a decode that was interrupted re-reads a layer it already holds; the bytes stay."""
    g = _geometry()
    op = _operands(g)
    cache = moe.DeepCacheMoE(g["experts"], torch.device("cpu"), scale=1.0)
    held = cache.decode_layer(0, (op["gate_up"], op["down"]))
    again = cache.decode_layer(0, (op["gate_up"], op["down"]))
    assert again == 0 and cache.decoded_bytes == held


@pytest.mark.parametrize("fp8", [False, True], ids=["dense", "fp8"])
def test_one_slot_of_work_per_call_matches_the_per_expert_path(fp8):
    g = _geometry()
    op = _operands(g)
    hidden = torch.randn(1, g["hidden"], dtype=torch.bfloat16)
    indices = torch.tensor([[5, 1, 7]], dtype=torch.long)
    weights = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.bfloat16)

    def rows(local_row):
        if fp8:
            return _fp8_rows(op, g, local_row)
        return op["gate_up"][local_row], op["down"][local_row]

    got = moe._run_experts_fixed(hidden, weights, indices, g["experts"], rows, g["experts"])
    want = _reference(g, op, hidden, indices, weights, fp8=fp8)

    assert torch.equal(got, want), f"max {(got - want).abs().max().item()}"
    assert got.dtype == hidden.dtype and got.shape == hidden.shape


def test_a_repeated_expert_runs_once_per_slot_not_once_per_id():
    """The loop runs a repeated draw once; the fixed path runs it `top_k` times, as slot order.

    The answers agree because the weight is split across the repeats, so a path that dropped the
    repeat would be short by exactly the dropped weight -- which is what the second assertion
    measures rather than a tolerance.
    """
    g = _geometry()
    op = _operands(g)
    hidden = torch.randn(1, g["hidden"], dtype=torch.bfloat16)
    indices = torch.tensor([[3, 3, 6]], dtype=torch.long)
    local = g["experts"]
    weights = torch.tensor([[0.25, 0.25, 0.5]], dtype=torch.bfloat16)

    def rows(local_row):
        return op["gate_up"][local_row], op["down"][local_row]

    got = moe._run_experts_fixed(hidden, weights, indices, g["experts"], rows, local)
    want = _reference(g, op, hidden, indices, weights)

    assert torch.equal(got, want)
    once = moe.swiglu_expert(hidden, op["gate_up"][3], op["down"][3]) * weights[0, 0]
    assert not torch.equal(got, once.to(got.dtype)), "the second draw of expert 3 was skipped"


def test_a_foreign_sentinel_contributes_nothing_and_reads_in_bounds():
    """Under TP a route to another rank arrives as `num_experts`; the weight is the zeroing.

    The row it reads is clamped, which matters because the sentinel itself indexes past the end of
    the local table.
    """
    g = _geometry()
    op = _operands(g)
    hidden = torch.randn(1, g["hidden"], dtype=torch.bfloat16)
    local = 3
    indices = torch.tensor([[1, g["experts"], g["experts"] + 1]], dtype=torch.long)
    weights = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.bfloat16)
    reads = []

    def rows(local_row):
        reads.append(local_row)
        return op["gate_up"][local_row], op["down"][local_row]

    got = moe._run_experts_fixed(
        hidden, weights, indices, g["experts"], rows, local, rows_are_local=False
    )
    want = moe.swiglu_expert(hidden, op["gate_up"][1], op["down"][1]) * weights[0, 0]

    assert all(0 <= row < local for row in reads), reads
    zeroed = torch.zeros(1, g["hidden"], dtype=torch.bfloat16)
    assert torch.equal(got, want.to(got.dtype).to(zeroed.device))
    # The zero weight, not a skipped call: the sentinel slots still ran an expert each.
    assert len(reads) == g["top_k"]


def test_the_shape_is_the_same_whatever_the_route():
    """What the fixed path buys: the call count and shapes do not depend on the draw."""
    g = _geometry()
    op = _operands(g)
    hidden = torch.randn(1, g["hidden"], dtype=torch.bfloat16)
    weights = torch.ones(1, g["top_k"], dtype=torch.bfloat16)
    shapes = []

    for draw in ([0, 1, 2], [4, 4, 4], [7, 0, 3]):
        indices = torch.tensor([draw], dtype=torch.long)

        def rows(local_row):
            gate_up, down = op["gate_up"][local_row], op["down"][local_row]
            shapes.append((tuple(hidden.shape), tuple(gate_up.shape), tuple(down.shape)))
            return gate_up, down

        moe._run_experts_fixed(hidden, weights, indices, g["experts"], rows, g["experts"])

    per_step = len(shapes) // 3
    assert per_step == g["top_k"]
    assert shapes[:per_step] == shapes[per_step : 2 * per_step] == shapes[2 * per_step :]