"""CPU checks that the rank-maximum exchange picks the token a full gather would.

The exchange is exact only if three things hold: local indices offset to global ones, ties
resolve to the lowest global index (which is the lowest rank, because shards are contiguous),
and the value comparison survives the float32 shipping cast.  Each is checked here against
``torch.cat(parts).argmax()`` computed directly, over vocabularies and value patterns that
force ties inside one shard, across shards, and at the last global index.
"""
from __future__ import annotations

import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch

from relicllm.models.qwen4_exp import runtime

LOW = -1e4


def _shard(peaks, width):
    """One shard's logits: `LOW` everywhere except `peaks` = {(local_index, value)}."""
    row = torch.full((width,), LOW)
    for index, value in peaks:
        row[index] = value
    return row


def _install(monkeypatch, rows):
    """Stand in for the collective: every rank receives every rank's shipped pair."""

    def gather(local, _ctx):
        batch = local.shape[0]
        out = [torch.empty_like(local) for _ in rows]
        for rank, row in enumerate(rows):
            if batch == 1:
                best, at = row.max(dim=-1)
                out[rank][0, 0] = best.to(torch.float32)
                out[rank][0, 1] = (at + rank * row.shape[-1]).to(torch.float32)
            else:
                best, at = row.max(dim=-1)
                out[rank][:, 0] = best.to(torch.float32)
                out[rank][:, 1] = (at + rank * row.shape[-1]).to(torch.float32)
        return out

    monkeypatch.setattr(runtime, "_argmax_all_gather", gather)


def _reference(rows):
    """The answer a full gather gives, shaped like the exchange's contract: one row a batch row."""
    return torch.cat(rows, dim=-1).reshape(1, -1).argmax(-1, keepdim=True)


def _run(monkeypatch, rows, rank):
    _install(monkeypatch, rows)
    ctx = SimpleNamespace(tp_world_size=len(rows), tp_rank=rank, argmax_parts=None)
    return runtime.argmax_sharded(rows[rank].unsqueeze(0), ctx)


def _check(monkeypatch, rows):
    expected = _reference(rows)
    for rank in range(len(rows)):
        assert torch.equal(_run(monkeypatch, rows, rank), expected), f"rank {rank}"


def test_unambiguous_winner_across_shards(monkeypatch):
    _check(monkeypatch, [
        _shard([(1, 2.0), (3, 1.0)], 5),
        _shard([(0, 5.0), (2, 0.5)], 5),
        _shard([(1, 2.0), (4, 4.0)], 5),
    ])


def test_tie_across_shards_takes_lowest_global_index(monkeypatch):
    # Ranks 0 and 2 peak at the same value; the earlier shard must win.
    rows = [
        _shard([(4, 7.0)], 6),
        _shard([(2, 7.0)], 6),
        _shard([(0, 7.0)], 6),
    ]
    expected = _reference(rows)
    assert expected.item() == 4
    _check(monkeypatch, rows)


def test_tie_inside_one_shard_takes_that_shards_first_index(monkeypatch):
    rows = [
        _shard([(2, 9.0), (0, 9.0)], 4),
        _shard([(1, 3.0), (3, 3.0)], 4),
    ]
    expected = _reference(rows)
    assert expected.item() == 0
    _check(monkeypatch, rows)


def test_last_global_index_is_reachable(monkeypatch):
    rows = [_shard([(0, 1.0)], 5), _shard([(4, 8.0)], 5)]
    expected = _reference(rows)
    assert expected.item() == 9
    _check(monkeypatch, rows)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shipped_cast_preserves_the_comparison(monkeypatch, dtype):
    rows = [
        _shard([(0, 1.0), (3, 2.0)], 4).to(dtype),
        _shard([(2, 1.5)], 4).to(dtype),
    ]
    _check(monkeypatch, rows)


def test_single_rank_never_exchanges(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("a one-rank vocabulary has nothing to exchange")

    monkeypatch.setattr(runtime, "_argmax_all_gather", forbidden)
    ctx = SimpleNamespace(tp_world_size=1, tp_rank=0, argmax_parts=None)
    logits = torch.tensor([[0.0, 3.0, 1.0]])
    assert runtime.argmax_sharded(logits, ctx).tolist() == [[1]]


def test_flag_defaults_off_and_gates_the_helper(monkeypatch):
    monkeypatch.delenv("POCKETLLM_QWEN4_ARGMAX_EXCHANGE", raising=False)
    assert runtime._argmax_exchange_enabled() is False
    monkeypatch.setenv("POCKETLLM_QWEN4_ARGMAX_EXCHANGE", "1")
    assert runtime._argmax_exchange_enabled() is True


def test_exchange_reuses_its_preallocated_buffers(monkeypatch):
    rows = [_shard([(0, 1.0)], 3), _shard([(2, 2.0)], 3)]
    _install(monkeypatch, rows)
    ctx = SimpleNamespace(tp_world_size=2, tp_rank=0, argmax_parts=None)
    first = runtime.argmax_sharded(rows[0].unsqueeze(0), ctx)
    held = ctx.argmax_parts
    second = runtime.argmax_sharded(rows[0].unsqueeze(0), ctx)
    assert ctx.argmax_parts is held
    assert torch.equal(first, second) and first.tolist() == [[5]]


def test_every_rank_of_a_three_row_batch_agrees(monkeypatch):
    """Three rows whose winners sit on different ranks, so a row swap cannot pass."""
    width = 4
    # One row per batch entry, one shard per rank: batch[row][rank] is what that rank holds.
    batch = [
        [_shard([(1, 3.0)], width), _shard([(0, 1.0)], width), _shard([(3, 9.0)], width)],
        [_shard([(0, 1.0)], width), _shard([(1, 5.0)], width), _shard([(2, 5.0)], width)],
        [_shard([(2, 0.5)], width), _shard([(3, 8.0)], width), _shard([(0, 2.0)], width)],
    ]
    rows = len(batch)
    expected = torch.stack([
        torch.cat([batch[row][rank] for rank in range(3)], dim=-1).argmax() for row in range(rows)
    ])
    assert expected.tolist() == [11, 5, 7]

    def gather(local, _ctx):
        # Every rank sees every rank's shipped (value, global index) pair.
        out = [torch.empty_like(local) for _ in range(3)]
        for rank in range(3):
            shard = torch.stack([batch[row][rank] for row in range(rows)])
            best, at = shard.max(dim=-1)
            out[rank][:, 0] = best.to(torch.float32)
            out[rank][:, 1] = (at + rank * width).to(torch.float32)
        return out

    monkeypatch.setattr(runtime, "_argmax_all_gather", gather)
    for rank in range(3):
        stacked = torch.stack([batch[row][rank] for row in range(rows)])
        ctx = SimpleNamespace(tp_world_size=3, tp_rank=rank, argmax_parts=None)
        assert torch.equal(runtime.argmax_sharded(stacked, ctx).reshape(-1), expected), f"rank {rank}"


def test_tensor_core_qsa_path_is_requested_by_default(monkeypatch):
    """A launch that says nothing about the kernel must get the tiled one, not the scalar one.

    The extension's selector takes the scalar path unless this variable is set at all, so an
    unset variable is not a neutral default -- it is the slow kernel.  An explicit 0 must keep
    working as an opt-out, and a run that already chose must not be overridden.
    """
    from relicllm.models.qwen4_exp import runtime as qwen_runtime

    monkeypatch.delenv("POCKETLLM_QWEN4_QSA_MMA", raising=False)
    qwen_runtime._default_tensor_core_qsa()
    assert qwen_runtime.os.environ["POCKETLLM_QWEN4_QSA_MMA"] == "1"

    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_MMA", "0")
    qwen_runtime._default_tensor_core_qsa()
    assert qwen_runtime.os.environ["POCKETLLM_QWEN4_QSA_MMA"] == "0"
