"""Opt-in indexer score geometry keeps every verify row at decode's GEMM N.

CPU tests pin dispatch, layout, row order and the single shared FP32 key cast. They do not
prove that a GPU cuBLAS reassociation exists or that this path fixes it; a complete 128-token
GPU comparison has to decide that, so no test here claims throughput or token parity.
"""
import torch
import pytest

from relicllm.models import qwen4_exp
from relicllm.models.qwen4_exp import attention
from relicllm.models.qwen4_exp.attention import (
    QSAIndexer,
    _indexer_score_rows,
    indexer_score_row_invariant_enabled,
)
from relicllm.models.qwen4_exp.config import Qwen4ExpTextConfig

INDEXER_FLAG = "POCKETLLM_QWEN4_INDEXER_ROW_INVARIANT"
ROWS_FLAG = "POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT"


def _inputs(rows, blocks=6, n_heads=4, head_dim=8, dtype=torch.bfloat16, seed=0):
    generator = torch.Generator().manual_seed(seed)
    keys = torch.randn(1, blocks, head_dim, generator=generator)
    query = torch.randn(1, rows, n_heads, head_dim, generator=generator).to(dtype)
    return keys, query


def _old_expression(keys, query):
    """The single batched GEMM the helper must preserve on the default path."""
    batch, rows, heads, dim = query.shape
    flat = query.float().reshape(batch, rows * heads, dim)
    scores = torch.matmul(keys.float(), flat.transpose(1, 2))
    return scores.reshape(batch, keys.shape[1], rows, heads).permute(0, 2, 1, 3)


def _serial_oracle(keys, query):
    """One GEMM per query row with N == heads: the geometry a decode step uses."""
    batch, rows, heads, dim = query.shape
    shared = keys.float()
    return torch.stack([
        torch.matmul(shared, query[:, row].float().reshape(batch, heads, dim).transpose(1, 2))
        for row in range(rows)
    ], dim=1)


def test_flag_requires_both_switches(monkeypatch):
    monkeypatch.delenv(INDEXER_FLAG, raising=False)
    monkeypatch.delenv(ROWS_FLAG, raising=False)
    assert indexer_score_row_invariant_enabled(4) is False
    monkeypatch.setenv(ROWS_FLAG, "1")
    assert indexer_score_row_invariant_enabled(4) is False  # indexer switch still off
    monkeypatch.setenv(INDEXER_FLAG, "1")
    assert indexer_score_row_invariant_enabled(4) is True
    monkeypatch.setenv(ROWS_FLAG, "0")
    assert indexer_score_row_invariant_enabled(4) is False
    monkeypatch.setenv(ROWS_FLAG, "1")
    for rows in (0, 1, 9, 16):
        assert indexer_score_row_invariant_enabled(rows) is False


@pytest.mark.parametrize("rows", [1, 2, 3, 4, 6, 8, 9])
def test_default_off_is_the_single_batched_gemm(monkeypatch, rows):
    monkeypatch.delenv(INDEXER_FLAG, raising=False)
    monkeypatch.setenv(ROWS_FLAG, "1")
    keys, query = _inputs(rows)
    calls = []
    original = torch.matmul

    def record(left, right):
        calls.append((tuple(left.shape), tuple(right.shape)))
        return original(left, right)

    monkeypatch.setattr(torch, "matmul", record)
    actual = _indexer_score_rows(keys, query, 1, rows, 4, 8)
    assert len(calls) == 1
    assert calls[0][1] == (1, 8, rows * 4)  # N is rows * heads, unchanged
    assert actual.shape == (1, rows, keys.shape[1], 4)
    assert torch.equal(actual, _old_expression(keys, query))


@pytest.mark.parametrize("rows", [2, 3, 4, 6, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_opt_in_runs_one_gemm_per_row_at_decode_width(monkeypatch, rows, dtype):
    monkeypatch.setenv(ROWS_FLAG, "1")
    monkeypatch.setenv(INDEXER_FLAG, "1")
    keys, query = _inputs(rows, dtype=dtype, seed=rows)
    calls = []
    original = torch.matmul

    def record(left, right):
        calls.append((left, right))
        return original(left, right)

    monkeypatch.setattr(torch, "matmul", record)
    actual = _indexer_score_rows(keys, query, 1, rows, 4, 8)
    assert len(calls) == rows
    for row, (_left, right) in enumerate(calls):
        assert tuple(right.shape) == (1, 8, 4)  # decode N, not rows * heads
        assert torch.equal(right, query[:, row].float().reshape(1, 4, 8).transpose(1, 2))
    # Every row scores against one FP32 key tensor: one cast, not one copy per row.
    assert len({id(left) for left, _right in calls}) == 1
    assert calls[0][0].dtype is torch.float32
    assert torch.equal(calls[0][0], keys.float())
    assert actual.dtype is torch.float32
    assert actual.shape == (1, rows, keys.shape[1], 4)
    assert torch.equal(actual, _serial_oracle(keys, query))
    # The shared cast is out of place: neither input is mutated.
    assert torch.equal(query, _inputs(rows, dtype=dtype, seed=rows)[1])
    assert torch.equal(keys, _inputs(rows, dtype=dtype, seed=rows)[0])


def test_row_order_matches_the_batched_layout(monkeypatch):
    """A row-indexed fake GEMM proves stack order, not merely the output shape."""
    monkeypatch.setenv(ROWS_FLAG, "1")
    monkeypatch.setenv(INDEXER_FLAG, "1")
    keys, query = _inputs(5, blocks=3)
    calls = []

    def fake(left, right):
        assert left.shape == (1, 3, 8) and right.shape == (1, 8, 4)
        calls.append(len(calls))
        return torch.full((1, 3, 4), float(len(calls) - 1) * 10.0)

    monkeypatch.setattr(torch, "matmul", fake)
    actual = _indexer_score_rows(keys, query, 1, 5, 4, 8)
    assert len(calls) == 5
    expected = torch.stack(
        [torch.full((1, 3, 4), float(row) * 10.0) for row in range(5)], dim=1
    )
    assert torch.equal(actual, expected)


def test_queries_outside_the_verify_range_stay_batched(monkeypatch):
    monkeypatch.setenv(ROWS_FLAG, "1")
    monkeypatch.setenv(INDEXER_FLAG, "1")
    for rows in (1, 9):
        keys, query = _inputs(rows)
        seen = []
        original = torch.matmul

        def record(left, right, _seen=seen, _original=original):
            _seen.append(tuple(right.shape))
            return _original(left, right)

        monkeypatch.setattr(torch, "matmul", record)
        actual = _indexer_score_rows(keys, query, 1, rows, 4, 8)
        assert seen == [(1, 8, rows * 4)]
        assert torch.equal(actual, _old_expression(keys, query))


def test_head_count_and_dimension_are_preserved(monkeypatch):
    monkeypatch.setenv(ROWS_FLAG, "1")
    monkeypatch.setenv(INDEXER_FLAG, "1")
    keys, query = _inputs(4, blocks=7, n_heads=3, head_dim=16)
    actual = _indexer_score_rows(keys, query, 1, 4, 3, 16)
    assert actual.shape == (1, 4, 7, 3)
    assert torch.equal(actual, _serial_oracle(keys, query))


def _indexer_fixture(rows, *, heads=2, head_dim=8, hidden=16, budget=8, ratio=2):
    config = Qwen4ExpTextConfig(
        hidden_size=hidden,
        head_dim=head_dim,
        indexer_n_heads=heads,
        indexer_kv_heads=1,
        indexer_head_dim=head_dim,
        indexer_budget=budget,
        indexer_compress_ratio=ratio,
    )
    generator = torch.Generator().manual_seed(11)
    weights = {
        "index_qk_proj": torch.randn(heads * head_dim + head_dim, hidden, generator=generator),
        # The query norm normalizes each head's last axis, so its weight is one head wide.
        "q_layernorm": torch.randn(head_dim, generator=generator),
        "k_layernorm": torch.randn(head_dim, generator=generator),
    }
    indexer = QSAIndexer(config, weights)
    hidden_states = torch.randn(1, rows, hidden, generator=generator)
    positions = 64
    cos = torch.rand(1, positions, config.rotary_dim, generator=generator) * 2 - 1
    sin = torch.rand(1, positions, config.rotary_dim, generator=generator) * 2 - 1
    raw_keys = torch.randn(1, 12, head_dim, generator=generator)
    return indexer, hidden_states, cos, sin, raw_keys


@pytest.mark.parametrize("enabled", [False, True])
def test_indexer_call_dispatches_through_the_helper(monkeypatch, enabled):
    monkeypatch.setenv(ROWS_FLAG, "1")
    if enabled:
        monkeypatch.setenv(INDEXER_FLAG, "1")
    else:
        monkeypatch.delenv(INDEXER_FLAG, raising=False)
    indexer, hidden_states, cos, sin, raw_keys = _indexer_fixture(4)
    seen = []
    original = attention._indexer_score_rows

    def record(block_keys, query, batch_size, seq_len, n_heads, head_dim):
        seen.append((tuple(block_keys.shape), tuple(query.shape), batch_size, seq_len,
                     n_heads, head_dim))
        return original(block_keys, query, batch_size, seq_len, n_heads, head_dim)

    monkeypatch.setattr(attention, "_indexer_score_rows", record)
    selected = indexer(hidden_states, cos, sin, None, past_len=0, raw_keys_override=raw_keys)
    # 12 override keys + 4 query positions = 16 keys, ratio 2 -> 8 blocks.
    assert seen == [((1, 8, 8), (1, 4, 2, 8), 1, 4, 2, 8)]
    assert selected.dtype is torch.int32
    assert selected.shape == (1, 4, 9)  # min(kv_len, budget + ratio - 1)