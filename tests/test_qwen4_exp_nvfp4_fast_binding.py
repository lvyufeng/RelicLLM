"""The exact NVFP4 entry is opt-in and uses identical route preprocessing for all row spans."""
from types import SimpleNamespace

import pytest
import torch

from relicllm.models.qwen4_exp.moe import NvFp4SlotExpertMoE


@pytest.mark.parametrize("rows", [1, 4])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_exact_entry_dispatch_preserves_shape_dtype_routes_and_table_identity(monkeypatch, rows, enabled, available):
    monkeypatch.setenv("POCKETLLM_QWEN4_NVFP4_EXACT_FAST", "1" if enabled else "0")
    monkeypatch.setenv("POCKETLLM_QWEN4_NVFP4_ROWS", "1")
    calls = []
    table = tuple(torch.zeros(1) for _ in range(8))

    def operation(name, multi):
        def invoke(x, indices, weights, *args):
            assert x.shape == (rows, 16)
            assert x.dtype == weights.dtype == torch.float32
            assert indices.dtype == torch.int64
            assert x.is_contiguous() and indices.is_contiguous() and weights.is_contiguous()
            assert indices.shape == weights.shape == ((rows, 3) if multi else (3,))
            expected = torch.tensor([0, -1, 1]).repeat(rows, 1)
            assert torch.equal(indices.reshape(rows, 3), expected)
            assert all(a is b for a, b in zip(args[:6], table[:6]))
            assert args[6:10] == (0, 0.0, 1.0, 1.0)
            assert args[10] is table[6] and args[11] is table[7] and args[12] is False
            calls.append(name)
            return x + 1
        return invoke

    backend = NvFp4SlotExpertMoE.__new__(NvFp4SlotExpertMoE)
    backend._ops = SimpleNamespace(nvfp4_moe_token_forward=operation("token", False),
                                   nvfp4_moe_rows_forward=operation("rows", True))
    if available:
        backend._ops.nvfp4_moe_rows_forward_exact_fast = operation("exact", True)
    backend._table = lambda _layer: table
    backend.stage_calls = 0
    backend.world_size, backend.rank, backend.num_experts = 8, 1, 512
    x = torch.zeros(1, rows, 16, dtype=torch.bfloat16)
    ids = torch.tensor([1, 2, 9]).repeat(rows, 1)
    weights = torch.tensor([.25, .5, .25]).repeat(rows, 1)
    out = backend(0, x, ids, weights)
    assert out.shape == x.shape and out.dtype == x.dtype and torch.equal(out, x + 1)
    assert calls == (["exact"] if enabled and available else ["token" if rows == 1 else "rows"])
    assert backend.stage_calls == 1


def test_explicit_no_batching_keeps_existing_one_row_loop(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_NVFP4_EXACT_FAST", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_NVFP4_ROWS", "0")
    backend = NvFp4SlotExpertMoE.__new__(NvFp4SlotExpertMoE)
    backend._ops = SimpleNamespace(nvfp4_moe_rows_forward_exact_fast=lambda *_args: pytest.fail("batch disabled"))
    backend._table = lambda _layer: ()
    backend.stage_calls = 0
    backend.world_size, backend.rank, backend.num_experts = 8, 0, 512
    calls = []

    def one(_table, row, ids, weights):
        calls.append(tuple(row.shape))
        return row.float()

    backend._one_row = one
    x = torch.zeros(1, 4, 16, dtype=torch.bfloat16)
    out = backend(0, x, torch.zeros(4, 2, dtype=torch.long), torch.ones(4, 2))
    assert torch.equal(out, x)
    assert calls == [(1, 16)] * 4
