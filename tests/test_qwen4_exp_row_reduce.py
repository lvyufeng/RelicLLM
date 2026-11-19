"""Opt-in BF16 verify reductions keep decode's per-row collective element count."""
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from relicllm.models.qwen4_exp.runtime import make_all_reduce


@pytest.mark.parametrize("rows", [1, 2, 4, 8, 9])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("enabled", [False, True])
def test_row_collective_count_and_inplace_answer(monkeypatch, rows, dtype, enabled):
    monkeypatch.delenv("POCKETLLM_QWEN4_SKIP_REDUCE", raising=False)
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1" if enabled else "0")
    calls = []

    def collective(value, op):
        assert op == dist.ReduceOp.SUM
        calls.append((tuple(value.shape), value.numel()))
        value.add_(1)

    monkeypatch.setattr(dist, "all_reduce", collective)
    reduce = make_all_reduce(SimpleNamespace(tp_world_size=8))
    source = torch.zeros(1, rows, 16, dtype=dtype)
    actual = reduce(source)
    assert actual is source
    assert actual.shape == (1, rows, 16)
    assert actual.dtype is dtype
    assert torch.equal(actual, torch.ones_like(actual))
    split = enabled and dtype is torch.bfloat16 and 1 < rows <= 8
    assert len(calls) == (rows if split else 1)
    assert [count for _shape, count in calls] == ([16] * rows if split else [rows * 16])


def test_noncontiguous_input_is_materialized_once_without_losing_rows(monkeypatch):
    monkeypatch.delenv("POCKETLLM_QWEN4_SKIP_REDUCE", raising=False)
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    counts = []

    def collective(value, op):
        assert value.is_contiguous()
        counts.append(value.numel())
        value.add_(2)

    monkeypatch.setattr(dist, "all_reduce", collective)
    reduce = make_all_reduce(SimpleNamespace(tp_world_size=8))
    original = torch.zeros(4, 32, dtype=torch.bfloat16)[:, ::2]
    assert not original.is_contiguous()
    actual = reduce(original)
    assert counts == [16, 16, 16, 16]
    assert actual.is_contiguous()
    assert torch.equal(actual, torch.full_like(original, 2))


def test_no_collective_for_unsharded_blocks():
    assert make_all_reduce(SimpleNamespace(tp_world_size=1)) is None


@pytest.mark.parametrize("rows", [4, 5, 6, 8])
def test_experimental_rank_chunk_layout_preserves_rows_and_inplace_contract(monkeypatch, rows):
    """CPU storage with a fake device tag tests packing, NOT actual GPU reduction rounding."""
    class DeviceTaggedTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda")

    monkeypatch.delenv("POCKETLLM_QWEN4_SKIP_REDUCE", raising=False)
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_REDUCE_LAYOUT", "rank_chunk")
    base = (torch.arange(rows * 2560, dtype=torch.float32).reshape(1, rows, 2560) / 256).to(torch.bfloat16)
    original = base.clone().as_subclass(DeviceTaggedTensor)
    calls = []

    def collective(value, op):
        plain = value.as_subclass(torch.Tensor)
        assert op == dist.ReduceOp.SUM
        assert plain.shape == (8, rows, 320)
        assert torch.equal(plain, base.reshape(rows, 8, 320).permute(1, 0, 2))
        calls.append(tuple(plain.shape))
        plain.mul_(2)

    monkeypatch.setattr(dist, "all_reduce", collective)
    actual = make_all_reduce(SimpleNamespace(tp_world_size=8))(original)
    assert actual is original
    assert calls == [(8, rows, 320)]
    assert torch.equal(actual.as_subclass(torch.Tensor), base * 2)
