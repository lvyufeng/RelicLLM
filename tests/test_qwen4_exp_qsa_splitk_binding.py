"""The split-K decode path is chosen for decode shapes and nothing else.

The distinction matters more than it looks: the split-K kernel is written for one query row and has
no row loop, so a prefill call that reached it would compute a fraction of the work. A measurement
with that bug in it reported a large speedup with all-zero output, which is why the rule is pinned
here rather than left to the shape of whatever arrives.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

torch = pytest.importorskip("torch")

from relicllm.models.qwen4_exp.attention import should_use_qsa_splitk


def _shapes(rows: int = 1, selection: int = 2051):
    # Only the shapes matter to the predicate; the tensors are empty.
    query_rows = torch.empty(1, rows, 3, 256, dtype=torch.bfloat16)
    selected = torch.empty(1, rows, selection, dtype=torch.int32)
    return query_rows, selected


def test_one_query_row_is_the_decode_case():
    assert should_use_qsa_splitk(*_shapes()) is True


def test_a_prefill_row_batch_is_refused():
    # 512 query rows is what a prefill call actually passes; the kernel has no loop over them.
    assert should_use_qsa_splitk(*_shapes(rows=512, selection=2048)) is False


def test_a_selection_longer_than_the_kernel_handles_is_refused():
    assert should_use_qsa_splitk(*_shapes(selection=4096)) is True
    assert should_use_qsa_splitk(*_shapes(selection=4097)) is False


def test_the_path_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_SPLITK", "0")
    assert should_use_qsa_splitk(*_shapes()) is False


def test_the_path_is_on_by_default(monkeypatch):
    monkeypatch.delenv("POCKETLLM_QWEN4_QSA_SPLITK", raising=False)
    assert should_use_qsa_splitk(*_shapes()) is True
