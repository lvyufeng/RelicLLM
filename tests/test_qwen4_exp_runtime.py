"""Runtime topology helpers for Qwen4-Exp heterogeneous inference."""

from __future__ import annotations

import io
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.distributed as dist

from relicllm.models.qwen4_exp import runtime


def test_parse_cpu_list() -> None:
    assert runtime._parse_cpu_list("0-2,5,8-9\n") == [0, 1, 2, 5, 8, 9]


def test_cpu_affinity_splits_ranks_on_same_numa_node(monkeypatch) -> None:
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: set(range(16)), raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", lambda _pid, _cpus: None, raising=False)
    monkeypatch.setattr(runtime, "_cuda_numa_node", lambda device: int(device.index or 0) // 2)
    monkeypatch.setattr(
        runtime,
        "_physical_core_groups",
        lambda cpus: [[cpu] for cpu in sorted(cpus)],
    )

    real_open = open

    def fake_open(path, *args, **kwargs):
        if path.endswith("node0/cpulist"):
            return io.StringIO("0-7")
        if path.endswith("node1/cpulist"):
            return io.StringIO("8-15")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)

    assert runtime._cpu_affinity_for_device(torch.device("cuda:0"), 0, 4) == [0, 1, 2, 3]
    assert runtime._cpu_affinity_for_device(torch.device("cuda:1"), 1, 4) == [4, 5, 6, 7]
    assert runtime._cpu_affinity_for_device(torch.device("cuda:2"), 2, 4) == [8, 9, 10, 11]
    assert runtime._cpu_affinity_for_device(torch.device("cuda:3"), 3, 4) == [12, 13, 14, 15]


@pytest.mark.parametrize("seq_len", [None, 1, 3])
def test_tp_selector_normalizes_then_gathers_once(monkeypatch, seq_len) -> None:
    ctx = runtime.TPContext(0, 2, torch.device("cpu"), True)
    local = torch.tensor([[4.0, 0.0, 1.0], [0.0, 4.0, 1.0]])
    remote = torch.tensor([[0.0, 1.0, 8.0], [0.0, 8.0, 1.0]])
    if seq_len is None:
        logits = local
    else:
        logits = torch.zeros(2, seq_len, 3)
        logits[:, :-1, 0] = 100.0
        logits[:, -1] = local
    gathered = []

    def all_gather(parts, tensor):
        gathered.append(tensor.clone())
        assert tensor.shape == (2, 3), "the collective must see local, last-position logits"
        parts[0].copy_(tensor)
        parts[1].copy_(remote)

    monkeypatch.setattr(dist, "all_gather", all_gather)
    token = runtime.next_token_from_logits(logits, ctx, batch_size=2)

    assert len(gathered) == 1
    torch.testing.assert_close(gathered[0], local)
    torch.testing.assert_close(token, torch.tensor([[5], [4]]))
    assert token.shape == (2, 1)
    assert token.dtype == torch.long


@pytest.mark.parametrize("with_sequence_axis", [False, True])
def test_pp_head_broadcasts_canonical_batch_tokens(monkeypatch, with_sequence_axis) -> None:
    ctx = runtime.TPContext(1, 2, torch.device("cpu"), True, pp_size=2, pp_rank=1)
    logits = torch.tensor([[0.0, 7.0, 0.0], [0.0, 0.0, 7.0]])
    if with_sequence_axis:
        logits = logits[:, None]
    broadcasts = []

    def broadcast(token, src):
        assert token.shape == (2, 1), "PP must not broadcast a [B, 1, 1] token"
        broadcasts.append((src, token.clone()))

    def unexpected_gather(*_args, **_kwargs):
        raise AssertionError("a PP head already owns the whole vocabulary")

    monkeypatch.setattr(dist, "broadcast", broadcast)
    monkeypatch.setattr(dist, "all_gather", unexpected_gather)
    token = runtime.next_token_from_logits(logits, ctx, batch_size=2)

    assert len(broadcasts) == 1
    assert broadcasts[0][0] == 1
    torch.testing.assert_close(token, torch.tensor([[1], [2]]))


def test_pp_non_head_receives_one_token_per_row(monkeypatch) -> None:
    ctx = runtime.TPContext(0, 2, torch.device("cpu"), True, pp_size=2, pp_rank=0)

    def broadcast(token, src):
        assert src == 1
        assert token.shape == (2, 1)
        assert token.dtype == torch.long
        token.copy_(torch.tensor([[5], [6]]))

    monkeypatch.setattr(dist, "broadcast", broadcast)
    token = runtime.next_token_from_logits(None, ctx, batch_size=2)
    torch.testing.assert_close(token, torch.tensor([[5], [6]]))


@pytest.mark.parametrize("pp_head", [False, True])
@pytest.mark.parametrize("shape", [None, (8,), (2, 0), (2, 0, 8), (2, 1, 1, 8), (1, 8)])
def test_selector_rejects_missing_or_unsupported_logits_before_collectives(
    monkeypatch, pp_head, shape
) -> None:
    ctx = runtime.TPContext(
        int(pp_head), 2, torch.device("cpu"), True,
        pp_size=2 if pp_head else 1, pp_rank=int(pp_head),
    )

    def unexpected_collective(*_args, **_kwargs):
        raise AssertionError("invalid logits must fail before a collective")

    monkeypatch.setattr(dist, "broadcast", unexpected_collective)
    monkeypatch.setattr(dist, "all_gather", unexpected_collective)
    logits = None if shape is None else torch.zeros(shape)
    with pytest.raises(ValueError, match="logits"):
        runtime.next_token_from_logits(logits, ctx, batch_size=2)


@pytest.mark.parametrize("shape", [(2,), (2, 1, 1), (1, 2)])
def test_decode_token_rejects_implicit_reshape_or_broadcast(shape) -> None:
    with pytest.raises(ValueError, match="expected token shape"):
        runtime._decode_token(torch.zeros(shape, dtype=torch.long), batch_size=2)


def test_decode_token_requires_int64() -> None:
    with pytest.raises(TypeError, match="int64"):
        runtime._decode_token(torch.zeros(2, 1), batch_size=2)


def test_pp_graph_is_refused_before_invoking_the_builder(monkeypatch) -> None:
    from types import SimpleNamespace

    def unexpected_build(**_kwargs):
        raise AssertionError("pipeline graph capture must be refused before cache mutation")

    monkeypatch.setitem(
        sys.modules, "relicllm.models.qwen4_exp.graph_window", SimpleNamespace(build=unexpected_build)
    )
    # A device descriptor does not initialize CUDA. The guard must return before any
    # tensor allocation, synchronization or graph work, even for a CUDA PP launch.
    ctx = runtime.TPContext(0, 2, torch.device("cuda:0"), True, pp_size=2, pp_rank=0)
    assert runtime._build_decode_window(
        ctx=ctx, model=object(), cache=object(), start=2, steps=2,
        token=torch.tensor([[3]]), select=None,
    ) is None
