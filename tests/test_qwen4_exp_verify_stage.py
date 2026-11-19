"""The verify graph's batched PLE staging must stage exactly what the per-layer path stages.

`_BatchedPLEStager` replaces 48 calls to `NGramHasher.row_ids` and `HostNGramTable._gather_into` with
one stacked hash and a numpy gather. Both halves are integer bookkeeping, so the check is equality:
the same row ids, and the same bytes in the pinned buffers, for histories with and without EOS
boundaries, for a bf16 table and an FP8 one, and for ids that land in the divisor padding.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

torch = pytest.importorskip("torch")

from relicllm.models.qwen4_exp.layers import NGramHasher  # noqa: E402
from relicllm.models.qwen4_exp.verify_graph import _BatchedPLEStager  # noqa: E402
from relicllm.models.qwen4_exp.weights import HostNGramTable  # noqa: E402

EOS = 7


def _config(ngram_size: int = 3, heads_per_ngram: int = 2):
    heads = (ngram_size - 1) * heads_per_ngram
    primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167]

    def sizes(layer: int):
        chosen = [primes[(layer * heads + h) % len(primes)] for h in range(heads)]
        offsets, total = [], 0
        for size in chosen:
            offsets.append(total)
            total += size
        # Padding past the real rows, so some ids could land in the zeroed tail.
        return chosen, offsets, total + 17

    return SimpleNamespace(
        ngram_size=ngram_size,
        heads_per_ngram=heads_per_ngram,
        primary_eos_token_id=EOS,
        ngram_head_vocab_sizes=sizes,
        vocab_size=50_000,
        seed=1234,
    )


def _plan(config, layers: int, rows: int, dtype: torch.dtype):
    plan = []
    generator = torch.Generator().manual_seed(0)
    for layer in range(layers):
        hasher = NGramHasher(config, layer, torch.device("cpu"))
        total = hasher.padded_vocab_size
        per_shard = 37
        # Floor, so the ids in the last partial shard's span name a shard that does not exist and
        # stage zero rows -- the divisor-padding path -- for about one id in thirteen.
        shard_count = (total - 17) // per_shard
        values = torch.randn(shard_count * per_shard, 8, generator=generator)
        if dtype is torch.float8_e4m3fn:
            shards = [v.to(torch.float8_e4m3fn) for v in values.split(per_shard)]
            table = HostNGramTable(shards, device=torch.device("cpu"), dtype=torch.bfloat16,
                                   scale=torch.tensor(1.0))
        else:
            shards = [v.to(dtype) for v in values.split(per_shard)]
            table = HostNGramTable(shards, device=torch.device("cpu"), dtype=dtype)
        ids = hasher.row_ids(torch.zeros(1, config.ngram_size - 1 + rows, dtype=torch.long), rows)
        buffer = torch.empty(ids.numel(), 8, dtype=table._gather_dtype)
        plan.append((SimpleNamespace(hasher=hasher), table, buffer, tuple(ids.shape)))
    return plan


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("rows", [1, 2, 4, 5])
@pytest.mark.parametrize("layers,heads", [(1, 8), (6, 2)])
def test_batched_staging_matches_per_layer(dtype, rows, layers, heads):
    config = _config(heads_per_ngram=heads)
    plan = _plan(config, layers=layers, rows=rows, dtype=dtype)
    stager = _BatchedPLEStager(plan, rows)
    generator = torch.Generator().manual_seed(rows)
    for trial in range(20):
        history = torch.randint(0, 50_000, (1, config.ngram_size - 1 + rows), generator=generator)
        if trial % 3 == 0:
            history[0, trial % history.shape[1]] = EOS
        # `check` raises on any difference in the ids or in the staged bytes.
        stager.check(history)
        batched = stager.row_ids(history)
        for index, (ple, _table, _buffer, _shape) in enumerate(plan):
            assert torch.equal(batched[index], ple.hasher.row_ids(history, rows)[0])


@pytest.mark.parametrize("ngram_size", [2, 3, 4])
def test_numpy_hash_wraparound_and_every_eos_boundary(ngram_size):
    config = _config(ngram_size=ngram_size, heads_per_ngram=8)
    rows = 5
    plan = _plan(config, layers=3, rows=rows, dtype=torch.bfloat16)
    stager = _BatchedPLEStager(plan, rows)
    # These products exceed int64; Python big-integer arithmetic without re-signing would choose
    # different rows. Consecutive EOS boundaries and current-position EOS are separate cases.
    values = [0, 7, (1 << 62) + 33, (1 << 63) - 1, 248319, 19, 13, 5]
    history = torch.tensor([values[:ngram_size - 1 + rows]], dtype=torch.long)
    for eos_at in range(history.shape[1] + 1):
        candidate = history.clone()
        if eos_at < candidate.shape[1]:
            candidate[0, eos_at] = EOS
            if eos_at > 0:
                candidate[0, eos_at - 1] = EOS
        stager.check(candidate)


def test_numpy_staging_does_not_dispatch_torch_hash_operations(monkeypatch):
    config = _config(heads_per_ngram=8)
    plan = _plan(config, layers=1, rows=4, dtype=torch.bfloat16)
    stager = _BatchedPLEStager(plan, 4)
    history = torch.tensor([[11, 12, 13, 14, EOS, 16]], dtype=torch.long)
    stager.check(history)
    expected = [buffer.clone() for _ple, _table, buffer, _shape in plan]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the hot staging path dispatched a torch hash operation")

    monkeypatch.setattr(torch, "remainder", forbidden)
    monkeypatch.setattr(torch, "bitwise_xor", forbidden)
    monkeypatch.setattr(plan[0][0].hasher, "_shift_right_ignore_eos", forbidden)
    stager.stage(history)
    assert torch.equal(plan[0][2], expected[0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
