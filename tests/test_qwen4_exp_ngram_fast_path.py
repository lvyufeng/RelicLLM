"""The single-token n-gram path must equal the tensor path bit for bit.

The fast path exists because the tensor one builds a shifted copy of the history per n-gram
order to read one element out of each.  Equality is the whole contract, and the arithmetic is
int64 with wraparound -- a product that overflows changes sign, and `remainder` takes the
floor -- so the comparison has to include values large enough to overflow and EOS placements
that change which token a shift reads.
"""
from __future__ import annotations

import itertools
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import pytest
import torch

from relicllm.models.qwen4_exp.layers import NGramHasher, _signed64

EOS = 0
BIG = 151935


class _Config:
    ngram_size = 3
    heads_per_ngram = 2
    primary_eos_token_id = EOS
    seed = 7
    vocab_size = 151936

    def ngram_head_vocab_sizes(self, _index):
        # Two orders (bigram, trigram) x two heads, on distinct slices.
        return ([11, 13, 17, 19], [0, 11, 24, 41], 96)


def _hasher():
    return NGramHasher(_Config(), 0, torch.device("cpu"))


def _tensor_path(hasher, history, out_len=1):
    """The pre-existing implementation, kept here as the reference."""
    shifted = [hasher._shift_right_ignore_eos(history, s) for s in range(hasher.ngram_size)]
    blocks = []
    for ngram in range(2, hasher.ngram_size + 1):
        start = (ngram - 2) * hasher.heads_per_ngram
        end = start + hasher.heads_per_ngram
        mixed = shifted[0] * hasher.layer_multipliers[0]
        for position in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[position] * hasher.layer_multipliers[position])
        sizes = hasher.head_vocab_sizes[start:end].view(1, 1, -1)
        offsets = hasher.head_offsets[start:end].view(1, 1, -1)
        blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes) + offsets)
    return torch.cat(blocks, dim=-1)[:, -out_len:]


@pytest.mark.parametrize("combination", list(itertools.product([EOS, 1, 2, BIG], repeat=3)))
def test_every_eos_boundary_pattern_matches(combination):
    hasher = _hasher()
    history = torch.tensor([list(combination)], dtype=torch.long)
    fast = hasher.row_ids(history, 1)
    assert fast.shape == (1, 1, 4)
    assert torch.equal(fast, _tensor_path(hasher, history))


def test_batched_rows_keep_their_own_answers():
    hasher = _hasher()
    history = torch.tensor([[EOS, BIG, 7], [7, BIG, BIG], [1, 1, 1]], dtype=torch.long)
    fast = hasher.row_ids(history, 1)
    assert fast.shape == (3, 1, 4)
    for row in range(3):
        single = hasher.row_ids(history[row:row + 1], 1)
        assert torch.equal(fast[row:row + 1], single)


def test_a_longer_history_keeps_the_tensor_path():
    hasher = _hasher()
    history = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.long)
    assert torch.equal(hasher.row_ids(history, 1), _tensor_path(hasher, history))


def test_more_than_one_output_keeps_the_tensor_path():
    hasher = _hasher()
    history = torch.tensor([[1, 2, 3]], dtype=torch.long)
    # out_len > 1 cannot take the one-token path, and the reference is what it must return.
    assert torch.equal(hasher.row_ids(history, 2), _tensor_path(hasher, history, 2))


@pytest.mark.parametrize("factor", [1, 2, 4, 8])
def test_signed_wrapping_matches_int64_arithmetic(factor):
    # A product that overflows int64 must land on the value the tensor path holds, because a
    # sign flip changes what floor modulo returns and therefore which row is read.  The check
    # is against torch itself rather than against a second Python formula.
    huge = (1 << 62) + 12345
    product = huge * factor
    # Let torch do the multiply so the wrap is the hardware's, not a second Python formula.
    expected = int((torch.tensor([huge], dtype=torch.long) * factor).item())
    assert _signed64(product) == expected
    if product > (1 << 63) - 1:
        # The interesting cases are the ones where the mathematical product is not the value
        # held, which is what makes the wrap observable at all.
        assert expected != product, "this case is only interesting when the product wraps"


def test_rows_are_in_range_for_their_head_slice():
    hasher = _hasher()
    sizes = hasher.head_vocab_sizes.tolist()
    offsets = hasher.head_offsets.tolist()
    for combination in itertools.product([EOS, 1, BIG], repeat=3):
        history = torch.tensor([list(combination)], dtype=torch.long)
        rows = hasher.row_ids(history, 1)[0, 0].tolist()
        for head, value in enumerate(rows):
            assert offsets[head] <= value < offsets[head] + sizes[head]


class _ShardedTable:
    """`HostNGramTable._gather_into`'s routing without a checkpoint behind it."""

    from relicllm.models.qwen4_exp.weights import HostNGramTable as _H

    _gather_into = _H._gather_into
    _gather_rows_small = _H._gather_rows_small

    def __init__(self, shards, rows_per_shard):
        self._gather_shards = shards
        self.rows_per_shard = rows_per_shard
        self.head_dim = shards[0].shape[1]


def _vectorised_gather(table, flat, out):
    """The shard-by-shard form the small path replaced, kept as the reference."""
    shard_idx = torch.div(flat, table.rows_per_shard, rounding_mode="floor")
    local_idx = flat - shard_idx * table.rows_per_shard
    for shard in shard_idx.unique().tolist():
        picks = (shard_idx == shard).nonzero(as_tuple=True)[0]
        if shard >= len(table._gather_shards):
            out[picks] = 0
        else:
            out[picks] = table._gather_shards[shard].index_select(0, local_idx[picks])


@pytest.mark.parametrize("rows", [
    [0, 1, 2, 3],
    [63, 64, 127, 128],          # shard boundaries
    [191, 192, 5, 130],          # 192 and up are divisor padding
    [5, 5, 5, 5],                # duplicates
    [0],
])
def test_small_gather_matches_the_shard_by_shard_form(rows):
    torch.manual_seed(0)
    table = _ShardedTable([torch.randn(64, 8) for _ in range(3)], rows_per_shard=64)
    flat = torch.tensor(rows, dtype=torch.long)
    got = torch.full((len(rows), 8), -7.0)
    want = torch.full((len(rows), 8), -7.0)
    table._gather_into(flat, got)
    _vectorised_gather(table, flat, want)
    assert torch.equal(got, want)


def test_a_large_gather_still_uses_the_vectorised_form():
    torch.manual_seed(0)
    table = _ShardedTable([torch.randn(64, 8) for _ in range(3)], rows_per_shard=64)
    flat = torch.randint(0, 200, (512,), dtype=torch.long)
    got = torch.full((512, 8), -7.0)
    want = torch.full((512, 8), -7.0)
    table._gather_into(flat, got)
    _vectorised_gather(table, flat, want)
    assert torch.equal(got, want)
