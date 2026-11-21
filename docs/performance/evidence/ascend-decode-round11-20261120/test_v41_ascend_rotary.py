"""The NPU rotary form: a real `(cos, sin)` table instead of a complex64 one.

Ascend has no complex dtype, so `precompute_freqs_cis` builds the same host `torch.polar` table and
reinterprets it with `view_as_real` into `[s, d // 2, 2]` float32 for an NPU destination.
`apply_rotary_emb` dispatches on the table's dtype, so the old-device path is untouched.

What these tests can and cannot establish:

* the pair table is bit-identical to the complex table it was cut from -- that part is exact, and it
  is asserted exactly;
* the pair rotation equals the complex rotation on **BF16** inputs, which are the model's activation
  dtype, exactly -- also asserted exactly;
* on **FP32** inputs the two forms are not the same sequence of FP32 operations (ATen's complex
  multiply against two real products), so they may differ in the last place. That is asserted as a
  bound and not as equality, and the residual risk -- an FP32 value exactly on a BF16 tie -- is
  stated rather than papered over;
* the NPU case is opt-in through `RELIC_ASCEND_ROTARY_DEVICE` and skips without a leased card. Its
  assertions are about the device form and its agreement with the CPU oracle, not about a forward
  pass or any model output.
"""

from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")

from relicllm.models.deepseek_v4_1.attention import (  # noqa: E402
    apply_rotary_emb,
    apply_rotary_emb_pairs,
    precompute_freqs_cis,
)

DIM = 8  # a rope head width: 4 rotated pairs
SEQ = 6
NPU_ENV = "RELIC_ASCEND_ROTARY_DEVICE"


def _table(seqlen: int = SEQ, dim: int = DIM, *, original_seq_len: int = 0, pairs: bool):
    """The table `Attention` would build, forced to one of the two forms."""
    return precompute_freqs_cis(
        dim,
        seqlen,
        original_seq_len,
        10_000.0,
        40.0,
        32,
        1,
        None,
        pairs=pairs,
    )


def _complex_rotation(x: torch.Tensor, table: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """The old body of `apply_rotary_emb`, kept here as the oracle it has to stay equal to."""
    y = x
    x = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        table = table.conj()
    if x.ndim == 3:
        table = table.view(1, x.size(1), x.size(-1))
    else:
        table = table.view(1, x.size(1), 1, x.size(-1))
    y.copy_(torch.view_as_real(x * table).flatten(-2))
    return y


def _pairs_are_exact_cos_sin(table: torch.Tensor, complex_table: torch.Tensor) -> bool:
    return torch.equal(table, torch.view_as_real(complex_table).contiguous())


@pytest.mark.parametrize("shape", [(1, 1, DIM), (1, 1, 3, DIM)])
@pytest.mark.parametrize("inverse", [False, True])
def test_one_selected_positional_row_rotates_a_decode_token(shape, inverse):
    g = torch.Generator().manual_seed(708)
    x = torch.randn(shape, generator=g).to(torch.bfloat16)
    complex_row = _table(pairs=False)[3]
    pair_row = _table(pairs=True)[3]
    expected = _complex_rotation(x.clone(), complex_row, inverse)
    actual = x.clone()
    assert apply_rotary_emb(actual, pair_row, inverse) is actual
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


# -- the table ---------------------------------------------------------------------------------


def test_the_default_table_is_still_complex():
    table = precompute_freqs_cis(DIM, SEQ, 0, 10_000.0, 40.0, 32, 1)
    assert torch.is_complex(table)
    assert table.shape == (SEQ, DIM // 2)
    assert table.dtype == torch.complex64


def test_the_pair_table_carries_the_complex_table_bits():
    complex_table = _table(pairs=False)
    pair_table = _table(pairs=True)
    assert not torch.is_complex(pair_table)
    assert pair_table.dtype == torch.float32
    assert pair_table.shape == (SEQ, DIM // 2, 2)
    assert pair_table.is_contiguous()
    assert _pairs_are_exact_cos_sin(pair_table, complex_table)


def test_the_yarn_table_converts_the_same_way():
    """The compressed-position variant takes the ramp branch; the conversion must not care."""
    complex_table = _table(original_seq_len=128, pairs=False)
    pair_table = _table(original_seq_len=128, pairs=True)
    assert _pairs_are_exact_cos_sin(pair_table, complex_table)


# -- the rotation ------------------------------------------------------------------------------


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("shape", [(2, SEQ, DIM), (2, SEQ, 3, DIM)])
def test_bf16_pairs_match_the_complex_rotation_exactly(shape, inverse):
    """The model's dtype: this is the parity claim, and it is exact."""
    torch.manual_seed(7)
    pair_table, complex_table = _table(pairs=True), _table(pairs=False)
    x = torch.randn(*shape, dtype=torch.bfloat16)
    expected = _complex_rotation(x.clone(), complex_table, inverse)
    actual = apply_rotary_emb(x.clone(), pair_table, inverse)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("inverse", [False, True])
def test_fp32_pairs_agree_within_the_arithmetic_seam(inverse):
    """Not equality: ATen's complex multiply is a different FP32 expression than two real products.

    The bound is a couple of ulps of the result, which is what a reassociated FP32 product costs.
    A tighter assertion here would be a claim about ATen's complex kernel rather than about this
    function.
    """
    torch.manual_seed(11)
    pair_table, complex_table = _table(pairs=True), _table(pairs=False)
    x = torch.randn(2, SEQ, DIM, dtype=torch.float32)
    expected = _complex_rotation(x.clone(), complex_table, inverse)
    actual = apply_rotary_emb(x.clone(), pair_table, inverse)
    scale = max(float(expected.abs().max()), 1.0)
    bound = 4 * torch.finfo(torch.float32).eps * scale
    assert float((actual - expected).abs().max()) <= bound


def test_the_dispatch_does_not_change_the_complex_path():
    """A complex table must run the old body, byte for byte."""
    torch.manual_seed(3)
    complex_table = _table(pairs=False)
    x = torch.randn(2, SEQ, 3, DIM, dtype=torch.bfloat16)
    expected = _complex_rotation(x.clone(), complex_table)
    actual = apply_rotary_emb(x.clone(), complex_table)
    assert torch.equal(actual, expected)


def test_the_rotation_is_in_place_and_returns_the_same_tensor():
    torch.manual_seed(5)
    pair_table = _table(pairs=True)
    storage = torch.randn(2, SEQ, 2 * DIM, dtype=torch.bfloat16)
    rotated = storage[..., -DIM:]  # the tail slice the model actually rotates
    before = storage.clone()
    returned = apply_rotary_emb(rotated, pair_table)
    assert returned is rotated
    assert torch.equal(storage[..., :-DIM], before[..., :-DIM]), "the width outside rope moved"
    assert not torch.equal(storage[..., -DIM:], before[..., -DIM:])


@pytest.mark.parametrize("inverse", [False, True])
def test_pair_rows_slice_like_the_complex_table(inverse):
    """`Attention.forward` hands over `freqs_cis[pos.span(seqlen)]`, a row slice, not the table."""
    torch.manual_seed(13)
    full = _table(seqlen=SEQ + 4, pairs=True)
    complex_full = _table(seqlen=SEQ + 4, pairs=False)
    rows = slice(2, 2 + SEQ)
    x = torch.randn(1, SEQ, DIM, dtype=torch.bfloat16)
    expected = _complex_rotation(x.clone(), complex_full[rows], inverse)
    actual = apply_rotary_emb(x.clone(), full[rows], inverse)
    assert torch.equal(actual, expected)


def test_inverse_undoes_the_rotation_within_bf16():
    torch.manual_seed(17)
    pair_table = _table(pairs=True)
    x = torch.randn(1, SEQ, DIM, dtype=torch.bfloat16)
    rotated = apply_rotary_emb(x.clone(), pair_table)
    back = apply_rotary_emb(rotated, pair_table, inverse=True)
    # Two narrowings to bf16 happen on the way, so this is a tolerance and not an identity.
    assert torch.allclose(back.float(), x.float(), atol=2e-2, rtol=2e-2)


# -- refusals ----------------------------------------------------------------------------------


def test_a_mismatched_or_odd_width_is_refused():
    pair_table = _table(pairs=True)
    with pytest.raises(ValueError, match="odd"):
        apply_rotary_emb_pairs(torch.zeros(1, SEQ, DIM - 1, dtype=torch.bfloat16), pair_table)
    with pytest.raises(ValueError, match="rotary table"):
        apply_rotary_emb_pairs(torch.zeros(1, SEQ, DIM, dtype=torch.bfloat16), pair_table[:, :, :1])
    with pytest.raises(ValueError, match="rows"):
        apply_rotary_emb_pairs(torch.zeros(1, SEQ + 1, DIM, dtype=torch.bfloat16), pair_table)
    with pytest.raises(ValueError, match="-dimensional"):
        apply_rotary_emb_pairs(torch.zeros(SEQ, DIM, dtype=torch.bfloat16), pair_table)


# -- the device case ---------------------------------------------------------------------------


def _npu_device():
    name = os.environ.get(NPU_ENV)
    if not name:
        pytest.skip(f"set {NPU_ENV}=npu:0 on a leased card to run the device case")
    import torch_npu  # noqa: F401  (registers the device)

    return torch.device(name)


def test_the_npu_table_is_real_and_the_rotation_matches_the_cpu_oracle(monkeypatch):
    device = _npu_device()
    pair_table = precompute_freqs_cis(
        DIM, SEQ, 0, 10_000.0, 40.0, 32, 1, device, pairs=None
    )
    assert pair_table.device.type == "npu"
    assert not torch.is_complex(pair_table), "the NPU table must not be complex"
    assert pair_table.dtype == torch.float32 and pair_table.shape == (SEQ, DIM // 2, 2)

    torch.manual_seed(23)
    host_x = torch.randn(2, SEQ, 3, DIM, dtype=torch.bfloat16)
    cpu_pair = apply_rotary_emb(host_x.clone(), _table(pairs=True))
    cpu_complex = _complex_rotation(host_x.clone(), _table(pairs=False))

    # The NPU path must not reach a complex op at all: the dispatch reads the dtype, and the pair
    # rotation is real arithmetic. Fail loudly if anything calls one.
    def forbidden(*args, **kwargs):
        raise AssertionError("a complex op ran on the NPU rotary path")

    monkeypatch.setattr(torch, "view_as_complex", forbidden)
    monkeypatch.setattr(torch, "view_as_real", forbidden)

    stream = torch.npu.Stream(device=device)
    before_device = torch.npu.current_device()
    with torch.npu.stream(stream):
        device_x = host_x.to(device)
        actual = apply_rotary_emb(device_x, pair_table)
        consumed = actual + torch.zeros((), dtype=actual.dtype, device=device)
        assert torch.npu.current_device() == before_device, "rotary changed the current device"
    stream.synchronize()
    monkeypatch.undo()

    actual_cpu = consumed.to("cpu")
    # Same FP32 expression on both devices, compared at the model's dtype.
    assert torch.equal(actual_cpu, cpu_pair), "the NPU rotation disagrees with the CPU pair form"
    # And that form is the complex one at bf16, which is the CPU-verified claim above.
    assert torch.equal(actual_cpu, cpu_complex)

def test_the_single_write_matches_two_strided_writes_bit_for_bit():
    """The one-copy form is a data move of the same values, so it cannot change a bit.

    `apply_rotary_emb_pairs` used to write the rotated pairs through the strided `[..., 0::2]` and
    `[..., 1::2]` views. It now interleaves them and copies once, because a `copy_` from a strided
    source is a format conversion on Ascend. This reconstructs the two-write form for the same
    inputs and asserts BF16 bit equality, so the shortcut cannot become a different rotation.
    """
    torch = pytest.importorskip("torch")
    from relicllm.models.deepseek_v4_1.attention import apply_rotary_emb_pairs

    def two_writes(x, freqs, inverse=False):
        cosine = freqs[..., 0].to(torch.float32)
        sine = freqs[..., 1].to(torch.float32)
        if inverse:
            sine = -sine
        if x.ndim == 3:
            cosine, sine = cosine.unsqueeze(0), sine.unsqueeze(0)
        else:
            cosine, sine = cosine.unsqueeze(0).unsqueeze(2), sine.unsqueeze(0).unsqueeze(2)
        even, odd = x[..., 0::2], x[..., 1::2]
        ef, of = even.float(), odd.float()
        new_even = ef * cosine - of * sine
        new_odd = ef * sine + of * cosine
        even.copy_(new_even)
        odd.copy_(new_odd)
        return x

    torch.manual_seed(7)
    for shape, ndim in (((1, 1, 8, 64), 4), ((1, 1, 64), 3), ((1, 4, 8, 64), 4)):
        pairs = shape[-1] // 2
        x = torch.randn(*shape, dtype=torch.bfloat16)
        freqs = torch.randn(shape[1], pairs, 2, dtype=torch.float32)
        for inverse in (False, True):
            want = two_writes(x.clone(), freqs.clone(), inverse)
            got = apply_rotary_emb_pairs(x.clone(), freqs.clone(), inverse)
            assert torch.equal(want.view(torch.int16), got.view(torch.int16)), (shape, inverse)
