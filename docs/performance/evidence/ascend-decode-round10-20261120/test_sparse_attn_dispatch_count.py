"""The torch sparse attention's dispatch count, and the arithmetic it must not trade away.

`_sparse_attn_torch` is a transcription of the released kernel, so its *arithmetic* is pinned by
`test_sparse_attn_reference_arithmetic.py` against a recording of that kernel. What that test cannot
see is how many dispatches the transcription spends to get there, and on a decode step the block loop
used to rebuild its operand layouts twice per layer per block: `einsum` lowers to
`permute`/`unsqueeze`/`view`/`slice` traffic that is metadata-only and that a fused kernel would not
pay at all.

The layouts are now built once per call and the block loop drives `torch.bmm` on the folded rows
directly. That is the same folded shape, the same reduction axis and the same operands the `einsum`
produced, so the tensor-core accumulation order is unchanged and the output is bit-identical -- which
is the property the first test in this file measures, and the reason the change is allowed to exist.

The number below is a *dispatch* count, not a promise about throughput. On the Ascend box these ops
are metadata-only: they cost host dispatch and emit no kernel, and the host is not this model's
critical path, so removing them measured as no change in TPS. The count is pinned anyway so that a
future edit cannot quietly reintroduce the traffic, and so that the "metadata-only" claim has a
number attached to it.
"""

from __future__ import annotations

import collections

import pytest

torch = pytest.importorskip("torch")

from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

from relic_core.kernels.ops import _SPARSE_ATTN_BLOCK, _sparse_attn_torch  # noqa: E402


class _Counter(TorchDispatchMode):
    def __init__(self) -> None:
        self.calls: collections.Counter = collections.Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls[func._schema.name] += 1
        return func(*args, **(kwargs or {}))


def _decode_inputs(filled: int, heads: int = 8, topk: int = 128):
    """One decode row at the released geometry: head_dim 512, a partly filled window."""
    torch.manual_seed(1234)
    q = torch.randn(1, 1, heads, 512, dtype=torch.bfloat16)
    kv = torch.randn(1, topk, 512, dtype=torch.bfloat16)
    sink = torch.randn(heads, dtype=torch.float32)
    idxs = torch.arange(topk, dtype=torch.int32).view(1, 1, topk).contiguous()
    idxs[:, :, filled:] = -1
    return q, kv, sink, idxs


def _previous_form(q, kv, attn_sink, topk_idxs, softmax_scale):
    """The implementation before the layouts were hoisted, kept as the arithmetic reference."""
    b, s, h, d = q.shape
    topk = topk_idxs.size(-1)
    safe_idxs = topk_idxs.clamp_min(0)
    batch_idx = torch.arange(b, device=q.device).view(b, 1, 1).expand_as(safe_idxs)
    gathered = kv[batch_idx, safe_idxs].to(torch.float32)
    valid = topk_idxs >= 0
    q_f32 = q.to(torch.float32)
    scores_max = torch.full((b, s, h), -1e30, device=q.device, dtype=torch.float32)
    sum_exp = torch.zeros((b, s, h), device=q.device, dtype=torch.float32)
    acc_o = torch.zeros((b, s, h, d), device=q.device, dtype=torch.float32)
    for start in range(0, topk, _SPARSE_ATTN_BLOCK):
        stop = min(start + _SPARSE_ATTN_BLOCK, topk)
        scores = torch.einsum("bshd,bsnd->bshn", q_f32, gathered[:, :, start:stop, :]) * softmax_scale
        scores = scores.masked_fill(~valid[:, :, start:stop].unsqueeze(2), float("-inf"))
        max_prev = scores_max
        scores_max = torch.maximum(max_prev, scores.amax(dim=-1))
        scores_scale = torch.exp(max_prev - scores_max)
        numerator = torch.exp(scores - scores_max.unsqueeze(-1))
        sum_exp = sum_exp * scores_scale + numerator.sum(dim=-1)
        acc_o = acc_o * scores_scale.unsqueeze(-1) + torch.einsum(
            "bshn,bsnd->bshd",
            numerator.to(torch.bfloat16).to(torch.float32),
            gathered[:, :, start:stop, :],
        )
    sum_exp = sum_exp + torch.exp(attn_sink.to(torch.float32).view(1, 1, h) - scores_max)
    return (acc_o / sum_exp.unsqueeze(-1)).to(q.dtype)


@pytest.mark.parametrize(
    "label,kwargs",
    [
        ("window partly filled", dict(filled=40)),
        ("window full", dict(filled=128)),
        ("sixteen heads", dict(filled=64, heads=16)),
    ],
)
def test_the_hoisted_layouts_do_not_change_a_bit(label, kwargs):
    """The property the hoist is allowed to have: same inputs, same bf16 output bits."""
    q, kv, sink, idxs = _decode_inputs(**kwargs)
    want = _previous_form(q, kv, sink, idxs, 0.135)
    got = _sparse_attn_torch(q, kv, sink, idxs, 0.135)
    assert torch.equal(want.view(torch.int16), got.view(torch.int16)), label


@pytest.mark.parametrize("rows,topk,filled", [(1, 128, 40), (2, 128, 40), (1, 512, 40)])
def test_a_call_spends_fewer_dispatches_than_the_previous_form(rows, topk, filled):
    """Layouts built once beat layouts rebuilt per block, and the gap is a real number."""
    torch.manual_seed(0)
    q = torch.randn(1, rows, 8, 512, dtype=torch.bfloat16)
    kv = torch.randn(1, topk, 512, dtype=torch.bfloat16)
    sink = torch.randn(8, dtype=torch.float32)
    idxs = torch.arange(topk, dtype=torch.int32).view(1, 1, topk).expand(1, rows, topk).contiguous()
    idxs[:, :, filled:] = -1
    args = (q, kv, sink, idxs, 0.135)

    before, after = _Counter(), _Counter()
    with before:
        _previous_form(*args)
    with after:
        _sparse_attn_torch(*args)
    n_before, n_after = sum(before.calls.values()), sum(after.calls.values())

    assert n_after < n_before, (n_before, n_after)
    # the metadata-only traffic is what goes away; no matmul may
    for name in ("aten::bmm", "aten::mul", "aten::exp", "aten::amax", "aten::sum"):
        assert after.calls[name] == before.calls[name], name
    assert after.calls["aten::permute"] == 0


def test_the_decode_call_spends_no_permute_and_forty_fewer_dispatches():
    """The measured decode-step numbers, pinned: 110 dispatches became 70."""
    q, kv, sink, idxs = _decode_inputs(filled=40)
    args = (q, kv, sink, idxs, 0.135)
    before, after = _Counter(), _Counter()
    with before:
        _previous_form(*args)
    with after:
        _sparse_attn_torch(*args)
    assert sum(before.calls.values()) == 110
    assert sum(after.calls.values()) == 70
    assert after.calls["aten::permute"] == 0
