"""The per-position recurrent state a speculative rewind needs.

A verify round runs K positions in one forward and then keeps only an accepted prefix, so the linear
layers have to be put back to the state *after* position k rather than after the whole forward. The
fused scan already passes through every one of those states; `qwen4_exp_gated_delta_bf16_forward_states`
hands them back.

Two properties make that usable, and this module is what holds them:

1. Recording changes nothing. The output and the final state are bit-identical to the plain entry
   point, so asking for the states cannot move a token.
2. Each recorded state is the real one. The state after position j equals what a scan of the first
   j+1 rows from the same starting state produces, which is what makes a rewind exact rather than
   approximately right.

Skipped without a card and a built op, like the rest of the kernel-facing tests: there is no soft
path to compare against.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch

from relicllm.models.qwen4_exp.attention import cuda_gated_delta_rule


ROWS = 5
KEY_HEADS = 2
VALUE_HEADS = 4
DIM = 128


def _needs_kernel():
    if not torch.cuda.is_available():
        return "needs a CUDA device"
    if not torch.cuda.is_bf16_supported():
        return "needs a card that supports bf16"
    from relicllm.runtime.ops import load_ops

    ops = load_ops()
    if ops is None or not hasattr(ops, "qwen4_exp_gated_delta_bf16_forward_states"):
        return "needs the built gated-delta states entry point"
    return None


pytestmark = pytest.mark.skipif(
    _needs_kernel() is not None, reason=str(_needs_kernel() or "")
)


def _inputs(rows=ROWS, seed=0):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    query = torch.randn(1, rows, KEY_HEADS, DIM, device=device, dtype=torch.bfloat16)
    key = torch.randn(1, rows, KEY_HEADS, DIM, device=device, dtype=torch.bfloat16)
    value = torch.randn(1, rows, VALUE_HEADS, DIM, device=device, dtype=torch.bfloat16)
    # The gate is per value head, not per key head: the kernel checks its last two dims against
    # `value`, which is what the decay applies to.
    gate = torch.randn(1, rows, VALUE_HEADS, device=device, dtype=torch.float32) * 0.1
    beta = torch.rand(1, rows, VALUE_HEADS, device=device, dtype=torch.bfloat16)
    state = torch.randn(1, VALUE_HEADS, DIM, DIM, device=device, dtype=torch.float32) * 0.05
    return query, key, value, gate, beta, state


def test_recording_does_not_change_the_output_or_final_state():
    query, key, value, gate, beta, state = _inputs()
    plain = cuda_gated_delta_rule(
        query, key, value, gate, beta, initial_state=state, with_states=False
    )
    assert plain is not None, "the fused scan declined these inputs"
    output_plain, state_plain = plain

    recorded = cuda_gated_delta_rule(
        query, key, value, gate, beta, initial_state=state, with_states=True
    )
    assert recorded is not None
    output_rec, state_rec, per_position = recorded

    assert torch.equal(output_plain, output_rec), "output moved when the states were requested"
    assert torch.equal(state_plain, state_rec), "final state moved when the states were requested"
    assert per_position.shape == (ROWS, VALUE_HEADS, DIM, DIM)
    assert per_position.dtype is torch.float32
    # The last recorded state is the one that comes out. Indexing a row drops the batch axis, and
    # the state keeps it, so put it back rather than compare tensors of different rank.
    assert torch.equal(per_position[-1].unsqueeze(0), state_rec)


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_a_recorded_state_is_the_state_a_shorter_scan_reaches(position):
    """The property a rewind depends on: state[j] is what stopping after j+1 rows would produce."""
    query, key, value, gate, beta, state = _inputs()
    recorded = cuda_gated_delta_rule(
        query, key, value, gate, beta, initial_state=state, with_states=True
    )
    assert recorded is not None
    _, _, per_position = recorded

    keep = position + 1
    shorter = cuda_gated_delta_rule(
        query[:, :keep],
        key[:, :keep],
        value[:, :keep],
        gate[:, :keep],
        beta[:, :keep],
        initial_state=state,
        with_states=False,
    )
    assert shorter is not None, "the shorter scan declined"
    _, state_shorter = shorter

    assert torch.equal(per_position[position].unsqueeze(0), state_shorter), (
        f"recorded state {position} is not the state a {keep}-row scan reaches"
    )


@pytest.mark.parametrize("rows", [2, 4])
@pytest.mark.parametrize("value_heads", [4, 6])
def test_recorded_scan_matches_serial_single_token_outputs_and_states(rows, value_heads, monkeypatch):
    """Clear the scan itself of one-row-vs-K differences before changing upstream projections.

    Six value heads over two key heads is the released model's TP8 shard geometry; four also
    exercises the smaller synthetic shape used by the original state-log tests.
    """
    monkeypatch.setitem(globals(), "VALUE_HEADS", value_heads)
    query, key, value, gate, beta, state = _inputs(rows=rows, seed=37)
    recorded = cuda_gated_delta_rule(query, key, value, gate, beta, initial_state=state, with_states=True)
    assert recorded is not None
    output, final, per_position = recorded
    current = state
    serial_outputs = []
    for position in range(rows):
        one = cuda_gated_delta_rule(
            query[:, position:position + 1], key[:, position:position + 1],
            value[:, position:position + 1], gate[:, position:position + 1],
            beta[:, position:position + 1], initial_state=current, with_states=False,
        )
        assert one is not None
        out, current = one
        serial_outputs.append(out)
        assert torch.equal(per_position[position].unsqueeze(0), current)
    assert torch.equal(output, torch.cat(serial_outputs, dim=1))
    assert torch.equal(final, current)
