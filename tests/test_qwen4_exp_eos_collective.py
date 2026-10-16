"""The end-of-sequence exit has to be agreed, not local.

`generate` is entered by every rank, and `on_step` is the collective that ends it -- the caller
broadcasts rank 0's decision, which is what stops one rank leaving its peers inside a per-layer
`all_reduce`. The EOS exit used to bypass that call and break on its own, and the token it tests is
each rank's own `argmax`, so a step where the ranks disagree let one rank leave while the others
walked into the next `on_step` and sat in its broadcast. The measured failure was
`WorkNCCL(OpType=BROADCAST, NumelIn=1, NumelOut=1, Timeout(ms)=600000) ran for 600018 ms` followed by
`taking the entire process down`, on 8xV100 and 8xRTX 4090 alike.

These tests pin the contract rather than the accident: with `local_eos` given, the loop must not
break on EOS by itself, must record the finding, and must let the caller's predicate be the thing
that ends it.
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import torch

from relicllm.models.qwen4_exp import runtime


class _ScriptedModel:
    """A model whose next token is dictated, so the loop can be driven without a checkpoint.

    `generate` only asks the model for `make_cache`, `forward` and `profiler`, and it reads the
    token off `gather_logits(...).argmax(...)`. Returning a one-hot row makes that argmax exact, so
    the script is the token stream and nothing else is in the way.
    """

    def __init__(self, tokens: list[int], vocab: int = 16) -> None:
        self._tokens = list(tokens)
        self._index = 0
        self._vocab = vocab
        self.profiler = None

    def make_cache(self, *, batch_size: int, max_seq_len: int):
        return object()

    def forward(self, ids, *, cache=None, past_len=0, last_token_only=False):
        # `(batch, seq, vocab)`, because the loop reads `step[:, -1]` and then `argmax(-1)`. A
        # 2-D return would make that argmax a scalar and the scripted token unreachable.
        row = torch.zeros(ids.shape[0], ids.shape[1], self._vocab)
        token = self._tokens[min(self._index, len(self._tokens) - 1)]
        self._index += 1
        row[:, -1, token] = 10.0
        return row


def _ctx():
    return runtime.TPContext(0, 1, torch.device("cpu"), False)


def _run(model, *, budget, eos, local_eos, on_step):
    return runtime.generate(
        model,
        _ctx(),
        torch.tensor([[1, 2]], dtype=torch.long),
        max_new_tokens=budget,
        chunk_size=8,
        eos_token_ids=eos,
        on_step=on_step,
        local_eos=local_eos,
    )


def test_eos_does_not_end_the_loop_on_its_own_when_a_collective_is_present():
    """The regression: the loop must keep stepping so every rank reaches the same `on_step`.

    The script emits the EOS token on the first step and a different token after it, so a loop that
    broke locally would stop at one token while one that defers would take the second step and be
    stopped by the predicate. The step count is the assertion.
    """
    model = _ScriptedModel([7, 3, 3, 3])
    local_eos = [False]
    steps: list[int] = []

    def on_step() -> bool:
        steps.append(1)
        return local_eos[0]

    tokens, stats = _run(model, budget=4, eos=(7,), local_eos=local_eos, on_step=on_step)

    assert stats["stopped"] == "eos"
    # Two predicate calls: one that ran the first step, one that read the flag and ended the loop.
    # A local break would have made it one.
    assert len(steps) == 2, f"the collective was consulted {len(steps)} times, expected 2"
    assert tokens == [7], "the loop took a step it should not have after agreeing to stop"


def test_the_flag_is_what_names_the_stop_reason():
    """`cancel` and `eos` ride the same broadcast, so the flag is what tells them apart."""
    model = _ScriptedModel([7, 3, 3])
    tokens, stats = _run(model, budget=4, eos=(7,), local_eos=[False], on_step=lambda: True)
    assert stats["stopped"] == "cancel"
    assert tokens == []


def test_without_the_flag_the_old_local_exit_still_stands():
    """World size 1 and the CLI entry point pass no flag, and must keep breaking on EOS."""
    model = _ScriptedModel([7, 3, 3])
    calls: list[int] = []
    tokens, stats = _run(
        model, budget=4, eos=(7,), local_eos=None, on_step=lambda: (calls.append(1), False)[1]
    )
    assert stats["stopped"] == "eos"
    assert tokens == [7]
    assert len(calls) == 1, "the local path still asks once, then breaks on the token"