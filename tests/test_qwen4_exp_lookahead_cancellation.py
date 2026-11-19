"""Cooperative cancellation contracts for the lookahead (multi-token round) decoder.

The fake model and fake recordings come from the lookahead contract fixture: they emit a known
greedy sequence on CPU, so every expected token list and every poll count is derived from the
control flow rather than from the implementation's own output. These tests prove the host-side
cancellation protocol -- where the predicate is asked, how an EOS is deferred, and that no trunk or
head work runs after a confirmed stop. They are not a GPU collective test and cannot prove that a
real eight-rank deployment never disagrees; that is a separate measured check.
"""
from __future__ import annotations

import pytest

from relicllm.models.qwen4_exp import runtime
from test_qwen4_exp_lookahead_contract import _install, _run


def _lrun(model, ctx, *, max_new=8, drafts=3, captured=False, eos=(), callback=None,
          on_step=None, local_eos=None):
    return runtime.generate_mtp_draft_lookahead(
        model, ctx, __import__("torch").tensor([[7, 8]]),
        max_new_tokens=max_new, draft_tokens=drafts, verify_graph=captured,
        eos_token_ids=eos, on_token=callback, on_step=on_step, local_eos=local_eos,
    )


def test_default_none_still_derives_the_same_stop_reasons(monkeypatch):
    """With neither argument the old control flow and the old `stopped` names must stand."""
    model, ctx, _clock = _install(monkeypatch)
    output, stats = _run(model, ctx, max_new=4)
    assert output == [10, 11, 12, 13]
    assert stats["stopped"] == "length"

    model, ctx, _clock = _install(monkeypatch, eos_position=3)
    seen = []
    output, stats = _run(model, ctx, max_new=6, eos=(3,), callback=seen.append)
    assert output == seen == [10, 3]
    assert stats["stopped"] == "eos"


@pytest.mark.parametrize("captured", [False, True])
def test_cancel_before_the_first_emitted_token_does_no_decode_work(monkeypatch, captured):
    model, ctx, _clock = _install(monkeypatch, captured=captured)
    calls: list[int] = []
    seen: list[int] = []

    def on_step() -> bool:
        calls.append(1)
        return True

    output, stats = _lrun(model, ctx, max_new=8, captured=captured, callback=seen.append,
                          on_step=on_step, local_eos=[False])
    assert output == seen == []
    assert stats["stopped"] == "cancel"
    assert stats["decode_tokens"] == 0.0
    assert len(calls) == 1, "one collective poll, then nothing else may run"
    # The prompt still had to be prefilled; nothing after that poll may consume a position.
    assert model.calls == [(0, 2)], "no decode forward may run after a confirmed cancel"


@pytest.mark.parametrize("captured", [False, True])
def test_zero_budget_returns_before_asking_the_predicate(monkeypatch, captured):
    model, ctx, _clock = _install(monkeypatch, captured=captured)
    calls: list[int] = []

    def on_step() -> bool:
        calls.append(1)
        return False

    output, stats = _lrun(model, ctx, max_new=0, captured=captured, on_step=on_step,
                          local_eos=[False])
    assert output == [] and stats["stopped"] == "length"
    assert calls == [], "a request with nothing to generate has no round boundary to poll"


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("eos_position", [2, 3, 4, 5, 6])
def test_local_eos_defers_to_the_collective_poll_and_emits_the_same_prefix(
        monkeypatch, captured, eos_position):
    """The EOS token is still delivered, but the exit goes through the shared predicate."""
    model, ctx, _clock = _install(monkeypatch, captured=captured, eos_position=eos_position)
    seen: list[int] = []

    def on_token(token):
        seen.append(token)

    def on_step() -> bool:
        # The caller's real predicate is a collective; here the flag *is* the decision.
        return local_eos[0]

    local_eos = [False]
    output, stats = _lrun(model, ctx, max_new=10, captured=captured, eos=(3,),
                          callback=seen.append, on_step=on_step, local_eos=local_eos)
    expected = list(range(10, eos_position + 8)) + [3]
    assert output == seen == expected
    assert stats["stopped"] == "eos"
    assert local_eos[0] is True, "the caller must be able to observe why the request ended"
    assert stats["decode_tokens"] == len(output) - 1
    if captured:
        assert model.verify_graph.released and model.draft_graph.released


@pytest.mark.parametrize("eos_position", [2, 3, 4])
def test_eos_does_not_end_the_loop_without_the_cooperative_flag(monkeypatch, eos_position):
    """No flag means the old local exit, which is what world size 1 and the CLI rely on."""
    model, ctx, _clock = _install(monkeypatch, eos_position=eos_position)
    calls: list[int] = []

    def on_step() -> bool:
        calls.append(1)
        return False

    output, stats = _lrun(model, ctx, max_new=10, eos=(3,), on_step=on_step, local_eos=None)
    expected = list(range(10, eos_position + 8)) + [3]
    assert output == expected
    assert stats["stopped"] == "eos"
    # One poll per completed round boundary: the local EOS break is never conditioned on a poll.
    assert len(calls) >= 1


def test_the_poll_schedule_does_not_depend_on_whether_the_token_is_eos(monkeypatch):
    """The safety property: EOS only sets a flag, it never changes how many polls happen.

    Two runs share one predicate that stops at the second call. The first run's very first
    generated token is EOS, the second run's is not; both must ask exactly twice, because a rank
    whose own logits happened to be EOS must not skip a poll its peers take.
    """
    counts = []
    for eos, eos_position in [((3,), 2), ((), None)]:
        model, ctx, _clock = _install(monkeypatch, eos_position=eos_position)
        calls: list[int] = []

        def on_step() -> bool:
            calls.append(1)
            return len(calls) >= 2

        _output, stats = _lrun(model, ctx, max_new=8, eos=eos, on_step=on_step,
                               local_eos=[False])
        counts.append(len(calls))
        assert counts[-1] == 2, f"stopped at poll {counts[-1]} instead of the second poll"
        assert stats["stopped"] in {"eos", "cancel"}
    assert counts[0] == counts[1]


def test_first_token_eos_exits_before_any_trunk_forward(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True, eos_position=2)
    local_eos = [False]

    def on_step() -> bool:
        return local_eos[0]

    output, stats = _lrun(model, ctx, max_new=8, captured=True, eos=(3,), on_step=on_step,
                          local_eos=local_eos)
    assert output == [3]
    assert stats["stopped"] == "eos"
    # Only the prompt's prefill forward ran; the EOS token needs no decode forward at all.
    assert model.calls == [(0, 2)]
    assert stats["trunk_steps"] == 0.0


@pytest.mark.parametrize("captured", [False, True])
def test_mid_round_cancel_skips_the_head_catch_up_and_commits_the_prefix(monkeypatch, captured):
    """A stop recorded by the accepted prefix's callbacks must skip the catch-up for that round."""
    model, ctx, _clock = _install(monkeypatch, captured=captured)
    seen: list[int] = []

    def on_token(token):
        seen.append(token)

    def on_step() -> bool:
        # The first token is emitted, then the round's accepted drafts; the stop is only visible
        # once two tokens have been delivered, which is after the drafts' callbacks.
        return len(seen) >= 2

    output, stats = _lrun(model, ctx, max_new=10, captured=captured, callback=on_token,
                          on_step=on_step, local_eos=[False])
    assert output == seen == [10, 11, 12, 13]
    assert stats["stopped"] == "cancel"
    # The round ran: the trunk verified it and the accepted prefix was delivered.
    assert stats["trunk_steps"] == 1.0
    assert stats["drafts_accepted"] == 3.0
    # ...but no head catch-up and no next-round draft may run after the confirmed stop.
    assert sum(stats["head_kept_prefix_histogram"]) == 0.0
    assert stats["head_replay_kept_prefix_histogram"].count(0) == len(
        stats["head_replay_kept_prefix_histogram"])
    if captured:
        assert model.draft_graph.calls == [], "no chain/catch-up replay after the cancel"
        assert model.verify_graph.released and model.draft_graph.released
    else:
        # The prefill seeding call plus the round's single-token chain links; a catch-up is the
        # one head call that would run over the verified rows and it must not appear.
        assert all(length == 1 for _position, length in model.mtp.calls[1:]), \
            "no catch-up forward may run after the confirmed stop"
    # The committed prefix is coherent: logical length matches the rewound token history.
    assert stats["logical_total_tokens"] == 2 + len(output)
    assert model.cache.length == len(model.cache.state)


def test_round_boundary_cancel_after_a_committed_round_keeps_the_next_token_unemitted(monkeypatch):
    """The top-of-loop poll is what protects the token a round already produced."""
    model, ctx, _clock = _install(monkeypatch)
    seen: list[int] = []
    polls: list[int] = []

    def on_token(token):
        seen.append(token)

    def on_step() -> bool:
        polls.append(1)
        # Stop only once the first round (five tokens: 10 plus a bonus) has been delivered.
        return len(seen) >= 5

    output, stats = _lrun(model, ctx, max_new=12, callback=on_token, on_step=on_step,
                          local_eos=[False])
    assert output == seen == [10, 11, 12, 13, 14]
    assert stats["stopped"] == "cancel"
    assert stats["decode_tokens"] == 4.0
    # The predicate was asked again at the next round boundary, and the round's token was dropped.
    assert len(output) == 5
    assert model.cache.length == len(model.cache.state)


def test_bonus_eos_exits_locally_because_the_bonus_row_is_identical_on_every_rank(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, eos_position=4)
    local_eos = [False]

    def on_step() -> bool:
        return local_eos[0]

    output, stats = _lrun(model, ctx, max_new=10, eos=(3,), on_step=on_step,
                          local_eos=local_eos)
    assert output == [10, 11, 3]
    assert stats["stopped"] == "eos"
    assert local_eos[0] is True, "the terminal bonus still publishes the EOS to the caller"


@pytest.mark.parametrize("captured", [False, True])
def test_native_tail_cancel_still_accounts_for_every_trunk_step(monkeypatch, captured):
    budget, drafts = 12, 3
    native = 2 + budget
    model, ctx, _clock = _install(monkeypatch, captured=captured, native_limit=native)
    seen: list[int] = []

    def on_token(token):
        seen.append(token)

    def on_step() -> bool:
        return len(seen) >= 6

    output, stats = _lrun(model, ctx, max_new=budget, captured=captured, callback=on_token,
                          on_step=on_step, local_eos=[False])
    # One round can deliver up to `drafts + 1` tokens before the next poll, so the cancellation
    # trails the last poll by at most that many tokens -- never by a whole extra round.
    assert output == seen == list(range(10, 10 + len(output)))
    assert 6 <= len(output) <= 6 + drafts, f"stopped after {len(output)} tokens, past the bound"
    assert stats["stopped"] == "cancel"
    assert stats["logical_total_tokens"] == 2 + len(output)
    assert stats["trunk_steps"] == (stats["graph_rounds"] + stats["tail_graph_steps"]
                                    + stats["plain_tail_steps"] + stats["mtp_eager_steps"])


def test_a_preset_flag_never_shortens_the_loop_without_the_predicate_agreeing(monkeypatch):
    """`local_eos` only names the stop reason; only the collective predicate may end the loop.

    This is the other half of the deferral contract: a rank that believes it saw EOS must not
    leave while its peers keep decoding, so a set flag that the predicate does not act on yet
    must not shorten anything.
    """
    model, ctx, _clock = _install(monkeypatch, captured=True)
    local_eos = [True]
    calls: list[int] = []

    def on_step() -> bool:
        calls.append(1)
        return len(calls) >= 4

    output, stats = _lrun(model, ctx, max_new=12, captured=True, on_step=on_step,
                          local_eos=local_eos)
    assert len(calls) == 4, "the loop must keep polling while the predicate says no"
    assert len(output) > 1, "a set flag alone may not truncate the sequence"
    # The flag is what names the exit, and the predicate is what made it happen.
    assert stats["stopped"] == "eos"


def test_cancel_without_the_flag_still_rides_the_same_polls(monkeypatch):
    """The predicate is asked whether or not the caller passes `local_eos`."""
    model, ctx, _clock = _install(monkeypatch)
    calls: list[int] = []

    def on_step() -> bool:
        calls.append(1)
        return len(calls) >= 2

    output, stats = _lrun(model, ctx, max_new=8, on_step=on_step, local_eos=None)
    assert len(calls) == 2
    assert stats["stopped"] == "cancel"
    assert output == [10], "the round's token was dropped at the next boundary"
    assert model.cache.length == len(model.cache.state)


def test_budget_and_eos_on_the_same_token_keeps_naming_the_eos(monkeypatch):
    """A deferred EOS must not be relabelled as a length stop when the budget also lands there."""
    model, ctx, _clock = _install(monkeypatch, eos_position=2)
    local_eos = [False]

    def on_step() -> bool:
        return local_eos[0]

    # Budget of one: the prefill-produced token is both the last allowed token and the EOS.
    output, stats = _lrun(model, ctx, max_new=1, eos=(3,), on_step=on_step, local_eos=local_eos)
    assert output == [3]
    assert stats["stopped"] == "eos"


def test_callback_exception_still_releases_and_never_polls_after_it(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True)
    calls: list[int] = []

    def on_step() -> bool:
        calls.append(1)
        return False

    def fail(_token):
        raise RuntimeError("injected callback failure")

    with pytest.raises(RuntimeError, match="callback failure"):
        _lrun(model, ctx, max_new=8, captured=True, callback=fail, on_step=on_step,
              local_eos=[False])
    assert len(calls) == 1, "the poll before the emission is the last one that may run"
    assert model.verify_graph.released and model.draft_graph.released


@pytest.mark.parametrize("captured", [False, True])
def test_local_eos_without_on_step_keeps_the_plain_local_exit(monkeypatch, captured):
    """A caller that passes the flag but no predicate gets the documented local behaviour."""
    model, ctx, _clock = _install(monkeypatch, captured=captured, eos_position=3)
    local_eos = [False]
    seen: list[int] = []
    output, stats = _lrun(model, ctx, max_new=10, captured=captured, eos=(3,),
                          callback=seen.append, on_step=None, local_eos=local_eos)
    assert output == seen == [10, 3]
    assert stats["stopped"] == "eos"