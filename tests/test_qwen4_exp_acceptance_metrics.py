"""Host-only acceptance evidence: prefix and per-position survival contracts.

Fake models produce fully known greedy sequences, so each expected histogram/counter is derived
from the round control flow in the test, not from the implementation's own output. These CPU tests
prove bookkeeping and JSON shape; checkpoint token parity and CUDA timing remain separate.
"""
from __future__ import annotations

import json

import pytest

from relicllm.models.qwen4_exp import verify_graph as graphs
from test_qwen4_exp_lookahead_contract import _install, _run, _Verify


class _BumpedVerify(_Verify):
    """Verifier whose first row always disagrees, to script rejections deterministically."""

    def replay(self, position, feed):
        rows = super().replay(position, feed)
        rows[0] = rows[0] + 1
        return rows


def _fields(stats):
    return {
        "verified": stats["accepted_prefix_histogram_verified_by_width"],
        "delivered": stats["accepted_prefix_histogram_delivered_by_width"],
        "proposed": stats["draft_position_proposed"],
        "survived_verified": stats["draft_position_survived_verified"],
        "survived_delivered": stats["draft_position_survived_delivered"],
        "head": stats["head_kept_prefix_histogram"],
        "head_replay": stats["head_replay_kept_prefix_histogram"],
    }


def test_eager_rounds_record_full_acceptance_by_actual_width(monkeypatch):
    model, ctx, _clock = _install(monkeypatch)
    # Round 1 accepts its draft (11) and the bonus 12 becomes the next consumed token; round 2
    # accepts draft 13. Every round proposes exactly one draft and every one survives.
    output, stats = _run(model, ctx, max_new=4, drafts=1)
    assert output == [10, 11, 12, 13]
    assert _fields(stats) == {
        "verified": {"1": [0, 2]},
        "delivered": {"1": [0, 2]},
        "proposed": [2],
        "survived_verified": [2],
        "survived_delivered": [2],
        # One eager catch-up, kept = accepted prefix + the consumed token = 2. Index 0 unused.
        "head": [0, 0, 1],
        "head_replay": [0, 0, 0],
    }
    assert stats["drafts_proposed"] == 2 and stats["drafts_accepted"] == 2


def test_captured_catch_up_is_attributed_to_the_replay_histogram(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True)
    output, stats = _run(model, ctx, max_new=4, drafts=1, captured=True)
    assert output == [10, 11, 12, 13]
    # Identical prefix outcomes to the eager run; only the catch-up was served by a recording.
    assert _fields(stats) == {
        "verified": {"1": [0, 2]},
        "delivered": {"1": [0, 2]},
        "proposed": [2],
        "survived_verified": [2],
        "survived_delivered": [2],
        "head": [0, 0, 1],
        "head_replay": [0, 0, 1],
    }


def test_budget_clipping_keeps_verified_prefix_separate_from_delivery(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True)
    # A recorded round always drafts the full width; the last round verifies one more draft than
    # this request may deliver, so the two histograms must disagree by exactly that token.
    output, stats = _run(model, ctx, max_new=3, drafts=3, captured=True)
    assert output == [10, 11, 12]
    assert _fields(stats) == {
        "verified": {"3": [0, 0, 0, 1]},
        "delivered": {"3": [0, 0, 1, 0]},
        "proposed": [1, 1, 1],
        "survived_verified": [1, 1, 1],
        "survived_delivered": [1, 1, 0],
        "head": [0, 0, 0, 0, 0],
        "head_replay": [0, 0, 0, 0, 0],
    }
    assert stats["drafts_accepted_verified"] == 3 and stats["drafts_accepted"] == 2


def test_accepted_eos_clips_delivery_but_not_the_verified_prefix(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True, eos_position=4)
    output, stats = _run(model, ctx, max_new=8, drafts=3, captured=True, eos=(3,))
    assert output == [10, 11, 3]
    assert stats["stopped"] == "eos"
    # All three drafts were verified; the accepted EOS at draft position one ends delivery there.
    assert _fields(stats) == {
        "verified": {"3": [0, 0, 0, 1]},
        "delivered": {"3": [0, 0, 1, 0]},
        "proposed": [1, 1, 1],
        "survived_verified": [1, 1, 1],
        "survived_delivered": [1, 1, 0],
        "head": [0, 0, 0, 0, 0],
        "head_replay": [0, 0, 0, 0, 0],
    }


def test_every_rejected_draft_is_counted_without_inventing_survivors(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True)
    monkeypatch.setattr(graphs, "VerifyGraph", _BumpedVerify)
    # Both rounds reject their single draft. The rejected rounds still advance by the verified
    # bonus token, so the output is a different (but fully known) sequence -- never dropped rounds.
    output, stats = _run(model, ctx, max_new=3, drafts=1, captured=True)
    assert output == [10, 12, 13]
    assert _fields(stats) == {
        "verified": {"1": [2, 0]},
        "delivered": {"1": [2, 0]},
        "proposed": [2],
        "survived_verified": [0],
        "survived_delivered": [0],
        # The first round ran one catch-up at kept = 1; the second ended by emitting its bonus.
        "head": [0, 1, 0],
        "head_replay": [0, 1, 0],
    }
    assert stats["draft_rejected"] == 2 and stats["drafts_accepted_verified"] == 0


@pytest.mark.parametrize("budget", [4, 6, 7])
def test_immediate_native_one_row_tail_records_no_draft_or_head_evidence(monkeypatch, budget):
    model, ctx, _clock = _install(monkeypatch, captured=True, native_limit=2 + budget)
    output, stats = _run(model, ctx, max_new=budget, drafts=3, captured=True)
    assert output == list(range(10, 10 + budget))
    assert stats["plain_tail_steps"] > 0 and stats["draft_rounds"] == 0
    # Speculation is retired before the boundary, so no round and no catch-up ever execute; the
    # evidence must stay empty rather than report the prefill head seeding as accepted drafts.
    assert _fields(stats) == {
        "verified": {}, "delivered": {},
        "proposed": [0, 0, 0], "survived_verified": [0, 0, 0],
        "survived_delivered": [0, 0, 0],
        # Bucket 0 is unused; kept spans 1..draft_tokens + 1.
        "head": [0, 0, 0, 0, 0], "head_replay": [0, 0, 0, 0, 0],
    }


def test_boundary_after_speculation_does_not_claim_a_skipped_catch_up(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, native_limit=11)
    output, stats = _run(model, ctx, max_new=9, drafts=3)
    assert output == list(range(10, 19))
    assert stats["plain_tail_steps"] > 0 and stats["draft_rounds"] == 1
    # The single round accepted all three drafts, then the next round could not speculate at the
    # boundary: its bonus was emitted by the one-row tail, so no head catch-up may be recorded.
    assert _fields(stats) == {
        "verified": {"3": [0, 0, 0, 1]},
        "delivered": {"3": [0, 0, 0, 1]},
        "proposed": [1, 1, 1],
        "survived_verified": [1, 1, 1],
        "survived_delivered": [1, 1, 1],
        "head": [0, 0, 0, 0, 0],
        "head_replay": [0, 0, 0, 0, 0],
    }
    assert stats["drafts_accepted_verified"] == stats["drafts_accepted"] == 3


@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("drafts", [1, 2, 3])
@pytest.mark.parametrize("budget", [3, 4, 5, 7, 9])
def test_ledger_reconciles_with_the_existing_round_accounting(monkeypatch, captured, drafts, budget):
    model, ctx, _clock = _install(monkeypatch, captured=captured)
    output, stats = _run(model, ctx, max_new=budget, drafts=drafts, captured=captured)
    assert output == list(range(10, 10 + budget))
    fields = _fields(stats)
    rounds = int(stats["draft_rounds"])
    # Every executed round lands in exactly one bucket of each histogram.
    assert sum(sum(row) for row in fields["verified"].values()) == rounds
    assert sum(sum(row) for row in fields["delivered"].values()) == rounds
    # A round cannot deliver more than it verified, and neither can exceed what it proposed.
    assert all(delivered <= verified <= proposed for delivered, verified, proposed in zip(
        fields["survived_delivered"], fields["survived_verified"], fields["proposed"]))
    assert stats["drafts_accepted"] <= stats["drafts_accepted_verified"] <= stats["drafts_proposed"]
    # The head histograms are a subset of the rounds; replays are a subset of executed catch-ups.
    assert sum(fields["head"]) <= rounds
    assert sum(fields["head_replay"]) <= sum(fields["head"])
    assert len(fields["head"]) == len(fields["head_replay"]) == drafts + 2
    assert stats["drafts_proposed"] == sum(fields["proposed"])


def test_zero_budget_reports_empty_but_json_serializable_evidence(monkeypatch):
    model, ctx, _clock = _install(monkeypatch)
    output, stats = _run(model, ctx, max_new=0, drafts=3)
    assert output == []
    fields = _fields(stats)
    assert fields["verified"] == {} and fields["delivered"] == {}
    assert fields["proposed"] == [] and fields["head"] == []
    assert json.loads(json.dumps(stats))["draft_position_proposed"] == []


def test_evidence_is_json_serializable_for_durable_probe_output(monkeypatch):
    model, ctx, _clock = _install(monkeypatch, captured=True)
    _output, stats = _run(model, ctx, max_new=5, drafts=2, captured=True)
    restored = json.loads(json.dumps(stats))
    for name in ("accepted_prefix_histogram_verified_by_width",
                 "accepted_prefix_histogram_delivered_by_width",
                 "draft_position_proposed", "draft_position_survived_verified",
                 "draft_position_survived_delivered", "head_kept_prefix_histogram",
                 "head_replay_kept_prefix_histogram"):
        assert restored[name] == stats[name]
    # Width keys are decimal strings of the proposed width and every bucket is an int count, so a
    # consumer never has to coerce JSON object keys or guess the bucket layout at read time.
    for field in ("accepted_prefix_histogram_verified_by_width",
                  "accepted_prefix_histogram_delivered_by_width"):
        for width_key, row in stats[field].items():
            assert width_key == str(int(width_key)) and len(row) == int(width_key) + 1
            assert all(isinstance(count, int) for count in row)
    assert all(isinstance(count, int) for count in stats["head_kept_prefix_histogram"])