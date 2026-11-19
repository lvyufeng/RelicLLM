"""PD resume skips prefill and restores the same first-token and private-head seeds."""
from types import SimpleNamespace
import hashlib

import pytest
import torch

from relicllm.models.qwen4_exp import runtime, pd_state
from test_qwen4_exp_lookahead_contract import _install


def test_prefill_hook_runs_before_first_callback_and_sees_the_complete_seed(monkeypatch):
    model, ctx, _clock = _install(monkeypatch)
    events = []
    states = []

    def prefilling(state):
        events.append("prefill")
        states.append(state)
        assert state["past"] == 2 and state["cache"].length == 2
        assert state["next_token"] == 10 and state["first_draft"] == 11
        assert state["head_cache"] is not None and state["chain_stack"] is not None

    output, stats = runtime.generate_mtp_draft_lookahead(
        model, ctx, torch.tensor([[7,8]]), max_new_tokens=4, draft_tokens=3,
        on_prefill=prefilling, on_token=lambda token: events.append(token),
    )
    assert output == [10,11,12,13]
    assert events == ["prefill",10,11,12,13] and len(states) == 1
    assert stats["pd_prefilled"] is False
    assert stats["pd_restore_s"] == stats["pd_source_prefill_s"] == 0.0


def test_resumed_loop_does_not_forward_the_prompt_and_preserves_output(monkeypatch):
    model, ctx, clock = _install(monkeypatch)
    original = model.make_cache

    def make_cache(**kwargs):
        cache = original(**kwargs)
        cache.share_rope_table = lambda: None
        return cache

    model.make_cache = make_cache
    model.rope = lambda positions, dtype: (
        torch.ones(1, positions.shape[-1], 1, dtype=dtype),
        torch.zeros(1, positions.shape[-1], 1, dtype=dtype),
    )
    observed = []

    def thaw(bundle, **kwargs):
        observed.append(kwargs)
        cache = kwargs["cache"]
        assert cache.rope_width == cache.width and cache.cos is not None
        cache.length = 2
        cache.state = [7,8]
        cache.layers[0].length = 2
        clock.advance(0.5)
        mixed = model.logits(4, 1)
        return {"past":2,"next_token":10,"first_draft":11,"first_conf":1.0,
                "chain_stack":mixed,"prefill_s":17.0}

    monkeypatch.setattr(pd_state,"thaw_state",thaw)
    output, stats = runtime.generate_mtp_draft_lookahead(
        model, ctx, torch.tensor([[7,8]]), max_new_tokens=4, draft_tokens=3,
        prefilled_state=SimpleNamespace(past=2),
    )
    assert output == [10,11,12,13]
    assert model.calls and all(start >= 2 for start, _width in model.calls)
    assert not any(start == 0 for start, _width in model.mtp.calls)
    assert len(observed)==1
    expected = hashlib.sha256(torch.tensor([[7,8]]).numpy().tobytes()).hexdigest()
    assert observed[0]["expected_metadata"] == {"input_sha256":expected}
    assert stats["pd_prefilled"] and stats["pd_source_prefill_s"]==17.0
    assert stats["pd_restore_s"]==0.5


def test_resumed_wrong_position_is_refused_before_import(monkeypatch):
    model, ctx, _clock = _install(monkeypatch)
    original = model.make_cache

    def make_cache(**kwargs):
        cache = original(**kwargs)
        cache.share_rope_table = lambda: None
        return cache

    model.make_cache = make_cache
    model.rope = lambda positions, dtype: (
        torch.ones(1, positions.shape[-1], 1),torch.zeros(1, positions.shape[-1],1))
    monkeypatch.setattr(pd_state,"thaw_state",lambda *_args, **_kwargs: pytest.fail("must validate position first"))
    with pytest.raises(ValueError,match="position"):
        runtime.generate_mtp_draft_lookahead(
            model,ctx,torch.tensor([[7,8]]),max_new_tokens=4,draft_tokens=3,
            prefilled_state=SimpleNamespace(past=3),
        )
    assert not model.calls and not model.mtp.calls
