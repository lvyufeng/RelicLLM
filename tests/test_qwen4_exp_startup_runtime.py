"""Runtime startup recording stays in measured decode and retains its owners until cleanup."""
import pytest

from relicllm.models.qwen4_exp import startup_graph
from test_qwen4_exp_lookahead_contract import _install, _run


FLAG = "POCKETLLM_QWEN4_MTP_STARTUP_GRAPH"


def install(monkeypatch, enabled, native_limit=19):
    model, ctx, clock = _install(monkeypatch, captured=True, native_limit=native_limit)
    monkeypatch.setenv(FLAG, "1" if enabled else "0")
    monkeypatch.setenv("POCKETLLM_QWEN4_MTP_TAIL_GRAPH", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_MTP_DEFER_GRAPH_RELEASE", "0")
    seen, recordings = [], []

    class Startup:
        def __init__(self, *, start, drafts, first_draft, chain_stack, **kwargs):
            assert first_draft == model.token(start + 1)
            self.start, self.drafts = start, drafts
            self.replays, self.released = 0, False
            self.at_build = clock.now
            clock.advance(3)
            recordings.append(self)

        def replay(self):
            assert seen, "startup generation must occur after the first callback"
            assert not self.released
            clock.advance(1)
            self.replays += 1
            return [model.token(self.start + i + 1) for i in range(self.drafts)]

        def release(self):
            self.released = True
            self.tokens_at_release = len(seen)
            clock.advance(.5)

    monkeypatch.setattr(startup_graph, "StartupDraftGraph", Startup)
    return model, ctx, seen, recordings


@pytest.mark.parametrize("enabled", [False, True])
def test_startup_replay_keeps_full_native_sequence_and_charges_actual_generation(monkeypatch, enabled):
    model, ctx, seen, built = install(monkeypatch, enabled)
    output, stats = _run(model, ctx, max_new=17, captured=True, callback=seen.append)
    assert output == seen == list(range(10, 27))
    assert stats["logical_total_tokens"] == stats["arena_tokens"] == 19
    assert stats["tail_graph_steps"] > 0 and stats["plain_tail_steps"] == 0
    assert stats["startup_graph_built"] is enabled
    assert stats["startup_graph_replays"] == int(enabled)
    assert stats["startup_head_s"] == (1 if enabled else 2)
    assert stats["graph_build_s"] == (14 if enabled else 11)
    assert stats["ttft_s"] >= stats["prefill_s"] + stats["graph_build_s"]
    assert len(model.mtp.calls) == (1 if enabled else 3)
    if enabled:
        assert len(built) == 1 and built[0].replays == 1 and built[0].released
        assert built[0].tokens_at_release == len(output)


def test_callback_failure_before_startup_still_releases_unused_recording(monkeypatch):
    model, ctx, seen, built = install(monkeypatch, True)

    def fail(token):
        seen.append(token)
        raise RuntimeError("injected first callback failure")

    with pytest.raises(RuntimeError, match="first callback"):
        _run(model, ctx, max_new=17, captured=True, callback=fail)
    assert len(built) == 1 and built[0].replays == 0 and built[0].released


def test_immediate_native_tail_does_not_build_or_replay_startup_chain(monkeypatch):
    model, ctx, seen, built = install(monkeypatch, True, native_limit=9)
    output, stats = _run(model, ctx, max_new=7, captured=True, callback=seen.append)
    assert output == seen == list(range(10, 17))
    assert built == []
    assert not stats["startup_graph_built"] and stats["startup_graph_replays"] == 0
    assert stats["startup_head_s"] == 0
