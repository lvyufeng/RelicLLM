"""Runtime deferral restores graph bindings at the boundary but frees owners after delivery.

CPU fake-clock contracts do not prove CUDA parity or throughput; GPU complete-output tests do.
"""
import pytest

from relicllm.models.qwen4_exp import verify_graph as graphs
from test_qwen4_exp_lookahead_contract import _Draft, _Verify, _install, _run


FLAG = "POCKETLLM_QWEN4_MTP_DEFER_GRAPH_RELEASE"


def fixture(monkeypatch, enabled):
    model, ctx, clock = _install(monkeypatch, captured=True, native_limit=19)
    monkeypatch.setenv("POCKETLLM_QWEN4_MTP_TAIL_GRAPH", "1")
    monkeypatch.setenv(FLAG, "1" if enabled else "0")
    seen, events = [], []

    class Verify(_Verify):
        def retire_bindings(self):
            events.append(("retire_verify", self.rows, len(seen)))
            clock.advance(.05)

        def release(self):
            events.append(("release_verify", self.rows, len(seen)))
            super().release()

    class Draft(_Draft):
        def retire_bindings(self):
            events.append(("retire_draft", self.drafts, len(seen)))
            clock.advance(.05)

        def release(self):
            events.append(("release_draft", self.drafts, len(seen)))
            super().release()

    monkeypatch.setattr(graphs, "VerifyGraph", Verify)
    monkeypatch.setattr(graphs, "DraftGraph", Draft)
    return model, ctx, seen, events


@pytest.mark.parametrize("enabled", [False, True])
def test_deferred_native_retirement_keeps_complete_output_and_reports_cleanup(monkeypatch, enabled):
    model, ctx, seen, events = fixture(monkeypatch, enabled)
    output, stats = _run(model, ctx, max_new=17, captured=True, callback=seen.append)
    assert output == seen == list(range(10, 27))
    assert stats["tail_graph_steps"] > 0 and stats["plain_tail_steps"] == 0
    assert stats["logical_total_tokens"] == stats["arena_tokens"] == 19
    assert stats["request_cleanup_s"] >= .5
    assert stats["decode_s_including_cleanup"] == pytest.approx(stats["decode_s"] + stats["request_cleanup_s"])
    assert stats["decode_tps_including_cleanup"] == pytest.approx(16 / stats["decode_s_including_cleanup"])
    assert stats["decode_tps_including_cleanup"] < stats["decode_tps_excluding_setup"]
    if enabled:
        assert [name for name, _rows, _tokens in events[:2]] == ["retire_draft", "retire_verify"]
        assert all(tokens == 17 for name, _rows, tokens in events if name.startswith("release"))
        assert stats["deferred_graph_cleanup_s"] == pytest.approx(1)
        assert stats["boundary_retire_s"] == pytest.approx(.1)
        assert stats["deferred_boundary_graphs"] == 2
    else:
        assert not any(name.startswith("retire") for name, _rows, _tokens in events)
        assert any(tokens < 17 for name, _rows, tokens in events if name.startswith("release"))
        assert stats["deferred_graph_cleanup_s"] == 0
        assert stats["deferred_boundary_graphs"] == 0
    assert model.verify_graph.released and model.draft_graph.released


def test_callback_failure_after_metadata_retirement_releases_every_retained_owner(monkeypatch):
    model, ctx, seen, events = fixture(monkeypatch, True)

    def callback(token):
        seen.append(token)
        if any(name.startswith("retire") for name, _rows, _tokens in events):
            raise RuntimeError("injected tail callback failure")

    with pytest.raises(RuntimeError, match="tail callback"):
        _run(model, ctx, max_new=17, captured=True, callback=callback)
    releases = [name for name, _rows, _tokens in events if name.startswith("release")]
    assert releases.count("release_draft") == 1
    assert releases.count("release_verify") == 2
    assert model.verify_graph.released and model.draft_graph.released
