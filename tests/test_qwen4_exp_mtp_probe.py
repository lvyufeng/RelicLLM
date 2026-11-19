"""Hermetic evidence/validation tests for the MTP probe; no torch or checkpoint required."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

PROBE = Path(__file__).with_name("probe_qwen4_mtp_nvfp4.py")
spec = importlib.util.spec_from_file_location("mtp_probe_under_test", PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_import_has_no_torch_dependency():
    # The helpers must remain usable without the accelerator environment. Importing another test
    # may already have imported torch, so assert this probe has no eager torch global instead.
    assert "torch" not in vars(probe)


def test_plain_graph_is_independent_of_lookahead_graph():
    args = probe._parser().parse_args(["--model", "/checkpoint"])
    assert args.graph_window == 32
    assert probe._arm_order({"la", "lag", "lagq"}, [2, 3]) == [
        ("la", 2, False), ("la", 3, False), ("lag", 2, True), ("lag", 3, True),
        ("lagq", 2, True), ("lagq", 3, True),
    ]


def test_full_native_capacity_is_not_a_full_native_prompt():
    plan = probe._arena_plan(262016, 128, 262144, {"plain", "lag"}, [3])
    assert plan["actual_input_tokens"] == 262016
    assert plan["requested_logical_total_tokens"] == 262144
    assert plan["largest_arena_tokens"] == 262144
    assert plan["arena_tokens_by_mode"] == {"plain": 262144, "draft3": 262144}
    assert plan["requested_arena_tokens_by_mode"]["draft3"] == 262154
    assert plan["arena_clamped_by_mode"]["draft3"]
    assert plan["full_arena_reaches_256k"] and plan["logical_total_reaches_256k"]
    assert not plan["input_prompt_is_at_least_256k"]
    with pytest.raises(ValueError, match="largest permitted input.*262016"):
        probe._arena_plan(262144, 128, 262144, {"lag"}, [3])
    with pytest.raises(ValueError, match="above native maximum"):
        probe._arena_plan(262017, 128, 262144, {"lag"}, [3])
    # A clamped speculative arena is legal while logical prompt + output remains native.
    assert probe._arena_plan(262007, 128, 262144, {"lag"}, [3])["native_limit_respected"]


def test_plan_uses_largest_draft_and_validates_arguments():
    assert probe._arena_plan(100, 10, 150, {"lagq"}, [1, 7])["largest_arena_tokens"] == 128
    with pytest.raises(ValueError, match="positive"):
        probe._arena_plan(100, 0, 150, {"plain"}, [3])
    with pytest.raises(ValueError, match="--arms"):
        probe._arena_plan(100, 1, 150, {"typo"}, [3])
    with pytest.raises(ValueError, match="lookahead"):
        probe._arena_plan(100, 1, 150, {"lag"}, [])


@pytest.mark.parametrize("reference,output,expected", [
    ([1, 2], [1, 2], None), ([1, 2], [1], 1), ([1], [1, 2], 1),
    ([1, 2], [1, 3], 1), ([], [1], 0), ([1], [], 0), ([], [], None),
])
def test_length_only_mismatch_is_not_silently_accepted(reference, output, expected):
    assert probe._first_mismatch(reference, output) == expected


@pytest.mark.parametrize("ids,requested,reason,valid", [
    ([1, 2, 3], 3, "budget", True), ([1, 9], 4, "eos", True),
    ([1, 2], 4, "incomplete", False), ([1, 9, 2], 4, "incomplete", False),
    ([1, 9, 2, 3], 4, "budget", False), ([], 4, "incomplete", False),
    ([1, 2, 3, 4, 5], 4, "incomplete", False), ([1, 2, 9], 3, "eos", True),
])
def test_completion_requires_budget_or_terminal_eos(ids, requested, reason, valid):
    completion = probe._completion(ids, requested, (9,))
    assert completion["stop_reason"] == reason
    assert completion["complete"] is valid


def _material():
    return {"schema": probe.SCHEMA, "requested_output_tokens": 3, "input_sha256": "input"}


def _reference():
    material = _material()
    return {"kind": "plain_reference", "schema": probe.SCHEMA,
            "fingerprint": probe._hash_json(material), "material": material,
            "record": {"arm": "plain", "output_ids": [1, 2, 3],
                       "output_sha256": probe._token_hash([1, 2, 3]),
                       "valid": True, "required_execution_mode_ran": True}}


def test_reference_fingerprint_and_full_output_are_checked():
    document = _reference()
    assert probe._validate_reference(document, _material(), (9,))["output_ids"] == [1, 2, 3]
    with pytest.raises(ValueError, match="fingerprint"):
        probe._validate_reference(document, {**_material(), "input_sha256": "other"}, (9,))
    document["record"]["output_ids"].pop()
    with pytest.raises(ValueError, match="output hash"):
        probe._validate_reference(document, _material(), (9,))
    document["record"]["output_sha256"] = probe._token_hash([1, 2])
    with pytest.raises(ValueError, match="incomplete"):
        probe._validate_reference(document, _material(), (9,))


def test_reference_round_trip_is_atomic_and_exact(tmp_path):
    path = tmp_path / "reference.json"
    probe._write_reference(path, _material(), _reference()["record"])
    loaded = json.loads(path.read_text())
    assert probe._validate_reference(loaded, _material(), (9,))["output_ids"] == [1, 2, 3]
    assert list(tmp_path.glob("*.tmp-*")) == []


def test_reference_rejects_failed_execution_mode():
    document = _reference()
    document["record"]["required_execution_mode_ran"] = False
    with pytest.raises(ValueError, match="execution/validation"):
        probe._validate_reference(document, _material(), (9,))


def test_baseline_reference_survives_only_speculative_source_change(monkeypatch):
    monkeypatch.setattr(probe, "_math_environment", lambda: {})
    kwargs = {"checkpoint": {}, "source_hashes": {"runtime.py": "one", "verify_graph.py": "one"},
              "input_ids": [1, 2], "native": 262144, "budget": 2048, "eos": (9,),
              "args": SimpleNamespace(max_new_tokens=3, chunk_size=512, graph_window=32),
              "world_size": 8, "torch_version": "torch", "cuda_version": "cuda"}
    material = probe._reference_material(**kwargs)
    kwargs["source_hashes"] = {"runtime.py": "one", "verify_graph.py": "two"}
    assert probe._reference_material(**kwargs) == material
    kwargs["source_hashes"] = {"runtime.py": "two", "verify_graph.py": "two"}
    assert probe._reference_material(**kwargs) != material


def test_rates_expose_old_prefill_numerator_without_calling_it_a_win():
    evidence = probe._rate_evidence([1, 2, 3], {"decode_s": 0.1, "generated_tokens": 3,
                                               "decode_tps": 30, "graph_build_s": 2})
    assert evidence["runtime_rate_numerator_tokens"] == 3
    assert evidence["emitted_minus_prefill_token_numerator"] == 2
    assert evidence["emitted_minus_prefill_token_tps"] == 20
    assert evidence["decode_wall_s_including_graph_setup"] == 2.1
    assert evidence["measured_not_projected"] is True


def _native_tail_stats():
    return {"graph_rounds": 6, "plain_tail_steps": 4, "trunk_steps": 10, "mtp_eager_steps": 0,
            "arena_tokens": 128, "native_max_context": 128, "logical_total_tokens": 128,
            "plain_tail_start": 123, "plain_tail_reason": "native_boundary"}


def test_captured_execution_accepts_only_proved_native_tail():
    plan = probe._arena_plan(100, 28, 128, {"lag"}, [3])
    evidence = probe._captured_execution(_native_tail_stats(), plan, 3, emitted=28)
    assert evidence["valid"]
    assert evidence["graph_rounds"] + evidence["native_eager_tail_steps"] == evidence["trunk_steps"]
    assert evidence["tail_is_native_boundary"]


@pytest.mark.parametrize("changed", [
    {"graph_rounds": 0}, {"mtp_eager_steps": 1}, {"trunk_steps": 11},
    {"plain_tail_reason": "arena_boundary"}, {"plain_tail_reason": "drafts_disabled"},
    {"plain_tail_start": 120, "plain_tail_steps": 7, "trunk_steps": 13},
    {"plain_tail_start": None}, {"plain_tail_steps": 5, "trunk_steps": 11},
    {"arena_tokens": 138}, {"native_max_context": 129}, {"logical_total_tokens": 127},
    {"graph_rounds": 6.5, "trunk_steps": 10.5},
])
def test_captured_execution_rejects_arbitrary_fallback_and_false_layout(changed):
    plan = probe._arena_plan(100, 28, 128, {"lag"}, [3])
    assert not probe._captured_execution({**_native_tail_stats(), **changed}, plan, 3, 28)["valid"]


def test_captured_execution_without_tail_still_requires_every_trunk_round_captured():
    plan = probe._arena_plan(100, 20, 140, {"lag"}, [3])
    stats = {"graph_rounds": 6, "plain_tail_steps": 0, "trunk_steps": 6, "mtp_eager_steps": 0,
             "arena_tokens": 130, "native_max_context": 140, "logical_total_tokens": 120,
             "plain_tail_start": None, "plain_tail_reason": ""}
    assert probe._captured_execution(stats, plan, 3, 20)["valid"]
    assert not probe._captured_execution({**stats, "trunk_steps": 7}, plan, 3, 20)["valid"]


def test_rate_evidence_uses_explicit_wall_scope_without_doublecounting_alignment():
    stats = {"decode_s": 0.1, "decode_s_excluding_setup": 0.1, "decode_wall_s": 2.6,
             "decode_tokens": 2, "decode_tps": 20, "graph_build_s": 2,
             "graph_alignment_s": 0.5, "graph_setup_s": 2.5, "decode_wall_tps": 2 / 2.6}
    evidence = probe._rate_evidence([1, 2, 3], stats)
    assert evidence["graph_setup_s"] == 2.5
    assert evidence["graph_alignment_s"] == 0.5
    assert evidence["decode_wall_s_including_graph_setup"] == 2.6
    assert evidence["emitted_minus_prefill_token_wall_tps"] == 2 / 2.6
    # Plain runtime's decode_s already includes its own capture/rebuild work.
    evidence = probe._rate_evidence([1, 2, 3], {"decode_s": 1, "decode_wall_s": 1,
                                               "graph_setup_s": 0.5, "decode_tokens": 2})
    assert evidence["decode_wall_s_including_graph_setup"] == 1


def test_cleanup_rate_is_explicit_and_not_folded_into_existing_setup_scope():
    evidence = probe._rate_evidence([1, 2, 3], {
        "decode_s": .1, "decode_wall_s": 2.6, "graph_setup_s": 2.5,
        "request_cleanup_s": .4, "deferred_graph_cleanup_s": .3,
    })
    assert evidence["emitted_minus_prefill_token_tps"] == 20
    assert evidence["decode_wall_s_including_graph_setup"] == 2.6
    assert evidence["runtime_decode_s_including_cleanup"] == .5
    assert evidence["emitted_minus_prefill_token_tps_including_cleanup"] == 4
    assert evidence["deferred_graph_cleanup_s"] == .3
    missing = probe._rate_evidence([1, 2], {"decode_s": .1})
    assert missing["runtime_decode_s_including_cleanup"] is None
    assert missing["emitted_minus_prefill_token_tps_including_cleanup"] is None


@pytest.mark.parametrize("eager,graph", [(0, 4), (2, 2), (4, 0)])
def test_native_tail_partition_records_graph_and_eager_steps_separately(eager, graph):
    plan = probe._arena_plan(100, 28, 128, {"lag"}, [3])
    stats = {**_native_tail_stats(), "plain_tail_steps": eager,
             "tail_graph_steps": graph, "tail_graph_built": True}
    evidence = probe._captured_execution(stats, plan, 3, 28, tail_graph_requested=True)
    assert evidence["valid"]
    assert evidence["native_tail_steps_total"] == 4
    assert evidence["tail_graph_steps"] == graph
    assert evidence["native_eager_tail_steps"] == eager
    assert evidence["native_tail_fully_captured"] is (eager == 0)


@pytest.mark.parametrize("changed", [
    {"tail_graph_built": False}, {"tail_graph_built": 1},
    {"tail_graph_steps": 5, "trunk_steps": 11},
    {"tail_graph_steps": -1, "plain_tail_steps": 5},
    {"tail_graph_steps": 3.5, "plain_tail_steps": 0.5},
    {"mtp_eager_steps": 1}, {"plain_tail_reason": "arena_boundary"},
    {"plain_tail_start": 120, "tail_graph_steps": 7, "trunk_steps": 13},
])
def test_captured_tail_rejects_forged_counts_and_missing_build_proof(changed):
    plan = probe._arena_plan(100, 28, 128, {"lag"}, [3])
    stats = {**_native_tail_stats(), "plain_tail_steps": 0,
             "tail_graph_steps": 4, "tail_graph_built": True, **changed}
    assert not probe._captured_execution(stats, plan, 3, 28, tail_graph_requested=True)["valid"]


def test_captured_tail_requires_explicit_request_and_metadata():
    plan = probe._arena_plan(100, 28, 128, {"lag"}, [3])
    stats = {**_native_tail_stats(), "plain_tail_steps": 0,
             "tail_graph_steps": 4, "tail_graph_built": True}
    assert not probe._captured_execution(stats, plan, 3, 28)["valid"]
    for missing in ("tail_graph_built", "tail_graph_steps"):
        incomplete = {key: value for key, value in stats.items() if key != missing}
        assert not probe._captured_execution(incomplete, plan, 3, 28,
                                             tail_graph_requested=True)["valid"]
    # Default-off runtime snapshots may omit graph-tail fields, but the ordinary exact eager
    # native suffix is still valid and is never presented as a fully captured tail.
    evidence = probe._captured_execution(_native_tail_stats(), plan, 3, 28)
    assert evidence["valid"] and not evidence["native_tail_fully_captured"]


@pytest.mark.parametrize("field", ["graph_rounds", "tail_graph_steps", "trunk_steps", "plain_tail_start"])
def test_captured_evidence_rejects_nonfinite_numbers(field):
    plan = probe._arena_plan(100, 28, 128, {"lag"}, [3])
    stats = {**_native_tail_stats(), "tail_graph_steps": 0, "tail_graph_built": False,
             field: float("inf")}
    assert not probe._captured_execution(stats, plan, 3, 28)["valid"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
