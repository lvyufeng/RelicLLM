"""Hermetic contract tests for the every-row verify probe; no checkpoint or GPU is required.

The probe's own helpers decide whether a K-row verify forward reproduces K independent single-row
decode steps. These tests pin that decision: a difference in *any* row of *any* layer must be
reported at that row and layer (not only row zero), a rewind window must be compared after the dtype
cast `rewind` performs, and a PLE carry must be sliced at the kept prefix. Comparison helpers are
exercised with real tensors, reference parsing with the durable artifact shapes the MTP probe writes.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

PROBE = Path(__file__).with_name("probe_qwen4_exp_all_rows.py")
spec = importlib.util.spec_from_file_location("all_rows_probe_under_test", PROBE)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def _outline(rows: int, width: int = 4, seed: int = 7) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(rows, width, generator=generator)


def _fake_layer(index: int, *, indexer: bool = True, ple: bool = True):
    class FakeLayer:
        def __init__(self) -> None:
            self.layer_idx = index
            self.router = lambda flat: (flat[:, :2] + 1.0, torch.zeros(flat.shape[0], 2,
                                                                      dtype=torch.long))
            self.attn = SimpleNamespace(
                indexer=(lambda *args, **kwargs: torch.zeros(1, 1, 3, dtype=torch.int32))
                if indexer else None)
            self.ple = (lambda *args, **kwargs: args[0]) if ple else None

        def __call__(self, value):
            return value

    return FakeLayer()


# ------------------------------------------------------------------------------------------------
# Module and reference-artifact contracts
# ------------------------------------------------------------------------------------------------


def test_import_has_no_eager_torch_dependency():
    # The helpers must import in an environment without the accelerator stack. Another test may
    # already have imported torch, so assert this module has no torch global instead of a clean DOF.
    assert "torch" not in vars(probe)


def test_token_hash_matches_the_mtp_probe_material_hash():
    other = Path(__file__).with_name("probe_qwen4_mtp_nvfp4.py")
    spec = importlib.util.spec_from_file_location("mtp_probe_for_hash", other)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ids = [1596, 1144, 4087, 0, 262143]
    assert probe.token_hash(ids) == module._token_hash(ids)


@pytest.mark.parametrize("ids", [[1, 2, 3], [0], [262143, 5, 6, 7]])
def test_plain_reference_artifact_round_trips(tmp_path, ids):
    document = {
        "kind": "plain_reference", "schema": "qwen4-exp-mtp-probe-v2", "fingerprint": "f",
        "material": {"input_sha256": "input-hash", "actual_input_tokens": 8192},
        "record": {"arm": "plain", "output_ids": ids, "output_sha256": probe.token_hash(ids)},
    }
    path = tmp_path / "plain.json"
    path.write_text(json.dumps(document))
    loaded = probe.load_teacher_reference(path)
    assert loaded["ids"] == ids
    assert loaded["input_sha256"] == "input-hash"
    assert loaded["input_tokens"] == 8192
    assert loaded["source_kind"] == "plain_reference"


def test_jsonl_selects_the_plain_arm_and_keeps_the_start_input_hash(tmp_path):
    lines = [
        {"kind": "qwen4_exp_mtp_start", "input_sha256": "start-hash", "actual_input_tokens": 8192},
        {"kind": "qwen4_exp_mtp_arm_result", "arm": "lagq_draft3", "output_ids": [9, 9, 9]},
        {"kind": "qwen4_exp_mtp_arm_result", "arm": "plain", "output_ids": [4, 5, 6],
         "output_sha256": probe.token_hash([4, 5, 6])},
    ]
    path = tmp_path / "arms.jsonl"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    loaded = probe.load_teacher_reference(path)
    assert loaded["ids"] == [4, 5, 6]
    assert loaded["input_sha256"] == "start-hash"
    assert loaded["source_kind"] == "probe_plain_arm"


def test_generic_output_ids_document_is_accepted(tmp_path):
    path = tmp_path / "generic.json"
    path.write_text(json.dumps({"output_ids": [7, 8], "output_sha256": probe.token_hash([7, 8])}))
    loaded = probe.load_teacher_reference(path)
    assert loaded["ids"] == [7, 8] and loaded["source_kind"] == "generic_output_ids"


@pytest.mark.parametrize("document, message", [
    ({"kind": "plain_reference", "record": {"output_ids": []}}, "nonempty"),
    ({"kind": "plain_reference", "record": {"output_ids": [1, -2]}}, "non-negative"),
    ({"kind": "plain_reference", "record": {"output_ids": [1, True]}}, "non-negative"),
    ({"kind": "plain_reference", "record": {"output_ids": [1, 2],
                                            "output_sha256": "not-the-hash"}}, "hash"),
    ({"kind": "qwen4_exp_mtp_start", "input_sha256": "only-start"}, "no plain output"),
])
def test_invalid_reference_documents_are_rejected(tmp_path, document, message):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match=message):
        probe.load_teacher_reference(path)


def test_missing_reference_file_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        probe.load_teacher_reference(tmp_path / "absent.json")


# ------------------------------------------------------------------------------------------------
# Window planning and argument validation
# ------------------------------------------------------------------------------------------------


def test_select_window_uses_distinct_teacher_tokens_and_their_predictions():
    ids = list(range(100, 200))
    window = probe.select_window(ids, deliver_prefix=64, rows=6)
    assert window["prefix_ids"] == list(range(100, 164))
    assert window["feed_ids"] == [164, 165, 166, 167, 168, 169]
    # Window row j must predict the token after it in the reference, so the last row's prediction
    # needs one more reference token than the window itself.
    assert window["teacher_predictions"] == [165, 166, 167, 168, 169, 170]


def test_select_window_last_row_has_no_prediction_when_reference_ends():
    window = probe.select_window([5, 6, 7, 8], deliver_prefix=1, rows=3)
    assert window["feed_ids"] == [6, 7, 8]
    assert window["teacher_predictions"] == [7, 8]


@pytest.mark.parametrize("deliver_prefix, rows, ids", [
    (3, 4, [1, 2, 3]), (-1, 2, [1, 2, 3]), (0, 0, [1, 2, 3]),
])
def test_select_window_rejects_impossible_windows(deliver_prefix, rows, ids):
    with pytest.raises(ValueError):
        probe.select_window(ids, deliver_prefix, rows)


@pytest.mark.parametrize("rows", [4, 5, 6, 8])
def test_plan_arena_keeps_the_logical_total_and_adds_only_slack(rows):
    plan = probe.plan_arena(8192, 64, rows, native_max=262144)
    assert plan["logical_total_tokens"] == 8192 + 64 + rows
    assert plan["arena_tokens"] == 8192 + 64 + rows + 2 * rows + 16
    assert not plan["arena_clamped"]


def test_plan_arena_clamps_a_fitting_request_but_rejects_an_oversized_total():
    # 261632 + 460 + 6 plus the 6-row/16 slack span is 262126, still inside the native maximum, so
    # this request does not need clamping at all.
    plan = probe.plan_arena(261632, 460, 6, native_max=262144)
    assert plan["logical_total_tokens"] == 262098
    assert plan["arena_tokens"] == 262126 and not plan["arena_clamped"]
    # A longer prefix whose logical total still fits, but whose rewind slack would not: the arena is
    # clamped to the native maximum instead of allocating past it.
    clamped = probe.plan_arena(261632, 490, 6, native_max=262144)
    assert clamped["logical_total_tokens"] == 262128
    assert clamped["requested_arena_tokens"] == 262156
    assert clamped["arena_tokens"] == 262144 and clamped["arena_clamped"]
    with pytest.raises(ValueError, match="exceeds the native maximum"):
        probe.plan_arena(261632, 508, 6, native_max=262144)


def test_validate_args_rejects_out_of_range_rows_and_negative_prefixes():
    good = SimpleNamespace(model="/ckpt", teacher_reference="/reference.json", rows=6,
                           context_tokens=8192, deliver_prefix=64, chunk_size=512, arena_slack=16)
    probe.validate_args(good)
    for change in ({"rows": 1}, {"rows": 9}, {"context_tokens": 0}, {"deliver_prefix": -1},
                   {"chunk_size": 0}, {"arena_slack": -1}, {"model": ""},
                   {"teacher_reference": ""}):
        with pytest.raises(ValueError):
            probe.validate_args(SimpleNamespace(**{**vars(good), **change}))


def test_parser_defaults_match_the_failing_width_five_case():
    args = probe._parser().parse_args(["--model", "/ckpt", "--teacher-reference", "/ref.jsonl"])
    assert (args.context_tokens, args.deliver_prefix, args.rows, args.chunk_size) == (8192, 64, 6, 512)
    assert args.skip_states is False and args.skip_aux is False
    assert args.allow_teacher_mismatch is False and args.allow_input_mismatch is False
    assert args.tensor_past is False
    captured = probe._parser().parse_args(
        ["--model", "/ckpt", "--teacher-reference", "/ref.jsonl", "--tensor-past"])
    assert captured.tensor_past is True


def test_flag_snapshot_keeps_only_experiment_flags():
    flags = probe.flag_snapshot({
        "POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT": "1", "POCKETLLM_GRAPH_RETURN_MARGIN": "0",
        "PATH": "/usr/bin", "POCKETLLM_NCCL_ID_PATH": "/tmp/id",
    })
    assert flags == {"POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT": "1",
                     "POCKETLLM_GRAPH_RETURN_MARGIN": "0"}


# ------------------------------------------------------------------------------------------------
# Exact comparisons: every row, every layer
# ------------------------------------------------------------------------------------------------


def test_compare_tensor_is_exact_and_reports_the_first_bit_difference():
    reference = tensor = _outline(4, 8)
    assert probe.compare_tensor(reference, tensor.clone())["equal"]
    changed = tensor.clone()
    changed[2, 5] += 0.5
    result = probe.compare_tensor(reference, changed)
    assert not result["equal"]
    assert result["differing_elements"] == 1 and result["max_abs"] == pytest.approx(0.5)


def test_compare_tensor_flattens_only_a_leading_dimension_difference():
    reference = _outline(1, 6)
    candidate = reference.reshape(1, 2, 3)
    assert probe.compare_tensor(reference, candidate)["equal"]
    with pytest.raises(ValueError):
        probe.compare_tensor(_outline(1, 6), _outline(1, 7))


@pytest.mark.parametrize("rows", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("bad_row", [0, 1, 2])
def test_every_row_is_compared_not_only_row_zero(rows, bad_row):
    if bad_row >= rows:
        pytest.skip("row outside this span")
    expected = _outline(rows, 12)
    actual = expected.clone()
    actual[bad_row, -1] += 1.0
    differences = probe.layer_row_differences([(3, expected)], [(3, actual)])
    assert [(entry["layer"], entry["row"]) for entry in differences] == [(3, bad_row)]
    assert differences[0]["differing_elements"] == 1


@pytest.mark.parametrize("rows", [2, 6, 8])
def test_every_layer_is_compared_and_layer_order_wins(rows):
    tensors = {layer: _outline(rows, 8, seed=layer) for layer in (0, 1, 2)}
    candidate = {layer: value.clone() for layer, value in tensors.items()}
    last = rows - 1
    candidate[1][last, 0] += 1.0
    candidate[2][0, 0] += 1.0
    differences = probe.layer_row_differences(
        [(layer, tensors[layer]) for layer in (0, 1, 2)],
        [(layer, candidate[layer]) for layer in (0, 1, 2)])
    assert [(entry["layer"], entry["row"]) for entry in differences] == [(1, last), (2, 0)]
    first = probe.first_layer_row_difference(differences)
    assert first["layer"] == 1 and first["row"] == last


def test_missing_layer_and_shape_mismatch_are_reported_and_equal_spans_are_clean():
    expected = _outline(3, 5)
    assert probe.layer_row_differences([(0, expected)], [(0, expected.clone())]) == []
    missing = probe.layer_row_differences([(0, expected)], [])
    assert missing == [{"layer": 0, "row": None, "missing": True}]
    mismatch = probe.layer_row_differences([(0, expected)], [(0, _outline(3, 6))])
    assert mismatch[0]["row"] is None and "shape_mismatch" in mismatch[0]
    assert probe.first_layer_row_difference([]) is None


# ------------------------------------------------------------------------------------------------
# Rewind contracts: recurrent state and PLE carry
# ------------------------------------------------------------------------------------------------


def test_state_comparison_matches_after_the_rewind_dtype_cast():
    live = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
    recorded = live.to(torch.float32)  # exactly representable, so the cast is lossless
    assert probe.state_comparison(recorded, live)["equal"]
    assert not probe.state_comparison(recorded + 0.25, live)["equal"]
    # A different leading singleton is the recorded-window shape, compared by value.
    assert probe.state_comparison(live.reshape(1, 1, 3), live.reshape(3))["equal"]
    assert probe.state_comparison(None, None)["equal"]
    assert not probe.state_comparison(None, live)["equal"]


@pytest.mark.parametrize("state_len", [2, 6, 10])
@pytest.mark.parametrize("rows", [2, 6])
def test_ple_carry_slices_reproduce_the_batch_window_for_every_kept_prefix(state_len, rows):
    # The batch log window is [carry(state_len), x_0 .. x_{rows-1}]; serial decode after keeping j
    # positions must carry exactly the slice a rewind would take from it.
    batch = _outline(1, state_len + rows, seed=state_len + rows)
    for kept in range(1, rows + 1):
        serial_live = probe.ple_carry_slice(batch, state_len, kept).clone()
        assert probe.ple_carry_comparison(batch, serial_live, state_len, kept)["equal"]
    # A one-row eager log is only `state_len + 1` wide, so it is legal at kept == 1 alone -- which
    # is exactly the first serial step of the span.
    serial_log = batch[..., :state_len + 1].clone()
    first_live = probe.ple_carry_slice(batch, state_len, 1).clone()
    assert probe.ple_carry_comparison(serial_log, first_live, state_len, 1)["equal"]


def test_ple_carry_detects_a_single_flipped_element_at_the_kept_boundary():
    state_len, rows = 6, 4
    batch = _outline(1, state_len + rows)
    live = probe.ple_carry_slice(batch, state_len, 3).clone()
    live[0, 2] += 1.0
    result = probe.ple_carry_comparison(batch, live, state_len, 3)
    assert not result["equal"] and result["differing_elements"] == 1


def test_ple_carry_rejects_a_prefix_that_does_not_fit_the_window():
    with pytest.raises(ValueError, match="does not fit"):
        probe.ple_carry_slice(_outline(1, 8), state_len=6, kept=3)


# ------------------------------------------------------------------------------------------------
# Observer installation, pool reset and call delegation
# ------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rows", [2, 6, 8])
def test_observers_record_every_layer_every_row_and_restore_the_originals(rows):
    layers = [_fake_layer(index) for index in range(3)]
    model = SimpleNamespace(layers=layers)
    originals = [(layer, layer.router, layer.attn.indexer, layer.ple) for layer in layers]
    store = {"layers": [], "indexer": [], "router": [], "ple": []}
    saved = probe.install_observers(model, store, aux=True)
    try:
        batch = _outline(rows, 4)
        for index, layer in enumerate(model.layers):
            # The wrapper must delegate attributes and return the inner call's exact value.
            assert layer.layer_idx == index
            assert torch.equal(layer(batch), batch)
            layer.router(batch)
            layer.attn.indexer(batch)
            layer.ple(batch)
        assert [index for index, _ in store["layers"]] == [0, 1, 2]
        # A router observer stores the call's (weights, indices) pair beside the layer index.
        assert [index for index, _weights, _indices in store["router"]] == [0, 1, 2]
        assert all(weights.shape == (rows, 2) and indices.shape == (rows, 2)
                   for _index, weights, indices in store["router"])
        assert [index for index, _ in store["indexer"]] == [0, 1, 2]
        assert [index for index, _ in store["ple"]] == [0, 1, 2]
        assert all(value.shape == (1, 1, 3) for _index, value in store["indexer"])
    finally:
        probe.restore_observers(saved)
    assert model.layers == layers
    for layer, router, indexer, ple in originals:
        assert layer.router is router and layer.attn.indexer is indexer and layer.ple is ple


def test_aux_observers_are_optional_and_absent_objects_are_skipped():
    model = SimpleNamespace(layers=[_fake_layer(0, indexer=False, ple=False)])
    store = {"layers": [], "indexer": [], "router": [], "ple": []}
    saved = probe.install_observers(model, store, aux=False)
    try:
        model.layers[0](_outline(2, 4))
        assert len(store["layers"]) == 1 and store["router"] == []
    finally:
        probe.restore_observers(saved)


def test_reset_indexer_pools_invalidates_every_request_owned_pool():
    class Pool:
        def __init__(self) -> None:
            self.calls = 0

        def invalidate(self) -> None:
            self.calls += 1

    pools = [Pool(), Pool()]
    cache = SimpleNamespace(layers=[SimpleNamespace(indexer_pool=pools[0]), None,
                                    SimpleNamespace(indexer_pool=None),
                                    SimpleNamespace(indexer_pool=pools[1])])
    assert probe.reset_indexer_pools(cache) == 2
    assert [pool.calls for pool in pools] == [1, 1]