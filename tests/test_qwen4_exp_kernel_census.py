"""Unit tests for the main-verifier census helpers.

Two CPUs-only halves:

* `tests/analyze_qwen4_exp_kernel_census.py` -- driven with small synthetic Chrome traces, so the
  correlation join, the innermost-range attribution and the non-lossy classification are checked
  without a checkpoint. The regression these guard is concrete: the previous analyzer reported the
  literal families ``std::enable_if``, ``""``, ``void`` and ``void at::native::``, which were
  truncation artifacts hiding about a fifth of the measured verify busy time.
* `tests/probe_qwen4_exp_linear_hotspots.py` -- the recorder, the geometry extraction and the
  delegation guarantee. The wrapper test runs real CPU tensors through the installed observer and
  compares against the unpatched call, so "delegates unchanged" is asserted by identity of result
  values rather than by reading the implementation.

Nothing here needs CUDA, a checkpoint, or a distributed group; the two entry points that do are
guarded by explicit CUDA checks that these tests do not exercise.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analyze_qwen4_exp_kernel_census as census_module
import probe_qwen4_exp_linear_hotspots as probe_module

# A real verify-round kernel name, kept at full length on purpose: it is a cuBLAS GEMV whose mangled
# text begins with `std::enable_if`, which is exactly why the old analyzer could not name it.
GEMV_FULL_NAME = (
    "std::enable_if<!(false), void>::type internal::gemvx::kernel<int, int, int, 64, 64, 8, 4, 1, "
    "true, false, cublasGemvParams_t<CUDA_R_16BF>>(cublasGemvParams_t<CUDA_R_16BF>)"
)
ELEMENTWISE_FULL_NAME = (
    "void at::native::vectorized_elementwise_kernel<4, "
    "at::native::binary_internal::MulFunctor<float>, std::array<char*, 6ul>>(int, "
    "at::native::binary_internal::MulFunctor<float>, std::array<char*, 6ul>)"
)


def _trace(*events) -> dict:
    return {"traceEvents": list(events)}


def _range(name: str, ts: float, dur: float) -> dict:
    return {"cat": "user_annotation", "name": name, "ts": ts, "dur": dur}


def _launch(correlation: int, ts: float) -> dict:
    return {"cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": ts, "dur": 1.0,
            "args": {"correlation": correlation}}


def _kernel(correlation: int, name: str, ts: float, dur: float,
            cat: str = "kernel") -> dict:
    return {"cat": cat, "name": name, "ts": ts, "dur": dur, "args": {"correlation": correlation}}


# ------------------------------------------------------------------------------------------
# Classification
# ------------------------------------------------------------------------------------------


def test_full_gemv_name_classifies_as_gemvx_not_enable_if():
    assert census_module.classify(GEMV_FULL_NAME) == "cublas_gemvx"


@pytest.mark.parametrize("name,family", [
    ("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "nccl"),
    ("ncclDevKernel_AllGather_RING_LL", "nccl"),
    ("void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_64x64_16x6_tn_align4>(void*)", "cutlass"),
    ("void cublasLt::splitKreduce_kernel<float, float, float, __nv_bfloat16>(int, int)", "cublaslt"),
    ("ampere_sgemm_32x128_tn", "gemm"),
    ("qwen4_exp_gated_delta_bf16_forward_kernel<1>(float const*, float const*)", "gdn_recurrent"),
    ("normalize_qk_bf16_kernel(c10::BFloat16 const*, c10::BFloat16 const*)", "gdn_recurrent"),
    ("qwen4_exp_hc_activation_kernel(float*, int)", "hyper_connection"),
    ("qwen4_exp_inject_kernel<float, 4>(float*, float const*)", "hyper_connection"),
    ("nvfp4_exact_gate_up_kernel(float const*, long const*, float const*)", "nvfp4_moe"),
    ("nvfp4_exact_down_kernel(float const*, long const*, float const*)", "nvfp4_moe"),
    ("qwen4_exp_grouped_rms_norm_kernel<float>(float*, float const*)", "norm"),
    ("qwen4_exp_qsa_splitk_kernel(float*, float const*)", "qsa_attention"),
    ("qwen4_exp_scatter_kernel_v2(float*, long const*)", "qwen4_exp_other"),
    ("void at::native::radixSortKVInPlace<float, long, 8, true, false>(int, int)", "topk_sort"),
    ("void at::native::sbtopk::gatherTopK<float, long, 2, false>(int, int)", "topk_sort"),
    ("void at::native::mbtopk::computeBlockDigitCounts<float, long, 4>(int, int)", "topk_sort"),
    ("at_cuda_detail::cub::detail::scan_by_key::DeviceScanByKeyKernel<long, long>(int)", "topk_sort"),
    (ELEMENTWISE_FULL_NAME, "elementwise"),
    ("void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float>>(int)", "reduction"),
    ("void reduce_1Block_kernel<float, float, 256>(float*, float*, int)", "reduction"),
    ("void dot_kernel<float, float, 256>(float*, float*, int)", "reduction"),
    ("memcpy32_post(void*, void const*)", "copy"),
    ("void at::native::memcpy_and_sync(void*, void const*, int)", "copy"),
])
def test_family_rules(name, family):
    assert census_module.classify(name) == family


def test_unmatched_name_is_unclassified_and_not_merged():
    assert census_module.classify("some_private_solver_kernel_v3(float*)") == "unclassified"
    assert census_module.classify("") == "unclassified"


# ------------------------------------------------------------------------------------------
# Join, span and busy accounting
# ------------------------------------------------------------------------------------------


def test_round_join_span_busy_and_outside_launches():
    # Chrome trace clocks are microseconds; the report converts only at the end.
    trace = _trace(
        _range("round.verify", 100_000.0, 50_000.0),
        _range("round.draft", 200_000.0, 20_000.0),
        # Two verify replays: 4.0 ms and 1.5 ms spans, each with two busy kernels.
        _launch(1, 101_000.0),
        _kernel(1, GEMV_FULL_NAME, 101_000.0, 2_000.0),
        _kernel(1, ELEMENTWISE_FULL_NAME, 104_000.0, 1_000.0),
        _launch(2, 111_000.0),
        _kernel(2, GEMV_FULL_NAME, 111_000.0, 1_500.0),
        _kernel(2, ELEMENTWISE_FULL_NAME, 112_000.0, 500.0),
        # One draft replay.
        _launch(3, 201_000.0),
        _kernel(3, "ncclDevKernel_AllGather_RING_LL", 201_000.0, 800.0),
        # A startup/tail replay outside every round range.
        _launch(4, 300_000.0),
        _kernel(4, "ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 300_000.0, 400.0),
    )
    document = census_module.census(trace)
    assert document["schema"] == census_module.SCHEMA
    verify = document["rounds"]["round.verify"]
    assert verify["replays"] == 2
    assert verify["kernels_per_replay"] == 2.0
    # Spans: replay 1 = (104_000+1_000-101_000) us = 4.0 ms; replay 2 = 1.5 ms.
    assert verify["span_ms_mean"] == pytest.approx((4.0 + 1.5) / 2)
    # Busy: (2.0+1.0)=3.0 and (1.5+0.5)=2.0 -> 2.5 ms mean.
    assert verify["busy_ms_mean"] == pytest.approx(2.5)
    assert verify["busy_ms_sum"] == pytest.approx(5.0)
    draft = document["rounds"]["round.draft"]
    assert draft["replays"] == 1
    assert draft["families"][0]["family"] == "nccl"
    assert document["launches_outside_rounds"]["count"] == 1
    assert document["kernels_without_correlation"] == 0
    # The outside launch was seen, so its kernel is outside-round work, not a broken join.
    assert document["kernels_outside_rounds"] == 1
    assert document["kernels_unmatched_correlation"] == 0
    # The profiled span is explicitly labelled, never presented as a floor.
    assert document["instrumented_scope"]["span_is_instrumented_not_floor"] is True


def test_nested_ranges_attribute_to_the_innermost():
    trace = _trace(
        _range("round.verify", 100_000.0, 100_000.0),
        _range("round.draft", 120_000.0, 10_000.0),
        _launch(7, 121_000.0),
        _kernel(7, GEMV_FULL_NAME, 121_000.0, 1_000.0),
    )
    document = census_module.census(trace)
    assert "round.draft" in document["rounds"]
    assert document["rounds"]["round.draft"]["kernels_sum"] == 1
    assert document["rounds"].get("round.verify", {}).get("kernels_sum", 0) == 0


def test_kernels_without_correlation_are_counted_not_dropped():
    trace = _trace(
        _range("round.verify", 100_000.0, 50_000.0),
        _launch(1, 101_000.0),
        {"cat": "kernel", "name": GEMV_FULL_NAME, "ts": 101_000.0, "dur": 1_000.0},
        _kernel(99, GEMV_FULL_NAME, 102_000.0, 1_000.0),  # correlation matches no launch
        _kernel(1, ELEMENTWISE_FULL_NAME, 103_000.0, 500.0),
    )
    document = census_module.census(trace)
    assert document["kernels_without_correlation"] == 1
    assert document["kernels_unmatched_correlation"] == 1
    assert document["rounds"]["round.verify"]["kernels_sum"] == 1


def test_launch_without_correlation_is_reported_outside_not_inside_the_round():
    trace = _trace(
        _range("round.verify", 100_000.0, 10_000.0),
        {"cat": "cuda_runtime", "name": "cudaGraphLaunch", "ts": 101_000.0, "dur": 1_000.0},
    )
    document = census_module.census(trace)
    # A round all of whose launches are unusable is reported as absent rather than as zero-work.
    assert "round.verify" not in document["rounds"]
    assert document["launches_outside_rounds"]["count"] == 1
    assert document["launches_outside_rounds"]["by_name"] == {"cudaGraphLaunch:no-correlation": 1}


# ------------------------------------------------------------------------------------------
# Full names: nothing is truncated, nothing is silently bucketed
# ------------------------------------------------------------------------------------------


def test_full_names_are_retained_verbatim_and_distinct_gemv_variants_stay_separate():
    variant = GEMV_FULL_NAME.replace("64, 64, 8", "128, 128, 8")
    trace = _trace(
        _range("round.verify", 100_000.0, 50_000.0),
        _launch(1, 101_000.0),
        _kernel(1, GEMV_FULL_NAME, 101_000.0, 2_000.0),
        _kernel(1, variant, 102_000.0, 1_000.0),
        _kernel(1, ELEMENTWISE_FULL_NAME, 103_000.0, 500.0),
    )
    document = census_module.census(trace)
    names = [row["full_name"] for row in document["rounds"]["round.verify"]["top_kernels"]]
    assert GEMV_FULL_NAME in names
    assert variant in names
    assert ELEMENTWISE_FULL_NAME in names
    for name in names:
        assert len(name) > 90 or name == ELEMENTWISE_FULL_NAME, (
            "a real kernel name shorter than 90 characters is suspicious in this fixture")
    families = {row["family"]: row for row in document["rounds"]["round.verify"]["families"]}
    assert families["cublas_gemvx"]["distinct_full_names"] == 2
    assert families["cublas_gemvx"]["busy_ms_per_replay"] == pytest.approx(3.0)
    assert families["cublas_gemvx"]["share_of_busy"] == pytest.approx(3.0 / 3.5)


def test_no_truncation_artifact_family_is_ever_emitted():
    trace = _trace(
        _range("round.verify", 100_000.0, 50_000.0),
        _launch(1, 101_000.0),
        _kernel(1, GEMV_FULL_NAME, 101_000.0, 2_000.0),
        _kernel(1, ELEMENTWISE_FULL_NAME, 102_000.0, 1_000.0),
        _kernel(1, "void at::native::unusual_internal_kernel<int>(int)", 103_000.0, 500.0),
        _kernel(1, "<unnamed>::mystery_kernel(float*)", 104_000.0, 250.0),
    )
    document = census_module.census(trace)
    reported = {row["family"] for row in document["rounds"]["round.verify"]["families"]}
    for artifact in ("std::enable_if", "", "void", "void at::native::"):
        assert artifact not in reported
    # The names no rule matches are surfaced verbatim so a human can extend the rules.
    assert "<unnamed>::mystery_kernel(float*)" in \
        document["rounds"]["round.verify"]["unclassified_full_names"]


# ------------------------------------------------------------------------------------------
# CLI surface
# ------------------------------------------------------------------------------------------


def test_cli_accepts_bare_list_trace_and_writes_json_out(tmp_path):
    trace = _trace(
        _range("round.verify", 100_000.0, 10_000.0),
        _launch(1, 101_000.0),
        _kernel(1, GEMV_FULL_NAME, 101_000.0, 1_000.0),
    )
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(trace["traceEvents"]))
    output = tmp_path / "census.json"
    assert census_module.main([str(source), "--top", "1",
                               "--json-out", str(output)]) == 0
    document = json.loads(output.read_text())
    assert document["schema"] == census_module.SCHEMA
    verify = document["rounds"]["round.verify"]
    assert verify["top_kernels_shown"] == 1
    assert verify["top_kernels_total"] == 1
    assert verify["top_kernels"][0]["full_name"] == GEMV_FULL_NAME


def test_cli_rejects_a_document_without_events(tmp_path):
    source = tmp_path / "summary.json"
    source.write_text(json.dumps({"round.verify": {"replays": 3}}))
    with pytest.raises(ValueError):
        census_module.main([str(source)])


def test_census_rejects_top_zero(tmp_path):
    source = tmp_path / "trace.json"
    source.write_text(json.dumps(_trace()["traceEvents"]))
    with pytest.raises(SystemExit):
        census_module.main([str(source), "--top", "0"])


# ------------------------------------------------------------------------------------------
# Projection recorder: geometry extraction
# ------------------------------------------------------------------------------------------


class _FakeTensor:
    """The duck-typed tensor surface the recorder is allowed to touch.

    Deliberately does not pretend to be a torch tensor, so a passing test shows the recorder needs
    only `shape`, `dtype`, `stride()` and `is_contiguous()` and therefore cannot be changing
    arithmetic through some other attribute.
    """

    def __init__(self, shape, dtype="torch.bfloat16", strides=None):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._strides = strides

    def stride(self):
        if self._strides is not None:
            return self._strides
        strides = []
        running = 1
        for dimension in reversed(self.shape):
            strides.append(running)
            running *= dimension
        return tuple(reversed(strides))

    def is_contiguous(self):
        return True


def test_describe_linear_reads_the_projection_geometry():
    geometry = probe_module.describe_linear(_FakeTensor((4, 2560)), _FakeTensor((31040, 2560)))
    assert geometry["rows"] == 4
    assert geometry["in_features"] == 2560
    assert geometry["out_features"] == 31040
    assert geometry["x_ndim"] == 2
    assert geometry["hidden_matches_weight"] is True
    assert geometry["x_dtype"] == "torch.bfloat16"
    assert geometry["x_contiguous"] is True
    assert geometry["has_bias"] is False


def test_describe_linear_folds_leading_dimensions_and_keeps_ndim():
    geometry = probe_module.describe_linear(_FakeTensor((1, 4, 2560)),
                                            _FakeTensor((80, 2560)))
    assert geometry["rows"] == 4
    assert geometry["x_ndim"] == 3
    assert geometry["x_shape"] == [1, 4, 2560]


def test_describe_linear_flags_a_hidden_size_mismatch_instead_of_raising():
    geometry = probe_module.describe_linear(_FakeTensor((4, 128)), _FakeTensor((8, 2560)))
    assert geometry["hidden_matches_weight"] is False
    assert geometry["in_features"] == 2560


def test_describe_linear_rejects_non_matrix_weights_and_non_tensors():
    assert probe_module.describe_linear(_FakeTensor((4, 8)), _FakeTensor((8,))) is None
    assert probe_module.describe_linear(object(), _FakeTensor((8, 4))) is None


# ------------------------------------------------------------------------------------------
# Projection recorder: phases, nesting, capacity
# ------------------------------------------------------------------------------------------


def _frame():
    """A real caller frame so `caller_site` gets a genuine code object."""
    import inspect
    return inspect.currentframe().f_back


def _record_once(recorder, *, kind="F_linear", x=None, weight=None, frame=None,
                 capturing=False):
    """Record one complete intercepted call the way the wrapper does: open, then close."""
    key = recorder.record(kind=kind, args=(x, weight), kwargs={},
                          frame=frame if frame is not None else _frame(),
                          capturing=capturing)
    recorder.finish_call(key)
    return key


def test_recorder_records_prefill_but_skips_steady_eager_decode():
    recorder = probe_module.ProjectionRecorder(root=Path("/repo"))
    frame = _frame()
    x = _FakeTensor((4, 2560))
    weight = _FakeTensor((31040, 2560))
    _record_once(recorder, x=x, weight=weight, frame=frame)
    assert recorder.document()["calls_total"] == 1
    recorder.set_phase(probe_module.PHASE_DECODE_SETUP)
    _record_once(recorder, x=x, weight=weight, frame=frame)
    assert recorder.document()["calls_total"] == 1
    assert recorder.document()["skipped_steady_calls"] == 1
    # A capture pass after prefill is recorded: that call is what becomes the replay.
    _record_once(recorder, x=x, weight=weight, frame=frame, capturing=True)
    document = recorder.document()
    assert document["calls_total"] == 2
    assert document["calls_by_phase"] == {"prefill": 1, "decode_setup": 1}


def test_recorder_attributes_a_nested_row_loop_to_its_parent_projection():
    recorder = probe_module.ProjectionRecorder(root=Path("/repo"))
    frame = _frame()
    x = _FakeTensor((4, 2560))
    weight = _FakeTensor((80, 2560))
    parent = recorder.record(kind="row_invariant_linear", args=(x, weight), kwargs={},
                             frame=frame, capturing=False)
    assert parent is not None
    for _ in range(4):
        inner = recorder.record(kind="F_linear", args=(_FakeTensor((2560,)), weight),
                                kwargs={}, frame=frame, capturing=False)
        assert inner is None  # a nested call opens nothing, so it cannot pop its parent
        recorder.finish_call(inner)
    recorder.finish_call(parent)
    document = recorder.document()
    assert document["calls_total"] == 1
    assert document["serial_children_total"] == 4
    assert document["entries"][0]["kind"] == "row_invariant_linear"


def test_recorder_second_top_level_call_after_a_finished_one_opens_its_own_entry():
    recorder = probe_module.ProjectionRecorder(root=Path("/repo"))
    frame = _frame()
    x = _FakeTensor((4, 2560))
    weight = _FakeTensor((80, 2560))
    # The same source line twice: `caller_site` keys on the calling line, so two further calls from
    # one call site must share an entry instead of opening one per iteration.
    for _ in range(2):
        _record_once(recorder, x=x, weight=weight, frame=frame)
    document = recorder.document()
    assert document["calls_total"] == 2
    assert document["serial_children_total"] == 0
    assert len(document["entries"]) == 1
    assert document["entries"][0]["calls"] == 2


def test_recorder_separates_distinct_call_sites():
    recorder = probe_module.ProjectionRecorder(root=Path("/repo"))
    frame = _frame()
    x = _FakeTensor((4, 2560))
    weight = _FakeTensor((80, 2560))
    _record_once(recorder, x=x, weight=weight, frame=frame)
    _record_once(recorder, x=x, weight=weight, frame=frame)
    document = recorder.document()
    # Two different lines are two call sites, which is what makes a site report actionable.
    assert len(document["entries"]) == 2
    assert {entry["calls"] for entry in document["entries"]} == {1}


def test_recorder_capacity_is_reported_not_silently_dropped():
    recorder = probe_module.ProjectionRecorder(root=Path("/repo"), max_entries=1)
    frame = _frame()
    _record_once(recorder, x=_FakeTensor((4, 2560)), weight=_FakeTensor((80, 2560)), frame=frame)
    _record_once(recorder, x=_FakeTensor((4, 2560)), weight=_FakeTensor((160, 2560)), frame=frame)
    document = recorder.document()
    assert document["skipped_capacity"] == 1
    assert len(document["entries"]) == 1


def test_recorder_rejects_unknown_phase():
    recorder = probe_module.ProjectionRecorder(root=Path("/repo"))
    with pytest.raises(ValueError):
        recorder.set_phase("steady")


def test_hot_projection_shapes_dedupe_across_sites_and_carry_one_call_bytes():
    recorder = probe_module.ProjectionRecorder(root=Path("/repo"))
    frame = _frame()
    for _ in range(2):
        _record_once(recorder, x=_FakeTensor((4, 2560)), weight=_FakeTensor((80, 2560)),
                     frame=frame)
    document = recorder.document()
    hot = document["hot_projection_shapes"]
    assert len(hot) == 1
    assert hot[0]["calls"] == 2
    assert hot[0]["weight_bytes_one_call"] == 80 * 2560 * 2
    assert hot[0]["x_ndim"] == 2


def test_caller_site_is_relative_to_the_named_root():
    frame = _frame()
    site = probe_module.caller_site(frame, root=Path.cwd())
    assert site.startswith("tests/") or site.startswith(".."), site
    # `_frame` returns its caller, so the reported function is the test that asked for it.
    assert site.split(":")[-1] == "test_caller_site_is_relative_to_the_named_root"


# ------------------------------------------------------------------------------------------
# Delegation: the observer must return the original arithmetic's own result
# ------------------------------------------------------------------------------------------


def test_observer_patches_linear_delegates_and_restores():
    torch = pytest.importorskip("torch")
    import torch.nn.functional as functional

    recorder = probe_module.ProjectionRecorder(root=Path.cwd())
    original = functional.linear
    x = torch.randn(4, 8, dtype=torch.bfloat16)
    weight = torch.randn(16, 8, dtype=torch.bfloat16)
    expected = functional.linear(x, weight)
    with probe_module.observe_projection_calls(recorder) as installed:
        assert installed is recorder
        assert functional.linear is not original
        observed = functional.linear(x, weight)
    assert functional.linear is original, "the observer must restore the original callable"
    assert torch.equal(observed, expected)
    document = recorder.document()
    assert document["calls_total"] == 1
    entry = document["entries"][0]
    assert entry["kind"] == "F_linear"
    assert (entry["rows"], entry["in_features"], entry["out_features"]) == (4, 8, 16)


def test_observer_restores_the_callables_even_when_the_body_raises():
    pytest.importorskip("torch")
    import torch.nn.functional as functional

    recorder = probe_module.ProjectionRecorder(root=Path.cwd())
    original = functional.linear
    with pytest.raises(RuntimeError, match="boom"):
        with probe_module.observe_projection_calls(recorder):
            raise RuntimeError("boom")
    assert functional.linear is original


def test_observer_delegates_row_invariant_linear_and_counts_the_row_loop_as_children(monkeypatch):
    torch = pytest.importorskip("torch")
    from relicllm.models.qwen4_exp import layers

    monkeypatch.setenv("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "1")
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    weight = torch.randn(4, 8, dtype=torch.bfloat16)
    expected = layers.row_invariant_linear(x, weight)
    recorder = probe_module.ProjectionRecorder(root=Path.cwd())
    with probe_module.observe_projection_calls(recorder):
        observed = layers.row_invariant_linear(x, weight)
    assert torch.equal(observed, expected)
    document = recorder.document()
    entries = [entry for entry in document["entries"]
               if entry["kind"] == "row_invariant_linear"]
    assert len(entries) == 1
    assert entries[0]["calls"] == 1
    assert entries[0]["serial_children"] == 2
    # The inner single-row calls must not be reported as independent projections.
    assert all(entry["rows"] != 1 for entry in document["entries"])


# ------------------------------------------------------------------------------------------
# Micro driver: shape extraction, errpaths, exclusive evidence writes
# ------------------------------------------------------------------------------------------


def _observation(shapes, schema=probe_module.SCHEMA):
    return {"schema": schema, "hot_projection_shapes": shapes}


def test_micro_shapes_dedupes_and_keeps_x_ndim():
    shapes = probe_module.micro_shapes(_observation([
        {"rows": 4, "in_features": 2560, "out_features": 80, "weight_dtype": "torch.bfloat16",
         "x_ndim": 2},
        {"rows": 4, "in_features": 2560, "out_features": 80, "weight_dtype": "torch.bfloat16",
         "x_ndim": 2},
        {"rows": 4, "in_features": 2560, "out_features": 80, "weight_dtype": "torch.bfloat16",
         "x_ndim": 3},
    ]))
    assert len(shapes) == 2
    assert {shape["x_ndim"] for shape in shapes} == {2, 3}


def test_micro_shapes_rejects_a_foreign_schema():
    with pytest.raises(ValueError, match="observation document"):
        probe_module.micro_shapes({"schema": "something-else", "hot_projection_shapes": []})


def test_micro_shapes_rejects_an_empty_or_negative_shape():
    with pytest.raises(ValueError, match="no hot_projection_shapes"):
        probe_module.micro_shapes(_observation([]))
    with pytest.raises(ValueError, match="must be positive"):
        probe_module.micro_shapes(_observation([
            {"rows": 0, "in_features": 2560, "out_features": 80,
             "weight_dtype": "torch.bfloat16", "x_ndim": 2}]))


def test_micro_entry_point_refuses_to_load_a_model_and_saved_shapes_together(tmp_path):
    observation = tmp_path / "obs.json"
    observation.write_text(json.dumps(_observation([
        {"rows": 4, "in_features": 8, "out_features": 4, "weight_dtype": "torch.bfloat16",
         "x_ndim": 2}])))
    with pytest.raises(SystemExit, match="saved shapes only"):
        probe_module.main(["--micro-from", str(observation), "--micro-out",
                           str(tmp_path / "micro.json"), "--model", "/nope"])


def test_micro_entry_point_requires_an_output_path(tmp_path):
    observation = tmp_path / "obs.json"
    observation.write_text(json.dumps(_observation([
        {"rows": 4, "in_features": 8, "out_features": 4, "weight_dtype": "torch.bfloat16",
         "x_ndim": 2}])))
    with pytest.raises(SystemExit, match="--micro-out"):
        probe_module.main(["--micro-from", str(observation)])


def test_observation_pass_requires_model_and_out():
    with pytest.raises(SystemExit, match="--model and --out"):
        probe_module.main([])


def test_evidence_write_never_replaces_an_existing_file(tmp_path):
    target = tmp_path / "evidence.json"
    target.write_text("previous\n")
    with pytest.raises(FileExistsError, match="already exists"):
        probe_module._write_exclusive(target, {"schema": probe_module.SCHEMA})
    assert target.read_text() == "previous\n"


def test_evidence_write_is_atomic_and_json_stable(tmp_path):
    target = tmp_path / "evidence.json"
    document = {"schema": probe_module.SCHEMA, "rows": [{"rows": 4, "in_features": 8}]}
    probe_module._write_exclusive(target, document)
    assert json.loads(target.read_text()) == document
    assert list(tmp_path.iterdir()) == [target], "a temporary file must not survive the write"


# ------------------------------------------------------------------------------------------
# The serial micro must not become a batched-arithmetic allowlist by accident
# ------------------------------------------------------------------------------------------


def test_serial_row_tensor_restores_the_single_row_leading_shape():
    torch = pytest.importorskip("torch")
    flat = torch.randn(3, 8)
    # Every row is handed over with all leading dimensions set to one, never flattened to a
    # bare vector: that leading shape is what makes a recorded call a one-row projection.
    for ndim in (2, 3, 4):
        row = probe_module.serial_row_tensor(flat, 1, ndim)
        assert row.shape == (1,) * (ndim - 1) + (8,), (ndim, row.shape)
        assert torch.equal(row.reshape(-1), flat[1])
        assert row.is_contiguous()


def test_micro_note_declares_serial_only_and_excludes_a_batched_candidate():
    source = Path(probe_module.__file__).read_text()
    assert '"serial_only": True' in source
    note = source.split('"note": (', 1)[1].split("),", 1)[0]
    assert "No bmm or batched replacement is measured" in note.replace("\n", " ")
    # The two frozen command surfaces must both exist as separate modes.
    assert "--micro-from" in source and '"--out"' in source


def test_census_and_probe_schemas_are_pinned_constants():
    assert census_module.SCHEMA == "qwen4-exp-kernel-census-v1"
    assert probe_module.SCHEMA == "qwen4-exp-linear-hotspots-v1"
    assert probe_module.MICRO_SCHEMA == "qwen4-exp-linear-hotspot-micro-v1"
    assert os.path.basename(census_module.__file__) == "analyze_qwen4_exp_kernel_census.py"
    assert os.path.basename(probe_module.__file__) == "probe_qwen4_exp_linear_hotspots.py"