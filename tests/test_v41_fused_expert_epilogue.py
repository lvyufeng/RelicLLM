"""Explicit expert-epilogue ownership and unchanged CPU/prefill dispatch."""
import pytest
import torch

from relicllm.models.deepseek_v4_1 import ascend_packed_experts as packed
from relicllm.models.deepseek_v4_1.fused_expert_epilogue import FusedExpertEpilogue


def test_cpu_body_is_original_and_import_or_install_does_not_touch_Engram(monkeypatch):
    calls = []
    result = object()
    def original(x, resident, **options):
        calls.append((x, resident, options))
        return result
    def forbidden(*args, **options):
        pytest.fail("CPU input entered an NPU epilogue")
    monkeypatch.setattr(packed, "packed_expert_forward", original)
    x = torch.ones((1, 64), dtype=torch.bfloat16)
    resident = object()
    with FusedExpertEpilogue(operator=forbidden) as owner:
        assert packed.packed_expert_forward(x, resident, backend="torch") is result
        assert packed.packed_expert_forward(x.expand(2, 64), resident, prefill=True) is result
        assert owner.describe()["fused_expert_calls"] == 0
        assert owner.describe()["original_passthroughs"] == 2
    assert len(calls) == 2 and packed.packed_expert_forward is original


def test_epilogue_install_has_one_owner_and_close_is_idempotent():
    original = packed.packed_expert_forward
    first, second = FusedExpertEpilogue(), FusedExpertEpilogue()
    with first:
        with pytest.raises(RuntimeError, match="another fused epilogue"):
            second.install()
        with pytest.raises(RuntimeError, match="already installed"):
            first.install()
    first.close()
    assert packed.packed_expert_forward is original


def test_outer_graph_owner_must_release_before_epilogue_close():
    original = packed.packed_expert_forward
    owner = FusedExpertEpilogue().install()
    replacement = packed.packed_expert_forward
    try:
        packed.packed_expert_forward = lambda *args, **options: None
        with pytest.raises(RuntimeError, match="release outer expert graphs"):
            owner.close()
    finally:
        packed.packed_expert_forward = replacement
        owner.close()
    assert packed.packed_expert_forward is original


def test_invalid_epilogue_operator_is_refused_before_a_module_patch():
    original = packed.packed_expert_forward
    with pytest.raises(TypeError, match="must be callable"):
        FusedExpertEpilogue(operator="invalid").install()
    assert packed.packed_expert_forward is original
