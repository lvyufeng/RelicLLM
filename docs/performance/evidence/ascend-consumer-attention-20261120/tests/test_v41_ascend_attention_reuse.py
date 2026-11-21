"""CPU contracts for immutable call-scoped original FP8 attention reuse."""
from types import SimpleNamespace
from threading import Thread

import pytest
import torch
from torch import nn

from relic_core.kernels import block_fp8_quant
from relicllm.models.deepseek_v4_1.ascend_attention_reuse import AttentionFP8Reuse


class Attention(nn.Module):
    def forward(self, x, *args, **kwargs):
        flat = x.reshape(-1, x.shape[-1]).contiguous()
        first = block_fp8_quant.quantize_block_fp8(flat, backend="torch")
        second = block_fp8_quant.quantize_block_fp8(x.reshape(-1, x.shape[-1]), backend="torch")
        return first, second


def test_same_storage_different_views_reuse_exact_original_bytes():
    x = torch.linspace(-1.2, 1.7, 64).reshape(1, 64).bfloat16()
    original = block_fp8_quant.quantize_block_fp8
    expected = original(x, backend="torch")
    with AttentionFP8Reuse(backend="torch", certify_immutable_inputs=True) as controller:
        with controller.scope():
            first = block_fp8_quant.quantize_block_fp8(x, backend="torch")
            second = block_fp8_quant.quantize_block_fp8(x.view(1, 64), backend="torch")
            assert first[0] is second[0] and first[1] is second[1]
            assert all(torch.equal(a, b) for a, b in zip(first, expected))
        assert controller.prepared is None
        with controller.scope():
            third = block_fp8_quant.quantize_block_fp8(x, backend="torch")
            assert third[0] is not first[0]
        assert controller.stats["quantizer_calls"] == 2
        assert controller.stats["reuses"] == 1
    assert block_fp8_quant.quantize_block_fp8 is original


def test_input_mutation_invalidates_instead_of_reusing_stale_codes():
    x = torch.ones(1, 32, dtype=torch.bfloat16)
    original = block_fp8_quant.quantize_block_fp8
    with AttentionFP8Reuse(backend="torch", certify_immutable_inputs=True) as controller:
        with controller.scope():
            first = block_fp8_quant.quantize_block_fp8(x, backend="torch")
            x[0, 3] = -4
            second = block_fp8_quant.quantize_block_fp8(x, backend="torch")
            expected = original(x, backend="torch")
            assert second[0] is not first[0]
            assert all(torch.equal(a, b) for a, b in zip(second, expected))
            assert controller.stats["reuses"] == 0


def test_code_mutation_is_refused_inside_scope():
    x = torch.ones(1, 32, dtype=torch.bfloat16)
    with AttentionFP8Reuse(backend="torch", certify_immutable_inputs=True) as controller:
        with controller.scope():
            codes, _ = block_fp8_quant.quantize_block_fp8(x, backend="torch")
            codes[0, 2] = 0
            with pytest.raises(RuntimeError, match="preparation changed"):
                block_fp8_quant.quantize_block_fp8(x, backend="torch")


def test_other_thread_and_outside_scope_do_not_borrow_preparation():
    x = torch.ones(1, 32, dtype=torch.bfloat16)
    results = []
    with AttentionFP8Reuse(backend="torch", certify_immutable_inputs=True) as controller:
        outside = block_fp8_quant.quantize_block_fp8(x, backend="torch")
        with controller.scope():
            inside = block_fp8_quant.quantize_block_fp8(x, backend="torch")
            thread = Thread(target=lambda: results.append(block_fp8_quant.quantize_block_fp8(x, backend="torch")))
            thread.start(); thread.join()
            assert outside[0] is not inside[0]
            assert results[0][0] is not inside[0]
            assert controller.stats["outside_scope_bypasses"] == 2


def test_prefill_bypasses_and_method_descriptor_restores():
    attention = Attention()
    model = SimpleNamespace(layers=[SimpleNamespace(attn=attention)])
    original = attention.forward
    with AttentionFP8Reuse(model, backend="torch", certify_immutable_inputs=True) as controller:
        first, second = attention(torch.ones(1, 1, 32, dtype=torch.bfloat16))
        assert first[0] is second[0]
        assert controller.stats["scopes"] == 1
        first, second = attention(torch.ones(1, 3, 32, dtype=torch.bfloat16))
        assert first[0] is not second[0]
        assert controller.stats["scopes"] == 1
    assert "forward" not in attention.__dict__
    assert attention.forward.__func__ is original.__func__


def test_scope_failure_clears_and_nested_or_early_close_is_refused():
    with AttentionFP8Reuse(backend="torch", certify_immutable_inputs=True) as controller:
        with pytest.raises(RuntimeError, match="intentional"):
            with controller.scope():
                with pytest.raises(RuntimeError, match="nested"):
                    with controller.scope():
                        pass
                with pytest.raises(RuntimeError, match="scope first"):
                    controller.close()
                raise RuntimeError("intentional")
        assert controller.prepared is None


def test_scope_retention_is_bounded_and_extra_inputs_bypass_safely():
    inputs = [torch.full((1, 32), float(index), dtype=torch.bfloat16) for index in range(10)]
    with AttentionFP8Reuse(backend="torch", certify_immutable_inputs=True) as controller:
        with controller.scope():
            for value in inputs:
                block_fp8_quant.quantize_block_fp8(value, backend="torch")
            assert len(controller.prepared) == 8
            assert controller.stats["peak_live_preparations"] == 8
            assert controller.stats["unsupported_bypasses"] == 2


def test_explicit_source_immutability_is_required():
    with pytest.raises(ValueError, match="immutable"):
        AttentionFP8Reuse(backend="torch")


def test_foreign_quantizer_owner_is_not_overwritten_on_close():
    controller = AttentionFP8Reuse(backend="torch", certify_immutable_inputs=True).install()
    installed = block_fp8_quant.quantize_block_fp8
    foreign = lambda *_: None
    block_fp8_quant.quantize_block_fp8 = foreign
    with pytest.raises(RuntimeError, match="owner"):
        controller.close()
    assert block_fp8_quant.quantize_block_fp8 is foreign
    block_fp8_quant.quantize_block_fp8 = installed
    controller.close(); controller.close()
