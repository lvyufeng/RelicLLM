"""HC FP32 coefficient fusion keeps nonlinear code, original sources and descriptors."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from relicllm.models.deepseek_v4_1 import modules
from relicllm.models.deepseek_v4_1.ascend_hc_coefficients import FusedHCDecodeCoefficients


class Block:
    def __init__(self):
        self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps, self.norm_eps = 4, 20, 1e-6, 1e-6
        for name in ('attn', 'ffn'):
            setattr(self, f'hc_{name}_fn', torch.randn(24, 128))
            setattr(self, f'hc_{name}_scale', torch.ones(3))
            setattr(self, f'hc_{name}_base', torch.randn(24))
    def hc_mixes(self, x, fn, scale, base):
        source = x.flatten(2).float()
        projected = F.linear(source, fn) * torch.rsqrt(source.square().mean(-1, keepdim=True) + self.norm_eps)
        return modules.hc_split_sinkhorn(projected, scale, base, self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)


def fixture():
    torch.manual_seed(72)
    block=Block()
    return SimpleNamespace(layers=[block]), block, torch.randn(1,1,4,32).bfloat16()


def test_cpu_binding_keeps_original_coefficient_math_and_nonlinear_provider():
    model, block, x=fixture()
    args=(block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    expected=block.hc_mixes(x,*args)
    with FusedHCDecodeCoefficients(model,backend='torch',certify_immutable_sources=True) as owner:
        actual=block.hc_mixes(x,*args)
        assert all(torch.equal(a.view(torch.int32),b.view(torch.int32)) for a,b in zip(actual,expected))
        assert owner.split_provider is modules.hc_split_sinkhorn
        assert owner.describe()['original_HC_FP32_weights_not_narrowed']
    assert 'hc_mixes' not in block.__dict__


def test_prefill_passthrough_and_previous_instance_override_are_restored():
    model, block, x=fixture()
    previous=block.hc_mixes
    block.hc_mixes=previous
    with FusedHCDecodeCoefficients(model,backend='torch',certify_immutable_sources=True) as owner:
        block.hc_mixes(x.expand(1,2,4,32), block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base)
        assert owner.passthrough==1
    assert block.hc_mixes is previous


def test_original_fp32_source_mutation_is_not_accepted():
    model, block, x=fixture()
    with FusedHCDecodeCoefficients(model,backend='torch',certify_immutable_sources=True):
        block.hc_attn_fn.add_(1)
        with pytest.raises(RuntimeError,match='immutable'):
            block.hc_mixes(x,block.hc_attn_fn,block.hc_attn_scale,block.hc_attn_base)


def test_immutability_certificate_is_required_for_versionless_inference_sources():
    model, _, _=fixture()
    with pytest.raises(ValueError,match='immutable'):
        FusedHCDecodeCoefficients(model,backend='torch')


def test_invalid_original_coefficient_dtype_and_duplicate_blocks_are_refused():
    model, block, _=fixture()
    block.hc_attn_fn=block.hc_attn_fn.bfloat16()
    with pytest.raises(ValueError,match='FP32'):
        FusedHCDecodeCoefficients(model,backend='torch',certify_immutable_sources=True)
    model, block, _=fixture()
    model.layers.append(block)
    with pytest.raises(ValueError,match='distinct'):
        FusedHCDecodeCoefficients(model,backend='torch',certify_immutable_sources=True)


def test_foreign_method_owner_is_never_overwritten_on_close():
    model, block, _=fixture()
    owner=FusedHCDecodeCoefficients(model,backend='torch',certify_immutable_sources=True).install()
    replacement=block.hc_mixes
    foreign=lambda *args:None
    block.hc_mixes=foreign
    with pytest.raises(RuntimeError,match='another owner'):
        owner.close()
    assert block.hc_mixes is foreign
    block.hc_mixes=replacement
    owner.close()


def test_rebound_hc_source_cannot_reuse_the_old_certified_triplet():
    model, block, x = fixture()
    original = (block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base)
    with FusedHCDecodeCoefficients(model, backend='torch', certify_immutable_sources=True):
        block.hc_attn_fn = block.hc_attn_fn.clone()
        with pytest.raises(RuntimeError, match='immutable'):
            block.hc_mixes(x, *original)
