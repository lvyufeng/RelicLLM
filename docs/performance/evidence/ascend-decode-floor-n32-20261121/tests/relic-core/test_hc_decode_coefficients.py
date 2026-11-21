"""CPU FP32 HC projection/RMS oracle and source/geometry guards."""
import pytest
import torch
import torch.nn.functional as F

from relic_core.kernels.hc_decode_coefficients import hc_project_rms


def values():
    generator = torch.Generator().manual_seed(2041)
    return torch.randn(1, 1, 4, 64, generator=generator).bfloat16(), torch.randn(24, 256, generator=generator)


def test_explicit_cpu_oracle_retains_original_fp32_math_and_shape():
    x, weight = values()
    source = x.flatten(2).float()
    expected = F.linear(source, weight) * torch.rsqrt(source.square().mean(-1, keepdim=True) + 1e-6)
    actual = hc_project_rms(x, weight, norm_eps=1e-6, backend="torch")
    assert actual.dtype == torch.float32 and actual.shape == (1, 1, 24)
    assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))


def test_native_projection_never_uses_cpu_as_an_implicit_fallback():
    x, weight = values()
    with pytest.raises(ValueError, match="Triton Ascend"):
        hc_project_rms(x, weight, norm_eps=1e-6)


@pytest.mark.parametrize("epsilon", [True, 0, -1, float('nan'), float('inf')])
def test_invalid_original_epsilon_is_refused(epsilon):
    x, weight = values()
    with pytest.raises(ValueError, match="epsilon"):
        hc_project_rms(x, weight, norm_eps=epsilon, backend="torch")


def test_original_fp32_coefficients_are_not_narrowed_to_bf16():
    x, weight = values()
    with pytest.raises(ValueError, match="FP32"):
        hc_project_rms(x, weight.bfloat16(), norm_eps=1e-6, backend="torch")


def test_prefill_is_not_silently_run_by_the_decode_kernel():
    x, weight = values()
    with pytest.raises(ValueError, match="one contiguous"):
        hc_project_rms(x.expand(1, 2, 4, 64), weight, norm_eps=1e-6, backend="torch")
