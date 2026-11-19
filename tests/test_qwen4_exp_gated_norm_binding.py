"""The optional GDN epilogue keeps torch's statistic and activation unchanged."""
from types import SimpleNamespace
import pytest
import torch
import torch.nn.functional as F
from relicllm.models.qwen4_exp import layers


def reference(x, gate, weight, eps, activation):
    h=x.float()
    h=h*torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+eps)
    h=weight*h.to(x.dtype)
    h=h*(torch.sigmoid(gate.float()) if activation=="sigmoid" else F.silu(gate.float()))
    return h.to(x.dtype)


@pytest.mark.parametrize("activation", ["sigmoid","silu"])
@pytest.mark.parametrize("rows", [6,24,30,36,48])
@pytest.mark.parametrize("gate_dtype", [torch.bfloat16,torch.float32])
def test_binding_keeps_inverse_activation_and_bf16_boundaries(monkeypatch,activation,rows,gate_dtype):
    x=torch.randn(rows,128,dtype=torch.bfloat16)
    gate=torch.randn_like(x,dtype=gate_dtype)
    weight=torch.randn(128,dtype=torch.bfloat16)
    expected=reference(x,gate,weight,1e-6,activation)
    calls=[]

    def kernel(value,inverse,w,activated):
        calls.append((value,inverse,w,activated))
        assert torch.equal(inverse,torch.rsqrt(x.float().pow(2).mean(-1,keepdim=True)+1e-6))
        assert torch.equal(activated,torch.sigmoid(gate.float()) if activation=="sigmoid" else F.silu(gate.float()))
        h=(value.float()*inverse).to(torch.bfloat16)
        h=(w*h).to(torch.bfloat16)
        return (h*activated).to(torch.bfloat16)

    monkeypatch.setenv("POCKETLLM_QWEN4_GATED_NORM_EXACT","1")
    monkeypatch.setattr(torch.Tensor,"device",property(lambda _self: torch.device("cuda",0)))
    monkeypatch.setattr(layers,"load_ops",lambda: SimpleNamespace(qwen4_exp_gated_rms_norm_exact=kernel))
    actual=layers.RMSNormGated(weight,1e-6,activation=activation)(x,gate)
    assert torch.equal(actual,expected) and len(calls)==1


def test_default_off_does_not_load_an_extension(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_GATED_NORM_EXACT","0")
    monkeypatch.setattr(layers,"load_ops",lambda: pytest.fail("default-off must not reach CUDA"))
    x=torch.randn(6,128,dtype=torch.bfloat16);g=torch.randn_like(x);w=torch.randn(128,dtype=torch.bfloat16)
    assert torch.equal(layers.RMSNormGated(w,1e-6)(x,g),reference(x,g,w,1e-6,"silu"))


def test_enabled_cpu_keeps_the_old_path(monkeypatch):
    monkeypatch.setenv("POCKETLLM_QWEN4_GATED_NORM_EXACT","1")
    monkeypatch.setattr(layers,"load_ops",lambda: pytest.fail("CPU must keep fallback"))
    x=torch.randn(6,128,dtype=torch.bfloat16);g=torch.randn_like(x);w=torch.randn(128,dtype=torch.bfloat16)
    assert torch.equal(layers.RMSNormGated(w,1e-6)(x,g),reference(x,g,w,1e-6,"silu"))


@pytest.mark.parametrize("unsupported", ["missing_entry","noncontiguous","dtype","activation","empty"])
def test_unsupported_dispatch_falls_back(monkeypatch,unsupported):
    monkeypatch.setenv("POCKETLLM_QWEN4_GATED_NORM_EXACT","1")
    x=torch.randn(6,128,dtype=torch.bfloat16);g=torch.randn_like(x);w=torch.randn(128,dtype=torch.bfloat16)
    activation="silu"
    if unsupported=="noncontiguous":x=torch.randn(6,256,dtype=torch.bfloat16)[:,::2]
    if unsupported=="dtype":x=x.float()
    if unsupported=="activation":activation="other"
    if unsupported=="empty":x=x[:0];g=g[:0]
    monkeypatch.setattr(torch.Tensor,"device",property(lambda _self: torch.device("cuda",0)))
    if unsupported=="missing_entry":monkeypatch.setattr(layers,"load_ops",lambda: SimpleNamespace())
    else:monkeypatch.setattr(layers,"load_ops",lambda: pytest.fail("unsupported must not look up extension"))
    assert layers._cuda_gated_rms_norm_exact(x,g,w,1e-6,activation) is None
