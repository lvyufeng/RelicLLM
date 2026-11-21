"""Complete-FFN graph adaptation and dispatch ownership on explicit CPU stubs."""
from threading import RLock
from types import SimpleNamespace
import types

import pytest
import torch

from relicllm.models.deepseek_v4_1.ascend_global_ffn import FullFFNGlobalDispatch, _OWNER


def fixture():
    owner=FullFFNGlobalDispatch.__new__(FullFFNGlobalDispatch)
    owner.installed=True;owner.adapted=[]
    ffn=SimpleNamespace(dim=32,layer_id=0)
    setattr(ffn,_OWNER,owner)
    sink=SimpleNamespace(ffn_in=torch.ones(1,1,32,dtype=torch.bfloat16),weights=torch.ones(1,2),indices=torch.tensor([[3,1]]),
                         y_in=None,x_out=None,residual=torch.ones(1,1,32,dtype=torch.bfloat16),ffn_post=None,ffn_comb=None)
    block=SimpleNamespace(ffn=ffn,hc_post=lambda x,residual,post,comb:x+residual)
    layer=SimpleNamespace(block=block,sink=sink,ready=False,_routed=lambda:None,_body_b=lambda:None)
    calls=[]
    owner.forward_routed_and_shared=lambda ffn,x,w,ids:(calls.append((ffn,x,w,ids)) or torch.full((1,32),2.0))
    return owner,layer,calls


def test_adapted_body_computes_complete_ffn_once_then_only_hc_fold():
    owner,layer,calls=fixture()
    original=(layer._routed,layer._body_b)
    owner.adapt_decode_graphs(SimpleNamespace(layers=[layer]))
    layer._routed();layer._body_b()
    assert len(calls)==1 and torch.equal(layer.sink.x_out,torch.full((1,1,32),3.0,dtype=torch.bfloat16))
    assert owner.adapted[0][1:3]==original
    with pytest.raises(RuntimeError,match='once'):
        owner.adapt_decode_graphs(SimpleNamespace(layers=[layer]))


def test_existing_capture_is_not_silently_rebound_to_different_ffn_arithmetic():
    owner,layer,_=fixture();layer.ready=True
    with pytest.raises(ValueError,match='capture'):
        owner.adapt_decode_graphs(SimpleNamespace(layers=[layer]))
    assert not owner.adapted


def test_foreign_full_ffn_owner_is_not_implicitly_adopted():
    owner,layer,_=fixture();setattr(layer.block.ffn,_OWNER,object())
    with pytest.raises(ValueError,match='capture'):
        owner.adapt_decode_graphs(SimpleNamespace(layers=[layer]))


class LifecycleFFN:
    def __init__(self, layer):
        self.dim = 32
        self.n_activated_experts = 2
        self.layer_id = layer
        self.tp = SimpleNamespace(world=8, rank=0)
        self.routed = SimpleNamespace(dim=32, topk=2, layer_id=layer)
        self.fail_install = False

    def __setattr__(self, name, value):
        if name == 'forward' and getattr(self, 'fail_install', False):
            object.__setattr__(self, 'fail_install', False)
            raise RuntimeError('injected FFN install failure')
        object.__setattr__(self, name, value)

    def forward(self, x, image_mask=None):
        return x


class LifecycleResidency:
    def __init__(self, model, **kwargs):
        self.device = torch.device('cpu')
        self.world, self.rank = 8, 0
        self.entries = [(block.ffn, block.ffn.routed, 64) for block in model.layers]
        self.workspace = SimpleNamespace(nbytes=64)
        self.replacements, self.close_policies = [], []

    def install(self):
        for ffn, original, _ in self.entries:
            shard = SimpleNamespace(banks={}, dim=32, topk=2, layer_id=ffn.layer_id)
            ffn.routed = shard
            self.replacements.append((ffn, original, shard))
        return self

    def close(self, *, restore_residency=None):
        self.close_policies.append(restore_residency)
        for ffn, original, shard in self.replacements:
            ffn.routed = original
            shard.banks.clear()
        self.replacements.clear()

    def describe(self):
        return {'expert_TP':2, 'expert_EP':4, 'resident_bytes':0,
                'one_TP_leader_emits_into_original_global_MoE_reduce':True,
                'FP32_down_TP_reduction_before_BF16_narrowing':True,
                'numerical_contract':'old routed-only fixture', 'graphs':0, 'calls':0}


def lifecycle(monkeypatch, count=1):
    from relicllm.models.deepseek_v4_1 import ascend_global_ffn as module
    monkeypatch.setattr(module, 'ExpertTPDispatch', LifecycleResidency)
    monkeypatch.setattr(torch, 'npu', SimpleNamespace(synchronize=lambda *args:None), raising=False)
    model = SimpleNamespace(layers=[SimpleNamespace(ffn=LifecycleFFN(layer), decode_graph=None) for layer in range(count)])
    owner = module.FullFFNGlobalDispatch(model, consumer=object())
    return model, owner


def test_complete_ffn_close_restores_descriptors_and_is_idempotent(monkeypatch):
    model, owner = lifecycle(monkeypatch)
    ffn = model.layers[0].ffn
    original = ffn.forward
    owner.install()
    assert 'forward' in ffn.__dict__
    owner.close(restore_residency=False)
    assert 'forward' not in ffn.__dict__ and ffn.forward == original
    assert not hasattr(ffn, _OWNER) and owner.slots is None
    assert owner.residency.close_policies == [False]
    owner.close()
    assert owner.residency.close_policies == [False]


def test_complete_ffn_install_failure_rolls_back_original_residency(monkeypatch):
    model, owner = lifecycle(monkeypatch, count=2)
    model.layers[1].ffn.fail_install = True
    with pytest.raises(RuntimeError, match='injected FFN install'):
        owner.install()
    assert all('forward' not in block.ffn.__dict__ and not hasattr(block.ffn, _OWNER) for block in model.layers)
    assert all(block.ffn.routed is original for block, (_, original, _) in zip(model.layers, owner.residency.entries))
    assert owner.residency.close_policies == [True] and owner.closed and owner.slots is None


def test_sync_failure_still_retires_full_ffn_owners_without_device_reload(monkeypatch):
    model, owner = lifecycle(monkeypatch)
    owner.install()
    def fail(*args):
        raise RuntimeError('injected global FFN sync failure')
    monkeypatch.setattr(torch.npu, 'synchronize', fail)
    with pytest.raises(RuntimeError, match='sync failure'):
        owner.close()
    assert owner.closed and not owner.installed and owner.slots is None
    assert 'forward' not in model.layers[0].ffn.__dict__ and not hasattr(model.layers[0].ffn, _OWNER)
    assert owner.residency.close_policies == [False]


def test_active_decode_recordings_must_release_before_global_ffn_close(monkeypatch):
    model, owner = lifecycle(monkeypatch)
    owner.install()
    layer = SimpleNamespace(block=model.layers[0], ready=False, graph_a=object(), graph_b=None,
                            _routed=lambda:None, _body_b=lambda:None)
    model.layers[0].decode_graph = layer
    original = (layer._routed, layer._body_b)
    owner.adapt_decode_graphs(SimpleNamespace(layers=[layer]))
    with pytest.raises(RuntimeError, match='release adapted decode graphs'):
        owner.close()
    assert owner.installed and not owner.residency.close_policies
    model.layers[0].decode_graph = None
    layer.graph_a = None
    owner.close()
    assert (layer._routed, layer._body_b) == original


def test_foreign_forward_override_is_not_erased_by_global_ffn_close(monkeypatch):
    model, owner = lifecycle(monkeypatch)
    owner.install()
    ffn = model.layers[0].ffn
    installed = ffn.forward
    foreign = lambda x:x
    ffn.forward = foreign
    with pytest.raises(RuntimeError, match='another owner'):
        owner.close()
    assert ffn.forward is foreign and owner.installed
    ffn.forward = installed
    owner.close()


def test_global_description_does_not_claim_the_bypassed_TP_leader_path(monkeypatch):
    _, owner = lifecycle(monkeypatch)
    owner.install()
    row = owner.describe()
    assert row['one_global_FFN_collective_per_layer']
    assert 'one_TP_leader_emits_into_original_global_MoE_reduce' not in row['residency']
    assert row['residency']['placement_metadata_only_not_the_TP_dispatch_execution_path']
    owner.close()
