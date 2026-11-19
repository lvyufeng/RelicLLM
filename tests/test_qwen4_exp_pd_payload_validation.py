"""Malformed pooled payloads and layout aliases fail before touching live state."""
import copy
import pytest
import torch
from relicllm.models.qwen4_exp import pd_state
from test_qwen4_exp_pd_state import backbone_cache


@pytest.mark.parametrize("mutation", ["pool_shape", "pool_dtype", "duplicate_index", "unknown_kind"])
def test_consistent_but_wrong_payload_geometry_never_starts_copy(monkeypatch, mutation):
    source=backbone_cache(12)
    bundle=pd_state.freeze_state(cache=source,past=12,next_token=5)
    bundle.manifest=copy.deepcopy(bundle.manifest)
    key="backbone.layer.1.indexer_keys"
    if mutation.startswith("pool_"):
        tensor=bundle.tensors[key]
        tensor=tensor[:,:2].clone() if mutation=="pool_shape" else tensor.to(torch.bfloat16)
        bundle.tensors[key]=tensor
        entry=next(e for e in bundle.manifest["tensors"] if e["name"]==key)
        entry.update(shape=list(tensor.shape),dtype=pd_state._dtype_name(tensor.dtype),
                     nbytes=tensor.numel()*tensor.element_size(),sha256=pd_state.tensor_hashes(bundle)[key])
    elif mutation=="duplicate_index":
        bundle.manifest["backbone"]["layers"][1]=copy.deepcopy(bundle.manifest["backbone"]["layers"][0])
    else:
        bundle.manifest["backbone"]["layers"][1]["kind"]="unknown"
    target=backbone_cache(0,seed=19)
    copies=[]
    original=torch.Tensor.copy_

    def observed(tensor,*args,**kwargs):
        copies.append(tensor)
        return original(tensor,*args,**kwargs)

    monkeypatch.setattr(torch.Tensor,"copy_",observed)
    with pytest.raises(pd_state.PDStateError):
        pd_state.thaw_state(bundle,cache=target)
    assert not copies
