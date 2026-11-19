"""Eager-completed indexer prefixes are valid even before the arena is replay-ready."""
import torch
from relicllm.models.qwen4_exp import pd_state
from test_qwen4_exp_pd_state import backbone_cache, head_cache, model_stub, Indexer


def test_eager_completed_pool_is_transferred_and_rebound_without_recompute():
    source=backbone_cache(12)
    head=head_cache(8)
    source.layers[1].indexer_pool.tensor_ready=False
    head.layers[0].indexer_pool.tensor_ready=False
    original=source.layers[1].indexer_pool.keys[:,:3].clone()
    head_original=head.layers[0].indexer_pool.keys[:,:2].clone()
    bundle=pd_state.freeze_state(cache=source,head_cache=head,past=12,next_token=17)
    assert "backbone.layer.1.indexer_keys" in bundle.tensors
    assert "head.layer.0.indexer_keys" in bundle.tensors
    assert torch.equal(bundle.tensors["backbone.layer.1.indexer_keys"],original)
    target=backbone_cache(0,seed=23)
    target_head=head_cache(0,seed=24)
    target.layers[1].indexer_pool=None
    target_head.layers[0].indexer_pool=None
    report=pd_state.thaw_state(bundle,cache=target,head_cache=target_head,
                              model=model_stub(backbone_indexer=Indexer(),head_indexer=Indexer()))
    assert report["indexer_pool_rebound"]==2 and report["indexer_pool_needs_recompute"]==0
    assert report["head_rope_table_ready"] and target_head.cos is None
    assert torch.equal(target.layers[1].indexer_pool.keys[:,:3],original)
    assert torch.equal(target_head.layers[0].indexer_pool.keys[:,:2],head_original)
    assert target.layers[1].indexer_pool.tensor_ready
    assert target_head.layers[0].indexer_pool.tensor_ready


def test_invalidated_pool_has_no_completed_prefix_to_export():
    source=backbone_cache(12)
    source.layers[1].indexer_pool.invalidate()
    bundle=pd_state.freeze_state(cache=source,past=12,next_token=17)
    assert "backbone.layer.1.indexer_keys" not in bundle.tensors
