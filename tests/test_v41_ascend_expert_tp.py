"""Original-byte TP2/EP4 resharding, memory and reversible-dispatch contracts."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from relicllm.models.deepseek_v4_1.ascend_expert_tp import ExpertTPDispatch, shard_original_projection
from relicllm.models.deepseek_v4_1.ascend_packed_experts import AscendPackedRoutedExperts
from relicllm.models.deepseek_v4_1.ascend_tile_major_experts import unpack_code_tiles, unpack_scale_tiles


class Checkpoint:
    def __init__(self, layers=2):
        self.root = "fixture://original-expert-tp"
        generator = torch.Generator().manual_seed(991)
        self.data = {}
        for layer in range(layers):
            for expert in range(8):
                for name in ("w1", "w3", "w2"):
                    n, k = (128, 256) if name == "w2" else (256, 128)
                    key = f"layers.{layer}.ffn.experts.{expert}.{name}.weight"
                    self.data[key] = torch.randint(0, 256, (n, k // 2), generator=generator, dtype=torch.uint8)
                    self.data[key.removesuffix("weight") + "scale"] = torch.full((n, k // 32), 120, dtype=torch.uint8)

    def packed(self, key):
        return self.data[key]

    def is_quantized(self, key):
        return key.removesuffix("weight") + "scale" in self.data

    def is_packed_fp4(self, key):
        return key.endswith("weight")

    def block_size(self, key):
        return 32


def model(rank=0, layers=2):
    checkpoint = Checkpoint(layers)
    blocks = []
    for layer in range(layers):
        store = AscendPackedRoutedExperts(checkpoint, layer, n_experts=8, dim=128,
            inter_dim=256, topk=4, device="cpu", dtype=torch.bfloat16,
            rank=rank, world=8, backend="torch", preload=True)
        blocks.append(SimpleNamespace(ffn=SimpleNamespace(routed=store)))
    return SimpleNamespace(layers=blocks), checkpoint


@pytest.mark.parametrize("rank", range(8))
def test_tp2_ep4_residency_is_same_original_bytes_not_a_second_bank(rank):
    tree, checkpoint = model(rank)
    originals = [block.ffn.routed for block in tree.layers]
    previous = sum(store.resident_bytes for store in originals)
    old_values = {key: value.clone() for key, value in checkpoint.data.items()}
    with ExpertTPDispatch(tree, tp_world=2, rank=rank, world=8, backend="torch", capture_graphs=False,
                          reduce_fp32=lambda value: None) as controller:
        assert controller.describe()["expert_EP"] == 4
        assert controller.describe()["expert_TP"] == 2
        assert controller.describe()["resident_bytes"] == previous
        assert all(store._arenas is None for store in originals)
        for block, source in zip(tree.layers, originals):
            shard = block.ffn.routed
            assert shard.inter_dim == 128 and shard.partial
            assert shard.ep_rank == rank // 2 and shard.tp_rank == rank % 2
            for slot, expert in enumerate(range(rank // 2, 8, 4)):
                for name in ("w1", "w3", "w2"):
                    n, k = (128, 128)
                    key = f"layers.{source.layer_id}.ffn.experts.{expert}.{name}.weight"
                    code, scales = shard_original_projection(checkpoint.packed(key), checkpoint.packed(key.removesuffix("weight") + "scale"),
                        projection=name, tp_rank=rank % 2, tp_world=2, full_inter_dim=256)
                    assert torch.equal(code, unpack_code_tiles(shard.banks[name][0][slot], n, k))
                    assert torch.equal(scales, unpack_scale_tiles(shard.banks[name][1][slot], n, k))
        assert all(torch.equal(value, old_values[key]) for key, value in checkpoint.data.items())
    assert all(block.ffn.routed is original and original.allocated for block, original in zip(tree.layers, originals))


def test_decode_dispatch_reduce_gets_fp32_before_original_bf16_boundary():
    tree, _ = model(0, layers=1)
    seen = []
    def reduce(value):
        seen.append((value.dtype, tuple(value.shape)))
        value.mul_(2)
    with ExpertTPDispatch(tree, rank=0, world=8, tp_world=2, backend="torch", capture_graphs=False, reduce_fp32=reduce) as controller:
        x = torch.randn(1, 128).bfloat16()
        weights = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        indices = torch.tensor([[0, 1, 4, 5]])
        result = tree.layers[0].ffn.routed.forward(x, weights, indices)
        assert seen == [(torch.float32, (4, 128))]
        assert result.dtype == torch.float32 and result.abs().max() > 0
        with pytest.raises(ValueError, match="decode-only"):
            tree.layers[0].ffn.routed.forward(x.expand(2, -1), weights.expand(2, -1), indices.expand(2, -1))
        assert controller.calls == 1


def test_nonleader_still_reduces_but_returns_zero_to_original_global_sum():
    tree, _ = model(1, layers=1)
    calls = []
    with ExpertTPDispatch(tree, rank=1, world=8, tp_world=2, backend="torch", capture_graphs=False,
                          reduce_fp32=lambda value: calls.append(value.dtype)):
        result = tree.layers[0].ffn.routed.forward(torch.ones(1, 128, dtype=torch.bfloat16),
            torch.ones(1, 4), torch.tensor([[0, 1, 4, 5]]))
        assert calls == [torch.float32] and not result.count_nonzero()


def test_failed_byte_certificate_restores_previous_original_residency():
    tree, checkpoint = model(0, layers=2)
    originals = [block.ffn.routed for block in tree.layers]
    checkpoint.data["layers.1.ffn.experts.4.w1.scale"].fill_(255)
    manager = ExpertTPDispatch(tree, rank=0, world=8, tp_world=2, backend="torch", capture_graphs=False, reduce_fp32=lambda value: None)
    with pytest.raises(ValueError, match="certified"):
        manager.install()
    assert all(block.ffn.routed is store and store.allocated for block, store in zip(tree.layers, originals))


def test_close_can_drop_private_hybrid_residency_without_loading_again():
    tree, _ = model(0, layers=1)
    old = tree.layers[0].ffn.routed
    manager = ExpertTPDispatch(tree, rank=0, world=8, tp_world=2, backend="torch", capture_graphs=False,
                              reduce_fp32=lambda value: None, restore_residency=False).install()
    manager.close(); manager.close()
    assert tree.layers[0].ffn.routed is old and not old.allocated


def test_changed_source_owner_is_refused_before_releasing_any_bank():
    tree, _ = model(0, layers=2)
    originals = [block.ffn.routed for block in tree.layers]
    manager = ExpertTPDispatch(tree, rank=0, world=8, tp_world=2, backend="torch", capture_graphs=False, reduce_fp32=lambda value: None)
    foreign = object()
    tree.layers[1].ffn.routed = foreign
    with pytest.raises(RuntimeError, match="owner"):
        manager.install()
    assert originals[0].allocated and originals[1].allocated
    assert tree.layers[1].ffn.routed is foreign


def test_graph_replay_cannot_silently_cast_routing_metadata():
    tree, _ = model(0, layers=1)
    with ExpertTPDispatch(tree, rank=0, world=8, tp_world=2, backend="torch", capture_graphs=False, reduce_fp32=lambda value: None):
        store = tree.layers[0].ffn.routed
        with pytest.raises(ValueError, match="dtypes"):
            store.forward(torch.ones(1, 128, dtype=torch.bfloat16), torch.ones(1, 4, dtype=torch.float64), torch.tensor([[0, 1, 4, 5]]))


def test_sync_failure_restores_original_store_ownership_without_reload(monkeypatch):
    tree, _ = model(0, layers=1)
    old = tree.layers[0].ffn.routed
    manager = ExpertTPDispatch(tree, rank=0, world=8, tp_world=2, backend="torch", capture_graphs=False,
                              reduce_fp32=lambda value: None, restore_residency=False).install()
    def fail(device):
        raise RuntimeError("device sync failed")
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=fail), raising=False)
    manager.backend = "triton"  # cleanup-only fixture; not an NPU execution claim
    with pytest.raises(RuntimeError, match="sync failed"):
        manager.close()
    assert tree.layers[0].ffn.routed is old
    assert manager.closed and not manager.installed and manager.workspace is None


def test_unused_owner_close_releases_workspace_without_releasing_original_store():
    tree, _ = model(0, layers=1)
    original = tree.layers[0].ffn.routed
    manager = ExpertTPDispatch(tree, rank=0, world=8, tp_world=2, backend="torch", capture_graphs=False, reduce_fp32=lambda value: None)
    manager.close()
    assert manager.closed and manager.workspace is None and original.allocated


def test_owned_tp_group_is_destroyed_once_after_replay_storage_retires(monkeypatch):
    tree, _ = model(0, layers=1)
    manager = ExpertTPDispatch(tree, rank=0, world=8, tp_world=2, backend="torch", capture_graphs=False,
                              restore_residency=False, reduce_fp32=lambda value: None).install()
    sentinel = object()
    calls = []
    manager.tp_group = sentinel
    monkeypatch.setattr("relicllm.models.deepseek_v4_1.ascend_expert_tp.dist.destroy_process_group", lambda group: calls.append(group))
    manager.close(); manager.close()
    assert calls == [sentinel] and manager.tp_group is None


def test_existing_expert_body_graph_owner_is_not_implicitly_released(monkeypatch):
    from relicllm.models.deepseek_v4_1 import ascend_packed_experts
    tree, _ = model(0, layers=1)
    def foreign(*args, **kwargs):
        return None
    foreign._relic_routed_expert_graph_owner = object()
    monkeypatch.setattr(ascend_packed_experts, "packed_expert_forward", foreign)
    with pytest.raises(RuntimeError, match="expert-body graphs"):
        ExpertTPDispatch(tree, rank=0, world=8, tp_world=2, backend="torch", capture_graphs=False)
    assert tree.layers[0].ffn.routed.allocated


@pytest.mark.parametrize("width", [True, False, 0, 3, 16])
def test_invalid_tp_width_does_not_change_original_residency(width):
    tree, _ = model(0, layers=1)
    with pytest.raises(ValueError, match="geometry"):
        ExpertTPDispatch(tree, rank=0, world=8, tp_world=width, backend="torch", capture_graphs=False)
    assert tree.layers[0].ffn.routed.allocated


def test_sharding_preserves_low_nibble_order_and_original_k32_columns():
    codes = torch.arange(128 * 128).reshape(128, 128).to(torch.uint8)
    scales = torch.arange(128 * 8).reshape(128, 8).to(torch.uint8)
    left = shard_original_projection(codes, scales, projection="w2", tp_rank=0, tp_world=2, full_inter_dim=256)
    right = shard_original_projection(codes, scales, projection="w2", tp_rank=1, tp_world=2, full_inter_dim=256)
    assert torch.equal(torch.cat([left[0], right[0]], -1), codes)
    assert torch.equal(torch.cat([left[1], right[1]], -1), scales)
    with pytest.raises(ValueError, match="K32"):
        shard_original_projection(codes, scales, projection="w2", tp_rank=0, tp_world=2, full_inter_dim=288)
