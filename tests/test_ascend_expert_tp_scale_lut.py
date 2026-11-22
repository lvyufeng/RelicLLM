"""Load-time LUT certification and compact-owner lifetime tests using CPU/metadata spies."""
import inspect
from types import SimpleNamespace

import pytest
import torch

from relicllm.models.deepseek_v4_1 import ascend_expert_tp as tp


class Store:
    def __init__(self, layer_id=0):
        self.layer_id, self.dim, self.inter_dim, self.topk, self.n_experts = layer_id, 128, 256, 4, 8
        self.device = SimpleNamespace(type="npu", index=0)
        self.rank, self.world, self.dtype, self.swiglu_limit = 0, 8, torch.bfloat16, 10.0
        self._arenas, self.releases, self.preloads = object(), 0, 0
        self.checkpoint = None

    def release(self):
        self.releases += 1
        self._arenas = None

    def preload(self):
        self.preloads += 1
        self._arenas = object()

    def _key(self, expert, name):
        return f"layer.{self.layer_id}.expert.{expert}.{name}.weight"

    def _scale_key(self, expert, name):
        return f"layer.{self.layer_id}.expert.{expert}.{name}.scale"


class Consumer:
    cube_packed, group_k, output_rows, block_dim = True, 4, 32, 24
    weight_format = tp.COMPACT_BANK_FORMAT

    def __init__(self, *, lut=True):
        if lut:
            self.scale_min, self.scale_max = 117, 127
        self.prepared = []

    def prepare_device_metadata(self, device):
        self.prepared.append(device)

    def __call__(self, *args, **kwargs):
        pytest.fail("runtime metadata tests must not call a native consumer")


@pytest.fixture
def admission(monkeypatch):
    monkeypatch.setattr(tp, "AscendPackedRoutedExperts", Store)
    monkeypatch.setattr(tp.dist, "is_available", lambda: True)
    monkeypatch.setattr(tp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tp.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(tp.dist, "get_world_size", lambda: 8)

    def forbidden(*args, **kwargs):
        pytest.fail("admission must stop before any real device or collective work")

    monkeypatch.setattr(tp.dist, "new_group", forbidden)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=forbidden, mem_get_info=forbidden), raising=False)

    def workspace(dim, inter_dim, topk, **kwargs):
        return SimpleNamespace(dim=dim, inter_dim=inter_dim, topk=topk, nbytes=16, **kwargs)

    monkeypatch.setattr(tp, "make_tile_moe_workspace", workspace)

    def make(consumer=None, *, layers=1, bank_format=tp.COMPACT_BANK_FORMAT):
        originals = [Store(index) for index in range(layers)]
        model = SimpleNamespace(layers=[SimpleNamespace(ffn=SimpleNamespace(routed=store)) for store in originals])
        manager = tp.ExpertTPDispatch(model, consumer=Consumer() if consumer is None else consumer,
            bank_format=bank_format, capture_graphs=False, rank=0, world=8)
        return manager, originals

    return make


def test_original_constructor_api_and_default_remain_unchanged():
    parameters = inspect.signature(tp.ExpertTPDispatch.__init__).parameters
    assert set(parameters) == {"self", "model", "consumer", "tp_world", "backend", "rank", "world",
        "reduce_fp32", "capture_graphs", "restore_residency", "reserve_bytes", "max_workspace_bytes", "bank_format"}
    assert parameters["bank_format"].default == tp.ORIGINAL_BANK_FORMAT
    assert parameters["consumer"].default is None
    assert parameters["capture_graphs"].default is True


def test_ordinary_n32_consumer_has_no_range_certificate_or_new_keyword(admission, monkeypatch):
    manager, _ = admission(Consumer(lut=False))
    assert tp.ExpertTPDispatch._consumer_scale_lut_range(manager.consumer) is None
    assert not manager.scale_lut_certified and not hasattr(manager.workspace, "compact_n64")
    seen = []
    monkeypatch.setattr(tp, "tp_moe_down_partials", lambda *args, **kwargs: seen.append(kwargs) or object())
    monkeypatch.setattr(tp, "finish_tp_moe", lambda *args, **kwargs: None)
    manager._execute(SimpleNamespace(banks={}, n_experts=8, swiglu_limit=10.0), object(), object(), object())
    assert "certified_scale_lut_operands" not in seen[0]


@pytest.mark.parametrize("minimum,maximum", [(None, 127), (117, None), (True, 127), (117.0, 127), (116, 127), (117, 128)])
def test_n32_lut_advertised_bounds_are_checked_before_workspace(admission, minimum, maximum):
    consumer = Consumer()
    consumer.scale_min, consumer.scale_max = minimum, maximum
    with pytest.raises(ValueError, match="117..127"):
        admission(consumer)


@pytest.mark.parametrize("scale", [116, 128])
def test_n32_actual_source_range_precedes_converter_and_copy(admission, monkeypatch, scale):
    from relic_core.kernels import cube_packed_format as cube

    manager, originals = admission()
    original = originals[0]
    code = torch.zeros((256, 64), dtype=torch.uint8)
    scales = torch.full((256, 4), scale, dtype=torch.uint8)
    original.checkpoint = SimpleNamespace(is_packed_fp4=lambda key: True, block_size=lambda key: 32,
        packed=lambda key: scales if key.endswith("scale") else code)
    manager.backend, manager.device = "torch", torch.device("cpu")
    monkeypatch.setattr(cube, "convert_fp4_cube", lambda *args: pytest.fail("invalid scales must not be converted"))
    with pytest.raises(ValueError, match="117..127"):
        manager._make_bank(original, original.inter_dim)
    assert not manager.scale_lut_certified and original.releases == 0


@pytest.mark.parametrize("fail_second_layer", [False, True])
def test_install_certifies_only_after_every_owned_bank_completes(admission, monkeypatch, fail_second_layer):
    from relicllm.models.deepseek_v4_1 import ascend_graphs

    manager, originals = admission(layers=2)
    events = []
    monkeypatch.setattr(torch.npu, "synchronize", lambda device: events.append("fence"))
    monkeypatch.setattr(tp.dist, "new_group", lambda *args, **kwargs: object())
    monkeypatch.setattr(tp.dist, "destroy_process_group", lambda group: None)
    monkeypatch.setattr(ascend_graphs, "npu_graph_api", lambda: SimpleNamespace(
        graph_pool_handle=lambda: object(), memory_allocated=lambda: 0))

    def load(store, full_mid):
        assert not manager.scale_lut_certified
        assert manager.consumer.prepared  # Tables finish before any bank replacement.
        events.append(store.layer_id)
        if store.layer_id == 1 and fail_second_layer:
            raise RuntimeError("incomplete bank load")
        return SimpleNamespace(banks={}, resident_bytes=0)

    monkeypatch.setattr(manager, "_make_bank", load)
    if fail_second_layer:
        with pytest.raises(RuntimeError, match="incomplete bank load"):
            manager.install()
        assert not manager.scale_lut_certified and not manager.installed and manager.closed
        assert all(ffn.routed is store for ffn, store, _ in manager.entries)
        assert all(store.preloads == 1 for store in originals)
    else:
        manager.install()
        assert manager.scale_lut_certified and manager.installed
        assert events == ["fence", 0, 1]
        seen = []
        monkeypatch.setattr(tp, "tp_moe_down_partials", lambda *args, **kwargs: seen.append(kwargs) or object())
        monkeypatch.setattr(tp, "finish_tp_moe", lambda *args, **kwargs: None)
        monkeypatch.setattr(manager, "reduce_fp32", lambda value: None)
        manager._execute(SimpleNamespace(banks={}, n_experts=8, swiglu_limit=10.0), object(), object(), object())
        assert seen[0]["certified_scale_lut_operands"] is True
        manager.close(restore_residency=False)


@pytest.mark.parametrize("rows", [32, 64])
def test_compact_failed_fence_retains_resources_until_successful_close_retry(admission, monkeypatch, rows):
    consumer = Consumer()
    consumer.output_rows = rows
    consumer.weight_format = tp.COMPACT_BANK_FORMAT if rows == 32 else tp.COMPACT_BANK_FORMAT64
    manager, originals = admission(consumer, bank_format=consumer.weight_format)
    original, ffn, workspace = originals[0], manager.entries[0][0], manager.workspace
    graph_api, graph_pool, tp_group = object(), object(), object()
    events = []
    content = {"w1": (object(), object()), "w3": (object(), object()), "w2": (object(), object())}

    class Banks(dict):
        def clear(self):
            assert not manager.graphs and manager.workspace is workspace
            events.append("banks-retired")
            super().clear()

    class Graphs(dict):
        def clear(self):
            assert manager.workspace is workspace and shard.banks == content
            events.append("graphs-retired")
            super().clear()

    shard = SimpleNamespace(banks=Banks(content))
    ffn.routed = shard
    manager.replacements.append((ffn, original, shard))
    manager.graphs = Graphs({0: object()})
    manager.graph_api, manager.graph_pool, manager.tp_group = graph_api, graph_pool, tp_group
    manager.installed = True

    def fail(device):
        events.append("failed-fence")
        raise RuntimeError("fence failed")

    monkeypatch.setattr(torch.npu, "synchronize", fail)
    monkeypatch.setattr(tp.dist, "destroy_process_group", lambda group: events.append("group-retired"))
    with pytest.raises(RuntimeError, match="fence failed"):
        manager.close(restore_residency=False)
    assert events == ["failed-fence"]
    assert manager.installed and not manager.closed and ffn.routed is shard
    assert manager.workspace is workspace and manager.graphs and shard.banks == content
    assert manager.graph_api is graph_api and manager.graph_pool is graph_pool and manager.tp_group is tp_group
    assert manager.replacements == [(ffn, original, shard)] and original.releases == 0

    def succeed(device):
        assert manager.workspace is workspace and manager.graphs and shard.banks == content
        events.append("successful-fence")

    monkeypatch.setattr(torch.npu, "synchronize", succeed)
    manager.close(restore_residency=False)
    assert events == ["failed-fence", "successful-fence", "graphs-retired", "banks-retired", "group-retired"]
    assert ffn.routed is original and not manager.graphs and not shard.banks and not manager.replacements
    assert manager.workspace is manager.graph_api is manager.graph_pool is manager.tp_group is None
    assert not manager.installed and manager.closed


def test_original_noncompact_failed_fence_keeps_the_legacy_cleanup_policy(admission, monkeypatch):
    consumer = Consumer(lut=False)
    consumer.weight_format, consumer.cube_packed, consumer.output_rows = tp.ORIGINAL_BANK_FORMAT, False, 16
    manager, originals = admission(consumer, bank_format=tp.ORIGINAL_BANK_FORMAT)
    original, ffn = originals[0], manager.entries[0][0]
    shard = SimpleNamespace(banks={"w1": (object(), object())})
    ffn.routed = shard
    manager.replacements.append((ffn, original, shard))
    manager.graphs[0] = object()
    manager.installed = True

    def fail(device):
        raise RuntimeError("legacy fence failed")

    monkeypatch.setattr(torch.npu, "synchronize", fail)
    with pytest.raises(RuntimeError, match="legacy fence failed"):
        manager.close(restore_residency=False)
    assert ffn.routed is original and not shard.banks and not manager.graphs
    assert manager.workspace is None and not manager.installed and manager.closed
