"""N64 K128 admission and CPU byte-copy oracles; no native kernels execute."""
from types import SimpleNamespace

import pytest
import torch

from relicllm.models.deepseek_v4_1 import ascend_expert_tp as tp


class Store:
    def __init__(self, dim=128, inter_dim=256, device=None):
        self.layer_id, self.dim, self.inter_dim, self.topk, self.n_experts = 0, dim, inter_dim, 4, 8
        self.device = SimpleNamespace(type="npu", index=0) if device is None else device
        self.rank, self.world, self.dtype, self.swiglu_limit = 0, 8, torch.bfloat16, 10.0
        self._arenas, self.releases = object(), 0
        self.checkpoint = None

    def release(self):
        self.releases += 1
        self._arenas = None

    def preload(self):
        self._arenas = object()

    def _key(self, expert, name):
        return f"expert.{expert}.{name}.weight"

    def _scale_key(self, expert, name):
        return f"expert.{expert}.{name}.scale"


class Consumer:
    cube_packed, group_k, output_rows, block_dim = True, 4, 64, 24
    weight_format = tp.COMPACT_BANK_FORMAT64
    scale_min, scale_max = 117, 127

    def __call__(self, *args, **kwargs):
        pytest.fail("metadata/copy tests must not invoke a native consumer")

    def prepare_device_metadata(self, device):
        pytest.fail("admission must not prepare native metadata")


class Checkpoint:
    def __init__(self, store, scale=120):
        self.data = {}
        for expert in range(store.n_experts):
            for name in ("w1", "w3", "w2"):
                n, k = (store.dim, store.inter_dim) if name == "w2" else (store.inter_dim, store.dim)
                self.data[store._key(expert, name)] = (torch.arange(n * k // 2).reshape(n, k // 2) + expert).to(torch.uint8)
                self.data[store._scale_key(expert, name)] = torch.full((n, k // 32), scale, dtype=torch.uint8)

    def is_packed_fp4(self, key):
        return key.endswith("weight")

    def block_size(self, key):
        return 32

    def packed(self, key):
        return self.data[key]


@pytest.fixture
def admission(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("admission must stop before native work")

    monkeypatch.setattr(tp, "AscendPackedRoutedExperts", Store)
    monkeypatch.setattr(tp.dist, "is_available", lambda: True)
    monkeypatch.setattr(tp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tp.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(tp.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(tp.dist, "new_group", forbidden)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=forbidden, mem_get_info=forbidden), raising=False)
    workspaces = []

    def workspace(dim, inter_dim, topk, **kwargs):
        value = SimpleNamespace(dim=dim, inter_dim=inter_dim, topk=topk, nbytes=16, **kwargs)
        workspaces.append(value)
        return value

    monkeypatch.setattr(tp, "make_tile_moe_workspace", workspace)

    def make(consumer=None, *, bank_format=tp.COMPACT_BANK_FORMAT64, dim=128, inter_dim=256, device=None):
        original = Store(dim, inter_dim, device)
        model = SimpleNamespace(layers=[SimpleNamespace(ffn=SimpleNamespace(routed=original))])
        manager = tp.ExpertTPDispatch(model, consumer=Consumer() if consumer is None else consumer,
            bank_format=bank_format, rank=0, world=8, capture_graphs=False)
        return manager, original

    return make, workspaces


def test_n64_is_explicit_and_enables_only_its_workspace_flag(admission):
    make, workspaces = admission
    manager, original = make()
    assert workspaces[0].group_k == 4 and workspaces[0].weight_rows == 64
    assert workspaces[0].compact_n64 is True and original.releases == 0
    assert manager.bank_format == "relic.fp4.cube-n64-k128.v1"
    assert manager.describe()["resident_encoding_changed"] and not manager.scale_lut_certified


@pytest.mark.parametrize("field,value", [("weight_format", tp.COMPACT_BANK_FORMAT), ("output_rows", 32),
    ("output_rows", 64.0), ("group_k", True), ("group_k", 8), ("cube_packed", False),
    ("scale_min", 116), ("scale_min", 117.0), ("scale_max", 128), ("scale_max", True)])
def test_n64_mismatch_is_refused_before_workspace_or_ownership(admission, field, value):
    make, workspaces = admission
    consumer = Consumer()
    setattr(consumer, field, value)
    with pytest.raises(ValueError):
        make(consumer)
    assert not workspaces


def test_n64_consumer_does_not_replace_the_original_default(admission):
    make, workspaces = admission
    with pytest.raises(ValueError, match="original-format"):
        make(bank_format=tp.ORIGINAL_BANK_FORMAT)
    assert not workspaces


@pytest.mark.parametrize("dim,inter_dim", [(160, 256), (128, 320), (192, 256)])
def test_n64_both_projections_must_have_complete_n64_k128_geometry(admission, dim, inter_dim):
    make, workspaces = admission
    with pytest.raises(ValueError, match="N64/K128"):
        make(dim=dim, inter_dim=inter_dim)
    assert not workspaces


def test_n64_refuses_cpu_residency_at_admission(admission):
    make, workspaces = admission
    with pytest.raises(ValueError, match="NPU"):
        make(device=torch.device("cpu"))
    assert not workspaces


def test_install_rechecks_the_lut_header_before_releasing_banks(admission):
    make, _ = admission
    consumer = Consumer()
    manager, original = make(consumer)
    consumer.scale_max = 128
    with pytest.raises(ValueError, match="117..127"):
        manager.install()
    assert original.releases == 0


@pytest.mark.parametrize("scale", [117, 120, 127])
def test_n64_inverse_preserves_source_bytes_and_exact_residency(admission, scale):
    from relic_core.kernels import cube_packed_format64 as cube

    make, _ = admission
    manager, original = make()
    original.checkpoint = Checkpoint(original, scale)
    before = {key: value.clone() for key, value in original.checkpoint.data.items()}
    manager.backend, manager.device = "torch", torch.device("cpu")
    bank = manager._make_bank(original, original.inter_dim)
    assert bank.resident_bytes == 2 * tp.packed_expert_bytes(128, 128)
    assert bank.bank_format == tp.COMPACT_BANK_FORMAT64
    for slot, expert in enumerate(range(0, 8, 4)):
        for name in ("w1", "w3", "w2"):
            codes, scales = bank.banks[name]
            geometry = cube.cube_geometry64(128, 128)
            payload = cube.CubePackedFP464(codes[slot].reshape(geometry.codes_shape),
                                          scales[slot].reshape(geometry.scales_shape), geometry)
            actual = cube.recover_fp4_cube64(payload)
            expected = tp.shard_original_projection(before[original._key(expert, name)],
                before[original._scale_key(expert, name)], projection=name, tp_rank=0, tp_world=2,
                full_inter_dim=256)
            assert all(torch.equal(value, source) for value, source in zip(actual, expected))
    assert all(torch.equal(value, before[key]) for key, value in original.checkpoint.data.items())
    assert original.releases == 0 and not manager.scale_lut_certified


@pytest.mark.parametrize("scale", [116, 128])
def test_actual_n64_lut_range_is_checked_before_conversion(admission, monkeypatch, scale):
    from relic_core.kernels import cube_packed_format64 as cube

    make, _ = admission
    manager, original = make()
    original.checkpoint = Checkpoint(original, scale)
    manager.backend, manager.device = "torch", torch.device("cpu")
    monkeypatch.setattr(cube, "convert_fp4_cube64", lambda *args: pytest.fail("invalid bytes must not be converted"))
    with pytest.raises(ValueError, match="117..127"):
        manager._make_bank(original, original.inter_dim)
    assert original.releases == 0 and not manager.scale_lut_certified
