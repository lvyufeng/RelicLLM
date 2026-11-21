"""Compact TP admission and CPU copy oracles; no native consumer executes here."""
from types import SimpleNamespace

import pytest
import torch

from relicllm.models.deepseek_v4_1 import ascend_expert_tp as tp


class MetadataStore:
    def __init__(self, layer_id=0, *, dim=128, inter_dim=256, device="npu:0"):
        self.layer_id, self.dim, self.inter_dim = layer_id, dim, inter_dim
        # The NPU label is metadata only: do not construct or allocate an NPU tensor.
        self.device = SimpleNamespace(type="npu", index=0) if device == "npu:0" else torch.device(device)
        self.rank, self.world, self.n_experts, self.topk = 0, 8, 8, 4
        self.dtype, self.swiglu_limit = torch.bfloat16, 10.0
        self._arenas = object()
        self.releases = 0
        self.checkpoint = None

    def release(self):
        self.releases += 1
        self._arenas = None

    def preload(self):
        self._arenas = object()

    def _key(self, expert, name):
        return f"layers.{self.layer_id}.ffn.experts.{expert}.{name}.weight"

    def _scale_key(self, expert, name):
        return self._key(expert, name).removesuffix("weight") + "scale"


class Consumer:
    def __init__(self, *, compact=True, group_k=4, output_rows=32, weight_format=None):
        self.cube_packed, self.group_k, self.output_rows = compact, group_k, output_rows
        self.block_dim = 24
        self.weight_format = (tp.COMPACT_BANK_FORMAT if compact else tp.ORIGINAL_BANK_FORMAT) if weight_format is None else weight_format

    def __call__(self, *args, **kwargs):
        pytest.fail("a metadata/copy oracle must not invoke a native consumer")


def forbidden(*args, **kwargs):
    pytest.fail("admission and CPU copy oracles must not invoke native/device work")


@pytest.fixture
def admission(monkeypatch):
    monkeypatch.setattr(tp, "AscendPackedRoutedExperts", MetadataStore)
    monkeypatch.setattr(tp.dist, "is_available", lambda: True)
    monkeypatch.setattr(tp.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(tp.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(tp.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(tp.dist, "new_group", forbidden)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=forbidden, mem_get_info=forbidden), raising=False)
    workspaces = []

    def workspace(dim, inter_dim, topk, **kwargs):
        result = SimpleNamespace(dim=dim, inter_dim=inter_dim, topk=topk, nbytes=16, **kwargs)
        workspaces.append(result)
        return result

    monkeypatch.setattr(tp, "make_tile_moe_workspace", workspace)

    def make(*, consumer=None, bank_format=tp.COMPACT_BANK_FORMAT, backend="triton", dim=128, inter_dim=256, device="npu:0"):
        original = MetadataStore(dim=dim, inter_dim=inter_dim, device=device)
        model = SimpleNamespace(layers=[SimpleNamespace(ffn=SimpleNamespace(routed=original))])
        manager = tp.ExpertTPDispatch(model, consumer=consumer, bank_format=bank_format, backend=backend,
                                      rank=0, world=8, tp_world=2, capture_graphs=False)
        return manager, original

    return make, workspaces


def test_compact_constructor_accepts_only_explicit_matching_consumer(admission):
    make, workspaces = admission
    manager, original = make(consumer=Consumer())
    assert manager.bank_format == tp.COMPACT_BANK_FORMAT
    assert workspaces[0].weight_rows == 32 and workspaces[0].group_k == 4
    assert original.releases == 0 and original._arenas is not None
    assert not manager.installed and not manager.replacements
    report = manager.describe()
    assert report["source_checkpoint_files_unchanged"]
    assert report["resident_encoding_changed"]
    assert report["packed_weight_and_scale_resident_byte_count_unchanged"]


@pytest.mark.parametrize("consumer, bank_format, backend", [
    (None, tp.COMPACT_BANK_FORMAT, "triton"),
    (Consumer(compact=False), tp.COMPACT_BANK_FORMAT, "triton"),
    (Consumer(group_k=2), tp.COMPACT_BANK_FORMAT, "triton"),
    (Consumer(group_k=4.0), tp.COMPACT_BANK_FORMAT, "triton"),
    (Consumer(group_k=True), tp.COMPACT_BANK_FORMAT, "triton"),
    (Consumer(output_rows=16), tp.COMPACT_BANK_FORMAT, "triton"),
    (Consumer(output_rows=32.0), tp.COMPACT_BANK_FORMAT, "triton"),
    (Consumer(output_rows=True), tp.COMPACT_BANK_FORMAT, "triton"),
    (Consumer(weight_format="unknown-v2"), tp.COMPACT_BANK_FORMAT, "triton"),
    (Consumer(), tp.COMPACT_BANK_FORMAT, "torch"),
    (Consumer(), tp.ORIGINAL_BANK_FORMAT, "triton"),
    (Consumer(compact=False, weight_format=tp.COMPACT_BANK_FORMAT), tp.ORIGINAL_BANK_FORMAT, "triton"),
    (Consumer(), "unknown-v2", "triton"),
])
def test_incompatible_formats_are_refused_before_workspace_or_ownership(admission, consumer, bank_format, backend):
    make, workspaces = admission
    with pytest.raises(ValueError, match="format|consumer"):
        make(consumer=consumer, bank_format=bank_format, backend=backend)
    assert not workspaces


@pytest.mark.parametrize("dim, inter_dim, device", [(160, 256, "npu:0"), (128, 320, "npu:0"), (128, 256, "cpu")])
def test_compact_geometry_is_refused_before_workspace_or_groups(admission, dim, inter_dim, device):
    make, workspaces = admission
    with pytest.raises(ValueError, match="N32/K128|NPU residency"):
        make(consumer=Consumer(), dim=dim, inter_dim=inter_dim, device=device)
    assert not workspaces


def test_default_original_format_still_accepts_legacy_consumer(admission):
    make, _ = admission
    legacy = SimpleNamespace(group_k=4, output_rows=32, block_dim=24)
    manager, original = make(consumer=legacy, bank_format=tp.ORIGINAL_BANK_FORMAT)
    assert manager.bank_format == tp.ORIGINAL_BANK_FORMAT
    assert not manager.describe()["resident_encoding_changed"]
    assert original.releases == 0


@pytest.mark.parametrize("field, value", [("weight_format", "unknown-v2"), ("cube_packed", False), ("group_k", 2), ("output_rows", 16)])
def test_install_rechecks_consumer_before_any_group_or_bank_release(admission, field, value):
    make, _ = admission
    consumer = Consumer()
    manager, original = make(consumer=consumer)
    setattr(consumer, field, value)
    with pytest.raises(ValueError, match="consumer"):
        manager.install()
    assert original.releases == 0 and original._arenas is not None
    assert not manager.replacements and not manager.installed


def test_install_rechecks_compact_geometry_before_any_bank_release(admission):
    make, _ = admission
    manager, original = make(consumer=Consumer())
    original.inter_dim = 320
    with pytest.raises(ValueError, match="geometry"):
        manager.install()
    assert original.releases == 0 and original._arenas is not None


class Checkpoint:
    def __init__(self, store):
        self.data = {}
        for expert in range(store.n_experts):
            for name in ("w1", "w3", "w2"):
                n, k = (store.dim, store.inter_dim) if name == "w2" else (store.inter_dim, store.dim)
                self.data[store._key(expert, name)] = (torch.arange(n * k // 2).reshape(n, k // 2) + expert).to(torch.uint8)
                self.data[store._scale_key(expert, name)] = torch.full((n, k // 32), 120, dtype=torch.uint8)

    def packed(self, key):
        return self.data[key]

    def is_packed_fp4(self, key):
        return key.endswith("weight")

    def block_size(self, key):
        return 32


def test_compact_bank_cpu_copy_oracle_preserves_bytes_shape_and_source(admission, monkeypatch):
    from relic_core.kernels import cube_packed_format as compact

    make, _ = admission
    manager, original = make(consumer=Consumer())
    original.checkpoint = Checkpoint(original)
    before = {key: value.clone() for key, value in original.checkpoint.data.items()}
    real_convert = compact.convert_fp4_cube
    calls = []

    def convert(code, scale):
        calls.append((tuple(code.shape), tuple(scale.shape)))
        return real_convert(code, scale)

    monkeypatch.setattr(compact, "convert_fp4_cube", convert)
    monkeypatch.setattr(tp, "pack_code_tiles", forbidden)
    monkeypatch.setattr(tp, "pack_scale_tiles", forbidden)
    # Exercise only byte loading on CPU; no installation, forward, or NPU call.
    manager.backend, manager.device = "torch", torch.device("cpu")
    bank = manager._make_bank(original, original.inter_dim)
    assert len(calls) == 6  # Two EP-owned experts, three bounded projections each.
    assert bank.bank_format == tp.COMPACT_BANK_FORMAT and bank.cache_stats["bank_format"] == bank.bank_format
    assert bank.resident_bytes == 2 * tp.packed_expert_bytes(original.dim, original.inter_dim // 2)
    for slot, expert in enumerate(range(0, original.n_experts, 4)):
        for name, n, k in (("w1", 128, 128), ("w3", 128, 128), ("w2", 128, 128)):
            codes, scales = bank.banks[name]
            assert codes.shape == (2, n // 16, k // 32, 16, 16)
            assert scales.shape == (2, n // 16, k // 32, 16)
            geometry = compact.cube_geometry(n, k)
            payload = compact.CubePackedFP4(codes[slot].reshape(geometry.codes_shape),
                                             scales[slot].reshape(geometry.scales_shape), geometry)
            recovered = compact.recover_fp4_cube(payload)
            expected = tp.shard_original_projection(before[original._key(expert, name)],
                before[original._scale_key(expert, name)], projection=name, tp_rank=0, tp_world=2,
                full_inter_dim=original.inter_dim)
            assert all(torch.equal(actual, source) for actual, source in zip(recovered, expected))
    assert all(torch.equal(value, before[key]) for key, value in original.checkpoint.data.items())
    assert original.releases == 0


def test_original_scale_certificate_precedes_compact_conversion(admission, monkeypatch):
    from relic_core.kernels import cube_packed_format as compact

    make, _ = admission
    manager, original = make(consumer=Consumer())
    original.checkpoint = Checkpoint(original)
    original.checkpoint.data[original._scale_key(0, "w1")].fill_(255)
    manager.backend, manager.device = "torch", torch.device("cpu")
    monkeypatch.setattr(compact, "convert_fp4_cube", forbidden)
    with pytest.raises(ValueError, match="certified"):
        manager._make_bank(original, original.inter_dim)
    assert original.releases == 0
