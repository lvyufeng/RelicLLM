"""Metadata and library guards for the explicit AscendC tile experiment."""
import hashlib
from types import SimpleNamespace

import pytest
import torch

from relic_core.kernels import ascendc_online_tile_cube as native
from relic_core.kernels.online_tile_cube import pack_expert_tile_bank


class FakeLaunch:
    def __call__(self, *args):
        pytest.fail("native launch must not happen during CPU metadata tests")


class FakeQuery:
    def __call__(self):
        return 1


@pytest.fixture
def consumer(tmp_path, monkeypatch):
    path = tmp_path / "explicit-native-kernel.so"
    path.write_bytes(b"not executable; CDLL is mocked for metadata tests")
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(native.ctypes, "CDLL", lambda _: SimpleNamespace(relic_online_tile_cube_launch=FakeLaunch(), relic_online_tile_cube_k_blocks_per_tile=FakeQuery()))
    return native.AscendCTileCubeConsumer(path, sha256=checksum, block_dim=4)


def tensors():
    b, s = pack_expert_tile_bank(torch.zeros((2, 64, 32), dtype=torch.uint8),
                               torch.full((2, 64, 2), 127, dtype=torch.uint8))
    return torch.ones((1, 64), dtype=torch.bfloat16), b, s, torch.tensor([0, -1], dtype=torch.int32)


def test_reject_unverified_library_before_loading(tmp_path, monkeypatch):
    path = tmp_path / "kernel.so"
    path.write_bytes(b"private build")
    monkeypatch.setattr(native.ctypes, "CDLL", lambda _: pytest.fail("unverified library was loaded"))
    with pytest.raises(ValueError, match="checksum"):
        native.AscendCTileCubeConsumer(path, sha256="0" * 64)


@pytest.mark.parametrize("block_dim", [0, 25, True, 1.5])
def test_reject_invalid_native_grid(tmp_path, block_dim):
    path = tmp_path / "kernel.so"
    path.write_bytes(b"private build")
    with pytest.raises(ValueError, match="block_dim"):
        native.AscendCTileCubeConsumer(path, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), block_dim=block_dim)


def test_no_cpu_native_fallback(consumer):
    with pytest.raises(ValueError, match="NPU"):
        consumer(*tensors(), world=1, certified_bf16_operands=True)


@pytest.mark.parametrize("n_output", [16, 32, 48])
def test_partial_extent_is_not_allowed_to_write_outside_output(consumer, n_output):
    with pytest.raises(ValueError, match="physical output extent"):
        consumer(*tensors(), world=1, n_output=n_output, certified_bf16_operands=True)


def test_ring_residency_is_small_but_not_zero(consumer):
    report = consumer.describe()
    assert report["global_expanded_weight_scratch_bytes"] == 8192
    assert report["whole_matrix_expanded_weight_scratch_bytes"] == 0
    assert report["native_full_K_accumulation_differs_from_exact_default"]


def test_native_library_abi_is_explicit(consumer):
    assert consumer._launch.restype is None
    assert len(consumer._launch.argtypes) == 15


class ConfiguredQuery:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def explicit_variant(tmp_path, monkeypatch, *, group, preload=None, bits=None):
    path = tmp_path / "metadata-only-variant.so"
    path.write_bytes(b"metadata-only; no native library is executed")
    symbols = {"relic_online_tile_cube_launch": FakeLaunch(),
               "relic_online_tile_cube_k_blocks_per_tile": ConfiguredQuery(group)}
    if bits is not None:
        symbols["relic_online_tile_cube_direct_bf16_bits"] = ConfiguredQuery(bits)
    if preload is not None:
        symbols["relic_online_tile_cube_preload_activation"] = ConfiguredQuery(preload)
    monkeypatch.setattr(native.ctypes, "CDLL", lambda _: SimpleNamespace(**symbols))
    return native.AscendCTileCubeConsumer(path, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.mark.parametrize("group", [1, 2, 4, 8])
def test_configured_ring_uses_exact_query_geometry(tmp_path, monkeypatch, group):
    operator = explicit_variant(tmp_path, monkeypatch, group=group)
    assert operator.ring_elements == 512 * group
    assert operator.describe()["global_expanded_weight_scratch_bytes"] == 24 * 512 * group * 4
    assert operator.describe()["activation_L1_preload_requested"] is False


@pytest.mark.parametrize("preload", [0, 1])
def test_optional_l1_feature_query_preserves_old_abi(tmp_path, monkeypatch, preload):
    operator = explicit_variant(tmp_path, monkeypatch, group=8, preload=preload)
    assert operator.preload_activation is bool(preload)
    assert len(operator._launch.argtypes) == 15
    assert operator.describe()["GM_ring_B1_and_MTE1_reuse_fences_preserved"]


def test_refuse_unknown_l1_policy_or_unbounded_k_geometry(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="boolean"):
        explicit_variant(tmp_path, monkeypatch, group=8, preload=2)
    with pytest.raises(ValueError, match="unsupported"):
        explicit_variant(tmp_path, monkeypatch, group=16)


@pytest.mark.parametrize("bits", [None, 0, 1])
def test_direct_bit_decode_optional_query_preserves_existing_library_abi(tmp_path, monkeypatch, bits):
    operator = explicit_variant(tmp_path, monkeypatch, group=4, bits=bits)
    assert operator.direct_bf16_bits is (bits == 1)
    assert operator.describe()["direct_original_BF16_bit_decoder"] is (bits == 1)
    assert len(operator._launch.argtypes) == 15


def test_unknown_direct_bit_decoder_policy_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="boolean"):
        explicit_variant(tmp_path, monkeypatch, group=4, bits=2)


def test_activation_l1_admission_matches_real_geometry():
    up = native.activation_l1_layout(5120, 6, group_k=8, shared=True)
    down = native.activation_l1_layout(2304, 6, group_k=8, shared=False)
    assert up["total_bytes"] == 180224 and up["fits"]
    assert down["total_bytes"] == 458752 and down["fits"]
    too_large = native.activation_l1_layout(2304, 8, group_k=8, shared=False)
    assert too_large["total_bytes"] == 606208 and not too_large["fits"]


@pytest.mark.parametrize("kwargs", [dict(k=True, slots=6), dict(k=31, slots=6),
                                    dict(k=128, slots=6, group_k=8), dict(k=5120, slots=9),
                                    dict(k=5120, slots=6, shared=1)])
def test_l1_admission_refuses_ambiguous_shapes(kwargs):
    with pytest.raises(ValueError):
        native.activation_l1_layout(**kwargs)


@pytest.mark.parametrize("world", [2**32, 2**32 + 1])
def test_positive_integer_world_cannot_wrap_in_the_native_abi(consumer, world):
    with pytest.raises(ValueError, match="uint32 C ABI"):
        consumer(*tensors(), world=world, certified_bf16_operands=True)
