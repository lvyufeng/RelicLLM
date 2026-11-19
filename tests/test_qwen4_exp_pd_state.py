"""CPU contracts for the prefill/decode state hand-off format.

These tests use the real cache classes with small synthetic geometry. They prove that every piece
of sequence state an exact resume reads survives a freeze/thaw bit-for-bit, that import writes into
the destination's existing allocations, and that a stale or mismatched bundle is refused before any
live tensor is touched. They do not prove cross-host token parity: that needs the real checkpoint on
two machines, which is the parent's GPU step.
"""
from __future__ import annotations

import copy
import json

import pytest
import torch

from relicllm.models.qwen4_exp.attention import (
    GatedDeltaNetCache,
    QSAAttentionCache,
    _QSAIndexerPool,
)
from relicllm.models.qwen4_exp.model import PLEState, Qwen4ExpCache
from relicllm.models.qwen4_exp import pd_state


# ---------------------------------------------------------------------------------------------
# Small synthetic geometry. The real model's shapes are covered by the parent's GPU run; these
# tests are about transport, not about a particular width.
# ---------------------------------------------------------------------------------------------


def config():
    return type("Config", (), {
        "linear_num_value_heads": 3, "linear_num_key_heads": 2,
        "linear_key_head_dim": 8, "linear_value_head_dim": 8,
        "linear_conv_kernel_dim": 4,
        "num_key_value_heads": 1, "head_dim": 8, "indexer_head_dim": 8,
    })()


class Indexer:
    """Stand-in for `QSAIndexer`: only the attributes the pool signature reads."""

    def __init__(self, ratio=4):
        self.compress_ratio = ratio


def backbone_cache(past: int, *, seed: int = 0, width: int = 64, ratio: int = 4) -> Qwen4ExpCache:
    """A cache with one GDN layer, one QSA layer, one empty slot and one PLE layer."""
    torch.manual_seed(seed)
    cfg = config()
    gdn = GatedDeltaNetCache(cfg, 1, torch.device("cpu"), torch.bfloat16)
    gdn.conv_state.normal_()
    gdn.recurrent_state.normal_()
    gdn.initialized = True
    gdn.length = past

    qsa = QSAAttentionCache(cfg, 1, width, torch.device("cpu"), torch.bfloat16)
    qsa.k.normal_()
    qsa.v.normal_()
    qsa.index_k.normal_()
    qsa.length = past
    qsa.host_position = past
    pool = _QSAIndexerPool(qsa.index_k, ratio)
    pool.keys.normal_()
    pool.completed = past // ratio
    pool.completed_tensor.fill_(past // ratio)
    pool.tensor_ready = True
    qsa.indexer_pool = pool

    cache = Qwen4ExpCache(
        layers=[gdn, qsa, None],
        ple_states={0: PLEState(torch.zeros(1, 6, 9, dtype=torch.bfloat16))},
        length=past,
    )
    cache.ple_states[0].conv_state.normal_()
    cache.ple_token_history = torch.tensor([[1, 2]], dtype=torch.long)
    cache.cos = torch.randn(3, 1, 4)
    cache.sin = torch.randn(3, 1, 4)
    return cache


def head_cache(past: int = 0, *, seed: int = 0, width: int = 64, ratio: int = 4) -> Qwen4ExpCache:
    torch.manual_seed(seed)
    layer = QSAAttentionCache(config(), 1, width, torch.device("cpu"), torch.bfloat16)
    layer.k.normal_()
    layer.v.normal_()
    layer.index_k.normal_()
    layer.length = past
    layer.cos = torch.randn(3, 1, 4)
    layer.sin = torch.randn(3, 1, 4)
    pool = _QSAIndexerPool(layer.index_k, ratio)
    pool.keys.normal_()
    pool.completed = past // ratio
    pool.completed_tensor.fill_(past // ratio)
    pool.tensor_ready = True
    layer.indexer_pool = pool
    cache = Qwen4ExpCache(layers=[layer], length=0)
    return cache


def model_stub(*, backbone_indexer=None, head_indexer=None, layers=3):
    entries = []
    for _ in range(layers):
        entries.append(type("Layer", (), {"attn": type("Attn", (), {"indexer": backbone_indexer})()})())
    head = type("Head", (), {
        "layer": type("HeadLayer", (), {"attn": type("HeadAttn", (), {"indexer": head_indexer})()})(),
    })()
    return type("Model", (), {"layers": entries, "mtp": head})()


def freeze(cache, head=None, **kwargs):
    options = dict(cache=cache, past=cache.length, next_token=1234)
    options.update(kwargs)
    if head is not None:
        options["head_cache"] = head
    return pd_state.freeze_state(**options)


# ---------------------------------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------------------------------


def test_freeze_captures_prefixes_and_skips_derived_tables():
    source = backbone_cache(past=12)
    bundle = freeze(source, head=head_cache(past=8), first_draft=77, first_conf=0.5,
                    chain_stack=torch.arange(6, dtype=torch.bfloat16).reshape(1, 1, 6),
                    prefill_s=1.25, tp_rank=2, tp_world_size=8, native_max_context=262144,
                    metadata={"weights_sha256": "abc"})
    names = set(bundle.tensor_names())
    # Prefixes, not arenas: 12 of 64 positions.
    assert bundle.tensors["backbone.layer.1.k"].shape == (1, 1, 12, 8)
    assert bundle.tensors["backbone.layer.1.index_k"].shape == (1, 12, 8)
    assert bundle.tensors["backbone.layer.1.indexer_keys"].shape == (1, 3, 8)
    # Derived or process-local state must never be in the payload.
    assert not {"backbone.cos", "backbone.sin", "backbone.layer.1.indexer_pool"}.intersection(names)
    assert "head.layer.0.k" in names and bundle.tensors["head.layer.0.k"].shape == (1, 1, 8, 8)
    assert bundle.manifest["past"] == 12
    assert bundle.manifest["cache_length"] == 12
    assert bundle.manifest["tp_rank"] == 2 and bundle.manifest["tp_world_size"] == 8
    assert bundle.manifest["mtp"]["first_draft"] == 77
    assert bundle.manifest["metadata"] == {"weights_sha256": "abc"}
    assert bundle.manifest["backbone"]["layers"][0]["initialized"] is True
    assert bundle.manifest["backbone"]["layers"][2]["kind"] == "none"
    assert all(t.is_contiguous() and t.device.type == "cpu" for t in bundle.tensors.values())
    # The bundle owns its bytes: mutating the live arena afterwards must not change it.
    before = bundle.tensors["backbone.layer.1.k"].clone()
    source.layers[1].k.fill_(99.0)
    assert torch.equal(bundle.tensors["backbone.layer.1.k"], before)


def test_thaw_restores_state_in_place_and_keeps_destination_addresses():
    source = backbone_cache(past=12, seed=1)
    source_head = head_cache(past=8, seed=2)
    bundle = freeze(source, head=source_head, first_draft=77, first_conf=0.5,
                    chain_stack=torch.arange(6, dtype=torch.bfloat16).reshape(1, 1, 6))

    destination = backbone_cache(past=0, seed=3)
    destination_head = head_cache(past=0, seed=4)
    pointers = {
        "conv": destination.layers[0].conv_state.data_ptr(),
        "recurrent": destination.layers[0].recurrent_state.data_ptr(),
        "k": destination.layers[1].k.data_ptr(),
        "index_k": destination.layers[1].index_k.data_ptr(),
        "pool": destination.layers[1].indexer_pool.keys.data_ptr(),
        "ple": destination.ple_states[0].conv_state.data_ptr(),
        "head_k": destination_head.layers[0].k.data_ptr(),
    }
    report = pd_state.thaw_state(
        bundle, cache=destination, head_cache=destination_head,
        model=model_stub(backbone_indexer=Indexer(), head_indexer=Indexer()),
        tp_rank=0, tp_world_size=1, native_max_context=262144,
    )
    assert report["past"] == 12 and report["next_token"] == 1234
    assert report["indexer_pool_rebound"] == 2 and report["indexer_pool_needs_recompute"] == 0
    assert report["chain_stack"] is not None
    assert torch.equal(report["chain_stack"], bundle.tensors["mtp.chain_stack"])

    # Exactly the imported prefix, bit for bit.
    assert torch.equal(destination.layers[1].k[:, :, :12], source.layers[1].k[:, :, :12])
    assert torch.equal(destination.layers[1].v[:, :, :12], source.layers[1].v[:, :, :12])
    assert torch.equal(destination.layers[1].index_k[:, :12], source.layers[1].index_k[:, :12])
    assert torch.equal(destination.layers[0].conv_state, source.layers[0].conv_state)
    assert torch.equal(destination.layers[0].recurrent_state, source.layers[0].recurrent_state)
    assert torch.equal(destination.ple_states[0].conv_state, source.ple_states[0].conv_state)
    assert torch.equal(destination.ple_token_history, source.ple_token_history)
    assert torch.equal(destination.layers[1].indexer_pool.keys[:, :3],
                       source.layers[1].indexer_pool.keys[:, :3])
    assert torch.equal(destination_head.layers[0].k[:, :, :8], source_head.layers[0].k[:, :, :8])

    # In place: every live buffer keeps the address a recording would have captured.
    assert destination.layers[0].conv_state.data_ptr() == pointers["conv"]
    assert destination.layers[0].recurrent_state.data_ptr() == pointers["recurrent"]
    assert destination.layers[1].k.data_ptr() == pointers["k"]
    assert destination.layers[1].index_k.data_ptr() == pointers["index_k"]
    assert destination.layers[1].indexer_pool.keys.data_ptr() == pointers["pool"]
    assert destination.ple_states[0].conv_state.data_ptr() == pointers["ple"]
    assert destination_head.layers[0].k.data_ptr() == pointers["head_k"]

    # Flags and lengths the next step reads.
    assert destination.length == 12
    assert destination.layers[0].initialized is True
    assert destination.layers[0].length == 12
    assert destination.layers[1].length == 12 and destination.layers[1].host_position == 12
    assert destination.layers[0].state_log is None
    pool = destination.layers[1].indexer_pool
    assert pool.tensor_ready is True and pool.completed == 3
    assert int(pool.completed_tensor) == 3
    assert pool.signature is not None
    # The signature is this process's, never the source's: a copied one would not validate.
    assert pool.signature[1][0] == destination.layers[1].index_k.data_ptr()
    assert destination_head.length == 0


def test_thaw_high_water_rows_past_the_prefix_are_untouched():
    source = backbone_cache(past=12, seed=5)
    bundle = freeze(source)
    destination = backbone_cache(past=0, seed=6)
    keep = destination.layers[1].k[:, :, 12:].clone()
    pd_state.thaw_state(bundle, cache=destination)
    assert torch.equal(destination.layers[1].k[:, :, 12:], keep)


def test_without_a_model_the_pool_is_left_for_one_recompute():
    source = backbone_cache(past=12, seed=7)
    bundle = freeze(source)
    destination = backbone_cache(past=0, seed=8)
    report = pd_state.thaw_state(bundle, cache=destination)
    assert report["indexer_pool_rebound"] == 0
    assert report["indexer_pool_needs_recompute"] == 1
    pool = destination.layers[1].indexer_pool
    assert pool.tensor_ready is False and pool.completed == 0


# ---------------------------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------------------------


def test_bytes_round_trip_is_bit_identical_and_pickle_free():
    source = backbone_cache(past=12, seed=9)
    bundle = freeze(source, head=head_cache(past=8, seed=10), first_draft=5,
                    chain_stack=torch.arange(6, dtype=torch.bfloat16).reshape(1, 1, 6))
    data = pd_state.bundle_to_bytes(bundle)
    restored = pd_state.bundle_from_bytes(data)
    assert restored.manifest["past"] == bundle.manifest["past"]
    assert pd_state.tensor_hashes(restored) == pd_state.tensor_hashes(bundle)
    # A zip of JSON plus raw element bytes: every entry is the manifest or a named tensor blob, so
    # no pickle opcode can be reached by loading it.
    import io, zipfile
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        names = archive.namelist()
    assert "manifest.json" in names
    assert all(name == "manifest.json" or name.startswith("t/") for name in names)
    destination = backbone_cache(past=0, seed=11)
    pd_state.thaw_state(restored, cache=destination, head_cache=head_cache(past=0, seed=110))
    assert torch.equal(destination.layers[1].k[:, :, :12], source.layers[1].k[:, :, :12])


def test_save_and_load_bundle_round_trips_through_a_file(tmp_path):
    bundle = freeze(backbone_cache(past=8, seed=12), first_draft=3)
    path = tmp_path / "nested" / "state.bin"
    written = pd_state.save_bundle(bundle, path)
    assert path.is_file() and written == path.stat().st_size
    loaded = pd_state.load_bundle(path)
    assert pd_state.tensor_hashes(loaded) == pd_state.tensor_hashes(bundle)


def test_corrupted_payload_is_rejected():
    bundle = freeze(backbone_cache(past=8, seed=13))
    data = bytearray(pd_state.bundle_to_bytes(bundle))
    data[len(data) // 2] ^= 0xFF
    with pytest.raises(pd_state.PDStateError):
        pd_state.bundle_from_bytes(bytes(data))


def test_truncated_or_non_zip_bytes_are_rejected():
    with pytest.raises(pd_state.PDStateError):
        pd_state.bundle_from_bytes(b"not a zip")
    data = pd_state.bundle_to_bytes(freeze(backbone_cache(past=8, seed=14)))
    with pytest.raises(pd_state.PDStateError):
        pd_state.bundle_from_bytes(data[: len(data) // 3])


def test_manifest_tensor_list_must_match_the_payload():
    bundle = freeze(backbone_cache(past=8, seed=15))
    broken = copy.deepcopy(bundle)
    broken.tensors.pop(sorted(broken.tensors)[0])
    # Serialization rewrites the manifest from the tensors it writes; the disagreement this test is
    # about is between a manifest and a payload, so thaw is where it has to be caught.
    destination = backbone_cache(past=0, seed=16)
    with pytest.raises(pd_state.PDStateError):
        pd_state.thaw_state(broken, cache=destination)
    assert destination.length == 0


# ---------------------------------------------------------------------------------------------
# Refusals: nothing may be written before every check passes
# ---------------------------------------------------------------------------------------------


def _guard(monkeypatch, destination):
    calls = []

    def record(self, *args, **kwargs):
        calls.append(type(self).__name__)
        return original(self, *args, **kwargs)

    original = torch.Tensor.copy_
    monkeypatch.setattr(torch.Tensor, "copy_", record)
    return calls


@pytest.mark.parametrize("mutation, message", [
    (lambda m: m.__setitem__("tp_rank", 3), "tensor-parallel rank"),
    (lambda m: m.__setitem__("tp_world_size", 4), "tensor-parallel width"),
    (lambda m: m.__setitem__("native_max_context", 8192), "native context"),
    (lambda m: m.__setitem__("version", 99), "version"),
    (lambda m: m.__setitem__("format", "other"), "format"),
    (lambda m: m["backbone"]["layers"][0].__setitem__("kind", "qsa"), "GatedDeltaNetCache"),
    (lambda m: m["backbone"]["layers"][1].__setitem__("length", 40), "outside"),
    (lambda m: m["backbone"]["layers"][2].__setitem__("kind", "gdn"), "destination"),
    (lambda m: m["metadata"].__setitem__("weights_sha256", "wrong"), "metadata"),
])
def test_mismatched_bundle_is_refused_before_any_copy(monkeypatch, mutation, message):
    source = backbone_cache(past=12, seed=17)
    bundle = freeze(source, tp_rank=0, tp_world_size=1, native_max_context=262144,
                    metadata={"weights_sha256": "abc"})
    broken = copy.deepcopy(bundle)
    mutation(broken.manifest)
    destination = backbone_cache(past=0, seed=18)
    calls = _guard(monkeypatch, destination)
    with pytest.raises(pd_state.PDStateError, match=message):
        pd_state.thaw_state(broken, cache=destination, tp_rank=0, tp_world_size=1,
                            native_max_context=262144,
                            expected_metadata={"weights_sha256": "abc"})
    assert calls == []
    assert destination.length == 0


def test_past_beyond_destination_arena_is_refused():
    source = backbone_cache(past=12, seed=19)
    bundle = freeze(source)
    narrow = backbone_cache(past=0, seed=20, width=8)
    with pytest.raises(pd_state.PDStateError):
        pd_state.thaw_state(bundle, cache=narrow)
    assert narrow.length == 0


def test_source_and_destination_arena_width_must_agree():
    bundle = freeze(backbone_cache(past=12, seed=21, width=64))
    wider = backbone_cache(past=0, seed=22, width=128)
    with pytest.raises(pd_state.PDStateError, match="arena"):
        pd_state.thaw_state(bundle, cache=wider)


def test_head_state_must_match_the_given_caches():
    source = backbone_cache(past=12, seed=23)
    with_head = freeze(source, head=head_cache(past=8, seed=24))
    with pytest.raises(pd_state.PDStateError, match="head_cache"):
        pd_state.thaw_state(with_head, cache=backbone_cache(past=0, seed=25))
    without_head = freeze(source)
    with pytest.raises(pd_state.PDStateError, match="head"):
        pd_state.thaw_state(without_head, cache=backbone_cache(past=0, seed=26),
                            head_cache=head_cache(past=0, seed=27))


def test_destination_dtype_must_match():
    bundle = freeze(backbone_cache(past=12, seed=28))
    destination = backbone_cache(past=0, seed=29)
    with torch.inference_mode(False):
        destination.layers[0].conv_state = destination.layers[0].conv_state.float()
    with pytest.raises(pd_state.PDStateError, match="dtype"):
        pd_state.thaw_state(bundle, cache=destination)


def test_cache_length_disagreeing_with_past_is_refused_at_freeze():
    source = backbone_cache(past=12, seed=30)
    with pytest.raises(pd_state.PDStateError, match="disagrees"):
        pd_state.freeze_state(cache=source, past=10, next_token=1)


def test_non_json_metadata_is_refused():
    with pytest.raises(pd_state.PDStateError, match="JSON"):
        freeze(backbone_cache(past=8, seed=31), metadata={"bad": object()})


# ---------------------------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------------------------


def test_hashes_identify_replica_groups_without_assuming_them():
    first = freeze(backbone_cache(past=12, seed=32))
    second = freeze(backbone_cache(past=12, seed=32))
    third = freeze(backbone_cache(past=12, seed=33))
    assert pd_state.bundle_digest(first) == pd_state.bundle_digest(second)
    assert pd_state.bundle_digest(first) != pd_state.bundle_digest(third)
    assert all(pd_state.identical_tensors(first, second).values())
    differences = pd_state.identical_tensors(first, third)
    # The GDN layer is per-rank unique; a dedup scheme is allowed to find either answer, but the
    # helper has to report it rather than assume it.
    assert differences["backbone.layer.0.recurrent_state"] is False


def test_bundle_digest_includes_metadata_and_scalars():
    base = freeze(backbone_cache(past=8, seed=34), metadata={"a": 1})
    other = freeze(backbone_cache(past=8, seed=34), metadata={"a": 2})
    assert pd_state.bundle_digest(base) != pd_state.bundle_digest(other)


# ---------------------------------------------------------------------------------------------
# Whole-state coverage guards
# ---------------------------------------------------------------------------------------------


def test_every_live_state_tensor_is_either_exported_or_explicitly_derived():
    source = backbone_cache(past=12, seed=35)
    bundle = freeze(source)
    names = set(bundle.tensor_names())
    exported = {
        "backbone.layer.0.conv_state", "backbone.layer.0.recurrent_state",
        "backbone.layer.1.k", "backbone.layer.1.v", "backbone.layer.1.index_k",
        "backbone.layer.1.indexer_keys", "backbone.ple.0.conv_state",
        "backbone.ple_token_history",
    }
    assert exported <= names
    # Anything a decode step mutates and reads later must be in the payload...
    assert not any(name.endswith(("cos", "sin")) for name in names)


def test_frozen_prefix_shape_follows_the_layer_length_not_the_arena():
    source = backbone_cache(past=12, seed=36, width=64)
    source.layers[1].length = 9
    bundle = freeze(source, past=12)
    assert bundle.tensors["backbone.layer.1.k"].shape[-2] == 9
    assert bundle.manifest["backbone"]["layers"][1]["length"] == 9
    destination = backbone_cache(past=0, seed=37, width=64)
    report = pd_state.thaw_state(bundle, cache=destination)
    assert report["past"] == 12 and destination.length == 12
    assert destination.layers[1].length == 9


def test_prefix_slice_of_a_multi_head_arena_exports_contiguous_bytes():
    """A strided source prefix must still travel as plain contiguous element bytes.

    At the real per-rank geometry the KV arena has one head, so its prefix happens to be
    contiguous; a wider-head arena does not have that accident, and the format must not depend on
    it. The payload is the same values in the same order, never a stride the consumer must honour.
    """
    torch.manual_seed(42)
    layer = QSAAttentionCache(config(), 1, 64, torch.device("cpu"), torch.bfloat16,
                              num_kv_heads=2)
    layer.k.normal_()
    layer.v.normal_()
    layer.index_k.normal_()
    layer.length = 12
    cache = Qwen4ExpCache(layers=[layer], length=12)
    source_slice = layer.k[:, :, :12]
    assert not source_slice.is_contiguous()
    bundle = pd_state.freeze_state(cache=cache, past=12, next_token=1)
    for name, tensor in bundle.tensors.items():
        assert tensor.is_contiguous(), name
    assert torch.equal(bundle.tensors["backbone.layer.0.k"], source_slice.contiguous())


def test_head_state_is_addressed_through_the_head_module_path():
    """The rebind for the head pool reads `model.mtp.layer.attn.indexer`, as the real head does."""
    source = backbone_cache(past=12, seed=43)
    source_head = head_cache(past=8, seed=44)
    bundle = freeze(source, head=source_head)
    destination = backbone_cache(past=0, seed=45)
    destination_head = head_cache(past=0, seed=46)
    head_indexer = Indexer()
    model = model_stub(backbone_indexer=Indexer(), head_indexer=head_indexer)
    assert model.mtp.layer.attn.indexer is head_indexer
    report = pd_state.thaw_state(bundle, cache=destination, head_cache=destination_head, model=model)
    assert report["indexer_pool_rebound"] == 2
    head_pool = destination_head.layers[0].indexer_pool
    assert head_pool.signature[0] == id(head_indexer)


def test_host_kv_round_trips_when_pinned_memory_is_available(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("pinned host KV needs a CUDA build of torch")
    monkeypatch.setenv("POCKETLLM_QWEN4_HOST_KV", "1")
    source = backbone_cache(past=12, seed=38)
    assert source.layers[1].host_kv is True
    bundle = freeze(source)
    assert bundle.manifest["backbone"]["layers"][1]["host_kv"] is True
    destination = backbone_cache(past=0, seed=39)
    pd_state.thaw_state(bundle, cache=destination)
    assert destination.layers[1].host_kv is True
    assert torch.equal(destination.layers[1].k[:, :, :12], source.layers[1].k[:, :, :12])


def test_bits_survive_not_just_values():
    """-0.0, infinities and a NaN must come back with the same bit patterns.

    A numeric equality check cannot see the difference between +0.0 and -0.0 and treats every NaN
    as unequal; the transfer contract is exact element bits, so the comparison is on the int16
    storage view.
    """
    source = backbone_cache(past=12, seed=70)
    exotic = torch.tensor([-0.0, 0.0, float("inf"), float("-inf"), float("nan")],
                          dtype=torch.bfloat16)
    flat = source.layers[0].conv_state.reshape(-1)
    flat[: exotic.numel()] = exotic
    bundle = freeze(source)
    stored = bundle.tensors["backbone.layer.0.conv_state"]
    assert torch.equal(stored.view(torch.int16), source.layers[0].conv_state.view(torch.int16))

    destination = backbone_cache(past=0, seed=71)
    pd_state.thaw_state(bundle, cache=destination)
    assert torch.equal(destination.layers[0].conv_state.view(torch.int16),
                       source.layers[0].conv_state.view(torch.int16))
    restored = destination.layers[0].conv_state.reshape(-1)[: exotic.numel()]
    assert torch.equal(restored.view(torch.int16), exotic.view(torch.int16))


@pytest.mark.parametrize("drop", ["past", "next_token", "tensors", "backbone"])
def test_malformed_manifest_is_a_bundle_error_not_a_keyerror(drop):
    bundle = freeze(backbone_cache(past=8, seed=50))
    broken = copy.deepcopy(bundle)
    broken.manifest.pop(drop)
    destination = backbone_cache(past=0, seed=51)
    with pytest.raises(pd_state.PDStateError):
        pd_state.thaw_state(broken, cache=destination)
    assert destination.length == 0


def test_inconsistent_destination_pool_geometry_is_refused_before_copy():
    bundle = freeze(backbone_cache(past=12, seed=52))
    destination = backbone_cache(past=0, seed=53)
    layer = destination.layers[1]
    with torch.inference_mode(False):
        layer.indexer_pool = _QSAIndexerPool(layer.index_k, 8)
    with pytest.raises(pd_state.PDStateError, match="pool"):
        pd_state.thaw_state(bundle, cache=destination)


def test_missing_destination_pool_is_created_and_filled():
    """The real decode-side cache comes from `make_cache`, where `indexer_pool` is still None."""
    source = backbone_cache(past=12, seed=60)
    bundle = freeze(source)
    destination = backbone_cache(past=0, seed=61)
    destination.layers[1].indexer_pool = None
    report = pd_state.thaw_state(bundle, cache=destination,
                                 model=model_stub(backbone_indexer=Indexer()))
    pool = destination.layers[1].indexer_pool
    assert pool is not None and pool.tensor_ready is True and pool.completed == 3
    assert torch.equal(pool.keys[:, :3], source.layers[1].indexer_pool.keys[:, :3])
    assert report["indexer_pool_rebound"] == 1


def test_destination_without_a_rope_table_keeps_bytes_but_defers_pool_reuse():
    """A cache that has not run a forward has no RoPE table; the pool must not claim validity."""
    source = backbone_cache(past=12, seed=62)
    bundle = freeze(source)
    destination = backbone_cache(past=0, seed=63)
    destination.cos = None
    destination.sin = None
    report = pd_state.thaw_state(bundle, cache=destination,
                                 model=model_stub(backbone_indexer=Indexer()))
    pool = destination.layers[1].indexer_pool
    assert report["indexer_pool_rebound"] == 0
    assert report["indexer_pool_needs_recompute"] == 1
    assert pool.tensor_ready is False
    # The imported bytes are still there, so the one-time recompute starts from the right prefix.
    assert torch.equal(pool.keys[:, :3], source.layers[1].indexer_pool.keys[:, :3])
    assert report["backbone_rope_table_ready"] is False


def test_imported_state_survives_a_second_freeze():
    """A resumed decode can itself hand off again; the round trip must be stable."""
    source = backbone_cache(past=12, seed=40)
    first = freeze(source, first_draft=9)
    middle = backbone_cache(past=0, seed=41)
    # Rebind the pools, as the real resume path does: without that the second freeze would have to
    # omit them and the comparison below would be about a slower path, not about transport.
    pd_state.thaw_state(first, cache=middle,
                        model=model_stub(backbone_indexer=Indexer()))
    second = freeze(middle, first_draft=9)
    assert pd_state.tensor_hashes(first) == pd_state.tensor_hashes(second)
    assert json.dumps(second.manifest, sort_keys=True) == json.dumps(first.manifest, sort_keys=True)