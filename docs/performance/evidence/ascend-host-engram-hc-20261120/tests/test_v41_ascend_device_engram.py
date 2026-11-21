"""Reversible CPU logic oracle for host-table/device-compute Engram dispatch."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from relic_core.kernels.original_fp8_engram import original_engram_gate, original_fp8_engram_projection
from relicllm.models.deepseek_v4_1 import mma, modules
from relicllm.models.deepseek_v4_1.ascend_device_engram import HostTableDeviceEngram


class Model(nn.Module):
    def __init__(self, *, quantized=False):
        super().__init__()
        generator = torch.Generator().manual_seed(71)
        cfg = SimpleNamespace(dim=32, hc_mult=1, norm_eps=1e-6)
        layout = SimpleNamespace(layer_ids=(1,), max_ngram_size=2, n_heads=1, head_dim=32)
        table = modules.ResidentEngramTable(torch.randn(16, 32, generator=generator).bfloat16(),
                                           torch.ones(16, 1))
        engram = modules.Engram(cfg, 1, layout, table, device="cpu", quantized=quantized)
        with torch.no_grad():
            engram.wkv.weight.copy_(torch.randn(64, 32, generator=generator).bfloat16() * 0.0625)
            engram.q_weight.copy_(torch.randn(1, 32, generator=generator).bfloat16())
            engram.k_weight.copy_(torch.randn(1, 32, generator=generator).bfloat16())
            if quantized:
                codes = engram.wkv.weight.detach().to(torch.float8_e4m3fn)
                engram.wkv.weight.copy_(codes.float().bfloat16())
                mma.attach_source_codes(engram.wkv.weight, codes, torch.ones(2, 1))
        block = nn.Module()
        block.engram = engram
        self.layers = nn.ModuleList([block])

    def forward(self, x, start_pos, hash_ids, mask=None):
        return self.layers[0].engram(x, hash_ids[:, :, 0, :], mask)


def inputs(seed=5, tokens=1):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(1, tokens, 1, 32, generator=generator).bfloat16()
    ids = torch.randint(0, 16, (1, tokens, 1, 1), generator=generator)
    return x, ids


def oracle(model, x, hashes, mask=None):
    engram = model.layers[0].engram
    lookup = engram.embed.lookup(hashes[:, :, 0, :], torch.device("cpu")).reshape(1, 32)
    kv = original_fp8_engram_projection(lookup, engram.wkv.weight.detach().t().contiguous(),
        quantized=engram._quantized, certified_original_bf16_weight=True, backend="torch")
    channel = engram.q_weight.detach().float() * engram.k_weight.detach().float()
    mask = torch.ones(1, 1, dtype=torch.bool) if mask is None else mask
    return original_engram_gate(x, kv, channel, mask, eps=engram.eps)


@pytest.mark.parametrize("quantized", [False, True])
def test_nonzero_decode_matches_explicit_original_operand_oracle(quantized):
    model = Model(quantized=quantized)
    original = model.forward
    e = model.layers[0].engram
    original_engram = e.forward
    table = e.embed
    weights = [p.clone() for p in e.parameters()]
    with HostTableDeviceEngram(model, backend="torch", capture_graphs=False) as manager:
        for seed in (0, 1, 7, 19):
            x, hashes = inputs(seed)
            expected = oracle(model, x, hashes)
            actual = model(x, 0, hashes)
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
        assert manager.describe()["prefetched_lookups"] == 4
        assert manager.describe()["hidden_state_D2H_bytes"] == 0
        assert manager.describe()["lookup_H2D_bytes"] == 0  # CPU oracle, not a device claim
        assert e.embed is table
        assert all(torch.equal(p, old) for p, old in zip(e.parameters(), weights))
    assert model.forward.__func__ is original.__func__
    assert model.forward.__self__ is original.__self__
    assert "forward" not in model.__dict__
    assert e.forward.__func__ is original_engram.__func__
    assert e.forward.__self__ is original_engram.__self__
    assert "forward" not in e.__dict__
    assert not hasattr(model, "_host_table_device_engram_owner")


def test_mask_updates_and_exported_outputs_are_not_overwritten():
    model = Model()
    with HostTableDeviceEngram(model, backend="torch", capture_graphs=False) as manager:
        x, hashes = inputs(1)
        first = model(x, 0, hashes, torch.zeros(1, 1, dtype=torch.bool))
        frozen = first.clone()
        assert torch.equal(first, x)
        y, other = inputs(2)
        second = model(y, 1, other)
        assert torch.equal(second, oracle(model, y, other))
        assert torch.equal(first, frozen)
        assert manager.stats["decode_calls"] == 2


def test_prefill_passthrough_retains_original_forward():
    model = Model()
    x, hashes = inputs(11, tokens=3)
    expected = model(x, 0, hashes)
    with HostTableDeviceEngram(model, backend="torch", capture_graphs=False) as manager:
        actual = model(x, 0, hashes)
        assert torch.equal(actual, expected)
        assert manager.stats["passthrough_calls"] == 1
        assert manager.stats["decode_calls"] == 0


def test_column_tp_gathers_complete_bf16_kv_before_gate():
    model = Model()
    calls = []
    holder = {}
    def gather(output, local):
        entry = next(iter(holder["manager"].entries.values()))
        full = entry.lookup @ model.layers[0].engram.wkv.weight.detach().t().contiguous()
        assert torch.equal(local, full[:, :32].contiguous().view(torch.int8).reshape(-1))
        output.copy_(full.view(torch.int8).reshape(-1))
        calls.append((output.dtype, local.dtype, local.numel()))
    manager = HostTableDeviceEngram(model, backend="torch", capture_graphs=False, rank=0, world=2, gather=gather)
    holder["manager"] = manager
    with manager:
        x, hashes = inputs()
        assert torch.equal(model(x, 0, hashes), oracle(model, x, hashes))
        entry = next(iter(manager.entries.values()))
        assert entry.weight_kn.shape == (32, 32)
        assert calls == [(torch.int8, torch.int8, 64)]
        assert manager.describe()["WKV_output_channel_TP"] == 2


def test_original_source_mutation_is_refused():
    model = Model()
    with HostTableDeviceEngram(model, backend="torch", capture_graphs=False):
        with torch.no_grad():
            model.layers[0].engram.wkv.weight.add_(1)
        x, hashes = inputs()
        with pytest.raises(RuntimeError, match="sources|parameters"):
            model(x, 0, hashes)


def test_prefetched_id_mismatch_is_refused():
    model = Model()
    with HostTableDeviceEngram(model, backend="torch", capture_graphs=False) as manager:
        x, hashes = inputs(2)
        manager._prepare(hashes)
        other = (hashes + 1) % 16
        entry = next(iter(manager.entries.values()))
        with pytest.raises(RuntimeError, match="IDs"):
            manager._forward(entry, x, other[:, :, 0, :], None)


def test_lookup_errors_restore_original_binding():
    model = Model()
    original = model.forward
    def failure(*_):
        raise RuntimeError("lookup failed")
    model.layers[0].engram.embed.lookup = failure
    with pytest.raises(RuntimeError, match="lookup failed"):
        with HostTableDeviceEngram(model, backend="torch", capture_graphs=False):
            x, hashes = inputs()
            model(x, 0, hashes)
    assert model.forward.__func__ is original.__func__
    assert model.forward.__self__ is original.__self__
    assert "forward" not in model.__dict__


def test_foreign_forward_owner_is_not_overwritten_on_close():
    model = Model()
    manager = HostTableDeviceEngram(model, backend="torch", capture_graphs=False).install()
    replacement = model.forward
    foreign = lambda *_: None
    model.forward = foreign
    with pytest.raises(RuntimeError, match="owner"):
        manager.close()
    assert model.forward is foreign
    model.forward = replacement
    manager.close()
    manager.close()


@pytest.mark.parametrize("options", [{"max_device_bytes": 1}, {"capture_graphs": True},
                                    {"world": 2}, {"rank": 2, "world": 1}])
def test_bad_budget_graph_or_tp_contracts_fail_closed(options):
    model = Model()
    base = {"backend": "torch", "capture_graphs": False, **options}
    with pytest.raises(ValueError):
        HostTableDeviceEngram(model, **base)


def test_quantized_tp_cannot_split_original_scale_blocks():
    model = Model(quantized=True)
    with pytest.raises(ValueError, match="scale blocks"):
        HostTableDeviceEngram(model, backend="torch", capture_graphs=False, rank=0, world=4, gather=lambda *_: None)


def test_unused_controller_releases_snapshots_and_cannot_install_later():
    model = Model()
    manager = HostTableDeviceEngram(model, backend="torch", capture_graphs=False)
    assert manager.entries
    manager.close(); manager.close()
    assert not manager.entries and manager.closed
    assert "forward" not in model.__dict__
    with pytest.raises(RuntimeError, match="fresh"):
        manager.install()


@pytest.mark.parametrize("target", ["model", "engram"])
def test_install_refuses_a_forward_changed_after_snapshot(target):
    model = Model()
    manager = HostTableDeviceEngram(model, backend="torch", capture_graphs=False)
    owner = model if target == "model" else model.layers[0].engram
    foreign = lambda *_: None
    owner.forward = foreign
    with pytest.raises(RuntimeError, match="owner"):
        manager.install()
    manager.close()
    assert owner.forward is foreign


def test_table_geometry_and_hash_layer_source_mutation_is_refused():
    model = Model()
    with HostTableDeviceEngram(model, backend="torch", capture_graphs=False):
        model.layers[0].engram.layer_hash_index = 1
        x, hashes = inputs()
        with pytest.raises(RuntimeError, match="parameters|sources"):
            model(x, 0, hashes)


def test_original_table_value_mutation_is_refused():
    model = Model()
    with HostTableDeviceEngram(model, backend="torch", capture_graphs=False):
        model.layers[0].engram.embed.weight.add_(1)
        x, hashes = inputs()
        with pytest.raises(RuntimeError, match="parameters|sources"):
            model(x, 0, hashes)


def test_lookup_failure_still_fences_device_before_restoration(monkeypatch):
    from concurrent.futures import Future
    model = Model()
    manager = HostTableDeviceEngram(model, backend="torch", capture_graphs=False).install()
    pending = Future(); pending.set_exception(RuntimeError("worker failed"))
    manager.pending = {1: ((), pending)}
    calls = []
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda device: calls.append(device)), raising=False)
    manager.backend = "triton"  # exercise cleanup only; this is not a device execution claim
    with pytest.raises(RuntimeError, match="worker failed"):
        manager.close()
    assert calls == [torch.device("cpu")]
    assert not manager.entries and not manager.installed
    assert "forward" not in model.__dict__


def test_wrong_default_group_tp_extent_is_refused(monkeypatch):
    import relicllm.models.deepseek_v4_1.ascend_device_engram as binding
    model = Model()
    monkeypatch.setattr(binding.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(binding.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(binding.dist, "get_world_size", lambda: 8)
    with pytest.raises(ValueError, match="process group"):
        HostTableDeviceEngram(model, backend="torch", capture_graphs=False, rank=0, world=2)


def test_cpu_prefetch_owner_is_rejected(monkeypatch):
    model = Model()
    def existing(*_):
        pass
    existing._engram_prefetch = object()
    monkeypatch.setattr(modules, "quantized_linear", existing)
    manager = HostTableDeviceEngram(model, backend="torch", capture_graphs=False)
    with pytest.raises(RuntimeError, match="CPU WKV prefetch"):
        manager.install()
