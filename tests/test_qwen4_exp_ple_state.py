"""Request-owned PLE state, independently checked without a checkpoint or GPU."""

from __future__ import annotations

from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from relicllm.models.qwen4_exp import model as model_module
from relicllm.models.qwen4_exp.attention import slot_layout
from relicllm.models.qwen4_exp.config import Qwen4ExpTextConfig
from relicllm.models.qwen4_exp.model import (
    PLELayer, PLEState, Qwen4ExpCache, Qwen4ExpDecoderLayer, Qwen4ExpModel,
    restore_cache, snapshot_cache,
)


@pytest.fixture(autouse=True)
def reference_schedule(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(model_module, "_SCHEDULE", "by_layer")
    with torch.device("cpu"):
        yield


def _config(layers: int = 2, kernel: int = 4, ple_ids=None) -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        hidden_size=4, hc_count=2, hc_lowrank=2, vocab_size=32,
        num_hidden_layers=layers, layer_types=["linear_attention"] * layers,
        num_attention_heads=2, num_key_value_heads=1, head_dim=4,
        partial_rotary_factor=0.5, ple_layer_ids=list(range(1, layers + 1)) if ple_ids is None else ple_ids,
        ple_embed_dim=3, ple_conv_kernel_size=kernel, ngram_size=3,
        heads_per_ngram=1, ngram_vocab_size_base=17, make_ngram_vocab_size_divisible_by=4,
        eos_token_id=[0], pad_token_id=0,
    )


def _ple(config: Qwen4ExpTextConfig, index: int = 0) -> PLELayer:
    generator = torch.Generator().manual_seed(2048 + index)
    rows = config.ngram_head_vocab_sizes(index)[2]
    table = torch.randn(rows, config.ple_embed_dim, generator=generator) * 0.3
    width = config.ngram_heads * config.ple_embed_dim
    weights = {
        "key_proj": torch.randn(config.hc_hidden_size, width, generator=generator) * 0.2,
        "value_proj": torch.randn(config.hidden_size, width, generator=generator) * 0.2,
        "norm_key": torch.randn(config.hc_hidden_size, generator=generator) * 0.05,
        "norm_query": torch.randn(config.hc_hidden_size, generator=generator) * 0.05,
        "norm_conv": torch.randn(config.hc_hidden_size, generator=generator) * 0.05,
        "conv1d": torch.randn(config.hc_hidden_size, 1, config.ple_conv_kernel_size, generator=generator) * 0.2,
    }
    return PLELayer(config, weights, index, torch.device("cpu"), ngram_table=lambda ids: F.embedding(ids, table))


def _streams(config: Qwen4ExpTextConfig, batch: int, length: int) -> torch.Tensor:
    return torch.randn(batch, length, config.hc_hidden_size, generator=torch.Generator().manual_seed(811))


def _tokens(batch: int, length: int, offset: int = 0) -> torch.Tensor:
    tokens = (torch.arange(length).view(1, -1) + torch.arange(batch).view(-1, 1) * 5 + offset) % 30 + 1
    if length > 4:
        tokens[:, 4] = 0
    return tokens.long()


def _history(config: Qwen4ExpTextConfig, tokens: torch.Tensor) -> torch.Tensor:
    return torch.cat((tokens.new_full((tokens.shape[0], config.ngram_size - 1), config.primary_eos_token_id), tokens), dim=1)


def _norm_reference(values: torch.Tensor, weight: torch.Tensor, config: Qwen4ExpTextConfig) -> torch.Tensor:
    groups = values.float().reshape(*values.shape[:-1], config.hc_count, config.hidden_size)
    groups = groups / torch.sqrt(groups.square().mean(dim=-1, keepdim=True) + config.rms_norm_eps)
    return (groups * (1 + weight.float().reshape(config.hc_count, config.hidden_size))).flatten(-2)


def _conv_reference(ple: PLELayer, values: torch.Tensor) -> torch.Tensor:
    # Explicit causal cross-correlation; no production convolution or state helper.
    result = torch.zeros_like(values)
    for t in range(values.shape[1]):
        for tap in range(ple.conv_kernel_size):
            source = t - (ple.conv_kernel_size - 1 - tap) * ple.conv_dilation
            if source >= 0:
                result[:, t] += values[:, source] * ple.conv1d_weight[:, 0, tap]
    return F.silu(result)


def _ple_reference(ple: PLELayer, values: torch.Tensor, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Hash/table lookup is not under test; projections, norms, gate and carry are independent.
    config = ple.config
    embeddings = ple.ngram_table(ple.hasher.row_ids(history, values.shape[1])).flatten(-2)
    key = _norm_reference(F.linear(embeddings, ple.key_proj), ple.norm_key.weight, config).unflatten(-1, (config.hc_count, config.hidden_size))
    query = _norm_reference(values, ple.norm_query.weight, config).unflatten(-1, (config.hc_count, config.hidden_size))
    gate = (key * query).sum(dim=-1, keepdim=True) / (config.hidden_size ** 0.5)
    gate = gate.sign() * gate.abs().clamp_min(1e-6).sqrt()
    value = F.linear(embeddings, ple.value_proj)
    gated = (gate.sigmoid() * value.unsqueeze(-2)).flatten(-2)
    normalized = _norm_reference(gated, ple.norm_conv.weight, config)
    return gated + _conv_reference(ple, normalized), normalized


def _tail(values: torch.Tensor, span: int) -> torch.Tensor:
    values = values.transpose(1, 2)
    if not span:
        return values[..., :0]
    return F.pad(values, (max(0, span - values.shape[-1]), 0))[..., -span:]


class _KeepStreams:
    def __init__(self, config: Qwen4ExpTextConfig) -> None:
        self.config = config

    def __call__(self, values: torch.Tensor):
        mixed = values.unflatten(-1, (self.config.hc_count, self.config.hidden_size)).mean(dim=-2)
        return mixed, values, values.new_ones((*values.shape[:-1], self.config.hc_count))


def _decoder(config: Qwen4ExpTextConfig, idx: int) -> Qwen4ExpDecoderLayer:
    # Exercise the real decoder __call__, with zero non-PLE blocks to isolate state ownership.
    layer = object.__new__(Qwen4ExpDecoderLayer)
    layer.config, layer.layer_idx, layer.is_linear = config, idx, True
    layer.ple = _ple(config, config.ple_layer_index(idx))
    layer.attn_hc, layer.mlp_hc = _KeepStreams(config), _KeepStreams(config)
    layer.attn = lambda mixed, cache: torch.zeros_like(mixed)
    layer.all_reduce = None
    layer.router = lambda flat: (flat.new_ones((flat.shape[0], 1)), torch.zeros(flat.shape[0], 1, dtype=torch.long))
    def moe(idx, flat, indices, weights):
        assert indices.dtype == torch.long and weights.dtype == flat.dtype
        return torch.zeros_like(flat)
    layer.moe = moe
    layer.shared_expert = lambda flat: torch.zeros_like(flat)
    layer.shared_expert_gate = torch.zeros(1, config.hidden_size)
    layer.make_cache = lambda batch, length, device, dtype: SimpleNamespace(length=0)
    return layer


def _model(config: Qwen4ExpTextConfig | None = None, local_ids=None) -> Qwen4ExpModel:
    config = config or _config()
    local_ids = list(range(config.num_hidden_layers)) if local_ids is None else local_ids
    layers = [_decoder(config, idx) if idx in local_ids else None for idx in range(config.num_hidden_layers)]
    generator = torch.Generator().manual_seed(99)
    embedding = torch.randn(config.vocab_size, config.hidden_size, generator=generator) * 0.3
    head = torch.randn(config.vocab_size, config.hidden_size, generator=generator) * 0.2
    return Qwen4ExpModel(
        config, layers, embed_tokens=lambda ids: F.embedding(ids, embedding), lm_head=head,
        final_mixer=lambda values: values.unflatten(-1, (config.hc_count, config.hidden_size)).mean(dim=-2),
        device=torch.device("cpu"), dtype=torch.float32,
    )


def _model_reference(model: Qwen4ExpModel, tokens: torch.Tensor):
    hidden = model.embed_tokens(tokens).repeat(1, 1, model.config.hc_count)
    history, states = _history(model.config, tokens), {}
    for idx, layer in enumerate(model.layers):
        if layer is not None:
            contribution, carry = _ple_reference(layer.ple, hidden, history)
            hidden = hidden + contribution
            states[idx] = _tail(carry, layer.ple.short_conv_state_len)
    logits = F.linear(model.final_mixer(hidden), model.lm_head)
    return logits, states


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("schedule", ["by_layer", "by_outer"])
def test_request_caches_interleave_without_ple_leakage(batch: int, schedule: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(model_module, "_SCHEDULE", schedule)
    model = _model()
    tokens = {"a": _tokens(batch, 12), "b": _tokens(batch, 11, 7)}
    expected = {name: _model_reference(model, ids) for name, ids in tokens.items()}
    caches = {"a": model.make_cache(batch, 32)}
    first = model.forward(tokens["a"][:, :2], cache=caches["a"], past_len=0)
    torch.testing.assert_close(first, expected["a"][0][:, :2], rtol=2e-5, atol=2e-6)
    saved_a = {idx: state.conv_state.clone() for idx, state in caches["a"].ple_states.items()}
    caches["b"] = model.make_cache(batch, 32)
    for idx, state in caches["a"].ple_states.items():
        torch.testing.assert_close(state.conv_state, saved_a[idx], rtol=0, atol=0)
        assert state.conv_state.data_ptr() != caches["b"].ple_states[idx].conv_state.data_ptr()
    pointers = {name: {idx: state.conv_state.data_ptr() for idx, state in cache.ple_states.items()} for name, cache in caches.items()}
    positions = {"a": 2, "b": 0}
    for name, count in (("b", 3), ("a", 1), ("b", 1), ("a", 4), ("b", 4), ("a", 5), ("b", 3)):
        other = "b" if name == "a" else "a"
        untouched = {idx: state.conv_state.clone() for idx, state in caches[other].ple_states.items()}
        start, end = positions[name], positions[name] + count
        actual = model.forward(tokens[name][:, start:end], cache=caches[name], past_len=start)
        torch.testing.assert_close(actual, expected[name][0][:, start:end], rtol=2e-5, atol=2e-6)
        for idx, state in caches[other].ple_states.items():
            torch.testing.assert_close(state.conv_state, untouched[idx], rtol=0, atol=0)
        positions[name] = end
    for name, cache in caches.items():
        assert cache.length == tokens[name].shape[1]
        torch.testing.assert_close(cache.ple_token_history, tokens[name][:, -2:])
        for idx, state in cache.ple_states.items():
            assert state.conv_state.data_ptr() == pointers[name][idx]
            torch.testing.assert_close(state.conv_state, expected[name][1][idx], rtol=2e-5, atol=2e-6)
    assert all(layer.ple.conv_state is None for layer in model.layers)


def test_make_cache_does_not_reset_existing_request_or_standalone_state() -> None:
    model = _model()
    legacy = torch.full((1, model.config.hc_hidden_size, 9), 7.0)
    model.layers[0].ple.conv_state = legacy
    first = model.make_cache(1, 16)
    assert model.layers[0].ple.conv_state is legacy
    first.ple_states[0].conv_state.fill_(3)
    previous = first.ple_states[0].conv_state.clone()
    second = model.make_cache(1, 16)
    assert model.layers[0].ple.conv_state is legacy
    torch.testing.assert_close(first.ple_states[0].conv_state, previous, rtol=0, atol=0)
    assert torch.count_nonzero(second.ple_states[0].conv_state) == 0


def test_cache_ple_owners_use_absolute_local_layer_ids() -> None:
    model = _model(_config(layers=4, ple_ids=[2, 4]), local_ids=[1, 3])
    cache = model.make_cache(2, 16)
    assert cache.layers[0] is None and cache.layers[2] is None
    assert set(cache.ple_states) == {1, 3}
    for state in cache.ple_states.values():
        assert state.conv_state.shape == (2, model.config.hc_hidden_size, 9)
        assert state.conv_state.dtype == model.dtype and state.conv_state.device == model.device
        assert torch.count_nonzero(state.conv_state) == 0


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("initialized", [False, True])
def test_short_initial_ple_window_is_zero_padded_to_nine(batch: int, initialized: bool) -> None:
    config, ple = _config(), _ple(_config())
    values = _streams(config, batch, 2)
    state = PLEState(torch.zeros(batch, config.hc_hidden_size, 9) if initialized else None)
    actual = ple._short_conv(values, use_cache=True, state_cache=state)
    torch.testing.assert_close(actual, _conv_reference(ple, values), rtol=2e-5, atol=2e-6)
    assert state.conv_state.shape == (batch, config.hc_hidden_size, 9)
    torch.testing.assert_close(state.conv_state, _tail(values, 9), rtol=0, atol=0)
    assert ple.conv_state is None


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("chunks", [(1, 1, 2, 3, 4), (2, 3, 6)])
def test_full_and_chunked_ple_match_independent_reference(batch: int, chunks: tuple[int, ...]) -> None:
    config, ple = _config(), _ple(_config())
    values, tokens = _streams(config, batch, sum(chunks)), _tokens(batch, sum(chunks))
    expected, carry = _ple_reference(ple, values, _history(config, tokens))
    state = PLEState(torch.zeros(batch, config.hc_hidden_size, 9))
    pointer, past, pieces = state.conv_state.data_ptr(), 0, []
    history = tokens.new_zeros((batch, 2))
    for count in chunks:
        window = torch.cat((history, tokens[:, past:past + count]), dim=1)
        pieces.append(ple(values[:, past:past + count], window, count, use_cache=True, state_cache=state))
        history, past = window[:, -2:].clone(), past + count
    torch.testing.assert_close(torch.cat(pieces, dim=1), expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(state.conv_state, _tail(carry, 9), rtol=2e-5, atol=2e-6)
    assert state.conv_state.data_ptr() == pointer and ple.conv_state is None


def test_legacy_standalone_cached_and_uncached_api() -> None:
    config, ple = _config(), _ple(_config())
    values, tokens = _streams(config, 1, 8), _tokens(1, 8)
    expected, _ = _ple_reference(ple, values, _history(config, tokens))
    uncached = ple(values, _history(config, tokens), 8, use_cache=False)
    torch.testing.assert_close(uncached, expected, rtol=2e-5, atol=2e-6)
    first = ple(values[:, :2], _history(config, tokens[:, :2]), 2, use_cache=True)
    pointer = ple.conv_state.data_ptr()
    second = ple(values[:, 2:], tokens, 6, use_cache=True)
    torch.testing.assert_close(torch.cat((first, second), dim=1), expected, rtol=2e-5, atol=2e-6)
    assert ple.conv_state.shape == (1, config.hc_hidden_size, 9)
    assert ple.conv_state.data_ptr() == pointer
    ple.reset_cache()
    assert ple.conv_state is None


def test_uncached_call_ignores_request_and_standalone_carry() -> None:
    config, ple = _config(), _ple(_config())
    values, tokens = _streams(config, 1, 3), _tokens(1, 3)
    state = PLEState(torch.full((1, config.hc_hidden_size, 9), 99.0))
    ple.conv_state = -state.conv_state.clone()
    expected, _ = _ple_reference(ple, values, _history(config, tokens))
    actual = ple(values, _history(config, tokens), 3, use_cache=False, state_cache=state)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert torch.all(state.conv_state == 99) and torch.all(ple.conv_state == -99)


def test_zero_length_convolution_carry() -> None:
    config = _config(kernel=1)
    ple, values = _ple(config), _streams(config, 1, 3)
    state = PLEState(torch.zeros(1, config.hc_hidden_size, 0))
    actual = ple._short_conv(values, use_cache=True, state_cache=state)
    torch.testing.assert_close(actual, _conv_reference(ple, values), rtol=2e-5, atol=2e-6)
    assert state.conv_state.shape == (1, config.hc_hidden_size, 0)


def test_inference_allocated_ple_buffer_updates_in_place() -> None:
    config, ple = _config(), _ple(_config())
    with torch.inference_mode():
        state = PLEState(torch.zeros(1, config.hc_hidden_size, 9))
    pointer = state.conv_state.data_ptr()
    values = _streams(config, 1, 2)
    ple._short_conv(values, use_cache=True, state_cache=state)
    assert state.conv_state.data_ptr() == pointer
    torch.testing.assert_close(state.conv_state, _tail(values, 9), rtol=0, atol=0)


class _NoScalarReads(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func != torch.ops.aten._local_scalar_dense.default, "state position was read on the host"
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("tensor_position", [False, True])
def test_ple_slot_reads_current_and_writes_next(batch: int, tensor_position: bool) -> None:
    config, ple = _config(), _ple(_config())
    initial = _streams(config, batch, 9).transpose(1, 2).contiguous()
    visible = torch.full_like(initial, 1000)
    state = PLEState(visible)
    state.state_slots = slot_layout({"conv_state": [initial.clone(), torch.zeros_like(initial), torch.zeros_like(initial)]})
    state.state_slot = torch.tensor(0) if tensor_position else 0
    first = _streams(config, batch, 1)
    expected_first = _conv_reference(ple, torch.cat((initial.transpose(1, 2), first), dim=1))[:, -1:]
    expected_carry = torch.cat((initial, first.transpose(1, 2)), dim=-1)[..., -9:]
    with _NoScalarReads():
        actual = ple._short_conv(first, use_cache=True, state_cache=state)
        repeated = ple._short_conv(first, use_cache=True, state_cache=state)
    torch.testing.assert_close(actual, expected_first, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(actual, repeated, rtol=0, atol=0)
    stack = state.state_slots["conv_state"].stack
    torch.testing.assert_close(stack[0], initial, rtol=0, atol=0)
    torch.testing.assert_close(stack[1], expected_carry, rtol=0, atol=0)
    if tensor_position:
        state.state_slot.fill_(1)
    else:
        state.state_slot = 1
    second = first * -0.4
    expected_second = _conv_reference(ple, torch.cat((expected_carry.transpose(1, 2), second), dim=1))[:, -1:]
    with _NoScalarReads():
        actual_second = ple._short_conv(second, use_cache=True, state_cache=state)
    torch.testing.assert_close(actual_second, expected_second, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(stack[2], torch.cat((expected_carry, second.transpose(1, 2)), dim=-1)[..., -9:], rtol=0, atol=0)
    torch.testing.assert_close(state.conv_state, visible, rtol=0, atol=0)
    assert torch.all(state.conv_state == 1000) and ple.conv_state is None


def test_snapshot_restore_tracks_ple_and_history_in_place() -> None:
    layer = SimpleNamespace(conv_state=torch.arange(6.0).reshape(1, 2, 3), recurrent_state=torch.arange(4.0).reshape(1, 2, 2), initialized=False, length=5)
    ple = PLEState(torch.arange(72.0).reshape(1, 8, 9))
    cache = Qwen4ExpCache(layers=[layer, None], ple_token_history=torch.tensor([[2, 3]]), length=7, ple_states={5: ple})
    snapshot = snapshot_cache(None, cache)
    assert snapshot[0] is None and len(snapshot[1]) == 4
    assert snapshot[1][1] == 7 and set(snapshot[1][3]) == {5}
    pointers = (layer.conv_state.data_ptr(), layer.recurrent_state.data_ptr(), ple.conv_state.data_ptr(), cache.ple_token_history.data_ptr())
    layer.conv_state.fill_(99)
    layer.recurrent_state.fill_(99)
    layer.initialized, layer.length, cache.length = True, 99, 99
    ple.conv_state.fill_(99)
    cache.ple_token_history.fill_(99)
    restore_cache(snapshot, None, cache)
    torch.testing.assert_close(layer.conv_state, torch.arange(6.0).reshape(1, 2, 3), rtol=0, atol=0)
    torch.testing.assert_close(layer.recurrent_state, torch.arange(4.0).reshape(1, 2, 2), rtol=0, atol=0)
    torch.testing.assert_close(ple.conv_state, torch.arange(72.0).reshape(1, 8, 9), rtol=0, atol=0)
    torch.testing.assert_close(cache.ple_token_history, torch.tensor([[2, 3]]), rtol=0, atol=0)
    assert not layer.initialized and layer.length == 5 and cache.length == 7
    assert pointers == (layer.conv_state.data_ptr(), layer.recurrent_state.data_ptr(), ple.conv_state.data_ptr(), cache.ple_token_history.data_ptr())
    assert ple.conv_state.data_ptr() != snapshot[1][3][5]["conv_state"].data_ptr()


def test_snapshot_restore_restores_none_state_and_history() -> None:
    layer = SimpleNamespace(conv_state=None, recurrent_state=None, initialized=False)
    cache = Qwen4ExpCache(layers=[layer], ple_states={0: PLEState()})
    snapshot = snapshot_cache(cache)
    layer.conv_state, layer.recurrent_state = torch.ones(1), torch.ones(1)
    cache.ple_states[0].conv_state = torch.ones(1, 8, 9)
    cache.ple_token_history = torch.tensor([[3, 4]])
    restore_cache(snapshot, cache)
    assert layer.conv_state is None and layer.recurrent_state is None
    assert cache.ple_states[0].conv_state is None and cache.ple_token_history is None


def test_legacy_three_field_snapshots_remain_compatible() -> None:
    layer = SimpleNamespace(conv_state=torch.ones(1, 1, 2), initialized=False)
    cache = Qwen4ExpCache(layers=[layer], length=4, ple_states={0: PLEState(torch.ones(1, 8, 9))})
    legacy = (snapshot_cache(cache)[0][:3],)
    layer.conv_state.fill_(9)
    cache.length, cache.ple_token_history = 9, torch.tensor([[3, 4]])
    cache.ple_states[0].conv_state.fill_(9)
    restore_cache(legacy, cache)
    assert cache.length == 4 and cache.ple_token_history is None
    assert torch.all(layer.conv_state == 1) and torch.all(cache.ple_states[0].conv_state == 9)


@pytest.mark.parametrize("owner", ["layer", "ple"])
def test_snapshot_rejects_slot_backed_layer_and_ple_owners(owner: str) -> None:
    state = PLEState(torch.zeros(1, 8, 9))
    state.state_slots = slot_layout({"conv_state": [state.conv_state.clone(), state.conv_state.clone()]})
    cache = Qwen4ExpCache(layers=[state] if owner == "layer" else [], ple_states={0: state} if owner == "ple" else {})
    with pytest.raises(RuntimeError, match="state_slots"):
        snapshot_cache(cache)


def test_snapshot_can_recreate_missing_ple_owner() -> None:
    cache = Qwen4ExpCache(ple_states={3: PLEState(torch.ones(1, 8, 9))})
    snapshot = snapshot_cache(cache)
    cache.ple_states.clear()
    restore_cache(snapshot, cache)
    assert set(cache.ple_states) == {3} and torch.all(cache.ple_states[3].conv_state == 1)


def test_uncached_model_forward_is_stateless() -> None:
    model, tokens = _model(), _tokens(2, 7)
    expected, _ = _model_reference(model, tokens)
    actual = model.forward(tokens, cache=None)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    assert all(layer.ple.conv_state is None for layer in model.layers)


def test_outer_schedule_accepts_cache_none() -> None:
    model, tokens = _model(), _tokens(1, 3)
    hidden = model.embed_tokens(tokens).repeat(1, 1, model.config.hc_count)
    cos, sin = model._rope_cache(tokens.shape[1], 1)
    actual = model._forward_outer_phased(hidden, list(enumerate(model.layers)), None, cos, sin, 0, _history(model.config, tokens), lambda label: model_module._noop_context())
    expected, _ = _model_reference(model, tokens)
    torch.testing.assert_close(F.linear(model.final_mixer(actual), model.lm_head), expected, rtol=2e-5, atol=2e-6)
    assert all(layer.ple.conv_state is None for layer in model.layers)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("seq_len", [1, 3])
def test_by_phase_is_rejected_before_mutating_cache(cached: bool, seq_len: int, monkeypatch: pytest.MonkeyPatch) -> None:
    model = _model()
    cache = model.make_cache(1, 16) if cached else None
    monkeypatch.setattr(model_module, "_SCHEDULE", "by_phase")
    with pytest.raises(RuntimeError, match="by_phase.*layer dependencies"):
        model.forward(_tokens(1, seq_len), cache=cache, past_len=0)
    if cache is not None:
        assert cache.length == 0 and torch.all(cache.ple_token_history == 0)
        assert all(torch.count_nonzero(state.conv_state) == 0 for state in cache.ple_states.values())


def test_private_cross_layer_phase_entry_is_rejected() -> None:
    model = _model()
    with pytest.raises(RuntimeError, match="by_phase.*layer dependencies"):
        model._forward_phased(None, [], None, None, None, 0, None, None)



def test_cpu_ids_do_not_allocate_pinned_transfer_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    model, tokens = _model(), _tokens(2, 3)
    empty = torch.empty
    def no_pinned_allocation(*args, **kwargs):
        assert not kwargs.get("pin_memory", False), "CPU ids should not allocate pinned storage"
        return empty(*args, **kwargs)
    monkeypatch.setattr(torch, "empty", no_pinned_allocation)
    actual = model._pinned_ids(tokens)
    assert actual.device.type == "cpu" and actual.data_ptr() == tokens.data_ptr()
    torch.testing.assert_close(actual, tokens, rtol=0, atol=0)


def test_ple_history_remains_cpu_under_non_cpu_factory_default() -> None:
    model = _model()
    # Meta is an allocation-free device, not an accelerator execution path.
    with torch.device("meta"):
        cache = model.make_cache(1, 16)
    assert cache.ple_token_history.device.type == "cpu"
    assert all(state.conv_state.device.type == "cpu" for state in cache.ple_states.values())
    assert torch.all(cache.ple_token_history == model.config.primary_eos_token_id)
