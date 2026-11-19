"""Position-sensitive QSA regressions with an independent per-query oracle."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from relicllm.models.qwen4_exp.attention import QSAAttention, QSAAttentionCache, QSAIndexer
from relicllm.models.qwen4_exp.config import Qwen4ExpTextConfig


@pytest.fixture(autouse=True)
def cpu_qsa_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "POCKETLLM_QWEN4_INDEXER_BUDGET", "POCKETLLM_QWEN4_QSA_FIXED_WIDTH",
        "POCKETLLM_QWEN4_HOST_KV", "POCKETLLM_QWEN4_ATTN_PHASE_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_CUDA", "0")
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_FALLBACK_ROWS", "2")


def _config(budget: int = 8) -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        hidden_size=16, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, partial_rotary_factor=0.5, indexer_n_heads=2,
        indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=budget,
        indexer_compress_ratio=4,
    )


def _weights(config: Qwen4ExpTextConfig) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20261110)
    def rand(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator) * 0.1
    hidden, dim = config.hidden_size, config.head_dim
    qk_proj = rand(24, hidden) * 3
    # Opposite query heads make ReLU-summed scores positive and cutoff ties rare.
    qk_proj[8:16] = -qk_proj[:8]
    return {
        "index_qk_proj": qk_proj,
        "q_layernorm": rand(8), "k_layernorm": rand(8),
        "q_proj": rand(config.num_attention_heads * dim * 2, hidden),
        "k_proj": rand(config.num_key_value_heads * dim, hidden),
        "v_proj": rand(config.num_key_value_heads * dim, hidden),
        "o_proj": rand(hidden, config.num_attention_heads * dim),
        "q_norm": rand(dim), "k_norm": rand(dim),
    }


def _hidden(batch: int, length: int) -> torch.Tensor:
    return torch.randn(batch, length, 16, generator=torch.Generator().manual_seed(914677))


def _rope(length: int, batch: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    # Distinct frequencies and batch phases exercise partial, non-identity RoPE.
    frequency = torch.tensor([0.71, 0.19])
    batch_ids = torch.arange(batch).view(batch, 1, 1)
    positions = torch.arange(length).view(1, length, 1)
    phase = positions * frequency * (1 + 0.17 * batch_ids) + 0.23 * batch_ids
    phase = torch.cat((phase, phase), dim=-1)
    return phase.cos(), phase.sin()


def _rms_reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normalized = x.float() / torch.sqrt(x.float().square().mean(dim=-1, keepdim=True) + eps)
    return (normalized * (1 + weight.float())).to(x.dtype)


def _rotate_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Apply the two-dimensional rotation directly, without the production helper.
    rotary_dim = cos.shape[-1]
    half = rotary_dim // 2
    first, second = x[..., :half], x[..., half:rotary_dim]
    return torch.cat((
        first * cos[..., :half] - second * sin[..., :half],
        second * cos[..., half:] + first * sin[..., half:],
        x[..., rotary_dim:],
    ), dim=-1)


def _raw_keys(hidden: torch.Tensor, weights: dict[str, torch.Tensor]) -> torch.Tensor:
    return F.linear(hidden, weights["index_qk_proj"])[..., 16:]


def _selection_reference(
    hidden: torch.Tensor, raw_keys: torch.Tensor,
    weights: dict[str, torch.Tensor], config: Qwen4ExpTextConfig,
    cos: torch.Tensor, sin: torch.Tensor, past_len: int,
    *, query_cos: torch.Tensor | None = None, query_sin: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score only complete *visible* blocks, independently for each query."""
    batch, seq_len, _ = hidden.shape
    dim, ratio, budget = 8, 4, config.indexer_budget
    width = raw_keys.shape[1] if raw_keys.shape[1] <= budget else min(raw_keys.shape[1], budget + ratio - 1)
    result = torch.full((batch, seq_len, width), -1, dtype=torch.int32)
    queries = F.linear(hidden, weights["index_qk_proj"])[..., :16].reshape(batch, seq_len, 2, dim)
    queries = _rms_reference(queries, weights["q_layernorm"], config.rms_norm_eps)
    for b in range(batch):
        table_batch = b if cos.shape[0] > 1 else 0
        for t in range(seq_len):
            visible = past_len + t + 1
            if raw_keys.shape[1] <= budget:
                picked = torch.arange(min(visible, raw_keys.shape[1]))
            else:
                count = visible // ratio
                pooled = raw_keys[b, :count * ratio].reshape(count, ratio, dim).float().mean(dim=1).to(raw_keys.dtype)
                pooled = _rms_reference(pooled, weights["k_layernorm"], config.rms_norm_eps)
                starts = torch.arange(count) * ratio
                keys = _rotate_reference(pooled, cos[table_batch, starts], sin[table_batch, starts])
                if query_cos is None:
                    q_cos, q_sin = cos[table_batch, past_len + t], sin[table_batch, past_len + t]
                else:
                    query_batch = b if query_cos.shape[0] > 1 else 0
                    q_cos, q_sin = query_cos[query_batch, t], query_sin[query_batch, t]
                query = _rotate_reference(queries[b, t], q_cos, q_sin)
                scores = (query.float().unsqueeze(1) * keys.float().unsqueeze(0)).sum(dim=-1).relu().sum(dim=0) / math.sqrt(dim)
                topk = min(budget // ratio, count)
                blocks = scores.topk(topk).indices
                if count > topk:
                    # A strict cutoff makes the expected set independent of tie ordering.
                    ordered = scores.sort(descending=True).values
                    assert ordered[topk - 1] - ordered[topk] > 1e-6
                picked = (blocks[:, None] * ratio + torch.arange(ratio)).flatten()
                picked = torch.cat((picked, torch.arange(count * ratio, visible)))
            result[b, t, :picked.numel()] = picked.to(torch.int32)
    return result.sort(dim=-1).values


def _assert_selection(actual: torch.Tensor, expected: torch.Tensor, past_len: int) -> None:
    assert actual.dtype == torch.int32
    torch.testing.assert_close(actual.sort(dim=-1).values, expected, rtol=0, atol=0)
    for b in range(actual.shape[0]):
        for t in range(actual.shape[1]):
            live = actual[b, t][actual[b, t] >= 0]
            assert torch.all(live < past_len + t + 1)
            assert live.unique().numel() == live.numel()


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("past_len,seq_len", [(0, 19), (10, 3), (11, 1), (12, 5), (15, 3)])
@pytest.mark.parametrize("oversized_table", [False, True])
def test_indexer_uses_absolute_query_rows_without_cache(batch: int, past_len: int, seq_len: int, oversized_table: bool) -> None:
    config, weights = _config(), _weights(_config())
    hidden = _hidden(batch, past_len + seq_len)
    raw = _raw_keys(hidden, weights)
    cos, sin = _rope(64 if oversized_table else past_len + seq_len, batch)
    current = hidden[:, past_len:]
    expected = _selection_reference(current, raw, weights, config, cos, sin, past_len)
    actual = QSAIndexer(config, weights)(
        current, cos, sin, None, past_len=past_len,
        raw_keys_override=raw[:, :past_len] if past_len else None,
    )
    _assert_selection(actual, expected, past_len)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("past_len", [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16])
def test_fixed_width_integer_and_tensor_positions_hide_future_blocks(batch: int, past_len: int) -> None:
    config, weights = _config(), _weights(_config())
    hidden = _hidden(batch, 48)
    raw = _raw_keys(hidden, weights)
    cos, sin = _rope(64)  # A shared table must broadcast over batch-two queries.
    current = hidden[:, past_len:past_len + 1]
    expected = _selection_reference(current, raw, weights, config, cos, sin, past_len)
    selections = []
    for position in (past_len, torch.tensor(past_len)):
        cache = SimpleNamespace(index_k=raw.clone(), length=past_len, kv_width=48)
        selections.append(QSAIndexer(config, weights)(current, cos, sin, cache, past_len=position))
    for selected in selections:
        assert selected.shape == (batch, 1, 11)
        _assert_selection(selected, expected, past_len)
    torch.testing.assert_close(selections[0], selections[1], rtol=0, atol=0)
    if past_len < config.indexer_budget:
        live = selections[1][0, 0][selections[1][0, 0] >= 0]
        assert set(live.tolist()) == set(range(past_len + 1))


def test_explicit_query_rope_is_separate_from_block_table() -> None:
    config, weights = _config(), _weights(_config())
    past_len, seq_len = 14, 3
    hidden = _hidden(2, past_len + seq_len)
    current, raw = hidden[:, past_len:], _raw_keys(hidden, weights)
    cos, sin = _rope(64, 2)
    query_cos, query_sin = cos[:, 16:19], sin[:, 16:19]
    expected = _selection_reference(current, raw, weights, config, cos, sin, past_len, query_cos=query_cos, query_sin=query_sin)
    ordinary = _selection_reference(current, raw, weights, config, cos, sin, past_len)
    assert not torch.equal(expected, ordinary), "the fixture must detect ignored explicit query rows"
    actual = QSAIndexer(config, weights)(
        current, cos, sin, None, past_len=torch.tensor(past_len),
        raw_keys_override=raw[:, :past_len], query_cos=query_cos, query_sin=query_sin,
    )
    _assert_selection(actual, expected, past_len)


def test_default_2048_budget_uses_current_query_at_block_boundary() -> None:
    config = _config(budget=Qwen4ExpTextConfig().indexer_budget)
    assert config.indexer_budget == 2048
    weights = _weights(config)
    # Opposite query heads avoid zero-score cutoff ties near the large budget.
    weights["index_qk_proj"][8:16] = -weights["index_qk_proj"][:8]
    past_len, seq_len = 2054, 3
    hidden = _hidden(1, past_len + seq_len)
    current, raw = hidden[:, past_len:], _raw_keys(hidden, weights)
    cos, sin = _rope(4096)
    expected = _selection_reference(current, raw, weights, config, cos, sin, past_len)
    indexer = QSAIndexer(config, weights)
    assert indexer.budget == 2048
    actual = indexer(current, cos, sin, None, past_len=past_len, raw_keys_override=raw[:, :past_len])
    _assert_selection(actual, expected, past_len)
    assert [int((row >= 0).sum()) for row in actual[0]] == [2051, 2048, 2049]


def _kv_reference(
    hidden: torch.Tensor, weights: dict[str, torch.Tensor], config: Qwen4ExpTextConfig,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, seq_len, _ = hidden.shape
    keys = F.linear(hidden, weights["k_proj"]).reshape(batch, seq_len, config.num_key_value_heads, config.head_dim)
    keys = _rms_reference(keys, weights["k_norm"], config.rms_norm_eps)
    keys = _rotate_reference(keys, cos.unsqueeze(2), sin.unsqueeze(2)).transpose(1, 2)
    values = F.linear(hidden, weights["v_proj"]).reshape(batch, seq_len, config.num_key_value_heads, config.head_dim).transpose(1, 2)
    return keys, values


def _attention_reference(
    hidden: torch.Tensor, weights: dict[str, torch.Tensor], config: Qwen4ExpTextConfig,
    cos: torch.Tensor, sin: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    batch, seq_len, _ = hidden.shape
    packed = F.linear(hidden, weights["q_proj"]).reshape(batch, seq_len, config.num_attention_heads, 2 * config.head_dim)
    query, gate = packed.chunk(2, dim=-1)
    query = _rms_reference(query, weights["q_norm"], config.rms_norm_eps)
    query = _rotate_reference(query, cos.unsqueeze(2), sin.unsqueeze(2))
    output = torch.empty_like(query)
    groups = config.num_attention_heads // config.num_key_value_heads
    for b in range(batch):
        for t in range(seq_len):
            live = selected[b, t][selected[b, t] >= 0].long()
            for h in range(config.num_attention_heads):
                kv_head = h // groups
                scores = (keys[b, kv_head, live] * query[b, t, h]).sum(dim=-1) / math.sqrt(config.head_dim)
                output[b, t, h] = (scores.softmax(dim=-1)[:, None] * values[b, kv_head, live]).sum(dim=0)
    return F.linear((output * gate.sigmoid()).flatten(2), weights["o_proj"])


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("tensor_position", [False, True])
@pytest.mark.parametrize("query_shift", [0, 2])
def test_attention_forwards_current_rope_with_fixed_oversized_table(batch: int, tensor_position: bool, query_shift: int) -> None:
    config, weights = _config(), _weights(_config())
    past_len, seq_len, width = 14, 3, 48
    hidden = _hidden(batch, past_len + seq_len)
    current, raw = hidden[:, past_len:], _raw_keys(hidden, weights)
    cos, sin = _rope(64, batch)
    query_cos = cos[:, past_len + query_shift:past_len + query_shift + seq_len]
    query_sin = sin[:, past_len + query_shift:past_len + query_shift + seq_len]
    keys, values = _kv_reference(hidden, weights, config, cos[:, :hidden.shape[1]], sin[:, :hidden.shape[1]])
    current_keys, current_values = _kv_reference(current, weights, config, query_cos, query_sin)
    keys[:, :, past_len:], values[:, :, past_len:] = current_keys, current_values
    expected_selected = _selection_reference(current, raw, weights, config, cos, sin, past_len, query_cos=query_cos, query_sin=query_sin)
    expected = _attention_reference(current, weights, config, query_cos, query_sin, keys, values, expected_selected)
    cache = QSAAttentionCache(config, batch, width, torch.device("cpu"), torch.float32)
    cache.index_k[:, :past_len] = raw[:, :past_len]
    cache.k[:, :, :past_len], cache.v[:, :, :past_len] = keys[:, :, :past_len], values[:, :, :past_len]
    cache.cos, cache.sin, cache.kv_width, cache.length = cos, sin, width, past_len
    if tensor_position:
        caller_cos, caller_sin, position = query_cos, query_sin, torch.tensor(past_len)
    else:
        caller_cos = torch.cat((cos[:, :past_len], query_cos), dim=1)
        caller_sin = torch.cat((sin[:, :past_len], query_sin), dim=1)
        position = past_len
    actual = QSAAttention(config, weights)(current, caller_cos, caller_sin, cache, past_len=position)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    # The independent norm divides by sqrt instead of multiplying by rsqrt.
    torch.testing.assert_close(cache.k[:, :, past_len:past_len + seq_len], current_keys, rtol=1e-6, atol=2e-7)
    torch.testing.assert_close(cache.v[:, :, past_len:past_len + seq_len], current_values, rtol=0, atol=0)


@pytest.mark.parametrize("cache_kind", ["none", "without_rope_table"])
def test_attention_without_shared_rope_table_remains_supported(cache_kind: str) -> None:
    config, weights = _config(), _weights(_config())
    past_len = 0 if cache_kind == "none" else 10
    seq_len = 19 if cache_kind == "none" else 3
    hidden = _hidden(2, past_len + seq_len)
    current, raw = hidden[:, past_len:], _raw_keys(hidden, weights)
    cos, sin = _rope(hidden.shape[1], 2)
    keys, values = _kv_reference(hidden, weights, config, cos, sin)
    selected = _selection_reference(current, raw, weights, config, cos, sin, past_len)
    expected = _attention_reference(current, weights, config, cos[:, past_len:], sin[:, past_len:], keys, values, selected)
    cache = None
    if cache_kind != "none":
        cache = QSAAttentionCache(config, 2, 32, torch.device("cpu"), torch.float32)
        cache.index_k[:, :past_len] = raw[:, :past_len]
        cache.k[:, :, :past_len], cache.v[:, :, :past_len] = keys[:, :, :past_len], values[:, :, :past_len]
        cache.length = past_len
    actual = QSAAttention(config, weights)(current, cos, sin, cache, past_len=past_len)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


class _NoScalarReads(TorchDispatchMode):
    """Reject tensor-to-host scalar extraction on the fixed-width graph path."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        assert func != torch.ops.aten._local_scalar_dense.default, "position was read on the host"
        return func(*args, **(kwargs or {}))


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("explicit_rows", [False, True])
def test_tensor_position_replays_fixed_shapes_without_host_reads(batch: int, explicit_rows: bool) -> None:
    config, weights = _config(), _weights(_config())
    hidden = _hidden(batch, 48)
    raw = _raw_keys(hidden, weights)
    cos, sin = _rope(64, batch)
    cache = SimpleNamespace(index_k=raw.clone(), length=0, kv_width=48)
    indexer = QSAIndexer(config, weights)
    position = torch.tensor(0)
    for past_len in (10, 11, 12, 14, 15, 16):
        position.fill_(past_len)
        current = hidden[:, past_len:past_len + 1]
        expected = _selection_reference(current, raw, weights, config, cos, sin, past_len)
        query_rope = {}
        if explicit_rows:
            query_rope = {
                "query_cos": cos.index_select(1, position.reshape(1)),
                "query_sin": sin.index_select(1, position.reshape(1)),
            }
        with _NoScalarReads():
            actual = indexer(current, cos, sin, cache, past_len=position, **query_rope)
        assert actual.shape == (batch, 1, 11)
        _assert_selection(actual, expected, past_len)


@pytest.mark.parametrize("provided", ["cos", "sin"])
def test_explicit_query_rope_requires_both_tables(provided: str) -> None:
    config, weights = _config(), _weights(_config())
    cos, sin = _rope(64)
    query_rope = {f"query_{provided}": (cos if provided == "cos" else sin)[:, 14:15]}
    with pytest.raises(ValueError, match="query_cos and query_sin must be supplied together"):
        QSAIndexer(config, weights)(_hidden(1, 1), cos, sin, None, past_len=14, **query_rope)
