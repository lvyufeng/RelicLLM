"""CPU recompute oracles for the opt-in, request-owned QSA pooled-key cache.

These tests prove cache arithmetic and position/write invariants, not CUDA graph
or checkpoint quality. The default remains off until the device-side oracle runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from relicllm.models.qwen4_exp.attention import QSAAttention, QSAAttentionCache, QSAIndexer
from relicllm.models.qwen4_exp.config import Qwen4ExpTextConfig
from relicllm.models.qwen4_exp.layers import RMSNorm, apply_rotary_pos_emb


@pytest.fixture(autouse=True)
def cpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "POCKETLLM_QWEN4_INDEXER_BUDGET", "POCKETLLM_QWEN4_QSA_FIXED_WIDTH",
        "POCKETLLM_QWEN4_HOST_KV", "POCKETLLM_QWEN4_ATTN_PHASE_PROFILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("POCKETLLM_QWEN4_INCREMENTAL_INDEXER", "1")
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_CUDA", "0")
    monkeypatch.setenv("POCKETLLM_QWEN4_QSA_FALLBACK_ROWS", "2")


def _config(budget: int = 8) -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        hidden_size=16, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, partial_rotary_factor=0.5, indexer_n_heads=2,
        indexer_kv_heads=1, indexer_head_dim=8, indexer_budget=budget,
        indexer_compress_ratio=4,
    )


def _weights(config: Qwen4ExpTextConfig, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20261110)

    def rand(*shape: int) -> torch.Tensor:
        return (torch.randn(*shape, generator=generator) * 0.1).to(dtype)

    dim, hidden = config.head_dim, config.hidden_size
    projection = rand(24, hidden) * 3
    # Opposite heads keep the ReLU-summed score away from systematic zero ties.
    projection[8:16] = -projection[:8]
    return {
        "index_qk_proj": projection,
        "q_layernorm": rand(8), "k_layernorm": rand(8),
        "q_proj": rand(config.num_attention_heads * dim * 2, hidden),
        "k_proj": rand(config.num_key_value_heads * dim, hidden),
        "v_proj": rand(config.num_key_value_heads * dim, hidden),
        "o_proj": rand(hidden, config.num_attention_heads * dim),
        "q_norm": rand(dim), "k_norm": rand(dim),
    }


def _hidden(batch: int, length: int, dtype: torch.dtype, seed: int = 914677) -> torch.Tensor:
    return torch.randn(
        batch, length, 16, generator=torch.Generator().manual_seed(seed)
    ).to(dtype)


def _rope(length: int, batch: int = 1, dtype: torch.dtype = torch.float32):
    frequencies = torch.tensor([0.71, 0.19])
    batch_ids = torch.arange(batch).view(batch, 1, 1)
    positions = torch.arange(length).view(1, length, 1)
    phase = positions * frequencies * (1 + 0.17 * batch_ids) + 0.23 * batch_ids
    phase = torch.cat((phase, phase), dim=-1)
    return phase.cos().to(dtype), phase.sin().to(dtype)


class _CountingNorm:
    def __init__(self, norm: RMSNorm) -> None:
        self.norm = norm
        self.groups: list[int] = []

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.groups.append(x.shape[1])
        return self.norm(x)


class _NoHostOrDynamicOps(TorchDispatchMode):
    """A CPU structural guard, not a substitute for an actual CUDA capture."""

    def __init__(self, pool_shape: tuple | None = None) -> None:
        super().__init__()
        self.pool_shape = pool_shape
        self.trace: list[tuple] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name.split("::")[-1]
        assert name not in {
            "_local_scalar_dense", "nonzero", "masked_select", "_unique2",
            "unique_dim", "unique_consecutive",
        }, f"host read or dynamic output in replay path: {func}"
        if name == "clone" and self.pool_shape is not None:
            # The established selection expansion reshapes a broadcast bool
            # mask, which legitimately clones a small fixed-size mask. Forbid
            # the large floating key pool, not that unrelated existing op.
            assert not (args[0].dtype == torch.float32 and tuple(args[0].shape) == self.pool_shape), "full key pool cloned"
        output = func(*args, **(kwargs or {}))
        tensors = [value for value in tree_flatten(output)[0] if isinstance(value, torch.Tensor)]
        self.trace.append((str(func), tuple((tuple(x.shape), x.dtype) for x in tensors)))
        return output


class _Pair:
    """Run identical writes against enabled caching and the default-off recompute path."""

    def __init__(
        self, monkeypatch: pytest.MonkeyPatch, *, batch: int = 1,
        dtype: torch.dtype = torch.float32, capacity: int = 67,
        fixed: bool = False, budget: int = 8, table_batch: int | None = None,
        table_length: int | None = None,
    ) -> None:
        self.monkeypatch = monkeypatch
        self.config = _config(budget)
        self.weights = _weights(self.config, dtype)
        self.hidden = _hidden(batch, capacity, dtype)
        self.fast = QSAIndexer(self.config, self.weights)
        self.old = QSAIndexer(self.config, self.weights)
        self.fast_norm = _CountingNorm(self.fast.k_layernorm)
        self.old_norm = _CountingNorm(self.old.k_layernorm)
        self.fast.k_layernorm = self.fast_norm
        self.old.k_layernorm = self.old_norm
        self.fast_cache = QSAAttentionCache(self.config, batch, capacity, torch.device("cpu"), dtype)
        self.old_cache = QSAAttentionCache(self.config, batch, capacity, torch.device("cpu"), dtype)
        if fixed:
            self.fast_cache.kv_width = self.old_cache.kv_width = capacity
        self.cos, self.sin = _rope(table_length or capacity + 29, table_batch or batch, dtype)

    def seed_arena(self) -> None:
        raw = F.linear(self.hidden, self.weights["index_qk_proj"])[..., 16:]
        self.fast_cache.index_k.copy_(raw)
        self.old_cache.index_k.copy_(raw)

    def _assert_complete_keys(self, past_len: int, seq_len: int) -> None:
        pool = self.fast_cache.indexer_pool
        width = getattr(self.fast_cache, "kv_width", 0) or past_len + seq_len
        width = min(width, self.fast_cache.index_k.shape[1])
        count = min((past_len + seq_len) // 4, width // 4)
        if pool is not None and pool.completed is not None:
            count = min(count, pool.completed)
        if pool is None or width <= self.config.indexer_budget or count == 0:
            return
        # Spell out the old full-recompute arithmetic independently of the new
        # helper, including both low-precision casts and the partial-RoPE phase.
        raw = self.fast_cache.index_k[:, :count * 4].reshape(-1, count, 4, 8)
        pooled = raw.float().mean(dim=2).to(raw.dtype)
        pooled = RMSNorm(self.weights["k_layernorm"], self.config.rms_norm_eps)(pooled)
        starts = torch.arange(count) * 4
        expected = apply_rotary_pos_emb(
            pooled.unsqueeze(2), self.cos[:, starts], self.sin[:, starts], unsqueeze_dim=2
        ).squeeze(2).float()
        assert pool.keys.dtype == torch.float32
        torch.testing.assert_close(pool.keys[:, :count], expected, rtol=0, atol=0)

    def step(
        self, past_len: int, seq_len: int = 1, *, position: torch.Tensor | None = None,
        current: torch.Tensor | None = None, query_shift: int | None = None,
        raw_keys_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int, list[tuple]]:
        current = self.hidden[:, past_len:past_len + seq_len] if current is None else current
        kwargs = {"past_len": position if position is not None else past_len}
        if query_shift is not None:
            begin = past_len + query_shift
            kwargs.update(query_cos=self.cos[:, begin:begin + seq_len], query_sin=self.sin[:, begin:begin + seq_len])
        if raw_keys_override is not None:
            kwargs["raw_keys_override"] = raw_keys_override
        before = len(self.fast_norm.groups)
        arena = self.fast_cache.index_k
        guard = _NoHostOrDynamicOps((arena.shape[0], arena.shape[1] // 4, arena.shape[2]))
        with self.monkeypatch.context() as env:
            env.setenv("POCKETLLM_QWEN4_INCREMENTAL_INDEXER", "1")
            if position is None:
                actual = self.fast(current, self.cos, self.sin, self.fast_cache, **kwargs)
            else:
                with guard:
                    actual = self.fast(current, self.cos, self.sin, self.fast_cache, **kwargs)
        with self.monkeypatch.context() as env:
            env.setenv("POCKETLLM_QWEN4_INCREMENTAL_INDEXER", "0")
            if position is None:
                expected = self.old(current, self.cos, self.sin, self.old_cache, **kwargs)
            else:
                with _NoHostOrDynamicOps():
                    expected = self.old(current, self.cos, self.sin, self.old_cache, **kwargs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(self.fast_cache.index_k, self.old_cache.index_k, rtol=0, atol=0, equal_nan=True)
        assert self.old_cache.indexer_pool is None
        for row in range(seq_len):
            live = actual[:, row][actual[:, row] >= 0]
            assert torch.all(live < past_len + row + 1)
            assert torch.all(live < self.fast_cache.index_k.shape[1])
        self._assert_complete_keys(past_len, seq_len)
        self.fast_cache.length = self.old_cache.length = past_len + seq_len
        work = sum(self.fast_norm.groups[before:])
        return actual, work, guard.trace


def test_explicit_zero_keeps_native_2048_budget_without_allocating_pool(monkeypatch) -> None:
    """`0` is the opt-out, and it must still answer exactly what the pool answers.

    The default is on -- the pool is what stops a step re-pooling the whole history to answer
    about one position -- so this test names the opt-out rather than the absence of a setting.
    """
    config = _config(Qwen4ExpTextConfig().indexer_budget)
    weights = _weights(config, torch.float32)
    indexer = QSAIndexer(config, weights)
    cache = QSAAttentionCache(config, 1, 2060, torch.device("cpu"), torch.float32)
    hidden = _hidden(1, 2055, torch.float32)
    cos, sin = _rope(4096)
    monkeypatch.setenv("POCKETLLM_QWEN4_INCREMENTAL_INDEXER", "0")
    assert indexer.budget == 2048 and indexer.block_topk == 512
    selected = indexer(hidden, cos, sin, cache, past_len=0)
    assert selected.shape == (1, 2055, 2051)
    assert cache.indexer_pool is None
    assert not any("pool" in name or "completed" in name for name in indexer.__dict__)
    # Twelve QSA layers each store 65536 x 128 float32 values at the native
    # 256K geometry, independently of the number of tensor-parallel query heads.
    native = Qwen4ExpTextConfig()
    bytes_per_rank = (native.num_hidden_layers // native.full_attention_interval
                      * (262144 // native.indexer_compress_ratio) * native.indexer_head_dim * 4)
    assert bytes_per_rank / 2**30 == 0.375


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("fixed", [False, True])
def test_integer_sequence_matches_recompute_and_only_normalizes_new_groups(monkeypatch, batch, dtype, fixed) -> None:
    pair = _Pair(monkeypatch, batch=batch, dtype=dtype, fixed=fixed)
    pointer = None
    for past_len in range(36):
        _, work, _ = pair.step(past_len)
        pool = pair.fast_cache.indexer_pool
        if pointer is None:
            pointer = pool.keys.data_ptr()
        assert pool.keys.data_ptr() == pointer
        assert pool.keys.shape == (batch, 67 // 4, 8)
        if past_len >= 9:
            assert work == int((past_len + 1) % 4 == 0)
            assert pair.old_norm.groups[-1] == (67 // 4 if fixed else (past_len + 1) // 4)


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("fixed", [False, True])
def test_prefill_chunks_cross_partial_r4_groups_without_repooling_prefix(monkeypatch, batch, dtype, fixed) -> None:
    pair = _Pair(monkeypatch, batch=batch, dtype=dtype, fixed=fixed)
    past_len, completed = 0, 0
    for size in (3, 2, 7, 1, 3, 8, 5, 6):
        _, work, _ = pair.step(past_len, size)
        raw_len = 67 if fixed else past_len + size
        if raw_len <= 8:
            assert work == 0
            completed = 0
        else:
            newly_complete = (past_len + size) // 4
            assert work == newly_complete - min(completed, past_len // 4)
            completed = newly_complete
        past_len += size


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("explicit_rows", [False, True])
def test_scalar_tensor_warmup_then_fixed_steady_work_has_no_host_or_dynamic_ops(monkeypatch, batch, dtype, explicit_rows) -> None:
    pair = _Pair(monkeypatch, batch=batch, dtype=dtype, fixed=True, table_batch=1)
    pair.step(0, 19)
    position = torch.tensor(19)
    # Setup may use a fake token even at the final token of an r4 group. The
    # first real step must replace it before that newly complete group is scored.
    _, warmup_work, _ = pair.step(19, position=position, current=pair.hidden[:, 19:20] * -3,
                                 query_shift=0 if explicit_rows else None)
    assert warmup_work == 67 // 4 + 1
    pointer = pair.fast_cache.indexer_pool.keys.data_ptr()
    traces = []
    for past_len in range(19, 36):
        position.fill_(past_len)
        pair.fast_cache.length = pair.old_cache.length = 999
        _, work, trace = pair.step(past_len, position=position, query_shift=0 if explicit_rows else None)
        assert work == 1
        assert pair.old_norm.groups[-1] == 67 // 4
        assert pair.fast_cache.indexer_pool.keys.data_ptr() == pointer
        traces.append(trace)
    assert all(trace == traces[0] for trace in traces[1:])


@pytest.mark.parametrize("seq_len", [2, 3, 4, 5, 8, 9])
@pytest.mark.parametrize("batch", [1, 2])
def test_tensor_multitoken_boundaries_use_fixed_touched_group_count(monkeypatch, seq_len, batch) -> None:
    pair = _Pair(monkeypatch, batch=batch, fixed=True, capacity=47)
    pair.seed_arena()
    position = torch.tensor(11)
    touched = 1 + (seq_len + 4 - 2) // 4
    _, work, _ = pair.step(11, seq_len, position=position)
    assert work == 47 // 4 + touched
    traces = []
    for past_len in (8, 9, 10, 11, 12, 13, 16, 19, 22, 47 - seq_len):
        position.fill_(past_len)
        current = _hidden(batch, seq_len, torch.float32, seed=12300 + past_len)
        _, work, trace = pair.step(past_len, seq_len, position=position, current=current)
        assert work == touched
        traces.append(trace)
    assert all(trace == traces[0] for trace in traces[1:])


@pytest.mark.parametrize("tensor_position", [False, True])
def test_full_key_table_is_independent_of_explicit_shifted_query_rows(monkeypatch, tensor_position) -> None:
    pair = _Pair(monkeypatch, batch=2, fixed=True)
    pair.step(0, 19)
    position = torch.tensor(19) if tensor_position else None
    pair.step(19, 3, position=position, query_shift=0)
    ordinary, _, _ = pair.step(19, 3, position=position, query_shift=0)
    shifted, work, _ = pair.step(19, 3, position=position, query_shift=2)
    assert not torch.equal(ordinary, shifted), "the fixture must detect incorrect query/table coupling"
    assert work == (2 if tensor_position else 1)


@pytest.mark.parametrize("change", ["length", "pointer", "dtype", "batch", "inplace"])
@pytest.mark.parametrize("tensor_position", [False, True])
def test_key_table_changes_rebuild_once_before_steady_steps(monkeypatch, change, tensor_position) -> None:
    pair = _Pair(monkeypatch, batch=2, dtype=torch.bfloat16, fixed=True, table_batch=1)
    pair.step(0, 19)
    position = torch.tensor(19) if tensor_position else None
    pair.step(19, position=position)
    pool_pointer = pair.fast_cache.indexer_pool.keys.data_ptr()
    if change == "length":
        pair.cos, pair.sin = _rope(128, dtype=torch.bfloat16)
    elif change == "pointer":
        pair.cos, pair.sin = pair.cos.clone(), pair.sin.clone()
    elif change == "dtype":
        pair.cos, pair.sin = pair.cos.float(), pair.sin.float()
    elif change == "batch":
        pair.cos, pair.sin = _rope(pair.cos.shape[1], batch=2, dtype=torch.bfloat16)
    else:
        pair.cos.mul_(0.75)
        pair.sin.mul_(0.75)
    if position is not None:
        position.fill_(20)
    _, work, _ = pair.step(20, position=position)
    assert work == (67 // 4 + 1 if tensor_position else 21 // 4)
    assert pair.fast_cache.indexer_pool.keys.data_ptr() == pool_pointer
    if position is not None:
        position.fill_(21)
    _, work, _ = pair.step(21, position=position)
    assert work == (1 if tensor_position else 0)


@pytest.mark.parametrize("change", ["pointer", "capacity", "dtype", "batch", "stride"])
@pytest.mark.parametrize("tensor_position", [False, True])
def test_raw_arena_layout_changes_invalidate_conservatively(monkeypatch, change, tensor_position) -> None:
    pair = _Pair(monkeypatch, fixed=True)
    pair.step(0, 19)
    position = torch.tensor(19) if tensor_position else None
    pair.step(19, position=position)
    old_pointer = pair.fast_cache.indexer_pool.keys.data_ptr()
    for cache in (pair.fast_cache, pair.old_cache):
        raw = cache.index_k
        if change == "pointer":
            cache.index_k = raw.clone()
        elif change == "capacity":
            replacement = torch.zeros(1, 79, 8)
            replacement[:, :67].copy_(raw)
            cache.index_k, cache.kv_width = replacement, 79
        elif change == "dtype":
            cache.index_k = raw.to(torch.bfloat16)
        elif change == "batch":
            cache.index_k = raw.expand(2, -1, -1).clone()
        else:
            replacement = torch.zeros(1, 67, 16)
            replacement[..., ::2].copy_(raw)
            cache.index_k = replacement[..., ::2]
    if change == "batch":
        pair.hidden = _hidden(2, 67, torch.float32)
    elif change == "dtype":
        # The tensor write's source and arena must have matching dtypes, just
        # as on the established index_copy_ path after a model dtype change.
        pair.hidden = pair.hidden.to(torch.bfloat16)
        projection = pair.weights["index_qk_proj"].to(torch.bfloat16)
        pair.weights["index_qk_proj"] = projection
        pair.fast.qk_proj = pair.old.qk_proj = projection
    if position is not None:
        position.fill_(20)
    _, work, _ = pair.step(20, position=position)
    width = pair.fast_cache.index_k.shape[1]
    assert work == (width // 4 + 1 if tensor_position else 21 // 4)
    if change in {"capacity", "batch"}:
        assert pair.fast_cache.indexer_pool.keys.data_ptr() != old_pointer
    if position is not None:
        position.fill_(21)
    _, work, _ = pair.step(21, position=position)
    assert work == (1 if tensor_position else 0)


@pytest.mark.parametrize("tensor_position", [False, True])
def test_padded_arena_with_only_valid_prefix_rope_never_reads_future_table_rows(monkeypatch, tensor_position) -> None:
    pair = _Pair(monkeypatch, batch=2, fixed=True, capacity=67, table_length=21)
    pair.seed_arena()
    position = torch.tensor(18) if tensor_position else None
    pair.step(18, position=position)
    if position is not None:
        position.fill_(19)
    _, work, _ = pair.step(19, position=position)
    assert work == 1
    # Growing/replacing the table in graph preparation may rebuild once. The
    # subsequent captured-shaped steps must not rebuild the entire padded span.
    pair.cos, pair.sin = _rope(96, batch=2)
    if position is not None:
        position.fill_(20)
    _, work, _ = pair.step(20, position=position)
    assert work == (67 // 4 + 1 if tensor_position else 21 // 4)
    if position is not None:
        position.fill_(21)
    _, work, _ = pair.step(21, position=position)
    assert work == (1 if tensor_position else 0)


@pytest.mark.parametrize("fixed", [False, True])
def test_new_request_reuses_inference_arena_and_pool_without_stale_completed_prefix(monkeypatch, fixed) -> None:
    with torch.inference_mode():
        pair = _Pair(monkeypatch, batch=2, dtype=torch.bfloat16, fixed=fixed)
        pair.step(0, 27)
        raw_pointer = pair.fast_cache.index_k.data_ptr()
        pool_pointer = pair.fast_cache.indexer_pool.keys.data_ptr()
        pair.fast_cache.length = pair.old_cache.length = 0
        pair.hidden = _hidden(2, 67, torch.bfloat16, seed=44213)
        past_len = 0
        for size in (3, 4, 6, 5, 7, 8):
            pair.step(past_len, size)
            assert pair.fast_cache.index_k.data_ptr() == raw_pointer
            assert pair.fast_cache.indexer_pool.keys.data_ptr() == pool_pointer
            past_len += size


@pytest.mark.parametrize("rewrite", ["integer", "tensor", "handback"])
def test_speculative_suffix_rewind_and_fake_current_are_overwritten_before_visible(monkeypatch, rewrite) -> None:
    with torch.inference_mode():
        pair = _Pair(monkeypatch, batch=2, dtype=torch.bfloat16, fixed=True)
        pair.step(0, 17)
        position = torch.tensor(17)
        for past_len in range(17, 29):
            position.fill_(past_len)
            pair.step(past_len, position=position, current=pair.hidden[:, past_len:past_len + 1] * -3)
        pointer = pair.fast_cache.indexer_pool.keys.data_ptr()
        # Restore logical lengths only, like graph handback or a speculative
        # accepted-prefix rollback. Neither raw nor pooled suffixes are cloned.
        pair.fast_cache.length = pair.old_cache.length = 17
        for past_len in range(17, 36):
            position.fill_(past_len)
            tensor = rewrite == "tensor" or (rewrite == "handback" and past_len >= 24)
            _, work, _ = pair.step(past_len, position=position if tensor else None)
            assert work == (1 if tensor else int((past_len + 1) % 4 == 0))
            assert pair.fast_cache.indexer_pool.keys.data_ptr() == pointer


@pytest.mark.parametrize("tensor_position", [False, True])
def test_final_incomplete_arena_group_and_raw_shorter_than_r4_fall_back_safely(monkeypatch, tensor_position) -> None:
    pair = _Pair(monkeypatch, fixed=True, capacity=19)
    pair.step(0, 16)
    position = torch.tensor(16) if tensor_position else None
    for past_len in range(16, 19):
        if position is not None:
            position.fill_(past_len)
        selected, _, _ = pair.step(past_len, position=position)
        assert torch.all(selected[selected >= 0] < 19)
        assert pair.fast_cache.indexer_pool.keys.shape[1] == 19 // 4
    short = _Pair(monkeypatch, capacity=3, fixed=True, budget=1)
    for past_len in range(3):
        position = torch.tensor(past_len) if tensor_position else None
        selected, work, _ = short.step(past_len, position=position)
        assert work == 0
        assert selected.shape == (1, 1, 3)
        assert set(selected[0, 0][selected[0, 0] >= 0].tolist()) == set(range(past_len + 1))


@pytest.mark.parametrize("cache_present", [False, True])
@pytest.mark.parametrize("tensor_position", [False, True])
def test_raw_overrides_and_cache_none_use_conservative_full_recompute(monkeypatch, cache_present, tensor_position) -> None:
    pair = _Pair(monkeypatch, fixed=True)
    pair.step(0, 19)
    raw_prefix = pair.fast_cache.index_k[:, :19].clone()
    position = torch.tensor(19) if tensor_position else None
    if cache_present:
        pair.step(19, position=position, raw_keys_override=raw_prefix)
        assert pair.fast_cache.indexer_pool.completed == 0
        assert pair.fast_norm.groups[-1] == 67 // 4
        if position is not None:
            position.fill_(20)
        _, work, _ = pair.step(20, position=position)
        assert work == (67 // 4 + 1 if tensor_position else 21 // 4)
    else:
        outputs = []
        for enabled, indexer in (("1", pair.fast), ("0", pair.old)):
            monkeypatch.setenv("POCKETLLM_QWEN4_INCREMENTAL_INDEXER", enabled)
            with _NoHostOrDynamicOps():
                outputs.append(indexer(
                    pair.hidden[:, 19:20], pair.cos, pair.sin, None,
                    past_len=position if position is not None else 19, raw_keys_override=raw_prefix,
                ))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        assert pair.fast_norm.groups[-1] == 20 // 4


def test_untracked_eager_arena_mutation_and_borrowed_owner_rebuild_prefix(monkeypatch) -> None:
    pair = _Pair(monkeypatch, fixed=True)
    pair.step(0, 19)
    for cache in (pair.fast_cache, pair.old_cache):
        cache.index_k[:, 1:3].mul_(-2)
    _, work, _ = pair.step(19)
    assert work == 20 // 4
    # Inference tensors have no mutation counter. A second cache using the same
    # arena still changes its Python owner, forcing conservative eager rebuild.
    with torch.inference_mode():
        pair = _Pair(monkeypatch, fixed=True)
        pair.step(0, 19)
        borrowed = SimpleNamespace(index_k=pair.fast_cache.index_k, kv_width=67, length=19)
        other = QSAIndexer(pair.config, pair.weights)
        other(pair.hidden[:, 19:20] * -3, pair.cos, pair.sin, borrowed, past_len=19)
        pair.old_cache.index_k.copy_(pair.fast_cache.index_k)
        _, work, _ = pair.step(20)
        assert work == 21 // 4
        assert borrowed.indexer_pool is not pair.fast_cache.indexer_pool
        assert borrowed.indexer_pool.keys.data_ptr() != pair.fast_cache.indexer_pool.keys.data_ptr()


def test_feature_toggle_write_forgets_cached_validity(monkeypatch) -> None:
    pair = _Pair(monkeypatch, fixed=True)
    pair.step(0, 19)
    pointer = pair.fast_cache.indexer_pool.keys.data_ptr()
    monkeypatch.setenv("POCKETLLM_QWEN4_INCREMENTAL_INDEXER", "0")
    pair.fast(pair.hidden[:, :3] * -3, pair.cos, pair.sin, pair.fast_cache, past_len=0)
    pair.old(pair.hidden[:, :3] * -3, pair.cos, pair.sin, pair.old_cache, past_len=0)
    assert pair.fast_cache.indexer_pool.completed == 0
    pair.step(3, 18)
    assert pair.fast_cache.indexer_pool.keys.data_ptr() == pointer


@pytest.mark.parametrize("tensor_position", [False, True])
def test_native_2048_budget_long_prefix_then_decode_matches_recompute(monkeypatch, tensor_position) -> None:
    pair = _Pair(monkeypatch, dtype=torch.bfloat16, fixed=tensor_position, capacity=2083,
                 budget=2048, table_length=4096)
    past_len = 0
    for size in (1024, 1024, 7):
        pair.step(past_len, size)
        past_len += size
    position = torch.tensor(past_len) if tensor_position else None
    if position is not None:
        pair.step(past_len, position=position, current=pair.hidden[:, past_len:past_len + 1] * -3)
    for past_len in range(2055, 2067):
        if position is not None:
            position.fill_(past_len)
        selected, work, _ = pair.step(past_len, position=position)
        assert selected.shape == (1, 1, 2051)
        assert work == (1 if tensor_position else int((past_len + 1) % 4 == 0))
        assert pair.fast.budget == 2048


@pytest.mark.parametrize("past_len,seq_len", [(9, 2), (10, 7), (11, 9), (12, 5)])
def test_integer_rewind_chunk_invalidates_only_completed_suffix_at_p_over_r4(monkeypatch, past_len, seq_len) -> None:
    pair = _Pair(monkeypatch, batch=2, dtype=torch.bfloat16, fixed=True)
    pair.step(0, 35)
    current = _hidden(2, seq_len, torch.bfloat16, seed=119)
    _, work, _ = pair.step(past_len, seq_len, current=current)
    assert work == (past_len + seq_len) // 4 - past_len // 4
    for position in range(past_len + seq_len, past_len + seq_len + 5):
        _, work, _ = pair.step(position)
        assert work == int((position + 1) % 4 == 0)


def test_scalar_new_request_and_nan_future_groups_are_hidden_until_fully_written(monkeypatch) -> None:
    with torch.inference_mode():
        pair = _Pair(monkeypatch, fixed=True, capacity=19)
        pair.fast_cache.index_k.fill_(float("nan"))
        pair.old_cache.index_k.fill_(float("nan"))
        position = torch.tensor(0)
        for past_len in range(16):
            position.fill_(past_len)
            _, work, _ = pair.step(past_len, position=position)
            assert work == (19 // 4 + 1 if past_len == 0 else 1)
        pointer = pair.fast_cache.indexer_pool.keys.data_ptr()
        # A new scalar-position request starts at zero over the same stale
        # arena. Every formerly complete group must be refreshed before exposure.
        pair.hidden = _hidden(1, 19, torch.float32, seed=719)
        pair.fast_cache.length = pair.old_cache.length = 0
        for past_len in range(19):
            position.fill_(past_len)
            _, work, _ = pair.step(past_len, position=position)
            assert work == 1
            assert pair.fast_cache.indexer_pool.keys.data_ptr() == pointer


@pytest.mark.parametrize("tensor_position", [False, True])
def test_fixed_span_larger_than_actual_arena_keeps_final_group_in_bounds(monkeypatch, tensor_position) -> None:
    pair = _Pair(monkeypatch, fixed=True, capacity=19)
    pair.fast_cache.kv_width = pair.old_cache.kv_width = 99
    pair.step(0, 16)
    position = torch.tensor(16) if tensor_position else None
    for past_len in range(16, 19):
        if position is not None:
            position.fill_(past_len)
        selected, _, _ = pair.step(past_len, position=position)
        assert selected.shape == (1, 1, 11)
        assert pair.fast_cache.indexer_pool.keys.shape == (1, 19 // 4, 8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tensor_position", [False, True])
def test_attention_outputs_match_old_recompute_with_cache_owned_key_table(monkeypatch, dtype, tensor_position) -> None:
    pair = _Pair(monkeypatch, batch=2, dtype=dtype, fixed=True, table_batch=1)
    fast_attention = QSAAttention(pair.config, pair.weights)
    old_attention = QSAAttention(pair.config, pair.weights)
    fast_cache, old_cache = pair.fast_cache, pair.old_cache
    fast_cache.cos = old_cache.cos = pair.cos
    fast_cache.sin = old_cache.sin = pair.sin
    for past_len, size in ((0, 19), (19, 1), (20, 3), (23, 1), (24, 3), (27, 1)):
        position = torch.tensor(past_len) if tensor_position and past_len > 0 else past_len
        current = pair.hidden[:, past_len:past_len + size]
        # The query rows can deliberately differ from the absolute-position key
        # table; the current P0 caller contract must survive cached key reuse.
        query_cos = pair.cos[:, past_len + 2:past_len + size + 2]
        query_sin = pair.sin[:, past_len + 2:past_len + size + 2]
        outputs = []
        for enabled, attention, cache in (("1", fast_attention, fast_cache), ("0", old_attention, old_cache)):
            monkeypatch.setenv("POCKETLLM_QWEN4_INCREMENTAL_INDEXER", enabled)
            outputs.append(attention(current, query_cos, query_sin, cache, past_len=position))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(fast_cache.k, old_cache.k, rtol=0, atol=0)
        torch.testing.assert_close(fast_cache.v, old_cache.v, rtol=0, atol=0)


def test_the_completed_prefix_pool_is_on_by_default(monkeypatch) -> None:
    """A launch that says nothing must get the pool, not the per-step full re-pool.

    Measured on eight RTX 4090s at the native 2048 budget with every decode step replayed and
    the tokens matching: 8192 positions 37.53 -> 38.15 tok/s excluding setup, 65536
    30.96 -> 32.80, 262080 19.24 -> 22.34.  The gain grows with the context because the cost it
    removes -- scoring the whole history to answer about one position -- grows with it.
    """
    config = _config(Qwen4ExpTextConfig().indexer_budget)
    indexer = QSAIndexer(config, _weights(config, torch.float32))
    cache = QSAAttentionCache(config, 1, 2060, torch.device("cpu"), torch.float32)
    hidden = _hidden(1, 2055, torch.float32)
    cos, sin = _rope(4096)
    monkeypatch.delenv("POCKETLLM_QWEN4_INCREMENTAL_INDEXER", raising=False)
    selected = indexer(hidden, cos, sin, cache, past_len=0)
    assert selected.shape == (1, 2055, 2051)
    assert cache.indexer_pool is not None


def test_the_pool_and_the_opt_out_choose_the_same_positions(monkeypatch) -> None:
    """The default must not change the answer, only the work it takes to reach it."""
    config = _config(Qwen4ExpTextConfig().indexer_budget)
    weights = _weights(config, torch.float32)
    hidden = _hidden(1, 2055, torch.float32)
    cos, sin = _rope(4096)

    def run(value: str):
        monkeypatch.setenv("POCKETLLM_QWEN4_INCREMENTAL_INDEXER", value)
        indexer = QSAIndexer(config, weights)
        cache = QSAAttentionCache(config, 1, 2060, torch.device("cpu"), torch.float32)
        return indexer(hidden, cos, sin, cache, past_len=0)

    assert torch.equal(run("1"), run("0"))
