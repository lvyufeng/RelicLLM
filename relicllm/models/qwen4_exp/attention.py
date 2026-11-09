"""Qwen4-Exp attention paths: GatedDeltaNet linear attention and QSA attention.

Both keep their own cache state. GatedDeltaNet carries a depthwise conv window
plus a (num_v_heads, head_k_dim, head_v_dim) recurrent matrix; QSA layers carry
a regular KV cache plus the indexer's raw pre-pool key cache.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from relicllm.models.qwen4_exp.config import Qwen4ExpTextConfig
from relicllm.models.qwen4_exp.layers import (
    RMSNorm,
    RMSNormGated,
    apply_rotary_pos_emb,
    prefill_linear,
)
from relicllm.runtime.ops import load_ops


def l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """FLA-compatible L2 norm: eps is added to the squared sum, not to the norm."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


# ---------------------------------------------------------------------------
# Gated delta rule
# ---------------------------------------------------------------------------


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sequential delta-rule scan in float32.

    query/key: (batch, seq, heads, k_dim); value: (batch, seq, heads, v_dim);
    g/beta: (batch, seq, heads); state: (batch, heads, k_dim, v_dim).

    This is O(seq) launches, which is fine for decode (seq==1) but slow for
    prefill; `chunk_gated_delta_rule` is the prefill path.
    """
    initial_dtype = query.dtype
    query = l2norm(query)
    key = l2norm(key)
    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    )

    batch_size, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    query = query * (1 / (k_dim**0.5))

    out = torch.zeros(batch_size, num_heads, seq_len, v_dim, dtype=torch.float32, device=value.device)
    state = (
        torch.zeros(batch_size, num_heads, k_dim, v_dim, dtype=torch.float32, device=value.device)
        if initial_state is None
        else initial_state.to(torch.float32)
    )

    for i in range(seq_len):
        q_t, k_t, v_t = query[:, :, i], key[:, :, i], value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
        state = state * g_t
        kv_mem = (state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q_t.unsqueeze(-1)).sum(dim=-2)

    return out.transpose(1, 2).contiguous().to(initial_dtype), state


def cuda_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """SM75 recurrent scan for the real BF16, D=128, batch-one path.

    The kernel retains the float32 recurrent state and pre-normalizes Q/K in
    float32, but shards each 128x128 state across sub-warps so a long prefill is
    one CUDA launch rather than Python-controlled chunk algebra. Unsupported
    shapes return ``None`` and leave the established PyTorch path untouched.
    """
    if (
        query.device.type != "cuda"
        or query.dtype is not torch.bfloat16
        or query.shape[0] != 1
        or query.shape[-1] != 128
        or key.shape != query.shape
        or value.shape[0] != 1
        or value.shape[1] != query.shape[1]
        or value.shape[-1] != 128
        or value.shape[2] % query.shape[2] != 0
        or beta.dtype is not torch.bfloat16
        or g.dtype is not torch.float32
    ):
        return None

    ext = load_ops()
    if ext is None or not hasattr(ext, "qwen4_exp_gated_delta_bf16_forward"):
        return None

    if initial_state is None:
        initial_state = torch.zeros(
            query.shape[0],
            value.shape[2],
            query.shape[-1],
            value.shape[-1],
            device=query.device,
            dtype=torch.float32,
        )
    groups_per_block = int(os.environ.get("POCKETLLM_QWEN4_GDN_GROUPS_PER_CTA", "1"))
    if groups_per_block not in (1, 2, 4, 8):
        groups_per_block = 1
    output, state = ext.qwen4_exp_gated_delta_bf16_forward(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        initial_state.contiguous(),
        1e-6,
        groups_per_block,
    )
    return output, state


def _run_table(selection: torch.Tensor):
    """The selection as sorted runs: fixed shapes all the way, no host read.

    The block indexer picks whole blocks of `compress_ratio`, so the 2048 positions it names arrive
    as a couple of hundred runs of eight-odd rows -- measured at 248 runs over 2051 rows at 8K.
    Staging a run at a time is what turns 2048 transfers a layer into 248, and the measured
    difference is the DMA engine's per-transfer cost: 7.7 ms of staging a token against the 1.8 ms
    the bytes alone account for.

    Everything is sized by the row count, which is the worst case -- one run a row -- so the table
    never has to grow and the live count stays on the device where a capture can use it.
    """
    flat = selection.clamp_min(0)
    ordered, _ = torch.sort(flat)
    breaks = torch.ones_like(ordered, dtype=torch.bool)
    breaks[1:] = (ordered[1:] - ordered[:-1]) != 1
    run_id = (torch.cumsum(breaks.to(torch.int64), dim=0) - 1).to(torch.int64)
    capacity = ordered.numel()
    device = ordered.device
    sentinel = torch.iinfo(torch.int32).max
    slots = torch.arange(capacity, dtype=torch.int32, device=device)
    run_start = torch.full((capacity,), sentinel, dtype=torch.int32, device=device)
    run_start.scatter_reduce_(0, run_id, ordered.to(torch.int32), reduce="amin")
    run_dst = torch.full((capacity,), sentinel, dtype=torch.int32, device=device)
    run_dst.scatter_reduce_(0, run_id, slots, reduce="amin")
    run_rows = torch.zeros((capacity,), dtype=torch.int32, device=device)
    run_rows.scatter_add_(0, run_id, torch.ones(capacity, dtype=torch.int32, device=device))
    run_count = (run_id[-1] + 1).reshape(1).to(torch.int32)
    return ordered, run_start, run_rows, run_dst, run_count


def _gather_runs(arena: torch.Tensor, table):
    """Stage whole runs of a host arena, laid out in sorted order."""
    ext = load_ops()
    gather = getattr(ext, "qwen4_exp_gather_runs_bf16", None) if ext is not None else None
    if gather is None:
        return None
    ordered, run_start, run_rows, run_dst, run_count = table
    rows = int(ordered.numel())
    heads = arena.shape[1]
    staged = [
        gather(
            arena.data_ptr() + head * arena.stride(1) * arena.element_size(),
            run_start, run_rows, run_dst, run_count, rows,
        )
        for head in range(heads)
    ]
    return torch.stack(staged, dim=0).unsqueeze(0)


def _gather_host_rows(arena: torch.Tensor, rows: torch.Tensor, device) -> torch.Tensor:
    """Copy `rows` of a host-resident `[1, heads, positions, 256]` arena onto the device.

    Falls back to a host-side gather when the extension that does it in one launch is not built:
    the fallback has to read the index list back, which is the slow path this exists to avoid, but
    it keeps a source checkout without the kernel usable.
    """
    from relic_core.kernels.cuda_loader import load_cuda_kernel

    native = load_cuda_kernel()
    gather = getattr(native, "qwen4_exp_gather_rows_bf16", None) if native is not None else None
    heads = arena.shape[1]
    if gather is None:
        picked = arena.index_select(2, rows.to("cpu"))
        return picked.to(device, non_blocking=True)
    staged = [
        gather(arena.data_ptr() + head * arena.stride(1) * arena.element_size(), rows)
        for head in range(heads)
    ]
    # `[rows, 256]` a head, so the head axis goes back in front of the position axis.
    return torch.stack(staged, dim=0).unsqueeze(0)


def _scatter_host_rows(arena: torch.Tensor, rows: torch.Tensor, at: torch.Tensor) -> None:
    """Write `rows` into a host-resident `[1, heads, positions, 256]` arena at `at`."""
    from relic_core.kernels.cuda_loader import load_cuda_kernel

    native = load_cuda_kernel()
    scatter = getattr(native, "qwen4_exp_scatter_rows_bf16", None) if native is not None else None
    batch, heads = arena.shape[0], arena.shape[1]
    if scatter is None:
        arena[:, :, at.to("cpu").tolist()] = rows.to("cpu", non_blocking=True)
        return
    row_bytes = arena.stride(2) * arena.element_size()
    for b in range(batch):
        for head in range(heads):
            offset = b * arena.stride(0) * arena.element_size() + head * row_bytes
            scatter(rows[b, head].contiguous(), arena.data_ptr() + offset, at)


def _stage_selected(cache, selected: torch.Tensor, device):
    """Bring just the positions this call reads back onto the device.

    `selected` is `[b, q, w]` of positions with `-1` marking unused slots, and the layers below
    index `key`/`value` with it.  With the arena in host memory that indexing cannot happen where
    it used to, so the rows it names are copied across -- the union of every query in the call, so
    a chunk of queries shares one transfer -- and `selected` is rewritten to address the copy.
    """
    if selected.shape[1] == 1:
        # A decode step has one query, so its own selection list *is* the staging order and the
        # rewrite is the identity.
        #
        # The rows come across in a kernel rather than through `index_select`, because the index
        # list lives on the device and reading it back to slice a host tensor is a synchronous
        # device-to-host copy on every layer -- measured at 12% of a decode step, and fatal to a
        # capture, which cannot contain one.  Under UVA the arena's host address is one a kernel
        # can dereference, so the gather runs on the card and the traffic is the rows themselves.
        # Measured at par with the per-row gather and not better: the sort and the three scatter
        # reductions the table costs are the DMA transactions it saves.  Kept behind a flag because
        # the reasoning is right and the balance may differ on a card whose per-transfer cost is
        # higher, but it is not the default while it buys nothing here.
        table = (
            _run_table(selected[0, 0])
            if os.environ.get("POCKETLLM_QWEN4_RUN_STAGING") == "1"
            else None
        )
        if table is not None:
            staged_k = _gather_runs(cache.k, table)
            staged_v = _gather_runs(cache.v, table)
            if staged_k is not None:
                # Runs are staged in sorted order, so a selected position addresses the slot it
                # sorted into; `-1` still marks a slot the mask drops.
                remapped = torch.searchsorted(table[0], selected.clamp_min(0))
                return staged_k, staged_v, remapped.masked_fill(selected < 0, -1)
        rows = selected[0, 0].clamp_min(0).to(torch.int32)
        staged_k = _gather_host_rows(cache.k, rows, device)
        staged_v = _gather_host_rows(cache.v, rows, device)
        width = selected.shape[-1]
        remapped = torch.arange(width, device=selected.device, dtype=selected.dtype)
        remapped = remapped.view(1, 1, -1).masked_fill(selected < 0, -1)
        return staged_k, staged_v, remapped
    live = selected[selected >= 0]
    if live.numel() == 0:
        raise RuntimeError("a host-resident KV cache needs at least one live position to stage")
    # A position past the arena is an `index_select` on the host that faults on the *device*, an
    # assert whose message names IndexKernel rather than the caller.  Naming it here is the
    # difference between a one-line fix and a bisect: the arena is sized for the request, and a
    # selection that reaches past it means the caller and the cache disagree about how long the
    # context is.
    width = int(cache.k.shape[2])
    if int(live.max()) >= width:
        raise RuntimeError(
            f"QSA selection reaches position {int(live.max())} but the arena holds {width}; "
            f"cache.length={int(cache.length)}, selected={tuple(selected.shape)}"
        )
    union = torch.unique(live)                      # sorted, on the device
    rows = union.to("cpu")
    if int(union.max()) >= int(cache.k.shape[2]):
        raise RuntimeError(
            f"staged position {int(union.max())} is past the arena ({int(cache.k.shape[2])})"
        )
    staged_k = cache.k.index_select(2, rows).to(device, non_blocking=True)
    staged_v = cache.v.index_select(2, rows).to(device, non_blocking=True)
    # `searchsorted` is exact because `union` is sorted and every live index is in it.
    remapped = torch.searchsorted(union, selected.clamp_min(0))
    return staged_k, staged_v, remapped.masked_fill(selected < 0, -1)


def _write_kv(cache, key: torch.Tensor, value: torch.Tensor, past_len, seq_len: int) -> None:
    """Put this step's keys and values at `past_len`.

    A Python `past_len` is a slice, which is what the eager loop has always done.  A tensor
    `past_len` cannot slice -- torch would read its value on the host, and inside a capture that is
    an error rather than a slow path -- so it goes through `index_copy_` against the position, which
    is the form a recording can replay at any offset.
    """
    # Pinned, not merely "on the host": a model that is running on the CPU has its arena there
    # too, and that is the ordinary path, not an offload -- it has no card to gather into.
    if (
        cache is not None
        and getattr(cache, "host_kv", False)
        and getattr(cache, "k", None) is not None
        and cache.k.device.type == "cpu"
    ):
        # A host-resident arena, written through a kernel rather than a slice.
        #
        # A slice would bake the offset into a recording: the Python integer becomes a fixed address
        # in the captured graph, so every replay would write its rows to the position the capture
        # happened at.  The offset therefore has to be a device value the recording reads, which is
        # what the position tensor already is -- it just has to travel with the data.
        if isinstance(past_len, torch.Tensor):
            at = (past_len.reshape(()) + torch.arange(seq_len, device=key.device)).to(torch.int32)
        else:
            at = torch.arange(
                past_len, past_len + seq_len, dtype=torch.int32, device=key.device
            )
        # Only the integer position can be checked here: reading a position tensor back to compare
        # it would be a host read, which a capture refuses.  The kernel writes where the tensor
        # says, and the arena is sized for the generation the caller laid out.
        width = int(cache.k.shape[2])
        if not isinstance(past_len, torch.Tensor) and int(past_len) >= width:
            raise RuntimeError(f"writing KV at {past_len} is past the arena ({width})")
        _scatter_host_rows(cache.k, key, at)
        _scatter_host_rows(cache.v, value, at)
        return
    if isinstance(past_len, torch.Tensor):
        # One row at a time: `index_copy_` takes a 1-D index, and the position is a fact about the
        # step rather than about a row, so the same offsets serve every row.
        offsets = past_len.reshape(()) + torch.arange(seq_len, device=key.device)
        for row in range(key.shape[0]):
            cache.k[row].index_copy_(1, offsets, key[row])
            cache.v[row].index_copy_(1, offsets, value[row])
        return
    cache.k[:, :, past_len : past_len + seq_len] = key
    cache.v[:, :, past_len : past_len + seq_len] = value


def _fixed_kv_width() -> int:
    """The key width a recording should see, or 0 to keep the length-dependent one.

    A capture bakes the tensor shapes it was given, so reading `cache.k[:length]` fixes both the
    key length and everything downstream of it: one graph per position.  Reading a constant-width
    arena and letting the selection mask say what is visible makes the shapes position-independent,
    at a measured cost -- the kernel runs in 94 us at width 54 and 349 us at 512, about 24 ms a
    step across the layers.  Off by default, so the eager loop pays none of it.
    """
    try:
        return int(os.environ.get("POCKETLLM_QWEN4_QSA_FIXED_WIDTH", "0"))
    except ValueError:
        return 0


def _kv_width_of(cache) -> int:
    """The arena width this cache's reads use: its own if it has one, else the environment's.

    The cache is the right scope for it. A capture needs the arena and an eager step does not, and
    one process can be asked for both -- a server that captured one request's decode and then
    serves another eagerly would otherwise charge the eager request the arena's ~24 ms a step for
    the rest of the process's life, because the environment is process-wide and `_fixed_kv_width`
    is read once a layer.

    `kv_width` is set on the cache before the first step that needs it, which for a capture is
    before the probe that sizes the logits buffer -- see `graph_window.DecodeGraph`.
    """
    width = getattr(cache, "kv_width", 0) if cache is not None else 0
    return int(width) if width else _fixed_kv_width()


def cuda_qsa_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    selected_indices: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor | None:
    """Indexed BF16 GQA attention for the real TP4 sparse-attention shape."""
    enabled = os.environ.get("POCKETLLM_QWEN4_QSA_CUDA", "1").lower() in {
        "1",
        "true",
        "yes",
    }
    if (
        not enabled
        or query.device.type != "cuda"
        or query.dtype is not torch.bfloat16
        or key.dtype is not torch.bfloat16
        or value.dtype is not torch.bfloat16
        or query.shape[0] != 1
        or query.shape[-1] != 256
        or key.shape != value.shape
        or key.shape[0] != query.shape[0]
        or key.shape[-1] != query.shape[-1]
        or query.shape[1] % key.shape[1] != 0
        or query.shape[1] // key.shape[1] > 12
        or selected_indices.shape[:2] != (query.shape[0], query.shape[2])
    ):
        return None

    ext = load_ops()
    if ext is None or not hasattr(ext, "qwen4_exp_qsa_bf16_forward"):
        return None
    return ext.qwen4_exp_qsa_bf16_forward(
        query.transpose(1, 2).contiguous(),
        key.contiguous(),
        value.contiguous(),
        selected_indices.contiguous().to(torch.int32),
        float(softmax_scale),
    )


def chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    chunk_size: int = 64,
    state_log: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Chunked (WY-representation) delta rule for prefill.

    Ports upstream's `torch_chunk_gated_delta_rule`: within a chunk the
    interactions are resolved by a forward substitution on the strictly-lower
    triangular attention matrix, then chunks are chained through the recurrent
    state.  Mathematically identical to the sequential scan, ~chunk_size fewer
    kernel launches.
    """
    initial_dtype = query.dtype
    query = l2norm(query)
    key = l2norm(key)
    query, key, value, beta, g = (
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    )

    batch_size, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_len = seq_len + pad_size

    query = query * (1 / (k_dim**0.5))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape into chunks: (batch, heads, n_chunks, chunk_size, dim)
    def to_chunks(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(batch_size, num_heads, total_len // chunk_size, chunk_size, -1)

    query, key, value, k_beta, v_beta = (to_chunks(x) for x in (query, key, value, k_beta, v_beta))
    g = g.reshape(batch_size, num_heads, total_len // chunk_size, chunk_size)
    g_cum = g.cumsum(dim=-1)

    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0
    )
    decay_mask = ((g_cum.unsqueeze(-1) - g_cum.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    # Forward substitution to invert (I + strictly-lower-triangular).
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g_cum.exp().unsqueeze(-1))

    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_dim, v_dim, dtype=torch.float32, device=value.device)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    core_attn_out = torch.zeros_like(value)
    mask_strict = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1
    )

    for i in range(0, total_len // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask_strict, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g_cum[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        g_last = g_cum[:, :, i, -1, None, None].exp()
        decay_k = (g_cum[:, :, i, -1, None] - g_cum[:, :, i]).exp()
        last_recurrent_state = (
            last_recurrent_state * g_last + (k_i * decay_k[..., None]).transpose(-1, -2) @ v_new
        )
        if state_log is not None and i < total_len // chunk_size:
            # One entry per chunk, which is one entry per position when the caller asks for
            # `chunk_size=1`.  A speculative verify needs exactly that: it consumes K drafts in a
            # single forward and then has to put these layers back to whichever position the trunk
            # accepted, and the rolling state has no other way to be rewound.
            state_log.append(last_recurrent_state)

    core_attn_out = core_attn_out.reshape(batch_size, num_heads, total_len, v_dim)
    core_attn_out = core_attn_out[:, :, :seq_len]
    return core_attn_out.transpose(1, 2).contiguous().to(initial_dtype), last_recurrent_state


# ---------------------------------------------------------------------------
# GatedDeltaNet layer
# ---------------------------------------------------------------------------


def _state_slots(cache) -> dict | None:
    """The per-position arrays this cache keeps its linear-layer state in, or None.

    A single mutable pair of buffers is right for a chain of steps run one after another and wrong
    for a recording: a replay writes the state it computes, so running the same recording twice
    applies the update twice.  When the caller keeps one entry a position instead, a step reads its
    own entry and writes the next -- which is what makes a recording repeatable.
    """
    return getattr(cache, "state_slots", None)


class SlotEntries:
    """One field's per-position entries, and the stack a step indexes.

    The stack is the store: a read and the write that follows it go through the same memory, and
    the entries are only the material it is built from.  `slot_layout` builds it, so there is no
    moment at which the two can disagree.
    """

    __slots__ = ("entries", "stack")

    def __init__(self, entries: list) -> None:
        self.entries = list(entries)
        self.stack = torch.stack(self.entries) if self.entries else None

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index):
        return self.stack[index]


def slot_layout(fields: dict) -> dict:
    """Lay out per-position state: `{"conv_state": [t0, t1, ...], ...}` -> SlotEntries a field.

    Called before the first step, so the stack a recording points at exists and is filled before
    anything can write through it.
    """
    return {name: SlotEntries(values) for name, values in fields.items()}


def _slot_read(entry: "SlotEntries", slot) -> torch.Tensor:
    """The entry a step starts from: its own position, selected on the device."""
    if isinstance(slot, torch.Tensor):
        # `index_select` with a 1-D index: indexing a stack by a 0-dim tensor is advanced
        # indexing, which returns a copy -- fine for a read, useless for the write that follows.
        return entry.stack.index_select(0, slot.reshape(1))[0]
    return entry.stack[slot]


def _slot_store(entry: "SlotEntries", slot, value: torch.Tensor) -> None:
    """Put `value` at `slot + 1` of the stack, on the device, in place.

    A capture writes through the addresses it recorded, so this is an in-place update of the stack,
    which is also the only store: nothing reads the list the stack was built from.
    """
    if isinstance(slot, torch.Tensor):
        entry.stack.index_copy_(0, slot.reshape(1) + 1, value.unsqueeze(0))
        return
    entry.stack[slot + 1].copy_(value)


def _state_at(cache, name: str):
    """The value a step reads for `name`: its own entry when the cache is laid out by position.

    `conv_state` is a window this layer always has -- the first step convolves it against zeros.
    `recurrent_state` is different: before the first step there is no recurrence to continue, and
    passing a zero tensor instead of None would make the rule start one where the caller meant it
    to start from nothing.
    """
    if cache is None:
        return None
    slots = _state_slots(cache)
    if slots is not None and name in slots:
        return _slot_read(slots[name], cache.state_slot)
    value = getattr(cache, name, None)
    if name == "recurrent_state" and not getattr(cache, "initialized", False):
        return None
    return value


def _update_state(cache, name: str, value: torch.Tensor) -> None:
    """Put `value` into `cache.<name>`, keeping the buffer when it already fits.

    A decode step can be captured as a CUDA graph, and a graph holds the addresses of the tensors
    it was recorded against.  Assigning a fresh tensor to the cache every step leaves the recorded
    address pointing at memory the cache no longer names: the replay writes it, the cache keeps
    stale data, and the next step reads a history the graph never produced.

    With `state_slots` set, `name` is written to `slots[name][slot + 1]` -- the entry the *next*
    step will read -- and the visible attribute is left alone, so a step's output depends only on
    its own entry.
    """
    slots = _state_slots(cache)
    if slots is not None and name in slots:
        entry = slots[name]
        if entry.stack.shape[1:] != value.shape:
            raise RuntimeError(
                f"{name} slot is {tuple(entry.stack.shape[1:])}, "
                f"the layer produced {tuple(value.shape)}"
            )
        _slot_store(entry, cache.state_slot, value)
        return
    live = getattr(cache, name, None)
    if live is not None and live.shape == value.shape:
        live.copy_(value)
    else:
        setattr(cache, name, value)


class GatedDeltaNetCache:
    """Depthwise conv window + recurrent state for one linear-attention layer.

    Sized from the *shard's* head counts, not the config's, so a TP rank only
    allocates the slice of the conv/recurrent state it actually owns.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        num_v_heads: int | None = None,
        num_k_heads: int | None = None,
    ) -> None:
        v_heads = config.linear_num_value_heads if num_v_heads is None else num_v_heads
        k_heads = config.linear_num_key_heads if num_k_heads is None else num_k_heads
        conv_dim = 2 * k_heads * config.linear_key_head_dim + v_heads * config.linear_value_head_dim
        self.conv_state = torch.zeros(
            batch_size, conv_dim, config.linear_conv_kernel_dim - 1, device=device, dtype=dtype
        )
        self.recurrent_state = torch.zeros(
            batch_size,
            v_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            device=device,
            dtype=torch.float32,
        )
        self.initialized = False
        #: When set to a list, the layer appends its post-position recurrent state here instead of
        #: only keeping the last one.  A speculative verify turns this on so a partially accepted
        #: round can be rewound; `None` everywhere else, and the step is then bit-identical.
        self.state_log = None


class GatedDeltaNet:
    def _fp32_weight(self, name: str) -> torch.Tensor:
        """A weight of this layer widened to float32 once, not once a step."""
        weight = getattr(self, name)
        cache = self.__dict__.setdefault("_fp32_weights", {})
        held = cache.get(name)
        if held is None or held[0] is not weight or held[1] != weight.device:
            widened = weight.float()
            cache[name] = (weight, weight.device, widened)
            return widened
        return held[2]

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        weights: dict[str, torch.Tensor],
        *,
        num_v_heads: int | None = None,
        num_k_heads: int | None = None,
    ) -> None:
        self.config = config
        # TP shards the head dimension; the shard's head counts override config.
        self.num_v_heads = config.linear_num_value_heads if num_v_heads is None else num_v_heads
        self.num_k_heads = config.linear_num_key_heads if num_k_heads is None else num_k_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim

        self.in_proj_qkv = weights["in_proj_qkv"]
        self.in_proj_z = weights["in_proj_z"]
        self.in_proj_b = weights["in_proj_b"]
        self.in_proj_a = weights["in_proj_a"]
        self.conv1d_weight = weights["conv1d"]  # (conv_dim, 1, kernel)
        self.dt_bias = weights["dt_bias"]
        self.A_log = weights["A_log"]
        self.out_proj = weights["out_proj"]
        self.norm = RMSNormGated(
            weights["norm"], config.rms_norm_eps, activation=config.output_gate_type or config.hidden_act
        )
        self.cuda_rule_enabled = os.environ.get(
            "POCKETLLM_QWEN4_GDN_CUDA", "1"
        ).lower() in {"1", "true", "yes"}
        self.last_profile: dict[str, float] = {}

    def __call__(self, hidden_states: torch.Tensor, cache: GatedDeltaNetCache | None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        profile = os.environ.get("POCKETLLM_QWEN4_ATTN_PHASE_PROFILE", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        if profile:
            import time

            torch.cuda.synchronize(hidden_states.device)
            started = time.perf_counter()

        mixed_qkv = prefill_linear(hidden_states, self.in_proj_qkv).transpose(1, 2)  # (b, conv_dim, seq)
        z = prefill_linear(hidden_states, self.in_proj_z).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = prefill_linear(hidden_states, self.in_proj_b)
        a = prefill_linear(hidden_states, self.in_proj_a)
        if profile:
            torch.cuda.synchronize(hidden_states.device)
            after_projection = time.perf_counter()

        conv_weight = self.conv1d_weight.squeeze(1)
        channels = conv_weight.shape[0]
        if cache is not None:
            # Prepend the cached window so the causal conv sees real history, then
            # keep only the new positions and refresh the window.
            window = _state_at(cache, "conv_state")
            padded = torch.cat([window.to(mixed_qkv.dtype), mixed_qkv], dim=-1)
            # In place when the state already exists: rebinding it would leave the old buffer --
            # the one a captured step recorded the address of -- written and unreferenced, and the
            # cache pointing at a tensor no kernel had touched.
            _update_state(cache, "conv_state", padded[:, :, -(self.conv_kernel_size - 1) :])
            conv_in = padded
        else:
            conv_in = F.pad(mixed_qkv, (self.conv_kernel_size - 1, 0))
        conv_out = F.conv1d(conv_in, conv_weight.unsqueeze(1), None, padding=0, groups=channels)
        mixed_qkv = F.silu(conv_out[:, :, -seq_len:])
        if profile:
            torch.cuda.synchronize(hidden_states.device)
            after_conv = time.perf_counter()

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        # float32 on A_log: in bf16 exp() of a large magnitude saturates to inf.
        #
        # The widening is of a *weight*, so it is the same value every step: done here it was two
        # conversions a layer, seventy-two a token, of tensors that never change.  It is kept per
        # device so a model that is moved still widens the tensor it is holding.
        g = -self._fp32_weight("A_log").exp() * F.softplus(
            a.float() + self._fp32_weight("dt_bias")
        )

        initial_state = _state_at(cache, "recurrent_state")
        cuda_result = (
            cuda_gated_delta_rule(query, key, value, g, beta, initial_state)
            if self.cuda_rule_enabled and seq_len > 1
            else None
        )
        if cuda_result is None:
            repeat = self.num_v_heads // self.num_k_heads
            if repeat > 1:
                query = query.repeat_interleave(repeat, dim=2)
                key = key.repeat_interleave(repeat, dim=2)
            # A verify round has to be able to put these layers back to a *position*, not to the
            # end of the forward.  The chunked rule only returns the state at each chunk boundary,
            # so recording per-position states means asking for one position a chunk -- the same
            # arithmetic, minus the intra-chunk parallelism a 2-3 token sequence never had anyway.
            log = getattr(cache, "state_log", None) if cache is not None else None
            if log is not None and seq_len > 1:
                # Two rolling values have to be recoverable per position, not just one: the
                # recurrence (returned by the rule) and the depthwise conv's window, which is the
                # tail of the conv input.  The window is a slice of `padded`, so it is read off
                # rather than recorded by the rule.
                log.clear()
                rec_log: list = []
                rule = chunk_gated_delta_rule
                core_attn_out, last_state = rule(
                    query, key, value, g, beta, initial_state, 1, rec_log
                )
                span = self.conv_kernel_size - 1
                for j in range(seq_len):
                    end = padded.shape[-1] - seq_len + j + 1
                    log.append((
                        padded[:, :, end - span : end],
                        rec_log[j] if j < len(rec_log) else last_state,
                    ))
            else:
                rule = recurrent_gated_delta_rule if seq_len == 1 else chunk_gated_delta_rule
                core_attn_out, last_state = rule(query, key, value, g, beta, initial_state)
        else:
            core_attn_out, last_state = cuda_result
        if profile:
            torch.cuda.synchronize(hidden_states.device)
            after_core = time.perf_counter()
        if cache is not None:
            _update_state(cache, "recurrent_state", last_state)
            cache.initialized = True

        core_attn_out = self.norm(
            core_attn_out.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim)
        ).reshape(batch_size, seq_len, -1)
        output = prefill_linear(core_attn_out, self.out_proj)
        if profile:
            torch.cuda.synchronize(hidden_states.device)
            finished = time.perf_counter()
            self.last_profile = {
                "projection": after_projection - started,
                "conv": after_conv - after_projection,
                "core": after_core - after_conv,
                "output": finished - after_core,
            }
        return output


# ---------------------------------------------------------------------------
# QSA sparse attention
# ---------------------------------------------------------------------------


class QSAAttentionCache:
    """KV cache plus the indexer's pre-pool raw key cache for one QSA layer."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        batch_size: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        *,
        num_kv_heads: int | None = None,
    ) -> None:
        kv_heads = config.num_key_value_heads if num_kv_heads is None else num_kv_heads
        #: Keys and values in pinned host memory instead of the card.
        #:
        #: A sparse-attention layer reads 2048 positions a query whatever the context is, so the
        #: arena does not have to be on the device for the *compute* to be fast -- it only has to
        #: be reachable.  262144 positions of k and v are 3.2 GiB a rank at this model's geometry,
        #: against the ~3 GiB a fully resident expert bank leaves; the same arena in host memory
        #: costs a 2 MiB gather a layer a step, which a step hides.
        host_kv = os.environ.get("POCKETLLM_QWEN4_HOST_KV") == "1"
        arena = dict(device="cpu" if host_kv else device, dtype=dtype)
        if host_kv:
            arena["pin_memory"] = True
        self.k = torch.zeros(batch_size, kv_heads, max_seq_len, config.head_dim, **arena)
        self.v = torch.zeros(batch_size, kv_heads, max_seq_len, config.head_dim, **arena)
        self.host_kv = host_kv
        #: Where the next write lands, as a plain integer.  A host-resident arena cannot be written
        #: through the tensor-position path -- that one indexes with a device tensor -- and reading
        #: the tensor back to get an integer would be a synchronisation, which is exactly what a
        #: capture cannot contain.  The caller that drives the step already knows the position as
        #: an integer, so it hands it over here instead.
        self.host_position = 0
        self.index_k = torch.zeros(
            batch_size, max_seq_len, config.indexer_head_dim, device=device, dtype=dtype
        )
        self.length = 0


class QSAIndexer:
    """Picks a token budget per query from mean-pooled key blocks.

    Keys are averaged in groups of `compress_ratio`, RoPE'd at the group start
    position, then scored against the (multi-head, ReLU-summed) index queries.
    The top `budget / compress_ratio` blocks expand back to token indices; the
    ragged tail beyond the last complete block is always visible.
    """

    def __init__(self, config: Qwen4ExpTextConfig, weights: dict[str, torch.Tensor], *, n_heads: int | None = None) -> None:
        self.n_heads = config.indexer_n_heads if n_heads is None else n_heads
        self.kv_heads = config.indexer_kv_heads
        self.head_dim = config.indexer_head_dim
        # The sparse budget is the knob that trades retrieval breadth for a step's cost: the layer
        # reads `budget` positions whatever the context is, so halving it halves the attention and
        # the staging.  Overridable so the trade can be priced against quality rather than assumed.
        override = os.environ.get("POCKETLLM_QWEN4_INDEXER_BUDGET")
        self.budget = int(override) if override else config.indexer_budget
        self.compress_ratio = config.indexer_compress_ratio
        self.block_topk = self.budget // self.compress_ratio
        self.qk_proj = weights["index_qk_proj"]
        self.q_layernorm = RMSNorm(weights["q_layernorm"], config.rms_norm_eps)
        self.k_layernorm = RMSNorm(weights["k_layernorm"], config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: QSAAttentionCache | None,
        *,
        past_len: int,
        raw_keys_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return padded selected token indices with ``-1`` marking invalid slots."""
        batch_size, seq_len, _ = hidden_states.shape
        qk = prefill_linear(hidden_states, self.qk_proj)
        q, token_k = torch.split(
            qk, [self.n_heads * self.head_dim, self.kv_heads * self.head_dim], dim=-1
        )
        q = self.q_layernorm(q.reshape(batch_size, seq_len, -1, self.head_dim))
        q = apply_rotary_pos_emb(q, cos[:, -seq_len:], sin[:, -seq_len:], unsqueeze_dim=2)
        new_raw_keys = token_k.reshape(batch_size, seq_len, self.head_dim)

        if cache is not None:
            # Same reason as the QSA write: a Python slice pins the offset into anything that
            # records it, so a tensor position goes through `index_copy_` instead.
            if isinstance(past_len, torch.Tensor):
                offsets = past_len.reshape(()) + torch.arange(seq_len, device=hidden_states.device)
                for row in range(batch_size):
                    cache.index_k[row].index_copy_(0, offsets, new_raw_keys[row])
                width = _kv_width_of(cache)
                if width > 0:
                    raw_keys = cache.index_k[:, :width]
                else:
                    # `cache.length` is still the previous step's: this layer runs before the
                    # attention that advances it, so narrowing to it drops the key just written.
                    # The integer path reads `past_len + seq_len`, and this has to match it.
                    raw_keys = cache.index_k[:, : int(past_len.reshape(())) + seq_len]
            else:
                cache.index_k[:, past_len : past_len + seq_len] = new_raw_keys
                width = _kv_width_of(cache)
                raw_keys = (cache.index_k[:, :width] if width > 0
                            else cache.index_k[:, : past_len + seq_len])
        elif raw_keys_override is not None:
            raw_keys = torch.cat([raw_keys_override, new_raw_keys], dim=1)
        else:
            raw_keys = new_raw_keys

        kv_len = raw_keys.shape[1]
        # Every query at absolute position p sees keys [0, p]; block boundaries
        # are therefore query-dependent only through how many keys are visible.
        num_blocks_total = kv_len // self.compress_ratio
        if kv_len <= self.budget:
            selected_width = kv_len
            key_ids = torch.arange(kv_len, device=hidden_states.device)
            visible_keys = past_len + torch.arange(seq_len, device=hidden_states.device) + 1
            indices = key_ids.view(1, 1, -1).expand(batch_size, seq_len, -1)
            return indices.masked_fill(
                key_ids.view(1, 1, -1) >= visible_keys.view(1, -1, 1), -1
            ).to(torch.int32)

        selected_width = min(
            kv_len,
            self.budget + self.compress_ratio - 1,
        )
        if num_blocks_total == 0:
            key_ids = torch.arange(kv_len, device=hidden_states.device)
            visible_keys = past_len + torch.arange(seq_len, device=hidden_states.device) + 1
            indices = key_ids.view(1, 1, -1).expand(batch_size, seq_len, -1)
            indices = indices.masked_fill(
                key_ids.view(1, 1, -1) >= visible_keys.view(1, -1, 1), -1
            )
            if kv_len < selected_width:
                indices = F.pad(indices, (0, selected_width - kv_len), value=-1)
            return indices[:, :, :selected_width].to(torch.int32)

        # Pool + RoPE all complete blocks once (shared across queries).
        block_tokens = torch.arange(
            num_blocks_total * self.compress_ratio, device=hidden_states.device
        ).view(num_blocks_total, self.compress_ratio)
        pooled = raw_keys[:, : num_blocks_total * self.compress_ratio].reshape(
            batch_size, num_blocks_total, self.compress_ratio, self.head_dim
        )
        pooled = pooled.float().mean(dim=2).to(raw_keys.dtype)
        pooled = self.k_layernorm(pooled)
        # cos/sin cover every absolute position in the cache, so a block's RoPE
        # phase comes from its first token's position.
        group_starts = block_tokens[:, 0]
        block_keys = apply_rotary_pos_emb(
            pooled.unsqueeze(2), cos[:, group_starts], sin[:, group_starts], unsqueeze_dim=2
        ).squeeze(2)  # (batch, blocks, head_dim)

        # scores: (batch, seq, blocks) = sum over index heads of relu(q . k)
        scores = torch.einsum("bqhd,bkd->bqhk", q.float(), block_keys.float())
        scores = torch.relu(scores).sum(dim=2) / math.sqrt(self.head_dim)

        query_abs = past_len + torch.arange(seq_len, device=hidden_states.device)
        visible_keys = query_abs + 1  # causal: keys [0, p] inclusive
        blocks_visible = visible_keys // self.compress_ratio  # per-query complete-block count
        block_ids = torch.arange(num_blocks_total, device=hidden_states.device)
        block_ok = block_ids.view(1, -1) < blocks_visible.view(-1, 1)  # (seq, blocks)
        scores = scores.masked_fill(~block_ok.unsqueeze(0), float("-inf"))

        topk = min(self.block_topk, num_blocks_total)
        selected_blocks = scores.topk(topk, dim=-1).indices  # (batch, seq, topk)
        # -inf rows mean "fewer visible blocks than topk"; drop those picks.
        selected_valid = torch.gather(block_ok.unsqueeze(0).expand(batch_size, -1, -1), 2, selected_blocks)

        block_expand = block_tokens[selected_blocks]  # (batch, seq, topk, compress_ratio)
        valid_expand = selected_valid.unsqueeze(-1).expand_as(block_expand)
        selected_tokens = block_expand.reshape(batch_size, seq_len, -1)
        selected_tokens = selected_tokens.masked_fill(
            ~valid_expand.reshape(batch_size, seq_len, -1), -1
        )

        # Append every possible partial-block tail slot. Invalid positions stay -1,
        # which keeps the output rectangular without constructing a dense mask.
        tail_offsets = torch.arange(
            self.compress_ratio - 1, device=hidden_states.device
        ).view(1, 1, -1)
        tail_start = (blocks_visible * self.compress_ratio).view(1, seq_len, 1)
        tail_tokens = tail_start + tail_offsets
        tail_valid = tail_tokens < visible_keys.view(1, seq_len, 1)
        tail_tokens = tail_tokens.expand(batch_size, -1, -1).masked_fill(
            ~tail_valid.expand(batch_size, -1, -1), -1
        )
        selected_tokens = torch.cat([selected_tokens, tail_tokens], dim=-1)
        if selected_tokens.shape[-1] < selected_width:
            selected_tokens = F.pad(
                selected_tokens,
                (0, selected_width - selected_tokens.shape[-1]),
                value=-1,
            )
        return selected_tokens[:, :, :selected_width].to(torch.int32)


class QSAAttention:
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        weights: dict[str, torch.Tensor],
        *,
        num_heads: int | None = None,
        num_kv_heads: int | None = None,
        indexer_heads: int | None = None,
    ) -> None:
        self.config = config
        self.num_heads = config.num_attention_heads if num_heads is None else num_heads
        self.num_kv_heads = config.num_key_value_heads if num_kv_heads is None else num_kv_heads
        self.head_dim = config.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = weights["q_proj"]  # (heads * head_dim * 2, hidden): value+gate packed
        self.k_proj = weights["k_proj"]
        self.v_proj = weights["v_proj"]
        self.o_proj = weights["o_proj"]
        self.q_norm = RMSNorm(weights["q_norm"], config.rms_norm_eps)
        self.k_norm = RMSNorm(weights["k_norm"], config.rms_norm_eps)
        self.indexer = QSAIndexer(config, weights, n_heads=indexer_heads)
        self.last_profile: dict[str, float] = {}

    def __call__(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache: QSAAttentionCache | None,
        *,
        past_len: int,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape
        profile = os.environ.get("POCKETLLM_QWEN4_ATTN_PHASE_PROFILE", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        if profile:
            import time

            torch.cuda.synchronize(hidden_states.device)
            started = time.perf_counter()
        # The indexer rotates each pooled block by its *absolute* position, so it needs the whole
        # table.  What the caller passes is the rows this step computes -- a slice when the
        # position is an integer, and just the current rows when it is a tensor -- which is right
        # for the query and wrong for the indexer.  Past the indexer budget that difference is an
        # out-of-bounds index rather than a wrong phase, which is how it was found.
        table_cos = cache.cos if cache is not None and getattr(cache, "cos", None) is not None else cos
        table_sin = cache.sin if cache is not None and getattr(cache, "sin", None) is not None else sin
        selected = self.indexer(hidden_states, table_cos, table_sin, cache, past_len=past_len)
        if profile:
            torch.cuda.synchronize(hidden_states.device)
            after_indexer = time.perf_counter()

        qg = prefill_linear(hidden_states, self.q_proj).view(batch_size, seq_len, -1, self.head_dim * 2)
        query, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(batch_size, seq_len, -1)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(
            prefill_linear(hidden_states, self.k_proj).view(batch_size, seq_len, -1, self.head_dim)
        ).transpose(1, 2)
        value = prefill_linear(hidden_states, self.v_proj).view(batch_size, seq_len, -1, self.head_dim).transpose(1, 2)

        cur_cos, cur_sin = cos[:, -seq_len:], sin[:, -seq_len:]
        query, key = apply_rotary_pos_emb(query, cur_cos, cur_sin, k=key, unsqueeze_dim=1)

        if cache is not None:
            _write_kv(cache, key, value, past_len, seq_len)
            cache.length = past_len + seq_len
            if getattr(cache, "host_kv", False):
                key, value, selected = _stage_selected(cache, selected, hidden_states.device)
            else:
                width = _kv_width_of(cache)
                if width > 0:
                    key = cache.k[:, :, :width]
                    value = cache.v[:, :, :width]
                else:
                    key = cache.k[:, :, : cache.length]
                    value = cache.v[:, :, : cache.length]
        if profile:
            torch.cuda.synchronize(hidden_states.device)
            after_projection = time.perf_counter()

        indexed_out = cuda_qsa_attention(
            query, key, value, selected, self.scaling
        )
        if indexed_out is None:
            # Bound fallback workspace by query rows and keep the KV-head axis
            # explicit. Repeating gathered K/V across every query head allocates
            # several GiB even at 2K and can OOM before the optional extension is
            # available.
            query_rows = max(
                1,
                int(os.environ.get("POCKETLLM_QWEN4_QSA_FALLBACK_ROWS", "16")),
            )
            key_tokens = key.transpose(1, 2)
            value_tokens = value.transpose(1, 2)
            chunks = []
            for begin in range(0, seq_len, query_rows):
                end = min(seq_len, begin + query_rows)
                selected_chunk = selected[:, begin:end]
                safe_selected = selected_chunk.clamp_min(0).to(torch.long)
                batch_index = torch.arange(
                    batch_size, device=hidden_states.device
                ).view(batch_size, 1, 1).expand_as(safe_selected)
                gathered_key = key_tokens[batch_index, safe_selected]
                gathered_value = value_tokens[batch_index, safe_selected]
                query_chunk = query.transpose(1, 2)[:, begin:end]
                query_chunk = query_chunk.unflatten(
                    2,
                    (self.num_kv_heads, self.num_kv_groups),
                )
                scores = torch.einsum(
                    "bqkgd,bqskd->bqkgs",
                    query_chunk,
                    gathered_key,
                ) * self.scaling
                scores = scores.masked_fill(
                    (selected_chunk < 0).unsqueeze(2).unsqueeze(2),
                    torch.finfo(scores.dtype).min,
                )
                probabilities = F.softmax(
                    scores,
                    dim=-1,
                    dtype=torch.float32,
                ).to(query.dtype)
                chunk = torch.einsum(
                    "bqkgs,bqskd->bqkgd",
                    probabilities,
                    gathered_value,
                )
                chunks.append(
                    chunk.flatten(2, 3)
                )
            indexed_out = torch.cat(chunks, dim=1)
        attn_out = indexed_out.reshape(batch_size, seq_len, -1)
        if profile:
            torch.cuda.synchronize(hidden_states.device)
            after_core = time.perf_counter()

        attn_out = attn_out * torch.sigmoid(gate)
        output = prefill_linear(attn_out, self.o_proj)
        if profile:
            torch.cuda.synchronize(hidden_states.device)
            finished = time.perf_counter()
            self.last_profile = {
                "indexer": after_indexer - started,
                "projection": after_projection - after_indexer,
                "core": after_core - after_projection,
                "output": finished - after_core,
            }
        return output
