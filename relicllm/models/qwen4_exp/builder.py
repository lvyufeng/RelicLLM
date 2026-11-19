"""Assemble a `Qwen4ExpModel` from checkpoint tensors.

Two entry points:

* `build_from_state_dict` — everything resident on one device. Used by the
  parity tests against upstream `transformers` goldens.
* `build_heterogeneous` — dense weights sharded across GPUs, routed experts and
  the PLE table left in host RAM. This is the TP4 path for the real 335 GiB
  checkpoint.
"""

from __future__ import annotations

import os

import torch

from relicllm.models.qwen4_exp.config import Qwen4ExpTextConfig
from relicllm.models.qwen4_exp.layers import GatedResidual
from relicllm.models.qwen4_exp.model import (
    PLELayer,
    Qwen4ExpDecoderLayer,
    Qwen4ExpModel,
    TPShard,
)
from relicllm.models.qwen4_exp.moe import (
    DeepCacheMoE,
    DeepFallbackMoE,
    DeviceExpertMoE,
    Fp8SlotExpertMoE,
    NvFp4SlotExpertMoE,
    HostExpertMoE,
    InMemoryMoE,
    ShardedMoE,
)
from relicllm.models.qwen4_exp.weights import (
    DeviceEmbedding,
    HostEmbedding,
    HostNGramTable,
    Qwen4ExpCheckpoint,
)

LM = "model.language_model"

#: Room the MTP pin leaves on a card for everything else.  The pin is a copy of one layer's
#: experts, but building it needs the source rows and the target buffer at once.
_MTP_PIN_HEADROOM_BYTES = 3 << 30

# Layer-local tensor suffixes, grouped by the sub-module that consumes them.
_HC_SUFFIXES = ("hc_norm", "input_mix_weight_down", "input_mix_weight_up", "block_inject_weight")
_LINEAR_ATTN_SUFFIXES = (
    "in_proj_qkv.weight",
    "in_proj_z.weight",
    "in_proj_b.weight",
    "in_proj_a.weight",
    "conv1d.weight",
    "dt_bias",
    "A_log",
    "out_proj.weight",
    "norm.weight",
)
_QSA_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "o_proj.weight",
    "q_norm.weight",
    "k_norm.weight",
    "indexer.index_qk_proj.weight",
    "indexer.q_layernorm.weight",
    "indexer.k_layernorm.weight",
)
_PLE_SUFFIXES = (
    "key_proj.weight",
    "value_proj.weight",
    "norm_key.weight",
    "norm_query.weight",
    "norm_conv.weight",
    "conv1d.weight",
)


def _strip_weight(name: str) -> str:
    return name[: -len(".weight")] if name.endswith(".weight") else name


def collect_layer_weights(
    getter,
    config: Qwen4ExpTextConfig,
    layer_idx: int,
    *,
    prefix: str = LM,
) -> dict[str, torch.Tensor]:
    """Gather one decoder layer's non-expert tensors into short keys.

    `getter(full_name)` returns a tensor already on the target device/dtype.
    Keys come back as e.g. `linear_attn.in_proj_qkv`, `self_attn.q_norm`,
    `attn_hyper_connection.hc_norm`, `mlp.gate`.
    """
    base = f"{prefix}.layers.{layer_idx}"
    out: dict[str, torch.Tensor] = {}

    for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
        for suffix in _HC_SUFFIXES:
            out[f"{hc}.{suffix}"] = getter(f"{base}.{hc}.{suffix}.weight")

    if config.is_linear_layer(layer_idx):
        for suffix in _LINEAR_ATTN_SUFFIXES:
            out[f"linear_attn.{_strip_weight(suffix)}"] = getter(f"{base}.linear_attn.{suffix}")
    else:
        for suffix in _QSA_SUFFIXES:
            short = _strip_weight(suffix)
            # The indexer's tensors are consumed by QSAAttention's weight dict
            # under their bare names.
            short = short[len("indexer.") :] if short.startswith("indexer.") else short
            out[f"self_attn.{short}"] = getter(f"{base}.self_attn.{suffix}")

    out["mlp.gate"] = getter(f"{base}.mlp.gate.weight")
    out["mlp.shared_expert_gate"] = getter(f"{base}.mlp.shared_expert_gate.weight")
    for proj in ("gate_proj", "up_proj", "down_proj"):
        out[f"mlp.shared_expert.{proj}"] = getter(f"{base}.mlp.shared_expert.{proj}.weight")
    return out


def _ple_weights(getter, layer_idx: int, *, prefix: str = LM) -> dict[str, torch.Tensor]:
    base = f"{prefix}.layers.{layer_idx}.ple"
    return {_strip_weight(s): getter(f"{base}.{s}") for s in _PLE_SUFFIXES}


def layer_range_for(n_layers: int, pp_size: int, pp_rank: int) -> tuple[int, int]:
    """The ``[start, end)`` layer slice one pipeline stage owns.

    Balanced and contiguous: stage ``i`` takes layers ``[i*n//p, (i+1)*n//p)``, so the first
    ``n % p`` stages carry one extra layer. Contiguous rather than strided because each stage
    preloads its experts as one sequential read of the checkpoint, and because a stage's layers are
    the unit the PLE tables and the KV caches are grouped by.

    It lives here, in the module that actually lays the layers out, so the checkpoint preload and
    the model build cannot disagree about which layers a stage owns.
    """
    if pp_size <= 1:
        return 0, n_layers
    return (pp_rank * n_layers) // pp_size, ((pp_rank + 1) * n_layers) // pp_size


def _tp_shard(
    config: Qwen4ExpTextConfig, world_size: int, rank: int
) -> tuple[TPShard, dict[str, int]]:
    """This rank's heads, and the kv window that binds them.

    Pulled out of `build_heterogeneous` so the MTP head is divided by the same rule rather than by a
    copy of it: the head's output goes through the same lm_head, so a rank that took a different
    slice of q heads would contribute a differently-sharded partial to the same all-reduce.

    GQA: a q head is bound to the kv head `q // num_kv_groups`, so slicing q heads dictates which kv
    heads this rank needs.  With 24 q / 2 kv on 4 ranks each rank owns 6 q heads that all share one
    kv head, so kv is sliced (not replicated) and the group count stays consistent.
    """
    q_per_rank = config.num_attention_heads // world_size
    global_groups = config.num_attention_heads // config.num_key_value_heads
    kv_start = (rank * q_per_rank) // global_groups
    kv_end = ((rank + 1) * q_per_rank - 1) // global_groups + 1
    kv_per_rank = kv_end - kv_start
    if q_per_rank % kv_per_rank != 0:
        raise ValueError(
            f"TP world_size={world_size} splits {config.num_attention_heads} q heads into "
            f"{q_per_rank} per rank, which does not group evenly onto {kv_per_rank} kv heads"
        )
    shard = TPShard(
        attn_heads=q_per_rank,
        attn_kv_heads=kv_per_rank,
        # The indexer's score is a sum over its heads and its mask must be
        # identical on every rank, so it is replicated rather than sharded.
        indexer_heads=config.indexer_n_heads or 0,
        linear_v_heads=config.linear_num_value_heads // world_size,
        linear_k_heads=config.linear_num_key_heads // world_size,
    )
    return shard, {"kv_start": kv_start, "kv_end": kv_end}


def _shard_layer_dense(
    store_view,
    base: str,
    config: Qwen4ExpTextConfig,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    world_size: int,
    kv: dict[str, int],
    *,
    is_linear: bool,
) -> dict[str, torch.Tensor]:
    """One decoder layer's non-expert tensors, sharded for `rank` of `world_size`.

    `store_view(name)` returns the checkpoint's tensor.  Shared by the main stack and the MTP head:
    the dense half of an MTP layer is an ordinary QSA layer, and two spellings of its sharding is
    exactly how the head would end up reading a different slice of q/o than the layers whose state
    it drafts from.
    """
    key_dim = config.key_dim
    v_dim = config.value_dim
    k_per = key_dim // world_size
    v_per = v_dim // world_size

    def dense(name: str) -> torch.Tensor:
        return store_view(name).to(device, dtype=dtype)

    def shard_rows(name: str, groups: int, group_size: int) -> torch.Tensor:
        """Split a [out, in] weight's output rows into `groups` head-groups and keep this rank's."""
        full = store_view(name)
        per_rank = groups // world_size
        start = rank * per_rank * group_size
        end = start + per_rank * group_size
        return full[start:end].to(device, dtype=dtype)

    def shard_cols(name: str, groups: int, group_size: int) -> torch.Tensor:
        """Split a [out, in] weight's input columns to match a row-sharded input."""
        full = store_view(name)
        per_rank = groups // world_size
        start = rank * per_rank * group_size
        end = start + per_rank * group_size
        return full[:, start:end].to(device, dtype=dtype)

    weights: dict[str, torch.Tensor] = {}
    for hc in ("attn_hyper_connection", "mlp_hyper_connection"):
        for suffix in _HC_SUFFIXES:
            weights[f"{hc}.{suffix}"] = dense(f"{base}.{hc}.{suffix}.weight")

    if is_linear:
        # qkv is packed [k_dim, k_dim, v_dim]; shard each section by heads.
        qkv = store_view(f"{base}.linear_attn.in_proj_qkv.weight")
        weights["linear_attn.in_proj_qkv"] = torch.cat(
            [
                qkv[rank * k_per : (rank + 1) * k_per],
                qkv[key_dim + rank * k_per : key_dim + (rank + 1) * k_per],
                qkv[2 * key_dim + rank * v_per : 2 * key_dim + (rank + 1) * v_per],
            ],
            dim=0,
        ).to(device, dtype=dtype)
        weights["linear_attn.in_proj_z"] = shard_rows(
            f"{base}.linear_attn.in_proj_z.weight",
            config.linear_num_value_heads,
            config.linear_value_head_dim,
        )
        weights["linear_attn.in_proj_b"] = shard_rows(
            f"{base}.linear_attn.in_proj_b.weight", config.linear_num_value_heads, 1
        )
        weights["linear_attn.in_proj_a"] = shard_rows(
            f"{base}.linear_attn.in_proj_a.weight", config.linear_num_value_heads, 1
        )
        # conv1d is depthwise over the packed conv_dim, so it shards the same way as in_proj_qkv.
        conv = store_view(f"{base}.linear_attn.conv1d.weight")
        weights["linear_attn.conv1d"] = torch.cat(
            [
                conv[rank * k_per : (rank + 1) * k_per],
                conv[key_dim + rank * k_per : key_dim + (rank + 1) * k_per],
                conv[2 * key_dim + rank * v_per : 2 * key_dim + (rank + 1) * v_per],
            ],
            dim=0,
        ).to(device, dtype=dtype)
        v_head_per_rank = config.linear_num_value_heads // world_size
        weights["linear_attn.dt_bias"] = store_view(f"{base}.linear_attn.dt_bias")[
            rank * v_head_per_rank : (rank + 1) * v_head_per_rank
        ].to(device, dtype=dtype)
        weights["linear_attn.A_log"] = store_view(f"{base}.linear_attn.A_log")[
            rank * v_head_per_rank : (rank + 1) * v_head_per_rank
        ].to(device, dtype=dtype)
        # norm is per-head-dim (shared across heads), so it stays whole.
        weights["linear_attn.norm"] = dense(f"{base}.linear_attn.norm.weight")
        weights["linear_attn.out_proj"] = shard_cols(
            f"{base}.linear_attn.out_proj.weight",
            config.linear_num_value_heads,
            config.linear_value_head_dim,
        )
    else:
        # q_proj packs [value, gate] per head, so its head stride is 2*head_dim.
        weights["self_attn.q_proj"] = shard_rows(
            f"{base}.self_attn.q_proj.weight", config.num_attention_heads, config.head_dim * 2
        )
        # Slice the kv heads this rank's q heads actually attend to.
        kv_lo = kv["kv_start"] * config.head_dim
        kv_hi = kv["kv_end"] * config.head_dim
        weights["self_attn.k_proj"] = store_view(f"{base}.self_attn.k_proj.weight")[kv_lo:kv_hi].to(
            device, dtype=dtype
        )
        weights["self_attn.v_proj"] = store_view(f"{base}.self_attn.v_proj.weight")[kv_lo:kv_hi].to(
            device, dtype=dtype
        )
        weights["self_attn.o_proj"] = shard_cols(
            f"{base}.self_attn.o_proj.weight", config.num_attention_heads, config.head_dim
        )
        weights["self_attn.q_norm"] = dense(f"{base}.self_attn.q_norm.weight")
        weights["self_attn.k_norm"] = dense(f"{base}.self_attn.k_norm.weight")
        # The indexer's scores are a sum over its heads, so sharding heads would need its own
        # reduce. Keep it replicated: 640x2560 per layer.
        weights["self_attn.index_qk_proj"] = dense(f"{base}.self_attn.indexer.index_qk_proj.weight")
        weights["self_attn.q_layernorm"] = dense(f"{base}.self_attn.indexer.q_layernorm.weight")
        weights["self_attn.k_layernorm"] = dense(f"{base}.self_attn.indexer.k_layernorm.weight")

    weights["mlp.gate"] = dense(f"{base}.mlp.gate.weight")
    weights["mlp.shared_expert_gate"] = dense(f"{base}.mlp.shared_expert_gate.weight")
    # Shared expert: column-parallel gate/up, row-parallel down, so its partial sums fold into the
    # same all-reduce as the routed experts.
    inter_per_rank = config.shared_expert_intermediate_size // world_size
    gp = store_view(f"{base}.mlp.shared_expert.gate_proj.weight")
    up = store_view(f"{base}.mlp.shared_expert.up_proj.weight")
    dn = store_view(f"{base}.mlp.shared_expert.down_proj.weight")
    lo, hi = rank * inter_per_rank, (rank + 1) * inter_per_rank
    weights["mlp.shared_expert.gate_proj"] = gp[lo:hi].to(device, dtype=dtype)
    weights["mlp.shared_expert.up_proj"] = up[lo:hi].to(device, dtype=dtype)
    weights["mlp.shared_expert.down_proj"] = dn[:, lo:hi].to(device, dtype=dtype)
    return weights


def _final_mixer(getter, config: Qwen4ExpTextConfig, *, prefix: str = LM) -> GatedResidual:
    base = f"{prefix}.hyper_connection_mixer"
    return GatedResidual(
        config,
        getter(f"{base}.hc_norm.weight"),
        getter(f"{base}.input_mix_weight_down.weight"),
        getter(f"{base}.input_mix_weight_up.weight"),
        None,
    )


def _mtp_shard_config(config: Qwen4ExpTextConfig) -> Qwen4ExpTextConfig:
    """`config`, with layer 0 declared as the MTP block's attention flavour.

    Only `_tp_shard` and `_shard_layer_dense` read the flavour, and both are asked about layer 0.
    A shallow copy with its own one-entry `layer_types` is the whole of the override.
    """
    import copy

    out = copy.copy(config)
    out.layer_types = list(getattr(config, "mtp_layer_types", None) or ["qwen_sparse_attention"])
    return out


def _mtp_dense_weights(
    checkpoint: Qwen4ExpCheckpoint,
    config: Qwen4ExpTextConfig,
    dtype: torch.dtype,
    device: torch.device,
    rank: int,
    world_size: int,
    kv_window: dict[str, int],
    prefix: str,
) -> dict[str, torch.Tensor]:
    """The MTP block's dense tensors, sharded by the main layers' rules, in a layer's short keys."""
    is_linear = _mtp_shard_config(config).is_linear_layer(0)
    weights = _shard_layer_dense(
        checkpoint.store.view,
        f"{prefix}.layers.0",
        config,
        dtype,
        device,
        rank,
        world_size,
        kv_window,
        is_linear=is_linear,
    )
    # The two fusion projections and the mixer that feeds lm_head.  Replicated rather than sharded:
    # `pre_fc_norm_hidden` normalizes whole streams that every rank already holds in full, and the
    # mixer's output is reduced by the same all-reduce the layer's MLP block was.  Together they are
    # 3 x 2560 x 2560 bf16 plus the low-rank mixer, 40 MiB a rank.
    for name in ("pre_fc_norm_hidden", "pre_fc_norm_embedding"):
        weights[name] = checkpoint.store.view(f"{prefix}.{name}.weight").to(device, dtype=dtype)
    for name in ("fc_hidden", "fc_embedding"):
        weights[name] = checkpoint.store.view(f"{prefix}.{name}.weight").to(device, dtype=dtype)
    mixer = f"{prefix}.hyper_connection_mixer"
    for suffix in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up"):
        weights[f"hyper_connection_mixer.{suffix}"] = checkpoint.store.view(
            f"{mixer}.{suffix}.weight"
        ).to(device, dtype=dtype)
    return weights


class _MTPExpertReader:
    """Serves the MTP layer's routed experts to a `HostExpertMoE` as if they were layer 0's.

    The MTP block is one layer and it is not `model.language_model.layers.0`, so its expert rows do
    not live where every other reader looks. Rather than teach `HostExpertMoE` a second naming
    scheme -- which would put a checkpoint layout decision inside the staging policy -- this is the
    reader seam it already takes, with the layer id ignored. That is also what keeps the ownership
    rule identical to the main layers': the inner `HostExpertMoE` wraps this in the same
    `ShardedMoE`, which masks by `expert_id % world_size == rank` before the reader is ever asked.

    `mtp_layer_idx` is what the checkpoint's MTP stack numbers its layers by.  The `layer_idx` a
    caller passes in is ignored: the head has one layer and there is nothing to disambiguate.
    """

    def __init__(self, checkpoint, *, mtp_layer_idx: int = 0) -> None:
        self.checkpoint = checkpoint
        self.mtp_layer_idx = mtp_layer_idx
        #: `HostExpertMoE.pin_layers` asks the shard which ids this rank owns; answering None makes
        #: it consider all of them, which is right here because the MTP shard is its own table.
        self.expert_shard = None

    def expert_rows(self, layer_idx: int, expert_id: int):
        return self.checkpoint.mtp_expert_rows(self.mtp_layer_idx, expert_id)


def build_mtp_head(
    config: Qwen4ExpTextConfig,
    checkpoint: Qwen4ExpCheckpoint,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    rank: int = 0,
    world_size: int = 1,
    all_reduce=None,
    mtp_layer_idx: int = 0,
    prefix: str = "mtp",
    pin_experts: int = 0,
):
    """Load the MTP block and its expert backend, sharded the same way the main layers are.

    The dense half is sharded by exactly the rules `build_heterogeneous` uses (q heads, kv heads
    derived from them, the indexer replicated, the shared expert column/row parallel) and reduced by
    the same `all_reduce`, because the head's output goes through the same lm_head and therefore has
    to be the same full-vocabulary quantity on every rank.

    The expert half is deliberately *not* the main stack's backend object.  Those experts live under
    `mtp.layers.0`, with their own rows and their own format -- on the ModelOpt NVFP4 checkpoint the
    main experts are NVFP4 and the MTP experts are still FP8 -- so the head gets its own
    `HostExpertMoE` over `_MTPExpertReader`, wrapped in the same `ShardedMoE`.
    """
    from relicllm.models.qwen4_exp.mtp import MTPHead  # noqa: PLC0415 -- avoids a cycle at import

    device = torch.device(device)
    config_for_layer = _mtp_shard_config(config)
    shard, kv_window = _tp_shard(config_for_layer, world_size, rank)

    weights = _mtp_dense_weights(
        checkpoint, config, dtype, device, rank, world_size, kv_window, prefix
    )
    reader = _MTPExpertReader(checkpoint, mtp_layer_idx=mtp_layer_idx)
    inner = HostExpertMoE(reader, config.num_experts, device, dtype)
    if pin_experts:
        # The head runs once a round under speculative decoding, and an unpinned head reads its
        # experts over PCIe on every one of those calls.  It is one layer -- ~0.30 GiB a rank at
        # TP4, ~0.4 GiB of FP8 codes at TP4 on the NVFP4 checkpoint -- so pinning it is cheap
        # *relative to a card with room*, and not affordable on one without: the 4x2080 Ti box runs
        # the trunk's whole NVFP4 bank resident and has ~1.7 GiB left, which the pin plus its load
        # transient does not fit in.
        #
        # So the budget is what the card can actually spare, taken now rather than promised by the
        # caller, and a head that does not fit is left staging: staging is slower, not wrong, and
        # `mtp_accuracy` does not care which it got.
        # The pin has to fit *and* leave the run somewhere to work: on the 4x2080 Ti box the
        # trunk's NVFP4 bank takes 20.1 of 21.48 GiB, a 0.4 GiB pin lands, and the first
        # activation afterwards is the 20 MiB that runs out.  So a pin is only taken when the card
        # has room for its own copy plus a reserve, rather than whenever it has room for the copy.
        if device.type == "cuda":
            free = int(torch.cuda.mem_get_info(device)[0]) - _MTP_PIN_HEADROOM_BYTES
        else:
            free = 0
        budget = min(int(pin_experts), max(0, free))
        if budget > 0:
            inner.pin_layers([mtp_layer_idx], budget_bytes=budget)
        elif rank == 0 and torch.distributed.is_initialized():
            spare = int(torch.cuda.mem_get_info(device)[0]) if device.type == "cuda" else 0
            print(
                f"[qwen4exp] mtp experts stay in host RAM: only {spare / (1 << 20):.0f} MiB "
                f"spare, and pinning needs {_MTP_PIN_HEADROOM_BYTES / (1 << 30):.0f} GiB of "
                "headroom left over for the run itself",
                flush=True,
            )
    moe = ShardedMoE(inner, rank, world_size) if world_size > 1 else inner
    return MTPHead(
        config,
        weights,
        moe,
        device=device,
        tp_shard=shard,
        all_reduce=all_reduce,
    )


class _TensorStore:
    """Adapts a plain dict of tensors to the `getter` protocol."""

    def __init__(self, tensors: dict[str, torch.Tensor], device, dtype) -> None:
        self.tensors = tensors
        self.device = device
        self.dtype = dtype

    def __call__(self, name: str) -> torch.Tensor:
        t = self.tensors[name]
        # Integer buffers (n-gram multipliers, vocab sizes) must keep their dtype.
        if t.is_floating_point():
            return t.to(self.device, dtype=self.dtype)
        return t.to(self.device)


def build_from_state_dict(
    config: Qwen4ExpTextConfig,
    state_dict: dict[str, torch.Tensor],
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    prefix: str = LM,
    profiler=None,
) -> Qwen4ExpModel:
    """Single-device model with every expert resident. Small models / tests only."""
    device = torch.device(device)
    getter = _TensorStore(state_dict, device, dtype)

    gate_up = {}
    down = {}
    for layer_idx in range(config.num_hidden_layers):
        gate_up[layer_idx] = getter(f"{prefix}.layers.{layer_idx}.mlp.experts.gate_up_proj")
        down[layer_idx] = getter(f"{prefix}.layers.{layer_idx}.mlp.experts.down_proj")
    moe = InMemoryMoE(config.num_experts, gate_up, down)

    layers = []
    for layer_idx in range(config.num_hidden_layers):
        ple = None
        ple_index = config.ple_layer_index(layer_idx)
        if ple_index is not None:
            table_key = f"{prefix}.layers.{layer_idx}.ple.ple_embedding.ngram_embedding.weight"
            shards = [state_dict[table_key]] if table_key in state_dict else _gather_shards(
                state_dict, f"{prefix}.layers.{layer_idx}.ple.ple_embedding.ngram_embedding"
            )
            ple_scale = (
                state_dict.get(f"{prefix}.layers.{layer_idx}.ple.ple_embedding.ngram_embedding.weight_scale")
                if hasattr(state_dict, "get")
                else None
            )
            table = HostNGramTable(shards, device=device, dtype=dtype, scale=ple_scale)
            ple = PLELayer(
                config,
                _ple_weights(getter, layer_idx, prefix=prefix),
                ple_index,
                device,
                ngram_table=table,
            )
        layers.append(
            Qwen4ExpDecoderLayer(
                config,
                layer_idx,
                collect_layer_weights(getter, config, layer_idx, prefix=prefix),
                moe,
                device=device,
                ple=ple,
            )
        )

    embed_weight = getter(f"{prefix}.embed_tokens.weight")
    return Qwen4ExpModel(
        config,
        layers,
        embed_tokens=DeviceEmbedding(embed_weight),
        lm_head=getter("lm_head.weight"),
        final_mixer=_final_mixer(getter, config, prefix=prefix),
        device=device,
        dtype=dtype,
        profiler=profiler,
    )


def _gather_shards(state_dict: dict[str, torch.Tensor], base: str) -> list[torch.Tensor]:
    keys = [k for k in state_dict if k.startswith(f"{base}.shard_")]
    keys.sort(key=lambda k: int(k.split("shard_")[1].split(".")[0]))
    if not keys:
        raise KeyError(f"no n-gram embedding shards under {base}")
    return [state_dict[k] for k in keys]


def build_heterogeneous(
    config: Qwen4ExpTextConfig,
    checkpoint: Qwen4ExpCheckpoint,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    rank: int = 0,
    world_size: int = 1,
    all_reduce=None,
    expert_cache_capacity: int = 0,
    prefix: str = LM,
    pp_size: int = 1,
    pp_rank: int = 0,
    profiler=None,
    expert_deep_cache: float | None = None,
    expert_fused_fp8: bool = False,
    expert_resident_layers: bool = False,
    expert_nvfp4: bool = False,
    expert_reserve_bytes: int = 3 << 30,
    mtp: bool = False,
    mtp_pin_bytes: int = 0,
) -> Qwen4ExpModel:
    """TP-sharded dense weights on GPU, routed experts + PLE table in host RAM.

    Sharding plan (heads divide evenly by 4 for this config):
      * QSA: 24 q-heads / 2 kv-heads / 4 indexer heads -> per rank 6 / 2 / 1.
        kv heads (2) do not divide by 4, so KV is replicated and only q/o are
        split; that costs a little duplicate KV cache but keeps the reduce simple.
      * GatedDeltaNet: 48 v-heads / 16 k-heads -> 12 / 4 per rank.
      * Shared expert and hyper-connections: replicated (they are tiny).
      * Routed experts: round-robin by expert id across ranks.
      * lm_head: column-sharded, gathered by the caller.

    ``pp_size > 1`` splits the *layers* across ranks instead: this rank builds only
    ``[layer_start, layer_end)`` and every sharding rule above collapses to one rank
    (``world_size`` is then the TP width, which the caller passes as 1). The embed stays
    on the first stage and the final mixer plus lm_head on the last, so those two stages
    are the only ones that pay for them.
    """
    device = torch.device(device)
    store = checkpoint.store

    def dense(name: str) -> torch.Tensor:
        return store.load(name, device=device, dtype=dtype)

    shard, kv_window = _tp_shard(config, world_size, rank)
    kv_start, kv_end = kv_window["kv_start"], kv_window["kv_end"]

    # The resident mode takes the *same* shard and drops the staging policy: the
    # `expert_weights` seam is the only thing that changes, so the dispatch loop,
    # the accumulation order and the dequantization are shared with the host path
    # rather than reimplemented.  It is selected on the shard's *placement* rather
    # than on a flag of its own -- `preload_experts` was told where to put this
    # rank's experts, and only a device-resident shard can answer `rows()` with
    # device tensors.  That keeps the two paths from being able to disagree: there
    # is one decision, made where the bytes are, and a host shard can only ever
    # produce the host backend.
    expert_shard = getattr(checkpoint, "expert_shard", None)
    if expert_deep_cache is not None:
        # The deep cache holds the layers the runtime could decode on the card; the rest of this
        # rank's share is read per slot from the checkpoint's own bytes by the host stove, so a
        # layer that did not fit is slow and correct rather than absent.
        host_moe = HostExpertMoE(
            checkpoint, config.num_experts, device, dtype, cache_capacity=expert_cache_capacity
        )
        inner_moe = DeepFallbackMoE(
            DeepCacheMoE(
                config.num_experts,
                device,
                scale=expert_deep_cache,
                world_size=world_size,
                rank=rank,
            ),
            host_moe,
        )
    elif expert_fused_fp8 and os.environ.get("POCKETLLM_QWEN4_NVFP4_EXPERTS", "0") == "1":
        # The same placement and the same seam as the FP8 fused path, one arithmetic apart: the
        # table is quantized to NVFP4 at load (about 7.85 GiB a rank against 14.06) and the decode
        # runs the NVFP4 kernel pair. Opt-in because it changes the numbers: the reconstruction
        # reaches cosine 0.988 at the expert-MLP output, which is a quality decision, not a
        # performance one, and no token-identity check can gate it.
        if expert_shard is None or getattr(expert_shard, "device", None) is None:
            raise ValueError("the NVFP4 expert bank needs the experts resident on the device")
        inner_moe = NvFp4SlotExpertMoE(
            expert_shard, config.num_experts, device, num_layers=config.num_hidden_layers
        )
    elif expert_fused_fp8:
        if expert_shard is None or getattr(expert_shard, "device", None) is None:
            raise ValueError("expert_fused_fp8 needs the experts resident on the device")
        # The same placement as the loop below, one arithmetic apart: the fused kernel reads the
        # FP8 codes and their scales in the load. That is a different set of numbers, so it is
        # asked for and never inferred from the placement.
        inner_moe = Fp8SlotExpertMoE(expert_shard, config.num_experts, device)
    elif expert_shard is not None and getattr(expert_shard, "device", None) is not None:
        inner_moe = DeviceExpertMoE(expert_shard, config.num_experts, device)
    else:
        inner_moe = HostExpertMoE(
            checkpoint,
            config.num_experts,
            device,
            dtype,
            cache_capacity=expert_cache_capacity,
        )

    moe = ShardedMoE(inner_moe, rank, world_size) if world_size > 1 else inner_moe
    # How many leading layers have their experts on the device and therefore no H2D in a step.
    # A capture has to stop there; the caller reads this rather than re-deriving the boundary.

    # Under pipeline parallelism this rank builds only its own stage's layers and leaves a `None`
    # placeholder for every other. The list stays full length on purpose: `layer_idx` is the
    # absolute id a layer keys its KV cache, PLE table and expert cache by, so a short list indexed
    # 0..2 would be a *second*, relative numbering and a mis-index would land on another layer's
    # cache silently.
    layer_start, layer_end = layer_range_for(config.num_hidden_layers, pp_size, pp_rank)

    layers: list[Qwen4ExpDecoderLayer | None] = []
    for layer_idx in range(config.num_hidden_layers):
        if not (layer_start <= layer_idx < layer_end):
            layers.append(None)
            continue
        base = f"{prefix}.layers.{layer_idx}"
        weights = _shard_layer_dense(
            store.view,
            base,
            config,
            dtype,
            device,
            rank,
            world_size,
            kv_window,
            is_linear=config.is_linear_layer(layer_idx),
        )

        ple = None
        ple_index = config.ple_layer_index(layer_idx)
        if ple_index is not None:
            shard_keys = checkpoint.ngram_shard_keys(layer_idx)
            table = HostNGramTable(
                [store.view(k) for k in shard_keys],
                device=device,
                dtype=dtype,
                scale=checkpoint.ngram_scale(layer_idx),
            )
            ple = PLELayer(
                config,
                {_strip_weight(s): dense(f"{base}.ple.{s}") for s in _PLE_SUFFIXES},
                ple_index,
                device,
                ngram_table=table,
            )

        layers.append(
            Qwen4ExpDecoderLayer(
                config,
                layer_idx,
                weights,
                moe,
                device=device,
                ple=ple,
                tp_shard=shard,
                all_reduce=all_reduce,
            )
        )

    # 1.18 GiB each: embedding stays in host RAM (gathered per step), lm_head is
    # column-sharded so each rank holds 300 MiB.
    #
    # Under pipeline parallelism only the boundary stages need them: the first owns the embedding
    # (it is the only stage that sees token ids) and the last owns the head and the final mixer that
    # feeds it. Building them everywhere would put a 1.18 GiB lm_head on 15 stages that cannot use
    # it. The head is *not* vocab-sharded on that path: `world_size` is the TP width there, and
    # under PP the head belongs to one stage undivided.
    owns_embedding = pp_size <= 1 or pp_rank == 0
    owns_head = pp_size <= 1 or pp_rank == pp_size - 1
    embed_weight = store.view(f"{prefix}.embed_tokens.weight") if owns_embedding else None
    lm_head = None
    final_mixer = None
    if owns_head:
        lm_head_full = store.view("lm_head.weight")
        if pp_size > 1:
            lm_head = lm_head_full.to(device, dtype=dtype)
        else:
            vocab_per_rank = lm_head_full.shape[0] // world_size
            lm_head = lm_head_full[rank * vocab_per_rank : (rank + 1) * vocab_per_rank].to(
                device, dtype=dtype
            )
        final_mixer = _final_mixer(dense, config, prefix=prefix)

    if expert_nvfp4:
        # The whole expert set goes on the card, so nothing stages during a decode and the step
        # stops being a PCIe relay. Done after the dense half is loaded for the same reason the
        # pinning block below is: the budget is what is left, not what was free at the start.
        from relicllm.models.qwen4_exp.nvfp4_moe import build_bank  # noqa: PLC0415
        from relicllm.runtime.ops import load_ops  # noqa: PLC0415

        ops = load_ops()
        if ops is None or not hasattr(ops, "nvfp4_moe_token_forward"):
            raise RuntimeError(
                "--expert-nvfp4 needs relic-core's ModelOpt NVFP4 MoE kernel, and this build has "
                "no nvfp4_moe_token_forward"
            )
        bank = build_bank(
            ops,
            checkpoint.root,
            rank=rank,
            world_size=world_size,
            layers=config.num_hidden_layers,
            hidden=config.hidden_size,
            inter=config.moe_intermediate_size,
            experts=config.num_experts,
            device=device,
            verbose=(rank == 0),
        )
        inner_moe.use_nvfp4(bank)
        if device.type == "cuda" and torch.distributed.is_initialized() and rank == 0:
            free, _total = torch.cuda.mem_get_info(device)
            print(
                f"[qwen4exp] nvfp4 experts: {len(bank)}/{config.num_hidden_layers} layers "
                f"resident, {free / (1 << 30):.1f} GiB still free",
                flush=True,
            )

    if expert_resident_layers:
        # Budgeted against what the card has left once the dense half is on it: pinning the experts
        # first spent the room the embedding was going to need, and the model died on
        # `embed_weight.to(device)` with 27 layers already held.
        free, _total = torch.cuda.mem_get_info(device)
        reserve = int(expert_reserve_bytes)
        budget = max(0, int(free) - reserve)
        pinned = inner_moe.pin_layers(range(config.num_hidden_layers), budget)
        if device.type == "cuda" and torch.distributed.is_initialized() and rank == 0:
            print(
                f"[qwen4exp] pinned {pinned}/{config.num_hidden_layers} layers' experts "
                f"on the device ({budget / (1 << 30):.1f} GiB of room, "
                f"{reserve / (1 << 30):.1f} GiB held back)",
                flush=True,
            )
        # After the pinning, not before: the capture boundary is the count this decided.
        moe.resident_layer_prefix = int(getattr(inner_moe, "_resident_layers", 0) or 0)

    head = None
    if mtp:
        # Built after the main experts so that on a card that is already tight the head's own
        # failure is reported against the memory that is actually left rather than against a moment
        # before the main stack took its share.  Not under pipeline parallelism: the head reads the
        # backbone's final stream state and feeds the lm_head, so it belongs on the stage that owns
        # both, and only the last stage does.
        if pp_size > 1 and pp_rank != pp_size - 1:
            if rank == 0:
                print(
                    f"[qwen4exp] --mtp ignored on pipeline stage {pp_rank}/{pp_size}: the head "
                    "runs on the stage that owns the last layer",
                    flush=True,
                )
        else:
            head = build_mtp_head(
                config,
                checkpoint,
                device=device,
                dtype=dtype,
                rank=rank,
                world_size=world_size,
                all_reduce=all_reduce,
                # Everything free after the trunk's bank is fair game: the head is one layer,
                # and a head that has to stage its experts is a draft that costs more than the
                # step it saves.
                pin_experts=max(0, int(mtp_pin_bytes)),
            )
            if rank == 0:
                print(
                    "[qwen4exp] mtp head loaded "
                    f"({'fp8' if checkpoint.mtp_fp8_experts() else 'packed'} expert rows)",
                    flush=True,
                )

    model = Qwen4ExpModel(
        config,
        layers,
        # The token embedding goes on the card, not in host RAM.  It is 1.18 GiB and a decode
        # step reads one row of it; keeping it on the host makes every step cross PCIe for that
        # row, and an unpinned crossing is one a captured graph will not record.  The card has
        # the room: on 8x3090 the resident experts leave 9.2 GiB free a rank.
        embed_tokens=DeviceEmbedding(embed_weight.to(device, dtype=dtype)) if owns_embedding else None,
        lm_head=lm_head,
        final_mixer=final_mixer,
        device=device,
        dtype=dtype,
        pp_size=pp_size,
        pp_rank=pp_rank,
        profiler=profiler,
    )
    if head is not None:
        # The head shares the embedding rather than holding a second copy: it is the same 1.18 GiB
        # table, and a copy would be the largest single allocation on a card that is already full.
        head.embed_tokens = model.embed_tokens
        model.mtp = head
    return model
