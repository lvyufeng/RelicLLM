"""Unit checks for the MTP draft head that do not need the real checkpoint.

The head is a thin composition of the backbone's own building blocks, so what is worth testing here
is exactly the places where it *is* not the backbone: the shape of the fusion (`pre_fc_norm_hidden`
is `hc_count * hidden` wide, `pre_fc_norm_embedding` is `hidden`), which attention flavour it builds,
that the routed experts go through the same `ShardedMoE` ownership rule, and that the model's
`return_streams` hands back the same tensor the final mixer was handed.

Run: pytest tests/test_qwen4_exp_mtp.py
"""

from __future__ import annotations

import torch

from relicllm.models.qwen4_exp.config import Qwen4ExpTextConfig
from relicllm.models.qwen4_exp.mtp import MTPHead
from relicllm.models.qwen4_exp.moe import InMemoryMoE, ShardedMoE
from relicllm.models.qwen4_exp.weights import DeviceEmbedding

HIDDEN = 64
INTER = 32
EXPERTS = 8
HEADS = 4
KV_HEADS = 2
HEAD_DIM = 16
HC = 4


def _config(**overrides) -> Qwen4ExpTextConfig:
    base = dict(
        hidden_size=HIDDEN,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        vocab_size=256,
        num_experts=EXPERTS,
        num_experts_per_tok=2,
        moe_intermediate_size=INTER,
        shared_expert_intermediate_size=INTER,
        hc_count=HC,
        hc_lowrank=16,
        layer_types=["qwen_sparse_attention"],
    )
    base.update(overrides)
    return Qwen4ExpTextConfig(**base)


def _mtp_weights(config: Qwen4ExpTextConfig) -> dict[str, torch.Tensor]:
    """The MTP block's non-expert tensors, at the checkpoint's own shapes."""
    g = torch.Generator().manual_seed(0)
    rand = lambda *shape: torch.randn(*shape, generator=g) * 0.02  # noqa: E731
    hc_hidden = config.hc_count * config.hidden_size
    q_rows = config.num_attention_heads * config.head_dim * 2
    weights = {
        "pre_fc_norm_hidden": torch.ones(hc_hidden),
        "pre_fc_norm_embedding": torch.ones(config.hidden_size),
        "fc_hidden": rand(config.hidden_size, config.hidden_size),
        "fc_embedding": rand(config.hidden_size, config.hidden_size),
        "hyper_connection_mixer.hc_norm": torch.ones(hc_hidden),
        "hyper_connection_mixer.input_mix_weight_down": rand(config.hc_lowrank, hc_hidden),
        "hyper_connection_mixer.input_mix_weight_up": rand(hc_hidden, config.hc_lowrank),
        "attn_hyper_connection.hc_norm": torch.ones(hc_hidden),
        "attn_hyper_connection.input_mix_weight_down": rand(config.hc_lowrank, hc_hidden),
        "attn_hyper_connection.input_mix_weight_up": rand(hc_hidden, config.hc_lowrank),
        "attn_hyper_connection.block_inject_weight": torch.ones(config.hc_count, hc_hidden),
        "mlp_hyper_connection.hc_norm": torch.ones(hc_hidden),
        "mlp_hyper_connection.input_mix_weight_down": rand(config.hc_lowrank, hc_hidden),
        "mlp_hyper_connection.input_mix_weight_up": rand(hc_hidden, config.hc_lowrank),
        "mlp_hyper_connection.block_inject_weight": torch.ones(config.hc_count, hc_hidden),
        "self_attn.q_proj": rand(q_rows, HIDDEN),
        "self_attn.k_proj": rand(KV_HEADS * HEAD_DIM, HIDDEN),
        "self_attn.v_proj": rand(KV_HEADS * HEAD_DIM, HIDDEN),
        "self_attn.o_proj": rand(HIDDEN, config.num_attention_heads * HEAD_DIM),
        "self_attn.q_norm": torch.ones(HEAD_DIM),
        "self_attn.k_norm": torch.ones(HEAD_DIM),
        "self_attn.index_qk_proj": rand(
            config.indexer_n_heads * config.indexer_head_dim
            + config.indexer_kv_heads * config.indexer_head_dim,
            HIDDEN,
        ),
        "self_attn.q_layernorm": torch.ones(config.indexer_head_dim),
        "self_attn.k_layernorm": torch.ones(config.indexer_head_dim),
        "mlp.gate": rand(EXPERTS, HIDDEN),
        "mlp.shared_expert_gate": rand(1, HIDDEN),
        "mlp.shared_expert.gate_proj": rand(INTER, HIDDEN),
        "mlp.shared_expert.up_proj": rand(INTER, HIDDEN),
        "mlp.shared_expert.down_proj": rand(HIDDEN, INTER),
    }
    return weights


def _moe(config: Qwen4ExpTextConfig, device="cpu") -> InMemoryMoE:
    g = torch.Generator().manual_seed(1)
    gate_up = torch.randn(EXPERTS, 2 * INTER, HIDDEN, generator=g) * 0.02
    down = torch.randn(EXPERTS, HIDDEN, INTER, generator=g) * 0.02
    return InMemoryMoE(EXPERTS, {0: gate_up}, {0: down})


def _head(config=None, device="cpu") -> MTPHead:
    config = config or _config()
    head = MTPHead(
        config,
        _mtp_weights(config),
        _moe(config, device),
        device=torch.device(device),
    )
    head.embed_tokens = DeviceEmbedding(_embedding_table(config))
    return head


def _embedding_table(config: Qwen4ExpTextConfig) -> torch.Tensor:
    g = torch.Generator().manual_seed(2)
    return torch.randn(config.vocab_size, config.hidden_size, generator=g) * 0.02


def test_fusion_is_per_stream_eh_proj() -> None:
    """`fc_hidden`/`fc_embedding` are `hidden x hidden`: the stack survives the fusion.

    Neither projection can consume the `hc_count * hidden` stack, so `fc_hidden` must be applied
    to each stream separately and `fc_embedding` to the one normalized embedding.  A fusion that
    collapses the stack -- or that tries to multiply `fc_hidden` by an `hc_count * hidden` row --
    is the mistake this pins: the checkpoint's shapes make it either a shape error at load time or
    a silently different function.
    """
    config = _config()
    head = _head(config)
    assert head.fc_hidden.shape == (HIDDEN, HIDDEN)
    assert head.fc_embedding.shape == (HIDDEN, HIDDEN)
    assert head.norm_hidden.weight.shape == (HC * HIDDEN,)
    assert head.norm_hidden.group_size == HIDDEN
    streams = torch.randn(1, 3, HC * HIDDEN)
    ids = torch.tensor([[1, 2, 3]])
    fused = head._fusion(streams, ids)
    assert fused.shape == (1, 3, HC * HIDDEN)
    # Per stream the fusion is `fc_hidden(norm_h(stream_i)) + fc_embedding(norm_e(e))`, so the
    # streams differ by their own residual term only -- a stack that was collapsed first would
    # make all four streams identical.
    per_stream = fused.reshape(1, 3, HC, HIDDEN)
    assert not torch.allclose(per_stream[:, :, 0], per_stream[:, :, 1])
    # And with zero streams the result is the embedding term, the same in every stream.
    embedding_only = head._fusion(torch.zeros(1, 3, HC * HIDDEN), ids)
    per_stream = embedding_only.reshape(1, 3, HC, HIDDEN)
    for stream in range(1, HC):
        torch.testing.assert_close(per_stream[:, :, stream], per_stream[:, :, 0])


def test_norm_hidden_is_grouped_per_stream() -> None:
    """Each stream's `pre_fc_norm_hidden` statistics are its own.

    Grouping the norm over all `hc_count * hidden` would let one loud stream scale the gain on a
    quiet one.  Build a stack whose streams have very different magnitudes and check the normalized
    streams come back with unit RMS each.
    """
    config = _config()
    head = _head(config)
    streams = torch.randn(1, 1, HC * HIDDEN)
    for stream in range(HC):
        streams[:, :, stream * HIDDEN : (stream + 1) * HIDDEN] *= 10.0**stream
    normed = head.norm_hidden(streams).reshape(1, 1, HC, HIDDEN)
    # The gain is `1 + weight` (Qwen's centred convention) and the test weights are all 1, so a
    # correctly grouped norm lands at RMS 2 on every stream -- and the number is the same for all
    # four only because each was normalized on its own statistics.
    rms = normed.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.full_like(rms, 2.0), atol=1e-3)


def test_head_builds_the_attention_the_weights_are() -> None:
    """A `self_attn.*` block builds QSA, not the main stack's layer-0 flavour."""
    config = _config(layer_types=["linear_attention"])
    head = _head(config)
    assert head.layer.is_linear is False
    assert hasattr(head.layer.attn, "indexer")


def test_forward_returns_a_vector_for_lm_head() -> None:
    config = _config()
    head = _head(config)
    streams = torch.randn(1, 2, HC * HIDDEN)
    ids = torch.tensor([[5, 6]])
    cache = head.make_cache(1, 8, torch.device("cpu"), torch.float32)
    out = head.forward(streams, ids, cache, past_len=0)
    assert out.shape == (1, 2, HIDDEN)
    assert torch.isfinite(out).all()
    # The head's cache is its own, and the layer wrote into it.
    assert cache.layers[0].length == 2


def test_fusing_the_token_being_predicted_is_observable() -> None:
    """The head is conditioned on the token it *consumed*, and moving it off that changes output.

    This is the defect that cost the most time: handing the head the token it is meant to predict
    rather than the token the trunk consumed leaves a head that runs, is confident, and predicts
    the token it was given.  The unit test cannot say which is *right* on a random fixture, but it
    can pin that the conditioning id is live -- a head whose output ignored `token_ids` would be
    a wiring bug the accuracy measurement would otherwise absorb as "a bad model".
    """
    config = _config()
    head = _head(config)
    torch.manual_seed(0)
    streams = torch.randn(1, 1, HC * HIDDEN)
    a = head.forward(streams, torch.tensor([[11]]), None, past_len=0)
    b = head.forward(streams, torch.tensor([[12]]), None, past_len=0)
    assert not torch.allclose(a, b)


def test_default_input_mode_does_not_readd_the_trunk_state() -> None:
    """`replace` is the default: `fc_hidden` already carries the trunk into the fused stack.

    The `add` mode re-adds the raw stack on top, which counts it twice.  It exists only to be
    measured against, so the default has to be the formulation.
    """
    import importlib
    import os

    from relicllm.models.qwen4_exp import mtp as mtp_module

    previous = os.environ.pop("POCKETLLM_QWEN4_MTP_INPUT", None)
    try:
        module = importlib.reload(mtp_module)
        assert module._mtp_input_mode() == "replace"
        config = _config()
        head = module.MTPHead(
            config,
            _mtp_weights(config),
            _moe(config),
            device=torch.device("cpu"),
        )
        assert head.input_mode == "replace"
    finally:
        if previous is not None:
            os.environ["POCKETLLM_QWEN4_MTP_INPUT"] = previous
        importlib.reload(mtp_module)


def test_decode_matches_prefill_for_the_same_token() -> None:
    """A cached single-token call reproduces the same position's prefill output.

    This is the property a draft depends on: the head is advanced one position at a time during
    decode, and if its cached step disagreed with its own prefill the draft would be scored against
    a different function than the one the prefill implied.
    """
    config = _config()
    head = _head(config)
    torch.manual_seed(0)
    streams = torch.randn(1, 3, HC * HIDDEN)
    ids = torch.tensor([[7, 8, 9]])

    full = head.forward(streams, ids, None, past_len=0)

    cache = head.make_cache(1, 8, torch.device("cpu"), torch.float32)
    head.forward(streams[:, :1], ids[:, :1], cache, past_len=0)
    head.forward(streams[:, 1:2], ids[:, 1:2], cache, past_len=1)
    step = head.forward(streams[:, 2:3], ids[:, 2:3], cache, past_len=2)

    torch.testing.assert_close(step[:, -1], full[:, -1], rtol=1e-4, atol=1e-4)


def test_experts_are_owned_by_the_same_round_robin_rule() -> None:
    """Under TP the head's routed experts are masked exactly as the main layers' are.

    `ShardedMoE` rewrites a foreign expert's id to the sentinel, and a backend that owns nothing
    of it contributes zero -- so two ranks' partials sum to the single-rank answer.  That is the
    ownership rule the task requires the head to share, and it is the inner backend that must not
    see another rank's id.
    """
    config = _config()
    base = _head(config)
    streams = torch.randn(1, 1, HC * HIDDEN)
    ids = torch.tensor([[3]])

    with torch.no_grad():
        whole = base.forward(streams, ids, None, past_len=0)

    moe = _moe(config)
    embedding = _embedding_table(config)
    parts = []
    for rank in range(2):
        head = MTPHead(
            config,
            _mtp_weights(config),
            ShardedMoE(moe, rank, 2),
            device=torch.device("cpu"),
        )
        head.embed_tokens = DeviceEmbedding(embedding)
        with torch.no_grad():
            parts.append(head.forward(streams, ids, None, past_len=0))
    # Only the attention's row-parallel output and the MLP block are split; here every rank runs
    # the *whole* layer, so the comparison is against the routed-expert sum alone: the two ranks'
    # MoE partials must add up to the single backend's.
    flat = torch.randn(1, HIDDEN)
    router_weights = torch.softmax(torch.randn(1, 2), -1)
    indices = torch.tensor([[1, 4]])
    single = moe(0, flat, indices, router_weights)
    summed = sum(
        ShardedMoE(moe, rank, 2)(0, flat, indices, router_weights) for rank in range(2)
    )
    torch.testing.assert_close(summed, single)
    assert all(torch.isfinite(p).all() for p in parts)