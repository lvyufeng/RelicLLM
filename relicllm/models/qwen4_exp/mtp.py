"""The MTP draft head: one extra decoder layer that predicts the token after next.

Qwen3.8-Flash-Next ships a single-layer multi-token-prediction head after the main stack.  It is
not a second small model: it is one ordinary decoder layer plus two projections that fuse the
backbone's residual stack with the *next* token's embedding, and one hyper-connection mixer that
turns the layer's stream back into a single vector for the existing (shared, frozen) `lm_head`.

The shapes pin down the formulation, and they only make sense one way:

* `mtp.pre_fc_norm_hidden` is `hc_count * hidden` wide, because `hnorm` normalizes the whole
  hyper-connection residual stack -- one RMS statistic a stream, like every other HC norm here.
* `mtp.pre_fc_norm_embedding` is `hidden` wide, because a token embedding is one vector.
* `mtp.fc_hidden` and `mtp.fc_embedding` are each `hidden x hidden`.  Neither is
  `hidden x (hc_count * hidden)`, so neither projection consumes the raw stack: `fc_hidden` is
  applied to each stream's normalized vector independently, and `fc_embedding` to the single
  normalized embedding, which is then broadcast to `hc_count` copies.  The published
  "`fc_hidden(norm_h(h)) + fc_embedding(norm_e(e))`" is a per-stream equation, and its result is
  `hc_count` wide again.

The block then runs exactly as a trunk layer does -- two hyper-connections around attention and one
around the MoE -- on that fused stack, and the final mixer collapses the streams for `lm_head`.
Its residual input is the trunk's *last* stack, so the head is a tail layer appended to the trunk
rather than a reader of an already-collapsed `hidden`.

Two things are easy to get wrong here and both leave a head that runs, returns confident logits and
predicts nothing:

* **which token conditions the draft.**  The head drafts the successor of the token the trunk just
  consumed, so it is fused with *that* token's embedding.  Fusing with the token being predicted
  hands the head its own answer, and the symptom is a head that echoes its input.
* **whether the trunk state is counted once.**  `fc_hidden` already carries it into `eh_proj`'s
  output, so that output *is* the block's residual rather than something to add to the trunk's.

`runtime.mtp_accuracy` scores both, on the next-token metric *and* on the echo metric, because the
echo metric rates a broken head highly.

None of that is re-implemented here.  The routed experts go through the *same* `MoEBackend` the
main layers use, which is what puts them through `ShardedMoE` with the same
`expert_id % world_size == rank` ownership, whether that backend stages FP8 codes from the host or
reads a whole-device NVFP4 bank.  The layer is a `Qwen4ExpDecoderLayer` fed the `mtp.layers.0.*`
weights under the same short keys, and the mixer is the same `GatedResidual` the backbone's final
mixer is, so there is no second copy of the attention, router or hyper-connection mathematics.
"""

from __future__ import annotations

import os

import torch

from relicllm.models.qwen4_exp.attention import QSAAttentionCache
from relicllm.models.qwen4_exp.layers import GatedResidual, MRoPE, RMSNorm, prefill_linear
from relicllm.models.qwen4_exp.model import Qwen4ExpCache, Qwen4ExpDecoderLayer, TPShard

#: The checkpoint prefix of the MTP block.  It is *not* under `model.language_model`: the draft
#: layer sits beside the text model in the multimodal archive, so its names are top-level.
MTP_PREFIX = "mtp"

#: The one layer index the per-layer loops use for the head.  A model with more MTP layers would
#: number them from here; the checkpoint ships one.
MTP_LAYER_INDEX = 0


def _mtp_input_mode() -> str:
    """How the fused stack is used: `replace` (default) or `add`.

    `replace` is the formulation: `eh_proj`'s output becomes the block's residual, carrying the
    trunk state through `fc_hidden`.  `add` additionally re-adds the raw trunk stack, counting it
    twice, and exists only so the two can be measured against each other on the next-token metric
    (`runtime.mtp_accuracy`).
    """
    mode = os.environ.get("POCKETLLM_QWEN4_MTP_INPUT", "replace").strip().lower()
    return mode if mode in {"add", "replace"} else "replace"


def _attention_flavour(config, weights: dict[str, torch.Tensor]) -> str:
    """Which attention the MTP block's weights are, as the checkpoint declares it.

    `config.mtp_layer_types` is the declaration; `config.layer_types` is *not*, because it belongs
    to the main stack, where layer 0 is a GatedDeltaNet layer on this checkpoint.  The weights are
    the tie-break rather than the truth: a config that says full-attention and a block with no
    `self_attn.q_proj` would build a layer that then fails to find its weights, and vice versa, so
    a disagreement is an error worth raising here instead of three frames down.
    """
    declared = list(getattr(config, "mtp_layer_types", None) or ["qwen_sparse_attention"])[0]
    declared = "qwen_sparse_attention" if declared == "full_attention" else declared
    shipped = "qwen_sparse_attention" if "self_attn.q_proj" in weights else "linear_attention"
    if declared != shipped:
        raise ValueError(
            f"the MTP config declares {declared!r} but the checkpoint's mtp.layers.0 weights are "
            f"{shipped!r}"
        )
    return declared


class MTPHead:
    """One MTP layer, its two fusion projections, and the mixer that feeds the shared `lm_head`."""

    def __init__(
        self,
        config,
        weights: dict[str, torch.Tensor],
        moe,
        *,
        device: torch.device,
        tp_shard: TPShard | None = None,
        all_reduce=None,
    ) -> None:
        self.config = config
        self.device = device
        self.moe = moe
        # `pre_fc_norm_hidden` normalizes a whole hyper-connection stack, so it is grouped like the
        # HC norms: one RMS statistic a stream, not one across all `hc_count * hidden`.  Grouping it
        # the other way still produces finite numbers, which is why it is pinned by
        # `test_norm_hidden_is_grouped_per_stream` against a stack whose streams differ in scale.
        self.norm_hidden = RMSNorm(
            weights["pre_fc_norm_hidden"], config.rms_norm_eps, group_size=config.hidden_size
        )
        self.norm_embedding = RMSNorm(weights["pre_fc_norm_embedding"], config.rms_norm_eps)
        self.fc_hidden = weights["fc_hidden"]
        self.fc_embedding = weights["fc_embedding"]
        layer_config = _layer_config(config, weights)
        self.layer = Qwen4ExpDecoderLayer(
            layer_config,
            MTP_LAYER_INDEX,
            weights,
            moe,
            device=device,
            tp_shard=tp_shard or TPShard.single(config),
            all_reduce=all_reduce,
        )
        self.final_mixer = GatedResidual(
            config,
            weights["hyper_connection_mixer.hc_norm"],
            weights["hyper_connection_mixer.input_mix_weight_down"],
            weights["hyper_connection_mixer.input_mix_weight_up"],
            None,
        )
        # The MTP layer rotates at its own theta: the checkpoint may carry a different base for the
        # draft head, and rotating it with the backbone's would misplace every position.
        theta = getattr(config, "mtp_rope_theta", None) or config.rope_theta
        self.rope = MRoPE(config, device, rope_theta=theta)
        self.input_mode = _mtp_input_mode()
        #: Filled by `build_heterogeneous` with the model's token embedding, which the head shares
        #: rather than holds (it is the same 1.18 GiB table).
        self.embed_tokens = None

    # -- caches -----------------------------------------------------------------

    def make_cache(self, batch_size: int, max_seq_len: int, device, dtype) -> Qwen4ExpCache:
        """The MTP layer's own KV cache, kept apart from the backbone's.

        Use `share_cache` instead when drafting more than one token a round; this one is right for
        the single-token case, where the head only ever attends over what the backbone has already
        written and its own keys never have to be visible to anybody.
        """
        return Qwen4ExpCache(
            layers=[self.layer.make_cache(batch_size, max_seq_len, device, dtype)]
        )

    def attach_rope_table(self, cache: Qwen4ExpCache, width: int, dtype: torch.dtype) -> None:
        """Give the head's sparse-attention layer the absolute RoPE table its indexer reads.

        The indexer rotates every pooled key block by the block's *absolute* start, and it reads
        that phase from `cache.cos` when the layer cache has one, falling back to the rows the
        forward computed for the current queries. The trunk publishes its table
        (`Qwen4ExpCache.share_rope_table`); the head's own cache never had one, so every block was
        rotated with a clamped current-query row -- block 0 with the first query's phase, every
        later block with the last one's -- and, because that table is a fresh tensor each call,
        the indexer's completed-block pool failed its layout check and was rebuilt from scratch on
        every call. Built with the head's own theta, which may differ from the trunk's.
        """
        layer = cache.layers[MTP_LAYER_INDEX]
        device = layer.index_k.device
        positions = torch.arange(int(width), device=device).view(1, 1, -1).expand(3, 1, -1)
        # The forward's activation dtype, so a table row equals the row `forward` computes itself.
        layer.cos, layer.sin = self.rope(positions, dtype)

    def share_cache(self, backbone_cache: Qwen4ExpCache) -> Qwen4ExpCache:
        """A head cache that reads and writes the backbone's own KV arena.

        Drafting `k` tokens in a row is a chain of head calls, and each link has to attend over the
        *backbone's* keys for every position before it -- a head that keeps its own arena sees only
        its own earlier drafts, and by the second link it is answering from a history the real
        sequence does not have.  Measured: with a private cache the first draft of a round is
        accepted almost every time and the second one essentially never.

        Sharing the arena fixes that, and it is safe for the same reason it is possible: the keys
        are addressed by absolute position, so the head reading slot `p` reads what the backbone
        wrote there, and the draft's own key for position `p + 1` is written into a slot the
        verification step is about to overwrite with whatever the trunk actually chose there.  A
        rejected draft therefore costs nothing to undo.

        The arena is only borrowed, never freed: the cache returned here owns no storage, so it
        must not outlive the backbone's.
        """
        source = backbone_cache.layers[0]
        head_layer = self.layer.make_cache(
            batch_size=1, max_seq_len=1, device=source.k.device, dtype=source.k.dtype
        )
        if not isinstance(source, QSAAttentionCache):
            raise RuntimeError(
                "the MTP head drafts multi-token rounds over a QSA backbone layer; "
                f"this one is {type(source).__name__}"
            )
        # The head's own attention is QSA over the same positions, so every arena it addresses is
        # one the backbone already keeps: borrow all three rather than allocate a second copy.
        head_layer.k = source.k
        head_layer.v = source.v
        head_layer.index_k = source.index_k
        return Qwen4ExpCache(layers=[head_layer])

    # -- forward ----------------------------------------------------------------

    def _fusion(self, streams: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        """`eh_proj(concat(e_norm, h_norm))` = `fc_embedding(e) + fc_hidden(h)`, per stream.

        This is where the checkpoint's shapes only make sense one way.  `pre_fc_norm_hidden` is
        `hc_count * hidden` wide because `hnorm` normalizes the whole residual stack, one statistic
        a stream, exactly as the trunk's hyper-connection norms do.  `fc_hidden` is `hidden x
        hidden`, not `hidden x (hc_count * hidden)`: it is applied to *each* stream's normalized
        vector independently, which is the point of pairing it with `fc_embedding` -- the token
        embedding is a fact about the next token, shared by every stream, so it is normalized once
        and broadcast to `hc_count` copies while the residual side stays per stream.

        The result is therefore `hc_count` wide again: a stack of "what the next token embedding
        and this stream jointly suggest", which is what the draft layer consumes.
        """
        embed = self.embed_tokens(token_ids).to(streams.dtype)
        # `enorm` normalizes one `hidden` vector; broadcasting after the norm is the same as
        # normalizing `hc_count` identical copies, and avoids materializing them first.
        embedding = self.norm_embedding(embed)
        embedding = embedding.unsqueeze(-2).expand(
            *embedding.shape[:-1], self.config.hc_count, self.config.hidden_size
        )

        stack = self.norm_hidden(streams).unflatten(-1, (self.config.hc_count, self.config.hidden_size))
        # Each matrix takes one `hidden`-wide row, so the two GEMMs are fused over the stream axis:
        # `(stack * hc, hidden)` in, `(stack * hc, hidden)` out.
        flat = stack.reshape(-1, self.config.hidden_size)
        projected_hidden = prefill_linear(flat, self.fc_hidden)
        projected_embedding = prefill_linear(
            embedding.reshape(-1, self.config.hidden_size), self.fc_embedding
        )
        return (projected_hidden + projected_embedding).reshape_as(streams)

    def forward(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        cache: Qwen4ExpCache | None = None,
        past_len: int = 0,
        return_streams: bool = False,
    ) -> torch.Tensor:
        """Run the head; returns `[b, seq, hidden]`, the vector `lm_head` takes.

        `hidden_states` is the backbone's *stream* state, `[b, seq, hc_count * hidden]` -- what
        `Qwen4ExpModel.forward` holds before `final_mixer`, the only point at which the state is
        still four streams.  `token_ids` are the ids the head conditions on: the token at the
        the token the trunk *consumed* at that position -- the one whose successor is being drafted
        -- not the token being predicted.  `cache` is the head's own (see `make_cache`) and
        `past_len` is
        where this call's positions start in it.
        """
        assert self.embed_tokens is not None, "the MTP head needs the model's embedding"
        # Callers hand ids in either the layer convention `[b, seq]` or the flat `[seq]` a decode
        # step of one row produces; normalizing here keeps the caller from having to care.
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        batch_size, seq_len = token_ids.shape
        hidden_states = hidden_states.reshape(batch_size, seq_len, -1)
        # Only this call's positions: the layer rotates the query rows it is handed. A device
        # `past_len` is a captured call's position; `arange` over tensor bounds would read it back.
        if isinstance(past_len, torch.Tensor):
            position_ids = past_len.reshape(()) + torch.arange(seq_len, device=token_ids.device)
        else:
            position_ids = torch.arange(past_len, past_len + seq_len, device=token_ids.device)
        position_ids = position_ids.view(1, 1, -1).expand(3, batch_size, -1)
        cos, sin = self.rope(position_ids, hidden_states.dtype)

        fused = self._fusion(hidden_states, token_ids)
        # The fused stack *is* the block's residual input -- `eh_proj` output becomes `res_hc`, and
        # the block then mixes it through its own two hyper-connections.  Adding `hidden_states` on
        # top would count the trunk state twice, because `fc_hidden` has already carried it into
        # `fused`: the layer would see roughly "2x state + a token hint", which is not the
        # formulation and lands the head at a plausible-looking but wrong operating point.
        # `POCKETLLM_QWEN4_MTP_INPUT=add` keeps that older reading available to measure against.
        stream = self.layer(
            fused + hidden_states if self.input_mode == "add" else fused,
            cos=cos,
            sin=sin,
            cache=None if cache is None else cache.layers[MTP_LAYER_INDEX],
            past_len=past_len,
            token_history=None,
            use_cache=cache is not None,
        )
        mixed = self.final_mixer(stream)
        # `stream` is the four-stream stack the next draft step has to be conditioned on: drafting
        # several tokens from one backbone step is a chain of head calls, and each link feeds the
        # previous link's stack back in as `hidden_states`.  Handing back the mixed (collapsed)
        # vector instead would lose the streams the fusion reads.
        return (mixed, stream) if return_streams else mixed


def _layer_config(config, weights: dict[str, torch.Tensor]):
    """`config` with layer index 0 declared as the attention flavour the MTP weights are.

    A shallow copy whose `layer_types` is a fresh one-entry list: `Qwen4ExpDecoderLayer` asks
    `is_linear_layer(0)` and nothing else of the list, so this is the whole of the lie.
    """
    import copy

    out = copy.copy(config)
    out.layer_types = [_attention_flavour(config, weights)]
    return out


__all__ = ["MTPHead", "MTP_PREFIX", "MTP_LAYER_INDEX"]