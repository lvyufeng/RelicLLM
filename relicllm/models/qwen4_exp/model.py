"""Qwen4-Exp decoder stack.

Weights arrive as plain tensors from a provider (see `weights.py`), so the same
module code serves the single-GPU reference path and the TP4 heterogeneous path
— the latter simply hands over sharded tensors and a host-resident MoE backend.
"""

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from relicllm.models.qwen4_exp.attention import (
    GatedDeltaNet,
    GatedDeltaNetCache,
    QSAAttention,
    QSAAttentionCache,
)
from relicllm.models.qwen4_exp.config import Qwen4ExpTextConfig
from relicllm.models.qwen4_exp.layers import (
    DenseMLP,
    GatedResidual,
    MRoPE,
    NGramHasher,
    RMSNorm,
    TopKRouter,
    inject_into_streams,
    prefill_linear,
)
from relicllm.models.qwen4_exp.moe import MoEBackend


@contextmanager
def _noop_context():
    """No-op context manager when profiler is disabled."""
    yield


#: How far a sequence's rotary table grows past the last position it was asked for, so a decode
#: step over a few hundred tokens extends it a handful of times rather than every step.
_ROPE_GROW = 256


# Decode drives the layers in one of three orders; `by_layer` is the reference answer.
#
# `by_phase` runs every layer's block `p` before any layer's block `p + 1`, so the stream carries 48
# partial pieces and a host read is paid once.  Measured on the 8x3090 box at batch 1 it was not
# faster (0.118 s/step against the loop's 0.198 in a resident-only run that also had the rope table
# rebuilt every step; once the table was fixed both schedules sat at ~0.198), so the pieces of a
# phase, not the number of drains, are what the step is made of.  `by_outer` defers only the
# outermost stream mix, which is the smallest version of the same idea.
_SCHEDULE = os.environ.get("POCKETLLM_QWEN4_SCHEDULE", "by_layer")


@dataclass
class Qwen4ExpCache:
    """Per-layer cache state plus the PLE token-history window.

    `ple_token_history` holds the last `ngram_size - 1` token ids so decode steps
    can hash n-grams that reach back before the current token.
    """

    layers: list[object] = field(default_factory=list)
    ple_token_history: torch.Tensor | None = None
    length: int = 0
    #: The rotary table grown so far -- absolute positions `[0, width)`, so every step reuses it.
    cos: torch.Tensor | None = None
    sin: torch.Tensor | None = None


class PLELayer:
    """Injects hashed n-gram features into every hyper-connection stream.

    The n-gram embedding is projected to one value and `hc_count` keys; the
    normalized stream activations gate the value per stream, then a dilated
    depthwise conv adds local lexical context.  The dilation is `ngram_size`, so
    the conv state spans `(kernel - 1) * ngram_size` positions.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        weights: dict[str, torch.Tensor],
        ple_layer_index: int,
        device: torch.device,
        *,
        ngram_table: object,
    ) -> None:
        self.config = config
        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count
        self.hasher = NGramHasher(config, ple_layer_index, device)
        self.ngram_table = ngram_table
        self.key_proj = weights["key_proj"]
        self.value_proj = weights["value_proj"]
        self.norm_key = RMSNorm(weights["norm_key"], config.rms_norm_eps, group_size=config.hidden_size)
        self.norm_query = RMSNorm(weights["norm_query"], config.rms_norm_eps, group_size=config.hidden_size)
        self.norm_conv = RMSNorm(weights["norm_conv"], config.rms_norm_eps, group_size=config.hidden_size)
        self.conv1d_weight = weights["conv1d"]  # (hc_hidden, 1, kernel)
        self.conv_dilation = config.ngram_size
        self.conv_kernel_size = config.ple_conv_kernel_size
        self.short_conv_state_len = (self.conv_kernel_size - 1) * self.conv_dilation
        self.conv_state: torch.Tensor | None = None

    def reset_cache(self) -> None:
        self.conv_state = None

    def _short_conv(self, hidden_states: torch.Tensor, *, use_cache: bool) -> torch.Tensor:
        seq_len = hidden_states.shape[1]
        x = hidden_states.transpose(1, 2)  # (b, hc_hidden, seq)
        if use_cache and self.conv_state is not None:
            x = torch.cat([self.conv_state.to(x.dtype), x], dim=-1)
        if use_cache:
            # In place when the shape allows: a captured step holds this buffer's address, and
            # rebinding the attribute every step leaves that address written and unread.
            state = x[:, :, -self.short_conv_state_len :]
            live = self.conv_state
            if live is not None and live.shape == state.shape:
                # Inside inference mode whatever the caller's context: the buffer was made there,
                # and an in-place update to an inference tensor is refused outside it.
                with torch.inference_mode():
                    live.copy_(state)
            else:
                with torch.inference_mode(False):
                    self.conv_state = state.clone()
        x = F.pad(x, (self.short_conv_state_len, 0))
        x = x[..., -(self.short_conv_state_len + seq_len) :]
        channels = self.conv1d_weight.shape[0]
        x = F.conv1d(
            x,
            self.conv1d_weight,
            None,
            padding=0,
            dilation=self.conv_dilation,
            groups=channels,
        )
        return F.silu(x).transpose(1, 2)

    def __call__(
        self,
        hidden_states: torch.Tensor,
        token_history: torch.Tensor,
        out_len: int,
        *,
        use_cache: bool,
    ) -> torch.Tensor:
        row_ids = self.hasher.row_ids(token_history, out_len)
        embeddings = self.ngram_table(row_ids).flatten(-2).to(hidden_states.dtype)
        key_normed = self.norm_key(prefill_linear(embeddings, self.key_proj)).unflatten(
            -1, (self.hc_count, self.hidden_size)
        )
        value = prefill_linear(embeddings, self.value_proj)
        query_normed = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        # Signed sqrt keeps the gate's sign while compressing its magnitude.
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_value_normed = self.norm_conv(gated_value.flatten(-2))
        gated_value = gated_value.flatten(-2)
        return gated_value + self._short_conv(gated_value_normed, use_cache=use_cache)


class Qwen4ExpDecoderLayer:
    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layer_idx: int,
        weights: dict[str, torch.Tensor],
        moe: MoEBackend,
        *,
        device: torch.device,
        ple: PLELayer | None = None,
        tp_shard: "TPShard | None" = None,
        all_reduce=None,
    ) -> None:
        self.config = config
        self.layer_idx = layer_idx
        self.is_linear = config.is_linear_layer(layer_idx)
        self.ple = ple
        self.moe = moe
        # Row-parallel outputs (attention o_proj/out_proj and the MLP sum) are
        # partial sums on each rank; the hyper-connection streams themselves are
        # replicated, so the reduce must land on the block output *before* it is
        # injected back into the streams.
        self.all_reduce = all_reduce
        shard = tp_shard or TPShard.single(config)

        if self.is_linear:
            self.attn = GatedDeltaNet(
                config,
                {k[len("linear_attn.") :]: v for k, v in weights.items() if k.startswith("linear_attn.")},
                num_v_heads=shard.linear_v_heads,
                num_k_heads=shard.linear_k_heads,
            )
        else:
            self.attn = QSAAttention(
                config,
                {k[len("self_attn.") :]: v for k, v in weights.items() if k.startswith("self_attn.")},
                num_heads=shard.attn_heads,
                num_kv_heads=shard.attn_kv_heads,
                indexer_heads=shard.indexer_heads,
            )

        self.attn_hc = GatedResidual(
            config,
            weights["attn_hyper_connection.hc_norm"],
            weights["attn_hyper_connection.input_mix_weight_down"],
            weights["attn_hyper_connection.input_mix_weight_up"],
            weights["attn_hyper_connection.block_inject_weight"],
        )
        self.mlp_hc = GatedResidual(
            config,
            weights["mlp_hyper_connection.hc_norm"],
            weights["mlp_hyper_connection.input_mix_weight_down"],
            weights["mlp_hyper_connection.input_mix_weight_up"],
            weights["mlp_hyper_connection.block_inject_weight"],
        )

        self.router = TopKRouter(weights["mlp.gate"], config.num_experts_per_tok, norm_topk_prob=config.norm_topk_prob)
        self.shared_expert = DenseMLP(
            weights["mlp.shared_expert.gate_proj"],
            weights["mlp.shared_expert.up_proj"],
            weights["mlp.shared_expert.down_proj"],
        )
        self.shared_expert_gate = weights["mlp.shared_expert_gate"]

    def make_cache(self, batch_size: int, max_seq_len: int, device: torch.device, dtype: torch.dtype):
        if self.is_linear:
            return GatedDeltaNetCache(
                self.config,
                batch_size,
                device,
                dtype,
                num_v_heads=self.attn.num_v_heads,
                num_k_heads=self.attn.num_k_heads,
            )
        return QSAAttentionCache(
            self.config,
            batch_size,
            max_seq_len,
            device,
            dtype,
            num_kv_heads=self.attn.num_kv_heads,
        )

    def __call__(
        self,
        hidden_states: torch.Tensor,
        *,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cache,
        past_len: int,
        token_history: torch.Tensor | None,
        use_cache: bool,
        profiler=None,
        defer_final_mix: bool = False,
    ) -> torch.Tensor:
        prof_scope = profiler.scope if profiler else lambda x: _noop_context()

        with prof_scope(f"layer_{self.layer_idx}"):
            if defer_final_mix and isinstance(hidden_states, tuple):
                block_output, hyper_input, inject = hidden_states
                if hyper_input is None:
                    hidden_states = block_output
                else:
                    with prof_scope("deferred_inject"):
                        hidden_states = inject_into_streams(block_output, hyper_input, inject)
            if self.ple is not None:
                assert token_history is not None, "PLE layers need the token id history"
                with prof_scope("ple"):
                    ple_out = self.ple(
                        hidden_states, token_history, hidden_states.shape[1], use_cache=use_cache
                    )
                hidden_states = hidden_states + ple_out

            with prof_scope("attn_hc"):
                mixed, hyper_input, inject = self.attn_hc(hidden_states)

            with prof_scope("attention"):
                if self.is_linear:
                    attn_out = self.attn(mixed, cache)
                else:
                    attn_out = self.attn(mixed, cos, sin, cache, past_len=past_len)

            if profiler is not None and hasattr(self.attn, "last_profile"):
                for name, elapsed in self.attn.last_profile.items():
                    profiler.add_external(f"attention_{name}", elapsed)
                self.attn.last_profile.clear()

            if self.all_reduce is not None:
                with prof_scope("attn_reduce"):
                    attn_out = self.all_reduce(attn_out)

            with prof_scope("attn_inject"):
                hidden_states = inject_into_streams(attn_out, hyper_input, inject)

            with prof_scope("mlp_hc"):
                mixed, hyper_input, inject = self.mlp_hc(hidden_states)

            flat = mixed.reshape(-1, mixed.shape[-1])

            with prof_scope("router"):
                weights, indices = self.router(flat)

            with prof_scope("moe"):
                routed = self.moe(self.layer_idx, flat, indices, weights)

            with prof_scope("shared_expert"):
                shared = self.shared_expert(flat)
                shared = torch.sigmoid(
                    prefill_linear(flat, self.shared_expert_gate)
                ) * shared

            mlp_out = (routed + shared).reshape(mixed.shape)

            if self.all_reduce is not None:
                with prof_scope("mlp_reduce"):
                    mlp_out = self.all_reduce(mlp_out)

            if defer_final_mix:
                # `by_outer` carries the stream as its pieces and mixes them once at the end.
                return mlp_out, hyper_input, inject
            with prof_scope("mlp_inject"):
                return inject_into_streams(mlp_out, hyper_input, inject)


@dataclass
class TPShard:
    """Head counts owned by this rank after tensor-parallel splitting."""

    attn_heads: int
    attn_kv_heads: int
    indexer_heads: int
    linear_v_heads: int
    linear_k_heads: int

    @classmethod
    def single(cls, config: Qwen4ExpTextConfig) -> "TPShard":
        return cls(
            attn_heads=config.num_attention_heads,
            attn_kv_heads=config.num_key_value_heads,
            indexer_heads=config.indexer_n_heads or 0,
            linear_v_heads=config.linear_num_value_heads,
            linear_k_heads=config.linear_num_key_heads,
        )


class Qwen4ExpModel:
    """The 48-layer decoder stack with 4-stream hyper-connections."""

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        layers: list[Qwen4ExpDecoderLayer | None],
        *,
        embed_tokens,
        lm_head: torch.Tensor | None,
        final_mixer: GatedResidual | None,
        device: torch.device,
        dtype: torch.dtype,
        pp_size: int = 1,
        pp_rank: int = 0,
        profiler=None,
    ) -> None:
        self.config = config
        self.layers = layers
        self.embed_tokens = embed_tokens
        self.lm_head = lm_head
        self.final_mixer = final_mixer
        self.device = device
        self.dtype = dtype
        self.profiler = profiler
        #: ``layers`` is full length with ``None`` at the slots this stage does not own, so an entry
        #: index is always the absolute layer id. Under ``pp_size == 1`` nothing is ``None``.
        self.pp_size = pp_size
        self.pp_rank = pp_rank
        self.owns_embedding = embed_tokens is not None
        self.owns_head = lm_head is not None
        self.rope = MRoPE(config, device)
        self._cos: torch.Tensor | None = None
        self._sin: torch.Tensor | None = None

    @property
    def local_layers(self) -> list[Qwen4ExpDecoderLayer]:
        """The layers this rank actually holds, in order, with the placeholders dropped."""
        return [layer for layer in self.layers if layer is not None]

    def make_cache(self, batch_size: int, max_seq_len: int) -> Qwen4ExpCache:
        # One slot per absolute layer id, `None` where this stage owns no layer -- the same
        # convention as `self.layers`, so `cache.layers[idx]` stays valid for the id it is asked for.
        cache = Qwen4ExpCache(
            layers=[
                layer.make_cache(batch_size, max_seq_len, self.device, self.dtype)
                if layer is not None
                else None
                for layer in self.layers
            ]
        )
        for layer in self.layers:
            if layer is not None and layer.ple is not None:
                layer.ple.reset_cache()
        cache.ple_token_history = torch.full(
            (batch_size, self.config.ngram_size - 1),
            self.config.primary_eos_token_id,
            dtype=torch.long,
        )
        return cache

    def _rope_cache(
        self, total_len: int, batch_size: int, cache: Qwen4ExpCache | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """cos/sin for absolute positions [0, total_len). The QSA indexer needs
        every past position, not just the current ones, so we keep the full span
        and grow it in place."""
        if cache is None:
            # A one-shot forward: the table is this call's alone, so it is not kept.
            position_ids = torch.arange(total_len, device=self.device).view(1, 1, -1).expand(3, batch_size, -1)
            cos, sin = self.rope(position_ids, self.dtype)
            return cos[:, :total_len], sin[:, :total_len]
        # The table belongs to the sequence, so it lives on the cache and spans what was grown
        # for it: kept on the model, the next call sees the previous call's *slice* and rebuilds
        # the whole span every step, which at a few hundred tokens is most of a decode step.
        # A caller that laid the whole generation out -- a capture has to, because a table that
        # grows on demand reallocates inside the recording -- says how wide the table is, and the
        # slice is then taken at a position that may be a device tensor.
        fixed = getattr(cache, "rope_width", 0)
        if isinstance(total_len, torch.Tensor):
            # `cos` is `(3, positions, head_dim/2)`: the span is dim 1, and reading the last dim
            # here compared the half-head width against the span and refused every capture, which
            # is what this cost a round to find.
            if not fixed or cache.cos is None or cache.cos.shape[1] < fixed:
                raise RuntimeError(
                    "a tensor past_len needs cache.rope_width set and the table built for it"
                )
            # A slice boundary has to be a host integer, and reading one off the device is what a
            # capture refuses -- so the row is taken by index on the device instead.  The result
            # is (3, 1, head_dim/2), the shape the eager path's slice produces.
            if total_len.dim() == 0:
                total_len = total_len.reshape(1)
            return (
                cache.cos.index_select(1, total_len),
                cache.sin.index_select(1, total_len),
            )
        width = -(-total_len // _ROPE_GROW) * _ROPE_GROW
        if cache.cos is None or cache.cos.shape[0] != batch_size or cache.cos.shape[1] < width:
            width = max(width, fixed, total_len)
            position_ids = torch.arange(width, device=self.device).view(1, 1, -1).expand(3, batch_size, -1)
            cache.cos, cache.sin = self.rope(position_ids, self.dtype)
        return cache.cos[:, :total_len], cache.sin[:, :total_len]

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        cache: Qwen4ExpCache | None = None,
        past_len: int = 0,
        last_token_only: bool = False,
    ) -> torch.Tensor | None:
        """Run this rank's layers.

        Under pipeline parallelism this is one stage of a chain: the first stage embeds, each stage
        runs its own layers and hands the hidden state to the next, and only the last stage returns
        logits -- every other stage returns ``None``. That ``None`` is the contract the caller reads
        to know it should not gather or argmax. ``pp_size == 1`` keeps the old single-stage path.

        The token ids travel with the hidden state, not just once at the front: every stage's PLE
        layers need the running token history, and on the decode path each step's ids are produced by
        the *last* stage's argmax, so a stage that did not receive them would build a history that
        silently diverges from the one that produced the token.
        """
        prof_scope = self.profiler.scope if self.profiler else lambda x: _noop_context()

        if self.pp_size > 1:
            return self._forward_stage(
                input_ids, cache=cache, past_len=past_len, last_token_only=last_token_only
            )
        return self._forward_all(
            input_ids, cache=cache, past_len=past_len, last_token_only=last_token_only
        )

    # -- pipeline stage helpers -------------------------------------------------

    def _activation_shape(self, batch_size: int, seq_len: int) -> tuple[int, int, int]:
        # `[batch, seq, hc_count * hidden]`, the layout the layers themselves take: each unpacks it
        # with `unflatten(-1, (hc_count, hidden))` rather than being handed the two axes separately.
        return (batch_size, seq_len, self.config.hc_hidden_size)

    def _send_activation(self, hidden: torch.Tensor, input_ids: torch.Tensor, dst: int) -> None:
        import torch.distributed as dist

        dist.send(hidden.contiguous(), dst=dst)
        # The ids ride the same hop so every stage's token history is the sequence that actually
        # ran, not one it reconstructed. Moved to this rank's device first: the prompt ids arrive on
        # the host, and NCCL cannot send a CPU tensor.
        dist.send(input_ids.to(device=self.device, dtype=torch.long).contiguous(), dst=dst)

    def _recv_activation(
        self, input_ids: torch.Tensor, src: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import torch.distributed as dist

        # `input_ids` is the same tensor the sender's `forward` was called with, so its leading two
        # axes are the activation's batch and sequence. Read them with `shape[:2]` rather than an
        # unpack so a caller that hands in something with a trailing axis cannot confuse this.
        batch_size, seq_len = input_ids.shape[:2]
        hidden = torch.empty(
            self._activation_shape(batch_size, seq_len), device=self.device, dtype=self.dtype
        )
        ids = torch.empty((batch_size, seq_len), device=self.device, dtype=torch.long)
        dist.recv(hidden, src=src)
        dist.recv(ids, src=src)
        return hidden, ids

    def _forward_stage(
        self,
        input_ids: torch.Tensor,
        *,
        cache: Qwen4ExpCache | None,
        past_len: int,
        last_token_only: bool,
    ) -> torch.Tensor | None:
        prof_scope = self.profiler.scope if self.profiler else lambda x: _noop_context()

        if self.pp_rank == 0:
            hidden = self.embed_tokens(input_ids).to(self.dtype)
            hidden = hidden.repeat(1, 1, self.config.hc_count)
            ids = input_ids
        else:
            hidden, ids = self._recv_activation(input_ids, self.pp_rank - 1)

        hidden = self._forward_layers(hidden, ids, cache=cache, past_len=past_len, prof_scope=prof_scope)

        if self.pp_rank + 1 < self.pp_size:
            self._send_activation(hidden, ids, self.pp_rank + 1)
            return None

        if last_token_only:
            hidden = hidden[:, -1:]
        with prof_scope("final_mixer"):
            hidden = self.final_mixer(hidden)
        with prof_scope("lm_head"):
            return F.linear(hidden, self.lm_head)

    def _forward_all(
        self,
        input_ids: torch.Tensor,
        *,
        cache: Qwen4ExpCache | None,
        past_len: int,
        last_token_only: bool,
    ) -> torch.Tensor:
        prof_scope = self.profiler.scope if self.profiler else lambda x: _noop_context()

        with prof_scope("embed"):
            hidden = self.embed_tokens(input_ids).to(self.dtype)
            hidden = hidden.repeat(1, 1, self.config.hc_count)

        hidden = self._forward_layers(
            hidden, input_ids, cache=cache, past_len=past_len, prof_scope=prof_scope
        )

        if last_token_only:
            hidden = hidden[:, -1:]
        with prof_scope("final_mixer"):
            hidden = self.final_mixer(hidden)
        with prof_scope("lm_head"):
            return F.linear(hidden, self.lm_head)

    def _forward_outer_phased(
        self,
        hidden: torch.Tensor,
        layers: list,
        cache: Qwen4ExpCache,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_len: int,
        token_history: torch.Tensor | None,
        prof_scope,
    ) -> torch.Tensor:
        """`by_outer`: the stream is never mixed inside a layer, only once, at the end.

        A layer's body is bound to the stream at its first use, so this is the per-layer loop up to
        that first site: the stream is carried from one layer to the next *as its three partial
        pieces*, and every mix the loop does at the end of a layer is done once afterwards, in
        order.  That is arithmetic on the same pieces, so the only thing it can change is where
        the wait is paid, which is the point: it says whether the loop's cost is its dependency
        chain or its energy.
        """
        block_output, hyper_input, inject = hidden, None, None
        deferred = []
        for idx, layer in layers:
            block_output, hyper_input, inject = layer(
                (block_output, hyper_input, inject),
                cos=cos,
                sin=sin,
                cache=cache.layers[idx],
                past_len=past_len,
                token_history=token_history,
                use_cache=True,
                profiler=self.profiler,
                defer_final_mix=True,
            )
            deferred.append((block_output, hyper_input, inject))
        streams = hidden
        for deferred_block, hyper_input, inject in deferred:
            streams = inject_into_streams(deferred_block, hyper_input, inject)
        return streams

    def _forward_phased(
        self,
        hidden: torch.Tensor,
        layers: list,
        cache: Qwen4ExpCache,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_len: int,
        token_history: torch.Tensor | None,
        prof_scope,
    ) -> torch.Tensor:
        """One decode token through every layer, a block at a time across all of them.

        The schedule is the per-layer loop's, flattened: layer `i`'s attention is still run before
        layer `i`'s hyper-connection output, and layer `i` finishes before layer `i + 1` reads the
        stream.  What changes is the order *the host issues* its work in.  A block's output is a
        host-dependent wait -- an expert row read off the host, or a collective -- and in the
        per-layer loop every such wait is paid before the next layer's independent compute is
        queued.  Here every layer's row read is issued in one block pass, so the stream drains once
        instead of once a layer, and the two collectives are issued 48 times and waited on
        together.  The answer is the loop's: the queue order between independent reads and copies
        changes, not the arithmetic.  It also removes a per-layer host round trip in this tree,
        which makes the blocks that run on the host (experts, rope) the whole of what is left.
        """
        n = len(layers)
        h = hidden

        with prof_scope("dec_ple_hc"):
            for i, (_, layer) in enumerate(layers):
                if layer.ple is not None:
                    h = h + layer.ple(h, token_history, 1, use_cache=True)
                layer._phased_hc = layer.attn_hc(h)
        with prof_scope("dec_attn"):
            for i, (idx, layer) in enumerate(layers):
                mixed = layer._phased_hc[0]
                if layer.is_linear:
                    attn = layer.attn(mixed, cache.layers[idx])
                else:
                    attn = layer.attn(mixed, cos, sin, cache.layers[idx], past_len=past_len)
                ar = layer.all_reduce
                layer._phased_attn = attn if ar is None else ar(attn)
        with prof_scope("dec_attn_inject"):
            for _, layer in layers:
                mixed, hyper_input, inject = layer._phased_hc
                h = inject_into_streams(layer._phased_attn, hyper_input, inject)
                mixed, hyper_input, inject = layer.mlp_hc(h)
                layer._phased_hc = (mixed, hyper_input, inject)
        with prof_scope("dec_router"):
            for _, layer in layers:
                flat = layer._phased_hc[0].reshape(-1, layer._phased_hc[0].shape[-1])
                layer._phased_route = layer.router(flat)
        with prof_scope("dec_moe"):
            for idx, layer in layers:
                flat = layer._phased_hc[0].reshape(-1, layer._phased_hc[0].shape[-1])
                layer._phased_moe = layer.moe(idx, flat, *layer._phased_route)
        with prof_scope("dec_shared"):
            for _, layer in layers:
                flat = layer._phased_hc[0].reshape(-1, layer._phased_hc[0].shape[-1])
                layer._phased_moe = layer._phased_moe + torch.sigmoid(
                    prefill_linear(flat, layer.shared_expert_gate)
                ) * layer.shared_expert(flat)
        with prof_scope("dec_mlp_reduce"):
            for _, layer in layers:
                mlp_out = layer._phased_moe.reshape(layer._phased_hc[0].shape)
                ar = layer.all_reduce
                layer._phased_moe = mlp_out if ar is None else ar(mlp_out)
        with prof_scope("dec_mlp_inject"):
            for _, layer in layers:
                _, hyper_input, inject = layer._phased_hc
                h = inject_into_streams(layer._phased_moe, hyper_input, inject)
                layer._phased_hc = layer._phased_attn = layer._phased_moe = layer._phased_route = None
        return h

    def _pinned_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """`input_ids` in host memory, through a buffer whose address a graph can keep.

        One row a step, so the buffer is sized per call and kept: a token id is 8 bytes and this
        runs once a forward, which is not worth a growing pool to avoid allocating.
        """
        rows, width = int(input_ids.shape[0]), int(input_ids.shape[1])
        buffer = self.__dict__.get("_ids_pinned")
        if buffer is None or buffer.shape[0] < rows or buffer.shape[1] < width:
            # Allocated outside inference mode: a buffer made inside it is an inference tensor,
            # and an inference tensor refuses the in-place update the next call makes.
            with torch.inference_mode(False):
                buffer = torch.empty(rows, width, dtype=torch.long, pin_memory=True)
            self.__dict__["_ids_pinned"] = buffer
        # The rows a batch asks for, not a fixed single row: a decode at batch two used to write
        # a two-row tensor into a one-row buffer and fail on the broadcast.
        pinned = buffer[:rows, :width]
        pinned.copy_(input_ids, non_blocking=True)
        return pinned

    def _forward_by_layer(
        self,
        hidden: torch.Tensor,
        layers: list,
        cache: Qwen4ExpCache | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_len: int,
        token_history: torch.Tensor | None,
    ) -> torch.Tensor:
        """The per-token loop every call has always run: one layer end to end, then the next."""
        for idx, layer in layers:
            hidden = layer(
                hidden,
                cos=cos,
                sin=sin,
                cache=cache.layers[idx],
                past_len=past_len,
                token_history=token_history,
                use_cache=cache is not None,
                profiler=self.profiler,
            )
        return hidden

    def _forward_layers(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        cache: Qwen4ExpCache | None,
        past_len: int,
        prof_scope,
    ) -> torch.Tensor:
        """The shared middle: rope, PLE history, this rank's layers, cache bookkeeping.

        ``idx`` is the absolute layer id in both branches, which is the whole reason ``self.layers``
        keeps its ``None`` placeholders: it is what indexes each layer's KV slot and its entry in the
        host expert cache, so a compacted list would silently renumber them.
        """
        batch_size, seq_len = input_ids.shape
        total_len = past_len + seq_len

        with prof_scope("rope_cache"):
            cos, sin = self._rope_cache(total_len, batch_size, cache)

        token_history = None
        if any(layer is not None and layer.ple is not None for layer in self.layers):
            with prof_scope("token_history"):
                # Token ids and the n-gram hash live on the CPU (the PLE table is
                # host-resident), so keep the history there too.  The crossing goes into a pinned
                # buffer: a pageable one is a copy the driver stages through the CPU, and a
                # captured step refuses it outright rather than slow down for it.
                # A caller that keeps the window itself -- a capture has to, because the ids it
                # is about to embed do not exist when the recording is made -- hands over the
                # whole row, and no device-to-host copy happens here at all.
                staged = (
                    getattr(cache, "ple_history_at", None) if cache is not None else None
                )
                if staged is not None:
                    token_history = staged(input_ids)
                else:
                    ids_cpu = self._pinned_ids(input_ids)
                    if cache is not None and cache.ple_token_history is not None:
                        token_history = torch.cat([cache.ple_token_history, ids_cpu], dim=1)
                    else:
                        pad = torch.full(
                            (batch_size, self.config.ngram_size - 1),
                            self.config.primary_eos_token_id,
                            dtype=torch.long,
                        )
                        token_history = torch.cat([pad, ids_cpu], dim=1)

        # How the layers are driven. `by_layer` is the per-token loop every call has always run.
        # `by_phase` runs the same blocks in the same order with the same inputs, but walks the
        # layers inside each phase, so a host read only has to cross the stream once and a
        # collective is not waited on before the next layer's compute can be queued. It is what a
        # decode step of this size wants; prefill keeps the per-layer loop, whose host traffic it
        # already covers with a chunk.
        layers = [(idx, layer) for idx, layer in enumerate(self.layers) if layer is not None]
        if cache is not None and seq_len == 1 and len(layers) > 1 and _SCHEDULE != "by_layer":
            run = self._forward_phased if _SCHEDULE == "by_phase" else self._forward_outer_phased
            hidden = run(hidden, layers, cache, cos, sin, past_len, token_history, prof_scope)
        else:
            hidden = self._forward_by_layer(hidden, layers, cache, cos, sin, past_len, token_history)

        if cache is not None:
            cache.length = total_len
            if token_history is not None and getattr(cache, "ple_history_at", None) is None:
                cache.ple_token_history = token_history[:, -(self.config.ngram_size - 1) :].clone()
        return hidden
