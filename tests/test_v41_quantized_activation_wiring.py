"""The wiring that threads the quantized-activation flag from the config to the call sites.

`tests/test_v41_quantized_activation_path.py` pins the *primitive* -- `quantized_linear` and the
block-32 `block_fp8_act_quant` beneath it -- against the reference's stated arithmetic. This file
pins the other half: that the flag the config decides actually arrives at the right call sites, that
it is recorded once at construction rather than re-derived from a weight, and that the sites the
reference leaves on the plain bf16 path are left there.

The distinction matters because the two failures are invisible from either end. A primitive that is
correct but never called is the port's *original* bug: every token still decodes, every tolerance
still passes at the widths the tests use, and the only symptom is a logit that is close but not
equal. And a site wired by mistake -- `wo_a`, `weights_proj`, `wk`, the compressor, or one of the
K/Q/V cache quantizations -- moves a tensor the reference never quantized, which is an error of the
same size in the other direction and just as quiet.

So there are four things checked here, none of which needs a card or the reference runtime:

* **The flag reaches both halves.** A dense site (`Attention`, and the `Engram`'s `wkv`) and the
  routed store (`ResidentRoutedExperts` through `MoE`) each produce a different forward under
  `dtype="fp8"` / `expert_dtype="fp4"` than under `dtype="bf16"`, on identical weights.
* **Which sites, exactly.** Every `quantized=True` call in a forward of a layer that runs both a
  compressor and an indexer uses one of the reference's fp8 weights, and none uses a bf16-declared
  one. That is the "not changed" list made executable.
* **The two orderings the reference fixes.** `w1` and `w3` each quantize the *same raw* activation --
  independently, not folded -- and `w2` quantizes the post-`.to(dtype)` value, not the fp32
  intermediate. Both are asserted at the primitive's input rather than through a tolerance, because
  a folded or mis-ordered path can be close enough to pass an output comparison.
* **The seam is still the seam.** `quantized=False` is `torch.equal` to the arithmetic the port ran
  before, and `expert_forward`'s positional signature is unchanged, which is what keeps
  `tests/test_models_deepseek_v4_1_modules.py` and the bf16 parity build where they were.

Every dimension here is a multiple of 32 -- the fp8 activation block -- so a site that has the flag
set does not fail on a ragged last block and be read as a wiring bug. The file skips without torch,
the same guard every V4.1 sibling uses.
"""

from __future__ import annotations

import pytest

# Before any project import: a torch-less host has to skip this file rather than fail to collect it.
torch = pytest.importorskip("torch")

import torch.nn.functional as F  # noqa: E402

from relicllm.models.deepseek_v4_1 import attention as attention_module  # noqa: E402
from relicllm.models.deepseek_v4_1 import kernels, modules as modules_module  # noqa: E402
from relicllm.models.deepseek_v4_1.attention import (  # noqa: E402
    Attention,
    SharedAttentionRuntime,
)
from relicllm.models.deepseek_v4_1.config import V41TextConfig  # noqa: E402
from relicllm.models.deepseek_v4_1.engram import EngramLayout  # noqa: E402
from relicllm.models.deepseek_v4_1.modules import (  # noqa: E402
    Engram,
    MoE,
    ResidentEngramTable,
    expert_forward,
)

# Geometry. Every GEMM's contraction axis here is 32, 64 or 192, all multiples of the fp8 block, and
# `attn_quant` size is what makes a missing flag show up as a wrong number rather than a `ValueError`
# from a ragged last block.
DIM = 64
INTER_DIM = 64
N_EXPERTS = 4
TOPK = 2
ENGRAM_HEAD_DIM = 32
ENGRAM_ROWS = 64

# One config the whole file shares: attention geometry from the attention sibling's `TOY`, MoE fields
# from the modules sibling, Engram fields from its `_ENGRAM_FIELDS`. Layer 2 is both a KV source and
# an index source, so a forward of it exercises the compressor (excluded from this seam) *and* the
# indexer's `wq_b` (part of it) -- which is the pair the "which sites, exactly" test needs.
_BASE = dict(
    dim=DIM,
    moe_inter_dim=INTER_DIM,
    n_routed_experts=N_EXPERTS,
    n_shared_experts=1,
    n_activated_experts=TOPK,
    score_func="sqrtsoftplus",
    route_scale=1.5,
    swiglu_limit=10.0,
    n_layers=6,
    n_mtp_layers=0,
    n_heads=4,
    head_dim=32,
    rope_head_dim=8,
    q_lora_rank=32,
    o_groups=2,
    o_lora_rank=16,
    window_size=4,
    compress_ratios=(0, 0, 2, 2, 1, 1),
    kv_source_layers=(2, 4),
    index_source_layers=(2, 4, 5),
    index_n_heads=2,
    index_head_dim=32,
    index_topk=16,
    candidate_source_layer=4,
    candidate_topk_blocks=16,
    candidate_block_size=2,
    norm_eps=1e-6,
    hc_mult=2,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    rope_theta=10000.0,
    compress_rope_theta=160000.0,
    rope_factor=40.0,
    beta_fast=32,
    beta_slow=1,
    original_seq_len=512,
    max_position_embeddings=64,
    engram_layer_ids=(0,),
    engram_num_embeddings=(ENGRAM_ROWS,),
    engram_max_ngram_size=2,
    engram_vocab_size=32,
    engram_n_heads=2,
    engram_head_dim=ENGRAM_HEAD_DIM,
)


def _cfg(**overrides) -> V41TextConfig:
    values = dict(_BASE)
    values.update(overrides)
    return V41TextConfig(**values)


def _fill(module: torch.nn.Module, seed: int = 0) -> torch.nn.Module:
    """Give every parameter a reproducible value.

    Every module here is built out of `torch.empty`, and a twin comparison is only a comparison if
    both twins hold the *same* numbers -- the flag is meant to be the only difference. Drawing from
    one seeded generator in parameter order gives two modules of the same architecture identical
    weights, which is what the "reaches both halves" tests rest on.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            values = torch.randn(parameter.shape, generator=generator, dtype=torch.float32) * 0.2
            parameter.copy_(values.to(parameter.dtype))
    return module


def _linear_expert(x, w1, w2, w3, limit, weights):
    """`expert_forward`'s `quantized=False` path written out, independent of the module under test."""
    gate = F.linear(x, w1).float()
    up = F.linear(x, w3).float()
    if limit > 0:
        up = up.clamp(min=-limit, max=limit)
        gate = gate.clamp(max=limit)
    out = F.silu(gate) * up
    if weights is not None:
        out = weights * out
    return F.linear(out.to(x.dtype), w2)


def test_the_dense_flag_reaches_the_attention_projections() -> None:
    """Twin-built `Attention`, `fp8` against `bf16`, identical weights: the fp8 forward differs.

    Layer 0 compresses nothing, so the four projections -- `wq_a`, `wq_b`, `wkv`, `wo_b` -- are the
    whole of this layer's quantized set and the difference cannot come from a compressor or an
    indexer. If the flag were dropped anywhere in the chain the two forwards would be `torch.equal`,
    which is the failure this is written to catch.
    """
    x = torch.randn(1, 4, DIM, dtype=attention_module.LINEAR_DTYPE)
    bf16_attn = _fill(Attention(0, _cfg(dtype="bf16")), seed=0)
    fp8_attn = _fill(Attention(0, _cfg(dtype="fp8")), seed=0)

    assert bf16_attn._quantized is False
    assert fp8_attn._quantized is True

    out_bf16 = bf16_attn(x, 0, SharedAttentionRuntime())
    out_fp8 = fp8_attn(x, 0, SharedAttentionRuntime())

    gap = (out_fp8.float() - out_bf16.float()).abs().max().item()
    assert gap > 1e-4, f"the fp8 and bf16 attention forwards differ by only {gap}, too little to be quant"

    # The negative control: two bf16 twins are bit-identical, so the gap above is the flag and not the
    # fill drawing different weights for the two builds.
    out_bf16_second = _fill(Attention(0, _cfg(dtype="bf16")), seed=0)(x, 0, SharedAttentionRuntime())
    assert torch.equal(out_bf16, out_bf16_second)


def test_the_dense_flag_reaches_the_engram_wkv() -> None:
    """`Engram.wkv` is declared fp8 in the reference too, so it takes the same flag."""
    layout = EngramLayout.from_config(_cfg().__dict__)
    torch.manual_seed(0)
    table = ResidentEngramTable(
        weight=torch.randn(ENGRAM_ROWS, ENGRAM_HEAD_DIM).to(torch.float8_e4m3fn),
        scale=torch.ones(ENGRAM_ROWS, 1).to(torch.float8_e8m0fnu),
    )
    x = torch.randn(1, 4, 2, DIM, dtype=modules_module.LINEAR_DTYPE)
    ids = torch.randint(0, ENGRAM_ROWS, (1, 4, layout.n_hash_columns))

    off = _fill(Engram(_cfg(), 0, layout, table, quantized=False), seed=0)
    on = _fill(Engram(_cfg(), 0, layout, table, quantized=True), seed=0)
    assert off._quantized is False and on._quantized is True

    a = off(x, ids)
    b = on(x, ids)
    assert not torch.allclose(a.float(), b.float(), atol=1e-5), (
        "the Engram's wkv runs the same arithmetic with and without the flag"
    )


def test_the_routed_flag_reaches_the_store_and_the_shared_expert() -> None:
    """`MoE` records the flag on both halves, and it changes the routed forward's numbers.

    The two halves read different config fields -- the shared expert follows `dtype`, the routed
    store follows `expert_dtype` -- so a port that wired one and not the other is caught here rather
    than by a tolerance somewhere downstream.
    """
    moe_bf16 = _fill(MoE(_cfg(dtype="bf16", expert_dtype=None), 0, N_EXPERTS, TOPK), seed=1)
    moe_fp8 = _fill(MoE(_cfg(dtype="fp8", expert_dtype="fp4"), 0, N_EXPERTS, TOPK), seed=1)

    assert moe_bf16.routed._quantized is False
    assert moe_bf16.shared_experts._quantized is False
    assert moe_fp8.routed._quantized is True
    assert moe_fp8.shared_experts._quantized is True

    x = torch.randn(2, DIM, dtype=modules_module.LINEAR_DTYPE)
    weights = torch.ones(2, TOPK, dtype=torch.float32)
    indices = torch.arange(TOPK).repeat(2, 1)

    off = moe_bf16.routed(x, weights, indices)
    on = moe_fp8.routed(x, weights, indices)
    gap = (on.float() - off.float()).abs().max().item()
    assert gap > 1e-4, f"the routed store's two builds differ by only {gap}"


def test_no_scales_is_still_exactly_the_linear_it_replaced() -> None:
    """The seam the whole design rests on: `quantized=False` is `F.linear`, bit for bit.

    A module whose flag is off has to produce the arithmetic the port ran before the fix, or the bf16
    parity build moves for a reason that has nothing to do with the reference. `torch.equal` and not
    `allclose`: this is the same call, not a good approximation of it.
    """
    torch.manual_seed(2)
    x = torch.randn(3, DIM, dtype=modules_module.LINEAR_DTYPE)
    w1 = torch.randn(INTER_DIM, DIM, dtype=modules_module.LINEAR_DTYPE) * 0.2
    w2 = torch.randn(DIM, INTER_DIM, dtype=modules_module.LINEAR_DTYPE) * 0.2
    w3 = torch.randn(INTER_DIM, DIM, dtype=modules_module.LINEAR_DTYPE) * 0.2
    weights = torch.ones(3, 1, dtype=torch.float32)

    off = expert_forward(x, w1, w2, w3, 10.0, weights, quantized=False)
    assert torch.equal(off, _linear_expert(x, w1, w2, w3, 10.0, weights))

    on = expert_forward(x, w1, w2, w3, 10.0, weights, quantized=True)
    assert not torch.equal(on, off), "turning the flag on changed nothing, so the flag is not threaded"


def test_w1_and_w3_each_quantize_the_same_activation_separately(monkeypatch) -> None:
    """The reference is two `Linear` calls, not one: `x` goes through the block quantizer twice.

    Folding them -- quantizing once and reusing the codes -- is the natural optimization and it is a
    different number, so this asserts the *input* to the quantizer on both projections rather than a
    tolerance on the output. The fp8 activation that feeds `w1` and the one that feeds `w3` are the
    same raw `x`, arrived at independently.
    """
    seen: list[torch.Tensor] = []
    real = kernels.block_fp8_act_quant

    def spy(value, block_size=32):
        seen.append(value.detach().clone())
        return real(value, block_size)

    monkeypatch.setattr(kernels, "block_fp8_act_quant", spy)

    torch.manual_seed(3)
    x = torch.randn(2, DIM, dtype=modules_module.LINEAR_DTYPE)
    w1 = torch.randn(INTER_DIM, DIM, dtype=modules_module.LINEAR_DTYPE) * 0.2
    w2 = torch.randn(DIM, INTER_DIM, dtype=modules_module.LINEAR_DTYPE) * 0.2
    w3 = torch.randn(INTER_DIM, DIM, dtype=modules_module.LINEAR_DTYPE) * 0.2

    # swiglu_limit 0 keeps the clamp out of the way so the three quantizations are unambiguous.
    expert_forward(x, w1, w2, w3, 0.0, None, quantized=True)

    assert len(seen) == 3, f"{len(seen)} quantizations, expected one each for w1, w3 and w2"
    assert torch.equal(seen[0], x), "w1 did not quantize the raw activation"
    assert torch.equal(seen[1], x), "w3 did not quantize the raw activation"
    # Two separate calls on the same value, which a folded path would collapse into one.
    assert seen[0].shape == seen[1].shape
    assert not torch.equal(seen[2], x), "w2 quantized the activation, not the gate/up product"


def test_w2_quantizes_the_post_cast_activation_not_the_fp32_one(monkeypatch) -> None:
    """The reference passes `w2` the value *after* `.to(dtype)`, so the quantizer sees the cast one.

    Doing the block quantization on the fp32 intermediate and narrowing afterwards is a different
    number, and the difference is below what an output comparison at these widths would notice. The
    discriminator here is the dtype the primitive was handed plus the exact value, both of which the
    reference fixes and neither of which a tolerance can see.
    """
    seen: list[torch.Tensor] = []
    real = kernels.block_fp8_act_quant

    def spy(value, block_size=32):
        seen.append(value.detach().clone())
        return real(value, block_size)

    monkeypatch.setattr(kernels, "block_fp8_act_quant", spy)

    torch.manual_seed(4)
    x = torch.randn(2, DIM, dtype=modules_module.LINEAR_DTYPE)
    w1 = torch.randn(INTER_DIM, DIM, dtype=modules_module.LINEAR_DTYPE) * 0.2
    w2 = torch.randn(DIM, INTER_DIM, dtype=modules_module.LINEAR_DTYPE) * 0.2
    w3 = torch.randn(INTER_DIM, DIM, dtype=modules_module.LINEAR_DTYPE) * 0.2
    limit = 10.0

    expert_forward(x, w1, w2, w3, limit, None, quantized=True)

    gate = F.linear(x, w1).float().clamp(max=limit)
    up = F.linear(x, w3).float().clamp(min=-limit, max=limit)
    mid_fp32 = F.silu(gate) * up
    mid_cast = mid_fp32.to(x.dtype)

    # The third quantization is w2's, and it is the post-cast value: an fp32 input would have left
    # `seen[-1].dtype` at fp32, which is the bug this pins.
    assert seen[-1].dtype == x.dtype, f"w2's activation reached the quantizer at {seen[-1].dtype}"
    assert torch.equal(seen[-1], mid_cast), "w2 did not quantize the post-cast activation"


def test_only_the_reference_fp8_sites_are_quantized(monkeypatch) -> None:
    """Make the "not changed" list executable: which weights the flag lands on, and which it does not.

    A layer 2 forward runs the compressor and the indexer beside the attention projections, so the
    set of weights the quantizer is handed in one prefill is the whole seam at once. It has to be
    exactly `wq_a`, `wq_b`, `wkv`, `wo_b` and `Indexer.wq_b`, and none of `wo_a` (bf16), the indexer's
    `weights_proj`/`wk` (bf16), or the compressor's `wkv`/`wgate` -- all of which the reference leaves
    on the plain path.
    """
    quantized_weights: list[torch.Tensor] = []
    real = kernels.quantized_linear

    def spy(x, weight, *, quantized=False, block_size=32):
        if quantized:
            quantized_weights.append(weight)
        return real(x, weight, quantized=quantized, block_size=block_size)

    monkeypatch.setattr(modules_module, "quantized_linear", spy)
    monkeypatch.setattr(attention_module, "quantized_linear", spy)

    attn = _fill(Attention(2, _cfg(dtype="fp8")), seed=0)
    assert attn._quantized is True
    assert attn.compressor is not None and attn.indexer is not None, "layer 2 has to run both side modules"
    x = torch.randn(1, 4, DIM, dtype=attention_module.LINEAR_DTYPE)
    attn(x, 0, SharedAttentionRuntime())

    used = {id(weight) for weight in quantized_weights}
    expected = {
        id(attn.wq_a.weight),
        id(attn.wq_b.weight),
        id(attn.wkv.weight),
        id(attn.wo_b.weight),
        id(attn.indexer.wq_b.weight),
    }
    excluded = {
        id(attn.wo_a.weight),  # declared bf16 in the reference
        id(attn.indexer.weights_proj.weight),  # bf16
        id(attn.indexer.wk.weight),  # bf16
        id(attn.compressor.wkv.weight),  # not part of this seam
        id(attn.compressor.wgate.weight),
    }
    assert used == expected, f"quantized sites were {len(used)}, expected {len(expected)}"
    assert used.isdisjoint(excluded), "a bf16-declared weight was quantized"


def test_expert_forward_keeps_its_positional_signature() -> None:
    """The new keyword arguments sit after `weights` and default off, which is what the callers rely on.

    `tests/test_models_deepseek_v4_1_modules.py` calls `expert_forward` with six positional arguments
    and the routed stores pass a seventh positionally for the routing weights; both have to keep
    meaning what they meant before the flag existed.
    """
    torch.manual_seed(5)
    x = torch.randn(2, DIM, dtype=modules_module.LINEAR_DTYPE)
    w1 = torch.randn(INTER_DIM, DIM, dtype=modules_module.LINEAR_DTYPE)
    w2 = torch.randn(DIM, INTER_DIM, dtype=modules_module.LINEAR_DTYPE)
    w3 = torch.randn(INTER_DIM, DIM, dtype=modules_module.LINEAR_DTYPE)
    weights = torch.ones(2, 1, dtype=torch.float32)

    default = expert_forward(x, w1, w2, w3, 10.0, weights)
    assert torch.equal(default, expert_forward(x, w1, w2, w3, 10.0, weights, quantized=False))
    assert torch.equal(default, expert_forward(x, w1, w2, w3, 10.0, weights, False))