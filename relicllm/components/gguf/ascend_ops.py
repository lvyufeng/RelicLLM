"""The Ascend answer to :func:`relicllm.runtime.ops.load_ops`.

`relicllm/runtime/ops.py` is the door a second hardware backend arrives through, and until now
nothing had walked through it: ``_PROVIDERS`` was empty, ``load_ops()`` returned ``None`` on an
Ascend host, and every call site took the branch it was written for -- probe, find nothing, fall
back or refuse. This module is the provider, and it is plain torch on whatever device the tensors
are already on, so it is the *correct* answer on Ascend and not a fast one: the CUDA kernels keep
the quantized blocks in registers and never materialize a weight, while this decodes a block into
the compute dtype and calls ``torch.matmul``. A fused kernel belongs in relic-core; what belongs
here is a statement of the arithmetic the runtime has to reproduce.

## Why it sits beside the bindings rather than beside `load_ops`

Because it is IQ4_NL-shaped, and the device plane may name no container format. That is not a
preference: `tests/test_package_boundaries.py` refuses a `runtime/` module that imports
`relicllm.loader`, and `docs/architecture/package_layers.md` gives the reason -- `runtime/` answers
"which accelerator is this process on" and names no vendor, no model and no format, so that a second
hardware family changes those files and nothing else. Decoding this format needs the block layout
and the codebook out of `relicllm.loader.gguf`, which sits a layer above, so this module sits
above it too -- in the package that owns the GGUF raw-block bindings' call sites, the same address
`components/gguf/generation.py` gives for itself.

The cost is that `load_ops("ascend")` answers ``None`` until something registers this module, and
that is the seam's documented behaviour rather than a hole in it: `runtime/ops.py` says a platform
with no provider is the case the probing call sites were written for. The registration lives in
`components/gguf/quantized_ops.py`, happens on import and only on an Ascend host, and is therefore
in place before any module that reaches for one of these bindings can use it.

## The GGUF raw-block group, and the two formats it needs

`Xing4.0-29B-A4B`'s released GGUF is 18.7 GiB, 243 tensors of ``iq4_nl`` and one of ``q6_k``, and its
MoE is 64 *resident* experts -- the whole checkpoint fits on one card, which is why it is the natural
first model for a 32 GiB 910. Five bindings carry it end to end
(``gguf_quant_gemm_forward``, ``gguf_quant_gemm_prefill_forward``, ``gguf_quant_gemm_pair_forward``,
``gguf_quant_embedding_forward`` and ``gguf_moe_prefill_grouped_forward``); the sixth thing the
model reaches for, ``xing4_hyper_connection_forward``, is deliberately absent here because its call
site already treats a missing op as "use the eager arithmetic", which is the same sum.

The trunk is ``iq4_nl``: 40 blocks of bf16 attention, ``iq4_nl`` dense MLPs and ``iq4_nl`` routed
expert stacks. The one ``q6_k`` tensor is ``output.weight``, the vocabulary projection, and it is
why the second decoder is here at all -- the model reads it through the same binding as everything
else, so a provider that answered one format would load the checkpoint and fail at the last layer of
every forward. It is one statement on both sides: the CUDA kernel switches on the runtime type id,
and :data:`relicllm.loader.gguf.quant_types.GGUF_DENSE_TYPE_IDS` carries both ids.

What makes IQ4_NL decodable in a few lines of torch is the loader's own fold, not a coincidence.
``relicllm/loader/gguf/iq4_nl.fold_to_runtime_span`` regroups eight native 18-byte blocks into one
144-byte row element so every raw-block kernel sees the uniform 256-weight geometry the other ten
formats have. The bytes are unchanged and in order, so decoding a runtime row element is a reshape
back to ``(..., 8, 18)`` and then the reference decoder's own arithmetic -- the same one
``relicllm/loader/gguf/iq4_nl.dequantize_blocks`` performs, checked against it by
`tests/test_ascend_ops_gguf_raw.py`. Q6_K is already 256-weight, so its blocks arrive unfolded as
``(..., 210)`` and the reference is ``tensor_reader._read_q6_k_rows``.

## What this does not do

* **No other GGUF format.** Every other type id raises :class:`NotImplementedError` by name rather
  than decoding as one of these two, which would read a block at the wrong stride and produce a
  number rather than an error.
* **No ``xing4_hyper_connection_forward``.** See above; ``missing_bindings`` reports it, and it is
  the one absence that is a decision.
* **No claim about speed.** The decode is chunked only so a 152k-row vocabulary does not need a
  temporary per element; it is still a dequantize-then-multiply, and the honest next step is a
  kernel in relic-core.

## Where the numbers come from

The IQ4_NL codebook is read out of the vendored llama.cpp header through ``iq4_nl.kvalues()`` rather
than written here, so there is one statement of it in the tree. IQ4_NL carries one fp16 scale per 32
weights; Q6_K carries one fp16 super-block scale per 256 and sixteen int8 sub-scales under it, which
is why it decodes in fp32 and narrows once at the end -- its ``sub-scale * code`` products reach
2**12 and are not exact in fp16 the way a codebook entry is.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

import torch

from relicllm.loader.gguf import iq4_nl
from relicllm.loader.gguf.quant_types import GGUF_DENSE_TYPE_IDS
from relicllm.runtime.ops import BINDINGS, register_provider

__all__ = [
    "answered_bindings",
    "gguf_moe_prefill_grouped_forward",
    "gguf_quant_embedding_forward",
    "gguf_quant_gemm_forward",
    "gguf_quant_gemm_pair_forward",
    "gguf_quant_gemm_prefill_forward",
    "ops",
    "register",
]


#: The two GGUF runtime type ids this provider decodes. Named rather than inlined because the
#: arithmetic below is each format's own block layout and not a generic reader. The ids come from the
#: loader's dispatch table, which is the same one the CUDA kernel switches on, so the two sides
#: cannot drift about which id means which format.
_IQ4_NL_TYPE_ID = int(iq4_nl.IQ4_NL_FILE_TYPE_ID)
_Q6_K_TYPE_ID = int(GGUF_DENSE_TYPE_IDS["q6_k"])
_DECODED_TYPE_IDS = (_IQ4_NL_TYPE_ID, _Q6_K_TYPE_ID)

#: A native IQ4_NL block: 32 weights in 18 bytes, one fp16 scale then 16 packed nibbles.
_NATIVE_BLOCK_ELEMS = int(iq4_nl.QK_IQ4_NL)
_NATIVE_BLOCK_BYTES = int(iq4_nl.IQ4_NL_BLOCK_BYTES)

#: The row element the loader folds eight native blocks into, and what the kernels index rows by.
_RUNTIME_SPAN_ELEMS = 8 * _NATIVE_BLOCK_ELEMS
_RUNTIME_SPAN_BYTES = 8 * _NATIVE_BLOCK_BYTES

#: A native Q6_K super-block: 256 weights in 210 bytes -- 128 of four-bit low halves, 64 of the two
#: high bits, 16 int8 sub-scales and one fp16 super-block scale. The file's own type id is 14
#: (``reader.GGML_TYPES``) and the runtime id above is 8, which is why the name and not the number is
#: what the two tables agree on; `tests/test_ascend_ops_gguf_raw.py` pins the geometry against both.
_Q6_K_BLOCK_ELEMS = 256
_Q6_K_BLOCK_BYTES = 210

#: Native blocks decoded per chunk. Bound rather than picked for speed: the nibble indices are
#: ``int64`` for torch's benefit, so a whole-vocabulary decode in one go would allocate two
#: eight-byte temporaries per weight on the way to a two-byte one. 1<<20 blocks is 33.5M weights a
#: chunk and about 400 MB of transient, which is small enough to be invisible next to the weights.
_DECODE_CHUNK_BLOCKS = 1 << 20


@lru_cache(maxsize=1)
def _codebook() -> torch.Tensor:
    """The 16 signed codebook entries, as a host ``int8`` tensor.

    Read from the vendored header by the loader's own reader, so a change to llama.cpp's table
    reaches both this decoder and the numpy reference at once.
    """
    return torch.from_numpy(iq4_nl.kvalues().copy())


def _compute_dtype(x: torch.Tensor) -> torch.dtype:
    """The width to dequantize into, which is the activation's own when it is one this can hold.

    The kernels dispatch on fp16/bf16/fp32 and accumulate in float; mirroring the input's width
    keeps a fp16 GEMM on the cube unit instead of falling into a vector fp32 path, and keeps a fp32
    one exact.
    """
    if x.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return x.dtype
    return torch.float32


def _require_supported(type_id: int, where: str) -> None:
    """Refuse a type id this provider has no decoder for, by name and with the two it does.

    The failure this prevents is the quiet one: the formats here are 32-, 144- and 210-byte blocks
    and reading one as another produces a number rather than an error, and a wrong number in a MoE
    layer or a vocabulary projection is a wrong model.
    """
    if int(type_id) not in _DECODED_TYPE_IDS:
        raise NotImplementedError(
            f"the Ascend provider decodes iq4_nl (type {_IQ4_NL_TYPE_ID}) and q6_k "
            f"(type {_Q6_K_TYPE_ID}) only; {where} asked for type {int(type_id)}. Decoding it as "
            f"either would read its blocks at the wrong stride, so this refuses instead."
        )


def _decode_weights(
    blocks: torch.Tensor, row_elems: int, type_id: int, out_dtype: torch.dtype, where: str
) -> torch.Tensor:
    """The dense ``[..., row_elems]`` weight the raw blocks name, whichever format they are.

    The dispatch is here rather than at each call site so the bindings stay a statement of the
    arithmetic around the dot, and so that ``where`` -- which binding asked -- survives into the
    refusal.
    """
    _require_supported(type_id, where)
    if int(type_id) == _IQ4_NL_TYPE_ID:
        return _iq4_nl_weights(blocks, int(row_elems), out_dtype)
    return _q6_k_weights(blocks, int(row_elems), out_dtype)


def _iq4_nl_weights(blocks: torch.Tensor, row_elems: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Decode raw IQ4_NL row elements into a dense ``[..., row_elems]`` weight.

    ``blocks`` is ``(..., spans, 144)`` of uint8 -- the loader's folded geometry, which is eight
    native 18-byte blocks end to end. The nibbles are not interleaved: ``qs[j] & 0x0f`` is weight
    ``j`` and ``qs[j] >> 4`` is weight ``j + 16``, so a block decodes to the low nibbles followed by
    the high ones. Every product is exact in the compute dtype as far as the scale allows: a
    codebook entry is a small integer and the software only ever multiplies it by one fp16.
    """
    row_elems = int(row_elems)
    spans = int(blocks.shape[-2])
    if int(blocks.shape[-1]) != _RUNTIME_SPAN_BYTES:
        raise ValueError(
            f"IQ4_NL row elements are {_RUNTIME_SPAN_BYTES} bytes, got {int(blocks.shape[-1])}. The "
            f"loader folds native {_NATIVE_BLOCK_BYTES}-byte blocks into this shape; bytes that did "
            f"not come through fold_to_runtime_span are not what this decodes."
        )
    if spans * _RUNTIME_SPAN_ELEMS < row_elems:
        raise ValueError(
            f"IQ4_NL blocks cover {spans * _RUNTIME_SPAN_ELEMS} weights, which is less than the "
            f"{row_elems} a row holds"
        )

    native = blocks.reshape(-1, _NATIVE_BLOCK_BYTES)
    table = _codebook().to(device=blocks.device, dtype=out_dtype)
    chunks: list[torch.Tensor] = []
    for start in range(0, int(native.shape[0]), _DECODE_CHUNK_BLOCKS):
        chunk = native[start : start + _DECODE_CHUNK_BLOCKS]
        # The scale is bytes 0..1 of the block and is the one field that is not a nibble.
        scale = chunk[:, :2].contiguous().view(torch.float16).reshape(-1).to(out_dtype)
        qs = chunk[:, 2:]
        low = table[(qs & 0x0F).to(torch.int64)]
        high = table[(qs >> 4).to(torch.int64)]
        chunks.append(torch.cat((low, high), dim=-1) * scale.unsqueeze(-1))

    weights = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    weights = weights.reshape(*blocks.shape[:-2], spans * _RUNTIME_SPAN_ELEMS)
    if int(weights.shape[-1]) != row_elems:
        # Only reachable for a row that is not a whole number of spans; every width in Xing4.0
        # (3584 and 1024) is, and fold_to_runtime_span refuses the others before they get here.
        weights = weights[..., :row_elems].contiguous()
    return weights


@lru_cache(maxsize=1)
def _q6_k_lane_scale_index() -> torch.Tensor:
    """``is = l // 16`` for ``l`` in 0..31: which of the eight sub-scales each lane position reads.

    ggml's ``dequantize_row_q6_K`` walks a 128-weight half in 32 lanes and takes its four outputs'
    scales from ``sc[is + 0]``, ``sc[is + 2]``, ``sc[is + 4]`` and ``sc[is + 6]``, so the lane axis
    selects scale slots 0..7 in that pattern. Written as one index array rather than two ``cat``s
    because the four sub-lanes are then one indexed gather each.
    """
    return torch.cat((torch.zeros(16, dtype=torch.int64), torch.ones(16, dtype=torch.int64)))


def _q6_k_weights(blocks: torch.Tensor, row_elems: int, out_dtype: torch.dtype) -> torch.Tensor:
    """Decode raw Q6_K super-blocks into a dense ``[..., row_elems]`` weight.

    ``blocks`` is ``(..., spans, 210)`` of uint8 in the loader's geometry -- Q6_K is already
    256-weight, so unlike IQ4_NL these arrive unfolded and there is no fold to undo. The six bits a
    weight live in two places: its low four in ``ql`` and its top two in ``qh``, whose two-bit fields
    are packed four to a byte and walked in the order the four sub-lanes consume them. The value is
    stored biased by 32.

    The whole decode is fp32 and narrows once at the end, unlike the IQ4_NL decoder above, because
    ``sub-scale * code`` reaches ``127 * 31`` and a small integer past 2048 is not an fp16. Computing
    the product in the compute dtype would round there and in the super-block scale as well; this
    rounds once, which is what the loader's reference path does too.
    """
    row_elems = int(row_elems)
    spans = int(blocks.shape[-2])
    if int(blocks.shape[-1]) != _Q6_K_BLOCK_BYTES:
        raise ValueError(
            f"Q6_K super-blocks are {_Q6_K_BLOCK_BYTES} bytes, got {int(blocks.shape[-1])}. "
            f"`read_quantized_matrix_blocks` hands them over at the format's own geometry; bytes that "
            f"did not come through it are not what this decodes."
        )
    if spans * _Q6_K_BLOCK_ELEMS < row_elems:
        raise ValueError(
            f"Q6_K blocks cover {spans * _Q6_K_BLOCK_ELEMS} weights, which is less than the "
            f"{row_elems} a row holds"
        )

    native = blocks.reshape(-1, _Q6_K_BLOCK_BYTES)
    lane_scales = _q6_k_lane_scale_index().to(blocks.device)
    chunks: list[torch.Tensor] = []
    for start in range(0, int(native.shape[0]), _DECODE_CHUNK_BLOCKS):
        chunk = native[start : start + _DECODE_CHUNK_BLOCKS]
        ql = chunk[:, 0:128]
        qh = chunk[:, 128:192]
        # The sub-scales are int8 and the two super-block bytes are one fp16, so both are widened
        # rather than read as unsigned 0..255.
        sub_scales = chunk[:, 192:208].contiguous().view(torch.int8).to(torch.float32)
        super_scale = chunk[:, 208:210].contiguous().view(torch.float16).reshape(-1).to(torch.float32)
        out = torch.empty(chunk.shape[0], _Q6_K_BLOCK_ELEMS, dtype=torch.float32, device=blocks.device)
        for half in range(2):
            base = half * 128
            ql_half = ql[:, half * 64 : (half + 1) * 64]
            qh_half = qh[:, half * 32 : (half + 1) * 32]
            sc_half = sub_scales[:, half * 8 : (half + 1) * 8]
            low_first = ql_half[:, 0:32]
            low_second = ql_half[:, 32:64]
            high = qh_half[:, 0:32]
            q1 = ((low_first & 0x0F) | ((high & 0x03) << 4)).to(torch.int16) - 32
            q2 = ((low_second & 0x0F) | (((high >> 2) & 0x03) << 4)).to(torch.int16) - 32
            q3 = ((low_first >> 4) | (((high >> 4) & 0x03) << 4)).to(torch.int16) - 32
            q4 = ((low_second >> 4) | (((high >> 6) & 0x03) << 4)).to(torch.int16) - 32
            scale = super_scale.unsqueeze(-1)
            out[:, base + 0 : base + 32] = scale * sc_half[:, lane_scales + 0] * q1.to(torch.float32)
            out[:, base + 32 : base + 64] = scale * sc_half[:, lane_scales + 2] * q2.to(torch.float32)
            out[:, base + 64 : base + 96] = scale * sc_half[:, lane_scales + 4] * q3.to(torch.float32)
            out[:, base + 96 : base + 128] = scale * sc_half[:, lane_scales + 6] * q4.to(torch.float32)
        chunks.append(out)

    weights = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    weights = weights.reshape(*blocks.shape[:-2], spans * _Q6_K_BLOCK_ELEMS)
    if int(weights.shape[-1]) != row_elems:
        # Only reachable for a row that is not a whole number of super-blocks; every width in
        # Xing4.0 (3584 and 1024) is.
        weights = weights[..., :row_elems].contiguous()
    return weights.to(out_dtype)


def _dense_gemm(
    x: torch.Tensor, blocks: torch.Tensor, row_elems: int, type_id: int, where: str
) -> torch.Tensor:
    """``x @ dequant(blocks).T``, in the accumulator the CUDA kernel returns and not its round trip.

    The CUDA implementations write their result as bf16 and the call sites then narrow it again, so
    returning the fp32 accumulator here is *more* accurate than the reference path rather than a
    departure from it. Nothing compares the two on one host -- there is no host with both -- and a
    second rounding is a loss on every caller.
    """
    dtype = _compute_dtype(x)
    weight = _decode_weights(blocks, int(row_elems), type_id, dtype, where)
    flat = x.reshape(-1, int(row_elems)).to(dtype)
    out = flat @ weight.t()
    return out.reshape(*x.shape[:-1], int(weight.shape[0])).float()


def gguf_quant_gemm_forward(
    x: torch.Tensor,
    blocks: torch.Tensor,
    row_elems: int,
    type_id: int,
    signed_grid: torch.Tensor,
) -> torch.Tensor:
    """``[M, K]`` (or ``[B, S, K]``) by a raw-block weight, for the single-token and generic calls.

    ``signed_grid`` is the IQ2/IQ1 codebook and is unused for either format here: IQ4_NL's values are
    a 16-entry table the kernel holds and Q6_K has no codebook at all, which is why the call sites
    pass an empty int8 tensor. It is in the signature because the kernels' is.
    """
    del signed_grid
    return _dense_gemm(x, blocks, int(row_elems), int(type_id), "gguf_quant_gemm")


def gguf_quant_gemm_prefill_forward(
    x: torch.Tensor,
    blocks: torch.Tensor,
    row_elems: int,
    type_id: int,
    signed_grid: torch.Tensor,
) -> torch.Tensor:
    """The prefill arm of :func:`gguf_quant_gemm_forward`, and the same arithmetic.

    The two are separate bindings because the CUDA side picks a different tiling for rows > 1. There
    is nothing to pick here, and answering both with one implementation is the statement that they
    compute the same thing rather than an omission.
    """
    del signed_grid
    return _dense_gemm(x, blocks, int(row_elems), int(type_id), "gguf_quant_gemm_prefill")


def gguf_quant_gemm_pair_forward(
    x: torch.Tensor,
    blocks0: torch.Tensor,
    row_elems0: int,
    type_id0: int,
    blocks1: torch.Tensor,
    row_elems1: int,
    type_id1: int,
    signed_grid: torch.Tensor,
) -> torch.Tensor:
    """Two raw GEMMs over one activation, stacked on a leading axis.

    The CUDA implementation returns ``torch.stack({out0, out1}, 0)`` and the call site unpacks it
    with ``y0, y1 = ...``, so the leading axis is the contract rather than a convenience.
    """
    del signed_grid
    first = _dense_gemm(x, blocks0, int(row_elems0), int(type_id0), "gguf_quant_gemm_pair(first)")
    second = _dense_gemm(x, blocks1, int(row_elems1), int(type_id1), "gguf_quant_gemm_pair(second)")
    return torch.stack((first, second), dim=0)


def gguf_quant_embedding_forward(
    token_ids: torch.Tensor,
    blocks: torch.Tensor,
    row_elems: int,
    type_id: int,
    signed_grid: torch.Tensor,
) -> torch.Tensor:
    """Selected rows of a raw IQ4_NL matrix, for an embedding that is quantized.

    Xing4.0 does not reach this one -- its ``token_embd.weight`` is dense and read through
    ``read_dense`` -- but it completes the GGUF group, and it answers both formats this provider
    decodes rather than only the one the model happens to store.
    """
    del signed_grid
    ids = token_ids.reshape(-1).to(torch.int64)
    weight = _decode_weights(blocks, int(row_elems), int(type_id), torch.float32, "gguf_quant_embedding")
    out = weight.index_select(0, ids)
    return out.reshape(*token_ids.shape, int(row_elems)).float()


def _swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """``silu(gate) * up`` with the kernel's optional clamp, applied where the kernel applies it.

    The clamp is off in this checkpoint (``swiglu_limit`` is 0.0) and it is written anyway because
    ``gguf_route_swiglu_quantize_hidden_16_kernel`` clamps *before* the activation -- the gate's
    positive tail only, the up half on both sides -- and a port that clamped the product instead
    would differ from the kernel it is standing in for.
    """
    if limit > 0.0:
        gate = torch.clamp(gate, max=limit)
        up = torch.clamp(up, -limit, limit)
    return torch.nn.functional.silu(gate) * up


def gguf_moe_prefill_grouped_forward(
    x: torch.Tensor,
    route_tokens: torch.Tensor,
    route_weights_sorted: torch.Tensor,
    seg_starts: torch.Tensor,
    w1_blocks: torch.Tensor,
    w3_blocks: torch.Tensor,
    w2_blocks: torch.Tensor,
    w1_row_elems: int,
    w1_type_id: int,
    w3_row_elems: int,
    w3_type_id: int,
    w2_row_elems: int,
    w2_type_id: int,
    signed_grid: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    """Every route of a routed MoE in one call: gate, up, activation, down, weighted and summed.

    The kernel's shape is the point. It counts *routes*, not tokens -- ``seg_starts`` names the
    expert of each segment of the route axis and ``route_tokens[r]`` names the activation row route
    ``r`` reads -- so this walks the same segments in the same order and adds each expert's down
    projection into the token it came from. The result is ``[tokens, dim]`` fp32, which is what the
    kernel returns and what the caller's own scatter then reduces over top-k.

    ``route_weights_sorted`` is applied to the activation, before the down projection, because that
    is where ``route_swiglu_cast_kernel`` applies it. Scaling after the projection is the same sum
    in exact arithmetic and a different one in fp16, and this is meant to be the kernel's sum.
    """
    del signed_grid
    # Refused up front rather than where the loop reaches each one: an empty route set returns
    # before the loop, and a format this cannot decode is a property of the weights, not of how many
    # routes happened to arrive.
    _require_supported(w1_type_id, "gguf_moe_prefill_grouped w1")
    _require_supported(w3_type_id, "gguf_moe_prefill_grouped w3")
    _require_supported(w2_type_id, "gguf_moe_prefill_grouped w2")

    tokens, dim = int(x.shape[0]), int(x.shape[1])
    n_experts = int(w1_blocks.shape[0])
    out = torch.zeros(tokens, dim, dtype=torch.float32, device=x.device)
    if tokens == 0 or n_experts == 0:
        return out

    seg = seg_starts.reshape(-1).to(torch.int64)
    route_rows = route_tokens.reshape(-1).to(torch.int64)
    route_weights = route_weights_sorted.reshape(-1).to(torch.float32)
    # One device-to-host read for the whole table rather than two per expert: the counts are what
    # the loop is over, and asking for them one at a time is 64 stream drains a layer that a
    # captured graph could not replay through.
    counts = (seg[1 : n_experts + 1] - seg[:n_experts]).tolist()
    dtype = _compute_dtype(x)
    limit = float(swiglu_limit)

    start = 0
    for expert, count in enumerate(counts):
        end = start + int(count)
        if end > start:
            rows = route_rows[start:end]
            gate = _decode_weights(w1_blocks[expert], int(w1_row_elems), int(w1_type_id), dtype, "gguf_moe_prefill_grouped w1")
            up = _decode_weights(w3_blocks[expert], int(w3_row_elems), int(w3_type_id), dtype, "gguf_moe_prefill_grouped w3")
            down = _decode_weights(w2_blocks[expert], int(w2_row_elems), int(w2_type_id), dtype, "gguf_moe_prefill_grouped w2")
            activations = x.index_select(0, rows).to(dtype)
            hidden = _swiglu(
                activations @ gate.t(), activations @ up.t(), limit
            ) * route_weights[start:end].unsqueeze(-1).to(dtype)
            out.index_add_(
                0, rows, (hidden @ down.t()).to(torch.float32)
            )
        start = end
    return out


_BINDING_FUNCTIONS = (
    "gguf_moe_prefill_grouped_forward",
    "gguf_quant_embedding_forward",
    "gguf_quant_gemm_forward",
    "gguf_quant_gemm_pair_forward",
    "gguf_quant_gemm_prefill_forward",
)


@lru_cache(maxsize=1)
def ops() -> Any:
    """The operator module :func:`load_ops` hands back for ``ascend``.

    A namespace of the names above rather than the module object itself, so the surface is the list
    and not "whatever a future edit adds to this file" -- the same reason ``BINDINGS`` is measured
    on the CUDA side.
    """
    module = sys.modules[__name__]
    return SimpleNamespace(**{name: getattr(module, name) for name in _BINDING_FUNCTIONS})


def answered_bindings() -> tuple[str, ...]:
    """Which of :data:`relicllm.runtime.ops.BINDINGS` this provider answers, in declaration order.

    The complement of ``missing_bindings``, and the useful direction to print: on Ascend the
    interesting list is the five that work, not the forty-one that belong to other families.
    """
    provider = ops()
    return tuple(name for name in BINDINGS if hasattr(provider, name))


def register() -> None:
    """Teach :func:`load_ops` this is the ``ascend`` answer. Idempotent.

    Called on import by `components/gguf/quantized_ops.py` when the host is an Ascend one, and
    called by hand by the tests, which have to be able to ask for the provider without a 910 under
    them.
    """
    register_provider("ascend", ops)
