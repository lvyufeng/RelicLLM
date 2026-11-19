"""Qwen4-Exp building blocks: norms, RoPE, hyper-connections, PLE, MoE.

Numerics follow the upstream `transformers` implementation exactly (including
where it forces float32), because the tests check bit-level agreement against
goldens captured from it.  Nothing here is TP-aware; sharding happens in
`runtime.py` by handing sliced weights to these modules.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn.functional as F

from relicllm.models.qwen4_exp.config import Qwen4ExpTextConfig
from relicllm.runtime.ops import load_ops


# ---------------------------------------------------------------------------
# Norms
# ---------------------------------------------------------------------------


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes"}


def _dense_fp16_enabled(rows: int) -> bool:
    min_rows = max(
        1,
        int(os.environ.get("POCKETLLM_QWEN4_DENSE_FP16_MIN_ROWS", "128")),
    )
    return _env_enabled("POCKETLLM_QWEN4_DENSE_FP16", "0") and rows >= min_rows


def _dense_fp16_fp32_output_enabled(rows: int) -> bool:
    min_rows = max(
        1,
        int(os.environ.get("POCKETLLM_QWEN4_DENSE_FP16_MIN_ROWS", "128")),
    )
    return (
        _env_enabled("POCKETLLM_QWEN4_DENSE_FP16_FP32_OUTPUT", "0")
        and rows >= min_rows
    )


def dequant_fp8_block(
    codes: torch.Tensor,
    scale: torch.Tensor,
    block: tuple[int, int] = (128, 128),
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Reference dequantization for a block-scaled FP8 matrix."""
    from relicllm.models.qwen4_exp.quant import dequantize_block_fp8

    return dequantize_block_fp8(codes, scale, block, out_dtype)


def _fp16_fp32_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """FP16 tensor-core GEMM with an FP32 accumulator and input dtype output."""
    flat = x.reshape(-1, x.shape[-1]).to(torch.float16)
    output = torch.mm(
        flat,
        weight.to(torch.float16).t(),
        out_dtype=torch.float32,
    )
    return output.to(x.dtype).reshape(
        *x.shape[:-1],
        weight.shape[0],
    )


def row_invariant_verify_enabled(rows: int) -> bool:
    """Opt into decode-shaped arithmetic for small BF16 calls, including short prefill chunks.

    This is an arithmetic contract, not a precision change: ordinary one-row BF16 operations
    are recorded independently for 2..8 rows. The environment switch is explicit and off by
    default. Larger prefills and ordinary one-row decode keep their existing dispatch.
    """
    return 2 <= rows <= 8 and os.environ.get("POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT", "0") == "1"


def row_invariant_linear(x: torch.Tensor, weight: torch.Tensor, bias=None) -> torch.Tensor:
    """Run small BF16 batches through the same one-row `F.linear` shape as greedy decode.

    cuBLAS may choose different arithmetic at M=1 and M=K. Keeping each recorded call at M=1
    preserves the decode operation without a global torch monkeypatch or FP16 weight mirror.
    Leading dimensions are restored exactly; no values are read back to the host.
    """
    rows = x.numel() // x.shape[-1]
    if (not row_invariant_verify_enabled(rows) or x.dtype is not torch.bfloat16
            or weight.dtype is not torch.bfloat16 or x.device != weight.device):
        return F.linear(x, weight, bias)
    if os.environ.get("POCKETLLM_QWEN4_LINEAR_OUT", "0") == "1":
        from relicllm.models.qwen4_exp.linear_out import linear_out_enabled, qwen4_exp_linear_out
        if linear_out_enabled():
            output = qwen4_exp_linear_out(x, weight, bias)
            if output is not None:
                return output
    flat = x.reshape(rows, x.shape[-1])
    single_shape = (1,) * (x.ndim - 1) + (x.shape[-1],)
    output = torch.cat([
        F.linear(flat[row:row + 1].reshape(single_shape).contiguous(), weight, bias).reshape(1, -1)
        for row in range(rows)
    ], dim=0)
    return output.reshape(*x.shape[:-1], weight.shape[0])


def row_invariant_sequence_sum(x: torch.Tensor) -> torch.Tensor:
    """Keep PLE's `[batch, seq, hc, hidden]` reduction at the one-query decode geometry.

    The reduction tree may depend on the number of output groups, even with BF16 operands and
    identical values. Split the sequence axis, not the hc axis: decode reduces four hc groups in
    one operation. No host values or altered accumulation dtype are introduced.
    """
    if (x.ndim != 4 or x.dtype is not torch.bfloat16
            or not row_invariant_verify_enabled(x.shape[1])):
        return x.sum(dim=-1, keepdim=True)
    return torch.cat([
        x[:, row:row + 1].contiguous().sum(dim=-1, keepdim=True)
        for row in range(x.shape[1])
    ], dim=1)


def row_invariant_conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
    """Keep small depthwise causal convolutions on their one-output decode shape.

    The no-padding, stride-one window includes dilation gaps (PLE uses dilation=ngram_size).
    Every slice is made contiguous, reproducing the per-channel window layout of a decode
    call instead of handing cuDNN a different K-output width/stride. Other cases are untouched.
    """
    step = stride[0] if isinstance(stride, (tuple, list)) else stride
    pad = padding[0] if isinstance(padding, (tuple, list)) else padding
    dil = dilation[0] if isinstance(dilation, (tuple, list)) else dilation
    span = dil * (weight.shape[-1] - 1) + 1 if isinstance(dil, int) and dil > 0 else 0
    outputs = x.shape[-1] - span + 1 if span else 0
    if (x.ndim != 3 or weight.ndim != 3 or step != 1 or pad != 0 or not span
            or groups != x.shape[1] or weight.shape[0] != groups or weight.shape[1] != 1
            or x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16
            or x.device != weight.device or not row_invariant_verify_enabled(outputs)):
        return F.conv1d(x, weight, bias, stride, padding, dilation, groups)
    return torch.cat([
        F.conv1d(x[..., row:row + span].contiguous(), weight, bias, 1, 0, dilation, groups)
        for row in range(outputs)
    ], dim=-1)


def prefill_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Use tensor-core GEMM for BF16 projections, including explicit FP16 mode.

    The environment gates are an optimization for BF16 checkpoints.  When the
    whole model was deliberately loaded as FP16, matching FP16 inputs/weights
    are already the native SM75 fast path and must not be cast back to BF16.
    """
    rows = x.numel() // x.shape[-1]
    same_device = weight.device == x.device
    if (row_invariant_verify_enabled(rows) and same_device
            and x.dtype is torch.bfloat16 and weight.dtype is torch.bfloat16):
        return row_invariant_linear(x, weight)
    native_fp16 = (
        x.device.type == "cuda"
        and x.dtype is torch.float16
        and weight.dtype is torch.float16
        and same_device
    )
    if native_fp16:
        if _dense_fp16_fp32_output_enabled(rows):
            return _fp16_fp32_linear(x, weight)
        return F.linear(x, weight)

    supported_bf16 = (
        x.device.type == "cuda"
        and x.dtype is torch.bfloat16
        and weight.dtype is torch.bfloat16
        and same_device
    )
    if supported_bf16 and _dense_fp16_fp32_output_enabled(rows):
        return _fp16_fp32_linear(x, weight)
    if supported_bf16 and _dense_fp16_enabled(rows):
        return F.linear(x.to(torch.float16), weight.to(torch.float16)).to(torch.bfloat16)
    return F.linear(x, weight)


def _hc_fp16_components(rows: int) -> tuple[bool, bool, bool]:
    min_rows = max(
        1,
        int(os.environ.get("POCKETLLM_QWEN4_HC_FP16_MIN_ROWS", "128")),
    )
    if rows < min_rows:
        return False, False, False
    all_enabled = _env_enabled("POCKETLLM_QWEN4_HC_FP16", "0")
    return (
        all_enabled or _env_enabled("POCKETLLM_QWEN4_HC_FP16_MIX_DOWN", "0"),
        all_enabled or _env_enabled("POCKETLLM_QWEN4_HC_FP16_MIX_UP", "0"),
        all_enabled or _env_enabled("POCKETLLM_QWEN4_HC_FP16_INJECT", "0"),
    )


def _hc_fp16_enabled(rows: int) -> bool:
    return any(_hc_fp16_components(rows))


def _hc_fp16_fp32_output_enabled(rows: int) -> bool:
    return (
        _hc_fp16_enabled(rows)
        and _env_enabled("POCKETLLM_QWEN4_HC_FP16_FP32_OUTPUT", "0")
    )


def _hc_cuda_enabled(rows: int) -> bool:
    """Whether the fused hyper-connection kernels may run, at this row count.

    On by default, and the row floor is 1.  The floor used to be 128 with the switch off, which
    meant a one-row decode step never reached the fused kernels at all -- the same shape of miss as
    the sparse-attention dispatch and the completed-prefix pool.

    Priced with three interleaved rounds at 8192 positions, both arms inside one script:

    | arm | tok/s (three rounds) | median | ITL p50 |
    |---|---|---|---|
    | elementwise | 37.92, 37.73, 37.65 | 37.73 | 25.89 ms |
    | fused | 38.31, 38.27, 38.44 | 38.31 | 25.47 ms |

    Identical tokens in all six runs, a within-arm spread of 0.27 tok/s and a between-arm gap of
    0.58.  A single run had earlier put this at +0.5% and dismissed it; the effect is real at 1.5%,
    which is the same lesson the fused norm taught -- one throughput run resolves about 3-4%.
    """
    min_rows = max(
        1,
        int(os.environ.get("POCKETLLM_QWEN4_HC_CUDA_MIN_ROWS", "1")),
    )
    return (
        _env_enabled("POCKETLLM_QWEN4_HC_CUDA", "1")
        and rows >= min_rows
    ) or _hc_fp16_enabled(rows)


def _cuda_hc_activation(
    name: str,
    x: torch.Tensor,
    groups: int,
) -> torch.Tensor | None:
    """Fuse an HC scale-then-activation pair without changing its rounding.

    The reference path divides a BF16 GEMM result by `hc_count` and then applies
    the activation in BF16.  Both kernels keep the divide and the activation in
    float32 and round once on store, which is bit-identical to the reference on
    every value measured so far, so this stays on the exact path.
    """
    if (
        not _hc_cuda_enabled(x.numel() // x.shape[-1])
        or x.device.type != "cuda"
        or x.dtype is not torch.bfloat16
        or not x.is_contiguous()
    ):
        return None
    ext = load_ops()
    if ext is None or not hasattr(ext, name):
        return None
    return getattr(ext, name)(x, groups)


def _rms_norm_cuda_enabled(rows: int) -> bool:
    """Whether the fused norm kernel may run.

    Deliberately *not* `_hc_cuda_enabled`: that predicate exists to decide when the
    hyper-connection kernels are worth their launch at a given row count, and it defaults off with
    a 128-row floor, which is a different question from whether this kernel may run.  Split out so
    the two decisions cannot be conflated again.

    On by default.  An earlier single measurement of this said it was inside the noise and it was
    left off; three interleaved rounds say otherwise, and the difference is the measurement rather
    than the kernel.  On eight RTX 4090s at 8192 positions, alternating arms within one script so a
    change in the box cannot be mistaken for a change in the kernel:

    | arm | tok/s (three rounds) | median | ITL p50 |
    |---|---|---|---|
    | torch path | 37.85, 37.56, 37.65 | 37.65 | 25.88 ms |
    | fused norm | 39.00, 38.77, 38.77 | 38.77 | 25.13 ms |

    Every one of the six runs produced the same tokens.  The within-arm spread is 0.3 tok/s and the
    between-arm gap 1.1, so a 3% effect that a single run could not resolve is plain here.  The
    fused kernel replaces about eight dispatches a call with one launch, and while a recorded step
    has no dispatches to remove, those eight were not all views: the `pow`, `mean`, `rsqrt` and two
    multiplies are four to six real kernels, several times a layer, forty-eight layers.
    """
    return _env_enabled("POCKETLLM_QWEN4_RMS_NORM_CUDA", "1") and rows >= 1


def _cuda_grouped_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
) -> torch.Tensor | None:
    if (
        not _rms_norm_cuda_enabled(x.numel() // x.shape[-1])
        or x.device.type != "cuda"
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or x.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or weight.dtype is not x.dtype
        or weight.device != x.device
        or x.shape[-1] != weight.numel()
        or x.shape[-1] % group_size != 0
    ):
        return None
    ext = load_ops()
    if ext is None or not hasattr(ext, "qwen4_exp_grouped_rms_norm"):
        return None
    return ext.qwen4_exp_grouped_rms_norm(x, weight, group_size, eps)


class RMSNorm:
    """RMSNorm with upstream's `(1 + weight)` convention and optional grouping.

    `group_size` normalizes each contiguous group independently, which is how
    the hyper-connection norms treat their `hc_count * hidden_size` input: each
    residual stream gets its own statistics.
    """

    def __init__(self, weight: torch.Tensor, eps: float, *, group_size: int | None = None) -> None:
        self.weight = weight
        self.eps = eps
        self.group_size = group_size

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Packed Q/gate projections leave gaps between heads (or between query rows). A one-query
        # indexer slice is contiguous and uses the fused norm, while a wider slice can fall back
        # to torch with another reduction tree. Preserve each decode slice's *original* strides:
        # copying a gapped Q-head slice would incorrectly select a fused kernel decode never used.
        if (x.ndim == 4 and x.shape[0] == 1 and x.dtype is torch.bfloat16
                and not x.is_contiguous() and row_invariant_verify_enabled(x.shape[1])):
            return torch.cat([self(x[:, row:row + 1]) for row in range(x.shape[1])], dim=1)
        # An ungrouped norm is one group covering the whole last dimension, which is the same
        # arithmetic the kernel already does: `gain = 1 + weight` over a single group.
        group_size = self.group_size if self.group_size is not None else int(x.shape[-1])
        output = _cuda_grouped_rms_norm(
            x,
            self.weight,
            group_size,
            self.eps,
        )
        if output is not None:
            return output
        in_dtype = x.dtype
        h = x.to(torch.float32)
        if self.group_size is not None:
            h = h.reshape(*h.shape[:-1], -1, self.group_size)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.group_size is not None:
            h = h.flatten(-2)
        # Qwen stores the norm scale centered on zero: the effective gain is
        # (1 + weight), and the multiply happens in float32 before the cast back.
        return (h * self._gain()).to(in_dtype)

    def _gain(self) -> torch.Tensor:
        """`1 + weight` in float32, computed once.

        The weight does not change, and this norm runs several times a layer on a decode step
        that is already host-bound: the cast and the add were two more dispatches a call.
        """
        gain = self.__dict__.get("_gain_cache")
        if gain is None or gain.device != self.weight.device:
            gain = (1.0 + self.weight.float()).contiguous()
            self.__dict__["_gain_cache"] = gain
        return gain


def _cuda_gated_rms_norm_exact(x, gate, weight, eps: float, activation: str):
    """Fuse only proven rounding/multiply boundaries; torch retains statistics and activation."""
    if (os.environ.get("POCKETLLM_QWEN4_GATED_NORM_EXACT", "0") != "1"
            or x.device.type != "cuda" or x.dtype is not torch.bfloat16
            or weight.dtype is not torch.bfloat16 or weight.device != x.device
            or gate.device != x.device or gate.dtype not in (torch.bfloat16, torch.float32)
            or x.ndim < 2 or x.numel() == 0 or gate.shape != x.shape or weight.ndim != 1
            or x.shape[-1] != weight.numel() or x.shape[-1] > 8192
            or not (x.is_contiguous() and gate.is_contiguous() and weight.is_contiguous())
            or activation not in ("sigmoid", "silu")):
        return None
    ext = load_ops()
    fused = getattr(ext, "qwen4_exp_gated_rms_norm_exact", None) if ext is not None else None
    if fused is None:
        return None
    # Keep the reduction tree, rsqrt and CUDA activation exactly as the existing path. The
    # independent native entry preserves norm->BF16, BF16 weight multiplication and final BF16.
    h = x.to(torch.float32)
    inverse = torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    act = torch.sigmoid if activation == "sigmoid" else F.silu
    activated = act(gate.to(torch.float32))
    return fused(x, inverse, weight, activated)


class RMSNormGated:
    """Per-head RMSNorm followed by a gate activation (GatedDeltaNet output).

    Upstream multiplies by `weight` (not `1 + weight`) here and applies the gate
    activation in float32.
    """

    def __init__(self, weight: torch.Tensor, eps: float, *, activation: str = "silu") -> None:
        self.weight = weight
        self.eps = eps
        self.activation = activation

    def __call__(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        fused = _cuda_gated_rms_norm_exact(x, gate, self.weight, self.eps, self.activation)
        if fused is not None:
            return fused
        in_dtype = x.dtype
        h = x.to(torch.float32)
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = self.weight * h.to(in_dtype)
        act = torch.sigmoid if self.activation == "sigmoid" else F.silu
        h = h * act(gate.to(torch.float32))
        return h.to(in_dtype)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k: torch.Tensor | None = None,
    unsqueeze_dim: int = 1,
):
    """Partial RoPE: rotate the leading `cos.shape[-1]` dims, pass the rest."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]

    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    q_out = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    if k is None:
        return q_out
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    k_out = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_out, k_out


class MRoPE:
    """Interleaved 3D (text/height/width) rotary embedding.

    For text-only inference all three sections carry the same positions, so the
    interleaving is a no-op numerically, but we keep the general path so image
    and video position ids work later without a second code path.
    """

    def __init__(self, config: Qwen4ExpTextConfig, device: torch.device, *, rope_theta: float | None = None) -> None:
        dim = config.rotary_dim
        theta = config.rope_theta if rope_theta is None else rope_theta
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        self.mrope_section = list(config.mrope_section)
        self.attention_scaling = 1.0

    def __call__(self, position_ids: torch.Tensor, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: (3, batch, seq) or (batch, seq). Returns cos/sin (batch, seq, rotary_dim)."""
        if position_ids.ndim == 2:
            position_ids = position_ids[None].expand(3, -1, -1)
        inv_freq = self.inv_freq[None, None, :, None].expand(3, position_ids.shape[1], -1, 1)
        pos = position_ids[:, :, None, :].float()
        freqs = (inv_freq.float() @ pos).transpose(2, 3)  # (3, batch, seq, dim/2)
        freqs = self._interleave(freqs)
        emb = torch.cat((freqs, freqs), dim=-1)
        return (emb.cos() * self.attention_scaling).to(dtype), (emb.sin() * self.attention_scaling).to(dtype)

    def _interleave(self, freqs: torch.Tensor) -> torch.Tensor:
        """Rearrange chunked [TTT..HHH..WWW] frequencies into [THWTHW..TT]."""
        out = freqs[0].clone()
        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            out[..., idx] = freqs[dim, ..., idx]
        return out


# ---------------------------------------------------------------------------
# Hyper-connections
# ---------------------------------------------------------------------------


class GatedResidual:
    """Mixes the `hc_count` residual streams into one block input.

    The streams stay separate across the whole network; each block reads a
    low-rank-gated mean of them and writes its output back into every stream
    scaled by a learned per-stream injection weight.  With `block_inject_weight`
    absent (the final mixer) only the mixed input is returned.
    """

    def __init__(
        self,
        config: Qwen4ExpTextConfig,
        hc_norm: torch.Tensor,
        mix_down: torch.Tensor,
        mix_up: torch.Tensor,
        block_inject: torch.Tensor | None,
    ) -> None:
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.hc_norm = RMSNorm(hc_norm, config.rms_norm_eps, group_size=config.hidden_size)
        self.mix_down = mix_down
        self.mix_up = mix_up
        self.block_inject = block_inject
        #: The fp16 copies of the three mixer weights, made on first use and kept.
        #:
        #: The fp16 path used to convert them on every call, which is a per-token cast of every
        #: layer's mixer -- 96 mixers a step on this model, about 1.9 GB of traffic a token for
        #: weights that never change.  They are built during the first step, which runs before any
        #: capture, so the recording sees a stable address and no allocation happens mid-replay.
        self._fp16_weights: dict[int, torch.Tensor] = {}

    def _weight_fp16(self, tensor: torch.Tensor) -> torch.Tensor:
        """The fp16 view of a mixer weight, optionally kept between steps.

        Off by default: the copies are 1.9 GiB across the model's 96 mixers, and on a card that is
        already holding every expert in VRAM that is enough to turn a run into an OOM.  On when
        asked, so the saving can be measured against the same binary.
        """
        if os.environ.get("POCKETLLM_QWEN4_HC_FP16_CACHE") != "1":
            return tensor.to(torch.float16)
        cached = self._fp16_weights.get(id(tensor))
        if cached is None or cached.device != tensor.device:
            cached = tensor.to(torch.float16)
            self._fp16_weights[id(tensor)] = cached
        return cached

    def __call__(self, hyper_input: torch.Tensor):
        normed = self.hc_norm(hyper_input)
        rows = hyper_input.numel() // hyper_input.shape[-1]
        fp16_down, fp16_up, fp16_inject = _hc_fp16_components(rows)
        fp16_supported = (
            normed.device.type == "cuda"
            and normed.dtype is torch.bfloat16
            and self.mix_down.dtype is torch.bfloat16
            and self.mix_up.dtype is torch.bfloat16
            and (
                self.block_inject is None
                or self.block_inject.dtype is torch.bfloat16
            )
        )
        if not fp16_supported:
            fp16_down = fp16_up = fp16_inject = False

        fp32_output = (
            fp16_supported and _hc_fp16_fp32_output_enabled(rows)
        )
        compute_input_fp16 = (
            normed.to(torch.float16)
            if (fp16_down or fp16_inject) and not fp32_output
            else None
        )
        if fp16_down:
            if fp32_output:
                down = F.silu(
                    _fp16_fp32_linear(normed, self.mix_down)
                    / self.hc_count
                )
            else:
                assert compute_input_fp16 is not None
                down = F.silu(
                    F.linear(
                        compute_input_fp16,
                        self._weight_fp16(self.mix_down),
                    )
                    / self.hc_count
                )
        else:
            raw_down = row_invariant_linear(normed, self.mix_down)
            down = _cuda_hc_activation(
                "qwen4_exp_hc_silu",
                raw_down,
                self.hc_count,
            )
            if down is None:
                down = F.silu(raw_down / self.hc_count)

        if fp16_up:
            if fp32_output:
                w = torch.sigmoid(
                    _fp16_fp32_linear(
                        down.to(normed.dtype),
                        self.mix_up,
                    )
                )
            else:
                up_input = (
                    down
                    if down.dtype is torch.float16
                    else down.to(torch.float16)
                )
                w = torch.sigmoid(
                    F.linear(up_input, self._weight_fp16(self.mix_up))
                ).to(normed.dtype)
        else:
            up_input = (
                down
                if down.dtype is normed.dtype
                else down.to(normed.dtype)
            )
            fused = _hc_up_mix_fused(up_input, self.mix_up, normed, self.hc_count, self.hidden_size)
            if fused is not None:
                mixed = fused
                w = None
            else:
                w = torch.sigmoid(row_invariant_linear(up_input, self.mix_up))

        if w is not None:
            w = w.unflatten(-1, (self.hc_count, self.hidden_size))
            mixed = (w * normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        if self.block_inject is None:
            return mixed
        if fp16_inject:
            if fp32_output:
                inject = 2 * torch.sigmoid(
                    _fp16_fp32_linear(normed, self.block_inject)
                    / self.hc_count
                )
            else:
                assert compute_input_fp16 is not None
                inject = 2 * torch.sigmoid(
                    F.linear(
                        compute_input_fp16,
                        self._weight_fp16(self.block_inject),
                    )
                    / self.hc_count
                )
                inject = inject.to(normed.dtype)
        else:
            raw_inject = row_invariant_linear(normed, self.block_inject)
            inject = _cuda_hc_activation(
                "qwen4_exp_hc_inject_gate",
                raw_inject,
                self.hc_count,
            )
            if inject is None:
                inject = 2 * torch.sigmoid(raw_inject / self.hc_count)
        return mixed, hyper_input, inject


def _hc_up_mix_fused(down, mix_up, normed, hc_count: int, hidden: int):
    """The up projection, its sigmoid gate and the stream fold as one kernel, or None.

    The tensor form materialises `hc_count * hidden` weights a mixer and then reads them back
    twice, and there are 96 mixers a token.  Returning None keeps a checkout without the extension
    on the tensor path.
    """
    if (
        down.device.type != "cuda"
        or down.dtype is not torch.bfloat16
        or mix_up.dtype is not torch.bfloat16
        or normed.dtype is not torch.bfloat16
        or not (down.is_contiguous() and mix_up.is_contiguous() and normed.is_contiguous())
        or mix_up.shape != (hc_count * hidden, down.shape[-1])
    ):
        return None
    ext = load_ops()
    exact = None
    if os.environ.get("POCKETLLM_QWEN4_HC_UP_MIX_EXACT") == "1":
        # Experimental: the four stream dot products share one activation load. The lane-strided
        # FMA sequence, the shuffle tree, the sigmoid and the fold order are the qualified entry's,
        # so the qualification gate is raw-bit equality against that entry rather than a tolerance.
        exact = getattr(ext, "qwen4_exp_hc_up_mix_exact", None) if ext is not None else None
        if exact is None:
            raise RuntimeError(
                "POCKETLLM_QWEN4_HC_UP_MIX_EXACT=1 but the loaded native module does not export "
                "qwen4_exp_hc_up_mix_exact; load the experimental build or unset the switch"
            )
    fused = exact
    if fused is None:
        fused = getattr(ext, "qwen4_exp_hc_up_mix", None) if ext is not None else None
    if fused is None:
        return None
    # The kernel works one row a block, so the leading axes are folded into rows here and put back
    # after: a mixer is called on `[batch, seq, hc * hidden]` and answers `[batch, seq, hidden]`.
    lead = normed.shape[:-1]
    down_rows = down.reshape(-1, down.shape[-1])
    normed_rows = normed.reshape(-1, normed.shape[-1])
    if exact is not None:
        out = exact(
            down_rows,
            mix_up,
            normed_rows,
            hc_count,
            hidden,
            _hc_up_mix_exact_threads(),
        )
    else:
        out = fused(down_rows, mix_up, normed_rows, hc_count, hidden)
    return out.reshape(*lead, hidden)


def _hc_up_mix_exact_threads() -> int:
    """Block width for the experimental up-mix, refused here rather than inside a capture."""
    raw = os.environ.get("POCKETLLM_QWEN4_HC_UP_MIX_THREADS", "128")
    try:
        threads = int(raw)
    except ValueError as exc:
        raise RuntimeError(
            f"POCKETLLM_QWEN4_HC_UP_MIX_THREADS must be an integer, got {raw!r}"
        ) from exc
    if threads not in (64, 128, 256, 512):
        raise RuntimeError("POCKETLLM_QWEN4_HC_UP_MIX_THREADS must be 64, 128, 256 or 512")
    return threads


def inject_into_streams(
    block_output: torch.Tensor, hyper_input: torch.Tensor, injection_weights: torch.Tensor
) -> torch.Tensor:
    """Scatter a block's output back into all hyper-connection streams."""
    groups = injection_weights.shape[-1]
    if (
        _hc_cuda_enabled(
            block_output.numel() // block_output.shape[-1]
        )
        and block_output.device.type == "cuda"
        and block_output.is_contiguous()
        and hyper_input.is_contiguous()
        and injection_weights.is_contiguous()
        and block_output.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and hyper_input.dtype is block_output.dtype
        and injection_weights.dtype is block_output.dtype
        and block_output.device == hyper_input.device == injection_weights.device
        and hyper_input.shape[:-1] == block_output.shape[:-1]
        and injection_weights.shape[:-1] == block_output.shape[:-1]
        and hyper_input.shape[-1] == groups * block_output.shape[-1]
    ):
        ext = load_ops()
        if ext is not None and hasattr(ext, "qwen4_exp_inject"):
            return ext.qwen4_exp_inject(
                block_output,
                hyper_input,
                injection_weights,
                groups,
            )
    injection = block_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    return hyper_input + injection.flatten(-2)


# ---------------------------------------------------------------------------
# PLE hashed n-gram embeddings
# ---------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


_MASK64_BITS = (1 << 64) - 1
_SIGN64_BIT = 1 << 63


def _signed64(value: int) -> int:
    """`value` as the int64 the tensor path would hold, wrapping exactly as it does."""
    value &= _MASK64_BITS
    return value - (1 << 64) if value >= _SIGN64_BIT else value


def build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> torch.Tensor:
    """Odd per-position hash multipliers, derived deterministically from the seed."""
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    multipliers = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        multipliers.append(2 * (_splitmix64(value) % half_bound) + 1)
    return torch.tensor(multipliers, dtype=torch.long)


class NGramHasher:
    """Maps token ids to hashed n-gram embedding row ids.

    Rows are looked up in a table whose `total_vocab_size` is the concatenation
    of one distinct-prime slice per (n-gram order, head).  The hash mixes the
    shifted token ids by XOR of per-position multiples, then takes a remainder
    per head.  Token history crossing an EOS boundary is masked to EOS so
    n-grams never straddle documents.
    """

    def __init__(self, config: Qwen4ExpTextConfig, ple_layer_index: int, device: torch.device) -> None:
        # The hash runs on int64 token ids and its output feeds a host-side
        # gather, so it stays on the CPU regardless of the compute device: doing
        # it on the GPU would only add a device round-trip.
        device = torch.device("cpu")
        self.ngram_size = config.ngram_size
        self.context_len = config.ngram_size - 1
        self.heads_per_ngram = config.heads_per_ngram
        self.eos_token_id = config.primary_eos_token_id
        sizes, offsets, padded = config.ngram_head_vocab_sizes(ple_layer_index)
        self.head_vocab_sizes = torch.tensor(sizes, dtype=torch.long, device=device)
        self.head_offsets = torch.tensor(offsets, dtype=torch.long, device=device)
        self.padded_vocab_size = padded
        self.layer_multipliers = build_layer_multipliers(
            config.vocab_size, config.ngram_size, ple_layer_index, config.seed
        ).to(device)

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
        )
        position_in_segment = positions.unsqueeze(0) - (previous_eos + 1)
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def _row_ids_one_token(self, token_history: torch.Tensor) -> torch.Tensor:
        """``row_ids(history, 1)`` for a history that is exactly one n-gram window.

        The tensor path builds a full shifted copy of the history per n-gram order -- a cummax, a
        concatenate, two gathers and a select each -- to then read a single element out of each.
        At a decode step the window is `ngram_size` columns wide and one row is wanted, so the same
        answer is a handful of integer operations in CPython, and it costs about a tenth as much.

        The arithmetic is int64 with wraparound because that is what the tensor path does, and the
        wraparound is load-bearing: a product that overflows changes the sign, and ``remainder``
        takes the floor, so a "mathematically equal" big-integer product would differ in the row
        it selects.  As a result the Python product and XOR are re-signed to int64 exactly as the
        tensor arithmetic would be, and the modulo is Python's floor modulo, which is what
        ``torch.remainder`` computes for a positive divisor.
        """
        rows = token_history.tolist()
        eos = self.eos_token_id
        multipliers = [int(value) for value in self.layer_multipliers.tolist()]
        sizes = [int(value) for value in self.head_vocab_sizes.tolist()]
        offsets = [int(value) for value in self.head_offsets.tolist()]
        heads = self.heads_per_ngram
        span = self.ngram_size
        last = span - 1
        answer = []
        for row in rows:
            # `shifted[s]` at the last position: the token `s` places back, provided that token is
            # in the same document -- an EOS between the two makes it an EOS instead.
            shifted = []
            for shift in range(span):
                source = last - shift
                if source < 0:
                    shifted.append(eos)
                    continue
                crossed = eos in row[source + 1:last]
                shifted.append(eos if crossed else row[source])
            head_ids = []
            for ngram in range(2, span + 1):
                start = (ngram - 2) * heads
                mixed = _signed64(multipliers[0] * shifted[0])
                for position in range(1, ngram):
                    mixed = _signed64(mixed ^ _signed64(multipliers[position] * shifted[position]))
                for head in range(start, start + heads):
                    head_ids.append(mixed % sizes[head] + offsets[head])
            answer.append(head_ids)
        # `[batch, out_len, heads]`, the shape the tensor path returns.
        return torch.tensor(answer, dtype=torch.long).unsqueeze(1)

    def row_ids(self, token_history: torch.Tensor, out_len: int) -> torch.Tensor:
        """token_history: (batch, context_len + out_len) -> (batch, out_len, ngram_heads)."""
        token_history = token_history.to("cpu", dtype=torch.long)
        if out_len == 1 and token_history.shape[1] == self.ngram_size:
            # The whole history is one window and one row is wanted: the tensor path would build
            # `ngram_size` shifted copies to read one element from each.
            return self._row_ids_one_token(token_history)
        shifted = [self._shift_right_ignore_eos(token_history, s) for s in range(self.ngram_size)]
        blocks = []
        for ngram in range(2, self.ngram_size + 1):
            start = (ngram - 2) * self.heads_per_ngram
            end = start + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for position in range(1, ngram):
                mixed = torch.bitwise_xor(mixed, shifted[position] * self.layer_multipliers[position])
            sizes = self.head_vocab_sizes[start:end].view(1, 1, -1)
            offsets = self.head_offsets[start:end].view(1, 1, -1)
            blocks.append(torch.remainder(mixed.unsqueeze(-1), sizes) + offsets)
        return torch.cat(blocks, dim=-1)[:, -out_len:]


# ---------------------------------------------------------------------------
# MoE
# ---------------------------------------------------------------------------


class TopKRouter:
    """Softmax-over-all-experts router with renormalized top-k weights."""

    def __init__(self, weight: torch.Tensor, top_k: int, *, norm_topk_prob: bool = True) -> None:
        self.weight = weight
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

    def __call__(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = row_invariant_linear(hidden_states, self.weight)
        probs = F.softmax(logits, dim=-1, dtype=torch.float32)
        top_value, indices = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_value = top_value / top_value.sum(dim=-1, keepdim=True)
        return top_value.to(logits.dtype), indices


class DenseMLP:
    """SwiGLU MLP used by the shared expert."""

    def __init__(self, gate_proj: torch.Tensor, up_proj: torch.Tensor, down_proj: torch.Tensor) -> None:
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.silu(
            prefill_linear(x, self.gate_proj)
        ) * prefill_linear(x, self.up_proj)
        return prefill_linear(hidden, self.down_proj)


def swiglu_expert(
    x: torch.Tensor, gate_up_proj: torch.Tensor, down_proj: torch.Tensor
) -> torch.Tensor:
    """One expert's fused-gate SwiGLU: gate_up is [2*inter, hidden] row-packed."""
    gate, up = row_invariant_linear(x, gate_up_proj).chunk(2, dim=-1)
    return row_invariant_linear(F.silu(gate) * up, down_proj)
