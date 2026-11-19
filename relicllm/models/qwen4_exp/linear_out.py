"""Cat-free replacement for the strict row-invariant BF16 linear loop, narrowly qualified.

The strict verifier runs small BF16 batches (2..8 flattened rows) through independent one-row
`F.linear` calls so that each row keeps decode's M=1 projection arithmetic. Today that loop
materialises one output per row and concatenates them, which costs a `torch.cat` and one output
allocation per row inside a CUDA graph.

This module offers an optional variant that keeps the same per-row calls but writes each result
straight into a slice of one preallocated output through ATen's `linear.out` overload. It removes
the per-row result allocations and the concatenation; it does **not** change the requested
per-row geometry.

Two staging modes exist, and the difference between them is not cosmetic:

* **Two-dimensional staging** (`POCKETLLM_QWEN4_LINEAR_OUT_2D=1`). Each row is handed to the kernel
  as a plain `(1, in_features)` matrix. On 8xRTX 4090 (`sm_89`, torch 2.11) this reproduced the
  reference loop's bits on **every** tested real projection shape, with and without bias, and
  survived cancellation-heavy vectors and a captured graph replay.
* **Preserved rank** (the previous default when the 2D switch was absent). Each row keeps the
  caller's leading singleton dimensions. On the same 8xRTX 4090 bit oracle this **failed**: 22 of
  269 cases disagreed with the reference while executed against the same operands, including 20 of
  the 22 rank-three real-projection configurations (the hyper-connection `10240 -> 320` mix was the
  only rank-three shape that matched), the adversarial reduction vectors, and the captured replay.
  The two-dimensional staging passed all of those cases.

Because preserved rank is a *rejected* mode on the only device that has been probed, this module no
longer serves it in production. A call with three or more dimensions returns ``None`` (pure
fallback to the reference loop) unless the caller has explicitly selected two-dimensional staging.
A two-dimensional input is staged as two-dimensional either way, so it stays accepted: at rank two
the preserved-rank request and the matrix request are the same request.

The CPU build shows the same split from the other side -- `linear.out` agrees with the composite
`F.linear` at rank two and differs by several BF16 ULPs at rank three -- which is why CPU equality
was never the promotion argument. The device oracle is.

Qualification scope: this is exact on the 8xRTX 4090 host the oracle ran on. It is **not** a claim
about the RTX 2080 Ti (`sm_75`) build, where no device oracle for this path exists. The switch
therefore still defaults off and the reference loop stays the default for every unqualified shape.
"""

from __future__ import annotations

import os

import torch

#: Explicit opt-in. Default off: the arithmetic of `linear.out` is only qualified by an on-device
#: oracle for the shapes in use, never by reasoning about the composite it replaces.
_ENABLE_ENV = "POCKETLLM_QWEN4_LINEAR_OUT"
#: The pre-existing strict-row switch. This helper is an implementation detail of that contract and
#: must never widen it, so both switches have to be on.
_STRICT_ENV = "POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT"
#: How each row is handed to the kernel. ``1`` stages every row as a plain two-dimensional matrix,
#: which is the mode the device oracle accepted. Unset preserves the caller's rank, which the device
#: oracle rejected at rank three and above; the helper refuses to serve that mode for production.
_STAGE_2D_ENV = "POCKETLLM_QWEN4_LINEAR_OUT_2D"

_MIN_ROWS = 2
_MAX_ROWS = 8


def _flag(name: str) -> bool:
    """A runtime switch, spelled exactly ``1``.

    Every strict-row switch in this runtime (`POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT` and the
    reduction layouts beside it) is read as ``== "1"``. Reading a truthy set here instead would let
    a launch enable this experiment through a spelling that enables nothing else, which is the kind
    of split-brain default a measurement then cannot explain.
    """
    return os.environ.get(name, "0") == "1"


def _enabled() -> bool:
    """Whether the caller opted in to the out-overload variant, and kept strict rows on."""
    return _flag(_ENABLE_ENV) and _flag(_STRICT_ENV)


def linear_out_enabled() -> bool:
    """Whether the *qualified* replacement of the model's strict-row loop may be bound.

    All three switches at once: the experiment, the strict-row contract it sits inside, and the
    two-dimensional staging the device oracle accepted. A caller that binds the model path on this
    predicate never serves the rejected preserved-rank mode, and with any of the three unset it
    keeps the reference `torch.cat` loop.

    Kept separate from :func:`qwen4_exp_linear_out` because the two answer different questions: this
    one is "may this path replace the loop", the other is "can this particular call be served".
    """
    return _enabled() and _stage_two_dimensional_rows()


def _available_overloads() -> list:
    """The overload names this torch build registered for `aten::linear`."""
    overloads = getattr(torch.ops.aten.linear, "overloads", None)
    return list(overloads()) if callable(overloads) else []


def _linear_out_op():
    """ATen's `linear.out`, or a clear failure when this torch build does not expose it.

    Returning ``None`` here would make an explicitly requested path silently fall back, and the
    caller would have no way to tell "not available" from "not beneficial" in its measurements.
    A call the guards refuse never reaches this lookup, so a refused shape cannot raise on a build
    without the overload.
    """
    names = _available_overloads()
    if "out" not in names:
        raise RuntimeError(
            "POCKETLLM_QWEN4_LINEAR_OUT=1 needs torch.ops.aten.linear.out, which this torch build "
            f"does not provide (it has {sorted(names)}); unset the switch to keep the cat-based path"
        )
    return torch.ops.aten.linear.out


def _stage_two_dimensional_rows() -> bool:
    """Whether rows are staged as two-dimensional matrices instead of keeping the caller's rank."""
    return _flag(_STAGE_2D_ENV)


def qwen4_exp_linear_out(x, weight, bias=None):
    """One-row `linear` calls written into a single preallocated output, or ``None``.

    Returns ``None`` whenever this build, these operands or this environment are outside the frozen
    experiment, so a caller can fall back to the reference implementation without a branch of its
    own. Every accepted call issues exactly one M=1 `linear.out` per flattened row and writes them
    into one contiguous allocation with the caller's leading dimensions.

    **Rank above two requires `POCKETLLM_QWEN4_LINEAR_OUT_2D=1`.** The preserved-rank request is a
    mode the 8xRTX 4090 bit oracle rejected (22 of 269 cases disagreed with the reference on the
    same operands while every two-dimensional case matched), so it is refused here rather than
    offered and measured as if it were a candidate. A two-dimensional input is accepted either way,
    because at rank two the preserved-rank request and the matrix request are identical; that is the
    only geometry where the two staging spellings cannot disagree.

    The oracle's qualification belongs to the eight-card `sm_89` host it ran on. Preserving rank
    three or more on another architecture is not covered by it and is not offered.
    """
    if not _enabled():
        return None
    if not isinstance(x, torch.Tensor) or not isinstance(weight, torch.Tensor):
        return None
    if bias is not None and not isinstance(bias, torch.Tensor):
        return None
    if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
        return None
    if x.device != weight.device or x.ndim < 2 or weight.ndim != 2:
        return None
    # The rejected mode. Checked before the overload lookup so a refused call is a pure fallback:
    # it must not depend on, or raise from, a kernel this build may not even expose.
    if x.ndim > 2 and not _stage_two_dimensional_rows():
        return None
    if x.numel() == 0 or x.shape[-1] != weight.shape[1]:
        return None
    rows = x.numel() // x.shape[-1]
    if not _MIN_ROWS <= rows <= _MAX_ROWS:
        return None
    if bias is not None:
        if (bias.ndim != 1 or bias.shape[0] != weight.shape[0]
                or bias.dtype is not torch.bfloat16 or bias.device != x.device):
            return None
    # An `out=` kernel cannot carry an autograd graph. The runtime calls this under
    # `torch.inference_mode()`, where the check is false and costs nothing.
    if torch.is_grad_enabled() and (
        x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    ):
        return None

    op = _linear_out_op()
    out_features = weight.shape[0]
    # The reference loop's geometry. Two-dimensional staging hands the kernel a plain matrix, which
    # is the request the device oracle qualified and the shape the composite `F.linear` folds each
    # row down to anyway. The preserved-rank branch is reachable only at `x.ndim == 2`, where it is
    # the same request; it is kept because it is literally the reference loop's spelling.
    flat = x.reshape(rows, x.shape[-1])
    if x.ndim == 2 or _stage_two_dimensional_rows():
        input_single = (1, x.shape[-1])
        output_single = (1, out_features)
    else:  # pragma: no cover - unreachable: rank above two without 2D was refused above
        input_single = (1,) * (x.ndim - 1) + (x.shape[-1],)
        output_single = (1,) * (x.ndim - 1) + (out_features,)
    buffer = torch.empty(rows, out_features, dtype=x.dtype, device=x.device)

    for row in range(rows):
        staged = flat[row:row + 1].reshape(input_single).contiguous()
        # A view of this row inside the one allocation, shaped as the kernel's output rank.
        target = buffer[row:row + 1].reshape(output_single)
        op(staged, weight, bias, out=target)

    return buffer.reshape(*x.shape[:-1], out_features)


__all__ = ["linear_out_enabled", "qwen4_exp_linear_out"]