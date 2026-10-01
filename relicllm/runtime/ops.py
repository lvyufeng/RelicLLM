"""The one door to the operator surface, so a second hardware backend has somewhere to arrive.

Every kernel this runtime calls arrives as a named binding on one extension module, and until now
each call site asked for it directly: ``load_cuda_kernel()`` from ``relic_core``, at fifty-six places
across seventeen files. That was a reasonable spelling while CUDA was the only answer. It stops being
one the moment a second answer exists, because the question "give me the ops" has a different reply
on different hardware and the *asking* should not.

This module is that question, asked once. It is deliberately small: a registry, a lookup, and a list
of the bindings a provider has to supply. It is not a wrapper, and it does not translate anything --
what :func:`load_ops` returns on CUDA is the exact object ``load_cuda_kernel()`` returns, by identity,
so every ``hasattr(ops, "<binding>")`` probe the call sites already use keeps working unchanged.

## Why the call sites probe instead of requiring

Two postures already exist in the tree, and this seam has to keep both working:

* **Probe and fall back.** ``models/xing4_0/hyper_connection.py`` says it best: *"``None`` rather
  than raising: this is one kernel of many in a tree that also builds for Ascend, and a forward pass
  that falls back to the eager arithmetic is correct, just slower."* Those call sites hold the module
  as an optional attribute and branch per call.
* **Require.** ``components/gguf/quantized_ops.py`` and ``models/mimo_v2/device_experts.py`` raise
  when the binding is absent, because there the op *is* the implementation and there is no slower
  correct answer.

Both are why an unimplemented platform answers ``None`` rather than raising here. A provider that
does not exist yet is exactly the case the first posture was written for, so on a host with no CUDA
and no registered provider every probing call site takes the branch it already has.

## The surface a provider must supply

:data:`BINDINGS` is measured rather than recalled: it is the set of names the built extension exports
*and* that this tree names anywhere under ``src/``. That was 46 of the extension's 57 public names at
the time of writing -- the other eleven are exports no runtime here calls. The list is a contract and
a checklist, not a gate: :func:`missing_bindings` reports the difference so a partial provider can say
what it has, and nothing here refuses to hand back a provider that is incomplete.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from relic_core.kernels.cuda_loader import load_cuda_kernel

from .device import probe_accelerator


#: The bindings this tree calls, which is what a provider for a new platform has to answer for.
#:
#: Derived by intersecting the built extension's public names with the names referenced anywhere
#: under ``src/``, so it is what the runtime uses and not what the extension offers. It is grouped
#: by the model family that reaches it, because that is how a partial port is actually staged --
#: a Qwen-only provider wants to know which thirteen of these are Qwen's.
BINDINGS: tuple[str, ...] = (
    # Attention and its surroundings, shared by several families.
    "fused_rms_norm_forward",
    "fused_kv_rope_actquant_inplace",
    "fused_q_rmsnorm_rope_inplace",
    "fused_o_inverse_rope_inplace",
    # DeepSeek-V4: the sparse indexer, the int8/Q8 path, and the hyper-connection.
    "c4_topk_from_scores",
    "fused_c4_indexer_decode_forward",
    "int8_gemm_pair_forward",
    "fp4_weight_to_int8_forward",
    "q8_0_gemm_forward",
    "wo_a_int8_forward",
    "hc_split_pre_forward",
    "hc_post_forward",
    # The MoE group, in its int8, fp4 and prefill shapes.
    "moe_group_routes",
    "moe_finalize_reduce_forward",
    "moe_single_token_int8_forward",
    "moe_single_token_fp4_forward",
    "moe_multi_token_fp4_forward",
    "moe_prefill_int8_grouped_forward",
    "moe_prefill_int8_grouped_gemm_forward",
    "moe_prefill_int8_grouped_gemm_bucketed_forward",
    "moe_prefill_fp4_grouped_gemm_forward",
    # The GGUF path: Q2_K and IQ-quantized experts, single-token and prefill.
    "gguf_quant_gemm_forward",
    "gguf_quant_gemm_pair_forward",
    "gguf_quant_gemm_prefill_forward",
    "gguf_quant_embedding_forward",
    "gguf_single_token_route_slots",
    "gguf_moe_prefill_grouped_forward",
    "gguf_moe_single_token_iq1m_forward",
    "gguf_moe_single_token_iq2_q2k_forward",
    # MiMo-V2.6-Flash.
    "mimo_rope_rows",
    "mimo_decode_attention",
    "mimo_noaux_tc_route",
    # MiniMax-M2.
    "fused_minimax_rope_halfsplit_inplace",
    # Qwen4-Exp, including the ternary/hadamard path.
    "qwen4_exp_qsa_bf16_forward",
    "qwen4_exp_gated_delta_bf16_forward",
    "qwen4_exp_moe_prefill_bf16_forward",
    "qwen4_exp_grouped_rms_norm",
    "qwen4_exp_inject",
    "qwen4_exp_hc_inject_gate",
    "qwen4_exp_hc_silu",
    "hadamard128_forward",
    # Xing4.0's hyper-connection.
    "xing4_hyper_connection_forward",
    # The collectives the DeepSeek-V4 decode path builds for itself, in place of NCCL.
    "custom_allreduce_open",
    "custom_allreduce_inplace",
    "custom_allreduce_close",
    "custom_allreduce_ipc_handle",
)


#: Platform -> the callable that produces its operator module. Empty, and that is the current state
#: of Ascend support rather than an oversight: this is the seam a provider arrives through, and the
#: provider itself is a later piece of work.
_PROVIDERS: dict[str, Callable[[], Any]] = {}


def register_provider(platform: str, loader: Callable[[], Any]) -> None:
    """Teach :func:`load_ops` where the operators for ``platform`` come from.

    ``loader`` is called on every :func:`load_ops` and should do its own caching, the way
    ``load_cuda_kernel`` does. Registering a platform that already has one replaces it, which is what
    makes the seam testable: a test registers a plain object under a name, asks for it, and takes it
    back out.
    """
    _PROVIDERS[platform] = loader


def unregister_provider(platform: str) -> None:
    """Drop a registration, if there is one. The other half of what makes the seam testable."""
    _PROVIDERS.pop(platform, None)


def registered_platforms() -> tuple[str, ...]:
    """The platforms a provider has been registered for, for a caller that wants to report it."""
    return tuple(sorted(_PROVIDERS))


@lru_cache(maxsize=1)
def _host_platform() -> str:
    """What this host is, asked once.

    The default argument to :func:`load_ops` reaches :func:`probe_accelerator`, which costs about
    sixty microseconds on a CUDA-only box -- most of it the *failing* ``import torch_npu`` inside
    ``_npu_available``, which CPython does not cache however many times it fails. That is expensive
    for a question whose answer is a property of the process, and the call sites that use the
    default are inside decode loops: ``models/deepseek_v4/runtime.py``'s ``rotate_activation``
    resolves the module per layer per step, and ``models/minimax_m2/architecture.py``'s
    ``_apply_rope`` twice per layer.

    Latching it is the smallest fix that removes the cost, and it is deliberately the *only* thing
    latched: the provider lookup beside it stays live, so ``register_provider`` and
    ``unregister_provider`` take effect on the next call with no invalidation to get wrong. The
    extension module was never the expensive half -- ``load_cuda_kernel`` is a cached global at a
    few nanoseconds.
    """
    return probe_accelerator().platform


def load_ops(platform: str | None = None) -> Any | None:
    """The operator module for ``platform``, or ``None`` when nothing is built for it.

    ``None`` as the argument asks the host, which is the same question ``--device auto`` asks and
    gets the same answer: :func:`relicllm.backends.factory.select_backend` resolves the flag through
    the same probe, so a launch and its operator lookup agree about which platform they are on. It
    is asked once per process -- see :func:`_host_platform` -- so calling this in a loop is a
    dictionary lookup and an attribute fetch, not a host probe.

    Three answers, in order:

    * a registered provider for the platform, if there is one;
    * on CUDA, ``relic_core``'s extension -- **the same object** ``load_cuda_kernel()`` returns, not
      a copy, so identity comparisons and the call sites' ``hasattr`` probes behave exactly as they
      did before this module existed;
    * ``None`` otherwise, which is not a failure but the answer the probing call sites were written
      for. See the module docstring: a forward pass that falls back to eager arithmetic is correct,
      just slower, and that is what a platform with no provider gets.

    A ``None`` return is indistinguishable from "the extension failed to load", for the reason
    ``cuda_loader`` documents -- it swallows every failure into the same ``None``, and this module
    does not add a second place that guesses why. What it does add is a *platform* to the question,
    so the answer "there is nothing for this hardware" is now expressible at all.
    """
    resolved = _host_platform() if platform is None else platform
    provider = _PROVIDERS.get(resolved)
    if provider is not None:
        return provider()
    if resolved == "cuda":
        return load_cuda_kernel()
    return None


def missing_bindings(provider: Any) -> tuple[str, ...]:
    """Which of :data:`BINDINGS` ``provider`` does not answer for, in declaration order.

    A diagnostic rather than a check, and it is asked this way round on purpose: a provider is
    usually partial while it is being built, and the useful question is which parts, not whether it
    is finished. Nothing in the runtime calls this -- a call site's own ``hasattr`` asks the only
    question it needs, about the one binding it is about to use.
    """
    return tuple(name for name in BINDINGS if not hasattr(provider, name))


__all__ = [
    "BINDINGS",
    "load_ops",
    "missing_bindings",
    "register_provider",
    "registered_platforms",
    "unregister_provider",
]
