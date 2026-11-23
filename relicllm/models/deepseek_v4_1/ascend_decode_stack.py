"""The registered pure-B1 decode stack: what composes it, and how to install it.

This module is the reproducible statement of one measured configuration — the registered
**14.287598 tok/s at 69.9922 ms per token** on 8x Ascend 910B3 (65536 MB per card) — so the
configuration lives in the repository rather than in a launcher script on one host.

It installs nothing by design. `install_available()` installs the components that need no extra
argument, and reports the rest, because several of them require a model instance or a dispatch object
the caller owns. A binding installed silently at import time would be a serving default, and every
component here is opt-in.

Three kinds of component appear below and they are not interchangeable, so each is labelled:

* **bit-exact** — the fused path reproduces the reference tensor. Validated by comparison, and the
  component tests are the gate.
* **approved deviation** — the output tensor differs from the reference and the difference was
  accepted explicitly, with its size recorded. These are the two that are not bit-exact.
* **layout / dispatch** — the tensor is the same and only the operand layout or the number of
  launches changes.

The gains below are the step-level deltas measured on the registered lane, each as the difference
between two columns of one trial rather than across trials; the control means of those trials stayed
inside 12.449932 to 12.479610, which is what makes them comparable. They are recorded to say what each
component was worth, not to claim that a fresh host will reproduce the sum.
"""
from __future__ import annotations

#: Components that need no argument beyond the bindings they patch, in the order they were measured.
INSTALLABLE = (
    "relicllm.models.deepseek_v4_1.ascend_indexer_sort_binding",
    "relicllm.models.deepseek_v4_1.ascend_row_norm_binding",
    "relicllm.models.deepseek_v4_1.ascend_rope_binding",
    "relicllm.models.deepseek_v4_1.ascend_window_idxs_binding",
    "relicllm.models.deepseek_v4_1.ascend_narrow_head_binding",
)

#: Components that need caller-owned arguments, with what they need.
NEEDS_WIRING = {
    "relicllm.models.deepseek_v4_1.ascend_sparse_attention_layout":
        "a model's attention modules, or the module's own install() scope",
    "relicllm.models.deepseek_v4_1.ascend_attention_reuse":
        "the model, a backend, and certify_immutable_inputs=True",
    "relicllm.models.deepseek_v4_1.fused_activation_binding":
        "the model and an explicit backend",
    "relicllm.models.deepseek_v4_1.ascend_inline_tp_graphs":
        "the expert TP dispatch object to record against",
}

#: What each installed component displaces, its deviation class and its measured step-level delta.
COMPONENTS = (
    ("ascend_row_norm_binding", "RMSNorm.forward", "bit-exact", "+2.52%"),
    ("ascend_rope_binding", "apply_rotary_emb_pairs", "bit-exact", "+8.25%"),
    ("ascend_window_idxs_binding", "get_window_topk_idxs (decode)", "bit-exact", "+9.50% then +12.31%"),
    ("ascend_narrow_head_binding", "Head.forward", "approved deviation", "+11.10%"),
    ("ascend_indexer_sort_binding", "Indexer.forward selection", "bit-exact", "+14.74% cumulative"),
    ("ascend_sparse_attention_layout", "attention sparse_attn operand layout", "layout", "included above"),
    ("ascend_inline_tp_graphs", "the expert TP layer body", "dispatch", "included above"),
)

#: The registered result, and the ceiling the objective reached, stated once so the two cannot drift.
REGISTERED = {
    "tokens_per_second": 14.287598,
    "milliseconds_per_token": 69.9922,
    "hardware": "8x Ascend 910B3, 65536 MB per card",
    "batching": "one request, one token per forward, no speculation and no MTP",
    "reported_ceiling_at_acceptance_length_one": 36.9,
    "objective_met": False,
}

APPROVED_DEVIATIONS = {
    "ascend_narrow_head_binding": {
        "logits_dtype": "bfloat16 instead of float32",
        "measured_broad_deviation": "1,034,216 of 1,034,240 values at up to 0.0623 on every step",
        "checkpoint_payload_unchanged": True,
    },
}


def install_available():
    """Install every component that needs no caller argument; report what still needs wiring.

    Returns ``(installed, skipped)``: the owners that were installed, and the modules that were not
    installed because they need arguments only the caller can supply.
    """
    import importlib

    installed = []
    skipped = []
    for name in INSTALLABLE:
        module = importlib.import_module(name)
        installed.append((name, module.install()))
    for name, requirement in NEEDS_WIRING.items():
        skipped.append((name, requirement))
    return installed, skipped


def describe():
    """What this stack claims, and what it does not."""
    return {
        "registered": dict(REGISTERED),
        "components": [{"module": module, "displaces": displaces, "class": kind, "measured": measured}
                       for module, displaces, kind, measured in COMPONENTS],
        "approved_deviations": {key: dict(value) for key, value in APPROVED_DEVIATIONS.items()},
        "installable": list(INSTALLABLE),
        "needs_wiring": dict(NEEDS_WIRING),
        "uninstalled_at_import": True,
    }