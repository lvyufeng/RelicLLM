"""The decode-stack bindings: what they displace, what they refuse, and what they restore.

Each binding is install/close around a real module attribute, so these tests are mostly about the undo
path and the refusals — the arithmetic itself is covered by relic-core's component tests, which
compare the fused kernels to the eager oracles on the device.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from relicllm.models.deepseek_v4_1 import ascend_decode_stack as stack  # noqa: E402
from relicllm.models.deepseek_v4_1 import ascend_narrow_head_binding as narrow  # noqa: E402
from relicllm.models.deepseek_v4_1 import ascend_rope_binding as rope  # noqa: E402
from relicllm.models.deepseek_v4_1 import ascend_row_norm_binding as row_norm  # noqa: E402
from relicllm.models.deepseek_v4_1 import ascend_window_idxs_binding as window  # noqa: E402

# Each binding keeps its owner attribute somewhere different: a class for the two that patch a
# method, the module for the two that patch a module-level function. The tuple records both the
# object being patched and the object carrying the owner.
_ALL = (
    (row_norm, "attention", "RMSNorm", "forward", "class"),
    (rope, "attention", None, "apply_rotary_emb_pairs", "module"),
    (window, "attention", None, "get_window_topk_idxs", "module"),
    (narrow, "modules", "Head", "forward", "module"),
)


def _target(module_name, attribute):
    import importlib

    return importlib.import_module("relicllm.models.deepseek_v4_1." + module_name), attribute


@pytest.mark.parametrize("binding, host, owner_class, attribute, owner_scope", _ALL)
def test_install_and_close_round_trip(binding, host, owner_class, attribute, owner_scope):
    module, name = _target(host, attribute)
    container = getattr(module, owner_class) if owner_class else module
    original = getattr(container, name)
    owner = binding.install()
    try:
        assert getattr(container, name) is not original, "install did not replace the attribute"
        described = owner.describe()
        assert described["installed"] is True
        assert described["numerical_or_performance_acceptance_claimed"] is False
    finally:
        owner.close()
    assert getattr(container, name) is original, "close did not restore the original object"


@pytest.mark.parametrize("binding, host, owner_class, attribute, owner_scope", _ALL)
def test_a_second_install_is_refused(binding, host, owner_class, attribute, owner_scope):
    owner = binding.install()
    try:
        with pytest.raises(RuntimeError, match="already installed"):
            binding.install()
    finally:
        owner.close()


@pytest.mark.parametrize("binding, host, owner_class, attribute, owner_scope", _ALL)
def test_close_is_refused_from_another_thread(binding, host, owner_class, attribute, owner_scope):
    owner = binding.install()
    failure = []

    def _close():
        try:
            owner.close()
        except RuntimeError as exc:
            failure.append(str(exc))

    try:
        worker = threading.Thread(target=_close)
        worker.start()
        worker.join()
        assert failure and "thread that installed it" in failure[0]
    finally:
        owner.close()


@pytest.mark.parametrize("binding, host, owner_class, attribute, owner_scope", _ALL)
def test_close_is_refused_when_the_attribute_moved_behind_it(binding, host, owner_class, attribute, owner_scope):
    module, name = _target(host, attribute)
    container = getattr(module, owner_class) if owner_class else module
    original = getattr(container, name)
    owner = binding.install()
    try:
        setattr(container, name, original)
        with pytest.raises(RuntimeError, match="no longer the one this binding installed"):
            owner.close()
    finally:
        setattr(container, name, original)
        holder = container if owner_scope == "class" else module
        for leftover in (binding._OWNER, binding._FALLBACK):
            if hasattr(holder, leftover):
                delattr(holder, leftover)


def test_the_stack_installs_only_what_needs_no_arguments():
    installed, skipped = stack.install_available()
    try:
        assert len(installed) == len(stack.INSTALLABLE)
        assert {name for name, _ in skipped} == set(stack.NEEDS_WIRING)
        # Each installed component has to be closable, or the caller cannot undo the whole set.
        for name, owner in installed:
            assert hasattr(owner, "close"), name
    finally:
        for _, owner in installed:
            try:
                owner.close()
            except RuntimeError:
                pass


def test_the_stack_reports_the_registered_result_and_its_deviation_classes():
    described = stack.describe()
    registered = described["registered"]
    assert registered["tokens_per_second"] == 14.287598
    assert registered["batching"].startswith("one request, one token")
    assert registered["objective_met"] is False
    classes = {row["class"] for row in described["components"]}
    assert classes == {"bit-exact", "approved deviation", "layout", "dispatch"}
    # The one broad deviation has to be recorded as such rather than hidden in a bit-exact story.
    deviation = described["approved_deviations"]["ascend_narrow_head_binding"]
    assert deviation["logits_dtype"].startswith("bfloat16")
    assert deviation["checkpoint_payload_unchanged"] is True


def test_importing_the_stack_installs_nothing():
    """A module that patched on import would be a serving default, which every component is not."""
    from relicllm.models.deepseek_v4_1 import attention, modules

    assert getattr(modules, narrow._OWNER, None) is None
    assert getattr(attention, row_norm._OWNER, None) is None
    assert getattr(attention, rope._OWNER, None) is None
    assert getattr(attention, window._OWNER, None) is None
    assert stack.describe()["uninstalled_at_import"] is True