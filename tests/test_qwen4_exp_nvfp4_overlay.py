"""CPU-only contract tests for the additive NVFP4 operator overlay installer.

These tests never build, load or execute a real extension: the module loader is injected through
``nvfp4_overlay._MODULE_LOADER`` and the "shared objects" are small files whose bytes are never
opened by a dynamic loader. The one exception is
``test_default_loader_reports_a_non_extension_file``, which exercises the *real* loader's failure
path on a file that is not a loadable extension -- that path needs no CUDA, no torch device and no
build, and it must fail rather than be swallowed.

Real-overlay behaviour (dlopen of the experimental ``.so``, identity of the base's own function
objects across a real load, and any GPU result) is the parent's remote measurement and is **not**
claimed here.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import types
from pathlib import Path

import pytest

from relicllm.models.qwen4_exp import nvfp4_overlay as ov

COMPACT, STORE = ov.OVERLAY_OPS
MARKER = ov.MARKER

#: Attributes a qualified base module carries. Their identity must survive an install unchanged.
BASELINE_NAMES = (
    "nvfp4_moe_rows_forward",
    "nvfp4_moe_rows_forward_exact_fast",
    "nvfp4_moe_token_forward",
    "nvfp4_exact_decode_e2m1",
    "nvfp4_exact_decode_e4m3",
    "qwen4_exp_gated_delta_bf16_forward",
    "qwen4_exp_gated_delta_bf16_forward_states",
    "qwen4_exp_qsa_splitk_rows_bf16_forward",
    "qwen4_exp_grouped_rms_norm",
    "qwen4_exp_fp8_linear",
)


def _op(label: str):
    """A distinct callable per name, so ``is`` comparisons can attribute every attribute."""

    def impl(*args, **kwargs):
        return f"{label}"

    impl.__name__ = label
    return impl


@pytest.fixture(autouse=True)
def clean_overlay_state():
    """No install may leak its registry, alias or namespace into the next test."""
    with ov._LOCK:
        ov._ACTIVE_BASES.clear()
        ov._ACTIVE_PATHS.clear()
    yield
    with ov._LOCK:
        ov._ACTIVE_BASES.clear()
        ov._ACTIVE_PATHS.clear()
    for name in [name for name in sys.modules if name.startswith(ov._NAMESPACE_PREFIX)]:
        sys.modules.pop(name, None)


@pytest.fixture
def base(tmp_path):
    """A stand-in for the already-loaded qualified extension module."""
    module = types.ModuleType("cuda_kernel")
    path = tmp_path / "cuda_kernel_base.so"
    path.write_bytes(b"\x7fELF-qualified-base-placeholder")
    module.__file__ = str(path)
    for name in BASELINE_NAMES:
        setattr(module, name, _op(name))
    return module


def _overlay_module(*, extra=None, drop=()):
    """A stand-in for the experimental overlay module, exposing exactly what a real one would."""
    module = types.ModuleType("overlay_1e")
    for name in ov.OVERLAY_OPS:
        if name not in drop:
            setattr(module, name, _op("overlay:" + name))
    for name, value in (extra or {}).items():
        setattr(module, name, value)
    return module


@pytest.fixture
def loader(monkeypatch):
    """Inject the fake loader and record what it was asked for."""
    state = {"calls": [], "module": None, "error": None}

    def load(full_name, path, parent_name):
        state["calls"].append((full_name, path, parent_name))
        if state["error"] is not None:
            raise state["error"]
        module = state["module"]
        return _overlay_module() if module is None else module

    monkeypatch.setattr(ov, "_MODULE_LOADER", load)
    return state


def _overlay_file(tmp_path, name="overlay_1e.so"):
    path = tmp_path / name
    path.write_bytes(b"\x7fELF-experimental-overlay-placeholder")
    return path


# ------------------------------------------------------------------------------------------------
# The narrow install contract
# ------------------------------------------------------------------------------------------------


def test_install_adds_exactly_the_two_allowlisted_ops_and_nothing_else(base, loader, tmp_path):
    before = dict(vars(base))
    handle = ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    after = dict(vars(base))

    assert set(after) - set(before) == set(ov.OVERLAY_OPS) | {MARKER}
    for name in before:
        assert after[name] is before[name], f"{name} identity changed; the base was not isolated"
    for name in ov.OVERLAY_OPS:
        assert after[name] is getattr(handle.module, name)
    handle.close()


def test_an_extra_attribute_in_the_overlay_never_reaches_the_base(base, loader, tmp_path):
    loader["module"] = _overlay_module(extra={"backdoor_op": _op("backdoor")})
    handle = ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    assert not hasattr(base, "backdoor_op")
    handle.close()
    assert not hasattr(base, "backdoor_op")


def test_base_keeps_its_foreign_attributes_and_baseline_callables(base, loader, tmp_path):
    base.custom_attribute = object()
    identity = base.nvfp4_moe_rows_forward_exact_fast
    handle = ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    assert base.nvfp4_moe_rows_forward_exact_fast is identity
    assert handle.identity["base_has_exact_fast"] is True
    handle.close()
    assert base.custom_attribute is not None
    assert not hasattr(base, COMPACT)


# ------------------------------------------------------------------------------------------------
# Identity and alias evidence
# ------------------------------------------------------------------------------------------------


def test_identity_records_both_hashes_the_alias_and_the_installed_ops(base, loader, tmp_path):
    overlay = _overlay_file(tmp_path)
    handle = ov.install_nvfp4_overlay(base, overlay)
    identity = handle.identity

    assert identity["schema"] == ov.SCHEMA
    assert identity["overlay_sha256"] == hashlib.sha256(overlay.read_bytes()).hexdigest()
    assert identity["base_sha256"] == hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()
    assert identity["overlay_path"] == str(overlay.resolve())
    assert identity["base_path"] == str(Path(base.__file__).resolve())
    assert identity["installed_ops"] == ov.OVERLAY_OPS
    assert identity["replaced_ops"] == ()
    handle.close()


def test_identity_is_json_serialisable_stable_and_not_shared(base, loader, tmp_path):
    handle = ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    first = handle.identity
    json.dumps(first)  # a benchmark line must be able to record it
    assert handle.identity == first
    first["overlay_sha256"] = "tampered"
    assert handle.identity["overlay_sha256"] != "tampered"
    handle.close()


def test_alias_uses_the_pyinit_leaf_and_a_hash_namespace(base, loader, tmp_path):
    overlay = _overlay_file(tmp_path)
    handle = ov.install_nvfp4_overlay(base, overlay)
    alias = handle.identity["module_alias"]
    parent, _, leaf = alias.rpartition(".")

    # CPython requires PyInit_<leaf>; setup.py names this extension `cuda_kernel`.
    assert leaf == "cuda_kernel"
    assert parent == f"{ov._NAMESPACE_PREFIX}{hashlib.sha256(overlay.read_bytes()).hexdigest()[:16]}"
    assert loader["calls"][0][0] == alias
    assert sys.modules[alias] is handle.module
    space = sys.modules[parent]
    assert isinstance(space, types.ModuleType) and space.__path__ == []
    handle.close()


def test_alias_and_namespace_are_removed_on_close(base, loader, tmp_path):
    handle = ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    alias = handle.identity["module_alias"]
    parent = alias.rpartition(".")[0]
    handle.close()
    assert alias not in sys.modules and parent not in sys.modules


def test_active_identity_helper_reports_the_open_overlay(base, loader, tmp_path):
    assert ov.active_identity(base) is None
    handle = ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    assert ov.active_identity(base) == handle.identity
    assert base.__dict__[MARKER] == handle.identity
    handle.close()
    assert ov.active_identity(base) is None
    assert ov.active_identity(object()) is None


# ------------------------------------------------------------------------------------------------
# Restoration
# ------------------------------------------------------------------------------------------------


def test_close_deletes_new_attributes_restores_baseline_and_marks_closed(base, loader, tmp_path):
    baseline = {name: getattr(base, name) for name in BASELINE_NAMES}
    handle = ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    handle.close()

    assert handle.closed is True
    for name in ov.OVERLAY_OPS:
        assert name not in vars(base)
    assert MARKER not in vars(base)
    for name, value in baseline.items():
        assert getattr(base, name) is value


def test_close_is_idempotent(base, loader, tmp_path):
    handle = ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    handle.close()
    handle.close()
    assert handle.closed is True
    assert not hasattr(base, COMPACT)


def test_close_restores_a_pre_existing_allowlisted_attribute(base, loader, tmp_path):
    sentinel = _op("pre-existing-store")
    base.nvfp4_moe_rows_forward_exact_fast_store = sentinel
    handle = ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))

    assert base.nvfp4_moe_rows_forward_exact_fast_store is not sentinel
    assert handle.identity["replaced_ops"] == (STORE,)
    handle.close()
    assert base.nvfp4_moe_rows_forward_exact_fast_store is sentinel


def test_close_restores_a_pre_existing_identity_marker(base, loader, tmp_path):
    previous = {"schema": "older-overlay", "overlay_sha256": "0" * 64}
    setattr(base, MARKER, previous)
    handle = ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    assert base.__dict__[MARKER] == handle.identity
    handle.close()
    assert base.__dict__[MARKER] is previous


def test_context_manager_restores_after_a_raising_body(base, loader, tmp_path):
    with pytest.raises(RuntimeError, match="body failed"):
        with ov.install_nvfp4_overlay(base, _overlay_file(tmp_path)) as handle:
            assert hasattr(base, COMPACT)
            raise RuntimeError("body failed")
    assert handle.closed is True
    assert not hasattr(base, COMPACT) and MARKER not in vars(base)


def test_repeated_install_close_cycles_leave_no_state(base, loader, tmp_path):
    overlay = _overlay_file(tmp_path)
    for _ in range(3):
        handle = ov.install_nvfp4_overlay(base, overlay)
        assert hasattr(base, STORE)
        handle.close()
    assert not any(name.startswith(ov._NAMESPACE_PREFIX) for name in sys.modules)
    assert len(loader["calls"]) == 3
    assert ov._ACTIVE_BASES == {} and ov._ACTIVE_PATHS == {}


# ------------------------------------------------------------------------------------------------
# Refusals and rollback
# ------------------------------------------------------------------------------------------------


def test_missing_overlay_entry_is_refused_and_rolled_back(base, loader, tmp_path):
    loader["module"] = _overlay_module(drop=(STORE,))
    before = dict(vars(base))
    with pytest.raises(ov.NVFP4OverlayError, match=STORE):
        ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    assert {name: value is before[name] for name, value in vars(base).items()} == {
        name: True for name in before
    }
    assert set(vars(base)) == set(before)
    assert not any(name.startswith(ov._NAMESPACE_PREFIX) for name in sys.modules)


def test_loader_error_propagates_unchanged_and_rolls_back(base, loader, tmp_path):
    loader["error"] = ImportError("dynamic module does not define module export function")
    before = set(vars(base))
    with pytest.raises(ImportError, match="PyInit|module export function|dynamic module"):
        ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    assert set(vars(base)) == before
    assert not any(name.startswith(ov._NAMESPACE_PREFIX) for name in sys.modules)


def test_a_loader_returning_a_non_module_is_refused(base, loader, tmp_path):
    loader["module"] = 42
    with pytest.raises(ov.NVFP4OverlayError, match="not a module"):
        ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    assert set(vars(base)) == set(BASELINE_NAMES) | {"__name__", "__doc__", "__package__",
                                                     "__loader__", "__spec__", "__file__"}


@pytest.mark.parametrize("value", [None, 42, "cuda_kernel"])
def test_base_must_be_a_loaded_module(base, loader, tmp_path, value):
    with pytest.raises(ov.NVFP4OverlayError, match="base_module"):
        ov.install_nvfp4_overlay(value, _overlay_file(tmp_path))
    assert ov._ACTIVE_BASES == {}


def test_overlay_path_must_be_an_existing_regular_file(base, loader, tmp_path):
    with pytest.raises(ov.NVFP4OverlayError, match="not an existing file"):
        ov.install_nvfp4_overlay(base, tmp_path / "absent.so")
    with pytest.raises(ov.NVFP4OverlayError, match="not a regular file"):
        ov.install_nvfp4_overlay(base, tmp_path)
    with pytest.raises(ov.NVFP4OverlayError, match="filesystem path"):
        ov.install_nvfp4_overlay(base, 17)


def test_overlay_path_must_be_named_like_an_extension(base, loader, tmp_path):
    wrong = tmp_path / "overlay.txt"
    wrong.write_bytes(b"not an extension")
    with pytest.raises(ov.NVFP4OverlayError, match="compiled extension"):
        ov.install_nvfp4_overlay(base, wrong)


def test_overlay_may_not_be_the_base_file_itself(base, loader):
    with pytest.raises(ov.NVFP4OverlayError, match="base module's own file"):
        ov.install_nvfp4_overlay(base, base.__file__)


def test_second_install_on_one_base_is_refused_until_the_first_closes(base, loader, tmp_path):
    overlay = _overlay_file(tmp_path)
    first = ov.install_nvfp4_overlay(base, overlay)
    with pytest.raises(ov.NVFP4OverlayError, match="already has an open NVFP4 overlay"):
        ov.install_nvfp4_overlay(base, overlay)
    # The refused attempt must not have disturbed the live installation.
    assert first.identity["overlay_sha256"] == hashlib.sha256(overlay.read_bytes()).hexdigest()
    assert base.__dict__[MARKER] == first.identity
    first.close()
    second = ov.install_nvfp4_overlay(base, overlay)
    assert second.closed is False
    second.close()


def test_one_overlay_cannot_be_installed_on_two_bases(base, loader, tmp_path):
    overlay = _overlay_file(tmp_path)
    other = types.ModuleType("cuda_kernel")
    other.__file__ = str(tmp_path / "other_base.so")
    Path(other.__file__).write_bytes(b"\x7fELF-other-base")
    handle = ov.install_nvfp4_overlay(base, overlay)
    with pytest.raises(ov.NVFP4OverlayError, match="already installed on another base"):
        ov.install_nvfp4_overlay(other, overlay)
    assert not hasattr(other, COMPACT)
    handle.close()
    ov.install_nvfp4_overlay(other, overlay).close()


def test_a_refused_install_does_not_break_the_live_overlay(base, loader, tmp_path):
    handle = ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    with pytest.raises(ov.NVFP4OverlayError):
        ov.install_nvfp4_overlay(base, tmp_path / "missing.so")
    assert base.__dict__[MARKER] == handle.identity
    assert hasattr(base, COMPACT)
    handle.close()
    assert not hasattr(base, COMPACT)


# ------------------------------------------------------------------------------------------------
# Real dynamic-loader failure path (no build, no CUDA, no device)
# ------------------------------------------------------------------------------------------------


def test_default_loader_reports_a_non_extension_file(tmp_path):
    """The real loader must not swallow a dlopen failure, and must leave no alias behind."""
    module = types.ModuleType("cuda_kernel")
    base_path = tmp_path / "cuda_kernel_base.so"
    base_path.write_bytes(b"\x7fELF-placeholder")
    module.__file__ = str(base_path)
    bogus = tmp_path / "not_really_an_extension.so"
    bogus.write_bytes(b"this is not a loadable shared object\n")

    with pytest.raises((ImportError, OSError)):
        ov.install_nvfp4_overlay(module, bogus)
    assert not any(name.startswith(ov._NAMESPACE_PREFIX) for name in sys.modules)
    assert ov._ACTIVE_BASES == {} and ov._ACTIVE_PATHS == {}
    assert not hasattr(module, COMPACT) and MARKER not in vars(module)


# ------------------------------------------------------------------------------------------------
# Source-level contracts: helper stays stdlib-only and never rebinds the native module
# ------------------------------------------------------------------------------------------------

_STDLIB_ROOTS = {
    "__future__",
    "hashlib",
    "importlib",
    "os",
    "pathlib",
    "sys",
    "threading",
    "types",
}


def _source() -> str:
    return (Path(ov.__file__)).read_text()


def test_helper_imports_no_torch_or_runtime_package():
    tree = ast.parse(_source())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                roots.add("<relative>")
            elif node.module:
                roots.add(node.module.split(".")[0])
    assert roots <= _STDLIB_ROOTS, f"unexpected import roots: {sorted(roots - _STDLIB_ROOTS)}"


def test_helper_never_rebinds_the_loader_native_module():
    # The qualified module object is supplied by the caller; this helper must not act as a loader.
    # Prose in the module docstring may *name* the loader cache, so this is a code-level check:
    # no identifier, attribute or assignment in the module may touch it.
    tree = ast.parse(_source())
    touched = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            touched.add(node.id)
        elif isinstance(node, ast.Attribute):
            touched.add(node.attr)
    assert "_NATIVE_MOD" not in touched
    assert "load_cuda_kernel" not in touched
    assert "cuda_loader" not in touched


def test_allowlist_is_exactly_the_two_experimental_entries():
    assert ov.OVERLAY_OPS == (
        "nvfp4_moe_rows_forward_exact_fast_compact",
        "nvfp4_moe_rows_forward_exact_fast_store",
    )
    assert ov.LEAF_MODULE == "cuda_kernel"


def test_loader_native_module_is_untouched_across_install_and_close(base, loader, tmp_path):
    cuda_loader = pytest.importorskip("relic_core.kernels.cuda_loader")
    before = cuda_loader._NATIVE_MOD
    handle = ov.install_nvfp4_overlay(base, _overlay_file(tmp_path))
    assert cuda_loader._NATIVE_MOD is before
    handle.close()
    assert cuda_loader._NATIVE_MOD is before


def test_failed_close_stays_reserved_and_can_be_retried(base, loader, tmp_path):
    class RefusingModule(types.ModuleType):
        def __setattr__(self, name, value):
            if name == COMPACT and getattr(self, "_fail_restore", False):
                raise RuntimeError("synthetic restoration refusal")
            super().__setattr__(name, value)

    module = RefusingModule("cuda_kernel")
    vars(module).update(vars(base))
    previous = _op("previous compact")
    setattr(module, COMPACT, previous)
    path = _overlay_file(tmp_path)
    handle = ov.install_nvfp4_overlay(module, path)
    module._fail_restore = True
    with pytest.raises(RuntimeError, match="restoration refusal"):
        handle.close()
    assert not handle.closed
    assert ov.active_identity(module) is not None
    assert handle.identity["module_alias"] in sys.modules
    with pytest.raises(ov.NVFP4OverlayError, match="open NVFP4 overlay"):
        ov.install_nvfp4_overlay(module, path)
    module._fail_restore = False
    handle.close()
    assert handle.closed
    assert getattr(module, COMPACT) is previous
    assert not hasattr(module, STORE)
    assert ov.active_identity(module) is None


def test_changing_the_binary_during_load_rolls_back_before_install(base, monkeypatch, tmp_path):
    path = _overlay_file(tmp_path)
    before = dict(vars(base))

    def changing_loader(alias, source, parent):
        source.write_bytes(b"replacement during synthetic load")
        return _overlay_module()

    monkeypatch.setattr(ov, "_MODULE_LOADER", changing_loader)
    with pytest.raises(ov.NVFP4OverlayError, match="changed while loading"):
        ov.install_nvfp4_overlay(base, path)
    assert vars(base) == before
    assert not ov._ACTIVE_BASES and not ov._ACTIVE_PATHS
    assert not any(name.startswith(ov._NAMESPACE_PREFIX) for name in sys.modules)
