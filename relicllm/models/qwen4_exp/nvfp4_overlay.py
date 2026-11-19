"""Isolate experimental NVFP4 operator entries from the qualified native binary.

Why this module exists
----------------------
The qualified native build (the one whose complete native 512-token greedy output, callbacks and
all eight ranks were verified) is the only build allowed to provide the *existing* operators: dense
BF16 projections, GDN, QSA, the FP8/NVFP4 quantization helpers, the exact-fast rows entry and every
table/dequant binding. A newer experimental build adds two candidate entries for the MoE rows path
(``..._exact_fast_compact`` and ``..._exact_fast_store``) and changes nothing else it is asked to
change -- but its "raw bits pass" from a standalone numpy comparison is not enough to accept the
whole library. Replacing the loaded native module wholesale would silently swap *every* operator,
which is exactly how a benchmark stops answering the question it claims to answer: when a native
first-token divergence appears, there is no way to tell whether the two new entries or any of the
other several hundred rebound functions changed the arithmetic.

So the runtime contract here is deliberately narrow and additive:

* the base module is the already-loaded, qualified extension object -- this module never loads a
  base and never mutates ``relic_core.kernels.cuda_loader._NATIVE_MOD``;
* the experimental shared object is loaded a second time under a unique hash-qualified alias, and
  only the two allowlisted callables are copied onto the base module;
* every other attribute of the base module keeps its identity (``is``), so any divergence caused by
  the two new entries is isolated to the two new entries;
* :meth:`NVFP4OverlayHandle.close` restores the exact previous state: attributes that did not exist
  are deleted again and attributes that existed are put back by value.

Why loading the same sources twice is safe
------------------------------------------
``relic_core/csrc/cuda_kernel.cpp`` only defines free functions inside ``PYBIND11_MODULE``; there is
no ``py::class_`` object registry and no ``TORCH_LIBRARY`` static registration (both checked in the
source, and enforced by a test here). Importing a second build of that source therefore has no
duplicate-registration failure mode. The one hard requirement is CPython's: a dynamically loaded
extension must export ``PyInit_<leaf>`` where ``<leaf>`` is the last dot-separated component of the
module name used with ``importlib``. The project's ``setup.py`` names the extension ``cuda_kernel``
and the vendored ``setup.py`` defines ``TORCH_EXTENSION_NAME=cuda_kernel``, so the aliased module is
always ``<namespace>.cuda_kernel``. A synthetic namespace package is registered in ``sys.modules``
for the prefix so the dotted name is well formed and repeatable.

Lifetime, and what ``close`` does not do
----------------------------------------
The loaded shared object is kept alive by the handle and stays in ``sys.modules`` for the lifetime of
the handle. Closing the handle removes the alias, but this module cannot and does not unload the
shared library: CPython never ``dlclose``s extension modules, so the mapping -- and therefore the
device code any live ``torch.cuda.CUDAGraph`` captured -- remains valid after ``close``.

Two consequences a caller has to respect, both by design rather than as accidents:

* **The caller quiesces before closing.** Keep the handle open until every model and CUDA graph that
  may replay these entries has been released, then close. Nothing here knows about graph pools, so
  ``close`` cannot check that ordering for the caller; closing early is safe for the *device* code
  that is already mapped, but it is the caller's job to make sure nothing issues a new replay after
  the attributes are gone.
* **After ``close`` the two entries are no longer reachable from Python.** The attributes are removed
  again, so a runtime that still asks for one (``getattr(base, "…_store", None)``) sees absence and
  must fail loudly -- it cannot silently call a stale binding. That is the intended failure mode:
  an overlay is a live installation, not a permanent upgrade of the base module.

Overlay identity in a benchmark record
--------------------------------------
This installer deliberately does **not** mutate process environment or any global configuration: an
import-time side effect would make it ambiguous which build a run used. Recording the overlay is the
caller's responsibility. The recommended contract -- because a graph/request cache key must include
the overlay identity and not only the math flags -- is for the launcher to export
``POCKETLLM_QWEN4_NVFP4_OVERLAY_SHA256`` with ``handle.identity["overlay_sha256"]`` before the
runtime loads, and to close the handle after the graph pool and model are released. The marker
attribute written here is for recording, not for behaviour.

Failure policy
--------------
Every refusal and every failure raises ``NVFP4OverlayError`` (except that a load failure keeps its
original exception type, e.g. ``ImportError`` from the dynamic loader) after rolling back every
attribute, marker and ``sys.modules`` entry this call created. There is no silent fallback: an
overlay that cannot be installed leaves the process exactly as it was.

Nothing here imports ``torch`` or ``relicllm``. The base module is supplied already loaded, so this
helper is safe to import from a launcher before a runtime is constructed.
"""

from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import os
import sys
import threading
import types
from pathlib import Path

__all__ = [
    "OVERLAY_OPS",
    "NVFP4OverlayError",
    "NVFP4OverlayHandle",
    "active_identity",
    "install_nvfp4_overlay",
]

#: Schema tag recorded in ``handle.identity`` and on the base module's marker attribute.
SCHEMA = "qwen4-nvfp4-overlay-v1"

#: The only operators an overlay may add. Both must exist in the overlay module; nothing else is
#: ever copied onto the base module.
OVERLAY_OPS = (
    "nvfp4_moe_rows_forward_exact_fast_compact",
    "nvfp4_moe_rows_forward_exact_fast_store",
)

#: Leaf name of the aliased extension module. It must equal the `setup.py` extension name, because
#: CPython resolves the loader entry point as ``PyInit_<last component of the module name>``.
LEAF_MODULE = "cuda_kernel"

#: Metadata attribute written on the base module so a benchmark record can cite which overlay ran.
#: It is not consulted by any operator and carries no behaviour.
MARKER = "_qwen4_nvfp4_overlay_identity"

#: Prefix of the synthetic namespace package holding the alias.
_NAMESPACE_PREFIX = "_qwen4_nvfp4_overlay_"

#: A qualified baseline entry used only to describe the base module in ``identity``.
_BASELINE_OP = "nvfp4_moe_rows_forward_exact_fast"

_CHUNK = 1 << 20

#: Serialises registry bookkeeping and rollback. An install can load a shared object, which is slow,
#: but overlays are installed at start-up and never on a request path, so holding the lock for the
#: whole call keeps the "is this base already overlayed" decision and its mutation atomic.
_LOCK = threading.RLock()

#: Active installations, keyed by ``id(base)`` and by resolved overlay path. Handles are kept
#: strongly: a handle owns the base module, so the id cannot be recycled while it is registered, and
#: a caller that forgets ``close`` leaks visibly instead of corrupting a later install.
_ACTIVE_BASES: dict = {}
_ACTIVE_PATHS: dict = {}

#: Test-only injection point: ``(full_name, path, parent_name) -> module``. Left ``None`` in
#: production, where the real ``importlib`` extension loader is used.
_MODULE_LOADER = None

_ABSENT = object()


class NVFP4OverlayError(RuntimeError):
    """An NVFP4 operator overlay was refused, or could not be installed."""


# ------------------------------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """The content hash recorded in ``identity``; no device work and no import side effects."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolved_file(value, what: str) -> Path:
    """``value`` as an existing regular file, symlinks resolved, or a clear refusal."""
    try:
        candidate = Path(os.fspath(value))
    except TypeError as exc:
        raise NVFP4OverlayError(
            f"{what} must be a filesystem path, got {type(value).__name__}"
        ) from exc
    try:
        resolved = candidate.expanduser().resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise NVFP4OverlayError(f"{what} is not an existing file: {candidate}") from exc
    if not resolved.is_file():
        raise NVFP4OverlayError(f"{what} is not a regular file: {resolved}")
    return resolved


def _require_extension_suffix(path: Path, what: str) -> None:
    """Refuse a path that is not named like something ``importlib`` can load as an extension."""
    suffixes = tuple(importlib.machinery.EXTENSION_SUFFIXES)
    if not path.name.endswith(suffixes):
        raise NVFP4OverlayError(
            f"{what} must be a compiled extension ({', '.join(suffixes)}); got {path.name!r}"
        )


def _same_file(left, right) -> bool:
    """Whether two paths name the same file, tolerating a path that no longer resolves."""
    try:
        return Path(os.fspath(left)).resolve() == Path(os.fspath(right)).resolve()
    except (OSError, RuntimeError, TypeError):
        return False


def _base_source(base) -> Path | None:
    """Where the already-loaded base extension lives, when it can be determined."""
    raw = getattr(base, "__file__", None)
    if not raw:
        return None
    try:
        return Path(os.fspath(raw)).resolve()
    except (OSError, RuntimeError, TypeError):
        return None


def _require_base(base):
    """The base must be the already-loaded qualified extension module object."""
    if base is None:
        raise NVFP4OverlayError(
            "base_module is None; pass the already-loaded qualified native extension module "
            "(the object the loader caches), not a path"
        )
    if not isinstance(base, types.ModuleType):
        raise NVFP4OverlayError(
            "base_module must be a loaded extension module, got "
            f"{type(base).__name__}; an overlay copies two callables onto an existing module and "
            "does not create one"
        )
    return base


def _ensure_namespace(parent_name: str) -> bool:
    """Register a synthetic namespace package for the alias; return whether this call created it."""
    existing = sys.modules.get(parent_name)
    if existing is not None:
        if isinstance(existing, types.ModuleType) and hasattr(existing, "__path__"):
            return False
        raise NVFP4OverlayError(
            f"the overlay namespace {parent_name!r} is already occupied by a foreign object; "
            "refusing to replace it"
        )
    package = types.ModuleType(parent_name)
    package.__doc__ = "Synthetic namespace for an aliased NVFP4 overlay extension."
    package.__package__ = parent_name
    package.__path__ = []
    sys.modules[parent_name] = package
    return True


def _default_module_loader(full_name: str, path: Path, parent_name: str):
    """Load ``path`` as the extension module ``full_name`` with the real dynamic loader.

    The dotted name is what makes ``PyInit_cuda_kernel`` the required entry point, so this is the
    one place that cannot be replaced by a plain ``module_name`` import. Errors are intentionally
    not caught: an import failure is the truthful answer and the caller rolls back around it.
    """
    existing = sys.modules.get(full_name)
    if isinstance(existing, types.ModuleType):
        origin = getattr(existing, "__file__", None)
        if origin and _same_file(origin, path):
            return existing
        # A stale binding for another build under the same alias: drop it before re-loading.
        sys.modules.pop(full_name, None)
    spec = importlib.util.spec_from_file_location(full_name, str(path))
    if spec is None or spec.loader is None:
        raise NVFP4OverlayError(
            f"cannot build an extension module spec for {path}; the file is not a loadable "
            "CPython extension"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_overlay_module(full_name: str, path: Path, parent_name: str):
    """Load through the test injection point when present, else the real extension loader."""
    loader = _MODULE_LOADER
    if loader is None:
        return _default_module_loader(full_name, path, parent_name)
    return loader(full_name, path, parent_name)


def _reports_attributes(module) -> bool:
    """A loaded overlay must expose named attributes; a bare object is a broken loader."""
    return isinstance(module, types.ModuleType) or hasattr(module, "__dict__")


def _restore(target, name: str, previous, had: bool):
    """Build the undo action for one attribute this install overwrote or created."""

    def restore() -> None:
        if had:
            setattr(target, name, previous)
        elif name in vars(target):
            delattr(target, name)

    return restore


def _refuse_active(base, path: Path) -> None:
    """Refuse a second concurrent overlay on one base, or one overlay bound to two bases."""
    current = _ACTIVE_BASES.get(id(base))
    if current is not None and not current.closed:
        raise NVFP4OverlayError(
            "this base module already has an open NVFP4 overlay; close that handle (after the "
            "graphs that replay its entries are released) before installing another"
        )
    other = _ACTIVE_PATHS.get(str(path))
    if other is not None and not other.closed:
        raise NVFP4OverlayError(
            f"the overlay {path} is already installed on another base module; it cannot be "
            "isolated into two bases at once"
        )


# ------------------------------------------------------------------------------------------------
# Handle
# ------------------------------------------------------------------------------------------------


class NVFP4OverlayHandle:
    """One installed overlay: the loaded entries, their evidence, and how to undo the install.

    Keep the handle open for as long as any CUDA graph may replay the two overlay entries; the
    caller owns that ordering. ``close`` is idempotent and never unloads the shared object.
    """

    __slots__ = (
        "_base",
        "_identity",
        "_module",
        "_overlay_path",
        "_parent_name",
        "_alias",
        "_previous",
        "_created_namespace",
        "_closed",
    )

    def __init__(
        self,
        *,
        base,
        identity: dict,
        module,
        overlay_path: Path,
        parent_name: str,
        alias: str,
        previous: dict,
        created_namespace: bool,
    ) -> None:
        self._base = base
        self._identity = dict(identity)
        self._module = module
        self._overlay_path = overlay_path
        self._parent_name = parent_name
        self._alias = alias
        self._previous = previous
        self._created_namespace = created_namespace
        self._closed = False

    # -- evidence ---------------------------------------------------------------------------

    @property
    def identity(self) -> dict:
        """A fresh, JSON-serialisable record of what this install did. Mutation is not shared."""
        return dict(self._identity)

    @property
    def module(self):
        """The loaded overlay module. Kept alive by this handle."""
        return self._module

    @property
    def overlay_path(self) -> Path:
        return self._overlay_path

    @property
    def closed(self) -> bool:
        return self._closed

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"NVFP4OverlayHandle(alias={self._alias!r}, {state})"

    # -- lifetime ---------------------------------------------------------------------------

    def close(self) -> None:
        """Restore the base module exactly as it was, then drop the alias. Idempotent.

        Attributes are restored *before* the ``sys.modules`` entries are removed, so no observer can
        see a base module whose overlay attributes are gone but whose alias is still registered.
        The shared object stays mapped: removal only drops this module binding, and CPython never
        unloads an extension, so already-captured graphs keep their device code.

        The caller must quiesce first: release the model and the graph pool that may replay these
        entries, then close. After this returns the two entries are no longer reachable from Python,
        so a later ``getattr(base, "…_store", None)`` is ``None`` and the runtime must refuse loudly
        rather than call a stale binding.
        """
        # Keep the installation reserved until restoration has actually completed. If a module
        # subclass refuses an attribute change, close remains retryable instead of advertising a
        # completed teardown with residual bindings still installed.
        with _LOCK:
            if self._closed:
                return
            base = self._base
            for name in reversed(OVERLAY_OPS):
                previous, had = self._previous[name]
                if had:
                    setattr(base, name, previous)
                elif name in vars(base):
                    delattr(base, name)
            marker_previous, marker_had = self._previous[MARKER]
            if marker_had:
                setattr(base, MARKER, marker_previous)
            elif MARKER in vars(base):
                delattr(base, MARKER)
            if _ACTIVE_BASES.get(id(base)) is self:
                _ACTIVE_BASES.pop(id(base), None)
            if _ACTIVE_PATHS.get(str(self._overlay_path)) is self:
                _ACTIVE_PATHS.pop(str(self._overlay_path), None)
            if sys.modules.get(self._alias) is self._module:
                sys.modules.pop(self._alias, None)
            if self._created_namespace:
                space = sys.modules.get(self._parent_name)
                if isinstance(space, types.ModuleType) and not getattr(space, "__dict__", {}).get(
                    "__file__"
                ):
                    sys.modules.pop(self._parent_name, None)
            self._closed = True

    def __enter__(self) -> "NVFP4OverlayHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False


# ------------------------------------------------------------------------------------------------
# Install
# ------------------------------------------------------------------------------------------------


def install_nvfp4_overlay(base_module, overlay_path) -> NVFP4OverlayHandle:
    """Copy the two allowlisted experimental entries onto an already-loaded qualified module.

    ``base_module`` is the qualified extension module object the runtime already holds, and
    ``overlay_path`` is the experimental shared object. Every other attribute of ``base_module`` is
    left untouched, including its identity, so a divergence can be attributed to these two entries.

    Returns an open :class:`NVFP4OverlayHandle`; use it as a context manager or call ``close`` after
    the graphs that replay the entries have been released. Raises ``NVFP4OverlayError`` for a
    refused install; a dynamic-loader failure keeps its own exception type.
    """
    base = _require_base(base_module)
    path = _resolved_file(overlay_path, "overlay_path")
    _require_extension_suffix(path, "overlay_path")
    base_path = _base_source(base)
    if base_path is not None and _same_file(base_path, path):
        raise NVFP4OverlayError(
            f"overlay_path is the base module's own file ({path}); an overlay must be a separate "
            "build or the two entries would not be isolated from the existing ones"
        )

    overlay_sha = _sha256_file(path)
    base_sha = _sha256_file(base_path) if base_path is not None and base_path.is_file() else None
    parent_name = f"{_NAMESPACE_PREFIX}{overlay_sha[:16]}"
    alias = f"{parent_name}.{LEAF_MODULE}"

    with _LOCK:
        _refuse_active(base, path)
        undo: list = []
        try:
            created_namespace = _ensure_namespace(parent_name)
            if created_namespace:
                undo.append(lambda: sys.modules.pop(parent_name, None)
                            if isinstance(sys.modules.get(parent_name), types.ModuleType)
                            and sys.modules[parent_name].__dict__.get("__file__") is None
                            else None)

            module = _load_overlay_module(alias, path, parent_name)
            if not _reports_attributes(module):
                raise NVFP4OverlayError(
                    f"loading {path} produced {type(module).__name__}, not a module with attributes"
                )
            if sys.modules.get(alias) is not module:
                sys.modules[alias] = module
                undo.append(lambda: sys.modules.pop(alias, None)
                            if sys.modules.get(alias) is module
                            else None)

            if _sha256_file(path) != overlay_sha:
                raise NVFP4OverlayError(
                    f"the overlay {path} changed while loading; native snapshots must stay immutable"
                )
            missing = [name for name in OVERLAY_OPS if not callable(getattr(module, name, None))]
            if missing:
                raise NVFP4OverlayError(
                    f"the overlay {path} does not define {', '.join(missing)}; it is not the "
                    "experimental build this installer accepts"
                )

            previous: dict = {}
            replaced: list = []
            for name in OVERLAY_OPS:
                had = name in vars(base)
                previous[name] = (vars(base).get(name, _ABSENT), had)
                if had:
                    replaced.append(name)
                setattr(base, name, getattr(module, name))
                undo.append(_restore(base, name, previous[name][0], had))

            marker_had = MARKER in vars(base)
            marker_previous = vars(base).get(MARKER, _ABSENT)
            previous[MARKER] = (marker_previous, marker_had)
            identity = {
                "schema": SCHEMA,
                "base_path": None if base_path is None else str(base_path),
                "base_sha256": base_sha,
                "overlay_path": str(path),
                "overlay_sha256": overlay_sha,
                "module_alias": alias,
                "installed_ops": tuple(OVERLAY_OPS),
                "replaced_ops": tuple(replaced),
                "base_has_exact_fast": callable(getattr(base, _BASELINE_OP, None)),
            }
            setattr(base, MARKER, dict(identity))
            undo.append(_restore(base, MARKER, marker_previous, marker_had))

            handle = NVFP4OverlayHandle(
                base=base,
                identity=identity,
                module=module,
                overlay_path=path,
                parent_name=parent_name,
                alias=alias,
                previous=previous,
                created_namespace=created_namespace,
            )
            _ACTIVE_BASES[id(base)] = handle
            undo.append(lambda: _ACTIVE_BASES.pop(id(base), None)
                        if _ACTIVE_BASES.get(id(base)) is handle
                        else None)
            _ACTIVE_PATHS[str(path)] = handle
            undo.append(lambda: _ACTIVE_PATHS.pop(str(path), None)
                        if _ACTIVE_PATHS.get(str(path)) is handle
                        else None)
            return handle
        except BaseException:
            # Roll back in reverse so a partial install leaves no attribute, marker or module entry.
            # A rollback failure must never replace the error that caused it.
            for action in reversed(undo):
                try:
                    action()
                except Exception:
                    pass
            raise


def active_identity(base_module) -> dict | None:
    """The identity an open overlay recorded on ``base_module``, or ``None`` when not overlayed.

    A benchmark line can record this to show which overlay (if any) the measured build used.
    """
    if not isinstance(base_module, types.ModuleType):
        return None
    value = vars(base_module).get(MARKER)
    return dict(value) if isinstance(value, dict) else None