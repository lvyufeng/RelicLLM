from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "relicllm"

REMOVED_IMPORT_PREFIXES = (
    "relicllm.gguf",
    "relicllm.models.moe",
    "relicllm.runtime.deepseek_v4",
    "relicllm.runtime.moe",
    "relicllm.moe",
    "relicllm.moe_model",
)

RUNTIME_FORBIDDEN_IMPORT_PREFIXES = (
    "safetensors",
    "relicllm.loader",
    "relicllm.models",
)

LOADER_FORBIDDEN_IMPORT_PREFIXES = (
    "relicllm.models",
    "relicllm.runtime",
)

COMPONENTS_MOE_ALLOWED_MODEL_IMPORTS = {
    # Exactly the specs `relicllm/components/moe/registry.py::_init_specs` imports to build its
    # `general.architecture` table. GLM-DSA was registered there and left out of this set, so the
    # test failed on a registry that is doing the one thing it is allowed to do.
    "relicllm.models.deepseek_v4.spec",
    "relicllm.models.glm_dsa.spec",
    "relicllm.models.minimax_m2.spec",
}

COMPONENTS_MOE_FORBIDDEN_MODEL_MODULES = (
    ".generation",
    ".loader",
    ".moe_runtime",
    ".moe_server",
    ".partition",
    ".runtime",
)

#: The package stack, lowest layer first. A module may import anything in its own layer or below;
#: an import of a strictly higher layer is the shape this test exists to catch. The stack is written
#: the way the runtime is built — each layer may only name the ones under it — so the number is a
#: contract, not a preference. `docs/architecture/package_layers.md` is the page behind it.
LAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("device", ("runtime",)),
    ("primitives", ("loader", "encoding")),
    ("protocol", ("api", "protocol")),
    ("support", ("components",)),
    ("models", ("models",)),
    ("adapters", ("backends",)),
    ("hosts", ("cli", "server", "bench", "triage")),
)

#: Layer index by package directory, plus the top-level modules (`relicllm/engine.py`,
#: `relicllm/choices.py`, …), which sit with the hosts — they are entry points, not a layer.
_LAYER_INDEX: dict[str, int] = {}
for _index, (_name, _packages) in enumerate(LAYERS):
    for _package in _packages:
        _LAYER_INDEX[_package] = _index
_ROOT_LAYER = len(LAYERS) - 1

#: Pairs that are allowed to point upward, each with the reason it is not a leak. Both are the
#: sanctioned *bidirectional declarations* upstream draws the same way (vLLM's `KVCacheSpec`,
#: SGLang's `pool_configurator`): a small declaration flows down, a discovery or a spec flows back
#: up. They are listed here so the exception is visible in one place rather than implicit in a
#: passing test; the finer rule for the first one lives in
#: `test_components_moe_only_imports_model_specs_for_registry_discovery`.
ALLOWED_UPWARD: dict[tuple[str, str], str] = {
    ("components", "models"): (
        "registry.py discovers MoE specs by importing them; the allowed set is pinned below."
    ),
}


def _python_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*.py")
        if "__pycache__" not in path.parts
    )


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _starts_with_any(module: str, prefixes: tuple[str, ...]) -> bool:
    return any(module == prefix or module.startswith(prefix + ".") for prefix in prefixes)


def _format_violations(violations: list[tuple[Path, str]]) -> str:
    return "\n".join(f"{path.relative_to(REPO_ROOT)} imports {module}" for path, module in violations)


def _package_of(path: Path) -> str:
    """The layer-bearing package a source file belongs to, or `""` for a top-level module."""
    relative = path.relative_to(PACKAGE_ROOT)
    if len(relative.parts) == 1:
        return ""
    return relative.parts[0]


def _layer_of(package: str) -> int:
    return _LAYER_INDEX.get(package, _ROOT_LAYER)


def test_removed_namespaces_are_not_imported_from_source() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(PACKAGE_ROOT):
        for module in _imported_modules(path):
            if _starts_with_any(module, REMOVED_IMPORT_PREFIXES):
                violations.append((path, module))

    assert not violations, _format_violations(violations)


def test_imports_only_reach_downward_through_the_package_stack() -> None:
    """No module imports a package that sits above it in :data:`LAYERS`."""
    violations: list[tuple[Path, str]] = []
    for path in _python_files(PACKAGE_ROOT):
        source = _package_of(path)
        if path.name == "__init__.py" and source == "":
            # `relicllm/__init__.py` is the package's own front door; it is a host by construction.
            source = ""
        for module in _imported_modules(path):
            if not module.startswith("relicllm."):
                continue
            parts = module.split(".")
            target = parts[1] if len(parts) > 1 else ""
            if target == source or target not in _LAYER_INDEX:
                continue
            if _layer_of(target) <= _layer_of(source):
                continue
            if (source, target) in ALLOWED_UPWARD:
                continue
            violations.append((path, module))

    assert not violations, (
        "these imports reach upward in the layer stack (`docs/architecture/package_layers.md`):\n"
        + _format_violations(violations)
    )


def test_runtime_stays_model_and_checkpoint_format_agnostic() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(PACKAGE_ROOT / "runtime"):
        for module in _imported_modules(path):
            if _starts_with_any(module, RUNTIME_FORBIDDEN_IMPORT_PREFIXES):
                violations.append((path, module))

    assert not violations, _format_violations(violations)


def test_loader_does_not_depend_on_runtime_or_model_packages() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(PACKAGE_ROOT / "loader"):
        for module in _imported_modules(path):
            if _starts_with_any(module, LOADER_FORBIDDEN_IMPORT_PREFIXES):
                violations.append((path, module))

    assert not violations, _format_violations(violations)


def test_components_moe_only_imports_model_specs_for_registry_discovery() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(PACKAGE_ROOT / "components" / "moe"):
        for module in _imported_modules(path):
            if not _starts_with_any(module, ("relicllm.models",)):
                continue
            if path.name == "registry.py" and module in COMPONENTS_MOE_ALLOWED_MODEL_IMPORTS:
                continue
            violations.append((path, module))

    assert not violations, _format_violations(violations)


def test_components_moe_does_not_import_model_runtime_loader_or_servers() -> None:
    violations: list[tuple[Path, str]] = []
    for path in _python_files(PACKAGE_ROOT / "components" / "moe"):
        for module in _imported_modules(path):
            if module.startswith("relicllm.models") and module.endswith(COMPONENTS_MOE_FORBIDDEN_MODEL_MODULES):
                violations.append((path, module))

    assert not violations, _format_violations(violations)