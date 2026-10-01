"""Where the answer to "can this runtime do X" lives, and the invariants that keep it there.

Capability used to be computed in six ``capabilities()`` methods, the refusal of a checkpoint was
written twice per adapter -- once as a predicate for ``auto`` and once as a raise for an explicit
``--backend`` -- three adapters carried their own copy of the same ignored-options set, and
``supports_prefix_caching`` meant "the store exists" on two runtimes and "the byte budget is
positive" on a third. Every test here pins one of those, and the source scans are deliberate: the
defect was never a wrong value, it was a second place to put one.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from relicllm.api import ConfigurationError, EngineArgs, UnsupportedFeatureError
from relicllm.backends import capabilities as declared
from relicllm.backends.capabilities import (
    IGNORED_OPTIONS,
    RUNTIMES,
    Verdict,
    declared_capabilities,
    identify,
    route,
)
from relicllm.backends.factory import select_backend

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKENDS_ROOT = REPO_ROOT / "relicllm" / "backends"


# --------------------------------------------------------------------------------------------------
# one place
# --------------------------------------------------------------------------------------------------


def test_every_backend_name_has_a_declaration() -> None:
    from relicllm.api.types import _BACKENDS

    assert set(RUNTIMES) == set(_BACKENDS) - {"auto"}
    # And `auto` is a routing rule over them rather than a sixth runtime.
    assert set(declared.AUTO_ORDER) == set(RUNTIMES)
    assert declared.AUTO_ORDER[-1] == "torch"


def test_the_wire_type_is_constructed_in_one_module() -> None:
    """`BackendCapabilities(...)` outside the declaration module is a runtime answering the
    question for itself again -- which is exactly how the six answers drifted apart."""
    offenders = sorted(
        path.relative_to(REPO_ROOT).as_posix()
        for path in BACKENDS_ROOT.glob("*.py")
        if path.name != "capabilities.py" and re.search(r"\bBackendCapabilities\(", path.read_text())
    )

    assert not offenders, (
        "these adapters build a BackendCapabilities themselves instead of reading the "
        f"declaration: {offenders}"
    )


def test_the_ignored_options_have_one_definition() -> None:
    """Three adapters used to carry a frozenset literal with the same keys. Accepting an option and
    ignoring it is a deliberate list, and there is one of them -- so the scan is for a second
    *literal*, not for a second mention: importing the shared set is the fix, not the bug.

    The pinned set is the second half of that. A key added here is a key every *model* options
    parser stops refusing, so it is the one place a launch option can become universally accepted
    by accident. The four the CLI and the supervisor fill in, plus the two batch flags the factory
    reads, are the whole list; nothing else belongs."""
    keys = sorted(IGNORED_OPTIONS)
    literal = re.compile(r"frozenset\(\s*\{[^}]*\}\)", re.S)
    offenders = []
    for path in sorted(BACKENDS_ROOT.glob("*.py")):
        if path.name == "capabilities.py":
            continue
        for found in literal.findall(path.read_text()):
            if all(f'"{key}"' in found for key in keys):
                offenders.append(path.relative_to(REPO_ROOT).as_posix())

    assert not offenders, f"an adapter writes the ignored-options set out again: {offenders}"
    assert set(keys) == {
        # Filled in by the CLI on every serve command, and by the supervisor for a sharded one.
        "engine_kind",
        "routed_experts_device",
        "pd_mode",
        "nccl_id_path",
        # Read by the factory to refuse a batch path no runtime here owns. They are in this set so
        # a launch can name them without the model option parser refusing them, the same treatment
        # as the four.
        "enable_batching",
        "scheduler_timeout_ms",
    }


def test_the_rejection_and_the_routing_predicate_are_the_same_call() -> None:
    """The two functions that could disagree are now one, so asking twice has to give the same
    answer twice -- and the third verdict, "no evidence", is what makes that possible."""
    args = EngineArgs(model="/nonexistent/checkpoint", backend="v41")

    first = identify("v41", args)
    second = identify("v41", args)

    assert first == second
    assert first.verdict is Verdict.UNKNOWN
    # No evidence is not a refusal, which is why an explicit `--backend` on a path that cannot be
    # read reaches the adapter's own loader error instead of a routing complaint here.
    assert declared.refusal("v41", args) == ""


# --------------------------------------------------------------------------------------------------
# prefix caching, one meaning
# --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(RUNTIMES))
def test_the_prefix_capability_is_the_declaration_anded_with_the_gate(name: str) -> None:
    """One question on every runtime: will a repeated prefix be resumed in this configuration?"""
    open_gate = declared_capabilities(name, reads_prefix_cache=True)
    closed_gate = declared_capabilities(name, reads_prefix_cache=False)

    assert open_gate.supports_prefix_caching is RUNTIMES[name].reads_prefix_cache
    assert closed_gate.supports_prefix_caching is False


def _write_config(directory: Path, payload: dict) -> None:
    (directory / "config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_a_gguf_is_evidence_against_a_safetensors_only_runtime(tmp_path) -> None:
    _write_config(tmp_path, {"model_type": "deepseek_v41"})
    (tmp_path / "model.gguf").write_bytes(b"GGUF")

    verdict = identify("v41", EngineArgs(model=str(tmp_path), backend="v41", model_format="gguf"))

    assert verdict.verdict is Verdict.REFUSED
    assert "GGUF" in verdict.reason


def test_a_foreign_config_is_evidence_against_and_no_config_is_neither(tmp_path) -> None:
    """The distinction the two-callables-per-adapter design did not have: a checkpoint that
    presents no evidence is not one that has been disproved."""
    _write_config(tmp_path, {"model_type": "qwen3_5"})
    foreign = identify("v41", EngineArgs(model=str(tmp_path), backend="v41"))

    assert foreign.verdict is Verdict.REFUSED
    assert "DeepSeek-V4.1-Flash" in foreign.reason

    missing = identify("v41", EngineArgs(model=str(tmp_path / "nothing"), backend="v41"))
    assert missing.verdict is Verdict.UNKNOWN


def test_an_unreadable_checkpoint_reaches_the_adapter_instead_of_the_router(tmp_path) -> None:
    """`UNKNOWN` is not `READ`: it must not be routed here by `auto`, and it must not be refused
    for an explicit backend either."""
    args = EngineArgs(model=str(tmp_path / "nothing"), backend="auto")

    assert identify("v41", args).verdict is Verdict.UNKNOWN
    assert route(args) == "torch"
    assert select_backend(EngineArgs(model=str(tmp_path / "nothing"), backend="v41")) == "v41"


def test_xing4_refuses_safetensors_at_selection_rather_than_at_the_loader(tmp_path) -> None:
    """Its weights come from the GGUF and the safetensors directory beside it supplies the tokenizer
    and the config, so `--model-format safetensors` was never something this runtime could serve.
    The auto path had always refused it; the explicit path used to let it through selection and
    fail later with "no .gguf file at or under ..."."""
    _write_config(tmp_path, {"model_type": "xing4_0"})
    args = EngineArgs(model=str(tmp_path), backend="xing4", model_format="safetensors")

    assert identify("xing4", args).verdict is Verdict.REFUSED
    with pytest.raises(UnsupportedFeatureError, match="weights from the GGUF"):
        select_backend(args)


def test_xing4_tells_its_two_causes_apart_when_both_apply(tmp_path) -> None:
    """A checkpoint that is not a Xing4 export *and* was asked for with the wrong format has two
    reasons to be refused, and the answer is the one about the checkpoint: "wrong flag" would send
    the caller to fix `--model-format` on a model this runtime cannot serve either way."""
    _write_config(tmp_path, {"model_type": "qwen3_5"})
    args = EngineArgs(model=str(tmp_path), backend="xing4", model_format="safetensors")

    assert "Xing4.0-29B-A4B checkpoints only" in identify("xing4", args).reason


def test_a_header_that_was_read_and_disagrees_is_evidence_against(tmp_path) -> None:
    """The distinction the empty-string return could not express: a `.gguf` whose header parsed and
    named another architecture is refused *on what it named*, while one that could not be read at
    all is not a refusal -- it reaches the adapter, whose loader error names the file it choked
    on. Both used to look identical, so the explicit path refused neither."""
    from tests.gguf_test_utils import write_gguf

    readable = tmp_path / "elsewhere.gguf"
    write_gguf(readable, metadata={"general.architecture": "qwen35"}, tensors=[])
    refused_on_the_header = EngineArgs(model=str(readable), backend="xing4")
    verdict = identify("xing4", refused_on_the_header)

    assert verdict.verdict is Verdict.REFUSED
    assert "'qwen35'" in verdict.reason

    unreadable = tmp_path / "not-a-gguf.gguf"
    unreadable.write_bytes(b"not a gguf at all")
    assert identify("xing4", EngineArgs(model=str(unreadable), backend="xing4")).verdict is Verdict.UNKNOWN


def test_the_generic_runtime_identifies_everything_and_is_asked_last() -> None:
    """The loop has no separate fallback: `torch` is the last candidate and the answer to "nothing
    else claimed it", so `route` cannot run off the end."""
    assert identify("torch", EngineArgs(model="/nonexistent", backend="auto")).routes_here
    assert declared.AUTO_ORDER.index("torch") == len(declared.AUTO_ORDER) - 1


def test_the_capability_refusal_does_not_shadow_the_checkpoint_refusal() -> None:
    """Two reasons to refuse, and the checkpoint's is the more specific one: it says which backend
    to use instead, and it fires first."""
    with pytest.raises(UnsupportedFeatureError, match="safetensors shards only"):
        select_backend(
            EngineArgs(model="missing", backend="v41", model_format="gguf", max_batch_size=8)
        )


def test_a_contradiction_in_the_arguments_is_still_a_configuration_error() -> None:
    """The refusal added here is about a runtime's declaration; the argument check stays where it
    was, so the two are not reported as each other."""
    with pytest.raises(ConfigurationError, match="serialized session runs one request"):
        EngineArgs(model="m", backend="mimo", max_batch_size=2, enable_batching=False)
