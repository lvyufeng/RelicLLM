"""Which accelerator this process runs on, and what the code above it is allowed to assume.

`--device` was accepted, validated, forwarded to every rank and then read by nothing: `auto` was
never resolved, and `RuntimeCapabilities.devices` -- the set each runtime declares it can run on --
was compared against nothing, so `--backend mimo --device cpu` was accepted by a runtime whose own
declaration says `("cuda",)`.

These tests are about the plane that answers both, and they are written so that the Ascend arm is
reachable **on a CUDA box**. That is not a convenience: this repository's Ascend support has to be
developable somewhere, and the only machine that can run this suite has four RTX 2080 Ti and no
CAN N. So the probe takes its answers as arguments and the index resolution takes a callable, and
neither proof needs a device of the kind it is proving.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from relicllm.api import ConfigurationError, EngineArgs, UnsupportedFeatureError
from relicllm.backends.factory import select_backend
from relicllm.cli import DEVICE_PLATFORMS
from relicllm.runtime import device as plane


def _torch():
    """torch, or a skip. Only the tests that build a `torch.device` need it."""
    return pytest.importorskip("torch")


def _args(**overrides) -> EngineArgs:
    base = dict(model="checkpoint", backend="torch")
    base.update(overrides)
    return EngineArgs(**base)


def _accelerator(platform: str) -> plane.Accelerator:
    """The Accelerator for `platform`, built the way the probe builds it."""
    if platform == "cuda":
        return plane.probe_accelerator(cuda_available=True, npu_available=False)
    if platform == "ascend":
        return plane.probe_accelerator(cuda_available=False, npu_available=True)
    return plane.probe_accelerator(cuda_available=False, npu_available=False)


# ------------------------------------------------------------------------ the two tables

#: What each platform is called at each layer below this module. Written as data rather than as
#: three separate assertions because the point is that the three agree: a run that picked the Ascend
#: device type and the NCCL collective has a device it cannot address, and the failure arrives
#: inside a collective rather than at the call that chose it.
_THE_TABLE = {
    "cuda": ("cuda", "nccl"),
    "ascend": ("npu", "hccl"),
    "cpu": ("cpu", "gloo"),
}


def test_every_platform_names_its_device_type_and_its_collective() -> None:
    """The 910's two answers, which are the two a port gets wrong first.

    `ascend` is `npu` to torch because that is the name `torch_npu` gives the PrivateUse1 backend,
    and `hccl` is CANN's collective library where CUDA has NCCL. Both are strings, so both are
    checkable on a host that has never seen either.
    """
    for platform, (device_type, collective) in _THE_TABLE.items():
        assert plane.torch_device_type(platform) == device_type
        assert plane.process_group_backend(platform) == collective


def test_a_platform_that_is_not_one_is_refused_by_name() -> None:
    """`tpu` is a real device type in neither table, and the message says what the set is.

    This is the same refusal `EngineArgs` makes, made again for the callers that reach the tables
    without going through the flag -- a model module resolving its own device, say.
    """
    for bad in ("tpu", "xpu", "", "auto!"):
        with pytest.raises(plane.DeviceError, match="is not a platform"):
            plane.torch_device_type(bad)
        with pytest.raises(plane.DeviceError, match="is not a platform"):
            plane.process_group_backend(bad)


def test_the_platform_set_is_spelled_once() -> None:
    """`--device`'s `--help`, its parser and its validator read one tuple.

    They used to read two literals that had to agree -- `relicllm/cli.py` rendered the set and
    `relicllm/api/types.py` validated the value that rendering had just offered -- with the device
    plane about to add a third.
    """
    assert DEVICE_PLATFORMS is plane.PLATFORMS
    assert plane.PLATFORMS == ("auto", "cuda", "ascend", "cpu")
    assert set(plane.ACCELERATORS) == {"cuda", "ascend"}


# ------------------------------------------------------------------------ the probe


def test_the_probe_takes_its_answers_rather_than_finding_them() -> None:
    """Every host is reachable from every other host, which is what makes the Ascend arm testable.

    This is the whole reason the answers are parameters: the suite that has to prove the Ascend
    decision runs on a machine with no Ascend anything, so `npu_available=True` is not a mock of the
    hardware, it is the same call the 910 host makes.
    """
    assert _accelerator("cuda").is_accelerator
    assert _accelerator("ascend").platform == "ascend"
    assert _accelerator("ascend").distributed_backend == "hccl"
    assert not _accelerator("cpu").is_accelerator


def test_cuda_wins_when_a_host_has_both() -> None:
    """Stated rather than left to dict ordering: four NVIDIA cards do not want a default of `npu`.

    A host with cards from two vendors is a host where the caller has to say which, and `--device`
    is how they say it. The default is the conservative reading of an ambiguous host.
    """
    both = plane.probe_accelerator(cuda_available=True, npu_available=True)
    assert both.platform == "cuda"


def test_the_unnarrowed_probe_answers_about_this_host() -> None:
    """With no answers supplied the probe reads the machine, and its answer is in the vocabulary.

    Deliberately weak: what this host *is* belongs to this host, but that the answer is one of the
    four, and that it agrees with torch about CUDA, are properties of the probe.
    """
    torch = _torch()
    probed = plane.probe_accelerator()
    assert probed.platform in plane.PLATFORMS
    assert (probed.platform == "cuda") is bool(torch.cuda.is_available())


# ------------------------------------------------------------------------ resolving `auto`


def test_auto_is_resolved_and_anything_else_is_left_alone() -> None:
    """`auto` asks the build; an explicit value is the caller's answer and not this function's.

    Resolving `auto` is the behaviour the flag's own help text has claimed since it was introduced
    and which nothing implemented -- the flag was stored and never read.
    """
    ascend = _accelerator("ascend")
    assert plane.resolve_platform("auto", ascend) == "ascend"
    assert plane.resolve_platform("cuda", ascend) == "cuda"  # the caller's problem, not this one
    assert plane.resolve_platform("cpu", ascend) == "cpu"


def test_a_requested_platform_outside_the_vocabulary_is_refused() -> None:
    with pytest.raises(plane.DeviceError, match="device must be one of"):
        plane.resolve_platform("cuda:2", _accelerator("cuda"))


# ---------------------------------------------------------------- the device type, and its absence


def test_an_unregistered_device_type_is_refused_and_torch_is_not_asked() -> None:
    """The Ascend experience on a box without `torch_npu`, stated as a refusal rather than a wall.

    `torch.device("npu")` raises, because an unregistered type cannot be named at all -- `torch_npu`
    is what renames torch's PrivateUse1 backend to `npu`. torch's own message for it lists the types
    it does know and mentions neither the platform that was asked for nor the package that would
    provide it; this is the message an operator can act on, and this box is exactly where it fires.
    """
    assert not plane.device_type_registered("npu")
    assert plane.device_type_registered("cuda")
    assert plane.device_type_registered("cpu")

    with pytest.raises(plane.DeviceError) as raised:
        plane.canonical_device("npu:1")
    message = str(raised.value)
    assert "npu" in message and "torch_npu" in message and "ascend" in message


# ------------------------------------------------------------------------ canonical_device


def test_no_device_passes_through() -> None:
    """`None` is an answer, and this function does not know what it means.

    For one caller it is "keep the weights on the host"; for another it is "there is nothing to
    place". Resolving it here would decide a policy the callers disagree about.
    """
    assert plane.canonical_device(None) is None


def test_an_unindexed_card_is_resolved_through_the_caller() -> None:
    """`torch.device("cuda")` is not `torch.device("cuda", 0)`, and forty nodes share one rope table.

    A table keyed on the unindexed form has its card decided by whichever node was built first. The
    index is read through a callable so the branch is testable without a card -- which is the same
    trade `visible_card_count(env=...)` makes for the environment.
    """
    torch = _torch()
    assert plane.canonical_device("cuda", current_index=lambda: 2) == torch.device("cuda", 2)
    assert plane.canonical_device(torch.device("cuda"), current_index=lambda: 1) == torch.device(
        "cuda", 1
    )
    # Already indexed: the card the caller named is the card they meant.
    assert plane.canonical_device("cuda:3", current_index=lambda: 2) == torch.device("cuda", 3)


def test_a_host_device_is_left_alone() -> None:
    """`cpu` has no card to resolve, and asking for one would be asking a host device for a rank."""
    torch = _torch()
    assert plane.canonical_device("cpu") == torch.device("cpu")
    assert plane.canonical_device(torch.device("cpu")) == torch.device("cpu")


def test_an_unknown_device_type_is_torchs_business_to_refuse() -> None:
    """A type torch knows but this plane does not -- `meta` -- is passed through, not re-judged."""
    torch = _torch()
    assert plane.canonical_device("meta") == torch.device("meta")


# ------------------------------------------------- the two policies beside the resolution


def test_a_strict_caller_gets_the_card_it_asked_for() -> None:
    """`require_device` resolves the same way `canonical_device` does, because it *is* that call.

    The caller here is a runtime that has to put weights somewhere, so an unindexed `cuda` has to
    come back as the card this process is on -- which is the half of the five copies that was
    identical in all five, and the half that is now written once.
    """
    torch = _torch()
    device = plane.require_device(
        "cuda", platform="cuda", accelerator=_accelerator("cuda"), current_index=lambda: 2
    )
    assert device == torch.device("cuda", 2)
    assert plane.require_device(
        torch.device("cuda", 3), platform="cuda", accelerator=_accelerator("cuda")
    ) == torch.device("cuda", 3)


def test_a_strict_caller_is_refused_a_card_of_another_kind() -> None:
    """The branch the llama.cpp-shaped `device="cuda"` default produces on an Ascend host.

    `require_device` is told which platform it needs and answers a device of the other kind by
    name, rather than by handing back something the caller will use as if it were right -- which is
    what a shared resolution with no policy on top of it would do.
    """
    torch = _torch()
    with pytest.raises(plane.DeviceError, match=r"resolved to .*cpu.*not one"):
        plane.require_device("cpu", platform="cuda", accelerator=_accelerator("cuda"))
    assert plane.require_device(
        "cuda:0", platform="cuda", accelerator=_accelerator("cuda")
    ) == torch.device("cuda", 0)


def test_a_strict_caller_is_refused_on_a_host_that_has_no_such_card() -> None:
    """The other refusal: the device is right, the *host* is not -- and this one named a different
    exception in the copy it replaces.

    `_canonical_cuda_device` raised `RuntimeError("CUDA is not available ...")` here and
    `ValueError` one line above, so a caller had to catch two types for one question. Both are
    `DeviceError` now, and the message says which platform the host does have and which device type
    torch would need -- which on an Ascend box is `torch_npu`'s name, not this host's.
    """
    _torch()
    with pytest.raises(plane.DeviceError, match=r"has no ascend accelerator.*'npu'"):
        plane.require_device("cuda:0", platform="ascend", accelerator=_accelerator("cpu"))
    with pytest.raises(plane.DeviceError, match="has no cuda accelerator"):
        plane.require_device("cuda:0", platform="cuda", accelerator=_accelerator("cpu"))


def test_a_strict_caller_on_an_ascend_host_still_needs_torch_npu() -> None:
    """The boundary this suite cannot cross, stated as a refusal rather than as a skip.

    On a host whose accelerator *is* an Ascend one, `require_device("npu:0")` still cannot build
    the device here: naming `npu` is `torch_npu`'s job, and this box has no `torch_npu`. So the
    claim that is testable on a CUDA box is the refusal, and the reading it names is the one an
    operator needs -- which package would make this call succeed.
    """
    with pytest.raises(plane.DeviceError, match="torch_npu"):
        plane.require_device("npu:0", platform="ascend", accelerator=_accelerator("ascend"))


def test_a_strict_caller_needs_an_accelerator_platform() -> None:
    """`require_device(..., platform="cpu")` is a contradiction: a host platform is not a card."""
    with pytest.raises(plane.DeviceError, match="needs an accelerator platform"):
        plane.require_device("cpu", platform="cpu", accelerator=_accelerator("cpu"))


def test_the_nullable_reading_answers_none_when_the_host_has_no_card() -> None:
    """`accelerator_device` is the third policy: no card is an answer, not a failure.

    This is the loader's question -- a checkpoint read into card memory with a host-memory fallback
    is a different thing from a runtime that requires a card, and the two must not be folded onto
    one refusal or a legitimate host-resident load becomes an exception.
    """
    _torch()
    assert plane.accelerator_device(accelerator=_accelerator("cpu")) is None
    assert plane.accelerator_device("cuda:1", accelerator=_accelerator("cpu")) is None


def test_the_nullable_reading_answers_the_current_card_when_asked_for_nothing() -> None:
    """No argument means "whichever card this process is on" -- and `None` is what a host answers.

    Two `None`s on the same line of a signature, saying different things: the argument's `None`
    asks for the current card, and the return's `None` says there is no card to give.
    """
    torch = _torch()
    device = plane.accelerator_device(accelerator=_accelerator("cuda"), current_index=lambda: 1)
    assert device == torch.device("cuda", 1)
    assert plane.accelerator_device(
        "cuda", platform="cuda", accelerator=_accelerator("cuda"), current_index=lambda: 1
    ) == torch.device("cuda", 1)


def test_the_nullable_reading_still_refuses_a_device_the_caller_named_wrongly() -> None:
    """Only the *host* is allowed to answer `None`; a named device of the wrong kind is a mistake."""
    _torch()
    with pytest.raises(plane.DeviceError, match="not a cuda device"):
        plane.accelerator_device("cpu", accelerator=_accelerator("cuda"))


# ------------------------------------------------------------- the declaration, enforced


def test_a_platform_the_runtime_does_not_declare_is_refused() -> None:
    """`devices=(...)` is a declaration with teeth now, and this is where the teeth are.

    `--backend mimo --device cpu` used to be accepted by a runtime whose own declaration says
    `("cuda",)`: the run then proceeded on a platform the adapter has no branch for, which is a
    failure that arrives some minutes later inside a model rather than at the flag.
    """
    with pytest.raises(UnsupportedFeatureError, match=r"devices=\('cuda',\)"):
        select_backend(_args(backend="mimo", device="cpu"), accelerator=_accelerator("cpu"))
    with pytest.raises(UnsupportedFeatureError, match="runtime that serves 'ascend'"):
        select_backend(_args(backend="mimo"), accelerator=_accelerator("ascend"))


def test_a_platform_the_runtime_declares_is_selected() -> None:
    """The other half: the check refuses `cuda`-only runtimes elsewhere and leaves the rest running.

    `torch` declares `("cuda", "cpu")`, so it is accepted on either of its own platforms -- which is
    what makes this a check on the declaration rather than a check on the host.
    """
    assert select_backend(_args(), accelerator=_accelerator("cuda")) == "torch"
    assert select_backend(_args(backend="torch"), accelerator=_accelerator("cpu")) == "torch"
    assert (
        select_backend(_args(backend="mimo"), accelerator=_accelerator("cuda")) == "mimo"
    )


def test_a_cardless_host_still_serves_the_runtimes_that_declare_a_host_platform() -> None:
    """`torch` declares `("cuda", "cpu")`, so `auto` on a host with neither accelerator runs.

    The suite already runs this way on a card-bearing host -- `CUDA_VISIBLE_DEVICES=""` is how a
    CPU-only test is written -- and a runtime that declares `cpu` is the one that stays usable when
    the cards are not there. That spelling is a *platform* change and not only a card change, which
    is the part worth knowing before using it on a runtime that declares none: `auto` resolves from
    `torch.cuda.is_available()`, so emptying the variable answers `cpu` exactly as a cardless host
    does.
    """
    assert select_backend(_args(backend="torch"), accelerator=_accelerator("cpu")) == "torch"
    # ... and `mimo`, which declares `("cuda",)` and no host platform, is the one that stops.
    with pytest.raises(UnsupportedFeatureError, match="--device auto resolved to"):
        select_backend(_args(backend="mimo"), accelerator=_accelerator("cpu"))


# ----------------------------------------------------------------- binding a card, and waiting


def test_binding_a_card_is_a_no_op_on_a_host_platform() -> None:
    """`cpu` has no card to point at, and asking is not an error.

    A launch with no accelerator in it reaches here legitimately -- `EngineArgs` has already refused
    a `--device-ids` beside `--device cpu` -- so the honest answer is to do nothing rather than to
    raise on a no-op.
    """
    _torch()
    plane.bind_device(0, platform="cpu")


def test_binding_an_unregistered_platform_is_refused_by_name() -> None:
    """The Ascend experience on this box, and the reason it is a refusal and not an `ImportError`.

    `_torch_module_for` reaches `torch.npu` through an `import torch_npu`, so without the check this
    would surface as a `ModuleNotFoundError` from the middle of whichever setup function happened to
    bind first -- naming a module the operator never asked for rather than the platform they did.
    """
    _torch()
    for call in (
        lambda: plane.bind_device(0, platform="ascend"),
        lambda: plane.synchronize(platform="ascend"),
        lambda: plane.device_count(platform="ascend"),
    ):
        with pytest.raises(plane.DeviceError) as raised:
            call()
        assert "torch_npu" in str(raised.value) and "'ascend'" in str(raised.value)


def test_binding_a_cuda_card_points_this_process_at_it() -> None:
    """And puts it back, so the rest of the suite runs on the card it started on.

    The index is **non-zero and different from the one the process is already on**, which is the
    half that makes this a test. Card 0 is where a fresh process already is, so asserting
    ``current_device() == 0`` after binding 0 passes against a `bind_device` that does nothing at
    all, and against one that drops its argument and always binds 0 -- which is every way this
    function can be wrong. Binding a card the process is not on is what pins both the call and its
    argument, and this host has four of them to move between.
    """
    torch = _torch()
    count = torch.cuda.device_count()
    if count < 2:
        pytest.skip(f"needs two cards to tell a bind from a no-op; this host has {count}")
    original = torch.cuda.current_device()
    target = 1 if original != 1 else 0
    try:
        plane.bind_device(target, platform="cuda")
        assert torch.cuda.current_device() == target, (
            "bind_device did not move this process to the card it was handed. Card 0 is where a "
            "process starts, so a test that binds 0 cannot tell this apart from a no-op"
        )
    finally:
        torch.cuda.set_device(original)


def test_the_card_count_is_zero_on_a_host_platform_and_a_number_here() -> None:
    """`0` rather than an error, so the launcher's "a world of N needs the cards" check reads the
    same on both platforms: the count is the fact and the refusal is the caller's sentence."""
    torch = _torch()
    assert plane.device_count(platform="cpu") == 0
    assert plane.device_count(platform="cuda") >= (1 if torch.cuda.is_available() else 0)


def test_synchronize_is_a_no_op_on_a_host_platform() -> None:
    """What the call sites write by hand today as
    `torch.cuda.synchronize if dev.type == "cuda" else lambda *a: None`."""
    _torch()
    plane.synchronize(platform="cpu")


# ------------------------------------------------ the two entry points that bind and join


def test_setup_dist_names_what_it_needs_when_the_host_has_neither_accelerator(monkeypatch) -> None:
    """`relicllm/runtime/generation.py`'s entry point, on a host with no card of either kind.

    The sentence was `"GGUF raw-block runtime requires CUDA"`, which named one vendor for a runtime
    that is about to be able to run on another. What it needs is an accelerator; whose is the device
    plane's question.
    """
    from relicllm.runtime import generation

    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.setattr(generation, "probe_accelerator", lambda: _accelerator("cpu"))
    with pytest.raises(RuntimeError, match="needs an accelerator"):
        generation.setup_dist()


def test_setup_dist_returns_a_device_of_the_platforms_type(monkeypatch) -> None:
    """`world, rank, local_rank, device` -- and the device is the accelerator's, not `cuda`'s.

    Single rank, so no process group is built and the test needs no rendezvous; the claim is about
    the tuple the entry point hands on, which is where every downstream `torch.device(...)` starts.

    `local_rank` is 0 here because nothing set `LOCAL_RANK`, so the assertion on
    `current_device()` says the entry point bound the card it was told to -- which is a fact about
    the *environment*, not about the binding. `test_binding_a_cuda_card_points_this_process_at_it`
    is the one that moves a process between cards and can tell a bind from a no-op.
    """
    torch = _torch()
    from relicllm.runtime import generation

    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(generation, "probe_accelerator", lambda: _accelerator("cuda"))

    world, rank, local_rank, device = generation.setup_dist()
    assert (world, rank, local_rank) == (1, 0, 0)
    assert device.type == "cuda" and device.index == 0
    assert torch.cuda.current_device() == 0


def test_the_qwen4_entry_point_binds_the_card_the_plane_named(monkeypatch) -> None:
    """One of the five entry points, called for real, on whatever host is reading this.

    This is the call the move onto the plane got wrong: `relicllm/models/qwen4_exp/runtime.py` read
    `probe_accelerator()` with no import for it, so `init_distributed()` -- the first thing this
    runtime's own entry point runs -- died on a `NameError` before it could print a line. Single
    rank, so nothing rendezvouses and no weights are loaded; the claim is the context the caller
    then builds a model on.
    """
    from relicllm.models.qwen4_exp import runtime as qwen4

    for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"):
        monkeypatch.delenv(name, raising=False)
    accelerator = plane.probe_accelerator()

    ctx = qwen4.init_distributed()
    assert (ctx.rank, ctx.world_size, ctx.initialized) == (0, 1, False)
    assert ctx.device.type == accelerator.torch_device_type
    if accelerator.is_accelerator:
        assert ctx.device.index == 0


# ------------------------------------------------------------------------ the call sites

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _shipped_python_files() -> list[Path]:
    """Every `.py` this repository ships: `src/` and `relicllm/`, and not `tests/`.

    The suite calls the plane too, but a test that drops an import fails in the test rather than in
    a process that was asked to serve a request, so it is not what this check is watching.
    """
    paths: list[Path] = []
    for root in (_REPO_ROOT / "src", _REPO_ROOT / "relicllm"):
        paths.extend(path for path in root.rglob("*.py") if "__pycache__" not in path.parts)
    return sorted(paths)


def _bound_names(tree: ast.AST) -> set[str]:
    """Every name this module brings into scope, by any route.

    Imports, assignments, arguments, comprehension and loop targets, `except ... as`, definitions.
    Being generous is deliberate: the check is for a name that is *read* and bound nowhere, and a
    false negative hides a regression while a false positive is a test nobody trusts.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            bound.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            bound.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
    return bound


def test_every_caller_of_the_plane_imports_what_it_calls() -> None:
    """A module that calls the plane without importing it is a `NameError`, and only at the call.

    That is the shape of the one defect the move onto the plane left behind: the missing import in
    `relicllm/models/qwen4_exp/runtime.py` was invisible to the whole suite, because nothing under
    `tests/` imports that module and an unread name is a name nobody notices. A static check is
    what catches it without loading six engines to ask each one a question about its own text.

    Read from the module's own `__all__`, so a name added to the plane is covered the day it is
    added rather than the day someone remembers to list it here.
    """
    vocabulary = frozenset(plane.__all__)
    offenders: dict[str, set[str]] = {}
    for path in _shipped_python_files():
        if path == _REPO_ROOT / "src" / "runtime" / "device.py":
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        used = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        missing = (used & vocabulary) - _bound_names(tree)
        if missing:
            offenders[str(path.relative_to(_REPO_ROOT))] = missing

    assert not offenders, (
        "these modules call the device plane without importing it: "
        + "; ".join(f"{name} uses {', '.join(sorted(names))}" for name, names in offenders.items())
        + ". The import belongs with the module's other imports, not inside the function that uses "
        "it -- which is where it would have to be for a call site to stay working by accident"
    )
