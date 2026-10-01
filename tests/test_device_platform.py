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

import pytest

from relicllm.api import ConfigurationError, EngineArgs, UnsupportedFeatureError
from relicllm.backends.factory import select_backend
from relicllm.cli import DEVICE_PLATFORMS
from src.runtime import device as plane


def _torch():
    """torch, or a skip. Only the tests that build a `torch.device` need it."""
    return pytest.importorskip("torch")


def _args(**overrides) -> EngineArgs:
    base = dict(model="checkpoint", backend="cpp")
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
    """The other half: the check refuses `cuda`-only runtimes on an NPU and leaves the rest running.

    `cpp` declares both accelerators, so the same `auto` resolves to `ascend` and is accepted --
    which is what makes this a check on the declaration rather than a check on the host.
    """
    assert select_backend(_args(), accelerator=_accelerator("cuda")) == "cpp"
    assert select_backend(_args(), accelerator=_accelerator("ascend")) == "cpp"
    assert select_backend(_args(device="ascend"), accelerator=_accelerator("ascend")) == "cpp"
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
    # ... and `cpp`, which declares no host platform, is the one that stops.
    with pytest.raises(UnsupportedFeatureError, match="--device auto resolved to"):
        select_backend(_args(backend="cpp"), accelerator=_accelerator("cpu"))
