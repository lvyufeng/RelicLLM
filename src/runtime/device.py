"""Which accelerator this process runs on, and the names torch and the collectives use for it.

RelicLLM targets two hardware families that share no runtime: NVIDIA's Turing cards (``sm_75``) and
Ascend's 910 series. Nothing above this module should have to know which one it is on, and until now
nothing did -- because every route was CUDA, spelled ``torch.cuda`` at three hundred and thirty call
sites. This module is the one place that answers the question, so that those three hundred and thirty
are decisions made *here* rather than assumptions made *there*.

Three things are deliberately pure, and each is pure for a reason a test can use:

* The **tables** are strings. ``platform -> device type`` and ``platform -> collective`` are the two
  facts a port gets wrong first, and neither needs a device object to state.
* The **probe** takes its answers as arguments. ``probe_accelerator(npu_available=True)`` returns the
  Ascend answer on a host with no Ascend anything, so the branch is reachable in this suite.
* The **index resolution** takes a callable. An unindexed ``"cuda"`` means "whichever card is
  current", which is a question only the card can answer -- so it is asked through a parameter,
  defaulted to the real probe, rather than from inside the function.

The last two are the pattern the serving side already uses for the same problem:
:func:`relicllm.backends.runtime_engine.visible_card_count` takes the environment as a mapping so a
test can hand it one, and ``tests/test_runtime_engine.py`` says why in its docstring -- *"a test that
imports torch to check it would skip on exactly the hosts where the string route is the one that
runs."*

## Where this lives, and why it is not in ``relicllm``

``CLAUDE.md`` names ``relicllm/protocol/`` as the home for a shared piece when `src/` would otherwise
reach back into ``relicllm/``, and `src/` -> ``relicllm/`` is currently one file. The exception is
this module's reason for existing: the device question is asked by twenty-one files under
``src/models/`` and by ``relicllm/backends/``, so sitting it in ``relicllm/protocol/`` would add
twenty-one reverse edges -- precisely the outcome that rule exists to prevent. Here, the model side is
an intra-package import and the serving side is the direction that already runs.

That is also why nothing here imports ``relicllm``. The refusal is :class:`DeviceError`, a
``ValueError``, which is the convention this tree already uses (``src/loader/gguf/prism_hadamard.py``
is the nearest example); the serving layer translates it where it needs its own error type.

## What this module can and cannot check on a CUDA host

A device *type* is registered with torch by whoever implements it, and an unregistered one cannot be
named at all: ``torch.device("npu")`` raises, because ``torch_npu`` is what renames torch's
PrivateUse1 backend to ``"npu"``. So on a CUDA box every Ascend *decision* is testable and no Ascend
*object* is. :func:`canonical_device` is written to make that boundary read as a refusal rather than
as torch's own ``RuntimeError: Expected one of cpu, cuda, ...`` -- which is the message an operator
who asked for Ascend on a box without ``torch_npu`` gets today.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch is imported lazily: this module is on the ``--help`` path
    import torch


class DeviceError(ValueError):
    """A device question this build cannot answer: an unknown platform, an unregistered type, or a
    card this host does not have.

    A ``ValueError`` because that is what the five copies of this question raised before they were
    folded here -- ``_canonical_cuda_device``, ``_cuda_quant_device`` and the two inline versions of
    it in the GLM-DSA and MiniMax loaders -- so a caller that already catches a bad device keeps
    catching it. The one exception is that function's "no CUDA here" branch, which was a
    ``RuntimeError`` and is now this; nothing in the tree catches either, and one type for every
    refusal the plane makes is worth more than the difference between the two.
    """


#: The spelling ``--device`` accepts. ``auto`` is a question, not an answer: it is resolved by
#: :func:`probe_accelerator` and never stored, which is why it is here and not in
#: :data:`ACCELERATORS`.
PLATFORMS: tuple[str, ...] = ("auto", "cuda", "ascend", "cpu")

#: The platforms that are a card of some kind. The complement of ``cpu``, and the set a runtime
#: declaring ``devices=...`` chooses from when it means "this needs an accelerator".
ACCELERATORS: tuple[str, ...] = ("cuda", "ascend")

#: The device type torch knows this platform by. Pure data: naming the type is not the same as being
#: able to construct one, and keeping them apart is what lets the mapping be tested on any host.
#: ``ascend`` is ``npu`` because that is the name ``torch_npu`` gives torch's PrivateUse1 backend.
_DEVICE_TYPE: dict[str, str] = {"cuda": "cuda", "ascend": "npu", "cpu": "cpu"}

#: The collective library, which is not a detail a caller can leave to the device: NCCL speaks to
#: CUDA and HCCL to Ascend, and asking ``torch.distributed`` for the wrong one fails inside a
#: collective rather than at the call that asked for it.
_COLLECTIVE: dict[str, str] = {"cuda": "nccl", "ascend": "hccl", "cpu": "gloo"}


@dataclass(frozen=True, slots=True)
class Accelerator:
    """What this host can serve, and the names the layers under it use for it.

    One value rather than three calls, because the three answers have to agree: a run that chose the
    Ascend device type and the NCCL collective has a device it cannot address.
    """

    #: ``cuda``, ``ascend`` or ``cpu``. Never ``auto`` -- resolving is what produces this.
    platform: str
    #: What to name in ``torch.device(...)``: ``cuda``, ``npu`` or ``cpu``.
    torch_device_type: str
    #: The ``torch.distributed`` backend for this platform: ``nccl``, ``hccl`` or ``gloo``.
    distributed_backend: str

    @property
    def is_accelerator(self) -> bool:
        return self.platform in ACCELERATORS


def _cuda_available() -> bool:
    """Whether torch can reach a CUDA device."""
    try:
        import torch
    except ImportError:  # pragma: no cover - relicllm requires torch; the CLI's --help does not
        return False
    return bool(torch.cuda.is_available())


def _npu_available() -> bool:
    """Whether torch can reach an Ascend device.

    ``import torch_npu`` is the whole test, and it is asked here rather than at module scope: the
    package is absent on a CUDA host and must stay absent from its import graph. The name check is a
    second reading of the same fact, and the one that says *which* privateuse backend was registered
    rather than that one was -- torch reports ``privateuseone`` until something renames it.
    """
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return torch._C._get_privateuse1_backend_name() == "npu"


def probe_accelerator(
    *,
    cuda_available: bool | None = None,
    npu_available: bool | None = None,
) -> Accelerator:
    """What this host can serve, preferring an accelerator over the host.

    Each answer may be supplied rather than probed, which is how the Ascend arm is reachable on a
    machine that has never seen an Ascend device: ``probe_accelerator(cuda_available=False,
    npu_available=True)`` is the same call the 910 host makes, and it needs nothing installed to
    make it. Both halves have to be named -- supplying only ``npu_available`` leaves ``cuda`` to the
    real probe, which on a four-card box answers ``True`` and wins by the rule below, so the call
    would return CUDA while reading as though it asked for Ascend.

    CUDA wins when both are present. A host carrying cards from two vendors is a host where the
    caller has to say which, and ``--device`` is how they say it -- but a default that reached for
    the NPU on a box with four NVIDIA cards would be a poor one, so the order is stated here rather
    than left to dict ordering.
    """
    cuda = _cuda_available() if cuda_available is None else bool(cuda_available)
    npu = _npu_available() if npu_available is None else bool(npu_available)
    if cuda:
        return Accelerator("cuda", _DEVICE_TYPE["cuda"], _COLLECTIVE["cuda"])
    if npu:
        return Accelerator("ascend", _DEVICE_TYPE["ascend"], _COLLECTIVE["ascend"])
    return Accelerator("cpu", _DEVICE_TYPE["cpu"], _COLLECTIVE["cpu"])


def _checked(platform: str) -> str:
    if platform not in _DEVICE_TYPE:
        raise DeviceError(
            f"{platform!r} is not a platform; expected one of "
            f"{', '.join(name for name in PLATFORMS if name != 'auto')}"
        )
    return platform


def torch_device_type(platform: str) -> str:
    """The name torch knows ``platform`` by: ``ascend`` is ``npu``.

    Pure, and separate from ``torch.device`` on purpose. Asking for the *name* cannot fail;
    constructing the *device* fails on any torch that has not registered it, and a caller who
    conflated the two would learn that at the wrong end of a traceback.
    """
    return _DEVICE_TYPE[_checked(platform)]


def process_group_backend(platform: str) -> str:
    """The collective backend for ``platform``: ``nccl``, ``hccl`` or ``gloo``.

    ``hccl`` is CANN's collective library and the Ascend counterpart of NCCL. ``gloo`` is the host
    backend -- the honest answer for ``cpu``, and also what the existing runtimes already use for
    their control-plane subgroups beside a device collective.
    """
    return _COLLECTIVE[_checked(platform)]


def resolve_platform(requested: str, accelerator: Accelerator) -> str:
    """``requested`` as a concrete platform, taking the ``auto`` answer from ``accelerator``.

    ``auto`` is the only input this changes, and that is the whole point of the function: the flag
    has been accepted and stored since it was introduced without anything ever reading it, so a
    launch that asked the build a question was answered by nobody.

    An explicit value is returned as given even when the host disagrees, because refusing it is a
    different question asked somewhere that can answer it. Whether a *runtime* can serve a platform
    is a statement about that runtime's declaration; whether a *build* can is a statement about
    torch's registered device types. Both are checked where those things are known, and folding
    either in here would make this the only place that could answer -- and the only place that
    cannot.
    """
    if requested not in PLATFORMS:
        raise DeviceError(f"device must be one of {', '.join(PLATFORMS)}, got {requested!r}")
    return accelerator.platform if requested == "auto" else requested


def device_type_registered(name: str) -> bool:
    """Whether torch can name this device type at all.

    ``torch.device`` refuses an unregistered type with a ``RuntimeError`` listing the ones it knows,
    and torch publishes no predicate for the question -- so the constructor is the predicate, and
    this is where its exception is caught rather than three layers up.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - see _cuda_available
        return False
    try:
        torch.device(name)
    except (RuntimeError, TypeError):
        return False
    return True


def canonical_device(
    device: torch.device | str | None,
    *,
    current_index: Callable[[], int] | None = None,
) -> torch.device | None:
    """``device`` with an unindexed accelerator resolved to the card this process is on.

    ``torch.device("cuda")`` is not the same key as ``torch.device("cuda", 0)``: it lands wherever
    the current device happens to point, and a table keyed on it is a table whose card is decided by
    whichever node was built first. Resolving here rather than at each lookup is what keeps forty
    nodes on one card's roperies instead of forty-some -- which is the reason the original of this
    function was written, and the reason the four copies of it that followed were.

    ``None`` passes through. Every caller that can be handed no device has an answer for it, and this
    function does not know what that answer is: for one it is "keep the weights on the host", for
    another "there is nothing to place". ``current_index`` is a callable so the branch is testable
    without a card of the right kind; it defaults to the real reading.

    A device type torch has not registered is refused by name. That is the whole of the Ascend
    experience on a host without ``torch_npu``, and torch's own message for it -- "Expected one of
    cpu, cuda, ipu, xpu, ..." -- names neither the platform that was asked for nor the package that
    would provide it.
    """
    if device is None:
        return None
    import torch

    if isinstance(device, str):
        name = device.split(":")[0].strip().lower()
        if not device_type_registered(name):
            platform = next((p for p, t in _DEVICE_TYPE.items() if t == name), None)
            raise DeviceError(
                f"device {device!r} names the {name!r} device type, which this torch build has not "
                f"registered. "
                + (
                    f"That is the {platform!r} platform: it needs torch_npu installed here."
                    if platform
                    else "Install the package that provides it."
                )
            )
    resolved = torch.device(device)
    if resolved.index is None and resolved.type in _DEVICE_TYPE.values() and resolved.type != "cpu":
        index = current_index() if current_index is not None else _current_index(resolved.type)
        resolved = torch.device(resolved.type, index)
    return resolved


def require_device(
    device: torch.device | str,
    *,
    platform: str = "cuda",
    accelerator: Accelerator | None = None,
    current_index: Callable[[], int] | None = None,
) -> torch.device:
    """``device`` as a card of ``platform``, or a refusal naming which half is wrong.

    Two callers in this tree need a card and cannot proceed without one -- the MiniMax-M2
    device-resident cache and the GLM-DSA raw-block runtime -- and each wrote its own version of
    this check next to its own copy of the resolution. The resolution is now
    :func:`canonical_device`'s, and what is left here is the *policy*, which is the half that
    genuinely differs between callers:

    * a device of the wrong kind is refused, which is the case the llama.cpp-shaped default
      ``device="cuda"`` produces on a host whose accelerator is an NPU;
    * a host that has no such card is refused, because a caller that requires one has no fallback
      to be handed.

    Both were ``ValueError`` and ``RuntimeError`` respectively in the copies; both are
    :class:`DeviceError` here, so that a caller can catch one type for every refusal the plane
    makes. Nothing in the tree catches either.

    ``accelerator`` is injected rather than probed for the reason every other function here takes
    its answers: it is what makes the Ascend arm reachable from a host that has never seen an
    Ascend device. A caller who supplies one is answering "is this host an Ascend host", and the
    device it hands in is then resolved on the host's behalf rather than checked against it.
    """
    resolved_platform = _checked(platform)
    if resolved_platform not in ACCELERATORS:
        raise DeviceError(f"require_device needs an accelerator platform, not {resolved_platform!r}")
    host = probe_accelerator() if accelerator is None else accelerator
    if host.platform != resolved_platform:
        raise DeviceError(
            f"this host has no {resolved_platform} accelerator to put the tensor on -- the device "
            f"plane answers {host.platform!r} here. The device type torch would need is "
            f"{torch_device_type(resolved_platform)!r}"
            + (" (torch_npu, which is what registers it)" if resolved_platform == "ascend" else "")
        )
    resolved = canonical_device(device, current_index=current_index)
    expected = torch_device_type(resolved_platform)
    if resolved is None or resolved.type != expected:
        raise DeviceError(
            f"a {resolved_platform} device was required and {device!r} resolved to "
            f"{resolved!r}, which is not one"
        )
    return resolved


def accelerator_device(
    device: torch.device | str | None = None,
    *,
    platform: str = "cuda",
    accelerator: Accelerator | None = None,
    current_index: Callable[[], int] | None = None,
) -> torch.device | None:
    """The card of ``platform`` to put this tensor on, or ``None`` when the host has none.

    The third policy, and the one the loader uses: ``None`` is not a failure, it is the answer
    "keep this on the host". A checkpoint that is read into card memory and can fall back to host
    memory is a different thing from a runtime that *requires* a card, and folding the two
    together would turn a legitimate host-resident load into an exception.

    No argument means "whichever card this process is on", which is how the original read the
    current device, and it is the same question :func:`canonical_device` resolves for an unindexed
    name. That is why ``device=None`` is expanded here rather than passed through: for this policy
    ``None`` in the *argument* means the current card while ``None`` out means no card at all.
    """
    resolved_platform = _checked(platform)
    if resolved_platform not in ACCELERATORS:
        raise DeviceError(f"accelerator_device needs an accelerator platform, not {resolved_platform!r}")
    host = probe_accelerator() if accelerator is None else accelerator
    if host.platform != resolved_platform:
        return None
    if device is None:
        device = torch_device_type(resolved_platform)
    resolved = canonical_device(device, current_index=current_index)
    if resolved is None or resolved.type != torch_device_type(resolved_platform):
        raise DeviceError(
            f"{device!r} is not a {resolved_platform} device, and only the *host* is allowed to "
            f"answer None here -- a device that is named and of the wrong kind is a caller's mistake "
            f"rather than a host without the card"
        )
    return resolved


def _current_index(device_type: str) -> int:
    """The card torch is currently pointed at, for ``device_type``.

    Split out because it is the one line of :func:`canonical_device` that cannot run without a card
    of the right kind -- so it is the line a test replaces by passing ``current_index``.
    """
    import torch

    if device_type == "cuda":
        return int(torch.cuda.current_device())
    import torch_npu  # noqa: F401  the import is what registers the backend

    return int(torch.npu.current_device())


__all__ = [
    "ACCELERATORS",
    "PLATFORMS",
    "Accelerator",
    "DeviceError",
    "accelerator_device",
    "canonical_device",
    "device_type_registered",
    "probe_accelerator",
    "process_group_backend",
    "require_device",
    "resolve_platform",
    "torch_device_type",
]
