"""Which accelerator this process runs on, and the names torch and the collectives use for it.

RelicLLM targets two hardware families that share no runtime: NVIDIA's CUDA cards and Ascend's 910
series. Nothing above this module should have to know which one it is on, and until now nothing did
-- because every route was CUDA, spelled ``torch.cuda`` at three hundred and thirty call sites. This
module is the one place that answers the question, so that those three hundred and thirty are
decisions made *here* rather than assumptions made *there*.

The two halves of that question are answered separately, because they are separately true. *Which
platform* is :func:`probe_accelerator`: ``cuda``, ``ascend`` or ``cpu``. *Which card, within CUDA* is
:func:`probe_card_capability`, and it is a second question because the NVIDIA side is no longer one
card: the tree was written for Turing (``sm_75``, RTX 2080 Ti) and now also ships on Ada (``sm_89``,
RTX 4090), which carry different tensor-core instructions and which ``relic-core`` already branches
between. A module that answered only the first would leave every caller to re-derive the second from
``torch.cuda`` at its own call site -- which is the shape of the problem this module exists to end,
met again one level down.

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
``relicllm/models/`` and by ``relicllm/backends/``, so sitting it in ``relicllm/protocol/`` would add
twenty-one reverse edges -- precisely the outcome that rule exists to prevent. Here, the model side is
an intra-package import and the serving side is the direction that already runs.

That is also why nothing here imports ``relicllm``. The refusal is :class:`DeviceError`, a
``ValueError``, which is the convention this tree already uses (``relicllm/loader/gguf/prism_hadamard.py``
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


#: The three-digit compute capability the kernel libraries use, ``major * 100 + minor * 10``, so
#: that ``sm_75`` is ``750`` and ``sm_89`` is ``890``. Spelled here rather than left implicit
#: because ``relic-core``'s headers compare against exactly this (``GGML_CUDA_CC_TURING`` is
#: ``750``), and a second arithmetic convention for the same two cards is a footgun.
def _compute_capability(major: int, minor: int) -> int:
    return major * 100 + minor * 10


@dataclass(frozen=True, slots=True)
class CardCapability:
    """What the CUDA card under this process is, in the terms a kernel choice is made in.

    This exists because :class:`Accelerator` answers *which platform* and nothing answered *which
    card*, and the two NVIDIA families this tree runs on do not merely differ in speed. Turing
    (``sm_75``, the RTX 2080 Ti) reaches an 8-bit matrix multiply through a sequence of ``m16n8k8``
    instructions; Ada (``sm_89``, the RTX 4090) does it in one ``m16n8k16``. ``relic-core`` already
    branches on the difference -- ``ops.py``'s ``_auto_impl`` sends an FP8 op down a Triton path at
    or above ``major`` 8 and a Torch path below it -- so a caller that needs to know which side of
    that fork it is on has had no way to ask. This is the way.

    Two edges worth knowing before a caller relies on the predicates. **The cut is relic-core's, not
    a hardware fact**: ``major >= 8`` is the line that repository forks on, and it is the whole of
    what the predicates claim. **And the reading is what the torch build supports, not what is
    plugged in**: :func:`probe_card_capability` asks ``torch.cuda.get_device_capability``, which
    reports the capability the torch binary was compiled for, so a build without ``sm_89`` device
    code answers ``(7, 5)`` on the very 4090 this descriptor exists to name. The descriptor reports
    what it read; it does not check that the card and the build agree.

    **A card this module has never heard of is not a refusal.** :func:`probe_card_capability`
    reports whatever torch reports; ``known`` is only "this is one of the cards the tree was
    validated on", and an unknown card keeps its real numbers so a caller can still decide. The
    *default* value is where the caution lives: :data:`UNKNOWN_CAPABILITY` is Turing, the oldest
    thing the tree supports, so a caller handed no card at all takes the branch that assumes
    nothing the newer card has -- which is the safe direction for a capability gate to fail in.
    """

    #: Whether this is one of the cards the tree was validated against, not whether it is usable.
    known: bool
    major: int
    minor: int
    #: What the driver calls the card, for a log line and for a golden fixture's record. Never
    #: load-bearing. It holds the recognised name when the capability is one of the known two, the
    #: driver's string when the probe read an unrecognised card, and empty when the caller supplied
    #: the capability itself and no name.
    name: str = ""

    @property
    def cc(self) -> int:
        """The three-digit capability: ``750`` for ``sm_75``, ``890`` for ``sm_89``."""
        return _compute_capability(self.major, self.minor)

    @property
    def supports_fp8_tensor_core(self) -> bool:
        """Whether ``relic-core`` would send an FP8 matmul down its Triton path for this card.

        ``major >= 8`` is the cut ``relic-core``'s ``ops.py`` uses to choose between its Triton and
        Torch FP8 paths. It is the whole of what this property claims: it mirrors *that fork*, not a
        hardware feature table. ``relic-core`` also gates the Triton arm on ``_USE_TRITON`` (false
        when the triton import fails) and asks the same capability question only when
        ``torch.cuda.is_available()``, neither of which is represented here -- so at ``major >= 8``
        with triton unimportable, this reads True while that fork takes the Torch path.
        """
        return self.major >= 8

    @property
    def supports_fp4_tensor_core(self) -> bool:
        """Whether a 4-bit matmul is a tensor-core op here: Blackwell (``sm_10x``) and later.

        Neither card this repository ships for has one. It is here because "4-bit" is a name a
        model file can carry regardless of the card, and a caller that assumed FP8's answer also
        covered FP4 would be wrong on the 4090. Same caveat as
        :attr:`supports_fp8_tensor_core`: the cut mirrors the fork above, and Blackwell's own
        sub-version has no full-rate 4-bit path.
        """
        return self.major >= 10


#: The cards the tree has been validated on, by compute capability. Membership sets ``known``; it
#: decides nothing else, and a card absent from this map is still probed and still reports its real
#: numbers.
KNOWN_CAPABILITIES: dict[int, str] = {750: "Turing / RTX 2080 Ti", 890: "Ada / RTX 4090"}

#: What a caller with no card gets. Turing is the floor of everything the tree supports, so a
#: capability gate reading this takes the conservative branch -- no FP8, no FP4 -- rather than
#: assuming whatever the newer card added.
UNKNOWN_CAPABILITY = CardCapability(known=False, major=7, minor=5, name="")


def probe_card_capability(
    *,
    index: int = 0,
    capability: tuple[int, int] | None = None,
    name: str | None = None,
) -> CardCapability:
    """The card at ``index``, or :data:`UNKNOWN_CAPABILITY` when there is none to read.

    Both halves of the answer may be supplied instead of probed, which is the convention every
    other probe here follows and the reason this is testable off the card:

    - ``capability=(8, 9)`` is what the RTX 4090 host answers, and handing it in makes the Ada
      branch reachable from the 2080 Ti box that has no Ada device -- the same trick
      :func:`probe_accelerator` uses to make the Ascend arm reachable on a CUDA host.
    - ``capability`` given as ``None`` means "ask torch", not "no card". A host that genuinely has
      no CUDA device answers through the probe failing, which returns the unknown value rather
      than raising: a question with no card to answer it is not an error, it is the host half of
      the same choice :func:`accelerator_device` makes with ``None``.

    The name is read from torch only when the capability was; a caller who supplied the numbers
    gets its own label or none, because cobbling a name from ``torch.cuda`` while the caller is
    describing a card that may not be on this host would report a *different* card's name.
    """
    if capability is None:
        try:
            import torch
        except ImportError:  # pragma: no cover - see _cuda_available
            return UNKNOWN_CAPABILITY
        if not torch.cuda.is_available():  # pragma: no cover - the sm_75/ascend hosts answer here
            return UNKNOWN_CAPABILITY
        major, minor = torch.cuda.get_device_capability(index)
        card_name = torch.cuda.get_device_name(index)
    else:
        major, minor = capability
        card_name = name or ""
    known_name = KNOWN_CAPABILITIES.get(_compute_capability(int(major), int(minor)))
    return CardCapability(
        known=known_name is not None,
        major=int(major),
        minor=int(minor),
        name=known_name or card_name,
    )


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


def _torch_module_for(device_type: str):
    """``torch.cuda``, ``torch.npu`` or ``torch.cpu`` -- the namespace holding a card's verbs.

    Torch gives each accelerator a namespace with the same shape, so reading it once is what lets
    :func:`bind_device` and :func:`synchronize` be one function each rather than a branch per call
    site. The import of ``torch_npu`` is what registers the backend and is therefore load-bearing for
    the ``getattr`` to find anything.
    """
    import torch

    if device_type == "cuda":
        return torch.cuda
    if device_type == "cpu":
        return torch.cpu
    import torch_npu  # noqa: F401

    return getattr(torch, device_type, None)


def _accelerator_module(platform: str) -> Any | None:
    """The torch namespace for ``platform``, or ``None`` for a host platform.

    Where :func:`bind_device` and :func:`synchronize` agree, which is the part worth writing once: a
    platform whose device type this build has not registered is refused by name, so an Ascend launch
    without ``torch_npu`` reads as that rather than as an ``AttributeError`` on a namespace that does
    not exist -- or, worse, as an ``ImportError`` from the middle of a collective.
    """
    device_type = torch_device_type(platform)
    if device_type == "cpu":
        return None
    if not device_type_registered(device_type):
        raise DeviceError(
            f"the {platform!r} platform needs the {device_type!r} device type, which this torch "
            f"build has not registered. That is what torch_npu provides."
        )
    return _torch_module_for(device_type)


def bind_device(index: int, *, platform: str | None = None) -> None:
    """Point this thread at card ``index`` for the platform this process runs on.

    The per-thread binding is not bookkeeping. ``torch`` reads a *thread's* current device, so a rank
    that resolved its card correctly in one thread and then runs its forwards in another allocates on
    card 0 -- which is why the serving bridge binds inside the run thread rather than where the
    runtime was constructed, and why each ``setup_dist`` body binds once before anything else.

    A host platform has no card to bind, so this is a no-op there rather than a refusal: the caller
    that reached here is a launch with no accelerator in it, and `EngineArgs` has already refused a
    `--device-ids` beside `--device cpu` before any of this ran.
    """
    resolved = probe_accelerator().platform if platform is None else platform
    module = _accelerator_module(resolved)
    if module is not None:
        module.set_device(int(index))


def synchronize(*, platform: str | None = None) -> None:
    """Wait for the accelerator's queued work, on the platform this process runs on.

    The host platform has nothing to wait for, so this is the no-op the call sites already write by
    hand as ``torch.cuda.synchronize if dev.type == "cuda" else lambda *a: None``.
    """
    resolved = probe_accelerator().platform if platform is None else platform
    module = _accelerator_module(resolved)
    if module is not None:
        module.synchronize()


def device_count(*, platform: str | None = None) -> int:
    """How many cards of this process's platform are visible, or ``0`` on a host platform.

    ``0`` rather than an error, so that the "a world of N needs the cards, and this host has none"
    check a launcher already makes reads the same on both platforms -- the count is the fact, and the
    refusal is the caller's sentence.
    """
    resolved = probe_accelerator().platform if platform is None else platform
    module = _accelerator_module(resolved)
    return 0 if module is None else int(module.device_count())


__all__ = [
    "ACCELERATORS",
    "KNOWN_CAPABILITIES",
    "PLATFORMS",
    "UNKNOWN_CAPABILITY",
    "Accelerator",
    "CardCapability",
    "DeviceError",
    "accelerator_device",
    "bind_device",
    "canonical_device",
    "device_count",
    "device_type_registered",
    "probe_accelerator",
    "probe_card_capability",
    "process_group_backend",
    "require_device",
    "resolve_platform",
    "synchronize",
    "torch_device_type",
]
