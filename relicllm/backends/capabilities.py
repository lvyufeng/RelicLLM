"""What each runtime is, and which checkpoints are its, declared once.

Before this the answer lived in several places that could disagree. Capability was computed by hand
in six ``capabilities()`` methods; the *rejection* of a checkpoint was written twice per adapter --
once as a predicate for ``auto`` and once as a raise for an explicit ``--backend``; three adapters
carried their own copy of the same ``_IGNORED_OPTIONS`` set; and ``supports_prefix_caching`` meant
"the store exists" on two runtimes and "the byte budget is positive" on a third, so the same field
answered two different questions depending on which backend you asked.

This module is the one place that answers "can this runtime do X" and "is this checkpoint its". The
adapters read it, the dispatcher reads it, and a runtime that has more to say than its declaration
says it by naming the field (see :func:`declared_capabilities`).

Two things are deliberately *not* here:

* **Whether a runtime is available** -- whether the native module imports, whether a card is
  present. That is a property of the machine, not of the runtime, and it changes between two
  processes on the same box.
* **Anything a configuration decides.** A declaration is static; the one capability that genuinely
  depends on the configuration is prefix reuse, and the value is resolved per instance because the
  runtime is the only thing that knows whether its gate is open. See
  :attr:`RuntimeCapabilities.reads_prefix_cache` for the single question that field asks.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any

from relicllm.api import BackendCapabilities, EngineArgs, UnsupportedFeatureError
from relicllm.protocol.contract import ServedFields


# ---------------------------------------------------------------------------------------------
# Recognising a checkpoint
# ---------------------------------------------------------------------------------------------
#
# These read a checkpoint's own headers. They are here rather than in the dispatcher because the
# declaration is what uses them, and a second copy of "does this config say qwen3_5" is how the
# routing question and the refusal question drift apart -- which is the defect this module exists
# to remove.

_QWEN35_TYPES = {"qwen3_5", "qwen3_5_text"}
#: The original DeepSeek-V4. Deliberately *not* ``deepseek_v41``: the two are different
#: architectures with different runtimes. The exact set keeps the ``model_type`` side apart, and the
#: digit boundary in :func:`_names_model` is what keeps the ``architectures`` side from swallowing
#: V4.1 -- which is what the `v41` runtime exists to serve.
_DEEPSEEK_V4_TYPES = {"deepseek_v4", "deepseek_v4_text"}
#: The GGUF ``general.architecture`` the DeepSeek-V4 export carries. It drops the underscore the
#: config's ``model_type`` has, so it is its own constant rather than a reuse of the set above.
_DEEPSEEK_V4_GGUF = "deepseek4"
_V41_TYPES = {"deepseek_v41", "deepseek_v41_text"}
_MIMO_TYPES = {"mimo_v2", "mimo_v2_text"}
_XING4_TYPES = {"xing4_0", "xing4_0_text"}
#: Qwen3.8-Flash-Next. Unlike Qwen3.5's, this checkpoint nests its text stack under ``text_config``
#: beside ``vision_config``, so both keys are its own name -- and the nested walk in
#: :func:`_names_model` is what reaches the inner one.
_QWEN4EXP_TYPES = {"qwen4_exp", "qwen4_exp_text"}


def config_path_for(path: str, explicit: str | None = None) -> Path:
    candidate = Path(explicit) if explicit else Path(path) / "config.json"
    if candidate.is_dir():
        candidate = candidate / "config.json"
    return candidate


def read_config(path: str, explicit: str | None = None) -> dict[str, Any] | None:
    """A checkpoint's ``config.json``, or ``None`` when there is not a readable one.

    ``None`` means *no evidence*, and every caller has to treat it that way: a checkpoint whose
    config cannot be read is not one that has been disproved.
    """
    try:
        config = config_path_for(path, explicit)
        if not config.is_file():
            return None
        with config.open(encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError):
        return None


def _names_model(config: Mapping[str, Any], types: set[str], architectures: str) -> bool:
    """Whether a config names one of ``types``, at the root or nested.

    The nesting walk is shared because the reason for it is: the released DeepSeek-V4.1 and
    Qwen3.5 checkpoints nest their text stack, so ``model_type`` is the architecture at the root
    and again inside ``text_config``. A wrapper config is still the architecture.

    ``architectures`` is matched up to a digit boundary rather than anywhere in the string. The
    architecture names are not decorated versions of one another, they are generations:
    ``DeepseekV4ForCausalLM`` and ``DeepseekV41ForCausalLM`` differ by a *digit*, so a plain
    substring test makes V4's ``deepseekv4`` a prefix of V4.1's value and one runtime's predicate
    answers for the other's checkpoint. Stopping at the first digit is what keeps ``V4`` from
    reading ``V41`` -- the config's own names put a letter, the end of the string, or a
    capital boundary after the digit in every release here, so this is not a tuning knob.
    """
    value = str(config.get("model_type") or "").lower()
    if value in types:
        return True
    declared = config.get("architectures", ())
    if isinstance(declared, (list, tuple)):
        pattern = re.compile(re.escape(architectures) + r"(?!\d)")
        if any(pattern.search(str(item).lower()) for item in declared):
            return True
    nested = config.get("text_config")
    return isinstance(nested, dict) and _names_model(nested, types, architectures)


def is_qwen35_config(config: Mapping[str, Any]) -> bool:
    return _names_model(config, _QWEN35_TYPES, "qwen3_5")


def is_v41_config(config: Mapping[str, Any]) -> bool:
    return _names_model(config, _V41_TYPES, "deepseekv41")


def is_mimo_config(config: Mapping[str, Any]) -> bool:
    """Whether a config describes MiMo-V2.6-Flash.

    Unlike V4.1's, this checkpoint's text stack is not nested: ``model_type`` and ``architectures``
    are at the root and describe the language model, with the vision and audio towers under their
    own keys. The nested walk happens anyway because a MiMo release under someone else's multimodal
    wrapper should not stop being a MiMo release -- and it walks ``text_config`` as well as
    ``text_encoder_config``, which is the pair the dispatcher accepted before this was moved here.
    """
    if _names_model(config, _MIMO_TYPES, "mimov2"):
        return True
    for key in ("text_encoder_config", "text_config"):
        nested = config.get(key)
        if isinstance(nested, dict) and is_mimo_config(nested):
            return True
    return False


def is_xing4_config(config: Mapping[str, Any]) -> bool:
    """Whether a config describes Xing4.0-29B-A4B.

    The released checkpoint is not nested and its ``model_type`` is the name the GGUF's own
    ``general.architecture`` carries, so the two artifacts the release ships are recognized by the
    same string.
    """
    return _names_model(config, _XING4_TYPES, "xing4")


def is_qwen4_exp_config(config: Mapping[str, Any]) -> bool:
    """Whether a config describes Qwen3.8-Flash-Next.

    This release is a conditional-generation wrapper: the root says ``model_type=qwen4_exp`` and
    ``architectures=["Qwen4ExpForConditionalGeneration"]`` beside a ``vision_config``, and the text
    stack is nested under ``text_config`` with its own ``qwen4_exp_text``. The nested walk in
    :func:`_names_model` is what reads through to it, which is why this is a one-line predicate
    despite the two-level shape.
    """
    return _names_model(config, _QWEN4EXP_TYPES, "qwen4exp")


def is_deepseek_v4_config(config: Mapping[str, Any]) -> bool:
    """Whether a config describes the original DeepSeek-V4 (not V4.1).

    ``deepseek_v4`` and ``deepseek_v41`` are two architectures. The ``model_type`` side is kept
    apart by the exact set; the ``architectures`` side -- where the releases put
    ``DeepseekV4ForCausalLM`` and ``DeepseekV41ForCausalLM`` -- is kept apart by the digit boundary
    in :func:`_names_model`, since a plain substring test would read V4 inside V4.1.
    """
    return _names_model(config, _DEEPSEEK_V4_TYPES, "deepseekv4")


def requested_gguf(args: EngineArgs) -> bool:
    """Whether the caller is pointing at a GGUF, by declaration or by shape."""
    return args.model_format == "gguf" or (
        args.model_format == "auto" and checkpoint_has_gguf(args.checkpoint_dir)
    )


def checkpoint_has_gguf(path: str) -> bool:
    """Whether the requested checkpoint is visibly a GGUF model."""
    try:
        candidate = Path(path)
        if candidate.is_file():
            return candidate.suffix.lower() == ".gguf"
        if not candidate.is_dir():
            return False
        return any(candidate.glob("*.gguf"))
    except OSError:
        return False


def gguf_architecture(path: str) -> str | None:
    """``general.architecture`` out of a GGUF, or ``None`` when there is no answer.

    Read rather than assumed: the file the launcher points at may be any GGUF, so this is what
    separates "a Xing4 export" from "a file ending in .gguf". The two failures are kept apart --
    ``None`` is *no evidence* (not a GGUF, more than one of them, a header that will not parse) and
    an empty string is impossible, so a caller can refuse on a header it read rather than on one it
    could not. Where it is ``None`` the caller falls through to the adapter, whose GGUF reader
    reports the precise error when it is the one that ends up with the file.
    """
    try:
        candidate = Path(path)
        if candidate.is_dir():
            found = sorted(candidate.glob("*.gguf"))
            if len(found) != 1:
                return None
            candidate = found[0]
        if not candidate.is_file() or candidate.suffix.lower() != ".gguf":
            return None
        from relicllm.loader.gguf.bundle import read_gguf_bundle

        metadata = read_gguf_bundle(candidate).metadata
        return str(metadata.get("general.architecture") or "") or None
    except Exception:
        return None


# ---------------------------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------------------------


class Verdict(Enum):
    """What a checkpoint says about which runtime should serve it."""

    #: This runtime is the one for it.
    READ = "read"
    #: No evidence either way. Not enough to be routed here, and not enough to refuse either --
    #: a caller who named this backend by hand gets the backend's own loader error, which names
    #: the file it could not read.
    UNKNOWN = "unknown"
    #: Evidence *against*. A refusal, and the sentence says what the evidence was.
    REFUSED = "refused"


@dataclass(frozen=True)
class Identification:
    verdict: Verdict
    reason: str = ""

    @property
    def routes_here(self) -> bool:
        return self.verdict is Verdict.READ

    @property
    def refuses(self) -> bool:
        return self.verdict is Verdict.REFUSED


READ = Identification(Verdict.READ)
UNKNOWN = Identification(Verdict.UNKNOWN)


def refused(reason: str) -> Identification:
    return Identification(Verdict.REFUSED, reason)


#: The ``backend_options`` keys every launch carries and no *model* options parser has a use for.
#:
#: The CLI fills in ``engine_kind``, ``routed_experts_device`` and ``pd_mode`` on every serve
#: command and the supervisor adds ``nccl_id_path`` for a sharded one, so all four arrive whether or
#: not the selected adapter reads them. Accepting them is what makes one launch command line work
#: for every backend. Every *other* unknown key is a refusal, because a tuning option that silently
#: does nothing is how a run ends up measured on the wrong lever.
#:
#: ``enable_batching`` and ``scheduler_timeout_ms`` are carried for the same reason and are
#: *not* ignored: the factory reads the first to refuse a batch path no runtime here owns, and
#: the second is the timeout a launch may name. They are here so a launch can name them without
#: the model option parser refusing them as unknown, which is the treatment every other key still
#: gets.
IGNORED_OPTIONS = frozenset(
    {
        "engine_kind",
        "routed_experts_device",
        "pd_mode",
        "nccl_id_path",
        "enable_batching",
        "scheduler_timeout_ms",
    }
)


@dataclass(frozen=True)
class RuntimeCapabilities:
    """One runtime, declared. See the module docstring for what belongs here.

    The fields mirror :class:`relicllm.api.BackendCapabilities`, which is the shape a client
    reads; :func:`declared_capabilities` is the one function that turns one into the other.
    """

    name: str
    models: tuple[str, ...] = ()
    model_formats: tuple[str, ...] = ()
    devices: tuple[str, ...] = ()
    #: Whether this runtime can ever hold more than one request at a time. False is a promise that
    #: a width above 1 is refused rather than accepted and ignored.
    supports_batch: bool = False
    supports_streaming: bool = True
    supports_cancellation: bool = False
    #: Which request fields this runtime's answer actually applies, audited by
    #: :func:`relicllm.protocol.contract.audit` against every body the runtime is asked to serve.
    #:
    #: It is one table for two questions that used to have two answers: the wire declaration
    #: (:func:`declared_capabilities` reads its ``logprobs`` and ``structured_outputs`` from here)
    #: and the refusal a request gets when it names a field the runtime will not apply. A runtime
    #: that declares a field and does not apply it is the failure this field exists to make
    #: impossible, which is why it is a ``ServedFields`` -- the very object :func:`audit` refuses
    #: from -- rather than a handful of booleans this module would then have to keep in step.
    served_fields: ServedFields = ServedFields()
    supports_speculative_decoding: tuple[str, ...] = ()
    #: Whether this runtime has a prefix-resumption path *at all*. Whether it is open in a given
    #: configuration is the instance's answer, not this one's -- see
    #: :func:`declared_capabilities`'s ``reads_prefix_cache``.
    reads_prefix_cache: bool = False
    #: ``backend_options`` keys this runtime accepts and ignores.
    ignores_options: frozenset[str] = IGNORED_OPTIONS
    #: What ``auto`` needs to route here, and what an explicit ``--backend`` refuses on if the
    #: answer is :attr:`Verdict.REFUSED`. One predicate answers both, which is the point: the two
    #: questions used to be two functions and could disagree.
    identifies: Callable[[EngineArgs], Identification] = lambda _args: UNKNOWN
    #: Prose the dispatcher merges into ``details``: what this runtime executes on, what its
    #: scheduler is, how it cancels. A declaration's details are static; an instance adds its own.
    details: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------------------------


def _is_gguf_only(args: EngineArgs, runtime: str) -> Identification | None:
    """The shared first rule for the safetensors-only runtimes.

    Returns the refusal when the checkpoint is a GGUF, ``None`` when it is not -- so a caller can
    chain it ahead of the architecture rule and read the two causes apart, which is what makes the
    refusal name the thing that actually refused.
    """
    if requested_gguf(args):
        return refused(
            f"backend={runtime!r} reads the checkpoint's safetensors shards only; "
            f"a GGUF checkpoint must use backend='torch'"
        )
    return None


def _identify_v41_or_mimo(
    args: EngineArgs, runtime: str, predicate: Callable[[Mapping[str, Any]], bool], name: str
) -> Identification:
    gguf = _is_gguf_only(args, runtime)
    if gguf is not None:
        return gguf
    config = read_config(args.checkpoint_dir, args.config_path)
    if config is None:
        return UNKNOWN
    if not predicate(config):
        return refused(f"backend={runtime!r} serves {name} checkpoints only")
    return READ


_XING4_ONLY = (
    "backend='xing4' serves Xing4.0-29B-A4B checkpoints only: a directory whose "
    "config.json says model_type=xing4_0, or a .gguf whose general.architecture is xing4_0"
)

_XING4_GGUF_ONLY = (
    "backend='xing4' reads its weights from the GGUF; the safetensors directory beside it "
    "supplies the tokenizer and the config only, so --model-format safetensors is not a "
    "checkpoint this backend can serve"
)


def _identify_xing4(args: EngineArgs) -> Identification:
    """Whether this checkpoint is one ``xing4`` can serve, asked architecture first.

    The two causes are told apart because the caller acts on them differently: "this is not a
    Xing4 checkpoint" says stop, and "this is one, but you asked for the wrong ``--model-format``"
    says change a flag. Asking the format first would report the second for a checkpoint the first
    is true of, which is why the readability of the config comes before either.

    The format rule itself is not a formality: this runtime always reads its weights from the GGUF,
    so ``--model-format safetensors`` is never something it can serve. The auto path had always
    refused that pair; the explicit path used to let it through selection and fail at the loader
    with "no .gguf file at or under ...", which is the failure this moves earlier.
    """
    config = read_config(args.checkpoint_dir, args.config_path)
    if config is not None and not is_xing4_config(config):
        return refused(_XING4_ONLY)
    if str(args.model_format).lower() == "safetensors":
        return refused(_XING4_GGUF_ONLY)
    if config is not None:
        return READ
    # No config to read, so the GGUF's own header is the only evidence left -- and its absence is
    # no evidence at all rather than evidence against.
    architecture = gguf_architecture(args.checkpoint_dir)
    if architecture is None:
        return UNKNOWN
    if architecture == "xing4_0":
        return READ
    return refused(f"{_XING4_ONLY} (this one declares general.architecture={architecture!r})")


_QWEN4EXP_ONLY = (
    "backend='qwen4_exp' serves Qwen3.8-Flash-Next checkpoints only: a directory whose "
    "config.json says model_type=qwen4_exp (or nests a text_config that does)"
)


def _identify_qwen4exp(args: EngineArgs) -> Identification:
    """Whether this checkpoint is one ``qwen4_exp`` can serve, asked architecture first.

    Safetensors-only, like the other specific runtimes here: this release ships shards, and the
    ``torch`` runtime is the one that reads a GGUF. The two causes are told apart so a caller reads
    the flag to change rather than the runtime to change -- the same split ``xing4``'s identify
    makes for its own format rule.
    """
    gguf = _is_gguf_only(args, "qwen4_exp")
    if gguf is not None:
        return gguf
    config = read_config(args.checkpoint_dir, args.config_path)
    if config is None:
        return UNKNOWN
    if not is_qwen4_exp_config(config):
        return refused(_QWEN4EXP_ONLY)
    return READ


_TORCH_ONLY = (
    "backend='torch' serves DeepSeek-V4 checkpoints only: a directory whose config.json says "
    "model_type=deepseek_v4, or a .gguf whose general.architecture is deepseek4 or a "
    "'deepseek4_'-prefixed variant of it"
)


def _is_deepseek_v4_gguf(architecture: str | None) -> bool:
    """Whether a GGUF ``general.architecture`` names the DeepSeek-V4 export family.

    The base export carries ``deepseek4``; a variant suffixes it, and this host's MTP build says
    ``deepseek4_mtp_support``. All of those are V4 checkpoints for the same reason the config's
    ``model_type=deepseek_v4`` is, so the family is matched and not just the one spelling -- refusing
    a V4 variant as "not V4" is the wrong answer in the direction that loses a servable model. The
    separator is required so the prefix cannot reach into a differently-numbered neighbour, which is
    the same generation-digit trap the config-side match stops at.
    """
    if not architecture:
        return False
    return architecture == _DEEPSEEK_V4_GGUF or architecture.startswith(_DEEPSEEK_V4_GGUF + "_")


def _identify_torch(args: EngineArgs) -> Identification:
    """Whether this checkpoint is one the ``torch`` runtime can serve.

    The name says which framework it is built on and nothing about scope: this runtime is
    hard-wired to DeepSeek-V4 -- it loads ``relicllm.models.deepseek_v4`` through the legacy
    serving engine, with no architecture dispatch anywhere. So it identifies DeepSeek-V4 and
    refuses everything else, which is the same evidence rule the other runtimes use. A checkpoint
    nothing claims no longer falls through to here and gets read as V4; it is refused, and
    :func:`route` reports that no runtime serves it.
    """
    config = read_config(args.model, args.config_path)
    if config is not None and is_deepseek_v4_config(config):
        return READ
    architecture = gguf_architecture(args.model)
    if _is_deepseek_v4_gguf(architecture):
        return READ
    if config is None and architecture is None:
        # No evidence either way: not enough to route here, not enough to refuse.
        return UNKNOWN
    # Evidence against -- either a config that named a different architecture or a GGUF header that
    # did, and each is worth saying so the caller reads what was read.
    detail = (
        f"this one declares model_type={config.get('model_type')!r}"
        if config is not None
        else f"this one declares general.architecture={architecture!r}"
    )
    return refused(f"{_TORCH_ONLY} ({detail})")


RUNTIMES: dict[str, RuntimeCapabilities] = {
    "torch": RuntimeCapabilities(
        name="torch",
        # Named for the framework it is built on, and it declares the one architecture it serves --
        # which is what makes it a candidate rather than a catch-all in `auto` and in triage.
        models=("deepseek_v4",),
        model_formats=("safetensors", "gguf"),
        devices=("cuda", "cpu"),
        supports_batch=False,
        supports_cancellation=True,
        # The filling of the table that can be read off this file. `choices` is the host's fan-out
        # (`relicllm/choices.py::expanded`) rather than anything a runtime does, so every entry
        # takes it. `logit_bias` and `parallel_tool_calls` are applied nowhere here. Streaming
        # logprobs are off everywhere: a streamed chunk carries its token's text and no ranking,
        # and the DeepSeek-V4 payload builder raises rather than pretending otherwise.
        #
        # `structured_outputs` is the one this runtime answers differently from the rest, and what
        # it means is stated on the field: the checkpoint's own instruction template
        # (`relicllm/encoding/deepseek_v4.py::response_format_template`) reaches the prompt through
        # `protocol/chat.py::prepare_messages`. There is no logit mask behind it -- the runtime has
        # no per-token sampler to hang one on -- and it is declared because the prompt instruction
        # is what the field buys here.
        served_fields=ServedFields(
            choices=True,
            stop=True,
            logprobs=True,
            penalties=True,
            repetition_penalty=True,
            min_p=True,
            structured_outputs=True,
        ),
        reads_prefix_cache=True,
        identifies=_identify_torch,
        details={
            "execution": "existing src/ runtime",
            "scheduler": "legacy serving queue",
            "cancellation": "safe boundary only",
        },
    ),
    "v41": RuntimeCapabilities(
        name="v41",
        models=("deepseek_v4_1", "deepseek_v41"),
        model_formats=("safetensors",),
        devices=("cpu", "cuda"),
        supports_cancellation=True,
        # `structured_outputs` for the same reason `torch` declares it, and a different encoder:
        # this checkpoint ships its own (`<checkpoint>/encoding/encoding.py`) and renders the
        # schema block under `role == "system"`, which is where
        # `protocol/chat.py::prepare_messages` now attaches the field. `logprobs` is False here
        # where it is True on `torch`: this runtime's loop does not produce a ranking.
        served_fields=ServedFields(choices=True, stop=True, structured_outputs=True),
        reads_prefix_cache=True,
        identifies=lambda args: _identify_v41_or_mimo(
            args, "v41", is_v41_config, "DeepSeek-V4.1-Flash"
        ),
        details={
            "execution": "relicllm/models/deepseek_v4_1 PyTorch runtime",
            "scheduler": "one mutable KV state, serialized at the backend boundary",
            "cancellation": "per-step collective; not inside the prompt's forward",
        },
    ),
    "mimo": RuntimeCapabilities(
        name="mimo",
        models=("mimo_v2", "mimo_v2_6"),
        model_formats=("safetensors",),
        devices=("cuda",),
        supports_cancellation=True,
        # `stop` is served on both routes: the streamed one through `TokenStreamer`, and the serial
        # one through `RuntimeAdapter._serial_result`. `response_format` is refused rather than
        # dropped -- this encoder renders no schema block, so accepting the field would be the
        # silent ignore the contract exists to prevent.
        served_fields=ServedFields(choices=True, stop=True),
        reads_prefix_cache=True,
        identifies=lambda args: _identify_v41_or_mimo(
            args, "mimo", is_mimo_config, "MiMo-V2.6-Flash"
        ),
        # No static prose: this runtime's description is entirely about the run -- the deal, the
        # arena a slot, the resident bands -- so the adapter supplies all of it.
        details={},
    ),
    "xing4": RuntimeCapabilities(
        name="xing4",
        models=("xing4_0",),
        model_formats=("gguf",),
        devices=("cuda",),
        supports_cancellation=True,
        # The same two fields `mimo` serves, for the same two reasons.
        served_fields=ServedFields(choices=True, stop=True),
        reads_prefix_cache=True,
        identifies=_identify_xing4,
        # No static prose, for the reason `mimo` has none.
        details={},
    ),
    "qwen4_exp": RuntimeCapabilities(
        name="qwen4_exp",
        models=("qwen4_exp",),
        model_formats=("safetensors",),
        devices=("cuda",),
        # One mutable KV cache serves one sequence, exactly as for the runtimes above, so a width
        # above 1 is refused rather than accepted and ignored.
        supports_batch=False,
        supports_cancellation=True,
        # The same two fields as the other two specific runtimes. This one samples greedily with
        # `argmax`, so half the sampling table has nothing to apply to -- which is the reason the
        # field is refused rather than accepted and ignored.
        served_fields=ServedFields(choices=True, stop=True),
        # The first of the specific runtimes with no prefix store: `models/qwen4_exp/runtime.py`
        # forward-passes its whole prompt, and resuming it needs a store that module does not have.
        # False here so a client reads "a repeated prefix is *not* resumed", which is the honest
        # answer and not a silent default.
        reads_prefix_cache=False,
        identifies=_identify_qwen4exp,
        # Static prose only; the adapter adds the run's own numbers.
        details={
            "execution": "relicllm/models/qwen4_exp PyTorch runtime, GatedDeltaNet + QSA",
            "scheduler": "one mutable KV state, serialized at the backend boundary",
        },
    ),
}


#: The order ``auto`` asks them in: the architecture-specific readers before the widest one. Every
#: entry now identifies a checkpoint *and refuses the others*, so the list is a set of claims rather
#: than a fallback chain -- and a checkpoint no entry claims is refused, not read by the last one.
AUTO_ORDER: tuple[str, ...] = ("v41", "mimo", "xing4", "qwen4_exp", "torch")


def runtime_capabilities(name: str) -> RuntimeCapabilities:
    try:
        return RUNTIMES[name]
    except KeyError as exc:
        raise KeyError(
            f"no capability declaration for backend {name!r}; declared: {sorted(RUNTIMES)}"
        ) from exc


def served_fields(name: str) -> ServedFields:
    """Which request fields the runtime ``name`` applies, as a declaration.

    The same read :func:`declared_capabilities` makes for the wire booleans and the adapters make
    for their audit, so a runtime's answer to "does the answer apply this field" is one value in
    one place.
    """
    return runtime_capabilities(name).served_fields


def declared_capabilities(
    name: str,
    *,
    details: Mapping[str, Any] | None = None,
    reads_prefix_cache: bool = True,
    **overrides: Any,
) -> BackendCapabilities:
    """The wire declaration for one runtime, from its declaration plus what only it knows.

    ``reads_prefix_cache`` is the gate, and it is a parameter rather than a field a caller can set
    directly because the question is fixed: **will a repeated prefix be resumed in this
    configuration, instead of being forwarded again?** That is the only reading a client can act
    on, and it is the reading the field used to lack -- the two runtimes that reported "the store
    object exists" and the one that reported "the byte budget is positive" were answering about
    their configuration while the field is documented as being about reuse.

    A runtime whose gate is open passes ``True`` (the default). One that can resume but is not
    configured to passes ``False``, and so does one on a code path that does not resume.

    ``overrides`` names any other field the instance knows better than the declaration does -- the
    architectures a build actually registered, the device backend it linked, whether it built a
    scheduler. Naming the field is deliberate: it is a short, greppable list of the places a
    runtime disagrees with its own declaration, which is what keeps the declaration honest.
    """
    declaration = replace(runtime_capabilities(name), **overrides)
    return BackendCapabilities(
        name=declaration.name,
        models=declaration.models,
        model_formats=declaration.model_formats,
        devices=declaration.devices,
        supports_batch=declaration.supports_batch,
        supports_streaming=declaration.supports_streaming,
        supports_cancellation=declaration.supports_cancellation,
        supports_embeddings=False,
        # Derived from the served-field table rather than declared a second time. A runtime that
        # told `/v1/models` it ranks log probabilities and then refused `logprobs` would be
        # answering two questions about one capability with two values; this is the read that
        # makes them the same value.
        supports_logprobs=declaration.served_fields.logprobs,
        supports_structured_outputs=declaration.served_fields.structured_outputs,
        supports_prefix_caching=declaration.reads_prefix_cache and reads_prefix_cache,
        supports_speculative_decoding=declaration.supports_speculative_decoding,
        details={**declaration.details, **(details or {})},
    )


def identify(name: str, args: EngineArgs) -> Identification:
    """What the checkpoint in hand says about ``name``."""
    return runtime_capabilities(name).identifies(args)


def no_runtime_serves(args: EngineArgs) -> UnsupportedFeatureError:
    """The refusal for a checkpoint no runtime identifies.

    Every entry in :data:`AUTO_ORDER` claims an architecture and refuses the rest, so "nothing
    matched" is now a real answer rather than an impossible branch. It is raised instead of routing
    the checkpoint into the last runtime on the list, which is how an unclaimed checkpoint used to
    be read as DeepSeek-V4.
    """
    return UnsupportedFeatureError(
        "no backend serves this checkpoint; auto tried "
        f"{', '.join(AUTO_ORDER)} -- pass --backend to name one"
    )


def route(args: EngineArgs) -> str:
    """The runtime ``auto`` selects, or raise when no entry claims the checkpoint."""
    for name in AUTO_ORDER:
        if runtime_capabilities(name).identifies(args).routes_here:
            return name
    raise no_runtime_serves(args)


def refusal(name: str, args: EngineArgs) -> str:
    """The sentence refusing ``args`` for an explicitly requested ``name``, or the empty string."""
    verdict = runtime_capabilities(name).identifies(args)
    return verdict.reason if verdict.refuses else ""
