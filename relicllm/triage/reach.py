"""Gate 1: what is missing between this checkpoint and a run, recorded but never ranked.

Three questions, and the answer to each is a *note*. That is a ruling, not a shortcut: a missing
GGUF kernel is a file in relic-core that an agent can write, a missing loader is a reader that does
not exist yet, and neither makes a model unservable. The only thing that makes a model impossible in
this package is that it does not fit, which is gate 0's question and is asked there. Keeping
reachability out of the tier function's signature (:func:`relicllm.triage.tier.tier`) is how that
stays true rather than being restated in a docstring and quietly depended on.

What gate 1 is *for*, then:

* it is the difference between "no adapter" and "an adapter, and a slow path" -- this session
  established that the roster's three fastest models at an 8k input, 1819 and 753 and 639 tok/s, have
  no serving path at all, while the fastest one that is served is far below them. A tool that
  reported only a tier would lose that;
* it names the work. "needs a kernel for ``ptq1_0`` in relic-core" is actionable in a way that
  "demo" is not, and under the cost-out-of-scope ruling it is *only* a work item.

The adapter answer comes from :data:`relicllm.backends.capabilities.RUNTIMES` -- the same table the
dispatcher routes on -- rather than from a second copy of "which runtime reads DeepSeek-V4.1". Two
copies of that question is precisely the defect the capabilities module documents having removed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from relicllm.backends.capabilities import RUNTIMES
from relicllm.loader.gguf.reader import GGML_TYPES
from relicllm.loader.gguf.quant_types import (
    GGUF_DENSE_TYPE_IDS,
    GGUF_TERNARY_FILE_TYPE_IDS,
)
from relicllm.triage.checkpoint import FORMAT_GGUF, CheckpointInventory
from relicllm.triage.keys import architecture_key

__all__ = ["Reachability", "reachability", "runtime_for_architecture"]


@dataclass(frozen=True)
class Reachability:
    """What stands between this checkpoint and a run. Every field is a note, not a rank."""

    adapter: str | None
    """The runtime's name from ``RUNTIMES``, or ``None`` when nothing declared reads it."""

    adapter_reason: str

    loader: bool
    loader_reason: str

    missing_kernel_types: tuple[str, ...]
    """GGUF block formats the checkpoint carries that no raw-block kernel consumes.

    Empty for safetensors, and empty for a GGUF whose types the dispatch table covers. A non-empty
    entry is a work item, not a verdict: the torch path can still dequantize these, more slowly.
    """

    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def needs_work(self) -> bool:
        return self.adapter is None or not self.loader or bool(self.missing_kernel_types)

    def work_items(self) -> tuple[str, ...]:
        """What a person would have to build, in the vocabulary of where it would be built.

        Named for the repository rather than for the symptom, because a kernel belongs in
        ``relic-core`` and an adapter belongs here, and the difference decides who picks it up.
        """
        items: list[str] = []
        if self.adapter is None:
            items.append("a backend adapter in relicllm/backends/ that declares this architecture")
        elif self.adapter == "torch":
            items.append(
                f"a per-architecture backend, for speed rather than feasibility: {self.adapter_reason}"
            )
        if self.missing_kernel_types:
            items.append(
                "raw-block kernels in relic-core for "
                + ", ".join(self.missing_kernel_types)
                + " (the torch path dequantizes them today, so this is a speed item)"
            )
        if not self.loader:
            items.append(self.loader_reason)
        return tuple(items)


def runtime_for_architecture(architecture: str, *, model_format: str) -> tuple[str | None, str]:
    """Which declared runtime reads this architecture, and on what grounds.

    The table is asked in its own order, so the answer cannot drift from what the dispatcher does.
    ``models`` is the architecture's own name in each runtime's spelling; every runtime now names
    the architecture it reads, so there is no catch-all left and an architecture no entry declares
    is reported as unserved rather than attributed to one that reads everything.

    **Both spellings are tried**, because the table's ``models`` entries are not all canonical keys.
    Xing4 declares ``("xing4_0",)`` -- the checkpoint's own ``model_type`` -- while the canonical key
    for that string is ``xing4``; asking with the key alone misses it and silently downgrades a model
    with a dedicated adapter, which is a wrong answer about the most actionable field this module
    reports.
    """
    key = architecture_key(architecture)
    candidates = {key, architecture.lower()}
    wrong_format: list[str] = []
    for name, runtime in RUNTIMES.items():
        if not candidates & set(runtime.models):
            continue
        if model_format not in runtime.model_formats:
            # This runtime knows the architecture but not in this format, so it is not the answer --
            # but it is not a *refusal* either, and returning here would be a lie about the
            # dispatcher. `capabilities.route()` skips exactly this runtime and keeps going, and so
            # must this: a DeepSeek-V4.1 GGUF has no `v41` adapter and no other runtime declares it,
            # so the report says so rather than naming a runtime that would misload it.
            wrong_format.append(f"{name} (only as {', '.join(runtime.model_formats)})")
            continue
        return name, f"{name} declares {sorted(candidates & set(runtime.models))[0]}"
    if wrong_format:
        return None, (
            f"{', '.join(wrong_format)} declares {architecture} but not as {model_format}, and no "
            f"runtime reads {model_format}"
        )
    return None, f"no declared runtime reads {model_format}"


def _missing_kernel_types(inventory: CheckpointInventory) -> tuple[str, ...]:
    """GGUF types the checkpoint carries that no *raw-block kernel* consumes.

    Three sets have to be told apart, and conflating any two of them produces a useless field:

    * ``GGML_TYPES`` is what the reader can interpret at all. A plain scalar type has one element per
      block, and this module's reason for reporting a type is that something has to unpack blocks
      rather than read a word -- so ``f32``, ``bf16`` and ``i64`` are not candidates whatever else
      is true of them. Reporting them, which a naive ``carried - GGUF_DENSE_TYPE_IDS`` does, puts
      "missing kernels" on nearly every GGUF in existence.
    * ``decodable_type_names()`` is what ``read_tensor`` will hand back. A type outside *that* is one
      the loader cannot even deliver, which is a harder problem than a missing kernel and is reported
      by :func:`_unaddressable_types` instead of being merged in here.
    * ``GGUF_DENSE_TYPE_IDS`` is the raw-block *runtime's* dispatch table: ten block formats, and a
      name in it is a claim that a kernel switches on its id.

    So the answer is block-quantized types that are not in the dispatch table -- which is the ternary
    pair today, and also `q3_k` and `q8_0` on the GLM GGUF. Those two are consumed by the torch path,
    by dequantizing, so they are *speed* items rather than feasibility ones; the ternary pair is
    neither consumed nor decoded, because ``read_tensor`` refuses it by name rather than silently
    upcasting to F16, which would cost ten times the memory and look like it had worked.
    """
    if inventory.format != FORMAT_GGUF:
        return ()
    geometry = {name: (elements, _bytes) for name, elements, _bytes in GGML_TYPES.values()}
    consumed = set(GGUF_DENSE_TYPE_IDS)
    missing: list[str] = []
    for name in sorted(inventory.bytes_by_dtype):
        if name in consumed:
            continue
        elements = geometry.get(name, (None, None))[0]
        if elements == 1:
            # A plain scalar type -- f32, bf16, i64. Nothing has to unpack it, so whether it is in
            # the *block* dispatch table is not a question worth asking about it. This is the check
            # that keeps ``bf16`` off the list for almost every GGUF ever written.
            continue
        missing.append(name)
    return tuple(missing)


def _unaddressable_types(inventory: CheckpointInventory) -> tuple[str, ...]:
    """Block formats the loader cannot even hand back bytes for -- a harder gap than a kernel.

    The set that answers this is ``decodable_type_names()``, read out of ``read_tensor``'s own
    dispatch. It is deliberately *not* ``GGUF_ADDRESSABLE_TYPE_NAMES``, which is the raw-block
    runtime's kernel table: a format can have no kernel and still decode perfectly, and calling that
    "its bytes cannot be reached at all" is a false alarm on a working file. GLM-5.2's Q2_K build
    carries ``q3_k`` and ``q8_0``, both of which decode, and both of which this note used to
    announce as unreachable one line under a note saying the torch path reads them.

    Plain scalar types need no exclusion here: they *are* in the reader's set, which is the point of
    asking the reader rather than a block-format table.
    """
    if inventory.format != FORMAT_GGUF:
        return ()
    from relicllm.loader.gguf.tensor_reader import decodable_type_names

    return tuple(sorted(set(inventory.bytes_by_dtype) - decodable_type_names()))


def reachability(inventory: CheckpointInventory) -> Reachability:
    """Gate 1 for one checkpoint. Free, so it runs on every checkpoint unconditionally."""
    adapter, adapter_reason = runtime_for_architecture(
        inventory.architecture or "", model_format=inventory.format
    )
    loader = inventory.format in ("safetensors", FORMAT_GGUF)
    loader_reason = (
        "a header reader exists for both formats this package understands"
        if loader
        else f"no reader for {inventory.format}"
    )

    missing = _missing_kernel_types(inventory)
    unaddressable = _unaddressable_types(inventory)
    notes: list[str] = []
    ternary = sorted(set(missing) & set(GGUF_TERNARY_FILE_TYPE_IDS))
    if ternary:
        notes.append(
            f"{', '.join(ternary)} is a ternary pack this fork addresses but does not interpret: "
            "the loader refuses it by name rather than upcasting, so a kernel is the whole of the work"
        )
    other = sorted(set(missing) - set(ternary))
    if other:
        notes.append(
            f"{', '.join(other)} has no raw-block kernel; the torch path dequantizes it, which works "
            "and costs the expansion"
        )
    if unaddressable:
        notes.append(
            f"{', '.join(unaddressable)} is not in the loader's addressable set, so its bytes cannot "
            "be reached at all until a reader knows its geometry"
        )
    if adapter is None:
        notes.append(
            "no backend adapter: the model can be loaded and run by hand, but not served through "
            "relicllm's request path until one exists"
        )
    if inventory.unknown:
        notes.extend(inventory.unknown)

    return Reachability(
        adapter=adapter,
        adapter_reason=adapter_reason,
        loader=loader,
        loader_reason=loader_reason,
        missing_kernel_types=missing,
        notes=tuple(notes),
    )
