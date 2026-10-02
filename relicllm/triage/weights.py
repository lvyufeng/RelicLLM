"""The precision ladder, and the difference between a file's bytes and the bytes a card ends up with.

Two rules, both learned the hard way, and this module exists to hold them.

**A label is not a size.** "Official FP8 release" and "smaller than native" are different claims and
the first does not imply the second. DeepSeek-V4-Flash's ``w8a8`` directory is **1.835x larger** than
the native checkpoint -- 292.9 GiB against 159.6 GiB, at an identical 69,187 tensors -- because it
quantizes activations too and stores them alongside. So the ladder is built by *measuring sibling
artifacts*, header by header, and never by reading a directory name for a number.

**A file's bytes are not the card's bytes.** ``sm_75`` has no FP8 or FP4 tensor core, so those
formats are storage-only and the arithmetic runs after a conversion. Which conversion, and therefore
which factor, is a property of the runtime and not of the format, and this tree contains both
answers:

* MiMo's routed experts are MXFP4 and are **never expanded** -- the packed ``[N, K/2]`` uint8 array
  beside its ``[N, K/32]`` E8M0 scale is the ABI the kernels already take
  (``relicllm/models/mimo_v2/loader.py:17``), so 4 bits stay 4 bits.
* DeepSeek's FP4 experts are **doubled**: ``_convert_fp4_to_int8`` dequantizes to bf16 and
  requantizes to INT8 (``relicllm/models/deepseek_v4/loader.py:141``), because the INT8 kernels are
  what this hardware has.
* MiMo's FP8 *dense* tiles are doubled too, dequantized to the compute dtype at load
  (``relicllm/models/mimo_v2/loader.py:414``).

A rule table with a ``file:line`` on every entry is what keeps those three from being one guess. A
combination that is not in the table gets a factor of **1.0 and an ``assumed`` note** rather than a
plausible-looking number, because the direction of the error is not knowable in general and a fit is
the wrong place to find out.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from relicllm.triage.checkpoint import (
    FORMAT_GGUF,
    ROLE_ATTENTION,
    ROLE_DENSE_MLP,
    ROLE_EMBEDDING,
    ROLE_LOOKUP_TABLE,
    ROLE_ROUTED_EXPERT,
    CheckpointInventory,
    read_inventory,
)
from relicllm.triage.keys import architecture_key
from relicllm.triage.kv import Confidence

__all__ = [
    "DIRECTORY_QUANT_SUFFIXES",
    "ExpansionRule",
    "PrecisionOption",
    "expansion_for",
    "on_card_bytes",
    "precision_ladder",
    "sibling_artifacts",
]


@dataclass(frozen=True)
class ExpansionRule:
    """How much bigger one family of tensors gets between the file and the card."""

    factor: float
    reason: str
    source: str
    confidence: Confidence
    requires_block_scale: str | None = None
    """A scale dtype the same role must *also* carry for this rule to apply.

    The licence to have two rules under one key. Packed FP4 and true INT8 both arrive as
    ``I8`` tensors and differ by a factor of two on a card, so the dtype alone cannot separate
    them; what does is the scale that rides beside the payload -- an MXFP4/F8_E8M0 block scale on
    one, a per-row F32 scale on the other. ``None`` means the rule applies on the dtype alone.
    """

    @property
    def expands(self) -> bool:
        return self.factor != 1.0

    def describe(self) -> str:
        return f"x{self.factor:g} -- {self.reason} ({self.source})"


#: ``(architecture_key, role, storage_dtype)`` -> the rules that may apply, first match winning.
#: ``None`` in a slot means *any*.
#:
#: Every entry is a reading of the loader that has to consume the bytes, cited so a disagreement is
#: settled by opening a file rather than by argument. The ABSENT combinations are as deliberate as
#: the present ones: there is no entry for a GGUF's ``iq*``/``q*`` types because what those expand
#: to depends on a kernel's own internal representation, which a header cannot see.
#:
#: **A key that cannot fire is worse than a missing one.** ``I8`` and ``U8`` are the same width and
#: the wrong spelling is invisible: the table looks like it covers a family, the lookup falls to the
#: permissive default, and the report says *assumed* on a case somebody thought they had answered.
#: The four DeepSeek entries below were keyed on ``U8`` and had never once been reached -- this
#: roster's safetensors files carry the packed payload as ``I8``, and ``U8`` appears only in MiMo.
_EXPANSION_RULES: dict[tuple[str, str | None, str], tuple[ExpansionRule, ...]] = {
    ("mimo", ROLE_ROUTED_EXPERT, "U8"): (
        ExpansionRule(
            factor=1.0,
            reason="MXFP4 expert weights are consumed verbatim by the fp4 MoE kernel and never expanded",
            source="relicllm/models/mimo_v2/loader.py:17",
            confidence=Confidence.DERIVED,
        ),
    ),
    ("mimo", None, "F8_E4M3"): (
        ExpansionRule(
            factor=2.0,
            reason="FP8 block weights are dequantized to the compute dtype at load",
            source="relicllm/models/mimo_v2/loader.py:414",
            confidence=Confidence.DERIVED,
        ),
    ),
    ("deepseek_v4", ROLE_ROUTED_EXPERT, "I8"): (
        # Only the packed form doubles. The `w8a8` sibling carries true INT8 experts -- 264.00 GiB
        # of them, against this release's 132.00 -- and those are copied through unchanged, so the
        # same dtype has to answer differently and the E8M0 block scale is what tells them apart.
        ExpansionRule(
            factor=2.0,
            reason=(
                "packed FP4 experts are requantized to INT8, and the arena the CPU expert cache "
                "materializes holds every expert of a layer, so the bank is twice the packed codes"
            ),
            source=(
                "relicllm/models/deepseek_v4/loader.py:141, "
                "relicllm/components/moe/cpu_backend.py:629"
            ),
            confidence=Confidence.DERIVED,
            requires_block_scale="F8_E8M0",
        ),
        ExpansionRule(
            factor=1.0,
            reason="already INT8 in the file, so the arena copies it through without a widening",
            source="relicllm/models/deepseek_v4/loader.py:373",
            confidence=Confidence.DERIVED,
        ),
    ),
    ("deepseek_v4_1", ROLE_ROUTED_EXPERT, "I8"): (
        # V4.1 is *not* V4 here and the difference is the whole reason both are written out rather
        # than sharing one entry. `CheckpointRoutedExperts` reads packed rows out of the mapping and
        # expands them into a bounded `LINEAR_DTYPE` cache, so what has to be reachable -- on the
        # card or behind a bank -- is the packed file, not a widened copy of it. V4 materializes
        # every expert at INT8 in host RAM; V4.1 does not, and asserting the same 2x would put
        # 275.67 GiB of experts on the books as 520 GiB of host memory that nothing ever allocates.
        ExpansionRule(
            factor=1.0,
            reason=(
                "packed FP4 rows are streamed from the checkpoint and expanded into a bounded "
                "bf16 cache, so the stored bytes are what must be reachable"
            ),
            source="relicllm/models/deepseek_v4_1/loader.py:407",
            confidence=Confidence.DERIVED,
        ),
    ),
    ("deepseek_v4", ROLE_ATTENTION, "F8_E4M3"): (
        ExpansionRule(
            factor=2.0,
            reason="FP8 attention tiles are dequantized to bf16 before the GEMM on this hardware",
            source="relicllm/models/deepseek_v4/loader.py:146",
            confidence=Confidence.DERIVED,
        ),
    ),
    ("deepseek_v4", ROLE_DENSE_MLP, "F8_E4M3"): (
        ExpansionRule(
            factor=2.0,
            reason="FP8 dense tiles are dequantized to bf16 before the GEMM on this hardware",
            source="relicllm/models/deepseek_v4/loader.py:385",
            confidence=Confidence.DERIVED,
        ),
    ),
    ("deepseek_v4_1", ROLE_ATTENTION, "F8_E4M3"): (
        ExpansionRule(
            factor=2.0,
            reason=(
                "`V41Checkpoint.weight(key, dtype=LINEAR_DTYPE)` dequantizes fp8 and casts to the "
                "parameter's width, and this backbone builds every attention linear in bf16"
            ),
            source="relicllm/models/deepseek_v4_1/loader.py:257, relicllm/models/deepseek_v4_1/attention.py:99",
            confidence=Confidence.DERIVED,
        ),
    ),
    ("deepseek_v4_1", ROLE_DENSE_MLP, "F8_E4M3"): (
        ExpansionRule(
            factor=2.0,
            reason=(
                "the shared expert is built in LINEAR_DTYPE and is fp8 in the checkpoint, so it "
                "widens in the same call the attention tiles do"
            ),
            source="relicllm/models/deepseek_v4_1/modules.py:487",
            confidence=Confidence.DERIVED,
        ),
    ),
}

#: A directory suffix that names a quantized release of the model beside it. Used only to *find*
#: artifacts to measure -- never to decide what they are, which is what the header read is for.
DIRECTORY_QUANT_SUFFIXES = (
    "-fp8", "-fp4", "-nvfp4", "-mxfp4", "-int8", "-int4", "-w8a8", "-w4a16", "-awq", "-gptq",
    "-gguf", "-bf16", "-fp16",
)

#: The file's own dtype, for a release whose *format* is GGUF: the quant is in the type name, not in
#: a directory name, so this path reports it as the label.
_GGUF_DTYPE_LABELS = {
    "iq4_nl": "GGUF IQ4_NL", "q4_k": "GGUF Q4_K", "q5_k": "GGUF Q5_K", "q6_k": "GGUF Q6_K",
    "q8_0": "GGUF Q8_0", "iq2_xxs": "GGUF IQ2_XXS", "iq2_xs": "GGUF IQ2_XS",
    "iq3_xxs": "GGUF IQ3_XXS", "iq1_m": "GGUF IQ1_M", "iq4_xs": "GGUF IQ4_XS",
    "ptq1_0": "GGUF PTQ1_0 (ternary)", "pq2_0": "GGUF PQ2_0 (ternary)",
    "f16": "GGUF F16", "bf16": "GGUF BF16", "f32": "GGUF F32",
}


def _block_scales(inventory: CheckpointInventory, role: str) -> frozenset[str]:
    """The dtypes of the scale tensors riding beside this role's payload."""
    return frozenset(inventory.bytes_by_role_and_dtype.get(role, {}))


def expansion_for(inventory: CheckpointInventory, *, role: str, dtype: str) -> ExpansionRule:
    """The on-card multiplier for one family, or 1.0 with the reason it had to be assumed."""
    arch = architecture_key(inventory.architecture or "")
    scales = _block_scales(inventory, role)
    for key in ((arch, role, dtype), (arch, None, dtype), (arch, role, None)):
        rules = _EXPANSION_RULES.get(key)  # type: ignore[arg-type]
        if rules is None:
            continue
        for rule in rules:
            # First match wins, so a conditional rule must come before the unconditional one it
            # refines -- which is how `I8` answers 2x for packed FP4 and 1x for true INT8.
            if rule.requires_block_scale is None or rule.requires_block_scale in scales:
                return rule
    return ExpansionRule(
        factor=1.0,
        reason=(
            f"no expansion rule for {arch or 'an unrecognised architecture'}/{role}/{dtype}; the "
            "file's bytes are used as they are, which understates a format this hardware converts"
        ),
        source="assumed",
        confidence=Confidence.ASSUMED,
    )


def on_card_bytes(
    inventory: CheckpointInventory,
) -> tuple[dict[str, int], dict[str, ExpansionRule]]:
    """Per-role bytes once the runtime has converted them, plus the rule applied to each family.

    Priced over the checkpoint's own ``(role, dtype)`` crossings rather than against a single
    dominant dtype, because the dominant one is often not the one that converts. MiMo is the case
    that shows why: 149.8 of its 161 GiB are the experts' packed ``U8``, which a kernel consumes
    verbatim, and the 3.6 GiB that actually double are FP8 attention tiles -- a per-role lookup
    against the global dominant would price those as ``U8`` and quietly lose the doubling.

    Returning the rules rather than only the total is the point: a report has to be able to say *why*
    a 4-bit checkpoint is costed at 8 bits, or the number reads as an error.
    """
    arch = architecture_key(inventory.architecture or "")
    rules: dict[str, ExpansionRule] = {}
    per_role: dict[str, int] = {}
    for role, by_dtype in inventory.bytes_by_role_and_dtype.items():
        converted = 0
        for dtype, size in by_dtype.items():
            rule = expansion_for(inventory, role=role, dtype=dtype)
            rules[f"{arch}/{role}/{dtype}".lstrip("/")] = rule
            converted += int(size * rule.factor)
        per_role[role] = converted
    return per_role, rules


def _dominant_dtype(inventory: CheckpointInventory) -> str:
    if not inventory.bytes_by_dtype:
        return ""
    return max(inventory.bytes_by_dtype.items(), key=lambda item: item[1])[0]


@dataclass(frozen=True)
class PrecisionOption:
    """One artifact's precision, measured rather than labelled."""

    label: str
    path: str
    format: str
    precision: str
    """What the header says it is -- a dtype histogram's dominant entry, or the GGUF type."""

    bytes_total: int
    on_card_bytes: int
    resident_on_card_bytes: int
    """The on-card bytes that cannot be banked: everything but routed experts and lookup tables."""

    confidence: Confidence
    evidence: str
    basis: tuple[str, ...] = field(default_factory=tuple)
    """The expansion rules that were applied, in words. Empty means nothing was converted."""

    unknown: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def gib(self) -> float:
        return self.bytes_total / 1024 ** 3

    @property
    def resident_gib(self) -> float:
        return self.resident_on_card_bytes / 1024 ** 3


_QUANT_SIDE_SUFFIXES = (
    "weight_scale_inv",
    "weight_scale",
    "weight_global_scale",
    "input_global_scale",
    "weight_packed",
    "weight",
    "scales",
    "scale",
)
"""The parts of a tensor name a quantization owns: the payload, and the scales beside it.

``Qwen3.8-27B-FP8`` and ``Qwen3.8-27B-NVFP4`` are the same model, and comparing their names directly
says they are not: NVFP4 stores ``...down_proj.weight_packed`` where FP8 stores ``...down_proj.weight``,
calls the per-block scale ``weight_scale`` where FP8 calls it ``weight_scale_inv``, and adds two
global scales FP8 does not have. Stripping these takes the pair from 0.405 to 0.974 while every pair
of genuinely different models stays at 0.014 or below.
"""

_CHECKPOINT_MARKERS = ("config.json",)


def _tensor_base(name: str) -> str:
    """A tensor name with its quantization-side suffixes removed."""
    parts = name.split(".")
    while parts and parts[-1] in _QUANT_SIDE_SUFFIXES:
        parts.pop()
    return ".".join(parts)


def _structural_base(name: str) -> str:
    """A tensor name reduced to what names a *model* rather than a release's packing of it.

    Two things come off, each a freedom the exporting tool has:

    * the quantization-side suffixes (:func:`_tensor_base`), because FP8 and NVFP4 spell the scale
      tensors differently;
    * every all-digit segment, because one release keeps its experts **stacked** -- a single
      ``...mlp.experts.gate_up_proj`` per layer -- where another writes them out one at a time as
      ``...mlp.experts.0.gate_proj``. Layer indices go the same way: every layer carries the same
      names at every depth, so dropping the index loses nothing a model identity needs.

    Measured, over the whole box: Qwen3.8-Flash-Next against its own FP8 export rises from 0.020 to
    0.970 -- the *entire* gap is the stacked-versus-split expert layout. Both sides reduce to about
    228 distinct structural names and they agree on 225 of them; the five that differ are
    ``experts.gate_up_proj`` on the stacked side, ``experts.gate_proj`` and ``experts.up_proj`` on the
    split one, and one per-layer-embedding name the FP8 export carries and the native does not. Every
    other pair of artifacts that shares an architecture key is unchanged or falls, the next highest
    being 0.560 between two different 27B training runs of one family.
    """
    return ".".join(part for part in _tensor_base(name).split(".") if not part.isdigit())


def same_model(left: CheckpointInventory, right: CheckpointInventory) -> bool:
    """Whether two artifacts' headers prove they are the same model at different precisions.

    Measured, not named. A directory name is how a candidate is *found*; this is what decides whether
    it may be reported, and it is deliberately hard to pass. The architecture key alone is not
    enough: Bonsai's GGUF and Qwen3.8-27B both declare ``qwen35``, so the key pairs a ternary Bonsai
    with an FP8 Qwen and would file one model's byte count under the other's name -- the worst answer
    this report can give. The tensor-name sets separate them cleanly where nothing else does:
    quantizations of one model overlap at 0.97-1.00 after normalisation, unrelated models at 0.014 --
    Bonsai against Qwen3.8-27B at 0.000, because a GGUF renames the entire tree rather than
    repacking it.

    **What this cannot separate**, and why the threshold is where it is: two training runs of one
    family at one size -- ``Qwen3.8-27B-DFlash2`` against ``-DSpark``, two *different* models -- name
    their tensors identically and land at 0.560. Nothing in a header distinguishes them, so this
    function says ``True`` and is wrong. It is wrong at the same rate with and without the digit
    stripping; the stripping is what lets the FP8 export through, and the architecture key is what
    keeps the mistake inside one family.
    """
    if architecture_key(left.architecture or "") != architecture_key(right.architecture or ""):
        return False
    if not left.tensor_names or not right.tensor_names:
        return False
    left_names = {_structural_base(name) for name in left.tensor_names}
    right_names = {_structural_base(name) for name in right.tensor_names}
    overlap = len(left_names & right_names) / len(left_names | right_names)
    return overlap >= 0.5


def sibling_artifacts(path: str) -> tuple[str, ...]:
    """Other releases of the same model sitting beside ``path``.

    This is a *search*, not a verdict: everything it returns is measured by :func:`read_inventory`
    and reported only if :func:`same_model` accepts it. Names are used to find candidates because
    there is no registry of releases, and one of four shapes is what a downloaded second precision
    actually looks like here:

    * **A file.** ``Ternary-Bonsai-2-27B-PTQ1_0.gguf`` beside ``...-PQ2_0.gguf`` -- a directory of
      free-standing GGUF quants, so the family is the files. A file that is one *shard* of a quant
      is the same case one level up: ``GLM-5.2-UD-Q2_K_XL-00003-of-00007.gguf`` resolves to the
      seven-file release it belongs to, and the releases beside *that* are the alternatives. Getting
      this wrong is loud rather than quiet -- the six sibling shards each resolve back to the same
      seventeen-file bundle and the ladder prints seven identical rows -- but it is still wrong, and
      the one precision worth seeing never appears.
    * **A directory of quant directories.** ``GLM-5.2-GGUF/UD-Q2_K_XL`` beside ``UD-Q4_K_M``; the
      caller may point at either level. Sidecars like ``.sha256`` sit in the same directory and do
      not make it a checkpoint, so the test is what a checkpoint *has* rather than what surrounds it.
    * **A checkpoint directory.** ``Qwen3.8-27B-FP8`` is a complete model in its own right, and its
      variants are directories beside it -- which means the search has to look *outside* it. That is
      the case the previous version of this function missed, and it missed every one of them.
    * **A sharded quant.** ``GLM-5.2-GGUF/UD-Q2_K_XL`` holds the shards of one quant, not alternatives
      to each other; the alternatives are the sibling directories one level up.
    """
    root = os.path.abspath(path)

    if os.path.isfile(root):
        if not root.endswith(".gguf"):
            return ()
        from relicllm.loader.gguf.bundle import resolve_gguf_bundle

        if len(resolve_gguf_bundle(root)) > 1:
            # One shard of a set. The thing with alternatives is the *release* it belongs to, whose
            # directory is the one this file sits in -- `.../UD-Q2_K_XL/...-00003-of-00007.gguf` --
            # so the search moves up one level and looks across that directory's siblings.
            holding = os.path.dirname(root)
            return _quant_directories_beside(os.path.dirname(holding), os.path.basename(holding))
        parent = os.path.dirname(root)
        return tuple(
            os.path.join(parent, name)
            for name in sorted(os.listdir(parent))
            if name.endswith(".gguf") and name != os.path.basename(root)
        )

    entries = sorted(os.listdir(root))
    ggufs = [name for name in entries if name.endswith(".gguf")]
    if ggufs:
        from relicllm.loader.gguf.bundle import resolve_gguf_bundle

        standalone = [len(resolve_gguf_bundle(os.path.join(root, name))) == 1 for name in ggufs]
        if len(ggufs) > 1 and all(standalone):
            return tuple(os.path.join(root, name) for name in ggufs)
        return _quant_directories_beside(os.path.dirname(root), os.path.basename(root))

    if not _holds_a_checkpoint(root, entries):
        return tuple(
            os.path.join(root, name)
            for name in entries
            if os.path.isdir(os.path.join(root, name))
        )
    return _quant_directories_beside(os.path.dirname(root), os.path.basename(root))


def _holds_a_checkpoint(root: str, entries: list[str]) -> bool:
    """Whether this directory is a checkpoint rather than a crate of them.

    ``config.json`` or a tensor file settles it. The alternative -- "every entry is a directory" --
    was how this used to be decided, and it is decided wrong by any sidecar: ``GLM-5.2-GGUF`` holds
    ``UD-Q2_K_XL`` and ``UD-Q4_K_M`` beside their ``.sha256`` files, so the directory of quants read
    as neither a checkpoint nor a crate and its two quants were never compared.
    """
    return any(
        name.endswith((".gguf", ".safetensors")) or name in _CHECKPOINT_MARKERS for name in entries
    )


def _quant_directories_beside(parent: str, stem: str) -> tuple[str, ...]:
    """Directories next to ``stem`` that are plausibly another precision of the same model.

    Three ways in, and only the last one is a name:

    * a name carrying a quantization suffix -- ``...-FP8`` beside ``...-NVFP4``;
    * a directory holding any ``.gguf``, which is what carries the GGUF releases: ``GLM-5.2-GGUF``
      holds ``UD-Q2_K_XL`` beside ``UD-Q4_K_M``, and neither name carries a suffix any list would
      contain -- evidence a name never is;
    * the *base* name this one was derived from, when ``stem`` itself ends in a suffix. Without it
      the search runs one way only: pointing at ``Qwen3.8-27B-FP8`` finds the NVFP4 build and never
      the checkpoint both were quantized from, whose directory carries no suffix to be found by and
      no GGUF to hold.

    What is *accepted* is still decided by the headers in :func:`precision_ladder`, never here.
    """
    if not os.path.isdir(parent):
        return ()
    found: list[str] = []
    stem_lower = stem.lower()
    for suffix in DIRECTORY_QUANT_SUFFIXES:
        if not stem_lower.endswith(suffix):
            continue
        base = stem[: len(stem) - len(suffix)]
        candidate = os.path.join(parent, base)
        # Longest match first would be tidier, but collecting is enough: `-NVFP4` also ends in
        # `-fp4`, and only one of the two candidate names can exist on disk.
        if base and os.path.isdir(candidate) and candidate not in found:
            found.append(candidate)
    for name in sorted(os.listdir(parent)):
        if name == stem:
            continue
        candidate = os.path.join(parent, name)
        if not os.path.isdir(candidate):
            continue
        named = any(suffix in name.lower() for suffix in DIRECTORY_QUANT_SUFFIXES)
        # A directory that cannot be listed is not a candidate; `lost+found` is mode 700 and sits
        # beside checkpoints on this box, so an unguarded listing turns a search into a crash.
        try:
            entries = os.listdir(candidate)
        except OSError:
            continue
        holds_gguf = any(entry.endswith(".gguf") for entry in entries)
        if (named or holds_gguf) and candidate not in found:
            found.append(candidate)
    return tuple(found)


def _unreadable(path: str, label: str, error: BaseException) -> PrecisionOption:
    """A release whose headers will not read. Reported as such, never as zero bytes."""
    return PrecisionOption(
        label=label,
        path=os.path.abspath(path),
        format="unknown",
        precision="unreadable",
        bytes_total=0,
        on_card_bytes=0,
        resident_on_card_bytes=0,
        confidence=Confidence.ASSUMED,
        evidence=f"{type(error).__name__}: {error}",
        unknown=(f"{os.path.abspath(path)} could not be read: {error}",),
    )


def _label_for(inventory: CheckpointInventory, path: str) -> str:
    """What to call this artifact on a report: its own quant, or ``native``.

    For a GGUF the quant is in the tensor types, because the file *is* the artifact; for a
    safetensors release it is in the directory name, because the header has no field for it. Nothing
    downstream reads this string for a number -- the bytes come from the header either way -- which
    is what lets a label stay a label instead of becoming evidence.
    """
    if inventory.format == FORMAT_GGUF:
        dominant = _dominant_dtype(inventory)
        return _GGUF_DTYPE_LABELS.get(dominant, f"GGUF {dominant}")
    name = os.path.basename(path).lower()
    for suffix in DIRECTORY_QUANT_SUFFIXES:
        if suffix in name:
            return suffix.lstrip("-").upper()
    return "native"


def _option_for(path: str, label: str | None = None) -> PrecisionOption:
    try:
        inventory = read_inventory(path)
    except (FileNotFoundError, ValueError, OSError) as error:
        return _unreadable(path, label or os.path.basename(os.path.abspath(path)), error)

    if label is None:
        label = _label_for(inventory, path)
    per_role, rules = on_card_bytes(inventory)
    total = sum(per_role.values())
    offload = sum(per_role.get(role, 0) for role in (ROLE_ROUTED_EXPERT, ROLE_LOOKUP_TABLE))
    conversion = tuple(
        f"{name}: {rule.describe()}" for name, rule in sorted(rules.items()) if rule.expands
    )
    unknown: list[str] = list(inventory.unknown)
    # Reported by *storage format* rather than by (role, dtype) pair: the pairs run to dozens and
    # name the same three or four formats over and over. What a reader needs is "these formats have
    # no conversion rule here", which is one line, plus which way the resulting error points.
    assumed = sorted({name.rsplit("/", 1)[-1] for name, rule in rules.items() if rule.confidence is Confidence.ASSUMED})
    if assumed:
        unknown.append(
            "no on-card conversion rule for " + ", ".join(assumed) + " on this hardware, so the "
            "file's bytes are used as they are; a format the runtime converts would be larger"
        )
    return PrecisionOption(
        label=label,
        path=os.path.abspath(path),
        format=inventory.format,
        precision=_dominant_dtype(inventory),
        bytes_total=inventory.total_bytes,
        on_card_bytes=total,
        resident_on_card_bytes=total - offload,
        confidence=Confidence.MEASURED,
        evidence=f"header inventory of {os.path.abspath(path)}: {inventory.tensor_count} tensors",
        basis=conversion,
        unknown=tuple(unknown),
    )


def precision_ladder(path: str, *, include_siblings: bool = True) -> tuple[PrecisionOption, ...]:
    """Every precision of this model that is on the box, measured, largest number of them first.

    The artifact the caller named is the first entry whatever happens -- including when it will not
    read, in which case it is reported unreadable rather than as zero bytes. Siblings are searched by
    name and then *kept only if their headers prove they are the same model*
    (:func:`same_model`): a release directory sitting beside another is not evidence that they are
    the same model, and reporting an unrelated checkpoint's byte count under this model's name would
    be the worst kind of answer.
    """
    primary = _option_for(path)
    if not include_siblings:
        return (primary,)

    try:
        primary_inventory = read_inventory(path)
    except (FileNotFoundError, ValueError, OSError):
        return (primary,)

    options = [primary]
    seen = {primary.path}
    for sibling in sibling_artifacts(path):
        resolved = os.path.abspath(sibling)
        if resolved in seen:
            continue
        seen.add(resolved)
        option = _option_for(sibling)
        if option.precision == "unreadable":
            continue
        try:
            sibling_inventory = read_inventory(sibling)
        except (FileNotFoundError, ValueError, OSError):
            continue
        if not same_model(primary_inventory, sibling_inventory):
            continue
        options.append(option)
    return tuple(options)
