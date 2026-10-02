"""Read a checkpoint's headers and answer the one question a fit test needs: how many bytes, of what.

**No weights are read.** For safetensors this is each shard's 8-byte length plus its JSON header
(:meth:`relicllm.loader.safetensors.MmapSafetensors._index_headers`); for GGUF it is the metadata
block plus the tensor table. A 475 GiB checkpoint opens in under a second, which is what makes the
static gate free and therefore unconditional.

The split that matters is **resident versus offloadable**, not the file's byte total. Routed experts
are the part a runtime can stage from host memory -- the 137-458 GiB banks in ``/dev/shm`` -- so a
model whose expert bytes exceed the cards can still fit. Everything else (attention, embeddings,
norms, shared experts) has to be somewhere the compute can reach on every step. A tool that reports
one byte total and calls it a fit is answering the wrong question for every MoE in this roster.

The byte total is also not the on-card total, and the two differ in *both* directions. A published
"official quant" can be larger than the checkpoint it quantizes -- DeepSeek-V4-Flash's ``w8a8``
directory is 1.835x the native one at an identical tensor count -- so the ladder this feeds must be
measured per artifact and never inferred from a label. In the other direction ``sm_75`` has no
FP8/FP4 tensor cores, so those formats are storage-only and the kernels dequantize before the GEMM;
DeepSeek's loader goes further and requantizes official FP4 experts to INT8, at
``relicllm/models/deepseek_v4/loader.py:141``, which *doubles* the bytes they occupy on the card
relative to how they sit on disk.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Any, Mapping

from relicllm.triage.keys import (
    architecture_of,
    flatten_gguf,
    int_at,
)

__all__ = [
    "CheckpointInventory",
    "FORMAT_GGUF",
    "FORMAT_SAFETENSORS",
    "OFFLOADABLE_ROLES",
    "ROLE_ATTENTION",
    "ROLE_DENSE_MLP",
    "ROLE_EMBEDDING",
    "ROLE_LOOKUP_TABLE",
    "ROLE_OTHER",
    "ROLE_ROUTED_EXPERT",
    "checkpoint_paths",
    "read_inventory",
    "role_of",
]


FORMAT_SAFETENSORS = "safetensors"
FORMAT_GGUF = "gguf"

ROLE_EMBEDDING = "embedding"
ROLE_ROUTED_EXPERT = "routed_expert"
ROLE_LOOKUP_TABLE = "lookup_table"
ROLE_ATTENTION = "attention"
ROLE_DENSE_MLP = "dense_mlp"
ROLE_OTHER = "other"

#: The roles a runtime can put in host memory and reach from there, rather than keeping resident on
#: every rank. They earn it for *different* reasons and a report should not merge them:
#:
#: * a routed expert bank is worth offloading because the router touches a fraction of it per token
#:   -- the bytes are there, the traffic is not;
#: * a lookup table is worth offloading because it is a gather, not a matmul, so the host can hold
#:   the whole thing and hand back rows. DeepSeek-V4.1's two Engram tables are 189.1 GiB, one
#:   ``layers.N.engram.embed.weight`` of ``(384006168, 256)`` per table, which is 39% of a 475 GiB
#:   checkpoint and the single largest thing in it -- larger than all its routed experts together
#:   only slightly, and 11x larger than everything else combined. A tool that folded this into
#:   "dense" would call V4.1 impossible on arithmetic that is 189 GiB wrong.
OFFLOADABLE_ROLES = frozenset({ROLE_ROUTED_EXPERT, ROLE_LOOKUP_TABLE})

#: Name fragments that mark a tensor as belonging to a routed expert bank, in the spellings the
#: GGUF and HuggingFace checkpoints in this roster use. Checked before anything else, because a
#: routed expert's name also contains ``mlp`` and would otherwise land in the dense bucket.
#:
#: The leading dots on ``.experts.`` are load-bearing. Without them ``ffn.shared_experts.w1``
#: matches, and a *shared* expert is kept resident by definition -- it is the part of a MoE layer
#: every token goes through. Misclassifying it moves bytes out of the resident total, which is the
#: direction that makes a model look like it fits.
_ROUTED_MARKERS = (
    "ffn_gate_exps", "ffn_down_exps", "ffn_up_exps",  # GGUF, DeepSeek/MiniMax/Xing4
    ".experts.",  # HF and V4.1's own layout, which spells it ``ffn.experts.N.w1``
    "expert_w1", "expert_w2", "expert_w3",
    ".moe_mlp.experts",  # Qwen MoE
)

#: The gather tables. Checked before :data:`_EMBEDDING_MARKERS` because their names contain
#: ``embed`` but they are not the input embedding -- different size class, different placement, and
#: in the one checkpoint that has them, the difference is 183 GiB.
_LOOKUP_TABLE_MARKERS = (
    "engram.embed",           # DeepSeek-V4.1's Engram tables, 189.1 GiB
    "ngram_embedding",        # Qwen3.8-Flash-Next's PLE tables, 95.4 GiB, read by HostNGramTable
    "ple_embedding",
    "wte_table", "n_gram_embed",
)

_EMBEDDING_MARKERS = (
    "embed_tokens", "token_embd", "wte.", "wpe.", "lm_head", "output.weight", "output_norm",
    "word_embeddings", "embed.weight", ".embed.", "head.weight",
)

_ATTENTION_MARKERS = (
    "self_attn", ".attn.", "attn_", "attention", "linear_attn", "ssm_", "mamba", "moe_gate",
    "ffn_gate_inp", "gate.weight", "e_score_correction",
)

_DENSE_MLP_MARKERS = ("mlp.", "ffn_", "feed_forward", "shared_experts", "shexp", "dense_mlp")


def role_of(name: str) -> str:
    """Which placement unit one tensor belongs to, from its name.

    Deliberately a substring test rather than a per-architecture table: names differ across the two
    formats and nine architectures, and a table would need an entry per pair. The order is what
    carries the meaning -- routed experts first because their names also contain ``mlp``, attention
    before dense because a MoE router's name also contains ``ffn``, and attention before embedding
    because ``output.weight`` is a marker for a GGUF's ``lm_head`` and also a substring of
    ``blk.N.attn_output.weight``, which is that layer's output projection and not an embedding at
    all. Measured over seven real checkpoints, those 41, 79 and 16 names are the *only* ones that
    match both lists, so attention-first is a total fix rather than a reordering that trades one
    misclassification for another.
    """
    lowered = name.lower()
    if any(marker in lowered for marker in _LOOKUP_TABLE_MARKERS):
        return ROLE_LOOKUP_TABLE
    if any(marker in lowered for marker in _ROUTED_MARKERS):
        return ROLE_ROUTED_EXPERT
    if any(marker in lowered for marker in _ATTENTION_MARKERS):
        return ROLE_ATTENTION
    if any(marker in lowered for marker in _EMBEDDING_MARKERS):
        return ROLE_EMBEDDING
    if any(marker in lowered for marker in _DENSE_MLP_MARKERS):
        return ROLE_DENSE_MLP
    return ROLE_OTHER


@dataclass(frozen=True)
class CheckpointInventory:
    """What a checkpoint's headers say, and nothing that required reading a weight."""

    path: str
    format: str
    """``safetensors`` | ``gguf``."""

    architecture: str | None
    """As the checkpoint spells it -- see :func:`relicllm.triage.keys.architecture_of`."""

    config: Mapping[str, Any]
    """``config.json``, or GGUF metadata flattened to bare keys (:func:`~.keys.flatten_gguf`)."""

    tensor_names: tuple[str, ...]

    elements_by_tensor: Mapping[str, int]
    """Logical element count per tensor, from the header's own shapes. Exact for both formats: a
    GGUF block-quantized tensor reports its elements, not its packed bytes."""

    bytes_by_role: Mapping[str, int]

    bytes_by_role_and_dtype: Mapping[str, Mapping[str, int]]
    """The two histograms *crossed*, which is the granularity a conversion rule needs.

    A global histogram cannot answer "what does MiMo's attention cost after loading" -- MiMo's
    dominant dtype is the experts' packed ``U8`` at 149.8 GiB of 161, so a per-role lookup against
    the global dominant would price its 3.6 GiB of FP8 attention tiles as if they were the experts'
    verbatim-consumed format and miss that the loader doubles them. One extra dict avoids that.
    """

    bytes_by_dtype: Mapping[str, int]
    """The file's own dtype names -- ``BF16``/``F8_E4M3`` for safetensors, GGML type names for GGUF.
    This *is* the precision ladder, and it is read rather than inferred from a directory name."""

    tensor_count: int
    total_bytes: int

    unknown: tuple[str, ...] = ()
    """What could not be read. A list rather than a default, so a report can never print a guess as
    a measurement."""

    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_gib(self) -> float:
        return self.total_bytes / 1024 ** 3

    @property
    def routed_expert_bytes(self) -> int:
        return int(self.bytes_by_role.get(ROLE_ROUTED_EXPERT, 0))

    @property
    def lookup_table_bytes(self) -> int:
        return int(self.bytes_by_role.get(ROLE_LOOKUP_TABLE, 0))

    @property
    def offloadable_bytes(self) -> int:
        """Everything a host-memory bank can hold: routed experts plus gather tables."""
        return int(sum(self.bytes_by_role.get(role, 0) for role in OFFLOADABLE_ROLES))

    @property
    def resident_bytes(self) -> int:
        """Bytes that have to be somewhere the compute reaches on every step.

        Not a lower bound on what a card must hold -- the offloadable part still has to be *staged*
        from somewhere, and a bank that is itself paged out is no better than the disk. It is the
        part of the arithmetic that cannot be moved, which is what a card count has to cover.
        """
        return int(self.total_bytes - self.offloadable_bytes)

    @property
    def has_routed_experts(self) -> bool:
        return self.routed_expert_bytes > 0

    @property
    def layer_count(self) -> int | None:
        return int_at(self.config, "num_hidden_layers", "n_layers", "block_count")

    @property
    def parameter_count(self) -> int:
        """Elements, from the header dimensions -- not a byte total divided by a dtype width.

        Nothing in this tree counted parameters before; the sum is exact for both formats because
        both headers carry every tensor's shape, and a GGUF's block-quantized tensors report their
        logical element count rather than their packed byte count.
        """
        total = 0
        for size in self.elements_by_tensor.values():
            total += size
        return total



def _safetensors_inventory(path: str) -> CheckpointInventory:
    from relicllm.loader.safetensors import MmapSafetensors

    reader = MmapSafetensors(path)
    config = _hf_config(path) or {}
    bytes_by_role: dict[str, int] = {}
    bytes_by_dtype: dict[str, int] = {}
    by_role_and_dtype: dict[str, dict[str, int]] = {}
    elements: dict[str, int] = {}
    for key, entry in reader.entries.items():
        role = role_of(key)
        bytes_by_role[role] = bytes_by_role.get(role, 0) + entry.nbytes
        bytes_by_dtype[entry.dtype] = bytes_by_dtype.get(entry.dtype, 0) + entry.nbytes
        per_dtype = by_role_and_dtype.setdefault(role, {})
        per_dtype[entry.dtype] = per_dtype.get(entry.dtype, 0) + entry.nbytes
        count = 1
        for dim in entry.shape:
            count *= int(dim)
        elements[key] = count
    return CheckpointInventory(
        path=os.path.abspath(path),
        format=FORMAT_SAFETENSORS,
        architecture=architecture_of(config),
        config=config,
        tensor_names=tuple(reader.entries),
        bytes_by_role=bytes_by_role,
        bytes_by_role_and_dtype=by_role_and_dtype,
        bytes_by_dtype=bytes_by_dtype,
        tensor_count=len(reader.entries),
        total_bytes=reader.nbytes_total(),
        elements_by_tensor=elements,
    )


def _hf_config(path: str) -> dict[str, Any] | None:
    from relicllm.backends.capabilities import read_config

    return read_config(path)


def _gguf_inventory(path: str) -> CheckpointInventory:
    from relicllm.loader.gguf.bundle import read_gguf_bundle

    bundle = read_gguf_bundle(path)
    config = flatten_gguf(bundle.metadata)
    bytes_by_role: dict[str, int] = {}
    bytes_by_dtype: dict[str, int] = {}
    by_role_and_dtype: dict[str, dict[str, int]] = {}
    elements: dict[str, int] = {}
    unknown_bytes = 0
    for tensor in bundle.tensors:
        role = role_of(tensor.name)
        size = tensor.nbytes
        if size is None:
            # A tensor whose type this build does not know: countable in elements, not in bytes.
            unknown_bytes += 1
            size = 0
        bytes_by_role[role] = bytes_by_role.get(role, 0) + int(size)
        bytes_by_dtype[tensor.type_name] = bytes_by_dtype.get(tensor.type_name, 0) + int(size)
        per_dtype = by_role_and_dtype.setdefault(role, {})
        per_dtype[tensor.type_name] = per_dtype.get(tensor.type_name, 0) + int(size)
        elements[tensor.name] = tensor.elements
    unknown: list[str] = []
    if unknown_bytes:
        unknown.append(
            f"{unknown_bytes} tensors carry a GGML type this build has no width for; their bytes "
            "are missing from every total below"
        )
    return CheckpointInventory(
        path=os.path.abspath(path),
        format=FORMAT_GGUF,
        architecture=architecture_of(config),
        config=config,
        tensor_names=tuple(tensor.name for tensor in bundle.tensors),
        bytes_by_role=bytes_by_role,
        bytes_by_role_and_dtype=by_role_and_dtype,
        bytes_by_dtype=bytes_by_dtype,
        tensor_count=len(bundle.tensors),
        total_bytes=sum(bytes_by_dtype.values()),
        unknown=tuple(unknown),
        elements_by_tensor=elements,
    )


def checkpoint_paths(path: str) -> tuple[str, ...]:
    """Resolve what the caller pointed at into the artifacts to read.

    A file is itself. A directory that holds a GGUF is that GGUF. A directory that holds a
    safetensors checkpoint is that directory. The GGUF case is the one worth spelling out: a quant
    release is often a *directory of directories* -- ``GLM-5.2-GGUF/UD-Q2_K_XL/`` beside
    ``UD-Q4_K_M/`` -- and pointing at the parent is a thing a person does. Listing them is not the
    same as choosing between them, so when there is more than one the caller gets all of them and
    the report says so.
    """
    root = os.path.abspath(path)
    if os.path.isfile(root):
        return (root,)
    entries = sorted(os.listdir(root))
    direct = [name for name in entries if name.endswith(".gguf")]
    if direct:
        # A sharded GGUF directory holds *one* checkpoint in N files, and handing the caller seven
        # paths would make shards look like alternative quantizations. `resolve_gguf_bundle` is the
        # loader's own answer to "which files are one model", and it is asked of a *member*: a file
        # that belongs to a shard set resolves back to the whole set, and a file that stands alone
        # resolves to itself. That is what separates Bonsai's directory -- two complete quants of one
        # model, PTQ1_0 beside PQ2_0 -- from a seven-shard upload of one, which look identical when
        # the directory itself is asked.
        from relicllm.loader.gguf.bundle import resolve_gguf_bundle

        members = tuple(os.path.join(root, name) for name in direct)
        if len(members) > 1 and all(len(resolve_gguf_bundle(member)) == 1 for member in members):
            return members
        shards = resolve_gguf_bundle(root)
        if len(shards) > 1:
            return (root,)  # one bundle; the reader resolves it
        return members
    nested: list[str] = []
    for name in entries:
        candidate = os.path.join(root, name)
        if os.path.isdir(candidate) and any(
            item.endswith(".gguf") for item in os.listdir(candidate)
        ):
            nested.append(candidate)
    if nested:
        return tuple(nested)
    return (root,)


def _looks_like_gguf(path: str) -> bool:
    """Whether this path is a GGUF release, at either of the two depths one arrives at.

    ``checkpoint_has_gguf`` is the runtime's own probe and it is a single-level glob, which is right
    for what the runtime does -- it is handed one file. A triage run is handed whatever a person has
    on disk, and a quant *crate* is the common shape: ``GLM-5.2-GGUF/UD-Q2_K_XL/`` beside
    ``UD-Q4_K_M/``, with no ``.gguf`` at the top level at all. Asking only the top level sends that
    directory to the safetensors reader, which reports ``no safetensors checkpoint at ...`` -- a true
    sentence about the wrong format, when the answer the caller needs is the two releases by name.
    """
    from relicllm.backends.capabilities import checkpoint_has_gguf

    if checkpoint_has_gguf(path):
        return True
    if not os.path.isdir(path):
        return False
    try:
        return any(
            os.path.isdir(os.path.join(path, name))
            and any(item.endswith(".gguf") for item in os.listdir(os.path.join(path, name)))
            for name in os.listdir(path)
        )
    except OSError:
        return False


def read_inventory(path: str) -> CheckpointInventory:
    """Header-only inventory of the checkpoint at ``path``.

    Format is decided by what is there rather than by what the caller said: a ``.gguf`` file or a
    directory containing one is GGUF, and anything else that holds safetensors is read as such. The
    two formats have genuinely different readers, so guessing wrong produces a confusing failure
    rather than a wrong number -- the safetensors reader raises ``FileNotFoundError`` on a GGUF
    directory and vice versa. A checkpoint whose shards are genuinely missing raises for the same
    reason: an inventory of 34 of 41 shards is not a smaller checkpoint, it is an unreadable one,
    and reporting a byte total for it would be a wrong answer rather than a refused one.

    **A truncated header is one of those failures, and it arrives as the wrong exception type.** A
    half-finished download is a normal thing to find on this box, and the two readers report it in
    their own vocabulary -- ``struct.error`` from a safetensors shard shorter than its own 8-byte
    length prefix, ``EOFError`` from a GGUF that ends mid-metadata. Neither is a ``ValueError``, and
    the CLI tells them apart by type, so an unreadable checkpoint was exiting ``2`` (*usage error*)
    instead of ``3`` (*could not be read*) -- the one distinction a script branching on ``$?`` makes.
    They are normalized here, at the single place both formats are dispatched from.
    """
    try:
        if os.path.isfile(path) and path.endswith(".gguf"):
            return _gguf_inventory(path)
        if _looks_like_gguf(path):
            candidates = checkpoint_paths(path)
            if len(candidates) > 1:
                names = ", ".join(os.path.basename(c) for c in candidates)
                raise ValueError(
                    f"{path} holds {len(candidates)} GGUF releases ({names}); point at one of them"
                )
            return _gguf_inventory(candidates[0])
        return _safetensors_inventory(path)
    except (EOFError, struct.error) as error:
        raise ValueError(
            f"{os.path.abspath(path)} could not be read: the header is truncated or malformed "
            f"({type(error).__name__}: {error})"
        ) from error
