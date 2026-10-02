"""Config-key reading, shared by every gate, so no two of them can disagree about what a key means.

Two shapes arrive here and they are close enough to share a reader but not close enough to be
treated as one thing:

* a HuggingFace ``config.json``, whose keys are bare -- ``num_hidden_layers``, ``head_dim``;
* a GGUF metadata table, whose keys are namespaced by architecture -- ``qwen35.block_count``,
  ``glm-dsa.attention.head_count``.

:func:`flatten_gguf` copies the namespaced keys to their bare form once, so a caller written against
one shape reads the other unchanged. That indirection is worth a module because the alternative --
each reader carrying both spellings for every key -- is nine chances to typo one and fall through to
a default, and a silent fall-through to a default is the failure this whole package is built to
prevent. It has already happened once: GLM-5.2's KV cost read as correct off nothing but defaults
that happened to match, because the prefixed keys were never seen.

The other rule here is that these readers never raise. A key that is missing, of the wrong type or
of a type this repository has never seen is *no evidence*, and :func:`int_at` returning ``None``
rather than ``0`` is what lets a caller tell "the checkpoint said zero" from "the checkpoint did not
say" -- a distinction that decides whether a fit result is measured or assumed.
"""

from __future__ import annotations

from typing import Any, Mapping

__all__ = [
    "architecture_key",
    "architecture_of",
    "flatten_gguf",
    "float_at",
    "int_at",
    "is_gguf_metadata",
    "list_at",
    "string_at",
    "text_configs",
]


def is_gguf_metadata(config: Mapping[str, Any]) -> bool:
    """Whether this table came from a GGUF rather than a ``config.json``.

    Keyed on ``general.architecture``: GGUF requires it, and no HuggingFace config uses that name.
    """
    return isinstance(config.get("general.architecture"), str)


def flatten_gguf(config: Mapping[str, Any]) -> Mapping[str, Any]:
    """Present a GGUF metadata table under its own bare keys.

    Returns ``config`` itself when it is not GGUF, so an HF config costs nothing. Where a bare key
    already exists the bare key wins, which keeps an explicitly-set value from being shadowed by the
    namespaced copy of itself.
    """
    arch = config.get("general.architecture")
    if not isinstance(arch, str) or not arch:
        return config
    prefix = f"{arch}."
    derived = {key[len(prefix):]: value for key, value in config.items() if key.startswith(prefix)}
    if not derived:
        return config
    return {**derived, **config}


def text_configs(config: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """The tables a key is looked for in, most specific first.

    The released DeepSeek-V4.1, MiMo and Qwen checkpoints nest their text stack, so the layer count
    is at the root *and* again under ``text_config``; a wrapper config is still the architecture.
    Both spellings of the nest are here because the checkpoints use both.
    """
    tables: list[Mapping[str, Any]] = [config]
    for key in ("text_config", "text_encoder_config", "language_config"):
        nested = config.get(key)
        if isinstance(nested, Mapping):
            tables.append(nested)
    return tuple(tables)


def int_at(config: Mapping[str, Any], *keys: str, default: int | None = None) -> int | None:
    """First present, integer-coercible value among ``keys``, at the root or under the text nest.

    ``None`` means the checkpoint did not say, which is not the same as its having said zero.
    """
    for table in text_configs(config):
        for key in keys:
            if key not in table:
                continue
            value = table[key]
            # A GGUF array carries its length in a `length` attribute and does not support int().
            if hasattr(value, "length"):
                return int(value.length)
            if isinstance(value, bool):
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
    return default


def float_at(config: Mapping[str, Any], *keys: str, default: float | None = None) -> float | None:
    """As :func:`int_at`, without the array-length case: a scalar float or nothing."""
    for table in text_configs(config):
        for key in keys:
            if key not in table:
                continue
            value = table[key]
            if isinstance(value, bool):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def list_at(config: Mapping[str, Any], *keys: str) -> list[Any] | None:
    """First list-valued entry among ``keys``, or ``None`` when none of them is a list.

    A GGUF stores arrays as objects with a ``length`` and an iterable body rather than as a Python
    list, and converting one is not free, so it is done here once rather than at each call site.
    """
    for table in text_configs(config):
        for key in keys:
            if key not in table:
                continue
            value = table[key]
            if isinstance(value, (list, tuple)):
                return list(value)
            if hasattr(value, "length"):
                try:
                    return [value[index] for index in range(int(value.length))]
                except Exception:
                    continue
    return None


def string_at(config: Mapping[str, Any], *keys: str) -> str | None:
    """First non-empty string among ``keys``."""
    for table in text_configs(config):
        for key in keys:
            value = table.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def architecture_of(config: Mapping[str, Any]) -> str | None:
    """What this checkpoint calls its own architecture, in whichever of the three ways it says it.

    ``model_type`` (HF), ``architectures`` (HF, a list whose first entry is the class), and
    ``general.architecture`` (GGUF). Returned lower-cased, exactly as the checkpoint spells it;
    mapping the several spellings onto one key is :func:`architecture_key`'s job.
    """
    for table in text_configs(config):
        for key in ("model_type", "general.architecture", "architectures"):
            value = table.get(key)
            if isinstance(value, str) and value:
                return value.lower()
            if isinstance(value, (list, tuple)) and value and isinstance(value[0], str):
                return str(value[0]).lower()
    return None


def architecture_key(arch: str) -> str:
    """Map the several spellings this repository uses onto one builder key.

    The same checkpoint calls itself one thing in its ``config.json`` and another in its GGUF, and
    the same model is spelled with a dash, an underscore and neither: Xing4's config says
    ``xing4_0``, Bonsai's GGUF says ``qwen35`` where its HF sibling says ``qwen3_5``, and
    DeepSeek-V4.1 appears as ``deepseek_v41``, ``deepseekv41`` and ``deepseek_v4_1``.

    Note the order. The V4.1 test has to come before the V4 one -- ``deepseek_v41`` contains
    ``deepseek_v4`` -- and MiMo before MiniMax, because a substring test for ``minimax`` would
    otherwise never be reached for a name that contains neither. Qwen3.5's prefix test is last of
    the Qwens for the same reason.

    An unrecognised name is returned cleaned and unchanged, which is what sends it to the generic
    path and marks its answer *assumed* rather than *derived*.
    """
    plain = arch.replace("-", "_").lower()
    if "deepseek_v41" in plain or "deepseekv41" in plain or plain == "deepseek_v4_1":
        return "deepseek_v4_1"
    if "deepseek4" in plain or "deepseek_v4" in plain:
        return "deepseek_v4"
    if "mimo" in plain:
        return "mimo"
    if "minimax" in plain:
        return "minimax"
    if "glm" in plain:
        return "glm_dsa"
    if "qwen4exp" in plain or "qwen4_exp" in plain or "flash_next" in plain:
        return "qwen4_exp"
    if "xing4" in plain:
        return "xing4"
    if "qwen3_5" in plain or "qwen35" in plain or plain.startswith("qwen3"):
        return "qwen3_5"
    return plain
