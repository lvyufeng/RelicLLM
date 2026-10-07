"""The KV declaration: its arithmetic, its per-architecture publishers, and the line it holds.

The taxonomy (`relicllm/runtime/kv_spec.py`) exists to answer one question two callers ask in two
ways -- "what does one more token cost" (the fit test) and "how many bytes does a block hold" (a
paged allocator) -- without either having to run the model. These tests pin the arithmetic those
answers rest on, then check every served architecture actually publishes one, against the same
checked-in config snapshots `tests/test_triage_kv_geometry.py` reads, so nothing here needs a
checkpoint.

The last test is the one that matters most: the declaration is only the authority if the consumer
reads it, and a silent fall-through to the older computation would pass every other test here.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from relicllm.runtime.kv_spec import (
    FullAttentionSpec,
    KVKind,
    MLASpec,
    SlidingWindowSpec,
    StateSpec,
    dtype_size,
    to_torch_dtype,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "triage" / "kv_geometry.json"

#: The architectures this tree serves that publish a declaration, and the module that publishes it.
#: The same table `relicllm/triage/kv.py::_DECLARED` holds, restated here so a publisher that moves
#: or disappears is a failure rather than a silently skipped case.
PUBLISHERS = {
    "deepseek_v4": "relicllm.models.deepseek_v4.kv_spec",
    "deepseek_v4_1": "relicllm.models.deepseek_v4_1.kv_spec",
    "mimo": "relicllm.models.mimo_v2.kv_spec",
    "qwen4_exp": "relicllm.models.qwen4_exp.kv_spec",
    "xing4": "relicllm.models.xing4_0.kv_spec",
}


def _fixture_models() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())["models"]


def _publisher(architecture: str):
    import importlib

    return importlib.import_module(PUBLISHERS[architecture])


def _served() -> list[dict[str, Any]]:
    return [m for m in _fixture_models() if m["architecture"] in PUBLISHERS]


# --------------------------------------------------------------------------------------------
# The arithmetic
# --------------------------------------------------------------------------------------------


def test_a_full_attention_page_is_the_gqa_pair_times_dtype() -> None:
    """K and V are stored separately: ``heads * (key + value) * dtype`` a token."""
    spec = FullAttentionSpec(
        name="key", layer_ids=(0, 1, 2), dtype="bfloat16", num_kv_heads=4, head_dim=192, v_head_dim=128
    )

    assert spec.page_size_bytes(1) == 4 * (192 + 128) * 2
    assert spec.values_per_token == 4 * (192 + 128)
    # Three layers share it, so the group's marginal cost is three times one layer's.
    assert spec.bytes_per_token_whole_model() == 3 * 4 * (192 + 128) * 2


def test_a_full_attention_value_follows_the_key_width_when_unstated() -> None:
    """``v_head_dim=None`` is the common case, not a missing value: the value is key-width."""
    spec = FullAttentionSpec(name="k", layer_ids=(0,), dtype="bfloat16", num_kv_heads=2, head_dim=256)
    paired = FullAttentionSpec(
        name="k", layer_ids=(0,), dtype="bfloat16", num_kv_heads=2, head_dim=256, v_head_dim=256
    )

    assert spec.values_per_token == paired.values_per_token == 2 * (256 + 256)


def test_a_sliding_window_margins_zero_and_still_allocates_its_ring() -> None:
    """The two costs this taxonomy keeps apart: a ring grows with nothing and reserves everything."""
    ring = SlidingWindowSpec(
        name="window",
        layer_ids=(0, 1),
        dtype="bfloat16",
        num_kv_heads=1,
        head_dim=512,
        v_head_dim=0,
        sliding_window=128,
    )

    assert ring.kind is KVKind.SLIDING_WINDOW
    assert ring.values_per_token == 0, "a ring costs nothing more however long the context runs"
    assert ring.bytes_per_token_whole_model() == 0
    # One latent a token per layer, 128 slots, no value tensor.
    assert ring.page_size_bytes(128) == 128 * 512 * 2
    assert ring.allocated_bytes(128) == 2 * 128 * 512 * 2


def test_mla_stores_no_value_and_divides_by_the_compress_ratio() -> None:
    """``head_dim // ratio`` answers an absorbed cache (ratio 1) and a compressed one alike."""
    absorbed = MLASpec(name="latent", layer_ids=(0,), dtype="bfloat16", head_dim=576, compress_ratio=1)
    compressed = MLASpec(
        name="compress", layer_ids=(0,), dtype="bfloat16", head_dim=512, compress_ratio=4
    )

    assert absorbed.values_per_token == 576, "an absorbed latent is stored whole, no head axis"
    assert compressed.values_per_token == 512 // 4
    # A page of four tokens is one latent, and latents are not doubled: there is no value tensor.
    assert compressed.page_size_bytes(4) == 1 * (512 // 4) * 2


def test_a_compressed_page_holds_at_least_one_latent() -> None:
    """A block smaller than the ratio still holds the partial group the compressor is filling."""
    spec = MLASpec(name="compress", layer_ids=(0,), dtype="bfloat16", head_dim=512, compress_ratio=128)

    assert spec.page_size_bytes(1) == 1 * (512 // 128) * 2


def test_a_state_has_no_token_axis() -> None:
    """Upstream's MambaSpec row: shaped by head counts and kernel width, never by context."""
    state = StateSpec(
        name="state",
        layer_ids=(0, 1, 2, 3),
        dtype="bfloat16",
        shapes=((40, 3), (8, 128, 128)),
        dtypes=("bfloat16", "float32"),
    )

    assert state.kind is KVKind.STATE
    assert state.values_per_token == 0
    assert state.page_size_bytes() == 40 * 3 * 2 + 8 * 128 * 128 * 4
    # One page for the whole state, and four layers of them.
    assert state.allocated_bytes() == 4 * (40 * 3 * 2 + 8 * 128 * 128 * 4)


def test_every_dtype_the_table_names_has_a_torch_spelling() -> None:
    """A spec that reported two bytes and asked torch for a four-byte dtype would double its cache."""
    for name in ("bool", "int8", "uint8", "float8_e4m3fn", "float8_e5m2", "float16", "bfloat16",
                 "float32", "float64"):
        assert dtype_size(name) > 0
        to_torch_dtype(name)  # raises if the two tables disagree about what exists


def test_an_unknown_dtype_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError):
        dtype_size("float7")


# --------------------------------------------------------------------------------------------
# The publishers
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", _served(), ids=lambda m: m["name"])
def test_a_served_architecture_declares_its_shape(model: dict[str, Any]) -> None:
    """Every served architecture publishes a non-empty declaration that names its own layers."""
    specs = _publisher(model["architecture"]).kv_spec(model["config"])

    assert specs, f"{model['name']} declares nothing"
    for spec in specs:
        assert spec.layer_ids, f"{spec.name} names no layer"
        assert max(spec.layer_ids) < 200, "a layer id that large is a unit mix-up, not a layer"


@pytest.mark.parametrize("model", _served(), ids=lambda m: m["name"])
def test_the_declared_cache_matches_the_verified_geometry(model: dict[str, Any]) -> None:
    """The declaration's marginal cost is the number verified against the checkpoint on 2026-10-02."""
    specs = _publisher(model["architecture"]).kv_spec(model["config"])
    derived = sum(spec.bytes_per_token_whole_model() for spec in specs)

    assert derived == model["expected_bytes_per_token_whole_model"], (
        f"{model['name']}: declaration gives {derived} B/token, "
        f"the verified fixture says {model['expected_bytes_per_token_whole_model']}"
    )


def test_the_declarations_are_readable_without_importing_torch() -> None:
    """#129's first acceptance item, checked the way the device tests check theirs: in a fresh process.

    A publisher that imported torch would still pass every test above -- the numbers would be right --
    while making a fit check on a host that never loads the model pay for a torch import it does not
    need. The check has to be a subprocess, because torch is already in this one.
    """
    program = (
        "import sys;"
        + "".join(f"import {module};" for module in PUBLISHERS.values())
        + "sys.exit(1 if 'torch' in sys.modules else 0)"
    )
    completed = subprocess.run(
        [sys.executable, "-c", program], cwd=REPO_ROOT, capture_output=True, text=True
    )

    assert completed.returncode == 0, "a KV declaration imported torch:\n" + completed.stderr


# --------------------------------------------------------------------------------------------
# The line the declaration has to hold
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("model", _served(), ids=lambda m: m["name"])
def test_triage_costs_the_declared_geometry_rather_than_recomputing_it(model: dict[str, Any]) -> None:
    """The consumer must read the declaration; a fall-through to the old builder is the failure mode.

    `relicllm/triage/kv.py` can compute the same number two ways. This asserts the served
    architectures take the declared one, so a change that quietly stops reaching it -- an import that
    raises, a config shape the publisher cannot read -- fails here instead of agreeing by coincidence.
    """
    from relicllm.triage.kv import _DECLARED, _from_declaration

    assert model["architecture"] in _DECLARED, "a served architecture triage does not know about"

    declared = _from_declaration(model["architecture"], model["config"])

    assert declared is not None, (
        f"{model['name']}: triage fell back to its own computation instead of the declaration"
    )
    assert any("kv_spec.py" in source for source in declared.sources)
    assert declared.values_per_token_per_layer * declared.dtype_bytes == (
        model["expected_bytes_per_token_whole_model"]
    )