"""The nine hand-verified KV geometries, pinned so a change to one is a decision rather than a drift.

Every entry in ``fixtures/triage/kv_geometry.json`` was read off a real checkpoint and then checked
against the allocation site in code, and each carries the ``file.py:NNN`` it came from. The test is
hermetic: it reads checked-in config snapshots, never a checkpoint, so it runs on any machine.

Why this test exists at all. The obvious formula -- ``n_layers * n_kv_heads * head_dim * 2`` -- is
wrong for four of these nine, by between 3.5x and infinity, and the mistakes are all in the same
direction (overstating the cache). It also disagrees with the code on GLM-5.2 by 56.9x, in the other
direction. Those are not typos to be caught by review; they are the reason the numbers live in a
fixture with their provenance beside them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from relicllm.triage.kv import (
    REPLICATED,
    SHARDED,
    SINGLE_CARD,
    bytes_per_token_per_rank,
    kv_geometry,
)


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "triage" / "kv_geometry.json"


def _models() -> list[dict[str, Any]]:
    return json.loads(FIXTURE.read_text())["models"]


def _case_id(model: dict[str, Any]) -> str:
    return model["name"]


@pytest.fixture(params=_models(), ids=_case_id)
def model(request: pytest.FixtureRequest) -> dict[str, Any]:
    return request.param


def test_whole_model_bytes_per_token_matches_the_pinned_value(model: dict[str, Any]) -> None:
    geometry = kv_geometry(model["config"], architecture=model["architecture"])
    derived = geometry.values_per_token_per_layer * geometry.dtype_bytes

    assert derived == model["expected_bytes_per_token_whole_model"], (
        f"{model['name']}: derived {derived} B/token, fixture says "
        f"{model['expected_bytes_per_token_whole_model']}. Source: {model['source']}"
    )


def test_sharding_and_preallocation_are_recorded_per_model(model: dict[str, Any]) -> None:
    geometry = kv_geometry(model["config"], architecture=model["architecture"])

    assert geometry.sharding == model["expected_sharding"], (
        f"{model['name']}: sharding is {geometry.sharding!r}, fixture says "
        f"{model['expected_sharding']!r} — from {geometry.sharding_source}"
    )
    assert geometry.preallocated is model["expected_preallocated"], (
        f"{model['name']}: preallocated is {geometry.preallocated}, fixture says "
        f"{model['expected_preallocated']}"
    )


def test_every_derived_geometry_names_the_code_it_came_from(model: dict[str, Any]) -> None:
    geometry = kv_geometry(model["config"], architecture=model["architecture"])

    assert geometry.sources, f"{model['name']}: a derived geometry must cite its allocation site"
    for layer in geometry.layers:
        assert layer.source, f"{model['name']}: a layer class must cite its allocation site"


def test_layer_classes_account_for_every_layer(model: dict[str, Any]) -> None:
    """A class list that does not sum to the layer count has silently dropped some layers.

    ``indexer`` classes are excluded: DeepSeek's second cache is a *second* buffer on layers the
    attention classes have already counted, so including it would double-count them.

    The pinned count is the **trunk** where the fixture carries one, because a GGUF's
    ``block_count`` includes the trailing NextN/MTP blocks the runtime does not build -- GLM-5.2
    declares 79 blocks and builds 78. The subtraction is not applied here: a test that repeated the
    code's own rule would agree with the code whatever the rule said, so the trunk depth is a
    hand-verified number in the fixture like every other expectation.
    """
    geometry = kv_geometry(model["config"], architecture=model["architecture"])
    config = model["config"]
    declared = (
        model.get("expected_trunk_layers")
        or config.get("num_hidden_layers")
        or config.get("glm-dsa.block_count")
        or config.get("qwen35.block_count")
    )
    counted = sum(layer.count for layer in geometry.layers if layer.kind != "indexer")

    assert counted == declared, f"{model['name']}: {counted} layer classes for {declared} layers"


def test_hybrid_models_cost_less_than_treating_every_layer_as_full() -> None:
    """The property the module is built around, asserted on the models where it is dramatic.

    MiMo is 39 sliding-window layers of 48 and Qwen3.8-27B is 48 linear layers of 64. Costing those
    as full attention -- the default for a model with no recognised architecture -- would overstate
    the first by 5.3x and the second by 4x, which is the difference between a fit and a non-fit.
    """
    cases = (
        ("MiMo-V2.6-Flash", {"hybrid_layer_pattern": [0] + [1] * 47, "num_hidden_layers": 48,
                             "num_key_value_heads": 4, "head_dim": 192, "v_head_dim": 128}, "mimo"),
        ("Qwen3.8-27B", {"full_attention_interval": 4, "num_hidden_layers": 64,
                         "num_key_value_heads": 4, "head_dim": 256}, "qwen3_5"),
    )
    for name, config, architecture in cases:
        geometry = kv_geometry(config, architecture=architecture)
        full = sum(layer.count for layer in geometry.layers if layer.kind == "full")
        per_full_layer = next(layer.values_per_token for layer in geometry.layers if layer.kind == "full")

        assert 0 < full < config["num_hidden_layers"], f"{name}: expected a hybrid split, got {full}"
        assert geometry.values_per_token_per_layer == full * per_full_layer
        assert geometry.values_per_token_per_layer < config["num_hidden_layers"] * per_full_layer


def test_a_gguf_metadata_table_is_read_through_its_architecture_prefix() -> None:
    """GGUF namespaces every key; one reader has to serve both formats or a key silently defaults."""
    gguf = {
        "general.architecture": "glm-dsa",
        "glm-dsa.block_count": 79,
        "glm-dsa.attention.head_count": 64,
        "glm-dsa.attention.key_length_mla": 256,
        "glm-dsa.attention.value_length_mla": 256,
    }
    geometry = kv_geometry(gguf)

    assert sum(layer.count for layer in geometry.layers) == 79
    assert geometry.layers[0].values_per_token == 64 * (256 + 256)


def test_a_gguf_without_layer_types_is_read_from_the_checkpoints_own_tensors() -> None:
    """The fallback that keeps a GGUF honest: a GGUF carries no layer_types at all."""
    names = [
        "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
        "blk.1.ssm_a", "blk.1.ssm_conv1d.weight", "blk.1.attn_qkv.weight",
    ]
    geometry = kv_geometry(
        {"general.architecture": "qwen35", "qwen35.block_count": 2,
         "qwen35.attention.head_count_kv": 4, "qwen35.attention.key_length": 256},
        tensor_names=names,
    )

    # Which class is which, not just how many of each: `blk.0` holds attention and `blk.1` the
    # state-space block, so a marker list with the two swapped still yields one class of each kind
    # and the same total. Pinning the order and which class carries the values is what catches that.
    assert [(layer.kind, layer.count) for layer in geometry.layers] == [("full", 1), ("linear", 1)]
    assert geometry.layers[0].values_per_token == 4 * (256 + 256)
    assert geometry.layers[1].values_per_token == 0, "a linear layer holds no growing KV"
    assert geometry.values_per_token_per_layer == 4 * (256 + 256), "one growing layer of the two"


def test_an_unknown_architecture_is_marked_assumed_rather_than_guessed_silently() -> None:
    from relicllm.triage.kv import Confidence

    geometry = kv_geometry({"model_type": "something-new", "num_hidden_layers": 8,
                            "num_key_value_heads": 2, "head_dim": 64})

    assert geometry.confidence is Confidence.ASSUMED
    assert geometry.notes, "an assumed geometry must say what it assumed"


def test_rank_width_only_divides_the_models_that_shard() -> None:
    """Replication is not a detail: at TP4 it costs four times what dividing would."""
    sharded = kv_geometry({"layer_types": ["full_attention"], "num_hidden_layers": 1,
                           "num_key_value_heads": 4, "head_dim": 128,
                           "architectures": ["Qwen3.8-27B"]})
    replicated = kv_geometry({"n_kv_heads": 8, "head_dim": 128, "num_hidden_layers": 1,
                              "model_type": "minimax-m2"})

    assert sharded.sharding == SHARDED
    assert replicated.sharding == REPLICATED
    # 1024 values x 2 bytes = 2048 B per token whole-model; TP4 gives each of four ranks a quarter.
    assert bytes_per_token_per_rank(sharded, tp_width=1) == 2048
    assert bytes_per_token_per_rank(sharded, tp_width=4) == 512
    # A replicated cache is the whole thing on every rank, whatever the width.
    assert bytes_per_token_per_rank(replicated, tp_width=1) == bytes_per_token_per_rank(
        replicated, tp_width=4
    )


def test_the_glm_discrepancy_is_carried_not_smoothed_over() -> None:
    """The tool reports the code's allocation; the disagreement with the model card is a note."""
    entry = next(m for m in _models() if m["name"].startswith("GLM"))
    geometry = kv_geometry(entry["config"], architecture=entry["architecture"])
    declared = entry["declared_latent_values_per_token"] * sum(
        layer.count for layer in geometry.layers
    )
    factor = geometry.values_per_token_per_layer / declared

    assert factor == pytest.approx(entry["expected_discrepancy_factor"], rel=0.01)
    assert any("over-allocation" in note for note in geometry.notes), geometry.notes
