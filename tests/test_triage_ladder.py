"""The precision ladder: finding the other releases, and proving they are the same model.

Every case here is a layout, not a checkpoint: directories, names and tensor-name sets written into
a tmp dir. The real roster has all of these shapes on one host -- a second precision beside a
checkpoint, a crate of quant subdirectories with ``.sha256`` sidecars between them, two free-standing
GGUF quants in one directory, and two different models that declare the same ``model_type`` -- and
the search has to survive each of them without either crashing or pairing the wrong two models.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from relicllm.triage import read_inventory
from relicllm.triage.weights import precision_ladder, same_model, sibling_artifacts

GIB = 1024 ** 3


def write_checkpoint(root: Path, tensors: dict[str, tuple[str, tuple[int, ...]]], *, model_type: str) -> Path:
    """A header-only safetensors checkpoint: an 8-byte length, a JSON header, and no data."""
    root.mkdir(parents=True, exist_ok=True)
    header: dict[str, object] = {}
    offset = 0
    for name, (dtype, shape) in tensors.items():
        width = {"BF16": 2, "F16": 2, "F32": 4, "F8_E4M3": 1, "I8": 1, "U8": 1}.get(dtype, 2)
        elements = 1
        for dim in shape:
            elements *= int(dim)
        size = elements * width
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    blob = json.dumps(header).encode()
    (root / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    (root / "config.json").write_text(
        json.dumps({"model_type": model_type, "num_hidden_layers": 1, "head_dim": 64})
    )
    return root


def test_a_quantisation_that_renames_its_tensors_is_still_the_same_model(tmp_path: Path) -> None:
    """The pair that made an architecture-only test insufficient, in miniature.

    FP8 keeps ``down_proj.weight`` beside ``down_proj.weight_scale_inv``; NVFP4 stores the same
    projection as ``weight_packed`` with a ``weight_scale`` and two global scales. Comparing the names
    as written says these are different models -- measured on the real pair, 0.405 -- and stripping
    the quantization-side suffixes takes it to 0.974.
    """
    fp8 = read_inventory(
        str(
            write_checkpoint(
                tmp_path / "M-FP8",
                {
                    "model.layers.0.mlp.down_proj.weight": ("F8_E4M3", (64, 64)),
                    "model.layers.0.mlp.down_proj.weight_scale_inv": ("F32", (1,)),
                },
                model_type="qwen3_5",
            )
        )
    )
    nvfp4 = read_inventory(
        str(
            write_checkpoint(
                tmp_path / "M-NVFP4",
                {
                    "model.layers.0.mlp.down_proj.weight_packed": ("U8", (64, 32)),
                    "model.layers.0.mlp.down_proj.weight_scale": ("F8_E4M3", (64, 1)),
                    "model.layers.0.mlp.down_proj.input_global_scale": ("F32", (1,)),
                },
                model_type="qwen3_5",
            )
        )
    )

    assert same_model(fp8, nvfp4)


def test_a_quantisation_that_splits_the_experts_is_still_the_same_model(tmp_path: Path) -> None:
    """The pair that made suffix-stripping alone insufficient, in miniature.

    ``Qwen3.8-Flash-Next`` keeps each layer's experts *stacked* -- one
    ``...mlp.experts.gate_up_proj`` -- and its own FP8 export writes them out one at a time as
    ``...mlp.experts.0.gate_proj.weight`` beside a ``weight_scale_inv``. The two artifacts share
    every non-expert tensor name and almost no expert ones, and on the real pair the overlap is 0.020
    against 1560 shared names out of 76,923. Dropping the expert and layer indices takes it to 0.970,
    and no other pair on the box rises above 0.560.
    """
    stacked = read_inventory(
        str(
            write_checkpoint(
                tmp_path / "flash",
                {
                    "model.language_model.embed_tokens.weight": ("BF16", (64, 64)),
                    "model.language_model.layers.0.mlp.experts.gate_up_proj": ("BF16", (64, 64)),
                    "model.language_model.layers.0.mlp.experts.down_proj": ("BF16", (64, 64)),
                    "mtp.layers.0.self_attn.q_proj.weight": ("BF16", (64, 64)),
                },
                model_type="qwen4_exp",
            )
        )
    )
    split = read_inventory(
        str(
            write_checkpoint(
                tmp_path / "flash-fp8",
                {
                    "model.language_model.embed_tokens.weight": ("F8_E4M3", (64, 64)),
                    "model.language_model.layers.0.mlp.experts.0.gate_proj.weight": ("F8_E4M3", (64, 64)),
                    "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv": ("F32", (1,)),
                    "model.language_model.layers.0.mlp.experts.0.down_proj.weight": ("F8_E4M3", (64, 64)),
                    "mtp.layers.0.self_attn.q_proj.weight": ("F8_E4M3", (64, 64)),
                },
                model_type="qwen4_exp",
            )
        )
    )

    assert same_model(stacked, split)


def test_two_models_sharing_a_model_type_are_not_the_same_model(tmp_path: Path) -> None:
    """The failure the architecture key alone allows, and the reason names are measured instead.

    Bonsai's GGUF declares ``qwen35`` and Qwen3.8-27B declares the same, so a rule keyed on the
    architecture would file Bonsai's 5.53 GiB under Qwen3.8-27B's name -- a different model's byte
    count presented as this one's, which is the answer this module exists to refuse.
    """
    left = read_inventory(
        str(
            write_checkpoint(
                tmp_path / "left",
                {"model.layers.0.self_attn.q_proj.weight": ("BF16", (64, 64))},
                model_type="qwen3_5",
            )
        )
    )
    right = read_inventory(
        str(
            write_checkpoint(
                tmp_path / "right",
                {"blk.0.attn_q.weight": ("BF16", (64, 64))},
                model_type="qwen3_5",
            )
        )
    )

    assert left.architecture == right.architecture
    assert not same_model(left, right)


def test_a_second_precision_beside_a_checkpoint_is_found(tmp_path: Path) -> None:
    """The layout that made the old search dead: the variants are *outside* the checkpoint directory.

    ``Qwen3.8-27B-FP8`` is a complete model in its own right, so a search that stays inside it finds
    nothing. Its ``-NVFP4`` sibling is one level up, and so is every ``-w8a8`` build on this box.
    """
    tensor = {"model.layers.0.self_attn.q_proj.weight": ("BF16", (64, 64))}
    primary = write_checkpoint(tmp_path / "Model", tensor, model_type="qwen3_5")
    variant = write_checkpoint(tmp_path / "Model-NVFP4", tensor, model_type="qwen3_5")
    # Named like a quantization and *found* like one, but its headers describe another model.
    unrelated = write_checkpoint(
        tmp_path / "Model-Bonsai-FP8",
        {"blk.0.attn_q.weight": ("BF16", (64, 64))},
        model_type="qwen3_5",
    )

    siblings = sibling_artifacts(str(primary))

    assert os.path.abspath(str(variant)) in siblings
    assert os.path.abspath(str(unrelated)) in siblings, "found by name -- and that is all a name is for"
    assert [o.path for o in precision_ladder(str(primary))] == [
        os.path.abspath(str(primary)),
        os.path.abspath(str(variant)),
    ]


def test_the_search_runs_back_from_a_quant_to_the_release_it_was_made_from(tmp_path: Path) -> None:
    """``Qwen3.8-27B-FP8`` has to find the checkpoint it was quantized from.

    That directory is the one shape the two name tests cannot reach: it carries no quantization
    suffix, and a safetensors checkpoint holds no GGUF. So the only name for it is the one the quant's
    own name implies -- strip the suffix the search matched -- and with the search one-directional
    the ``-FP8`` build finds the NVFP4 build and never the native release underneath both.
    """
    tensor = {"model.layers.0.self_attn.q_proj.weight": ("BF16", (64, 64))}
    base = write_checkpoint(tmp_path / "Model", tensor, model_type="qwen3_5")
    variant = write_checkpoint(tmp_path / "Model-FP8", tensor, model_type="qwen3_5")

    assert [o.path for o in precision_ladder(str(variant))] == [
        os.path.abspath(str(variant)),
        os.path.abspath(str(base)),
    ]


def test_a_crate_of_quant_directories_is_read_through_its_sidecars(tmp_path: Path) -> None:
    """``GLM-5.2-GGUF`` holds two quant directories *and* their ``.sha256`` files.

    An earlier version asked whether every entry was a directory, which the sidecars answer "no", so
    the crate looked like neither a checkpoint nor a crate and its quants were never compared.
    """
    crate = tmp_path / "Model-GGUF"
    for name in ("UD-Q2_K_XL", "UD-Q4_K_M"):
        (crate / name).mkdir(parents=True)
        (crate / name / "shard.gguf").touch()
    (crate / "UD-Q2_K_XL.sha256").write_text("deadbeef\n")

    assert sorted(os.path.basename(p) for p in sibling_artifacts(str(crate))) == ["UD-Q2_K_XL", "UD-Q4_K_M"]


def test_two_free_standing_gguf_quants_in_one_directory_are_alternatives(tmp_path: Path) -> None:
    """Bonsai ships ``PTQ1_0`` beside ``PQ2_0``; both are whole models, neither is a shard."""
    crate = tmp_path / "Bonsai-gguf"
    crate.mkdir(parents=True)
    for name in ("Ternary-Bonsai-2-27B-PTQ1_0.gguf", "Ternary-Bonsai-2-27B-PQ2_0.gguf"):
        (crate / name).touch()

    assert sibling_artifacts(str(crate / "Ternary-Bonsai-2-27B-PTQ1_0.gguf")) == (
        str(crate / "Ternary-Bonsai-2-27B-PQ2_0.gguf"),
    )


def test_an_ambiguous_directory_says_which_artifacts_it_holds(tmp_path: Path) -> None:
    """Two whole GGUF quants in one directory is a question, not a checkpoint.

    The reader must not answer it by reading both as one model -- their tensor tables overlap and the
    loader would raise a duplicate-tensor error that says nothing about what to do. Naming them is
    the answer, and it is a different answer from ``IMPOSSIBLE``.
    """
    crate = tmp_path / "Bonsai-gguf"
    crate.mkdir(parents=True)
    for name in ("Ternary-Bonsai-2-27B-PTQ1_0.gguf", "Ternary-Bonsai-2-27B-PQ2_0.gguf"):
        (crate / name).touch()

    with pytest.raises(ValueError) as failure:
        read_inventory(str(crate))

    assert "2 GGUF releases" in str(failure.value)
    assert "point at one of them" in str(failure.value)


def test_an_unlistable_sibling_does_not_turn_the_search_into_a_crash(tmp_path: Path) -> None:
    """``/mnt/data2/lost+found`` is mode 700 and sits beside the checkpoints on this box."""
    primary = write_checkpoint(
        tmp_path / "Model",
        {"model.layers.0.self_attn.q_proj.weight": ("BF16", (64, 64))},
        model_type="qwen3_5",
    )
    locked = tmp_path / "Model-FP8"
    locked.mkdir()
    locked.chmod(0o000)
    try:
        siblings = sibling_artifacts(str(primary))
    finally:
        locked.chmod(0o755)

    assert os.path.abspath(str(locked)) not in siblings
