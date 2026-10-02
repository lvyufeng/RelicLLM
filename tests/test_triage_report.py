"""The report and the CLI, end to end, on a synthetic checkpoint built in a tmp dir.

Hermetic on purpose. Every real checkpoint this tool was developed against lives on one host, and a
test that needed one would skip there and pass nowhere else. The safetensors format is small enough
to write by hand -- an 8-byte length, a JSON header, and a data section this tool never reads, since
everything here is header arithmetic -- so the whole pipeline runs from gate 0 to the exit code
without a card, a checkpoint or a run.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from relicllm.triage import Tier, assess, read_inventory, role_of
from relicllm.triage.checkpoint import (
    ROLE_ATTENTION,
    ROLE_DENSE_MLP,
    ROLE_EMBEDDING,
    ROLE_LOOKUP_TABLE,
    ROLE_OTHER,
    ROLE_ROUTED_EXPERT,
)
from relicllm.triage.fit import hardware_profile_for

GIB = 1024 ** 3


def write_checkpoint(
    root: Path,
    *,
    architecture: str = "something-new",
    tensors: dict[str, tuple[str, tuple[int, ...]]] | None = None,
) -> Path:
    """Write a header-only safetensors checkpoint plus a ``config.json``.

    The data section is zero bytes for every tensor because the tool reads no weights: a shard's
    header is the 8-byte length followed by JSON, and ``TensorEntry.nbytes`` comes from the two
    ``data_offsets`` rather than from the file's size. That is what makes the static gate free, and
    writing the test this way exercises exactly that property.
    """
    root.mkdir(parents=True, exist_ok=True)
    tensors = tensors or {"model.embed_tokens.weight": ("BF16", (64, 64))}
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
        json.dumps(
            {
                "model_type": architecture,
                "num_hidden_layers": 2,
                "num_key_value_heads": 1,
                "head_dim": 64,
            }
        )
    )
    return root


def test_role_classification_separates_what_a_bank_can_hold_from_what_it_cannot() -> None:
    """The split the whole fit rests on, asserted on both formats' spellings.

    ``shared_experts`` is the case worth having a test for: it contains ``experts`` and is *not*
    offloadable, and reading it as one moves bytes out of the resident total -- the direction that
    makes a model look like it fits.
    """
    assert role_of("model.layers.3.mlp.experts.7.down_proj.weight") is ROLE_ROUTED_EXPERT
    assert role_of("blk.12.ffn_gate_exps.weight") is ROLE_ROUTED_EXPERT
    assert role_of("layers.5.ffn.shared_experts.w1.weight") is ROLE_DENSE_MLP
    assert role_of("layers.1.engram.embed.weight") is ROLE_LOOKUP_TABLE
    assert role_of("model.layers.7.ple.ple_embedding.ngram_embedding.shard_3.weight") is ROLE_LOOKUP_TABLE
    assert role_of("model.embed_tokens.weight") is ROLE_EMBEDDING
    assert role_of("model.layers.0.self_attn.q_proj.weight") is ROLE_ATTENTION
    assert role_of("model.layers.0.something_novel.weight") is ROLE_OTHER


def test_the_inventory_counts_tensors_and_bytes_from_headers_alone(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(
        tmp_path / "ckpt",
        tensors={
            "model.embed_tokens.weight": ("BF16", (32, 32)),
            "model.layers.0.mlp.experts.0.down_proj.weight": ("I8", (64, 64)),
        },
    )

    inventory = read_inventory(str(checkpoint))

    assert inventory.tensor_count == 2
    assert inventory.parameter_count == 32 * 32 + 64 * 64
    assert inventory.bytes_by_role[ROLE_EMBEDDING] == 32 * 32 * 2
    assert inventory.bytes_by_role[ROLE_ROUTED_EXPERT] == 64 * 64 * 1
    assert inventory.resident_bytes == 32 * 32 * 2
    assert inventory.has_routed_experts


def test_an_unreadable_checkpoint_is_candidate_and_never_impossible(tmp_path: Path) -> None:
    """A missing shard is a claim about the disk, not about the model.

    Gate 0 has not run, so nothing has been disproved. Reporting ``IMPOSSIBLE`` here would be the
    tool's worst failure mode: a confident answer derived from a filesystem error.
    """
    report = assess(str(tmp_path / "does-not-exist"))

    assert report.unreadable is not None
    assert report.verdict.tier is Tier.CANDIDATE
    assert not report.fits
    assert "do not read" in report.upgrade()


def test_an_unknown_architecture_is_reported_assumed_rather_than_refused(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "ckpt", architecture="something-new")

    report = assess(str(checkpoint), hardware=hardware_profile_for(host_memory_gib=64.0))

    assert report.geometry.confidence.value == "assumed"
    assert any("assumed" in note or "conservative" in note for note in report.geometry.notes)
    assert report.verdict.tier is Tier.CANDIDATE


def test_a_checkpoint_whose_resident_weights_exceed_every_card_is_impossible(tmp_path: Path) -> None:
    """The one definitive answer, on a tensor set small enough to write by hand.

    20 GiB of attention weights, no experts, one 22 GiB card whose usable budget is 18.7 GiB.
    Nothing about a missing adapter or kernel enters into it -- this is gate 0 alone, and that is
    the point.
    """
    checkpoint = write_checkpoint(
        tmp_path / "big",
        tensors={"model.layers.0.self_attn.q_proj.weight": ("BF16", (4_000_000, 2684))},
    )

    report = assess(
        str(checkpoint),
        hardware=hardware_profile_for(gpu_count=1, gpu_memory_gib=22.0),
        include_siblings=False,
    )

    assert report.verdict.tier is Tier.IMPOSSIBLE
    assert report.verdict.decided_by == "min_cards"
    assert report.best_fit is None


def test_the_report_is_json_serialisable_and_carries_every_basis(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "ckpt")
    report = assess(str(checkpoint), hardware=hardware_profile_for(host_memory_gib=64.0))

    payload = json.loads(json.dumps(report.to_dict()))

    assert payload["tier"] == "candidate"
    assert payload["hardware"]["host_memory_gib"] == 64.0
    assert payload["kv"]["sharding_source"]
    assert payload["precision_ladder"], "the ladder must report the artifact it was pointed at"
    assert payload["precision_ladder"][0]["evidence"]
    assert "reserve fraction" in " ".join(payload["fits"][0]["basis"])


def test_the_text_report_prints_the_reserve_it_used(tmp_path: Path) -> None:
    """A card count is meaningless without the reserve, because the two in use differ by 1.1 GiB."""
    checkpoint = write_checkpoint(tmp_path / "ckpt")

    text = assess(
        str(checkpoint), hardware=hardware_profile_for(host_memory_gib=64.0), reserve_fraction=0.10
    ).format_text()

    # Both the fraction and the GiB it leaves, because the fraction alone is a prefix of 0.15 and
    # would keep passing if the report switched back to it. 22 x 0.9 = 19.8 GiB against 18.7 at 0.15.
    assert "reserve 0.1 ->" in text
    assert "19.8 GiB/card usable" in text


def test_the_tier_a_measurement_earns_and_the_gap_an_incomplete_one_leaves(tmp_path: Path) -> None:
    checkpoint = write_checkpoint(tmp_path / "ckpt")
    box = hardware_profile_for(gpu_count=1, gpu_memory_gib=22.0, host_memory_gib=64.0)

    from relicllm.triage import Measured

    partial = assess(str(checkpoint), hardware=box, measured=Measured(ttft_1k_seconds=8.0))
    assert partial.verdict.tier is Tier.CANDIDATE
    assert "tpot_seconds" in partial.verdict.reason

    passing = assess(
        str(checkpoint),
        hardware=box,
        measured=Measured(
            ttft_1k_seconds=8.0, tpot_seconds=0.12, prefill_8k_tok_per_second=400.0, source="test"
        ),
    )
    assert passing.verdict.tier is Tier.PRODUCTION


# ---------------------------------------------------------------------------------------------
# the CLI
# ---------------------------------------------------------------------------------------------


def test_the_cli_prints_a_brief_row_and_an_exit_code(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from relicllm.cli.triage import EXIT_OK, main

    checkpoint = write_checkpoint(tmp_path / "ckpt")

    code = main(["--brief", "--host-memory-gib", "64", str(checkpoint)])

    assert code == EXIT_OK
    assert "CANDIDATE" in capsys.readouterr().out


def test_the_cli_brief_row_says_unreadable_rather_than_a_tier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The brief row and the exit code have to agree, and ``CANDIDATE`` is not what a disk error means.

    ``CANDIDATE`` is a claim that the static gates were passed. A checkpoint whose headers did not
    read has not been through them, so printing the tier there is the same conflation
    ``EXIT_UNREADABLE`` exists to prevent -- in the one output format a person reads as a list.

    The fixture is a *file* that will not parse, not a directory of two releases: the CLI expands a
    directory before it reads anything (``pointing at the parent`` is how the two quants of one model
    get compared), so an ambiguous directory never reaches this branch and would have tested the
    expansion instead of the row.
    """
    checkpoint = write_checkpoint(tmp_path / "ckpt")
    (checkpoint / "model.safetensors").write_bytes(struct.pack("<Q", 8) + b"not-json")

    from relicllm.cli.triage import EXIT_UNREADABLE, main

    assert main(["--brief", str(checkpoint)]) == EXIT_UNREADABLE
    out = capsys.readouterr().out
    assert out.startswith("UNREADABLE")
    assert "JSONDecodeError" in out


def test_a_truncated_header_is_unreadable_rather_than_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A half-finished download is the ordinary way a checkpoint fails, and it must not read as one.

    Both readers report it in their own vocabulary — ``struct.error`` from a shard shorter than its
    own 8-byte length prefix, ``EOFError`` from a GGUF that ends mid-metadata — and neither is a
    ``ValueError``. The CLI tells the two apart by type, so a truncated checkpoint was exiting ``2``
    (usage error) rather than ``3`` (could not be read), which is the one distinction a script
    branching on ``$?`` makes.
    """
    from relicllm.cli.triage import EXIT_UNREADABLE, main

    gguf = tmp_path / "truncated.gguf"
    gguf.write_bytes(b"GGUF")  # the magic, and then nothing
    stub = write_checkpoint(tmp_path / "ckpt")
    (stub / "model.safetensors").write_bytes(b"\x00\x00")

    for bad in (gguf, stub):
        assert main(["--brief", str(bad)]) == EXIT_UNREADABLE, bad
        assert capsys.readouterr().out.startswith("UNREADABLE"), bad


def test_the_cli_exits_non_zero_on_impossible_so_a_script_can_branch(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from relicllm.cli.triage import EXIT_IMPOSSIBLE, main

    checkpoint = write_checkpoint(
        tmp_path / "big",
        tensors={"model.layers.0.self_attn.q_proj.weight": ("BF16", (4_000_000, 2684))},
    )

    code = main(["--brief", "--gpu-count", "1", "--no-siblings", str(checkpoint)])

    assert code == EXIT_IMPOSSIBLE
    assert "IMPOSSIBLE" in capsys.readouterr().out


def test_the_cli_reads_a_measurement_file_and_reports_what_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from relicllm.cli.triage import EXIT_USAGE, main

    checkpoint = write_checkpoint(tmp_path / "ckpt")
    results = tmp_path / "results.json"
    results.write_text(json.dumps({"ttft_1k_seconds": 9.0}))

    code = main(["--brief", "--host-memory-gib", "64", "--measured", str(results), str(checkpoint)])

    assert code == 0
    assert "CANDIDATE" in capsys.readouterr().out

    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps({"unrelated": 1}))
    assert main(["--measured", str(empty), str(checkpoint)]) == EXIT_USAGE
    assert "none of" in capsys.readouterr().err


def test_the_cli_json_matches_the_object_the_library_builds(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from relicllm.cli.triage import main

    checkpoint = write_checkpoint(tmp_path / "ckpt")

    assert main(["--json", "--host-memory-gib", "64", str(checkpoint)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["tier"] == "candidate"
    assert payload["reachability"]["loader"] is True
