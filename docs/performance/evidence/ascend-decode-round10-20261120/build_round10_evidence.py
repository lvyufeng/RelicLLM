"""Copy the late-round receipts: what the launch flood is made of, and what does not move the number.

Nine earlier bundles each froze one optimisation or one measurement. The rounds after them stopped
producing gains and started producing *limits*, and those are worth freezing for the same reason: a
later reader should be able to check the claim that a knob is dead instead of re-running it.

Three receipts and the two tests that pin them:

* `kernel-inventory` -- the CANN profile of an eager decode step: operator counts, and the same table
  grouped by operator and input shape, which is how the per-token dtype conversions were attributed.
* `tile-major` -- the tile-major original-byte residency, accepted as exact on the card and measured
  at 3.05x the row-major store in isolation.
* `reverify` -- two independent 16-forward runs of the shipped candidate, including the per-step
  quality entries that are computed against the exact reference reduction inside the same process.

The tile-major *whole-model* result (0.2395 TPS, about 21x slower than the native Cube expert path)
was produced on the private root and its receipt is not in this tree; the number is recorded in
`.coord/performance-state.json` under `round14_findings` together with the run directory it came from.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "docs/performance/evidence/ascend-decode-round10-20261120"
SOURCES = {
    "kernel-inventory": ".coord/perf-kernel-inventory/profile-20261119T115321Z",
    "tile-major-isolated": ".coord/perf-tile-major-results/tile-major-runtime-acceptance-20261118T060535Z",
    "candidate-reverify": ".coord/perf-reverify-20261119T122859Z",
}
FILES = (
    ".coord/build_round10_evidence.py",
    "../relic-core-ascend-online-unpack/tests/test_sparse_attn_dispatch_count.py",
    "tests/test_v41_tile_major_switch.py",
)
SUFFIXES = {".json", ".log", ".csv", ".py"}


def main():
    if TARGET.exists():
        raise RuntimeError("the round-10 evidence bundle already exists; do not overwrite it")
    TARGET.mkdir(parents=True)
    for label, relative in SOURCES.items():
        source = ROOT / relative
        if not source.is_dir():
            raise RuntimeError(f"missing measured receipt directory: {source}")
        destination = TARGET / label
        destination.mkdir()
        for path in sorted(source.iterdir()):
            if path.is_file() and path.suffix in SUFFIXES:
                shutil.copyfile(path, destination / path.name)
    for relative in FILES:
        source = (ROOT / relative).resolve()
        if not source.is_file():
            raise RuntimeError(f"missing probe or test: {source}")
        shutil.copyfile(source, TARGET / source.name)
    files = [
        {"path": str(path.relative_to(TARGET)), "bytes": path.stat().st_size,
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        for path in sorted(TARGET.rglob("*")) if path.is_file()
    ]
    manifest = {
        "milestone": ("round 10: the launch flood is the numerical contract, and the host-side knobs "
                      "that look free do not move the number"),
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
        "verified_candidate_TPS": 5.1042,
        "verified_candidate_TPS_range": [5.0327, 5.1584],
        "measurements": {
            "aten_ops_per_token": 5704,
            "dtype_conversions_per_token": 5704,
            "copies_per_token": 3041,
            "conversions_by_shape_per_token": {
                "[1,1,5120]": 328, "[1,5120]": 297, "[1,1,8,64]": 276, "[1,32]": 264,
                "[1,2304]": 224, "[1,288]": 200, "[1,1,4,5120]": 167,
            },
            "sparse_attn_dispatches_per_call": {"before_hoist": 110, "after_hoist": 70},
        },
        "null_results": {
            "tile_major_whole_model_TPS": 0.2395,
            "tile_major_isolated_speedup": 3.05,
            "allocator_expandable_segments_TPS": 5.1358,
            "OMP_threads": {"2": 5.1119, "4": 5.0940, "8": 5.1231, "16": 5.0891},
            "metadata_only_dispatch_removal_TPS": 5.1052,
        },
        "verification": {
            "exact_control_reproduced_bit_exactly": {
                "max_abs_difference": 2.173757076263428,
                "relative_L2_difference": 0.1565176397562027,
                "reference_to_candidate_KL": 0.001334548112936318,
            },
            "relic_core_suite": "1143 passed, 13 pre-existing CUDA-environment failures (proven identical with and without the change)",
            "v41_suite_on_target": "529 passed, 19 skipped",
            "v41_suite_local": "562 passed, 39 skipped",
        },
        "target_TPS": 50.0,
        "target_met": False,
    }
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {TARGET.relative_to(ROOT)}: {manifest['file_count']} files, "
          f"{manifest['total_bytes']} bytes")


if __name__ == "__main__":
    main()
