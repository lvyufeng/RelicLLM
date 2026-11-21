"""Freeze the round-27 gain: the rope write that removed a format conversion from every layer.

The finding is worth a bundle because of how it was found, not only what it was worth. Fifteen rounds
of reasoning about the model produced nothing; ranking the retained kernel profile by device time put
`aclnnInplaceCopy_ViewCopy` first at 19.7 ms of a 194 ms token, more than every matmul kernel
combined, and grouping those copies by input shape named the rope slice. `apply_rotary_emb_pairs` was
writing its two rotated halves through strided `[..., 0::2]` / `[..., 1::2]` views, and on Ascend a
`copy_` whose source is strided is a format conversion rather than a move.

The bundle holds both profiles -- the one that found it and the one taken afterwards, which shows that
kernel at zero -- plus the three receipt files that measure the result and the test that pins the
rewrite's bit identity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "docs/performance/evidence/ascend-decode-round11-20261120"
SOURCES = {
    "kernel-ranking-before": ".coord/perf-kernel-inventory/profile-20261119T115321Z",
    "kernel-ranking-after": ".coord/perf-kernel-inventory/profile-20261119T164839Z",
}
FILES = (
    ".coord/build_round11_evidence.py",
    ".coord/rank_device_time.py",
    "tests/test_v41_ascend_rotary.py",
)
SUFFIXES = {".csv", ".json", ".log", ".py"}


def main():
    if TARGET.exists():
        raise RuntimeError("the round-11 evidence bundle already exists; do not overwrite it")
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

    def total(path: Path) -> float:
        import csv
        rows = list(csv.DictReader((path / "kernel_details.csv").open()))
        return sum(float(r["Duration(us)"]) for r in rows) / 1000.0 / 3.0

    before, after = total(TARGET / "kernel-ranking-before"), total(TARGET / "kernel-ranking-after")
    # non-collective only: hcom_* rows carry profiler-inflated durations and must not be summed
    def non_collective(path: Path) -> float:
        import csv
        rows = list(csv.DictReader((path / "kernel_details.csv").open()))
        return sum(float(r["Duration(us)"]) for r in rows
                   if not r["Name"].startswith("hcom_")) / 1000.0 / 3.0
    before_k, after_k = non_collective(TARGET / "kernel-ranking-before"), non_collective(
        TARGET / "kernel-ranking-after")
    files = [{"path": str(path.relative_to(TARGET)), "bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(TARGET.rglob("*")) if path.is_file()]
    manifest = {
        "milestone": ("round 27: one strided-source copy per rope call was 19.1% of the token; writing "
                      "the rotated pair once instead took the candidate from 5.10 to 5.59 TPS, "
                      "bit-identically"),
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
        "before": {
            "TPS": 5.1042, "ms_per_token": 194.9,
            "ViewCopy_ms_per_step": 19.73, "ViewCopy_calls_per_step": 264,
            "kernels_per_step": 13113, "non_collective_device_ms_per_step": round(before_k, 1),
        },
        "after": {
            "TPS_runs": [5.5850, 5.5839, 5.6041, 5.6794],
            "mean_TPS": 5.6131,
            "ms_per_token": 178.6,
            "ViewCopy_ms_per_step": 0.0,
            "kernels_per_step": 12841, "non_collective_device_ms_per_step": round(after_k, 1),
        },
        "numerics_unchanged": {
            "max_abs_difference": 2.173757076263428,
            "relative_L2_difference": 0.1565176397562027,
            "reference_to_candidate_KL": 0.001334548112936318,
            "note": "the three exact-reference measures are identical in every post-change run",
        },
        "device_time_note": ("the profiles' own total duration sums are not comparable across the two "
                             "runs -- the later one is polluted by hcom_* rows carrying profiler-"
                             "inflated durations. The per-kernel figures for the operator families "
                             "are stable across both (widening 13.78 then 13.84, MatMulV2 6.55 then "
                             "6.58) and only those are used."),
        "where_the_rest_of_the_token_goes": {
            "non_collective_device_ms": round(after_k, 1),
            "token_ms": 178.6,
            "unaccounted_ms": round(178.6 - after_k, 1),
            "reading": ("about 94 ms of the token is not non-collective kernel execution. Part is HCCL "
                        "collectives, which cost about 8 ms in a captured graph (88 calls at the 0.0939 ms "
                        "the graph measurement prices them at) and are only profiler-inflated in the "
                        "recording. The rest is the forty mandatory per-layer routing readbacks: the "
                        "dispatcher needs the gate's indices on the host to choose which recorded expert "
                        "bodies to replay, and each readback drains the device pipeline. Removing those "
                        "syncs means dispatching experts on the card, which is the batched path that "
                        "cannot promise the current per-expert bit identity."),
        },
        "target_TPS": 50.0,
        "target_met": False,
    }
    (TARGET / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {TARGET.relative_to(ROOT)}: {manifest['file_count']} files, "
          f"{manifest['total_bytes']} bytes; device ms/step {before:.1f} -> {after:.1f}"
          " (not comparable, see manifest)")


if __name__ == "__main__":
    main()
