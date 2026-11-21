"""Copy the round-3 attribution receipts into a small versioned evidence bundle.

These are measurements that explain the remaining cost of the verified 4.934599 TPS candidate: a
timing ablation that splits the token, the probe that closes the hardware-4-bit avenue, and the
run that shows the bounded operand cache loses to the recorded expert bodies.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
TARGET = ROOT / "docs/performance/evidence/ascend-decode-round3-20261119"
SOURCES = {
    # Eight rank JSONs of the private timing ablation, plus the run that holds the timed window.
    "timing-ablation": ".coord/perf-ablation-results/ablation-20261119T074707Z",
    "operand-cache-loses-to-recorded-bodies": ".coord/perf-cache-no-graph-results/cache-20261119T084111Z",
    "hardware-4bit-probe": ".coord/perf-int4-probe/int4-support-v2-20261119T083847Z",
}
FILES = (
    ".coord/profile_actual_cube.py",
    ".coord/benchmark_reduce_floor.py",
    ".coord/benchmark_int4_support_v2.py",
)
SUFFIXES = {".json", ".log"}


def main():
    if TARGET.exists():
        raise RuntimeError("the round-3 evidence bundle already exists; do not overwrite it")
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
        source = ROOT / relative
        if not source.is_file():
            raise RuntimeError(f"missing probe source: {source}")
        shutil.copyfile(source, TARGET / source.name)
    files = [{"path": str(path.relative_to(TARGET)), "bytes": path.stat().st_size,
              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
             for path in sorted(TARGET.rglob("*")) if path.is_file()]
    manifest = {
        "milestone": "round 3: attribution of the remaining cost of the 4.934599 TPS candidate",
        "file_count": len(files),
        "total_bytes": sum(row["bytes"] for row in files),
        "files": files,
        "verified_candidate_TPS": 4.93459894468458,
        "exact_baseline_TPS": 0.19444076009188987,
        "meets_50_TPS": False,
        "timing_ablation": {
            "numbers_are_wrong_on_purpose": True,
            "window": "8 forwards after 3 warmups, context 24, all eight ranks",
            "baseline_TPS": 5.081,
            "no_routing_host_read_TPS": 5.659,
            "no_expert_work_TPS": 10.705,
            "expert_share_of_token": "52.5%",
            "routing_read_share_of_token": "10.2%",
        },
        "findings": [
            "the routed-expert bodies are half of the token; the forty per-layer routing drains are "
            "a tenth more",
            "captured collectives cost 0.0939 ms each and are about 3.6% of a token, so they are not "
            "the bottleneck",
            "a 4 GiB per-expert decoded-operand cache without recordings measured 3.169 TPS against "
            "4.935 TPS with recordings, so the cache loses",
            "hardware W4A16 cannot consume the checkpoint's E2M1 codes because that code-to-value map "
            "is not two's-complement int4; the INT8 widening is inherent to the original format",
        ],
        "numerical_contract": {
            "exact_default_path": "unchanged",
            "later_changes_added_zero_FP32_bit_differences": True,
            "complete_Engram_CPU": True,
            "serving_guards_changed": False,
            "checkpoint_rewritten": False,
        },
    }
    with (TARGET / "manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps({"evidence": str(TARGET), "files": len(files),
                      "bytes": manifest["total_bytes"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())