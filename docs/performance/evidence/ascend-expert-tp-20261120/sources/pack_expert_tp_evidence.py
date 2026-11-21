"""Bundle bounded TP evidence into one compressed transfer; exclude tensor payloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938")
DEST = ROOT / "logs/expert-TP2-EP4-evidence-20261120-v1.zip"


def exactly_one(pattern):
    paths = list((ROOT / "logs").glob(pattern))
    if len(paths) != 1:
        raise RuntimeError(f"ambiguous required evidence: {pattern}, {paths}")
    return paths[0]


def main():
    picked = {}
    def include(path, name):
        if not path.is_file() or path.suffix == ".pt":
            raise RuntimeError("missing receipt or forbidden tensor transfer")
        if path.stat().st_size > 2**20:
            raise RuntimeError("unexpected oversized evidence file")
        if name in picked:
            raise RuntimeError("duplicate evidence destination")
        picked[name] = path
    for tag, pattern in (
        ("generation-audit", "compare-original-expert-TP2-EP4-models-20*T*Z"),
        ("matched-input-audit", "compare-original-expert-TP2-EP4-matched-inputs-20*T*Z"),
        ("original-real-replay", "original-expert-TP2-EP4-real-replay-20*T*Z"),
        ("final-real-replay", "original-expert-TP2-EP4-final-replay-20*T*Z"),
    ):
        directory = exactly_one(pattern)
        status = json.loads((directory / "summary.json").read_bytes())
        if status["returncode"] or not status["cleanup"]["complete"]:
            raise RuntimeError("only independently completed evidence is retained")
        for path in directory.iterdir():
            if path.is_file() and path.suffix in (".json", ".log"):
                include(path, tag + "/" + path.name)
    for tag, pattern, keep_sources in (
        ("TP2-sustained", "original-expert-tp2-host-table-HC-steps64-20*T*Z", True),
        ("TP2-short", "original-expert-tp2-host-table-HC-20*T*Z", False),
        ("EP8-short", "original-expert-ep8-host-table-HC-20*T*Z", False),
        ("TP2-matched-input", "original-expert-tp2-forced-host-table-HC-20*T*Z", True),
        ("EP8-matched-input", "original-expert-ep8-forced-host-table-HC-20*T*Z", False),
        ("native-matched-input", "original-expert-native-forced-host-table-HC-20*T*Z", False),
    ):
        directory = exactly_one(pattern)
        include(directory / "preconditions.json", tag + "/preconditions.json")
        for path in directory.iterdir():
            if path.is_file() and (path.name == "summary.json" or path.name.startswith("original-expert-") and path.suffix in (".json", ".log")):
                include(path, tag + "/" + path.name)
        if keep_sources:
            for path in (directory / "source-snapshots").rglob("*"):
                if path.is_file():
                    include(path, tag + "/source-snapshots/" + path.relative_to(directory / "source-snapshots").as_posix())
            include(directory / "profile-expert-tp-generated-rank-0.py", tag + "/generated-source-rank-0.py")
    include(ROOT / "logs/new-lease-safety-20261120T115137Z.json", "safety/beginning.json")
    safety = sorted((ROOT / "logs").glob("new-lease-safety-20*T*Z.json"))[-1]
    include(safety, "safety/ending.json")
    for name in ("probe_expert_tp2_ep4.py", "compare_expert_tp_models.py", "compare_expert_tp_matched.py",
                 "pack_expert_tp_evidence.py", "perf_round2_safety.py"):
        include(ROOT / "source" / name, "sources/" + name)
    for relative in ("RelicLLM/relicllm/models/deepseek_v4_1/ascend_expert_tp.py",
                     "relic-core/relic_core/kernels/expert_tp_moe.py",
                     "relic-core/relic_core/kernels/ascendc_online_tile_cube.py"):
        include(ROOT / "source" / relative, "end-of-round-sources/" + relative)
    files = {name: {"origin": str(path), "bytes": path.stat().st_size,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in sorted(picked.items())}
    if len(files) > 160 or sum(row["bytes"] for row in files.values()) > 16 * 2**20:
        raise RuntimeError("bounded receipt bundle exceeded expected size")
    manifest = {"tensor_payloads_not_transferred": True, "files": files,
                "accepted_K128_library_SHA256": hashlib.sha256((ROOT / "cache/online-tile-cube-k128-v2/lib/librelic_online_tile_cube.so").read_bytes()).hexdigest()}
    with DEST.open("xb") as stream, zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, path in sorted(picked.items()):
            archive.write(path, name)
        archive.writestr("transfer-manifest.json", json.dumps(manifest, indent=2))
    with zipfile.ZipFile(DEST) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("archive failed CRC validation")
    print(json.dumps({"archive": str(DEST), "sha256": hashlib.sha256(DEST.read_bytes()).hexdigest(),
                      "files": len(files), "bytes": DEST.stat().st_size}), flush=True)


if __name__ == "__main__":
    main()
