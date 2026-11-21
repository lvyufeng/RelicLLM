"""Compress only completed profiler receipts/traces in the exact private run directory."""
import argparse
import json
from pathlib import Path
import zipfile

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938/logs").resolve()

parser = argparse.ArgumentParser()
parser.add_argument("run")
args = parser.parse_args()
source = (ROOT / args.run).resolve(strict=True)
if source.parent != ROOT or not source.name.startswith("current-bottleneck-"):
    raise RuntimeError("only exact task-private current-bottleneck runs may be archived")
summary = json.loads((source / "summary.json").read_text())
if summary["returncode"] != 0 or not summary["cleanup"]["complete"]:
    raise RuntimeError("only completed, cleaned profiler runs may be archived")
files = [p for p in source.glob("*.json") if p.is_file()]
files += [p for p in source.glob("profiler-rank*/**/*") if p.is_file() and
          p.name in ("kernel_details.csv", "trace_view.json", "op_statistic.csv", "memory_record.csv")]
archive = source / "profile-bundle.zip"
with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as stream:
    for path in files:
        stream.write(path, path.relative_to(source))
print(json.dumps({"archive": str(archive), "bytes": archive.stat().st_size,
                  "files": len(files), "uncompressed_bytes": sum(p.stat().st_size for p in files)}), flush=True)
