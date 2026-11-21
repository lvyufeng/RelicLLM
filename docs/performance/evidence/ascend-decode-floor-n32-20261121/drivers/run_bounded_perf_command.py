"""Bounded generic command runner for root-owned optimization probes."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path("/data1/relic-v41-perf-20261118-0938")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--project", choices=["relic-core", "RelicLLM"], default="RelicLLM")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    if not arguments:
        raise ValueError("a bounded probe needs an actual command")
    path = ROOT / "source/bounded_task.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != "7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a":
        raise RuntimeError("unreviewed cleanup helper")
    spec = importlib.util.spec_from_file_location("root_probe_safety", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    output = ROOT / "logs" / (args.name + "-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(exist_ok=False)
    env = dict(os.environ, RELIC_PERF_OUTPUT=str(output), RELIC_CPU_GEMV_CACHE=str(ROOT / "cache/cpu"))
    command = [sys.executable, *arguments]
    result = helper.bounded_attempt(args.name, command, args.timeout, output, env, cwd=ROOT / "source" / args.project)
    with (output / "summary.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"output": str(output), "returncode": result["returncode"],
                      "owned_cleanup_complete": result["cleanup"]["complete"]}), flush=True)
    return 0 if result["returncode"] == 0 and result["cleanup"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
