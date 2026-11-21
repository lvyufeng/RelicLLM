"""Run a checkpoint-backed expert comparison with owned-child cleanup."""
from __future__ import annotations
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path("/data1/relic-expert-gemm-20261117-1046")


def main():
    path=ROOT/"source/bounded_task.py"
    if hashlib.sha256(path.read_bytes()).hexdigest()!="7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a":
        raise RuntimeError("unreviewed cleanup helper")
    spec=importlib.util.spec_from_file_location("expert_probe_safety",path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output=ROOT/"logs"/f"original-expert-{stamp}"
    output.mkdir(exist_ok=False)
    result=helper.bounded_attempt("original-expert",[sys.executable,"-u",str(ROOT/"source/probe_original_expert.py"),
             "--report",str(output/"report.json"),"--rows","1"],900,output,dict(os.environ),
             cwd=ROOT/"source/RelicLLM")
    with (output/"summary.json").open("x") as handle:json.dump(result,handle,indent=2)
    return 0 if result["returncode"]==0 and result["cleanup"]["complete"] else 1


if __name__=="__main__":raise SystemExit(main())
