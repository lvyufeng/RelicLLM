"""Actual bounded original-checkpoint inference, with source and cleanup receipts."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path("<workspace>/relic-v41-infer-20261117-1424")
CHECKPOINT=Path("<checkpoint>")


def main():
    path=ROOT/"source/bounded_task.py"
    if hashlib.sha256(path.read_bytes()).hexdigest()!="7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a":
        raise RuntimeError("unreviewed cleanup helper")
    spec=importlib.util.spec_from_file_location("inference_safety",path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    stamp=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output=ROOT/"logs"/f"inference-{stamp}"
    output.mkdir(exist_ok=False)
    command=[sys.executable,"-m","torch.distributed.run","--standalone","--nnodes=1",
             "--nproc-per-node=8","-m","relicllm.cli.generate_v41","--checkpoint",str(CHECKPOINT),
             "--expert-device","npu:0","--expert-world","8","--expert-pool-rows","0",
             "--expert-hot-rows","0","--no-expert-batched","--expert-deal","id",
             "--no-decode-graphs","--threads","4","--max-seq-len","128",
             "--max-new-tokens","4","--temperature","0","--prompt","The capital of France is",
             "--dump-logits",str(output/"generated-logits.pt")]
    hashes={}
    for project,prefix in (("relic-core","relic_core/kernels"),("RelicLLM","relicllm/models/deepseek_v4_1")):
        for source in (ROOT/"source"/project/prefix).glob("*.py"):
            hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
    with (output/"preconditions.json").open("x") as handle:
        json.dump({"source_hashes":hashes,"python":sys.executable,"original_checkpoint":str(CHECKPOINT),
                   "requested_prompt":"The capital of France is","requested_new_tokens":4,
                   "reference_reduction_enabled":os.environ.get("POCKETLLM_V41_MMA_EMULATE"),
                   "checkpoint_rewritten":False,"serving_guard_bypassed":False,
                   "inference_requested":True,"generation_success_claimed_before_execution":False},handle,indent=2)
    result=helper.bounded_attempt("original-inference",command,1200,output,dict(os.environ),
                                  cwd=ROOT/"source/RelicLLM")
    with (output/"summary.json").open("x") as handle:json.dump(result,handle,indent=2)
    return 0 if result["returncode"]==0 and result["cleanup"]["complete"] else 1


if __name__=="__main__":raise SystemExit(main())
