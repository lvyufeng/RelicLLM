"""Bounded actual NPU checks for inference call sites, without CUDA graph claims."""
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT=Path("/data1/relic-v41-infer-20261117-1424")


def main():
    path=ROOT/"source/bounded_task.py"
    if hashlib.sha256(path.read_bytes()).hexdigest()!="7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a":raise RuntimeError("cleanup helper changed")
    spec=importlib.util.spec_from_file_location("infer_ops_cleanup",path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    output=ROOT/"logs"/("inference-ops-"+dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(exist_ok=False)
    env=dict(os.environ,RELIC_INFERENCE_TEST_DEVICE="npu:0",RELIC_ASCEND_ROTARY_DEVICE="npu:0",RELIC_ASCEND_ENGRAM_DEVICE="npu:0")
    command=[sys.executable,"-m","pytest","-p","no:cacheprovider","tests/test_v41_ascend_activation.py",
             "tests/test_v41_ascend_rotary.py","tests/test_v41_host_engram.py","tests/test_models_deepseek_v4_1_loader.py",
             "-q","-rs","--junitxml="+str(output/"pytest.xml")]
    result=helper.bounded_attempt("inference-ops",command,900,output,env,cwd=ROOT/"source/RelicLLM")
    with (output/"summary.json").open("x") as f:json.dump(result,f,indent=2)
    return 0 if result["returncode"]==0 and result["cleanup"]["complete"] else 1


if __name__=="__main__":raise SystemExit(main())
