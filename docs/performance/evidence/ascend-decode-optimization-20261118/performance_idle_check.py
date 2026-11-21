"""Check private processes and immutable host inputs before releasing the lease."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938")
CHECKPOINT = Path("<checkpoint>")


def main():
    safety = ROOT / "source/bounded_task.py"
    if hashlib.sha256(safety.read_bytes()).hexdigest() != "7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a":
        raise RuntimeError("unreviewed ownership helper")
    spec = importlib.util.spec_from_file_location("performance_idle_ownership", safety)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    checked = []
    for path in sorted((ROOT / "logs").glob("*/summary.json")):
        row = json.loads(path.read_bytes())
        if "ownership_token" not in row:
            continue
        live = helper.OwnedProcesses(row["ownership_token"]).scan()
        if live:
            raise RuntimeError(f"owned children are still live for {path}: {live}")
        checked.append({"receipt": str(path), "returncode": row.get("returncode"),
                        "cleanup_complete": row.get("cleanup", {}).get("complete"), "live_owned": []})
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = proc.joinpath("cmdline").read_bytes()
            state = helper.process_identity(int(proc.name))["state"]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if state != "Z" and b"python" in command.split(b"\0")[0] and str(ROOT).encode() in command:
            raise RuntimeError(f"a private Python process still runs: {int(proc.name)}")
    cann_path = Path("/usr/local/Ascend/cann-9.2.0/set_env.sh")
    cann = hashlib.sha256(cann_path.read_bytes()).hexdigest()
    if cann != "bbbda1c0f925b0e8673e8c83958d6bb333e0e7d3ed0eb389db1457d92b3c3b12":
        raise RuntimeError("shared CANN environment changed")
    links = {name: str(Path(name).resolve()) for name in
             ("/usr/local/Ascend/cann", "/usr/local/Ascend/ascend-toolkit/latest")}
    if set(links.values()) != {"/usr/local/Ascend/cann-9.2.0"}:
        raise RuntimeError("shared CANN links changed")
    baseline = json.loads(Path("<workspace>/relic-v41-20261116-210/logs/model-attempts/original-fp4-fp8-tp8-verified/preconditions.json").read_bytes())["checkpoint_stats"]
    current = {}
    for name in baseline:
        st = (CHECKPOINT / name).stat()
        current[name] = {"st_dev": st.st_dev, "st_ino": st.st_ino, "st_size": st.st_size,
                         "st_mtime_ns": st.st_mtime_ns, "st_ctime_ns": st.st_ctime_ns}
    if current != baseline:
        raise RuntimeError("a checkpoint identity or timestamp changed")
    prior = json.loads(Path("<workspace>/relic-expert-gemm-20261117-1046/logs/final-expert-host-check-after-final-tests.json").read_bytes())
    jenkins = subprocess.check_output(["ps", "-p", "115417", "-o", "pid,lstart,args"], text=True)
    if jenkins != prior["jenkins_agent"]:
        raise RuntimeError("the shared Jenkins agent identity changed")
    npu = subprocess.check_output(["npu-smi", "info"], text=True)
    if npu.count("No running processes found in NPU") != 8:
        raise RuntimeError("one or more NPUs still have live processes")
    import torch
    import torch_npu
    import triton
    if torch.__version__ != "2.10.0+cpu" or torch_npu.__version__ != "2.10.0.post7":
        raise RuntimeError("the verified Torch/NPU stack changed")
    patch = Path("<workspace>/relic-online-unpack-20261117-0534/venv311/lib/python3.11/site-packages/triton/backends/ascend/npu_utils.cpp")
    patch_sha = hashlib.sha256(patch.read_bytes()).hexdigest()
    if patch_sha != "92cbdcaa5d2ba02f8dd7b05363428f8ab0cd94d943264dd06cd1457ed4ee7d50":
        raise RuntimeError("the reviewed private Triton compatibility source changed")
    result = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "receipts": checked,
              "owned_processes_live": [], "private_python_processes_live": [],
              "empty_NPU_process_snapshots": 8, "checkpoint_stat_identity_count": len(current),
              "checkpoint_stat_identities_unchanged": True, "full_checksum_pass_claimed": False,
              "CANN_SHA": cann, "CANN_links": links, "jenkins_agent": jenkins,
              "private_Triton_patch_SHA": patch_sha, "torch": torch.__version__,
              "torch_npu": torch_npu.__version__, "triton": triton.__version__,
              "no_device_reset_or_shared_cache_cleanup": True, "lease_release": "pending fleet release"}
    target = ROOT / "logs" / ("idle-safety-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    with target.with_suffix(".json").open("x") as handle:
        json.dump(result, handle, indent=2)
    with target.with_suffix(".npu.log").open("x") as handle:
        handle.write(npu)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
