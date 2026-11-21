"""Final scoped host/process/isolation check before releasing the operator-test lease."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("<workspace>/relic-expert-gemm-20261117-1046")
CHECKPOINT = Path("<checkpoint>")
CANN_SHA256 = "bbbda1c0f925b0e8673e8c83958d6bb333e0e7d3ed0eb389db1457d92b3c3b12"
SAFETY_SHA256 = "7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a"


def main():
    source = ROOT / "source/bounded_task.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != SAFETY_SHA256:
        raise RuntimeError("unreviewed ownership-check helper")
    spec = importlib.util.spec_from_file_location("unpack_final_ownership", source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    live, receipts = [], []
    for path in sorted((ROOT / "logs").glob("*/summary.json")):
        row = json.loads(path.read_bytes())
        if "ownership_token" not in row:
            continue
        observed = helper.OwnedProcesses(row["ownership_token"]).scan()
        live.extend(observed)
        receipts.append({"path": str(path), "observed_live_processes": observed,
                         "recorded_returncode": row.get("returncode"),
                         "recorded_cleanup_complete": row.get("cleanup", {}).get("complete")})
    if live:
        raise RuntimeError("token-attributed operator work remains active: " + str(live))
    private_python = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = proc.joinpath("cmdline").read_bytes()
            state = helper.process_identity(int(proc.name))["state"]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if state != "Z" and b"python" in command.split(b"\0")[0] and str(ROOT).encode() in command:
            private_python.append({"pid": int(proc.name), "command": command.replace(b"\0", b" ").decode()})
    if private_python:
        raise RuntimeError("private Python task is still live: " + str(private_python))
    cann = hashlib.sha256(Path("/usr/local/Ascend/cann-9.2.0/set_env.sh").read_bytes()).hexdigest()
    if cann != CANN_SHA256:
        raise RuntimeError("shared CANN environment script changed")
    links = {p: str(Path(p).resolve()) for p in ("/usr/local/Ascend/cann", "/usr/local/Ascend/ascend-toolkit/latest")}
    if set(links.values()) != {"/usr/local/Ascend/cann-9.2.0"}:
        raise RuntimeError("shared CANN link targets changed")
    # Compare current payload identities/stats with the accepted prior original-
    # checkpoint attempt. No second hundreds-of-GB hash pass is represented here.
    baseline = json.loads(Path("<workspace>/relic-v41-20261116-210/logs/model-attempts/original-fp4-fp8-tp8-verified/preconditions.json").read_bytes())["checkpoint_stats"]
    current = {}
    for name in baseline:
        path = CHECKPOINT / name
        st = path.stat()
        current[name] = {"st_dev": st.st_dev, "st_ino": st.st_ino, "st_size": st.st_size,
                         "st_mtime_ns": st.st_mtime_ns, "st_ctime_ns": st.st_ctime_ns}
    if current != baseline:
        raise RuntimeError("original checkpoint payload identity/timestamp changed")
    npu = subprocess.check_output(["npu-smi", "info"], text=True)
    if npu.count("No running processes found in NPU") != 8:
        raise RuntimeError("final NPU process snapshot is not empty on all eight cards")
    import torch
    import torch_npu
    import triton
    if torch.__version__ != "2.10.0+cpu" or torch_npu.__version__ != "2.10.0.post7":
        raise RuntimeError("the exact Torch/NPU stack changed")
    patch_path = Path("<workspace>/relic-online-unpack-20261117-0534/venv311/lib/python3.11/site-packages/triton/backends/ascend/npu_utils.cpp")
    patch_sha = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    if patch_sha != "92cbdcaa5d2ba02f8dd7b05363428f8ab0cd94d943264dd06cd1457ed4ee7d50":
        raise RuntimeError("the reviewed private Triton compatibility source changed")
    result = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
              "python": sys.executable, "python_version": sys.version,
              "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "triton": triton.__version__, "private_triton_patch_sha256": patch_sha,
              "task_owned_live_processes": [], "private_python_processes": [],
              "receipt_count_checked": len(receipts), "receipts": receipts,
              "original_checkpoint_unchanged_by_stat_identity_check": True,
              "original_checkpoint_path": str(CHECKPOINT),
              "cann_env_sha256": cann, "cann_links": links,
              "jenkins_agent": subprocess.check_output(["ps", "-p", "115417", "-o", "pid,lstart,args"], text=True),
              "no_device_reset_or_shared_cleanup": True,
              "private_triton_patch_retained": "<workspace>/relic-online-unpack-20261117-0534/venv311/lib/python3.11/site-packages/triton/backends/ascend/npu_utils.cpp",
              "private_root_retained": str(ROOT), "lease_release": "pending fleet tool after final checks"}
    with (ROOT / "logs/final-expert-host-check-after-final-tests.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    with (ROOT / "logs/final-expert-npu-smi-after-final-tests.log").open("x") as handle:
        handle.write(npu)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
