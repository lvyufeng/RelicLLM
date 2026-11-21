"""Final owned-process, checkpoint and shared-host checks before lease release."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("<workspace>/relic-v41-infer-20261117-1424")
CHECKPOINT = Path("<checkpoint>")
SAFETY_SHA = "7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a"
CANN_SHA = "bbbda1c0f925b0e8673e8c83958d6bb333e0e7d3ed0eb389db1457d92b3c3b12"
TRITON_PATCH_SHA = "92cbdcaa5d2ba02f8dd7b05363428f8ab0cd94d943264dd06cd1457ed4ee7d50"


def main():
    safety = ROOT / "source/bounded_task.py"
    if hashlib.sha256(safety.read_bytes()).hexdigest() != SAFETY_SHA:
        raise RuntimeError("unreviewed ownership helper")
    spec = importlib.util.spec_from_file_location("final_inference_ownership", safety)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    receipts = []
    for path in sorted((ROOT / "logs").glob("*/summary.json")):
        row = json.loads(path.read_bytes())
        if "ownership_token" not in row:
            continue
        live = helper.OwnedProcesses(row["ownership_token"]).scan()
        if live:
            raise RuntimeError("task-owned children still live: " + str(live))
        receipts.append({"path": str(path), "live_owned_processes": live,
                         "returncode": row.get("returncode"),
                         "recorded_cleanup_complete": row.get("cleanup", {}).get("complete")})
    remaining = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            command = proc.joinpath("cmdline").read_bytes()
            state = helper.process_identity(int(proc.name))["state"]
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if state != "Z" and b"python" in command.split(b"\0")[0] and str(ROOT).encode() in command:
            remaining.append({"pid": int(proc.name), "command": command.replace(b"\0", b" ").decode()})
    if remaining:
        raise RuntimeError("a task-private Python process remains active: " + str(remaining))
    cann = hashlib.sha256(Path("/usr/local/Ascend/cann-9.2.0/set_env.sh").read_bytes()).hexdigest()
    links = {name: str(Path(name).resolve()) for name in
             ("/usr/local/Ascend/cann", "/usr/local/Ascend/ascend-toolkit/latest")}
    if cann != CANN_SHA or set(links.values()) != {"/usr/local/Ascend/cann-9.2.0"}:
        raise RuntimeError("shared CANN environment or links changed")
    baseline_path = Path("<workspace>/relic-v41-20261116-210/logs/model-attempts/original-fp4-fp8-tp8-verified/preconditions.json")
    baseline = json.loads(baseline_path.read_bytes())["checkpoint_stats"]
    current = {}
    for name in baseline:
        st = (CHECKPOINT / name).stat()
        current[name] = {"st_dev": st.st_dev, "st_ino": st.st_ino, "st_size": st.st_size,
                         "st_mtime_ns": st.st_mtime_ns, "st_ctime_ns": st.st_ctime_ns}
    if current != baseline:
        raise RuntimeError("a verified original checkpoint identity/timestamp changed")
    prior = json.loads(Path("<workspace>/relic-expert-gemm-20261117-1046/logs/final-expert-host-check-after-final-tests.json").read_bytes())
    jenkins = subprocess.check_output(["ps", "-p", "115417", "-o", "pid,lstart,args"], text=True)
    if jenkins != prior["jenkins_agent"]:
        raise RuntimeError("the shared Jenkins agent identity/command changed")
    npu = subprocess.check_output(["npu-smi", "info"], text=True)
    if npu.count("No running processes found in NPU") != 8:
        raise RuntimeError("one or more NPUs still have a live process")
    import torch
    import torch_npu
    import triton
    if torch.__version__ != "2.10.0+cpu" or torch_npu.__version__ != "2.10.0.post7":
        raise RuntimeError("the exact Torch/NPU stack changed")
    patch = Path("<workspace>/relic-online-unpack-20261117-0534/venv311/lib/python3.11/site-packages/triton/backends/ascend/npu_utils.cpp")
    patch_sha = hashlib.sha256(patch.read_bytes()).hexdigest()
    if patch_sha != TRITON_PATCH_SHA:
        raise RuntimeError("the reviewed private Triton compatibility source changed")
    result = {"timestamp": dt.datetime.now(dt.timezone.utc).isoformat(), "python": sys.executable,
              "python_version": sys.version, "torch": torch.__version__, "torch_npu": torch_npu.__version__,
              "triton": triton.__version__, "private_triton_patch_sha256": patch_sha,
              "task_owned_live_processes": [], "private_python_processes": [], "receipts": receipts,
              "receipt_count_checked": len(receipts), "checkpoint_stat_identity_count": len(current),
              "original_checkpoint_unchanged_by_stat_identity_check": True,
              "new_full_checkpoint_checksum_pass_claimed": False,
              "cann_env_sha256": cann, "cann_links": links, "jenkins_agent": jenkins,
              "jenkins_matches_preceding_acceptance": True, "empty_npu_process_snapshots": 8,
              "no_device_reset_or_shared_cleanup": True, "retained_private_root": str(ROOT),
              "lease_release": "pending fleet-tool release after this check"}
    with (ROOT / "logs/final-inference-host-check.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    with (ROOT / "logs/final-inference-npu-smi.log").open("x") as handle:
        handle.write(npu)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
