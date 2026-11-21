"""Scoped private-work safety receipt at a new lease boundary, without shared writes."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/data1/relic-v41-perf-20261118-0938")
CHECKPOINT = Path("/data1/relic-v41-20261116-210/checkpoint")
SAFETY_SHA = "7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a"
CANN_SHA = "bbbda1c0f925b0e8673e8c83958d6bb333e0e7d3ed0eb389db1457d92b3c3b12"


def main():
    if subprocess.check_output(["hostname"], text=True).strip() != "ascend216":
        raise RuntimeError("the authenticated control channel reached a different host")
    source = ROOT / "source/bounded_task.py"
    if hashlib.sha256(source.read_bytes()).hexdigest() != SAFETY_SHA:
        raise RuntimeError("unreviewed process ownership helper")
    spec = importlib.util.spec_from_file_location("perf_round2_ownership", source)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    live, receipts = [], []
    for path in sorted((ROOT / "logs").glob("*/summary.json")):
        row = json.loads(path.read_bytes())
        if "ownership_token" not in row:
            continue
        current = helper.OwnedProcesses(row["ownership_token"]).scan()
        live.extend(current)
        receipts.append({"path": str(path), "returncode": row.get("returncode"),
                         "cleanup_complete": row.get("cleanup", {}).get("complete"),
                         "live_processes": current})
    if live:
        raise RuntimeError("private token-attributed jobs remain active: " + str(live))
    private = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            argv = proc.joinpath("cmdline").read_bytes()
            identity = helper.process_identity(int(proc.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if identity["state"] != "Z" and b"python" in argv.split(b"\0")[0] and str(ROOT).encode() in argv:
            private.append({"pid": int(proc.name), "command": argv.replace(b"\0", b" ").decode()})
    if private:
        raise RuntimeError("private Python jobs remain: " + str(private))
    baseline = json.loads(Path("/data1/relic-v41-20261116-210/logs/model-attempts/original-fp4-fp8-tp8-verified/preconditions.json").read_bytes())["checkpoint_stats"]
    current = {}
    for name in baseline:
        st = (CHECKPOINT / name).stat()
        current[name] = {"st_dev": st.st_dev, "st_ino": st.st_ino, "st_size": st.st_size,
                         "st_mtime_ns": st.st_mtime_ns, "st_ctime_ns": st.st_ctime_ns}
    if current != baseline:
        raise RuntimeError("the original checkpoint stat identity changed")
    cann = hashlib.sha256(Path("/usr/local/Ascend/cann-9.2.0/set_env.sh").read_bytes()).hexdigest()
    if cann != CANN_SHA:
        raise RuntimeError("the shared CANN environment changed")
    links = {name: str(Path(name).resolve()) for name in ("/usr/local/Ascend/cann", "/usr/local/Ascend/ascend-toolkit/latest")}
    if set(links.values()) != {"/usr/local/Ascend/cann-9.2.0"}:
        raise RuntimeError("shared CANN links changed")
    npu = subprocess.check_output(["npu-smi", "info"], text=True)
    if npu.count("No running processes found in NPU") != 8:
        raise RuntimeError("not all eight NPU process tables are empty")
    report = {"timestamp_UTC": dt.datetime.now(dt.timezone.utc).isoformat(),
              "hostname": "ascend216", "private_live_jobs": [], "private_job_receipts": receipts,
              "all_eight_NPU_process_tables_empty": True, "checkpoint_stat_identity_unchanged": True,
              "checkpoint_files_compared": len(current), "no_new_full_checkpoint_checksum_claim": True,
              "CANN_env_sha256": cann, "CANN_links": links,
              "jenkins_agent_identity": subprocess.check_output(["ps", "-p", "115417", "-o", "pid,lstart,args"], text=True),
              "no_shared_configuration_change_or_device_reset": True}
    destination = ROOT / "logs" / ("new-lease-safety-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json")
    with destination.open("x") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"receipt": str(destination), "checks_passed": True, "checkpoint_files": len(current),
                      "prior_private_receipts": len(receipts)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
