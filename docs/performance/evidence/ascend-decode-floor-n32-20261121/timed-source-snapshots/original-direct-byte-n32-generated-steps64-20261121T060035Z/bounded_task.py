"""Bounded model attempts after complete, independently verified checkpoint receipts.

Only task-owned processes inheriting a fresh per-attempt environment token are
signaled, using Linux pidfds after checking UID, start time, PGID and session.
This includes reparented workers in separate sessions. This is not a universal
host process arbiter: descendants that clear their token before being observed,
and an external SIGKILL of this harness, cannot be handled reliably. Run only in
a quiescent task-owned directory, under the separately established fleet lease.
No source changes, capability bypass, resident banks, or success claims occur.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import ctypes
import datetime as dt
import errno
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import secrets
import signal
import socket
import stat
import subprocess
import sys
import time

ROOT = Path("<workspace>/relic-v41-20261116-210")
HF_REVISION = "314113f8a7dadb388215396ba5fbee2c610a04d6"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
PLATFORM_VARIANT_METADATA = {
    ".gitattributes": {
        "hf_bytes": 1701,
        "hf_blob_id": "9b59a7525c7b7870bd2cbdc41a4e91fd63fcb6cb",
        "modelscope_bytes": 2265,
        "modelscope_sha256": "df4c454c9953a39b58beb44d4e78fd66f201b8823f21494e7502a9d5ee8e2f9d",
    },
}
WEIGHT_BYTES = 510296708312
WEIGHTS = {f"model-{i:05d}-of-00048.safetensors" for i in range(1, 49)}
TOKEN_ENV = "RELIC_V41_ATTEMPT_TOKEN"
INTERRUPTION_SIGNALS = tuple(getattr(signal, name) for name in ("SIGINT", "SIGTERM", "SIGHUP")
                             if hasattr(signal, name))
CANN_PYTHONPATH = ("/usr/local/Ascend/cann-9.2.0/python/site-packages",
                   "/usr/local/Ascend/cann-9.2.0/opp/built-in/op_impl/ai_core/tbe")
LIMITATIONS = ["external harness SIGKILL cannot run cleanup",
               "unobserved descendants that clear the token can escape attribution"]
CRITICAL_ENV = ("PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE",
                "ASCEND_RT_VISIBLE_DEVICES", "RELICLLM_ASCEND_DISPATCH",
                "DEEPSEEK_V41_RESIDENT_EXPERTS", "DEEPSEEK_V41_PIN_RESIDENT_EXPERTS",
                "DEEPSEEK_FORCE_TORCH_GEMM", "DEEPSEEK_FP8_IMPL", "DEEPSEEK_FP4_IMPL",
                "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TMPDIR", "HOME",
                "ASCEND_HOME_PATH", "LD_LIBRARY_PATH", "ASCEND_CACHE_PATH",
                "ASCEND_WORK_PATH", "ASCEND_PROCESS_LOG_PATH")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def real_path(path, *, directory=False):
    path = Path(os.path.abspath(path))
    require(path.resolve() == path, f"aliased path refused: {path}")
    require(path.is_dir() if directory else path.is_file(), f"missing path: {path}")
    return path


def new_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def checkpoint_stats(root, entries):
    checkpoint = real_path(root / "checkpoint", directory=True)
    result = {}
    for name, expected in entries.items():
        relative = Path(name)
        require(relative.parts and not relative.is_absolute() and ".." not in relative.parts
                and relative.as_posix() == name, f"unsafe payload name: {name}")
        path = real_path(checkpoint / relative)
        require(path.is_relative_to(checkpoint), f"payload escaped checkpoint: {name}")
        info = path.stat()
        require(stat.S_ISREG(info.st_mode) and info.st_size == expected["Size"],
                f"current payload size/type mismatch: {name}")
        result[name] = {key: getattr(info, key) for key in
                        ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")}
    return result


def validate_preconditions(verification_path, audit_path, root=ROOT):
    root = real_path(root, directory=True)
    checkpoint = real_path(root / "checkpoint", directory=True)
    verification_path, audit_path = real_path(verification_path), real_path(audit_path)
    require(all(path.is_relative_to(root / "logs") for path in (verification_path, audit_path)),
            "receipts must be in the private logs directory")
    verification_bytes, audit_bytes = verification_path.read_bytes(), audit_path.read_bytes()
    verified, audit = json.loads(verification_bytes), json.loads(audit_bytes)
    hf_bytes = real_path(root / "hf-model.json").read_bytes()
    ms_bytes = real_path(root / "logs/ms-model-pinned.json").read_bytes()
    hf, ms = json.loads(hf_bytes), json.loads(ms_bytes)
    hf_entries = {row["rfilename"]: row for row in hf["siblings"]}
    blobs = [row for row in ms["Data"]["Files"] if row.get("Type") == "blob"]
    entries = {row["Path"]: row for row in blobs}
    require(len(entries) == len(blobs) == len(hf_entries) == len(hf["siblings"]) == 90,
            "expected unique 90-file snapshot inventories")
    require(hf["id"] == MODEL and hf["sha"] == HF_REVISION, "wrong pinned HF snapshot")
    require(set(hf_entries) - set(entries) == {".gitignore"}
            and set(entries) - set(hf_entries) == {"configuration.json"}, "snapshot inventories differ")
    require({name for name in entries if name.endswith(".safetensors")} == WEIGHTS,
            "wrong released shard filename set")
    require(sum(entries[name]["Size"] for name in WEIGHTS) == WEIGHT_BYTES, "wrong shard bytes")
    require(verified["passed"] is True and verified["model"] == MODEL
            and verified["hf_revision"] == HF_REVISION, "payload verification has not passed")
    require(Path(verified["checkpoint_dir"]) == checkpoint
            and Path(audit["checkpoint_dir"]) == checkpoint, "receipt is for another checkpoint")
    require(verified["hf_manifest_sha256"] == sha256(hf_bytes)
            and verified["modelscope_manifest_sha256"] == sha256(ms_bytes), "manifest receipt mismatch")
    verifier = real_path(root / "verify_modelscope_checkpoint.py")
    require(verified["verifier_sha256"] == sha256(verifier.read_bytes()), "wrong payload verifier source")
    rows = {row["path"]: row for row in verified["files"]}
    require(len(rows) == len(verified["files"]) == 90 and set(rows) == set(entries),
            "payload verification inventory is incomplete or duplicated")
    require(verified["hf_files"] == verified["modelscope_files"] == 90, "wrong verified file counts")
    variants = {name: record for name, record in PLATFORM_VARIANT_METADATA.items() if name in entries}
    require(verified.get("platform_variant_metadata", {}) == variants,
            "platform metadata variance receipt must exactly match the pinned allowed record")
    lfs_count = 0
    for name, entry in entries.items():
        row = rows[name]
        require(row["passed"] is True and row["expected_bytes"] == entry["Size"]
                and row["expected_sha256"] == row["actual_sha256"] == entry["Sha256"],
                f"payload receipt does not verify {name}")
        source = hf_entries.get(name)
        if name in variants:
            variant = variants[name]
            require(source is not None and not source.get("lfs")
                    and source["size"] == variant["hf_bytes"] and source["blobId"] == variant["hf_blob_id"]
                    and entry["Size"] == variant["modelscope_bytes"]
                    and entry["Sha256"] == variant["modelscope_sha256"],
                    f"pinned platform metadata identity mismatch: {name}")
            require(row.get("platform_variant_metadata") == variant
                    and row.get("hf_content_identity_claimed") is False
                    and "expected_hf_blob_id" not in row and "actual_hf_blob_id" not in row,
                    f"platform metadata row must disclose no HF content identity: {name}")
            continue  # Only this exact .gitattributes pair differs; its full MS SHA was checked above.
        require("platform_variant_metadata" not in row, f"unapproved platform metadata override: {name}")
        if source is not None:
            require(source["size"] == entry["Size"], f"cross-source size mismatch: {name}")
            if source.get("lfs"):
                lfs_count += 1
                require(source["lfs"]["sha256"] == entry["Sha256"]
                        and source["lfs"]["size"] == entry["Size"], f"LFS mismatch: {name}")
            else:
                require(row["expected_hf_blob_id"] == row["actual_hf_blob_id"] == source["blobId"],
                        f"non-LFS blob receipt mismatch: {name}")
    require(verified["lfs_entries_cross_checked"] == lfs_count
            and verified["bytes_verified"] == sum(row["Size"] for row in entries.values()),
            "incomplete verified byte/LFS totals")
    index = audit["index"]
    require(audit["complete"] is True and audit["shards"] == 48 and audit["tensors"] == 96085,
            "header audit is incomplete")
    require(index["indexed"] is True and index["shipped_tensors"] == 96085
            and index["pending_shards"] == [] and len(index["local_shards"]) == 48
            and set(index["local_shards"]) == WEIGHTS, "header index coverage is incomplete")
    require(audit["checks"] and all(row["status"] == "pass" and row["passed"] is True
                                   for row in audit["checks"]), "header audit has failed/undecided checks")
    stats = checkpoint_stats(root, entries)
    verified_at = dt.datetime.fromisoformat(verified["timestamp"])
    require(verified_at.tzinfo is not None, "payload receipt timestamp must include a timezone")
    verified_ns = int(verified_at.timestamp() * 1_000_000_000)
    require(all(max(row["st_mtime_ns"], row["st_ctime_ns"]) <= verified_ns for row in stats.values()),
            "checkpoint files changed after payload verification")
    return {"verification_report": str(verification_path), "header_audit_report": str(audit_path),
            "verification_report_sha256": sha256(verification_bytes),
            "header_audit_sha256": sha256(audit_bytes), "checkpoint_stats": stats,
            "platform_variant_metadata": variants}, entries


def launch_environment(source, root=ROOT):
    env = dict(source)
    exact = {"RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "GROUP_RANK", "ROLE_RANK",
             "ROLE_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "TP_RANK", "TP_WORLD_SIZE",
             "NCCL_ID_PATH", "POCKETLLM_NCCL_ID_PATH", "POCKETLLM_WORKER_BACKEND",
             "PYTHONOPTIMIZE", "PYTHONHOME", "LD_PRELOAD", "CUDA_VISIBLE_DEVICES", TOKEN_ENV}
    for key in list(env):
        if key in exact or key.startswith("TORCHELASTIC_"):
            env.pop(key)
    env.update({"DEEPSEEK_V41_RESIDENT_EXPERTS": "0", "DEEPSEEK_V41_PIN_RESIDENT_EXPERTS": "0",
                "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7", "RELICLLM_ASCEND_DISPATCH": "aclnn",
                "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1"})
    # Keep only the CANN Python paths from the already-sourced, validated environment.
    # Dropping these would create artificial missing `te`/op-compiler failures.
    sourced_paths = source.get("PYTHONPATH", "").split(os.pathsep)
    paths = [str(root / "src/relic-core"), str(root / "src/RelicLLM")]
    paths.extend(path for path in CANN_PYTHONPATH if path in sourced_paths)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def pidfd_syscall(operation, *arguments):
    """Linux UAPI fallback for Python builds that omitted the optional wrappers.

    asm-generic/unistd.h defines pidfd_send_signal=424 and pidfd_open=434;
    x86_64/unistd_64.h uses the same numbers. Do not infer other ABIs' numbers.
    Native Python wrappers remain preferred. Kernel/seccomp/permission errors are
    propagated, never converted to a less precise PID-based kill fallback.
    """
    require(sys.platform.startswith("linux") and platform.machine() in {"aarch64", "x86_64"}
            and ctypes.sizeof(ctypes.c_void_p) == 8,
            "pidfd syscall fallback supports Linux aarch64/x86_64 LP64 only")
    numbers = {"pidfd_open": 434, "pidfd_send_signal": 424}
    require(operation in numbers, "unsupported pidfd syscall operation")
    libc = ctypes.CDLL(None, use_errno=True)
    syscall = libc.syscall
    syscall.restype = ctypes.c_long
    ctypes.set_errno(0)
    value = syscall(ctypes.c_long(numbers[operation]), *arguments)
    if value == -1:
        code = ctypes.get_errno() or errno.EIO
        raise OSError(code, f"{operation}: {os.strerror(code)}")
    return int(value)


def pidfd_open(pid):
    native = getattr(os, "pidfd_open", None)
    if native is not None:
        return native(pid, 0)
    return pidfd_syscall("pidfd_open", ctypes.c_int(pid), ctypes.c_uint(0))


def pidfd_send_signal(fd, signum):
    native = getattr(signal, "pidfd_send_signal", None)
    if native is not None:
        return native(fd, signum, None, 0)
    return pidfd_syscall("pidfd_send_signal", ctypes.c_int(fd), ctypes.c_int(signum),
                         ctypes.c_void_p(None), ctypes.c_uint(0))


def require_procfs_pidfds():
    require(sys.platform.startswith("linux") and Path("/proc/self/environ").is_file(),
            "Linux procfs is required before starting any child")
    fd = pidfd_open(os.getpid())
    try:
        pidfd_send_signal(fd, 0)
    finally:
        os.close(fd)


def process_identity(pid):
    directory = Path("/proc") / str(pid)
    text = (directory / "stat").read_text()
    fields = text.rsplit(")", 1)[1].split()
    return {"pid": pid, "starttime": int(fields[19]), "pgid": int(fields[2]),
            "sid": int(fields[3]), "state": fields[0], "uid": directory.stat().st_uid}


class OwnedProcesses:
    def __init__(self, token):
        self.token = f"{TOKEN_ENV}={token}".encode()
        self.seen = {}
        self.errors = []
        self.signals = []
        self.token_losses = {}
        self.ownership_observations = []

    def record_token_loss(self, identity, reason):
        """Retain the failure, but allow later proof that this exact identity terminated."""
        key = (identity["pid"], identity["starttime"])
        self.token_losses[key] = {"identity": dict(identity), "reason": reason}
        self.ownership_observations.append({"identity": dict(identity), "event": "token_unreadable",
                                            "reason": reason})

    def reconcile_token_losses(self):
        unresolved, reconciled = [], []
        for lost in self.token_losses.values():
            identity = lost["identity"]
            proof = None
            try:
                current = process_identity(identity["pid"])
                if current["starttime"] != identity["starttime"]:
                    proof = "original identity gone; PID has a different start time"
                elif current["state"] == "Z":
                    proof = "same-starttime identity rechecked as zombie"
                    self.seen[(current["pid"], current["starttime"])] = current
            except (FileNotFoundError, ProcessLookupError):
                proof = "original identity rechecked as absent from procfs"
            except OSError:
                pass  # Unreadable is not evidence of exit, and never permits a signal.
            if proof is None:
                unresolved.append(lost["reason"])
            else:
                reconciled.append({**lost, "terminal_proof": proof})
        return unresolved, reconciled

    def has_token(self, pid):
        return self.token in (Path("/proc") / str(pid) / "environ").read_bytes().split(b"\0")

    def scan(self):
        live = []
        for directory in Path("/proc").iterdir():
            if not directory.name.isdigit() or int(directory.name) == os.getpid():
                continue
            pid = int(directory.name)
            try:
                if directory.stat().st_uid != os.getuid():
                    continue
                identity = process_identity(pid)
                if identity["uid"] != os.getuid():
                    continue
                key = (pid, identity["starttime"])
                if identity["state"] == "Z":
                    if key in self.seen:
                        self.seen[key] = identity
                    continue
                if self.has_token(pid):
                    self.seen[key] = identity
                    live.append(identity)
                elif key in self.seen:
                    self.record_token_loss(identity, f"observed owned process lost its token: {pid}")
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (PermissionError, OSError) as error:
                # An unreadable same-UID process could be an owned worker: fail closed.
                self.errors.append(f"procfs ownership scan uncertain for {pid}: {error}")
        return live

    def send(self, identity, signum):
        pid, fd = identity["pid"], None
        try:
            fd = pidfd_open(pid)
            current = process_identity(pid)
            keys = ("starttime", "uid", "pgid", "sid")
            if any(current[key] != identity[key] for key in keys):
                return  # PID/session changed; rescan rather than signal an old identity.
            if current["state"] == "Z":
                return
            if not self.has_token(pid):
                self.record_token_loss(current, f"refused signal after owned token changed: {pid}")
                return
            pidfd_send_signal(fd, signum)
            self.signals.append({**current, "signal": signal.Signals(signum).name})
        except (FileNotFoundError, ProcessLookupError):
            pass
        except OSError as error:
            self.errors.append(f"could not signal validated owned PID {pid}: {error}")
        finally:
            if fd is not None:
                os.close(fd)

    def cleanup(self, child, graces, poll_interval):
        live = self.scan()
        for signum, grace in zip((signal.SIGINT, signal.SIGTERM, signal.SIGKILL), graces):
            child.poll()  # Reap the launcher, not just its workers.
            if not live and child.returncode is not None:
                break
            for identity in live:
                self.send(identity, signum)
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                child.poll()
                live = self.scan()
                if not live and child.returncode is not None:
                    break
                time.sleep(poll_interval)
        # A second empty scan avoids claiming success from a single transient reading.
        child.poll()
        live = self.scan()
        if not live and child.returncode is not None:
            time.sleep(poll_interval)
            live = self.scan()
        unresolved, reconciled = self.reconcile_token_losses()
        errors = sorted(set(self.errors + unresolved))
        return {"complete": not live and child.returncode is not None and not errors,
                "live_owned_processes": live, "processes_seen": list(self.seen.values()),
                "ownership_errors": errors, "signals_sent": self.signals,
                "ownership_observations": self.ownership_observations,
                "terminal_reconciliations": reconciled,
                "launcher_reaped": child.returncode is not None, "limitations": LIMITATIONS}


class AttemptInterrupted(KeyboardInterrupt):
    pass


@contextmanager
def interruption_handlers(ignore=False):
    previous = {}
    def handle(signum, _frame):
        raise AttemptInterrupted(f"harness received {signal.Signals(signum).name}")
    try:
        for signum in INTERRUPTION_SIGNALS:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, signal.SIG_IGN if ignore else handle)
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@contextmanager
def deferred_spawn_signals():
    """Defer interruption until Popen has returned a child handle; do not block child signals."""
    previous, pending = {}, []
    try:
        for signum in INTERRUPTION_SIGNALS:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, lambda received, _frame: pending.append(received))
        yield
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    if pending:
        raise AttemptInterrupted(f"harness received {signal.Signals(pending[0]).name} during spawn")


def bounded_attempt(name, command, timeout_s, output_dir, env, *, cwd=None,
                    cleanup_graces=(30, 20, 10), poll_interval=0.1):
    require_procfs_pidfds()
    started = time.monotonic()
    token = secrets.token_hex(32)
    child_env = dict(env, **{TOKEN_ENV: token})
    owned, child, error = OwnedProcesses(token), None, None
    result = {"name": name, "command": command, "timeout_s": timeout_s,
              "cleanup_graces_s": list(cleanup_graces), "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "cwd": str(cwd or ROOT / "src/RelicLLM"), "timed_out": False,
              "ownership_token": token, "critical_environment": {key: env.get(key) for key in CRITICAL_ENV},
              "generation_success_claimed": False, "serving_success_claimed": False}
    try:
        with (output_dir / f"{name}.stdout.log").open("x") as stdout, \
             (output_dir / f"{name}.stderr.log").open("x") as stderr:
            with deferred_spawn_signals():
                child = subprocess.Popen(command, cwd=result["cwd"], env=child_env, stdin=subprocess.DEVNULL,
                                         stdout=stdout, stderr=stderr, start_new_session=True)
                result.update(pid=child.pid, process_group=child.pid)
            deadline = started + timeout_s
            while child.poll() is None and time.monotonic() < deadline:
                owned.scan()
                time.sleep(poll_interval)
            result["timed_out"] = child.poll() is None
    except BaseException as caught:
        error = caught
        result["harness_error"] = {"type": type(caught).__name__, "message": str(caught)}
    finally:
        # Keep catchable interrupts deferred through the receipt, not just through cleanup.
        with interruption_handlers(ignore=True):
            try:
                result["cleanup"] = owned.cleanup(child, cleanup_graces, poll_interval) if child else {
                    "complete": True, "launcher_started": False}
            except BaseException as cleanup_error:
                result["cleanup"] = {"complete": False, "error": repr(cleanup_error)}
            result["returncode"] = None if child is None else child.poll()
            result["elapsed_s"] = time.monotonic() - started
            result["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
            new_json(output_dir / f"{name}.json", result)
            print(json.dumps(result), flush=True)
    require(result["cleanup"]["complete"], f"{name} cleanup incomplete/uncertain; abort before any next attempt")
    if error is not None:
        raise error
    return result


def require_free_port(port=18541):
    with socket.socket() as probe:
        probe.settimeout(1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as error:
            raise RuntimeError(f"localhost port {port} is occupied/unavailable; do not disturb it") from error


def build_commands(root=ROOT):
    generation = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
                  "--nproc-per-node=8", "-m", "relicllm.cli.generate_v41", "--checkpoint",
                  str(root / "checkpoint"), "--expert-device", "npu:0", "--expert-world", "8",
                  "--expert-pool-rows", "0", "--no-expert-batched", "--no-decode-graphs",
                  "--threads", "8", "--max-seq-len", "128", "--max-new-tokens", "4",
                  "--temperature", "0", "--prompt", "The capital of France is"]
    serving = [sys.executable, "-m", "relicllm", "serve", "--model", str(root / "checkpoint"),
               "--backend", "v41", "--device", "ascend", "--device-ids", "0,1,2,3,4,5,6,7",
               "--tensor-parallel-size", "8", "--max-model-len", "128", "--host", "127.0.0.1",
               "--port", "18541", "--backend-option", "decode_graphs=false",
               "--backend-option", "resident_engram=false", "--backend-option", "resident_experts=false"]
    return generation, serving


def runtime_stack(root=ROOT):
    require(Path(sys.prefix).resolve() == root / "venv", "use the task-owned virtual environment")
    import torch
    import torch_npu
    require(torch.__version__.split("+")[0] == "2.10.0"
            and importlib.metadata.version("torch-npu") == "2.10.0.post7", "wrong exact runtime stack")
    for module in (torch, torch_npu):
        require(Path(module.__file__).resolve().is_relative_to(root / "venv"), "package outside private venv")
    require(torch.npu.is_available() and torch.npu.device_count() == 8, "expected eight available NPUs")
    return {"torch": torch.__version__, "torch_npu": importlib.metadata.version("torch-npu"),
            "torch_path": torch.__file__, "torch_npu_path": torch_npu.__file__,
            "npu_names": [torch.npu.get_device_name(i) for i in range(8)], "python": sys.executable}


def run_attempts(output, env, meta, entries, root=ROOT):
    generation, serving = build_commands(root)
    results = []
    try:
        results.append(bounded_attempt("generation", generation, 900, output, env))
        require(results[-1]["cleanup"]["complete"], "generation cleanup incomplete")
        require(checkpoint_stats(root, entries) == meta["checkpoint_stats"], "checkpoint changed between attempts")
        require_free_port()  # Immediately before serving, not before the generation's 900-second budget.
        results.append(bounded_attempt("serving", serving, 90, output, env))
        require(results[-1]["cleanup"]["complete"], "serving cleanup incomplete")
    finally:
        new_json(output / "summary.json", {"attempts": results, "operator_review_required": True,
                 "receipt_files": [f"{name}.json" for name in ("generation", "serving")
                                   if (output / f"{name}.json").exists()],
                 "generation_success_claimed": False, "serving_success_claimed": False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verification-report", type=Path, required=True)
    parser.add_argument("--audit-report", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[a-z0-9-]{3,64}", args.run_id) is not None, "unsafe run identifier")
    require_procfs_pidfds()
    meta, entries = validate_preconditions(args.verification_report, args.audit_report)
    meta["runtime_stack"] = runtime_stack()
    env = launch_environment(os.environ)
    output = ROOT / "logs/model-attempts" / args.run_id
    require(output.resolve() == output, "aliased output directory")
    output.mkdir(parents=True, exist_ok=False)
    meta.update(attempt_script_sha256=sha256(Path(__file__).read_bytes()), resident_experts=False,
                resident_engram=False, decode_graphs=False, capability_guard_bypassed=False,
                critical_environment={key: env.get(key) for key in CRITICAL_ENV}, limitations=LIMITATIONS)
    new_json(output / "preconditions.json", meta)
    with interruption_handlers():
        run_attempts(output, env, meta, entries)


if __name__ == "__main__":
    main()
