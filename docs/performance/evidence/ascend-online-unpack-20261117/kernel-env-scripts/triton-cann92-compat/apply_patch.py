"""Apply the approved CANN 9.x compatibility patch to the private venv311 Triton source.

Scope: exactly one file,
``<private venv311>/lib/python3.11/site-packages/triton/backends/ascend/npu_utils.cpp``.

Guards, in order, all fail closed before any write:

1. The private root and venv resolve to the expected absolute paths (no aliases).
2. The installed file is a regular, non-symlink file whose sha256 is the recorded
   wheel original (2d6e5851...). A different file is refused, not "repaired".
3. The backup ``npu_utils.cpp.orig.wheel`` does not already exist (exclusive
   create), so a re-run cannot overwrite the pristine copy.
4. The patched bytes written are compared against the recorded patched digest; a
   mismatch restores the backup immediately.

It then re-compiles the installed file standalone (g++ only, no torch, no device)
and records the outcome. CANN, the driver, shared headers, torch_npu and every
kernel file are untouched; no device operation is performed.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("<workspace>/relic-online-unpack-20261117-0534")
TARGET = ROOT / "venv311/lib/python3.11/site-packages/triton/backends/ascend/npu_utils.cpp"
BACKUP_NAME = "npu_utils.cpp.orig.wheel"
COMPAT = ROOT / "source/kernel-env-scripts/triton-cann92-compat"
PATCHED_SOURCE = COMPAT / "npu_utils_patched.cpp"
PY_INCLUDE = ROOT / "python311-base/python/include/python3.11"
CANN_INC = Path("/usr/local/Ascend/cann-9.2.0/aarch64-linux/pkg_inc")
ORIGINAL_SHA256 = "2d6e58513486c70e47a8a4149418f750f59568ffa9feb2c45ba3479016554f53"
PATCHED_SHA256 = "92cbdcaa5d2ba02f8dd7b05363428f8ab0cd94d943264dd06cd1457ed4ee7d50"


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    result = {"started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "target": str(TARGET),
              "expected_original_sha256": ORIGINAL_SHA256, "expected_patched_sha256": PATCHED_SHA256,
              "enum_substitution": None, "shared_environment_modified": False, "device_work": False}
    code = 1
    try:
        require(ROOT.is_dir() and not ROOT.is_symlink(), "private root missing or aliased")
        require(TARGET.resolve() == TARGET, f"target is an alias: {TARGET}")
        require(TARGET.is_file() and not TARGET.is_symlink(), "target is not a regular file")
        require(PATCHED_SOURCE.is_file(), "approved patched source is missing from the compat directory")

        original = TARGET.read_bytes()
        installed_sha = hashlib.sha256(original).hexdigest()
        result["installed_sha256_before"] = installed_sha
        require(installed_sha == ORIGINAL_SHA256,
                f"installed file is not the recorded wheel original: {installed_sha}")

        approved = PATCHED_SOURCE.read_bytes()
        require(hashlib.sha256(approved).hexdigest() == PATCHED_SHA256,
                "approved patched source does not match its recorded digest")

        backup = TARGET.with_name(BACKUP_NAME)
        with backup.open("xb") as handle:            # exclusive: never overwrite a pristine copy
            handle.write(original)
        require(sha256(backup) == ORIGINAL_SHA256, "backup does not reproduce the original bytes")
        result["backup"] = str(backup)
        result["backup_sha256"] = sha256(backup)

        staged = TARGET.with_name("npu_utils.cpp.pending")
        with staged.open("xb") as handle:
            handle.write(approved)
        require(sha256(staged) == PATCHED_SHA256, "staged patch bytes differ from the approved digest")
        os.replace(staged, TARGET)

        installed_after = sha256(TARGET)
        result["installed_sha256_after"] = installed_after
        if installed_after != PATCHED_SHA256:
            os.replace(backup, TARGET)
            raise RuntimeError("patched file failed verification; original restored")

        # Rebuild proof against the INSTALLED path, standalone (no torch, no device).
        include = ["-I", str(PY_INCLUDE), "-I", str(CANN_INC)]
        object_path = ROOT / "tmp/npu_utils_installed.o"
        compile_cmd = ["g++", "-std=c++17", "-fPIC", "-c", str(TARGET), "-o", str(object_path), *include]
        compiled = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=600)
        result["recompile"] = {"command": " ".join(compile_cmd), "returncode": compiled.returncode,
                                "object_bytes": object_path.stat().st_size if object_path.exists() else 0,
                                "stderr_sha256": hashlib.sha256(compiled.stderr.encode()).hexdigest()}
        require(compiled.returncode == 0 and result["recompile"]["object_bytes"] > 0,
                "patched installed file does not compile standalone")

        # The three other mappings must still be present verbatim.
        text = TARGET.read_text()
        for entry in ('{"LOW_POWER_TIMEOUT", rtLimitType_t::RT_LIMIT_TYPE_LOW_POWER_TIMEOUT}',
                      '{"DVG_WARP_STACK_SIZE", rtLimitType_t::RT_LIMIT_TYPE_SIMT_DVG_WARP_STACK_SIZE}',
                      '{"STACK_SIZE", rtLimitType_t::RT_LIMIT_TYPE_STACK_SIZE}'):
            require(entry in text, f"unrelated mapping lost: {entry}")
        require("WARP_STACK_SIZE is unsupported" in text, "explicit unsupported branch missing")
        limit_map = text.split("static const std::unordered_map")[1].split("};")[0]
        require("SIMT_WARP_STACK_SIZE" not in limit_map, "broken enumerator still mapped")
        require("RT_LIMIT_TYPE_SIMT_STACK_SIZE" not in text,
                "must not substitute the SIMT_STACK_SIZE enumerator")
        result.update({
            "preserved_mappings": 3,
            "unsupported_name_raises": "NotImplementedError",
            "invalid_name_raises": "ValueError",
            "rollback_command": f"cp -p {backup} {TARGET} && sha256sum {TARGET}",
            "files_touched": [str(TARGET), str(backup)],
            "cann_headers_or_libraries_touched": False,
            "torch_or_torch_npu_touched": False,
            "kernel_files_touched": False,
        })
        code = 0
    except Exception as error:  # noqa: BLE001 - receipt carries the precise failure
        result["error"] = repr(error)[:600]
    finally:
        result.update(finished_at=dt.datetime.now(dt.timezone.utc).isoformat(), returncode=code)
        receipt = ROOT / "logs/triton-cann92-compat.json"
        if receipt.exists():
            stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            receipt = receipt.with_name(f"triton-cann92-compat-{stamp}.json")
        receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=1))
        print("RECEIPT", receipt)
    return code


if __name__ == "__main__":
    raise SystemExit(main())