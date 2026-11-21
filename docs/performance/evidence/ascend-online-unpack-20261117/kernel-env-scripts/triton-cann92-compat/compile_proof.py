"""Compile-prove the Triton-Ascend host-driver patch without torch or any device.

Compiles npu_utils.cpp standalone (it includes only Python.h and CANN's
runtime/rt.h) twice: the pristine wheel copy, which must fail with the reported
missing enumerator, and the patched copy, which must compile. No torch import, no
NPU context, no device memory, no kernel. Nothing installed is modified.
"""
from __future__ import annotations

import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY_INCLUDE = Path("/data1/relic-online-unpack-20261117-0534/python311-base/python/include/python3.11")
CANN_INC = Path("/usr/local/Ascend/cann-9.2.0/aarch64-linux/pkg_inc")
OUT = Path("/data1/relic-online-unpack-20261117-0534/tmp")
EXPECTED_FAILURE = "RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE"


def compile_one(source: Path, tag: str) -> dict:
    obj = OUT / f"npu_utils_{tag}.o"
    command = ["g++", "-std=c++17", "-fPIC", "-c", str(source), "-o", str(obj),
               f"-I{PY_INCLUDE}", f"-I{CANN_INC}", "-Wall"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    return {"tag": tag, "source": str(source), "command": " ".join(command),
            "returncode": completed.returncode,
            "stderr_sha256": __import__("hashlib").sha256(completed.stderr.encode()).hexdigest(),
            "stderr_full": completed.stderr, "object_bytes": obj.stat().st_size if obj.exists() else 0}


def main() -> int:
    result = {"python_include": str(PY_INCLUDE), "cann_include": str(CANN_INC),
              "compiler": subprocess.run(["g++", "--version"], capture_output=True, text=True).stdout.splitlines()[0]}
    original = compile_one(HERE / "npu_utils_original.cpp", "orig")
    patched = compile_one(HERE / "npu_utils_patched.cpp", "patched")
    result["original"] = original
    result["patched"] = patched
    result["original_fails_for_reported_reason"] = (
        original["returncode"] != 0
        and f"‘{EXPECTED_FAILURE}’ is not a member of ‘rtLimitType_t’" in original["stderr_full"])
    result["patched_compiles"] = patched["returncode"] == 0 and patched["object_bytes"] > 0
    result["passed"] = result["original_fails_for_reported_reason"] and result["patched_compiles"]
    (HERE / "compile-proof.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"original", "patched"}}, indent=1))
    print("original rc", original["returncode"], "| failure line:",
          next((line.strip() for line in original["stderr_full"].splitlines() if EXPECTED_FAILURE in line), ""))
    print("patched rc", patched["returncode"], "object bytes", patched["object_bytes"])
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
