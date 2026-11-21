"""Bounded final unpack grid/bit/tail regression on the leased NPU."""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

ROOT = Path("/data1/relic-v41-infer-20261117-1424")


def main():
    path = ROOT / "source/bounded_task.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != "7101c6d6637fd0a1a91d6730ace772ba3c5dc91b7c3d3424aa2ef6b1f9d6442a":
        raise RuntimeError("cleanup helper changed")
    spec = importlib.util.spec_from_file_location("grid_infer_cleanup", path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    output = ROOT / "logs" / ("unpack-grid-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(exist_ok=False)
    env = dict(os.environ, RELIC_ONLINE_UNPACK_TEST_DEVICE="npu:0", RELIC_ONLINE_UNPACK_TEST_BACKEND="triton")
    tests = ["tests/test_online_unpack.py", "tests/test_ascend_unpack_grid.py",
             "tests/test_ascend_unpack_guard.py", "tests/test_ascend_unpack_stream.py",
             "tests/test_ascend_unpack_launch.py", "tests/test_ascend_unpack_math.py"]
    project = ROOT / "source/relic-core"
    hashes = {name: hashlib.sha256((project / name).read_bytes()).hexdigest()
              for name in tests + ["relic_core/kernels/unpack.py", "relic_core/kernels/ascend_unpack.py",
                                   "relic_core/kernels/_ascend_unpack_kernels.py"]}
    with (output / "tested-source-hashes.json").open("x") as handle:
        json.dump(hashes, handle, indent=2)
    command = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", *tests, "-q", "-rs",
               "--junitxml=" + str(output / "pytest.xml")]
    result = helper.bounded_attempt("unpack-grid", command, 900, output, env, cwd=project)
    with (output / "summary.json").open("x") as handle:
        json.dump(result, handle, indent=2)
    return 0 if result["returncode"] == 0 and result["cleanup"]["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
