"""Build three new private consumer variants; never rebuild the accepted binary."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/data1/relic-v41-perf-20261118-0938")
SOURCE = ROOT / "source/relic-core/relic_core/csrc/ascend/online_tile_cube"
VARIANTS = {"k256": (8, 0), "k128-preload-a": (4, 1), "k256-preload-a": (8, 1)}
OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])


def main():
    if SOURCE.resolve() != SOURCE:
        raise RuntimeError("the task-private native source root changed")
    directories = {name: ROOT / "cache" / ("online-tile-consumer-" + name + "-20261120-v2") for name in VARIANTS}
    if any(path.exists() for path in directories.values()):
        raise RuntimeError("a proposed variant build directory already exists; reconcile, do not overwrite")
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (SOURCE / "online_tile_cube.cpp", SOURCE / "CMakeLists.txt", SOURCE / "kernel_entry.cpp.in")}
    reports = {}
    for name, (group, preload) in VARIANTS.items():
        build = directories[name]
        command = ["cmake", "-S", str(SOURCE), "-B", str(build),
                   "-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0", "-DSOC_VERSION=Ascend910B3",
                   "-DRUN_MODE=npu", "-DCMAKE_BUILD_TYPE=Release", f"-DRELIC_TILE_K_BLOCKS={group}",
                   f"-DRELIC_TILE_PRELOAD_A={preload}"]
        print(json.dumps({"variant": name, "configure": command}), flush=True)
        subprocess.run(command, check=True)
        subprocess.run(["cmake", "--build", str(build), "--parallel", "2"], check=True)
        library = build / "lib/librelic_online_tile_cube.so"
        # Runtime symbol resolution and both geometry queries are checked by
        # the device probe after torch_npu has loaded its registered runtime.
        reports[name] = {"library": str(library), "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                         "original_K32_blocks": group, "activation_L1_preload_requested": bool(preload),
                         "source_hashes": sources, "accepted_k128_library_not_modified": True,
                         "host_device_queries_deferred_to_NPU_probe": True}
        print(json.dumps({"built": name, **reports[name]}), flush=True)
    with (OUTPUT / "consumer-variants-build.json").open("x") as stream:
        json.dump(reports, stream, indent=2)
    pointer = ROOT / "logs/consumer-variants-build-20261120-v2.json"
    with pointer.open("x") as stream:
        json.dump({"report": str(OUTPUT / "consumer-variants-build.json"), "variants": reports}, stream, indent=2)


if __name__ == "__main__":
    main()
