"""Build direct-bit decoders in fresh private roots; accepted consumers stay intact."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path("/data1/relic-v41-perf-20261118-0938")
SOURCE = ROOT / "source/relic-core/relic_core/csrc/ascend/online_tile_cube"
OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])
VARIANTS = {"k128-direct-bits": 4, "k256-direct-bits": 8}


def main():
    roots = {name: ROOT / "cache" / f"online-tile-consumer-{name}-20261121-v1" for name in VARIANTS}
    if SOURCE.resolve() != SOURCE or any(path.exists() for path in roots.values()):
        raise RuntimeError("native build sources or proposed fresh directories are not unambiguous")
    originals = {name: hashlib.sha256((ROOT / "cache" / name / "lib/librelic_online_tile_cube.so").read_bytes()).hexdigest()
                 for name in ("online-tile-cube-k128-v2", "online-tile-consumer-k256-20261120-v2")}
    sources = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (SOURCE / "online_tile_cube.cpp", SOURCE / "CMakeLists.txt", SOURCE / "kernel_entry.cpp.in")}
    rows = {}
    for name, group in VARIANTS.items():
        command = ["cmake", "-S", str(SOURCE), "-B", str(roots[name]),
                   "-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0", "-DSOC_VERSION=Ascend910B3",
                   "-DRUN_MODE=npu", "-DCMAKE_BUILD_TYPE=Release", f"-DRELIC_TILE_K_BLOCKS={group}",
                   "-DRELIC_TILE_PRELOAD_A=0", "-DRELIC_TILE_DIRECT_BITS=1"]
        print(json.dumps({"variant": name, "configure": command}), flush=True)
        subprocess.run(command, check=True)
        subprocess.run(["cmake", "--build", str(roots[name]), "--parallel", "2"], check=True)
        library = roots[name] / "lib/librelic_online_tile_cube.so"
        rows[name] = {"library": str(library), "sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                      "group_k": group, "direct_bits": True, "source_hashes": sources,
                      "preload_activation": False, "no_original_binary_overwrite": True}
    for name, digest in originals.items():
        if hashlib.sha256((ROOT / "cache" / name / "lib/librelic_online_tile_cube.so").read_bytes()).hexdigest() != digest:
            raise RuntimeError("an accepted native binary changed")
    path = OUTPUT / "direct-bit-consumer-build.json"
    with path.open("x") as stream:
        json.dump(rows, stream, indent=2)
    with (ROOT / "logs/direct-bit-consumer-build-20261121-v1.json").open("x") as stream:
        json.dump({"report": str(path), "variants": rows, "original_binary_hashes": originals}, stream, indent=2)
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
