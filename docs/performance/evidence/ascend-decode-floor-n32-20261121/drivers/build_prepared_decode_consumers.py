"""Hoist immutable AIV lookup/index tables into fresh private native builds."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('<workspace>/relic-v41-perf-20261118-0938')
SOURCE = ROOT / 'source/relic-core/relic_core/csrc/ascend/online_tile_cube'
OUTPUT = Path(os.environ['RELIC_PERF_OUTPUT'])
VARIANTS = {'k128-prepared-float':(4,0), 'k128-prepared-bits':(4,1)}


def main():
    roots = {name:ROOT/'cache'/f'online-tile-{name}-20261121-v1' for name in VARIANTS}
    if SOURCE.resolve() != SOURCE or any(path.exists() for path in roots.values()):
        raise RuntimeError('each prepared-table build must use a new unambiguous private root')
    original_paths = (
        ROOT/'cache/online-tile-cube-k128-v2/lib/librelic_online_tile_cube.so',
        ROOT/'cache/online-tile-consumer-k256-20261120-v2/lib/librelic_online_tile_cube.so',
        ROOT/'cache/online-tile-consumer-k128-direct-bits-20261121-v1/lib/librelic_online_tile_cube.so',
        ROOT/'cache/online-tile-consumer-k256-direct-bits-20261121-v1/lib/librelic_online_tile_cube.so')
    originals = {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in original_paths}
    payloads = {name:(SOURCE/name).read_bytes() for name in ('online_tile_cube.cpp','CMakeLists.txt','kernel_entry.cpp.in')}
    sources = {name:hashlib.sha256(payload).hexdigest() for name,payload in payloads.items()}
    rows = {}
    for name,(group,bits) in VARIANTS.items():
        root = roots[name]
        snapshot = root/'source'
        snapshot.mkdir(parents=True,exist_ok=False)
        for filename,payload in payloads.items():
            with (snapshot/filename).open('xb') as stream:stream.write(payload)
        build = root/'build'
        command = ['cmake','-S',str(snapshot),'-B',str(build),
            '-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0','-DSOC_VERSION=Ascend910B3',
            '-DRUN_MODE=npu','-DCMAKE_BUILD_TYPE=Release',f'-DRELIC_TILE_K_BLOCKS={group}',
            '-DRELIC_TILE_PRELOAD_A=0',f'-DRELIC_TILE_DIRECT_BITS={bits}','-DRELIC_TILE_PREPARED_TABLES=1']
        print(json.dumps({'variant':name,'configure':command}),flush=True)
        subprocess.run(command,check=True)
        subprocess.run(['cmake','--build',str(build),'--parallel','2'],check=True)
        library = build/'lib/librelic_online_tile_cube.so'
        rows[name] = {'library':str(library),'sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
            'group_k':group,'direct_bits':bool(bits),'prepared_tables':True,'source_hashes':sources,
            'source_snapshot':str(snapshot),'configure_command':command,'original_binaries_unchanged':True}
    for filename,digest in originals.items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != digest:
            raise RuntimeError('an existing native consumer was changed during the new builds')
    report = {'variants':rows,'original_binary_hashes':originals,'no_source_or_binary_overwrite':True,
              'runtime_device_queries_and_acceptance_are_deferred_until_torch_npu_is_loaded':True}
    with (OUTPUT/'prepared-decode-build.json').open('x') as stream:json.dump(report,stream,indent=2)
    with (ROOT/'logs/prepared-decode-build-20261121-v1.json').open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps(report),flush=True)


if __name__ == '__main__':
    main()
