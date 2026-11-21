"""Build immutable compact-layout and same-source original-layout N32 consumers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('<private-compact-root>')
SOURCE = ROOT / 'source/relic-core/relic_core/csrc/ascend/online_tile_cube'
OUTPUT = Path(os.environ['RELIC_PERF_OUTPUT'])


def main():
    if SOURCE.resolve() != SOURCE or subprocess.check_output(['hostname'], text=True).strip() != '<leased-910B3-host>':
        raise RuntimeError('private compact build root or leased host identity differs')
    payloads = {name: (SOURCE / name).read_bytes() for name in ('online_tile_cube.cpp', 'CMakeLists.txt', 'kernel_entry.cpp.in')}
    source_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    rows = {}
    for name, packed in (('original-control', 0), ('cube-packed', 1)):
        root = ROOT / 'cache' / name
        if root.exists():
            raise RuntimeError('never overwrite a prior native build')
        snapshot = root / 'source'
        snapshot.mkdir(parents=True, exist_ok=False)
        for filename, data in payloads.items():
            with (snapshot / filename).open('xb') as handle:
                handle.write(data)
        command = ['cmake', '-S', str(snapshot), '-B', str(root / 'build'),
                   '-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0', '-DSOC_VERSION=Ascend910B3',
                   '-DRUN_MODE=npu', '-DCMAKE_BUILD_TYPE=Release', '-DRELIC_TILE_K_BLOCKS=4',
                   '-DRELIC_TILE_OUTPUT_ROWS=32', '-DRELIC_TILE_PRELOAD_A=0', '-DRELIC_TILE_DIRECT_BITS=0',
                   '-DRELIC_TILE_PREPARED_TABLES=1', f'-DRELIC_TILE_CUBE_PACKED={packed}']
        print(json.dumps({'variant': name, 'configure': command}), flush=True)
        subprocess.run(command, check=True)
        subprocess.run(['cmake', '--build', str(root / 'build'), '--parallel', '2'], check=True)
        library = root / 'build/lib/librelic_online_tile_cube.so'
        rows[name] = {'library': str(library), 'sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
                      'cube_packed': bool(packed), 'group_k': 4, 'output_rows': 32,
                      'prepared_tables': True, 'source_hashes': source_hashes,
                      'source_snapshot': str(snapshot), 'configure_command': command}
    report = {'variants': rows, 'original_checkpoint_writes': 0, 'old_native_libraries_overwritten': False,
              'no_local_macOS_build_or_benchmark': True}
    for destination in (OUTPUT / 'compact-build.json', ROOT / 'logs/compact-build.json'):
        with destination.open('x') as handle:
            json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
