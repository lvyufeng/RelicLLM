"""Build private N32 controls and unsigned-byte/double-UB compact consumers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path('<private-compact-root>')
SOURCE = ROOT / 'source-v2/relic_core/csrc/ascend/online_tile_cube'
OUTPUT = Path(os.environ['RELIC_PERF_OUTPUT'])


def main():
    if SOURCE.resolve() != SOURCE or subprocess.check_output(['hostname'], text=True).strip() != '<leased-910B3-host>':
        raise RuntimeError('private v2 source or leased host differs')
    payloads = {name: (SOURCE / name).read_bytes() for name in ('online_tile_cube.cpp', 'CMakeLists.txt', 'kernel_entry.cpp.in')}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    rows = {}
    for name, packed, unsigned in (('original-control', 0, 0), ('cube-packed', 1, 1)):
        root = ROOT / 'cache-v2' / name
        if root.exists(): raise RuntimeError('never replace an existing native build')
        snapshot = root / 'source'
        snapshot.mkdir(parents=True, exist_ok=False)
        for filename, data in payloads.items():
            with (snapshot / filename).open('xb') as handle: handle.write(data)
        command = ['cmake', '-S', str(snapshot), '-B', str(root / 'build'),
                   '-DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0', '-DSOC_VERSION=Ascend910B3',
                   '-DRUN_MODE=npu', '-DCMAKE_BUILD_TYPE=Release', '-DRELIC_TILE_K_BLOCKS=4',
                   '-DRELIC_TILE_OUTPUT_ROWS=32', '-DRELIC_TILE_PRELOAD_A=0', '-DRELIC_TILE_DIRECT_BITS=0',
                   '-DRELIC_TILE_PREPARED_TABLES=1', f'-DRELIC_TILE_CUBE_PACKED={packed}',
                   f'-DRELIC_TILE_UNSIGNED_BYTE_CAST={unsigned}']
        subprocess.run(command, check=True)
        subprocess.run(['cmake', '--build', str(root / 'build'), '--parallel', '2'], check=True)
        library = root / 'build/lib/librelic_online_tile_cube.so'
        rows[name] = {'library': str(library), 'sha256': hashlib.sha256(library.read_bytes()).hexdigest(),
                      'cube_packed': bool(packed), 'unsigned_byte_cast': bool(unsigned), 'group_k': 4,
                      'output_rows': 32, 'prepared_tables': True, 'source_hashes': hashes,
                      'source_snapshot': str(snapshot), 'configure_command': command}
    report = {'variants': rows, 'previous_binaries_not_overwritten': True,
              'double_BF16_UB_reuses_unused_NZ_index_region': True, 'original_checkpoint_writes': 0}
    for destination in (OUTPUT / 'compact-build-v2.json', ROOT / 'logs/compact-build-v2.json'):
        with destination.open('x') as handle: json.dump(report, handle, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
