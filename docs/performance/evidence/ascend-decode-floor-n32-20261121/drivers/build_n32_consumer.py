"""Build one bounded N32/K128 consumer; existing N16 libraries are immutable controls."""
from __future__ import annotations

import inspect
from pathlib import Path

import build_prepared_decode_consumers as base


def main():
    base.VARIANTS = {'k128-prepared-n32':(4,0)}
    source = inspect.getsource(base.main)
    source = source.replace("'-DRELIC_TILE_PRELOAD_A=0',", "'-DRELIC_TILE_OUTPUT_ROWS=32','-DRELIC_TILE_PRELOAD_A=0',", 1)
    source = source.replace("'prepared_tables':True,", "'prepared_tables':True,'output_rows':32,", 1)
    source = source.replace('prepared-decode-build.json', 'prepared-n32-build.json', 1)
    source = source.replace('prepared-decode-build-20261121-v1.json', 'prepared-n32-build-20261121-v2.json', 1)
    source = source.replace("f'online-tile-{name}-20261121-v1'", "f'online-tile-{name}-20261121-v2'", 1)
    if '-DRELIC_TILE_OUTPUT_ROWS=32' not in source:
        raise RuntimeError('the bounded N32 build geometry was not selected')
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), base.__dict__)
    return base.main()


if __name__ == '__main__':
    main()
