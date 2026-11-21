"""Run compact-format regression on the leased Linux server, never on macOS."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path('<private-compact-root>')


def main():
    import torch_npu
    import pytest
    project = sys.argv[1] if len(sys.argv) == 2 else 'core'
    if project == 'core':
        directory = ROOT / 'source/relic-core'
        files = ['tests/test_cube_packed_format.py', 'tests/test_compact_decode_tables.py',
                 'tests/test_prepared_decode_tables.py', 'tests/test_ascendc_online_tile_cube.py',
                 'tests/test_ascendc_tile_moe.py', 'tests/test_expert_tp_moe.py']
    elif project == 'runtime':
        directory = ROOT / 'source/RelicLLM'
        files = ['tests/test_ascend_expert_tp_compact.py', 'tests/test_v41_ascend_expert_tp.py']
    else:
        raise ValueError('choose core/runtime')
    os.chdir(directory)
    sys.path.insert(0, str(directory))
    sources = {path: hashlib.sha256((directory / path).read_bytes()).hexdigest() for path in files}
    code = pytest.main(['-p', 'no:cacheprovider', '-q', '-rs', *files])
    with (Path(os.environ['RELIC_PERF_OUTPUT']) / f'{project}-regression-source-ids.json').open('x') as handle:
        json.dump({'project': project, 'directory': str(directory), 'test_sources': sources, 'pytest_returncode': code}, handle, indent=2)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
