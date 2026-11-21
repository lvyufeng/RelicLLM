"""Run a warmup-separated B=1 original/compact model trial within the live lease."""
from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import run_device_engram_trial as base

ROOT = Path('<private-compact-root>')


def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ('original-control', 'cube-packed') or sys.argv[2] not in ('16', '64'):
        raise SystemExit('choose original-control/cube-packed and exactly 16/64 timed forwards')
    variant, steps = sys.argv[1:]
    receipt = os.environ.get('RELIC_COMPACT_BUILD_RECEIPT', 'compact-build.json')
    if receipt not in ('compact-build.json', 'compact-build-v2.json'):
        raise ValueError('choose a known immutable compact build receipt')
    rows = json.loads((ROOT / 'logs' / receipt).read_bytes())['variants']
    row = rows[variant]
    library = Path(row['library'])
    if hashlib.sha256(library.read_bytes()).hexdigest() != row['sha256']:
        raise RuntimeError('the immutable native candidate changed')
    base.ROOT = ROOT
    base.LEASE_END = dt.datetime.fromisoformat('2026-11-21T16:18:38+00:00')
    base.LIMIT = 1200
    base.COMPACT_VARIANT = variant + ('-v2' if receipt == 'compact-build-v2.json' else '')
    base.COMPACT_LIBRARY, base.COMPACT_SHA = str(library), row['sha256']
    base.COMPACT_FORMAT = 'relic.fp4.cube-n32-k128.v1' if variant == 'cube-packed' else 'original-n16'
    source = inspect.getsource(base.main)
    source = source.replace('    name = "original-host-table-" + mode + "-engram-native-baseline"\n',
                            '    name = "compact-weight-format-" + COMPACT_VARIANT + "-host-table-HC"\n', 1)
    marker = '    command = [sys.executable, "-m", "torch.distributed.run"'
    if source.count(marker) != 1:
        raise RuntimeError('the private model environment boundary moved')
    source = source.replace(marker,
        '    env.update(RELIC_TILE_CONSUMER_LIBRARY=COMPACT_LIBRARY, RELIC_TILE_CONSUMER_SHA256=COMPACT_SHA,\n'
        '               RELIC_COMPACT_BANK_FORMAT=COMPACT_FORMAT, RELIC_EXPERT_TP="2",\n'
        '               RELIC_EXPERT_BODY_GRAPHS="0", RELIC_M1_DIRECT_ROUTING="0",\n'
        '               RELIC_DEVICE_MOE="0", RELIC_DEVICE_TILE_MOE="0",\n'
        '               RELIC_EXPERT_PROJECTION="loader", DEEPSEEK_V41_ASCEND_TILE_MAJOR="1",\n'
        '               RELIC_TP_MATCHED_TOKEN_INPUTS="0", RELIC_ATTENTION_FP8_REUSE="0")\n' + marker, 1)
    source = source.replace('str(ROOT / "source/profile_device_engram.py")', 'str(ROOT / "source/profile_compact_weight_model.py")', 1)
    source = source.replace('    snapshots = output / "source-snapshots"\n',
        '    extra_sources = ["profile_compact_weight_model.py", "profile_expert_tp.py", "run_compact_weight_model.py",\n'
        '        "RelicLLM/relicllm/models/deepseek_v4_1/ascend_expert_tp.py",\n'
        '        "relic-core/relic_core/kernels/ascendc_online_tile_cube.py",\n'
        '        "relic-core/relic_core/kernels/cube_packed_format.py",\n'
        '        "relic-core/relic_core/csrc/ascend/online_tile_cube/online_tile_cube.cpp",\n'
        '        "relic-core/relic_core/csrc/ascend/online_tile_cube/CMakeLists.txt",\n'
        '        "relic-core/relic_core/csrc/ascend/online_tile_cube/kernel_entry.cpp.in"]\n'
        '    for relative in extra_sources:\n'
        '        path = ROOT / "source" / relative\n'
        '        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()\n'
        '    snapshots = output / "source-snapshots"\n', 1)
    source = source.replace('    for relative in relative_sources:\n',
                            '    relative_sources += extra_sources\n    for relative in relative_sources:\n', 1)
    source = source.replace('"original_checkpoint_and_quantization_unchanged": True,',
        '"original_checkpoint_files_unchanged": True, "represented_weight_values_unchanged": True,\n'
        '                    "resident_encoding_changed": COMPACT_FORMAT != "original-n16",\n'
        '                    "bank_format": COMPACT_FORMAT, "native_library_sha256": COMPACT_SHA,', 1)
    source = source.replace('"latest_human_permission": "Engram tables host is sufficient; begin optimization",',
        '"latest_human_permission": "offline lossless same-size FP4/FP8 format conversion and leased dormant 910B3 validation",', 1)
    sys.argv = [sys.argv[0], 'device-hc', steps]
    attempt = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    generated = ROOT / 'source' / f'run_compact_weight_{variant}_{steps}_{attempt}.generated.py'
    if generated.exists():
        raise RuntimeError('a trial-specific source snapshot already exists')
    with generated.open('x') as handle: handle.write(source)
    exec(compile(source, str(generated), 'exec'), base.__dict__)
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
