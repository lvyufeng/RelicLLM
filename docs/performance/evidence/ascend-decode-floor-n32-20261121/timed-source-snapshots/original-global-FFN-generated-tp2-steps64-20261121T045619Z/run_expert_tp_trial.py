"""Bounded actual 40-layer expert TP2/EP4 versus same-codec EP8 model trial."""
from __future__ import annotations

import datetime as dt
import inspect
from pathlib import Path
import sys

import run_device_engram_trial as base

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938")


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in ("tp2", "ep8", "tp2-forced", "ep8-forced", "native-forced"):
        raise SystemExit("choose tp2/ep8 generation or tp2-forced/ep8-forced/native-forced identical-input latency")
    topology = sys.argv[1]
    base.TOPOLOGY = topology
    base.LEASE_END = dt.datetime.fromisoformat("2026-11-20T15:39:50+00:00")
    base.LIMIT = 1200
    sys.argv[1] = "device-hc"
    source = inspect.getsource(base.main)
    name_marker = '    name = "original-host-table-" + mode + "-engram-native-baseline"\n'
    if source.count(name_marker) != 1:
        raise RuntimeError("the task-private model naming boundary moved")
    source = source.replace(name_marker, '    name = "original-expert-" + TOPOLOGY + "-host-table-HC"\n', 1)
    env_marker = '    command = [sys.executable, "-m", "torch.distributed.run"'
    added = (
        '    env.update(RELIC_TILE_CONSUMER_LIBRARY=str(ROOT / "cache/online-tile-cube-k128-v2/lib/librelic_online_tile_cube.so"),\n'
        '               RELIC_TILE_CONSUMER_SHA256="0cd9111e373859c5b8d4ec09089a952467f26b6c88a3fb387842b11c0725b287",\n'
        '               RELIC_EXPERT_TP="2" if TOPOLOGY.startswith("tp2") else "1",\n'
        '               RELIC_EXPERT_BODY_GRAPHS="1" if TOPOLOGY == "native-forced" else "0",\n'
        '               RELIC_M1_DIRECT_ROUTING="1" if TOPOLOGY == "native-forced" else "0",\n'
        '               RELIC_DEVICE_MOE="0", RELIC_DEVICE_TILE_MOE="1" if TOPOLOGY.startswith("ep8") else "0",\n'
        '               RELIC_EXPERT_PROJECTION="cube" if TOPOLOGY == "native-forced" else "loader",\n'
        '               DEEPSEEK_V41_ASCEND_TILE_MAJOR="0" if TOPOLOGY == "native-forced" else "1",\n'
        '               RELIC_TP_MATCHED_TOKEN_INPUTS="1" if TOPOLOGY.endswith("forced") else "0",\n'
        '               RELIC_ATTENTION_FP8_REUSE="0")\n' + env_marker)
    if source.count(env_marker) != 1:
        raise RuntimeError("the task-private model environment boundary moved")
    source = source.replace(env_marker, added, 1)
    source = source.replace('str(ROOT / "source/profile_device_engram.py")', 'str(ROOT / "source/profile_expert_tp.py")', 1)
    extra = (
        '    extra_sources = ["run_expert_tp_trial.py", "profile_expert_tp.py",\n'
        '        "RelicLLM/relicllm/models/deepseek_v4_1/ascend_expert_tp.py",\n'
        '        "relic-core/relic_core/kernels/expert_tp_moe.py",\n'
        '        "relic-core/relic_core/kernels/ascendc_online_tile_cube.py",\n'
        '        "relic-core/relic_core/kernels/ascendc_tile_moe.py"]\n'
        '    for relative in extra_sources:\n'
        '        path = ROOT / "source" / relative\n'
        '        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()\n'
        '    snapshots = output / "source-snapshots"\n')
    source = source.replace('    snapshots = output / "source-snapshots"\n', extra, 1)
    source = source.replace('    for relative in relative_sources:\n', '    relative_sources += extra_sources\n    for relative in relative_sources:\n', 1)
    source = source.replace('"mode": mode, "command": command,',
                            '"mode": mode, "expert_topology": TOPOLOGY, "command": command,', 1)
    source = source.replace('"latest_human_permission": "Engram tables host is sufficient; begin optimization",',
        '"latest_human_permission": "directly implement and test expert TP2/EP4; original bytes and activation grids retained",'
        ' "explicit_TP_partial_accumulation_is_authorized_experiment": True,'
        ' "same_token_input_latency_not_generation_TPS": TOPOLOGY.endswith("forced"),', 1)
    exec(compile(source, str(Path(__file__).resolve()), "exec"), base.__dict__)
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
