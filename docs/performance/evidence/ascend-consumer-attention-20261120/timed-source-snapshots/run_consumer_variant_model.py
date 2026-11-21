"""Bounded full-model K128/K256 comparison with the same host-table Engram/HC stack.

Uses the established private model driver, preserving its exact prefill and
reference collection. Native K partition changes are explicitly numerical
experiments. The native-host-dispatch configuration is a separate control.
"""
from __future__ import annotations

import datetime as dt
import inspect
import json
from pathlib import Path
import sys

import run_device_engram_trial as base

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938")


def main():
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in ("k128", "k256", "native", "native-reuse"):
        raise SystemExit("choose k128, k256, native or native-reuse, optionally 16/64 timed forwards")
    mode = sys.argv[1]
    base.CONSUMER_MODE = mode
    base.LEASE_END = dt.datetime.fromisoformat("2026-11-20T13:37:20+00:00")
    selected = None
    if mode == "k128":
        selected = {"library": str(ROOT / "cache/online-tile-cube-k128-v2/lib/librelic_online_tile_cube.so"),
                    "sha256": "0cd9111e373859c5b8d4ec09089a952467f26b6c88a3fb387842b11c0725b287"}
    elif mode == "k256":
        selected = json.loads((ROOT / "logs/consumer-variants-build-20261120-v2.json").read_bytes())["variants"]["k256"]
    base.SELECTED_CONSUMER = selected
    sys.argv[1] = "device-hc"
    source = inspect.getsource(base.main)
    name_marker = '    name = "original-host-table-" + mode + "-engram-native-baseline"\n'
    if source.count(name_marker) != 1:
        raise RuntimeError("private naming boundary changed")
    source = source.replace(name_marker, '    name = "original-consumer-" + CONSUMER_MODE + "-host-table-HC"\n', 1)
    marker = '    command = [sys.executable, "-m", "torch.distributed.run"'
    if source.count(marker) != 1:
        raise RuntimeError("private environment boundary changed")
    add = (
        '    env["RELIC_ATTENTION_FP8_REUSE"] = "1" if CONSUMER_MODE == "native-reuse" else "0"\n'
        '    if SELECTED_CONSUMER is not None:\n'
        '        env.update(RELIC_TILE_CONSUMER_LIBRARY=SELECTED_CONSUMER["library"],\n'
        '                   RELIC_TILE_CONSUMER_SHA256=SELECTED_CONSUMER["sha256"],\n'
        '                   RELIC_DEVICE_TILE_MOE="1", RELIC_EXPERT_BODY_GRAPHS="0", RELIC_M1_DIRECT_ROUTING="0",\n'
        '                   RELIC_EXPERT_PROJECTION="loader", DEEPSEEK_V41_ASCEND_TILE_MAJOR="1")\n'
        + marker)
    source = source.replace(marker, add, 1)
    source_marker = '    snapshots = output / "source-snapshots"\n'
    add_sources = (
        '    extra_sources = ["run_consumer_variant_model.py",\n'
        '        "RelicLLM/relicllm/models/deepseek_v4_1/ascend_attention_reuse.py",\n'
        '        "relic-core/relic_core/csrc/ascend/online_tile_cube/online_tile_cube.cpp",\n'
        '        "relic-core/relic_core/csrc/ascend/online_tile_cube/CMakeLists.txt",\n'
        '        "relic-core/relic_core/csrc/ascend/online_tile_cube/kernel_entry.cpp.in",\n'
        '        "relic-core/relic_core/kernels/ascendc_online_tile_cube.py",\n'
        '        "relic-core/relic_core/kernels/ascendc_tile_moe.py"]\n'
        '    for relative in extra_sources:\n'
        '        path = ROOT / "source" / relative\n'
        '        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()\n'
        + source_marker)
    source = source.replace(source_marker, add_sources, 1)
    snapshot_marker = '    for relative in relative_sources:\n'
    source = source.replace(snapshot_marker, '    relative_sources += extra_sources\n' + snapshot_marker, 1)
    precondition_marker = '"mode": mode, "command": command,'
    source = source.replace(precondition_marker,
        '"mode": mode, "consumer_mode": CONSUMER_MODE, "consumer": SELECTED_CONSUMER, "command": command,', 1)
    exec(compile(source, str(Path(__file__).resolve()), "exec"), base.__dict__)
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
