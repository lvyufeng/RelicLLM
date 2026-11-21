"""Bounded original-checkpoint consumer comparison with optional matched inputs."""
from __future__ import annotations

import datetime as dt
import hashlib
import inspect
import json
from pathlib import Path
import sys

import run_device_engram_trial as base

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938")


def main():
    if len(sys.argv) not in (3, 4) or sys.argv[1] not in ("old", "bits", "aiv", "prepared", "prepared-bits", "n32") or sys.argv[2] not in ("generated", "matched"):
        raise SystemExit("choose old/bits/aiv/prepared/prepared-bits and generated/matched, optionally 16/64 forwards")
    mode, matched = sys.argv[1], sys.argv[2] == "matched"
    if mode == "aiv":
        probe_path = ROOT / "logs/original-direct-bit-consumer-probes-n8-20261121T033900Z/direct-bit-consumer-probe.json"
        if probe_path.exists():
            measured = json.loads(probe_path.read_bytes())["real_projections"]
            if any(row["paired_times_by_active_slots"]["3"]["AIV-direct"]["mean_ms"]
                   > 2 * row["paired_times_by_active_slots"]["3"]["k128-original"]["mean_ms"] for row in measured):
                print("SKIP full-model N8 AIV: actual original-weight projections are already much slower; keep the negative evidence", flush=True)
                return 0
    steps = sys.argv[3] if len(sys.argv) == 4 else "16"
    if matched and steps != "16":
        raise ValueError("the fixed-input control is the exact 16-input context24..40 window")
    library = ROOT / "cache/online-tile-cube-k128-v2/lib/librelic_online_tile_cube.so"
    sha = "0cd9111e373859c5b8d4ec09089a952467f26b6c88a3fb387842b11c0725b287"
    if mode == "bits":
        build = json.loads((ROOT / "logs/direct-bit-consumer-build-20261121-v1.json").read_bytes())["variants"]["k128-direct-bits"]
        library, sha = Path(build["library"]), build["sha256"]
    if mode.startswith("prepared"):
        name = "k128-prepared-bits" if mode == "prepared-bits" else "k128-prepared-float"
        build = json.loads((ROOT / "logs/prepared-decode-build-20261121-v1.json").read_bytes())["variants"][name]
        library, sha = Path(build["library"]), build["sha256"]
    if mode == "n32":
        build = json.loads((ROOT / "logs/prepared-n32-build-20261121-v2.json").read_bytes())["variants"]["k128-prepared-n32"]
        library, sha = Path(build["library"]), build["sha256"]
    if hashlib.sha256(library.read_bytes()).hexdigest() != sha:
        raise RuntimeError("the selected accepted or new bit decoder library changed")
    base.DIRECT_MODE, base.MATCHED_INPUTS = mode, matched
    base.CONSUMER_LIBRARY, base.CONSUMER_SHA = library, sha
    base.LEASE_END = dt.datetime.fromisoformat("2026-11-21T07:05:44+00:00")
    base.LIMIT = 1200
    sys.argv = [sys.argv[0], "device-hc", steps]
    source = inspect.getsource(base.main)
    marker = '    name = "original-host-table-" + mode + "-engram-native-baseline"\n'
    if source.count(marker) != 1:
        raise RuntimeError("the private bounded model trial naming boundary moved")
    source = source.replace(marker,
        '    name = "original-direct-byte-" + DIRECT_MODE + ("-matched" if MATCHED_INPUTS else "-generated")\n', 1)
    command = '    command = [sys.executable, "-m", "torch.distributed.run"'
    source = source.replace(command,
        '    env.update(RELIC_TILE_CONSUMER_LIBRARY=str(CONSUMER_LIBRARY), RELIC_TILE_CONSUMER_SHA256=CONSUMER_SHA,\n'
        '        RELIC_DIRECT_CONSUMER_MODE=DIRECT_MODE, RELIC_EXPERT_TP="2",\n'
        '        RELIC_EXPERT_BODY_GRAPHS="0", RELIC_M1_DIRECT_ROUTING="0",\n'
        '        RELIC_DEVICE_MOE="0", RELIC_DEVICE_TILE_MOE="0", RELIC_EXPERT_PROJECTION="loader",\n'
        '        DEEPSEEK_V41_ASCEND_TILE_MAJOR="1", RELIC_ATTENTION_FP8_REUSE="0",\n'
        '        RELIC_TP_MATCHED_TOKEN_INPUTS="1" if MATCHED_INPUTS else "0")\n' + command, 1)
    source = source.replace('str(ROOT / "source/profile_device_engram.py")', 'str(ROOT / "source/profile_direct_decoder_model.py")', 1)
    extra = (
        '    extra_sources = ["run_direct_decoder_model.py", "profile_direct_decoder_model.py", "direct_decoder_factory.py",\n'
        '        "profile_expert_tp.py", "RelicLLM/relicllm/models/deepseek_v4_1/ascend_expert_tp.py",\n'
        '        "relic-core/relic_core/kernels/expert_tp_moe.py", "relic-core/relic_core/kernels/ascendc_online_tile_cube.py",\n'
        '        "relic-core/relic_core/kernels/direct_aiv_gemv.py", "relic-core/relic_core/kernels/_ascend_direct_aiv_gemv_kernels.py"]\n'
        '    for relative in extra_sources:\n'
        '        path = ROOT / "source" / relative\n'
        '        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()\n'
        '    snapshots = output / "source-snapshots"\n')
    source = source.replace('    snapshots = output / "source-snapshots"\n', extra, 1)
    source = source.replace('    for relative in relative_sources:\n', '    relative_sources += extra_sources\n    for relative in relative_sources:\n', 1)
    source = source.replace('"mode": mode, "command": command,',
        '"mode": mode, "consumer_mode": DIRECT_MODE, "matched_input_latency_not_generation_TPS": MATCHED_INPUTS, "command": command,', 1)
    exec(compile(source, str(Path(__file__).resolve()), "exec"), base.__dict__)
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
