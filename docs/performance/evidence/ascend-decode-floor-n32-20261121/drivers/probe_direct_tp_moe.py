"""Eight-rank actual original-expert TP2 replay and complete routed-body timing."""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import probe_expert_tp2_ep4 as base

ROOT = Path("/data1/relic-v41-perf-20261118-0938")


def main():
    mode = os.environ["RELIC_DIRECT_CONSUMER_MODE"]
    if mode not in ("bits", "aiv"):
        raise ValueError("choose a native bit decoder or a separately tested no-ring AIV reduction")
    source = inspect.getsource(base.main)
    source = source.replace('    operator = AscendCTileCubeConsumer(LIBRARY, sha256=SHA)\n',
        '    reference_operator = AscendCTileCubeConsumer(LIBRARY, sha256=SHA)\n'
        '    if os.environ["RELIC_DIRECT_CONSUMER_MODE"] == "bits":\n'
        '        build = json.loads((ROOT / "logs/direct-bit-consumer-build-20261121-v1.json").read_bytes())["variants"]["k128-direct-bits"]\n'
        '        operator = AscendCTileCubeConsumer(build["library"], sha256=build["sha256"])\n'
        '    else:\n'
        '        from relic_core.kernels.direct_aiv_gemv import DirectAIVConsumer\n'
        '        operator = DirectAIVConsumer()\n', 1)
    source = source.replace('consumer=operator, certified_bf16_operands=True, backend="triton")',
                            'consumer=reference_operator, certified_bf16_operands=True, backend="triton")', 1)
    marker = '        row = {"rank": rank, "world": world,'
    timed = (
        '        import time\n'
        '        times = []\n'
        '        for _ in range(4):\n'
        '            shard.forward(x, weights, indices)\n'
        '        dist.barrier(); torch.npu.synchronize()\n'
        '        for _ in range(32):\n'
        '            start = time.perf_counter()\n'
        '            shard.forward(x, weights, indices)\n'
        '            torch.npu.synchronize()\n'
        '            times.append(1000 * (time.perf_counter() - start))\n'
        + marker)
    if source.count(marker) != 1:
        raise RuntimeError("actual expert test receipt boundary moved")
    source = source.replace(marker, timed, 1)
    save = '        with (OUTPUT / f"expert-TP2-EP4-rank-{rank}.json").open("x") as stream:\n'
    source = source.replace(save,
        '        row["consumer_mode"] = os.environ["RELIC_DIRECT_CONSUMER_MODE"]\n'
        '        row["consumer"] = operator.describe()\n'
        '        row["synchronized_routed_TP_body_ms"] = times\n'
        '        row["timing_scope"] = "routed projection/epilogue/subgroup reduction, not shared FFN or model TPS"\n'
        + save, 1)
    exec(compile(source, str(Path(__file__).resolve()), "exec"), base.__dict__)
    return base.main()


if __name__ == "__main__":
    main()
