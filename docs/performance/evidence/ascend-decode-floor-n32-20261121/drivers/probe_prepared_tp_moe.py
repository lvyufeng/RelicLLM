"""Eight-rank actual original-expert replay for the prepared-table consumer."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import probe_expert_tp2_ep4 as base


def main():
    mode = os.environ['RELIC_DIRECT_CONSUMER_MODE']
    if mode not in ('prepared','prepared-bits','n32'):
        raise ValueError('select only an independently gated prepared-table consumer')
    source = inspect.getsource(base.main)
    marker = '    operator = AscendCTileCubeConsumer(LIBRARY, sha256=SHA)\n'
    if source.count(marker) != 1:
        raise RuntimeError('the actual-expert consumer construction boundary moved')
    source = source.replace(marker,
        '    reference_operator = AscendCTileCubeConsumer(LIBRARY, sha256=SHA)\n'
        '    import json\n'
        '    name = "k128-prepared-n32" if os.environ["RELIC_DIRECT_CONSUMER_MODE"] == "n32" else ("k128-prepared-bits" if os.environ["RELIC_DIRECT_CONSUMER_MODE"] == "prepared-bits" else "k128-prepared-float")\n'
        '    metadata = "prepared-n32-build-20261121-v2.json" if os.environ["RELIC_DIRECT_CONSUMER_MODE"] == "n32" else "prepared-decode-build-20261121-v1.json"\n'
        '    build = json.loads((ROOT / "logs" / metadata).read_bytes())["variants"][name]\n'
        '    operator = AscendCTileCubeConsumer(build["library"], sha256=build["sha256"])\n'
        '    if not operator.prepared_decode_tables:\n'
        '        raise RuntimeError("the prepared-table native feature was not selected")\n', 1)
    source = source.replace('consumer=operator, certified_bf16_operands=True, backend="triton")',
                            'consumer=reference_operator, certified_bf16_operands=True, backend="triton")', 1)
    marker = '        row = {"rank": rank, "world": world,'
    if source.count(marker) != 1:
        raise RuntimeError('the actual-expert receipt boundary moved')
    source = source.replace(marker,
        '        import time\n'
        '        times = []\n'
        '        for _ in range(4):\n'
        '            shard.forward(x, weights, indices)\n'
        '        dist.barrier(); torch.npu.synchronize()\n'
        '        for _ in range(32):\n'
        '            start = time.perf_counter()\n'
        '            shard.forward(x, weights, indices)\n'
        '            torch.npu.synchronize()\n'
        '            times.append(1000 * (time.perf_counter() - start))\n' + marker, 1)
    save = '        with (OUTPUT / f"expert-TP2-EP4-rank-{rank}.json").open("x") as stream:\n'
    source = source.replace(save,
        '        row["consumer_mode"] = os.environ["RELIC_DIRECT_CONSUMER_MODE"]\n'
        '        row["consumer"] = operator.describe()\n'
        '        row["synchronized_routed_TP_body_ms"] = times\n'
        '        row["timing_scope"] = "routed projections/epilogue/FP32 subgroup reduction, not shared FFN or model TPS"\n' + save, 1)
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), base.__dict__)
    return base.main()


if __name__ == '__main__':
    main()
