"""Exhaustive and real-weight N32 consumption from untouched original N16 banks."""
from __future__ import annotations

import inspect
from pathlib import Path

import probe_direct_bit_decoders as base


def main():
    source = inspect.getsource(base.main)
    source = source.replace('logs/direct-bit-consumer-build-20261121-v1.json', 'logs/prepared-n32-build-20261121-v2.json', 1)
    source = source.replace('"AIV-direct": DirectAIVConsumer()', '', 1)
    source = source.replace('if not op.direct_bf16_bits or op.group_k != row["group_k"]:',
        'if not op.prepared_decode_tables or op.output_rows != 32 or op.group_k != row["group_k"]:', 1)
    source = source.replace('    corpus = []\n',
        '    for op in operators.values():\n'
        '        op.prepare_device_metadata(device)\n'
        '    corpus = []\n', 1)
    source = source.replace('("k128-direct-bits", "k256-direct-bits", "AIV-direct")', '("k128-prepared-n32",)', 1)
    pad = '        for block in range(k // 32):\n'
    if source.count(pad) != 1:
        raise RuntimeError('the exhaustive corpus padding boundary moved')
    source = source.replace(pad,
        '        code = torch.cat((code, code[:, :16]), dim=1)\n'
        '        scale = torch.cat((scale, scale[:, :16]), dim=1)\n' + pad, 1)
    source = source.replace('entry(op, x, bt, st, ids, 4016)', 'entry(op, x, bt, st, ids, 4032)', 1)
    source = source.replace('"BF16_values_checked": 4016 * k,',
        '"BF16_values_checked": 4032 * k, "unique_original_corpus_rows":4016, "duplicate_padding_rows":16,', 1)
    source = source.replace('(576, 64, 6, False)', '(1152, 64, 6, False)', 1)
    begin = '            if not torch.equal(values["k128-original"].view(torch.int16), values["k128-direct-bits"].view(torch.int16)):\n'
    first = source.index(begin)
    paired = source.index('"paired_times_by_active_slots":', first)
    last = source.index('\n', paired) + 1
    source = source[:first] + (
        '            for name, value in values.items():\n'
        '                if not torch.equal(value.view(torch.int16), values["k128-original"].view(torch.int16)):\n'
        '                    raise RuntimeError("N32 changed the real original N16 bank BF16 output")\n'
        '                if not torch.equal(states[name]["accum"].cpu().view(torch.int32), states["k128-original"]["accum"].cpu().view(torch.int32)):\n'
        '                    raise RuntimeError("N32 changed the same-K128 Cube FP32 accumulator")\n'
        '                if not torch.isfinite(value).all():\n'
        '                    raise RuntimeError("N32 produced nonfinite real original weights")\n'
        '            real.append({"projection":projection,"original_checkpoint_keys":keys,"local_TP2_shape":[n,k],\n'
        '                "same_K128_Cube_BF16_bit_differences":0,"same_K128_Cube_FP32_accumulator_bit_differences":0,\n'
        '                "original_N16_banks_and_K32_scales_unchanged":True,\n'
        '                "paired_times_by_active_slots":{str(active):paired_times(states,ids,active) for active in (0,1,3,6)}})\n'
    ) + source[last:]
    source = source.replace('direct-bit-consumer-probe.json', 'n32-consumer-probe.json', 1)
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), base.__dict__)
    return base.main()


if __name__ == '__main__':
    main()
