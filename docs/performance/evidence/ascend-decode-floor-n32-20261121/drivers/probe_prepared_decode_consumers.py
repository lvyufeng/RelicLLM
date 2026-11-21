"""Exhaustive prepared-table codec and same-Cube real-weight projection gates."""
from __future__ import annotations

import inspect
from pathlib import Path

import probe_direct_bit_decoders as base


def main():
    source = inspect.getsource(base.main)
    source = source.replace('logs/direct-bit-consumer-build-20261121-v1.json', 'logs/prepared-decode-build-20261121-v1.json', 1)
    source = source.replace(',\n                 "AIV-direct": DirectAIVConsumer()', '', 1)
    source = source.replace('"AIV-direct": DirectAIVConsumer()', '', 1)
    source = source.replace('if not op.direct_bf16_bits or op.group_k != row["group_k"]:',
        'if not op.prepared_decode_tables or op.direct_bf16_bits != row["direct_bits"] or op.group_k != row["group_k"]:', 1)
    source = source.replace('    corpus = []\n',
        '    for op in operators.values():\n'
        '        op.prepare_device_metadata(device)\n'
        '    corpus = []\n', 1)
    source = source.replace('("k128-direct-bits", "k256-direct-bits", "AIV-direct")',
                            '("k128-prepared-float", "k128-prepared-bits")', 1)
    source = source.replace('(576, 64, 6, False)', '(1152, 64, 6, False)', 1)
    begin = '            if not torch.equal(values["k128-original"].view(torch.int16), values["k128-direct-bits"].view(torch.int16)):\n'
    first = source.index(begin)
    paired = source.index('"paired_times_by_active_slots":', first)
    last = source.index('\n', paired) + 1
    source = source[:first] + (
        '            for name, value in values.items():\n'
        '                if not torch.equal(value.view(torch.int16), values["k128-original"].view(torch.int16)):\n'
        '                    raise RuntimeError("prepared decode tables changed the real same-Cube BF16 output")\n'
        '                if not torch.equal(states[name]["accum"].cpu().view(torch.int32), states["k128-original"]["accum"].cpu().view(torch.int32)):\n'
        '                    raise RuntimeError("prepared decode tables changed the same-Cube FP32 accumulator")\n'
        '                if not torch.isfinite(value).all():\n'
        '                    raise RuntimeError("prepared real original weights produced nonfinite values")\n'
        '            real.append({"projection":projection,"original_checkpoint_keys":keys,"local_TP2_shape":[n,k],\n'
        '                "same_Cube_BF16_bit_differences":0,"same_Cube_FP32_accumulator_bit_differences":0,\n'
        '                "immutable_tables_include_no_checkpoint_weights":True,\n'
        '                "paired_times_by_active_slots":{str(active):paired_times(states,ids,active) for active in (0,1,3,6)}})\n'
    ) + source[last:]
    source = source.replace('direct-bit-consumer-probe.json', 'prepared-decode-probe.json', 1)
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), base.__dict__)
    return base.main()


if __name__ == '__main__':
    main()
