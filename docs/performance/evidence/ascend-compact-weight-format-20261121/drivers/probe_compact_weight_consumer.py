"""Gate offline Cube-order FP4 on one leased 910B with real checkpoint projections."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch_npu
from relic_core.kernels.ascendc_online_tile_cube import AscendCTileCubeConsumer
from relic_core.kernels.cube_packed_format import convert_fp4_cube, recover_fp4_cube
from relic_core.kernels.fast_fused_fp8_activation_decode import quantize_decode_fp8_fast
from relic_core.kernels.online_tile_cube import pack_expert_tile_bank
from relic_core.kernels.unpack import unpack_fp4_to_int8, fp4_int8_scale
from relicllm.models.deepseek_v4_1.ascend_graphs import _record, npu_graph_api
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint, scale_key

ROOT = Path('<private-compact-root>')
OUTPUT = Path(os.environ['RELIC_PERF_OUTPUT'])


def bank_pair(code, scale, device):
    original = pack_expert_tile_bank(code, scale)
    converted = [convert_fp4_cube(c, s) for c, s in zip(code, scale)]
    for source_c, source_s, payload in zip(code, scale, converted):
        restored = recover_fp4_cube(payload)
        if not torch.equal(restored[0].view(torch.uint8), source_c.view(torch.uint8)) or not torch.equal(restored[1], source_s):
            raise RuntimeError('offline conversion lost original code/scale bytes')
    compact = tuple(torch.stack([getattr(p, attr) for p in converted]).reshape(target.shape).contiguous()
                    for attr, target in zip(('codes', 'scales'), original))
    if sum(t.numel() for t in compact) != code.numel() + scale.numel():
        raise RuntimeError('compact payload exceeded original FP4 plus scale budget')
    return {'original-control': tuple(t.to(device) for t in original),
            'cube-packed': tuple(t.to(device) for t in compact)}


def capture(op, x, pair, ids, n):
    ring = torch.empty(op.block_dim, 2, op.ring_elements, dtype=torch.bfloat16, device=x.device)
    accum = torch.empty(ids.numel(), n, dtype=torch.float32, device=x.device)
    state = {'op': op, 'ring': ring, 'accum': accum}
    def body():
        state['out'] = op(x, *pair, ids, world=1, certified_bf16_operands=True, scratch=ring, accumulation=accum)
    api = npu_graph_api()
    state['pool'] = api.graph_pool_handle()
    state['graph'] = _record(body, state['pool'], 2, api)
    return state


def compare(states):
    control = states['original-control']
    for state in states.values():
        if not torch.equal(state['out'].cpu().view(torch.int16), control['out'].cpu().view(torch.int16)):
            raise RuntimeError('compact physical layout changed same-Cube BF16 outputs')
        if not torch.equal(state['accum'].cpu().view(torch.int32), control['accum'].cpu().view(torch.int32)):
            raise RuntimeError('compact physical layout changed same-Cube FP32 accumulator bits')


def timed(states, ids, active, repeats=24):
    ids.copy_(torch.tensor(list(range(active)) + [-1] * (ids.numel() - active), dtype=torch.int32, device=ids.device))
    for state in states.values():
        for _ in range(4): state['graph'].replay()
    torch.npu.synchronize()
    samples = {name: [] for name in states}
    names = list(states)
    for repetition in range(repeats):
        for name in names if repetition % 2 else names[::-1]:
            start = time.perf_counter()
            states[name]['graph'].replay()
            torch.npu.synchronize()
            samples[name].append(1000 * (time.perf_counter() - start))
    compare(states)
    return {name: {'samples_ms': values, 'mean_ms': sum(values) / len(values)} for name, values in samples.items()}


@torch.inference_mode()
def main():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    device = torch.device('npu:0')
    rows = json.loads((ROOT / 'logs/compact-build.json').read_bytes())['variants']
    operators = {name: AscendCTileCubeConsumer(row['library'], sha256=row['sha256']) for name, row in rows.items()}
    if operators['cube-packed'].weight_format != 'relic.fp4.cube-n32-k128.v1':
        raise RuntimeError('compiled native candidate does not identify the compact format')
    for op in operators.values(): op.prepare_device_metadata(device)
    code = torch.arange(256).to(torch.uint8).reshape(16, 16).repeat(251, 4).unsqueeze(0)
    scales = torch.arange(2, 253).to(torch.uint8).repeat_interleave(16).reshape(1, 4016, 1).repeat(1, 1, 4)
    code = torch.cat((code, code[:, :16]), dim=1)
    scales = torch.cat((scales, scales[:, :16]), dim=1)
    for block in range(4): code[:, :, block * 16:(block + 1) * 16] = code[:, :, block * 16:(block + 1) * 16].roll(block, -1)
    expected = (unpack_fp4_to_int8(code, backend='torch').float()
                * fp4_int8_scale(scales, backend='torch').repeat_interleave(32, -1)).bfloat16()[0]
    pairs = bank_pair(code, scales, device)
    x = torch.zeros(1, 128, dtype=torch.bfloat16, device=device)
    ids = torch.zeros(1, dtype=torch.int32, device=device)
    states = {name: capture(op, x, pairs[name], ids, 4032) for name, op in operators.items()}
    for position in range(128):
        x.zero_(); x[0, position] = 1
        for state in states.values(): state['graph'].replay()
        torch.npu.synchronize()
        compare(states)
        if not torch.equal(states['cube-packed']['out'].cpu()[0].view(torch.int16), expected[:, position].view(torch.int16)):
            raise RuntimeError(f'compact code/scale corpus failed K={position}')
    corpus = {'original_E2M1_byte_values': 256, 'original_E8M0_scale_codes': [2, 252],
              'K_positions': 128, 'decoded_BF16_values_checked': 4032 * 128, 'differences': 0}
    del states, pairs, expected
    generator = torch.Generator().manual_seed(0xC0B1)
    dynamic = []
    for k, n, shared in ((256, 64, True), (256, 64, False), (1152, 64, False)):
        code = torch.randint(0, 256, (8, n, k // 2), generator=generator, dtype=torch.uint8)
        scales = torch.full((8, n, k // 32), 121, dtype=torch.uint8)
        pairs = bank_pair(code, scales, device)
        x = torch.zeros(1 if shared else 6, k, dtype=torch.bfloat16, device=device)
        ids = torch.zeros(6, dtype=torch.int32, device=device)
        states = {name: capture(op, x, pairs[name], ids, n) for name, op in operators.items()}
        for replay in range(32):
            active = (0, 1, 3, 6)[replay % 4]
            selected = torch.randperm(8, generator=generator)[:active].tolist() + [-1] * (6 - active)
            x.copy_(quantize_decode_fp8_fast(torch.randn(x.shape, generator=generator).bfloat16(), backend='torch').to(device))
            ids.copy_(torch.tensor(selected, dtype=torch.int32, device=device))
            for state in states.values(): state['graph'].replay()
            torch.npu.synchronize(); compare(states)
        for invalid in (-2, 8, 2**31 - 1):
            ids.fill_(invalid)
            for state in states.values(): state['graph'].replay()
            torch.npu.synchronize()
            if not all(bool(torch.isnan(state['out']).all()) for state in states.values()):
                raise RuntimeError('invalid compact source was not poisoned')
        ids.zero_()
        for invalid in (0, 1, 253, 254, 255):
            for pair in pairs.values(): pair[1].fill_(invalid)
            for state in states.values(): state['graph'].replay()
            torch.npu.synchronize()
            if not all(bool(torch.isnan(state['out']).all()) for state in states.values()):
                raise RuntimeError('invalid compact scale was not poisoned')
        dynamic.append({'K': k, 'N': n, 'shared': shared, 'changing_replays': 32,
                        'same_Cube_FP32_BF16_bit_differences': 0, 'invalid_sources_and_scales_poisoned': True})
        del states, pairs
    checkpoint = V41Checkpoint('<original-checkpoint>', device='cpu')
    real = []
    sidecar = OUTPUT / 'one-expert-compact-checkpoint'
    sidecar.mkdir(exist_ok=False)
    try:
        for projection in ('w1', 'w3', 'w2'):
            keys = [f'layers.0.ffn.experts.{expert}.{projection}.weight' for expert in range(8)]
            code = torch.stack([checkpoint.packed(key).view(torch.uint8).clone() for key in keys])
            scales = torch.stack([checkpoint.packed(scale_key(key)).view(torch.uint8).clone() for key in keys])
            if projection != 'w2': code, scales = code[:, :1152].contiguous(), scales[:, :1152].contiguous()
            else: code, scales = code[:, :, :576].contiguous(), scales[:, :, :36].contiguous()
            n, k = code.shape[1], code.shape[2] * 2
            payload = convert_fp4_cube(code[0], scales[0])
            persisted = {}
            for name, tensor in (('codes', payload.codes), ('scales', payload.scales)):
                data = tensor.numpy().tobytes()
                path = sidecar / f'{projection}.{name}.bin'
                with path.open('xb') as handle: handle.write(data); handle.flush(); os.fsync(handle.fileno())
                reread = path.read_bytes()
                if reread != data: raise RuntimeError('sidecar payload changed across persistence')
                persisted[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(), 'shape': list(tensor.shape)}
            pairs = bank_pair(code, scales, device)
            x = quantize_decode_fp8_fast(torch.randn(1 if projection != 'w2' else 6, k, generator=generator).bfloat16(), backend='torch').to(device)
            ids = torch.arange(6, dtype=torch.int32, device=device)
            states = {name: capture(op, x, pairs[name], ids, n) for name, op in operators.items()}
            real.append({'projection': projection, 'local_TP2_shape': [n, k], 'original_checkpoint_keys': keys,
                         'same_Cube_FP32_BF16_bit_differences': 0, 'sidecar': persisted,
                         'paired_times_by_active_slots': {str(active): timed(states, ids, active) for active in (0, 1, 3, 6)}})
            del states, pairs
    finally: checkpoint.close()
    report = {'corpus': corpus, 'dynamic': dynamic, 'real_projections': real,
              'operators': {name: op.describe() for name, op in operators.items()},
              'same_size_compact_conversion_is_byte_reversible': True, 'whole_expanded_checkpoint_bytes': 0,
              'original_checkpoint_writes': 0, 'not_model_TPS': True}
    with (OUTPUT / 'compact-probe.json').open('x') as handle: json.dump(report, handle, indent=2)
    with (sidecar / 'manifest.json').open('x') as handle: json.dump({'format': operators['cube-packed'].weight_format, 'projections': [row['sidecar'] for row in real]}, handle, indent=2)
    print(json.dumps({'corpus': corpus, 'dynamic': dynamic,
                      'active3_mean_ms': {r['projection']: {name: v['mean_ms'] for name, v in r['paired_times_by_active_slots']['3'].items()} for r in real}}), flush=True)


if __name__ == '__main__': main()
