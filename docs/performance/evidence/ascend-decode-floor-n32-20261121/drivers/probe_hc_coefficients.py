"""Actual FP32 HC coefficient/RMS projections; nonlinear providers stay original."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch.nn.functional as F
import torch_npu

from relic_core.kernels.hc_decode_coefficients import hc_project_rms
from relic_core.kernels.hc_sinkhorn_normalize import hc_split_sinkhorn_normalized
from relicllm.models.deepseek_v4_1.ascend_graphs import _record, npu_graph_api
from relicllm.models.deepseek_v4_1.config import load_config, resolve_config
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint

ROOT = Path('/data1/relic-v41-perf-20261118-0938')
CHECKPOINT = '/data1/relic-v41-20261116-210/checkpoint'
OUTPUT = Path(os.environ['RELIC_PERF_OUTPUT'])


def entry(body):
    state = {}
    def capture():
        state['out'] = body()
    api = npu_graph_api()
    state['pool'] = api.graph_pool_handle()
    state['graph'] = _record(capture, state['pool'], 2, api)
    return state


@torch.inference_mode()
def main():
    torch.npu.set_device(0); torch.set_num_threads(4)
    device = torch.device('npu:0')
    config = load_config(resolve_config(CHECKPOINT)).text
    checkpoint = V41Checkpoint(CHECKPOINT, device='cpu')
    generator = torch.Generator().manual_seed(0x4CF32)
    rows = []
    try:
        for layer in (0, 20, 39):
            for which in ('attn', 'ffn'):
                key = f'layers.{layer}.hc_{which}_fn'
                weights = checkpoint.packed(key).clone()
                scales = checkpoint.packed(f'layers.{layer}.hc_{which}_scale').clone()
                base = checkpoint.packed(f'layers.{layer}.hc_{which}_base').clone()
                if weights.dtype != torch.float32 or weights.shape != (24, 20480):
                    raise RuntimeError('HC probe must retain the original complete FP32 coefficient matrix')
                weight, scale, offset = weights.to(device), scales.to(device), base.to(device)
                x = torch.zeros(1, 1, 4, 5120, dtype=torch.bfloat16, device=device)
                def projection():
                    source = x.flatten(2).float()
                    return F.linear(source, weight) * torch.rsqrt(source.square().mean(-1, keepdim=True) + config.norm_eps)
                def original():
                    return hc_split_sinkhorn_normalized(projection(), scale, offset, 4, config.hc_sinkhorn_iters, config.hc_eps)
                states = {'original': entry(original)}
                for tile in (512, 1024, 2048):
                    def fused(tile=tile):
                        return hc_split_sinkhorn_normalized(hc_project_rms(x, weight, norm_eps=config.norm_eps, k_tile=tile),
                            scale, offset, 4, config.hc_sinkhorn_iters, config.hc_eps)
                    states[str(tile)] = entry(fused)
                differences = {name: {'projection_max_abs': 0.0, 'projection_max_relative_L2': 0.0,
                                     'HC_output_max_abs': 0.0, 'graph_eager_projection_bit_differences': 0} for name in ('512','1024','2048')}
                for replay in range(32):
                    value = torch.randn(1,1,4,5120,generator=generator).bfloat16()
                    if replay % 8 == 0:
                        value.zero_()
                    elif replay % 8 == 1:
                        value.mul_(8)
                    x.copy_(value.to(device))
                    for state in states.values():
                        state['graph'].replay()
                    torch.npu.synchronize()
                    reference = projection().cpu()
                    original_outputs = [v.cpu() for v in states['original']['out']]
                    for tile in ('512','1024','2048'):
                        projected = hc_project_rms(x, weight, norm_eps=config.norm_eps, k_tile=int(tile)).cpu()
                        delta = projected - reference
                        if not torch.isfinite(projected).all():
                            raise RuntimeError('HC FP32 projection produced a nonfinite coefficient')
                        metric = differences[tile]
                        metric['projection_max_abs'] = max(metric['projection_max_abs'], float(delta.abs().max()))
                        metric['projection_max_relative_L2'] = max(metric['projection_max_relative_L2'], float(delta.norm()/reference.norm().clamp_min(1e-30)))
                        outputs = [v.cpu() for v in states[tile]['out']]
                        eager = hc_split_sinkhorn_normalized(hc_project_rms(x, weight, norm_eps=config.norm_eps, k_tile=int(tile)),
                            scale, offset, 4, config.hc_sinkhorn_iters, config.hc_eps)
                        torch.npu.synchronize()
                        if any(not torch.equal(a.view(torch.int32), b.cpu().view(torch.int32)) for a,b in zip(outputs,eager)):
                            raise RuntimeError('HC same-provider eager/capture values differ')
                        metric['HC_output_max_abs'] = max(metric['HC_output_max_abs'], max(float((a-b).abs().max()) for a,b in zip(outputs,original_outputs)))
                # Matching input, alternating graph replay order; includes sync,
                # excludes quality copies and does not claim model TPS.
                for state in states.values():
                    for _ in range(3):
                        state['graph'].replay()
                torch.npu.synchronize()
                samples = {name:[] for name in states}
                names = list(states)
                for repeat in range(32):
                    for name in names if repeat % 2 else names[::-1]:
                        begin=time.perf_counter(); states[name]['graph'].replay(); torch.npu.synchronize()
                        samples[name].append(1000*(time.perf_counter()-begin))
                rows.append({'layer':layer,'projection':which,'original_key':key,'FP32_source_sha256':hashlib.sha256(weights.numpy().tobytes()).hexdigest(),
                    'original_FP32_coefficients_retained':True,'original_nonlinear_provider_retained':True,'changing_input_replays':32,
                    'differences':differences,'paired_graph_timings':{name:{'samples_ms':data,'mean_ms':sum(data)/len(data)} for name,data in samples.items()},
                    'native_projection_RMS_reduction_order_is_explicit_experiment':True,'not_model_TPS':True})
                del states,weight,scale,offset
    finally:
        checkpoint.close()
    with (OUTPUT/'HC-FP32-coefficient-probe.json').open('x') as stream:
        json.dump(rows,stream,indent=2)
    print(json.dumps({'HC_actual_graph_mean_ms':[{ 'layer':r['layer'],'projection':r['projection'], 'timings':{k:v['mean_ms'] for k,v in r['paired_graph_timings'].items()} } for r in rows]}),flush=True)


if __name__ == '__main__':
    main()
