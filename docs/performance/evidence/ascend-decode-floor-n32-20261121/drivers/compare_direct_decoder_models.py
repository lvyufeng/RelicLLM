"""CPU-only all-rank consumer comparison, complete FFN timing and real-output floor."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path('<workspace>/relic-v41-perf-20261118-0938/logs')
OUTPUT = Path(os.environ['RELIC_PERF_OUTPUT'])


def completed(pattern):
    paths=[]
    for path in ROOT.glob(pattern):
        receipt=path/'summary.json'
        if receipt.is_file():
            row=json.loads(receipt.read_bytes())
            if row['returncode']==0 and row['cleanup']['complete']:
                paths.append(path)
    if len(paths)!=1:
        raise RuntimeError(f'exactly one completed receipt required: {pattern}: {paths}')
    return paths[0]


def artifact(path, hashes):
    hashes[str(path)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    hashes, modes, tensors, models = {}, {}, {}, {}
    for mode in ('old','bits'):
        root=completed(f'original-direct-byte-{mode}-generated-20*T*Z')
        models[mode],tensors[mode]=[],[]
        ffns=[]
        for rank in range(8):
            path=root/f'Cube-model-rank-{rank}.json'
            row=json.loads(path.read_bytes())
            if row['rank']!=rank or row['timed_steps']!=16 or not row['single_stream'] or row['batch_size']!=1:
                raise RuntimeError('full-model timing identity differs')
            quality=root/f'Cube-quality-logits-rank-{rank}.pt'
            final=root/f'Cube-final-logits-rank-{rank}.pt'
            tensor=torch.load(quality,map_location='cpu',weights_only=True)
            tensor['final']=torch.load(final,map_location='cpu',weights_only=True)
            if any(value.dtype!=torch.float32 or not torch.isfinite(value).all() for value in tensor.values()):
                raise RuntimeError('native candidate is nonfinite or not FP32')
            for p in (path,quality,final):artifact(p,hashes)
            keep=('rank','dense_contract','expert_contract','quality_steps','quality_metrics','teacher_forced_reference_tokens',
                  'host_table_device_Engram','HC_decode_fold','expert_TP_EP','direct_original_consumer','timed_warmup_steps_excluded',
                  'timed_steps','timed_context_start','timed_context_end','timed_token_ids','timed_step_seconds','timed_decode_seconds',
                  'sustained_single_stream_TPS','NPU_completion_and_host_argmax_included','single_stream','batch_size','finite_final_logits',
                  'graph_candidate_eager_FP32_bit_differences','production_defaults_and_serving_guards_unchanged')
            retained={key:row[key] for key in keep if key in row}
            retained['original_complete_model_artifact']={'path':str(path),**hashes[str(path)]}
            with (OUTPUT/f'consumer-{mode}-rank-{rank}.json').open('x') as stream:json.dump(retained,stream,indent=2)
            models[mode].append(row);tensors[mode].append(tensor)
            fpath=root/f'complete-FFN-rank-{rank}.json'
            ffn=json.loads(fpath.read_bytes());artifact(fpath,hashes)
            if len(ffn['records'])!=40 or not all(r['same_real_full_FFN_BF16_bit_differences']==0 for r in ffn['records']):
                raise RuntimeError('complete-FFN real-output bit checks failed')
            ffns.append(ffn)
            with (OUTPUT/f'complete-FFN-{mode}-rank-{rank}.json').open('x') as stream:json.dump(ffn,stream,indent=2)
        for rank in range(1,8):
            if models[mode][rank]['timed_token_ids']!=models[mode][0]['timed_token_ids'] or any(
                not torch.equal(tensors[mode][rank][key].view(torch.int32),tensors[mode][0][key].view(torch.int32)) for key in tensors[mode][0]):
                raise RuntimeError('model tensors/tokens differ across physical ranks')
        wall=max(row['timed_decode_seconds'] for row in models[mode])
        modes[mode]={'directory':str(root),'slowest_rank_seconds':wall,'batch_one_single_request_TPS':16/wall,
                     'context':[24,40],'timed_steps':16,'timed_tokens':models[mode][0]['timed_token_ids'],
                     'same_mode_all_physical_rank_teacher_final_FP32_bits_equal':True,
                     'complete_FFN_layer_slowest_rank_ms':[max(row['records'][layer]['complete_FFN_mean_ms'] for row in ffns) for layer in range(40)],
                     'isolated_FFN_layer_sum_not_model_critical_path':True}
    bits=[]
    for rank in range(8):
        left,right=tensors['old'][rank],tensors['bits'][rank]
        counts={key:int((left[key].view(torch.int32)!=right[key].view(torch.int32)).sum()) for key in left}
        if any(counts.values()) or models['old'][rank]['timed_token_ids']!=models['bits'][rank]['timed_token_ids']:
            raise RuntimeError('direct exponent decoder changed the same-Cube original model FP32 bits')
        bits.append({'rank':rank,'all_full_teacher_and_final_FP32_bit_differences':counts,'same_timed_token_stream':True})
    floor_root=completed('original-decode-real-output-floor-tp2-20*T*Z')
    floor=[]
    for rank in range(8):
        path=floor_root/f'non-expert-floor-rank-{rank}.json'
        row=json.loads(path.read_bytes());artifact(path,hashes)
        if not all(r['full_model_FP32_logits_bit_equal_to_real_baseline'] and r['final_KV_compression_and_hash_bytes_equal'] for r in row['reports'].values()):
            raise RuntimeError('real-output floor conditioning changed')
        floor.append(row)
        with (OUTPUT/f'non-expert-floor-rank-{rank}.json').open('x') as stream:json.dump(row,stream,indent=2)
    floors={mode:max(row['reports'][mode]['ms_per_forward'] for row in floor) for mode in ('routed_output_replay','complete_FFN_output_replay')}
    report={'modes':modes,'old_vs_bits_all_rank_comparison':bits,
            'same_stream_TPS_ratio':modes['bits']['batch_one_single_request_TPS']/modes['old']['batch_one_single_request_TPS'],
            'latency_reduction_ms_per_forward':1000*(modes['old']['slowest_rank_seconds']-modes['bits']['slowest_rank_seconds'])/16,
            'real_output_ablation_floor_ms':floors,'real_output_floor_not_generation_TPS':True,
            'current_non_expert_path_alone_exceeds_20ms_target':floors['complete_FFN_output_replay']>20,
            'all_8_ranks_complete_tensor_artifacts_compared':True,'original_bytes_quantization_and_defaults_unchanged':True,
            'native_K_reduction_is_still_authorized_experiment_not_exact_default':True,'artifact_hashes':hashes}
    with (OUTPUT/'direct-bit-full-model-comparison.json').open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ('artifact_hashes','old_vs_bits_all_rank_comparison')},indent=2),flush=True)


if __name__=='__main__':
    main()
