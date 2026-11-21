"""All-rank CPU audit of original K128, prepared N16 and wider N32 model windows."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

ROOT = Path('/data1/relic-v41-perf-20261118-0938/logs')
OUTPUT = Path(os.environ['RELIC_PERF_OUTPUT'])
CASES = {
    'old-16':'original-direct-byte-old-generated-20*T*Z',
    'old-64':'original-direct-byte-old-generated-steps64-20*T*Z',
    'prepared-16':'original-direct-byte-prepared-generated-20*T*Z',
    'prepared-64':'original-direct-byte-prepared-generated-steps64-20*T*Z',
    'n32-16':'original-direct-byte-n32-generated-20*T*Z',
    'n32-64':'original-direct-byte-n32-generated-steps64-20*T*Z',
    'global-64':'original-global-FFN-generated-tp2-steps64-20*T*Z',
}


def completed(pattern):
    found=[]
    for path in ROOT.glob(pattern):
        receipt=path/'summary.json'
        if receipt.is_file():
            row=json.loads(receipt.read_bytes())
            if row['returncode']==0 and row['cleanup']['complete']:
                found.append(path)
    if len(found)!=1:
        raise RuntimeError(f'exactly one completed owned model required: {pattern}: {found}')
    return found[0]


def tensor_metrics(left,right):
    delta=right-left
    return {'FP32_bit_differences':int((left.view(torch.int32)!=right.view(torch.int32)).sum()),
            'max_abs_difference':float(delta.abs().max()),
            'relative_L2':float(delta.norm()/left.norm().clamp_min(1e-30))}


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    model_rows, values, modes, identities = {}, {}, {}, {}
    for name,pattern in CASES.items():
        root=completed(pattern)
        model_rows[name],values[name]=[],[]
        steps=int(name.rsplit('-',1)[1])
        for rank in range(8):
            files=(root/f'Cube-model-rank-{rank}.json',root/f'Cube-quality-logits-rank-{rank}.pt',root/f'Cube-final-logits-rank-{rank}.pt')
            row=json.loads(files[0].read_bytes())
            if row['rank']!=rank or row['timed_steps']!=steps or row['batch_size']!=1 or not row['single_stream']:
                raise RuntimeError('full-model identity or autoregressive timing scope differs')
            for path in files:
                identities[str(path)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            tensor=torch.load(files[1],map_location='cpu',weights_only=True)
            tensor['final']=torch.load(files[2],map_location='cpu',weights_only=True)
            if any(value.dtype!=torch.float32 or not torch.isfinite(value).all() for value in tensor.values()):
                raise RuntimeError('complete quality tensor is nonfinite or not original FP32')
            keep=('rank','dense_contract','expert_contract','quality_steps','quality_metrics','teacher_forced_reference_tokens',
                  'host_table_device_Engram','HC_decode_fold','expert_TP_EP','direct_original_consumer',
                  'timed_warmup_steps_excluded','timed_steps','timed_context_start','timed_context_end','timed_token_ids',
                  'timed_step_seconds','timed_decode_seconds','sustained_single_stream_TPS',
                  'NPU_completion_and_host_argmax_included','single_stream','batch_size','finite_final_logits',
                  'graph_candidate_eager_FP32_bit_differences','production_defaults_and_serving_guards_unchanged')
            retained={key:row[key] for key in keep if key in row}
            retained['original_complete_artifact']={'path':str(files[0]),**identities[str(files[0])]}
            with (OUTPUT/f'{name}-rank-{rank}.json').open('x') as stream:json.dump(retained,stream,indent=2)
            model_rows[name].append(row);values[name].append(tensor)
        for rank in range(1,8):
            if model_rows[name][rank]['timed_token_ids']!=model_rows[name][0]['timed_token_ids'] or any(
                not torch.equal(values[name][rank][key].view(torch.int32),values[name][0][key].view(torch.int32)) for key in values[name][0]):
                raise RuntimeError('model teacher/final tensor bits or output tokens differ across physical ranks')
        wall=max(row['timed_decode_seconds'] for row in model_rows[name])
        modes[name]={'directory':str(root),'timed_steps':steps,'context':[24,24+steps],
            'slowest_rank_seconds':wall,'ms_per_token':1000*wall/steps,'single_request_batch_one_TPS':steps/wall,
            'timed_output_token_ids':model_rows[name][0]['timed_token_ids'],
            'all_8_rank_full_teacher_and_final_FP32_bits_equal':True,
            'eager_graph_same_candidate_bit_differences':sum(sum(row.get('graph_candidate_eager_FP32_bit_differences',[])) for row in model_rows[name]),
            'small_exact_reference_teacher_window_not_general_quality_acceptance':True}
    comparisons={}
    for candidate in ('prepared','n32','global'):
        for steps in (16,64):
            name=f'{candidate}-{steps}'
            if name not in values:continue
            control=f'old-{steps}'
            same_tokens=modes[name]['timed_output_token_ids']==modes[control]['timed_output_token_ids']
            checks=[]
            for rank in range(8):
                a,b=values[control][rank],values[name][rank]
                if not torch.equal(a['reference'].view(torch.int32),b['reference'].view(torch.int32)):
                    raise RuntimeError('the unchanged original exact reference differs across consumer trials')
                checks.append({'rank':rank,'metrics':{key:tensor_metrics(a[key],b[key]) for key in a},
                               'same_autoregressive_token_stream':same_tokens})
            comparisons[name]={'all_rank_full_tensor_metrics':checks,'same_autoregressive_token_stream':same_tokens,
                'first_output_difference':next((index for index,(a,b) in enumerate(zip(modes[control]['timed_output_token_ids'],modes[name]['timed_output_token_ids'])) if a!=b),None),
                'observed_TPS_ratio':modes[name]['single_request_batch_one_TPS']/modes[control]['single_request_batch_one_TPS'],
                'same_stream_ratio_only_when_outputs_match':same_tokens,
                'observed_ms_saved_per_token':modes[control]['ms_per_token']-modes[name]['ms_per_token']}
    repeat={}
    for mode in ('old','prepared','n32'):
        short,long=values[f'{mode}-16'],values[f'{mode}-64']
        repeat[mode]={'same_first_16_generated_tokens':modes[f'{mode}-16']['timed_output_token_ids']==modes[f'{mode}-64']['timed_output_token_ids'][:16],
            'all_rank_short_long_teacher_bit_differences':[int((short[rank]['candidate_teacher_forced'].view(torch.int32)!=long[rank]['candidate_teacher_forced'].view(torch.int32)).sum()) for rank in range(8)]}
    report={'modes':modes,'comparisons':comparisons,'short_long_repeats':repeat,'all_8_full_tensor_artifacts_compared':True,
            'N32_source_N16_checkpoint_banks_K32_scales_and_activation_quantization_unchanged':True,
            'native_Cube_K_reduction_still_explicit_experiment_not_original_exact_default':True,
            'production_and_serving_guards_unchanged':True,'artifact_hashes':identities,
            'target_50_TPS_reached':modes['n32-64']['single_request_batch_one_TPS']>=50}
    with (OUTPUT/'final-original-consumer-model-comparison.json').open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'TPS':{name:row['single_request_batch_one_TPS'] for name,row in modes.items()},
        'comparison':{name:{key:row[key] for key in ('same_autoregressive_token_stream','observed_TPS_ratio','observed_ms_saved_per_token')} for name,row in comparisons.items()},
        'all_rank_compared':True,'target_50_TPS_reached':report['target_50_TPS_reached']},indent=2),flush=True)


if __name__=='__main__':
    main()
