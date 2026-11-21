"""CPU-only complete FFN/HC model audit, preserving generated versus fixed-input scope."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

ROOT=Path('/data1/relic-v41-perf-20261118-0938/logs')
OUTPUT=Path(os.environ['RELIC_PERF_OUTPUT'])


def completed(pattern):
    cases=[]
    for path in ROOT.glob(pattern):
        receipt=path/'summary.json'
        if receipt.exists():
            row=json.loads(receipt.read_bytes())
            if row['returncode']==0 and row['cleanup']['complete']:cases.append(path)
    if len(cases)!=1:raise RuntimeError(f'unambiguous completed model required {pattern}: {cases}')
    return cases[0]


def metrics(a,b):
    rows=[]
    for left,right in zip(a,b):
        diff=right-left
        p,q=left.double().log_softmax(-1),right.double().log_softmax(-1)
        rows.append({'FP32_bit_differences':int((left.view(torch.int32)!=right.view(torch.int32)).sum()),
                     'max_abs_difference':float(diff.abs().max()),'relative_L2':float(diff.norm()/left.norm().clamp_min(1e-30)),
                     'KL_FP64':float((p.exp()*(p-q)).sum()),'greedy_matches':int(left.argmax())==int(right.argmax())})
    return rows


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    modes,rows,tensors,hashes={},{},{},{}
    for mode,pattern in (
        ('old-generated','original-direct-byte-old-generated-20*T*Z'),
        ('HC-generated','original-HC-coefficient-generated-tp2-20*T*Z'),
        ('global-generated','original-global-FFN-generated-tp2-20*T*Z'),
        ('old-matched','original-direct-byte-old-matched-20*T*Z'),
        ('HC-matched','original-HC-coefficient-matched-tp2-20*T*Z'),
        ('global-matched','original-global-FFN-matched-tp2-20*T*Z')):
        root=completed(pattern);rows[mode]=[];tensors[mode]=[]
        for rank in range(8):
            path=root/f'Cube-model-rank-{rank}.json'
            row=json.loads(path.read_bytes())
            if row['rank']!=rank or row['timed_steps']!=16 or not row['single_stream'] or row['batch_size']!=1:
                raise RuntimeError('not the complete original model timing/rank configuration')
            paths=(path,root/f'Cube-quality-logits-rank-{rank}.pt',root/f'Cube-final-logits-rank-{rank}.pt')
            tensor=torch.load(paths[1],map_location='cpu',weights_only=True)
            tensor['final']=torch.load(paths[2],map_location='cpu',weights_only=True)
            if any(value.dtype!=torch.float32 or not torch.isfinite(value).all() for value in tensor.values()):
                raise RuntimeError('nonfinite candidate or incorrectly narrowed quality tensor')
            for artifact in paths:hashes[str(artifact)]={'bytes':artifact.stat().st_size,'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}
            keep=('rank','dense_contract','expert_contract','quality_steps','quality_metrics','teacher_forced_reference_tokens',
                  'host_table_device_Engram','HC_decode_fold','expert_TP_EP','HC_FP32_coefficients',
                  'timed_warmup_steps_excluded','timed_steps','timed_context_start','timed_context_end',
                  'timed_token_ids','timed_step_seconds','timed_decode_seconds','sustained_single_stream_TPS',
                  'single_request_forward_rate_not_generation_TPS','same_token_input_latency_not_generation_TPS','fixed_input_token_ids',
                  'NPU_completion_and_host_argmax_included','single_stream','batch_size','finite_final_logits',
                  'graph_candidate_eager_FP32_bit_differences','production_defaults_and_serving_guards_unchanged')
            retained={key:row[key] for key in keep if key in row};retained['original_model_artifact']={'path':str(path),**hashes[str(path)]}
            with (OUTPUT/f'{mode}-rank-{rank}.json').open('x') as stream:json.dump(retained,stream,indent=2)
            rows[mode].append(row);tensors[mode].append(tensor)
        for rank in range(1,8):
            if rows[mode][rank]['timed_token_ids']!=rows[mode][0]['timed_token_ids'] or any(
                not torch.equal(tensors[mode][rank][key].view(torch.int32),tensors[mode][0][key].view(torch.int32)) for key in tensors[mode][0]):
                raise RuntimeError('full candidate model quality/tokens differ across physical ranks')
        matched=mode.endswith('matched')
        if matched and ('sustained_single_stream_TPS' in rows[mode][0] or not rows[mode][0]['same_token_input_latency_not_generation_TPS']):
            raise RuntimeError('fixed-input forward timing cannot be called generation TPS')
        wall=max(row['timed_decode_seconds'] for row in rows[mode])
        modes[mode]={'directory':str(root),'wall_seconds':wall,'ms_per_forward':wall*1000/16,
                     'real_generation_TPS':None if matched else 16/wall,'fixed_input_not_generation_TPS':matched,
                     'output_token_ids':rows[mode][0]['timed_token_ids'],
                     'input_token_ids':rows[mode][0].get('fixed_input_token_ids'),
                     'same_candidate_all_8_rank_teacher_final_FP32_bits_equal':True,
                     'teacher_original_exact_8_greedy_match':all(m['greedy_matches'] for row in rows[mode] for m in row['quality_metrics'])}
    comparisons={}
    for candidate in ('HC','global'):
        pair=[]
        for rank in range(8):
            old,new=tensors['old-generated'][rank],tensors[candidate+'-generated'][rank]
            if not torch.equal(old['reference'].view(torch.int32),new['reference'].view(torch.int32)):
                raise RuntimeError('original exact model reference differs between trial configurations')
            a,b=rows['old-generated'][rank]['timed_token_ids'],rows[candidate+'-generated'][rank]['timed_token_ids']
            pair.append({'rank':rank,'same_generated_token_stream':a==b,'first_generated_output_difference':next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),None),
                         'teacher_metrics':metrics(old['candidate_teacher_forced'],new['candidate_teacher_forced']),
                         'final_FP32_bit_differences':int((old['final'].view(torch.int32)!=new['final'].view(torch.int32)).sum())})
        if modes['old-matched']['input_token_ids']!=modes[candidate+'-matched']['input_token_ids']:
            raise RuntimeError('conditioning tokens differ in the purported matched-input latency control')
        comparisons[candidate]={'all_8_rank_generated_quality':pair,
            'same_generated_stream':all(row['same_generated_token_stream'] for row in pair),
            'raw_generation_rate_ratio':modes[candidate+'-generated']['real_generation_TPS']/modes['old-generated']['real_generation_TPS'],
            'matched_input_latency_ratio':modes['old-matched']['wall_seconds']/modes[candidate+'-matched']['wall_seconds'],
            'matched_input_ms_saved':modes['old-matched']['ms_per_forward']-modes[candidate+'-matched']['ms_per_forward']}
    report={'modes':modes,'comparisons':comparisons,'all_8_rank_complete_teacher_and_final_tensor_artifacts_compared':True,
            'fixed_input_latencies_not_generation_TPS':True,'original_checkpoint_quantization_and_production_defaults_unchanged':True,
            'HC_projection_and_global_FP32_addition_order_are_explicit_reduction_experiments':True,
            'small_teacher_window_not_general_quality_acceptance':True,'artifact_hashes':hashes}
    with (OUTPUT/'full-FFN-HC-model-comparison.json').open('x') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({'modes':modes,'comparison_ratios':{k:{p:v[p] for p in ('same_generated_stream','raw_generation_rate_ratio','matched_input_latency_ratio','matched_input_ms_saved')} for k,v in comparisons.items()}},indent=2),flush=True)


if __name__=='__main__':main()
