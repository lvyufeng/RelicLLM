"""Compare all-rank sustained decode and full logits on the leased server."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import torch

ROOT = Path('<private-compact-root>')
CONTROL = ROOT / 'logs/compact-weight-format-original-control-host-table-HC-steps64-20261121T150232Z'


def bit_differences(a, b):
    if a.dtype != torch.float32 or b.dtype != torch.float32 or a.shape != b.shape:
        raise ValueError('compare identical full FP32 tensor shapes, never a truncated logit view')
    return int((a.view(torch.int32) != b.view(torch.int32)).sum())


def main():
    candidates = []
    for path in (ROOT / 'logs').glob('compact-weight-format-cube-packed-v2-host-table-HC-steps64-*/summary.json'):
        summary = json.loads(path.read_bytes())
        if summary.get('returncode') == 0 and summary.get('cleanup', {}).get('complete') is True:
            candidates.append(path.parent)
    if len(candidates) != 1:
        raise RuntimeError('require exactly one completed, cleaned-up compact candidate')
    candidate = candidates[0]
    if CONTROL.resolve() != CONTROL or candidate.resolve() != candidate:
        raise RuntimeError('model receipt directories must not escape the private root')
    results = []
    rows = {'control': [], 'candidate': []}
    identities = {}
    for rank in range(8):
        pair = {}
        tensors = {}
        for name, directory in (('control', CONTROL), ('candidate', candidate)):
            path = directory / f'Cube-model-rank-{rank}.json'
            data = path.read_bytes()
            row = json.loads(data)
            if (row['rank'] != rank or row['timed_steps'] != 64 or row['timed_warmup_steps_excluded'] != 4
                    or (row['timed_context_start'], row['timed_context_end']) != (24, 88)
                    or row['expert_TP_EP']['resident_bytes'] != 36097228800
                    or row['Engram_original_tables_host'] is not True):
                raise RuntimeError('the compared trials do not share the expected real model/timing/storage scope')
            rows[name].append(row)
            pair[name] = row
            quality = torch.load(directory / f'Cube-quality-logits-rank-{rank}.pt', map_location='cpu', weights_only=True)
            final = torch.load(directory / f'Cube-final-logits-rank-{rank}.pt', map_location='cpu', weights_only=True)
            tensors[name] = (quality, final)
            identities[f'{name}/rank-{rank}'] = {'json_sha256': hashlib.sha256(data).hexdigest(),
                'quality_file_sha256': hashlib.sha256((directory / f'Cube-quality-logits-rank-{rank}.pt').read_bytes()).hexdigest(),
                'final_file_sha256': hashlib.sha256((directory / f'Cube-final-logits-rank-{rank}.pt').read_bytes()).hexdigest()}
        old, new = tensors['control'], tensors['candidate']
        results.append({'rank': rank,
            'reference_FP32_bit_differences': bit_differences(old[0]['reference'], new[0]['reference']),
            'candidate_teacher_FP32_bit_differences': bit_differences(old[0]['candidate_teacher_forced'], new[0]['candidate_teacher_forced']),
            'final_FP32_bit_differences': bit_differences(old[1], new[1]),
            'all_logits_finite': all(bool(torch.isfinite(t).all()) for t in
                (old[0]['reference'], new[0]['reference'], old[0]['candidate_teacher_forced'], new[0]['candidate_teacher_forced'], old[1], new[1])),
            'same_64_generated_tokens': pair['control']['timed_token_ids'] == pair['candidate']['timed_token_ids'],
            'same_reference_tokens': pair['control']['teacher_forced_reference_tokens'] == pair['candidate']['teacher_forced_reference_tokens'],
            'same_native_contract': (pair['control']['expert_contract'], pair['control']['dense_contract']) ==
                (pair['candidate']['expert_contract'], pair['candidate']['dense_contract'])})
    timings = {}
    for name, group in rows.items():
        seconds = max(row['timed_decode_seconds'] for row in group)
        timings[name] = {'critical_rank_total_seconds': seconds, 'milliseconds_per_output_token': seconds / 64 * 1000,
                         'sustained_single_request_TPS': 64 / seconds, 'all_rank_TPS': [row['sustained_single_stream_TPS'] for row in group]}
    accepted = all(row['reference_FP32_bit_differences'] == row['candidate_teacher_FP32_bit_differences'] == row['final_FP32_bit_differences'] == 0
                   and row['all_logits_finite'] and row['same_64_generated_tokens'] and row['same_reference_tokens'] and row['same_native_contract']
                   for row in results)
    report = {'all_rank_comparison': results, 'accepted_same_provider_delta': accepted,
              'timing': timings, 'speedup_ratio': timings['candidate']['sustained_single_request_TPS'] / timings['control']['sustained_single_request_TPS'],
              'milliseconds_saved_per_token': timings['control']['milliseconds_per_output_token'] - timings['candidate']['milliseconds_per_output_token'],
              'control_directory': str(CONTROL), 'candidate_directory': str(candidate),
              'steps': 64, 'excluded_warmups': 4, 'context': [24, 88], 'batch': 1,
              'packed_expert_payload_bytes_per_rank': 36097228800,
              'source_checkpoint_files_not_rewritten': True, 'no_new_quantization': True,
              'exact_default_not_replaced': True, 'not_general_quality_acceptance': True,
              'raw_measurement_identities': identities,
              'target_50_TPS_reached': accepted and timings['candidate']['sustained_single_request_TPS'] >= 50}
    destination = Path(os.environ['RELIC_PERF_OUTPUT']) / 'compact-full-model-comparison.json'
    with destination.open('x') as handle: json.dump(report, handle, indent=2)
    print(json.dumps({'accepted': accepted, 'timing': timings, 'speedup_ratio': report['speedup_ratio'],
                      'milliseconds_saved_per_token': report['milliseconds_saved_per_token'], 'evidence': str(destination)}), flush=True)
    if not accepted:
        raise RuntimeError('compact format changed the same-provider real full model')
    return 0


if __name__ == '__main__': raise SystemExit(main())
