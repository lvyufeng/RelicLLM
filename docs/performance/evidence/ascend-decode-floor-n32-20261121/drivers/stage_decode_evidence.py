"""Stage a new immutable evidence bundle; previous dated bundles never change.

Only compact model audits are copied. Complete FP32 tensors and raw CANN traces
stay on the leased host with verified path/size/SHA256 identities. Source versions
captured by each bounded model runner remain distinguishable from final guards.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path('/data1/relic-v41-perf-20261118-0938')
LOGS = ROOT/'logs'
DEST = LOGS/'decoder-evidence-20261121-v1'
PREFIXES = (
 'original-decode-real-output-floor-tp2-', 'build-original-direct-bit-consumers-',
 'original-direct-bit-consumer-probes-', 'original-direct-bits-TP2-replay-',
 'original-direct-AIV-TP2-replay-', 'original-direct-byte-',
 'original-HC-FP32-coefficient-probe-', 'original-HC-coefficient-',
 'original-global-FFN-', 'compare-original-direct-byte-models-',
 'compare-original-full-FFN-HC-models-', 'current-profile-tp2-',
 'analyze-current-TP2-timeline-', 'build-prepared-AIV-decode-consumers-',
 'original-prepared-AIV-', 'build-original-N32-consumer-',
 'original-N32-', 'compare-final-original-consumers-'
)


def copy(source, relative):
    if not source.is_file():
        raise RuntimeError(f'the required evidence source is not a file: {source}')
    target=DEST/relative
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as stream:stream.write(source.read_bytes())


def main():
    if ROOT.resolve()!=ROOT or DEST.exists():
        raise RuntimeError('new evidence staging root must be exact, private and not already present')
    DEST.mkdir(exist_ok=False)
    receipts=[]
    for root in sorted(LOGS.iterdir()):
        if not root.is_dir() or '20261121T' not in root.name or not root.name.startswith(PREFIXES):
            continue
        summary=root/'summary.json'
        if not summary.is_file():
            continue
        row=json.loads(summary.read_bytes())
        if not row['cleanup']['complete']:
            raise RuntimeError('all retained jobs must have reconciled complete owned cleanup')
        receipts.append({'directory':str(root),'returncode':row['returncode'],'cleanup_complete':True})
        for file in sorted(root.iterdir()):
            if not file.is_file():continue
            if file.name.startswith('Cube-model-rank-') or file.suffix=='.pt':continue
            if file.suffix not in ('.json','.log','.py','.txt'):continue
            if file.name.startswith(('decode-scopes-rank-','profile-scope-rank-')):continue
            copy(file,Path('trials')/root.name/file.name)
        if root.name.startswith(('original-direct-byte-n32-generated-', 'original-direct-byte-old-generated-steps64-', 'original-global-FFN-generated-tp2-steps64-')):
            snapshots=root/'source-snapshots'
            for file in sorted(snapshots.rglob('*')):
                if file.is_file():copy(file,Path('timed-source-snapshots')/root.name/file.relative_to(snapshots))
    if not receipts:
        raise RuntimeError('evidence selection did not find any of this lease\'s jobs')
    metadata_files=('direct-bit-consumer-build-20261121-v1.json','prepared-decode-build-20261121-v1.json',
                    'prepared-n32-build-20261121-v1.json','prepared-n32-build-20261121-v2.json')
    for name in metadata_files:copy(LOGS/name,Path('build-metadata')/name)
    # Keep the new current-profile summaries and full raw artifact identities.
    raw=json.loads((LOGS/'current-TP2-timeline-20261121.json').read_bytes())
    compact={key:value for key,value in raw.items() if key not in ('event_scopes','CANN_kernels')}
    compact['event_scopes']=[{key:value for key,value in row.items() if key!='raw_scopes'} for row in raw['event_scopes']]
    compact['CANN_kernels']=[{key:value for key,value in row.items() if key not in ('collective_attributions_include_waiting_and_profiler_inflation','CPU_trace_host_scopes')}
                           for row in raw['CANN_kernels']]
    for target,original in zip(compact['CANN_kernels'],raw['CANN_kernels']):
        comm=original['collective_attributions_include_waiting_and_profiler_inflation']
        target['collective_calls_per_forward']=sum(row['calls_per_forward'] for row in comm)
        target['collective_duration_sum_ms_per_forward_includes_waiting']=sum(row['duration_ms_per_forward'] for row in comm)
        target['CPU_trace_host_scopes']=original['CPU_trace_host_scopes']
    identities={}
    profile=Path(raw['directory'])
    for pattern in ('profiler-rank*/**/kernel_details.csv','profiler-rank*/**/trace_view.json', 'decode-scopes-rank-*.json','profile-scope-rank-*.json'):
        for path in sorted(profile.glob(pattern)):
            identities[str(path)]={'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    identities[str(LOGS/'current-TP2-timeline-20261121.json')]={'bytes':(LOGS/'current-TP2-timeline-20261121.json').stat().st_size,
        'sha256':hashlib.sha256((LOGS/'current-TP2-timeline-20261121.json').read_bytes()).hexdigest()}
    compact['raw_CANN_and_scope_artifact_identities']=identities
    compact['compact_summary_not_a_copy_of_complete_raw_CANN_traces']=True
    with (DEST/'current-TP2-timeline-summary.json').open('x') as stream:json.dump(compact,stream,indent=2)
    scripts=('profile_decode_floor.py','run_decode_floor.py','probe_direct_bit_decoders.py', 'build_direct_bit_consumers.py',
        'probe_direct_tp_moe.py','direct_decoder_factory.py','profile_direct_decoder_model.py','run_direct_decoder_model.py',
        'probe_hc_coefficients.py','profile_hc_coefficient_model.py','run_hc_coefficient_model.py','probe_global_ffn_slots.py',
        'profile_global_ffn_model.py','run_global_ffn_model.py','compare_direct_decoder_models.py','compare_ffn_hc_models.py',
        'profile_current_tp2.py','run_current_tp2_profile.py','analyze_current_tp2_profile.py','profile_current_bottlenecks.py',
        'analyze_current_bottleneck_profile.py','build_prepared_decode_consumers.py','probe_prepared_decode_consumers.py',
        'probe_prepared_tp_moe.py','build_n32_consumer.py','probe_n32_consumer.py','compare_final_consumer_models.py',
        'bounded_task.py','run_bounded_perf_command.py','perf_round2_safety.py','stage_decode_evidence.py')
    for name in scripts:copy(ROOT/'source'/name,Path('drivers')/name)
    for key in ('prepared-n32-build-20261121-v1.json','prepared-n32-build-20261121-v2.json','prepared-decode-build-20261121-v1.json'):
        build=json.loads((LOGS/key).read_bytes())
        for name,row in build['variants'].items():
            directory=Path(row['source_snapshot'])
            for file in sorted(directory.iterdir()):
                if file.is_file():copy(file,Path('native-build-snapshots')/key/name/file.name)
    with (DEST/'selection.json').open('x') as stream:
        json.dump({'selected_owned_receipts':receipts,'previous_frozen_bundles_not_modified':True,
          'full_quality_tensors_remain_remote_and_are_verified_by_complete_audits':True,
          'raw_profiles_remain_remote_with_hashes':True,'target_50_TPS_claimed':False},stream,indent=2)
    paths=[path for path in DEST.rglob('*') if path.is_file()]
    print(json.dumps({'staged_root':str(DEST),'files':len(paths),'bytes':sum(path.stat().st_size for path in paths)}),flush=True)


if __name__=='__main__':main()
