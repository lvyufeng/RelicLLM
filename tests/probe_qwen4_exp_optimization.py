"""Full-output optimization A/B and warm graph ownership proof on a leased TP8 host.

Rates do not include prefill/model load; setup and cleanup remain separately reported. Traces are
never mixed into throughput evidence. All requests keep exactly the same target model/precision.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))


def _install_additive_op(base, path, name):
    """Install exactly one experimental binding on the qualified module, isolating every other entry.

    The shared object is loaded a second time under a hash-qualified alias, so ``name`` is the only
    attribute that changes: the qualified module keeps every other attribute by identity, and a
    divergence can therefore only come from ``name``. Returns the identity record and a closer that
    removes the attribute again.
    """
    resolved = Path(os.fspath(path)).expanduser().resolve()
    if not resolved.is_file():
        raise RuntimeError(f'the requested additive build is not a file: {resolved}')
    namespace = '_qwen4_additive_' + hashlib.sha256(str(resolved).encode()).hexdigest()[:16]
    package = sys.modules.get(namespace)
    if package is None:
        package = types.ModuleType(namespace)
        package.__path__ = []
        sys.modules[namespace] = package
    full = f'{namespace}.cuda_kernel'
    spec = importlib.util.spec_from_file_location(full, str(resolved))
    if spec is None or spec.loader is None:
        raise RuntimeError(f'{resolved} is not a loadable CPython extension')
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    operator = getattr(module, name, None)
    if operator is None:
        raise RuntimeError(f'the additive build does not export {name}')
    if name in vars(base):
        raise RuntimeError(f'the qualified module already exports {name}; refusing to shadow it')
    setattr(base, name, operator)
    identity = {'schema': 'qwen4-additive-op-v1', 'path': str(resolved),
                'sha256': hashlib.sha256(resolved.read_bytes()).hexdigest(),
                'installed_ops': [name], 'replaced_ops': []}

    def close():
        if name in vars(base):
            delattr(base, name)

    return identity, close


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',required=True)
    parser.add_argument('--context-tokens',type=int,default=8192)
    parser.add_argument('--max-new-tokens',type=int,default=128)
    parser.add_argument('--chunk-size',type=int,default=512)
    parser.add_argument('--draft-tokens',type=int,default=3)
    parser.add_argument('--variants',default='off,on,off')
    parser.add_argument('--switch',choices=['gated_norm','nvfp4_compact','nvfp4_store','linear_out','combined','hc_up_mix'],default='gated_norm')
    parser.add_argument('--warm-session',action='store_true')
    parser.add_argument('--same-prompt',action='store_true')
    parser.add_argument('--check-real-norm',action='store_true')
    parser.add_argument('--reuse-prefill',action='store_true')
    parser.add_argument('--reference-jsonl',type=Path)
    parser.add_argument('--nvfp4-overlay-so',type=Path)
    parser.add_argument('--hc-exact-overlay-so',type=Path)
    parser.add_argument('--output-jsonl',type=Path,required=True)
    args=parser.parse_args()
    variants=args.variants.split(',')
    if any(value not in ('off','on') for value in variants):parser.error('variants must be off/on')
    if args.warm_session and len(set(variants))!=1:parser.error('warm reuse must keep the captured flag unchanged')
    if args.reuse_prefill and args.warm_session:
        parser.error('a reused prefill diagnostic currently uses one fixed prompt, not the two-prompt warm reset test')
    if os.environ.get('POCKETLLM_QWEN4_ROUND_TRACE'):parser.error('tracing is not throughput proof')
    import torch
    if args.switch in ('linear_out','combined'):
        if 'out' not in torch.ops.aten.linear.overloads():
            raise RuntimeError('the requested linear-out arm needs the ATen linear.out overload')
        # The higher-rank out overload was rejected by the device oracle. A model A/B only
        # exercises the independently qualified two-dimensional M=1 staging, even when unset.
        os.environ['POCKETLLM_QWEN4_LINEAR_OUT_2D']='1'
    import torch.distributed as dist
    from transformers import AutoTokenizer
    from bench_qwen4_exp_baseline import _inputs
    from relicllm.models.qwen4_exp.runtime import (init_distributed,load_model,generate,
                                                  generate_mtp_draft_lookahead)
    from relicllm.models.qwen4_exp.graph_session import GraphSessionPool
    from relic_core.kernels.cuda_loader import load_cuda_kernel

    ctx=init_distributed()
    if ctx.world_size!=8 or ctx.pp_size!=1 or ctx.device.type!='cuda':
        raise RuntimeError('only an explicitly leased pure-TP8 CUDA host is qualified')
    overlay_handle=None
    overlay_env_before=os.environ.get('POCKETLLM_QWEN4_NVFP4_OVERLAY_SHA256')
    additive_identity=None
    additive_close=None
    try:
        if args.nvfp4_overlay_so is not None:
            from relicllm.models.qwen4_exp.nvfp4_overlay import install_nvfp4_overlay
            overlay_handle=install_nvfp4_overlay(load_cuda_kernel(),args.nvfp4_overlay_so)
            os.environ['POCKETLLM_QWEN4_NVFP4_OVERLAY_SHA256']=overlay_handle.identity['overlay_sha256']
        if args.hc_exact_overlay_so is not None:
            additive_identity,additive_close=_install_additive_op(load_cuda_kernel(),args.hc_exact_overlay_so,
                                                                  'qwen4_exp_hc_up_mix_exact')
        stream=args.output_jsonl.open('x') if ctx.rank==0 else None
        def emit(value):
            if ctx.rank==0:
                line=json.dumps(value);print(line,flush=True);stream.write(line+'\n');stream.flush()
        os.environ['POCKETLLM_QWEN4_GATED_NORM_EXACT']='0'
        os.environ['POCKETLLM_QWEN4_NVFP4_COMPACT']='0'
        os.environ['POCKETLLM_QWEN4_NVFP4_STORE']='0'
        os.environ['POCKETLLM_QWEN4_LINEAR_OUT']='0'
        os.environ['POCKETLLM_QWEN4_HC_UP_MIX_EXACT']='0'
        switch_key={'gated_norm':'POCKETLLM_QWEN4_GATED_NORM_EXACT',
                    'nvfp4_compact':'POCKETLLM_QWEN4_NVFP4_COMPACT',
                    'nvfp4_store':'POCKETLLM_QWEN4_NVFP4_STORE',
                    'combined':'POCKETLLM_QWEN4_NVFP4_STORE',
                    'linear_out':'POCKETLLM_QWEN4_LINEAR_OUT',
                    'hc_up_mix':'POCKETLLM_QWEN4_HC_UP_MIX_EXACT'}[args.switch]
        os.environ['POCKETLLM_QWEN4_MTP_NVFP4']='0'
        model,config=load_model(args.model,ctx,expert_resident=True,expert_fused_fp8=True,graph_window=32,mtp=True)
        # Quantize the private proposal head before every reference/arm. Otherwise the first arm
        # prefills an FP8 head and later arms prefill its one-way NVFP4 replacement, confounding the
        # norm A/B with a different proposal schedule near sensitive target-token ties.
        from relicllm.models.qwen4_exp.verify_graph import _quantize_head_experts
        _quantize_head_experts(model.mtp.moe, ctx.device)
        native=int(config.text_config.max_position_embeddings)
        if args.context_tokens+args.max_new_tokens>native:raise ValueError('request exceeds native total context')
        tokenizer=AutoTokenizer.from_pretrained(args.model,trust_remote_code=True,local_files_only=True)
        primary=_inputs(tokenizer,args.context_tokens)
        prompts=[primary]
        if args.warm_session and not args.same_prompt:
            secondary=primary.clone()
            # A different middle-of-prefix token proves state reset; the same geometry remains reusable.
            index=max(1,args.context_tokens//2)
            secondary[0,index]=primary[0,index-1]
            if torch.equal(primary,secondary):raise RuntimeError('secondary prompt must differ')
            prompts.append(secondary)
        extension=load_cuda_kernel()
        required_entry={'gated_norm':'qwen4_exp_gated_rms_norm_exact',
                        'nvfp4_compact':'nvfp4_moe_rows_forward_exact_fast_compact',
                        'nvfp4_store':'nvfp4_moe_rows_forward_exact_fast_store',
                        'combined':'nvfp4_moe_rows_forward_exact_fast_store',
                        'linear_out':'nvfp4_moe_rows_forward_exact_fast',
                        'hc_up_mix':'qwen4_exp_hc_up_mix_exact'}[args.switch]
        if extension is None or not hasattr(extension,required_entry):
            raise RuntimeError('the explicitly measured additive native entry must be loaded for every arm')
        norm_oracle={'calls':0,'different_elements':0}
        if args.check_real_norm:
            from relicllm.models.qwen4_exp import layers
            native_norm=layers._cuda_gated_rms_norm_exact
            def checked_norm(x,gate,weight,eps,activation):
                result=native_norm(x,gate,weight,eps,activation)
                if result is None or torch.cuda.is_current_stream_capturing():
                    return result
                h=x.float()
                h=h*torch.rsqrt(h.pow(2).mean(-1,keepdim=True)+eps)
                h=weight*h.to(x.dtype)
                activated=torch.sigmoid(gate.float()) if activation=='sigmoid' else torch.nn.functional.silu(gate.float())
                expected=(h*activated).to(x.dtype)
                norm_oracle['calls']+=1
                norm_oracle['different_elements']+=int((result.view(torch.int16)!=expected.view(torch.int16)).sum())
                return result
            layers._cuda_gated_rms_norm_exact=checked_norm
        source_names=['runtime.py','layers.py','model.py','moe.py','verify_graph.py','graph_session.py','linear_out.py','nvfp4_overlay.py']
        fingerprints={'kernel_sha256':hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
                      'native_ops_overlay':overlay_handle.identity if overlay_handle else None,
                      'additive_ops':additive_identity,
                      'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'reference_sha256':hashlib.sha256(args.reference_jsonl.read_bytes()).hexdigest()
                          if args.reference_jsonl else None,
                      'sources':{name:hashlib.sha256((ROOT/'relicllm/models/qwen4_exp'/name).read_bytes()).hexdigest()
                                 for name in source_names
                                 if (ROOT/'relicllm/models/qwen4_exp'/name).is_file()}}
        emit({'kind':'qwen4_optimization_start','schema':'qwen4-opt-v1','native_max_context':native,
              'input_tokens':args.context_tokens,'output_tokens':args.max_new_tokens,'variants':variants,
              'warm_session':args.warm_session,'switch':args.switch,'switch_key':switch_key,
              'fingerprints':fingerprints,'trace_enabled':False,
              'math_environment':{key:value for key,value in sorted(os.environ.items())
                                  if key.startswith('POCKETLLM_')}})
        references=[]
        for prompt_index,prompt in enumerate(prompts):
            if args.reference_jsonl and prompt_index==0:
                records=[json.loads(line) for line in args.reference_jsonl.read_text().splitlines()]
                valid=next(r for r in records if r.get('kind')=='qwen4_exp_mtp_arm_result'
                           and r.get('arm')=='plain' and r.get('valid'))
                reference=valid['output_ids']
                if len(reference)!=args.max_new_tokens:raise ValueError('reference length differs')
                starts=[r for r in records if r.get('kind')=='qwen4_exp_mtp_start']
                token_hash=hashlib.sha256(json.dumps(prompt.reshape(-1).tolist(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
                if starts and starts[0].get('input_sha256')!=token_hash:raise ValueError('reference prompt differs')
            else:
                callbacks=[]
                arena=min(args.context_tokens+args.max_new_tokens+2*args.draft_tokens+4,native)
                plain_cache=model.make_cache(batch_size=1,max_seq_len=arena)
                reference,plain_stats=generate(model,ctx,prompt,max_new_tokens=args.max_new_tokens,
                    chunk_size=args.chunk_size,graph_window=32,eos_token_ids=(),on_token=callbacks.append,
                    cache=plain_cache)
                del plain_cache
                if len(reference)!=args.max_new_tokens:raise RuntimeError('plain reference incomplete')
                emit({'kind':'qwen4_optimization_plain','prompt_index':prompt_index,'output_ids':reference,
                      'stats':plain_stats,'callback_equal':callbacks==reference})
            references.append(reference)
        os.environ['POCKETLLM_QWEN4_MTP_NVFP4']='1'
        prefix_bundles={}
        prefix_freeze_s={}
        def save_prefix(state,prompt_index):
            from relicllm.models.qwen4_exp.pd_state import freeze_state
            prompt=prompts[prompt_index]
            token_hash=hashlib.sha256(prompt.cpu().contiguous().numpy().tobytes()).hexdigest()
            started=time.perf_counter()
            prefix_bundles[prompt_index]=freeze_state(cache=state['cache'],head_cache=state['head_cache'],
                past=state['past'],next_token=state['next_token'],first_draft=state['first_draft'],
                chain_stack=state['chain_stack'],first_conf=state['first_conf'],prefill_s=state['prefill_s'],
                tp_rank=ctx.rank,tp_world_size=ctx.world_size,native_max_context=native,
                metadata={'input_sha256':token_hash})
            prefix_freeze_s[prompt_index]=time.perf_counter()-started
        pool=GraphSessionPool(model=model,capacity=1) if args.warm_session else None
        records=[]
        try:
            for trial,variant in enumerate(variants):
                os.environ[switch_key]='1' if variant=='on' else '0'
                if args.switch=='combined':
                    os.environ['POCKETLLM_QWEN4_LINEAR_OUT']='1' if variant=='on' else '0'
                prompt_index=trial%len(prompts);prompt=prompts[prompt_index]
                callbacks=[];start=time.perf_counter()
                output,stats=generate_mtp_draft_lookahead(model,ctx,prompt,
                    max_new_tokens=args.max_new_tokens,chunk_size=args.chunk_size,draft_tokens=args.draft_tokens,
                    verify_graph=True,eos_token_ids=(),on_token=callbacks.append,graph_session=pool,
                    prefilled_state=prefix_bundles.get(prompt_index) if args.reuse_prefill else None,
                    on_prefill=(lambda state: save_prefix(state,prompt_index))
                        if args.reuse_prefill and prompt_index not in prefix_bundles else None)
                wall=time.perf_counter()-start
                ranks=[None]*ctx.world_size
                dist.all_gather_object(ranks,{'rank':ctx.rank,'output_ids':output,'callback_ids':callbacks,
                                             'norm_oracle':dict(norm_oracle)})
                complete=(len(output)==args.max_new_tokens and output==references[prompt_index]
                          and ranks[0]['callback_ids']==output and all(r['output_ids']==output
                              and r['callback_ids']==(output if r['rank']==0 else []) for r in ranks))
                if args.check_real_norm:
                    complete=complete and all(r['norm_oracle']['different_elements']==0 for r in ranks)
                reused=bool(stats.get('graph_session_reused'))
                if args.warm_session and trial>0:
                    complete=complete and reused and stats.get('graph_build_s')==0
                record={'kind':'qwen4_optimization_result','trial':trial,'variant':variant,
                        'prompt_index':prompt_index,'switch':args.switch,'valid':complete,
                        'output_ids':output,'all_ranks':ranks,
                        'math_environment':{key:value for key,value in sorted(os.environ.items())
                                            if key.startswith('POCKETLLM_')},
                        'input_tokens':args.context_tokens,'requested_output_tokens':args.max_new_tokens,
                        'logical_total_tokens':args.context_tokens+args.max_new_tokens,'wall_s':wall,
                        'stats':stats,'pool_summary':pool.summary() if pool else None,
                        'prefix_reused':bool(args.reuse_prefill and trial>0),
                        'prefix_freeze_s':prefix_freeze_s.get(prompt_index,0.0),
                        'measurement_scope':('real-activation norm diagnostic; not throughput evidence'
                            if args.check_real_norm else 'full-output untraced A/B; setup and cleanup separately reported')}
                emit(record);records.append(record)
                if not complete:return 2
        finally:
            closed=time.perf_counter()
            if pool is not None:pool.close()
            emit({'kind':'qwen4_optimization_session_close','close_s':time.perf_counter()-closed,
                  'pool_summary':pool.summary() if pool else None})
            if stream:stream.close()
        return 0
    finally:
        try:
            if additive_close is not None or overlay_handle is not None:
                torch.cuda.synchronize(ctx.device)
        finally:
            if additive_close is not None:
                additive_close()
            if overlay_handle is not None:
                overlay_handle.close()
                if overlay_env_before is None:
                    os.environ.pop('POCKETLLM_QWEN4_NVFP4_OVERLAY_SHA256',None)
                else:
                    os.environ['POCKETLLM_QWEN4_NVFP4_OVERLAY_SHA256']=overlay_env_before



if __name__=='__main__':raise SystemExit(main())
