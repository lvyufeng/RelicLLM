"""Qualify the actual OpenAI HTTP adapter: warm reuse, disconnect cancellation, then fresh reset.

Runs only on a leased pure-TP8 CUDA host. The standard HTTP handler and standard backend worker
transport are used; diagnostics wrap the model loop but do not replace request/stream semantics.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import threading
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True);p.add_argument('--context-tokens',type=int,default=8192)
    p.add_argument('--max-new-tokens',type=int,default=128);p.add_argument('--port',type=int,default=63009)
    p.add_argument('--output-jsonl',type=Path,required=True)
    args=p.parse_args()
    import torch
    import torch.distributed as dist
    from bench_qwen4_exp_baseline import _inputs
    from relicllm.cli import build_parser,_args
    from relicllm.backends.qwen4_exp_backend import Qwen4ExpBackend
    from relicllm.models.qwen4_exp.runtime import generate
    from relicllm.models.qwen4_exp.verify_graph import _quantize_head_experts
    from relicllm.server.openai import RelicLLMHTTPServer,OpenAIHandler

    rank=int(os.environ['RANK'])
    parsed=build_parser().parse_args(['serve','--model',args.model,'--backend','qwen4_exp',
        '--tensor-parallel-size','8','--tensor-parallel-rank',str(rank),'--no-tensor-parallel-supervisor',
        '--device','cuda','--max-model-len','262144','--speculative-method','mtp','--speculative-tokens','3',
        '--backend-option','expert_resident=true','--backend-option','prefill_chunk=512'])
    backend=Qwen4ExpBackend(_args(parsed));backend.prepare()
    if backend._world!=8 or backend._ctx.pp_size!=1 or backend._device.type!='cuda':
        raise RuntimeError('this probe requires an explicitly leased TP8 CUDA host')
    model,ctx=backend._model,backend._ctx
    _quantize_head_experts(model.mtp.moe,ctx.device)
    tokenizer=backend._tokenizer
    text=tokenizer.decode(_inputs(tokenizer,args.context_tokens)[0].tolist(),skip_special_tokens=False)
    prompt_ids=[int(v) for v in tokenizer(text)['input_ids']]
    arena=len(prompt_ids)+args.max_new_tokens+10
    cache=model.make_cache(batch_size=1,max_seq_len=arena)
    reference,_plain=generate(model,ctx,torch.tensor([prompt_ids]),max_new_tokens=args.max_new_tokens,
                             chunk_size=512,graph_window=32,cache=cache,eos_token_ids=())
    del cache
    if len(reference)!=args.max_new_tokens:raise RuntimeError('the plain reference must be complete')
    stream=args.output_jsonl.open('x') if rank==0 else None
    records=[];condition=threading.Condition()
    def emit(record):
        if rank==0:
            line=json.dumps(record);print(line,flush=True);stream.write(line+'\n');stream.flush()
    emit({'kind':'qwen4_http_start','schema':'qwen4-http-graph-session-v1','input_tokens':len(prompt_ids),
          'reference_ids':reference,'session_enabled':True,'norm_fusion_enabled':False})
    original=backend._run_mtp
    def measured(*positional,**kwargs):
        delivered=[];callback=kwargs.get('on_token')
        if callback is not None:
            def observed(token):delivered.append(int(token));callback(token)
            kwargs['on_token']=observed
        output,stats=original(*positional,**kwargs)
        per_rank=[None]*ctx.world_size
        dist.all_gather_object(per_rank,{'rank':rank,'output_ids':output,'callback_ids':delivered})
        valid=(output==reference[:len(output)] and all(r['output_ids']==output
                    and r['callback_ids']==(output if r['rank']==0 else []) for r in per_rank))
        record={'kind':'qwen4_http_generation','request_index':len(records),'valid':valid,
                'output_ids':output,'stats':stats,'per_rank':per_rank,
                'pool_summary':backend._graph_session.summary() if backend._graph_session else None}
        with condition:
            records.append(record);condition.notify_all()
        emit(record)
        return output,stats
    backend._run_mtp=measured
    if rank!=0:
        try:backend.run_worker()
        finally:backend.close()
        dist.destroy_process_group();return 0

    server=RelicLLMHTTPServer(('127.0.0.1',args.port),OpenAIHandler,backend,'qwen38-nvfp4')
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    responses=[]
    payload={'model':'qwen38-nvfp4','prompt':text,'max_tokens':args.max_new_tokens,
             'temperature':0,'stream':True}
    try:
        for request_index in range(4):
            started=time.perf_counter();parts=[];disconnected=request_index==2
            request=urllib.request.Request(f'http://127.0.0.1:{args.port}/v1/completions',
                json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=600) as response:
                if response.status!=200:raise RuntimeError(f'HTTP status {response.status}')
                for raw in response:
                    line=raw.decode().strip()
                    if not line.startswith('data: '):continue
                    data=line[6:]
                    if data=='[DONE]':break
                    event=json.loads(data)
                    if 'error' in event:raise RuntimeError(str(event['error']))
                    piece=event.get('choices',[{}])[0].get('text','')
                    parts.append(piece)
                    if disconnected and piece:break
            if disconnected:
                with condition:
                    if not condition.wait_for(lambda:len(records)>request_index,timeout=120):
                        raise RuntimeError('the disconnected request did not unwind collectively')
            body=''.join(parts)
            valid=True if disconnected else body==backend._decode(reference)
            record={'kind':'qwen4_http_response','request_index':request_index,'valid':valid,
                    'intentional_disconnect':disconnected,'wall_s':time.perf_counter()-started,'text':body}
            responses.append(record);emit(record)
        valid=(len(records)==4 and all(r['valid'] for r in records) and all(r['valid'] for r in responses)
               and len(records[0]['output_ids'])==len(records[1]['output_ids'])==len(records[3]['output_ids'])==args.max_new_tokens
               and records[1]['stats']['graph_session_reused'] and records[1]['stats']['graph_build_s']==0
               and records[2]['stats']['stopped']=='cancel' and not records[2]['stats']['graph_session_retained']
               and not records[3]['stats']['graph_session_reused'])
        emit({'kind':'qwen4_http_summary','valid':valid,'requests':4,'worker_transport':'standard backend broadcast',
              'measurement_scope':'real local OpenAI SSE requests; no HTTP server deployment claim'})
    finally:
        server.shutdown();thread.join();server.server_close()
        backend.close();dist.barrier()
        emit({'kind':'qwen4_http_closed','active_mtp':backend._mtp_active,'backend_closed':backend._closed})
        stream.close();dist.destroy_process_group()
    return 0 if valid else 2


if __name__=='__main__':raise SystemExit(main())
