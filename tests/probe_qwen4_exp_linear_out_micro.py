"""Device-level verification for cat-free M=1 writes, isolated from model throughput claims."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(Path(__file__).resolve().parent))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--observation',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--samples',type=int,default=7);p.add_argument('--mutations',type=int,default=8)
    p.add_argument('--target-ms',type=float,default=60);args=p.parse_args()
    import torch
    import torch.nn.functional as F
    from relicllm.models.qwen4_exp.linear_out import qwen4_exp_linear_out
    from probe_qwen4_exp_linear_hotspots import micro_shapes
    os.environ['POCKETLLM_QWEN4_LINEAR_OUT']='1';os.environ['POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT']='1'
    torch.cuda.set_device(0);stream=args.output.open('x')
    def emit(row):stream.write(json.dumps(row)+'\n');stream.flush();print(json.dumps(row),flush=True)
    if args.observation:
        shapes=[s for s in micro_shapes(json.loads(args.observation.read_text()))
                if 2<=s['rows']<=8 and s['weight_dtype'].removeprefix('torch.')=='bfloat16']
    else:
        shapes=[{'rows':4,'in_features':k,'out_features':n,'x_ndim':ndim,'weight_dtype':'torch.bfloat16'}
                for ndim in [2,3] for k,n in [(2560,768),(2560,1024),(2560,2560),(2560,80),(80,2560),(10240,320),(10240,4)]]
    shapes=list({(s['rows'],s['in_features'],s['out_features'],s['x_ndim']):s for s in shapes}.values())
    emit({'kind':'linear_out_micro_start','schema':'linear-out-micro-v1','shapes':len(shapes),
          'torch':torch.__version__,'cuda':torch.version.cuda,
          'two_dimensional_rows':os.environ.get('POCKETLLM_QWEN4_LINEAR_OUT_2D','0'),
          'scope':'old M=1 linear+cat vs explicit ATen out; not model throughput'})
    try:
        for index,s in enumerate(shapes):
            rows,k,n,ndim=s['rows'],s['in_features'],s['out_features'],s['x_ndim']
            generator=torch.Generator(device='cuda').manual_seed(202611190+index)
            x=torch.randn(rows,k,dtype=torch.bfloat16,device='cuda',generator=generator).reshape((1,)*(ndim-2)+(rows,k))
            weight=torch.randn(n,k,dtype=torch.bfloat16,device='cuda',generator=generator)
            def old():
                flat=x.reshape(rows,k);one_shape=(1,)*(ndim-1)+(k,)
                return torch.cat([F.linear(flat[i:i+1].reshape(one_shape).contiguous(),weight).reshape(1,-1)
                                  for i in range(rows)]).reshape(*x.shape[:-1],n)
            def new():return qwen4_exp_linear_out(x,weight)
            different=[]
            for mutation in range(args.mutations):
                x.copy_(torch.randn(x.shape,dtype=x.dtype,device=x.device,generator=generator))
                if mutation%3==1:x.mul_(0.001)
                if mutation%3==2:x[...,::2].neg_()
                a,b=old(),new()
                if b is None:raise RuntimeError('explicit out helper did not accept its promised shape')
                different.append(int((a.view(torch.int16)!=b.view(torch.int16)).sum()))
            graphs={};outputs={}
            for mode,func in [('old',old),('out',new)]:
                func();torch.cuda.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):outputs[mode]=func()
                graphs[mode]=graph
            for _ in range(12):graphs['old'].replay();graphs['out'].replay()
            torch.cuda.synchronize()
            def timed(graph,count):
                first=torch.cuda.Event(enable_timing=True);last=torch.cuda.Event(enable_timing=True)
                first.record()
                for _ in range(count):graph.replay()
                last.record();last.synchronize();return first.elapsed_time(last)/count
            pilot=timed(graphs['old'],8);count=max(4,min(2048,int(args.target_ms/max(pilot,1e-6))))
            times={'old':[],'out':[]}
            for sample in range(args.samples):
                for mode in (['old','out'] if sample%2==0 else ['out','old']):times[mode].append(timed(graphs[mode],count))
            x.mul_(0.7).add_(0.01);graphs['old'].replay();graphs['out'].replay();torch.cuda.synchronize()
            captured=int((outputs['old'].view(torch.int16)!=outputs['out'].view(torch.int16)).sum())
            median={k:statistics.median(v) for k,v in times.items()}
            emit({'kind':'linear_out_micro_result','shape':s,'valid':max(different+[captured])==0,
                  'mutation_different_elements':different,'captured_mutation_different_elements':captured,
                  'ms_samples':times,'median_ms':median,'time_ratio':median['out']/median['old'],
                  'weight_bytes':weight.numel()*weight.element_size(),'replays_per_sample':count,
                  'scope':'one shape repeat; same-matrix cache residency, not measured HBM traffic or full-model speed'})
            del graphs,outputs,graph,x,weight,a,b;torch.cuda.empty_cache()
    finally:stream.close()
    return 0


if __name__=='__main__':raise SystemExit(main())
