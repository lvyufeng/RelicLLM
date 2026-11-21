"""Eight-rank real expert/collective codec proof for global per-expert FFN slots."""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch_npu

from relic_core.kernels.global_ffn_slots import prepare_global_ffn_slots, finish_global_ffn_slots
from relicllm.models.deepseek_v4_1.ascend_graphs import _record,npu_graph_api

OUTPUT=Path(os.environ['RELIC_PERF_OUTPUT'])


@torch.inference_mode()
def main():
    rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.npu.set_device(rank);torch.set_num_threads(4)
    dist.init_process_group('hccl',timeout=dt.timedelta(seconds=120))
    device=torch.device(f'npu:{rank}')
    dim=5120;topk=6
    down=torch.empty(topk,dim,dtype=torch.float32,device=device)
    selected=torch.empty(topk,dtype=torch.int32,device=device)
    ids=torch.empty(1,topk,dtype=torch.int64,device=device)
    shared=torch.empty(1,dim,dtype=torch.bfloat16,device=device)
    slots=torch.empty(topk+1,dim,dtype=torch.float32,device=device)
    route_error=torch.zeros(1,dtype=torch.int32,device=device)
    state={}
    def body():
        prepare_global_ffn_slots(down,selected,ids,shared,slots)
        dist.all_reduce(slots)
        state['out']=finish_global_ffn_slots(slots,route_error)
    api=npu_graph_api();pool=api.graph_pool_handle()
    down.zero_();selected.fill_(-1);ids.copy_(torch.arange(topk,device=device).reshape(1,-1));shared.zero_()
    graph=_record(body,pool,1,api)
    captured_output=state['out']
    generator=torch.Generator().manual_seed(410)
    records=[]
    try:
        for replay in range(32):
            picks=torch.randperm(32,generator=generator)[:topk]
            logical=torch.zeros(topk+1,dim)
            cpu=torch.zeros(topk,dim)
            owned=sorted([int(expert) for expert in picks if int(expert)%4==rank//2])
            selected.copy_(torch.tensor(owned+[-1]*(topk-len(owned)),dtype=torch.int32,device=device))
            ids.copy_(picks.reshape(1,-1).to(device))
            # Exactly representable FP32 dyadics; cancellation cases prove
            # this collective does not BF16-narrow each rank's down partial.
            for slot,expert in enumerate(owned):
                cpu[slot].fill_(1.00390625+expert/16 if rank%2==0 else -1.0)
            down.copy_(cpu.to(device));shared.fill_(rank/8)
            graph.replay();torch.npu.synchronize()
            for slot,expert in enumerate(sorted(picks.tolist())):
                logical[slot].fill_(.00390625+expert/16)
            logical[-1].fill_(sum(r/8 for r in range(world)))
            expected=finish_global_ffn_slots(logical,torch.zeros(1,dtype=torch.int32),backend='torch')
            actual=captured_output.cpu()
            if not torch.equal(actual.view(torch.int32),expected.view(torch.int32)):
                raise RuntimeError('global FP32 slot alignment or post-reduction narrowing failed')
            body();torch.npu.synchronize()
            if not torch.equal(actual.view(torch.int32),state['out'].cpu().view(torch.int32)):
                raise RuntimeError('global FFN eager/capture values differ')
            records.append({'replay':replay,'selected_global_expert_ids':picks.tolist(),'FP32_bit_differences':0})
        row={'rank':rank,'world':world,'shape':[7,5120],'changing_replays':32,
             'global_FP32_collective_before_per_expert_BF16_narrowing':True,'eager_graph_FP32_bits_equal':True,
             'all_global_expert_and_shared_slots_not_combined_early':True,'records':records,'not_model_TPS':True}
        with (OUTPUT/f'global-FFN-slots-rank-{rank}.json').open('x') as stream:json.dump(row,stream,indent=2)
    finally:
        dist.destroy_process_group()


if __name__=='__main__':main()
