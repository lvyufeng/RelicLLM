"""Original layer-zero routed experts: exact M1 scheduling microbenchmark."""
from __future__ import annotations

from functools import partial
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa:F401

from relic_core.kernels import unpack,vector_fp4_decode
from relicllm.models.deepseek_v4_1.ascend_packed_experts import AscendPackedRoutedExperts
from relicllm.models.deepseek_v4_1.ascend_m1_routing import M1RoutedDispatch
from relicllm.models.deepseek_v4_1.cube_experiment import CubePackedProjection
from relicllm.models.deepseek_v4_1.fused_activation_binding import FusedActivationBinding
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint


@torch.inference_mode()
def main():
    device=torch.device("npu:0")
    torch.npu.set_device(device)
    output=Path(os.environ["RELIC_PERF_OUTPUT"])
    checkpoint=V41Checkpoint("/data1/relic-v41-20261116-210/checkpoint")
    original_widen=unpack.unpack_fp4_to_int8
    unpack.unpack_fp4_to_int8=partial(vector_fp4_decode.unpack_fp4_vector,mode="arithmetic",tile=2048)
    store=None
    try:
        store=AscendPackedRoutedExperts(checkpoint,0,n_experts=384,dim=5120,inter_dim=2304,topk=6,
            swiglu_limit=8.0,device=device,dtype=torch.bfloat16,rank=0,world=8,backend="triton",
            preload=True,packed_gemv=CubePackedProjection(mode="weight_quant",cache_budget_bytes=0,backend="triton"))
        x=torch.randn(1,5120,generator=torch.Generator().manual_seed(1671)).bfloat16().to(device)
        weights=torch.tensor([[.1,.2,.3,.05,.25,.1]],dtype=torch.float32,device=device)
        indices=torch.tensor([[0,8,16,7,9,23]],dtype=torch.int64,device=device)
        records=[]
        with FusedActivationBinding():
            reference=store.forward(x,weights,indices).cpu().contiguous()
            for name in ("original_index_copy_dispatch","M1_direct_row_dispatch"):
                manager=None
                if name=="M1_direct_row_dispatch":
                    manager=M1RoutedDispatch([store]).install()
                try:
                    result=store.forward(x,weights,indices).cpu().contiguous()
                    differences=int((reference.view(torch.int32)!=result.view(torch.int32)).sum())
                    if differences:
                        raise RuntimeError(f"exact M1 routed scheduling changed {differences} FP32 result bits")
                    for _ in range(3):
                        store.forward(x,weights,indices)
                    torch.npu.synchronize(device)
                    samples=[]
                    for _ in range(12):
                        start=time.perf_counter()
                        store.forward(x,weights,indices)
                        torch.npu.synchronize(device)
                        samples.append(time.perf_counter()-start)
                    record={"name":name,"original_layer":0,"owned_experts_in_probe":[0,8,16],
                        "batch_size":1,"original_FP32_result_bit_differences":differences,
                        "median_seconds":statistics.median(samples),"seconds":samples,
                        "native_Cube_accumulation_same_in_both_modes":True,
                        "original_FP8_activation_contract_unchanged":True,"whole_model_TPS_claimed":False,
                        "binding":None if manager is None else manager.describe()}
                    records.append(record)
                    print(json.dumps(record),flush=True)
                finally:
                    if manager is not None:
                        manager.close()
            with (output/"original-M1-routing-benchmark.json").open("x") as handle:
                json.dump(records,handle,indent=2)
    finally:
        unpack.unpack_fp4_to_int8=original_widen
        if store is not None:
            store.release()
        checkpoint.close()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
