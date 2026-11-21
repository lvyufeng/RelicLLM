"""Measure launch-grid striding for exact original FP4 paired-byte decoding."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu  # noqa:F401

from relic_core.kernels.static_fp4_decode import unpack_fp4_pairs
from relic_core.kernels.unpack import unpack_fp4_to_int8
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint


@torch.inference_mode()
def main():
    device=torch.device("npu:0")
    torch.npu.set_device(device)
    out=Path(os.environ["RELIC_PERF_OUTPUT"])
    checkpoint=V41Checkpoint("/data1/relic-v41-20261116-210/checkpoint")
    raw=checkpoint.packed("layers.0.ffn.experts.0.w1.weight").view(torch.uint8).to(device)
    expected=unpack_fp4_to_int8(raw,backend="triton").cpu()
    records=[]
    for tile in (2048,4096,8192):
        for programs in (None,32,64,128):
            call=lambda:unpack_fp4_pairs(raw,backend="triton",tile=tile,programs=programs)
            result=call()
            torch.npu.synchronize(device)
            if not torch.equal(result.cpu(),expected):raise RuntimeError("strided paired decode changed original bytes")
            for _ in range(3):call()
            torch.npu.synchronize(device)
            samples=[]
            for _ in range(12):
                start=time.perf_counter()
                call()
                torch.npu.synchronize(device)
                samples.append(time.perf_counter()-start)
            record={"tile":tile,"programs":programs,"median_seconds":statistics.median(samples),"seconds":samples,
                    "original_weight_bit_identity":True,"whole_model_TPS_claimed":False}
            records.append(record)
            print(json.dumps(record),flush=True)
            with (out/"partial-pair-grid-benchmark.json").open("w") as handle:json.dump(records,handle,indent=2)
    with (out/"pair-grid-benchmark.json").open("x") as handle:json.dump(records,handle,indent=2)
    del raw,result,expected
    checkpoint.close()
    return 0


if __name__=="__main__":raise SystemExit(main())
