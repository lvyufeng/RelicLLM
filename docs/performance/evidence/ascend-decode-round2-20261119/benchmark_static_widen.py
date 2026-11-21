"""Original expert matrix: static FP4 widening bit identity and native Cube cost."""
import json
import os
from pathlib import Path
import statistics
import time

import torch
import torch_npu

from relic_core.kernels.static_fp4_decode import unpack_fp4_static,unpack_fp4_pairs
from relic_core.kernels.unpack import unpack_fp4_to_int8
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint


@torch.inference_mode()
def main():
    device=torch.device("npu:0")
    torch.npu.set_device(device)
    out=Path(os.environ["RELIC_PERF_OUTPUT"])
    all_bytes=torch.arange(256,dtype=torch.int64).to(torch.uint8).reshape(16,16)
    checkpoint=V41Checkpoint("<checkpoint>")
    raw=checkpoint.packed("layers.0.ffn.experts.0.w1.weight").view(torch.uint8).to(device)
    expected=unpack_fp4_to_int8(raw,backend="triton")
    tiny_expected=unpack_fp4_to_int8(all_bytes,backend="torch")
    rows=[]
    calls=[("old",1024,lambda:unpack_fp4_to_int8(raw,backend="triton"))]
    for tile in (512,1024,2048):
        tiny=unpack_fp4_static(all_bytes.to(device),backend="triton",tile=tile).cpu()
        print("STATIC_WIDEN_SMOKE",tile,flush=True)
        if not torch.equal(tiny,tiny_expected):raise RuntimeError("static NPU all-byte widened bits differ")
        calls.append(("static",tile,lambda tile=tile:unpack_fp4_static(raw,backend="triton",tile=tile)))
    for tile in (1024,2048,4096):
        tiny=unpack_fp4_pairs(all_bytes.to(device),backend="triton",tile=tile).cpu()
        if not torch.equal(tiny,tiny_expected):raise RuntimeError("packed-pair store byte identity failed")
        calls.append(("pair",tile,lambda tile=tile:unpack_fp4_pairs(raw,backend="triton",tile=tile)))
    for name,tile,call in calls:
        actual=call()
        torch.npu.synchronize(device)
        if not torch.equal(actual.cpu(),expected.cpu()):raise RuntimeError("original expert widened bytes differ")
        for _ in range(3):call()
        torch.npu.synchronize(device)
        samples=[]
        for _ in range(12):
            start=time.perf_counter()
            call()
            torch.npu.synchronize(device)
            samples.append(time.perf_counter()-start)
        row={"name":name,"tile":tile,"median_seconds":statistics.median(samples),"seconds":samples,
             "original_expert_value_bits_equal":True,"all_256_codes_verified":True,"output_bytes":actual.numel(),
             "whole_model_TPS_claimed":False}
        rows.append(row)
        print(json.dumps(row),flush=True)
        with (out/"partial-static-widen-benchmark.json").open("w") as handle:json.dump(rows,handle,indent=2)
    with (out/"original-static-widen-benchmark.json").open("x") as handle:json.dump(rows,handle,indent=2)
    del raw,expected,actual
    checkpoint.close()
    return 0


if __name__=="__main__":raise SystemExit(main())
