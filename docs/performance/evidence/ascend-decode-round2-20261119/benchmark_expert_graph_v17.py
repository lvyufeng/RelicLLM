"""Original resident experts: eager versus expert-body graphs with actual pool accounting."""
from __future__ import annotations

from functools import partial
import json
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch
import torch_npu  # noqa:F401

from relic_core.kernels import unpack, vector_fp4_decode
from relicllm.models.deepseek_v4_1.ascend_packed_experts import AscendPackedRoutedExperts
from relicllm.models.deepseek_v4_1.ascend_expert_graphs import RoutedExpertGraphs
from relicllm.models.deepseek_v4_1.ascend_m1_routing import M1RoutedDispatch
from relicllm.models.deepseek_v4_1.cube_experiment import CubePackedProjection
from relicllm.models.deepseek_v4_1.fused_activation_binding import FusedActivationBinding
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint


@torch.inference_mode()
def main():
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    output = Path(os.environ["RELIC_PERF_OUTPUT"])
    checkpoint = V41Checkpoint("/data1/relic-v41-20261116-210/checkpoint")
    original_widen = unpack.unpack_fp4_to_int8
    unpack.unpack_fp4_to_int8 = partial(vector_fp4_decode.unpack_fp4_vector, mode="arithmetic", tile=2048)
    store = round_ = None
    try:
        operator = CubePackedProjection(mode="weight_quant", cache_budget_bytes=0, backend="triton")
        store = AscendPackedRoutedExperts(checkpoint, 0, n_experts=384, dim=5120, inter_dim=2304,
            topk=6, swiglu_limit=8.0, device=device, dtype=torch.bfloat16, rank=0, world=8,
            backend="triton", preload=True, packed_gemv=operator)
        model = SimpleNamespace(layers=[SimpleNamespace(layer_id=0, ffn=SimpleNamespace(routed=store))])
        x = torch.randn(1,5120,generator=torch.Generator().manual_seed(1681)).bfloat16().to(device)
        weights = torch.tensor([[.1,.2,.3,.05,.25,.1]],dtype=torch.float32,device=device)
        ids = torch.tensor([[0,8,16,7,9,23]],dtype=torch.int64,device=device)
        records = []
        with FusedActivationBinding(), M1RoutedDispatch([store]):
            controls = []
            for step in range(3):
                controls.append(store.forward((x.float()+step*.03125).bfloat16(),weights*(step+1),ids).cpu().contiguous())
            for mode in ("eager_original_expert_body", "captured_original_expert_body"):
                if mode == "captured_original_expert_body":
                    round_ = RoutedExpertGraphs(model,warmup=1,max_graphs=64,max_pool_bytes=512*2**20).install()
                    capture = round_.capture_all()
                    with (output / "actual-expert-graph-capture.json").open("x") as handle:
                        json.dump(capture, handle, indent=2)
                for step, expected in enumerate(controls):
                    actual = store.forward((x.float()+step*.03125).bfloat16(),weights*(step+1),ids).cpu().contiguous()
                    differences = int((actual.view(torch.int32)!=expected.view(torch.int32)).sum())
                    if differences or not bool(torch.isfinite(actual).all()):
                        raise RuntimeError(f"expert graph changed {differences} FP32 bits at input/route test {step}")
                for _ in range(3):
                    store.forward(x,weights,ids)
                torch.npu.synchronize(device)
                samples = []
                for _ in range(12):
                    start = time.perf_counter()
                    store.forward(x,weights,ids)
                    torch.npu.synchronize(device)
                    samples.append(time.perf_counter()-start)
                record = {"mode":mode,"source_layer":0,"recorded_owned_experts":len(store.owned),
                          "active_owned_experts":[0,8,16],"input_and_route_tests":3,
                          "FP32_result_bit_differences":0,"median_seconds":statistics.median(samples),
                          "seconds":samples,"graph_state":None if round_ is None else round_.describe(),
                          "no_I8_operand_cache":True,"original_activation_quantization_unchanged":True,
                          "whole_model_TPS_claimed":False}
                records.append(record)
                print(json.dumps(record),flush=True)
            with (output / "original-expert-body-graph-benchmark.json").open("x") as handle:
                json.dump(records,handle,indent=2)
    finally:
        if round_ is not None:
            round_.release()
        unpack.unpack_fp4_to_int8 = original_widen
        if store is not None:
            store.release()
        checkpoint.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
