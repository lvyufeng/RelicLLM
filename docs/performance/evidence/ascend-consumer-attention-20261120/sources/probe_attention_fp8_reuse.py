"""Actual-checkpoint paired attention projections with exact original FP8 reuse."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch
import torch_npu

from relic_core.kernels import block_fp8_quant, unpack
from relic_core.kernels.fast_fused_fp8_activation_decode import quantize_decode_fp8_fast
from relicllm.models.deepseek_v4_1.ascend_attention_reuse import AttentionFP8Reuse
from relicllm.models.deepseek_v4_1.fused_activation_binding import FusedActivationBinding
from relicllm.models.deepseek_v4_1.ascend_graphs import _record, npu_graph_api
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint

OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])


@torch.inference_mode()
def main():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    device = torch.device("npu:0")
    checkpoint = V41Checkpoint("/data1/relic-v41-20261116-210/checkpoint", device="cpu")
    keys = [f"layers.0.attn.{name}.weight" for name in ("wq_a", "wkv")]
    weights = [checkpoint.weight(key, dtype=torch.bfloat16, device="cpu").t().contiguous().to(device) for key in keys]
    k = weights[0].shape[0]
    if weights[1].shape[0] != k:
        raise RuntimeError("the real Q/KV pair does not use a shared original activation geometry")
    checkpoint.close()
    x = torch.empty((1, k), dtype=torch.bfloat16, device=device)
    generator = torch.Generator().manual_seed(0xA641)
    x.copy_(torch.randn(1, k, generator=generator).bfloat16().to(device))
    outputs = {}
    def projections(name):
        values = []
        for weight in weights:
            codes, scales = block_fp8_quant.quantize_block_fp8(x.view(1, k), backend="triton")
            decoded = unpack.unpack_fp8_to_bf16(codes, scales, block_shape=(1, 32), backend="triton")
            values.append(decoded @ weight)
        outputs[name] = tuple(values)
        return outputs[name]
    api = npu_graph_api()
    with FusedActivationBinding(operator=quantize_decode_fp8_fast) as fused:
        pool = api.graph_pool_handle()
        base = _record(lambda: projections("base"), pool, 2, api)
        with AttentionFP8Reuse(certify_immutable_inputs=True) as reuse:
            def body():
                with reuse.scope():
                    return projections("reuse")
            candidate = _record(body, pool, 2, api)
            stored_candidate = outputs["reuse"]
            stored_base = outputs["base"]
            for case in range(32):
                source = torch.randn(1, k, generator=generator).bfloat16()
                if case % 8 == 0:
                    source.zero_()
                elif case % 8 == 1:
                    source.mul_(128)
                x.copy_(source.to(device))
                base.replay(); candidate.replay(); torch.npu.synchronize()
                for old, new in zip(stored_base, stored_candidate):
                    if not torch.equal(old.view(torch.int16), new.view(torch.int16)):
                        raise RuntimeError("call-scoped original FP8 preparation reuse changed native projection bits")
                if reuse.prepared is not None:
                    raise RuntimeError("a recording scope escaped its owner")
            samples = {"base": [], "reuse": []}
            for repeat in range(16):
                for name in (("base", "reuse") if repeat % 2 else ("reuse", "base")):
                    begin = time.perf_counter()
                    (base if name == "base" else candidate).replay()
                    torch.npu.synchronize()
                    samples[name].append(time.perf_counter() - begin)
            report = {"original_checkpoint_projection_keys": keys, "k": k,
                      "weight_shapes_KN": [list(weight.shape) for weight in weights],
                      "changed_input_graph_replays": 32, "BF16_projection_bit_differences": 0,
                      "original_E4M3_E8M0_quantization_unchanged": True,
                      "reuse": reuse.describe(), "fused": fused.describe(),
                      "paired_synchronized_seconds": samples,
                      "paired_mean_ms": {name: 1000 * sum(value) / len(value) for name, value in samples.items()},
                      "only_projection_pair_not_model_TPS": True}
            if reuse.stats["quantizer_calls"] != reuse.stats["reuses"] or reuse.stats["reuses"] == 0:
                raise RuntimeError("the candidate did not actually reuse one preparation per paired projection")
            with (OUTPUT / "attention-FP8-reuse-probe.json").open("x") as stream:
                json.dump(report, stream, indent=2)
            print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
