"""Same original-checkpoint TP2 decode with explicit direct-bit/AIV consumers.

The optional isolated complete-FFN pass retains real model inputs, router,
shared experts and both collectives; its timings are not generation TPS.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import textwrap
import time

import torch
import torch.distributed as dist

import profile_actual_cube as probe
import profile_expert_tp as tp_probe
from relicllm.models.deepseek_v4_1 import ascend_graphs, graphs

OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])


@torch.inference_mode()
def complete_ffn_probe(model, owner):
    records = []
    api = ascend_graphs.npu_graph_api()
    for block in model.layers:
        source = block.decode_graph.sink.ffn_in.view(1, block.ffn.dim).clone()
        expected = block.decode_graph.sink.observed_full_ffn.clone()
        state = {}
        ffn = block.ffn
        def body(ffn=ffn, source=source):
            weights, indices = ffn.gate(source)
            routed = owner._execute(ffn.routed, source, weights, indices)
            shared = ffn.shared_experts(source)
            state["output"] = ffn.tp.reduce(routed + shared).type_as(source).view(1, 1, ffn.dim)
        pool = api.graph_pool_handle()
        graph = ascend_graphs._record(body, pool, 1, api)
        graph.replay(); torch.npu.synchronize()
        differences = int((state["output"].view(torch.int16) != expected.view(torch.int16)).sum().item())
        if differences:
            raise RuntimeError(f"isolated complete FFN changed real layer{block.ffn.layer_id} BF16 output: {differences}")
        for _ in range(3):
            graph.replay()
        dist.barrier(); torch.npu.synchronize()
        samples = []
        for _ in range(16):
            began = time.perf_counter()
            graph.replay(); torch.npu.synchronize()
            samples.append(1000 * (time.perf_counter() - began))
        records.append({"layer": block.ffn.layer_id, "complete_FFN_mean_ms": sum(samples) / len(samples),
                        "samples_ms": samples, "router_shared_expert_TP_down_and_global_reductions_included": True,
                        "same_real_full_FFN_BF16_bit_differences": 0,
                        "context": int(os.environ.get("RELIC_PERF_TIMED_STEPS", "16")) + 23,
                        "timing_not_model_TPS_or_additive_critical_path": True})
        del graph, pool, state
    with (OUTPUT / f"complete-FFN-rank-{dist.get_rank()}.json").open("x") as stream:
        json.dump({"rank": dist.get_rank(), "consumer_mode": os.environ["RELIC_DIRECT_CONSUMER_MODE"],
                   "records": records, "all_40_real_layer_inputs_used": True,
                   "isolated_synchronized_graph_timings_not_model_TPS": True}, stream, indent=2)


def main():
    previous_sink = graphs.Sink
    class ObservedSink(previous_sink):
        __slots__ = ("observed_full_ffn",)
        def __init__(self):
            for name in (*previous_sink.__slots__, *self.__slots__):
                setattr(self, name, None)
    graphs.Sink = ObservedSink
    body_source = textwrap.dedent(inspect.getsource(graphs.LayerGraphs._body_b))
    marker = '    _put(sink, "x_out", block.hc_post(y, sink.residual, sink.ffn_post, sink.ffn_comb))\n'
    if body_source.count(marker) != 1:
        raise RuntimeError("the real shared/routed FFN store boundary moved")
    body_source = body_source.replace(marker, '    _put(sink, "observed_full_ffn", y)\n' + marker, 1)
    namespace = dict(graphs.__dict__)
    exec(compile(body_source, "complete-FFN-observer", "exec"), namespace)
    previous_b = ascend_graphs.NpuLayerGraphs._body_b
    previous_placement = tp_probe.placement.main
    ascend_graphs.NpuLayerGraphs._body_b = namespace["_body_b"]
    source = inspect.getsource(tp_probe.main)
    source = source.replace('from relic_core.kernels.ascendc_online_tile_cube import AscendCTileCubeConsumer',
                            'from direct_decoder_factory import make_consumer as AscendCTileCubeConsumer', 1)
    exec(compile(source, str(Path(__file__).resolve()), "exec"), tp_probe.__dict__)
    def placement_hook():
        candidate = inspect.getsource(probe.generate_probe)
        save = '        torch.save(final, OUTPUT/f"Cube-final-logits-rank-{rank}.pt")\n'
        if candidate.count(save) != 1:
            raise RuntimeError("the untimed complete FFN probe insertion point moved")
        candidate = candidate.replace(save, save +
            '        complete_ffn_probe(model, expert_tp_binding)\n', 1)
        receipt = '        row["expert_TP_EP"] = None if expert_tp_binding is None else expert_tp_binding.describe()\n'
        if candidate.count(receipt) != 1:
            raise RuntimeError("the private TP consumer receipt moved")
        candidate = candidate.replace(receipt, receipt +
            '        row["direct_original_consumer"] = expert_tp_binding.consumer.describe()\n', 1)
        probe.complete_ffn_probe = complete_ffn_probe
        generated = OUTPUT / f"direct-byte-generated-rank-{os.environ['LOCAL_RANK']}.py"
        with generated.open("x") as stream:
            stream.write(candidate)
        exec(compile(candidate, str(generated), "exec"), probe.__dict__)
        return previous_placement()
    tp_probe.placement.main = placement_hook
    try:
        return tp_probe.main()
    finally:
        ascend_graphs.NpuLayerGraphs._body_b = previous_b
        graphs.Sink = previous_sink
        tp_probe.placement.main = previous_placement


if __name__ == "__main__":
    raise SystemExit(main())
