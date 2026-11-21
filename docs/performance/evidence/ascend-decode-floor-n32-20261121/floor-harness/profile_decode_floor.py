"""Real-output replay ablation: an implementation latency floor, never generation TPS.

Captured routed and complete FFN values come from real fixed-input model forwards.
Attention, HC, Engram and head stay live; every KV/hash state is restored between
passes. Neither a zero expert result nor fabricated routing is substituted.
"""
from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import textwrap
import time
import types

import torch
import torch.distributed as dist

import profile_actual_cube as probe
import profile_expert_tp as tp_probe
from relicllm.models.deepseek_v4_1 import graphs, ascend_graphs

OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])


@torch.inference_mode()
def run_real_output_floor(model, front, forward, cache, hashes, baseline_tokens):
    inputs = probe.MATCHED_INPUTS
    if len(inputs) != 16 or len(model.layers) != 40:
        raise RuntimeError("the fixed-output floor must cover all 40 layers and 16 input tokens")
    layer_graphs = [block.decode_graph for block in model.layers]
    if any(layer is None or not layer.ready for layer in layer_graphs):
        raise RuntimeError("capture the real model before the floor ablation")
    before = []
    frames = []
    current = {"step": 0}
    def rewind():
        graphs.restore(model, cache)
        front.restore_prefix(hashes)
        probe.sync()
    def one(step):
        current["step"] = step
        _, logits, _ = forward(torch.tensor([[inputs[step]]]), 24 + step)
        probe.sync()
        return logits[0]
    try:
        rewind()
        for step in range(16):
            logits = one(step)
            token = int(logits.argmax().item())
            if token != baseline_tokens[step]:
                raise RuntimeError("recording pass changed the actual baseline continuation outputs")
            frames.append({"logits": logits.detach().float().clone(), "layers": [{
                "routed": layer.sink.y_in.clone(),
                "full_ffn": layer.sink.floor_full_ffn_actual.clone(),
                "ffn_input": layer.sink.ffn_in.clone(),
                "indices": layer.sink.indices.clone(),
                "weights": layer.sink.weights.clone(),
            } for layer in layer_graphs]})
        baseline_end_cache = graphs.snapshot(model)
        baseline_end_hashes = front.snapshot_prefix(model.max_seq_len)
        # Record one minimal complete-FFN replacement B graph per layer. Its
        # input address stays live; only recorded real output bytes are copied.
        for index, layer in enumerate(layer_graphs):
            staging = torch.empty_like(frames[0]["layers"][index]["full_ffn"])
            staging.copy_(frames[0]["layers"][index]["full_ffn"])
            before.append((layer, layer._routed, layer.graph_b, staging))
            def body(layer=layer, staging=staging):
                sink, block = layer.sink, layer.block
                graphs._put(sink, "x_out", block.hc_post(staging, sink.residual, sink.ffn_post, sink.ffn_comb))
            floor_graph = ascend_graphs._record(body, layer.pool, 1, ascend_graphs.npu_graph_api())
            before[-1] += (floor_graph,)
        reports = {}
        for mode in ("routed_output_replay", "complete_FFN_output_replay"):
            for index, (layer, original_routed, original_b, staging, floor_b) in enumerate(before):
                def replay(this, index=index, staging=staging, mode=mode):
                    frame = frames[current["step"]]["layers"][index]
                    if mode == "routed_output_replay":
                        graphs._put(this.sink, "y_in", frame["routed"])
                    else:
                        staging.copy_(frame["full_ffn"])
                layer._routed = types.MethodType(replay, layer)
                layer.graph_b = original_b if mode == "routed_output_replay" else floor_b
            # Verify every intermediate conditioning input and route outside
            # the timed window, plus full model output bits and final KV/hash.
            rewind()
            for step in range(16):
                value = one(step)
                if not torch.equal(value.float().view(torch.int32), frames[step]["logits"].view(torch.int32)):
                    raise RuntimeError(f"{mode} changed full-model logits at input{step}")
                for index, layer in enumerate(layer_graphs):
                    frame = frames[step]["layers"][index]
                    for name, tensor in (("ffn_input", layer.sink.ffn_in), ("indices", layer.sink.indices), ("weights", layer.sink.weights)):
                        if not torch.equal(tensor.view(torch.int8), frame[name].view(torch.int8)):
                            raise RuntimeError(f"{mode} changed layer{index} original {name}")
            final_cache = graphs.snapshot(model)
            if final_cache.keys() != baseline_end_cache.keys() or any(
                not torch.equal(final_cache[name].view(torch.int8), baseline_end_cache[name].view(torch.int8))
                for name in final_cache):
                raise RuntimeError("real-output ablation changed original KV/compression state bytes")
            final_hashes = front.snapshot_prefix(model.max_seq_len)
            if (final_hashes is None) != (baseline_end_hashes is None) or (
                    final_hashes is not None and not torch.equal(final_hashes, baseline_end_hashes)):
                raise RuntimeError("real-output ablation changed original host hash state")
            rewind()
            dist.barrier(); probe.sync()
            samples = []
            generated = []
            start = time.perf_counter()
            for step in range(16):
                began = time.perf_counter()
                value = one(step)
                generated.append(int(value.argmax().item()))
                samples.append(time.perf_counter() - began)
            wall = time.perf_counter() - start
            if generated != baseline_tokens:
                raise RuntimeError("timed floor no longer reproduces the real baseline outputs")
            reports[mode] = {"rank": dist.get_rank(), "timed_steps": 16, "context": [24, 40],
                "wall_seconds": wall, "ms_per_forward": wall * 1000 / 16,
                "timed_step_seconds": samples, "actual_output_token_ids": generated,
                "all_40_layer_conditioning_and_routes_verified": True,
                "full_model_FP32_logits_bit_equal_to_real_baseline": True,
                "final_KV_compression_and_hash_bytes_equal": True,
                "real_output_D2D_staging_included": True,
                "not_generation_TPS_or_achieved_model_performance": True,
                "retained_shared_expert_and_global_MoE_reduce": mode == "routed_output_replay"}
        row = {"rank": dist.get_rank(), "original_byte_TP2_EP4_baseline": True,
               "scope": "real-output ablation on the same 16 conditioning inputs, not production inference",
               "zero_expert_outputs_not_used": True, "attention_HC_Engram_head_remain_live": True,
               "reports": reports,
               "stored_frame_bytes": sum(t.numel() * t.element_size() for frame in frames for layer in frame["layers"] for t in layer.values())}
        with (OUTPUT / f"non-expert-floor-rank-{dist.get_rank()}.json").open("x") as stream:
            json.dump(row, stream, indent=2)
        if dist.get_rank() == 0:
            print(json.dumps({"non_expert_floor": reports}), flush=True)
        return row
    finally:
        probe.sync()
        for layer, original_routed, original_b, _, _ in before:
            layer._routed = original_routed
            layer.graph_b = original_b
        rewind()


def main():
    # Add an observable complete-FFN sink while preserving the original B body.
    source = textwrap.dedent(inspect.getsource(graphs.LayerGraphs._body_b))
    marker = '    _put(sink, "x_out", block.hc_post(y, sink.residual, sink.ffn_post, sink.ffn_comb))\n'
    if source.count(marker) != 1:
        raise RuntimeError("the original full-FFN store boundary moved")
    source = source.replace(marker, '    _put(sink, "floor_full_ffn_actual", y)\n' + marker, 1)
    namespace = dict(graphs.__dict__)
    exec(compile(source, "real-FFN-output-sink", "exec"), namespace)
    previous_sink = graphs.Sink
    class FloorSink(previous_sink):
        __slots__ = ("floor_full_ffn_actual",)
        def __init__(self):
            for name in (*previous_sink.__slots__, *self.__slots__):
                setattr(self, name, None)
    graphs.Sink = FloorSink
    previous_body = getattr(ascend_graphs.NpuLayerGraphs, "_body_b")
    previous_placement = tp_probe.placement.main
    ascend_graphs.NpuLayerGraphs._body_b = namespace["_body_b"]
    def install_hook():
        candidate = inspect.getsource(probe.generate_probe)
        snapshot_marker = '        start_wall = time.perf_counter()\n'
        if candidate.count(snapshot_marker) != 1:
            raise RuntimeError("the baseline wall timing boundary moved")
        candidate = candidate.replace(snapshot_marker,
            '        floor_cache = snapshot(model)\n'
            '        floor_hashes = front.snapshot_prefix(model.max_seq_len)\n' + snapshot_marker, 1)
        diagnostic_marker = '        torch.save(final, OUTPUT/f"Cube-final-logits-rank-{rank}.pt")\n'

        if candidate.count(diagnostic_marker) != 1:
            raise RuntimeError("the full-model final quality save boundary moved")
        candidate = candidate.replace(diagnostic_marker, diagnostic_marker +
            '        real_output_floor = run_real_output_floor(model, front, candidate_forward, floor_cache, floor_hashes, timed_tokens)\n', 1)
        probe.run_real_output_floor = run_real_output_floor
        generated = OUTPUT / f"floor-generated-rank-{os.environ['LOCAL_RANK']}.py"
        with generated.open("x") as stream:
            stream.write(candidate)
        exec(compile(candidate, str(generated), "exec"), probe.__dict__)
        return previous_placement()
    tp_probe.placement.main = install_hook
    try:
        return tp_probe.main()
    finally:
        ascend_graphs.NpuLayerGraphs._body_b = previous_body
        graphs.Sink = previous_sink
        tp_probe.placement.main = previous_placement


if __name__ == "__main__":
    raise SystemExit(main())
