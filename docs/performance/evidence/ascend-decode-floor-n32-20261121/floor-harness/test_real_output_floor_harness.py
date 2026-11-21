"""CPU lifecycle fixtures for the private real-output floor probe; no NPU timing claim."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import types
import time

import pytest
import torch

from relicllm.models.deepseek_v4_1 import graphs as actual_graphs

ROOT = Path(__file__).resolve().parent


class Replay:
    def __init__(self, body):
        self.body = body
    def replay(self):
        self.body()


class Front:
    def __init__(self):
        self.hashes = torch.zeros(1, 128, dtype=torch.int64)
    def snapshot_prefix(self, limit):
        return self.hashes[:, :limit].clone()
    def restore_prefix(self, value):
        self.hashes.copy_(value)


def fixture(tmp_path):
    blocks = []
    for index in range(40):
        sink = SimpleNamespace(ffn_in=None, residual=None, y_in=None, x_out=None,
            weights=None, indices=None, floor_full_ffn_actual=None, ffn_post=None, ffn_comb=None)
        block = SimpleNamespace(hc_post=lambda value, residual, post, comb: (value + residual).bfloat16())
        layer = SimpleNamespace(sink=sink, ready=True, block=block, pool=None)
        def routed(this, index=index):
            actual_graphs._put(this.sink, 'y_in', this.sink.ffn_in.view(1, 4).float() * (index + 1) / 100)
        layer._routed = types.MethodType(routed, layer)
        def body(layer=layer):
            value = (layer.sink.y_in + 1).bfloat16().view(1, 1, 4)
            actual_graphs._put(layer.sink, 'floor_full_ffn_actual', value)
            actual_graphs._put(layer.sink, 'x_out', layer.block.hc_post(value, layer.sink.residual, None, None))
        layer.graph_b = Replay(body)
        blocks.append(SimpleNamespace(decode_graph=layer))
    model = SimpleNamespace(layers=blocks, max_seq_len=128, cache=torch.zeros(1))
    front = Front()
    def forward(tokens, pos):
        value = torch.full((1, 1, 4), int(tokens.item()), dtype=torch.bfloat16)
        for index, block in enumerate(model.layers):
            layer = block.decode_graph
            for name, tensor in (('ffn_in', value), ('residual', value), ('weights', torch.ones(1, 2)),
                                 ('indices', torch.tensor([[index, index + 1]]))):
                actual_graphs._put(layer.sink, name, tensor)
            layer._routed(); layer.graph_b.replay()
            value = layer.sink.x_out
        model.cache.add_(1)
        front.hashes[:, pos] = int(tokens.item())
        return None, value.view(1, 4).float(), None
    tree = ast.parse((ROOT / 'profile_decode_floor.py').read_bytes())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_real_output_floor')
    namespace = dict(torch=torch, dist=SimpleNamespace(barrier=lambda:None,get_rank=lambda:0),
        probe=SimpleNamespace(MATCHED_INPUTS=list(range(16)),sync=lambda:None),
        graphs=SimpleNamespace(_put=actual_graphs._put,
            snapshot=lambda model: {'cache':model.cache.clone()},
            restore=lambda model,cache:model.cache.copy_(cache['cache'])),
        ascend_graphs=SimpleNamespace(npu_graph_api=lambda:None,
            _record=lambda body,pool,warmup,api:Replay(body)), types=types, time=time,json=json, OUTPUT=tmp_path)
    exec(compile(ast.Module(body=[function],type_ignores=[]),'floor-fixture','exec'),namespace)
    return namespace['run_real_output_floor'], model, front, forward


def test_recorded_real_outputs_preserve_logits_routes_cache_and_owners(tmp_path):
    operation, model, front, forward = fixture(tmp_path)
    originals = [(block.decode_graph._routed, block.decode_graph.graph_b) for block in model.layers]
    cache, hashes = {'cache':model.cache.clone()}, front.snapshot_prefix(128)
    report = operation(model, front, forward, cache, hashes, [0] * 16)
    for row in report['reports'].values():
        assert row['full_model_FP32_logits_bit_equal_to_real_baseline']
        assert row['final_KV_compression_and_hash_bytes_equal']
        assert row['all_40_layer_conditioning_and_routes_verified']
        assert row['not_generation_TPS_or_achieved_model_performance']
    assert all(block.decode_graph._routed == routed and block.decode_graph.graph_b is graph
               for block, (routed, graph) in zip(model.layers, originals))
    assert not model.cache.count_nonzero() and not front.hashes.count_nonzero()


def test_changed_baseline_output_is_refused_and_prefix_rewound(tmp_path):
    operation, model, front, forward = fixture(tmp_path)
    with pytest.raises(RuntimeError,match='baseline continuation'):
        operation(model,front,forward,{'cache':model.cache.clone()},front.snapshot_prefix(128),[1]*16)
    assert not model.cache.count_nonzero() and not front.hashes.count_nonzero()


def test_floor_sink_adds_owned_slot_without_dropping_original_slots():
    tree = ast.parse((ROOT / 'profile_decode_floor.py').read_bytes())
    main = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
    cls = next(node for node in main.body if isinstance(node,ast.ClassDef) and node.name=='FloorSink')
    namespace={'previous_sink':actual_graphs.Sink}
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'floor-sink-fixture','exec'),namespace)
    sink=namespace['FloorSink']()
    assert all(getattr(sink,name) is None for name in actual_graphs.Sink.__slots__)
    value=actual_graphs._put(sink,'floor_full_ffn_actual',torch.ones(1,4))
    actual_graphs._put(sink,'floor_full_ffn_actual',torch.full((1,4),2.0))
    assert sink.floor_full_ffn_actual is value and torch.equal(value,torch.full((1,4),2.0))
