"""Eight-card actual-expert TP2/EP4 capture, reduction and residency probe.

Original layer0 experts0..7 and real 5120/2304 geometry. Exact eager-vs-captured
TP2 outputs are mandatory; differences versus original complete-expert EP8 are
measured as the separately authorized native accumulation experiment.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch_npu

from relic_core.kernels.ascendc_online_tile_cube import AscendCTileCubeConsumer
from relic_core.kernels.ascendc_tile_moe import make_tile_moe_workspace, device_tile_fp4_moe
from relicllm.models.deepseek_v4_1.ascend_expert_tp import ExpertTPDispatch
from relicllm.models.deepseek_v4_1.ascend_tile_major_experts import AscendTileMajorRoutedExperts
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938")
OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])
LIBRARY = ROOT / "cache/online-tile-cube-k128-v2/lib/librelic_online_tile_cube.so"
SHA = "0cd9111e373859c5b8d4ec09089a952467f26b6c88a3fb387842b11c0725b287"


@torch.inference_mode()
def main():
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(rank)
    torch.set_num_threads(4)
    dist.init_process_group("hccl", timeout=dt.timedelta(seconds=180))
    checkpoint = V41Checkpoint("<checkpoint>", device="cpu")
    operator = AscendCTileCubeConsumer(LIBRARY, sha256=SHA)
    original = AscendTileMajorRoutedExperts(checkpoint, 0, n_experts=8, dim=5120, inter_dim=2304, topk=6,
        rank=rank, world=world, device=f"npu:{rank}", dtype=torch.bfloat16, backend="triton", preload=True)
    old_bytes = original.resident_bytes
    old_banks = {name: (arena.codes, arena.scales) for name, arena in original._arenas.items()}
    old_workspace = make_tile_moe_workspace(5120, 2304, 6, device=f"npu:{rank}", group_k=4)
    model = SimpleNamespace(layers=[SimpleNamespace(ffn=SimpleNamespace(routed=original))])
    manager = ExpertTPDispatch(model, consumer=operator, tp_world=2, capture_graphs=True,
                               restore_residency=False, reserve_bytes=0)
    generator = torch.Generator().manual_seed(0x2E41)
    x = torch.empty(1, 5120, dtype=torch.bfloat16, device=f"npu:{rank}")
    weights = torch.tensor([[0.1, 0.2, 0.125, 0.175, 0.15, 0.25]], dtype=torch.float32, device=x.device)
    indices = torch.empty(1, 6, dtype=torch.int64, device=x.device)
    records = []
    try:
        manager.install()
        if manager.describe()["resident_bytes"] != old_bytes:
            raise RuntimeError("TP2/EP4 did not preserve per-rank packed resident byte count")
        shard = model.layers[0].ffn.routed
        for step in range(32):
            host_x = torch.randn(1, 5120, generator=generator).bfloat16()
            host_ids = torch.randperm(8, generator=generator)[:6].reshape(1, 6)
            if step % 8 == 0:
                host_x.zero_()
            elif step % 8 == 1:
                host_x.mul_(4)
            x.copy_(host_x.to(x.device)); indices.copy_(host_ids.to(x.device))
            captured = shard.forward(x, weights, indices)
            eager = manager._execute(shard, x, weights, indices)
            torch.npu.synchronize()
            differences = int((captured.view(torch.int32) != eager.view(torch.int32)).sum().item())
            if differences or not bool(torch.isfinite(captured).all().item()):
                raise RuntimeError(f"TP2 graph/eager parity failed step{step}, {differences} FP32 values")
            if rank % 2 and bool(torch.count_nonzero(captured).item()):
                raise RuntimeError("a nonleader emitted duplicate EP contribution")
            tp_complete = captured.clone()
            dist.all_reduce(tp_complete)
            ep_complete = device_tile_fp4_moe(x, weights, indices, old_banks,
                rank=rank, world=world, n_experts=8, workspace=old_workspace,
                consumer=operator, certified_bf16_operands=True, backend="triton")
            dist.all_reduce(ep_complete)
            torch.npu.synchronize()
            if not bool(torch.isfinite(ep_complete).all().item()):
                raise RuntimeError("original EP8 real-expert reference is not finite")
            delta = tp_complete - ep_complete
            wire = tp_complete.view(torch.int8).flatten()
            peers = torch.empty(wire.numel() * world, dtype=torch.int8, device=x.device)
            dist.all_gather_into_tensor(peers, wire)
            torch.npu.synchronize()
            if not torch.equal(peers.reshape(world, -1), wire.expand(world, -1)):
                raise RuntimeError("global TP/EP output differs across physical ranks")
            records.append({"step": step, "expert_ids": host_ids.tolist()[0], "captured_vs_eager_TP_FP32_bit_differences": 0,
                            "TP_vs_EP8_FP32_bit_differences": int((tp_complete.view(torch.int32) != ep_complete.view(torch.int32)).sum().item()),
                            "TP_vs_EP8_max_abs": float(delta.abs().max().item()),
                            "TP_vs_EP8_relative_L2": float((delta.norm() / ep_complete.norm().clamp_min(1e-30)).item()),
                            "all_physical_rank_outputs_equal": True})
        row = {"rank": rank, "world": world, "real_original_expert_layer": 0, "expert_count": 8,
               "real_geometry": [5120, 2304, 6], "changed_input_graph_replays": 32,
               "eager_vs_captured_TP_FP32_bits_equal": True, "all_physical_rank_global_outputs_equal": True,
               "native_TP_partial_accumulation_is_explicit_changed_reduction_experiment": True,
               "original_FP4_E8M0_and_activation_quantization_unchanged": True,
               "FP32_TP_down_reduce_precedes_original_BF16_per_expert_narrowing": True,
               "not_model_TPS": True, "original_resident_bytes_per_rank": old_bytes,
               "controller": manager.describe(), "records": records}
        with (OUTPUT / f"expert-TP2-EP4-rank-{rank}.json").open("x") as stream:
            json.dump(row, stream, indent=2)
        if rank == 0:
            print(json.dumps({"controller": row["controller"], "replays": 32,
                              "TP_vs_EP8_max_relative_L2": max(item["TP_vs_EP8_relative_L2"] for item in records)}), flush=True)
    finally:
        manager.close(restore_residency=False)
        checkpoint.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
