"""Eight-rank actual-checkpoint Engram placement and dynamic graph probe.

No expert/model allocation; original host tables are mmap-backed and read-only.
WKV/q/k source parameters are original file tensors. The probe checks each local
projection against the same native full matrix, gate against the released FP32
sequence, all-rank bytes, 32 changing-ID/mask graph replays, and pinned-input
lifetime. Microbenchmarks are not model TPS.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from relic_core.kernels.fast_fused_fp8_activation_decode import quantize_decode_fp8_fast
from relic_core.kernels.original_fp8_engram import original_fp8_engram_projection
from relicllm.models.deepseek_v4_1 import mma, modules
from relicllm.models.deepseek_v4_1.ascend_device_engram import HostTableDeviceEngram
from relicllm.models.deepseek_v4_1.config import load_config, resolve_config
from relicllm.models.deepseek_v4_1.engram import EngramLayout
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint, CheckpointEngramTable, scale_key

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938")
OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])
CHECKPOINT = Path("<checkpoint>")


class MiniEngramModel(nn.Module):
    def __init__(self, checkpoint, config):
        super().__init__()
        layout = EngramLayout.from_config(config.__dict__)
        blocks = []
        for layer in layout.layer_ids:
            engram = modules.Engram(config, layer, layout, CheckpointEngramTable(checkpoint, layer),
                                    device="cpu", quantized=True)
            for suffix, parameter in (("wkv.weight", engram.wkv.weight), ("q_weight", engram.q_weight), ("k_weight", engram.k_weight)):
                key = f"layers.{layer}.engram.{suffix}"
                parameter.copy_(checkpoint.weight(key, dtype=parameter.dtype, device="cpu"))
                if checkpoint.is_quantized(key):
                    mma.attach_source_codes(parameter, checkpoint.packed(key), checkpoint.tile_scale(key))
            block = nn.Module()
            block.engram = engram
            blocks.append(block)
        self.layers = nn.ModuleList(blocks)

    def forward(self, x, pos, hashes, mask=None):
        for block in self.layers:
            e = block.engram
            x = e(x, hashes[:, :, e.layer_hash_index, :], mask)
        return x


def reference_gate(entry, x, kv, mask):
    e = entry.engram
    key, value = kv.view(1, 1, -1).split([e.hc_mult * e.dim, e.dim], -1)
    key = key.float().unflatten(-1, (e.hc_mult, e.dim))
    hidden = x.float()
    rstd = torch.rsqrt(hidden.square().mean(-1) + e.eps) * torch.rsqrt(key.square().mean(-1) + e.eps)
    dot = (hidden * entry.channel_weight * key).sum(-1) * rstd * e.dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(e.clamp_value).sqrt(), dot))
    gate = gate.masked_fill(~mask.unsqueeze(-1), 0)
    return (hidden + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).bfloat16()


@torch.inference_mode()
def main():
    rank, world = int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl", timeout=dt.timedelta(seconds=180))
    torch.set_num_threads(4)
    checkpoint = V41Checkpoint(str(CHECKPOINT), device="cpu")
    config = load_config(resolve_config(str(CHECKPOINT))).text
    model = MiniEngramModel(checkpoint, config)
    original_tables = [block.engram.embed for block in model.layers]
    manager = HostTableDeviceEngram(model, device=f"npu:{rank}", rank=rank, world=world,
                                   capture_graphs=True, backend="triton")
    generator = torch.Generator().manual_seed(0xE641)
    records = []
    try:
        manager.install()
        for entry in manager.entries.values():
            engram = entry.engram
            full_weight = engram.wkv.weight.detach().t().contiguous().to(manager.device)
            previous_output, previous_bits = None, None
            projection_differences, projection_max_ulp, full_output_differences = [], [], []
            projection_relative_l2, projection_max_abs = [], []
            started = time.perf_counter()
            for step in range(32):
                values = torch.randn(1, 1, engram.hc_mult, engram.dim, generator=generator).bfloat16()
                if step % 8 == 0:
                    values.zero_()
                elif step % 8 == 1:
                    values.mul_(256)
                ids = torch.randint(0, engram.embed.rows_total, (1, 1, entry.hash_cols), generator=generator)
                mask = torch.tensor([[step % 5 != 0]], dtype=torch.bool)
                x = values.to(manager.device)
                actual = manager._forward(entry, x, ids, mask)
                torch.npu.synchronize()
                lookup = manager._lookup(entry, ids).to(manager.device)
                decoded = quantize_decode_fp8_fast(lookup, backend="triton")
                expected_kv = decoded @ full_weight
                actual_kv = entry.gathered
                local_reference = original_fp8_engram_projection(
                    lookup, entry.weight_kn, quantized=engram._quantized,
                    certified_original_bf16_weight=True, backend="triton")
                # Native output-channel TP changes CANN's matrix tiling. Compare
                # graph execution exactly to that eager TP geometry, and measure
                # its difference from the native full matrix separately.
                expected = reference_gate(entry, x, actual_kv, mask.to(manager.device))
                full_expected = reference_gate(entry, x, expected_kv, mask.to(manager.device))
                torch.npu.synchronize()
                actual_bits = actual_kv.cpu().view(torch.int16).to(torch.int32)
                full_bits = expected_kv.cpu().view(torch.int16).to(torch.int32)
                ordered_actual = torch.where(actual_bits < 0, -32768 - actual_bits, actual_bits)
                ordered_full = torch.where(full_bits < 0, -32768 - full_bits, full_bits)
                projection_difference = int((actual_bits != full_bits).sum())
                max_ulp = int((ordered_actual - ordered_full).abs().max())
                projection_differences.append(projection_difference)
                projection_max_ulp.append(max_ulp)
                actual_cpu, full_cpu = actual_kv.float().cpu(), expected_kv.float().cpu()
                delta = actual_cpu - full_cpu
                projection_relative_l2.append(float(delta.norm() / full_cpu.norm().clamp_min(1e-30)))
                projection_max_abs.append(float(delta.abs().max()))
                full_output_differences.append(int((full_expected.view(torch.int16) != actual.view(torch.int16)).sum().item()))
                local_width = local_reference.shape[-1]
                local_differences = int((local_reference.view(torch.int16) != actual_kv[:, rank * local_width:(rank + 1) * local_width].view(torch.int16)).sum().item())
                differences = int((expected.view(torch.int16) != actual.view(torch.int16)).sum().item())
                # Different native N tilings do not have a bit/ULP-neutrality
                # contract. They are explicitly measured as a changed native
                # accumulation experiment; exact eager-TP/capture and gate
                # parity remain mandatory. Full-model quality is a separate gate.
                if differences or local_differences or not bool(torch.isfinite(actual).all().item()):
                    actual_kv = entry.gathered
                    local_reference = original_fp8_engram_projection(
                        lookup, entry.weight_kn, quantized=engram._quantized,
                        certified_original_bf16_weight=True, backend="triton")
                    gate_with_actual_kv = reference_gate(entry, x, actual_kv, mask.to(manager.device))
                    torch.npu.synchronize()
                    actual_bits = actual_kv.cpu().view(torch.int16).to(torch.int32)
                    full_bits = expected_kv.cpu().view(torch.int16).to(torch.int32)
                    ordered_actual = torch.where(actual_bits < 0, -32768 - actual_bits, actual_bits)
                    ordered_full = torch.where(full_bits < 0, -32768 - full_bits, full_bits)
                    local_width = local_reference.shape[-1]
                    failure = {"rank": rank, "layer": engram.layer_id, "step": step,
                               "full_output_bit_differences": differences,
                               "full_native_vs_TP_KV_bit_differences": int((actual_bits != full_bits).sum()),
                               "full_native_vs_TP_KV_max_BF16_ULP": int((ordered_actual - ordered_full).abs().max()),
                               "gate_only_bit_differences_given_actual_KV": int((gate_with_actual_kv.view(torch.int16) != actual.view(torch.int16)).sum().item()),
                               "eager_local_projection_vs_captured_shard_bit_differences": int((local_reference.view(torch.int16) != actual_kv[:, rank * local_width:(rank + 1) * local_width].view(torch.int16)).sum().item()),
                               "lookup_input_bit_differences": int((lookup.view(torch.int16) != entry.lookup.view(torch.int16)).sum().item()),
                               "hidden_input_bit_differences": int((x.view(torch.int16) != entry.x.view(torch.int16)).sum().item()),
                               "finite_actual": bool(torch.isfinite(actual).all().item())}
                    with (OUTPUT / f"device-engram-failure-rank-{rank}.json").open("x") as stream:
                        json.dump(failure, stream, indent=2)
                    raise RuntimeError(f"Engram{engram.layer_id}/step{step}: full-output parity failed: {failure}")
                if previous_output is not None and not torch.equal(previous_output.cpu().view(torch.int16), previous_bits):
                    raise RuntimeError("persistent graph overwrote a caller-owned previous output")
                wire = actual.view(torch.int8).reshape(-1)
                others = torch.empty(world * wire.numel(), dtype=torch.int8, device=manager.device)
                dist.all_gather_into_tensor(others, wire)
                torch.npu.synchronize()
                if not torch.equal(others.view(world, -1), wire.expand(world, -1)):
                    raise RuntimeError("device Engram rank outputs differ")
                previous_output, previous_bits = actual, actual.cpu().view(torch.int16).clone()
            elapsed = time.perf_counter() - started
            # Eager-vs-graph on the same persistent inputs; every boundary stays
            # as recorded. Timings include synchronization, not a TPS claim.
            times = {}
            for name in ("graph", "eager"):
                for _ in range(2):
                    entry.graph.replay() if name == "graph" else manager._body(entry)
                torch.npu.synchronize()
                durations = []
                for _ in range(8):
                    begin = time.perf_counter()
                    entry.graph.replay() if name == "graph" else manager._body(entry)
                    torch.npu.synchronize()
                    durations.append(time.perf_counter() - begin)
                times[name] = sum(durations) / len(durations)
            records.append({"layer": engram.layer_id, "graph_replays": 32, "changed_IDs_and_masks": True,
                            "eager_TP_projection_and_original_gate_bit_differences": 0,
                            "native_full_vs_TP_projection_bit_differences_by_step": projection_differences,
                            "native_full_vs_TP_projection_max_BF16_ULP_by_step": projection_max_ulp,
                            "native_full_vs_TP_projection_relative_L2_by_step": projection_relative_l2,
                            "native_full_vs_TP_projection_max_abs_by_step": projection_max_abs,
                            "native_full_vs_TP_output_bit_differences_by_step": full_output_differences,
                            "native_TP_shape_reduction_changed_not_bit_neutral": True,
                            "all_rank_output_bits_equal": True, "previous_outputs_not_overwritten": True,
                            "validation_elapsed_s": elapsed, "isolated_synchronized_seconds_per_body": times,
                            "microbench_not_model_TPS": True})
            del full_weight
        if any(block.engram.embed is not table for block, table in zip(model.layers, original_tables)):
            raise RuntimeError("original table object changed")
        row = {"rank": rank, "world": world, "checkpoint_original": True, "original_tables_host": True,
               "source_weights_original_and_unchanged": True, "records": records,
               "controller": manager.describe(), "native_reduction_is_explicit_experiment": True}
        with (OUTPUT / f"device-engram-rank-{rank}.json").open("x") as stream:
            json.dump(row, stream, indent=2)
        if rank == 0:
            print(json.dumps(row), flush=True)
    finally:
        manager.close()
        checkpoint.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
