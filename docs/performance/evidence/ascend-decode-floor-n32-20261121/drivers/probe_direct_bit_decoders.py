"""Exhaustive native BF16 codec and real-shape direct-bit/AIV consumer comparison."""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

import torch
import torch_npu

from relic_core.kernels.ascendc_online_tile_cube import AscendCTileCubeConsumer
from relic_core.kernels.direct_aiv_gemv import DirectAIVConsumer
from relic_core.kernels.fast_fused_fp8_activation_decode import quantize_decode_fp8_fast
from relic_core.kernels.online_tile_cube import pack_expert_tile_bank
from relic_core.kernels.unpack import unpack_fp4_to_int8, fp4_int8_scale
from relicllm.models.deepseek_v4_1.ascend_graphs import _record, npu_graph_api
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint, scale_key
from probe_online_tile_cube import CopyAudit

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938")
OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])


def entry(op, x, b, s, ids, n):
    ring = torch.empty(24, 2, op.ring_elements, dtype=torch.bfloat16, device=x.device)
    accum = torch.empty(ids.numel(), n, dtype=torch.float32, device=x.device)
    state = {"op": op, "ring": ring, "accum": accum}
    def body():
        state["out"] = op(x, b, s, ids, world=1, certified_bf16_operands=True, scratch=ring, accumulation=accum)
    api = npu_graph_api()
    state["pool"] = api.graph_pool_handle()
    state["graph"] = _record(body, state["pool"], 2, api)
    return state


def paired_times(states, ids, active, repeats=16):
    ids.copy_(torch.tensor(list(range(active)) + [-1] * (ids.numel() - active), dtype=torch.int32, device=ids.device))
    for state in states.values():
        for _ in range(3):
            state["graph"].replay()
    torch.npu.synchronize()
    samples = {name: [] for name in states}
    names = list(states)
    for repetition in range(repeats):
        for name in names if repetition % 2 else names[::-1]:
            start = time.perf_counter()
            states[name]["graph"].replay(); torch.npu.synchronize()
            samples[name].append(1000 * (time.perf_counter() - start))
    return {name: {"samples_ms": values, "mean_ms": sum(values) / len(values)} for name, values in samples.items()}


@torch.inference_mode()
def main():
    torch.npu.set_device(0); torch.set_num_threads(4)
    device = torch.device("npu:0")
    build = json.loads((ROOT / "logs/direct-bit-consumer-build-20261121-v1.json").read_bytes())["variants"]
    operators = {"k128-original": AscendCTileCubeConsumer(ROOT / "cache/online-tile-cube-k128-v2/lib/librelic_online_tile_cube.so",
                         sha256="0cd9111e373859c5b8d4ec09089a952467f26b6c88a3fb387842b11c0725b287"),
                 "AIV-direct": DirectAIVConsumer()}
    for name, row in build.items():
        op = AscendCTileCubeConsumer(row["library"], sha256=row["sha256"])
        if not op.direct_bf16_bits or op.group_k != row["group_k"]:
            raise RuntimeError("the native build does not expose the direct bit policy it recorded")
        operators[name] = op
    corpus = []
    for name in ("k128-direct-bits", "k256-direct-bits", "AIV-direct"):
        op = operators[name]
        k = 128 if name.startswith("k128") else 256
        code = torch.arange(256).to(torch.uint8).reshape(16, 16).repeat(251, 1).unsqueeze(0).repeat(1, 1, k // 32)
        scale = torch.arange(2, 253).to(torch.uint8).repeat_interleave(16).reshape(1, 4016, 1).repeat(1, 1, k // 32)
        for block in range(k // 32):
            code[:, :, block * 16:(block + 1) * 16] = code[:, :, block * 16:(block + 1) * 16].roll(block, -1)
        expected = (unpack_fp4_to_int8(code, backend="torch").float()
                    * fp4_int8_scale(scale, backend="torch").repeat_interleave(32, -1)).bfloat16()[0]
        bt, st = pack_expert_tile_bank(code, scale)
        bt, st = bt.to(device), st.to(device)
        x = torch.zeros(1, k, dtype=torch.bfloat16, device=device)
        ids = torch.zeros(1, dtype=torch.int32, device=device)
        state = entry(op, x, bt, st, ids, 4016)
        for position in range(k):
            x.zero_(); x[0, position] = 1
            state["graph"].replay(); torch.npu.synchronize()
            value = state["out"].cpu()[0]
            if not torch.equal(value.view(torch.int16), expected[:, position].view(torch.int16)):
                raise RuntimeError(f"original codec failed {name}, K position {position}")
        corpus.append({"variant": name, "all_K_positions": k, "BF16_values_checked": 4016 * k,
                       "E2M1_byte_values": 256, "E8M0_scales": [2, 252], "BF16_bit_differences": 0,
                       "operator": op.describe(), "not_model_TPS": True})
        del state, bt, st, expected
    generator = torch.Generator().manual_seed(0x50A1)
    dynamic = []
    for k, n, slots, shared in ((256, 64, 6, True), (256, 64, 6, False), (576, 64, 6, False)):
        available = {name: op for name, op in operators.items() if k % (32 * op.group_k) == 0}
        code = torch.randint(0, 256, (8, n, k // 2), generator=generator, dtype=torch.uint8)
        scales = torch.full((8, n, k // 32), 121, dtype=torch.uint8)
        dense = (unpack_fp4_to_int8(code, backend="torch").float()
                 * fp4_int8_scale(scales, backend="torch").repeat_interleave(32, -1)).bfloat16()
        bt, st = pack_expert_tile_bank(code, scales); bt, st = bt.to(device), st.to(device)
        x = torch.zeros(1 if shared else slots, k, dtype=torch.bfloat16, device=device)
        ids = torch.zeros(slots, dtype=torch.int32, device=device)
        states = {name: entry(op, x, bt, st, ids, n) for name, op in available.items()}
        for replay in range(32):
            active = (0, 1, 3, 6)[replay % 4]
            chosen = torch.randperm(8, generator=generator)[:active].tolist() + [-1] * (slots - active)
            source = quantize_decode_fp8_fast(torch.randint(-8, 9, (x.shape[0], k), generator=generator).bfloat16(), backend="torch")
            x.copy_(source.to(device)); ids.copy_(torch.tensor(chosen, dtype=torch.int32, device=device))
            expected = torch.zeros(slots, n, dtype=torch.bfloat16)
            for slot, expert in enumerate(chosen):
                if expert >= 0:
                    expected[slot] = (source[0 if shared else slot].float()[None] @ dense[expert].float().t())[0].bfloat16()
            for name, state in states.items():
                audit = CopyAudit()
                with audit:
                    state["graph"].replay()
                torch.npu.synchronize()
                if audit.d2h or not torch.equal(state["out"].cpu().view(torch.int16), expected.view(torch.int16)):
                    raise RuntimeError(f"changing-input bit/slot contract failed for {name}/{replay}")
        for invalid in (-2, 8, 2**31 - 1):
            ids.fill_(invalid)
            for name, state in states.items():
                state["graph"].replay(); torch.npu.synchronize()
                if not bool(torch.isnan(state["out"]).all().item()):
                    raise RuntimeError(f"invalid source not poisoned {name}")
        ids.zero_()
        for invalid_scale in (0, 1, 253, 254, 255):
            st.fill_(invalid_scale)
            for name, state in states.items():
                state["graph"].replay(); torch.npu.synchronize()
                if not bool(torch.isnan(state["out"]).all().item()):
                    raise RuntimeError(f"uncertified original scale not poisoned {name}/{invalid_scale}")
        dynamic.append({"K": k, "N": n, "shared": shared, "variants": list(states),
                        "changing_replays": 32, "BF16_bit_differences": 0, "routing_D2H_calls": 0,
                        "invalid_sources_and_scales_poisoned": True})
    checkpoint = V41Checkpoint("<checkpoint>", device="cpu")
    real = []
    try:
        for projection in ("w1", "w3", "w2"):
            keys = [f"layers.0.ffn.experts.{expert}.{projection}.weight" for expert in range(8)]
            code = torch.stack([checkpoint.packed(key).view(torch.uint8).clone() for key in keys])
            scales = torch.stack([checkpoint.packed(scale_key(key)).view(torch.uint8).clone() for key in keys])
            # Exact output slices for up/gate; matching input/K32 slices for down.
            if projection != "w2":
                code, scales = code[:, :1152].contiguous(), scales[:, :1152].contiguous()
            else:
                code, scales = code[:, :, :576].contiguous(), scales[:, :, :36].contiguous()
            n, k = code.shape[1], code.shape[2] * 2
            available = {name: op for name, op in operators.items() if k % (32 * op.group_k) == 0}
            bt, st = pack_expert_tile_bank(code, scales); bt, st = bt.to(device), st.to(device)
            shared = projection != "w2"
            x = quantize_decode_fp8_fast(torch.randn(1 if shared else 6, k, generator=generator).bfloat16(), backend="torch").to(device)
            ids = torch.arange(6, dtype=torch.int32, device=device)
            states = {name: entry(op, x, bt, st, ids, n) for name, op in available.items()}
            for state in states.values():
                state["graph"].replay()
            torch.npu.synchronize()
            values = {name: state["out"].cpu() for name, state in states.items()}
            if not torch.equal(values["k128-original"].view(torch.int16), values["k128-direct-bits"].view(torch.int16)):
                raise RuntimeError("integer BF16 bit construction changed the same-Cube real-weight reduction")
            delta = values["AIV-direct"].float() - values["k128-original"].float()
            if not all(torch.isfinite(value).all() for value in values.values()):
                raise RuntimeError("a real original-weight projection produced nonfinite values")
            real.append({"projection": projection, "original_checkpoint_keys": keys, "local_TP2_shape": [n, k],
                         "old_to_direct_bits_BF16_bit_differences": 0,
                         "AIV_to_Cube_BF16_bit_differences": int((values["AIV-direct"].view(torch.int16) != values["k128-original"].view(torch.int16)).sum()),
                         "AIV_to_Cube_max_abs": float(delta.abs().max()),
                         "AIV_reduction_is_explicit_changed_accumulation_experiment": True,
                         "paired_times_by_active_slots": {str(active): paired_times(states, ids, active) for active in (0, 1, 3, 6)}})
    finally:
        checkpoint.close()
    report = {"corpus": corpus, "dynamic": dynamic, "real_projections": real, "build": build,
              "not_model_TPS": True, "original_FP4_E8M0_and_activation_quantizer_unchanged": True}
    with (OUTPUT / "direct-bit-consumer-probe.json").open("x") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({"codec": corpus, "real_mean_ms_active6": {row["projection"]: {name: data["mean_ms"] for name, data in row["paired_times_by_active_slots"]["6"].items()} for row in real}}), flush=True)


if __name__ == "__main__":
    main()
