"""Same-operand native consumer grain/L1 comparison, not a model TPS claim."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import torch
import torch_npu

from relic_core.kernels.ascendc_online_tile_cube import AscendCTileCubeConsumer, activation_l1_layout
from relic_core.kernels.online_tile_cube import pack_expert_tile_bank
from relic_core.kernels.fast_fused_fp8_activation_decode import quantize_decode_fp8_fast
from relic_core.kernels.unpack import unpack_fp4_to_int8, fp4_int8_scale
from relicllm.models.deepseek_v4_1.ascend_graphs import _record, npu_graph_api
from relicllm.models.deepseek_v4_1.loader import V41Checkpoint, scale_key
from probe_online_tile_cube import CopyAudit

ROOT = Path("/data1/relic-v41-perf-20261118-0938")
OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])
OLD = ROOT / "cache/online-tile-cube-k128-v2/lib/librelic_online_tile_cube.so"
OLD_SHA = "0cd9111e373859c5b8d4ec09089a952467f26b6c88a3fb387842b11c0725b287"


def decoded(code, scale):
    return (unpack_fp4_to_int8(code, backend="torch").float()
            * fp4_int8_scale(scale, backend="torch").repeat_interleave(32, -1)).bfloat16()


def entry(operator, x, b, s, ids, n):
    ring = torch.empty((24, 2, operator.ring_elements), dtype=torch.bfloat16, device=x.device)
    accum = torch.empty((ids.numel(), n), dtype=torch.float32, device=x.device)
    state = {"operator": operator, "ring": ring, "accum": accum}
    def body():
        return operator(x, b, s, ids, world=1, certified_bf16_operands=True,
                        scratch=ring, accumulation=accum)
    state["body"] = body
    def capture_body():
        state["output"] = body()
        return state["output"]
    api = npu_graph_api()
    state["pool"] = api.graph_pool_handle()
    state["graph"] = _record(capture_body, state["pool"], 2, api)
    return state


def compare_preloads(values):
    differences = {}
    for base, candidate in (("k128", "k128-preload-a"), ("k256", "k256-preload-a")):
        count = int((values[base].view(torch.int16) != values[candidate].view(torch.int16)).sum())
        differences[base + "_preload_only"] = count
        if count:
            raise RuntimeError(f"activation reuse changed {count} BF16 values at {base}")
    return differences


def same_inputs_timing(states, ids, active, repeats=16):
    ids.copy_(torch.tensor(list(range(active)) + [-1] * (ids.numel() - active), dtype=torch.int32, device=ids.device))
    for state in states.values():
        for _ in range(3):
            state["graph"].replay()
    torch.npu.synchronize()
    samples = {name: [] for name in states}
    names = list(states)
    for repetition in range(repeats):
        order = names if repetition % 2 else names[::-1]
        for name in order:
            start = time.perf_counter()
            states[name]["graph"].replay()
            torch.npu.synchronize()
            samples[name].append(time.perf_counter() - start)
    return {name: {"seconds": values, "mean_ms": 1000 * sum(values) / len(values)} for name, values in samples.items()}


@torch.inference_mode()
def main():
    torch.npu.set_device(0)
    torch.set_num_threads(4)
    device = torch.device("npu:0")
    build = json.loads((ROOT / "logs/consumer-variants-build-20261120-v2.json").read_bytes())["variants"]
    operators = {"k128": AscendCTileCubeConsumer(OLD, sha256=OLD_SHA)}
    for name, row in build.items():
        operator = AscendCTileCubeConsumer(row["library"], sha256=row["sha256"])
        if operator.group_k != row["original_K32_blocks"] or operator.preload_activation != row["activation_L1_preload_requested"]:
            raise RuntimeError("the built host/device geometry query is inconsistent")
        operators[name] = operator
    generator = torch.Generator().manual_seed(0xC832)
    validations = []
    # Integer-friendly BF16/FP8 inputs and original FP4 values give an exact
    # FP32 sum oracle, independent of native MMAD K partitioning.
    for k, n, slots, shared in ((256, 64, 6, True), (256, 64, 6, False), (4096, 16, 8, False)):
        code = torch.randint(0, 256, (8, n, k // 2), generator=generator, dtype=torch.uint8)
        scale = torch.full((8, n, k // 32), 121, dtype=torch.uint8)
        weights = decoded(code, scale)
        bt, st = pack_expert_tile_bank(code, scale)
        bt, st = bt.to(device), st.to(device)
        rows = 1 if shared else slots
        host_x = quantize_decode_fp8_fast(torch.randint(-8, 9, (rows, k), generator=generator).bfloat16(), backend="torch")
        x = host_x.to(device)
        ids = torch.tensor(list(range(slots)), dtype=torch.int32, device=device)
        states = {name: entry(operator, x, bt, st, ids, n) for name, operator in operators.items()}
        for replay in range(32):
            count = (0, 1, min(3, slots), slots)[replay % 4]
            chosen = torch.randperm(8, generator=generator)[:count].tolist() + [-1] * (slots - count)
            host_x = quantize_decode_fp8_fast(torch.randint(-8, 9, (rows, k), generator=generator).bfloat16(), backend="torch")
            x.copy_(host_x.to(device)); ids.copy_(torch.tensor(chosen, dtype=torch.int32, device=device))
            expected = torch.zeros((slots, n), dtype=torch.bfloat16)
            for slot, index in enumerate(chosen):
                if index >= 0:
                    expected[slot] = (host_x[0 if shared else slot].float()[None] @ weights[index].float().t())[0].bfloat16()
            for name, state in states.items():
                audit = CopyAudit()
                with audit:
                    state["graph"].replay()
                torch.npu.synchronize()
                actual = state["output"].cpu()
                count_diff = int((actual.view(torch.int16) != expected.view(torch.int16)).sum())
                if count_diff or audit.d2h:
                    raise RuntimeError(f"dynamic {name} K{k}/replay{replay} parity failed: {count_diff}, D2H{audit.d2h}")
        for invalid in (-2, 8, 2**31 - 1):
            ids.fill_(invalid)
            for name, state in states.items():
                state["graph"].replay()
                torch.npu.synchronize()
                if not bool(torch.isnan(state["output"]).all().item()):
                    raise RuntimeError(f"{name} did not poison invalid source {invalid}")
        validations.append({"k": k, "n": n, "slots": slots, "shared": shared, "changed_input_graph_replays": 32,
                            "all_variants_BF16_bit_differences": 0, "routing_D2H": 0, "invalid_source_poisoning": True,
                            "activation_L1_geometry": {name: activation_l1_layout(k, slots, group_k=operator.group_k, shared=shared) for name, operator in operators.items()}})
        del states, bt, st, weights
    with (OUTPUT / "consumer-variants-dynamic-validation.json").open("x") as stream:
        json.dump(validations, stream, indent=2)
    print(json.dumps({"dynamic_validation": validations}), flush=True)
    # Actual checkpoint bytes, not a random-weight performance proxy.
    checkpoint = V41Checkpoint("/data1/relic-v41-20261116-210/checkpoint", device="cpu")
    records = []
    try:
        for projection in ("w1", "w3", "w2"):
            keys = [f"layers.0.ffn.experts.{expert}.{projection}.weight" for expert in range(8)]
            if any(key not in checkpoint or not checkpoint.is_packed_fp4(key) or checkpoint.block_size(key) != 32 for key in keys):
                raise RuntimeError("real checkpoint projection is not the expected original E2M1/K32 form")
            code = torch.stack([checkpoint.packed(key).view(torch.uint8).clone() for key in keys])
            grid = torch.stack([checkpoint.packed(scale_key(key)).view(torch.uint8).clone() for key in keys])
            # MXFP4 uses one original scale row per output row. It is not
            # the square N32 dense-FP8 grid and must not be expanded on N.
            if grid.shape != (*code.shape[:2], code.shape[-1] * 2 // 32):
                raise RuntimeError("the actual FP4 file does not carry its expected row/K32 scale grid")
            scale = grid
            if int(scale.min()) < 2 or int(scale.max()) > 252:
                raise RuntimeError("real original E8M0 weight range is not certified BF16")
            _, n, packed_k = code.shape
            k = packed_k * 2
            shared = projection != "w2"
            bt, st = pack_expert_tile_bank(code, scale)
            bt, st = bt.to(device), st.to(device)
            host_x = quantize_decode_fp8_fast(torch.randn((1 if shared else 6, k), generator=generator).bfloat16(), backend="torch")
            x = host_x.to(device)
            ids = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32, device=device)
            states = {name: entry(operator, x, bt, st, ids, n) for name, operator in operators.items()}
            for state in states.values():
                state["graph"].replay()
            torch.npu.synchronize()
            values = {name: state["output"].cpu() for name, state in states.items()}
            preload_diff = compare_preloads(values)
            grain_delta = values["k256"].float() - values["k128"].float()
            grain_diff = int((values["k256"].view(torch.int16) != values["k128"].view(torch.int16)).sum())
            if not all(torch.isfinite(value).all() for value in values.values()):
                raise RuntimeError("real projection result is nonfinite")
            timings = {}
            for active in (0, 1, 3, 6):
                timings[str(active)] = same_inputs_timing(states, ids, active)
            record = {"projection": projection, "original_checkpoint_keys": keys, "n": n, "k": k, "shared": shared,
                      "scale_range": [int(scale.min()), int(scale.max())], "preload_only_BF16_bit_differences": preload_diff,
                      "k128_vs_k256_bit_differences": grain_diff, "k128_vs_k256_max_abs": float(grain_delta.abs().max()),
                      "native_K_partitioning_is_explicit_experiment": True, "same_input_paired_graph_timings_by_active_count": timings,
                      "library_reports": {name: operator.describe() for name, operator in operators.items()},
                      "not_model_TPS": True, "no_original_checkpoint_or_activation_requantization": True}
            records.append(record)
            print(json.dumps({"projection": projection, "grain_bit_differences": grain_diff,
                              "preload_bit_differences": preload_diff,
                              "same_input_mean_ms": {active: {name: row["mean_ms"] for name, row in grouped.items()} for active, grouped in timings.items()}}), flush=True)
            del states, bt, st, code, grid, scale
    finally:
        checkpoint.close()
    with (OUTPUT / "consumer-variants-real-projection-comparison.json").open("x") as stream:
        json.dump({"validations": validations, "records": records, "operator_only_not_model_TPS": True,
                   "accepted_k128_sha256": OLD_SHA, "new_build_report": build}, stream, indent=2)


if __name__ == "__main__":
    main()
