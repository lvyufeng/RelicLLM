"""Exhaustive original-byte/scale K256 gate, including every K32 block position."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
import torch_npu

from relic_core.kernels.ascendc_online_tile_cube import AscendCTileCubeConsumer
from relic_core.kernels.online_tile_cube import pack_expert_tile_bank
from relic_core.kernels.unpack import unpack_fp4_to_int8, fp4_int8_scale

ROOT = Path("<workspace>/relic-v41-perf-20261118-0938")
OUTPUT = Path(os.environ["RELIC_PERF_OUTPUT"])


@torch.inference_mode()
def main():
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    builds = json.loads((ROOT / "logs/consumer-variants-build-20261120-v2.json").read_bytes())["variants"]
    records = []
    for name in ("k256", "k256-preload-a"):
        row = builds[name]
        operator = AscendCTileCubeConsumer(row["library"], sha256=row["sha256"])
        code = torch.arange(256).to(torch.uint8).reshape(16, 16).repeat(251, 1).unsqueeze(0).repeat(1, 1, 8)
        scales = torch.arange(2, 253).to(torch.uint8).repeat_interleave(16).reshape(1, 4016, 1).repeat(1, 1, 8)
        # Change every K32 block's lane permutation without changing the
        # covered original byte/scale corpus. This also checks tile order.
        for block in range(8):
            code[:, :, block * 16:(block + 1) * 16] = code[:, :, block * 16:(block + 1) * 16].roll(block, -1)
        expected = (unpack_fp4_to_int8(code, backend="torch").float()
                    * fp4_int8_scale(scales, backend="torch").repeat_interleave(32, -1)).bfloat16()[0]
        bt, st = pack_expert_tile_bank(code, scales)
        bt, st = bt.to(device), st.to(device)
        x = torch.zeros(1, 256, dtype=torch.bfloat16, device=device)
        selected = torch.zeros(1, dtype=torch.int32, device=device)
        ring = torch.empty(24, 2, operator.ring_elements, dtype=torch.bfloat16, device=device)
        accumulation = torch.empty(1, 4016, dtype=torch.float32, device=device)
        positions = list(range(256))
        checked = 0
        for position in positions:
            x.zero_(); x[0, position] = 1
            output = operator(x, bt, st, selected, world=1, scratch=ring, accumulation=accumulation,
                              certified_bf16_operands=True)
            torch.npu.synchronize()
            actual = output.cpu()[0]
            differences = int((actual.view(torch.int16) != expected[:, position].view(torch.int16)).sum())
            if differences:
                raise RuntimeError(f"{name} original byte/scale corpus at K{position}: {differences} differences")
            checked += actual.numel()
        records.append({"variant": name, "library_sha256": operator.sha256, "positions": positions,
                        "original_byte_values_covered": 256, "original_E8M0_scale_range": [2, 252],
                        "BF16_values_checked": checked, "BF16_bit_differences": 0,
                        "all_256_K_positions_checked": True, "each_original_K32_block_covered": True,
                        "not_model_TPS": True})
        del bt, st, x, ring, accumulation, expected
    with (OUTPUT / "K256-original-byte-scale-corpus.json").open("x") as stream:
        json.dump(records, stream, indent=2)
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
