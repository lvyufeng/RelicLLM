"""Gate and up must be quantized under the one global scale the kernel applies to both."""
from types import SimpleNamespace

import torch

from relicllm.models.qwen4_exp.moe import NvFp4SlotExpertMoE


def _unpack(packed, scales, expert_scale):
    levels = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32)
    codes = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8)
    codes[..., 0::2] = packed & 15
    codes[..., 1::2] = packed >> 4
    values = levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    block = scales.view(torch.float8_e4m3fn).float().repeat_interleave(16, dim=-1)
    return values * block * expert_scale.view(-1, 1, 1)


def test_joint_gate_up_scale_when_up_has_larger_range():
    # Expert 0's up projection has a much larger amax; expert 1 has the opposite ordering. The
    # NVFP4 kernel uses table[6] for BOTH projections, not a separate expert scale per matrix.
    gate = torch.stack([torch.full((16, 16), 0.5), torch.full((16, 16), 4.0)])
    up = torch.stack([torch.full((16, 16), 3.0), torch.full((16, 16), 0.5)])
    down = torch.ones(2, 16, 16)
    shard = SimpleNamespace(layer_tensors=lambda _layer: (torch.cat((gate, up), dim=1), down))
    backend = NvFp4SlotExpertMoE.__new__(NvFp4SlotExpertMoE)
    backend.shard = shard
    backend.device = torch.device("cpu")
    backend._tables = {}
    backend.cache_hits = 0
    table = backend._table(0)
    w1q, w1s, w3q, w3s, _w2q, _w2s, global_gate_up, _global_down = table
    torch.testing.assert_close(global_gate_up, torch.tensor([3.0 / 6, 4.0 / 6]))
    # Decode exactly as the kernel does: both matrices get the SAME global scale.
    restored_gate = _unpack(w1q, w1s, global_gate_up)
    restored_up = _unpack(w3q, w3s, global_gate_up)
    torch.testing.assert_close(restored_gate, gate, rtol=0.04, atol=0)
    torch.testing.assert_close(restored_up, up, rtol=0.04, atol=0)
    assert backend._table(0) is table
