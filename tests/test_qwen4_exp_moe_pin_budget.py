"""The expert pin weighs a layer before it moves it.

`pin_layers` is all-or-nothing a layer: a layer either has every one of its experts resident or it
keeps staging them. The budget therefore has to decide *before* the transfer. It used to decide
after -- move the whole layer to the device, then compare and drop it -- which under a tight budget
allocated the layer it was about to refuse and failed with an out-of-memory instead of declining.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from relicllm.models.qwen4_exp.moe import HostExpertMoE

PER_EXPERT_BYTES = 4096
NUM_EXPERTS = 4


class _RecordingReader:
    """A reader that hands out host rows and records which ones were asked for."""

    def __init__(self):
        self.num_experts = NUM_EXPERTS
        self.expert_shard = None
        self.reads = []
        self._gate_up = torch.zeros(PER_EXPERT_BYTES // 4, dtype=torch.float32)
        self._down = torch.zeros(2, dtype=torch.float32)

    def expert_rows(self, layer_idx, expert_id):
        self.reads.append((layer_idx, expert_id))
        return self._gate_up, self._down


def _make_moe(reader):
    moe = HostExpertMoE.__new__(HostExpertMoE)
    moe.reader = reader
    moe.num_experts = reader.num_experts
    moe.device = torch.device("cpu")
    moe.wide_bank = False
    moe._resident = {}
    moe._wide_keys = {}
    moe._resident_layers = 0
    moe._weight_nbytes = HostExpertMoE._weight_nbytes
    moe._wide = lambda w: w
    return moe


def _pin(reader, budget):
    # The success path ends in a device synchronise, which a CPU-only build cannot serve and this
    # test has no use for: the point is which transfers happen, not when they complete.
    original = torch.cuda.synchronize
    torch.cuda.synchronize = lambda *a, **k: None
    try:
        return _make_moe(reader).pin_layers([0], budget_bytes=budget)
    finally:
        torch.cuda.synchronize = original


def _layer_bytes():
    gate_up = PER_EXPERT_BYTES // 4 * 4
    return NUM_EXPERTS * (gate_up + 2 * 4)


def test_a_layer_over_budget_is_declined_after_being_weighed():
    reader = _RecordingReader()
    pinned = _pin(reader, _layer_bytes() - 1)
    assert pinned == 0, "a layer that does not fit must not be pinned"
    assert reader.reads, "the layer still has to be weighed to know it does not fit"


def test_a_layer_within_budget_is_pinned():
    reader = _RecordingReader()
    pinned = _pin(reader, _layer_bytes())
    assert pinned == 1, "a layer that fits must be pinned"


def test_a_zero_budget_pins_nothing_and_does_not_raise():
    reader = _RecordingReader()
    assert _pin(reader, 0) == 0, "a full card must decline the pin, not fail on it"
