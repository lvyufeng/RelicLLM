"""Semi-automated hardware-adaptation triage: can this checkpoint be served here, and at what tier?

Four gates, split by cost. Gates 0 and 1 are free -- a static fit calculation and a reachability
check -- so they run on every checkpoint unconditionally. Gate 2 is a measurement and costs
GPU-hours, so a human decides whether to pay it. Gate 3 is the rule that turns the two into one of
five answers:

``PRODUCTION`` | ``CHAT`` | ``DEMO`` | ``IMPOSSIBLE`` | ``CANDIDATE``

The design rule this package is built around: **a static calculation may fail a checkpoint
definitively, but its pass is only ever provisional.** It has no workspace measurement, no
fragmentation and no bank-miss penalty, so "it fits" is a necessary condition and never a sufficient
one. That is why gate 0's pass produces ``CANDIDATE`` and never a tier.

The other rule is that every derived number carries how it was arrived at -- ``measured``,
``derived`` or ``assumed`` -- so that an assumption cannot propagate silently into a card count.

Two rulings decided at the outset and load-bearing throughout:

1. **Cost is out of scope.** The only thing that makes a model impossible is that it does not fit. A
   missing kernel, loader or adapter is a note and a work item, never a downgrade -- an agent can
   write a kernel. :func:`~relicllm.triage.tier.tier` therefore does not take a reachability
   argument, and a test asserts that its signature stays that way.
2. **A host offload counts as placed.** Routed experts and gather tables in host RAM are placed, not
   unplaced; the tier then comes from measurement and the host requirement is recorded as a
   precondition. Where the bank is larger than host memory the excess comes back onto the cards, and
   that changes the card count rather than passing unnoticed.

Typical use::

    from relicllm.triage import assess, hardware_profile_for

    report = assess("/mnt/data3/DeepSeek-V4.1-Flash", hardware=hardware_profile_for(host_memory_gib=1007))
    print(report.verdict.tier)   # CANDIDATE -- nothing measured yet on this hardware
    print(report.upgrade())      # the command that would settle it

Or from a shell::

    python -m relicllm.cli.triage /mnt/data3/DeepSeek-V4.1-Flash --host-memory-gib 1007
"""

from __future__ import annotations

from relicllm.triage.checkpoint import (
    CheckpointInventory,
    read_inventory,
    role_of,
)
from relicllm.triage.fit import (
    FitResult,
    fit_all,
    fit_precision,
    hardware_profile_for,
)
from relicllm.triage.kv import (
    Confidence,
    KvGeometry,
    LayerGeometry,
    bytes_per_token_per_rank,
    kv_geometry,
)
from relicllm.triage.reach import Reachability, reachability
from relicllm.triage.report import Assessment, assess, dumps
from relicllm.triage.tier import DEFAULT_THRESHOLDS, Measured, Thresholds, Tier, Verdict, tier
from relicllm.triage.weights import PrecisionOption, precision_ladder

__all__ = [
    "Assessment",
    "CheckpointInventory",
    "Confidence",
    "DEFAULT_THRESHOLDS",
    "FitResult",
    "KvGeometry",
    "LayerGeometry",
    "Measured",
    "PrecisionOption",
    "Reachability",
    "Thresholds",
    "Tier",
    "Verdict",
    "assess",
    "bytes_per_token_per_rank",
    "dumps",
    "fit_all",
    "fit_precision",
    "hardware_profile_for",
    "kv_geometry",
    "precision_ladder",
    "reachability",
    "read_inventory",
    "role_of",
    "tier",
]
