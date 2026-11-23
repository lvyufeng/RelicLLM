"""Observe how the selected experts spread across the expert-parallel ranks.

The step spends about 19% of it waiting at collective barriers with every AI core idle. Waiting a
hundred times the median with nothing running means waiting for the slowest rank, and the expected
cause is that the experts a token selects do not land evenly on the ranks. This lane runs expert
parallelism four ways inside two-way tensor parallelism, so a token's handful of experts is spread over
only four owners and the counts per owner are small integers with real spread.

This module only observes. It records the `indices` each routed dispatch receives, keeps the clones on
device so no per-layer synchronisation distorts anything, and writes the counts once at the end. It
changes no routing decision, no weight and no value, and it is not part of any accepted lane.

The earlier attempt wrapped `device_selected_fp4_moe`, which this lane does not use at all; the record
says `device_selected_MoE: null` and `expert_TP_EP` is the live path.
"""
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path

_RECORDS: list = []
_DESCRIBED: dict = {}


def _flush():
    """Write the recorded routing decisions to the path the environment names."""
    target = os.environ.get('RELIC_ROUTING_PROBE_OUT')
    if not target:
        return
    payload = {'calls': len(_RECORDS), 'records': [], 'described': _DESCRIBED}
    rank = -1
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
    except Exception:
        rank = -1
    payload['rank'] = rank
    if _RECORDS:
        import torch

        stacked = torch.stack([t.reshape(-1) for t in _RECORDS]).to('cpu').to(torch.int64)
        payload['shape_per_call'] = list(_RECORDS[0].shape)
        payload['calls_recorded'] = int(stacked.shape[0])
        payload['experts_selected_per_call'] = int(stacked.shape[1])
        payload['distinct_ids_seen'] = int(torch.unique(stacked).numel())
        payload['id_min'] = int(stacked.min())
        payload['id_max'] = int(stacked.max())
        world = _DESCRIBED.get('ep_world')
        if world:
            # `route_owned_kernel` states the predicate literally: `owned = (ids % WORLD) == RANK`,
            # so rank r owns the experts whose id is congruent to r. (`expert // ep_world` is the
            # slot index inside a bank, which is a different thing and was the earlier mistake.)
            owners = torch.remainder(stacked, world)
            per_rank = torch.stack([(owners == r).sum(dim=1) for r in range(world)], dim=1)
            payload['per_rank_counts'] = per_rank.tolist()
            spread = per_rank.max(dim=1).values - per_rank.min(dim=1).values
            histogram = {}
            for value in range(int(spread.min()), int(spread.max()) + 1):
                histogram[str(value)] = int((spread == value).sum())
            payload['spread_histogram'] = histogram
            payload['calls_with_any_spread'] = int((spread > 0).sum())
            payload['calls_with_spread_at_least_2'] = int((spread >= 2).sum())
            payload['max_spread'] = int(spread.max())
            payload['mean_max_count'] = float(per_rank.max(dim=1).values.float().mean())
            payload['mean_min_count'] = float(per_rank.min(dim=1).values.float().mean())
            # The predicted idle time: every rank except the slowest waits for the difference.
            excess = (per_rank.max(dim=1).values.unsqueeze(1) - per_rank).sum(dim=1)
            payload['sum_excess_rank_experts'] = int(excess.sum())
            payload['mean_excess_rank_experts'] = float(excess.float().mean())
            payload['raw_ids'] = stacked.tolist()
    Path(target + '.%d.json' % rank).write_text(json.dumps(payload))


def install():
    """Wrap the expert-TP dispatch so the indices are recorded and then used unchanged."""
    from relicllm.models.deepseek_v4_1 import ascend_expert_tp

    existing = getattr(ascend_expert_tp, '_relic_routing_probe', None)
    if existing is not None:
        return existing
    original = ascend_expert_tp.tp_moe_down_partials

    def probed(x, weights, indices, banks, *args, **kwargs):
        for key in ('ep_world', 'world', 'ep_rank', 'rank', 'n_experts', 'topk'):
            if key in kwargs:
                _DESCRIBED[key] = kwargs[key]
        _RECORDS.append(indices.detach().clone())
        return original(x, weights, indices, banks, *args, **kwargs)

    ascend_expert_tp.tp_moe_down_partials = probed
    ascend_expert_tp._relic_routing_probe = {'installed': True}
    atexit.register(_flush)
    return ascend_expert_tp._relic_routing_probe