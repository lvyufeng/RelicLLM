"""Latest TP2 timeline attribution with replicated EP ownership accounted for."""
from __future__ import annotations

from collections import defaultdict
import inspect
import json

import analyze_current_bottleneck_profile as base


def main():
    source = inspect.getsource(base.analyze)
    marker = '    source_bytes_per_expert = 3 * (2304 * 5120 // 2 + 2304 * 5120 // 32)\n'
    source = source.replace(marker,
        '    tp_world = scopes[0]["expert_topology"]["TP"]\n'
        '    ep_world = scopes[0]["expert_topology"]["EP"]\n'
        '    if tp_world != 2 or ep_world != 4:\n'
        '        raise RuntimeError("this analysis requires the real TP2xEP4 configuration")\n' + marker, 1)
    source = source.replace('sum(values) != 6 or len(values) != 8', 'sum(values) != 6 * tp_world or len(values) != 8', 1)
    source = source.replace('"mean_owned_per_rank_per_layer": 6 / 8', '"mean_owned_per_rank_per_layer": 6 / ep_world', 1)
    source = source.replace('"original_packed_bytes_per_owned_expert_three_projections": source_bytes_per_expert,',
                            '"original_packed_bytes_per_owned_expert_three_projections": source_bytes_per_expert // tp_world,', 1)
    source = source.replace('"logical_FP4_weight_elements_per_expert_three_projections": 3 * 2304 * 5120,',
                            '"logical_FP4_weight_elements_per_expert_three_projections": 3 * 2304 * 5120 // tp_world,', 1)
    source = source.replace('"tile_ring_write_plus_read_bytes_per_owned_expert_if_tile_mode": 3 * 2304 * 5120 * 2 * 2,',
                            '"tile_ring_write_plus_read_bytes_per_owned_expert_if_tile_mode": 3 * 2304 * 5120 * 2 * 2 // tp_world,', 1)
    source = source.replace('"all_rank_six_expert_partition_verified": True,',
                            '"all_rank_six_expert_partition_verified": True, "TP":tp_world, "EP":ep_world, '
                            '"physical_counts_duplicate_each_EP_owner_tp_world_times":True,', 1)
    namespace = dict(base.__dict__)
    exec(compile(source, __file__, 'exec'), namespace)
    actual = namespace['analyze']
    def verified(root):
        for ep in range(4):
            left = json.loads((root / f'decode-scopes-rank-{ep*2}.json').read_bytes())
            right = json.loads((root / f'decode-scopes-rank-{ep*2+1}.json').read_bytes())
            if left['excluded_route_telemetry'] != right['excluded_route_telemetry']:
                raise RuntimeError('TP replicas did not use identical EP expert IDs and ownership')
        return actual(root)
    base.analyze = verified
    return base.main()


if __name__ == '__main__':
    main()
