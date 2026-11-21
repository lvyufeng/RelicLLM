"""Latest original-checkpoint TP2 timeline, with independent timing/profile windows."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import profile_actual_cube as probe
import profile_current_bottlenecks as scopes
import profile_expert_tp as tp


class TPScopes(scopes.ScopeRecorder):
    def result(self, steps):
        result = super().result(steps)
        shard = self.model.layers[0].ffn.routed
        for row in result['excluded_route_telemetry']:
            row['owned'] = sum(expert % shard.ep_world == shard.ep_rank for expert in row['global_ids'])
        result['expert_topology'] = {'TP':shard.tp_world, 'EP':shard.ep_world,
                                    'TP_rank':shard.tp_rank, 'EP_rank':shard.ep_rank}
        result['notes'].append('Engram is original host table lookup plus NPU WKV/gate; scope label is historical')
        return result


def main():
    scopes.ScopeRecorder = TPScopes
    previous = tp.placement.main
    def placement():
        source = inspect.getsource(probe.generate_probe)
        marker = '        position += 4\n'
        if source.count(marker) != 1:
            raise RuntimeError('the valid context24 profiling snapshot boundary moved')
        source = source.replace(marker, marker +
            '        profile_cache_boundary = snapshot(model)\n'
            '        profile_hash_boundary = front.snapshot_prefix(model.max_seq_len)\n'
            '        profile_first_token = token\n'
            '        def reset_profile_boundary():\n'
            '            restore(model, profile_cache_boundary)\n'
            '            front.restore_prefix(profile_hash_boundary)\n'
            '            sync()\n', 1)
        profile = 'run_kernel_profile(model, candidate_forward, position, first, OUTPUT, reset)'
        if source.count(profile) != 1:
            raise RuntimeError('the independent kernel-profile call boundary moved')
        source = source.replace(profile,
            'run_kernel_profile(model, candidate_forward, position, profile_first_token, OUTPUT, reset_profile_boundary, '
            'prefetch=prefetch, expected_tokens=timed_tokens)', 1)
        generated = Path(os.environ['RELIC_PERF_OUTPUT']) / f'TP2-profile-generated-rank-{os.environ["LOCAL_RANK"]}.py'
        with generated.open('x') as stream:
            stream.write(source)
        exec(compile(source, str(generated), 'exec'), probe.__dict__)
        probe.run_kernel_profile = scopes.run_current_profile
        return previous()
    tp.placement.main = placement
    try:
        return tp.main()
    finally:
        tp.placement.main = previous


if __name__ == '__main__':
    raise SystemExit(main())
