"""Fresh-lease TP2 profile with original host tables and native NPU Engram/HC folds."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

import run_expert_tp_trial as base


def main():
    sys.argv = [sys.argv[0], 'tp2', '16']
    source = inspect.getsource(base.main)
    source = source.replace('2026-11-20T15:39:50+00:00', '2026-11-21T07:05:44+00:00', 1)
    source = source.replace('original-expert-" + TOPOLOGY + "-host-table-HC', 'current-profile-" + TOPOLOGY + "-host-table-NPU-Engram-HC', 1)
    source = source.replace('str(ROOT / "source/profile_expert_tp.py")', 'str(ROOT / "source/profile_current_tp2.py")', 1)
    source = source.replace('env.update(RELIC_TILE_CONSUMER_LIBRARY=',
                            'env.update(RELIC_KERNEL_PROFILE="1", RELIC_TILE_CONSUMER_LIBRARY=', 1)
    source = source.replace('"run_expert_tp_trial.py", "profile_expert_tp.py",',
        '"run_expert_tp_trial.py", "profile_expert_tp.py", "run_current_tp2_profile.py", "profile_current_tp2.py", '
        '"profile_current_bottlenecks.py", "analyze_current_bottleneck_profile.py",', 1)
    if 'RELIC_KERNEL_PROFILE="1"' not in source:
        raise RuntimeError('the profile flag was not inserted into the bounded runner')
    exec(compile(source, str(Path(__file__).resolve()), 'exec'), base.__dict__)
    return base.main()


if __name__ == '__main__':
    raise SystemExit(main())
