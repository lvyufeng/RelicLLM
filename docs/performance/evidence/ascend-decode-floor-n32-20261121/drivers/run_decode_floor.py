"""Bounded real-output ablation of the current TP2/EP4 full decode path."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

import run_expert_tp_trial as driver


def main():
    sys.argv = [sys.argv[0], "tp2-forced"]
    source = inspect.getsource(driver.main)
    source = source.replace('2026-11-20T15:39:50+00:00', '2026-11-21T07:05:44+00:00', 1)
    source = source.replace('original-expert-" + TOPOLOGY + "-host-table-HC', 'original-decode-real-output-floor-tp2', 1)
    source = source.replace('str(ROOT / "source/profile_expert_tp.py")', 'str(ROOT / "source/profile_decode_floor.py")', 1)
    source = source.replace('"run_expert_tp_trial.py", "profile_expert_tp.py",',
                            '"run_expert_tp_trial.py", "profile_expert_tp.py", "profile_decode_floor.py", "run_decode_floor.py",', 1)
    exec(compile(source, str(Path(__file__).resolve()), "exec"), driver.__dict__)
    return driver.main()


if __name__ == "__main__":
    raise SystemExit(main())
