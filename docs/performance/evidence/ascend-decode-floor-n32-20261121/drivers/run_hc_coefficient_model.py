"""Bounded original-checkpoint HC projection experiment; prior native consumer unchanged."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

import run_expert_tp_trial as base


def main():
    if len(sys.argv) not in (2,3) or sys.argv[1] not in ('generated','matched'):
        raise SystemExit('choose generated or matched HC decode, optionally16/64 forwards')
    matched=sys.argv[1]=='matched'
    steps=sys.argv[2] if len(sys.argv)==3 else '16'
    sys.argv=[sys.argv[0],'tp2-forced' if matched else 'tp2',steps]
    source=inspect.getsource(base.main)
    source=source.replace('2026-11-20T15:39:50+00:00','2026-11-21T07:05:44+00:00',1)
    source=source.replace('original-expert-" + TOPOLOGY + "-host-table-HC','original-HC-coefficient-" + ("matched" if TOPOLOGY.endswith("forced") else "generated") + "-tp2',1)
    source=source.replace('str(ROOT / "source/profile_expert_tp.py")','str(ROOT / "source/profile_hc_coefficient_model.py")',1)
    source=source.replace('"run_expert_tp_trial.py", "profile_expert_tp.py",',
        '"run_expert_tp_trial.py", "profile_expert_tp.py", "run_hc_coefficient_model.py", "profile_hc_coefficient_model.py", '
        '"RelicLLM/relicllm/models/deepseek_v4_1/ascend_hc_coefficients.py", '
        '"relic-core/relic_core/kernels/hc_decode_coefficients.py", '
        '"relic-core/relic_core/kernels/_ascend_hc_decode_coefficients_kernels.py",',1)
    exec(compile(source,str(Path(__file__).resolve()),'exec'),base.__dict__)
    return base.main()


if __name__=='__main__':
    raise SystemExit(main())
