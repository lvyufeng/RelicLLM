"""Bounded complete-FFN global-slot experiment on the original checkpoint."""
from __future__ import annotations

import inspect
from pathlib import Path
import sys

import run_hc_coefficient_model as base


def main():
    if len(sys.argv) not in (2,3) or sys.argv[1] not in ('generated','matched'):
        raise SystemExit('choose generated or matched, optionally16/64 forwards')
    source=inspect.getsource(base.main)
    source=source.replace('profile_hc_coefficient_model.py','profile_global_ffn_model.py')
    source=source.replace('original-HC-coefficient-','original-global-FFN-')
    source=source.replace('"run_hc_coefficient_model.py", "profile_global_ffn_model.py",',
       '"run_global_ffn_model.py", "profile_global_ffn_model.py", '
       '"RelicLLM/relicllm/models/deepseek_v4_1/ascend_global_ffn.py", '
       '"relic-core/relic_core/kernels/global_ffn_slots.py", '
       '"relic-core/relic_core/kernels/_ascend_global_expert_slots_kernels.py",',1)
    exec(compile(source,str(Path(__file__).resolve()),'exec'),base.__dict__)
    return base.main()


if __name__=='__main__':raise SystemExit(main())
