"""Original TP2 residency with one full-FFN collective and separate expert slots."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import profile_actual_cube as probe
import profile_expert_tp as base


def main():
    source=inspect.getsource(base.main)
    source=source.replace('from relicllm.models.deepseek_v4_1.ascend_expert_tp import ExpertTPDispatch',
                          'from relicllm.models.deepseek_v4_1.ascend_global_ffn import FullFFNGlobalDispatch as ExpertTPDispatch',1)
    source=source.replace('capture_graphs=True, restore_residency=False).install()', 'restore_residency=False).install()',1)
    exec(compile(source,str(Path(__file__).resolve()),'exec'),base.__dict__)
    previous=base.placement.main
    def hook():
        candidate=inspect.getsource(probe.generate_probe)
        marker='            graph_driver = NpuDecodeGraphs(model)\n'
        if candidate.count(marker)!=1:
            raise RuntimeError('the original decode graph construction boundary moved')
        candidate=candidate.replace(marker,marker+'            expert_tp_binding.adapt_decode_graphs(graph_driver)\n',1)
        generated=Path(os.environ['RELIC_PERF_OUTPUT'])/f'global-FFN-generated-rank-{os.environ["LOCAL_RANK"]}.py'
        with generated.open('x') as stream:stream.write(candidate)
        exec(compile(candidate,str(generated),'exec'),probe.__dict__)
        return previous()
    base.placement.main=hook
    try:return base.main()
    finally:base.placement.main=previous


if __name__=='__main__':raise SystemExit(main())
