"""Current TP2 model with explicit HC FP32 coefficient fusion and original nonlinear code."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import profile_actual_cube as probe
import profile_expert_tp as tp_probe


def main():
    previous_placement = tp_probe.placement.main
    def hook():
        source = inspect.getsource(probe.generate_probe)
        marker = '    m1_binding = None\n'
        if source.count(marker) != 1:
            raise RuntimeError('private HC install boundary moved')
        source = source.replace(marker,
            '    from relicllm.models.deepseek_v4_1.ascend_hc_coefficients import FusedHCDecodeCoefficients\n'
            '    hc_coeff_owner = FusedHCDecodeCoefficients(model,certify_immutable_sources=True,k_tile=1024).install()\n' + marker,1)
        save = '        with (OUTPUT/f"Cube-model-rank-{rank}.json").open("x") as handle:\n'
        if source.count(save) != 1:
            raise RuntimeError('private HC model receipt boundary moved')
        source = source.replace(save,'        row["HC_FP32_coefficients"] = hc_coeff_owner.describe()\n' + save,1)
        cleanup = '        ascend_linear.quantized_linear_ascend = original_linear\n'
        source = source.replace(cleanup,'        hc_coeff_owner.close()\n' + cleanup,1)
        generated=Path(os.environ['RELIC_PERF_OUTPUT'])/f'HC-coefficients-generated-rank-{os.environ["LOCAL_RANK"]}.py'
        with generated.open('x') as stream:stream.write(source)
        exec(compile(source,str(generated),'exec'),probe.__dict__)
        return previous_placement()
    tp_probe.placement.main=hook
    try:
        return tp_probe.main()
    finally:
        tp_probe.placement.main=previous_placement


if __name__=='__main__':
    raise SystemExit(main())
