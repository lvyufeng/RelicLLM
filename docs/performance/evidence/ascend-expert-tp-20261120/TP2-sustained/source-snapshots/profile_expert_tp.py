"""Install expert TP after exact prefill/reference, before candidate graph capture."""
from __future__ import annotations

import inspect
import os
from pathlib import Path

import profile_device_engram as placement
import profile_actual_cube as probe


def main():
    original_generate = probe.generate_probe
    source = inspect.getsource(original_generate)
    source = source.replace("def generate_probe(", "def generate_probe(", 1)
    marker = '    m1_binding = None\n'
    if source.count(marker) != 1:
        raise RuntimeError("private candidate install boundary moved")
    addition = (
        '    expert_tp_binding = None\n'
        '    if os.environ.get("RELIC_EXPERT_TP", "1") != "1":\n'
        '        from relic_core.kernels.ascendc_online_tile_cube import AscendCTileCubeConsumer\n'
        '        from relicllm.models.deepseek_v4_1.ascend_expert_tp import ExpertTPDispatch\n'
        '        tp_consumer = AscendCTileCubeConsumer(os.environ["RELIC_TILE_CONSUMER_LIBRARY"],\n'
        '            sha256=os.environ["RELIC_TILE_CONSUMER_SHA256"], block_dim=24)\n'
        '        expert_tp_binding = ExpertTPDispatch(model, consumer=tp_consumer,\n'
        '            tp_world=int(os.environ["RELIC_EXPERT_TP"]), capture_graphs=True, restore_residency=False).install()\n'
        '        projection_contract = expert_tp_binding.describe()["numerical_contract"]\n'
        '        expert_projection_source = "original_expert_tensor_parallel_tile_consumer"\n'
        + marker)
    source = source.replace(marker, addition, 1)
    cleanup = '        ascend_linear.quantized_linear_ascend = original_linear\n'
    if source.count(cleanup) != 1:
        raise RuntimeError("private projection restore boundary moved")
    source = source.replace(cleanup,
        '        if expert_tp_binding is not None:\n'
        '            expert_tp_binding.close(restore_residency=False)\n' + cleanup, 1)
    save = '        with (OUTPUT/f"Cube-model-rank-{rank}.json").open("x") as handle:\n'
    source = source.replace(save,
        '        row["expert_TP_EP"] = None if expert_tp_binding is None else expert_tp_binding.describe()\n'
        + save, 1)
    # inspect.getsource must see this intermediate source when the placement
    # driver injects the independent Engram/HC owners in its next pass.
    generated = Path(os.environ["RELIC_PERF_OUTPUT"]) / f"profile-expert-tp-generated-rank-{os.environ['LOCAL_RANK']}.py"
    with generated.open("x") as stream:
        stream.write(source)
    exec(compile(source, str(generated), "exec"), probe.__dict__)
    return placement.main()


if __name__ == "__main__":
    raise SystemExit(main())
