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
    if source.count(save) != 1:
        raise RuntimeError("private TP receipt save boundary moved")
    if os.environ.get("RELIC_TP_MATCHED_TOKEN_INPUTS") == "1":
        # Exactly the observed EP8 control's context-24 input and output prefix.
        # Greedy selection is still completed and timed, but its result is not
        # fed back. This is same-token-input forward latency, NOT generation TPS.
        probe.MATCHED_INPUTS = [20, 14, 223, 21, 14, 77199, 223, 1457, 16, 1, 1119, 3295, 16, 1, 1119, 3295]
        loop = '        for step in range(TIMED_STEPS):\n            step_start = time.perf_counter()\n'
        if source.count(loop) != 1:
            raise RuntimeError("private uninstrumented timing loop moved")
        source = source.replace(loop,
            '        if TIMED_STEPS != len(MATCHED_INPUTS) or token != MATCHED_INPUTS[0]:\n'
            '            raise RuntimeError("matched input window has an inconsistent initial prefix")\n'
            '        for step in range(TIMED_STEPS):\n'
            '            token = MATCHED_INPUTS[step]\n'
            '            step_start = time.perf_counter()\n', 1)
        source = source.replace(save,
            '        row["same_token_input_latency_not_generation_TPS"] = True\n'
            '        row["fixed_input_token_ids"] = MATCHED_INPUTS\n'
            '        row["single_request_forward_rate_not_generation_TPS"] = row.pop("sustained_single_stream_TPS")\n'
            '        row["NPU_completion_and_host_argmax_included"] = True\n' + save, 1)
        source = source.replace('"sustained_single_stream_TPS", "timed_token_ids"',
                                '"single_request_forward_rate_not_generation_TPS", "timed_token_ids"', 1)
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
