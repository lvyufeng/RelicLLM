"""Same bounded full-checkpoint probe with opt-in host-table/NPU Engram compute.

Exact original prefill/reference runs before the binding is installed. The
original full model and dense/expert experiment are otherwise unchanged. This
private driver changes no serving default and explicitly updates placement
receipts so the former complete-CPU claim is not carried forward.
"""
from pathlib import Path
import inspect
import os

import profile_actual_cube as probe


def main():
    steps = int(os.environ.get("RELIC_PERF_TIMED_STEPS", "16"))
    if steps not in (16, 64):
        raise ValueError("this private benchmark permits exactly 16 or 64 timed forwards")
    probe.TIMED_STEPS = steps
    source = inspect.getsource(probe.generate_probe)
    marker = '    m1_binding = None\n'
    addition = (
        '    device_engram_binding = None\n'
        '    if os.environ.get("RELIC_HOST_TABLE_DEVICE_ENGRAM") == "1":\n'
        '        if prefetch is not None:\n'
        '            prefetch.close()\n'
        '            prefetch = None\n'
        '        from relicllm.models.deepseek_v4_1.ascend_device_engram import HostTableDeviceEngram\n'
        '        device_engram_binding = HostTableDeviceEngram(\n'
        '            model, device=next(model.parameters()).device, backend="triton",\n'
        '            capture_graphs=os.environ.get("RELIC_ENGRAM_GRAPHS", "1") == "1").install()\n'
        '    hc_fold_binding = None\n'
        '    if os.environ.get("RELIC_HC_DECODE_FOLD") == "1":\n'
        '        from relicllm.models.deepseek_v4_1.ascend_hc_decode import FusedHCDecodeFold\n'
        '        hc_fold_binding = FusedHCDecodeFold(model, backend="triton").install()\n'
        + marker)
    if source.count(marker) != 1:
        raise RuntimeError("the task-private install boundary moved")
    source = source.replace(marker, addition, 1)
    old_claim = '"checkpoint_original_and_not_rewritten": True, "Engram_complete_CPU": True,'
    new_claim = ('"checkpoint_original_and_not_rewritten": True, "Engram_complete_CPU": device_engram_binding is None,\n'
                 '               "Engram_original_tables_host": True,\n'
                 '               "host_table_device_Engram": None if device_engram_binding is None else device_engram_binding.describe(),')
    if source.count(old_claim) != 1:
        raise RuntimeError("the former Engram placement receipt is not uniquely identified")
    source = source.replace(old_claim, new_claim, 1)
    cleanup = '        if prefetch is not None:\n            prefetch.close(propagate_errors=False)\n'
    if source.count(cleanup) != 1:
        raise RuntimeError("the model cleanup boundary moved")
    source = source.replace(cleanup,
        '        if hc_fold_binding is not None:\n            hc_fold_binding.close()\n'
        '        if device_engram_binding is not None:\n            device_engram_binding.close()\n' + cleanup, 1)
    save = '        with (OUTPUT/f"Cube-model-rank-{rank}.json").open("x") as handle:\n'
    if source.count(save) != 1:
        raise RuntimeError("the model receipt save boundary moved")
    source = source.replace(save,
        '        row["HC_decode_fold"] = None if hc_fold_binding is None else hc_fold_binding.describe()\n'
        '        row["original_host_engram_state_counters_are_history"] = device_engram_binding is not None\n'
        '        if device_engram_binding is not None and row["exact_fused_original_activation_binding"] is not None:\n'
        '            row["exact_fused_original_activation_binding"]["complete_Engram_CPU"] = False\n'
        '            row["exact_fused_original_activation_binding"]["CPU_quantizer_bypass_preserved"] = True\n' + save, 1)
    exec(compile(source, str(Path(__file__).resolve()), "exec"), probe.__dict__)
    return probe.main()


if __name__ == "__main__":
    raise SystemExit(main())
