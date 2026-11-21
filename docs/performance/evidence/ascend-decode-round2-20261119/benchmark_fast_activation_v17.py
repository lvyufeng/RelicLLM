"""The same original-activation bit/timing batteries, with the static-shift fused API."""
import json
import os
from pathlib import Path

import benchmark_fused_activation_v16 as batteries
from relic_core.kernels.fast_fused_fp8_activation_decode import quantize_decode_fp8_fast


def main():
    batteries.quantize_decode_fp8 = quantize_decode_fp8_fast
    output = Path(os.environ["RELIC_PERF_OUTPUT"])
    with (output / "selected-fast-fused-activation-API.json").open("x") as handle:
        json.dump({"API": "quantize_decode_fp8_fast", "same_original_input_batteries": True,
                   "no_changed_activation_precision": True}, handle, indent=2)
    return batteries.main()


if __name__ == "__main__":
    raise SystemExit(main())
