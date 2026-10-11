# sm_75 relic-core GEMM probes

Read-only probes measuring three hazards in `relic_core.kernels.ops` on the sm_75 (Turing) box.
They produce the numbers in `docs/superpowers/notes/relic-core-gemm-block-size-hazards.md`, which is
the write-up; read that first. Nothing here is part of the runtime, and nothing here touches model
weights.

Run any of them with the relic-core checkout on `PYTHONPATH`:

```bash
PYTHONPATH=<relic-core checkout> python -I probe6.py
```

`-I` (isolated mode) is deliberate: these were written and run against an extracted checkpoint tree,
and isolated mode keeps the script's directory and the CWD off `sys.path`.

| Script | What it establishes |
|---|---|
| `probe6.py` | Finding 1: `fp4_gemm` hardcodes `act_block_size=128` and silently returns wrong numbers on V4.1's block-32 activation scales. Its three cases (block-32 usage, block-128 control, and the 128-wide re-read that reproduces the error) are the whole proof. |
| `probe2.py` | The per-element bf16-ulp table: the recipe vs fp64 at two shapes (the <= 0.5 ulp bound), and the fp16-weight-dequant penalty (642 / 539 ulp). This is where the note's 0.5-ulp and ~600-ulp fp16 figures come from. |
| `probe5.py` | The ulp figures: the fp32-accumulate recipe is 0.000 ulp against the fp64 truth at its shape, relic-core's `"torch"` soft GEMM is bitwise identical to it, and its `"triton"` one is 592 ulp. |
| `probe4.py` | Finding 4: a bare `float4_e2m1fn_x2 -> float32` cast asserts on sm_75, while `fp4_act_quant` itself is fine. Cases: `quant_only`, `cast_fp4`, `cast_fp4_cpu`, `fp4_roundtrip`, `fp8_cast`. |
| `probe3.py` | Finding 3: forcing `impl="triton"` on the e4m3 quantizer hard-fails on Turing. Cases: `fp8_triton`, `fp4_act_quant`, `fp4_act_quant_fp32`, `fp4_gemm`, `triton_gemm`. |
| `probe.py` | The capability-fork resolution (`_auto_impl`, `_choose_compute_dtype`) and a broad sweep of which paths work on sm_75. The context for the four findings. |

Measured 2026-10-11 against relic-core `9ce5e2ec4b69a868e422d8020e0a40f5655e68dd`, Python 3.11.14,
torch 2.9.1+cu128, triton 3.5.1, on 4x RTX 2080 Ti (`cuda:0` only).