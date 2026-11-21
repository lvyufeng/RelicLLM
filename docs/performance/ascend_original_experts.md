# Original-format experts on Ascend 910B

> Publication note: [the evidence publication policy](ascend_decode_optimization.md#published-evidence-and-measurement-identities)
> explains neutral paths/host labels and the separate original/published hash map. Publication cleanup
> does not rerun the recorded measurements.

This continues [online original-format unpack](ascend_online_unpack.md) with actual expert execution.
It does **not** replace the released checkpoint with W8A8/W4A8 weights or INT8 activations, and it
does not claim native FP4/FP8 arithmetic. The new kernels live in relic-core; RelicLLM owns the
weight cache, expert deal, activation/narrowing order and call sites.

The later [original-checkpoint generation record](ascend_original_inference.md) completes a real
TP8 backbone prefill/decode with host Engram. The measurements and limits below describe this
expert-only stage; they are not retroactively full-model results.

## What is implemented

- `quantize_block_fp8`: BF16/FP32 activation → raw E4M3 bytes and original-style E8M0 bytes, one scale
  per 32 elements. The scale is the IEEE-bit ceiling of FP32 `amax * fl32(1/448)`, after the original
  `1e-4` amax floor. It does not invoke an unsupported NPU float8 cast.
- `fp4_int8_block_gemm`: original FP8 activations × exact widened E2M1 weight values. The weight
  remains `INT8 = 2 * E2M1` plus the **original half-scale** grid. Before the reduction the kernel
  recovers E2M1 values and original scales as a matched pair; it does not mix bare values with the
  half-scale, which would halve each K-block result.
- `fp8_block_gemm`: dense E4M3 operands with original square or per-row E8M0 grids. Source codes are
  read directly, not recovered from an expanded BF16 parameter or requantized.
- `AscendRoutedExperts`: online unpack, a **single process-wide 512 MiB byte-budget LRU**, and cyclic
  expert-id ownership. It is not 512 MiB per layer. An expert larger than the budget is streamed
  without retaining it. Original checkpoint payloads are read-only.
- `ascend_linear`: the dense NPU quantized-linear binding. Missing source codes/scales are an explicit
  error rather than an unreported reference-reduction fallback.

The reduction is the port's explicit model of NVIDIA tensor-core behavior: two 16-term halves,
nominal-exponent anchors, truncation on a window-13 grid, first-half carry into the second-half
window, final renormalization, **separate** FP32 activation and weight rescaling, and K-order FP32
addition. An independent exact-rational oracle and the existing CPU port check it. This is not a
new measurement of the released TileLang kernel on NVIDIA hardware.

The full expert path keeps the original ordering:

1. Independently quantize the input for gate and up projections.
2. Narrow each GEMM output to BF16, then widen to FP32 for the clamp and SwiGLU.
3. Clamp up to `[-limit, limit]` and gate only above `limit`; compute `silu(gate) * up`.
4. Multiply the routing weight into the intermediate activation, not into the final expert output.
5. Cast to BF16 **before** the down projection's FP8 activation quantization.
6. Narrow that expert output to BF16; accumulate routed experts in ascending expert-id order into
   one FP32 sum. Expert-parallel stores set `partial=True` with the same ownership decision.

For NPU stores, constructor or execution errors propagate. The CUDA arena's existing allowed-to-fail
host fallback is not reused. Unsupported CUDA hot/pool/batched options or a non-cyclic deal are
refused by name. Existing CPU/CUDA bodies and the serving capability gate remain unchanged.

## Actual acceptance

The leased host was **fleet-host-2 / Ascend910B3 / CANN 9.2.0 / driver 26.1.0**. All hardware tests used
**torch 2.10.0+cpu / torch_npu 2.10.0.post7** in the retained private **CPython 3.11.18 / Triton Ascend
3.2.0** environment. The private host-driver compatibility patch from the online-unpack run remains
necessary and documented there; no stock-wheel or cross-ABI claim is made.

| Final check | Result | What it establishes |
| --- | --- | --- |
| GEMM and bit-expression suite | **31 passed, zero skips** | Independent FP32-bit parity; all E4M3 × E2M1 codes, scale endpoints, square/row layouts, sequential cancellation, invalid FP32 grids and memory guards |
| Activation quantizer suite | **44 passed, zero skips** | Original rounding contract, signed zero, extreme magnitudes, real NPU output/stream checks and explicit refusal of the NPU Torch-cast path |
| Runtime, unpack and legacy regression suite | **127 passed, 2 skipped** | Expert/SwiGLU/routing/cache/loader/dense integration and unchanged MMA/activation/packing behavior; the two skips are existing CUDA-capture cases |
| Independent rational oracle | **1668 checks passed** | A separate stdlib transcription of the activation and block-reduction contract, including the value/scale factor-of-two compensation |

The invocations above are **scoped suites**, not the entire repository suite. The NPU quantizer suite
contains portable/CPU cases as well as actual NPU comparisons; no skipped case is counted as a
hardware pass. Output sentinels and all scale bytes were checked after actual compilation, not merely
whether a compiler accepted the source.

Two problems were diagnosed rather than hidden:

- The initial CPU test oracle lost negative-zero sign through `Fraction`; the implementation's
  `0x80` code was correct. The oracle was fixed and both signed-zero paths rechecked.
- The first GEMM kernel used unsigned bit fields whose logical shifts lowered to an unsupported
  UINT32→UINT64 vector cast in the tested CANN compiler. Equivalent **signed 32-bit fields plus
  masks** compiled and passed exhaustive scalar-bit and NPU tests. No compiler binary, CANN header,
  driver or shared environment was changed to resolve this kernel issue.

## Original checkpoint-backed forward

The final probe read original FP4 payloads and original per-row E8M0 grids for **layer 0 experts
0, 127 and 383**. Each gate/up matrix has packed shape `(2304,2560)` and each down matrix
`(5120,1152)`, with the reference block-32 scale layouts. Per-tensor byte SHA-256 values and the
nine tested implementation source hashes are in the report.

For a fixed **one-row BF16 activation**, top-k 3 routing and all three projections:

- New NPU provider versus `CheckpointRoutedExperts(quantized=True)`: **0 / 5120 FP32 bit mismatches**,
  maximum absolute error **0.0**, NaN classifications equal.
- Cold and warm NPU output bits agree; ten further warm repeats on each provider were checked.
- Warm synchronized wall-time medians: **CPU reference 588.68 ms**, **NPU provider 111.61 ms** for the
  three-expert routed sum. These are different hardware/provider lanes, not a same-device kernel
  speedup or token-per-second measurement. Python routing/submission overhead is included.
- The NPU cache retained **119,439,360 bytes** for three experts within the probe's explicit
  128 MiB budget. The production default is a shared 512 MiB budget.

The earlier expert-0-only probe also matched all 5120 outputs exactly and measured a **single warm
call of 37.70 ms**; its first NPU call was **35.49 s**, including compilation and staging. That first
call must not be carried over as warm performance, and the one warm sample is not a distribution.

## Reproduction and retained evidence

The operator API and reduction contract are documented in the paired relic-core worktree at
`docs/guides/original_block_gemm.md`; these changes have not yet been committed or pushed.
In a fresh, leased, task-private environment with the same Python ABI and exact stack:

```bash
RELIC_EXPERT_TEST_DEVICE=npu:0 RELIC_EXPERT_TEST_BACKEND=triton \
  python -m pytest tests/test_block_gemm.py tests/test_block_gemm_math.py -q -rs
RELIC_BLOCK_FP8_QUANT_TEST_DEVICE=npu:0 \
  python -m pytest tests/test_block_fp8_quant.py tests/test_block_fp8_quant_math.py -q -rs
RELIC_ASCEND_EXPERTS_DEVICE=npu:0 \
  python -m pytest tests/test_v41_ascend_experts.py tests/test_v41_ascend_bindings.py -q -rs
```

The first two commands run from the relic-core checkout; the third runs from RelicLLM. No environment
resolver should replace the exact Torch/NPU pair. The original released checkpoint and the private
source/env/log trees remain available for a later separately leased run.

- [Final GEMM test log](evidence/ascend-original-experts-20261117/gemm-npu-20261117T044257Z/gemm-npu.stdout.log)
  and [machine-readable tests](evidence/ascend-original-experts-20261117/gemm-npu-20261117T044257Z/pytest.xml).
- [Final quantizer test log](evidence/ascend-original-experts-20261117/quant-npu-20261117T054459Z/quant-npu.stdout.log)
  and [tested source hashes](evidence/ascend-original-experts-20261117/quant-npu-20261117T054459Z/tested-source-hashes.json).
- [Final runtime regression log](evidence/ascend-original-experts-20261117/runtime-npu-20261117T054511Z/runtime-npu.stdout.log)
  and [owned-child cleanup receipt](evidence/ascend-original-experts-20261117/runtime-npu-20261117T054511Z/summary.json).
- [Original three-expert parity and all timing samples](evidence/ascend-original-experts-20261117/original-expert-20261117T054529Z/report.json),
  [independent rational oracle](evidence/ascend-original-experts-20261117/expert_scalar_oracle.py)
  and [oracle self-test log](evidence/ascend-original-experts-20261117/expert-scalar-oracle-selftest.log).
- [Final exact-stack and host integrity check](evidence/ascend-original-experts-20261117/final-expert-host-check-after-final-tests.json),
  [idle NPU snapshot](evidence/ascend-original-experts-20261117/final-expert-npu-smi-after-final-tests.log)
  and [confirmed lease release](evidence/ascend-original-experts-20261117/lease-release.json).

## Remaining limits

- **No full-backbone forward, generated token, service, model TPS or official TileLang hardware
  parity is claimed.** The checkpoint-backed forward here is the complete three-projection expert
  routed sum, not an entire model.
- The `v41` Ascend serving capability guard is **not relaxed**. Remaining model paths still need
  validation/adaptation, including compressed-KV/indexer FP4 activation quantization, Engram decode
  and source-code metadata through attention weight views. Fixing those requires their own tests,
  not removing a guard because an expert works.
- NPU tensor/expert-parallel loader selection is tested at the binding seam and CPU fixture sums;
  this continuation did not execute a new multi-process TP8 NPU model.
- The current routing path reads indices on the host once per call and is **not graph-capturable**.
  CPU versus NPU nonlinear arithmetic is independently checked on fixtures, not declared equal for
  every possible input from one checkpoint row.
- Stored subnormal scale bits are preserved and scale rounding is modeled explicitly. INT8 cannot
  retain FP4 negative-zero sign. Further device/operator FTZ behavior must not be generalized from
  the accepted fixtures.

No checkpoint conversion, native integer activation replacement, shared-cache deletion, Jenkins stop
or device reset was used. All subprocesses were confined to task-private outputs and tracked with
ownership tokens and bounded cleanup. The host is released after the final integrity check; retained
files alone do not grant permission for further NPU work.
