# Online original-format weight unpack on Ascend

> Publication note: [the evidence publication policy](ascend_decode_optimization.md#published-evidence-and-measurement-identities)
> explains neutral paths/host labels and the separate original/published hash map. Publication cleanup
> does not rerun the recorded measurements.

The new byte-level operators live in relic-core, not in the model runtime. They decode the original
released FP8/E8M0 payload to BF16, and widen original packed E2M1 values exactly to INT8 multiplied
by two with the original scale divided by two. No row-wise amax requantization or offline
replacement checkpoint is used.

## Runtime integration

`checkpoint_weights(..., online_unpack=None)` selects online decoding for **NPU BF16 parameters**.
The default CPU/CUDA load body is unchanged. For an online fill:

1. Read the checkpoint's original E4M3 and E8M0 storage as byte aliases.
2. Apply the tensor-parallel plan to those original codes and the scale grid.
3. Transfer the compact original payload to the parameter's device and decode there through
   `relic_core.kernels.unpack.unpack_fp8_to_bf16`.
4. Attach the file's own FP8 codes to the **actual Parameter object**, not a discarded detached
   wrapper. Attach the tiny original grid decoded to FP32; the NPU never casts an E8M0 container.

This avoids the previous load-time `mma.codes_from_scale` → FP8 numeric cast → `aclnnInplaceCopy`
561103 failure. The original codes are already in the file; recovering them from expanded weights
would be redundant and can lose source exponent information. The online path does not change the
reference GEMM reduction or activation quantization rules.

Explicit `online_unpack=True` refuses a FP32 parameter rather than silently introducing a BF16
intermediate. The runtime validates the release's divisible block-32 layout. Dense projections use
square block scales; row-scaled tables use one exponent per row and 32-wide K block. The generic
core operator supports additional ragged/batched layouts, but those are not attributed to this
released checkpoint.

`fp4_expert_int8(...)` exposes an exact widened representation and its original half-scale grid.
It does **not** invent an NPU routed-expert executor or make that representation compatible with an
arbitrary per-channel INT8 GEMM. INT8 cannot preserve the sign of FP4 negative zero; original packed
bytes remain unchanged.

## Measured acceptance so far

On leased fleet-host-2, Ascend 910B3 with PyTorch **2.10.0** and torch_npu **2.10.0.post7**:

- Core Torch/NPU unpack tests: **68 passed, zero skips**, 6.64 s. These include all 256 packed FP4
  bytes, all 65,536 E4M3/E8M0 code pairs, signed zeros, NaN classification, subnormal output, layouts,
  alias rejection and real CPU/NPU device mismatch checks.
- Runtime integration plus unchanged loader suite: **43 passed, zero skips**, 7.86 s: 24 new cases
  including four real NPU cases, plus 19 existing loader cases. Original source codes and FP32
  scales remain attached to the filled parameter; online loading does not enter code recovery.
- Scalar fused-kernel bit math and mocked launch wiring: **28 passed**, 1.62 s. These are host/unit
  checks, not evidence that an NPU kernel compiled or executed.

The original checkpoint matrix benchmark using the **Torch composition backend** measured ten
synchronized warm iterations:

| Matrix | Input → output | Median unpack wall time |
| --- | --- | ---: |
| `layers.0.ffn.experts.0.w1.weight` | packed `(2304,2560)` → INT8 `(2304,5120)` | **37.03 ms** |
| `layers.0.attn.wkv.weight` | E4M3 `(512,5120)` + E8M0 → BF16 | **27.89 ms** |

These are unpack-only measurements including Python launch overhead, not matmul, expert-forward,
token-per-second, or whole-model parity results. Original payload bytes and checkpoint files were
not rewritten by the benchmark.

## Fused backend acceptance and same-environment benchmark

The final fused Triton Ascend path passed **75/75 actual NPU tests**, zero skips, in **18.84 s**:
68 operator/validation cases, one real nondefault-stream case, and six tail-memory-guard cases.
The stream case consumes the result with the next Torch operation before explicit synchronization;
a separate receipt records matching Torch/Triton raw streams and target `npu / Ascend910B3`.
Finite BF16 bits, signed zeros, scale edge cases and output guards were checked, not merely whether
a kernel could launch.

The fused lane uses a separate private **CPython 3.11.18** environment with the same exact
**torch 2.10.0 / torch_npu 2.10.0.post7**, plus **triton-ascend 3.2.0**. The original CPython 3.12
runtime environment was not overwritten; a CPython-3.11 wheel was not forced into it.

Two failures were retained rather than counted as successes:

1. The public Triton 3.2.0 wheel's host driver referred to `RT_LIMIT_TYPE_SIMT_WARP_STACK_SIZE`, absent
   from installed CANN 9.2. The private copy removes only that optional mapping and reports requests
   as `NotImplementedError`; it does not fabricate a replacement enum. The original file, patch,
   both hashes and rollback command are retained. **Shared CANN headers/libraries, driver and old
   environment were not modified.** An unpatched 3.2.0 wheel on this host fails bootstrap, so the
   measured fused result must not be attributed to an unmodified stock installation.
2. Two stride-2 INT8 output stores triggered a Triton Ascend interleave-optimization compiler
   assertion. The final FP4 kernel emits a single contiguous output vector, uses `output_index//2`
   for its source byte and parity for the low/high nibble. After that change, both exhaustive values
   and sentinel tail-guards passed. The padded lane count selects 64-bit indexing; only actual
   output lanes are written.

A same-private-environment comparison used the same original weight keys and byte hashes, **20
synchronized warm iterations** per backend, including Python launch overhead:

| Original matrix | Torch composition | Fused Triton Ascend | Unpack-only speedup |
| --- | ---: | ---: | ---: |
| FP4 expert `(2304,2560)` packed → `(2304,5120)` INT8 | **35.76 ms** | **15.02 ms** | **2.38×** |
| FP8 attention `(512,5120)` + original scales → BF16 | **27.73 ms** | **7.67 ms** | **3.62×** |

The fused first calls including compilation were **2.22 s / 2.75 s** for those FP4/FP8 matrices.
These are matrix-specific unpack measurements, not a pure device-event rate, exhaustive model
benchmark, or model speedup. The earlier 37.03/27.89 ms Torch figures above came from the original
CPython 3.12 environment and are retained separately, not used as the paired comparison baseline.

## Reproducible evidence

- [Final 75-case fused test log](evidence/ascend-online-unpack-20261117/unpack-npu-triton-20261117T000413Z/npu-triton.stdout.log),
  [machine-readable tests](evidence/ascend-online-unpack-20261117/unpack-npu-triton-20261117T000413Z/pytest.xml)
  and [owned-process cleanup](evidence/ascend-online-unpack-20261117/unpack-npu-triton-20261117T000413Z/summary.json).
- [Torch/NPU 68-case log](evidence/ascend-online-unpack-20261117/unpack-npu-torch-20261116T222606Z/npu-torch.stdout.log)
  and [43-case runtime/legacy-loader log](evidence/ascend-online-unpack-20261117/online-loader-tests-20261116T233625Z/loader-integration.stdout.log).
- [Exact environment and nondefault-stream provenance](evidence/ascend-online-unpack-20261117/fused-stream-environment-20261117T000613Z.json),
  [private wheel/ABI manifest](evidence/ascend-online-unpack-20261117/kernel-env-manifest.json)
  and [private-driver patch/rollback](evidence/ascend-online-unpack-20261117/kernel-env-scripts/triton-cann92-compat/FINAL-compat-receipt.json).
- [Same-environment Torch benchmark](evidence/ascend-online-unpack-20261117/online-unpack-bench-torch-20261117T000213Z/benchmark.json)
  and [fused benchmark](evidence/ascend-online-unpack-20261117/online-unpack-bench-triton-20261117T000114Z/benchmark.json),
  including source weight hashes, all timing samples and named providers.

For a repeat, use a fresh task-private environment with the matching Python ABI, exact Torch/NPU
versions and documented private compatibility patch. Run `tests/test_online_unpack.py` with
`RELIC_ONLINE_UNPACK_TEST_DEVICE=npu:0` and `RELIC_ONLINE_UNPACK_TEST_BACKEND=triton`, plus the
stream and guard tests. The benchmark is `tests/bench_online_unpack.py --backend triton --device
npu:0 --checkpoint <original-directory> --report <fresh-report>`. A new fleet lease is required
before further NPU work; retained files alone are not permission to take the host.

## Final regression, packaging and cleanup checks

The expanded unchanged-semantics regression (loader, MMA reduction, activation quantization/wiring
and expert nibble packing) passed **81 cases**, with **two CUDA-capture tests skipped**, in 10.46 s.
Those skips are not NPU passes. Separately, **45 portable launch/math/packaging tests passed** in
1.69 s. Neither invocation was the full repository test suite.

Python-only relic-core installation was actually tested on the private Ascend environment:
`RELIC_CORE_BUILD_EXTENSIONS=0` produced an installed operator package with no native `.so`
extensions. The default CUDA build remains unchanged; this opt-in does not build an expert/GEMM
provider or make Triton available by itself.

The [final host check](evidence/ascend-online-unpack-20261117/final-operator-host-check.json)
found no live task-owned or private Python processes, unchanged original checkpoint file identities,
unchanged shared CANN environment/link targets, and the same resident Jenkins process. The
[final NPU snapshot](evidence/ascend-online-unpack-20261117/final-operator-npu-smi.log)
shows healthy idle cards and no running NPU processes on all eight. No shared cache deletion,
driver reset, Jenkins stop or CANN reinstall was used. The private compatibility patch and both
private Python environments remain under `<workspace>/relic-online-unpack-20261117-0534`.

`ts_release(host="fleet-host-2")` returned `released: true`; an immediate lease-list read confirmed
[fleet-host-2 absent](evidence/ascend-online-unpack-20261117/lease-release.json). Other agents' leases were
left untouched. Future NPU work requires a new lease even though the private files remain.

## Chitu findings and semantic limits

Primary-source inspection used the official [thu-pacman/chitu](https://github.com/thu-pacman/chitu)
commit `6461680a462a57b73b37ea19a3b74475516ef2f0`. Relevant files were TLS-fetched and verified against
pinned Git blob identities; the repository was not executed or installed.

- The [NPU FP4 dispatcher](https://github.com/thu-pacman/chitu/blob/6461680a462a57b73b37ea19a3b74475516ef2f0/chitu/ops/quant/blockfp4/matmul.py)
  uses external `cinfer_ascendc` grouped GEMM/GEMV with `computeType="fp4"`, multiplicative scales
  and empty offsets. The binary is not shipped in the inspected source tree, and the source does
  not establish its 910B or PyTorch-2.10 ABI support.
- The [FP4 converter](https://github.com/thu-pacman/chitu/blob/6461680a462a57b73b37ea19a3b74475516ef2f0/chitu/ops/quant/blockfp4/convert.py)
  confirms low-nibble-first ordering and E2M1 levels. No exact FP4→INT8 widening implementation was
  found in the inspected relevant conversion/dispatch subset.
- The [soft FP8 conversion](https://github.com/thu-pacman/chitu/blob/6461680a462a57b73b37ea19a3b74475516ef2f0/chitu/ops/triton_ops/quant/blockfp8/convert.py)
  uses integer bit relocation before wider floating computation. Its direct E8M0 `byte << 23`
  branch lacks special cases: byte zero becomes FP32 zero, not `2^-127`, and byte 255 becomes
  infinity, not NaN. Our independent bit/table implementation tests those boundaries explicitly;
  it does not blindly reuse that arithmetic or claim Chitu device parity.

Unpacking correct values does not validate downstream activation quantization, 32-element
partial-dot scale application, FP32 reduction order, routing, logits or generated tokens. The
existing `v41` Ascend capability refusal is unchanged. Completing an expert/GEMM provider remains
separate work. Downstream arithmetic may flush tiny values even when stored half-scale bits are
correct; the current tested NPU fixture preserved the low edge, but that is not a universal claim
about every device/operator.
