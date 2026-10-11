# Three block-size traps in `relic_core.kernels.ops`

**Status: measured on the sm_75 box, kept as a regression hazard, not a current blocker.** This note
records why the V4.1 activation-quantization fix (`quantized_linear` in
`relicllm/models/deepseek_v4_1/kernels.py`, see `quantized-activation-path.md`) is written in pure
torch and never calls relic-core's `fp8_gemm`/`fp4_gemm`. It is not that those kernels are unused
elsewhere -- it is that on the V4.1 path they are either wrong or unavailable, and two of the three
failures are silent.

The measurements below are from the sm_75 (4x RTX 2080 Ti) box, against `relic_core.kernels.ops`.
They de-risk the Turing leg specifically; the sm_89 numbers live in
`v41_dense_gemm_dtype.md` and `deepseek_v4_1_flash_device_experts.md`.

## The traps

### 1. `fp4_gemm` silently returns wrong numbers on block-32 scales

`ops.py` hardcodes `act_block_size=128` (the V3 convention) at the `fp4_gemm` call site. V4.1 uses
**block-32** activation scales, so handing `fp4_gemm` a V4.1 activation with its own scales produces a
wrong answer with no error:

- measured `max_rel_to_amax` **6.296e-01** -- 34410 bf16 ulp;
- the identical bytes read with a 128-wide scale expansion reproduce the *entire* error, which
  isolates the cause to the block width and nothing else.

This is the one to keep in mind: the activation-quant path this repository is adding is exactly
block-32, and the failure mode is a silent wrong answer on the acceptance surface rather than a
crash. Either thread `act_block_size` through `fp4_gemm` or do not call it on the V4.1 path.

### 2. `fp8_gemm` hardcodes block 128 too

`fp8_gemm` -- and its helper `_dequant_fp8_acts` -- default to block 128 and drop V4.1's
`block_size=32`. `soft_fp8_blockfp8_weight_dequant` *does* take 32 correctly, so the V4.1 GEMM is
assemble-able, just not by calling `fp8_gemm`.

### 3. `impl="triton"` cannot lower e4m3 on Turing

Forcing `impl="triton"` on `blockfp8_act_quant` on sm_75 hard-fails:

```
CompilationError: type fp8e4nv not supported in this architecture
```

Triton on Turing cannot lower e4m3 at all. A knob that can select triton on Turing must therefore be
gated on capability, not on an environment variable -- the capability fork already resolves
`_auto_impl("fp8_quant") -> "torch"` and `_auto_impl("fp8") -> "torch"` at `major < 8`, and
`_choose_compute_dtype -> fp16`.

## Why the fix sidesteps all three

`quantized_linear` computes `dequant(act) @ dequant(W).T` in fp32 and never calls `fp8_gemm` or
`fp4_gemm`. That is what makes it independent of every trap above: it does not pick a block size from
relic-core, and it does not go through a Triton lowering.

The fp32 accumulator is load-bearing, and this is the one place the measurements matter most. Against
the fp64 truth of the *same operands*, the recipe as `quantized_linear` writes it -- fp32 accumulate,
narrow once to bf16 -- is **0.000 ulp**: at these magnitudes the fp32 reduction is already exact
enough that its bf16 rounding is bit-identical to rounding the fp64 result, so the accumulator width
costs nothing. relic-core's own soft GEMM agrees with the recipe bit for bit when it takes the
`"torch"` implementation, and diverges by **592 ulp** (mean 2.97, `max_rel_to_amax` 5.5e-03) when it
takes the `"triton"` one -- which is the width the fp16 path computes at. The explicit
`.to(torch.float32)` in `quantized_linear` is therefore what keeps the soft path on the exact side of
that split; a later "optimization" that lets the accumulator follow the card's default width would
move it to the triton column, not by a little.

## A fourth trap, in the cast rather than the GEMM

A bare `torch.float4_e2m1fn_x2 -> torch.float32` cast asserts on sm_75
(`DynamicCast.h:78 fetch_and_cast false`), independent of any quantizer. `fp4_act_quant` itself is
fine -- hand-decoding its nibbles reproduces the input's amax. Use a levels-gather or
`soft_fp4_blockfp4_weight_dequant` instead of the cast.

## Reproducing

The probes are `scripts/v41_sm75_probes/probe.py`, `probe3.py`, `probe4.py`, `probe5.py` and
`probe6.py`. Each is read-only and touches no model weights. `probe6.py` is the one that produces
finding 1's numbers, and it is the one to run first:

```bash
cd scripts/v41_sm75_probes
PYTHONPATH=<relic-core checkout> python -I probe6.py
```

`probe6.py` prints three lines. Case A feeds `fp4_gemm` a block-32 activation and compares against
the block-32 fp64 truth -- that is the 6.296e-01. Case B feeds it block-128 activation scales and gets
0.000. Case C takes the *same* block-32 codes and reads them with a 128-wide scale expansion, and
reproduces A's error exactly -- which is what pins the cause to the block width rather than to
anything about the fp4 weight, the packing, or the scale values.

Minimal shape of the failure, stripped from `probe6.py`:

```python
x   = (torch.randn(128, 512) * 2.0).bfloat16().cuda()
w4  = torch.randn(512, 512) * 0.02                          # fp4 weight
codes = torch.clamp(torch.round(w4 / 0.02), -6, 6).abs().uint8()
raw = (codes[:, 0::2] | (codes[:, 1::2] << 4)).uint8()      # pack pairs, low nibble first
b   = raw.view(torch.int8).view(torch.float4_e2m1fn_x2)     # (512, 256), logical K=512
ws32 = <e8m0 (512, 16)>                                     # weight scales, block 32

q32, s32 = ref_act_quant(x, 32)                             # (128,512) e4m3 + (128,16) e8m0
out = O.fp4_gemm(q32, s32, b, ws32, torch.float32)          # reads s32 at block 128 internally
```

Both `K % 32 == 0` and `K % 128 == 0` hold, which is why the call does not raise: it degrades
numerically instead of failing a shape check, and the activation's tail scales are silently read
against the wrong blocks.

### Provenance

Measured on the sm_75 (4x RTX 2080 Ti) box, `cuda:0` only, 2026-10-11:

- relic-core `9ce5e2ec4b69a868e422d8020e0a40f5655e68dd` (2026-10-07), clean tree. The `act_block_size=128`
  in `fp4_gemm` and the `_dequant_fp8_acts(a, a_s, 128)` in `fp8_gemm` are both in `relic_core/kernels/ops.py`
  at that commit.
- Python 3.11.14, torch 2.9.1+cu128, triton 3.5.1.
- The reported ulp figures are bf16 ulp against an fp64 truth computed on the identical operands.

One thing here is **inference and not measurement**: `fp4_gemm`'s hardcoded 128 is arch-independent
code, so the silent wrong answer should reproduce on Ada as well as Turing -- `_auto_impl("fp4")`
returns `"triton"` there, but `soft_fp4_raise_to_fp8_blockfp4_gemm` passes `act_block_size=128` down
regardless of impl. It was measured on sm_75 only; treat the sm_89 claim as expected, not observed.

## What this note is not

It is not a claim that the V4.1 port is currently wrong. The fix as implemented never enters these
code paths. It is a record of the hazards the fix was designed around, so that a later change which
*does* reach for a relic-core GEMM on the V4.1 path -- a perf refactor, a fused kernel, a "just call
the library" cleanup -- has this to read first.