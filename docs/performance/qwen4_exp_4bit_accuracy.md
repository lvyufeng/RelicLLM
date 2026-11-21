# 4-bit for the qwen4-exp expert bank: NVFP4 against a Q4-style grid

> **Current status (2026-11-16): the 100 tok/s target is not achieved.** This is a chronological
> engineering log, not a set of equally current conclusions. The historical acceptance ceiling was
> overturned by correcting the MTP pairing, and both verify and draft graphs now run. A subsequent
> audit found a shared gate/up scale defect in the full-bank quantizer; correcting it changes the
> quantized reference and requires new accuracy and token-parity measurements. Historical rates and
> outputs below must not be relabeled as results of that correction. See the final sections for
> measurement scope, complete-output evidence, and open work.

The 256k step leaves 0.87 GiB free per rank, and a correct MTP verify round needs a fixed 4.6 GiB
(measured). The constraint that had to give was weight precision for the expert bank, which is
14.06 GiB a rank: at 4.5 bits a weight it becomes about 7.85 GiB, freeing 6.2 GiB. This records the
measurement that decides *which* 4-bit format, and the error it costs.

Everything below is measured on the real checkpoint (`<checkpoint>`), layers 0 and 20,
eight experts each spanning the shard, three random activations per expert: 48 samples.

## The formats cost the same, and the first comparison was unfair

Both of the candidates are 4.5 bits a weight:

| format | codes | block metadata | total |
|---|---|---|---|
| NVFP4 | 4 bits (E2M1, run through an E4M3 scale per 16) | 1 byte / 16 weights | 4.5 bits |
| Q4-style symmetric int4 | 4 bits (16 uniform steps) | 1 byte / 16 weights | 4.5 bits |

The first version of this comparison gave the int4 arm a float32 min **and** max per block -- 8 bytes
per 16 weights, four times the NVFP4 scale -- and int4 looked 1.75x more accurate. It was not; it had
four times the budget. With one E4M3 scale per block on both sides:

| measurement | NVFP4 (E2M1) | int4 (uniform, E4M3 scale) |
|---|---:|---:|
| weight relative L2, gate projection | 11.30% | 10.31% |
| expert-MLP output relative L2, mean | 18.94% | 17.10% |
| expert-MLP output relative L2, max | 22.43% | 20.69% |
| cosine similarity, worst sample | 0.9746 | 0.9785 |

**At equal bits the two codebooks are near enough to be a wash.** int4 is ahead by about a tenth of
the error, not by a factor, and that is the whole of its advantage.

## Also measured: the scale format barely matters

The kernel that will consume these weights decodes each scale byte with `nvfp4_e4m3`, so the block
scale is an E4M3 value rather than a power of two. The library's `fp4_act_quant` instead rounds
scales to powers of two (the MXFP4 rule). Measuring both:

| block scale rule | weight relative L2 | output relative L2 |
|---|---:|---:|
| power of two (MXFP4) | 11.47% | 18.87% |
| E4M3 (what the kernel reads) | 11.30% | 18.94% |

A tenth of a point apart in opposite directions on the two metrics, which is noise at 48 samples:
with 16-element blocks the block amax is well behaved enough that the scale rule is second-order.
The error is dominated by the 16 levels, not by how the block is scaled. It also means the first
measurement was not pessimised by using the library's power-of-two quantizer -- the format, not the
scale rule, sets the error.

## What this decides

**NVFP4, because it is the format the code already speaks.** The two are within a tenth on accuracy
and identical on footprint, so the tiebreak is engineering: `relic_core` already has

- `_quantize_fp4_codes` and `fp4_act_quant` for the quantization, with tie-to-even and sign-of-zero
  semantics pinned by `test_fp4_tie_rule.py` and `test_fp4_signed_zero.py`;
- `moe_single_token_fp4_forward`, a fused single-token MoE kernel in exactly the expert-MLP shape,
  whose layouts the plan can consume directly: weights `[E, N, K/2]`, scales `[E, N, K/16]` as
  E4M3 bytes.

The GGUF/Q4_K path has better prefill arithmetic (vendored INT8 MMA) but its decode is a general
DP4A GEMV rather than a MoE-fused kernel, so adopting it means writing the qwen4-exp slot-MoE wiring
that NVFP4 gets for free.

## What this does not decide

**Whether the model still produces acceptable output.** A ~19% relative error at the expert-MLP
output is large in itself, and how large it is *for the model* depends on what fraction of the
residual stream the MLP contributes -- which this measurement does not tell us. Two consequences for
the plan:

1. **Token identity cannot be the acceptance test for this change.** Every earlier change in this work
   was accepted only when it reproduced the reference tokens bit for bit. A 4-bit expert bank will
   not, so the gate has to be a quality measurement (logit drift against the FP8 baseline, or
   perplexity on a held-out slice), plus the separate question of whether the user accepts any drift
   at all.
2. **The step-time cost is already known and is not the problem.** The FP4 kernel measures 291.3 us
   against 224.9 us for int8 at this model's expert shape (dim 2560, inter 640, 64 experts, top-10),
   so it is 29% slower on a latency-bound path: about +0.7 ms a step, or +2.4%. That trades against
   a 6.2 GiB saving, which is the point.

## The wiring is not usable as it stands, and that was measured too

Format accuracy is one question; whether the kernel can consume weights we quantized is another, and
it is the one that decides whether any of this is reachable. Feeding the quantizer's own output to
the two FP4 entry points in relic-core, against real layer-0 experts and a dense float32 reference:

| entry point | scale convention tried | magnitude vs reference | cosine | verdict |
|---|---|---:|---:|---|
| `moe_single_token_fp4_forward` | E4M3 byte, `amax/6`, 16-group | **8.59x** | 0.967 | wrong |
| `moe_single_token_fp4_forward` | E4M3 byte, `amax/12` | 3.04x | 0.867 | wrong |
| `moe_single_token_fp4_forward` | E4M3 byte, `amax/24` | 1.29x | 0.587 | wrong |
| `nvfp4_moe_token_forward`, per-expert 1.0 | E4M3 byte, `amax/6`, 16-group | 0.326x | 0.932 | wrong |
| `nvfp4_moe_token_forward`, per-expert 0.5 | E4M3 byte, `amax/6`, 16-group | 0.039x | 0.911 | wrong |

The first row's factor is explained: `fp4_lut_x2` stores the E2M1 levels **doubled**
(`{0,1,2,3,4,6,8,12}`), so gate, up and down each come back 2x too large and the composed answer is
about 2^3 = 8.6. But dividing the scale down to compensate fixes the magnitude and **breaks the
direction** (cosine 0.967 -> 0.587), so it is not a single missing constant. The per-expert scalars on
the second entry do behave multiplicatively as `s^3` (0.326 at s=1, 0.039 at s=0.5 against a predicted
0.041), so that path is closer to a coherent convention, but even the best arm is 3x low with cosine
0.93 -- worse than the format itself explains (the accuracy measurement above puts the same weights at
cosine 0.98).

### Why this is not a one-line fix

Reading the sources rather than guessing, relic-core currently carries **two FP4 conventions at once**:

- a **legacy** one: E8M0 block scales over 32-element groups, E2M1 levels, which is what
  `moe_single_token_fp4_forward`'s docstring still describes and what
  `tests/test_moe_single_token_fp4.py` still constructs;
- a **ModelOpt** one: E4M3 block scales over 16-element groups plus two global scalars, which is what
  the kernels now decode (`fp4_block_scale` parses sign/4-bit-exponent/3-bit-mantissa; the comment
  claiming it is E8M0 is stale) and what `nvfp4_moe_token_forward` exposes.

The two do not agree, and the repository's own check for the legacy entry **does not run**: it builds
`dim/32` scales while the kernel requires `dim/16`, so it raises
`w1/w2 scale K must equal dim/16` before comparing anything. That means there is no working reference
in the tree for either convention, and the 4-bit path cannot be trusted until there is one.

### The convention is now pinned for one entry, and it is still not accurate enough

Repairing `tests/test_moe_single_token_fp4.py` (it built `dim/32` scales while the kernel requires
`dim/16`, so it raised before comparing anything) and moving its reference to the conventions the
kernel actually implements -- E4M3 scale bytes, 16-element groups, and the **doubled** codebook that
`fp4_lut_x2` stores -- makes it pass bit-exactly:

| | |
|---|---:|
| `y_kernel` norm | 59906.06640625 |
| `y_ref` norm | 59906.06640625 |
| max absolute difference | 0.00098 |
| p99 relative difference | 4.1e-06 |

So the contract for `moe_single_token_fp4_forward` is settled: `level * scale` with the level doubled
in the LUT, which means the producer must satisfy `2 * level * scale == level * amax/6`. The trap is
that both halves of that cannot live in the scale: normalising codes against `amax/12` pushes the
normalised magnitude to +-12 and clips it against E2M1's +-6. The working recipe is to choose codes
against `amax/6` and store the byte as `amax/12`.

On real weights that recipe reaches cosine **0.939** and relative L2 0.345 against the dense FP8
reference -- well short of the 0.98 the format itself should allow. The likely cause is E4M3
**subnormal precision**: these experts' block amax is around 0.05-0.2, so `amax/12` is 0.004-0.017
against a smallest normal E4M3 of 0.015625, putting most scale bytes into the 3-mantissa-bit
subnormal range at up to ~12% relative error on the scale.

That is exactly what the two-level entry exists to avoid, so it was tried next.

### The two-level entry does not close the gap either

`nvfp4_moe_token_forward` keeps the block byte O(1) and puts the magnitude in a per-expert float32
scalar, and its `nvfp4_e2m1` returns the true level rather than the doubled one. Recipe: for an
expert whose largest block amax is `A_max`, `w_scale = A_max/6` and `byte = A_block/A_max`, which
makes `level * byte * w_scale == level * A_block/6` exactly and keeps every byte inside (0, 1].

| measurement | result |
|---|---:|
| byte values produced | 25-54 as raw bytes, i.e. ~0.07-0.88 decoded |
| per-expert scale, median | 0.0309 |
| cosine against dense | **0.818** |
| relative L2 | 0.576 |
| effect of `x_scale` = 1.0 versus 127.0 | **none, byte-identical output** |

Two things follow. It is *worse* than the single-level path (0.818 against 0.939) and it is also
systematically low, and the `x_scale` argument -- which the kernel comment describes as the calibrated
activation scale -- changes nothing when varied across two orders of magnitude.

### A defect in relic-core, found from that anomaly

The `x_scale` non-effect is not a property of the format, it is a bug: in `nvfp4_moe_token_forward`,
`x_scale`, `down_scale` and `activation_fp4` appear **only in the argument list and never in the
body**. The wrapper launches `nvfp4_gate_up_kernel` and `nvfp4_down_kernel` with the raw float
activation and no scale, so the two scalars and the W4A4 switch are silently ignored. Anything that
believes it is configuring calibrated activation scaling through this entry is not.

Reading the two kernels settles what the entry actually computes, and it is simpler than the comments
around it suggest: activations stay **float32** through both stages, and a weight element is
`nvfp4_e2m1(code) * nvfp4_e4m3(scale_byte) * w_scale[expert]`, with `nvfp4_e2m1` returning the true
E2M1 level (the LUT is doubled but halved again on read). Only the weights are quantized; it is W4A16,
not W4A8.

Two of the recipe mistakes behind the earlier numbers were mine, not the kernel's, and both are worth
recording because they are easy to repeat:

- normalising the codes against the **block step** is what makes the codebook span its +-6; dividing
  only by `w_scale` makes the step equal the block amax, so the codes use +-1 of +-6 and six times the
  resolution is thrown away (0.818 -> 0.912 cosine on fixing it);
- the kernel applies **one** `w_scale1[local]` to both the gate and the up accumulator, so those two
  matrices must share a per-expert scale (0.912 -> 0.938).

### The mirror settles it: the kernel is wrong, the quantization is fine

A Python replica of what `nvfp4_gate_up_kernel` and `nvfp4_down_kernel` actually compute -- block by
block, `nvfp4_e2m1(code) * e4m3(byte) * w_scale`, SwiGLU, then the down stage -- compared against the
kernel on identical tensors:

| comparison | cosine | norm ratio | relative L2 |
|---|---:|---:|---:|
| kernel vs replica **with the route weight squared** | **1.000000** | **1.0000** | **0.000000** |
| kernel vs FP4 weights, route weight applied once | 0.9492 | 0.3357 | 0.6895 |
| FP4 weights (weight once) vs the float ideal | **0.9884** | 0.9916 | **0.1516** |

Three things are now settled at once.

1. **The replica is exact.** Cosine 1.0 and a relative L2 of zero is not agreement, it is identity, so
   the arithmetic below is what the kernel does and there is no unaccounted term left.
2. **The kernel applies the routing weight twice.** `nvfp4_gate_up_kernel` writes
   `act = silu(g)*u*weights[r]` and `nvfp4_down_kernel` then does `out += acc*weights[r]*w_scale2`.
   The output carries `w^2` instead of `w`, which is why every attempt to make it match a correct MoE
   came back with the direction right and the magnitude low. The FP8 slot path records this exact
   hazard and does not have it: *"No weight here: the kernel's down phase already multiplies each
   slot by `wts` ... Applying it again squares the routing weight and the answer diverges from the
   first token on."*
3. **The quantization was never the problem.** FP4 weights with the route weight applied once give
   **cosine 0.9884 and relative L2 0.152** against the float ideal. That is the number predicted for
   a 4.5-bit codebook (~11% weight error -> ~15% output error) and it is good enough to proceed on.
   The 0.938-0.94 that earlier sections were chasing was the `w^2` bug, not the format.

The fix is one term: drop `* weights[r]` from the gate/up epilogue and let the down stage own it, which
is the convention the FP8 path already documents. It was applied in
`relic_core/csrc/cuda_kernel_impl.cu` and the extension rebuilt (about fifty minutes, one translation
unit), and the mirror was run again as the acceptance test:

| comparison | before the fix | after the fix |
|---|---:|---:|
| kernel vs FP4 weights, weight once | cosine 0.9492, norm 0.3357 | **cosine 1.000000, norm 1.0000, rel L2 0.000000** |
| kernel vs replica with `w^2` | cosine 1.000000, norm 1.0000 | cosine 0.9492, norm 2.9787 |
| kernel vs the float ideal | cosine 0.9376, norm 0.3329 | **cosine 0.988444, norm 0.9916, rel L2 0.151621** |

The two comparisons swap exactly, which is what a correct fix looks like here: the identity moved from
the wrong reference to the right one, and the agreement with the float ideal went from 0.938 to 0.9884.
The repaired single-level test still passes (`max rel diff 4.3e-05`), so the change did not disturb the
other entry point.

**This closes the accuracy question.** FP4 weights on real expert weights reach cosine 0.9884 and
relative L2 0.152 against the float model -- the level a 4.5-bit codebook should reach -- once the
kernel applies the routing weight once. The 4-bit plan is therefore numerically viable, and the
quantization recipe in this document is the one that produces it.

### Where this leaves the validation

The four points of cosine that were unattributed above are attributed now: they are the squared route
weight, not the quantization. With that removed from the comparison the format measures 0.988, and the
recipe in this document (shared gate/up scale, codes against `ablock/6`, byte against
`w_scale`-normalised `ablock/amax`) is the one that produces it.

The next diagnostic is not another recipe tweak. It is a **mirror reference**: a Python re-implementation
of exactly what the two kernels do -- `nvfp4_e2m1(code) * e4m3(byte) * w_scale`, SwiGLU, then the down
stage -- compared against the kernel on identical tensors, in the same style as the repaired
single-level test that now passes bit-exactly. That separates "the kernel computes what it documents"
from "the quantization we feed it is good", which are the two questions currently tangled together.

### What selecting ModelOpt's convention would actually require

`nvfp4_moe_token_forward` takes `x_scale` and `down_scale` as global float32 scalars and one
`w_scale` per expert. The kernel comments say these come from `hf_quant_config.json` and that
`input_scale` was **calibrated**. We do not have that file, and we do not have an NVFP4 checkpoint to
take it from -- the whole reason for converting is that only `<checkpoint>` exists.

So the honest state of the 4-bit plan is:

1. The **format** is measured and acceptable-ish (11% weight error, and int4 is no better at equal
   bits), so the idea is not dead on quality grounds.
2. The **kernel wiring is not available**: two conventions, no working test, and the ModelOpt path
   needs calibration constants we would have to invent.
3. Reproducing ModelOpt's calibration means fitting per-tensor scales for `input`/`down` and
   per-expert scales for the three weight matrices -- a quantization-calibration task in its own
   right, with its own accuracy measurement, before the MoE wiring can even be exercised.

## Wiring the bank into the model: decode path in place, prefill is next

With the kernel fixed and its convention measured, the NVFP4 bank now has a home in the runtime:
`NvFp4SlotExpertMoE` in `moe.py`, selected by `POCKETLLM_QWEN4_NVFP4_EXPERTS=1` (default off, so the
shipping path is untouched). It mirrors `Fp8SlotExpertMoE`'s seam -- `encode_route`'s local-row code,
one fused launch pair a step -- and quantizes each layer's experts on first use, caching the table.
The quantization is the recipe measured above, moved into the class:

- codes chosen against `block_amax / 6`, so the codebook spans its +-6 rather than +-1;
- one E4M3 scale per 16 weights holding `block_amax / expert_amax`, landing in (0, 1] so nothing goes
  subnormal;
- one float32 scale per expert holding `expert_amax / 6`, shared between gate and up because the
  kernel applies a single `w_scale1[local]` to both accumulators;
- a foreign route stays negative, which both kernels skip, so no zeroed-weight dance is needed.

Building it found four things that were wrong in my own code and are worth recording because each is
easy to repeat:

1. the extension handle is on `load_ops()` from `relicllm.runtime.ops`, not on
   `relic_core.kernels.ops` -- the function is registered on the built module;
2. `FP8Tensor` is not subscriptable, so an expert range has to be taken from `.code` *and* `.scale`
   and the scale expanded to the code's last two dims by derived ratios, not by a hardcoded 128 tile;
3. the down table is `[experts, hidden, inter]`, so its row count is the model width -- transposing
   those two allocated the packed tables with 2560 and 320 where 640 and 1280 belonged, which
   surfaced as `size of tensor a (320) must match b (1280)`;
4. `hidden_states.reshape(1, -1)` is only valid on a decode step. A 512-row prefill chunk reaches the
   same method and fails on `reshape_as`, which is the current state:

```
RuntimeError: shape '[512, 2560]' is invalid for input of size 2560
```

So the class covers **decode only** and the next piece of work is the prefill dispatch that
`Fp8SlotExpertMoE` already has (`_grouped_prefill` then `_prefill`). Until that exists the flag cannot
produce a single end-to-end run: prefill always comes first, so no token and no memory number has been
measured through this path yet.

There is also a memory question that predates the dispatch: the builder fills the resident FP8 bank
during preload and only then chooses a backend, so turning the flag on currently costs FP8 *and* NVFP4
(14.06 GiB plus 7.85) rather than replacing one with the other. The room that the whole exercise is
for only appears once the bank is quantized *during* the fill. Both items are prerequisites for the
measurement the objective actually wants.

## End to end: the NVFP4 bank runs, and it is faster than the FP8 one

With the decode/prefill dispatch in place and the FP8 bytes released per layer as it is quantized,
the bank produced a full 8K run: 64 of 64 tokens, no out-of-memory, peak allocated 19.83 GiB.

Both arms were then run back to back with the *same* script, the flag being the only difference:

| 8K, 64 new tokens, graph mode, budget 2048 | NVFP4 | FP8 |
|---|---:|---:|
| streaming tok/s | **52.50** | 48.67 |
| emit ITL p50 | **18.17 ms** | 20.09 ms |
| tokens emitted | 64/64 | 64/64 |
| identical token prefix | **56 of 64** | -- |
| peak allocated | 19.83 GiB | -- |

The two arms agree on the first **56 tokens** and part ways at index 56 (`1414` against `11316`),
which is the behaviour 4-bit weights should have: a small logit perturbation that a greedy decode
tracks exactly until a near-tie tips the other way. It is a much better quality result than the
layer-level cosine alone predicted, and it is the number to quote for this horizon -- with the caveat
that 56 tokens is a short horizon and says nothing about divergence at a few hundred.

The direction is the opposite of the estimate this document opened with. A microbenchmark priced the
FP4 kernel 29% slower than the int8 one at the same shape, which would have been about +2.4% on the
step; the step instead got **7.4% faster** (48.87 -> 52.50 tok/s, p50 20.17 -> 18.17 ms). The reason
the estimate was wrong is the same one recorded in the performance page: a micro benchmark prices an
op, not a step. The FP4 kernel moves half the weight bytes, and the step is latency-bound on exactly
that traffic, so the arithmetic that made the op slower does not survive into the step.

The outputs also start with the same ids the FP8 path produces -- `1596, 1144, 4087, 1156, 579, 1622,
13, 14235, ...` -- which is a consistency check rather than an accuracy claim: 4-bit weights do change
tokens eventually (cosine 0.988 at the expert output), and any statement about quality has to come
from the layer-level measurement above, not from a matching prefix.

Two things had to be fixed to get here, both found by running rather than by reading:

- **The FP8 bank was still resident.** The builder fills the resident bank during preload and only
  then chooses a backend, so turning the flag on held 14.06 GiB of FP8 *and* 7.85 GiB of NVFP4 and
  the run died at 22.94 GiB of 23.52 on an 800 MiB request. The fix is to drop a layer's FP8 bytes
  as soon as its NVFP4 table exists, which keeps the peak at the FP8 bank's own size and lets it
  decline from there -- 14.06 + 0.17 at the worst point instead of the sum of the two banks.
- **Prefill had no path.** `nvfp4_moe_token_forward` is single-row by construction
  (`x.size(0) == 1`), so unlike the FP8 slot kernel there is no grouped form to hand a prefill chunk
  to. It currently loops a token at a time through the same kernel, which is correct and slow, and is
  a placeholder for the K-row extension the MTP verify round needs anyway. That is why the 8K run is
  reported and a 256K one is not yet: at roughly 33 us a kernel call the loop would need a couple of
  hours to prefill 256K, against a few minutes for the decode it is meant to measure.

## The expert kernel now takes K rows, and it is exact

The single-row limit was the one thing standing between the NVFP4 bank and both a usable prefill and a
capturable MTP verify round, so the kernel gained a K-row form in relic-core:
`nvfp4_moe_rows_forward` takes `x [rows, hidden]` and per-row routes `indices, weights [rows, k]`.
The implementation is deliberately not a loop inside the kernel -- one row is one block in the grid's
`z` dimension -- so a 512-token chunk is two launches instead of 1024, and the row index threads
through the four places that were implicitly zero before: the shared-memory load of `x`, the route
pointers, the `act` offsets and the `out` offsets. `nvfp4_moe_token_forward` is now the `K=1` wrapper
around it, which keeps the decode path and the diagnostic entry points unchanged.

The bar for this change is exactness, not closeness, because the arithmetic did not change: only which
row a block reads. Measured on real layer-0 experts with four different tokens and four different
route sets:

| check | result |
|---|---|
| 4 rows in one call versus 4 single-row calls | **max abs difference 0.0, `torch.equal` true** |
| `rows=1` versus `nvfp4_moe_token_forward` | **max abs difference 0.0, `torch.equal` true** |
| a row whose routes are all foreign | 0.0, contributes nothing, as in the single-row path |

And end to end, which is the check that matters for the model: the 8K run through the K-row prefill
produces `output_sha256 b14d7ec2724abcc9e9284b8835ed6ee126b64218932a556484ce42d0be33dba2`, **byte for
byte the same as the token-loop run**. So the prefill rewiring changed no token; it only stopped paying
a launch per token per layer.

| 8K, 64 tokens | row loop | K-row |
|---|---:|---:|
| streaming tok/s | 52.50 | 50.65 |
| emit ITL p50 | 18.17 ms | 18.19 ms |
| load | 88.9 s | 82.6 s |
| output sha256 | `b14d7ec2...` | `b14d7ec2...` (identical) |

The p50 is unchanged to two decimal places, which is the expected result since decode still goes
through the one-row entry; the difference in streaming tok/s is that metric including the prefill, and
both figures sit inside the run-to-run spread this box shows (32.9-34.3 tok/s at 256K on the FP8 arms).
What the K-row form buys is wall time on the prompt, which is the thing that made a 256K measurement
unreachable rather than merely slow.

## 256K at the native budget, with 6.15 GiB a card handed back

The measurement the whole exercise was for. 262015-token prompt, 64 new tokens, native
`indexer_budget=2048`, graph mode, the NVFP4 bank for all 48 layers:

| 256K, 64 new tokens | NVFP4 | FP8 baseline |
|---|---:|---:|
| streaming tok/s | **35.07** | 33.94 |
| emit ITL p50 | **28.07 ms** | 28.98 ms |
| peak allocated | **16.34 GiB** | 22.15 GiB |
| **free after the step** | **7.02 GiB** | 0.87 GiB |
| tokens emitted | 64/64 | 128/128 |
| required execution mode ran | true | true |

**Free memory went from 0.87 GiB to 7.02 GiB, +6.15 GiB**, against the ~6.2 GiB the plan predicted
from the format arithmetic (14.06 GiB of FP8 experts at 4.5 bits/weight against 7.85 GiB at 4.5
bits/weight with the scale bytes counted). The context, the budget and the decode are unchanged;
the outputs start `1596, 1144, 4087, 1156, 579, 1622, 13, 14235, ...`, the same ids the FP8 path
produces for this prompt.

The step also got slightly *faster*, not slower: 35.07 against 33.94 tok/s and p50 28.07 against
28.98 ms, both inside the 32.9-34.3 tok/s spread this box shows on FP8 arms, so the honest statement
is "no slower" rather than a claimed 3%. What matters is the 6.15 GiB.

### The allocation order was the thing that made 256K unreachable

Two out-of-memory runs came first, and the second one is the instructive one. Quantizing a layer on
first use frees its FP8 bytes, which is enough at 8K but not at 256K, because the **long-context arena
is allocated before the first token**: the FP8 bank (14.06 GiB) and the arena (about 8 GiB) are
resident together, so the run died at 22.66 GiB allocated with 800 MiB asked for, before a single
layer had been quantized. The earlier 8K run had hidden this, because there the arena is small enough
for both to coexist.

The fix is ordering, not arithmetic: quantize every layer in the constructor, before the arena asks
for anything. Peak then is the FP8 bank plus one layer's quantized table, declining per layer, and by
the time the arena is placed the resident set is 7.85 GiB instead of 14.06. That is why the measured
free memory at 256K is now larger than the FP8 bank ever was.

### What this unblocks

The MTP head needs a fixed 4.6 GiB that does not scale with context and did not fit beside an FP8
bank at 256K (0.87 GiB free). It fits in the 7.02 GiB this leaves. Together with the K-row kernel --
which is what makes the verify round capturable, since a verify round is K candidate rows through the
same expert kernel -- the two prerequisites for the last step are now both in place:

- current: 35.07 tok/s at 256K;
- target: 100 tok/s, i.e. **2.85x**;
- MTP measured at 1.33 tokens a pass, so 2.85x needs roughly 3.8 tokens a pass, or a cheaper verify
  than the 1.33 figure was taken with.

## The MTP head now fits -- and that is where the good news ends

Step 4 needs a draft head that is a fixed 4.6 GiB, independent of context, beside a bank that at 256K
left 0.87 GiB free on FP8. The first thing the freed memory buys is that the head is now measurable at
all, and the comparison is blunt:

| 8K, 64 tokens, draft head loaded | NVFP4 bank | FP8 bank |
|---|---|---|
| peak allocated | **16.22 GiB** | 22.66 GiB |
| free | **8.66 GiB** | 0.20 GiB |
| outcome | **ran to completion** | **out of memory** (`200 MiB` request) |

So the FP8 configuration cannot run the head at 8K at all -- that is the historical blocker this whole
exercise was aimed at, reproduced here. The NVFP4 bank runs it with 8.66 GiB to spare, which is the
memory half of step 4 delivered and measured.

The other half is not delivered, and the numbers say so directly:

| 8K, 64 tokens | plain decode | MTP draft=1 | draft=2 | draft=3 |
|---|---:|---:|---:|---:|
| decode tok/s, graph mode | **18.42** | 1.19 | 1.07 | 1.10 |
| decode tok/s, eager | 1.42 | 1.21 | 1.48 | 1.44 |
| tokens per trunk step | 1.0 | 0.89 | 1.14 | 1.14 |
| draft accept rate | -- | 0.78 | 0.79 | 0.64 |
| matches plain decode | -- | **no** (token 4) | **no** (token 5) | **no** (token 18) |

Three things are wrong, and they are separable.

1. **`tokens_per_trunk_step` is about 1.1, not the ~3 the objective wants, and not even the 1.33 this
   project measured before.** `trunk_steps` is 2x `draft_rounds` in every arm, which is the shape of a
   round that runs the trunk twice instead of replaying one captured step -- and consistent with that,
   the graph-mode arms are *15x slower* than plain decode rather than faster. The accept rates
   themselves are decent (0.64-0.79), so the drafts are not the problem; the cost of a round is.
2. **The output does not match plain decode**, which `generate_mtp_draft` promises it always will: a
   rejected draft is supposed to leave the trunk having produced the token an unspeculated decode
   would have. Divergence at token 4/5/18, varying with draft count, is a rollback that does not
   restore exactly what it thinks it restores. **It is not the capture**: it happens with
   `graph_window=0` too, so it is independent of the recording this task spent effort making possible.
3. Neither mode is faster than plain decode, so there is nothing here to tune yet -- the round cost
   has to be fixed before an accept rate can be turned into throughput.

The honest status of step 4 is therefore: the memory prerequisite is met and measured, the head runs,
and the speculation itself is not yet worth taking. The 100 tok/s target is not close: plain decode at
8K is 18.42 tok/s in this configuration and 52.50 in the ITL-measured one, and MTP would have to
multiply whichever number by ~2, not survive it.

One diagnostic is worth recording because it narrows item 2 cheaply: a verify round runs K rows
through the trunk in one pass, while plain decode runs one row. The expert kernel is bit-identical
either way (measured above), but attention and the gated-delta recurrence are not obviously
row-invariant in floating point, so a token drafted in a K-row pass can be judged against a one-row
answer that differs in the last bits. That would produce exactly this signature -- divergence that
depends on the number of drafts, not on the context -- and it is testable by diffing a single trunk
step against the same token decoded alone, without any speculation involved.

## Why a speculative verify cannot be token-identical on this model

The MTP divergence in the previous section was not the capture and not the rollback. Two probes
settle what it is.

**First, the trunk step is not row-invariant.** The same token, fed as the only row of a step and then
as row 0 of a K-row step, against two identically prefilled caches:

| rows in the step | max abs | mean abs | cosine | argmax |
|---|---:|---:|---:|---|
| 2 | 2.31 | 0.350 | 0.9784 | equal |
| 3 | 2.83 | 0.423 | 0.9692 | equal |
| 4 | 2.71 | 0.438 | 0.9663 | **flips** (308 -> 25) |

The control is clean: the *same* K-row batching used for both prefills agrees exactly
(max abs 0.0), so the harness is not the source, and the divergence grows with K.

**Second, it is shape-dependent and not a future leak.** Holding row 0 fixed and changing *only* the
second row's token:

| change | effect on row 0 |
|---|---|
| second row `760` -> `1421` | **max abs 0.0, cosine 1.0000** |
| batched at all, versus alone | max abs 2.31, cosine 0.9784 |

So row 0 does not see the rows after it -- the causal mask is doing its job -- but its own answer
changes purely because the batch has more than one row. That is batched-versus-single GEMM arithmetic,
not a mask bug.

### What that means for speculation

A cosine of 0.978 from a few rows of reassociation is too large to be the reassociation itself. This
model makes **two discrete selections per layer** -- top-10 of 512 routed experts, and the indexer's
top-k -- and a selection is discontinuous: a perturbation of one part in a thousand can move a
near-tie and swap which expert computes the token. After the swap the arithmetic is genuinely
different rather than slightly reordered, which is how a batched pass ends up 2% away from the
single-row one.

The consequence is not a bug to fix but a property to design around. `generate_mtp_draft` guarantees
its output equals plain decode's because a *rejected* draft leaves the trunk having produced what an
unspeculated decode would produce. That guarantee needs the verify pass's arithmetic for a position to
equal a one-row decode's arithmetic for that position, and this model does not have that property:
judging K drafts in one pass changes which experts fire, which changes the token at a near-tie, which
accumulates into the drift that was measured at token 4, 5 and 18 for one, two and three drafts.

So the objective's own rule -- token-changing changes do not count as done -- rules out the obvious
route to 100 tok/s here. Batched verification cannot be token-identical on this model, and verifying
one row at a time is plain decode with extra bookkeeping. Closing that gap would need the discrete
selections to be shape-invariant (compute routing and indexer top-k exactly as a one-row step would,
for every row of the batch), which is a redesign of how the batch is formed rather than a tuning
change, and it would have to be built on a model whose accept path is already cheap -- which, per the
2x trunk steps measured above, this one is not.

### The same probe on the FP8 bank: worse, so it is the model and not the 4-bit weights

| rows in the step | NVFP4 cosine | FP8 cosine | NVFP4 argmax | FP8 argmax |
|---|---:|---:|---|---|
| 2 | 0.9784 | **0.9609** | equal | **flips** |
| 3 | 0.9692 | **0.9470** | equal | **flips** |
| 4 | 0.9663 | **0.9381** | flips | **flips** |

The FP8 bank is *less* row-invariant than the NVFP4 one, and its argmax flips at two rows where the
NVFP4 one holds to three. So this property was there before the expert bank changed and the 4-bit
weights are not responsible for it -- if anything they are steadier, because four bits quantize the
weights to a coarser grid and a coarser grid moves fewer near-ties.

### It is expert routing, and the context length says so

At the 1024-token context this probe used, `indexer_budget=2048` is larger than the number of
positions, so the indexer's top-k keeps **every** token and cannot be the selection that flips. That
leaves the routed experts' top-10-of-512 per layer as the only discrete decision in the step, and it
is made on a router matmul whose result differs in the last bits between a one-row and a K-row GEMM.
One part in a thousand on the router margin is enough to exchange the tenth and eleventh expert, and
from that point the layer is computing with different weights -- which is the measured 2-4% cosine,
and the argmax flips that come with it.

That also names the one route that could still make a verify round token-identical: make the
*selection* shape-invariant even if the arithmetic is not, by computing each row's router logits the
way a one-row step would and using that choice for the batched expert pass. Selecting the same
experts removes the discontinuity; the remaining batched-versus-single difference is then ordinary
floating-point reassociation, which is small enough not to move an argmax over a few rows. It is a
real change to how a verify batch is formed rather than a tuning knob, and it should be tried on a
model whose accept path is already cheap -- this one spends two trunk steps a round, so speculation is
currently a loss before any of this matters.

### The rest of the attribution: it is the attention dispatch, and the expert kernel is cleared

The routing hypothesis was tested next, by recording every `TopKRouter` call for the same token fed
alone and as row 0 of a two-row step:

| measurement | result |
|---|---|
| layers that choose a **different expert set** when the row is batched | **19 of 48** |
| layers whose top-k differs only in order | 38 of 48 |
| example, layer 1 | `{225}` alone, `{383}` batched |

So batching does move the expert choice, exactly as the router-margin argument predicted. But making
the selection shape-invariant -- routing a small batch one row at a time so each row sees the same
M=1 GEMM it would see alone -- **did not restore row invariance**:

| rows | cosine, as found | cosine, invariant router |
|---|---:|---:|
| 2 | 0.9784 | 0.9778 |
| 3 | 0.9692 | 0.9805 |
| 4 | 0.9663 | 0.9680 |

And forcing the *expert kernel* back to one row a launch changed the result by **nothing at all** --
bit-identical to sixteen digits. That is a useful negative twice over: it clears the expert kernel as
a source (it is bit-exact either way, which is what the acceptance probe already said), and it leaves
the rest of the step.

What is left is a dispatch on token count, and there are three of them:

- `attention.py:476`, `_qsa_splitk_predicate`: when `query_rows == 1` and `selection == 1` and the
  selection is at most 4096 wide, the **split-K QSA kernel** runs; for two or more rows the generic
  path runs. Split-K divides the reduction across blocks and combines the partials, which sums in a
  different order by construction.
- `attention.py:256`, `_stage_selected`: one query makes the selection order the identity. This one is
  inactive here -- it belongs to the host-KV path, which these runs do not use.
- `model.py:980`: `seq_len == 1` with `by_outer` scheduling interleaves layers differently. Also
  inactive here, since these runs set `by_schedule=by_layer`.

So the chain is complete and it ends in a deliberate optimization: **a deck of one row uses a
decode-specialised attention kernel and a deck of K rows does not.** The last bits differ, MoE routing
amplifies that into a different expert at 19 of 48 layers, and the logits land 2% apart. Nothing in the
expert path, the 4-bit bank, or the rollback is responsible.

### What that means for the objective

A verify round is token-identical to plain decode only if the arithmetic for a position is the same in
both. Here it provably is not, and the reason is the very kernel that makes decode fast. The two ways
to close it are both costly and neither is a tuning change:

- extend the split-K QSA kernel to K rows, so a verify row gets the same reduction order a decode row
  gets -- a new kernel, and it also has to survive the graph capture;
- or drop split-K from decode so both paths share the generic one -- which gives back the decode
  speedup the whole performance table rests on.

And either would still have to answer the cost problem measured earlier: the accept path spends **two
trunk steps a round**, so speculation is a loss before token identity is even considered. Step 4 of the
objective therefore does not have a path that satisfies its own "no token-changing changes" rule with
the accept path as it stands.

## The accept path costs two trunk steps a round, and that caps the whole idea

The measured `tokens_per_trunk_step` of 0.89-1.14 is not a slow round; it is the round's definition.
`generate_mtp_draft` makes **two** trunk forwards per round:

- `runtime.py:1918`, `model.forward(consumed)` -- the trunk consumes the token it just chose, which
  produces the next token *and* the state the head drafts from;
- `runtime.py:1987`, `model.forward(verify_tokens)` -- one forward over every draft, which decides the
  accepted prefix.

and emits `1 + accepted_run` tokens for those two steps, so

```
tokens_per_trunk_step = (1 + accept_rate) / 2
```

which is what was measured: accept 0.78 gives 0.89, accept 0.79 gives 0.90, and the arms that accept
about 1.3 drafts give 1.14. The design is capped at `(1 + draft_tokens) / 2` even at **perfect**
acceptance -- draft 3 could reach 2.0 and no configuration reaches the 3 the objective asks for.

### The redundancy is visible in the code

`runtime.py:1987` asks for `return_streams=True` and then throws the streams away
(`verify_logits, _ = ...`), while `runtime.py:1918` spends a full forward re-consuming a token whose
successor the verify pass already computed: `chosen = [next_id] + rows` covers `K+1` positions, so the
verify forward's own logits already contain the bonus token that the next round's `next_id` is taken
from.

A loop that emits from the verify pass instead -- emit, draft from the carried state, verify once,
take `accepted_run + 1` tokens including the bonus -- costs **one** trunk step a round and yields
`1 + accepted_run`. At draft 3 and the 0.6-0.8 accept rates measured above, that is 2.5-3.4 tokens a
step, which is the objective's number.

Two things make it more than a reshuffle. The state a round drafts from has to be the *verify pass's*
stream at the accepted position, not the one from a separate forward, so the per-position gated-delta
state that already exists (`_begin_state_log`, `_rewind_to`, `_end_state_log`) has to be what the draft
reads. And the whole round still has to be token-identical to plain decode, which the previous two
sections show it currently is not, for a reason (the split-K QSA dispatch) that has nothing to do with
this loop.

So step 4 has two independent blockers, and this one is the cheaper: it is a restructure of the
speculation loop in `runtime.py`, not a kernel. It is also the one that decides whether the other is
worth doing -- a round that costs one step breaks even at 2 tokens and wins at 3, while a round that
costs two breaks even at 4 and cannot get there.

### Two more hypotheses eliminated, and the source already documents the conclusion

The split-K QSA dispatch was the best remaining suspect, and it is switchable
(`POCKETLLM_QWEN4_QSA_SPLITK=0`), so it was tested. Turning it **off** did not restore row
invariance -- it made it slightly worse (cosine 0.953 at two rows against 0.978 with it on) and moved
the single-row baseline's own answer (argmax 308 -> 25), so it changes both sides rather than being the
asymmetry.

At that point the answer was already written in `attention.py`, at the gated-delta dispatch:

> Preferring the recording path when a log is open looks like the fix for the rollback problem below --
> `state_log` is unreachable while the fused scan is taken ... It was tried and it is *not* the fix:
> forcing the torch path changes the verify forward's own arithmetic, so its rows stop agreeing with
> what plain decode would have answered and even draft width 1 -- where nothing needs rolling back --
> started diverging, at position 3 instead of not at all. **The two arms differ in more than state
> bookkeeping**, so the log cannot be had by switching this branch; the fused scan stays.

So the maintainers of this model hit the same wall from the other side: a verify forward and a decode
forward do not agree, the disagreement is not about state bookkeeping, and the resolution they chose
was to make the *state* exactly recoverable (the fused scan can hand over the per-position states)
while leaving the arithmetic difference in place.

That closes the attribution. The chain measured here -- cosine 0.978 at two rows, content-independent,
not the router, not the expert kernel (bit-identical either way), not split-K, and diagnosed in the
source as a decode-versus-verify arithmetic difference -- is one finding reached two ways.

### Therefore

A verify round cannot be made token-identical to plain decode on this model without unifying the
decode and verify arithmetic of the gated-delta recurrence, which the code records as having been
attempted and abandoned because forcing one path breaks the other. Since the objective says a change
that alters tokens does not count as done, **step 4 cannot be satisfied as specified**, independently
of the separate fact that the accept path also costs two trunk steps a round.

What is still worth doing is the part that does not depend on token identity: restructuring the
speculation loop to one trunk step a round would take `tokens_per_trunk_step` from `(1+a)/2` to
`1+a`, i.e. from about 0.9 to about 1.8 at the measured accept rates, and to about 3 at three drafts
and a 0.75 accept rate. That is real progress on the objective's throughput target, and it is
reportable as such, but it cannot be reported as *done* under the objective's own rule.

## Restructuring to one trunk step a round: the step count halves, then the drafts stop landing

`generate_mtp_draft_onestep` was written to the plan above -- the emitted token is fed to the verify
pass together with the drafts, so a round costs one trunk forward instead of two. Both loops were then
run back to back at 8K, eager, 64 tokens, three draft widths:

| arm | rounds | trunk steps | accept rate | tokens / step | matches plain |
|---|---:|---:|---:|---:|---|
| old, draft 1 | 36 | 72 | **0.778** | 0.889 | no (token 4) |
| old, draft 2 | 28 | 56 | 0.786 | 1.143 | no (token 5) |
| old, draft 3 | 28 | 56 | 0.643 | 1.143 | no (token 18) |
| **new, draft 1** | 57 | **57** | **0.105** | 1.123 | no (token 24) |
| **new, draft 2** | 56 | **56** | **0.107** | 1.143 | no (token 24) |
| **new, draft 3** | 55 | **55** | **0.127** | 1.164 | no (token 24) |

The restructure did exactly what it was meant to: **one trunk step a round**, 57 rounds against 57
steps where the old loop spent 72 steps on 36 rounds. So the `(1 + a) / 2` cap is gone and
`tokens_per_trunk_step` is now `1 + a` by construction.

But the accept rate fell from 0.78 to 0.11, which gave the saved step straight back: 1.12 tokens a step
either way. The cause is bookkeeping in the new loop, not the idea -- the head's cache is advanced once
per *drafting* call, while only `1 + accepted_run` of those positions become final, so it drifts by
`len(drafts) - accepted_run` each round and quickly drafts against a history the sequence does not
have. The old loop names the discipline this needs in its own comment: "the accept and reject paths
both end by feeding the head the same pair -- `(state at p, token p)`". The new loop restores the head
only on a **full** rejection, so a partial accept leaves it misaligned, and that is precisely the case
the old loop's pair-feeding existed to handle.

So the remaining work on this path is narrow and known: after each round, put `mtp_cache` back to the
settled prefix and feed it the accepted run plus the bonus pair, exactly as the reject path does. Until
that is done, the new loop is a correct-shape but worse-accuracy version of the old one, and it stays
opt-in (`generate_mtp_draft_onestep`, not the default path, which is unchanged).

### The trace that explains the collapsed accept rate, and the shape the fix has to take

Tracing both loops at a 1024-token context, one draft:

```
old   past=1025 drafts=[1144]        chosen=[1144]   accepted=1
new   past=1024 feed=[1596, 1596]    rows=[1144, 1144]  accepted=0
```

The new loop's draft is **1596**, the token it had just emitted -- the head was asked for the position
that was already known, one earlier than the drafts it needed to produce. That is a structural
consequence of the restructure, not a bookkeeping slip: after a round finishes, the newest trunk state
in hand is the one from *before* the bonus token, so the first head call can only reproduce the bonus.

The old loop avoids this by spending a forward to consume the emitted token, and that is why its draft
is 1144, the right token for the next position. So the two-step structure was not redundant after all --
which also means the `(1 + accepted) / 2` cap derived earlier stands for that loop.

### What the same trace says the one-step loop must do instead

The head is autoregressive, so it does not need the trunk to consume the bonus before it can draft past
it: given the trunk state and token for position `q`, its first call predicts `q + 1` (the bonus, which
is already known and can be checked and discarded), and its second call -- fed its own state and the
bonus -- predicts `q + 2`, which is genuinely unknown. So a round can be:

1. draft `K + 1` tokens from the carried state, discarding the first (it should equal the known bonus);
2. feed the bonus plus the `K` useful drafts in one trunk step;
3. accept the leading run of `draft[j] == rows[j - 1]`, exactly as the old loop compares them.

That costs one trunk step and yields `1 + accepted` tokens, with `K` useful drafts instead of `K + 1`.
At four drafting calls, a 0.6 accept rate and three useful drafts, that is about 2.8 tokens a trunk
step -- the objective's number -- which the two-step loop cannot reach at any accept rate.

The implementation is a small change to `generate_mtp_draft_onestep`: keep the first head output,
compare it to the emitted token as a check, then build the feed from the bonus and the remaining
drafts. It has not been measured yet, so it is not claimed.

### The aligned one-step loop: measured, and it is a real gain

Drafting one position earlier (the first head output is the already-known token, so it is a check
rather than a draft) and feeding `[emitted] + drafts` in one pass:

| arm | rounds | trunk steps | accept rate | tokens / step | decode tok/s | first divergence from plain |
|---|---:|---:|---:|---:|---:|---:|
| old, draft 1 | 36 | 72 | 0.778 | 0.889 | 1.217 | token 4 |
| old, draft 2 | 28 | 56 | 0.786 | 1.143 | 1.480 | token 5 |
| old, draft 3 | 28 | 56 | 0.643 | 1.143 | 1.445 | token 18 |
| **new, draft 1** | 46 | **46** | 0.370 | **1.391** | 1.659 | token 56 |
| **new, draft 2** | 40 | **40** | 0.400 | **1.600** | 1.810 | token 57 |
| **new, draft 3** | 38 | **38** | 0.395 | **1.684** | **1.840** | token 57 |

The structural goal is met exactly: **one trunk step a round** (38 rounds against 38 steps, where the
old loop spent 56 steps on 28 rounds), and the objective metric moves with it -- `tokens_per_trunk_step`
from 1.143 to **1.684** and decode throughput from 1.445 to **1.840 tok/s**, a 27% gain at equal
quality of drafting.

Two details are worth recording because they bound what is left.

The **accept rate fell** from 0.643 to 0.395. That is intrinsic, not a bug: the useful drafts now start
one position later and the first of them is conditioned on the head's own unverified output rather than
on a trunk state, so the chain compounds its own error. Buying the extra trunk step back costs draft
quality, and the trade is still worth taking -- 1.684 against 1.143.

The **output tracked plain decode much further** -- first divergence at token 56/57 against 4/5/18.
That is the same verify arithmetic as before, so the improvement is in how tokens are selected and
emitted rather than in the arithmetic, and it does not change the earlier finding that the two paths
are not bit-identical.

### The bound this puts on step 4

With the accept rate falling as the chain lengthens, the useful draft count saturates: draft 2 and
draft 3 both land at about 1.6-1.68 tokens a step, because the third drafted token is rarely right.
So the practical ceiling for this structure is near **1.7-2.0 tokens a trunk step**, not the 3 the
objective asks for.

At the measured 256K baseline of 35.07 tok/s, 1.684 tokens a step is about **59 tok/s** before paying
for the verify's extra rows, which at long context are not free -- attention over 262k positions scales
with the number of query rows. So step 4's 100 tok/s is out of reach on this path by roughly a factor
of two, independently of the token-identity blocker.

## 256K, measured: 1.45x on the eager step, and no token changed

The one-step loop at the real context, 262015-token prompt, 48 new tokens, eager trunk:

| 256K, 48 tokens | plain decode | draft 1 | draft 2 | draft 3 |
|---|---:|---:|---:|---:|
| decode tok/s | 1.375 | 1.799 | **1.993** | 1.851 |
| tokens per trunk step | 1.000 | 1.548 | **1.778** | 1.714 |
| trunk steps for 48 tokens | 48 | 31 | 27 | 28 |
| draft accept rate | -- | 0.548 | 0.481 | 0.464 |
| first divergence from plain | -- | **none** | **none** | **none** |
| peak allocated | 18.77 GiB | 19.15 | 19.15 | 19.15 |
| free after the run | 3.14 GiB | 0.32 | 0.32 | 0.32 |

Two results here matter more than the throughput number.

**The output was token-identical to plain decode for all 48 tokens, at all three draft widths.**
`first_mismatch` is null in every MTP arm. At 8K the same loop diverged at token 56 of 64, so the
decode-versus-verify arithmetic difference established earlier is real but its *consequence* is
context-dependent: at 256K the near-ties that the K-row pass moves did not change the chosen token for
48 steps. That is not a proof of identity -- 48 tokens is a short horizon and the 8K run shows it does
eventually tip -- but it is the first configuration measured where speculation did not change a token,
and it is the configuration the objective cares about.

**Speculation pays more at long context, not less.** The ratio is 1.993 / 1.375 = **1.45x** at 256K
against 1.29x at 8K, because a step's fixed cost (the weight traffic that dominates decode) is
amortised over `1 + accepted` tokens instead of one. The verify's extra rows do cost something -- the
ratio is below the 1.78 tokens a step that the raw metric shows -- but at 256K the marginal cost of a
row is smaller than the fixed cost it rides on.

### What that projects to, and why it is not the objective

The captured 256K baseline is **35.07 tok/s**. Applying the measured eager ratio, 1.45x, gives about
**51 tok/s** at 256K -- against the objective's 100. That is a large, real gain and it is short by a
factor of two.

The gap is now bounded from both sides and neither side is a tuning knob:

- `tokens_per_trunk_step` saturates near **1.8** because the accept rate falls as the draft chain
  lengthens (0.548 at one draft, 0.464 at three), so the objective's 3 is roughly 1.7x away on the
  metric itself;
- and the ratio between the metric and the wall-clock gain is about 0.82 (1.45 against 1.78), because
  the verify's rows are not free at long context.

Getting to 100 tok/s would need `tokens_per_trunk_step` around 3.5 at that ratio, which needs draft
acceptance well above anything the head produces on this model (0.46-0.55 measured).

### Why more drafts cannot close the gap either

The obvious remaining lever is a wider draft chain: `tokens_per_trunk_step` grows with the accepted run,
so more drafts should buy more tokens. The 256K measurement says it does not, and the arithmetic is
short enough to write out. Converting each arm to per-draft acceptance:

| drafts | tokens / step | accepted per round | accept per draft |
|---:|---:|---:|---:|
| 1 | 1.548 | 0.548 | **0.548** |
| 2 | 1.778 | 0.778 | **0.389** |
| 3 | 1.714 | 0.714 | **0.238** |

Per-draft acceptance falls by about a third for each link added -- the head's chain compounds its own
error, exactly as the collapsed-accept-rate analysis predicted -- so the accepted run saturates while
the verify keeps paying for rows. `tokens_per_trunk_step` peaks at **1.778 at two drafts** and is
already declining at three. Extrapolating the trend, an eight-draft chain would add well under one
accepted token while costing eight rows.

The cost side is measured from the same run: the plain arm spent 48 steps in 34.9 s (0.73 s a step) and
the two-draft arm spent 27 steps in 24.1 s (0.89 s a step), so a three-row verify costs **1.22x** a
one-row step. That 1.22 is the denominator of the gain, and it is why the wall-clock ratio (1.45) sits
below the metric (1.778).

### The objective's target, quantitatively

At the captured 256K baseline of 35.07 tok/s:

```
gain   = tokens_per_trunk_step / verify_cost_ratio = 1.778 / 1.22 = 1.46x
result = 35.07 * 1.46 = 51 tok/s        target = 100 tok/s
```

Reaching 100 needs a gain of 2.85x, which at a 1.22 verify ratio needs `tokens_per_trunk_step` of
**3.48**. The measured ceiling is 1.778 and it is *falling* with draft width, so the requirement is
about twice the best value the drafting head can produce, and no arrangement of the loop changes that
number -- it is a property of how well this model's head predicts.

That closes step 4 with numbers rather than an opinion: the memory half of the plan succeeded and is
measured (6.15 GiB freed, 256K intact, decode not slower), the expert-kernel work succeeded and is
measured (K rows, bit-exact), the speculation loop was restructured and improved (1.000 -> 1.778
tokens a step, 1.45x wall clock, no token changed in 48 steps at 256K), and the objective's 100 tok/s
is out of reach by roughly a factor of two because draft acceptance saturates.

## Outcome

What this work delivered, all of it measured on eight RTX 4090s with the native
`indexer_budget=2048`:

| item | result |
|---|---|
| NVFP4 expert bank, FP8 -> 4-bit, all 48 layers | **done**, cosine 0.9884 / relative L2 0.152 against the float expert output |
| memory freed per card at 256K | **6.15 GiB** (0.87 GiB free before, 7.02 GiB after) |
| 256K single-request decode | **35.07 tok/s, p50 28.07 ms** -- faster than the FP8 baseline's 33.94, not slower |
| 8K decode, same script both arms | **52.50 tok/s** against FP8's 48.67 |
| token agreement at 8K | identical for the first **56 of 64** tokens |
| K-row expert kernel (`nvfp4_moe_rows_forward`) | **done**, bit-exact against K single-row calls and against the token loop end to end |
| MTP head alongside the bank | **fits** (16.22 GiB peak, 8.66 GiB free); on the FP8 bank the same run runs out of memory |
| speculation loop, one trunk step a round | **done**, `tokens_per_trunk_step` 1.000 -> **1.778**, wall clock **1.45x** at 256K, 48/48 tokens unchanged |

What it did not deliver is the objective's last target: **100 tok/s at 256K**. The gap is about 2x and
it is not a matter of effort left on this path:

- `tokens_per_trunk_step` saturates at **1.778** (two drafts) because per-draft acceptance falls
  0.548 -> 0.389 -> 0.238 as the chain lengthens, so a wider chain adds rows faster than it adds
  accepted tokens;
- the verify's extra rows cost **1.22x** a one-row step at 256K, so the wall-clock gain is
  `1.778 / 1.22 = 1.46x`;
- `35.07 * 1.46 = 51 tok/s`, and reaching 100 would need `tokens_per_trunk_step` around **3.48**.

The one remaining structural lever is capturing the K-row verify in the decode graph, which
`DecodeGraph` cannot do today: it records a single position (`position.fill_(position)`,
`Set_lengths(position + 1)`, one output row), and the module's own comments record that an earlier
attempt to make the verify path recordable changed the verify's arithmetic and was abandoned. Wiring it
would be a rewrite of the capture's position and rollback core, and even if it succeeded it would
realise the measured 1.45x on the captured baseline rather than raise the ceiling -- roughly 51 tok/s,
still about half of the target.

So the memory half of the plan is complete and is the part that will keep its value: 4-bit experts cost
the model nothing measurable in throughput, hand back 6.15 GiB a card at 256K, and are what makes the
MTP head fit at all. The throughput half is bounded by how well this model's draft head predicts, and
that is a property of the checkpoint, not of the kernels or the loop.

---

## Re-evaluation: the head was mis-paired, and the device-time ceiling is wrong

The conclusion above rests on two measurements that a second look overturned: the head-pairing
convention and the cost of an extra verify row at 256K. Both were wrong, and the corrected numbers
reopen step 4.

### The head pairs with the *next* token, not the one consumed

The MTP head fuses the trunk's residual stack at position `i` with a token embedding, and the code
paired it with `t[i]` — the token consumed at `i` — to predict `t[i+1]`. The standard convention
(DeepSeek-V3, vLLM) pairs it with `t[i+1]` to predict `t[i+2]`: a genuine two-token lookahead, not a
re-derivation of the trunk's own next token.

A teacher-forced probe measured both conventions against the correct target on 256 generated tokens:

| pairing | target | accuracy | echo rate |
|---|---|---:|---:|
| A (current): head(h[i], t[i]) vs t[i+1] | 0.702 | — |
| **B (standard): head(h[i], t[i+1]) vs t[i+2]** | **0.854** | **0.020** |

B is the head the checkpoint was trained for. Its 2% echo rate says it is not just echoing its
input, and its 85.4% first-draft acceptance is the range DeepSeek-V3 MTP reports. The code's 47%
next-token agreement under convention A was the same head predicting `t[i+2]` against `t[i+1]` —
predicting the wrong target.

### The lookahead loop, with the pairing fixed

`generate_mtp_draft_lookahead` rewrites the round to the corrected convention and fixes two state
bugs the old loop carried: the PLE short-convolution window was never rewound on a partial accept,
and the catch-up call that re-seeds the head after verify reused the wrong pairing. Measured at 256K
over 48 generated tokens, against plain decode as the reference:

| arm | tokens/step | per-draft accept | first mismatch |
|---|---:|---:|---|
| plain (captured) | 1.000 | — | — |
| la_draft1 | 2.000 | 0.958 | none (48/48) |
| la_draft2 | **3.000** | 0.969 | none (48/48) |
| la_draft3 | **3.429** | 0.829 | none (48/48) |

The tokens-per-step target of "about 3" is met at K=2 and exceeded at K=3, and every arm matches
plain decode token for token over the full 48-token window.

### The verify cost at 256K is 1.5x, not 4x

The ceiling argument above assumed an extra query row re-reads the full 262K KV cache, because QSA
is the only attention flavour and the 1-row decode uses a split-K kernel. That is wrong: QSA is
*sparse* attention with a fixed `indexer_budget=2048`, so each query row selects 2048 keys, not
262K. The indexer scores the full key set, but that is a single GEMM over all rows.

A profiler probe summed device kernel time for 1, 2, 3, 4 rows at 256K (non-NCCL, max across ranks):

| rows | non-NCCL device ms |
|---:|---:|
| 1 | 13.54 |
| 2 | 19.37 |
| 3 | 20.26 |
| 4 | 20.69 |

The marginal cost of a 4th row is 0.4 ms; of a 2nd row, 5.8 ms. A 4-row verify costs 1.53x a
1-row step on device, not 4x. The captured plain step (28 ms) minus the 1-row device time (13.5 ms)
leaves ~14.5 ms of NCCL all-reduce and scheduling gaps, which is latency-bound on this PCIe-only
host (no NVLink) and roughly constant across row counts. So a captured 3-row verify is ~35 ms, about
1.24x a captured step.

### What this projects to

With the corrected acceptance (3.43 tokens/step at K=3) and the corrected verify cost (1.24x), a
captured MTP round projects to:

```
35.07 * (3.43 / 1.24) ≈ 97 tok/s
```

before the cost of the K head calls a round. If those are captured too — the head is one decoder
layer, its K-row path already exists, and NCCL collectives capture — they add roughly 1 ms a call,
landing a K=3 round near 85-90 tok/s. That is close enough to the 100 target that the remaining work
is the capture itself, not the acceptance or the verify cost.

### What remains

The capture is step 3's deliverable: a captured K-row trunk verify plus captured head chain and
catch-up, with per-position GDN/PLE state logged for rollback. The K-row expert kernel is done and
bit-exact. The QSA path accepts a device-tensor position and fixed-width arena. The PLE staging
(`ple_history_at` + pinned rows buffer) already works for one row and generalizes to K+1. The
remaining engineering is the graph class itself — recording the multi-row forward, threading the
kept-row index through the state rewind, and capturing the head chain — which is a rewrite of the
decode graph's position core, not a kernel change.

## Captured rounds and measurement corrections (2026-11-16)

The projected numbers above are not achieved throughput. `VerifyGraph` now records the K+1-row
trunk, vocabulary gather and greedy pick. Its per-position GDN/PLE outputs feed one rewind graph
per accepted prefix. `DraftGraph` records the catch-up and remaining head chain. The head has its
own absolute RoPE table, and its optional NVFP4 bank holds only this rank's experts.

Historical full-output measurements before the joint-scale correction:

- At 262015 input tokens, 48 returned tokens and K=3, both graphs matched the plain reference for
  all 48 tokens. The recorded rate was 65.856 tok/s, but its numerator incorrectly included the
  prefill-produced first token. On the same interval the corrected rate is **64.484 tok/s**.
- At 8192 input tokens, the new durable probe returned all 64 requested tokens, identical on every
  rank and through the rank-0 streaming callback. Its corrected rank-0 decode rate was **69.357
  tok/s**, explicitly excluding graph construction; including construction it was **11.022 tok/s**.
  These are short-run measurements, not a 256K or 100 tok/s result. Setup completion also differed
  across ranks, so a common-start setup barrier is required for the next comparisons.

The complete 8K evidence, including token ids, per-rank raw statistics, checkpoint metadata,
source hashes and loaded CUDA-binary hash, is in `tests/data/qwen4_exp/r355_smoke8k.jsonl`.
`tests/probe_qwen4_mtp_nvfp4.py` now streams a start/result record for each arm, checks callback and
rank agreement, treats length-only differences as mismatches, and refuses to continue after a failed
arm. Its plain reference can be reused only when its full input/configuration/source/kernel
fingerprint matches. This does not excuse a truncated generation or a changed numerical reference.

Host staging was also changed from repeated PyTorch integer operations to NumPy int64 arithmetic
and byte-preserving table views. Twenty equality tests cover the actual one-PLE-layer/eight-heads-
per-order layout as well as multiple layers, signed overflow, every EOS boundary, bf16/FP8 bytes and
padding rows. On the leased host, a synthetic hot-table CPU microbenchmark measured 3.945 ms for the
per-layer path and 0.768 ms for the new path. That is a staging microbenchmark, **not model TPS**.

### Exact context accounting

The checkpoint declares `max_position_embeddings=262144`; that is a native total-position limit,
not a promise of a 262144-token input followed by arbitrary generation. The earlier 262015-token
input was near that boundary. Similarly, a 262006-token input plus 128 returned tokens occupies
262134 logical positions even if its speculative arena has 262144 allocated slots. Allocation is
not visible context. Exact-native-boundary testing must use a safe short/eager tail rather than
silently increase the position limit or shorten the requested output.

### Joint scale correction invalidates the old quantized reference

The full-bank builder chose the gate's global scale first, then quantized up with the larger of its
own amax and the gate's. It discarded up's resulting global scale even when it was larger, while the
kernel applies **one** global scale to both projections. For such experts the reconstructed up
projection was therefore multiplied by `gate_amax / up_amax`. This is an unintended error, not the
expected 4-bit reconstruction drift.

The corrected producer computes `max(gate_amax, up_amax)` first and supplies it to both quantizers.
A CPU regression test fails on the old producer and passes on the repair, explicitly reconstructing
both projections under the scale the kernel consumes. This changes the trunk weights: plain and MTP
must be rebaselined together, and the earlier output hashes/rates cannot validate this revision.
`tests/probe_qwen4_exp_nvfp4_scales.py` compares both producers on real checkpoint experts; it must
run before recording fresh end-to-end quality/parity evidence. Reaching 100 tok/s remains open.

The completed real-weight audit sampled layers 0 and 20, experts 0/8/16/24/32/40/48/56 and three
activations per expert (16 experts, 48 expert-output samples). Two experts were affected. Layer 20,
expert 48 had `gate_amax/up_amax=0.71839`: its old output norm ratio was 0.71233 and relative L2
0.31709; the joint-scale producer gave norm ratio 1.00046 and relative L2 0.15531. Across the sample,
mean relative L2 changed from **0.166775 to 0.155103**. This checks the reconstruction defect and
expert-output error, not held-out task quality. Raw evidence is
`tests/data/qwen4_exp/r355_scale_audit.jsonl`.

The longer pre-correction run at 262006 input tokens also completed: all **128/128** returned ids
matched plain decode on all ranks and through the callback, with no EOS truncation. K=3 took 44
verify replays; the decode numerator was 127 tokens (first token excluded), giving **61.916 tok/s**
on rank 0. Delivered decode tokens per trunk call were **2.886**, not the earlier short-window
3.429. Setup-inclusive rate was 18.480 tok/s. Plain decode on the same input measured 34.624 tok/s
excluding setup and 22.215 tok/s including it. These scopes must not be mixed to claim a larger
speedup. Per-rank setup skew remains visible in this older run. All outputs and scopes are preserved
in `tests/data/qwen4_exp/r355_native256k.jsonl`; despite its historical filename, its logical span
was 262134, not 262144. A fresh quantized reference and exact-boundary run remain required.

### The corrected-weight 128-token smoke rejected MTP (round 357)

The next run used the joint-scale weights, common-start setup alignment, exact-native-tail support,
and the v2 evidence schema. At 8192 input tokens, plain decode completed all 128 requested tokens.
The captured MTP arm also returned 128 and agreed across ranks/callbacks, **but first differed from
plain at output index 67**. The probe correctly returned exit code 2 and did not launch the longer
native-boundary stage. Its 70.919 tok/s timing is **invalid as a lossless speed result**. The complete
failed evidence is `tests/data/qwen4_exp/r355v2_smoke8k.jsonl`.

A subsequent same-prefix, first-row diagnostic at 1024 input tokens confirms numerical row-count
dependence before the first layer has finished. Ordinary four-row vs one-row execution differed by
1.0625 maximum absolute logit value; making dense linear calls row-wise reduced that to 0.5391 but
did not eliminate it. This is not future-token leakage (the earlier changed-second-row experiment
left row 0 unchanged), nor evidence that a token-mismatching batched verifier is acceptable. Further
diagnostics compare depthwise-convolution dispatch and BF16 collective accumulation at equal inputs.
No runtime arithmetic change or 100 tok/s claim follows from this diagnostic alone.

### Exact arithmetic was isolated, rather than inferred from token prefixes (round 358)

The diagnostic was extended to log each reduction's input/output and separately control dense
linears, depthwise convolution, row-parallel sums and sparse-attention dispatch. On the same
committed 1024-token prefix with four verification rows:

| Diagnostic path | First differing layer | Maximum logit error |
|---|---:|---:|
| Ordinary batched forward | 0 | 1.0625 |
| One-row dense/conv, batched BF16 reductions | 0 | 0.5391 |
| One-row dense/conv and separate one-row reductions | 3 | 0.6875 |
| Same, plus the existing one-query QSA dispatch per row | **none across 48 layers** | **0.0 (bit-identical)** |

The first reduction in the second row above had **bit-identical inputs on each rank** but differing
outputs when the payload held four rows instead of one. This is a BF16 accumulation-order issue,
not a topology or NCCL-knob performance experiment. Coalescing the one-row collectives also changed
the result, so it is not a verified replacement for separate calls. At layer 3, QSA's batched kernel
uses different arithmetic from plain decode's split-K entry; a one-query call per verification row
removed the remaining difference. Neither change selects a new greedy baseline.

The serial GDN test additionally compared repeated one-token CUDA calls against the K-row state-
recording entry, using the actual TP8 geometry (two key heads, six value heads): **nine GPU tests
passed**, covering outputs and every recurrent state. The scan itself is not the observed source.

Complete diagnostic records are in `tests/data/qwen4_exp/r358_firstrow_arithmetic.json` and
`tests/data/qwen4_exp/r358_qsa_rowwise_diagnostic.json`. These are same-prefix numerical diagnostics,
not generated-sequence parity, long-context validation, or throughput. An opt-in production path
and complete 128-token comparison are the next checks; the extra per-row work may reduce speed.

The explicit `POCKETLLM_QWEN4_VERIFY_ROW_INVARIANT=1` implementation subsequently passed the
complete 8192-input/128-output smoke: all 128 ids match plain decode on all eight ranks and the
callback exactly matches returned output. No output token was discarded to obtain parity. This
correctness costs time: corrected decode throughput was **44.193 tok/s** for the speculative arm
versus **52.088 tok/s** for ordinary one-row decode, both excluding setup. The verify cost rose to
55.62 ms per round (51.24 ms device/replay interval); head cost was 5.88 ms, staging 4.26 ms and
rewind 0.31 ms. Decode tokens per trunk pass were 2.761. Setup-inclusive MTP rate was 13.945 tok/s.
These are measured results, not a 100 tok/s success. Evidence is
`tests/data/qwen4_exp/r358_smoke8k.jsonl`.

The native-total-boundary run uses **262016 input + 128 output = 262144 logical positions** and a
clamped 262144-position arena. It hands the final positions to ordinary one-row trunk decode before
speculative rows would overrun that arena. The probe validates that graph rounds plus the declared
native tail account for all trunk steps; arbitrary eager fallback is not reported as graph success.
Its GPU result must still be collected before claiming boundary parity.

That boundary result is now collected: **128/128 outputs match plain**, every rank agrees and
callbacks match the returned sequence. Both arms respected the native limit. The speculative arm
used 43 captured verify rounds then four one-row eager calls starting at position 262139; its
logical total and arena size are both 262144. Complete evidence is
`tests/data/qwen4_exp/r358_native256k.jsonl`.

The rate is **21.015 tok/s excluding initial graph setup**, versus the same run's plain
**34.127 tok/s excluding setup**. MTP including setup measured 10.324 tok/s. Its native tail is
currently eager, so four final calls are disproportionately expensive on a 128-output run. This is
an honest correctness success and a throughput failure, not the requested 100 tok/s deployment.

A subsequent controlled collective-layout diagnostic found a possible way to preserve BF16 sum
ordering without K separate collectives: packing `[K, 8, 320]` as `[8, K, 320]` keeps each one-row
rank-owner chunk together. At K=4/H=2560, this layout matched serial one-row reductions in both eager
and captured cancellation-heavy tests, while the ordinary flat K-row layout did not. It is exposed
only as the hardware-scoped experimental `POCKETLLM_QWEN4_VERIFY_REDUCE_LAYOUT=rank_chunk`, with
strict per-row reduction remaining the default. This is not a new precision baseline or an NCCL
setting sweep. Eight-trial diagnostic evidence is
`tests/data/qwen4_exp/r358_collective_layout_diagnostic.json`; a source-alias caveat invalidates the
trivial full-width layout's captured check but does not affect the nontrivial rank-chunk layout.
Larger adversarial trials and full-output GPU checks must validate this experimental layout before
using its rate as an exact speed result.

The independent-buffer, **32-trial** eager/captured collective diagnostic passed for rank-chunk
width 320, including adversarial cancellation inputs. The full 8K/128 generated-sequence comparison
then passed: all output/callback/rank checks remain exact and captured MTP measured **66.480 tok/s**
excluding setup versus plain **52.706 tok/s**. Verify time decreased from the separate-row run's
55.62 ms to **34.22 ms**; head cost was 6.55 ms and staging 2.01 ms. This validates that limited
workload/hardware scope, not general portability or 100 tok/s. Complete evidence is
`tests/data/qwen4_exp/r358_layout_smoke8k.jsonl`; independent numerical trials are in
`tests/data/qwen4_exp/r358_collective_layout32.json`.

The next opt-in change, `POCKETLLM_QWEN4_MTP_TAIL_GRAPH=1`, captures a one-row native-boundary trunk
step during initial setup, **before** the K-row verifier. Each graph owns separate pinned PLE
staging while sharing stable live recurrence buffers; reverse-order teardown restores the tail's
staging at the boundary. A one-row forward fully commits its state, so it needs neither per-position
GDN rewind slots nor an empty rewind graph. Its cost stays in initial graph setup, and reported
`tail_graph_steps` is separate from `plain_tail_steps` (eager-only). The reviewed snapshot is queued
for GPU parity testing only after the earlier immutable run exits. CPU regression tests pass, but no
GPU or throughput result is yet attributed to this tail change.

The packed-layout **native-total** run also passed all 128 ids/callbacks/ranks. It measured
**23.852 tok/s** excluding setup versus plain 34.122 tok/s; 43 verify rounds averaged 35.40 ms
and head work 8.80 ms, but four native-boundary eager calls consumed **3.001 seconds** of the
5.325-second decode interval. This explains why the shorter 8K workload's gain is not a native-total
throughput result. The complete evidence is `tests/data/qwen4_exp/r358_layout_native256k.jsonl`.
Capturing those tail calls is the next measured change, not a license to subtract their time and
report an unobserved rate.

The captured-tail native run is now complete: **128/128 tokens, callbacks and every rank match the
plain reference**, with 262016 input tokens, 128 returned tokens and exactly 262144 logical/arena
positions. It used 43 K-row verify replays plus four one-row tail replays and **zero eager tail
calls**. Those four tail replays took 0.176 seconds instead of the prior 3.001 seconds. Corrected
MTP throughput is **52.058 tok/s excluding initial setup**, versus plain **33.631 tok/s excluding
setup**; including setup MTP measured 12.227 tok/s. Output evidence and every raw scope are in
`tests/data/qwen4_exp/r358_tail_layout_native256k.jsonl`. Thus native context and sequence parity
are established for this configuration, but the requested 100 tok/s is still not achieved.

A new small-query split-K QSA entry is the next kernel-level experiment. It adds query-row address
planes to the existing 16-split partial/combine arithmetic and leaves the single-query kernels/API
unchanged. GPU tests must require bit equality against concatenated existing single-query calls,
including adversarial selections and captured replays, before its complete-output runtime rate can
be accepted. The runtime binding is opt-in (`POCKETLLM_QWEN4_QSA_SPLITK_ROWS=1`) and falls back to
the previously verified serial query loop when the new entry is unavailable. A fresh CUDA binary
fingerprint invalidates reused plain-reference artifacts; no old-reference comparison is silently
carried over a rebuild.

The new extension built successfully for sm_75/sm_86/sm_89 and **204 real CUDA tests passed, zero
skipped**, including 199 small-query exactness/guard cases and the existing QSA tests. Its binary
SHA256 is `5d7c8f6be236d7051812b9dd5c372734e6146ec4a7dc8e9b56c6cdbc7079c765`; validation provenance
is in `tests/data/qwen4_exp/r359_qsa_rows_cuda_validation.json`.

An isolated four-query/2051-selection CUDA-graph benchmark measured approximately 0.0753 ms for
four separate single-query calls versus 0.0285 ms for the new batched entry, with bit-identical
output. This is kernel timing, not decode TPS; complete evidence is
`tests/data/qwen4_exp/r359_qsa_rows_micro.json`. The complete 8K/128-output model comparison then
passed token/rank/callback checks and measured **68.579 tok/s excluding setup** versus plain
**50.337 tok/s** on the same run. That result is in `tests/data/qwen4_exp/r359_qsa_rows_smoke8k.jsonl`.
The native-total result still needs collection before attributing a 256K improvement to this op.

That native-total comparison has now completed with **128/128 greedy tokens, callbacks and all
ranks equal**: captured MTP measured **55.641 tok/s excluding setup**, versus plain **33.403 tok/s**.
It still uses 43 K-row rounds and four captured tail steps, with a 2.2825-second decode interval for
127 decode-produced tokens. Complete evidence is `tests/data/qwen4_exp/r359_qsa_rows_native256k.jsonl`.
This remains below 100 tok/s, and its setup-inclusive rate is 12.280 tok/s.

Fresh independent reviews identified a boundary-cleanup timing hazard: destroying verifier/draft
recordings and flushing cached allocations at the native handoff delays the first tail token and
shifts the delay among ranks' tail/unclassified timers. Their **combined** time is consistent across
ranks even when `tail_graph_s` alone is not. The next default-OFF experiment separates restoration of
cache/PLE metadata from destruction of device graph resources, retains all rewind/staging owners
through final token delivery, and measures request cleanup separately. No tail time is subtracted
from a reported rate merely because it appears to be cleanup.

A BF16 strided-batched linear microprobe is **rejected as a general exact replacement**: several
projection geometries initially matched the serial one-row oracle but differed after changing the
input values. The vocab projection's BMM path was not materially faster in that test. Evidence is
`tests/data/qwen4_exp/r360_bf16_linear_micro.jsonl`; no runtime math switch was made from it.

Cancellation-heavy 32-trial collective-layout tests also passed width 320 at five, six and eight
rows, in both eager and captured execution. The experimental rank-chunk allow-list now includes only
those tested row spans plus the previously proven four-row span; full generated-sequence comparisons
still gate each new draft depth. Diagnostics are in `tests/data/qwen4_exp/r360_collective_layout_spans.json`.

The depth sweep correctly stopped on a negative result: at 8K/128, depth 3 and depth 4 matched every
plain token, but **depth 5 changed token 67**, despite complete output and agreement among ranks and
callbacks. The approximately 59.0 tok/s depth-5 value is **invalid as an exact result**. Complete
accepted-prefix and per-position survival evidence is in
`tests/data/qwen4_exp/r360_defer_depth_smoke8k.jsonl`. A new default-OFF FP32 indexer-score flag uses
one-query GEMM geometry to investigate the wider-row difference; it does not change pool dtype,
precision settings or top-k ordering. Its GPU attribution remains unproven.

The native-total rerun used only qualified depths 3 and 4, and both matched all 128 greedy tokens,
callbacks and ranks. With metadata-only boundary retirement, depth 3 measured **66.680 tok/s before
post-delivery resource cleanup, 53.197 tok/s including cleanup**; depth 4 measured **62.986 and
59.511 tok/s** on those same scopes. Plain measured 34.073 tok/s excluding initial setup. Restoring
boundary bindings took about 0.2 ms, but resource destruction did not disappear: it is charged to the
separately reported request-cleanup interval. Evidence is `tests/data/qwen4_exp/r360_defer_native34.jsonl`.

The independently named NVFP4 operand-decoder fast path subsequently passed **252 real CUDA tests,
zero skipped**, including all 16 E2M1 and 256 E4M3 operand encodings and eager/captured comparisons
against the unchanged old MoE entry. Its binary SHA256 is
`c768c0582f169135fbc75f9d3b2304af606778d00e49cc6ba5dce5cac5203db9`; provenance is in
`tests/data/qwen4_exp/r361_nvfp4_exact_validation.json`. The new entry replaces divergent constant
loads/scale decoding, not weight precision or accumulation order. An isolated four-row, TP-like
route probe measured approximately **0.061 to 0.031 ms**, with bit-identical outputs, while a one-row
case changed little. Those are synthetic-kernel timings, not model TPS, preserved in
`tests/data/qwen4_exp/r361_nvfp4_exact_micro.jsonl`.

A separately captured startup chain now replays the same initial two/three one-query head calls
**after** first-token delivery. Its capture/warmup cost remains setup; its real replay is measured
inside decode. In the new complete 8K/128 run, depths 3/4 remained greedy-identical and measured
73.712/71.792 tok/s before post-delivery cleanup. The native-total measurement is still running; no
100 tok/s claim follows from an operator speedup or from the 8K result.

### Subsequent native result, longer-output correction and PD proof

The native-total 128-output run subsequently completed: depth 3 measured **72.679 tok/s excluding
setup/cleanup, 56.289 including cleanup**; depth 4 measured **71.964 / 66.560 tok/s**. Both delivered
all 128 greedy ids with matching callbacks and every rank, at 262016 + 128 = 262144 logical positions.
Evidence: [complete native 128-output results](https://github.com/lvyufeng/RelicLLM/blob/master/tests/data/qwen4_exp/r361_exact_startup_native256k.jsonl).
These are short-output qualifications, not a hardware ceiling or a universal production rate.

The extended 512-output test then **failed at output index 464**. Its earlier fast timing is not a
lossless result. Packed Q/gate projections exposed a missed arithmetic contract: a one-query indexer
norm can be contiguous and select fused CUDA RMSNorm, while a wider query slice falls back to torch;
a gapped QSA-head slice also changes torch reduction geometry as row count grows. The strict path
now retains each one-query slice's original strides and decode dispatch. It does not force a gapped
Q-head input into a fused kernel that ordinary decode did not use. One-query FP32 indexer scoring
was also retained for the corrected verification configuration.

The corrected **261632-input / 512-output native-total** run passed every id, callback and rank and
also matched the previous plain 512-token reference without redefining the answer. It measured
**55.702 tok/s excluding setup, 53.030 including request cleanup**, with 233 verifier rounds and five
captured one-query tail steps. Plain measured 37.001 tok/s excluding setup. The 512-output run's
2.147 tokens/trunk step is lower than the earlier 128-output workload's 2.702; throughput depends on
proposal acceptance as well as the kernel path. Evidence: [corrected complete native 512-output
result](https://github.com/lvyufeng/RelicLLM/blob/master/tests/data/qwen4_exp/r362_norm_fixed_native512.jsonl). **100 tok/s remains unachieved.**

A complete two-host PD handoff has separately passed at **8192 input + 128 output**. The receiving
TP8 host did not repeat prefill; all 128 outputs equal the source/plain reference, with every rank and
the streaming callback agreeing. Thirteen FP32 indexer prefixes were imported and rebound to local
storage, with zero historical-pool recomputation. Eager `tensor_ready=False` describes arena-wide
replay initialization, not an invalid completed prefix; that distinction is now preserved by the
serializer. The destination measured **66.390 / 63.582 tok/s** excluding setup / including cleanup,
not a native-256K PD rate. State and local recordings are distinct: CUDA graphs and process pointers
never cross hosts. The portable shared binary passed 247 CUDA tests on the older-GLIBC host.

The real 8K rank-local state is 1.317 GB, with **0.375 GB of distinct tensors proven by hashes**. The
prototype transferred the rank-local copies in 17.290 seconds of payload phases over the active 1GbE
link; deduplicated wire transport and asynchronous prefill overlap are not yet implemented. PD does
not add GPUs to single-request decode. Two complete replicas generally provide more aggregate
capacity for this prefill-dominated workload. The detailed scope, byte accounting and theoretical
analysis are in [the roofline and PD report](https://github.com/lvyufeng/RelicLLM/blob/master/tests/data/qwen4_exp/r362_roofline_pd_report.md).

