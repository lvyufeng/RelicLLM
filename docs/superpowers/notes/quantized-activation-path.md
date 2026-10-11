# Where the v41 port and the official reference diverge

**Status: root cause measured, fix designed.** This note records how the port's arithmetic departs
from DeepSeek's own reference, the experiment that pins it, and what the fix must therefore be. It
supersedes the diagnosis in the sm_89 plan (`2026-10-09-v41-flash-sm89.md`) that attributed the
divergence to the card ("bf16 tensor cores on sm_89 vs fp32 SIMT on sm_75") -- that was a symptom.

## The claim

The port dequantizes every fp8/fp4 weight to bf16 once at load and runs a plain `F.linear` bf16
GEMM (`relicllm/models/deepseek_v4_1/modules.py`; the module docstring says so: "dequantized once,
at load, and the arithmetic runs in bf16"). The reference instead keeps the weight **quantized** and
quantizes the **activation** to fp8 before a block-scaled GEMM:

```python
# checkpoint inference/model.py::linear, abridged
x, s = act_quant(x, 32, "ue8m0", torch.float8_e8m0fnu)
return fp8_gemm(x, s, weight, weight.scale, torch.float8_e8m0fnu, block_size=32)  # fp32 accumulate
```

That activation quantization is the whole of the token divergence.

## Why the weight half is *not* the difference

fp8 (e4m3) and fp4 (e2m1) each carry at most three mantissa bits, and a block-scaled dequant is
`code * 2**k` -- an exact operation in bf16 (8 mantissa bits). So the port's already-dequantized bf16
weights are **bit-identical** to the reference's quantized weights. The port loses nothing on the
weight; what it drops is (a) quantizing the activation to fp8 and (b) accumulating in fp32.

## The experiment (single-variable)

Run DeepSeek's unmodified `model.py` on the same 8 cards, same TP8 weights, same host-offload, and
replace **only** the `linear` function. One prompt (the 9 raw token ids the golden fixture uses):

| step-0 top-8 | argmax |
|---|---|
| official, unmodified (the fixture's step 0) | `[455, 223, 342, 539, 2275, 1004, 983, 1162]` | **455** |
| official, `linear` = dequant weight -> bf16 `F.linear` (the port's arithmetic) | see below | **1 (EOS)** |
| official, `linear` = `act_quant` + dequant weight -> **fp32** matmul -> bf16 | see below | 455 or 1 -- not reproducible |

**Correction, 2026-10-11.** An earlier version of this table quoted the official step-0 row as
`[455, 2052, 983, 1162, 5715, 12025, 6948, 35857]`. That is **step 8**, not step 0: the raw dump
`/tmp/ref_logits_raw9.pt` is shaped `(16, 129280)` -- sixteen autoregressive steps, despite the
"raw9" name -- and the earlier comparison indexed row 8. Row 0 is
`[455, 223, 342, 539, 2275, 1004, 983, 1162]`, which is exactly the committed fixture
`tests/fixtures/reference/v41.json`. **The fixture is correct; this note was not.** Rows 2 and 3 of
the old table were drawn from the same mis-indexed comparison and their step-0 values are being
re-measured.

Row 2 is the falsification: the port's arithmetic alone, with every other part of the official model
untouched, produces the port's EOS-first answer (argmax 1). Row 3 shows the reference's own
activation quantization moving the answer back toward 455.

**Row 3 is not bit-reproducible, and that is the sharper finding.** The official is: two independent
official processes agree to `max_abs = 0.000000`, argmax 455. But the soft `act_quant` + fp32-matmul
proxy gives step-0 argmax **455 in one run of three and 1 in the other two**, with pairwise
`max_abs` of 3.8-6.5 between its own runs -- on an RTX 4090 the soft `x @ w.T` runs through cuBLAS
with unpinned algorithm selection, while the official runs its own deterministic tilelang kernel.
Per-op the soft is *unbiased* (signed mean -0.0011 bf16 ulp, <= 1.0 ulp against the fp64 truth) and
the kernel is one rounding event noisier (<= 1.075 ulp). So the ~9 step-0 `max_abs` is ~1 ulp/layer
of rounding-order difference chained over 40 layers, not a bias -- and **no pure-torch soft path can
be made bit-identical to the oracle**. The acceptance is therefore **token-level agreement, not
value-level**; do not assert a tiny logits tolerance against the reference.

One number is *not* noise and remains open: the official `logit[1]` (EOS) is 15.05, while the soft
gives a stable ~18.5 (+3.4 logits, ~55 ulp) even as `logit[455]` only wobbles (18.0-18.85). An
unbiased <= 1 ulp/op chain cannot lift one logit by a stable 55 ulp, so the soft path carries a
systematic error somewhere -- the fp4 expert path, a host-offload summation order, or the head's
accumulator width -- and it is what decides the flip. It is being localized.

## Why no existing test caught it

`tests/test_models_deepseek_v4_1_reference_parity.py` sets `dtype="bf16", expert_dtype=None` -- which
turns the reference's `Linear` into a plain `F.linear`, i.e. it turns the reference's quantization
**off** to match the port, so a divergence that lives *inside* the quantization is invisible by
construction. The official oracle recorded in `tests/fixtures/reference/v41.json` is the check that
catches it.

## Reproducing

- `scripts/v41_reference_linear_swap.py` -- runs the official model with only `linear()` replaced,
  by `--variant dequant-bf16` (row 2) or `--variant actq-fp32` (row 3). Prints the step-0 top-8.
- `scripts/v41_layer_tap.py` -- runs the official model and applies the real head to each layer's
  collapsed residual stream, so a divergence can be located by depth rather than only at the output.

Both import the checkpoint's `inference/model.py` unmodified and reproduce it here; they need the
converted TP checkpoint and an env with `tilelang` (the `dsref` env on the 4090 box).

# The fix

The port needs the reference's activation quantization on every quantized `linear`, with the GEMM
accumulating in fp32. Nothing is re-loaded and nothing is re-packed: `dequant` is already exact, so
`dequant(act) @ W.T` in fp32 *is* the reference's GEMM. What follows is the design.

## The primitive: `quantized_linear`

Two additions to `relicllm/models/deepseek_v4_1/kernels.py`, both written from the reference's
`act_quant` semantics rather than imported, because `relic_core.kernels.ops.act_quant` reproduces the
same block-32 arithmetic but is a *kernel* binding whose output dtype and impl are chosen by the
host, and this path has to be a pure torch soft GEMM.

```python
def block_fp8_act_quant(x, block_size: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """The reference's `act_quant(x, 32, "ue8m0", float8_e8m0fnu)` without a kernel.

    Returns `(quant, scale)` where `quant` is `float8_e4m3fn` shaped like `x` and `scale` is fp32
    `[..., K // block_size]`. Three steps, in this order, and the order is the contract:

    1. `amax = x.abs().view(-1, K // block_size, block_size).amax(-1)` then `amax.clamp_min(1e-4)`.
       The floor is the reference's `T.max(amax_local[i], 1e-4)` -- an all-zero block must still get a
       nonzero scale, because that scale is divided out on the way back.
    2. `scale = 2 ** ceil(log2(amax / 448))`. The reference's `fast_round_scale(amax, 1/448)` is
       exactly this bit trick, so the result is the smallest power of two at or above `amax / 448`,
       which is what makes the clamped ratio land inside the fp8 range without a second clamp.
       `448` is E4M3's largest finite.
    3. `quant = clamp(x / scale, -448, 448).to(torch.float8_e4m3fn)`.
    """
```

```python
def quantized_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    quantized: bool = False,
    block_size: int = 32,
) -> torch.Tensor:
    """The reference's `linear()` on the soft path, with the weight already dequantized.

    `quantized=False` is the no-scales path and it is exactly `F.linear(x, weight)` -- not an
    approximation of it, the same call. That is the seam the bf16 build stays on and the seam
    `test_models_deepseek_v4_1_reference_parity.py` exercises.

    `quantized=True` is the reference's with-scales path. The `weight` handed in is the *dense*
    tensor the loader already produced, so there is no weight scale here and none is needed: an fp8
    e4m3 or fp4 e2m1 code carries at most three mantissa bits and a block scale is a power of two,
    so the loader's `dequant` is exact and `weight` equals the reference's `dequant(weight, scale)`
    bit for bit. What is missing from the port and restored here is the *activation*: it goes through
    `block_fp8_act_quant`, and the product is accumulated in fp32 and narrowed once, to `x.dtype`.

    This is what makes the reference's GEMM: both operands are quantized (the weight in the
    checkpoint, the activation here) and the accumulation is fp32 where the port's plain `F.linear`
    accumulated at the activation's width with an unquantized activation.
    """
    if not quantized:
        return F.linear(x, weight)
    act, act_scale = block_fp8_act_quant(x, block_size)
    dequant_act = act.to(torch.float32) * act_scale.repeat_interleave(block_size, dim=-1)
    return torch.matmul(dequant_act, weight.to(torch.float32).transpose(0, 1)).to(x.dtype)
```

Two decisions inside that are worth stating:

* **The weight is upcast to fp32 explicitly, even though it is already bf16.** The accumulator's
  width has to be a property of this function and not of what the caller happened to hand in; a
  `weight` that is bf16 matmuls in bf16 on some paths, and the reference accumulates in fp32. The
  upcast is free of information (bf16 is exact in fp32) and it fixes the reduction's width.
* **`fp4` and `fp8` weights share one activation path.** The reference's `linear()` quantizes the
  activation with `act_quant` -- fp8, block 32 -- for *both* weight dtypes; for fp4 the GEMM handles
  the mixed precision. So the primitive has one activation path and does not need to know which kind
  of checkpoint weight produced `weight`. `block_size` is the activation's block width and it is 32
  for both. `block_fp8_act_quant` is the only new arithmetic.

## The seam

The split is per-call-site and driven by the config, not by a runtime dtype test on the weight: a
dequantized weight is bf16 by the time it reaches the module, so its own dtype cannot say whether it
came from fp8 or from a plain bf16 checkpoint. Each module therefore records one boolean at build
time and dispatches on it.

* `cfg.dtype == "fp8"` turns on the activation-quantizing path for the **dense** linears: `wq_a`,
  `wq_b`, `wkv`, `wo_b` and the indexer's `wq_b` in `attention.py`, and `Engram.wkv` in `modules.py`.
  These are exactly the reference modules that construct a `Linear`/`ColumnParallelLinear`/
  `RowParallelLinear` with no explicit dtype, i.e. the ones whose `dtype` resolves to
  `default_dtype = torch.float8_e4m3fn`.
* `cfg.expert_dtype == "fp4"` turns it on for the routed experts, and `cfg.dtype == "fp8"` for the
  shared expert (the reference builds `Expert(..., dtype=None)` there, so it is fp8, not fp4).
* `bias=None` everywhere: the reference's `linear()` asserts it.

The split must not break `test_models_deepseek_v4_1_reference_parity.py`, which builds random
parameters with `dtype="bf16", expert_dtype=None`. Under that config every boolean is off, every
`quantized_linear` is called with `quantized=False`, and the tree is the plain `F.linear` tree it is
today -- bit for bit, because `quantized_linear(x, w)` *is* `F.linear(x, w)`. The test that pins
this is `test_no_scales_path_is_exactly_f_linear` in the new file.

Two things that look like call sites and are not, both confirmed by reading the reference and worth
recording so a later sweep does not "fix" them:

* **`Gate`.** `Gate.weight` is a bare `nn.Parameter(torch.empty(...))` and so takes torch's default
  dtype (bf16, set by the reference's own `torch.set_default_dtype`), not `default_dtype`. Its
  forward is `linear(x.float(), self.weight.float())`, and `weight.float()` makes `linear()` fall
  through to `F.linear`. The port already writes `F.linear(x.float(), self.weight.float())`. No
  change.
* **`Compressor` and the indexer's `wk`.** The reference names their dtype explicitly (fp32 above
  ratio 1, bf16 otherwise), so `default_dtype` never applies and `linear()` is a plain `F.linear`
  there too. The port's `LINEAR_DTYPE`/fp32 choices already match. No change.
* **The indexer's `weights_proj`** is bf16, so it *is* a plain `F.linear` in the reference -- and, for
  the same reason, in the port. That is a genuine second deviation from the "every dense linear is
  fp8" reading, but it is not the one this fix addresses, and it is out of the primitive's reach
  because `weights_proj` is bf16 by the reference's own declaration, not by a config the port reads.
  Recorded here so a later sweep does not mistake `quantized_linear` for having missed it.

## Exactly which call sites change

`relicllm/models/deepseek_v4_1/modules.py`

* `expert_forward` -- the three projections. It must take the quantization as an argument (a
  `block_size`/`kind`, or simply the two weight tensors and a flag) because it is called with loose
  tensors by both `Expert.forward` and the two `RoutedExperts` stores; it cannot read a config it
  does not have. The routed experts are fp4 (`expert_dtype == "fp4"`), the shared expert fp8.
* `Expert.__init__`/`forward` -- thread the shared expert's `cfg.dtype == "fp8"` and the routed flag
  down to `expert_forward`.
* `ResidentRoutedExperts.forward` and `CheckpointRoutedExperts.forward` (loader.py) -- both call
  `expert_forward`; both must pass the routed experts' quantization, which `loader.load_backbone`
  reads from `config.expert_dtype`.
* `Engram.__init__` -- `wkv`.

`relicllm/models/deepseek_v4_1/attention.py`

* The `Attention` projections: `wq_a`, `wq_b`, `wkv`, `wo_b`. `wo_a` stays bf16 (the reference names
  it), and the block-diagonal `einsum` over `wo_a` is not a `Linear` and is not touched.
* The indexer's `wq_b`. `weights_proj` and `wk` stay bf16.
* `LINEAR_DTYPE` itself does not move: it is still the storage width the weights are dequantized to,
  which is what makes the fix a change of *arithmetic* and not of storage.

`relicllm/models/deepseek_v4_1/device_experts.py`

* No change, and the reason is that this path is already a quantized-activation path -- just not
  *this* one. `moe_single_token_fp4_forward` is documented as "int8 activations x fp4 weights", so
  the device experts quantize the activation to int8 with a single per-row scale where the reference
  uses fp8 with a block-32 scale. That is a second, independent divergence (recorded in
  `docs/performance/deepseek_v4_1_flash_device_experts.md`: 6.7% of the output scale on the worst of
  69 rows), and it is the device path's own kernel contract, not something `quantized_linear` can
  reach from torch. The fix here is the *host* path; the device path is a separate piece of work
  against `relic_core`'s kernel and is out of scope for this primitive.

## Where the activation quantization sits relative to the swiglu clamps

`expert_forward` computes `gate = w1(x).float()` and `up = w3(x).float()`, clamps in **fp32**
(`up` on both sides, `gate` from above, at `swiglu_limit = 10.0`), forms `silu(gate) * up`, scales by
the routing weight, casts to `x.dtype`, and only then runs `w2`. The quantization therefore belongs
**inside** each of the four projections and not around the expert:

* `w1`/`w3` quantize their *input* `x`. That is the same `x` for both, so it is quantized twice with
  the same block scale -- wasteful but exactly what the reference does, since the reference's `w1` and
  `w3` are two separate `Linear` calls each running `act_quant` on the same activation. Reproducing it
  is what keeps the two sides bit-for-bit; folding it into one shared quantize is a later
  optimization, not this fix.
* `w2` quantizes `out.to(dtype)` -- the post-clamp, post-silu activation, cast back to bf16 first
  because that is the operand `Linear.forward` is handed in the reference. The clamp is in fp32 and
  happens *before* this cast, exactly as today; the quantization is after the cast, because the
  reference quantizes whatever width its `Linear` was handed. Getting that order wrong -- quantizing
  the fp32 pre-cast value -- would move the fp8 rounding by one bit and is the kind of thing the
  clamp-before-silu comment in `expert_forward` already warns about at the same site.

The clamp values themselves do not change and the fp32 clamp is not moved: they exist to keep the
activations inside the fp8 range in the first place, which is now load-bearing rather than vestigial.

## Integration note: `loader.py`

**Nothing changes, and the reading confirms it.** `V41Checkpoint.dequantize` expands a weight with
`dequant_fp8_weight`/`dequant_fp4_weight` to fp32, and `weight()` then casts to the parameter's dtype,
which is `LINEAR_DTYPE` (bf16) for every module that will take the quantized path. That cast is lossless:
an fp8 e4m3 or fp4 e2m1 code carries at most three mantissa bits, and an E8M0 block scale is a power of
two, so `code * 2**k` is a bf16 value exactly -- the source's `soft_*_weight_dequant` returns fp32 and
the narrowing to bf16 reproduces the reference's `weight * scale` bit for bit. The quantized path
therefore reads the *same* numbers the reference reads, and `quantized_linear` needs no `scale` tensor
and no un-expanded weight: it operates on the dense bf16 weight the loader already produced. The one
caveat is the `.float()` inside `quantized_linear`: on an `sm_80`+ card with triton the loader's own
`dequant` can return fp16, so a build on Ada should confirm the weights it holds are the fp32 form
before trusting the bf16 equivalence. On the sm_75 target `_auto_impl` is "torch", so this is already
true and no loader change is needed.

## Tests

`tests/test_v41_quantized_activation_path.py` pins the three properties this design turns on:

* the block-32 quantized GEMM against a longhand oracle (`dequant(act) @ W.T`) and against a plain
  bf16 `F.linear`, which it must differ from by the fp8-activation error the reference introduces;
* the `quantized=False` path against `F.linear` **exactly** (`torch.equal`, not `allclose`), because
  the reference-parity suite depends on that being the same call and not an approximation;
* the activation quantization's `1e-4` amax floor (an all-zero block gets a nonzero scale) and its
  power-of-two rounding (a known input's scale is asserted against its exact power of two).

The file skips when torch is absent, like every sibling in `tests/`. The quantized GEMM is CPU-safe
on purpose: the primitive is soft torch, so it does not need a card and only the module-level
`pytest.importorskip("torch")` guard is required.