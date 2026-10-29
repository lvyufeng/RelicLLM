# Qwen4-Exp Heterogeneous TP4 Performance

**Date:** 2026-08-31

**Hardware:** 4 x RTX 2080 Ti, TP4 over PCIe Gen3
**Model:** Qwen3.8-Flash-Next, real checkpoint at
`/mnt/data1/modelscope/Qwen/Qwen3.8-Flash-Next`

## Runtime architecture

The real model has 48 layers, hidden size 2560, 512 routed experts, and top-10
routing. One BF16 routed expert is 9.375 MiB, and the full routed-expert set is
about 225 GiB.

The routed experts are copied explicitly into host RAM during model startup:

- expert ownership is round-robin: `expert_id % world_size == rank`;
- every rank owns 128 experts per layer, or 56.25 GiB across 48 layers;
- the four disjoint rank shards together hold exactly one copy of the expert
  set;
- the shard is pinned by default, with an all-or-nothing pageable fallback;
- only active experts are copied to the rank's GPU and all MoE arithmetic runs
  on the GPU;
- the 95 GiB PLE table remains host-side.

This document uses *host-resident* only for the explicit 56.25 GiB/rank copy.
A warm mmap or page-cache hit is not treated as host residency. The latest warm
preload took about 53 seconds per rank; it is a one-time load cost and is not
included in steady-state throughput.

## Authoritative 8K TP4 prefill result

Measurement conditions:

- 8192 real tokenizer tokens;
- one 8192-token chunk;
- TP4;
- expert cache capacity 1024;
- pinned 56.25 GiB host expert shard per rank;
- first pass discarded as warmup;
- no synchronization-heavy profiler in the timed region;
- all rows below produced token 97792;
- preload excluded.

### Accuracy-first path

The accuracy-first candidate uses transient FP16 tensor-core dense projections
and the CUDA hyper-connection RMSNorm/injection path. Decode rows stay BF16
because all prefill precision gates require at least 128 rows.

| Path | Warm samples | Mean wall | Mean throughput | Peak/rank |
|---|---:|---:|---:|---:|
| BF16 baseline | 788.07 / 784.47 tok/s | 10.419 s | **786.27 tok/s** | 14.108 GiB |
| FP16 dense + CUDA HC | 951.04 / 947.77 tok/s | 8.629 s | **949.41 tok/s** | 13.839 GiB |
| BF16 baseline recheck | 770.43 / 767.59 tok/s | 10.653 s | **769.01 tok/s** | 14.108 GiB |

The candidate is a 20.7% gain over the first baseline mean and reaches a stable
about 950 tok/s, just below the 1000 tok/s target.

A 512-token prompt plus 32 forced-token decisions was checked on Chinese,
English, and code inputs. Chinese and code matched all greedy decisions. The
English arm changed decisions 4 and 23; the baseline margins were 0.7500 and
0.1875 respectively. Therefore this path is still opt-in rather than a strict
parity default.

The prefill precision switches do not affect single-token decode. With cache
1024 and a 512-token prefix, a separate 32-step forced-token A/B measured:

| Prompt | BF16 decode | Candidate decode |
|---|---:|---:|
| Chinese | 4.365 tok/s | 4.363 tok/s |
| English | 4.378 tok/s | 4.388 tok/s |
| Code | 4.367 tok/s | 4.370 tok/s |

This is within run noise and confirms the rows=1 BF16 fallback protects decode.

### Throughput-first path: target exceeded, not promoted

A more accurate FP16 GEMM form uses FP16 tensor-core inputs, FP32 accumulation,
and one BF16 output cast. Applying it to dense projections and the HC mix-down
and mix-up projections gives:

| Path | Warm samples | Mean wall | Mean throughput | Peak/rank |
|---|---:|---:|---:|---:|
| BF16 baseline | 784.68 / 780.69 / 777.29 tok/s | 10.491 s | **780.89 tok/s** | 14.108 GiB |
| FP32-output dense + HC down/up | 1063.42 / 1061.49 / 1059.48 tok/s | 7.718 s | **1061.46 tok/s** | 13.961 GiB |
| BF16 baseline recheck | 765.46 / 762.26 / 760.08 tok/s | 10.742 s | **762.60 tok/s** | 14.108 GiB |

This exceeds the 1000 tok/s goal by 6.1% and is 35.9% faster than the first
baseline mean. It is not promoted as a correctness-safe default:

- 512-token Chinese and code prompts matched all 33 decisions;
- the English prompt changed decisions 19 and 23;
- decision 19 was an exact baseline tie, but decision 23 had a baseline margin
  of 0.1875 and the candidate selected the other token with a 0.9375 margin;
- a 2048-token plus 8-decision matrix matched all decisions on all three
  prompts, so the mismatch is prompt- and margin-sensitive rather than a
  general runtime failure.

The correct status is therefore: **the performance target is reached by an
experimental precision mode, while the accuracy-first mode remains about
950 tok/s**.

The 8192-token chunk is also the correct operating point. A clean strict-BF16
sweep measured 154.70, 257.88, 409.14, 580.68, and 753.25 tok/s for chunks 512,
1024, 2048, 4096, and 8192 respectively. Smaller chunks do not improve
correctness and sharply increase repeated per-layer dispatch, communication,
and active-expert staging overhead.

## Remaining prefill attribution

A synchronization-heavy real TP4 profile is for attribution only. Under the
accuracy-first configuration, the main 8K phases were:

| Phase | Calls | Total | Per call |
|---|---:|---:|---:|
| Routed MoE | 48 | 7.777 s | 162.03 ms |
| Attention | 48 | 1.484 s | 30.92 ms |
| MLP hyper-connection | 48 | 0.863 s | 17.97 ms |
| Attention hyper-connection | 48 | 0.844 s | 17.58 ms |
| MLP all-reduce | 48 | 0.666 s | 13.88 ms |
| Attention all-reduce | 48 | 0.356 s | 7.41 ms |
| Router | 48 | 0.207 s | 4.31 ms |
| PLE | 1 | 0.149 s | 148.98 ms |

MoE attribution on rank 0 reported 4.189 seconds staging, 3.090 seconds
compute, 4652 active expert calls, and 42.59 GiB H2D. Turning off the existing
copy-stream prefetch reduced the no-profile path from about 950 tok/s to about
727 tok/s, so the overlap must remain enabled.

Attention sub-phases were:

| Sub-phase | Calls | Total | Per call |
|---|---:|---:|---:|
| Core | 48 | 0.808 s | 16.84 ms |
| Projection | 48 | 0.294 s | 6.12 ms |
| QSA indexer | 12 | 0.165 s | 13.73 ms |
| Output | 48 | 0.139 s | 2.89 ms |
| GatedDeltaNet convolution | 36 | 0.076 s | 2.12 ms |

The isolated 8K GatedDeltaNet layer measured about 20.2 ms with transient FP16
projections. Its recurrent core was 8.66 ms. Sweeping 1/2/4/8 work groups per
CTA did not improve the full layer, so CTA grouping is not the next lever.

A synchronization-free 512-token PyTorch trace also showed that all 96 NCCL
collectives ran on the default compute stream: NCCL occupied 327.46 ms, BF16
GEMMs 1.353 s, and pinned H2D copies 2.651 s, with zero timeline overlap between
NCCL and either GEMM or H2D. Moving NCCL to a separate stream reduced an
isolated 8192 x 2560 all-reduce from 7.54 to 7.28 ms, but an independent 0.58 ms
GEMM still did not overlap because both kernels saturate SM resources. The real
8K path gained only about 0.8%, so simple stream reassignment is not a path to
strict-parity 1000 tok/s. Projection slicing or a reduce-scatter formulation is
needed to create useful pipeline granularity.

## Numerical gates and rejected candidates

The parity harness generates one BF16 greedy chain per prompt, feeds that same
chain to every candidate, and reports the BF16 top-1/top-2 margin for every
mismatch. Exact ties are separated from real flips.

### Rejected or diagnostic precision paths

| Candidate | Kernel or 8K result | Parity verdict |
|---|---:|---|
| FP16 dense projections alone | fast | English decision 1 flips at BF16 margin 3.4375 |
| FP32-output dense projections alone | 885.48 tok/s | Only an exact-tie English decision changes; no non-tie flip in the 512+32 matrix |
| FP32-output dense + CUDA HC | 956.80 tok/s | English decision 23 flips at margin 0.1875; grouped RMSNorm is not bit-exact |
| FP32-output dense + HC mix-down only | 1013.28 tok/s | English decisions 0 and 1 flip at margins 4.3750 and 3.4375 |
| FP32-output dense + HC mix-up only | 1005.89 tok/s | English decision 0 flips at margin 4.3750 |
| FP32-output dense + HC injection only | 940.27 tok/s | English decision 23 flips at margin 0.1875 |
| FP16 dense + FP16 HC down/up | earlier 1043.35 tok/s was not reproducible | Chinese decision 5 flips at margin 0.9375 |
| FP32-output dense + HC down/up | 1061.46 tok/s | English decision 23 flips at margin 0.1875 |
| Full FP16 routed MoE | 2.65x kernel speedup | English decision 0 flips at margin 4.375 |
| FP16 gate / BF16 down MoE | 1.75x kernel speedup | real Chinese and English flips |
| BF16 gate / FP16 down MoE | 1.23x kernel speedup | English decision 23 flips at margin 0.1875 |

The routed-MoE FP16 experiments are kept opt-in and disabled by default. Their
small average activation error is not sufficient to preserve real generation.

### Exact-BF16 hyper-connection limits

Two scale-then-activation pairs now use dedicated CUDA kernels when
`POCKETLLM_QWEN4_HC_CUDA=1`: HC mix-down divide plus SiLU, and block-injection divide
plus `2 * sigmoid`. Direct tests at 1, 7, and 8192 rows are bit-identical to the
PyTorch BF16 expressions. At 8192 rows the isolated pairs improved from about
0.069 to 0.054 ms and from 0.083 to 0.049 ms respectively. Their aggregate
end-to-end ceiling is only about 5 ms per 8192-token prefill because each pair is
small relative to the model wall.

The larger CUDA grouped RMSNorm used by the same HC switch is not strictly
bit-exact to PyTorch: on real layer-0 weights at 8192 rows, 99.999541% of values
were equal and the maximum absolute difference was 0.0625. Therefore references
to the CUDA HC configuration mean an accuracy-tested near-parity path, not a
bit-exact implementation. The activation kernels themselves remain exact.

A steady-state TP4 run with two global 8K warmups is the usable end-to-end
measurement for the two activation fusions. With prompt=8192, chunk=8192,
TP4, and expert cache=1024, strict BF16 measured 769.12 / 765.31 tok/s
(767.21 mean), CUDA HC measured 801.09 / 799.17 tok/s (800.13 mean), and the
BF16 recheck measured 752.99 / 750.73 tok/s (751.86 mean). All arms produced
token 97792. This comparison includes the pre-existing non-exact CUDA grouped
RMSNorm and injection kernels; it does not isolate the roughly 5 ms activation
fusion gain. A later confirmatory run suffered an external machine-wide
slowdown from about 786 to 428 tok/s and is discarded rather than averaged into
these numbers. A 512-token Chinese prompt plus eight forced decode decisions
also kept every greedy decision unchanged; the full-logit delta reflects the
pre-existing grouped-RMSNorm approximation rather than rows=1 activation use.

An isolated strict-BF16 HC call at 8192 rows measured 25.363 ms. The two BF16
mix GEMMs consumed 7.071 and 7.610 ms; a kernel trace attributed 14.134 of
16.950 ms of CUDA time to `aten::mm`. On SM75 these GEMMs use
`magma_sgemmEx_kernel` at about 50% occupancy.

A separate optimistic compute-floor probe replaced every staged expert pair by
one GPU-resident pair, removing all 42.76 GiB of expert H2D while preserving the
same dispatch and GEMM shapes. At prompt=8192/chunk=8192/TP4, the real path
measured 779.93 tok/s (10.504 s) and the no-H2D floor measured 909.07 tok/s
(9.011 s). Perfect staging overlap therefore still cannot reach 1000 tok/s,
which requires 8.192 s; at least 0.819 s of compute and/or launch overhead must
also be removed. Elementwise or reduction fusion cannot supply that margin. The
remaining strict-path work must target the HC/attention/MoE compute structure,
not H2D overlap alone.

The weighted mean is already a 100%-occupancy BF16 multiply followed by a
100%-occupancy `MeanOps` reduction. Replacing `.mean(-2)` with
`.sum(-2) / hc_count` is bit-identical but slower. Hand-written left-to-right,
pairwise, and `einsum` reductions do not reproduce PyTorch's reduction order;
`einsum` is also much slower. These variants were rejected.

### Split-K and cached FP16 weight experiments

Splitting the K dimension into FP16 tensor-core partial GEMMs does not reproduce
the SM75 BF16 linear result. Equal-element fractions remained roughly
0.9984-0.9997 for every tested chunk size, while BF16 itself was not bit-equal
to an FP32 reference. Chunk sizes below 256 were also slower than BF16.

Caching FP16 mirrors of dense weights did not materially improve isolated GEMMs:
for example, the 8K GatedDeltaNet QKV projection was 2.360 ms with a transient
weight conversion and 2.365 ms with a cached FP16 weight. Permanent mirrors
would consume substantial VRAM without a measurable benefit, so weights remain
BF16 and conversion is transient.

## Confirmed architecture decisions

1. **Keep routed experts explicitly resident in host RAM.** Mechanical-disk mmap
   faults must never be part of steady-state inference.
2. **Keep per-rank round-robin ownership.** TP4 stages one disjoint 56.25 GiB
   shard per rank and never duplicates routed-expert work.
3. **Keep active-only H2D.** The full-local-layer grouped BF16 path increases
   H2D bytes by about 31.6% and regressed real 8K TP4 prefill to 577.13 tok/s.
   `POCKETLLM_QWEN4_MOE_GROUPED` stays disabled by default.
4. **Keep MoE copy-stream prefetch enabled.** It is worth about 950 versus
   727 tok/s in the current accuracy-first path.
5. **Keep prefill and decode precision paths separate.** Large prefill rows may
   use the experimental FP16 tensor-core modes; rows=1 remain BF16.
6. **Do not promote a candidate from token 0 alone.** Real forced-token matrices
   and baseline margins are required.

## Reproduction commands

Accuracy-first 8K A/B:

```bash
PYTHONPATH=. \
POCKETLLM_QWEN4_DENSE_FP16=1 \
POCKETLLM_QWEN4_HC_CUDA=1 \
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun \
  --standalone --nproc_per_node=4 \
  .scratch/bench_qwen4_exp_tp_fp16_matrix.py \
  --prompt-tokens 8192 --chunk-size 8192 \
  --expert-cache 1024 --repeat 3 --global-warmup 2 \
  --arms baseline,dense_hc_cuda,baseline_recheck
```

Throughput-first 8K A/B:

```bash
PYTHONPATH=. \
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun \
  --standalone --nproc_per_node=4 \
  .scratch/bench_qwen4_exp_tp_fp16_matrix.py \
  --prompt-tokens 8192 --chunk-size 8192 \
  --expert-cache 1024 --repeat 4 --global-warmup 2 \
  --arms baseline,dense_fp32_hc_down_up_fp32,baseline_recheck
```

Forced-token margin matrix:

```bash
PYTHONPATH=. \
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun \
  --standalone --nproc_per_node=4 \
  .scratch/check_qwen4_exp_tp_long_matrix.py \
  --prompt-tokens 512 --decode-tokens 32 --expert-cache 0 \
  --prompts chinese,english,code \
  --arms dense_fp32_hc_down_up_fp32
```

## Theoretical ceilings of the heterogeneous scheme

All constants in this section were measured on this machine (4x RTX 2080 Ti,
TP4, PCIe Gen3) with `.scratch/probe_*.py`. Rows marked *projected* combine
measured constants arithmetically and have not been observed end to end.

### Hardware roofs

| Roof | Measured |
|---|---:|
| Pinned H2D, one GPU active | 10.49-10.82 GiB/s |
| Pinned H2D, all four GPUs concurrent | 42.42 GiB/s aggregate (10.61 GiB/s per GPU) |
| GEMM, BF16, 8192x4096x4096 | 7.50 TFLOP/s |
| GEMM, FP16, 8192x4096x4096 | 54.71 TFLOP/s |
| DRAM copy | 494.5 GiB/s effective |
| NCCL all_reduce 8192x2560 BF16 | 7.146 ms (5.47 GiB/s) |
| NCCL all_reduce 1x2560 BF16 | 0.092 ms |

Two of these matter more than the rest. **H2D does not contend across the four
GPUs**: each rank keeps its full ~10.6 GiB/s while all four copy at once, so
per-rank staging math is additive, not shared. And **BF16 GEMM runs 7.3x slower
than FP16** on SM75, because Turing has no BF16 tensor-core path — this single
ratio dominates the prefill ceiling.

### Prefill ceiling

Summing the measured per-kernel costs of one 8192-token TP4 pass (36
`linear_attention` + 12 `qwen_sparse_attention` layers, per-rank shapes):

| Arm | Floor | Ceiling |
|---|---:|---:|
| BF16 everywhere | 7.217 s | 1135 tok/s |
| FP16 dense, BF16 MoE (this PR's arm) | 4.123 s | 1987 tok/s |
| FP16 dense + FP16 MoE (projected) | 2.979 s | 2750 tok/s |
| + tensor-core QSA core, 3.44x (projected) | 2.126 s | 3853 tok/s |

Measured end to end for comparison: BF16 786.27 tok/s, FP16 dense + CUDA HC
949.41 tok/s, no-H2D compute floor 909.07 tok/s. The shipped arm therefore runs
at **48% of its own GEMM floor**; the missing 1.44-2.1x is launch overhead,
elementwise traffic, and Python dispatch, not arithmetic.

Phase attribution of the floor:

| Phase | BF16 | FP16 dense |
|---|---:|---:|
| routed MoE (x48) | 1.326 s / 18.4% | 1.326 s / 32.2% |
| QSA core (x12) | 1.202 s / 16.7% | 1.202 s / 29.2% |
| hyper-connection (x48) | 1.444 s / 20.0% | 0.234 s / 5.7% |
| GDN dense proj (x36) | 1.154 s / 16.0% | 0.158 s / 3.8% |
| NCCL all_reduce (x96) | 0.686 s / 9.5% | 0.686 s / 16.6% |
| shared + router (x48) | 0.657 s / 9.1% | 0.095 s / 2.3% |
| GDN recurrence (x36) | 0.290 s / 4.0% | 0.290 s / 7.0% |
| QSA dense proj (x12) | 0.378 s / 5.2% | 0.053 s / 1.3% |
| QSA indexer (x12) | 0.081 s / 1.1% | 0.081 s / 2.0% |

The hyper-connection is the largest single BF16 phase at 20%, which is why the
FP16 path bought 20.7% end to end. Once dense projections are FP16, the profile
inverts: routed MoE and the QSA core become 61% of the floor together, and both
are still BF16.

The QSA core sustains only 1.03 TFLOP/s, 14% of the BF16 GEMM roof, because it
runs as two `einsum` calls with an FP32 softmax between them. This is the same
shape of problem the `cpp_engine` Qwen path already solved with an SM75
`m16n8k8` kernel, where the analogous exact-GQA prefill gained 3.44x.

**H2D is not the prefill bound and cannot become one soon.** A cold 8192-token
chunk touches all 128 local experts across 48 layers = 56.25 GiB/rank = 5.31 s
at 10.6 GiB/s. Measured exposed staging was only 1.492 s of a 10.504 s wall, so
3.81 s already hides behind compute. Staging becomes the wall only if compute
drops below 5.31 s, i.e. past **1544 tok/s**. Between today's 949 tok/s and
1544 tok/s, every gain must come from compute; above 1544 tok/s the scheme turns
PCIe-bound and needs quantized experts to go further.

### Decode ceiling

Decode is bound by bytes, not arithmetic, and the gap is 3x.

Per token per rank: top-10 of 512 experts, round-robin ownership, 48 layers =
120 expert copies = 1125 MiB = 1.099 GiB -> **103.6 ms of H2D**. Measured decode
compute at 8192 context is 35.0 ms (dense 11.2 + full attention 8.8 + routed MoE
4.4 + NCCL 8.8 + lm_head 1.8), i.e. 28.6 tok/s if H2D were free.

| Scenario | ms/token | Ceiling |
|---|---:|---:|
| Serial H2D + compute | 138.6 | 7.2 tok/s |
| Perfect H2D/compute overlap | 103.6 | 9.6 tok/s |
| + 31% resident cache, uniform routing | 71.5 | 14.0 tok/s |
| INT8 experts | 51.8 | 19.3 tok/s |
| INT8 + 62% resident | 35.0 | 28.6 tok/s |
| INT4 experts | 35.0 | 28.6 tok/s |
| Any scheme with no expert H2D | 35.0 | 28.6 tok/s |

**28.6 tok/s is the hard decode ceiling of this model on this hardware**, and it
is a compute number, not a transfer number. Nothing done to the expert pipeline
can pass it; INT4 experts and fully-GPU-resident experts land on exactly the
same 35.0 ms because at that point H2D has fallen under the compute time. Past
28.6 tok/s the levers are NCCL (8.8 ms, 25% — 96 small all_reduces), the
`linear_attention` dense projections (11.2 ms, 32%), and full attention (8.8 ms,
25%).

Residency is set by the memory budget: 22.0 GiB per card minus 2.54 GiB dense
per rank minus ~2 GiB KV and activations leaves 17.46 GiB, or 1907 of the 6144
local experts (31%). That is what makes the 14.0 tok/s row reachable with BF16
experts and no quantization.

Quantized residency changes the picture qualitatively:

| Expert dtype | Total | Per rank | + dense | Fits 22 GiB |
|---|---:|---:|---:|---|
| BF16 | 229.7 GiB | 57.4 GiB | 60.0 GiB | no |
| INT8 | 112.5 GiB | 28.1 GiB | 30.7 GiB | no |
| Q4_K-ish (4.5 bit) | 63.3 GiB | 15.8 GiB | 18.4 GiB | **yes** |
| INT4 | 56.2 GiB | 14.1 GiB | 16.6 GiB | **yes** |

At 4 bits the entire routed-expert set fits in GPU memory across the four
ranks, which removes the heterogeneous H2D path from decode altogether and puts
decode at its 28.6 tok/s compute floor — a 3.0x lift over the 9.6 tok/s
overlap-only ceiling. This is the one structural change that moves decode by
more than a fraction.

**Speculative decoding does not help the bytes.** Expert selection is nearly
independent across positions: verifying 8 tokens touches 74.7 distinct experts
per layer instead of 8x10=80, only 6.6% fewer bytes per token. Batched verify
amortizes compute, which decode is not bound by, and leaves the actual bound
untouched.

### Ranked levers

Prefill, in order of measured headroom:

1. FP16 or INT8 routed-expert MoE — 32.2% of the FP16-dense floor, still BF16.
2. Tensor-core QSA core — 29.2% of that floor at 14% of the GEMM roof; the
   `cpp_engine` `m16n8k8` kernel is the existing template.
3. Close the 1.44-2.1x gap between the 4.123 s floor and the 8.629 s wall —
   kernel fusion and launch reduction, not new math.
4. HC projection TP slicing — element-wise validated, real all-gather cost
   still unmeasured.

Decode, in order:

1. 4-bit routed experts, fully resident — 9.6 to 28.6 tok/s, 3.0x, and the only
   lever that removes the bound rather than shrinking it.
2. Fewer, larger NCCL calls — 8.8 ms of the 35.0 ms compute floor.
3. `linear_attention` dense projections — 11.2 ms, the largest compute phase.

### Probes

```bash
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
  .scratch/probe_h2d_roof.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
  .scratch/probe_compute_roof.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
  .scratch/probe_attn_roof.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/torchrun --nproc_per_node=4 \
  .scratch/probe_decode_layer.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/python .scratch/probe_gdn_and_budget.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/python .scratch/ceiling_model.py
/home/lvyufeng/miniconda3/envs/deepseek/bin/python .scratch/ceiling_breakdown.py
```

A caveat on one number: an earlier PyTorch loop measured the GDN recurrence at
596 ms per 8192 rows. That is the naive Python scan, not the shipped kernel,
which measures 8.05 ms. Only the kernel number is used above.

## Open issue: the 6.0x decode gap on RTX 4090

**Date:** 2026-10-17
**Hardware:** 8 x RTX 4090 (sm_89), TP8, torch 2.11.0+cu130, against 8 x RTX 3090 (sm_86), TP8.
**Status:** unexplained. Every cause proposed so far has been eliminated by measurement.

This section exists so the next person does not re-run the eliminations. It records what has
been ruled out, with the evidence; what is still standing; the claims that were **withdrawn**;
and the one open confound. Read the retractions before trusting the table.

### The observation

The same model, same checkpoint, same TP8, same 58-token prompt:

| | 8 x RTX 3090 | 8 x RTX 4090 |
|---|---|---|
| decode, steady state | **5.54 tok/s** | **0.9185 tok/s** |
| decode step | ~189 ms | ~1826 ms |

Both boxes were verified to run `world_size=8`, `nproc_per_node=8`, and 64 experts/layer per
rank. This is a real **6.0x**, and it is in the direction that a faster card should not be.

The 4090's steady state is 124 generated tokens stopped on a real end-of-sequence token
(`stopped: eos`), `--expert-resident`, `stage_bytes`/`stage_calls` 0/0, exit 0. The 3090's
number is on the staging path. That asymmetry matters and is addressed under *Open confound*.

### Eliminated, with the evidence

| Suspect | How it was eliminated |
|---|---|
| **The extension never loaded** | The `.so` was genuinely stale at first (built Oct 9, torch swapped Oct 17; `load_cuda_kernel()` returned `None`). After a rebuild the matched config returned `decode_s` **78.52 s -> 78.96 s**, a 0.56% difference over a 394 s wall. A torch fallback does not reproduce a fused-kernel timing to 1%. |
| **No fused kernel on the decode path** | The MoE extension is *not* on the decode path: `_grouped_bf16` is its only consumer and four gates close — `POCKETLLM_QWEN4_MOE_GROUPED` defaults `"0"` (`moe.py:222`), `rows < grouped_min_tokens` (128) fails at decode, `_dequantized_layer_tensors` returns `None` on an FP8 shard (`moe.py:284`), and world size 1. The attention QSA kernel *is* on the path, and was traced running: **2261 calls, 2261 `path=cuda`, 0 fallback** in a 24-token run — with the rate unchanged at 0.783 tok/s. |
| **Staging / PCIe** | Device-resident experts remove **100%** of the H2D (20.52 GiB, 4483 misses -> 0.00 GiB, 0 misses) and buy **+5.0% decode / +10.6% prefill**. On the 3090 the same mode gives **2.7x prefill**. The bytes were never the dominant term on sm_89. |
| **Different bytes per call** | `stage_bytes`/`stage_calls` is 4.6864-4.6897 MiB on every 4090 run against the 3090's 4.6881 — the same regime on both boxes, six runs. |
| **Silicon** | Clocks 2520-2805 MHz against a 3105 max; power 54-79 W against a 450 W cap; PCIe gen4 x16 full width; `SW Power Cap` and `HW Slowdown` both *Not Active* on all 8 cards; a bare GEMM reaches ~62 TFLOPS. Not throttled, not link-degraded. |
| **Torch version** | torch 2.11.0+cu130 is **13-15% slower** than 2.9.1+cu128 on this box — the hypothesis was backwards. It is a real regression, not a hidden fast path. |
| **Collectives** | An uncontended NCCL probe times the exact collective at 0.274 ms; 2 per layer x 48 = ~1.4% of a 4090 step. (The corresponding figure is ~22% of a *3090* step, because that step is 9.7x shorter — the absolute costs agree, the denominators do not. Neither share generalises.) |
| **Storage** | `/workspace` is local NVMe (`/dev/nvme1n1p1 ext4`) at 1.3 GB/s, 134x the 3090's 9.7 MB/s network path. |
| **NUMA / affinity** | 4 NUMA nodes (not 3), GPUs on nodes 0 and 2, every rank bound to its local 12 logical CPUs — which is the correct 24 physical cores per node over 4 GPUs. Not misfiring. |
| **Prefill-vs-decode division** | Both boxes use the same 58 prompt tokens, so `prefill_tps` divides by the same constant and is directly comparable. |

### Still standing

- **Both boxes run the same per-expert path.** `_grouped_bf16` is off by default, gated at
  `rows >= 128`, and returns `None` for an FP8 shard, so prefill and decode both run the
  per-expert loop, one expert at a time.
- **The 4090 does identical work at identical bytes and takes 6.0x longer.**
- **A candidate cost that no current counter sees.** `_expert_matrix` (`moe.py:56-69`, the
  dequantize call at `:67`) is invoked **inside** the per-expert loop (`moe.py:117-118` and
  `:148-149`) and dequantizes **both** expert weights, per expert, per layer, per step.
  `dequantize_block_fp8` (`quant.py:59`, `values = code.to(dtype)` at `:80`) materialises a
  **full dense bf16 copy** that is discarded after two GEMMs. One routed expert is 9.375 MiB
  in bf16 — exactly **2x** its 4.6875 MiB FP8 footprint, and `stage_bytes` counts only the
  FP8 H2D, so it is **blind to this entirely**. This is consistent with the resident result:
  residency changes where the codes live, not that they are dequantized per expert per step.

  **This is not yet shown to be the gap.** It is a real, unmeasured cost that predicts the
  observed uniformity across phases. It would only be *the* gap if the per-expert dequant is
  differentially expensive on sm_89; if it costs the same on both boxes, it is a cost but not
  the difference. The test is count and per-call time of the dequant on both boxes.

### Withdrawn claims

A record that keeps its retractions is worth more than one that looks clean. These were stated
during the investigation and are **not** evidence:

- **"The 4090 stages 65% more bytes"** (20.52 vs 12.43 GiB) — a **token-count artifact**:
  64 tokens against 32. Per-call volume is identical on both boxes. Withdrawn.
- **"~22,000 launches per step"** ceiling — not measured, withdrawn.
- **"Collectives are ~44% of a step"** — an artifact of `profiler.py`'s `scope()`, which
  synchronizes on both entry and exit, so every scope absorbs its predecessors' undrained work
  and reads high. The tell was two byte-identical `[1,1,10240]` reductions reporting 28.2% and
  3.5%. Withdrawn; see the measured ~1.4% above.
- **"Attention is 38.9% of a 4090 decode layer"** — same artifact. Re-measured with
  `attention.py`'s own self-timer (`POCKETLLM_QWEN4_ATTN_PHASE_PROFILE=1`, `time.perf_counter`
  between explicit syncs), attention is **~21.5%** (core 12.9 / projection 3.3 / output 2.4 /
  indexer 1.6 / conv 1.3). Withdrawn.
- **"Prefill is capped at ~7 tok/s, a low-arithmetic-intensity artifact"** — superseded:
  prefill is 19x more *efficient* per token than decode within a box (2.34 vs 45.38 ms/token),
  which is the healthy shape, but that does not explain a **cross-box** prefill gap at
  identical prompt length. The two comparisons are different and only the first was healthy.
- **"The 4090's load is 145 s vs the 3090's 21.7 s"** — `load_s` varied 82 / 120.8 / 145.0 s
  across three identical-config 4090 runs. **Neither box's load time should be quoted.**

### Open confound

There is **no apples-to-apples cross-box pair.** The 3090's figures are on **torch 2.6.0+cu124**
and its resident-mode run has not been taken at all; the 4090's are on **2.11.0+cu130**. Since
torch 2.11 is measurably 13-15% slower here, the two boxes are not on a common baseline. The
operator requires 2.11 on all legs. **Until the 3090 is re-baselined on 2.11 and run in
resident mode, the 6.0x is a real observation but not a clean experiment.**

### What would settle it

A per-expert marginal-cost instrument on the **real** path — cost as a function of expert count
at fixed rows, separating per-expert overhead from per-call overhead, with a **count and a time
per expert**. Two outcomes, both publishable:

- If the 4090 issues more launches per expert, the count explains it and the ~6.4 us flat
  per-launch law transfers.
- If the counts are **identical** and it is still 6x, the law does **not** transfer across
  architectures — "same op count, higher cost per launch on sm_89" — which is a different
  result and must be reported as such rather than forced onto the law.

Timing must not come from a `profiler.py` scope (see *Withdrawn claims*). The gate must be
printed into the log and asserted, so a stale `.so` aborts the run rather than silently
measuring a different graph.

## Steady-state single-request decode on RTX 3090: what the 150 ms step is made of

**Date:** 2026-10-19. **Box:** 8 x RTX 3090 (`ts-208`), TP8, expert-resident, FP8 checkpoint.
**Target this was measured against:** 50 tok/s on *one* request, which needs a 20 ms step.

The step is **146-151 ms**, so 6.6-6.8 tok/s. It is not the MoE, not the collectives, and not the
host's Python: it is **8,139 CUDA ops a step**, and at B=1 each of them is a GPU-blocking launch.

| | per decode step |
|---|---|
| CUDA ops (torch profiler) | **8,139** |
| `F.linear` calls alone | 911 |
| device time, summed over all ops | 104.3 ms |
| host `tolist()` / `.item()` syncs | 97 / 0, **1.6 ms total** |
| wall / GPU-observed | 146.2 / 146.2 ms (no host gap) |

The op count is the same for 8 ranks as for one, because every rank runs all 48 layers with no
tokens to parallelise. At ~18 us a launch (measured, `cudaLaunchKernel` 176 ms over 3,339 launches)
the launches alone are the step: 8,139 x 18 us = 146 ms.

**Eliminated this round, with the measurement that eliminated each:**

- *Routing read back to the host* (`unique.tolist()` twice an expert lists). 97 syncs a step and
  1.6 ms of the 146 -- worth removing for a capture, not worth a step. The device-side variant of
  the scan was written and removed on that number.
- *Batching.* A batch of 16/32/64 rows costs 251/277/309 ms a step against 151 at one, so the
  throughput path exists (`generate_batch`) but a request's own rate is unchanged. Not the target.
- *What the kernels are.* 783 D2D copies, 694 bfloat16 copy kernels, 326 sigmoid, 244 rsqrt, 220
  reductions -- the elementwise glue of PLE, hyper-connections, the router and the norm/activation
  pairs, each its own launch at [1, 2560], where a launch is ~64x the arithmetic.
- *The GDN kernel at decode.* It is gated to `seq_len > 1`; lifting the gate measured 6.64 -> 7.26
  tok/s but changed the emitted tokens (kernel vs the fp32 reference differ by 2e-4, enough to flip
  a greedy choice), so it was reverted rather than shipped silently.
- *The whole decode graph on one card.* 213.9 ms a step single-GPU against 146 on eight, so TP is
  already a win; the capture itself fails on the host reads the PLE and router make.

## Split reduce groups: 96 collectives at 10.5 ms, 2.2 ms in four strided groups

**Date:** 2026-10-19, 8 x RTX 3090, torch 2.6 / NCCL 2.21.5, `ts-208`.

A decode step's collective cost is **fixed latency per collective, not bytes**: 96 reduces of
`[1, 1, 2560]`, `[1, 1, 8192]` and `[1, 1, 22528]` each measure 121.5 / 116.2 / 125.0 us, and one
`[96, 1, 2560]` reduce of the same total bytes measures 449 us against 425 us for the 96 separate
ones. So the lever is the number of collectives *issued at once*, not their size.

Splitting the process group into **strided** sub-groups (`rank % k`) and issuing a subset each --
every rank creates the same group lists in the same order, and each reduces its own stride -- cuts
the wall time almost in proportion:

| groups | 96 reduces take | per collective |
|---|---|---|
| 1 | 10.5 ms | 109 us |
| 2 strided | 4.4 ms | 92 us |
| 4 strided | 2.2 ms | 92 us |

**What this does not do.** A TP all-reduce sums *across all eight ranks*; a sub-group sums within
itself. Splitting is not a way to reduce faster -- it is a measurement of what is left once the
per-collective latency is amortised, and the reason a true intranode implementation (one-shot or
two-shot over the PCIe topology, or a ring whose steps are smaller than a reduction latency) is the
term worth replacing. At four groups a step's 96 collectives would cost 2.2 ms.

**Not implemented in the runtime.** It needs an all-reduce that replaces the NCCL ring for this
topology, and a way to keep the layer dependencies intact; the probe above establishes the ceiling,
not a change.

## The all-reduces, measured

The same step, with events around each `all_reduce`: **73.1 ms of the ~150 ms step**, from **192
collectives** (two a layer) at **381 us each**. That is the single largest term after the launch
count, and it is a cross-rank latency that no graph can remove -- only fewer collectives can.

**Deferring them is not a local rewrite.** The hyper-connection streams are replicated, so a
reduced value is needed before the streams are *read* again, and the second reduce (the MLP block)
writes into those streams: the split `A = A_L + A_R` is consumed by the stream reads that follow.
So the first reduce must land before `mlp_hc` reads the streams -- one collective a layer is the
floor without moving the hyper-connection mix itself onto the device, which is a redesign rather
than a flag. A draft that deferred every reduce to the end of the step was written and reverted
unshipped: it would have produced wrong logits, silently.

**Two removals measured and rejected this round:**

- `torch.compile` on every layer's attention: **145.4 -> 145.2 ms a step.** The attention module is
  the largest single term (47 ms of compute) but compiling it fuses little, because the step's
  launches are spread over the whole graph and most of them are in the parts Triton will not take --
  the hyper-connection GEMMs, the router, the resident expert loop. There is no per-module win here.
- The resident path's per-expert weight *reads*: `rows()` builds two `FP8Tensor` views per expert per
  step, and those are 4 refcounted stores -- no launch. The step's CPU profile is 7,399 launches,
  4,637 `as_strided`, 1,550 `copy_`, 1,508 `reshape`: view bookkeeping, not something a faster read
  removes.

**What 20 ms a step requires, in order of how much of the step each removes:**

1. **One collective a layer, not two** (~36 ms), by making `mlp_hc` read the already-reduced
   streams and reducing only its block output.
2. **A step-wide CUDA graph** (up to ~100 ms of launches -> one replay), which needs the PLE n-gram
   hash and the router's selection on the device first.
3. **Fused glue kernels** (the 783 copies + 694 casts, ~15% of the device time) for the replay to
   be worth it.

The budget a step has to fit into is **20 ms**, and the terms measured above add to ~146: launches
~100, all-reduces 73, MoE 29. They overlap -- the collectives and the launches are not in series --
which is why the step is 146 and not 200, and why removing any one of them alone leaves the step
well above 20 ms.

None of these is a flag; each is a change to this model's decode path, and 2 needs 1 to be a single
stream of work a graph can hold.

## Where a decode actually stands on 8xRTX 3090

Measured on `ts-208` (8xRTX 3090, TP8, `/dev/shm` checkpoint), 128 tokens from a 53-token prompt,
`--expert-resident --expert-fused-fp8`:

```
decode  8.1 tok/s     122 ms a step
prefill 59 tok/s
```

That is the only configuration that answers correctly, and every number below is measured against
it.

### The step is bound by the host, not the card

A decode step issues **24,775 aten calls** -- about 516 a layer -- and dispatch alone is 3.72 us
each, so 92.7 ms of the 122 ms a step takes is the host queueing work. Device time is 42 ms. A
breakdown of the host side:

```
24,775 aten calls   92.7 ms   (mm 773, mul 1257, copy_ 1224, empty_strided 1107, to 1403 ...)
10,066 other        40.0 ms   (cudaLaunchKernel 6938 at 3.8 us)
```

There is no hot spot: the largest single entry, `cudaLaunchKernel`, is 26.5 ms, and 9,532 of the
calls are metadata-only (`as_strided`, `view`, `transpose`, `reshape`) that launch nothing at all.
Removing any one of them changes the step by a fraction of a percent.

### What each option was worth

| Attempt | Result |
|---|---|
| act kernel vectorised reads | device time halved; device was never the bound |
| `blocks_per_expert` 32 -> 48 | small gain, kept |
| NCCL protocol (LL / LL128 / Simple) | within 3%, noise |
| NCCL algorithm (Ring / Tree), channel count | no measurable change |
| dense FP16 linear path | **slower** than the default |
| `torch.compile` on the attention half | **53% slower**; on the whole model it fails -- triton 3.6 cannot bitcast e4m3 |
| RMSNorm gain precomputed | 0.4% |
| batching | see below: correct, and the only throughput lever there is |

### Batching: the one lever, and the caveat

The step is host-bound, so its cost barely moves with the number of rows: one row is 122 ms, eight
are 210 ms, twelve are 212 ms. Measured on identical rows, which decode the same tokens as a single
request does:

```
rows  1   122 ms/step    8.1 tok/s
rows  8   210 ms/step   38.1 tok/s
rows 12   212 ms/step   56.7 tok/s
```

That is throughput and not one request's latency -- each row still advances at about 4.8 tok/s when
there are eight of them -- but it is the only thing measured here that reaches 30 tok/s at all, and
it comes from the same fact that blocks the single-request target: the cards are idle while the
host queues work, so filling them with more rows is free.

The batched path used to crash outright: `_pinned_ids` allocated a one-row buffer and a two-row
batch failed writing into it. It follows the batch now.

**It is not yet correct, and the number above should be read with that.** Identical rows do not
all decode what a single sequence decodes: the answer goes off by one position at the sixth token,
and which rows are affected depends on the batch width -- with two rows both are wrong, with four
the first two agree and the last two do not. It is deterministic, and the top-1/top-2 margin at
that step is 5.0, so it is state rather than a near-tie in the logits. Making the id copy blocking
did not change it.

A batch does not take the fused path: `Fp8SlotExpertMoE`'s decode predicate is
`hidden_states.numel() == hidden`, true for one token and false for a batch, and the fast path
itself ends in `slot_rows.sum(0)`, which folds every row into one. So a batch is routed to
`_prefill` and grouped dispatch.

That route is not the cause, which was checked rather than assumed: the decoder hands the MoE
`flat = mixed.reshape(-1, hidden)`, so the dispatch's `token_idx = slot // top_k` indexes flattened
tokens and every row is kept apart. Where the divergence actually comes from is still open; it is
not the id copy (blocking it changed nothing) and not a near-tie in the logits (the margin is 5.0).

Until then, treat the batched throughput as an upper bound measured on a path that is not verified.

### The one thing that would get there, and why it does not

A captured decode step replays in **30 ms** -- 33 tok/s, which is the target -- so a CUDA graph is
the only mechanism that removes 90 ms of host queueing at once. Two shapes of it were built and
neither produced a correct token:

* **one graph per position.** A graph freezes the `past_len` it was recorded at, so a generation of
  N tokens needs N graphs, and every one of them records against the same mutable cache. Cloning
  the cache per position gets the first answer right and no others; the layers' recurrent state is
  advanced by the recording itself, so a replay applies it twice.
* **one step-major layout for the whole generation.** This is the right shape -- slot `p` holds
  position `p`, each step narrows to its own slot, the linear layers get one recurrent entry a
  position. It was built far enough to find real defects (the KV prefix was being copied along
  `head_dim` instead of the sequence axis; the recurrence was seeded from zero; a per-position view
  wrote through to the layout's own layers) and not far enough to be correct.

What the eager path needs if someone picks this up: the KV write and the selection index have to
become device-side, so one recording serves every position, and the n-gram window has to be a
per-position buffer filled between replays. `feature/qwen4exp-pipeline-parallel` carries a partial
`spec_decode.py` that starts from the same premise.

### What a single capture would need, concretely

The measured replay is 30 ms a step, so this is not a question of whether it would be fast enough.
It is a question of what has to change for one capture to serve every position, and that came down
to four things -- each of them identified by a failed attempt, none of them speculative:

1. **The KV write is a Python slice.** `cache.k[:, :, past_len : past_len + seq_len] = key` bakes the
   offset into the recording, so a graph recorded at position 53 writes position 53 forever. The
   write has to become `index_copy_`/`scatter_` against a device-side position tensor.
2. **The KV read is a Python slice too.** `key = cache.k[:, :, : cache.length]` fixes the key length
   in the graph, which is also what fixes the attention kernel's shapes. Reading a fixed-width arena
   and masking past `length` makes the shapes position-independent -- `cuda_qsa_attention` takes no
   `past_len` of its own, it infers everything from tensor shapes, so a constant-shape call is
   enough and no kernel change is needed.
3. **The recurrent state of the linear layers has to be indexed by position**, not advanced in
   place. Advancing in place is what made a replay apply the state update twice; a per-position
   array that each step reads and the next writes is the shape that works.
4. **The n-gram window is per position**, `ngram_size` ids ending in the token that step embeds. It
   cannot be computed during the recording, because the tokens do not exist yet, so it is a buffer
   filled between replays -- pinned, because a pageable crossing is refused inside a capture.

The head is vocabulary-sharded, so the argmax cannot be in the recording either: each position
records its logits and the choice is made on the gathered row outside, one sync a step.

### Where the capture actually stops

The four prerequisites above are in place and each is verified on its own: the KV write and read
take a device position, the linear state lives one entry a position, and the rope table can be
built before a recording. A whole step can now be captured and replayed in **30.4 ms** -- 32.9
tok/s, measured with the window fill, the replay, the token choice and the id copy all counted.

It decodes the wrong thing. The output is what says so, and nothing narrower does:

```
generate [1596, 1144, 310, 5707, 310, 1156, 328, 12675, ...]
capture  [1144, 310, 5707, 13, 271, 248069, 271, 248069, ...]
```

A layer-by-layer comparison was attempted and **did not measure what it claimed**: it read the
intermediate tensors of the capture itself rather than of a replay, so the numbers it produced were
of a pass that never ran. They are not recorded here as a finding. What the same harness did measure
reliably is the QSA kernel alone, which is bit-identical under a capture, so the attention kernel is
not the explanation.

The rest -- the slot index, the arena width, the n-gram window, TF32 -- was each changed and the
end-to-end output compared, and none of them changed it. That is a real exclusion, but a coarse one:
any of them could still matter in a way this comparison could not see.

What is left is that a replay and an eager step of the same input differ somewhere inside the fused
operators, and the recorded step's error compounds until a token goes wrong. Where inside them is
open, and finding it needs a probe this work did not build.

Over a generation the difference flips an argmax where two tokens are close: the first five agree
with eager and the sixth does not. So the honest position is that the speed is reachable and the
agreement is not, and both of those are measurements rather than expectations.

### A bug that invalidated a great deal of measurement

The fused expert backend answered `nan` for every decode step for a long time, because `ShardedMoE`
masks another rank's expert with `num_experts` as a sentinel and `encode_route` tested ownership
with `index % world_size == rank`: for the sentinel that is true on rank 0, and `512 // 8` is row 64
of a 64-row table. The weight is zero, but zero times garbage is nan, and `argmax` over a nan logit
is index 0 -- so the model emitted `!` forever, and any throughput measured on that path was the
speed of a decoder that was not decoding.



Measured on `ts-208` (8xRTX 3090, TP8, `/dev/shm` checkpoint), 128 tokens from a 53-token prompt,
`--expert-resident --expert-fused-fp8`:

```
decode  8.00 tok/s    122 ms a step
prefill 59 tok/s
```

It is the only configuration that produces a correct answer, and everything below is measured
against it.

**The step is bound by the host, not the card.** A decode step issues about 7,000 kernels;
`cudaLaunchKernel` alone accounts for 6,938 of them, and the CPU profile is 148 ms against 86 ms of
device time. The largest single device term is the collective: 96 all-reduces (two a layer) of
5 KB each, 15.3 ms of a step.

**Two things this box makes worse than they need to be.** The eight cards sit on two PCIe switches
across NUMA with no NVLink between them, so `NCCL_PROTO=LL` cuts a 5 KB all-reduce from 91.5 to
74.5 us; `Simple` and `LL128` are both worse for this payload. And the resident experts hold 14 GiB
a card, which is what limits how much a captured step could ever be sized to.

**What was tried for the host cost, and what it cost to try.** A decode step can be captured as a
CUDA graph and replayed -- `--graph-window N`. One position replays correctly and is 3x faster.
Two do not, and the reason is structural rather than a missing fix:

* a graph records the position it was captured at, so a decode of N tokens needs N graphs;
* every graph is recorded against the same mutable KV cache, and captures write it;
* so the second capture's writes are what the first graph's replay reads back.

Restoring the cache between positions was tried in several forms -- snapshot and restore of the KV,
of each layer's `length`, of the linear layers' recurrent state and `initialized` flag, of the PLE
history, plus pre-growing the rotary table so the graphs' recorded addresses survive, plus blanking
the tail past each position. With one position the answer is right; with two it is not, and with
all graphs sharing a pool it is a segfault.

Cloning the cache per position is the closest this got: three positions then replay correctly
(`[1144, 1144, 1144]`, matching eager), which is what named the shared cache as the cause. It is
still not a decode. A clone seeded from the window's opening state is missing every position the
window has already decoded, and the answer repeats itself (`'We!!"! I am Qwen is Qwen is Qwen'`);
seeding each clone from the previous one instead makes it worse (`'WeWeWeWeWeWe...'`), because the
capture has to freeze the token at each position and the token is what the step before it has yet
to produce.

That is the shape of the problem: an autoregressive decode's step reads a history its own
predecessor writes, and a graph freezes the addresses it read. The versions that work in the field
put the position on the device -- `past_len` as a tensor rather than a Python int -- so one graph
serves every position. Here the rope and the causal mask already accept a tensor
(`attention.py:557`), but the KV cache is written and read by slicing to `past_len`
(`attention.py:502`, `:660`), and a slice needs an int. Moving that is a change to the cache's
addressing, not a flag.

`--graph-window` therefore stays at 0 and is marked experimental. The pieces it needed -- the
warmup that keeps cuBLAS's algorithm selection out of the capture, the cache save/restore, and a
logits selector because the head is vocab-sharded and argmax has to follow the gather -- are in
`graph_window.py` and are the part of it that is right.

**And one bug that invalidated a great deal of measurement.** The fused expert backend answered
`nan` for every decode step for a long time, because `ShardedMoE` masks another rank's expert with
`num_experts` as a sentinel and `encode_route` tested ownership with `index % world_size == rank`:
for the sentinel that is true on rank 0, and `512 // 8` is row 64 of a 64-row table. The weight is
zero, but zero times garbage is nan. `argmax` over a nan logit is index 0, so the model emitted `!`
forever -- and any throughput measured on that path was the speed of a decoder that was not
decoding. The fix is one predicate; the lesson is that a perf number needs a correctness check
beside it before it means anything.

## The captured decode, and what a capture costs

The three prerequisites above were built and measured on 8×RTX 3090 (`ts-208`, TP8, `/dev/shm`
checkpoint, 128 tokens from a 53-token prompt). Two numbers matter, and they are different numbers.

**The step itself: 36.5 tok/s (27.4 ms).** A decode step captured as a CUDA graph and replayed,
with the fused expert kernel, reads 15× faster than the 2.84 tok/s the fused path started at. The
terms, each measured on the device timeline:

| Change | Effect |
|---|---|
| `fp8_expert_act_kernel` splitting an expert's rows over the grid, not one block a slot | 2.84 → 7.03 |
| reading the hidden width off the plane shape instead of unpacking the layer for it | → 7.90 |
| the act phase addressing the packed plane directly (no unpack at all) | device time a step 168.9 → 74.9 ms |
| a lane reading a whole 16-byte plane chunk, so a warp's loads are contiguous | 25 → 28.3 GB/s of code reads |
| `blocks_per_expert` retuned for the captured step (32 → 48) | plateau, 24 is measurably worse |
| `NCCL_PROTO=LL` for the 5 KB all-reduces | 91.5 → 74.5 µs each, 33.2 → 27.1 ms a step |

That last one is this box, not the model: the eight cards sit on two PCIe switches across NUMA with
no NVLink between them, so a ring of 5 KB blocks is a latency problem, and the low-latency protocol
is the one that fits. On a machine with a real fabric it will not matter.

**The generation: about 2.6 tok/s for a 128-token request.** A graph records the position it was
captured at -- `past_len` picks a rope slice and a KV cache slice on the host -- so one capture
serves one position, and a decode of N tokens needs N captures. A capture costs **0.30 s a
position**: 124 ms of warmup (one full forward, because cuBLAS selects its algorithm on first sight
of a shape and doing that inside a capture voids it) and 176 ms of recording. Sixty-four positions
is what the resident experts leave room for, so a 128-token decode captures in rounds.

Three hundred milliseconds a position against the 13 ms a position it saves means the capture pays
for itself only past about **2700 generated tokens**. Shorter requests are slower on the captured
path than eager, and that is the whole of the gap between the two numbers above.

The capture cost was attacked and did not come down:

- **Sharing one pool and one stream across the window** removed an out-of-memory at 80 positions
  and returns the memory at `release()`, but leaves the 0.30 s.
- **Reusing windows across requests** works when the caller brings its own KV arena (the second
  request's `build()` returns in 0.000 s against 5.89 s) and fails in practice: a window is filed
  under a fixed size to be reusable, and spending a whole window on a tail that has fewer tokens
  left overruns the card.
- **`torch.compile` before the capture** does not survive this model's decode path.

What would collapse it is one graph serving every position, which needs `past_len` off the host:
the rope and the causal mask already take it as a tensor (`attention.py:557`), but the KV cache is
written and read by slicing to it (`attention.py:502`, `:660`), and a slice needs an int. That is a
change to the cache's addressing, not a flag.

So: the step is fast enough, the setup is not, and the setup is the open problem.
