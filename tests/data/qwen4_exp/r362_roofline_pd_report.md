# Qwen3.8-Flash-Next: 100 tok/s feasibility and two-host PD

Measurements collected on 2026-11-17, Asia/Shanghai. Text-only inference, two separately leased
hosts with eight RTX 4090s each. Experts use NVFP4 weight storage; dense weights and KV remain BF16,
and recurrent state / cached indexer keys remain FP32. Native `indexer_budget=2048` is unchanged.

## Decision

**72 tok/s is not a demonstrated hardware ceiling.** It was a successful **128-token** measurement,
not a guarantee for arbitrary prompts or long outputs. 100 tok/s is not ruled out by the measured
memory bandwidth, but would require a substantial latency/acceptance improvement, not a precision
flag or PD switch.

**PD is technically feasible and has now passed a real cross-host handoff.** It does not combine
sixteen GPUs for decoding: the decode host still runs TP8. Its potential benefit is phase isolation
and scheduling, not an automatic increase in single-request tok/s.

For long prompts with short outputs, **two complete replicas are usually the better aggregate-
throughput choice** on these same two hosts. A dedicated prefill host becomes the bottleneck in PD.

## Qualified decode evidence

All rates below exclude model load and prompt prefill. “Steady decode” also excludes graph setup;
“with cleanup” charges resource release after final-token delivery. The numerator is the number of
decode-produced tokens, `N-1`, not the token already produced by prefill.

| Case | Steady decode | With request cleanup | Qualification |
|---|---:|---:|---|
| Earlier depth 3, 262016 input + 128 output | 72.679 tok/s | 56.289 tok/s | All 128 ids/callbacks/ranks matched |
| Earlier depth 4, same input/output | 71.964 tok/s | 66.560 tok/s | All 128 ids/callbacks/ranks matched |
| Corrected depth 3, 261632 input + 512 output | **55.702 tok/s** | **53.030 tok/s** | **All 512 ids/callbacks/ranks matched** |
| Plain quantized greedy, same 512-output request | 37.001 tok/s | — | All 512 ids matched |

Both native cases have **262144 logical input-plus-output positions**. This is not 262144 input
positions followed by additional generation.

The first extended test of the earlier fast path failed at output index **464**. Its rate is not a
lossless performance result. The corrected 512-output run matches **the previous plain 512-token
reference as well**, so the fix did not redefine the greedy answer to make the benchmark pass.

The corrected long run used 233 verify rounds and five one-query tail steps: 238 trunk steps for
511 decode-produced tokens, **2.147 tokens/trunk step**. Its decode interval was 9.174 s; graph
setup was 8.225 s, separately charged. Its MTP prefill took 511.511 s. These scopes must not be
collapsed into an end-to-end 55.7 tok/s claim.

Sources: [earlier native 128-output evidence](r361_exact_startup_native256k.jsonl),
[invalid earlier native 512-output evidence](r361_sustained_native512_INVALID.jsonl),
[corrected complete native 512-output evidence](r362_norm_fixed_native512.jsonl).

### Why the earlier “strict” verification could still drift

Packed Q/gate projections leave gaps between heads or query rows. A one-query indexer norm can be
contiguous and choose the fused CUDA RMSNorm, while a wider verification slice is noncontiguous
and falls back to torch. A gapped QSA-head norm can also use a different torch reduction geometry
as query count grows. The fix retains each one-query slice's original strides and decode dispatch;
it does not force a contiguous Q-head copy into a fused kernel decode did not use.

With that fix and one-query FP32 indexer scoring, a real six-row diagnostic compared every layer,
logit, recurrent carry and PLE carry on every rank with serial decode: **zero row-arithmetic
differences**. Its smaller diagnostic arena did not reproduce one old reference token; this was
reported rather than hidden. Native validation uses the same clamped arena for both arms, and
subsequently passed the complete 512-output test above.

## What the theoretical analysis does and does not establish

### Bandwidth

A 512 MiB device-copy probe measured about **937–938 GB/s per card**, counting one read plus one
write, on all eight cards of the second host. This is sustained copy bandwidth, **not measured
model bandwidth utilization**.

A token does not read the entire roughly eight-GiB expert bank on every GPU. With hidden size 2560,
intermediate size 640, 48 layers and top-10 routing, an idealized selected-expert budget is:

```
48 * 10 * (3 * 2560 * 640) * 0.5625 bytes = 1.327 GB per token across the TP group
```

That is about 166 MB/card on average before routing imbalance and expert reuse. Four verification
queries have about 664 MB/card of logical selected-expert reads before reuse. Adding dense weights,
logit projections, full FP32 indexer scans and selected KV reads gives a few GB per verification
round, not a full expert-bank sweep per token. An optimistic streaming-bandwidth budget is therefore
only several milliseconds. **It cannot justify calling a roughly 38–40 ms round a bandwidth ceiling.**
These are logical byte estimates, not a profiler-derived HBM byte count; L2 reuse, repeated reads,
route imbalance and small-kernel occupancy matter.

### Latency and acceptance

The captured replay diagnostic measured about **9386 kernels per verifier replay**. NCCL kernel
busy time was about 5.38 ms/replay in that diagnostic, but includes peer waiting and is not a proven
irreducible communication floor. The trace itself perturbs timing and was at 8K; its 74 ms profiled
span must not replace the unprofiled native round measurement.

The important ratio is:

```
tok/s = verified tokens delivered / (verify + head + rewind + boundary work)
```

MTP acceptance is workload-dependent. Depth 3 can advance at most four positions per trunk verify;
the earlier native 128-output run advanced 2.702 positions/trunk step, while the corrected 512-output
run advanced 2.147. Thus **72 is not a universal rate even with the same cards and quantization**.
Perfect acceptance at an unchanged approximately 38–40 ms round would place an optimistic
short-run rate around 100 tok/s, ignoring boundary/setup costs. That is a conditional calculation,
not a measured result or a promise of perfect acceptance.

Holding the tested workload fixed, reaching 100 tok/s would require:

- Earlier 128-output case: 1.747 s -> 1.270 s, **27.3% less decode time**.
- Corrected 512-output case: 9.174 s -> 5.110 s, **44.3% less decode time**.

Meaningful remaining work is decode-geometry-preserving fusion / parallel scheduling and better
verified proposal efficiency. Blind batched GEMM, reduced indexer precision or changed top-k order
cannot be counted as speedups if they change the greedy sequence. **100 tok/s has not been achieved.**

Sources: [real device-copy measurements](r361_pd_secondary_memory_copy.json),
[replay diagnostic summary](r361_round_trace_summary.json).

## Real PD handoff proof

The prefill host froze complete consumed-prefix state before recording or delivering generation.
The decode host loaded the same checkpoint configuration, Python code and native binary and
continued without running prompt prefill.

Passed on **8192 input + 128 output**:

- All 128 destination tokens equal source generation and the plain greedy reference.
- All eight ranks agree; the rank-0 streaming callback equals the returned sequence.
- GDN conv/recurrent state, QSA k/v/raw index keys, PLE carry/history, private MTP cache and first
  draft/recycled-stack seeds are transferred as typed bytes.
- **13 FP32 pooled-key prefixes were imported and rebound to local storage; zero needed historical
  recomputation.** Eager `tensor_ready=False` does not mean its completed prefix is invalid; this
  distinction was fixed and tested.
- Cache values are copied in place. CUDA graphs, source pointers and RoPE tables are not transported;
  tables/recordings are local to the receiving process.

Destination steady decode: **66.390 tok/s**, **63.582 tok/s with cleanup**, at 8K. State restore took
1.534 s in this cold prototype. The source measured 68.964 / 65.904 tok/s on those scopes. These are
not native-256K or multi-request PD-throughput measurements.

The first host's binary needed GLIBC_2.38 and could not load on the second host's GLIBC_2.35. A
private CUDA 13.0 build on the older host passed **247 CUDA tests, zero skipped**. That portable
binary, SHA256 `4ec35943965ab53aa181ea04a01364cd55a78ee72ffb3f71522c6879e8d72f6c`, was used
identically on both PD hosts. No operating-system libraries or shared CI configuration were changed.
Both hosts already had the checkpoint; no hundreds-of-GB model copy was necessary.

Sources: [PD source proof](r362_pd_source8k.jsonl), [PD destination proof](r362_pd_destination8k.jsonl),
[typed-state replica accounting](r362_pd_state_accounting.json),
[verified actual file transfer](r362_pd_state_transfer.json).

## State volume and network cost

The only active inter-host link observed is **1GbE**. A verified 256 MiB synthetic transfer achieved
0.786 Gbit/s. The real complete-state file transfer received **1.317 GB in 17.290 s of payload phases**,
about **0.609 Gbit/s**, with all file hashes verified. This is not the complete HTTP request latency.

| State size | Eight rank-local copies | Distinct bytes after proven replica grouping |
|---|---:|---:|
| Real 8K state | **1,316,667,520 B** | **374,661,136 B** |
| Full 262144-position state, geometry projection | about **38.50 GB** | about **8.40 GB** |

The deduplicated 8K count is measured from per-tensor hashes, not assumed. KV heads replicate in
rank groups 0–3 and 4–7; raw index keys / pooled keys replicate across ranks. GDN states are unique
per rank and all eight must travel. The prototype actually sent all rank-local copies; it has not
implemented deduplicated wire transport yet.

At the measured actual-file payload rate, deduplicated native state would take roughly **110 s**
without overlap; theoretical 1GbE serialization is about 67 s. Naive native copies would take roughly
**505 s**, before other costs. This is why a simple “prefill, then copy all KV” design is unattractive.

Chunk streaming is the potential remedy. A distinct 512-position slice is about **16.19 MB**, or
0.21 s at the measured file payload rate. Current prefill takes approximately a second per 512-token
chunk, so a deduplicated asynchronous transfer can in principle keep up. The final GDN state alone
is **115.46 MB across all ranks**. A conservative final-state-plus-last-chunk tail is about 132 MB,
**1.3–1.8 s**, not 0.1 s. Per-layer overlap could hide part of it, but has not been measured.

The 25GbE-capable Ethernet ports and the InfiniBand ports were administratively/physically DOWN;
there was no usable high-speed route. No NIC configuration was changed. Faster prefill would also
raise the required link bandwidth: at the observed file rate, the distinct-state stream saturates
around **2400 prefill tokens/s**.

## PD versus two complete replicas

For a batch-1 request let `P` be prefill, `D` decode, `S` non-hidden setup, `C` cleanup and `X` transfer.
Two complete replicas have an ideal aggregate rate `2/(P+S+D+C)`. A pipelined one-P/one-D system is
bounded by the slowest stage: `min(1/P, 1/X, 1/(S_D+import+D+C))`.

For approximately 500 s of long-context prefill and only 9 s of 512-token decode, PD is almost wholly
prefill-bound. It can approximately double throughput relative to **one** co-located server only
when the two phases are balanced; it does not beat **two** otherwise equivalent complete replicas
merely by moving prefill. Here the two-replica long-prompt aggregate capacity is close to twice the
one-P/one-D capacity. Continuous batching or a substantially faster prefill engine could change this
comparison; neither is implemented by this prototype.

Runtime/state/cancellation/CLI integration passed **438 focused CPU tests**, with two explicitly
reported skips, after review. The CUDA proof remains the separate real TP8 handoff above; CPU tests
are not GPU-parity evidence. The final manifest validator also rejects duplicate layer indices and
self-consistently hashed but wrong-shape/wrong-dtype pooled payloads before any live copy begins.

**Recommendation:** do not use PD as the route from 72 to 100 single-request tok/s. Prefer two
complete replicas for prefill-dominated aggregate throughput. Develop PD only for a measured
phase-isolation / latency-SLO need, with distinct-state streaming, local prebuilt recordings and
request-level cancellation/acknowledgment. The current prototype proves exact handoff, not streaming
overlap, concurrent throughput or a production HTTP PD service.
