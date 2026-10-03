# Performance roadmap: becoming the best on old hardware

**Date:** 2026-10-02
**Target hardware:** RTX 2080 Ti (`sm_75`), 22 GiB, PCIe Gen3, no NVLink; Ascend 910B second.

This page is the plan, not a run record. Every number in it is quoted from a run record elsewhere on
this site and is linked; nothing here was measured for this page, and no figure in it should be
carried anywhere without the record it came from. Its purpose is to say **what stands between this
runtime and the fastest inference on the hardware it targets**, and in what order to take it apart.

## The one diagnosis

**Decode across this tree is bound by host submission, not by arithmetic.** The sharpest statement
of it is already in a run record: on DeepSeek-V4.1-Flash, a decode step is 7,076 kernels in 122
distinct shapes and the device is busy **11%** of the wall
([remaining bottlenecks](deepseek_v4_1_flash_remaining_bottlenecks.md)). That is not one model's
defect. It is the shape of the whole stack:

- **Four independent Python decode loops.** `v41`, `mimo`, `xing4` and `torch` each carry their own
  per-token Python forward. The *adapter* layer converged onto `RuntimeAdapter`; the *model* layer
  did not, so `generate` between two runtimes was measured at **1.00 similarity**
  ([per-method duplication](../architecture/per_method_duplication_2026_09.md)) — two copies of one
  body, which is where the next fix goes to die.
- **No graph capture anywhere by default.** `xing4`'s decode graph is the existence proof and the
  exception: eager **178 ms** → graphed **38 ms** a step, **4.3×**, bit-identical
  ([decode launch gap](xing4_0_decode_launch_gap.md)). V4.1 has an implementation and it is off.
- **One host round trip a token.** `int(logits.argmax())` on every step, plus 541
  `.item()/.cpu()/.numpy()/.tolist()/.synchronize()` sites outside the tests.
- **Zero `torch.compile` / dynamo / inductor** in the tree.

The corollary matters more than the diagnosis: **a faster kernel cannot fix this.** The device is
idle 89% of the time. The levers that move decode here are ones that remove host work — graph
capture, de-synchronized sampling, and taking weights off the per-step path — which is why the
ordering below is by *host work removed per unit of effort*, not by op-level speedup.

## Baseline

Where each served model stands today. Read each number's own page for its conditions; they are not
comparable to each other.

| Model | Prefill | Decode | What limits it |
| --- | ---: | ---: | --- |
| [Qwen3.8-27B-FP8](../models/qwen3.8-27b-fp8.md) | 1818.65 @ 8192 | 43.99 | Nearest to the memory bandwidth; the one to defend |
| [Qwen3.8-Flash-Next](../models/qwen3.8-flash-next.md) | 786.27 @ 8192 chunk | 3.6 served | Expert H2D over PCIe; 28.6 ceiling with the experts off the bus |
| [Ternary-Bonsai-2-27B](../models/ternary-bonsai-2-27b.md) | 636.0 @ 4096 | 25.9 | 84% of its upstream reference |
| [MiMo-V2.6-Flash](../models/mimo-v2.6-flash.md) | 104.04 @ 262144 | 6.40 (8.53 resident) | Expert copy over PCIe |
| [DeepSeek-V4-Flash](../models/deepseek-v4.md) | ~401 @ 32K | ~3.7 | CPU MoE and the fp4 expansion |
| [DeepSeek-V4.1-Flash](../models/deepseek-v4.1-flash.md) | 150.3 @ 260244 | 3.48 | Python dispatch; 11% device-busy |
| [MiniMax-M2.7](../models/minimax-m2.7.md) | ~105 | 10.32 | Attention 71% (RoPE 29%), not the collectives |
| [Xing4.0-29B-A4B](../models/xing4.0-29b-a4b.md) | 75.22 @ 4096 | 6.72 | Host launch: 22,155 dispatches a step |
| [GLM-5.2](../models/glm-5.2.md) | ~0.79 | ~0.66 | Expert staging and per-layer NCCL, two floors |

Two structural gaps sit under all of them and are visible in the adapter layer rather than in any
one number: **no continuous batching** (every runtime declares `supports_batch=False` and serializes
behind one mutable KV cache) and **no paged KV** (every runtime preallocates a contiguous buffer).

## The plan

Ordered by **benefit ÷ cost**, and gated on measurement — a phase does not start until the one
before it can tell whether it worked.

### Phase 0 — a baseline that can be trusted (first, or nothing after it is)

Every optimization below needs an answer to "did that help", and today there is no reproducible way
to ask. ~70 `bench_*` / `probe_*` / `profile_*` scripts each answer a different question.

- **P0.1 `relicllm bench` — landed.** One command, one JSON out, on vLLM's terms: TTFT / TPOT / ITL /
  E2EL / goodput for a given (model, hardware, scenario). The client moved out of `tests/` into
  `relicllm/bench/`, gained a launcher that starts `relicllm serve` and waits for `/ready`, and a
  names-its-own-argv envelope; the metric definitions are unchanged from
  [serving latency metrics](../guides/latency_metrics.md). **The mechanism landed; the numbers did
  not.** No GPU run was spent this round, so there is no committed baseline yet — that is P0.2's
  deliverable, not an omission here.
- **P0.2 a performance regression gate.** `scripts/check_perf_baseline.py`, threshold-based against
  P0.1's JSON, `skip` rather than `fail` with no GPU — the same shape `scripts/check_test_baseline.py`
  already uses. Depends on a committed baseline from a P0.1 run.
- **P0.3 fix MiMo's prefill/decode seam — landed.** The `_drain` Xing4.0 already had is now at MiMo's
  seam too, so `prefill_seconds` and `decode_seconds` are a device fact on both. One line of behaviour
  and the same one-line guard off the device; the position is pinned hermetically in
  `tests/test_mimo_serving.py`. It makes an existing number trustworthy
  ([benchmarking rules](../guides/benchmarking.md)); its own effect is **not yet re-measured**, which
  belongs to a P0.1 run.

**Acceptance:** one command runs decode-only and 8k-prefill for every served runtime and writes a
committed baseline. The command exists; the run that fills the baseline in is the next step.

### Phase 1 — serving coverage (landed: qwen4_exp)

The cheapest large win is not making a model faster; it is serving one that is fast and unreachable.
Qwen3.8-Flash-Next measured **786.27 tok/s** of prefill and had **no serving path at all** — it could
answer one prompt and exit. That gap is now closed in
[the model page](../models/qwen3.8-flash-next.md) and its PR, and the finding is the general one:
this repository's fastest models have been its least reachable.

- **P1.2 adapter de-duplication.** Five adapters, one shape. Every future optimization in Phase 2 has
  to be written five times until this lands, so it is the multiplier on all of them.

**Acceptance:** a new runtime is a spec, not a loop.

### Phase 2 — take the host out of the loop (the largest single lever)

- **P2.1 V4.1's decode graph, on by default.** **Landed.** The implementation existed
  ([decode graph](deepseek_v4_1_flash_decode_graph.md)) and the split point around the host-resident
  expert call was already the design; what was missing was the served number on the copy-removed
  config, which the roadmap's own rule (先搬，量完再说) required before the default moved. Measured at
  [2.49× the decode TPOT and 3.3× the decode throughput](v41_decode_graph_default.md), against a
  ~1.55 s a request capture that break-even clears inside three tokens — so the backend's
  `decode_graphs` is now `true` and the flag is the control column. The larger win the number points
  at is *not* taken here: capture is still per request because the recordings must be released before
  the next prompt, and a persistent holder in the Xing4.0 shape is what would stop a load paying it
  once a request.
- **P2.2 de-synchronize sampling.** Replace the per-token host round trip with device-side sampling
  and a batched read-back. Applies to all four runtimes; independent of graphs.
- **P2.3 carry graph capture to `mimo` and `qwen4_exp`.** Follows P2.1's split-point answer.

**Acceptance:** a decode step whose device-busy fraction is measured, not estimated, and is above
the 11% it started at.

### Phase 3 — the offload path (for the host-resident-expert models)

- **P3.1 delete the fp4→bf16 CPU expansion.** It is **99.7%** of V4's 15–42 s/token host path
  ([host run](deepseek_v4_1_flash_host_run.md)). Keep the weights packed and go through a raw-fp4
  kernel, or expand on the device. This multiplies directly with P2.1, because the expansion is
  exactly what stops a graph covering the experts.
- **P3.2 page-lock MiMo's expert views.** Small and known.

**Scope discipline:** [the CPU offload profile](cpu_offload_profile.md) already falsified several
offload expectations with bandwidth numbers — decode ≥5 tok/s among them. Phase 3's targets are the
ones the records show are reachable, not the ones that sound good.

### Phase 4 — concurrency (largest architectural win, highest cost)

**Paged KV and continuous batching.** Every runtime today is "make one request as fast as possible";
none is "make the box as fast as possible". This is also the prerequisite for any SLO claim that
compares to vLLM or SGLang. It is last only because it is the largest change and because Phase 0 is
what will show whether it paid.

**Not in scope:** Ascend 910B (the torch runtimes declare `cuda`/`cpu`), GLM-5.2's cache
over-allocation (reported, tracked separately), and multi-node.

## What landed this round

Phase 1's first item. Qwen3.8-Flash-Next is served through `relicllm serve --backend qwen4_exp`,
selected by `--backend auto`; a fifth architecture now fits `RuntimeAdapter` without a second loop,
which is the Phase 1.2 claim tested early. The run also produced two facts that belong in this
roadmap rather than only in the PR: a cold expert preload costs **3,446 s a rank** against a 300 s
default, and decode at the default `--expert-cache 0` is **PCIe-bound at 3.6 tok/s** with device
utilization swinging between 7% and 94% a step — which is the Phase 2 diagnosis, in one more model,
with a flag that moves it.

Then Phase 0's mechanism: **P0.1** gave every number below a reproducible command and a record
(`relicllm bench`), and **P0.3** closed MiMo's prefill/decode seam. Neither spent a GPU run, so the
baseline P0.1 exists to produce is still **P0.2's** deliverable. The point of doing them first is the
one this page opened with — a phase does not start until the one before it can tell whether it
worked, and until P0.2 can, the optimizations in Phase 2 would be landed against numbers that cannot
show they helped.

Then Phase 2's first item. **P2.1** turned V4.1's decode graph on, and it is the first change on this
page argued from a number P0.1 produced rather than from a precedent: the graph's 1.43–1.56× had been
taken with the expert `_stage` copy still in the step, so the served measurement was run again on the
copy-removed config and the default was set from it — [2.49× the decode TPOT and 3.3× the decode
throughput](v41_decode_graph_default.md), and the capture cost read off the first request's TTFT
rather than assumed, which is what made break-even a fact (under three tokens) and the default
defensible. The un-banked configuration is stated on the page, and the absolute ms are not comparable
to the bank-attached figures — the delta is what the decision rests on.

## Where the detail is

- [DeepSeek-V4.1-Flash: what is left to optimize](deepseek_v4_1_flash_remaining_bottlenecks.md) —
  the 11%-device-busy measurement and the falsified overlap that followed it.
- [Xing4.0-29B-A4B decode launch gap](xing4_0_decode_launch_gap.md) — the 4.3× graph result.
- [CPU offload and prefetch: the measured ceiling](cpu_offload_profile.md) — what offload cannot do.
- [The command-line surface against vLLM and SGLang](../architecture/cli_surface_design_2026_09.md) —
  where this runtime's interface already agrees with the two systems it is measured against.
- [Per-method duplication across the adapters](../architecture/per_method_duplication_2026_09.md) —
  the 1.00 similarity that Phase 1.2 removes.
- [Benchmarking and reporting rules](../guides/benchmarking.md) — read before quoting any number here.