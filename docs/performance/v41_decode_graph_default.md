# DeepSeek-V4.1-Flash: the decode graph becomes the default

[The live page](deepseek_v4_1_flash_decode_graph_live.md) cleared the graph's *gate* and said, in its
own last bullet, that it was **not** an argument to turn the flag on: it measured a step with the
expert `_stage` copy still in the path, and its 1.43–1.56× was taken against that step. The copy has
since been removed ([device-experts page](deepseek_v4_1_flash_device_experts.md)), so the question the
default rests on — *what does the graph buy a served request now* — was open. This page is that
measurement, and on it the v41 backend's `decode_graphs` default moves from `false` to `true`.

**Decode falls from 923 to 370 ms a token — 2.49× — and output throughput from 0.72 to 2.36 tok/s,
3.3×.** The same graph is 1.34× on an 8192-token prompt, and it does a second thing the ratio does not
show: it removes the run-to-run jitter, because what the graph takes off the host is the expert-staging
variability that was the spread.

## Run record

| | |
| --- | --- |
| Model | DeepSeek-V4.1-Flash, released checkpoint, fp8 dense + packed-fp4 experts |
| Checkpoint | `/mnt/data3/DeepSeek-V4.1-Flash`, 48 shards, **no resident bank attached** |
| Configuration | TP4, one process a card, `--max-model-len 32768`, `--backend-option expert_pool_rows=148 --backend-option prefill_chunk=4096 --backend-option threads=22`, `temperature 0.0` |
| Tool | `relicllm bench serve` (the [P0.1 command](../guides/latency_metrics.md#invocation)), which launched and tore down `relicllm serve` itself |
| Scenarios | `decode` (128-in, 128-out) and `prefill-8k` (8192-in, 32-out), 8 prompts each, `--request-rate inf`, `--max-concurrency 1` |
| GPUs | 4 × RTX 2080 Ti, 22528 MiB each, idle before and after |
| Records | `tests/fixtures/perf/v41_decode_graph_eager.json`, `tests/fixtures/perf/v41_decode_graph_on.json` |
| Date | 2026-10-03 |

**The resident bank is not attached**, and that is the one thing to read this page against. The 457.8
GiB bank does not fit this host's `/dev/shm` alongside the MiMo bank, and the graph touches only the
dense tree half, so the bank changes the **absolute** ms and not the **delta** — which is this page's
subject. The absolute figures here are therefore a no-bank configuration and are **not** comparable to
the bank-attached numbers on the device-experts and live pages. If a later run wants the graph's delta
*with* the bank, it is a separate sitting.

## The measurement

Per-request TPOT (ms), successful requests, the two legs the same binary a flag apart:

```text
decode (128-in, 128-out)     TPOT median   output tok/s   per-request TPOT (ms)
  eager  --no-...graphs          923            0.72       528 766 772 923 1130 1825 2526
  graph  --decode-graphs         370            2.36       332 337 370 370  371  372  372

prefill-8k (8192-in, 32-out) TPOT median   output tok/s   per-request TPOT (ms)
  eager                          527            0.63       519 523 524 526 528 531 538 562
  graph                          393            0.69       387 388 390 392 393 398 405 411
```

**The graph's spread is 1.1×; the eager path's is 4.8×.** The eager decode column runs 528 → 2526 ms a
token across eight requests of an identical prompt, and the graphed column runs 332 → 372. The graph
is not only faster, it is repeatable — which is what makes the `decode` figure a number rather than a
range, and it is why the ratio here can be a point estimate where the live page had to give a range.

**Capture costs ~1.55 s, paid once a request, and it is recovered inside ten tokens.** The `decode`
scenario's first request is the one with no queue in front of it, so its TTFT is the cleanest read on
the capture pass: **585 ms eager against 2132 ms graphed**, a 1.55 s difference that is the forty
layers' capture. Every later request is the same as every other (the graphs are released and recaptured
per request — see below), but only the first is unqueued. At the measured saving of ~550 ms a decode
token, break-even is **under three tokens** on the decode path; at the prefill path's ~134 ms it is
about twelve. The default's own workload — a chat turn, tens of tokens — is an order of magnitude past
that.

**One request failed in the decode scenario, identically in both legs** — request 5, `Never received a
valid chunk to calculate TTFT`, in the eager and the graphed column alike. It is not the graph's, and
it is the reason both `decode` records read `status: partial` and a completed count of 7 of 8; the
`prefill-8k` scenarios are 8 of 8 in both legs.

## Why the default is a policy question and not just a ratio

`DecodeGraphs` is built fresh in `_decode_graphs` (`relicllm/models/deepseek_v4_1/generate.py`) and
released by the backend after every request (`v41_backend._release_graphs`). Release is **mandatory**:
an installed recording intercepts the next prompt's forward and dies on the sink's shape. So the graph
is not a one-time cost a served process pays on its first request the way Xing4.0's persistent holder
is — it is a **per-request** capture, and that is exactly why this page had to measure the served
number rather than inherit the step ratio. The 1.55 s is charged to every request's TTFT; it is
invisible in the decode column here only because eight 128-token prefill+decode requests each cost tens
of seconds, so 1.55 s is under 2%.

The corollary is a target, not a defect to fix in this change: **the capture is per request because the
recordings are tied to the request's cache state and must be released, and a serving load would pay it
once instead of once a request if the holder were made to outlive a request** — the Xing4.0 design. That
is a larger change and is not required for this default, whose benefit survives the per-request cost by
a wide margin on any generation longer than a few tokens.

## What this page does not say

- **No resident bank**, per the run record. The bank changes the absolute ms on both legs, not the
  delta; a bank-attached confirmation is a separate sitting.
- **It is the served number, not a step profile.** For the mechanism — 213 launches and 244 host API
  calls a block, the per-layer node counts, the graph-A → eager experts → graph-B split — read
  [the live page](deepseek_v4_1_flash_decode_graph_live.md); this page took the end-to-end reading the
  default needed.
- **It does not measure prefill's graph**, which does not exist. The prompt forward is eager on both
  legs, and `prefill-8k`'s 1.34× is the graph on that scenario's thirty-two decode tokens.