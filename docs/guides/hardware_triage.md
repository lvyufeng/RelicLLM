# Hardware adaptation triage

A checkpoint arrives — a new release, a quant, a model nobody has run here — and the question is
whether it is worth adapting to this hardware, and to what service level. This page describes the
tool that answers it, `relicllm.triage`, and the reasoning behind the four gates it runs.

The tool answers with one of five words: **production**, **chat**, **demo**, **impossible**, or
**candidate**. It never answers with a single number, because the number that decides a fit is not
the number that decides a tier, and the two are derived by different means.

```bash
python -m relicllm.cli.triage /mnt/data3/DeepSeek-V4.1-Flash --host-memory-gib 1007
python -m relicllm.cli.triage /mnt/data2/Xing4.0-29B-A4B-GGUF --brief
python -m relicllm.cli.triage /mnt/data3/GLM-5.2-GGUF/UD-Q2_K_XL --json > glm.json
```

`tier` is new vocabulary on this site. It is not a performance grade; it is the answer to a
*procurement* question — can this be put in front of users, and how many of them — and the four named
tiers are deliberately coarse. Nothing here replaces
[Benchmarking and reporting rules](benchmarking.md) or
[Serving latency metrics](latency_metrics.md); the thresholds are stated in that page's vocabulary
(TTFT, TPOT, aggregate prefill) so a triage row and a benchmark row can sit in one table.

## The five outcomes

| Outcome | Meaning |
| --- | --- |
| `PRODUCTION` | Meets the box-level aggregate throughput SLO **and** the single-stream latency SLO as measured. |
| `CHAT` | Meets the latency SLO; does not reach the throughput SLO. One conversation at a time, and acceptably. |
| `DEMO` | Produces correct tokens on this hardware and meets neither SLO. Usable to show, not to serve. |
| `IMPOSSIBLE` | Does not fit at **any** precision, on max cards plus max host memory. |
| `CANDIDATE` | Passes the free static gates and has no measurement yet. Not a rank — an absence of one. |

Two rulings shape all of this, and both are enforced in code rather than in prose.

**Cost is out of scope.** The only thing that makes a model `IMPOSSIBLE` is that it does not fit. A
missing GGUF kernel, a missing loader and a missing backend adapter are *notes* and work items, never
a downgrade — a kernel is a file in
[relic-core](https://github.com/lvyufeng/relic-core) that an agent can write. That is why
`relicllm/triage/tier.py`'s `tier()` takes no reachability argument at all, and why
`tests/test_triage_tier.py` asserts its signature: a parameter added back "for completeness" would
let a model silently drop a tier over work somebody could do.

**A host offload counts as placed.** A model whose routed experts and gather tables live in host RAM
is placed, not unplaced; the 137–458 GiB banks in `/dev/shm` are how the largest models in this
roster run at all. Its tier then comes from measurement, and the host requirement is recorded as a
precondition. Where the bank is *larger* than host memory the excess comes back onto the cards and
changes the card count, rather than being noted and forgotten.

## What each gate costs

The split is by cost, and the four gates are ordered by it.

| Gate | Cost | Runs | Answers |
| --- | --- | --- | --- |
| 0 — static fit | under a second, no GPU | always | Does it fit, at which precision, on how many cards, holding what context and batch |
| 1 — reachability | seconds, no GPU | always | Adapter, loader, kernels — recorded as work items |
| 2 — measurement | **GPU-hours** | only when a human decides | What it actually runs at |
| 3 — tiering | zero | always | Which of the five words |

Gate 0 reads headers only: each safetensors shard's 8-byte length and its JSON header
(`relicllm/loader/safetensors.py:191`), or a GGUF's metadata and tensor table
(`relicllm/loader/gguf/bundle.py`). No weight is read, so a 475 GiB checkpoint opens in under a
second — which is what makes the gate free and therefore unconditional.

Gate 2 is never run by this tool. `assess()` *takes* a measurement; it does not make one. Phase one
prints the command that would produce it, in this repository's own harness and with the SLOs handed
to `--goodput`, and a person decides whether the GPU-hours are worth it. A tool that ran a
measurement by default would be a tool nobody points at a new checkpoint.

**A load-bearing asymmetry.** Gate 0 may say `IMPOSSIBLE` definitively, because its arithmetic has no
workspace, no fragmentation and no bank-miss penalty to be wrong about — anything that fails on
these numbers fails harder in reality. But its *pass* is only provisional for exactly that reason:
Xing4 with its host stated at 0.42 GiB passes the default 8k/1 point with **0.006 GiB** to spare —
about 140 tokens — so the difference between that and a crash is the workspace this calculation does
not model. So a pass produces `CANDIDATE`, never a tier.

A checkpoint whose headers will not read is also `CANDIDATE`, and this is the sharpest version of the
same rule. A missing shard is a claim about the disk, not about the model; gate 0 has not run, so
nothing has been disproved. Reporting `IMPOSSIBLE` there would be the tool's worst failure mode — a
confident answer derived from a filesystem error. The CLI exits `3` for it, apart from `1` for a real
refusal.

## The static fit calculation

One budget, three terms:

```text
usable_per_card = per_card_bytes * (1 - reserve_fraction)
cards           = min n such that  even_shard(weights, n) + kv_per_rank(L, B, n) + workspace <= usable_per_card
kv_per_rank     = bytes_per_token_per_rank * L * B        # divided by n only if the runtime shards it
```

Two of the terms are less obvious than they look.

**The reserve is a parameter, not a constant**, and it moves a real number by more than it looks. On
a 22 GiB card, 15% leaves 18.70 GiB and 10% leaves 19.80 GiB. Xing4's released GGUF is 18.72 GiB on
card, of which 4.32 GiB is resident and the other 14.40 GiB is an expert bank, so the reserve is
paid out of the KV the card has left:

```text
reserve 0.15  usable 18.70 GiB   4.32 GiB resident   14.38 GiB of KV   334,988 tokens at B=1
reserve 0.10  usable 19.80 GiB   4.32 GiB resident   15.48 GiB of KV   360,619 tokens at B=1
```

1.1 GiB of KV is about 25,000 tokens, so the reserve is printed with every fit. Where it flips a
*verdict* rather than a context is narrower than that sounds: Xing4's runtime has no tensor-parallel
path, so an under-sized host bank lands back on the card whole and the fit fails only below 0.42 GiB
of stated host memory — and at a 10% reserve, never, because 18.72 GiB plus its 0.35 GiB of
8k cache is still under 19.80.

**Context and batch are one budget, not two numbers.** The card holds `weights + L * B * kv`, so
raising the batch lowers the context and the two trade off exactly. The report prints the curve, not
a pair, because the choice is a product decision. From the tool, on Xing4's actual numbers:

```text
L=  8192 B= 1   kv   0.352 GiB/rank   headroom   14.02 GiB
L= 32768 B= 8   kv  11.250 GiB/rank   headroom    3.13 GiB
L=131072 B= 1   kv   5.625 GiB/rank   headroom    8.75 GiB
L=131072 B= 8   kv  45.000 GiB/rank   headroom  -30.62 GiB
```

With the expert bank offloaded, the same 14.38 GiB of KV covers 334,988 tokens at batch 1 — past the
131,072 the grid stops at — and 32,768 at batch 8: one person reading a long document, or eight
people chatting, and the choice is locked at startup because the cache is allocated once.

**Whether another card helps depends on the runtime, not the model.** Weights split evenly; KV splits
only if the runtime shards it. `minimum_cards` therefore takes a `sharded` flag, and a model whose
cache is *replicated* gets no help at all from a second card. That asymmetry is what makes some
models fail definitively: 24 GiB of replicated cache per rank is over budget at four cards for the
same reason it is at one.

**What ran out first is named, and `workspace` is its own answer.** Every fit row reports which of
the three terms the budget gave up on: `weights` (the parameters alone do not fit), `workspace` (a
scratch allocation fills the card before any context is priced), `kv` (the cache is what pushed it
over), `host_bank` (the offload plan has nowhere to live), or `none`. `workspace` is not folded into
`kv` because the two ask for opposite things — a KV-bound fit wants a shorter context and will be
better at one, and a workspace-bound fit is asking for a smaller scratch allocation and will not move
at any context however short. Reporting the second as the first is how a 12 GiB scratch allocation
once came to be blamed on a KV that was 0.13 GiB.

## KV geometry is not layer count times head count

`n_layers * n_kv_heads * head_dim * 2` is wrong for most of this roster, wrong in the same direction,
and wrong by between 3.5× and infinity. Four of the nine models here are *hybrid*: most of their
layers hold no growing cache at all, because they are sliding-window rings or linear attention. The
judge is the checkpoint's own `layer_types` or `compress_ratios`, and no model card mentions it.

`relicllm/triage/kv.py` derives it, and `tests/fixtures/triage/kv_geometry.json` pins the answer for
each model with the `file.py:NNN` it came from. The verified per-token costs, whole-model before
sharding:

| Model | B/token | Shape |
| --- | ---: | --- |
| DeepSeek-V4-Flash | 6,880 | 2 sliding-window + 21 compressed at ratio 4 + 20 at ratio 128, plus an indexer |
| DeepSeek-V4.1-Flash | 3,200 | **only 4 layers own a growing buffer**; 36 are consumer-only rings |
| MiMo-V2.6-Flash | 23,040 | 9 global + 39 sliding-window |
| MiniMax-M2.7 | 253,952 | 62 uniform GQA layers — no hybrid split |
| GLM-5.2 | 5,111,808 | 78 trunk layers of the file's 79, per-head, as the runtime allocates them |
| Qwen3.8-Flash-Next | 27,648 | 12 GQA + 36 GatedDeltaNet, plus a 128-wide indexer per QSA layer |
| Xing4.0-29B-A4B | 46,080 | 40 absorbed-MLA latents of 576 |
| Ternary-Bonsai-2-27B | 65,536 | 16 full + 48 linear |
| Qwen3.8-27B | 65,536 | 16 full + 48 linear |

Four further things vary per model and are recorded rather than assumed.

**Sharding.** MiMo shards its cache by KV head; V4.1, MiniMax and GLM each hold a full copy on every
rank; qwen4_exp shards. Assuming sharding where the code replicates is a 4× error at TP4, so an
unrecognised architecture defaults to `replicated` — the conservative direction.

**Preallocation.** DeepSeek-V4, V4.1, MiMo, Xing4 and Qwen3.8-27B size the cache once from
`max_seq_len` at load, so `--max-model-len` is a **prepayment** paid whether or not the context is
used. MiniMax, GLM-5.2 and qwen4_exp size it per request. These support different questions and a
report that conflates them misleads in opposite directions.

**Extra caches.** DeepSeek-V4 and V4.1 attach a second per-token buffer — an `Indexer` — to their
compressed layers, separate from the attention cache and easy to miss. On V4-Flash it is 32 of the
6,880 values per token.

**The trunk, not the block count.** A file's `block_count` includes the trailing NextN/MTP blocks a
runtime does not build, and no cache is allocated for a block that is not built, so the layer count
has to come down before the cache is priced. GLM-5.2 declares 79 blocks and builds 78; Xing4's GGUF
declares 41 where its HF config says 40. DeepSeek states the same thing in a `compress_ratios` list
rather than in a block count — one entry per layer *including* the draft heads, so 46 entries against
a 43-layer trunk — which is where the `2 sliding-window` above comes from, and why reading the list
as written gives 5. MiMo is the counter-example that keeps this a table rather than a rule: it
declares `num_nextn_predict_layers: 3` and builds all 48 blocks, so the subtraction is keyed by
architecture — `_TRUNK_ONLY_ARCHITECTURES` for the block-count case, `_trim_to_backbone` for the
ratio list — with the `file.py:NNN` that performs it on the entry, and a note printed on every report
that applies it. The error is worth 2.5% of GLM's cache and 2.4% of Xing4's.

### GLM-5.2: the tool reports the code, not the architecture

`relicllm/models/glm_dsa/architecture.py:300-308` allocates per-head K and V over `n_heads` (=64)
while the checkpoint declares `attention.head_count_kv = 1`. The real MLA latent is
`kv_lora_rank(512) + rope(64) = 576` values per token per layer; the code allocates 32,768 — a
**56.9×** over-allocation. Triage reports the code's number, because the code's number is what has to
fit on a card, and it reports the disagreement in the same line:

```text
the runtime allocates a per-head cache (64 x (256+256) = 32768 values/token/layer) while the
checkpoint declares one shared latent (512+64 = 576), a 56.9x over-allocation. This report uses
the runtime's number because the runtime's number is what must fit on a card.
```

That is what makes GLM-5.2 `IMPOSSIBLE` on this box: 5,111,808 B/token × 8,192 tokens is 39.0 GiB,
against 18.7 GiB of usable budget per card. With the declared latent it would be 0.69 GiB and would
fit. Fixing `architecture.py` is a separate piece of work; the fixture pins
`expected_bytes_per_token_whole_model` (the code's 5,111,808), `declared_latent_values_per_token`
(576) and `expected_discrepancy_factor` (56.9) side by side, so that fixing the allocation turns a
test red rather than silently changing an answer.

### GGUF carries no `layer_types`

A GGUF has no layer-type list, so a hybrid model read from GGUF alone looks uniform — Qwen3.8-27B's
65,536 B/token would read as 262,144, a 4× overstatement. Two independent fallbacks recover it. The
checkpoint's own metadata often says it (`qwen35.full_attention_interval = 4`), and where it does
not, the tensor names do: `blk.N.ssm_a` exists exactly on the linear layers and `blk.N.attn_k.weight`
exactly on the attention ones. Counting them is a measurement, and the report says which of the two
was used. Where neither is available the answer is marked `assumed` and says so.

## The thresholds and the gaps they sit in

```text
chat:       TTFT@1k <= 15 s   AND   TPOT <= 250 ms          (4.0 tok/s)
production: box-level aggregate prefill >= 222 tok/s at input=8192, output=128
```

Both were chosen because they fall in a **measured gap** rather than because they are round numbers,
and a threshold that separates nothing is worse than no threshold at all.

**TPOT ≤ 250 ms** sits between two measured arms of the same checkpoint.
`docs/performance/deepseek_v4_1_flash_served_gate.md:133` records 4.50, 4.45 and 4.53 tok/s of decode
— 222, 225 and 221 ms — on the shipping configuration, and 3.87 tok/s — 258 ms — on the compared arm.
250 ms separates exactly those two, which is a line with a decision on both sides of it. The same page
records 16.65 s of ttft at a 1,364-token prompt for the same served path, against
`docs/architecture/bonsai_2_27b_design.md:88`'s 6.44 s at 4,096 tokens for Bonsai.

**`TTFT@1k ≤ 15 s`** is the latency floor a person will actually wait for before deciding a page is
broken. It admits Bonsai's whole measured table and refuses V4.1's served path at its shortest
prompt.

**Prefill ≥ 222 tok/s at 8k** is the throughput floor of a box that is meant to answer more than one
person. The measured 8k figures above the line are 1,818.65 tok/s for Qwen3.8-27B-FP8
(`docs/models/qwen3.8-27b-fp8.md:159`), 753.25 for qwen4_exp at a 512-token chunk
(`docs/performance/qwen4_exp_performance.md:105`) and 639.1 for Ternary-Bonsai
(`docs/models/ternary-bonsai-2-27b.md:139`); below it are the served paths, V4.1's 137.5–152.0 tok/s
of prefill. Nothing measured sits near 222.

**These are this box's numbers, not universal ones.** They were set against four RTX 2080 Ti and the
one machine whose paths `CLAUDE.md` lists; a different host means different thresholds, which is why
`Thresholds` is a dataclass and every field is a CLI flag. The basis string is printed with every
report so a threshold never appears without the measurements it was placed between.

## What the tool refuses to guess

Every derived number carries `measured`, `derived` or `assumed`, and the report prints the ones that
are not the first. Three refusals are worth naming, because each corresponds to a mistake this
session made before the tool existed.

**A label is not a size.** "Official FP8 release" and "smaller than native" are different claims.
DeepSeek-V4-Flash's `w8a8` directory is **1.835× larger** than the native checkpoint — 272.74 GiB
against 148.65 GiB, at an identical 69,187 tensors, because it quantizes activations too and keeps
the activation scales beside the weights. So the precision ladder is built by measuring sibling
artifacts header by header, never by reading a directory name for a number. A name is only how a
candidate is *found*; what *keeps* it is a header comparison, and it has to be a strong one.

The architecture key alone is not enough — Bonsai's GGUF and Qwen3.8-27B both declare `qwen35`, and
accepting that pair would file a ternary Bonsai's 5.53 GiB under Qwen3.8-27B's name. So two artifacts
are accepted as the same model only when their tensor-name sets overlap by at least half, after two
normalisations: the quantization-side suffixes come off (`weight` vs `weight_packed`,
`weight_scale_inv` vs `weight_scale`), and so does every all-digit path segment. The second is what
carries the biggest real pair on this box — `Qwen3.8-Flash-Next` keeps each layer's experts *stacked*
as one `...mlp.experts.gate_up_proj` where its own FP8 export writes them out one at a time as
`...mlp.experts.0.gate_proj.weight`, and the pair scores **0.020** as written against **0.970**
normalised. Every other pair that shares an architecture key is unchanged or falls; the next highest
is 0.560, between two 27B training runs of one family that no header can tell apart.

The search runs in both directions, which it did not at first: `...-FP8` implies the base release
beside it, and that directory carries no suffix to be found by and no GGUF to hold, so a
one-directional search finds the NVFP4 build and never the checkpoint both were quantized from.

**A file's bytes are not the card's bytes.** `sm_75` has no FP8 or FP4 tensor core, so a conversion
happens — and which conversion is a property of the runtime, not of the format. This tree contains
both answers: MiMo's MXFP4 routed experts are consumed **verbatim** by the fp4 MoE kernel and never
expanded (`relicllm/models/mimo_v2/loader.py:17`), while MiMo's FP8 dense tiles are dequantized at
load (`relicllm/models/mimo_v2/loader.py:414`) and DeepSeek's FP4 experts are requantized to INT8,
which *doubles* them (`relicllm/models/deepseek_v4/loader.py:141`). MiMo's 161.05 GiB of files
therefore cost 164.64 GiB of card. Expansion rules are a small table keyed by
`(architecture, role, dtype)` with a `file:line` on every entry, and a combination that is not in it
gets a factor of 1.0 **and a note saying the file's bytes may be too small**, because the direction
of the error is not knowable in general.

**What must be resident is not the file total, and it is not a two-way split either.** DeepSeek-V4.1
is 475.24 GiB of files, of which 17.27 GiB must be resident, 275.67 GiB are routed experts a bank can
stage, and **188.83 GiB are two Engram n-gram tables** — one `layers.N.engram.embed.weight` of
`(384006168, 256)` per table, read by `CheckpointEngramTable`
(`relicllm/models/deepseek_v4_1/loader.py:502`). Those tables are neither weights nor experts: they
are a *gather*, and `HostNGramTable` keeps Qwen3.8-Flash-Next's equivalent 95.37 GiB in host RAM for
the same reason. A tool that folded tables into "dense" would call V4.1 impossible on arithmetic that
is 188 GiB wrong.

## Running it over the roster

Eight of the nine models, as this box sees them, with its 1007 GiB of host memory stated — the
number is not optional in the reading, because four of these rows are host-bank placements and at
64 GiB of stated host three of them would be `IMPOSSIBLE` instead:

```text
CANDIDATE   Xing4.0-29B-A4B-GGUF              4.32 GiB resident off a 14.40 GiB bank, 1 card
IMPOSSIBLE  UD-Q2_K_XL (GLM-5.2 Q2_K)         kv-bound at the runtime's 56.9x allocation
CANDIDATE   DeepSeek-V4.1-Flash               17.27 GiB resident, 464.50 GiB bank, 1 card
CANDIDATE   Qwen3.8-Flash-Next                10.22 GiB resident, 325.06 GiB bank, 1 card
CANDIDATE   Qwen3.8-27B-FP8                   14.37 GiB per rank, no bank, 2 cards
CANDIDATE   MiMo-V2.6-Flash-RL                14.83 GiB on card, 149.81 GiB bank, 1 card
CANDIDATE   Ternary-Bonsai-2-27B (PTQ1_0)     5.53 GiB resident, 1 card
CANDIDATE   DeepSeek-V4-Flash-0731            14.62 GiB resident, 284.62 GiB bank, 1 card
```

`DeepSeek-V4.1-Flash` shows the conversion rules working: 10.74 GiB of weights sit on disk, and
17.27 GiB has to be resident, because its FP8 attention and shared-expert tiles are dequantized to
bf16 before the GEMM on this hardware and there is no kernel here that consumes them in place.

Every `CANDIDATE` is `CANDIDATE` for the same reason — nothing has been measured *through this
tool's own threshold set* — and that is the honest state. What the tool adds before any measurement
is the one row that is not a candidate, the card counts, the bank sizes, and the work items:

- **Ternary-Bonsai** carries `ptq1_0`, a ternary pack this fork addresses but does not interpret;
  `read_tensor` refuses it by name rather than silently upcasting to F16, which would cost ten times
  the memory and look like it had worked. A kernel in relic-core is the whole of the work.
- **GLM-5.2's Q2_K** carries `q3_k` and `q8_0`, which have no raw-block kernel; the torch path
  dequantizes them, so that is a speed item rather than a feasibility one.
- **Xing4**'s only block format is `iq4_nl`, which has a kernel, and its adapter is declared — it
  needs nothing. Its `xing4` adapter is found because `RUNTIMES` declares the checkpoint's own
  `model_type` spelling, which is not the same string as the canonical architecture key; asking with
  the key alone reports no adapter for it.
- **Xing4 fits only because the bank exists.** With the 14.40 GiB expert bank offloaded it has
  4.32 GiB resident and reaches 334,988 tokens at batch 1, 32,768 at batch 8 — 46,080 B/token of KV
  is the whole model's, and the 40 trunk layers the GGUF ships (41 blocks, one of them a draft layer
  the HF config does not count) are all of it. State a host smaller than that bank and the excess
  comes back onto the card whole: 18.34 GiB resident plus its 0.35 GiB of 8k cache against 18.70 GiB
  of budget is `IMPOSSIBLE` on one card, because the runtime has no tensor-parallel path to divide it
  across. The flip is between 0.41 and 0.42 GiB of stated host memory. An *unstated* host is neither
  of those answers: it is read as "nobody said", the bank is assumed to fit, and the report carries
  the assumption in its `unknown` list rather than in its verdict.

## What is deliberately not here

- **Cost.** Not dollars, not GPU-hours, not engineering days. The one disqualifier is "does not fit".
- **Gate 2's harness.** The tool prints a command built from `tests/bench_serving.py`'s real flags and
  `relicllm.cli`'s `serve` subcommand; it does not build a new one.
- **Multi-node and Ascend.** No runtime declares `ascend` — `RUNTIMES`' devices are `cuda` and `cpu`
  only — so the tool will report on this box's declaration, not a 910A's. The arithmetic itself is
  platform-neutral: `relicllm/runtime/device.py` declares `ascend`/`npu`/`hccl` in full and
  `probe_accelerator` returns the Ascend answer, so a 910A column is a *card count and card memory
  this file was not given*, not a platform the tree cannot execute. What is missing is a runtime
  declaration and an operator provider, both of which [Joining a machine, and joining a
  family](../architecture/joining_a_machine_or_a_family.md) sets out.
- **Fixing the GLM-5.2 allocation.** Triage reports it. The fix belongs in `glm_dsa/architecture.py`
  and is the cheapest single win in the roster: it would turn GLM's per-rank KV at 32k from 156.0 GiB
  into 2.74 GiB.
