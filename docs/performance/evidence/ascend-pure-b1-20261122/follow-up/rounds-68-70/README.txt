# Evidence: rounds 68 to 70

This section records what happened after the indexer-selection fusion was registered at **14.287598 tok/s
/ 69.9922 ms a token** — three negative results obtained by measurement, and one open lead written down
with its hazard so it is not approached carelessly later.

The registered result did **not** change: none of the work in these rounds was adopted. Every sha256 is
in `summary.json` under `manifest`.

## Result 1 — the o-projection transpose cannot be removed bit-exactly

`aclnnBatchMatMul_TransposeAiCore_Transpose` costs **1,171.6 us a forward** across 48 calls, dominated by a
`1024,4096 → 4096,1024` shape about once a layer. It comes from `attention.py:1688-1689`, which lays the
o-projection weight out as `[groups, o_lora_rank, d]` and then einsums over `d`, so the framework
materialises a transpose of a **constant** weight before every matmul.

Transposing a constant once per module looked free of consequence — a transpose changes no value — and that
reasoning was wrong. Tested on the device before touching the lane:

| variant | bit-identical to current code | eager cost |
| --- | --- | --- |
| contiguous pre-transpose, `gdr` einsum | 3 of 6, then 5 of 8 — **no** | 212.1 us, *slower* |
| transposed view handed to matmul | **0 of 8** | 99.6 us, 40% cheaper |

Differences are 1–2 of 2048 elements, max abs up to **0.5** on bf16. The reduction order on this build is
therefore **not layout-invariant**: the faster variant changes results in every trial, and the variant that
sometimes agrees is slower than what runs today.

So the 1,171.6 us cannot be removed bit-exactly and **the change was not adopted**. The faster variant is
recorded as a *new deviation class* — bf16 reduction-order deviation in the o-projection — needing explicit
user approval. The existing row-normalisation and narrow-head approvals do **not** cover it: different op,
different class.

## Result 2 — the expert projection is M=1 tile padding, and it is closed

The scale-LUT expert projection is **15.4%** of the step and had only ever been bounded by reasoning. The
bandwidth driver that answers it had been sitting in the evidence bundle since round 48, titled
**"written, not run"**. It was run for the first time, after verifying both pinned identities: the accepted
v2-n64 library (`7dce3e3d…`) and CANN (`bbbda1c0…`). The first library found on the box was the v1-n64
build (`c896289f`) and the hash guard correctly rejected it.

| variant | median | achieved |
| --- | --- | --- |
| 1 group, runtime bf16 | 366.3 us | 8.6 GB/s |
| 6 groups, runtime bf16 | 474.9 us | 39.8 GB/s |
| 6 groups, kernel fp32 | 474.0 us | 39.7 GB/s |

Five more experts cost only about 110 us, so the call is dominated by per-call cost rather than per-expert
bytes. Two cautions were needed before reading that as headroom:

* **the reference matters.** The harness reports a same-volume read-only `sum` at 3,481.8 us (**5.4 GB/s**)
  next to a same-volume `copy_` at 83.9 us (**448.3 GB/s**). The first is a reduction over an awkward shape,
  not a bandwidth measurement; taking it as the floor would understate the card by two orders of magnitude.
  Only the copy number is a practical reference.
* **the standalone numbers are host-bound.** The same kernel takes 474.9 us standalone and **101 us a call
  in the model profile** — a 4.7× gap, which is host launch and synchronisation the graph does not pay. The
  implied 344.5 us fixed cost is therefore *not* kernel-internal.

Sized in-graph instead: 18.8 MB in 101 us is **186.9 GB/s against a 448.3 GB/s practical reference — 2.40×
off**, not the order of magnitude the raw fixed-cost line suggests. A generous best case would recover about
7 ms of the 12.1 ms, and that best case is unreachable: the kernel runs **M=1**, one activation row
broadcast across expert groups, while the cube unit is built for M=16 tiles, so most of every loaded tile is
padding. That is a property of single-request decode, not of this kernel, and it is the same reason a larger
acceptance length is so much faster.

This closes "swap the LUT kernel" as a route to 50 tok/s. A better M=1 kernel may still find 1–2 ms.

## Result 3 — the lane is device-bound

The union of device task intervals is **287,045 us of a 314,467 us span (91.3%)**. Host-side time is
overlapped, so host restructuring has no headroom: every remaining gain must be less or faster **device**
work. This is why the `aten::copy_` host cost (95 calls, 4,277.7 us of host time, 63.1 us of device time a
forward) is not a target.

## Open lead — redundant FP8 activation quantisation

`fast_fused_fp8_activation_decode_kernel`: **322 calls and 3,390.8 us a forward**, at 10.53 us a call. Each
call quantises roughly 10 KB, so the cost is launch overhead rather than work.

`modules.py:203-204` calls `quantized_linear(x, w1)` and `quantized_linear(x, w3)` on the **same `x`**, so
one quantisation per MoE call is redundant; two MoE FFNs a layer over 40 layers is about **80 redundant
calls, roughly 840 us = 1.2%**. Reusing the first result is bit-exact by construction — quantising the same
unchanged tensor twice yields the same bytes.

**The hazard that stops a quick fix**: the runtime reuses activation buffers (`_reuse` does
`slot.copy_(value)` in `decode_pos.py`), so a cache keyed on tensor identity would go stale when a buffer is
overwritten. A correct fix dedupes **at the call site**, not by memoising inside the quantiser.

It was not attempted this round because `quantized_linear` is the soft/reference path while the device path
uses the fused kernel, and the profiler did not capture input shapes for this kernel — pinning the 322
calls' provenance needs a targeted capture first.

## What this leaves

Within the objective's constraints, every item with magnitude now has a measured verdict rather than an
opinion: communication 26.2% (closed twice), the LUT projection 15.4% (closed above), the o-projection
transpose 1.6% (closed), host-side work (closed by the device-bound measurement). The remaining identified
headroom is the ~1.2% quantisation lead and a possible 1–2 ms from a better M=1 LUT kernel.

The ceiling is unchanged and worth restating: a tuned community stack on the same hardware measures
**27.13 ms a step at an acceptance length of one**, so about 36.9 tok/s, and reaches 136.9 tok/s only by
emitting 4.10 tokens a step. This lane is at 14.2876 tok/s, roughly 39% of that ceiling.