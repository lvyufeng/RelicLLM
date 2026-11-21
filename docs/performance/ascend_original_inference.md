# Original-checkpoint generation on Ascend 910B

> Publication note: [the evidence publication policy](ascend_decode_optimization.md#published-evidence-and-measurement-identities)
> explains neutral paths/host labels and the separate original/published hash map. Publication cleanup
> does not rerun the recorded measurements.

This continues the [online unpack](ascend_online_unpack.md) and
[original-format expert](ascend_original_experts.md) work with a real **40-layer
DeepSeek-V4.1-Flash TP8 prefill and autoregressive decode**. Engram is enabled and
its **complete computation is on the host**, not just its large embedding table.
The released checkpoint remains read-only; there is no W8A8/W4A8 checkpoint
conversion and no native FP4/FP8 arithmetic claim.

## Verified delivered-source generation

The final delivered-source attempt on 2026-11-17 was
`inference-20261117T090245Z`, after withdrawing the unbound FMA experiment, on a
leased **8 × Ascend 910B3** host. The first successful attempt was
`inference-20261117T075737Z`:

```text
Prompt:       The capital of France is
Continuation:  Paris.
Token IDs:    [11111, 16, 1]
Stop:         EOS (1)
```

The untemplated prompt tokenizes to `[671, 6102, 294, 8760, 344]` (five tokens).
All eight ranks loaded the whole backbone and completed generation; rank 0
recorded the actual **3 × 129280** logits. A separate artifact verifier checked
that **every logit is finite** and that all three recorded tokens equal the
corresponding greedy argmax. This is actual model execution, not an expert-only
probe or a synthetic kernel benchmark. It is **not official-reference token or
logit parity**, a quality evaluation, a service acceptance test or a long-context
claim.

The final CLI measured **57.9 s** for its generation call, excluding model load;
rank 0's loader measured **10.3 s** with warm host page cache. Its post-prefill
loop measured **24.3 s** and printed `8091 ms/token` by dividing by three
generated tokens. There are only **two additional backbone decode forwards** in
this EOS-terminated sample: its first token comes from prefill and EOS needs no
further forward. Thus that printed quotient is **not per-forward latency or
stable decode TPS**.

The first successful generation call was **107.0 s**, and two intermediate
stable-source repeats were **58.3 s** each. This is a short-run warmness/history
record, **not an optimization speedup benchmark**. JIT/cache warmness, host
dispatch, expert staging, HCCL rendezvous and host Engram are included; none has
been isolated by these short runs. First-versus-final **3 × 129280 FP32 logits
are bit-identical**, maximum absolute difference **0.0**. Final-source hashes
were separately checked after generation. These repeat checks do not establish
official reference parity or invariance across prompts/hardware.

## Placement and unchanged format

- Dense backbone, attention/KV caches and routed-expert execution run on the eight
  NPUs, one tensor-parallel process per card.
- **Engram tables, hash state, hash IDs, `wkv`, gate weights and the entire
  gather/projection/normalization/gate computation are on CPU.** An NPU block
  constructs only its Engram submodule on CPU. The current small
  `[batch, tokens, hc_mult, dim]` activation, selected IDs and optional mask cross
  to CPU at its call boundary; the already narrowed BF16 result returns to the
  caller's NPU. The table is not copied to NPU or replicated into a new resident
  host bank per rank. `--resident-engram` is explicitly refused for an NPU
  backbone before bank/table allocation, preventing a silent per-rank 189.1 GiB
  host copy.
- Original FP8 weights retain source E4M3 code bytes and E8M0 grids for the core
  block GEMM. BF16 loading is an online expansion, not a newly quantized weight.
- Original FP4 expert bytes widen exactly to **INT8 = 2 × E2M1** with the original
  **half-scale** grid. The block GEMM recovers values/scales as a matched pair,
  keeps original FP8 activations and the accepted reduction/narrowing order.
- Routed experts use cyclic ID ownership and one process-wide **512 MiB** byte
  budget, not a cache budget per layer. CUDA pool/hot/batched paths are disabled
  for this NPU provider; no resident-checkpoint allocation was requested.
- Decode graphs are off. The serving capability gate is unchanged; running the
  model CLI does not grant an Ascend OpenAI-serving capability.

## Failures actually encountered and repaired

1. **Tokenizer dependency absent.** The retained CPython 3.11 kernel environment
   had no Transformers. Installation was confined to a new task-private package
   directory, without replacing Torch/NPU. A stalled large wheel download was
   interrupted only after matching the exact private-target pip process with
   UID/start-time/pidfd ownership checks. Installed pure-Python or declared
   compatible ABI3 artifacts were reused read-only from the earlier environment;
   CPython 3.12-specific extensions and its `site-packages` path were not copied
   into or appended to the 3.11 interpreter. Small native dependencies used 3.11
   wheels. The real checkpoint-local tokenizer probe then passed.
2. **Large original FP8 matrix exceeded Ascend's launch-grid limit.** The second
   actual TP8 attempt reached loading and failed with `grid should be less than
   65536`. All three unpack kernels now stride their remaining tiles through a
   grid bounded to **65535** programs. Pointer indexing still includes padded
   masked lanes and widens before multiplication. No environment flag was used
   to disable this compiler guard.
3. **FP4 activation kernel compilation/cache/sign issues.** The first real NPU
   suite identified non-constexpr globals; the next exposed a cached floating
   NaN comparing unequal to itself; finite inplace comparisons then exposed
   negative zero lost by arithmetic. Constants are explicit constexpr values,
   NaNs are constructed from integer bits, and final zero signs are restored
   by bit construction without turning a non-finite block into zero. No compiler
   check or numerical assertion was disabled.
4. **Decode selected a single RoPE row.** The third actual run completed prefill
   but its first compressed-indexer decode received `[d/2, 2]` rather than
   `[1, d/2, 2]`. The real-pair adapter now accepts that row form exactly when
   the sequence contains one token. Multi-position mismatches remain errors.

The original FP4 activation source selects E8M0 scales with the **bit ceiling**
of FP32 `amax * fl32(1/6)`, after `6 * 2^-126` flooring; it does not use the
legacy port's `round(log2(...))` branch. Compressed-KV E4M3 scales use the floor
`6 * 2^-9`, FP32 **division** by 6 and E4M3 RNE before normalization. For ratios
above E4M3's maximum, the new API explicitly models a **SATFINITE-style** scale
cast. The released TileLang source does not state overflow policy, so saturation
outside the finite range is an assumption, not newly established hardware
parity. Non-finite FP4 blocks likewise have an explicit extension policy.

## Real-pair RoPE correctness boundary

The NPU table is formed from the same **host `torch.polar` complex64 table** as
the existing path, viewed as real FP32 `(cos, sin)` pairs before device transfer.
There is no angle recomputation or NPU complex operation. Pairwise FP32 products
and sums rotate the activation and narrow back once into its storage; forward,
inverse, non-contiguous rotated tails and the selected-row decode shape are
covered by tests.

**Do not infer universal BF16 bit parity from seeded small cases.** Complex
multiply contraction differs between CPU builds. A large local hold-out corpus
found rare one-BF16-ulp differences between unfused real products and its complex
multiply; a separate ARM-host probe found a different per-component contraction
again. An exploratory explicit-FMA operator was therefore **not bound to this
successful model path** on an unproven cross-architecture parity claim. Its four
unbound source/test files were withdrawn from the delivered core checkout after
archiving both revisions and the failed comparisons. The
successful generation establishes finite autoregressive execution, not official
CUDA/TileLang bit equivalence. CPU/CUDA's existing complex path is unchanged.

## Scoped acceptance

| Check | Result | Scope |
| --- | --- | --- |
| Final source-derived FP4 activation suite | **75 passed, zero skips** | Portable/CPU cases plus actual NPU bytes, inplace BF16/FP32, E8M0/E4M3, block 16/32, signed zero, non-finite policy and stream/device checks |
| Runtime inference bindings | **60 passed, zero skips** | Actual NPU host Engram boundary, early resident-Engram refusal, window-KV FP8, real-pair RoPE and loader fixtures |
| Complete unpack grid/bit regression | **106 passed, zero skips** | Actual 65536 × 1024 matrix output bits/guards, forced-grid FP4/half-scale tails, independent byte contracts and portable launch/math checks |
| Checkpoint-backed generation | **3 greedy tokens, EOS, all logits finite** | Whole 40-layer TP8 prefill and two subsequent decode forwards |

These are scoped suites, **not the whole repository test suite**. A portable test
inside a hardware-requested invocation is not a hardware pass of its own.
Output guards, bytes and dtype/sign classifications were checked, not merely
compiler acceptance. No claim is made for graphs, a resident expert bank,
long-context inference, speculative decoding, serving, or a throughput
improvement versus the 2080 Ti runtime.

## Exact reproduction scope

The accepted stack is **torch 2.10.0+cpu / torch_npu 2.10.0.post7 / CANN 9.2.0 /
driver 26.1.0**, private **CPython 3.11.18 / Triton Ascend 3.2.0**. The previously
reviewed private Triton host-driver compatibility patch is retained; the shared
CANN installation, Jenkins agent and other users' resources were not modified.
This is not a stock-wheel or CPython-cross-ABI claim.

From the final runtime checkout under a fresh fleet lease and the compatible
private environment:

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DEEPSEEK_V41_RESIDENT_EXPERTS=0 DEEPSEEK_V41_PIN_RESIDENT_EXPERTS=0
export POCKETLLM_V41_MMA_EMULATE=1
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=8 \
  -m relicllm.cli.generate_v41 \
  --checkpoint <checkpoint> \
  --expert-device npu:0 --expert-world 8 \
  --expert-pool-rows 0 --expert-hot-rows 0 --no-expert-batched --expert-deal id \
  --no-decode-graphs --threads 4 --max-seq-len 128 --max-new-tokens 4 \
  --temperature 0 --prompt "The capital of France is" \
  --dump-logits /path/to/fresh-attempt/generated-logits.pt
```

Every attempt used a bounded owner-token-aware runner, separate logs and cleanup
receipts. Existing checkpoint file identities/timestamps are compared with the
previous fully verified original download; this final check is not represented
as a new 510 GB checksum pass. Retained paths do not authorize another NPU run.

## Retained acceptance receipts

- [Verified generated tokens and finite greedy logits](evidence/ascend-original-inference-20261117/inference-20261117T090245Z/validated-generation.json),
  [actual output](evidence/ascend-original-inference-20261117/inference-20261117T090245Z/original-inference.stdout.log),
  [per-rank load and generation times](evidence/ascend-original-inference-20261117/inference-20261117T090245Z/original-inference.stderr.log)
  and the [full logits tensor artifact](evidence/ascend-original-inference-20261117/inference-20261117T090245Z/generated-logits.pt).
- [Invocation, source hashes and unchanged-format preconditions](evidence/ascend-original-inference-20261117/inference-20261117T090245Z/preconditions.json)
  and [complete owned-child cleanup receipt](evidence/ascend-original-inference-20261117/inference-20261117T090245Z/summary.json).
- [Final FP4 activation test log](evidence/ascend-original-inference-20261117/fp4-activation-20261117T075023Z/fp4-activation.stdout.log)
  and [machine-readable test cases](evidence/ascend-original-inference-20261117/fp4-activation-20261117T075023Z/pytest.xml).
- [Actual NPU host-Engram/activation/rotary binding tests](evidence/ascend-original-inference-20261117/inference-ops-20261117T090232Z/inference-ops.stdout.log),
  [portable bounded-grid checks](evidence/ascend-original-inference-20261117/inference-grid-launch-tests.log)
  and [scoped CPU runtime regressions](evidence/ascend-original-inference-20261117/inference-final-runtime-cpu.log).
- [Independent ARM complex-contraction probe](evidence/ascend-original-inference-20261117/rotary-host-component-contractions.json)
  and the [unaccepted exploratory FMA comparison log](evidence/ascend-original-inference-20261117/rotary-fma-20261117T080755Z/rotary-fma.stdout.log).
  The latter is retained as failed evidence, **not** a production or parity pass.
- [First-versus-delivered repeated-logit and 36-source-hash verification](evidence/ascend-original-inference-20261117/inference-20261117T090245Z/repeat-comparison.json),
  [complete large-grid/bit/tail acceptance](evidence/ascend-original-inference-20261117/unpack-grid-20261117T090215Z/unpack-grid.stdout.log),
  [final host/checkpoint/Jenkins integrity check](evidence/ascend-original-inference-20261117/final-inference-host-check.json),
  [all eight idle-device process snapshots](evidence/ascend-original-inference-20261117/final-inference-npu-smi.log)
  and [confirmed fleet lease release](evidence/ascend-original-inference-20261117/lease-release.json).

The final integrity check found no live task-owned/private Python children across
**19 attempt receipts**, all **90** original checkpoint file identities/stat
timestamps unchanged, the same Jenkins agent and shared CANN environment, and no
running NPU process on any of the eight cards. The fleet lease was then released.
