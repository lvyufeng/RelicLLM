# Original-byte Ascend910B tile consumer

This is an **explicit numerical experiment**, not a default provider. It consumes the original
lossless 16-row E2M1/E8M0 byte bank, decodes each small tile to the exact BF16 value represented by
`INT8 x 2` with the original half-scale, and uses BF16 Cube. The native K accumulation differs from
the exact window-13 operator; production defaults and serving guards are not changed.

The tested configuration is Ascend910B3, aarch64, CANN 9.2.0, Torch 2.10.0 and torch_npu 2.10.0.post7.
The AIV producer and AIC consumer are paired one-to-one. Two bounded GM ring slots per worker
provide the handoff on 910B; no direct A5 UB-to-L1 assumption is made. Every original K32 scale is retained inside a K32/K64/K128/K256 compute tile. K32 and K128 have
prior hardware evidence; K256 is an explicit larger-grain experiment with independent codec,
changing-input replay, real-weight projection and whole-model comparison gates. K64 remains
structurally supported rather than an additional validated hardware result. No regrouping averages
or replaces the original per-row K32 scale grid.

## Separate private build

Build only in a writable private directory; do not modify the installed CANN or checkpoints:

```bash
cmake -S relic_core/csrc/ascend/online_tile_cube -B /private/build/online-tile-cube \
  -DASCEND_CANN_PACKAGE_PATH=/usr/local/Ascend/cann-9.2.0 \
  -DSOC_VERSION=Ascend910B3 -DRUN_MODE=npu -DRELIC_TILE_K_BLOCKS=4 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build /private/build/online-tile-cube --parallel 2
```

The CMake default remains one K32 block, with activation preloading disabled. To test K256,
pass `-DRELIC_TILE_K_BLOCKS=8` in a fresh private build root; never rebuild an accepted library
in place. `-DRELIC_TILE_PRELOAD_A=1` is a separate immutable-activation L1 experiment. Its
optional exported query preserves old-library compatibility. Admission uses 512 KiB L1 capacity
and the complete M16-padded activation footprint, including the weight double buffer; oversized
geometry retains the original per-tile A load. No global expanded activation buffer is introduced.
The real 5120/2304 projection probe found no meaningful L1-preload improvement, so it is not
selected for the K256 model trial. All GM-ring, cache-bypass and MTE1/Vector reuse fences remain.

The resulting library is `lib/librelic_online_tile_cube.so` under that build root. Host and device
compile the same configured source entry, so the exported scratch-geometry query cannot silently
report K32 when the kernel is actually K128. Compute the library's SHA256 before explicitly loading
it with `relic_core.kernels.ascendc_online_tile_cube.AscendCTileCubeConsumer`.

The callable takes original decoded BF16 activation `[1,K]` or `[slots,K]`, a packed bank
`[owned,N/16,K/32,16,16]`, original scale bank `[owned,N/16,K/32,16]` and device int32 selected IDs.
Only complete physical output tiles are supported; IDs use the cyclic `local = global // world`
deal. The owning router must validate rank membership and duplicates. `-1` is an empty slot; other
invalid source indices poison output rather than reading outside the bank.

`certified_bf16_operands=True` is a **load-time obligation**: certify that every original E8M0
scale is in 2..252. It is not inferred from hot-loop D2H reads. Invalid scale bytes also poison the
local operand. Original FP8 activation quantization belongs before this projection, not inside a
new dynamic-INT8 quantizer.

Use caller-owned `scratch` and `accumulation` for graph capture. Scratch is BF16
`[block_dim,2,512*group_k]`, accumulation is FP32 `[slots,N]`, and inputs/scratch must not alias.
Both tensors and their backing storage must outlive capture/replay. Sharing scratch requires one
serialized NPU stream; the RelicLLM experimental tile-routing binding enforces that ownership.

At 24 workers, K128 uses a **196,608-byte expanded tile ring**, not zero global scratch. The
complete 5120/2304/topk6 MoE workspace uses **430,188 bytes**, excluding the physical graph pool.
K256 doubles only that ring to **393,216 bytes** and uses **626,796 bytes** of shared MoE scratch;
no whole-matrix INT8/BF16 weight bank is allocated. The consumer returns the original per-projection
BF16 narrowing boundary. The MoE composition
retains original FP8 encode/decode, original FP32 SiLU and ascending-global-expert FP32 combine.

## Measured result and scope

The validated K128 library had SHA256
`0cd9111e373859c5b8d4ec09089a952467f26b6c88a3fb387842b11c0725b287` on the tested private build.
Rebuilding may change the checksum; a checksum alone is not numerical evidence.

- 128,512 original byte/scale BF16 fixture values: zero differences.
- 128 dynamic projection graph replays with 0/1/3/6 owned experts: zero fixture differences.
- 32 whole-MoE graph replays: zero fixture FP32 output differences and no routing D2H.
- Six-active-slot projection: 2.754 ms for N2304/K5120, 2.826 ms for N5120/K2304, 16 synchronized
  replays each. These are not token-throughput figures.
- Original 40-layer TP8/EP8 model, complete CPU Engram, 16 synchronized decode forwards after four
  excluded warmups: **4.777145 TPS**. This improves on the earlier 3.821369 TPS whole-matrix stage
  but loses to the earlier 5.723332 TPS native baseline. They are separate recorded runs, not a
  same-hour causal ablation. The 50 TPS objective is not reached.

## Later K256 experiment

The fresh matched K128/K256 model comparison keeps original host-resident tables, the explicit NPU
Engram path and fused HC folds equal. Sixteen synchronized batch-one autoregressive forwards at
context 24..40 measured **5.073273 TPS versus 5.753094 TPS** (1.134000x). All eight ranks agreed on
the same token stream and both teacher/final FP32 logit tensors matched bit for bit within the
pair. This is scoped native-consumer parity, not equivalence to the exact window-13 production
operator. The faster native on-chip-antiquantization control remained **6.149515 TPS**; therefore
K256 is not a recommendation to replace that control. The 50 TPS target is still not achieved.

Independent K256/K256-with-preload probes check all 256 original packed byte values, certified
E8M0 bytes 2..252 and every position in K256 (1,028,096 BF16 comparisons per variant), as well as
changing activation/selected-ID replay, empty-slot overwrites, invalid-source poisoning and an
oversized-L1 fallback geometry. No meaningful activation-preload speedup was measured. Libraries
and source hashes are pinned in the runtime's separate experiment evidence; rebuilding one is
not a substitute for its device acceptance checks.

The original checkpoint, original quantization grids, exact default and serving guards remain
unchanged. This implementation is not recommended as a replacement for the faster native baseline.
All upstream source inspection is pinned to vllm-ascend
`a38556f69529595e70598d0d6203009f5ac3293e`; the kernel itself is independently written, not copied.
