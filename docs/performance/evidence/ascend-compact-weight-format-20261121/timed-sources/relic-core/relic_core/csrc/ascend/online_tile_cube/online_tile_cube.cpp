// Original-byte online unpack with explicit AIV -> bounded GM ring -> L1 -> AIC ownership.
// This is an independent, opt-in Ascend910B prototype; no upstream kernel is copied.
#include "kernel_operator.h"
#include <cstdint>

using namespace AscendC;

namespace {
constexpr uint32_t M = 16;
#ifndef RELIC_TILE_OUTPUT_ROWS
#define RELIC_TILE_OUTPUT_ROWS 16
#endif
constexpr uint32_t N = RELIC_TILE_OUTPUT_ROWS;
static_assert(N == 16 || N == 32, "only bounded complete output tiles are supported");
constexpr uint32_t SOURCE_N = 16;
constexpr uint32_t N_TILES = N / SOURCE_N;
#ifndef RELIC_TILE_K_BLOCKS
#define RELIC_TILE_K_BLOCKS 1
#endif
constexpr uint32_t GROUP_K = RELIC_TILE_K_BLOCKS;
static_assert(GROUP_K == 1 || GROUP_K == 2 || GROUP_K == 4 || GROUP_K == 8,
              "retain an integral original K32 scale grid");
#ifndef RELIC_TILE_PRELOAD_A
#define RELIC_TILE_PRELOAD_A 0
#endif
constexpr uint32_t PRELOAD_A = RELIC_TILE_PRELOAD_A;
static_assert(PRELOAD_A <= 1, "activation reuse is an explicit build-time experiment");
#ifndef RELIC_TILE_DIRECT_BITS
#define RELIC_TILE_DIRECT_BITS 0
#endif
constexpr uint32_t DIRECT_BITS = RELIC_TILE_DIRECT_BITS;
static_assert(DIRECT_BITS <= 1, "direct BF16 bit decoding is an explicit build-time experiment");
#ifndef RELIC_TILE_PREPARED_TABLES
#define RELIC_TILE_PREPARED_TABLES 0
#endif
constexpr uint32_t PREPARED_TABLES = RELIC_TILE_PREPARED_TABLES;
static_assert(PREPARED_TABLES <= 1, "immutable decode tables require an explicit build-time opt-in");
#ifndef RELIC_TILE_CUBE_PACKED
#define RELIC_TILE_CUBE_PACKED 0
#endif
constexpr uint32_t CUBE_PACKED = RELIC_TILE_CUBE_PACKED;
static_assert(CUBE_PACKED <= 1, "offline compact layout is an explicit build-time experiment");
static_assert(CUBE_PACKED == 0 || (N == 32 && GROUP_K == 4 && DIRECT_BITS == 0 && PREPARED_TABLES == 1),
              "offline Cube-order FP4 requires the checked prepared N32/K128 floating decoder");
#ifndef RELIC_TILE_UNSIGNED_BYTE_CAST
#define RELIC_TILE_UNSIGNED_BYTE_CAST 0
#endif
constexpr uint32_t UNSIGNED_BYTE_CAST = RELIC_TILE_UNSIGNED_BYTE_CAST;
static_assert(UNSIGNED_BYTE_CAST <= 1, "unsigned byte indices are an explicit A2 experiment");
static_assert(UNSIGNED_BYTE_CAST == 0 || (PREPARED_TABLES == 1 && DIRECT_BITS == 0),
              "unsigned byte casts require the checked prepared floating decoder");
constexpr uint32_t L1_BYTES = 512 * 1024;
constexpr uint32_t K_BLOCK = 32;
constexpr uint32_t K = K_BLOCK * GROUP_K;
constexpr uint32_t SCALE_COUNT = N * GROUP_K;
constexpr uint32_t BYTES = N * K / 2;
constexpr uint32_t ELEMENTS = N * K;
constexpr uint32_t SOURCE_BYTES = SOURCE_N * K / 2;
constexpr uint32_t SOURCE_SCALES = SOURCE_N * GROUP_K;
constexpr uint32_t TABLE_SCALE = 256;
constexpr uint32_t TABLE_REPEAT = TABLE_SCALE + 256;
constexpr uint32_t TABLE_REPEAT_COUNT = DIRECT_BITS ? BYTES : ELEMENTS;
constexpr uint32_t TABLE_NZ = TABLE_REPEAT + TABLE_REPEAT_COUNT;
constexpr uint32_t TABLE_MASK = TABLE_NZ + BYTES;
constexpr uint32_t TABLE_POISON = TABLE_MASK + 256;
constexpr uint32_t TABLE_WORDS = (CUBE_PACKED ? TABLE_NZ : TABLE_MASK) + (DIRECT_BITS ? 512 : 0);
constexpr uint32_t L1_B_SLOT_BYTES = ELEMENTS * sizeof(bfloat16_t);
constexpr uint32_t L1_A_OFFSET = 2 * L1_B_SLOT_BYTES;
constexpr uint32_t SCALE_PAD = (SCALE_COUNT + 31) / 32 * 32;
constexpr uint32_t DECODED_OFFSET = BYTES + SCALE_PAD;
constexpr uint32_t HALF_OFFSET = DECODED_OFFSET + 2 * ELEMENTS * sizeof(uint16_t);
constexpr uint32_t INDEX_OFFSET = HALF_OFFSET + BYTES * sizeof(half);
constexpr uint32_t LOOKUP_OFFSET = INDEX_OFFSET + BYTES * sizeof(int32_t);
constexpr uint32_t PAIRS_OFFSET = LOOKUP_OFFSET + 256 * sizeof(uint32_t);
constexpr uint32_t VALUES_OFFSET = PAIRS_OFFSET + BYTES * sizeof(uint32_t);
constexpr uint32_t SCALED_OFFSET = VALUES_OFFSET + ELEMENTS * sizeof(float);
constexpr uint32_t NZ_OFFSET = SCALED_OFFSET + ELEMENTS * sizeof(bfloat16_t);
constexpr uint32_t SCALE_HALF_OFFSET = NZ_OFFSET + BYTES * sizeof(uint32_t);
constexpr uint32_t SCALE_INDEX_OFFSET = SCALE_HALF_OFFSET + SCALE_COUNT * sizeof(half);
constexpr uint32_t SCALE_FACTOR_OFFSET = SCALE_INDEX_OFFSET + SCALE_COUNT * sizeof(int32_t);
constexpr uint32_t SCALE_LOOKUP_OFFSET = SCALE_FACTOR_OFFSET + SCALE_COUNT * sizeof(float);
constexpr uint32_t SCALE_REPEAT_OFFSET = SCALE_LOOKUP_OFFSET + 256 * sizeof(uint32_t);
constexpr uint32_t EXPANDED_FACTOR_OFFSET = SCALE_REPEAT_OFFSET + ELEMENTS * sizeof(uint32_t);
constexpr uint32_t BIT_MASK_LOOKUP_OFFSET = EXPANDED_FACTOR_OFFSET + ELEMENTS * sizeof(float);
constexpr uint32_t BIT_POISON_LOOKUP_OFFSET = BIT_MASK_LOOKUP_OFFSET + 256 * sizeof(uint32_t);
static_assert((DIRECT_BITS ? BIT_POISON_LOOKUP_OFFSET + 256 * sizeof(uint32_t)
                          : EXPANDED_FACTOR_OFFSET + ELEMENTS * sizeof(float)) <= 128 * 1024,
              "the complete Vector decode scratch must fit the Ascend910B UB");
constexpr uint16_t READY_FLAG = 8;
constexpr uint16_t FREE_FLAG = 10;

__aicore__ inline uint16_t OriginalFp4Bits(uint8_t code, uint8_t scale)
{
    if (scale < 2 || scale > 252) return 0x7fc0;
    const uint32_t magnitude = code & 7;
    if (magnitude == 0) return 0; // The accepted I8x2 widening folds -0.
    const int32_t delta = magnitude >= 6 ? 2 : (magnitude >= 4 ? 1 : (magnitude >= 2 ? 0 : -1));
    const uint32_t mantissa = (magnitude == 3 || magnitude == 5 || magnitude == 7) ? 0x40 : 0;
    return static_cast<uint16_t>(((code & 8) << 12) | ((scale + delta) << 7) | mantissa);
}

class TileVector {
public:
    __aicore__ inline void Init(GM_ADDR b, GM_ADDR s, GM_ADDR selected, GM_ADDR stage, GM_ADDR out,
                                GM_ADDR tables, uint32_t tiles, uint32_t blocks, uint32_t world, uint32_t n)
    {
        packed_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(b));
        scales_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(s));
        selected_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(selected));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        n_ = n;
        sourceTiles_ = tiles;
        tiles_ = tiles / N_TILES;
        blocks_ = blocks / GROUP_K;
        world_ = world;
        raw_ = LocalTensor<uint8_t>(TPosition::VECCALC, 0, BYTES);
        scale_ = LocalTensor<uint8_t>(TPosition::VECCALC, BYTES, SCALE_PAD);
        decoded_ = LocalTensor<uint16_t>(TPosition::VECCALC, DECODED_OFFSET, 2 * ELEMENTS);
        signedHalf_ = LocalTensor<half>(TPosition::VECCALC, HALF_OFFSET, BYTES);
        indexes_ = LocalTensor<int32_t>(TPosition::VECCALC, INDEX_OFFSET, BYTES);
        lookup_ = LocalTensor<uint32_t>(TPosition::VECCALC, LOOKUP_OFFSET, 256);
        unitPairs_ = LocalTensor<uint32_t>(TPosition::VECCALC, PAIRS_OFFSET, BYTES);
        values_ = LocalTensor<float>(TPosition::VECCALC, VALUES_OFFSET, ELEMENTS);
        scaled_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, SCALED_OFFSET, CUBE_PACKED ? 2 * ELEMENTS : ELEMENTS);
        nzIndexes_ = LocalTensor<uint32_t>(TPosition::VECCALC, NZ_OFFSET, BYTES);
        scaleHalf_ = LocalTensor<half>(TPosition::VECCALC, SCALE_HALF_OFFSET, SCALE_COUNT);
        scaleIndexes_ = LocalTensor<int32_t>(TPosition::VECCALC, SCALE_INDEX_OFFSET, SCALE_COUNT);
        scaleFactors_ = LocalTensor<float>(TPosition::VECCALC, SCALE_FACTOR_OFFSET, SCALE_COUNT);
        scaleLookup_ = LocalTensor<uint32_t>(TPosition::VECCALC, SCALE_LOOKUP_OFFSET, 256);
        repeatScaleIndexes_ = LocalTensor<uint32_t>(TPosition::VECCALC, SCALE_REPEAT_OFFSET, ELEMENTS);
        expandedFactors_ = LocalTensor<float>(TPosition::VECCALC, EXPANDED_FACTOR_OFFSET, ELEMENTS);
        if constexpr (DIRECT_BITS != 0) {
            bitMasks_ = LocalTensor<uint32_t>(TPosition::VECCALC, VALUES_OFFSET, BYTES);
            bitMaskLookup_ = LocalTensor<uint32_t>(TPosition::VECCALC, BIT_MASK_LOOKUP_OFFSET, 256);
            bitPoisonLookup_ = LocalTensor<uint32_t>(TPosition::VECCALC, BIT_POISON_LOOKUP_OFFSET, 256);
            bitPoison_ = LocalTensor<uint32_t>(TPosition::VECCALC, VALUES_OFFSET + BYTES * sizeof(uint32_t), SCALE_COUNT);
            expandedPoison_ = LocalTensor<uint32_t>(TPosition::VECCALC, SCALE_REPEAT_OFFSET + BYTES * sizeof(uint32_t), BYTES);
        }
        bool prepared = false;
        if constexpr (PREPARED_TABLES != 0) {
            if (tables != nullptr) {
                // These small tables contain only immutable code/index metadata,
                // never expanded checkpoint weights or selected expert values.
                tables_.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(tables));
                DataCopy(lookup_, tables_, 256);
                DataCopy(scaleLookup_, tables_[TABLE_SCALE], 256);
                DataCopy(repeatScaleIndexes_, tables_[TABLE_REPEAT], TABLE_REPEAT_COUNT);
                if constexpr (CUBE_PACKED == 0) DataCopy(nzIndexes_, tables_[TABLE_NZ], BYTES);
                if constexpr (DIRECT_BITS != 0) {
                    DataCopy(bitMaskLookup_, tables_[TABLE_MASK], 256);
                    DataCopy(bitPoisonLookup_, tables_[TABLE_POISON], 256);
                }
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                prepared = true;
            }
        }
        if (!prepared) {
        // Retain the legacy ABI and default path when no prepared pointer is
        // supplied. The opt-in launch skips every scalar table-fill loop.
        for (uint32_t index = 0; index < (DIRECT_BITS ? BYTES : ELEMENTS); ++index) {
            const uint32_t scaleIndex = CUBE_PACKED ? (index / (N * K_BLOCK)) * N + (index / 16) % N
                : index / (DIRECT_BITS ? K_BLOCK / 2 : K_BLOCK);
            repeatScaleIndexes_.SetValue(index, scaleIndex * sizeof(uint32_t));
        }
        for (uint32_t index = 0; index < BYTES; ++index) {
            const uint32_t halfTile = index / (N * 8);
            const uint32_t sourceRow = (index % (N * 8)) / 8;
            const uint32_t sourceTile = sourceRow / SOURCE_N;
            const uint32_t row = sourceRow % SOURCE_N;
            const uint32_t pair = index % 8;
            const uint32_t originalBlock = halfTile / 2;
            const uint32_t half = halfTile % 2;
            nzIndexes_.SetValue(index, (sourceTile * GROUP_K * 256 + originalBlock * 256
                                       + row * 16 + half * 8 + pair) * sizeof(uint32_t));
        }
        for (uint32_t index = 0; index < 256; ++index) {
            const uint8_t byte = (index + (UNSIGNED_BYTE_CAST ? 0 : 128)) & 255;
            if constexpr (DIRECT_BITS != 0) {
                const uint16_t offset = static_cast<uint16_t>((static_cast<int32_t>(byte) - 127) * 128);
                scaleLookup_.SetValue(index, byte >= 2 && byte <= 252
                    ? static_cast<uint32_t>(offset) | (static_cast<uint32_t>(offset) << 16) : 0);
                bitPoisonLookup_.SetValue(index, byte >= 2 && byte <= 252 ? 0 : 0x7fc07fc0);
                bitMaskLookup_.SetValue(index, ((byte & 7) ? 0xffffu : 0u) |
                                               ((byte & 0x70) ? 0xffff0000u : 0u));
            } else {
                scaleLookup_.SetValue(index, byte >= 2 && byte <= 252 ? static_cast<uint32_t>(byte) << 23 : 0x7fc00000);
            }
            lookup_.SetValue(index, static_cast<uint32_t>(OriginalFp4Bits(byte & 15, 127)) |
                                    (static_cast<uint32_t>(OriginalFp4Bits(byte >> 4, 127)) << 16));
        }
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
        stage_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(stage));
        stage_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    }

    __aicore__ inline void Process(uint32_t groups, uint32_t owned)
    {
        const uint32_t owner = GetBlockIdx();
        for (uint32_t task = owner; task < groups * tiles_; task += GetBlockNum()) {
            const uint32_t group = task / tiles_;
            const uint32_t tile = task % tiles_;
            const int32_t expert = selected_.GetValue(group);
            if (expert < 0 || static_cast<uint32_t>(expert / world_) >= owned) {
                union { uint32_t bits; float value; } fill;
                fill.bits = expert == -1 ? 0 : 0x7fc00000;
                Duplicate(values_, fill.value, N);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy(output_[static_cast<uint64_t>(group) * n_ + tile * N], values_, N);
                SetFlag<HardEvent::MTE3_V>(0);
                WaitFlag<HardEvent::MTE3_V>(0);
                continue;
            }
            const uint32_t local = expert / world_;
            const uint64_t base = (static_cast<uint64_t>(local) * sourceTiles_ + tile * N_TILES) * blocks_;
            const uint64_t compactBase = (static_cast<uint64_t>(local) * tiles_ + tile) * blocks_;
            for (uint32_t block = 0; block < blocks_; ++block) {
                const uint32_t slot = block & 1;
                if (block >= 2) CrossCoreWaitFlag<0x2>(FREE_FLAG + slot);
                if constexpr (CUBE_PACKED != 0) {
                    // Offline layout already matches Mmad K16-major/N16-minor B.
                    // Scales stay one original byte per N row and K32 block.
                    const uint64_t source = compactBase + block;
                    DataCopy(raw_, packed_[source * BYTES], BYTES);
                    DataCopy(scale_, scales_[source * SCALE_COUNT], SCALE_COUNT);
                } else {
                // Wider output tiles read adjacent original N16 banks without
                // repacking or changing a single checkpoint code/scale byte.
                for (uint32_t sourceTile = 0; sourceTile < N_TILES; ++sourceTile) {
                    const uint64_t source = base + sourceTile * blocks_ + block;
                    DataCopy(raw_[sourceTile * SOURCE_BYTES], packed_[source * SOURCE_BYTES], SOURCE_BYTES);
                    DataCopyPad(scale_[sourceTile * SOURCE_SCALES], scales_[source * SOURCE_SCALES],
                                DataCopyExtParams{1, SOURCE_SCALES, 0, 0, 0},
                                DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
                }
                }
                SetFlag<HardEvent::MTE2_S>(0);
                WaitFlag<HardEvent::MTE2_S>(0);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                if constexpr (UNSIGNED_BYTE_CAST != 0) {
                    Cast(signedHalf_, raw_, RoundMode::CAST_NONE, BYTES);
                } else {
                    Cast(signedHalf_, raw_.ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, BYTES);
                }
                PipeBarrier<PIPE_V>();
                Cast(indexes_, signedHalf_, RoundMode::CAST_RINT, BYTES);
                PipeBarrier<PIPE_V>();
                if constexpr (UNSIGNED_BYTE_CAST == 0) {
                    Adds(indexes_, indexes_, static_cast<int32_t>(128), BYTES);
                    PipeBarrier<PIPE_V>();
                }
                ShiftLeft(indexes_.ReinterpretCast<uint32_t>(), indexes_.ReinterpretCast<uint32_t>(),
                          static_cast<uint32_t>(2), BYTES);
                PipeBarrier<PIPE_V>();
                Gather(unitPairs_, lookup_, indexes_.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), BYTES);
                PipeBarrier<PIPE_V>();
                if constexpr (DIRECT_BITS != 0) {
                    DecodeDirectBits();
                } else {
                Cast(values_, unitPairs_.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, ELEMENTS);
                PipeBarrier<PIPE_V>();
                if constexpr (UNSIGNED_BYTE_CAST != 0) {
                    Cast(scaleHalf_, scale_, RoundMode::CAST_NONE, SCALE_COUNT);
                } else {
                    Cast(scaleHalf_, scale_.ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, SCALE_COUNT);
                }
                PipeBarrier<PIPE_V>();
                Cast(scaleIndexes_, scaleHalf_, RoundMode::CAST_RINT, SCALE_COUNT);
                PipeBarrier<PIPE_V>();
                if constexpr (UNSIGNED_BYTE_CAST == 0) {
                    Adds(scaleIndexes_, scaleIndexes_, static_cast<int32_t>(128), SCALE_COUNT);
                    PipeBarrier<PIPE_V>();
                }
                ShiftLeft(scaleIndexes_.ReinterpretCast<uint32_t>(), scaleIndexes_.ReinterpretCast<uint32_t>(),
                          static_cast<uint32_t>(2), SCALE_COUNT);
                PipeBarrier<PIPE_V>();
                Gather(scaleFactors_.ReinterpretCast<uint32_t>(), scaleLookup_,
                       scaleIndexes_.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), SCALE_COUNT);
                PipeBarrier<PIPE_V>();
                Gather(expandedFactors_, scaleFactors_, repeatScaleIndexes_, static_cast<uint32_t>(0), ELEMENTS);
                PipeBarrier<PIPE_V>();
                Mul(values_, values_, expandedFactors_, ELEMENTS);
                PipeBarrier<PIPE_V>();
                const auto scaledTile = scaled_[CUBE_PACKED ? slot * ELEMENTS : 0];
                Cast(scaledTile, values_, RoundMode::CAST_RINT, ELEMENTS);
                PipeBarrier<PIPE_V>();
                }
                if constexpr (CUBE_PACKED != 0) {
                    SetFlag<HardEvent::V_MTE3>(0);
                    WaitFlag<HardEvent::V_MTE3>(0);
                    DataCopy(stage_[owner * 2 * ELEMENTS + slot * ELEMENTS], scaled_[slot * ELEMENTS], ELEMENTS);
                } else {
                Gather(decoded_[slot * ELEMENTS].ReinterpretCast<uint32_t>(),
                       scaled_.ReinterpretCast<uint32_t>(), nzIndexes_, static_cast<uint32_t>(0), BYTES);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy(stage_[owner * 2 * ELEMENTS + slot * ELEMENTS],
                         decoded_[slot * ELEMENTS].ReinterpretCast<bfloat16_t>(), ELEMENTS);
                }
                CrossCoreSetFlag<0x2, PIPE_MTE3>(READY_FLAG + slot);
                // The compact scaled BF16 uses two UB slots in the unused NZ
                // index space. FREE fences both UB and GM reuse after two blocks.
                // The single raw/scale UB slot cannot be overwritten until
                // the current vector decode has consumed it.
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
            }
            const uint32_t drains = blocks_ < 2 ? blocks_ : 2;
            for (uint32_t i = 0; i < drains; ++i) {
                const uint32_t block = blocks_ - drains + i;
                CrossCoreWaitFlag<0x2>(FREE_FLAG + (block & 1));
            }
        }
    }
private:
    __aicore__ inline void DecodeDirectBits()
    {
        // Metadata casts produce exact byte indices only. Scaled weight
        // values themselves never enter FP16/FP32 arithmetic on this path.
        Gather(bitMasks_, bitMaskLookup_, indexes_.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), BYTES);
        PipeBarrier<PIPE_V>();
        Cast(scaleHalf_, scale_.ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, SCALE_COUNT);
        PipeBarrier<PIPE_V>();
        Cast(scaleIndexes_, scaleHalf_, RoundMode::CAST_RINT, SCALE_COUNT);
        PipeBarrier<PIPE_V>();
        Adds(scaleIndexes_, scaleIndexes_, static_cast<int32_t>(128), SCALE_COUNT);
        PipeBarrier<PIPE_V>();
        ShiftLeft(scaleIndexes_.ReinterpretCast<uint32_t>(), scaleIndexes_.ReinterpretCast<uint32_t>(),
                  static_cast<uint32_t>(2), SCALE_COUNT);
        PipeBarrier<PIPE_V>();
        Gather(scaleFactors_.ReinterpretCast<uint32_t>(), scaleLookup_,
               scaleIndexes_.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), SCALE_COUNT);
        PipeBarrier<PIPE_V>();
        Gather(expandedFactors_.ReinterpretCast<uint32_t>(), scaleFactors_.ReinterpretCast<uint32_t>(),
               repeatScaleIndexes_, static_cast<uint32_t>(0), BYTES);
        PipeBarrier<PIPE_V>();
        // VADD int16 is modulo the lane width. Scaling changes only each
        // original E2M1 value's exponent; its sign and mantissa stay exact.
        Add(scaled_.ReinterpretCast<int16_t>(), unitPairs_.ReinterpretCast<int16_t>(),
            expandedFactors_.ReinterpretCast<int16_t>(), ELEMENTS);
        PipeBarrier<PIPE_V>();
        And(scaled_.ReinterpretCast<uint16_t>(), scaled_.ReinterpretCast<uint16_t>(),
            bitMasks_.ReinterpretCast<uint16_t>(), ELEMENTS);
        PipeBarrier<PIPE_V>();
        Gather(bitPoison_, bitPoisonLookup_, scaleIndexes_.ReinterpretCast<uint32_t>(),
               static_cast<uint32_t>(0), SCALE_COUNT);
        PipeBarrier<PIPE_V>();
        Gather(expandedPoison_, bitPoison_, repeatScaleIndexes_, static_cast<uint32_t>(0), BYTES);
        PipeBarrier<PIPE_V>();
        Or(scaled_.ReinterpretCast<uint16_t>(), scaled_.ReinterpretCast<uint16_t>(),
           expandedPoison_.ReinterpretCast<uint16_t>(), ELEMENTS);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<uint8_t> raw_, scale_;
    LocalTensor<uint16_t> decoded_;
    LocalTensor<half> signedHalf_, scaleHalf_;
    LocalTensor<int32_t> indexes_, scaleIndexes_;
    LocalTensor<uint32_t> lookup_, unitPairs_, nzIndexes_, scaleLookup_, repeatScaleIndexes_;
    LocalTensor<uint32_t> bitMasks_, bitMaskLookup_, bitPoisonLookup_, bitPoison_, expandedPoison_;
    LocalTensor<float> values_, scaleFactors_, expandedFactors_;
    LocalTensor<bfloat16_t> scaled_;
    GlobalTensor<bfloat16_t> stage_;
    GlobalTensor<uint8_t> packed_, scales_;
    GlobalTensor<int32_t> selected_;
    GlobalTensor<float> output_;
    GlobalTensor<uint32_t> tables_;
    uint32_t tiles_, sourceTiles_, blocks_, world_, n_;
};

class TileCube {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR selected, GM_ADDR out, GM_ADDR stage,
                                uint32_t tiles, uint32_t blocks, uint32_t world,
                                uint32_t n, bool shared, uint32_t groups)
    {
        activation_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
        selected_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(selected));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        stage_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(stage));
        stage_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        tiles_ = tiles / N_TILES; blocks_ = blocks / GROUP_K; world_ = world; n_ = n; shared_ = shared;
        const uint64_t rows = shared ? 1 : groups;
        const uint64_t activationElements = rows * M * blocks * K_BLOCK;
        preload_ = PRELOAD_A != 0 && activationElements * sizeof(bfloat16_t) <= L1_BYTES - L1_A_OFFSET;
        fullK_ = preload_ ? blocks * K_BLOCK : 0;
        sharedB_ = LocalTensor<bfloat16_t>(TPosition::B1, 0, 2 * ELEMENTS);
        sharedA_ = LocalTensor<bfloat16_t>(TPosition::A1, L1_A_OFFSET,
                                         preload_ ? static_cast<uint32_t>(activationElements) : M * K);
        a_ = LocalTensor<bfloat16_t>(TPosition::A2, 0, M * K);
        b_ = LocalTensor<bfloat16_t>(TPosition::B2, 0, ELEMENTS);
        c_ = LocalTensor<float>(TPosition::CO1, 0, M * N);
    }

    __aicore__ inline void PreloadActivation(uint32_t groups, uint32_t owned)
    {
        if (!preload_) return;
        Nd2NzParams convert;
        convert.ndNum = 1; convert.nValue = 1; convert.dValue = fullK_;
        convert.srcNdMatrixStride = 0; convert.srcDValue = fullK_;
        convert.dstNzC0Stride = M; convert.dstNzNStride = 1; convert.dstNzMatrixStride = 0;
        bool sharedLoaded = false;
        for (uint32_t group = 0; group < groups; ++group) {
            const int32_t expert = selected_.GetValue(group);
            if (expert < 0 || static_cast<uint32_t>(expert / world_) >= owned) continue;
            if (shared_ && sharedLoaded) break;
            const uint32_t row = shared_ ? 0 : group;
            DataCopy(sharedA_[static_cast<uint64_t>(row) * M * fullK_],
                     activation_[static_cast<uint64_t>(row) * fullK_], convert);
            sharedLoaded = true;
        }
        // All activation rows used by this worker are now immutable for this
        // kernel invocation. B1 and the GM ring retain their original fences.
        SetFlag<HardEvent::MTE2_MTE1>(0);
        WaitFlag<HardEvent::MTE2_MTE1>(0);
    }

    __aicore__ inline void Process(uint32_t groups, uint32_t owned)
    {
        PreloadActivation(groups, owned);
        const uint32_t owner = GetBlockIdx();
        for (uint32_t task = owner; task < groups * tiles_; task += GetBlockNum()) {
            const uint32_t group = task / tiles_;
            const uint32_t tile = task % tiles_;
            const int32_t expert = selected_.GetValue(group);
            if (expert < 0 || static_cast<uint32_t>(expert / world_) >= owned) continue;
            const uint64_t xBase = shared_ ? 0 : static_cast<uint64_t>(group) * blocks_ * K;
            for (uint32_t block = 0; block < blocks_; ++block) {
                CrossCoreWaitFlag<0x2>(READY_FLAG + (block & 1));
                DataCopy(sharedB_[(block & 1) * ELEMENTS],
                         stage_[owner * 2 * ELEMENTS + (block & 1) * ELEMENTS], ELEMENTS);
                CrossCoreSetFlag<0x2, PIPE_MTE2>(FREE_FLAG + (block & 1));
                if (!preload_) {
                    Nd2NzParams convert;
                    convert.ndNum = 1;
                    convert.nValue = 1;
                    convert.dValue = K;
                    convert.srcNdMatrixStride = 0;
                    convert.srcDValue = K;
                    convert.dstNzC0Stride = M;
                    convert.dstNzNStride = 1;
                    convert.dstNzMatrixStride = 0;
                    DataCopy(sharedA_, activation_[xBase + block * K], convert);
                }
                SetFlag<HardEvent::MTE2_MTE1>(0);
                WaitFlag<HardEvent::MTE2_MTE1>(0);
                LoadData2DParams load;
                load.startIndex = 0; load.repeatTimes = K / 16;
                load.srcStride = 1; load.dstGap = 0; load.ifTranspose = false;
                const uint32_t row = shared_ ? 0 : group;
                const uint64_t aBase = preload_ ? static_cast<uint64_t>(row) * M * fullK_ : 0;
                load.startIndex = preload_ ? block * (K / 16) : 0;
                LoadData(a_, sharedA_[aBase], load);
                // The B1 ring still contains only the current K tile, not the
                // complete K dimension; never reuse the full-A start index.
                // Mmad consumes the complete K16-major/N16-minor B tile.
                // Keep the native block order through the L1 -> L0B transfer,
                // rather than rearranging N32 into independent N16 panels.
                load.startIndex = 0;
                load.repeatTimes = (K / 16) * N_TILES;
                LoadData(b_, sharedB_[(block & 1) * ELEMENTS], load);
                // MTE2 must not replace the single activation L1 slot before
                // MTE1 has loaded its previous value into L0A.
                SetFlag<HardEvent::MTE1_MTE2>(0);
                WaitFlag<HardEvent::MTE1_MTE2>(0);
                SetFlag<HardEvent::MTE1_M>(0);
                WaitFlag<HardEvent::MTE1_M>(0);
                MmadParams params;
                params.m = M; params.n = N; params.k = K;
                params.cmatrixInitVal = block == 0;
                params.cmatrixSource = false;
                Mmad(c_, a_, b_, params);
                SetFlag<HardEvent::M_MTE1>(0);
                WaitFlag<HardEvent::M_MTE1>(0);
            }
            SetFlag<HardEvent::M_FIX>(0);
            WaitFlag<HardEvent::M_FIX>(0);
            FixpipeParamsV220 fix;
            fix.nSize = N; fix.mSize = 1; fix.srcStride = M;
            fix.dstStride = n_; fix.quantPre = QuantMode_t::NoQuant;
            Fixpipe<float, float, CFG_ROW_MAJOR>(output_[static_cast<uint64_t>(group) * n_ + tile * N], c_, fix);
            SetFlag<HardEvent::FIX_M>(0);
            WaitFlag<HardEvent::FIX_M>(0);
        }
    }
private:
    LocalTensor<bfloat16_t> sharedA_, sharedB_, a_, b_;
    LocalTensor<float> c_;
    GlobalTensor<bfloat16_t> activation_, stage_;
    GlobalTensor<int32_t> selected_;
    GlobalTensor<float> output_;
    uint32_t tiles_, blocks_, world_, n_, fullK_;
    bool shared_, preload_;
};
} // namespace

extern "C" __global__ __aicore__ void relic_online_tile_cube(
    GM_ADDR x, GM_ADDR b, GM_ADDR s, GM_ADDR selected, GM_ADDR out, GM_ADDR stage, GM_ADDR tables,
    uint32_t tiles, uint32_t blocks, uint32_t groups, uint32_t owned,
    uint32_t world, uint32_t n, uint32_t shared)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if ASCEND_IS_AIV {
        TileVector vector;
        vector.Init(b, s, selected, stage, out, tables, tiles, blocks, world, n);
        vector.Process(groups, owned);
    } else {
        TileCube cube;
        cube.Init(x, selected, out, stage, tiles, blocks, world, n, shared != 0, groups);
        cube.Process(groups, owned);
    }
}

extern "C" uint32_t relic_online_tile_cube_unsigned_byte_cast()
{
    return UNSIGNED_BYTE_CAST;
}

extern "C" uint32_t relic_online_tile_cube_compact_layout()
{
    return CUBE_PACKED;
}

extern "C" uint32_t relic_online_tile_cube_output_rows()
{
    return N;
}

extern "C" uint32_t relic_online_tile_cube_k_blocks_per_tile()
{
    return GROUP_K;
}

extern "C" uint32_t relic_online_tile_cube_preload_activation()
{
    return PRELOAD_A;
}

extern "C" uint32_t relic_online_tile_cube_direct_bf16_bits()
{
    return DIRECT_BITS;
}

extern "C" void relic_online_tile_cube_launch(uint32_t blockDim, void *stream,
    void *x, void *b, void *s, void *selected, void *out, void *stage,
    uint32_t tiles, uint32_t blocks, uint32_t groups, uint32_t owned,
    uint32_t world, uint32_t n, uint32_t shared)
{
    relic_online_tile_cube<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(x), reinterpret_cast<GM_ADDR>(b), reinterpret_cast<GM_ADDR>(s),
        reinterpret_cast<GM_ADDR>(selected), reinterpret_cast<GM_ADDR>(out), reinterpret_cast<GM_ADDR>(stage),
        nullptr, tiles, blocks, groups, owned, world, n, shared);
}

extern "C" uint32_t relic_online_tile_cube_prepared_tables()
{
    return PREPARED_TABLES;
}

extern "C" uint32_t relic_online_tile_cube_table_words()
{
    return PREPARED_TABLES ? TABLE_WORDS : 0;
}

extern "C" void relic_online_tile_cube_launch_tables(uint32_t blockDim, void *stream,
    void *x, void *b, void *s, void *selected, void *out, void *stage, void *tables,
    uint32_t tiles, uint32_t blocks, uint32_t groups, uint32_t owned,
    uint32_t world, uint32_t n, uint32_t shared)
{
    relic_online_tile_cube<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(x), reinterpret_cast<GM_ADDR>(b), reinterpret_cast<GM_ADDR>(s),
        reinterpret_cast<GM_ADDR>(selected), reinterpret_cast<GM_ADDR>(out), reinterpret_cast<GM_ADDR>(stage),
        reinterpret_cast<GM_ADDR>(tables), tiles, blocks, groups, owned, world, n, shared);
}
