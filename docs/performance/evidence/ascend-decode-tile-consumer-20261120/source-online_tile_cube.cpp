// Original-byte online unpack with explicit AIV -> bounded GM ring -> L1 -> AIC ownership.
// This is an independent, opt-in Ascend910B prototype; no upstream kernel is copied.
#include "kernel_operator.h"
#include <cstdint>

using namespace AscendC;

namespace {
constexpr uint32_t M = 16;
constexpr uint32_t N = 16;
#ifndef RELIC_TILE_K_BLOCKS
#define RELIC_TILE_K_BLOCKS 1
#endif
constexpr uint32_t GROUP_K = RELIC_TILE_K_BLOCKS;
static_assert(GROUP_K == 1 || GROUP_K == 2 || GROUP_K == 4, "retain an integral original K32 scale grid");
constexpr uint32_t K_BLOCK = 32;
constexpr uint32_t K = K_BLOCK * GROUP_K;
constexpr uint32_t SCALE_COUNT = N * GROUP_K;
constexpr uint32_t BYTES = N * K / 2;
constexpr uint32_t ELEMENTS = N * K;
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
static_assert(EXPANDED_FACTOR_OFFSET + ELEMENTS * sizeof(float) <= 128 * 1024,
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
                                uint32_t tiles, uint32_t blocks, uint32_t world, uint32_t n)
    {
        packed_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(b));
        scales_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t *>(s));
        selected_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(selected));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        n_ = n;
        tiles_ = tiles;
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
        scaled_ = LocalTensor<bfloat16_t>(TPosition::VECCALC, SCALED_OFFSET, ELEMENTS);
        nzIndexes_ = LocalTensor<uint32_t>(TPosition::VECCALC, NZ_OFFSET, BYTES);
        scaleHalf_ = LocalTensor<half>(TPosition::VECCALC, SCALE_HALF_OFFSET, SCALE_COUNT);
        scaleIndexes_ = LocalTensor<int32_t>(TPosition::VECCALC, SCALE_INDEX_OFFSET, SCALE_COUNT);
        scaleFactors_ = LocalTensor<float>(TPosition::VECCALC, SCALE_FACTOR_OFFSET, SCALE_COUNT);
        scaleLookup_ = LocalTensor<uint32_t>(TPosition::VECCALC, SCALE_LOOKUP_OFFSET, 256);
        repeatScaleIndexes_ = LocalTensor<uint32_t>(TPosition::VECCALC, SCALE_REPEAT_OFFSET, ELEMENTS);
        expandedFactors_ = LocalTensor<float>(TPosition::VECCALC, EXPANDED_FACTOR_OFFSET, ELEMENTS);
        for (uint32_t index = 0; index < ELEMENTS; ++index) {
            repeatScaleIndexes_.SetValue(index, (index / K_BLOCK) * sizeof(float));
        }
        for (uint32_t index = 0; index < BYTES; ++index) {
            const uint32_t originalBlock = index / 256;
            const uint32_t half = (index % 256) / 128;
            const uint32_t row = (index % 128) / 8;
            const uint32_t pair = index % 8;
            nzIndexes_.SetValue(index, (originalBlock * 256 + row * 16 + half * 8 + pair) * sizeof(uint32_t));
        }
        for (uint32_t index = 0; index < 256; ++index) {
            const uint8_t byte = (index + 128) & 255;
            scaleLookup_.SetValue(index, byte >= 2 && byte <= 252 ? static_cast<uint32_t>(byte) << 23 : 0x7fc00000);
            lookup_.SetValue(index, static_cast<uint32_t>(OriginalFp4Bits(byte & 15, 127)) |
                                    (static_cast<uint32_t>(OriginalFp4Bits(byte >> 4, 127)) << 16));
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
            const uint64_t base = (static_cast<uint64_t>(local) * tiles_ + tile) * blocks_;
            for (uint32_t block = 0; block < blocks_; ++block) {
                const uint32_t slot = block & 1;
                if (block >= 2) CrossCoreWaitFlag<0x2>(FREE_FLAG + slot);
                DataCopy(raw_, packed_[(base + block) * BYTES], BYTES);
                DataCopyPad(scale_, scales_[(base + block) * SCALE_COUNT],
                            DataCopyExtParams{1, SCALE_COUNT, 0, 0, 0},
                            DataCopyPadExtParams<uint8_t>{false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_S>(0);
                WaitFlag<HardEvent::MTE2_S>(0);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                Cast(signedHalf_, raw_.ReinterpretCast<int8_t>(), RoundMode::CAST_NONE, BYTES);
                PipeBarrier<PIPE_V>();
                Cast(indexes_, signedHalf_, RoundMode::CAST_RINT, BYTES);
                PipeBarrier<PIPE_V>();
                Adds(indexes_, indexes_, static_cast<int32_t>(128), BYTES);
                PipeBarrier<PIPE_V>();
                ShiftLeft(indexes_.ReinterpretCast<uint32_t>(), indexes_.ReinterpretCast<uint32_t>(),
                          static_cast<uint32_t>(2), BYTES);
                PipeBarrier<PIPE_V>();
                Gather(unitPairs_, lookup_, indexes_.ReinterpretCast<uint32_t>(), static_cast<uint32_t>(0), BYTES);
                PipeBarrier<PIPE_V>();
                Cast(values_, unitPairs_.ReinterpretCast<bfloat16_t>(), RoundMode::CAST_NONE, ELEMENTS);
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
                Gather(expandedFactors_, scaleFactors_, repeatScaleIndexes_, static_cast<uint32_t>(0), ELEMENTS);
                PipeBarrier<PIPE_V>();
                Mul(values_, values_, expandedFactors_, ELEMENTS);
                PipeBarrier<PIPE_V>();
                Cast(scaled_, values_, RoundMode::CAST_RINT, ELEMENTS);
                PipeBarrier<PIPE_V>();
                Gather(decoded_[slot * ELEMENTS].ReinterpretCast<uint32_t>(),
                       scaled_.ReinterpretCast<uint32_t>(), nzIndexes_, static_cast<uint32_t>(0), BYTES);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopy(stage_[owner * 2 * ELEMENTS + slot * ELEMENTS],
                         decoded_[slot * ELEMENTS].ReinterpretCast<bfloat16_t>(), ELEMENTS);
                CrossCoreSetFlag<0x2, PIPE_MTE3>(READY_FLAG + slot);
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
    LocalTensor<uint8_t> raw_, scale_;
    LocalTensor<uint16_t> decoded_;
    LocalTensor<half> signedHalf_, scaleHalf_;
    LocalTensor<int32_t> indexes_, scaleIndexes_;
    LocalTensor<uint32_t> lookup_, unitPairs_, nzIndexes_, scaleLookup_, repeatScaleIndexes_;
    LocalTensor<float> values_, scaleFactors_, expandedFactors_;
    LocalTensor<bfloat16_t> scaled_;
    GlobalTensor<bfloat16_t> stage_;
    GlobalTensor<uint8_t> packed_, scales_;
    GlobalTensor<int32_t> selected_;
    GlobalTensor<float> output_;
    uint32_t tiles_, blocks_, world_, n_;
};

class TileCube {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR selected, GM_ADDR out, GM_ADDR stage,
                                uint32_t tiles, uint32_t blocks, uint32_t world,
                                uint32_t n, bool shared)
    {
        activation_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(x));
        selected_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(selected));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(out));
        stage_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(stage));
        stage_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        tiles_ = tiles; blocks_ = blocks / GROUP_K; world_ = world; n_ = n; shared_ = shared;
        sharedB_ = LocalTensor<bfloat16_t>(TPosition::B1, 0, 2 * ELEMENTS);
        sharedA_ = LocalTensor<bfloat16_t>(TPosition::A1, L1_A_OFFSET, M * K);
        a_ = LocalTensor<bfloat16_t>(TPosition::A2, 0, M * K);
        b_ = LocalTensor<bfloat16_t>(TPosition::B2, 0, ELEMENTS);
        c_ = LocalTensor<float>(TPosition::CO1, 0, M * N);
    }

    __aicore__ inline void Process(uint32_t groups, uint32_t owned)
    {
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
                SetFlag<HardEvent::MTE2_MTE1>(0);
                WaitFlag<HardEvent::MTE2_MTE1>(0);
                LoadData2DParams load;
                load.startIndex = 0; load.repeatTimes = K / 16;
                load.srcStride = 1; load.dstGap = 0; load.ifTranspose = false;
                LoadData(a_, sharedA_, load);
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
    uint32_t tiles_, blocks_, world_, n_;
    bool shared_;
};
} // namespace

extern "C" __global__ __aicore__ void relic_online_tile_cube(
    GM_ADDR x, GM_ADDR b, GM_ADDR s, GM_ADDR selected, GM_ADDR out, GM_ADDR stage,
    uint32_t tiles, uint32_t blocks, uint32_t groups, uint32_t owned,
    uint32_t world, uint32_t n, uint32_t shared)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
    if ASCEND_IS_AIV {
        TileVector vector;
        vector.Init(b, s, selected, stage, out, tiles, blocks, world, n);
        vector.Process(groups, owned);
    } else {
        TileCube cube;
        cube.Init(x, selected, out, stage, tiles, blocks, world, n, shared != 0);
        cube.Process(groups, owned);
    }
}

extern "C" uint32_t relic_online_tile_cube_k_blocks_per_tile()
{
    return GROUP_K;
}

extern "C" void relic_online_tile_cube_launch(uint32_t blockDim, void *stream,
    void *x, void *b, void *s, void *selected, void *out, void *stage,
    uint32_t tiles, uint32_t blocks, uint32_t groups, uint32_t owned,
    uint32_t world, uint32_t n, uint32_t shared)
{
    relic_online_tile_cube<<<blockDim, nullptr, stream>>>(
        reinterpret_cast<GM_ADDR>(x), reinterpret_cast<GM_ADDR>(b), reinterpret_cast<GM_ADDR>(s),
        reinterpret_cast<GM_ADDR>(selected), reinterpret_cast<GM_ADDR>(out), reinterpret_cast<GM_ADDR>(stage),
        tiles, blocks, groups, owned, world, n, shared);
}
