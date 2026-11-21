"""Original E2M1/E8M0 scaled BF16 bits consumed directly by a single-row AIV GEMV."""
import triton
import triton.language as tl


@triton.jit
def original_bit_aiv_gemv_kernel(X, B, S, IDS, OUT,
    N: tl.constexpr, K: tl.constexpr, TILES: tl.constexpr, BLOCKS: tl.constexpr,
    SLOTS: tl.constexpr, OWNED: tl.constexpr, WORLD: tl.constexpr,
    SHARED: tl.constexpr, KT: tl.constexpr, NR: tl.constexpr, INDEX64: tl.constexpr):
    rows = tl.arange(0, NR)
    lanes = tl.arange(0, KT)
    for task in range(tl.program_id(0), SLOTS * ((N + NR - 1) // NR), tl.num_programs(0)):
        group = task // ((N + NR - 1) // NR)
        logical_tile = task % ((N + NR - 1) // NR)
        tile = logical_tile // (16 // NR)
        within = (logical_tile % (16 // NR)) * NR + rows
        expert = tl.load(IDS + group)
        column = logical_tile * NR + rows
        value = tl.full((NR,), 0.0, tl.float32)
        if expert >= 0 and expert // WORLD < OWNED:
            local = expert // WORLD
            if INDEX64:
                local = local.to(tl.int64)
                tile = tile.to(tl.int64)
            for start in range(0, K, KT):
                k = start + lanes
                kb = k // 32
                pair = (k % 32) // 2
                raw = tl.load(B + (local * TILES + tile) * BLOCKS * 256
                              + kb[None, :] * 256 + within[:, None] * 16 + pair[None, :],
                              mask=(k[None, :] < K) & (column[:, None] < N), other=0).to(tl.int32)
                scale = tl.load(S + (local * TILES + tile) * BLOCKS * 16
                                + kb[None, :] * 16 + within[:, None],
                                mask=(k[None, :] < K) & (column[:, None] < N), other=127).to(tl.int32)
                code = (raw >> (4 * (k[None, :] & 1))) & 15
                magnitude = code & 7
                delta = tl.where(magnitude >= 6, 2, tl.where(magnitude >= 4, 1, tl.where(magnitude >= 2, 0, -1)))
                mantissa = tl.where((magnitude == 3) | (magnitude == 5) | (magnitude == 7), 64, 0)
                bits = ((code & 8) << 12) | ((scale + delta) << 7) | mantissa
                bits = tl.where(magnitude == 0, 0, bits)
                bits = tl.where((scale >= 2) & (scale <= 252), bits, 0x7fc0)
                weight = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).to(tl.float32)
                source_row = 0 if SHARED else group
                activation = tl.load(X + source_row * K + k, mask=k < K, other=0).to(tl.float32)
                product = weight * activation[None, :]
                value = value + tl.sum(product, axis=1)
        elif expert != -1:
            value = tl.full((NR,), float("nan"), tl.float32)
        tl.store(OUT + group * N + column, value, mask=column < N)
