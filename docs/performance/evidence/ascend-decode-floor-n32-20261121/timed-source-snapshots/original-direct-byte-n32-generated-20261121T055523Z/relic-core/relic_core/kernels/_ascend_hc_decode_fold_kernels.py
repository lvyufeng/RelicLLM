"""Decode-only original-FP32 HC collapse and residual-fold kernels."""
import triton
import triton.language as tl


@triton.jit
def hc_decode_pre_kernel(X, PRE, OUT, DIM: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r0 = tl.load(X + index, index < DIM, other=0).to(tl.float32)
    r1 = tl.load(X + DIM + index, index < DIM, other=0).to(tl.float32)
    r2 = tl.load(X + 2 * DIM + index, index < DIM, other=0).to(tl.float32)
    r3 = tl.load(X + 3 * DIM + index, index < DIM, other=0).to(tl.float32)
    w0 = tl.load(PRE).to(tl.float32)
    w1 = tl.load(PRE + 1).to(tl.float32)
    w2 = tl.load(PRE + 2).to(tl.float32)
    w3 = tl.load(PRE + 3).to(tl.float32)
    # Multiplications must round before the explicit sequential reduction.
    value = ((r0 * w0 + r1 * w1) + r2 * w2) + r3 * w3
    value = tl.where(value == 0.0, 0.0, value)
    tl.store(OUT + index, value, index < DIM)


@triton.jit
def hc_decode_post_kernel(X, RESIDUAL, POST, COMB, OUT, DIM: tl.constexpr, BLOCK: tl.constexpr):
    output_head = tl.program_id(0)
    index = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    r0 = tl.load(RESIDUAL + index, index < DIM, other=0).to(tl.float32)
    r1 = tl.load(RESIDUAL + DIM + index, index < DIM, other=0).to(tl.float32)
    r2 = tl.load(RESIDUAL + 2 * DIM + index, index < DIM, other=0).to(tl.float32)
    r3 = tl.load(RESIDUAL + 3 * DIM + index, index < DIM, other=0).to(tl.float32)
    c0 = tl.load(COMB + output_head).to(tl.float32)
    c1 = tl.load(COMB + 4 + output_head).to(tl.float32)
    c2 = tl.load(COMB + 8 + output_head).to(tl.float32)
    c3 = tl.load(COMB + 12 + output_head).to(tl.float32)
    # Explicit sequential candidate for the strided four-input reduction;
    # current-device parity is mandatory before accepting this order.
    residual_value = ((r0 * c0 + r1 * c1) + r2 * c2) + r3 * c3
    residual_value = tl.where(residual_value == 0.0, 0.0, residual_value)
    source = tl.load(X + index, index < DIM, other=0).to(tl.float32)
    post = tl.load(POST + output_head).to(tl.float32)
    value = post * source + residual_value
    tl.store(OUT + output_head * DIM + index, value, index < DIM)
