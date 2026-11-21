"""Bounded FP32 coefficient/RMS work; original nonlinear providers stay separate."""
import triton
import triton.language as tl


@triton.jit
def hc_project_rms_kernel(X, W, OUT, K: tl.constexpr, EPS: tl.constexpr, KT: tl.constexpr):
    row = tl.program_id(0)
    lanes = tl.arange(0, KT)
    projection = tl.full((), 0.0, tl.float32)
    squares = tl.full((), 0.0, tl.float32)
    for begin in range(0, K, KT):
        k = begin + lanes
        x = tl.load(X + k, mask=k < K, other=0).to(tl.float32)
        coefficient = tl.load(W + row * K + k, mask=k < K, other=0)
        projection = projection + tl.sum(x * coefficient, axis=0)
        squares = squares + tl.sum(x * x, axis=0)
    value = projection * tl.rsqrt(squares / K + EPS)
    tl.store(OUT + row, value)
