"""Substitute the split-K sparse attention into a real decode run, and measure what it buys.

`probe_qwen4_exp_qsa_splitk.py` establishes that the split-K kernel is 7.84x faster than the shipped
one on the same inputs and agrees with a torch reference exactly. A per-call ratio is not a
throughput, though: the step is a recorded graph, the kernel is one item in it, and the review of
this work has already been burned once by trusting a proportion instead of a measurement. So this
probe replaces the op the model actually calls and runs the ordinary benchmark.

It does that by swapping `load_ops` in the attention module for a proxy that answers
`qwen4_exp_qsa_bf16_forward` with the split-K build and forwards every other name untouched. Nothing
else about the model changes: same weights, same budget, same graph, same tokens expected.

Run on a leased CUDA host, from the repository root, eight cards like the baseline it is compared
against:
  torchrun --nproc_per_node=8 --master_port=60641 tests/probe_qwen4_exp_splitk_decode.py \
      --context-tokens 8192 --max-new-tokens 256 --label splitk-8k
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kDim = 256;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
constexpr int kSplits = 16;
constexpr int kMaxChunk = 256;

__device__ __forceinline__ float ld(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

// One CTA per (split, kv_head): a slice of the selected positions reduced to a running max, a
// running sum and a partial output. flash-decoding's first pass.
template <int HeadsPerKv>
__global__ void qsa_splitk_partial_kernel(
        const __nv_bfloat16* __restrict__ query,
        const __nv_bfloat16* __restrict__ key,
        const __nv_bfloat16* __restrict__ value,
        const int32_t* __restrict__ selected,
        float* __restrict__ part_acc,
        float* __restrict__ part_m,
        float* __restrict__ part_l,
        int kv_len,
        int selected_len,
        float softmax_scale) {
    const int split = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int chunk = (selected_len + kSplits - 1) / kSplits;
    const int begin = split * chunk;
    const int count = min(chunk, selected_len - begin);

    __shared__ float q_sh[HeadsPerKv][kDim];
    __shared__ float scores[HeadsPerKv][kMaxChunk];
    __shared__ int32_t ids[kMaxChunk];

    const int first_head = kv_head * HeadsPerKv;
    const __nv_bfloat16* q_base = query + static_cast<int64_t>(first_head) * kDim;
    for (int i = tid; i < HeadsPerKv * kDim; i += kThreads) {
        q_sh[i / kDim][i % kDim] = ld(q_base + i);
    }
    for (int i = tid; i < count; i += kThreads) {
        ids[i] = selected[begin + i];
    }
    __syncthreads();

    const int64_t kv_head_base = static_cast<int64_t>(kv_head) * kv_len * kDim;

    // One warp a position. Every warp walks the same number of iterations so that the barrier
    // after this loop is reached by all of them.
    for (int base = 0; base < count; base += kWarps) {
        const int slot = base + warp;
        float acc[HeadsPerKv];
#pragma unroll
        for (int h = 0; h < HeadsPerKv; ++h) {
            acc[h] = 0.0f;
        }
        const int token = slot < count ? ids[slot] : -1;
        if (token >= 0 && token < kv_len) {
            const __nv_bfloat16* kp =
                key + kv_head_base + static_cast<int64_t>(token) * kDim;
            for (int d = lane; d < kDim; d += 32) {
                const float kv = ld(kp + d);
#pragma unroll
                for (int h = 0; h < HeadsPerKv; ++h) {
                    acc[h] += q_sh[h][d] * kv;
                }
            }
        }
#pragma unroll
        for (int h = 0; h < HeadsPerKv; ++h) {
            float v = acc[h];
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, off);
            }
            if (lane == 0) {
                const bool valid = token >= 0 && token < kv_len;
                scores[h][slot] = (valid && slot < count) ? v * softmax_scale : -INFINITY;
            }
        }
    }
    __syncthreads();

    // Three phases, not one: a thread that stored `exp(score - max)` while another was still
    // scanning for the maximum would hand it a score of at most 1.
    float m[HeadsPerKv];
    float l[HeadsPerKv];
    float out[HeadsPerKv];
#pragma unroll
    for (int h = 0; h < HeadsPerKv; ++h) {
        m[h] = -INFINITY;
        for (int slot = 0; slot < count; ++slot) {
            m[h] = fmaxf(m[h], scores[h][slot]);
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < HeadsPerKv; ++h) {
        for (int slot = tid; slot < count; slot += kThreads) {
            scores[h][slot] = scores[h][slot] == -INFINITY
                ? 0.0f
                : __expf(scores[h][slot] - m[h]);
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 0; h < HeadsPerKv; ++h) {
        l[h] = 0.0f;
        for (int slot = 0; slot < count; ++slot) {
            l[h] += scores[h][slot];
        }
        out[h] = 0.0f;
    }
    for (int slot = 0; slot < count; ++slot) {
        const int token = ids[slot];
        if (token < 0 || token >= kv_len) {
            continue;
        }
        const __nv_bfloat16* vp =
            value + kv_head_base + static_cast<int64_t>(token) * kDim;
        const float vd = ld(vp + tid);
#pragma unroll
        for (int h = 0; h < HeadsPerKv; ++h) {
            out[h] += scores[h][slot] * vd;
        }
    }

    const int64_t split_index = static_cast<int64_t>(kv_head) * kSplits + split;
#pragma unroll
    for (int h = 0; h < HeadsPerKv; ++h) {
        part_acc[split_index * HeadsPerKv * kDim + static_cast<int64_t>(h) * kDim + tid] = out[h];
    }
    if (tid < HeadsPerKv) {
        part_m[split_index * HeadsPerKv + tid] = m[tid];
        part_l[split_index * HeadsPerKv + tid] = l[tid];
    }
}

template <int HeadsPerKv>
__global__ void qsa_splitk_combine_kernel(
        const float* __restrict__ part_acc,
        const float* __restrict__ part_m,
        const float* __restrict__ part_l,
        __nv_bfloat16* __restrict__ output) {
    const int kv_head = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);

    float m = -INFINITY;
    for (int split = 0; split < kSplits; ++split) {
        m = fmaxf(m, part_m[(kv_head * kSplits + split) * HeadsPerKv + head]);
    }
    float acc = 0.0f;
    float denom = 0.0f;
    for (int split = 0; split < kSplits; ++split) {
        const int64_t idx = static_cast<int64_t>(kv_head) * kSplits + split;
        const float p = part_l[idx * HeadsPerKv + head];
        const float weight = p == 0.0f ? 0.0f : __expf(part_m[idx * HeadsPerKv + head] - m);
        denom += p * weight;
        acc += part_acc[idx * HeadsPerKv * kDim + head * kDim + tid] * weight;
    }
    acc = denom == 0.0f ? 0.0f : acc / denom;
    output[static_cast<int64_t>(kv_head * HeadsPerKv + head) * kDim + tid] =
        __float2bfloat16(acc);
}

torch::Tensor qsa_splitk_forward(
        const torch::Tensor& query,
        const torch::Tensor& key,
        const torch::Tensor& value,
        const torch::Tensor& selected,
        double softmax_scale) {
    c10::cuda::CUDAGuard guard(query.device());
    const int kv_len = static_cast<int>(key.size(2));
    const int kv_heads = static_cast<int>(key.size(1));
    const int query_heads = static_cast<int>(query.size(2));
    const int selected_len = static_cast<int>(selected.size(2));
    TORCH_CHECK(query.size(2) % kv_heads == 0, "query heads must divide by kv heads");
    TORCH_CHECK(query.size(3) == kDim, "head dim must be 256");
    TORCH_CHECK(selected_len <= kSplits * kMaxChunk, "selection too long for this build");
    const int heads_per_kv = query_heads / kv_heads;

    auto output = torch::empty(query.sizes(), query.options());
    auto fopts = query.options().dtype(torch::kFloat32);
    auto part_acc = torch::empty({kv_heads, kSplits, heads_per_kv, kDim}, fopts);
    auto part_m = torch::empty({kv_heads, kSplits, heads_per_kv}, fopts);
    auto part_l = torch::empty({kv_heads, kSplits, heads_per_kv}, fopts);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const dim3 grid(kSplits, kv_heads);

#define QSA_LAUNCH(HPK)                                                            \
    qsa_splitk_partial_kernel<HPK><<<grid, kThreads, 0, stream>>>(                 \
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),                  \
        reinterpret_cast<const __nv_bfloat16*>(key.data_ptr()),                    \
        reinterpret_cast<const __nv_bfloat16*>(value.data_ptr()),                  \
        selected.data_ptr<int32_t>(),                                              \
        part_acc.data_ptr<float>(), part_m.data_ptr<float>(),                       \
        part_l.data_ptr<float>(), kv_len, selected_len,                            \
        static_cast<float>(softmax_scale));                                        \
    qsa_splitk_combine_kernel<HPK><<<dim3(kv_heads, HPK), kDim, 0, stream>>>(      \
        part_acc.data_ptr<float>(), part_m.data_ptr<float>(),                       \
        part_l.data_ptr<float>(),                                                  \
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr()));

    switch (heads_per_kv) {
        case 1: QSA_LAUNCH(1); break;
        case 2: QSA_LAUNCH(2); break;
        case 3: QSA_LAUNCH(3); break;
        case 4: QSA_LAUNCH(4); break;
        case 6: QSA_LAUNCH(6); break;
        case 12: QSA_LAUNCH(12); break;
        default:
            TORCH_CHECK(false, "unsupported heads per kv head: ", heads_per_kv);
    }
#undef QSA_LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("qsa_splitk_forward", &qsa_splitk_forward, "split-K decode sparse attention");
}
"""


class _SplitKProxy:
    """Answer the QSA op with the split-K build and everything else with the shipped extension."""

    def __init__(self, inner, splitk):
        self._inner = inner
        self._splitk = splitk
        self.calls = 0
        self.fell_back = 0
        self._seen = set()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def qwen4_exp_qsa_bf16_forward(self, query, key, value, selected, softmax_scale):
        shape = (tuple(query.shape), tuple(key.shape), tuple(selected.shape))
        if shape not in self._seen:
            # The first end-to-end attempt produced zeros because it took the split-K path on a
            # prefill call -- 512 query rows a layer, each with its own 512-entry selection -- while
            # the kernel implements a single query row. The shapes decide applicability, so every
            # one of them is reported rather than assumed.
            self._seen.add(shape)
            print(f"# qsa shapes: q={shape[0]} {query.dtype} k={shape[1]} "
                  f"sel={shape[2]} {selected.dtype} scale={softmax_scale}", flush=True)
        # Decode only: one query row, one KV head a rank, a selection of at most 4096 positions.
        decode = (
            query.dtype is torch.bfloat16
            and query.is_cuda
            and query.size(3) == 256
            and query.size(1) == 1
            and selected.dim() == 3
            and selected.size(1) == 1
            and selected.size(2) <= 4096
        )
        if decode:
            self.calls += 1
            return self._splitk.qsa_splitk_forward(query, key, value, selected, softmax_scale)
        self.fell_back += 1
        return self._inner.qwen4_exp_qsa_bf16_forward(
            query, key, value, selected, softmax_scale)


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--splits", type=int, default=16)
    known, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest

    from torch.utils.cpp_extension import load_inline

    splitk = load_inline(
        name="qwen4_qsa_splitk_decode",
        cpp_sources="",
        cuda_sources=_CUDA_SOURCE,
        functions=None,
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=False,
    )

    from relicllm.models.qwen4_exp import attention as attention_module

    original = attention_module.load_ops
    state = {}

    def patched_load_ops():
        inner = original()
        if inner is None:
            return None
        if "proxy" not in state:
            state["proxy"] = _SplitKProxy(inner, splitk)
        return state["proxy"]

    attention_module.load_ops = patched_load_ops

    # The baseline script is the measurement; this probe only decides which kernel it runs. It is
    # loaded by path because `tests/` is not a package -- it has no `__init__.py`, the CWD is what
    # puts it on `sys.path`.
    import importlib.util

    bench_path = Path(__file__).resolve().parent / "bench_qwen4_exp_baseline.py"
    spec = importlib.util.spec_from_file_location("bench_qwen4_exp_baseline", bench_path)
    bench_qwen4_exp_baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench_qwen4_exp_baseline)

    try:
        bench_qwen4_exp_baseline.main()
    finally:
        attention_module.load_ops = original
        proxy = state.get("proxy")
        if proxy is not None:
            print(f"# split-K served {proxy.calls} calls, fell back {proxy.fell_back}", flush=True)


if __name__ == "__main__":
    main()